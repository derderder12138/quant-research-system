"""
SQLite 持久化模块 — 多用户支持。
每用户独立存储：分析记录、持仓、密码哈希。
"""

import os, sqlite3, hashlib, re
from typing import Optional, List, Dict, Any

DB_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DB_DIR, exist_ok=True)

_RATING_PATTERN = re.compile(r"\[(积极建仓|谨慎持有|观望等待|减仓回避)\]")


def _db_path(username: str) -> str:
    """每个用户独立的 SQLite 文件，确保数据隔离。"""
    safe = re.sub(r"[^a-zA-Z0-9_一-鿿]", "_", username)
    return os.path.join(DB_DIR, f"user_{safe}.db")


def _ensure_tables(username: str) -> sqlite3.Connection:
    """确保用户的表存在，返回连接。"""
    path = _db_path(username)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            fetch_success BOOLEAN NOT NULL,
            error_message TEXT DEFAULT '',
            technical_analysis TEXT DEFAULT '',
            fundamental_analysis TEXT DEFAULT '',
            final_report TEXT DEFAULT '',
            rating TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL DEFAULT '默认池',
            code TEXT NOT NULL,
            added_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_info (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


# ========== 用户管理 ==========

def _hash(password: str) -> str:
    return hashlib.sha256(f"quant_salt_{password}".encode()).hexdigest()


def register_user(username: str, password: str) -> bool:
    """注册新用户。已存在则返回 False。"""
    conn = _ensure_tables(username)
    exists = conn.execute("SELECT 1 FROM user_info WHERE username=?", (username,)).fetchone()
    if exists:
        conn.close()
        return False
    conn.execute("INSERT INTO user_info (username, password_hash) VALUES (?,?)",
                 (username, _hash(password)))
    conn.commit(); conn.close()
    return True


def verify_user(username: str, password: str) -> bool:
    """验证用户名密码。用户不存在时自动注册。"""
    conn = _ensure_tables(username)
    row = conn.execute("SELECT password_hash FROM user_info WHERE username=?", (username,)).fetchone()
    if row:
        ok = row[0] == _hash(password)
        conn.close()
        return ok
    # 新用户：用这个密码注册
    conn.execute("INSERT INTO user_info (username, password_hash) VALUES (?,?)",
                 (username, _hash(password)))
    conn.commit(); conn.close()
    return True


# ========== 分析结果 ==========

def init_db(db_path: str = "") -> None:
    """兼容旧接口——不再使用全局 DB，改为按用户名管理。保留空实现。"""
    pass


def save_result(username: str, result: Dict[str, Any]) -> int:
    conn = _ensure_tables(username)
    rating = None
    if result.get("data_fetch_success") and result.get("final_report"):
        rating = extract_rating(result["final_report"])
    elif not result.get("data_fetch_success"):
        rating = "数据失败"
    c = conn.execute("""
        INSERT INTO analysis_results (ticker, fetch_success, error_message, technical_analysis, fundamental_analysis, final_report, rating)
        VALUES (?,?,?,?,?,?,?)
    """, (result.get("ticker",""), result.get("data_fetch_success",False), result.get("error_message",""),
          result.get("technical_analysis",""), result.get("fundamental_analysis",""), result.get("final_report",""), rating or ""))
    conn.commit(); rid = c.lastrowid; conn.close()
    return rid


def get_results(username: str, ticker: Optional[str] = None, limit: int = 50) -> List[Dict]:
    path = _db_path(username)
    if not os.path.exists(path): return []
    conn = sqlite3.connect(path); conn.row_factory = sqlite3.Row
    if ticker:
        rows = conn.execute("SELECT * FROM analysis_results WHERE ticker=? ORDER BY created_at DESC LIMIT ?", (ticker, limit)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM analysis_results ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_summary(username: str) -> Dict:
    path = _db_path(username)
    if not os.path.exists(path): return {"total": 0, "success": 0, "failed": 0, "ratings": {}}
    conn = sqlite3.connect(path)
    total = conn.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0]
    success = conn.execute("SELECT COUNT(*) FROM analysis_results WHERE fetch_success=1").fetchone()[0]
    rows = conn.execute("SELECT rating, COUNT(*) FROM analysis_results WHERE rating!='' GROUP BY rating ORDER BY COUNT(*) DESC").fetchall()
    conn.close()
    return {"total": total, "success": success, "failed": total - success, "ratings": {r[0]: r[1] for r in rows}}


def extract_rating(final_report: str) -> Optional[str]:
    m = _RATING_PATTERN.search(final_report)
    return m.group(1) if m else None


# ========== 用户持仓 ==========

def get_watchlist(username: str, list_name: str = "默认池") -> List[str]:
    path = _db_path(username)
    if not os.path.exists(path): return []
    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT code FROM watchlists WHERE name=? ORDER BY added_at", (list_name,)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def add_to_watchlist(username: str, list_name: str, codes: List[str]) -> int:
    conn = _ensure_tables(username)
    added = 0
    for code in codes:
        code = code.strip().zfill(6)
        if not code.isdigit() or len(code) != 6: continue
        ex = conn.execute("SELECT 1 FROM watchlists WHERE name=? AND code=?", (list_name, code)).fetchone()
        if not ex:
            conn.execute("INSERT INTO watchlists (name,code) VALUES (?,?)", (list_name, code))
            added += 1
    conn.commit(); conn.close()
    return added


def remove_from_watchlist(username: str, list_name: str, codes: List[str]) -> int:
    conn = _ensure_tables(username)
    removed = 0
    for code in codes:
        c = conn.execute("DELETE FROM watchlists WHERE name=? AND code=?", (list_name, code.strip().zfill(6))).rowcount
        removed += c
    conn.commit(); conn.close()
    return removed


def get_watchlist_names(username: str) -> List[str]:
    path = _db_path(username)
    if not os.path.exists(path): return []
    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT DISTINCT name FROM watchlists ORDER BY name").fetchall()
    conn.close()
    return [r[0] for r in rows]
