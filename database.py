"""
数据库持久化模块 — 多用户支持 · 本地 SQLite + 云端 SQLite Cloud 自动切换。
部署到 Streamlit Cloud 时，在 Secrets 中设置 SQLITE_CLOUD_URL 即可自动使用云数据库。
"""

import os, hashlib, re
from typing import Optional, List, Dict, Any, Union

# 尝试导入 SQLite Cloud SDK
try:
    import sqlitecloud as _sqlitecloud
    _HAS_CLOUD = True
except ImportError:
    _HAS_CLOUD = False

# 标准库 sqlite3 始终可用
import sqlite3 as _sqlite3

_RATING_PATTERN = re.compile(r"\[(积极建仓|谨慎持有|观望等待|减仓回避)\]")

# 数据目录：优先用项目 data/，不可用时回退 /tmp
_DB_DIR = None
_CLOUD_URL = None


def _get_cloud_url() -> Optional[str]:
    """获取 SQLite Cloud 连接字符串（从环境变量或 Streamlit Secrets）。"""
    global _CLOUD_URL
    if _CLOUD_URL is not None:
        return _CLOUD_URL if _CLOUD_URL else None
    # 先查环境变量
    url = os.environ.get("SQLITE_CLOUD_URL", "")
    # 再查 Streamlit Secrets
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("SQLITE_CLOUD_URL", "")
        except Exception:
            pass
    _CLOUD_URL = url
    return url if url else None


def _is_cloud() -> bool:
    return _HAS_CLOUD and bool(_get_cloud_url())


def _connect(path: str = ":memory:") -> Any:
    """创建数据库连接。云端模式忽略 path，使用 SQLite Cloud。"""
    if _is_cloud():
        conn = _sqlitecloud.connect(_get_cloud_url())
        conn.execute("USE DATABASE quant_system")
        return conn
    return _sqlite3.connect(path)


def _get_db_dir() -> str:
    global _DB_DIR
    if _DB_DIR is not None:
        return _DB_DIR
    # 先尝试项目目录
    try:
        base = os.path.abspath(os.path.dirname(__file__))
    except Exception:
        base = "/tmp"
    d = os.path.join(base, "data")
    try:
        os.makedirs(d, exist_ok=True)
        # 写测试
        test = os.path.join(d, ".w")
        with open(test, "w") as f:
            f.write("1")
        os.remove(test)
        _DB_DIR = d
    except Exception:
        _DB_DIR = "/tmp/quant_data"
        os.makedirs(_DB_DIR, exist_ok=True)
    return _DB_DIR


def _db_path(username: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_一-鿿]", "_", username)
    return os.path.join(_get_db_dir(), f"user_{safe}.db")


def _ensure_tables(username: str):  # -> sqlite3.Connection
    path = _db_path(username)
    conn = _connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL,
        fetch_success INTEGER DEFAULT 0, error_message TEXT DEFAULT '',
        technical_analysis TEXT DEFAULT '', fundamental_analysis TEXT DEFAULT '',
        final_report TEXT DEFAULT '', rating TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now','localtime')))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS watchlists (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT DEFAULT '默认池',
        code TEXT NOT NULL, added_at TEXT DEFAULT (datetime('now','localtime')))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS user_info (
        username TEXT PRIMARY KEY, password_hash TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now','localtime')))""")
    conn.commit()
    return conn


# ========== 用户管理 ==========
def _hash(pw: str) -> str:
    return hashlib.sha256(f"qs_{pw}".encode()).hexdigest()


def verify_user(username: str, password: str) -> bool:
    conn = _ensure_tables(username)
    row = conn.execute("SELECT password_hash FROM user_info WHERE username=?", (username,)).fetchone()
    if row:
        ok = row[0] == _hash(password)
        conn.close()
        return ok
    conn.execute("INSERT INTO user_info(username,password_hash) VALUES (?,?)", (username, _hash(password)))
    conn.commit(); conn.close()
    return True


# ========== 分析结果 ==========
def init_db(db_path: str = "") -> None:
    pass


def save_result(username: str, result: Dict[str, Any]) -> int:
    conn = _ensure_tables(username)
    rating = ""
    if result.get("data_fetch_success") and result.get("final_report"):
        rating = extract_rating(result["final_report"]) or ""
    elif not result.get("data_fetch_success"):
        rating = "数据失败"
    c = conn.execute("""INSERT INTO analysis_results
        (ticker, fetch_success, error_message, technical_analysis, fundamental_analysis, final_report, rating, created_at)
        VALUES (?,?,?,?,?,?,?,datetime('now','localtime'))""",
        (result.get("ticker",""), 1 if result.get("data_fetch_success") else 0,
         result.get("error_message",""), result.get("technical_analysis",""),
         result.get("fundamental_analysis",""), result.get("final_report",""), rating))
    conn.commit(); rid = c.lastrowid; conn.close()
    return rid


def get_results(username: str, ticker: Optional[str] = None, limit: int = 50) -> List[Dict]:
    path = _db_path(username)
    if not _is_cloud() and not os.path.exists(path):
        return []
    conn = _connect(path)
    if ticker:
        rows = conn.execute("SELECT * FROM analysis_results WHERE ticker=? ORDER BY created_at DESC LIMIT ?", (ticker, limit)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM analysis_results ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    # 手动构造 dict，不依赖 row_factory（避免云端兼容问题）
    cols = ["id","ticker","fetch_success","error_message","technical_analysis","fundamental_analysis","final_report","rating","created_at"]
    results = []
    for row in rows:
        d = {}
        for i, col in enumerate(cols):
            d[col] = row[i] if i < len(row) else None
        results.append(d)
    conn.close()
    return results


def get_summary(username: str) -> Dict:
    path = _db_path(username)
    if not _is_cloud() and not os.path.exists(path):
        return {"total": 0, "success": 0, "failed": 0, "ratings": {}}
    conn = _connect(path)
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
    if not _is_cloud() and not os.path.exists(path):
        return []
    conn = _connect(path)
    rows = conn.execute("SELECT code FROM watchlists WHERE name=? ORDER BY added_at", (list_name,)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def add_to_watchlist(username: str, list_name: str, codes: List[str]) -> int:
    conn = _ensure_tables(username)
    added = 0
    for code in codes:
        code = code.strip().zfill(6)
        if not code.isdigit() or len(code) != 6:
            continue
        if not conn.execute("SELECT 1 FROM watchlists WHERE name=? AND code=?", (list_name, code)).fetchone():
            conn.execute("INSERT INTO watchlists (name,code) VALUES (?,?)", (list_name, code))
            added += 1
    conn.commit(); conn.close()
    return added


def remove_from_watchlist(username: str, list_name: str, codes: List[str]) -> int:
    conn = _ensure_tables(username)
    removed = 0
    for code in codes:
        removed += conn.execute("DELETE FROM watchlists WHERE name=? AND code=?", (list_name, code.strip().zfill(6))).rowcount
    conn.commit(); conn.close()
    return removed


# ========== 持仓交易记录 ==========
def save_position(username: str, ticker: str, shares: int, cost: float) -> None:
    conn = _ensure_tables(username)
    conn.execute("""CREATE TABLE IF NOT EXISTS positions
        (ticker TEXT PRIMARY KEY, shares INTEGER DEFAULT 0, cost REAL DEFAULT 0.0,
         updated_at TEXT DEFAULT (datetime('now','localtime')))""")
    conn.execute("INSERT OR REPLACE INTO positions (ticker, shares, cost, updated_at) VALUES (?,?,?,datetime('now','localtime'))",
                 (ticker, shares, cost))
    conn.commit(); conn.close()


def get_position(username: str, ticker: str) -> Dict:
    path = _db_path(username)
    if not os.path.exists(path): return {"shares": 0, "cost": 0.0}
    conn = _connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS positions
        (ticker TEXT PRIMARY KEY, shares INTEGER DEFAULT 0, cost REAL DEFAULT 0.0,
         updated_at TEXT DEFAULT (datetime('now','localtime')))""")
    row = conn.execute("SELECT shares, cost FROM positions WHERE ticker=?", (ticker,)).fetchone()
    conn.close()
    return {"shares": row[0], "cost": row[1]} if row else {"shares": 0, "cost": 0.0}


def get_all_positions(username: str) -> Dict[str, Dict]:
    path = _db_path(username)
    if not os.path.exists(path): return {}
    conn = _connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS positions
        (ticker TEXT PRIMARY KEY, shares INTEGER DEFAULT 0, cost REAL DEFAULT 0.0,
         updated_at TEXT DEFAULT (datetime('now','localtime')))""")
    rows = conn.execute("SELECT ticker, shares, cost FROM positions").fetchall()
    conn.close()
    return {r[0]: {"shares": r[1], "cost": r[2]} for r in rows}


# ========== 价格预警 ==========
def save_alert(username: str, ticker: str, price: float, direction: str) -> None:
    conn = _ensure_tables(username)
    conn.execute("""CREATE TABLE IF NOT EXISTS alerts
        (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL, price REAL NOT NULL,
         direction TEXT NOT NULL, active INTEGER DEFAULT 1,
         created_at TEXT DEFAULT (datetime('now','localtime')))""")
    conn.execute("INSERT INTO alerts (ticker, price, direction) VALUES (?,?,?)", (ticker, price, direction))
    conn.commit(); conn.close()


def get_alerts(username: str) -> List[Dict]:
    path = _db_path(username)
    if not os.path.exists(path): return []
    conn = _connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS alerts
        (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL, price REAL NOT NULL,
         direction TEXT NOT NULL, active INTEGER DEFAULT 1,
         created_at TEXT DEFAULT (datetime('now','localtime')))""")
    rows = conn.execute("SELECT ticker, price, direction, active FROM alerts WHERE active=1").fetchall()
    conn.close()
    return [{"ticker": r[0], "price": r[1], "direction": r[2], "active": r[3]} for r in rows]


def delete_alert(username: str, alert_id: int) -> None:
    conn = _ensure_tables(username)
    conn.execute("UPDATE alerts SET active=0 WHERE id=?", (alert_id,))
    conn.commit(); conn.close()


# ========== 股票笔记 ==========
def save_note(username: str, ticker: str, note: str) -> None:
    conn = _ensure_tables(username)
    conn.execute("""CREATE TABLE IF NOT EXISTS stock_notes
        (ticker TEXT PRIMARY KEY, note TEXT DEFAULT '', updated_at TEXT DEFAULT (datetime('now','localtime')))""")
    conn.execute("INSERT OR REPLACE INTO stock_notes (ticker, note, updated_at) VALUES (?,?,datetime('now','localtime'))",
                 (ticker, note))
    conn.commit(); conn.close()


def get_note(username: str, ticker: str) -> str:
    path = _db_path(username)
    if not os.path.exists(path): return ""
    conn = _connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS stock_notes
        (ticker TEXT PRIMARY KEY, note TEXT DEFAULT '', updated_at TEXT DEFAULT (datetime('now','localtime')))""")
    row = conn.execute("SELECT note FROM stock_notes WHERE ticker=?", (ticker,)).fetchone()
    conn.close()
    return row[0] if row else ""


def get_all_notes(username: str) -> Dict[str, str]:
    path = _db_path(username)
    if not os.path.exists(path): return {}
    conn = _connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS stock_notes
        (ticker TEXT PRIMARY KEY, note TEXT DEFAULT '', updated_at TEXT DEFAULT (datetime('now','localtime')))""")
    rows = conn.execute("SELECT ticker, note FROM stock_notes").fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def get_watchlist_names(username: str) -> List[str]:
    path = _db_path(username)
    if not _is_cloud() and not os.path.exists(path):
        return []
    conn = _connect(path)
    rows = conn.execute("SELECT DISTINCT name FROM watchlists ORDER BY name").fetchall()
    conn.close()
    return [r[0] for r in rows]
