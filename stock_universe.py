"""
全市场股票代码表 — AKShare 全量 A 股代码 → SQLite 缓存。
提供模糊搜索、分页查询、缓存刷新。首次运行自动从 AKShare 拉取。
"""

import os
import sqlite3
from typing import List, Dict, Optional, Tuple


UNIVERSE_DB = os.path.join(os.path.dirname(__file__), "data", "stock_universe.db")


def _ensure_db() -> None:
    """确保 stock_universe 表存在。"""
    db_dir = os.path.dirname(UNIVERSE_DB)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(UNIVERSE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_list (
            code TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            market TEXT NOT NULL,     -- SH / SZ
            board TEXT NOT NULL,      -- 主板/创业板/科创板/中小板
            updated_at TEXT DEFAULT (datetime('now'))
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
    conn.commit()
    conn.close()


def _classify_board(code: str) -> str:
    """根据代码推断所属板块。"""
    if code.startswith("688"):
        return "科创板"
    if code.startswith("300") or code.startswith("301"):
        return "创业板"
    if code.startswith("002") or code.startswith("003") or code.startswith("001"):
        return "中小板"
    if code.startswith(("60", "68")):
        return "主板(沪)"
    return "主板(深)"


def _classify_market(code: str) -> str:
    return "SH" if code.startswith(("60", "68")) else "SZ"


def refresh_universe(force: bool = False) -> int:
    """
    从 AKShare 拉取全量 A 股代码表并缓存至 SQLite。
    如果已缓存且 force=False，跳过刷新。
    返回总股票数。
    """
    _ensure_db()
    conn = sqlite3.connect(UNIVERSE_DB)
    count = conn.execute("SELECT COUNT(*) FROM stock_list").fetchone()[0]
    if count > 0 and not force:
        conn.close()
        return count

    print(f"[stock_universe] 正在从 AKShare 获取全量 A 股代码表...")
    import requests as _r
    _o = _r.Session.__init__
    def _p(s, *a, **k): _o(s, *a, **k); s.trust_env = False
    _r.Session.__init__ = _p

    import akshare as ak
    try:
        df = ak.stock_info_a_code_name()
    except Exception as e:
        print(f"[stock_universe] 获取代码表失败: {e}")
        conn.close()
        return 0

    conn.execute("DELETE FROM stock_list")
    for _, row in df.iterrows():
        code = str(row["code"]).zfill(6)
        name = row["name"]
        conn.execute(
            "INSERT OR REPLACE INTO stock_list (code, name, market, board) VALUES (?,?,?,?)",
            (code, name, _classify_market(code), _classify_board(code)),
        )
    conn.commit()
    conn.close()
    print(f"[stock_universe] 已缓存 {len(df)} 支股票。")
    return len(df)


def search_stocks(query: str, limit: int = 50) -> List[Dict]:
    """
    模糊搜索股票（代码或名称）。
    query: 搜索关键词（支持拼音首字母、代码片段、名称关键词）
    返回匹配的股票列表。
    """
    _ensure_db()
    conn = sqlite3.connect(UNIVERSE_DB)
    conn.row_factory = sqlite3.Row

    # 纯数字 → 优先代码匹配
    if query.isdigit():
        cursor = conn.execute(
            "SELECT * FROM stock_list WHERE code LIKE ? LIMIT ?",
            (f"%{query}%", limit),
        )
    else:
        cursor = conn.execute(
            "SELECT * FROM stock_list WHERE name LIKE ? LIMIT ?",
            (f"%{query}%", limit),
        )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_stock_info(code: str) -> Optional[Dict]:
    """获取单支股票基本信息。"""
    _ensure_db()
    conn = sqlite3.connect(UNIVERSE_DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM stock_list WHERE code = ?", (code,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_by_board(board: str, limit: int = 100) -> List[Dict]:
    """按板块筛选（创业板/科创板/主板(沪)/主板(深)/中小板）。"""
    _ensure_db()
    conn = sqlite3.connect(UNIVERSE_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM stock_list WHERE board = ? LIMIT ?",
        (board, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_watchlist(list_name: str = "默认池") -> List[str]:
    """获取自定义股票池的代码列表。"""
    _ensure_db()
    conn = sqlite3.connect(UNIVERSE_DB)
    rows = conn.execute(
        "SELECT code FROM watchlists WHERE name = ? ORDER BY added_at",
        (list_name,),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def add_to_watchlist(list_name: str, codes: List[str]) -> int:
    """向自定义股票池添加代码（去重）。"""
    _ensure_db()
    conn = sqlite3.connect(UNIVERSE_DB)
    added = 0
    for code in codes:
        code = code.strip().zfill(6)
        if not code.isdigit() or len(code) != 6:
            continue
        exists = conn.execute(
            "SELECT 1 FROM watchlists WHERE name=? AND code=?", (list_name, code)
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO watchlists (name, code) VALUES (?,?)",
                (list_name, code),
            )
            added += 1
    conn.commit()
    conn.close()
    return added


def remove_from_watchlist(list_name: str, codes: List[str]) -> int:
    """从自定义股票池移除代码。"""
    _ensure_db()
    conn = sqlite3.connect(UNIVERSE_DB)
    removed = 0
    for code in codes:
        code = code.strip().zfill(6)
        c = conn.execute(
            "DELETE FROM watchlists WHERE name=? AND code=?", (list_name, code)
        ).rowcount
        removed += c
    conn.commit()
    conn.close()
    return removed


def get_watchlist_names() -> List[str]:
    """获取所有自定义股票池名称。"""
    _ensure_db()
    conn = sqlite3.connect(UNIVERSE_DB)
    rows = conn.execute(
        "SELECT DISTINCT name FROM watchlists ORDER BY name"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_universe_stats() -> Dict:
    """获取全市场统计信息。"""
    _ensure_db()
    conn = sqlite3.connect(UNIVERSE_DB)
    total = conn.execute("SELECT COUNT(*) FROM stock_list").fetchone()[0]
    boards = {}
    for row in conn.execute("SELECT board, COUNT(*) FROM stock_list GROUP BY board"):
        boards[row[0]] = row[1]
    conn.close()
    return {"total": total, "boards": boards}
