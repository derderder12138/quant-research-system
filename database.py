"""
SQLite 持久化模块 —— 分析结果自动落库、历史查询。
零外部依赖（Python 标准库 sqlite3）。
"""

import sqlite3
import re
import os
from typing import Optional, List, Dict, Any


# CIO 输出中的评级标签提取正则
_RATING_PATTERN = re.compile(r"\[(积极建仓|谨慎持有|观望等待|减仓回避)\]")


def extract_rating(final_report: str) -> Optional[str]:
    """从 CIO 最终报告中提取评级标签。"""
    match = _RATING_PATTERN.search(final_report)
    if match:
        return match.group(1)
    return None


def init_db(db_path: str) -> None:
    """
    初始化数据库：创建目录和表（若不存在）。
    幂等操作——多次调用不会重复创建。
    """
    # 确保目录存在
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
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
    conn.commit()
    conn.close()


def save_result(db_path: str, result: Dict[str, Any]) -> int:
    """
    将单支股票的分析结果插入数据库。

    Args:
        db_path: SQLite 文件路径
        result: LangGraph 最终状态字典，包含 ticker, data_fetch_success,
                error_message, technical_analysis, fundamental_analysis, final_report

    Returns:
        int: 新插入记录的行 ID
    """
    # 确保目录和表存在（防御性编程）
    init_db(db_path)

    rating = None
    if result.get("data_fetch_success") and result.get("final_report"):
        rating = extract_rating(result["final_report"])
    elif not result.get("data_fetch_success"):
        rating = "数据失败"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO analysis_results
            (ticker, fetch_success, error_message, technical_analysis,
             fundamental_analysis, final_report, rating)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        result.get("ticker", ""),
        result.get("data_fetch_success", False),
        result.get("error_message", ""),
        result.get("technical_analysis", ""),
        result.get("fundamental_analysis", ""),
        result.get("final_report", ""),
        rating or ""
    ))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_results(
    db_path: str,
    ticker: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    查询历史分析结果。

    Args:
        db_path: SQLite 文件路径
        ticker: 可选，筛选特定股票代码
        limit: 最大返回条数

    Returns:
        List[Dict]: 结果列表
    """
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if ticker:
        cursor.execute(
            "SELECT * FROM analysis_results WHERE ticker = ? ORDER BY created_at DESC LIMIT ?",
            (ticker, limit)
        )
    else:
        cursor.execute(
            "SELECT * FROM analysis_results ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_summary(db_path: str) -> Dict[str, Any]:
    """
    获取数据库汇总统计。

    Returns:
        Dict: {"total": int, "success": int, "failed": int, "ratings": {...}}
    """
    if not os.path.exists(db_path):
        return {"total": 0, "success": 0, "failed": 0, "ratings": {}}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM analysis_results")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM analysis_results WHERE fetch_success = 1")
    success = cursor.fetchone()[0]

    failed = total - success

    cursor.execute(
        "SELECT rating, COUNT(*) FROM analysis_results "
        "WHERE rating != '' GROUP BY rating ORDER BY COUNT(*) DESC"
    )
    ratings = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()
    return {
        "total": total,
        "success": success,
        "failed": failed,
        "ratings": ratings
    }
