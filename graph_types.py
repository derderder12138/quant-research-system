"""
全局共享类型定义 —— 所有模块引用的唯一 State 类型来源。
从 test_env.py 提取，字段保持不变，确保向后兼容。
"""

from typing import TypedDict


class StockAgentState(TypedDict):
    """LangGraph 全局状态 —— 贯穿数据获取 → 双 Agent 分析 → CIO 终审的全链路"""
    ticker: str                     # A 股股票代码（如 "600519"）
    data_fetch_success: bool        # AKShare 数据获取是否成功
    error_message: str              # 失败时的错误信息
    raw_history_data: dict          # 原始行情数据（近 5 日 OHLCV）
    technical_analysis: str         # 技术面 Agent 输出
    fundamental_analysis: str       # 基本面 Agent 输出
    final_report: str               # CIO 最终综合决策报告
