"""
数据获取模块 —— AKShare 真实 A 股数据拉取（腾讯数据源）+ 代理屏蔽 + 重试容灾。
从 test_env.py 的 data_fetcher_node 提取核心逻辑，保持功能完全等价。
"""

import os
import time
import datetime
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# 模块级可配置参数，由 batch_runner 通过 configure_fetcher() 注入
_retry_attempts = 3
_retry_delay = 2

# ==== 根因修复：requests 在 Windows 上默认从注册表读取系统代理，
#     必须在导入 akshare 之前关闭 trust_env。 ====
import requests as _requests

_original_init = _requests.Session.__init__


def _patched_session_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.trust_env = False


_requests.Session.__init__ = _patched_session_init  # type: ignore[method-assign]

# ==== 辅助兜底：清空代理环境变量 ====
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

import akshare as ak


def configure_fetcher(retry_attempts: int = 3, retry_delay: int = 2) -> None:
    """供 batch_runner 调用的配置注入函数。"""
    global _retry_attempts, _retry_delay
    _retry_attempts = retry_attempts
    _retry_delay = retry_delay


def fetch_stock_data(
    ticker: str,
    retry_attempts: int = 3,
    retry_delay: int = 2
) -> Dict[str, Any]:
    """
    通过 AKShare（腾讯数据源）获取 A 股近 30 日行情数据，提取最近 5 个交易日的 OHLCV。

    Args:
        ticker: A 股代码（如 "600519"）
        retry_attempts: 最大重试次数
        retry_delay: 重试间隔（秒）

    Returns:
        {
            "data_fetch_success": bool,
            "raw_history_data": dict,      # 近 5 日数据，key 为日期
            "error_message": str
        }
    """
    # 根据股票代码自动添加 sh/sz 前缀（腾讯接口要求）
    if ticker.startswith(("60", "68")):
        tx_symbol = "sh" + ticker
    else:
        tx_symbol = "sz" + ticker

    for attempt in range(retry_attempts):
        try:
            end_date = datetime.datetime.now().strftime("%Y%m%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y%m%d")

            # 使用腾讯数据源（东方财富源在某些网络环境被封）
            df = ak.stock_zh_a_hist_tx(
                symbol=tx_symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=""
            )

            if df.empty:
                raise ValueError(f"未找到代码 {ticker} 的有效数据。")

            # 取最近 5 个交易日，列名映射为中文（兼容下游）
            df = df.tail(5)[["date", "open", "high", "low", "close", "volume"]]
            df.rename(columns={  # type: ignore[reportCallIssue]
                "date": "日期", "open": "开盘", "high": "最高",
                "low": "最低", "close": "收盘", "volume": "成交量"
            }, inplace=True)
            df.set_index("日期", inplace=True)

            return {
                "data_fetch_success": True,
                "raw_history_data": df.to_dict(orient="index"),  # type: ignore[reportCallIssue]
                "error_message": ""
            }

        except Exception as e:
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
            else:
                return {
                    "data_fetch_success": False,
                    "raw_history_data": {},
                    "error_message": str(e)
                }

    # 防御性编程
    return {
        "data_fetch_success": False,
        "raw_history_data": {},
        "error_message": "未知错误：重试循环异常退出"
    }


def data_fetcher_node(state: dict) -> dict:
    """
    LangGraph 节点封装 —— 供 graph_builder 直接注册使用。
    从 state 中读取 ticker，返回 state 更新字典。
    """
    ticker = state["ticker"]
    result = fetch_stock_data(ticker, retry_attempts=_retry_attempts, retry_delay=_retry_delay)

    status = "成功" if result["data_fetch_success"] else "失败"
    logger.info(f"[{ticker}] 数据获取: {status}")

    return result
