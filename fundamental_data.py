"""
基本面数据模块 — 腾讯实时行情中提取 PE/市值/52周高低等基础面数据。
"""

import re
from typing import Optional, Dict, List

import requests


def _get_tx_session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com"})
    return s


def get_fundamentals(tickers: List[str]) -> Dict[str, Dict]:
    """
    批量获取基本面数据：PE、市值、换手率、52周高低、阶段涨跌幅。

    Args:
        tickers: 股票代码列表

    Returns:
        {code: {pe, market_cap, ...}, ...}
    """
    if not tickers:
        return {}

    # 构建腾讯查询 URL（一次最多约 50 支）
    codes = []
    for t in tickers:
        prefix = "sh" if t.startswith(("60", "68")) else "sz"
        codes.append(f"{prefix}{t}")

    result = {}
    batch_size = 50

    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        url = "http://qt.gtimg.cn/q=" + ",".join(batch)

        try:
            s = _get_tx_session()
            r = s.get(url, timeout=10)
            r.encoding = "gbk"
            for line in r.text.strip().split("\n"):
                if not line.strip() or "=" not in line:
                    continue
                raw = line.split('"')[1] if '"' in line else ""
                if not raw:
                    continue
                parts = raw.split("~")
                if len(parts) < 50:
                    continue

                code = parts[2] if len(parts) > 2 else ""
                name = parts[1] if len(parts) > 1 else ""

                def _f(idx, default=""):
                    return parts[idx] if len(parts) > idx else default

                def _float(idx, default=0.0):
                    try:
                        return float(_f(idx))
                    except (ValueError, TypeError):
                        return default

                result[code] = {
                    "name": name,
                    "price": _float(3),
                    "change_pct": _float(32),
                    "pe": _float(39),               # 市盈率(动态)
                    "market_cap": _float(44),        # 总市值(亿)
                    "circ_market_cap": _float(45),   # 流通市值(亿)
                    "turnover_rate": _float(46),     # 换手率(%)
                    "high_52w": _float(47),          # 52周最高
                    "low_52w": _float(48),           # 52周最低
                    "amplitude": _float(49),         # 振幅(%)
                    "ytd_change": _float(62),        # 年初至今涨跌%
                    "q_change": _float(63),          # 近一季%
                    "hy_change": _float(64),         # 近半年%
                    "y1_change": _float(65),         # 近一年%
                    "y2_change": _float(66),         # 近两年%
                    "history_high": _float(67),      # 历史最高
                    "history_low": _float(68),       # 历史最低
                    "roe": _float(69),               # ROE(%)
                    "pb": 0.0,                       # PB 需另外计算或查财报
                }
        except Exception:
            continue

    return result


def get_single_fundamentals(ticker: str) -> Optional[Dict]:
    """获取单支股票基本面。"""
    result = get_fundamentals([ticker])
    return result.get(ticker)
