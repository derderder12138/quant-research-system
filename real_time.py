"""
实时行情模块 —— 新浪财经实时数据源 + 全市场搜索。
提供个股实时报价、市场概况、代码搜索、股票池管理。
"""

import re
import time
from typing import List, Dict, Optional, Tuple

import requests
from stock_universe import search_stocks, get_watchlist, add_to_watchlist, remove_from_watchlist, get_watchlist_names, get_universe_stats, refresh_universe  # noqa: E402


# 全局 Session（禁用代理，复用连接）
_SESSION: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.trust_env = False
        _SESSION.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://finance.sina.com.cn",
        })
    return _SESSION


def _sina_code(ticker: str) -> str:
    """将裸代码转为新浪前缀格式（sh600519 / sz000001）。"""
    return ("sh" if ticker.startswith(("60", "68")) else "sz") + ticker


def _parse_sina_line(line: str) -> Optional[Dict]:
    """
    解析单行新浪实时行情数据。
    返回 dict 或 None（解析失败时）。
    """
    m = re.match(r'var hq_str_(\w+)="(.+)"', line)
    if not m:
        return None
    code = m.group(1)
    fields = m.group(2).split(",")
    if len(fields) < 32:
        return None

    try:
        name = fields[0]
        open_p = float(fields[1]) if fields[1] else 0.0
        pre_close = float(fields[2]) if fields[2] else 0.0
        price = float(fields[3]) if fields[3] else 0.0
        high = float(fields[4]) if fields[4] else 0.0
        low = float(fields[5]) if fields[5] else 0.0
        volume = int(fields[8]) if fields[8] else 0        # 成交量(股)
        amount = float(fields[9]) if fields[9] else 0.0     # 成交额(元)
        change = price - pre_close
        change_pct = (change / pre_close * 100) if pre_close != 0 else 0.0

        # 裸代码（去掉 sh/sz 前缀）
        bare_code = code[2:]

        return {
            "code": bare_code,
            "name": name,
            "price": price,
            "open": open_p,
            "high": high,
            "low": low,
            "pre_close": pre_close,
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": volume,
            "amount": amount,
            "timestamp": time.strftime("%H:%M:%S"),
        }
    except (ValueError, IndexError):
        return None


def get_quotes_batched(tickers: List[str], batch_size: int = 80) -> Dict[str, Dict]:
    """分批获取实时行情，返回 {code: quote_dict} 映射。自动拆分大批量请求。"""
    result = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        quotes = get_realtime_quotes(batch)
        for q in quotes:
            if q.get("price", 0) > 0:
                result[q["code"]] = q
    return result


def get_realtime_quotes(tickers: List[str]) -> List[Dict]:
    """
    批量获取实时行情（新浪源，单次最多 ~80 支）。
    传入裸代码列表如 ['600519','000001']，返回行情 dict 列表。
    """
    if not tickers:
        return []

    all_results = []
    batch_size = 50  # 新浪单次上限安全值

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        codes = ",".join(_sina_code(t) for t in batch)
        url = f"https://hq.sinajs.cn/list={codes}"

        try:
            s = _get_session()
            r = s.get(url, timeout=10)
            r.encoding = "gbk"
            for line in r.text.strip().split("\n"):
                parsed = _parse_sina_line(line)
                if parsed:
                    all_results.append(parsed)
        except Exception:
            # 单批失败不中断整体，用 None 占位
            for t in batch:
                all_results.append({
                    "code": t, "name": "", "price": 0, "change": 0,
                    "change_pct": 0, "volume": 0, "amount": 0,
                    "timestamp": "", "error": str(Exception),
                })

    return all_results


def get_single_quote(ticker: str) -> Optional[Dict]:
    """获取单支股票实时行情。"""
    results = get_realtime_quotes([ticker])
    return results[0] if results else None


def get_index_quotes() -> List[Dict]:
    """
    获取三大指数实时行情：上证指数、深证成指、创业板指。
    新浪指数代码: s_sh000001, s_sz399001, s_sz399006
    """
    # 指数映射
    index_map = {
        "s_sh000001": "上证指数",
        "s_sz399001": "深证成指",
        "s_sz399006": "创业板指",
    }
    codes = ",".join(index_map.keys())
    url = f"https://hq.sinajs.cn/list={codes}"

    results = []
    try:
        s = _get_session()
        r = s.get(url, timeout=10)
        r.encoding = "gbk"
        for line in r.text.strip().split("\n"):
            m = re.match(r'var hq_str_(\w+)="(.+)"', line)
            if not m:
                continue
            fields = m.group(2).split(",")
            if len(fields) < 4:
                continue
            try:
                name = index_map.get(m.group(1), m.group(1))
                price = float(fields[1])
                change = float(fields[2])
                change_pct = float(fields[3])
                results.append({
                    "code": m.group(1),
                    "name": name,
                    "price": price,
                    "change": change,
                    "change_pct": change_pct,
                })
            except (ValueError, IndexError):
                continue
    except Exception:
        pass

    return results


def validate_tickers(tickers: List[str]) -> Tuple[List[str], List[str]]:
    """
    批量验证股票代码是否真实有效。
    返回 (valid_list, invalid_list)。
    """
    quotes = get_realtime_quotes(tickers)
    found_codes = {q["code"] for q in quotes if q.get("name") and q.get("price", 0) > 0}
    valid = [t for t in tickers if t in found_codes]
    invalid = [t for t in tickers if t not in found_codes]
    return valid, invalid


def get_top_active(limit: int = 20) -> List[Dict]:
    """
    获取成交最活跃的股票（基于新浪行业/概念板块的活跃股）。
    注：新浪没有直接的"全市场成交量排名"API，此方法使用常用活跃股池
    结合实时成交量排序。如需更精确的全市场排名，建议接入 AKShare 分页查询。
    """
    # 预置热门活跃股池（覆盖各大行业龙头，均为高流动性标的）
    hot_pool = [
        # 金融
        "601318", "600036", "000001", "601398", "601288", "600030", "601166",
        # 白酒消费
        "600519", "000858", "000568",
        # 新能源
        "300750", "601012", "600900", "002594",
        # 科技
        "000725", "002415", "688981", "002230",
        # 医药
        "600276", "300760", "000538",
        # 地产基建
        "000002", "601668",
        # 汽车
        "002594", "000625", "601238",
        # 半导体
        "002371", "603986",
        # 有色
        "601899", "600547",
        # 农业
        "002714", "300498",
    ]

    # 去重
    hot_pool = list(dict.fromkeys(hot_pool))

    # 获取实时行情并排序
    quotes = get_realtime_quotes(hot_pool)
    valid = [q for q in quotes if q.get("volume", 0) > 0]
    valid.sort(key=lambda x: x.get("volume", 0), reverse=True)

    return valid[:limit]
