"""
教学引擎 — 每日自动案例 + 白话解释 + 动态 Tooltip。
新手学堂的核心知识输出层。
"""

import datetime
from typing import Optional, Dict, List


def _fetch(ticker: str, days: int = 400):
    import requests as _r
    _o = _r.Session.__init__
    def _p(s, *a, **k): _o(s, *a, **k); s.trust_env = False
    _r.Session.__init__ = _p
    import akshare as ak
    import pandas as pd
    prefix = "sh" if ticker.startswith(("60", "68")) else "sz"
    end = datetime.datetime.now().strftime("%Y%m%d")
    start = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y%m%d")
    try:
        df = ak.stock_zh_a_hist_tx(symbol=prefix + ticker, start_date=start, end_date=end, adjust="qfq")
        if df.empty: return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ============================================
# 1. 每日教学案例自动扫描
# ============================================
def scan_daily_case() -> Optional[Dict]:
    """
    扫描沪深300成分股中最典型的今日形态，返回教学案例。
    优先级：金叉 > 底背离 > 布林缩口突破
    """
    candidates = [
        "600519", "000858", "600036", "601318", "000001", "300750",
        "601012", "600900", "000725", "002415", "600276", "601398",
        "601288", "600030", "000002", "601668", "002594", "002230",
        "002371", "300059", "600050", "601857", "600809", "000568",
        "000651", "002475", "688981", "300760", "600585", "000538",
    ]

    best = None
    for ticker in candidates:
        df = _fetch(ticker, days=200)
        if df.empty or len(df) < 60:
            continue

        # 计算指标
        df["MA5"] = df["close"].rolling(5).mean()
        df["MA10"] = df["close"].rolling(10).mean()
        df["MA20"] = df["close"].rolling(20).mean()

        # 金叉检测
        golden = (df["MA5"] > df["MA10"]) & (df["MA5"].shift(1) <= df["MA10"].shift(1))
        if golden.any():
            idx = df[golden].index[-1]
            date = df.loc[idx, "date"]
            days_ago = (df["date"].iloc[-1] - date).days
            if days_ago <= 2:
                info_q = _get_name(ticker)
                return {
                    "ticker": ticker,
                    "name": info_q.get("name", ticker),
                    "type": "金叉信号",
                    "date": str(date.date()),
                    "lesson": "均线金叉",
                    "explanation": f"今日{info_q.get('name',ticker)}的5日均线上穿10日均线，形成经典的金叉信号。金叉通常被视为短线买入时机——短期趋势开始强于中期趋势。但注意：单靠金叉不够，需结合成交量确认。",
                    "indicator": "MA5↑MA10",
                }

        # 底背离检测（价格新低 + RSI 未新低）
        df["RSI"] = _calc_rsi(df["close"])
        if len(df) >= 40:
            recent_20 = df.tail(20)
            prior_20 = df.tail(40).head(20)
            price_new_low = recent_20["low"].min() < prior_20["low"].min()
            rsi_new_low = recent_20["RSI"].min() < prior_20["RSI"].min()
            if price_new_low and not rsi_new_low:
                info_q = _get_name(ticker)
                return {
                    "ticker": ticker,
                    "name": info_q.get("name", ticker),
                    "type": "底背离",
                    "date": str(df["date"].iloc[-1].date()),
                    "lesson": "RSI底背离",
                    "explanation": f"{info_q.get('name',ticker)}近期价格创了新低，但RSI指标却没有跟随创新低——这是经典的底背离信号。说明虽然价格在跌，但下跌动能正在衰竭，可能接近底部。这是左侧交易者关注的信号。",
                    "indicator": "RSI底背离",
                }

    return best


def _get_name(ticker: str) -> Dict:
    try:
        from real_time import get_realtime_quotes
        q = get_realtime_quotes([ticker])
        if q and q[0].get("name"): return {"name": q[0]["name"]}
    except Exception:
        pass
    return {"name": ticker}


def _calc_rsi(close, period=14):
    import pandas as pd
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


# ============================================
# 2. 基本面白话翻译
# ============================================
def explain_fundamentals(fd: Dict) -> Dict[str, str]:
    """将基本面数据翻译成口语化的教学解释。"""
    explanations = {}
    pe = fd.get("pe", 0)
    if pe > 0:
        if pe < 15:
            explanations["PE"] = f"当前PE={pe:.1f}，属于低估区间。按现在的盈利能力，大约{pe:.0f}年能通过利润收回你的投资成本，在同行业中算便宜的。"
        elif pe < 30:
            explanations["PE"] = f"当前PE={pe:.1f}，处于合理区间。不算贵也不算便宜，市场给了一个中规中矩的估值。"
        else:
            explanations["PE"] = f"当前PE={pe:.1f}，偏高。市场愿意给这么高的估值，说明大家对它未来的增长有较高预期。但注意：高PE也意味着一旦业绩不达预期，股价容易大跌。"

    cap = fd.get("market_cap", 0)
    if cap > 0:
        if cap > 1000:
            explanations["市值"] = f"总市值{cap:,.0f}亿元，属于大盘蓝筹股。盘子大，波动小，适合稳健型投资者。"
        elif cap > 100:
            explanations["市值"] = f"总市值{cap:,.0f}亿元，属于中盘股。有一定成长空间，波动比大盘股大一些。"
        else:
            explanations["市值"] = f"总市值{cap:,.0f}亿元，属于小盘股。弹性大、波动大，风险和收益都相对较高。"

    roe = fd.get("roe", 0)
    if roe > 0:
        if roe > 15:
            explanations["ROE"] = f"ROE={roe:.1f}%，非常优秀。说明公司每100元净资产能赚{roe:.0f}元，赚钱效率高。巴菲特选股的核心指标之一。"
        elif roe > 8:
            explanations["ROE"] = f"ROE={roe:.1f}%，中等水平。公司赚钱效率尚可，但不算突出。"
        else:
            explanations["ROE"] = f"ROE={roe:.1f}%，偏低。公司用股东的每一块钱赚得不多，需要关注盈利能力。"

    return explanations


# ============================================
# 3. 动态 Tooltip（带实时数据）
# ============================================
def tooltip_macd(ticker: str) -> str:
    """MACD 动态解释——带上当前股票的实时状态。"""
    df = _fetch(ticker, days=200)
    if df.empty or len(df) < 30:
        return _tooltips["MACD"]
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    dif = ema12.iloc[-1] - ema26.iloc[-1]
    dea = (ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1]
    position = "金叉区域（多头占优）" if dif > dea else "死叉区域（空头占优）"
    return f"**MACD** = 指数平滑异同移动平均线。DIF(快线)上穿DEA(慢线)=金叉买入信号。\n\n📊 **{ticker} 当前状态**: DIF={dif:.3f} DEA={dea:.3f} → {position}。"


def tooltip_kdj(ticker: str) -> str:
    """KDJ 动态解释。"""
    df = _fetch(ticker, days=100)
    if df.empty or len(df) < 20:
        return _tooltips["KDJ"]
    low_n = df["low"].rolling(9).min()
    high_n = df["high"].rolling(9).max()
    rsv = (df["close"] - low_n) / (high_n - low_n) * 100
    k = rsv.ewm(com=2, adjust=False).mean().iloc[-1]
    d = pd.Series(rsv.ewm(com=2, adjust=False).mean()).ewm(com=2, adjust=False).mean().iloc[-1]
    j = 3 * k - 2 * d
    if k > 80:
        state = "超买区（K>80，短期过热，注意回调风险）"
    elif k < 20:
        state = "超卖区（K<20，短期超跌，可能有反弹机会）"
    else:
        state = "中性区域"
    return f"**KDJ** = 随机指标。K线>80=超买、K线<20=超卖。J线最灵敏，常先行拐头。\n\n📊 **{ticker} 当前**: K={k:.1f} D={d:.1f} J={j:.1f} → {state}。"


def tooltip_rsi(ticker: str) -> str:
    """RSI 动态解释。"""
    df = _fetch(ticker, days=100)
    if df.empty:
        return _tooltips["RSI"]
    rsi = _calc_rsi(df["close"]).iloc[-1]
    if rsi > 70:
        state = "超买（RSI>70，短期涨太多了，有回调压力）"
    elif rsi < 30:
        state = "超卖（RSI<30，短期跌太多了，有反弹机会）"
    else:
        state = "中性（30-70之间，正常波动）"
    return f"**RSI** = 相对强弱指标，0-100之间。>70=超买、<30=超卖。\n\n📊 **{ticker} 当前**: RSI={rsi:.1f} → {state}。"


# 静态 Tooltip（无实时数据时使用）
_tooltips = {
    "PE": "**PE(市盈率)** = 股价 ÷ 每股收益。\n简单理解：你花PE块钱买1块钱的利润。PE越低越便宜，但不同行业差别很大（银行5-15，科技30-60）。",
    "市值": "**总市值** = 股价 × 总股本。\n代表市场认为这家公司值多少钱。市值越大，股价越稳定（波动越小）。",
    "ROE": "**ROE(净资产收益率)** = 净利润 ÷ 净资产。\n衡量公司用股东的钱赚钱的效率。ROE>15%算优秀，巴菲特最看重的指标之一。",
    "MACD": "**MACD** = 指数平滑异同移动平均线。\nDIF(快线)上穿DEA(慢线)=买入信号。DIF下穿DEA=卖出信号。柱状图=两线差值，柱越长趋势越强。",
    "KDJ": "**KDJ** = 随机指标，0-100之间摆动。\nK线>80=超买(慎买)、K线<20=超卖(机会)。J线最灵敏，常先于K/D拐头。",
    "RSI": "**RSI** = 相对强弱指标，0-100之间。\nRSI>70=超买(短期涨太多了)、RSI<30=超卖(短期跌太多了)。50线是牛熊分界。",
    "BOLL": "**BOLL(布林带)** = 中轨(MA20) ± 2倍标准差。\n价格触及上轨=压力位，触及下轨=支撑位。带宽缩窄=即将变盘。",
    "MA": "**MA(移动平均线)** = 最近N天的平均收盘价。\nMA5=周线、MA20=月线、MA60=季线。短期上穿长期=金叉(买)，下穿=死叉(卖)。",
    "金叉": "**金叉** = 短期均线上穿长期均线。\n例如MA5上穿MA10，表示短期趋势开始强于中期趋势，是常见的买入信号。",
    "死叉": "**死叉** = 短期均线下穿长期均线。\n与金叉相反，是常见卖出信号。金叉死叉不是100%准确，需配合成交量和其他指标。",
    "换手率": "**换手率** = 当日成交量 ÷ 流通股本。\n代表股票活跃程度。换手率<1%=冷门，1-3%=正常，>5%=热门，>10%=异常活跃。",
    "夏普比率": "**夏普比率** = (收益率-无风险利率) ÷ 波动率。\n衡量每承担1单位风险能获得多少超额回报。>1=优秀，>2=极好，<0=不如存银行。",
}

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
