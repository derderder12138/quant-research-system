"""
交易信号引擎 — 凯里公式 + 多条件买卖提示。
凯里仓位 / MA10黏着 / 25日止损 / 突破回踩 / 前复权历史高点 / 50%回撤位。
"""

import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


def _fetch_history(ticker: str, days: int) -> pd.DataFrame:
    import requests as _r
    _o = _r.Session.__init__
    def _p(s, *a, **k): _o(s, *a, **k); s.trust_env = False
    _r.Session.__init__ = _p
    import akshare as ak
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
# 1. 凯里公式
# ============================================
def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> Dict:
    """
    凯里公式: f* = (bp - q) / b
    b = 盈亏比 (avg_win / |avg_loss|)
    p = 胜率 (win_rate)
    q = 1 - p

    Returns:
        kelly_full: 全凯里仓位比例
        kelly_half: 半凯里（保守）
        kelly_quarter: 四分之一凯里（非常保守）
        interpretation: 中文解读
    """
    if avg_loss >= 0 or avg_win <= 0:
        return {"kelly_full": 0, "kelly_half": 0, "kelly_quarter": 0, "interpretation": "数据无效（无亏损样本，不适用凯里公式）"}

    b = avg_win / abs(avg_loss)
    p = win_rate / 100.0
    q = 1 - p
    f = (b * p - q) / b if b > 0 else 0
    f = max(0, min(f, 1.0))

    if f == 0:
        interp = "凯里值为0，当前策略没有正期望值，建议暂时观望，不要开仓。"
    elif f < 0.15:
        interp = f"建议轻仓参与（{f*100:.1f}%），当前策略期望值偏低，控制风险为首要任务。"
    elif f < 0.3:
        interp = f"建议中等仓位（{f*100:.1f}%），策略具有正期望值，可适度参与。"
    else:
        interp = f"建议积极参与（{f*100:.1f}%），策略统计学优势明显，但仍需设置止损线。"

    return {
        "kelly_full": round(f * 100, 1),
        "kelly_half": round(f * 50, 1),
        "kelly_quarter": round(f * 25, 1),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "b_ratio": round(b, 2),
        "interpretation": interp,
    }


def kelly_from_trades(trades: List[Dict]) -> Dict:
    """从交易记录中计算凯里仓位。trades: [{win: bool, pnl_pct: float}, ...]"""
    if not trades:
        return kelly_criterion(50, 2, 1)
    wins = [t for t in trades if t.get("win")]
    losses = [t for t in trades if not t.get("win")]
    wr = len(wins) / len(trades) * 100 if trades else 50
    aw = np.mean([t["pnl_pct"] for t in wins]) if wins else 2.0
    al = np.mean([abs(t["pnl_pct"]) for t in losses]) if losses else -1.0
    return kelly_criterion(wr, aw, abs(al) if al != 0 else 1.0)


# ============================================
# 2. MA10 黏着判断
# ============================================
def check_ma10_sticky(df: pd.DataFrame, threshold: float = 0.02) -> Dict:
    """
    判断最新价是否黏着 MA10（连续N天价格在MA10的±threshold范围内）。

    Returns:
        is_sticky: 是否黏着
        days_sticky: 连续黏着天数
        ma10: 当前MA10
        price: 当前价
        deviation: 偏离度%
        signal: 操作建议
    """
    if df.empty or len(df) < 15:
        return {"is_sticky": False, "signal": "数据不足"}

    df = df.copy()
    df["MA10"] = df["close"].rolling(10).mean()
    df_valid = df.dropna(subset=["MA10"])
    if df_valid.empty:
        return {"is_sticky": False, "signal": "数据不足"}

    latest_close = df_valid["close"].iloc[-1]
    latest_ma10 = df_valid["MA10"].iloc[-1]
    deviation = (latest_close / latest_ma10 - 1) * 100

    # 连续黏着天数
    sticky_days = 0
    for i in range(len(df_valid) - 1, -1, -1):
        close_i = df_valid["close"].iloc[i]
        ma10_i = df_valid["MA10"].iloc[i]
        if abs(close_i / ma10_i - 1) <= threshold:
            sticky_days += 1
        else:
            break

    is_sticky = sticky_days >= 3 and abs(deviation) <= threshold * 100

    if is_sticky:
        signal = f"🟢 价格黏着MA10已{sticky_days}天（偏离{deviation:+.2f}%）→ 不减仓，继续持有"
    else:
        signal = f"🔴 价格不黏着MA10（偏离{deviation:+.2f}%）→ 减仓一半。剩余半仓以MA25为清仓线。"

    return {
        "is_sticky": is_sticky,
        "days_sticky": sticky_days,
        "ma10": round(float(latest_ma10), 2),
        "price": round(float(latest_close), 2),
        "deviation": round(deviation, 2),
        "threshold": threshold * 100,
        "signal": signal,
    }


# ============================================
# 3. MA25 止损位
# ============================================
def check_ma25_stop(df: pd.DataFrame) -> Dict:
    """检查价格是否跌破 MA25，决定是否清仓。"""
    if df.empty or len(df) < 30:
        return {"below_ma25": False, "signal": "数据不足"}

    df["MA25"] = df["close"].rolling(25).mean()
    df_valid = df.dropna(subset=["MA25"])
    if df_valid.empty:
        return {"below_ma25": False, "signal": "数据不足"}

    latest = df_valid.iloc[-1]
    prev = df_valid.iloc[-2]
    below_now = latest["close"] < latest["MA25"]
    crossed_down = below_now and prev["close"] >= prev["MA25"]

    if below_now:
        if crossed_down:
            signal = "🔴 今日跌破MA25 → 清仓信号！立即卖出剩余仓位。"
        else:
            signal = f"🔴 价格在MA25下方（{latest['close']:.2f} < {latest['MA25']:.2f}）→ 维持空仓或等待反弹。"
    else:
        signal = "🟢 价格在MA25上方，安全。"

    return {
        "below_ma25": below_now,
        "crossed_down": crossed_down,
        "ma25": round(float(latest["MA25"]), 2),
        "price": round(float(latest["close"]), 2),
        "signal": signal,
    }


# ============================================
# 4. 历史最高点（前复权）
# ============================================
def check_history_high(df: pd.DataFrame) -> Dict:
    """检查当前价距历史最高（前复权）的距离。"""
    if df.empty:
        return {"signal": "数据不足"}

    all_time_high = df["close"].max()
    all_time_high_date = df[df["close"] == all_time_high]["date"].iloc[0]
    latest = df["close"].iloc[-1]
    pct_from_high = (latest / all_time_high - 1) * 100

    if pct_from_high >= -1:  # 距高点 1% 以内
        signal = f"🔴 接近历史高点！当前{latest:.2f}距前复权历史最高{all_time_high:.2f}({all_time_high_date.date()})仅{pct_from_high:+.2f}%。建议分批止盈。"
    elif pct_from_high >= -5:
        signal = f"🟡 距历史高点 {abs(pct_from_high):.1f}%，关注突破力度。"
    else:
        signal = f"距历史高点 {abs(pct_from_high):.1f}%（{all_time_high_date.date()} 创下 {all_time_high:.2f}）。"

    return {
        "all_time_high": round(float(all_time_high), 2),
        "high_date": str(all_time_high_date.date()),
        "latest_price": round(float(latest), 2),
        "pct_from_high": round(pct_from_high, 2),
        "signal": signal,
    }


# ============================================
# 5. 50% 回撤位标记
# ============================================
def check_50pct_retracement(df: pd.DataFrame, lookback: int = 60) -> Dict:
    """
    日线最高/最低的50%位置标记。
    从近期高点回撤到50%位置 → 标记支撑，若跌破 → 卖出信号。
    """
    if df.empty or len(df) < lookback:
        return {"signal": "数据不足"}

    recent = df.tail(lookback)
    high = recent["high"].max()
    low = recent["low"].min()
    mid = (high + low) / 2
    latest = recent["close"].iloc[-1]

    above_mid = latest > mid
    if above_mid:
        signal = f"🟢 最新价{latest:.2f}在50%回撤位{mid:.2f}上方（区间{low:.2f}–{high:.2f}），支撑有效。"
    else:
        signal = f"🔴 跌破50%回撤位{mid:.2f}！该位置是重要支撑，失守建议减仓。"

    return {
        "range_high": round(float(high), 2),
        "range_low": round(float(low), 2),
        "mid_point": round(float(mid), 2),
        "latest_price": round(float(latest), 2),
        "above_mid": above_mid,
        "lookback_days": lookback,
        "signal": signal,
    }


# ============================================
# 6. 突破后回踩两次买入
# ============================================
def check_double_pullback(df: pd.DataFrame, lookback: int = 120) -> Dict:
    """
    检测"突破后回踩两次"形态：
    1. 价格先突破 lookback 日高点
    2. 之后回踩（回调）
    3. 回踩 2 次后 → 买入信号
    """
    if df.empty or len(df) < lookback:
        return {"signal": "数据不足", "buy_ready": False}

    # 找突破点：突破 N 日高点
    df = df.copy()
    df["high_N"] = df["high"].rolling(lookback // 3).max().shift(1)  # 突破前的高点
    df["breakout"] = df["close"] > df["high_N"]

    # 找到最近一次突破后的回踩
    breakout_rows = df[df["breakout"]]
    if breakout_rows.empty:
        return {
            "signal": "近期无突破信号，暂不适用此策略。",
            "buy_ready": False,
            "pullback_count": 0,
        }

    last_breakout_idx = breakout_rows.index[-1]
    post_breakout = df.loc[last_breakout_idx:].copy()

    if len(post_breakout) < 5:
        return {"signal": "突破不久，数据不足以判断回踩。", "buy_ready": False, "pullback_count": 0}

    # 计算回踩：价格在突破价格下方且形成低点
    breakout_price = df.loc[last_breakout_idx, "close"]
    pullbacks = 0
    in_pullback = False
    pullback_lows = []

    for i in range(len(post_breakout)):
        idx = post_breakout.index[i]
        close_i = post_breakout.loc[idx, "close"]
        if close_i < breakout_price and not in_pullback:
            in_pullback = True
            pullbacks += 1
            pullback_lows.append(float(close_i))
        elif close_i > breakout_price and in_pullback:
            in_pullback = False

    if pullbacks >= 2:
        signal = f"🟢 突破后已完成{pullbacks}次回踩 → 买入信号！回踩低点: {pullback_lows}。止损设在最近回踩低点下方。"
        buy_ready = True
    elif pullbacks == 1:
        signal = f"🟡 突破后已回踩1次，等待第二次回踩确认后买入。"
        buy_ready = False
    else:
        signal = "突破后暂无回踩，等待回调确认。"
        buy_ready = False

    return {
        "signal": signal,
        "buy_ready": buy_ready,
        "pullback_count": pullbacks,
        "breakout_price": round(float(breakout_price), 2),
        "pullback_lows": pullback_lows,
    }


# ============================================
# 7. 布林带缩口检测
# ============================================
def check_bollinger_squeeze(df: pd.DataFrame, period: int = 20, threshold: float = 0.05) -> Dict:
    """检测布林带缩口（带宽压缩→即将变盘）。带宽 = (上轨-下轨)/中轨。"""
    if df.empty or len(df) < period + 10:
        return {"squeeze": False, "signal": "数据不足"}
    df = df.copy()
    df["B_MID"] = df["close"].rolling(period).mean()
    df["B_STD"] = df["close"].rolling(period).std()
    df["B_WIDTH"] = (2 * df["B_STD"]) / df["B_MID"]  # 带宽百分比
    latest = df["B_WIDTH"].iloc[-1]
    avg_3m = df["B_WIDTH"].tail(60).mean() if len(df) >= 60 else df["B_WIDTH"].mean()
    is_squeeze = latest < avg_3m * 0.7  # 当前带宽低于3月均值的70%

    if is_squeeze:
        signal = f"🟡 布林带缩口中（带宽{latest*100:.2f}%，仅为3月均值{avg_3m*100:.2f}%的{latest/avg_3m*100:.0f}%）→ 变盘在即，密切关注方向。"
    else:
        signal = f"布林带宽度正常（{latest*100:.2f}%），无明显缩口信号。"

    return {"squeeze": is_squeeze, "bandwidth": round(float(latest)*100, 2), "signal": signal}


# ============================================
# 8. 量价背离检测
# ============================================
def check_gap_break(df: pd.DataFrame, hold_days: int = 3) -> Dict:
    """
    跳空缺口不破买入：
    1. 检测向上跳空缺口（今日最低 > 昨日最高）
    2. 缺口在 hold_days 天内未被回补 → 强支撑确认 → 买入信号

    Returns:
        gaps: 最近 60 天内检测到的所有缺口
        active_buy: 是否有符合条件的未补跳空买入信号
        latest_gap: 最近缺口信息
        signal: 操作建议
    """
    if df.empty or len(df) < 60:
        return {"gaps_found": 0, "signal": "数据不足"}

    df = df.copy()
    df["gap_up"] = df["low"] > df["high"].shift(1)   # 向上跳空
    df["gap_down"] = df["high"] < df["low"].shift(1)  # 向下跳空
    df["gap_high"] = df["high"].shift(1)  # 缺口上沿（昨天最高）
    df["gap_filled"] = False

    # 找最近 60 天内的向上跳空缺口
    gap_up_rows = df.tail(60)
    gap_up_rows = gap_up_rows[gap_up_rows["gap_up"]]

    active_gaps = []
    for idx in gap_up_rows.index:
        gap_date = df.loc[idx, "date"]
        gap_upper = df.loc[idx, "high"]  # 跳空日的最高点
        gap_lower = df.loc[idx, "low"]   # 跳空日的最低点
        yesterday_high = df["high"].shift(1).loc[idx] if idx > 0 else gap_upper

        # 检查跳空后是否回补（价格跌回缺口下方 = 回补）
        after_gap = df.loc[idx+1:]
        filled = any(after_gap["low"] <= yesterday_high) if len(after_gap) > 0 else False
        days_since = len(after_gap)

        if not filled and days_since >= hold_days:
            active_gaps.append({
                "date": str(gap_date.date()),
                "gap_price": round(float(yesterday_high), 2),
                "current_price": round(float(df["close"].iloc[-1]), 2),
                "days_held": days_since,
            })

    if active_gaps:
        latest = active_gaps[-1]
        signal = f"🟢 跳空缺口不破！{latest['date']} 形成缺口（支撑位 {latest['gap_price']:.2f}），已守住 {latest['days_held']} 天未回补 → 强支撑确认，可考虑买入。止损设在缺口下沿。"
    elif len(gap_up_rows) > 0:
        # 有跳空但已被回补
        signal = "近期有跳空缺口但已被回补，暂不符合缺口不破买入条件。"
    else:
        signal = "近 60 天内无向上跳空缺口，暂不适用此策略。"

    return {
        "gaps_found": len(active_gaps),
        "active_gaps": active_gaps,
        "signal": signal,
    }


def check_volume_divergence(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """检测量价背离：价格上涨但成交量萎缩→上涨乏力。"""
    if df.empty or len(df) < lookback + 5:
        return {"divergence": False, "signal": "数据不足"}
    recent = df.tail(lookback)
    price_up = recent["close"].iloc[-1] > recent["close"].iloc[0]
    vol_first_half = recent["volume"].iloc[:lookback//2].mean()
    vol_second_half = recent["volume"].iloc[lookback//2:].mean()
    vol_declining = vol_second_half < vol_first_half * 0.8

    if price_up and vol_declining:
        signal = "🔴 量价背离：近10日价格上涨但成交量萎缩20%+ → 上涨动力不足，警惕回调。"
        divergence = "bearish"
    elif not price_up and vol_declining:
        signal = "🟢 缩量下跌：量能衰竭，抛压减轻 → 可能接近底部。"
        divergence = "bullish"
    else:
        signal = "量价关系正常，未检测到明显背离。"
        divergence = "none"

    return {"divergence": divergence, "signal": signal}


# ============================================
# 9. 综合信号汇总
# ============================================
def get_all_signals(ticker: str) -> Dict:
    """
    获取一支股票的全部交易信号。
    返回给前端展示的完整信号字典。
    """
    df = _fetch_history(ticker, days=1600)
    if df.empty:
        return {"error": f"无法获取 {ticker} 的历史数据"}

    # 各项信号
    ma10 = check_ma10_sticky(df)
    ma25 = check_ma25_stop(df)
    hist_high = check_history_high(df)
    retrace = check_50pct_retracement(df)
    pullback = check_double_pullback(df)
    boll_sqz = check_bollinger_squeeze(df)
    vol_div = check_volume_divergence(df)
    gap_break = check_gap_break(df)

    # 从 DataFram 计算凯里所需数据
    df_valid = df.copy()
    df_valid["ret"] = df_valid["close"].pct_change()
    df_valid["MA5"] = df_valid["close"].rolling(5).mean()
    df_valid["MA25"] = df_valid["close"].rolling(25).mean()
    df_v = df_valid.dropna(subset=["MA5", "MA25"])

    # 简单模拟交易计算胜率/盈亏比
    trades = []
    in_pos = False
    entry = 0
    for i in range(1, len(df_v)):
        idx = df_v.index[i]
        prev = df_v.index[i - 1]
        # 金叉买入 / 死叉卖出
        if (df_v.loc[idx, "MA5"] > df_v.loc[idx, "MA25"] and
            df_v.loc[prev, "MA5"] <= df_v.loc[prev, "MA25"] and not in_pos):
            entry = df_v.loc[idx, "close"]
            in_pos = True
        elif (df_v.loc[idx, "MA5"] < df_v.loc[idx, "MA25"] and
              df_v.loc[prev, "MA5"] >= df_v.loc[prev, "MA25"] and in_pos):
            pnl = (df_v.loc[idx, "close"] / entry - 1) * 100
            trades.append({"win": pnl > 0, "pnl_pct": pnl})
            in_pos = False

    kelly = kelly_from_trades(trades)

    # 综合买卖建议
    buy_signals = 0
    sell_signals = 0

    if pullback.get("buy_ready"): buy_signals += 1
    if ma10.get("is_sticky"): sell_signals -= 1  # 黏着=不卖

    if not ma10.get("is_sticky"): sell_signals += 1  # 不黏着=减半仓
    if ma25.get("below_ma25"): sell_signals += 2  # 跌破25=清仓
    if hist_high.get("pct_from_high", -100) >= -1: sell_signals += 1  # 近历史高点
    if not retrace.get("above_mid", True): sell_signals += 1  # 跌破50%回撤
    if boll_sqz.get("squeeze"): sell_signals += 0  # 缩口=关注但不加分（方向不明）
    if vol_div.get("divergence") == "bearish": sell_signals += 2  # 量价背离=强卖出
    if vol_div.get("divergence") == "bullish": buy_signals += 1  # 缩量跌=潜在买入
    if gap_break.get("gaps_found", 0) > 0: buy_signals += 3  # 跳空不破=强买入

    if sell_signals >= 3:
        action = "🔴 卖出信号强烈，建议减仓或清仓。"
    elif sell_signals >= 1:
        action = "🟡 存在卖出信号，建议降低仓位。"
    elif buy_signals >= 1:
        action = "🟢 出现买入信号，可考虑建仓。"
    else:
        action = "⚪ 目前无明显买卖信号，建议观望。"

    return {
        "ticker": ticker,
        "kelly": kelly,
        "ma10_sticky": ma10,
        "ma25_stop": ma25,
        "history_high": hist_high,
        "retrace_50": retrace,
        "double_pullback": pullback,
        "bollinger_squeeze": boll_sqz,
        "volume_divergence": vol_div,
        "gap_break": gap_break,
        "trade_count": len(trades),
        "action": action,
        "buy_score": buy_signals,
        "sell_score": sell_signals,
    }
