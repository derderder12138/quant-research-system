"""
双均线策略引擎 — 25 日线 × 25 月线（MA25 / MA500）黄金交叉/死亡交叉。
包含信号生成、回测收益计算、可视化图表。
"""

import datetime
from typing import Optional, List, Dict, Tuple

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _fetch_history(ticker: str, days: int) -> pd.DataFrame:
    """从腾讯源拉取历史K线。"""
    import requests as _r
    _o = _r.Session.__init__
    def _p(s, *a, **k): _o(s, *a, **k); s.trust_env = False
    _r.Session.__init__ = _p

    import akshare as ak

    prefix = "sh" if ticker.startswith(("60", "68")) else "sz"
    end = datetime.datetime.now().strftime("%Y%m%d")
    start = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y%m%d")

    try:
        df = ak.stock_zh_a_hist_tx(symbol=prefix + ticker, start_date=start, end_date=end, adjust="")
        if df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def run_strategy(ticker: str) -> Optional[Dict]:
    """
    执行 25 日线 × 25 月线 双均线策略。

    Args:
        ticker: A 股代码

    Returns:
        {
            "df": 含信号的历史数据 DataFrame,
            "signals": [(date, type, price), ...],  # type: "buy" / "sell"
            "total_return": 策略总收益(%),
            "buy_hold_return": 买入持有收益(%),
            "annual_return": 策略年化收益(%),
            "max_drawdown": 最大回撤(%),
            "sharpe": 夏普比率,
            "win_rate": 信号胜率(%),
            "total_trades": 交易次数,
            "latest_signal": 最新信号,
            "latest_date": 最新信号日期,
        }
    """
    days_needed = 800  # 至少需要 500 个交易日（25 个月）+ 余量
    df = _fetch_history(ticker, days=days_needed + 100)

    if df.empty or len(df) < 500:
        return None

    # 计算均线
    df["MA25"] = df["close"].rolling(window=25).mean()      # 25 日线
    df["MA500"] = df["close"].rolling(window=500).mean()    # 25 月线（≈500 交易日）

    # 均线交叉信号
    df["diff"] = df["MA25"] - df["MA500"]          # 正值 = 短线上穿长线
    df["signal"] = 0
    # 金叉：diff 由负转正
    df.loc[(df["diff"] > 0) & (df["diff"].shift(1) <= 0), "signal"] = 1
    # 死叉：diff 由正转负
    df.loc[(df["diff"] < 0) & (df["diff"].shift(1) >= 0), "signal"] = -1

    # 提取信号点
    signals: List[Tuple] = []
    buy_signals = df[df["signal"] == 1]
    sell_signals = df[df["signal"] == -1]
    for _, row in buy_signals.iterrows():
        signals.append((row["date"], "buy", row["close"]))
    for _, row in sell_signals.iterrows():
        signals.append((row["date"], "sell", row["close"]))
    signals.sort(key=lambda x: x[0])

    # ---- 回测 ----
    # 策略逻辑：金叉买入持有至死叉卖出
    df_valid = df.dropna(subset=["MA25", "MA500"]).copy()
    df_valid["position"] = 0
    in_position = False
    for i in range(1, len(df_valid)):
        idx = df_valid.index[i]
        prev_idx = df_valid.index[i - 1]
        if df_valid.loc[idx, "signal"] == 1 and not in_position:
            df_valid.loc[idx, "position"] = 1
            in_position = True
        elif df_valid.loc[idx, "signal"] == -1 and in_position:
            df_valid.loc[idx, "position"] = 0
            in_position = False
        else:
            df_valid.loc[idx, "position"] = df_valid.loc[prev_idx, "position"]

    # 计算策略日收益
    df_valid["ret"] = df_valid["close"].pct_change()
    df_valid["strategy_ret"] = df_valid["ret"] * df_valid["position"].shift(1)

    # 策略绩效
    total_ret = (np.prod(1 + df_valid["strategy_ret"].fillna(0)) - 1) * 100
    bh_ret = (df_valid["close"].iloc[-1] / df_valid["close"].iloc[0] - 1) * 100

    trading_days = 252
    years = max(len(df_valid) / trading_days, 0.1)
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100

    # 最大回撤
    cumulative = np.cumprod(1 + df_valid["strategy_ret"].fillna(0))
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = np.min(drawdowns) * 100

    # 夏普
    excess = df_valid["strategy_ret"].fillna(0) - 0.025 / trading_days
    sharpe = np.sqrt(trading_days) * excess.mean() / excess.std() if excess.std() > 0 else 0

    # 胜率（按每次交易）
    trades = len(buy_signals)
    wins = 0
    if trades > 0:
        for _, buy_row in buy_signals.iterrows():
            buy_price = buy_row["close"]
            buy_date = buy_row["date"]
            # 找下一个死叉
            next_sell = sell_signals[sell_signals["date"] > buy_date]
            if not next_sell.empty:
                sell_price = next_sell.iloc[0]["close"]
                if sell_price > buy_price:
                    wins += 1
    win_rate = (wins / trades * 100) if trades > 0 else 0

    # 最新信号与当前状态
    latest_signal = "无"
    latest_date = None
    current_status = "未知"
    recent_signals = df_valid[df_valid["signal"] != 0]
    if not recent_signals.empty:
        last = recent_signals.iloc[-1]
        latest_signal = "🔴 死叉（卖出）" if last["signal"] == -1 else "🟢 金叉（买入）"
        latest_date = last["date"]

    # 当前均线状态（不管有没有交叉）
    last_ma25 = df_valid["MA25"].iloc[-1]
    last_ma500 = df_valid["MA500"].iloc[-1]
    gap_pct = (last_ma25 / last_ma500 - 1) * 100
    if last_ma25 > last_ma500:
        current_status = f"🟢 金叉持续中 | MA25:{last_ma25:.1f} > MA500:{last_ma500:.1f} (领先{gap_pct:+.1f}%)"
    else:
        current_status = f"🔴 死叉持续中 | MA25:{last_ma25:.1f} < MA500:{last_ma500:.1f} (落后{gap_pct:+.1f}%)"

    return {
        "df": df_valid,
        "signals": signals,
        "total_return": round(total_ret, 2),
        "buy_hold_return": round(bh_ret, 2),
        "annual_return": round(annual_ret, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "win_rate": round(win_rate, 1),
        "total_trades": trades,
        "latest_signal": latest_signal,
        "latest_date": latest_date,
        "current_status": current_status,
        "last_ma25": round(float(last_ma25), 1),
        "last_ma500": round(float(last_ma500), 1),
        "gap_pct": round(gap_pct, 1),
    }


def build_strategy_chart(ticker: str, name: str = "") -> Tuple[Optional[go.Figure], Optional[Dict]]:
    """
    构建双均线策略可视化图表。

    Returns:
        (Plotly Figure, 策略结果 Dict)
    """
    result = run_strategy(ticker)
    if result is None:
        return None, None

    df = result["df"]
    title = f"{ticker} {name} — 25日线 × 25月线 双均线策略" if name else f"{ticker} — 25日线 × 25月线 双均线策略"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.65, 0.35],
        subplot_titles=(title, "策略净值曲线 vs 买入持有"),
    )

    # K 线
    fig.add_trace(
        go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"],
                       close=df["close"], name="K线",
                       increasing_line_color="#e53935", decreasing_line_color="#43a047",
                       showlegend=False),
        row=1, col=1,
    )

    # 均线
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA25"], mode="lines",
                             name="MA25 (25日)", line=dict(color="#ff9800", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA500"], mode="lines",
                             name="MA500 (25月)", line=dict(color="#2196f3", width=2)), row=1, col=1)

    # 金叉/死叉标记
    buy_signals = df[df["signal"] == 1]
    sell_signals = df[df["signal"] == -1]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(x=buy_signals["date"], y=buy_signals["close"],
                                 mode="markers", name="金叉(买入)",
                                 marker=dict(symbol="triangle-up", size=12, color="#e53935")),
                      row=1, col=1)
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(x=sell_signals["date"], y=sell_signals["close"],
                                 mode="markers", name="死叉(卖出)",
                                 marker=dict(symbol="triangle-down", size=12, color="#43a047")),
                      row=1, col=1)

    # 净值曲线
    df["cum_strategy"] = np.cumprod(1 + df["strategy_ret"].fillna(0))
    df["cum_bh"] = df["close"] / df["close"].iloc[0]

    fig.add_trace(go.Scatter(x=df["date"], y=df["cum_strategy"], mode="lines",
                             name="策略净值", line=dict(color="#ff9800", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["cum_bh"], mode="lines",
                             name="买入持有", line=dict(color="#9e9e9e", width=1.5, dash="dash")),
                  row=2, col=1)

    # 布局
    fig.update_layout(
        template="plotly_white", height=700, hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="top", y=1.18, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="价格（元）", row=1, col=1)
    fig.update_yaxes(title_text="净值", row=2, col=1)

    return fig, result
