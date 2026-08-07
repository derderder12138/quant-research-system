"""
双均线策略引擎 + 多参数优化器。
支持单策略回测 + 批量 MA 组合扫描，找出最优配对。
"""

import datetime
from typing import Optional, List, Dict, Tuple

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
        df = ak.stock_zh_a_hist_tx(symbol=prefix + ticker, start_date=start, end_date=end, adjust="")
        if df.empty: return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ============================================
# 单策略回测
# ============================================
def _backtest_ma_pair(df: pd.DataFrame, short: int, long: int) -> Optional[Dict]:
    """回测一组 (short, long) 均线交叉策略。"""
    if len(df) < long + 5:
        return None

    df = df.copy()
    df["MA_S"] = df["close"].rolling(window=short).mean()
    df["MA_L"] = df["close"].rolling(window=long).mean()
    df["diff"] = df["MA_S"] - df["MA_L"]
    df["signal"] = 0
    df.loc[(df["diff"] > 0) & (df["diff"].shift(1) <= 0), "signal"] = 1   # 金叉
    df.loc[(df["diff"] < 0) & (df["diff"].shift(1) >= 0), "signal"] = -1  # 死叉

    df_valid = df.dropna(subset=["MA_S", "MA_L"]).copy()
    df_valid = df_valid.copy()
    df_valid.loc[:, "position"] = 0
    df_valid.loc[:, "ret"] = df_valid["close"].pct_change()
    df_valid.loc[:, "strategy_ret"] = 0.0
    if len(df_valid) < 10:
        return None

    # 持仓状态
    df_valid["position"] = 0
    in_pos = False
    for i in range(1, len(df_valid)):
        idx = df_valid.index[i]
        prev = df_valid.index[i - 1]
        if df_valid.loc[idx, "signal"] == 1 and not in_pos:
            df_valid.loc[idx, "position"] = 1; in_pos = True
        elif df_valid.loc[idx, "signal"] == -1 and in_pos:
            df_valid.loc[idx, "position"] = 0; in_pos = False
        else:
            df_valid.loc[idx, "position"] = df_valid.loc[prev, "position"]

    df_valid["ret"] = df_valid["close"].pct_change()
    df_valid["strategy_ret"] = df_valid["ret"] * df_valid["position"].shift(1)

    n = len(df_valid)
    years = max(n / 252, 0.1)
    total_ret = (np.prod(1 + df_valid["strategy_ret"].fillna(0)) - 1) * 100
    bh_ret = (df_valid["close"].iloc[-1] / df_valid["close"].iloc[0] - 1) * 100
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100
    excess = df_valid["strategy_ret"].fillna(0) - 0.025 / 252
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0

    cumulative = np.cumprod(1 + df_valid["strategy_ret"].fillna(0))
    running_max = np.maximum.accumulate(cumulative)
    max_dd = np.min((cumulative - running_max) / running_max) * 100

    # 交易次数 + 胜率
    buy_signals = df_valid[df_valid["signal"] == 1]
    sell_signals = df_valid[df_valid["signal"] == -1]
    trades = len(buy_signals)
    wins = 0
    if trades > 0:
        for _, br in buy_signals.iterrows():
            bp, bd = br["close"], br["date"]
            ns = sell_signals[sell_signals["date"] > bd]
            if not ns.empty and ns.iloc[0]["close"] > bp:
                wins += 1
    win_rate = (wins / trades * 100) if trades > 0 else 0

    # 最新信号
    recent = df_valid[df_valid["signal"] != 0]
    latest_signal = "无"
    latest_date = None
    if not recent.empty:
        last = recent.iloc[-1]
        latest_signal = "死叉(卖)" if last["signal"] == -1 else "金叉(买)"
        latest_date = last["date"]

    return {
        "short": short, "long": long,
        "total_return": round(total_ret, 2), "bh_return": round(bh_ret, 2),
        "annual_return": round(annual_ret, 2), "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2), "win_rate": round(win_rate, 1),
        "total_trades": trades, "latest_signal": latest_signal,
        "latest_date": latest_date,
        "excess": round(total_ret - bh_ret, 2),
    }


# ============================================
# 多参数优化器
# ============================================
def optimize_ma_pairs(ticker: str, name: str = "") -> Tuple[Optional[go.Figure], Optional[pd.DataFrame]]:
    """
    扫描所有 MA 组合，按夏普比率排名，返回热力图 + 排名表。
    """
    df = _fetch_history(ticker, days=1300)
    if df.empty or len(df) < 300:
        return None, None

    short_windows = [5, 8, 10, 13, 15, 20, 21, 25, 30, 34, 40, 50, 55, 60, 75, 89, 100, 120, 144, 150]
    long_windows = [20, 30, 40, 50, 60, 75, 89, 100, 120, 144, 150, 180, 200, 233, 250, 300, 377, 400, 500, 600]

    results = []
    for short in short_windows:
        for long in long_windows:
            if short >= long:
                continue
            r = _backtest_ma_pair(df, short, long)
            if r:
                results.append(r)

    if not results:
        return None, None

    df_results = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    # ---- 热力图 ----
    # 取 top 400 组合做热力图
    heat_data = df_results.head(400).pivot_table(
        index="short", columns="long", values="sharpe", aggfunc="first"
    )

    title = f"{ticker} {name} — MA 组合优化热力图 (夏普比率)" if name else f"{ticker} — MA 组合优化热力图"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
        subplot_titles=(title, "Top 20 策略排名"),
    )

    # 热力图
    fig.add_trace(
        go.Heatmap(
            z=heat_data.values, x=heat_data.columns.tolist(), y=heat_data.index.tolist(),
            colorscale="RdYlGn", zmid=0, colorbar=dict(title="夏普比率", len=0.6, y=0.75),
            hovertemplate="MA%{y} × MA%{x}<br>夏普: %{z:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # 排名表
    top20 = df_results.head(20).copy()
    fig.add_trace(
        go.Bar(
            x=top20["sharpe"], y=top20.apply(lambda r: f"MA{r['short']}×MA{r['long']}", axis=1),
            orientation="h", marker_color=np.where(top20["sharpe"] > 0, "#27ae60", "#e74c3c"),
            text=top20["sharpe"].round(2), textposition="outside",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_white", height=850,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_yaxes(title_text="短线 MA 周期", row=1, col=1)
    fig.update_xaxes(title_text="长线 MA 周期", row=1, col=1)
    fig.update_xaxes(title_text="夏普比率", row=2, col=1)

    return fig, df_results


# ============================================
# 单策略图表（保留原接口）
# ============================================
def run_strategy(ticker: str) -> Optional[Dict]:
    """执行 25日线 × 25月线(500日) 策略。"""
    df = _fetch_history(ticker, days=1300)
    if df.empty or len(df) < 500:
        return None
    return _backtest_ma_pair(df, 25, 500)


def build_strategy_chart(ticker: str, name: str = "") -> Tuple[Optional[go.Figure], Optional[Dict]]:
    """构建 25日×500日 策略可视化图表。"""
    result = run_strategy(ticker)
    if result is None:
        return None, None

    df = _fetch_history(ticker, days=1300)
    if df.empty:
        return None, None
    df["MA_S"] = df["close"].rolling(25).mean()
    df["MA_L"] = df["close"].rolling(500).mean()
    df["diff"] = df["MA_S"] - df["MA_L"]
    df["signal"] = 0
    df.loc[(df["diff"] > 0) & (df["diff"].shift(1) <= 0), "signal"] = 1
    df.loc[(df["diff"] < 0) & (df["diff"].shift(1) >= 0), "signal"] = -1
    df_valid = df.dropna(subset=["MA_S", "MA_L"])

    # 计算净值
    df_valid["position"] = 0
    in_pos = False
    for i in range(1, len(df_valid)):
        idx = df_valid.index[i]; prev = df_valid.index[i - 1]
        if df_valid.loc[idx, "signal"] == 1 and not in_pos:
            df_valid.loc[idx, "position"] = 1; in_pos = True
        elif df_valid.loc[idx, "signal"] == -1 and in_pos:
            df_valid.loc[idx, "position"] = 0; in_pos = False
        else:
            df_valid.loc[idx, "position"] = df_valid.loc[prev, "position"]
    df_valid["ret"] = df_valid["close"].pct_change()
    df_valid["strategy_ret"] = df_valid["ret"] * df_valid["position"].shift(1)
    df_valid["cum_strategy"] = np.cumprod(1 + df_valid["strategy_ret"].fillna(0))
    df_valid["cum_bh"] = df_valid["close"] / df_valid["close"].iloc[0]

    title = f"{ticker} {name} — 25日线 × 25月线" if name else f"{ticker} — 25日线 × 25月线"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.65, 0.35],
        subplot_titles=(title, "策略净值 vs 买入持有"),
    )
    fig.add_trace(go.Candlestick(x=df_valid["date"], open=df_valid["open"], high=df_valid["high"],
                   low=df_valid["low"], close=df_valid["close"], name="K线",
                   increasing_line_color="#e53935", decreasing_line_color="#43a047", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_valid["date"], y=df_valid["MA_S"], mode="lines",
                   name="MA25", line=dict(color="#ff9800", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_valid["date"], y=df_valid["MA_L"], mode="lines",
                   name="MA500(25月)", line=dict(color="#2196f3", width=2)), row=1, col=1)

    buy_s = df_valid[df_valid["signal"] == 1]
    sell_s = df_valid[df_valid["signal"] == -1]
    if not buy_s.empty:
        fig.add_trace(go.Scatter(x=buy_s["date"], y=buy_s["close"], mode="markers",
                       name="金叉", marker=dict(symbol="triangle-up", size=12, color="#e53935")), row=1, col=1)
    if not sell_s.empty:
        fig.add_trace(go.Scatter(x=sell_s["date"], y=sell_s["close"], mode="markers",
                       name="死叉", marker=dict(symbol="triangle-down", size=12, color="#43a047")), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_valid["date"], y=df_valid["cum_strategy"], mode="lines",
                   name="策略净值", line=dict(color="#ff9800", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_valid["date"], y=df_valid["cum_bh"], mode="lines",
                   name="买入持有", line=dict(color="#9e9e9e", width=1.5, dash="dash")), row=2, col=1)

    fig.update_layout(template="plotly_white", height=700, hovermode="x unified",
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="top", y=1.18, x=0))
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="净值", row=2, col=1)

    # 当前状态
    last_s = df_valid["MA_S"].iloc[-1]; last_l = df_valid["MA_L"].iloc[-1]
    gap = (last_s / last_l - 1) * 100
    status = f"🟢 金叉持续中 | MA25:{last_s:.1f} > MA500:{last_l:.1f} (+{gap:.1f}%)" if last_s > last_l else f"🔴 死叉持续中 | MA25:{last_s:.1f} < MA500:{last_l:.1f} ({gap:.1f}%)"
    result["current_status"] = status
    result["last_ma25"] = round(float(last_s), 1)
    result["last_ma500"] = round(float(last_l), 1)
    result["gap_pct"] = round(gap, 1)

    return fig, result
