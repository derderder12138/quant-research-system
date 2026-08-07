"""
自定义策略：5日线金叉25日线 + MACD多周期 + 25周线过滤 + 双级止损。
买入：MA5↑MA25 AND MACD日周月黄线>0 AND 价格>MA25周
卖出：跌穿MA10减半仓 | 跌穿MA25全清
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
    def _p(s,*a,**k):_o(s,*a,**k);s.trust_env=False
    _r.Session.__init__=_p
    import akshare as ak
    prefix="sh" if ticker.startswith(("60","68")) else "sz"
    end=datetime.datetime.now().strftime("%Y%m%d")
    start=(datetime.datetime.now()-datetime.timedelta(days=days)).strftime("%Y%m%d")
    try:
        df=ak.stock_zh_a_hist_tx(symbol=prefix+ticker,start_date=start,end_date=end,adjust="qfq")
        if df.empty:return pd.DataFrame()
        df["date"]=pd.to_datetime(df["date"]);return df.sort_values("date").reset_index(drop=True)
    except:return pd.DataFrame()


def _calc_macd(df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
    """计算 MACD: DIF/DEA/Histogram。黄线=DEA。"""
    ema12 = df[col].ewm(span=12, adjust=False).mean()
    ema26 = df[col].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()  # 黄线
    df["MACD_HIST"] = 2 * (df["DIF"] - df["DEA"])
    return df


def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """日线→周线。"""
    w = df.set_index("date").resample("W").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    return w.reset_index()


def _resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """日线→月线。"""
    m = df.set_index("date").resample("ME").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    return m.reset_index()


def run_custom_strategy(ticker: str) -> Optional[Dict]:
    """
    执行自定义多条件策略。

    买入条件（全部满足）：
      1. MA5 上穿 MA25（金叉）
      2. MACD 日线 DEA(黄线) > 0
      3. MACD 周线 DEA(黄线) > 0
      4. MACD 月线 DEA(黄线) > 0
      5. 价格 > MA25周（≈125日线）

    卖出条件：
      - 价格跌穿 MA10 → 减仓 50%
      - 价格跌穿 MA25 → 全部清仓
    """
    # 需要至少 800 天数据（月线 MACD 需要足够样本）
    df = _fetch_history(ticker, days=1600)
    if df.empty or len(df) < 200:
        return None

    # ---- 日线指标 ----
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA10"] = df["close"].rolling(10).mean()
    df["MA25"] = df["close"].rolling(25).mean()
    df["MA125"] = df["close"].rolling(125).mean()  # ≈25周线
    df = _calc_macd(df)

    # MACD 日线黄线位置
    df["MACD_DAY_OK"] = df["DEA"] > 0

    # MA5 金叉 MA25
    df["CROSS_UP"] = (df["MA5"] > df["MA25"]) & (df["MA5"].shift(1) <= df["MA25"].shift(1))

    # ---- 周线 MACD ----
    weekly = _resample_weekly(df)
    weekly = _calc_macd(weekly)
    weekly["MACD_WEEK_OK"] = weekly["DEA"] > 0

    # 向日线映射周线状态（取最近一周的值）
    df["MACD_WEEK_OK"] = False
    for i, row in weekly.iterrows():
        mask = df["date"] <= row["date"]
        if mask.any():
            last_idx = df[mask].index[-1]
            df.loc[last_idx:, "MACD_WEEK_OK"] = row["MACD_WEEK_OK"]

    # ---- 月线 MACD ----
    monthly = _resample_monthly(df)
    monthly = _calc_macd(monthly)
    monthly["MACD_MONTH_OK"] = monthly["DEA"] > 0
    df["MACD_MONTH_OK"] = False
    for i, row in monthly.iterrows():
        mask = df["date"] <= row["date"]
        if mask.any():
            last_idx = df[mask].index[-1]
            df.loc[last_idx:, "MACD_MONTH_OK"] = row["MACD_MONTH_OK"]

    # ---- 价格条件 ----
    df["PRICE_ABOVE_MA125"] = df["close"] > df["MA125"]

    # ---- 综合买入信号 ----
    df["BUY_SIGNAL"] = (
        df["CROSS_UP"] &
        df["MACD_DAY_OK"] &
        df["MACD_WEEK_OK"] &
        df["MACD_MONTH_OK"] &
        df["PRICE_ABOVE_MA125"]
    )

    # ---- 卖出信号 ----
    # 跌破 MA10 → 减半仓
    df["SELL_HALF"] = (df["close"] < df["MA10"]) & (df["close"].shift(1) >= df["MA10"].shift(1))
    # 跌破 MA25 → 全清
    df["SELL_ALL"] = (df["close"] < df["MA25"]) & (df["close"].shift(1) >= df["MA25"].shift(1))

    # ---- 回测 ----
    df_valid = df.dropna(subset=["MA5","MA10","MA25","MA125","DEA"]).copy()
    if len(df_valid) < 50:
        return None

    position = 0  # 0=空仓, 1=满仓, 0.5=半仓
    df_valid["position"] = 0.0
    buy_dates, sell_half_dates, sell_all_dates = [], [], []

    for i in range(1, len(df_valid)):
        idx = df_valid.index[i]
        prev = df_valid.index[i - 1]

        if df_valid.loc[idx, "BUY_SIGNAL"] and position == 0:
            position = 1.0
            buy_dates.append(df_valid.loc[idx, "date"])
        elif df_valid.loc[idx, "SELL_ALL"] and position > 0:
            position = 0.0
            sell_all_dates.append(df_valid.loc[idx, "date"])
        elif df_valid.loc[idx, "SELL_HALF"] and position == 1.0:
            position = 0.5
            sell_half_dates.append(df_valid.loc[idx, "date"])

        df_valid.loc[idx, "position"] = position

    df_valid["ret"] = df_valid["close"].pct_change()
    df_valid["strategy_ret"] = df_valid["ret"] * df_valid["position"].shift(1)

    # ---- 绩效 ----
    n = len(df_valid)
    years = max(n / 252, 0.1)
    total_ret = (np.prod(1 + df_valid["strategy_ret"].fillna(0)) - 1) * 100
    bh_ret = (df_valid["close"].iloc[-1] / df_valid["close"].iloc[0] - 1) * 100
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100
    excess = df_valid["strategy_ret"].fillna(0) - 0.025 / 252
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0
    cum = np.cumprod(1 + df_valid["strategy_ret"].fillna(0))
    max_dd = np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)) * 100

    # 胜率
    wins, losses = 0, 0
    for _, br in df_valid[df_valid["BUY_SIGNAL"]].iterrows():
        bd = br["date"]
        next_sell = df_valid[(df_valid["date"] > bd) & ((df_valid["SELL_ALL"]) | (df_valid["SELL_HALF"]))]
        if not next_sell.empty:
            sp = next_sell.iloc[0]["close"]
            if sp > br["close"]: wins += 1
            else: losses += 1
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    # ---- 最新状态 ----
    latest = df_valid.iloc[-1]
    conditions = {
        "MA5金叉MA25": bool(latest.get("CROSS_UP", False)) or (latest["MA5"] > latest["MA25"]),
        "MACD日线黄线>0": bool(latest["MACD_DAY_OK"]),
        "MACD周线黄线>0": bool(latest["MACD_WEEK_OK"]),
        "MACD月线黄线>0": bool(latest["MACD_MONTH_OK"]),
        "价格>25周线": bool(latest["PRICE_ABOVE_MA125"]),
    }
    all_met = all(conditions.values())
    pos_now = latest["position"]

    return {
        "df": df_valid,
        "total_return": round(total_ret, 2),
        "bh_return": round(bh_ret, 2),
        "annual_return": round(annual_ret, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "total_trades": total_trades,
        "buy_count": len(buy_dates),
        "conditions": conditions,
        "all_met": all_met,
        "position": pos_now,
        "latest_price": float(latest["close"]),
        "latest_ma5": float(latest["MA5"]),
        "latest_ma10": float(latest["MA10"]),
        "latest_ma25": float(latest["MA25"]),
        "latest_ma125": float(latest["MA125"]),
        "latest_dea_day": float(latest["DEA"]),
    }


def build_custom_chart(ticker: str, name: str = "") -> Tuple[Optional[go.Figure], Optional[Dict]]:
    """构建自定义策略可视化图表。"""
    result = run_custom_strategy(ticker)
    if result is None:
        return None, None

    df = result["df"]
    title = f"{ticker} {name} — 5×25金叉+MACD多周期+25周线策略" if name else f"{ticker} — 自定义多条件策略"

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(title, "MACD 日线 (黄线=DEA)", "策略净值"),
    )

    # K线 + 均线
    fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"],
                   low=df["low"], close=df["close"], name="K线",
                   increasing_line_color="#e53935", decreasing_line_color="#43a047", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA5"], mode="lines", name="MA5", line=dict(color="#ff9800",width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA10"], mode="lines", name="MA10", line=dict(color="#f44336",width=1,dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA25"], mode="lines", name="MA25", line=dict(color="#2196f3",width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA125"], mode="lines", name="MA125(25周)", line=dict(color="#9c27b0",width=2)), row=1, col=1)

    # 买入标记
    buy_pts = df[df["BUY_SIGNAL"]]
    if not buy_pts.empty:
        fig.add_trace(go.Scatter(x=buy_pts["date"], y=buy_pts["close"], mode="markers",
                       name="买入", marker=dict(symbol="triangle-up", size=14, color="#e53935")), row=1, col=1)
    sell_half = df[df["SELL_HALF"]]
    if not sell_half.empty:
        fig.add_trace(go.Scatter(x=sell_half["date"], y=sell_half["close"], mode="markers",
                       name="减半仓", marker=dict(symbol="triangle-down", size=10, color="#ff9800")), row=1, col=1)
    sell_all = df[df["SELL_ALL"]]
    if not sell_all.empty:
        fig.add_trace(go.Scatter(x=sell_all["date"], y=sell_all["close"], mode="markers",
                       name="清仓", marker=dict(symbol="x", size=12, color="#43a047")), row=1, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df["date"], y=df["DIF"], mode="lines", name="DIF", line=dict(color="#e53935",width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["DEA"], mode="lines", name="DEA(黄线)", line=dict(color="#ff9800",width=1.5)), row=2, col=1)
    colors = ["#e53935" if v >= 0 else "#43a047" for v in df["MACD_HIST"].fillna(0)]
    fig.add_trace(go.Bar(x=df["date"], y=df["MACD_HIST"], name="柱", marker_color=colors, showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#888", row=2, col=1)

    # 净值
    df["cum_strategy"] = np.cumprod(1 + df["strategy_ret"].fillna(0))
    df["cum_bh"] = df["close"] / df["close"].iloc[0]
    fig.add_trace(go.Scatter(x=df["date"], y=df["cum_strategy"], mode="lines", name="策略净值", line=dict(color="#ff9800",width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["cum_bh"], mode="lines", name="买入持有", line=dict(color="#9e9e9e",width=1,dash="dash")), row=3, col=1)

    fig.update_layout(template="plotly_white", height=900, hovermode="x unified",
                      margin=dict(l=10,r=10,t=60,b=10),
                      legend=dict(orientation="h", yanchor="top", y=1.22, x=0))
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="净值", row=3, col=1)

    return fig, result
