"""
交互式图表模块 — 同花顺风格暗色主题。
K线+量+KDJ+BOLL · 多时间范围 · 收益率分布。
"""

import datetime
from typing import Optional, Tuple

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
        df = ak.stock_zh_a_hist_tx(symbol=prefix + ticker, start_date=start, end_date=end, adjust="qfq")
        if df.empty: return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _calc_ma(df: pd.DataFrame, periods: list) -> pd.DataFrame:
    for p in periods:
        if len(df) >= p:
            df[f"MA{p}"] = df["close"].rolling(window=p).mean()
    return df


def _calc_kdj(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    """KDJ 指标。"""
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n) * 100
    df["K"] = rsv.ewm(com=2, adjust=False).mean()
    df["D"] = df["K"].ewm(com=2, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]
    return df


def _calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI 相对强弱指标。"""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def _calc_boll(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """BOLL 布林带。"""
    df["BOLL_MID"] = df["close"].rolling(n).mean()
    std = df["close"].rolling(n).std()
    df["BOLL_UP"] = df["BOLL_MID"] + 2 * std
    df["BOLL_DN"] = df["BOLL_MID"] - 2 * std
    return df


TIMEFRAME_DAYS = {
    "1 个月": 45, "3 个月": 100, "半年": 200,
    "1 年": 380, "3 年": 1200, "全部（5 年）": 2000,
}


def build_kline_chart(ticker: str, name: str = "", timeframe: str = "半年") -> Tuple[Optional[go.Figure], Optional[pd.DataFrame]]:
    """
    同花顺风格 K 线图：K线+布林带 / 成交量 / KDJ。
    """
    days = TIMEFRAME_DAYS.get(timeframe, 200)
    df = _fetch_history(ticker, days)
    if df.empty:
        return None, None

    df = _calc_ma(df, [5, 10, 20, 60])
    df = _calc_kdj(df)
    df = _calc_boll(df)
    df = _calc_rsi(df)
    df = df.dropna(subset=["MA5", "MA10"]).copy()
    df.reset_index(drop=True, inplace=True)

    stock_label = f"{ticker} {name}" if name else ticker

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.45, 0.12, 0.22, 0.21],
    )

    # ---- 面板1: K线 + 布林带 + 均线 ----
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="", increasing=dict(line=dict(color="#e83939", width=1), fillcolor="#e83939"),
        decreasing=dict(line=dict(color="#1aad19", width=1), fillcolor="#1aad19"),
        showlegend=False, hoverinfo="x+y+text",
    ), row=1, col=1)

    # 布林带
    fig.add_trace(go.Scatter(x=df["date"], y=df["BOLL_UP"], mode="lines",
        name="BOLL上轨", line=dict(color="#ff9800", width=0.8, dash="dot"), opacity=0.5, showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["BOLL_MID"], mode="lines",
        name="BOLL中轨", line=dict(color="#ff9800", width=1), opacity=0.6, showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["BOLL_DN"], mode="lines",
        name="BOLL下轨", line=dict(color="#ff9800", width=0.8, dash="dot"), opacity=0.5, showlegend=False), row=1, col=1)

    # 均线
    ma_colors = {"MA5": "#ffb340", "MA10": "#33a3ff", "MA20": "#e83939", "MA60": "#9b30ff"}
    for ma, color in ma_colors.items():
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df[ma], mode="lines",
                name=ma, line=dict(color=color, width=1.2), legendgroup="ma"), row=1, col=1)

    # ---- 面板2: 成交量 ----
    vol_colors = ["#e83939" if df["close"].iloc[i] >= df["open"].iloc[i] else "#1aad19"
                  for i in range(len(df))]
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="量",
        marker=dict(color=vol_colors, opacity=0.4), showlegend=False), row=2, col=1)

    # ---- 面板3: KDJ ----
    fig.add_trace(go.Scatter(x=df["date"], y=df["K"], mode="lines",
        name="K", line=dict(color="#e83939", width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["D"], mode="lines",
        name="D", line=dict(color="#33a3ff", width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["J"], mode="lines",
        name="J", line=dict(color="#ffb340", width=1, dash="dot")), row=3, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="#e83939", line_width=0.5, opacity=0.4, row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="#1aad19", line_width=0.5, opacity=0.4, row=3, col=1)

    # ---- 面板4: RSI ----
    fig.add_trace(go.Scatter(x=df["date"], y=df["RSI"], mode="lines",
        name="RSI(14)", line=dict(color="#9b30ff", width=1.5)), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#e83939", line_width=0.5, opacity=0.4, row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#1aad19", line_width=0.5, opacity=0.4, row=4, col=1)
    fig.add_hline(y=50, line_dash="solid", line_color="#666", line_width=0.3, row=4, col=1)

    # ---- 全局布局 ----
    fig.update_layout(
        template="plotly_dark", height=750, hovermode="x unified",
        margin=dict(l=10, r=20, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=10, color="#ccc"), bgcolor="rgba(0,0,0,0.3)"),
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        dragmode="pan",
    )
    grid = dict(showgrid=True, gridcolor="#333", gridwidth=0.5, zeroline=False)
    fig.update_yaxes(title_text=stock_label, row=1, col=1, **grid, tickformat=".2f")
    fig.update_yaxes(title_text="成交量", row=2, col=1, **grid, showticklabels=False)
    fig.update_yaxes(title_text="KDJ", row=3, col=1, **grid)
    fig.update_yaxes(title_text="RSI", row=4, col=1, **grid)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    return fig, df


def build_return_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
    """日收益率分布直方图。"""
    if df.empty or len(df) < 5:
        return None
    returns = df["close"].pct_change().dropna() * 100
    fig = go.Figure()
    colors = ["#e83939" if v >= 0 else "#1aad19" for v in returns]
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, marker_color=colors, opacity=0.7, name="日收益分布"))
    mean_ret = returns.mean()
    fig.add_vline(x=mean_ret, line_dash="dash", line_color="#ffb340",
                  annotation=dict(text=f"均值 {mean_ret:.2f}%", font=dict(color="#ccc")))
    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=30, b=10),
                      plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
                      xaxis_title="收益率 (%)", yaxis_title="频次",
                      font=dict(color="#999"))
    return fig
