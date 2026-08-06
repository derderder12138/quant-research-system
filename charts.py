"""
交互式图表模块 — Plotly K 线图 + 成交量 + 均线叠加。
支持多时间范围：1月/3月/半年/1年/3年/全部。
"""

import datetime
from typing import Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def _fetch_history(ticker: str, days: int) -> pd.DataFrame:
    """从腾讯源拉取历史K线数据。"""
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
        df = df.sort_values("date")
        return df
    except Exception:
        return pd.DataFrame()


def _calc_ma(df: pd.DataFrame, periods: list) -> pd.DataFrame:
    """计算移动平均线。"""
    for p in periods:
        if len(df) >= p:
            df[f"MA{p}"] = df["close"].rolling(window=p).mean()
    return df


TIMEFRAME_DAYS = {
    "1 个月": 45,
    "3 个月": 100,
    "半年": 200,
    "1 年": 380,
    "3 年": 1200,
    "全部（最多 5 年）": 2000,
}


def build_kline_chart(ticker: str, name: str = "", timeframe: str = "半年") -> Tuple[Optional[go.Figure], Optional[pd.DataFrame]]:
    """
    构建交互式 K 线图。

    Args:
        ticker: 股票代码
        name: 股票名称
        timeframe: 时间范围（TIMEFRAME_DAYS 的 key）

    Returns:
        (plotly Figure, DataFrame) 或 (None, None)
    """
    days = TIMEFRAME_DAYS.get(timeframe, 200)
    df = _fetch_history(ticker, days)
    if df.empty:
        return None, None

    df = _calc_ma(df, [5, 10, 20, 60])

    # 颜色
    colors = ["#e53935" if c >= o else "#43a047" for c, o in zip(df["close"], df["open"])]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker} {name}" if name else ticker, "成交量"),
    )

    # K 线
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing_line_color="#e53935",
            decreasing_line_color="#43a047",
        ),
        row=1, col=1,
    )

    # 均线
    ma_styles = {"MA5": ("#ff9800", 1.2), "MA10": ("#2196f3", 1.2), "MA20": ("#9c27b0", 1.5), "MA60": ("#795548", 1.5)}
    for ma_name, (color, width) in ma_styles.items():
        if ma_name in df.columns:
            fig.add_trace(
                go.Scatter(x=df["date"], y=df[ma_name], mode="lines", name=ma_name, line=dict(color=color, width=width), opacity=0.7),
                row=1, col=1,
            )

    # 成交量柱状图
    fig.add_trace(
        go.Bar(x=df["date"], y=df["volume"], name="成交量", marker_color=colors, opacity=0.35, showlegend=False),
        row=2, col=1,
    )

    # 布局
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=550,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="价格（元）", row=1, col=1)
    fig.update_yaxes(title_text="成交量（股）", row=2, col=1)

    return fig, df


def build_return_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
    """构建日收益率分布直方图。"""
    if df.empty or len(df) < 5:
        return None

    returns = df["close"].pct_change().dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns, nbinsx=50,
        marker_color=["#e53935" if v >= 0 else "#43a047" for v in returns],
        opacity=0.7, name="日收益率分布",
    ))
    mean_ret = returns.mean()
    fig.add_vline(x=mean_ret, line_dash="dash", line_color="#333", annotation_text=f"均值 {mean_ret:.2f}%")
    fig.update_layout(
        template="plotly_white", height=300, margin=dict(l=10, r=10, t=20, b=10),
        title="日收益率分布", xaxis_title="收益率 (%)", yaxis_title="频次",
    )
    return fig
