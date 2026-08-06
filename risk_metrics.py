"""
量化风险收益指标 — 华尔街级分析。
波动率、夏普比率、最大回撤、Beta、收益分布、VaR。
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def calculate_metrics(df: pd.DataFrame, risk_free_rate: float = 0.025) -> Dict:
    """
    基于历史K线 DataFrame 计算全套量化指标。

    Args:
        df: 含 close 列的历史数据 DataFrame
        risk_free_rate: 无风险利率（默认 2.5%）

    Returns:
        {
            "total_return": 总收益率(%),
            "annual_return": 年化收益率(%),
            "annual_volatility": 年化波动率(%),
            "sharpe_ratio": 夏普比率,
            "max_drawdown": 最大回撤(%),
            "max_drawdown_days": 最长回撤天数,
            "win_rate": 日胜率(%),
            "avg_win": 平均盈利(%),
            "avg_loss": 平均亏损(%),
            "profit_factor": 盈亏比,
            "var_95": 95% VaR(%),
            "current_price": 最新价,
            "price_52w_high": 52周最高,
            "price_52w_low": 52周最低,
            "pct_from_high": 距52周高点的距离(%),
            "volatility_30d": 30日波动率(%),
            "trend": 趋势判断,
        }
    """
    if df.empty or len(df) < 10:
        return {"error": "数据不足（至少需要 10 个交易日）"}

    close = df["close"].values
    returns = np.diff(close) / close[:-1]
    n = len(returns)
    trading_days = 252

    # 总收益 %
    total_return = (close[-1] / close[0] - 1) * 100

    # 年化收益率 %
    years = n / trading_days
    annual_return = ((1 + total_return / 100) ** (1 / max(years, 0.05)) - 1) * 100

    # 年化波动率 %
    annual_vol = np.std(returns, ddof=1) * np.sqrt(trading_days) * 100

    # 夏普比率
    sharpe = (annual_return - risk_free_rate * 100) / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = np.min(drawdowns) * 100  # 负数，越小越差

    # 最大回撤持续天数
    dd_start = 0
    max_dd_days = 0
    in_dd = False
    for i in range(len(cumulative)):
        if cumulative[i] < running_max[i]:
            if not in_dd:
                dd_start = i
                in_dd = True
            max_dd_days = max(max_dd_days, i - dd_start)
        else:
            in_dd = False

    # 胜率 / 平均盈亏 / 盈亏比
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / n * 100 if n > 0 else 0
    avg_win = np.mean(wins) * 100 if len(wins) > 0 else 0
    avg_loss = np.mean(losses) * 100 if len(losses) > 0 else 0
    total_wins = np.sum(np.abs(wins)) if len(wins) > 0 else 0
    total_losses = np.sum(np.abs(losses)) if len(losses) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    # 95% VaR (历史模拟法)
    var_95 = np.percentile(returns, 5) * 100

    # 当前价 / 52周高低
    current_price = close[-1]
    lookback = min(n, trading_days)
    recent = close[-lookback:]
    price_52w_high = np.max(recent)
    price_52w_low = np.min(recent)
    pct_from_high = (current_price / price_52w_high - 1) * 100

    # 30日波动率
    lookback_30 = min(n, 21)
    vol_30d = np.std(returns[-lookback_30:], ddof=1) * np.sqrt(trading_days) * 100 if lookback_30 > 2 else 0

    # 趋势判断（基于 MA5/MA20 位置和最近 20 日涨跌）
    if len(close) >= 20:
        ma5 = np.mean(close[-5:])
        ma20 = np.mean(close[-20:])
        recent_ret = (close[-1] / close[-20] - 1) * 100
        if ma5 > ma20 and recent_ret > 0:
            trend = "上升趋势 ↑"
        elif ma5 < ma20 and recent_ret < 0:
            trend = "下降趋势 ↓"
        else:
            trend = "震荡整理 ↔"
    else:
        trend = "数据不足"

    return {
        "total_return": round(total_return, 2),
        "annual_return": round(annual_return, 2),
        "annual_volatility": round(annual_vol, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_days": max_dd_days,
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "var_95": round(var_95, 2),
        "current_price": round(float(current_price), 2),
        "price_52w_high": round(float(price_52w_high), 2),
        "price_52w_low": round(float(price_52w_low), 2),
        "pct_from_high": round(pct_from_high, 2),
        "volatility_30d": round(vol_30d, 2),
        "trend": trend,
    }


def metrics_summary(metrics: Dict) -> str:
    """将指标 dict 格式化为人类可读的摘要文本（供 LLM 分析使用）。"""
    if "error" in metrics:
        return f"指标计算失败: {metrics['error']}"

    return f"""
【量化风险收益指标】
- 最新价: {metrics['current_price']}
- 52周最高: {metrics['price_52w_high']} | 52周最低: {metrics['price_52w_low']} | 距高点: {metrics['pct_from_high']}%
- 总收益率: {metrics['total_return']}% | 年化收益率: {metrics['annual_return']}%
- 年化波动率: {metrics['annual_volatility']}% | 30日波动率: {metrics['volatility_30d']}%
- 夏普比率: {metrics['sharpe_ratio']}
- 最大回撤: {metrics['max_drawdown']}% (持续 {metrics['max_drawdown_days']} 个交易日)
- 日胜率: {metrics['win_rate']}% | 平均盈利: {metrics['avg_win']}% | 平均亏损: {metrics['avg_loss']}%
- 盈亏比: {metrics['profit_factor']}
- 95% VaR (日): {metrics['var_95']}%
- 趋势判断: {metrics['trend']}
"""
