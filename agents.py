"""
Agent 节点模块 — 增强版。技术面/基本面/CIO 各自拉取相关 Skill 数据注入 Prompt。
"""

import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

_llm: Optional[ChatOpenAI] = None


def init_llm(model: str, api_key: str, api_base: str) -> ChatOpenAI:
    global _llm
    _llm = ChatOpenAI(model=model, api_key=api_key, base_url=api_base)  # type: ignore[reportArgumentType]
    return _llm


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1"),
        )  # type: ignore[reportArgumentType]
    return _llm


def _enrich_technical_context(ticker: str, raw_data: dict) -> str:
    """从各 Skill 模块拉取技术面数据，拼接成上下文。"""
    context = [f"股票: {ticker}\n近5日行情数据: {raw_data}"]
    try:
        from signals import check_ma10_sticky, check_ma25_stop, check_volume_divergence, check_bollinger_squeeze, check_gap_break
        from signals import _fetch_history
        import pandas as pd
        df = _fetch_history(ticker, days=400)  # type: ignore[reportCallIssue]
        if not df.empty and len(df) > 30:
            df["MA5"] = df["close"].rolling(5).mean()
            df["MA10"] = df["close"].rolling(10).mean()
            df["MA20"] = df["close"].rolling(20).mean()
            df["MA60"] = df["close"].rolling(60).mean()
            latest = df.iloc[-1]

            context.append(f"\n【均线系统】")
            context.append(f"MA5={latest['MA5']:.2f} MA10={latest['MA10']:.2f} MA20={latest['MA20']:.2f} MA60={latest['MA60']:.2f}")
            context.append(f"收盘={latest['close']:.2f} {'MA5>MA10>MA20(多头排列)' if latest['MA5']>latest['MA10']>latest['MA20'] else '非多头排列'}")

            # RSI
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            context.append(f"\nRSI(14)={rsi:.1f} {'超买>70' if rsi>70 else '超卖<30' if rsi<30 else '正常'}")

            # KDJ
            low_n = df["low"].rolling(9).min()
            high_n = df["high"].rolling(9).max()
            rsv = (df["close"] - low_n) / (high_n - low_n) * 100
            k = rsv.ewm(com=2, adjust=False).mean().iloc[-1]
            d = rsv.ewm(com=2, adjust=False).mean().ewm(com=2, adjust=False).mean().iloc[-1]
            j = 3*k - 2*d
            context.append(f"KDJ K={k:.1f} D={d:.1f} J={j:.1f} {'超买' if k>80 else '超卖' if k<20 else '正常'}")

            # MACD
            ema12 = df["close"].ewm(span=12, adjust=False).mean()
            ema26 = df["close"].ewm(span=26, adjust=False).mean()
            dif = ema12.iloc[-1] - ema26.iloc[-1]
            dea = (ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1]
            context.append(f"MACD DIF={dif:.3f} DEA={dea:.3f} {'金叉(多头)' if dif>dea else '死叉(空头)'}")

            # 成交量趋势
            vol_5 = df["volume"].tail(5).mean()
            vol_20 = df["volume"].tail(20).mean()
            context.append(f"量能: 5日均量={vol_5/1e4:.0f}万手 20日均量={vol_20/1e4:.0f}万手 {'放量' if vol_5>vol_20*1.2 else '缩量' if vol_5<vol_20*0.8 else '平量'}")

            # 交易信号摘要
            sigs = {}
            try:
                sigs["MA10黏着"] = check_ma10_sticky(df).get("signal", "")
                sigs["MA25止损"] = check_ma25_stop(df).get("signal", "")
                sigs["量价关系"] = check_volume_divergence(df).get("signal", "")
                sigs["布林缩口"] = check_bollinger_squeeze(df).get("signal", "")
                sigs["跳空缺口"] = check_gap_break(df).get("signal", "")
                context.append(f"\n【交易信号状态】")
                for k, v in sigs.items():
                    if v: context.append(f"{k}: {v[:80]}")
            except Exception:
                pass
    except Exception:
        pass
    return "\n".join(context)


def _enrich_fundamental_context(ticker: str) -> str:
    """拉取基本面数据。"""
    context = [f"股票: {ticker}"]
    try:
        from fundamental_data import get_single_fundamentals
        from industry import classify_stock
        fd = get_single_fundamentals(ticker)
        if fd and fd.get("pe", 0) > 0:
            context.append(f"PE(动)={fd['pe']:.2f} 总市值={fd['market_cap']:,.0f}亿 流通市值={fd['circ_market_cap']:,.0f}亿")
            context.append(f"ROE={fd.get('roe',0):.2f}% 换手率={fd['turnover_rate']:.2f}%")
            context.append(f"52周高={fd['high_52w']:.2f} 52周低={fd['low_52w']:.2f}")
            context.append(f"近1年涨幅={fd['y1_change']:+.2f}% 近半年={fd['hy_change']:+.2f}% 年初至今={fd['ytd_change']:+.2f}%")
            industry = classify_stock(ticker)
            context.append(f"行业: {industry}")
    except Exception:
        pass
    return "\n".join(context)


def technical_analyst_node(state: dict) -> dict:
    """技术分析 Agent —— 增强版：携带 RSI/KDJ/MACD/量能/信号数据。"""
    ticker = state["ticker"]
    raw_data = state["raw_history_data"]
    context = _enrich_technical_context(ticker, raw_data)

    prompt = f"""你是一位拥有20年经验的资深技术分析师。请根据以下A股真实数据给出专业技术分析。

{context}

请输出（200字以内）：
1. 当前趋势判断（多头/空头/震荡）及理由
2. 关键技术位（支撑位和压力位，基于均线和布林带）
3. 量能是否配合当前趋势
4. 短期操作参考（不要给买卖建议，只说技术面状态）"""

    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"technical_analysis": response.content}


def fundamental_analyst_node(state: dict) -> dict:
    """基本面分析 Agent —— 增强版：携带 PE/ROE/市值/行业/阶段涨幅。"""
    ticker = state["ticker"]
    context = _enrich_fundamental_context(ticker)

    prompt = f"""你是一位资深基本面分析师。请根据以下数据评估该公司的长期投资价值。

{context}

请输出（200字以内）：
1. 估值水平评判（PE相对行业是高是低，是否合理）
2. 盈利能力评估（ROE水平，赚钱效率）
3. 市场定价分析（市值规模、52周位置）
4. 行业地位与长期前景（是否处于好赛道）"""

    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"fundamental_analysis": response.content}


def cio_decision_node(state: dict) -> dict:
    """CIO 终审 Agent —— 综合技术面+基本面+量化指标+交易信号。"""
    ticker = state["ticker"]
    tech = state["technical_analysis"]
    fund = state["fundamental_analysis"]

    # 附加上下文：风险指标和交易信号
    extra = ""
    try:
        from risk_metrics import calculate_metrics
        from signals import _fetch_history
        df = _fetch_history(ticker, days=800)
        if not df.empty and len(df) > 60:
            m = calculate_metrics(df)
            if "error" not in m:
                extra = f"""
【量化风险指标】
年化收益={m['annual_return']:.1f}% 年化波动={m['annual_volatility']:.1f}% 夏普比率={m['sharpe_ratio']:.2f}
最大回撤={m['max_drawdown']:.1f}% 日胜率={m['win_rate']:.1f}% 盈亏比={m['profit_factor']:.2f}
30日波动={m['volatility_30d']:.1f}% 趋势={m['trend']}
"""
    except Exception:
        pass

    prompt = f"""你是一位资深首席投资官(CIO)。请综合以下三方面信息，做出最终投资决策。

股票: {ticker}

【技术面分析】:
{tech}

【基本面分析】:
{fund}
{extra}
请输出最终综合决策报告（250字以内）：
1. 核心矛盾：技术面信号与基本面价值的冲突点在哪里
2. 综合评级：[积极建仓/谨慎持有/观望等待/减仓回避]
3. 仓位建议：基于夏普比率{extra.split('夏普比率=')[1].split()[0] if '夏普比率' in extra else '未知'}和回撤水平
4. 关键风险提示
5. 最终评级标签（仅输出以下之一：[积极建仓/谨慎持有/观望等待/减仓回避]）

保持客观严谨、数据驱动、不带感情色彩。"""

    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": response.content}
