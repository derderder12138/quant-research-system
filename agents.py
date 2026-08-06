"""
Agent 节点模块 —— 技术面分析、基本面分析、CIO 终审决策。
从 test_env.py 提取，prompt 逻辑保持一致。
LLM 实例通过 init_llm() 统一注入，全局共享复用。
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 模块级 LLM 实例（全局复用）
_llm: ChatOpenAI = None


def init_llm(model: str, api_key: str, api_base: str) -> ChatOpenAI:
    """初始化并缓存全局 LLM 实例，避免每个节点重复创建。"""
    global _llm
    _llm = ChatOpenAI(model=model, api_key=api_key, base_url=api_base)
    return _llm


def _get_llm() -> ChatOpenAI:
    """获取已初始化的 LLM 实例，若未初始化则从环境变量自动创建。"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")
        )
    return _llm


def technical_analyst_node(state: dict) -> dict:
    """
    技术分析 Agent —— 基于近 5 日真实行情数据给出短线技术判断。
    输入：state["ticker"], state["raw_history_data"]
    输出：{"technical_analysis": str}
    """
    ticker = state["ticker"]
    raw_data = state["raw_history_data"]

    prompt = (
        f"你是一位专业的技术分析师。请根据以下 A 股真实数据 ({ticker}) 给出简短技术分析，"
        f"包含支撑位与趋势，控制在 100 字以内。\n数据: {raw_data}"
    )

    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"technical_analysis": response.content}


def fundamental_analyst_node(state: dict) -> dict:
    """
    基本面分析 Agent —— 基于公司长期价值与行业地位给出评估。
    输入：state["ticker"]
    输出：{"fundamental_analysis": str}
    """
    ticker = state["ticker"]

    prompt = (
        f"你是一位资深的基本面分析师。请对 A 股公司 ({ticker}) 的长期投资价值与行业地位"
        f"给出简短评估，控制在 100 字以内。"
    )

    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"fundamental_analysis": response.content}


def cio_decision_node(state: dict) -> dict:
    """
    CIO 终审决策引擎 —— 综合技术与基本面报告，输出最终投资决策。
    输入：state["ticker"], state["technical_analysis"], state["fundamental_analysis"]
    输出：{"final_report": str}
    """
    ticker = state["ticker"]
    tech_report = state["technical_analysis"]
    fund_report = state["fundamental_analysis"]

    prompt = f"""
你是一位资深的首席投资官 (CIO)。现在你需要综合技术分析师和基本面分析师的报告，对 A 股股票代码 {ticker} 做出最终的投资决策。

【技术分析报告】:
{tech_report}

【基本面分析报告】:
{fund_report}

请输出最终的综合决策报告，内容需包含：
1. 核心观点冲突分析（技术面短线信号 vs 基本面长线价值）
2. 最终投资评级（如：积极建仓 / 谨慎持有 / 观望等待）
3. 仓位控制建议与风控止损位
4. 最终评级标签（仅输出以下之一：[积极建仓/谨慎持有/观望等待/减仓回避]）

请保持客观、严谨，总字数控制在 200 字以内。
"""

    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": response.content}
