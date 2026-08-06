"""
LangGraph 图构建工厂 —— 组装 data_fetcher → [technical, fundamental] → cio 的全链路拓扑。
与 test_env.py 的 Phase 3 完全一致，铁律零修改。
"""

from langgraph.graph import StateGraph, END
from graph_types import StockAgentState
from data_fetcher import data_fetcher_node
from agents import technical_analyst_node, fundamental_analyst_node, cio_decision_node


# ==========================================
# 路由守卫：绝对熔断控制器（铁律——不可修改）
# ==========================================
def check_data_status(state: StockAgentState) -> str:
    """
    条件路由判断：数据成功 → 放行至分析节点；数据失败 → 直接 END 熔断。
    """
    if state["data_fetch_success"]:
        return "continue"
    else:
        return "end"


def build_graph() -> StateGraph:
    """
    构建并编译 LangGraph 计算图。

    拓扑结构（与 test_env.py 完全一致）：
        data_fetcher
            ├── (条件) check_data_status ── "continue" → technical_analyst
            │                            ── "end"      → END
            └── (无条件) → fundamental_analyst
        technical_analyst  ──→ cio_decision ──→ END
        fundamental_analyst ──→ cio_decision

    Returns:
        CompiledStateGraph: 已编译的图实例，可直接调用 .invoke(state)
    """
    workflow = StateGraph(StockAgentState)

    # 1. 注册所有节点
    workflow.add_node("data_fetcher", data_fetcher_node)
    workflow.add_node("technical_analyst", technical_analyst_node)
    workflow.add_node("fundamental_analyst", fundamental_analyst_node)
    workflow.add_node("cio_decision", cio_decision_node)

    # 2. 设定入口点
    workflow.set_entry_point("data_fetcher")

    # 3. 条件路由：数据成功 → 技术面分析；数据失败 → 熔断 END
    workflow.add_conditional_edges(
        "data_fetcher",
        check_data_status,
        {
            "continue": "technical_analyst",
            "end": END
        }
    )
    # 数据成功时间时触发基本面分析（与条件路由并行）
    workflow.add_edge("data_fetcher", "fundamental_analyst")

    # 4. 汇聚：技术面与基本面完成后，同时流入 CIO 决策节点
    workflow.add_edge("technical_analyst", "cio_decision")
    workflow.add_edge("fundamental_analyst", "cio_decision")

    # 5. CIO 决策完成后走向终点
    workflow.add_edge("cio_decision", END)

    return workflow.compile()
