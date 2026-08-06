import os
import sys
import time
import datetime
from typing import TypedDict
from pydantic import SecretStr

# 0. 根因修复：requests 库在 Windows 上默认从注册表读取系统代理，
#    即使清空 HTTP_PROXY 环境变量也会尝试通过代理连接，导致 AKShare 报 ProxyError。
#    必须在任何第三方库（尤其是 akshare）导入之前，猴子补丁 Session 关闭 trust_env。
import requests as _requests

_original_init = _requests.Session.__init__


def _patched_session_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.trust_env = False


_requests.Session.__init__ = _patched_session_init  # type: ignore[method-assign]

# 1. Windows GBK 兼容：强制 stdout 使用 UTF-8，否则 Emoji 会触发 UnicodeEncodeError
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[reportAttributeAccessIssue]

# 2. 辅助兜底：继续清空代理环境变量（双保险）
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
import akshare as ak

# 2. 确保加载 .env 文件中的大模型密钥
load_dotenv()

# ==========================================
# 第一阶段：定义全局数据状态 (State)
# ==========================================
class StockAgentState(TypedDict):
    ticker: str
    data_fetch_success: bool
    error_message: str
    raw_history_data: dict  
    technical_analysis: str 
    fundamental_analysis: str 
    final_report: str

# ==========================================
# 第二阶段：编写独立职能节点 (Nodes)
# ==========================================

# --- 节点 1：真实 A 股数据获取 Agent（腾讯数据源，绕过东方财富反爬屏蔽） ---
def data_fetcher_node(state: StockAgentState):
    print("-> 执行节点：[数据获取 Agent] 正在通过腾讯数据源拉取 A 股真实数据...")
    ticker = state["ticker"]

    # 根据股票代码自动添加 sh/sz 前缀（腾讯接口要求）
    if ticker.startswith(("60", "68")):
        tx_symbol = "sh" + ticker
    else:
        tx_symbol = "sz" + ticker

    for attempt in range(3):
        try:
            end_date = datetime.datetime.now().strftime("%Y%m%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y%m%d")

            # 使用腾讯数据源（东方财富源在此网络环境被封）
            df = ak.stock_zh_a_hist_tx(
                symbol=tx_symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=""
            )

            if df.empty:
                raise ValueError(f"未找到代码 {ticker} 的有效数据。")

            # 取最近 5 个交易日，列名从英文映射为中文（兼容下游）
            df = df.tail(5)[["date", "open", "high", "low", "close", "volume"]]
            df.rename(columns={  # type: ignore[reportCallIssue]
                "date": "日期", "open": "开盘", "high": "最高",
                "low": "最低", "close": "收盘", "volume": "成交量"
            }, inplace=True)
            df.set_index("日期", inplace=True)

            print(f"   状态：成功获取 {ticker} 的最新真实交易数据！")
            return {
                "data_fetch_success": True,
                "raw_history_data": df.to_dict(orient="index"),  # type: ignore[reportCallIssue]
                "error_message": ""
            }

        except Exception as e:
            print(f"   状态：第 {attempt+1} 次直连尝试失败，原因: {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                return {
                    "data_fetch_success": False,
                    "raw_history_data": {},
                    "error_message": str(e)
                }

# --- 路由守卫：绝对熔断控制器 ---
def check_data_status(state: StockAgentState):
    print("-> [系统主脑] 正在检查数据状态...")
    if state["data_fetch_success"]:
        print("   决策：数据获取成功，全面放行至多 Agent 并行分析。")
        return "continue"
    else:
        print("   决策：数据获取失败，触发绝对熔断，安全终止流程！")
        return "end"

# --- 节点 2：技术分析 Agent ---
def technical_analyst_node(state: StockAgentState):
    print("-> 执行节点：[技术分析 Agent] 正在思考...")
    ticker = state["ticker"]
    raw_data = state["raw_history_data"]
    
    prompt = f"你是一位专业的技术分析师。请根据以下 A 股真实数据 ({ticker}) 给出简短技术分析，包含支撑位与趋势，控制在 100 字以内。\n数据: {raw_data}"
    
    llm = ChatOpenAI(model="deepseek-chat", api_key=SecretStr(os.environ["OPENAI_API_KEY"]), base_url=os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com/v1"))
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"technical_analysis": response.content}

# --- 节点 3：基本面分析 Agent ---
def fundamental_analyst_node(state: StockAgentState):
    # 熔断守卫：数据获取失败时直接跳过，不消耗 LLM 调用
    if not state["data_fetch_success"]:
        print("-> [基本面分析 Agent] 数据获取失败，跳过分析（熔断保护）")
        return {"fundamental_analysis": ""}

    print("-> 执行节点：[基本面分析 Agent] 正在思考...")
    ticker = state["ticker"]
    
    prompt = f"你是一位资深的基本面分析师。请对 A 股公司 ({ticker}) 的长期投资价值与行业地位给出简短评估，控制在 100 字以内。"
    
    llm = ChatOpenAI(model="deepseek-chat", api_key=SecretStr(os.environ["OPENAI_API_KEY"]), base_url=os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com/v1"))
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"fundamental_analysis": response.content}

# --- 节点 4：CIO 决策汇总 Agent (终审引擎) ---
def cio_decision_node(state: StockAgentState):
    # 熔断守卫：数据获取失败时直接跳过，不消耗 LLM 调用
    if not state["data_fetch_success"]:
        print("-> [CIO 决策引擎] 数据获取失败，跳过决策（熔断保护）")
        return {"final_report": ""}

    print("-> 执行节点：[CIO 决策引擎] 正在综合各方意见...")
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
    
    请保持客观、严谨，总字数控制在 200 字以内。
    """
    
    llm = ChatOpenAI(model="deepseek-chat", api_key=SecretStr(os.environ["OPENAI_API_KEY"]), base_url=os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com/v1"))
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": response.content}


# ==========================================
# 第三阶段：编排计算图 (Graph Compilation)
# ==========================================
print("\n--- 开始编排多智能体工作流 ---")
workflow = StateGraph(StockAgentState)

# 1. 注册所有节点
workflow.add_node("data_fetcher", data_fetcher_node)
workflow.add_node("technical_analyst", technical_analyst_node)
workflow.add_node("fundamental_analyst", fundamental_analyst_node)
workflow.add_node("cio_decision", cio_decision_node)

# 2. 设定入口点
workflow.set_entry_point("data_fetcher")

# 3. 严格条件路由：只有数据成功（continue）时才流向技术面；失败则直接 END 结束
workflow.add_conditional_edges(
    "data_fetcher",         
    check_data_status,      
    {
        "continue": "technical_analyst", 
        "end": END                       
    }
)
# 数据成功时，同时触发基本面分析
workflow.add_edge("data_fetcher", "fundamental_analyst")

# 4. 汇聚逻辑：技术面与基本面完成后，同时流入 CIO 决策节点
workflow.add_edge("technical_analyst", "cio_decision")
workflow.add_edge("fundamental_analyst", "cio_decision")

# 5. CIO 决策完成后走向终点
workflow.add_edge("cio_decision", END)

app = workflow.compile()


# ==========================================
# 第四阶段：启动系统
# ==========================================
if __name__ == "__main__":
    initial_state: StockAgentState = {
        "ticker": "600519",  # 以贵州茅台为例
        "data_fetch_success": False,
        "error_message": "",
        "raw_history_data": {},
        "technical_analysis": "",
        "fundamental_analysis": "",
        "final_report": ""
    }
    
    print("========================================")
    print("🚀 启动全功能工业级多智能体量化系统...")
    print("========================================")
    
    final_state = app.invoke(initial_state)
    
    print("\n========================================")
    print("✅ 系统运行完毕！各 Agent 的分析与最终决策如下：")
    print("========================================")
    if final_state["data_fetch_success"]:
        print(f"【技术分析报告】:\n{final_state['technical_analysis']}\n")
        print(f"【基本面分析报告】:\n{final_state['fundamental_analysis']}\n")
        print("========================================")
        print(f"🎯【CIO 最终投资决策报告】:\n{final_state['final_report']}")
    else:
        print(f"❌ 流程已安全熔断，原因：{final_state['error_message']}")