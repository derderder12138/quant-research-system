# Project: Stock Agent (LangGraph + Multi-Agent Quant System)

## 1. 项目核心架构 (Architecture)
这是一个基于 Python、LangGraph 和 DeepSeek API 的工业级多智能体量化投研系统。
- `state_models.py` / `test_env.py`: 核心状态机与工作流定义。
- 关键状态 (`StockAgentState`): 包含 ticker, data_fetch_success, raw_history_data, technical_analysis, fundamental_analysis, final_report。

## 2. 严禁破坏的既定铁律 (Strict Rules)
- **绝对不要**随意修改已经跑通的 LangGraph 条件路由与安全熔断机制 (`check_data_status`)。
- **绝对不要**使用伪造的模拟数据代替真实接口：数据获取必须通过 AKShare 真实拉取，且自带容灾与重试。
- 采用 **双 Agent 并行 + CIO 终审** 的闭环结构，后续所有新增的 Agent（如风控 Agent、情绪 Agent）都必须汇聚到 CIO 节点或按既定规范接入图谱。

## 3. 常用操作命令 (Commands)
- 运行测试: `python test_env.py`