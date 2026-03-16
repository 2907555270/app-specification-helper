# graph.py
import logging
from typing import Any, Dict, List, Literal, Annotated

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage

from langchain_core.tools import tool as ToolType  # 仅用于类型提示

logger = logging.getLogger(__name__)

# 状态定义（使用 add_messages 自动追加消息）
class AgentState(Dict[str, Any]):
    messages: Annotated[List[BaseMessage], add_messages] = []


def build_interactive_graph(
    chain: Runnable,           # 预绑定的 prompt | llm.bind_tools
    tools_node: ToolNode,      # ToolNode(tools)
    custom_tool_names_to_end: List[str] = ["nl2sql_tool"]
) -> Any:
    """
    构建交互式 Agent 的 LangGraph
    - agent_node: 使用传入的 chain（prompt | bound_llm）
    - tools_node: 处理工具调用
    - nl2sql_tool 执行完后直接结束 graph
    """
    
    graph = StateGraph(state_schema=AgentState)

    # agent 节点：决策（澄清、回复、调用工具）
    def agent_node(state: AgentState) -> AgentState:
        # chain 已经预绑定了 prompt 和 tools
        response = chain.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.set_entry_point("agent")

    # agent 后：判断是否调用工具
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END}
    )

    # tools 后：自定义路由 - 如果是指定工具（如 nl2sql_tool），直接结束
    def post_tool_router(state: AgentState) -> Literal["agent", "__end__"]:
        messages = state["messages"]
        last_msg = messages[-1] if messages else None

        if isinstance(last_msg, ToolMessage):
            tool_name = last_msg.name or ""
            # 判断是否是需要直接结束的工具
            if any(name in tool_name.lower() for name in custom_tool_names_to_end):
                return END

        return "agent"  # 其他工具回 agent

    graph.add_conditional_edges(
        "tools",
        post_tool_router,
        {"agent": "agent", END: END}
    )

    # 编译 graph（带 checkpointer）
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)