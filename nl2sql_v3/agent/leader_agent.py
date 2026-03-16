import logging
from typing import Any, Dict, List, Optional

from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

from nl2sql_v3.agent.nl2sql_agent import NL2SQLAgent
from nl2sql_v3.config import config
from .graph import build_interactive_graph  # 从 graph.py 导入

logger = logging.getLogger(__name__)


class InteractiveNL2SQLAgent:
    """
    交互式 BI Agent：业务层 + 交互逻辑
    使用 graph.py 中的 graph 工厂构建 LangGraph
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        nl2sql_agent: Optional[NL2SQLAgent] = None,
        temperature: float = 0.2,
    ):
        # LLM
        if llm is None:
            llm = ChatOpenAI(
                api_key=config.services.llm.api_key,
                base_url=config.services.llm.base_url,
                model=config.services.llm.analyse_model,
                temperature=temperature,
            )
        self.llm = llm

        # NL2SQL Agent
        self.nl2sql_agent = nl2sql_agent or NL2SQLAgent(dialect=config.agent.sql_dialect)

        # 工具定义
        self.tools = [self._create_nl2sql_tool()]

        # Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能 BI 助手，与用户进行多轮自然对话，理解数据分析意图。
规则：
1. 根据整个对话历史，总结出当前最完整、精确的查询意图。
2. 如果意图清晰，直接调用 nl2sql_tool 生成 SQL。
3. 如果模糊/多轮，先澄清意图、确认细节，再回复用户（不要调用工具）。
4. 调用工具前，确保 query 是完整、自洽的自然语言描述。
5. 保持友好、专业、简洁。"""),
            ("placeholder", "{messages}"),
        ])

        # 提前绑定工具和组装 chain（只一次）
        self.bound_llm = self.llm.bind_tools(self.tools)
        self.chain = self.prompt | self.bound_llm

        # 构建 graph（传入 chain 和 tools_node）
        tools_node = ToolNode(self.tools)
        self.graph = build_interactive_graph(
            chain=self.chain,
            tools_node=tools_node,
            custom_tool_names_to_end=["nl2sql_tool"]  # 指定结束的工具名
        )

        # 会话管理
        self.threads: Dict[str, str] = {}

    def _create_nl2sql_tool(self):
        @tool
        def nl2sql_tool(
            query: str,
            dialect: str = "sqlite",
        ) -> Dict[str, Any]:
            """调用 NL2SQL 生成器，生成结构化 SQL 结果。"""
            result = self.nl2sql_agent.run(query=query, dialect=dialect)
            return result

        return nl2sql_tool

    def _get_thread_id(self, conversation_id: str) -> str:
        if conversation_id not in self.threads:
            self.threads[conversation_id] = f"thread_{conversation_id}_{id(self)}"
        return self.threads[conversation_id]

    def run(
        self,
        user_input: str,
        conversation_id: str = "default",
    ) -> Dict[str, Any]:
        thread_id = self._get_thread_id(conversation_id)
        config = {"configurable": {"thread_id": thread_id}}

        inputs = {"messages": [HumanMessage(content=user_input)]}

        final_output = None

        # 执行 graph
        for update in self.graph.stream(inputs, config, mode="update"):
            logger.info(f"Update: {update}")
        final_state = self.graph.get_state(config)

        messages = final_state.values.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, ToolMessage):
                final_output = last_msg.content  # nl2sql 结果
            elif isinstance(last_msg, AIMessage):
                final_output = last_msg.content
            else:
                final_output = str(last_msg)

        # 美化 nl2sql dict 输出
        if isinstance(final_output, dict):
            final_output = (
                f"SQL 生成结果：\n"
                f"SQL: {final_output.get('sql', '无')}\n"
                f"置信度: {final_output.get('confidence', '未知')}\n"
                f"解释: {final_output.get('explanation', '无')}\n"
                f"涉及表: {', '.join(final_output.get('selected_tables', []))}\n"
                f"使用字段: {', '.join(final_output.get('used_columns', []))}"
            )

        logger.info(f"[{conversation_id}] Final response: {str(final_output)[:100]}...")

        return {
            "output": final_output if isinstance(final_output, str) else str(final_output),
            "thread_id": thread_id,
        }


# 交互式客户端（不变）
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    agent = InteractiveNL2SQLAgent()
    
    conversation_id = "session_001"
    
    print("=" * 50)
    print("智能 BI 助手 - 多轮对话")
    print("输入问题即可对话，输入 'quit' 或 'exit' 退出")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break
            
            resp = agent.run(
                user_input=user_input,
                conversation_id=conversation_id
            )
            print(f"\n助手: {resp['output']}")
            
        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"\n错误: {e}")