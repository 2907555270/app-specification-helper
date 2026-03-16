import logging
import time
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from nl2sql_v3.agent.nl2sql_prompts import PromptTemplate, CompactPromptTemplate
from nl2sql_v3.agent.schema_builder import SchemaBuilder, CompactSchemaBuilder
from nl2sql_v3.config import config
from nl2sql_v3.models.table_info import TableInfo, ColumnInfo, ForeignKeyRelation, RelatedTable, to_table_info
from nl2sql_v3.recall.fusion import HybridRetriever
from nl2sql_v3.recall.base import RecallResult

logger = logging.getLogger(__name__)


class SQLGenerationResult(BaseModel):
    sql: str = Field(..., description="生成的 SQL 语句")
    confidence: float = Field(..., description="置信度 0.0~1.0", ge=0.0, le=1.0)
    explanation: str = Field(..., description="生成逻辑简要解释")
    selected_tables: List[str] = Field(..., description="实际使用的表名列表")
    used_columns: List[str] = Field(..., description="实际使用的字段列表")


class NL2SQLAgent:
    """
    单次 NL2SQL 生成器：
    - 输入：完整自然语言 query + dialect（可选）
    - 内部：表召回 → schema 构建 → prompt → structured LLM → 输出结构化结果
    - 不维护对话历史、不处理多轮
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        retriever: Optional[HybridRetriever] = None,
        schema_builder: Optional[SchemaBuilder] = None,
        prompt_template_class: type = CompactPromptTemplate,
        dialect: str = "sqlite",
        include_fewshot: bool = False,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        # LLM
        if llm is None:
            llm = ChatOpenAI(
                api_key=config.services.llm.api_key,
                base_url=config.services.llm.base_url,
                model=config.services.llm.coding_model,
                temperature=temperature,
            )
        self.llm = llm

        self.structured_llm = self.llm.with_structured_output(
            SQLGenerationResult,
            method="json_schema",
        )

        # 召回器（必须有）
        self.retriever = HybridRetriever()

        # Schema Builder & Prompt Template
        self.schema_builder = schema_builder or CompactSchemaBuilder()
        self.prompt_template_class = prompt_template_class
        self.dialect = dialect
        self.include_fewshot = include_fewshot
        self.max_retries = max_retries

    def _generate_sql(self, query: str, dialect: Optional[str] = None) -> SQLGenerationResult:
        """核心单次生成逻辑"""
        start_time = time.time()

        # 1. 表召回
        recalled_results: List[RecallResult] = self.retriever.retrieve(query)
        
        if not recalled_results:
            raise ValueError(f"没有召回任何相关表，query: {query}")
        
        recalled_tables: List[TableInfo] = []
        for r in recalled_results:
            recalled_tables.append(to_table_info(r))

        # 2. 构建 schema 文本
        schema_text = self.schema_builder.build_schema_text(recalled_tables)

        # 3. 构建 prompt（使用原有 PromptTemplate 的风格）
        messages = self.prompt_template_class.build_prompt(
            query=query,
            schema=schema_text,
            include_fewshot=self.include_fewshot,
            dialect=dialect if dialect else self.dialect,
        )

        # 4. 转成 LangChain 消息格式
        from langchain_core.messages import HumanMessage, SystemMessage

        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            else:
                langchain_messages.append(HumanMessage(content=msg["content"]))
        
        logger.info(f"NL2SQL input messages: {langchain_messages}")
        result: SQLGenerationResult = self.structured_llm.invoke(langchain_messages)

        llm_time = time.time() - start_time
        logger.info(
            f"SQL 生成成功 | confidence={result.confidence:.2f} | "
            f"time={llm_time:.2f}s | sql={result.sql[:80]}..."
        )

        return result

    def run(
        self,
        query: str,
        dialect: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        对外暴露的单次调用接口

        Args:
            query: 完整、独立的自然语言查询（由交互 Agent 总结）
            dialect: 可选，覆盖默认 dialect

        Returns:
            SQLGenerationResult 的 dict
        """
        if not query.strip():
            raise ValueError("查询不能为空")

        try:
            result = self._generate_sql(query, dialect=dialect)
            return result.model_dump()

        except Exception as e:
            logger.error(f"SQL 生成最终失败: {str(e)}", exc_info=True)
            # 可以在这里返回一个默认的低置信度结果，或者抛出自定义异常
            return {
                "sql": "",
                "confidence": 0.0,
                "explanation": f"生成失败：{str(e)}",
                "selected_tables": [],
                "used_columns": [],
            }


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    agent = NL2SQLAgent(
        dialect="sqlite",
        include_fewshot=True,
    )

    result = agent.run(
        query="how many clubs are there?"
    )
    print(result)