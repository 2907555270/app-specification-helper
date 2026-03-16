import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nl2sql_v3.agent.llm_client import get_llm_client, LangChainLLMClient
from nl2sql_v3.agent.prompts import CompactPromptTemplate, PromptTemplate, SQLGenerationResult
from nl2sql_v3.agent.schema_builder import CompactSchemaBuilder, SchemaBuilder
from nl2sql_v3.data.loader import MetadataLoader
from nl2sql_v3.models.table_info import TableInfo
from nl2sql_v3.recall.fusion import HybridRetriever
from nl2sql_v3.util.db_manager import db_manager as _db_manager


logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    schema_builder: SchemaBuilder = field(default_factory=CompactSchemaBuilder)
    prompt_template: type = CompactPromptTemplate
    llm_client: Optional[LangChainLLMClient] = None
    llm_provider: str = "openrouter"
    include_fewshot: bool = True
    temperature: float = 0.0
    max_retries: int = 2


class NL2SQLAgent:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

        if self.config.llm_client is None:
            self.config.llm_client = get_llm_client(
                provider=self.config.llm_provider,
                type="coding",
            )

    def run(
        self,
        query: str,
        recalled_tables: List[TableInfo],
    ) -> SQLGenerationResult:
        if not query:
            raise ValueError("Query cannot be empty")

        if not recalled_tables:
            raise ValueError("Recalled tables cannot be empty")

        tables_to_use = recalled_tables

        schema_text = self.config.schema_builder.build_schema_text(tables_to_use)

        messages = self.config.prompt_template.build_prompt(
            query=query,
            schema=schema_text,
            include_fewshot=self.config.include_fewshot,
        )

        llm_start = time.time()
        for attempt in range(self.config.max_retries):
            try:
                result = self.config.llm_client.chat_with_json_output(
                    messages=messages,
                    temperature=self.config.temperature,
                )
                llm_time = time.time() - llm_start

                sql_result = SQLGenerationResult(
                    sql=result.get("sql", ""),
                    confidence=result.get("confidence", 0.0),
                    explanation=result.get("explanation", ""),
                    selected_tables=result.get("selected_tables", []),
                    used_columns=result.get("used_columns", []),
                )

                logger.info(
                    f"SQL generated successfully: confidence={sql_result.confidence}, "
                    f"timings: llm={llm_time:.3f}s"
                )
                return sql_result

            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise

        raise RuntimeError("Failed to generate SQL after max retries")


class NL2SQLPipeline:
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        agent: Optional[NL2SQLAgent] = None,
        table_loader: Optional[MetadataLoader] = None,
        db_manager: Optional[Any] = None,
    ):
        self.retriever = retriever
        self.agent = agent or NL2SQLAgent()
        self.table_loader = table_loader
        self._table_cache: Dict[str, TableInfo] = {}
        
        if db_manager is None:
            self.db_manager = _db_manager
        else:
            self.db_manager = db_manager
 
    def _execute_sql(
        self, 
        sql: str, 
        tables: List[TableInfo],
        selected_tables: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not sql or not tables:
            return None
        
        target_tables = tables
        if selected_tables:
            target_tables = [t for t in tables if t.table_key in selected_tables]
            if not target_tables:
                logger.warning(f"Selected tables {selected_tables} not found in recalled tables, using all")
                target_tables = tables
        
        db_names = list(set(t.db_name for t in target_tables))
        if len(db_names) != 1:
            logger.warning(f"Multiple databases detected: {db_names}, skipping execution")
            return {"error": "Multiple databases not supported for execution"}
        
        db_name = db_names[0]
        
        result = self.db_manager.execute(db_name, sql)
        
        if result is None:
            return {"error": "SQL execution failed"}
        
        return result

    def run(
        self,
        query: str,
        return_tables: bool = False,
        execute_sql: bool = True,
    ) -> Dict[str, Any]:
        if not query:
            raise ValueError("Query cannot be empty")

        if self.retriever is None:
            raise ValueError("Retriever is not configured")

        timings = {}
        total_start = time.time()

        start = time.time()
        recalled_results = self.retriever.retrieve(query)
        timings["table_recall"] = time.time() - start
        logger.info(f"Recalled {len(recalled_results)} tables for query: {query}")

        recalled_tables = []
        start = time.time()
        for r in recalled_results:
            if r.table:
                recalled_tables.append(r.table)
            elif r.columns:
                from nl2sql_v3.models.table_info import ColumnInfo, ForeignKeyRelation, RelatedTable
                
                columns = []
                for col in r.columns:
                    columns.append(ColumnInfo(
                        name=col.get("name", ""),
                        name_cn=col.get("name_cn", ""),
                        type=col.get("type", ""),
                        is_primary_key=col.get("is_primary_key", False),
                        is_foreign_key=col.get("is_foreign_key", False),
                    ))
                
                foreign_keys = []
                for fk in r.foreign_keys:
                    foreign_keys.append(ForeignKeyRelation(**fk))
                
                related_tables = []
                for rt in r.related_tables:
                    related_tables.append(RelatedTable(**rt))
                
                table_info = TableInfo(
                    db_name=r.db_name,
                    table_name=r.table_name,
                    table_name_cn=r.table_name_cn,
                    columns=columns,
                    primary_keys=r.primary_keys,
                    foreign_keys=foreign_keys,
                    related_tables=related_tables,
                )
                recalled_tables.append(table_info)
        timings["table_conversion"] = time.time() - start

        if not recalled_tables:
            return {
                "query": query,
                "sql": "",
                "confidence": 0.0,
                "explanation": "No relevant tables found for the query",
                "selected_tables": [],
                "used_columns": [],
                "recalled_tables": [],
                "timings": timings,
            }

        start = time.time()
        sql_result = self.agent.run(
            query=query,
            recalled_tables=recalled_tables,
        )
        timings["llm_sql_generation"] = time.time() - start

        execution_result = None
        if execute_sql and sql_result.sql and recalled_tables:
            start = time.time()
            execution_result = self._execute_sql(
                sql_result.sql, 
                recalled_tables,
                selected_tables=sql_result.selected_tables,
            )
            timings["sql_execution"] = time.time() - start

        timings["total"] = time.time() - total_start

        result = {
            "query": query,
            "sql": sql_result.sql,
            "confidence": sql_result.confidence,
            "explanation": sql_result.explanation,
            "selected_tables": sql_result.selected_tables,
            "used_columns": sql_result.used_columns,
            "recalled_tables": [t.table_key for t in recalled_tables],
            "timings": timings,
        }

        if execution_result is not None:
            result["execution_result"] = execution_result

        if return_tables:
            result["recalled_table_objects"] = recalled_tables

        return result


def create_agent(
    llm_provider: str = "openrouter",
    use_compact_prompt: bool = True,
    use_compact_schema: bool = True,
    **kwargs,
) -> NL2SQLAgent:
    schema_builder = CompactSchemaBuilder() if use_compact_schema else SchemaBuilder()
    prompt_template = CompactPromptTemplate if use_compact_prompt else PromptTemplate

    config = AgentConfig(
        schema_builder=schema_builder,
        prompt_template=prompt_template,
        llm_provider=llm_provider,
        **kwargs,
    )

    return NL2SQLAgent(config=config)
