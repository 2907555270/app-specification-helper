import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from nl2sql_v3.agent.schema_builder import (
    SchemaBuilder,
    CompactSchemaBuilder,
    ColumnSchema,
    TableSchema,
)
from nl2sql_v3.agent.prompts import (
    PromptTemplate,
    CompactPromptTemplate,
    SQLGenerationResult,
)
from nl2sql_v3.agent.llm_client import (
    LangChainLLMClient,
    get_llm_client,
    LLMResponse,
)
from nl2sql_v3.agent.agent import (
    NL2SQLAgent,
    NL2SQLPipeline,
    AgentConfig,
    create_agent,
)
from nl2sql_v3.models.table_info import TableInfo, ColumnInfo


class TestSchemaBuilder:
    def test_column_schema_creation(self):
        col = ColumnSchema(
            name="user_id",
            type="int",
            comment="用户ID",
            is_primary_key=True,
        )
        assert col.name == "user_id"
        assert col.type == "int"
        assert col.is_primary_key is True

    def test_table_schema_creation(self):
        schema = TableSchema(
            db_name="test_db",
            table_name="users",
            table_comment="用户表",
            columns=[
                ColumnSchema(name="id", type="int", is_primary_key=True),
                ColumnSchema(name="name", type="varchar"),
            ],
            primary_keys=["id"],
        )
        assert schema.db_name == "test_db"
        assert len(schema.columns) == 2

    def test_schema_builder_build_schema_text(self):
        tables = [
            TableInfo(
                db_name="test_db",
                table_name="users",
                table_name_cn="用户表",
                columns=[
                    ColumnInfo(name="user_id", name_cn="用户ID", type="int", is_primary_key=True),
                    ColumnInfo(name="username", name_cn="用户名", type="varchar"),
                    ColumnInfo(name="email", name_cn="邮箱", type="varchar"),
                ],
                primary_keys=["user_id"],
            )
        ]

        builder = SchemaBuilder(max_columns_per_table=10, max_sample_rows=2)
        schema_text = builder.build_schema_text(tables)

        assert "test_db.users" in schema_text
        assert "用户表" in schema_text
        assert "user_id" in schema_text
        assert "username" in schema_text

    def test_compact_schema_builder(self):
        tables = [
            TableInfo(
                db_name="test_db",
                table_name="users",
                table_name_cn="用户表",
                columns=[
                    ColumnInfo(name="user_id", name_cn="用户ID", type="int", is_primary_key=True),
                    ColumnInfo(name="username", name_cn="用户名", type="varchar"),
                    ColumnInfo(name="email", name_cn="邮箱", type="varchar"),
                ],
                primary_keys=["user_id"],
            )
        ]

        builder = CompactSchemaBuilder()
        schema_text = builder.build_schema_text(tables)

        assert "## test_db.users" in schema_text
        assert "user_id: int [PK]" in schema_text


class TestPromptTemplate:
    def test_prompt_template_build_prompt(self):
        schema_text = "## db.users\nuser_id: int [PK]"
        query = "查询所有用户"

        messages = PromptTemplate.build_prompt(
            query=query,
            schema=schema_text,
            include_fewshot=True,
        )

        assert len(messages) > 0
        assert messages[0]["role"] == "system"
        assert "SQL" in messages[0]["content"]

        user_messages = [m for m in messages if m["role"] == "user"]
        assert any(query in m["content"] for m in user_messages)
        assert any(schema_text in m["content"] for m in user_messages)

    def test_compact_prompt_template(self):
        schema_text = "## db.users\nuser_id: int [PK]"
        query = "查询用户"

        messages = CompactPromptTemplate.build_prompt(
            query=query,
            schema=schema_text,
            include_fewshot=False,
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestLLMClient:
    @patch("nl2sql_v3.agent.llm_client.ChatOpenAI")
    def test_langchain_client_chat(self, mock_chat):
        mock_instance = Mock()
        mock_response = Mock()
        mock_response.content = "Hello"
        mock_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_instance

        client = LangChainLLMClient(api_key="test_key", base_url="http://test.com", model="test-model")
        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(messages)

        assert response.content == "Hello"
        mock_instance.invoke.assert_called_once()

    @patch("nl2sql_v3.agent.llm_client.ChatOpenAI")
    def test_langchain_client_json_output(self, mock_chat):
        mock_instance = Mock()
        mock_response = Mock()
        mock_response.content = '{"sql": "SELECT * FROM users", "confidence": 0.9}'
        mock_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_instance

        client = LangChainLLMClient(api_key="test_key", base_url="http://test.com", model="test-model")
        messages = [{"role": "user", "content": "Generate SQL"}]
        result = client.chat_with_json_output(messages)

        assert result["sql"] == "SELECT * FROM users"
        assert result["confidence"] == 0.9


class TestNL2SQLAgent:
    def test_agent_config_creation(self):
        config = AgentConfig()
        assert config.llm_provider == "openrouter"
        assert config.include_fewshot is True
        assert config.temperature == 0.0

    @patch("nl2sql_v3.agent.agent.get_llm_client")
    def test_agent_run_with_mock_llm(self, mock_get_client):
        mock_client = Mock()
        mock_client.chat_with_json_output.return_value = {
            "sql": "SELECT * FROM users",
            "confidence": 0.95,
            "explanation": "Simple select",
            "selected_tables": ["db.users"],
            "used_columns": ["user_id", "username"],
        }
        mock_get_client.return_value = mock_client

        tables = [
            TableInfo(
                db_name="db",
                table_name="users",
                columns=[
                    ColumnInfo(name="user_id", type="int", is_primary_key=True),
                    ColumnInfo(name="username", type="varchar"),
                ],
            )
        ]

        config = AgentConfig(llm_client=mock_client)
        agent = NL2SQLAgent(config=config)

        result = agent.run("查询所有用户", tables)

        assert result.sql == "SELECT * FROM users"
        assert result.confidence == 0.95
        assert "db.users" in result.selected_tables


class TestNL2SQLPipeline:
    @patch("nl2sql_v3.agent.agent.get_llm_client")
    def test_pipeline_run(self, mock_get_client):
        mock_client = Mock()
        mock_client.chat_with_json_output.return_value = {
            "sql": "SELECT * FROM users",
            "confidence": 0.9,
            "explanation": "Test",
            "selected_tables": ["db.users"],
            "used_columns": ["user_id"],
        }
        mock_get_client.return_value = mock_client

        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            Mock(
                db_name="db",
                table_name="users",
                table=TableInfo(
                    db_name="db",
                    table_name="users",
                    columns=[ColumnInfo(name="user_id", type="int")],
                ),
            )
        ]

        agent = NL2SQLAgent(config=AgentConfig(llm_client=mock_client))
        pipeline = NL2SQLPipeline(
            retriever=mock_retriever,
            agent=agent,
        )

        result = pipeline.run("查询用户")

        assert result["sql"] == "SELECT * FROM users"
        assert result["confidence"] == 0.9
        assert "db.users" in result["recalled_tables"]


class TestCreateAgent:
    @patch("nl2sql_v3.agent.agent.get_llm_client")
    def test_create_agent_with_compact_settings(self, mock_get_client):
        mock_get_client.return_value = Mock()

        agent = create_agent(
            llm_provider="openrouter",
            use_compact_prompt=True,
            use_compact_schema=True,
        )

        assert isinstance(agent, NL2SQLAgent)
        assert isinstance(agent.config.schema_builder, CompactSchemaBuilder)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
