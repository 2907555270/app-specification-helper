import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nl2sql_v3.models.table_info import TableInfo

logger = logging.getLogger(__name__)


@dataclass
class ColumnSchema:
    name: str
    type: str
    comment: str = ""
    is_primary_key: bool = False
    is_foreign_key: bool = False


@dataclass
class TableSchema:
    db_name: str
    table_name: str
    table_comment: str = ""
    columns: List[ColumnSchema] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)
    sample_rows: List[Dict[str, Any]] = field(default_factory=list)


class SchemaBuilder:
    def __init__(
        self,
        max_columns_per_table: int = 30,
        max_sample_rows: int = 3,
        include_foreign_keys: bool = True,
    ):
        self.max_columns_per_table = max_columns_per_table
        self.max_sample_rows = max_sample_rows
        self.include_foreign_keys = include_foreign_keys

    def build_schema_text(
        self,
        tables: List[TableInfo],
        sample_rows: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> str:
        if not tables:
            return ""

        schema_parts = []
        for table in tables:
            table_schema = self._build_single_table_schema(table, sample_rows)
            schema_parts.append(self._format_table_schema(table_schema))

        return "\n\n".join(schema_parts)

    def _build_single_table_schema(
        self,
        table: TableInfo,
        sample_rows: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> TableSchema:
        columns = []
        for col in table.columns[: self.max_columns_per_table]:
            columns.append(
                ColumnSchema(
                    name=col.name,
                    type=col.type,
                    comment=col.name_cn or "",
                    is_primary_key=col.is_primary_key,
                    is_foreign_key=col.is_foreign_key,
                )
            )

        foreign_keys = []
        if self.include_foreign_keys:
            for fk in table.foreign_keys:
                foreign_keys.append(
                    {
                        "column": fk.source_column,
                        "references": f"{fk.target_db}.{fk.target_table}.{fk.target_column}",
                    }
                )

        table_key = table.table_key
        sample_data = sample_rows.get(table_key, []) if sample_rows else []

        return TableSchema(
            db_name=table.db_name,
            table_name=table.table_name,
            table_comment=table.table_name_cn or "",
            columns=columns,
            primary_keys=table.primary_keys,
            foreign_keys=foreign_keys,
            sample_rows=sample_data[: self.max_sample_rows],
        )

    def _format_table_schema(self, schema: TableSchema) -> str:
        parts = []

        table_header = f"### {schema.db_name}.{schema.table_name}"
        if schema.table_comment:
            table_header += f" ({schema.table_comment})"
        parts.append(table_header)

        if schema.primary_keys:
            parts.append(f"**主键**: {', '.join(schema.primary_keys)}")

        if schema.foreign_keys:
            fk_lines = []
            for fk in schema.foreign_keys:
                fk_lines.append(f"{fk['column']} -> {fk['references']}")
            parts.append(f"**外键**: {', '.join(fk_lines)}")

        col_lines = []
        for col in schema.columns:
            col_str = f"- {col.name} ({col.type})"
            if col.is_primary_key:
                col_str += " [PK]"
            if col.is_foreign_key:
                col_str += " [FK]"
            if col.comment:
                col_str += f": {col.comment}"
            col_lines.append(col_str)

        parts.append("\n".join(col_lines))

        if schema.sample_rows:
            parts.append("\n**示例数据**:")
            for row in schema.sample_rows:
                row_str = ", ".join(f"{k}={v}" for k, v in row.items() if v is not None)
                parts.append(f"- {row_str}")

        return "\n".join(parts)


class CompactSchemaBuilder(SchemaBuilder):
    def __init__(self, max_columns_per_table: int = 15, max_sample_rows: int = 2):
        super().__init__(
            max_columns_per_table=max_columns_per_table,
            max_sample_rows=max_sample_rows,
            include_foreign_keys=True,
        )

    def _format_table_schema(self, schema: TableSchema) -> str:
        parts = []

        table_header = f"## {schema.db_name}.{schema.table_name}"
        if schema.table_comment:
            table_header += f" - {schema.table_comment}"
        parts.append(table_header)

        col_lines = []
        for col in schema.columns:
            pk_marker = "PK" if col.is_primary_key else ""
            fk_marker = "FK" if col.is_foreign_key else ""
            markers = " ".join(filter(None, [pk_marker, fk_marker]))
            marker_str = f" [{markers}]" if markers else ""

            comment_str = f" - {col.comment}" if col.comment else ""
            col_lines.append(f"{col.name}: {col.type}{marker_str}{comment_str}")

        parts.append(", ".join(col_lines))

        return "\n".join(parts)
