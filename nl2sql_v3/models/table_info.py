from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
    USE_PYDANTIC = True
except ImportError:
    from dataclasses import dataclass, field as dc_field
    from typing import List as TypingList
    BaseModel = object
    
    class Field:
        def __init__(self, default_factory=None, **kwargs):
            self.default_factory = default_factory or (lambda: [])
    
    def _make_field(default_factory):
        return default_factory
    
    USE_PYDANTIC = False


if USE_PYDANTIC:
    class ColumnInfo(BaseModel):
        name: str = ""
        name_cn: str = ""
        type: str = ""
        is_primary_key: bool = False
        is_foreign_key: bool = False

    class ForeignKeyRelation(BaseModel):
        source_db: str = ""
        source_table: str = ""
        source_column: str = ""
        target_db: str = ""
        target_table: str = ""
        target_column: str = ""

        @property
        def source_key(self) -> str:
            return f"{self.source_db}.{self.source_table}"

        @property
        def target_key(self) -> str:
            return f"{self.target_db}.{self.target_table}"

    class RelatedTable(BaseModel):
        table_name: str = ""
        db_name: str = ""
        join_columns: List[str] = Field(default_factory=list)

    class TableInfo(BaseModel):
        db_name: str
        table_name: str
        table_name_cn: str = ""
        columns: List[ColumnInfo] = Field(default_factory=list)
        primary_keys: List[str] = Field(default_factory=list)
        foreign_keys: List[ForeignKeyRelation] = Field(default_factory=list)
        related_tables: List[RelatedTable] = Field(default_factory=list)

        @property
        def table_key(self) -> str:
            return f"{self.db_name}.{self.table_name}"

        def to_es_doc(self) -> Dict[str, Any]:
            return {
                "db_name": self.db_name,
                "table_name": self.table_name,
                "table_name_cn": self.table_name_cn,
                "all_names": self._build_all_names(),
                "columns_name": [col.name for col in self.columns],
                "columns_name_cn": [col.name_cn for col in self.columns],
                "columns": [col.model_dump() for col in self.columns],
                "primary_keys": self.primary_keys,
                "foreign_keys": [fk.model_dump() for fk in self.foreign_keys],
                "related_tables": [rt.model_dump() for rt in self.related_tables],
            }

        def _build_all_names(self) -> str:
            parts = [self.db_name, self.table_name, self.table_name_cn]
            for col in self.columns:
                parts.extend([col.name, col.name_cn])
            return " ".join(filter(None, parts))
else:
    @dataclass
    class ColumnInfo:
        name: str = ""
        name_cn: str = ""
        type: str = ""
        is_primary_key: bool = False
        is_foreign_key: bool = False

    @dataclass
    class ForeignKeyRelation:
        source_db: str = ""
        source_table: str = ""
        source_column: str = ""
        target_db: str = ""
        target_table: str = ""
        target_column: str = ""

        @property
        def source_key(self) -> str:
            return f"{self.source_db}.{self.source_table}"

        @property
        def target_key(self) -> str:
            return f"{self.target_db}.{self.target_table}"

    @dataclass
    class RelatedTable:
        table_name: str = ""
        db_name: str = ""
        join_columns: TypingList[str] = dc_field(default_factory=list)

    @dataclass
    class TableInfo:
        db_name: str
        table_name: str
        table_name_cn: str = ""
        columns: TypingList[ColumnInfo] = dc_field(default_factory=list)
        primary_keys: TypingList[str] = dc_field(default_factory=list)
        foreign_keys: TypingList[ForeignKeyRelation] = dc_field(default_factory=list)
        related_tables: TypingList[RelatedTable] = dc_field(default_factory=list)

        @property
        def table_key(self) -> str:
            return f"{self.db_name}.{self.table_name}"

        def to_es_doc(self) -> Dict[str, Any]:
            return {
                "db_name": self.db_name,
                "table_name": self.table_name,
                "table_name_cn": self.table_name_cn,
                "all_names": self._build_all_names(),
                "columns_name": [col.name for col in self.columns],
                "columns_name_cn": [col.name_cn for col in self.columns],
                "columns": [{"name": c.name, "name_cn": c.name_cn, "type": c.type,
                             "is_primary_key": c.is_primary_key, "is_foreign_key": c.is_foreign_key}
                            for c in self.columns],
                "primary_keys": self.primary_keys,
                "foreign_keys": [{"source_db": fk.source_db, "source_table": fk.source_table,
                                  "source_column": fk.source_column, "target_db": fk.target_db,
                                  "target_table": fk.target_table, "target_column": fk.target_column}
                                 for fk in self.foreign_keys],
                "related_tables": [{"table_name": rt.table_name, "db_name": rt.db_name,
                                   "join_columns": rt.join_columns}
                                  for rt in self.related_tables],
            }

        def _build_all_names(self) -> str:
            parts = [self.db_name, self.table_name, self.table_name_cn]
            for col in self.columns:
                parts.extend([col.name, col.name_cn])
            return " ".join(filter(None, parts))
