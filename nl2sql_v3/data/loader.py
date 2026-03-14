import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from nl2sql_v3.client.api_client import dense_vector_client, sparse_vector_client
from nl2sql_v3.config import config
from nl2sql_v3.recall.base import QueryRecord, TableInfo

logger = logging.getLogger(__name__)


class MetadataLoader:
    def __init__(self, metadata_path: Optional[str] = None):
        if metadata_path:
            self.metadata_path = Path(metadata_path)
        else:
            self.metadata_path = config.data.get_metadata_path()
        self._tables: Optional[List[TableInfo]] = None

    def load(self) -> List[TableInfo]:
        if self._tables is not None:
            return self._tables

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        tables = []
        for item in raw_data:
            parts = item.strip().split("\n")
            if len(parts) < 2:
                continue

            db_name = parts[0].strip()
            table_name = parts[1].strip()
            columns = [col.strip() for col in parts[2:] if col.strip()]

            tables.append(
                TableInfo(db_name=db_name, table_name=table_name, columns=columns)
            )

        self._tables = tables
        logger.info(f"Loaded {len(tables)} tables from metadata")
        return tables

    def get_table(self, db_name: str, table_name: str) -> Optional[TableInfo]:
        tables = self.load()
        for table in tables:
            if table.db_name == db_name and table.table_name == table_name:
                return table
        return None

    def get_tables_by_db(self, db_name: str) -> List[TableInfo]:
        tables = self.load()
        return [t for t in tables if t.db_name == db_name]


class QueryLoader:
    def __init__(self, queries_path: Optional[str] = None):
        if queries_path:
            self.queries_path = Path(queries_path)
        else:
            self.queries_path = config.data.get_queries_path()
        self._queries: Optional[List[QueryRecord]] = None

    def load(self) -> List[QueryRecord]:
        if self._queries is not None:
            return self._queries

        if not self.queries_path.exists():
            raise FileNotFoundError(f"Queries file not found: {self.queries_path}")

        with open(self.queries_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        queries = []
        for item in raw_data:
            queries.append(
                QueryRecord(
                    db_name=item["db_name"],
                    question=item["question"],
                    tables=item["tables"],
                )
            )

        self._queries = queries
        logger.info(f"Loaded {len(queries)} queries")
        return queries

    def get_queries_by_db(self, db_name: str) -> List[QueryRecord]:
        queries = self.load()
        return [q for q in queries if q.db_name == db_name]


def build_index_documents(
    tables: List[TableInfo],
    use_sparse: bool = True,
    use_dense: bool = True,
) -> List[Dict]:
    documents = []
    for table in tables:
        doc = {
            "db_name": table.db_name,
            "table_name": table.table_name,
            "all_names": f"{table.db_name} {table.table_name} {' '.join(table.columns)}",
        }

        if use_sparse:
            try:
                sparse_result = sparse_vector_client.encode(
                    doc["all_names"], query_mode=False
                )
                doc["sparse_vector"] = sparse_result.get("sparse_id_vector", {})
            except Exception as e:
                logger.warning(f"Failed to generate sparse vector for {table.table_name}: {e}")
                doc["sparse_vector"] = {}

        if use_dense:
            try:
                dense_result = dense_vector_client.encode(doc["all_names"])
                doc["dense_vector"] = dense_result
            except Exception as e:
                logger.warning(f"Failed to generate dense vector for {table.table_name}: {e}")
                doc["dense_vector"] = []

        documents.append(doc)

    return documents


def stream_build_index_documents(
    tables: List[TableInfo],
    use_sparse: bool = True,
    use_dense: bool = True,
) -> Generator[Dict, None, None]:
    for table in tables:
        doc = {
            "db_name": table.db_name,
            "table_name": table.table_name,
            "all_names": f"{table.db_name} {table.table_name} {' '.join(table.columns)}",
        }
        if use_sparse:
            try:
                sparse_result = sparse_vector_client.encode(
                    doc["all_names"], query_mode=False
                )
                doc["sparse_vector"] = sparse_result.get("sparse_id_vector", {})
            except Exception as e:
                logger.warning(f"Failed to generate sparse vector for {table.table_name}: {e}")
                doc["sparse_vector"] = {}

        if use_dense:
            try:
                dense_result = dense_vector_client.encode(doc["all_names"])
                doc["dense_vector"] = dense_result
            except Exception as e:
                logger.warning(f"Failed to generate dense vector for {table.table_name}: {e}")
                doc["dense_vector"] = []

        yield doc


metadata_loader = MetadataLoader()
query_loader = QueryLoader()
