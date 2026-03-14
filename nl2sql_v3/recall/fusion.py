import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from nl2sql_v3.config import config
from nl2sql_v3.recall.base import RecallResult, TableInfo
from nl2sql_v3.client.es_client import es_client
from nl2sql_v3.client.api_client import sparse_vector_client, dense_vector_client

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        tables: List[TableInfo],
        weights: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
        use_keyword: bool = True,
        use_sparse: bool = True,
        use_dense: bool = True,
        filter_db_name: Optional[str] = None,
    ):
        self.tables = tables
        self.weights = weights or {
            "keyword": config.recall.weights.keyword,
            "sparse": config.recall.weights.sparse,
            "dense": config.recall.weights.dense,
        }
        self.top_k = top_k if top_k is not None else config.recall.top_k
        self.use_keyword = use_keyword
        self.use_sparse = use_sparse
        self.use_dense = use_dense
        self.filter_db_name = filter_db_name

    def retrieve(self, query: str, filter_db_name: Optional[str] = None) -> List[RecallResult]:
        if not query:
            return []

        target_db_name = filter_db_name or self.filter_db_name
        
        if target_db_name:
            db_tables = [t for t in self.tables if t.db_name == target_db_name]
            table_set = {(t.db_name, t.table_name) for t in db_tables}
        else:
            table_set = {(t.db_name, t.table_name) for t in self.tables}
        
        keyword_query = query if self.use_keyword else None
        sparse_vector = None
        dense_vector = None

        if self.use_sparse:
            try:
                sparse_result = sparse_vector_client.encode(query, query_mode=True)
                sparse_vector = sparse_result.get("sparse_id_vector", {})
            except Exception as e:
                logger.warning(f"Sparse vector encoding failed: {e}")

        if self.use_dense:
            try:
                dense_vector = dense_vector_client.encode(query)
            except Exception as e:
                logger.warning(f"Dense vector encoding failed: {e}")

        try:
            results = es_client.hybrid_search(
                query=keyword_query,
                sparse_vector=sparse_vector if sparse_vector else None,
                dense_vector=dense_vector if dense_vector else None,
                keyword_weight=self.weights.get("keyword"),
                sparse_weight=self.weights.get("sparse"),
                dense_weight=self.weights.get("dense"),
                size=self.top_k,
                filter_db_name=target_db_name,
            )
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

        recall_results = []
        for doc in results:
            db_name = doc.get("db_name", "")
            table_name = doc.get("table_name", "")

            if (db_name, table_name) in table_set:
                recall_results.append(
                    RecallResult(
                        db_name=db_name,
                        table_name=table_name,
                        score=doc.get("_score", 0.0),
                        match_type="hybrid",
                    )
                )

        recall_results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Hybrid retrieval complete: {len(recall_results)} results")
        return recall_results[:self.top_k]
