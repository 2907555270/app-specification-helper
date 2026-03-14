import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from nl2sql_v3.config import config
from nl2sql_v3.recall.base import RecallResult, TableInfo
from nl2sql_v3.client.es_client import es_client
from nl2sql_v3.client.api_client import sparse_vector_client, dense_vector_client, rerank_client

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        tables: List[TableInfo],
        weights: Optional[Dict[str, float]] = None,
        use_keyword: bool = True,
        use_sparse: bool = True,
        use_dense: bool = True,
        filter_db_name: Optional[str] = None,
        use_rerank: Optional[bool] = None,
    ):
        self.tables = tables
        self.weights = weights or {
            "keyword": config.recall.weights.keyword,
            "sparse": config.recall.weights.sparse,
            "dense": config.recall.weights.dense,
        }
        self.use_keyword = use_keyword
        self.use_sparse = use_sparse
        self.use_dense = use_dense
        self.filter_db_name = filter_db_name
        self.use_rerank = use_rerank if use_rerank is not None else config.recall.rerank_enabled
        self.rerank_top_k = config.recall.rerank_top_k
        self.rerank_threshold = config.recall.rerank_threshold
        self.hybrid_search_top_k = config.recall.hybrid_search_top_k

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
                size=self.hybrid_search_top_k,
                filter_db_name=target_db_name,
            )
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

        recall_results = []
        for doc in results:
            db_name = doc.get("db_name", "")
            table_name = doc.get("table_name", "")
            all_names = doc.get("all_names", "")

            if (db_name, table_name) in table_set:
                recall_results.append(
                    RecallResult(
                        db_name=db_name,
                        table_name=table_name,
                        all_names=all_names,
                        score=doc.get("_score", 0.0),
                        match_type="hybrid",
                    )
                )

        if self.use_rerank and recall_results:
            recall_results = self._rerank(query, recall_results)

        recall_results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Hybrid retrieval complete: {len(recall_results)} results")
        return recall_results

    def _rerank(self, query: str, results: List[RecallResult]) -> List[RecallResult]:
        if not results:
            return results

        table_docs = []
        for r in results:
            table_docs.append(f"{r.all_names}")

        try:
            rerank_result = rerank_client.rerank(
                query=query,
                documents=table_docs,
                top_k=self.rerank_top_k,
            )

            scores = rerank_result.get("scores", [])
            rankings = rerank_result.get("rankings", [])

            for idx, rank in enumerate(rankings):
                if 1 <= rank <= len(results):
                    score = scores[idx] if idx < len(scores) else 0.0
                    results[rank - 1].rerank_score = score

            results = [r for r in results if r.rerank_score >= self.rerank_threshold]
            results = results[:self.rerank_top_k]

        except Exception as e:
            logger.warning(f"Rerank failed: {e}")

        return results
