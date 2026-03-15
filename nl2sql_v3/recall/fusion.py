import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from nl2sql_v3.config import config
from nl2sql_v3.recall.base import RecallResult
from nl2sql_v3.client.es_client import es_client
from nl2sql_v3.client.api_client import sparse_vector_client, dense_vector_client, rerank_client, bge3_client

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
        use_keyword: bool = True,
        use_sparse: bool = True,
        use_dense: bool = True,
        use_rrf: Optional[bool] = None,
        use_rerank: Optional[bool] = None,
        use_bge3: bool = True,
    ):
        self.weights = weights or {
            "keyword": config.recall.weights.keyword,
            "sparse": config.recall.weights.sparse,
            "dense": config.recall.weights.dense,
        }
        self.top_k = top_k if top_k is not None else config.recall.top_k
        self.use_keyword = use_keyword
        self.use_sparse = use_sparse
        self.use_dense = use_dense
        self.use_rrf = use_rrf if use_rrf is not None else config.recall.rrf_enabled
        self.use_rerank = use_rerank if use_rerank is not None else config.recall.rerank_enabled
        self.use_bge3 = use_bge3
        self.rerank_top_k = config.recall.rerank_top_k
        self.rerank_threshold = config.recall.rerank_threshold
        self.hybrid_search_top_k = config.recall.hybrid_search_top_k

    def retrieve(self, query: str, filter_db_name: Optional[str] = None) -> List[RecallResult]:
        if not query:
            return []

        keyword_query = query if self.use_keyword else None
        sparse_vector = None
        dense_vector = None

        if self.use_bge3 and (self.use_sparse or self.use_dense):
            try:
                bge_result = bge3_client.encode(
                    query,
                    dense_output=self.use_dense,
                    sparse_output=self.use_sparse,
                )
                if self.use_dense:
                    dense_vecs = bge_result.get("dense_vecs", [])
                    dense_vector = dense_vecs[0] if dense_vecs else None
                if self.use_sparse:
                    sparse_vecs = bge_result.get("sparse_vecs", [])
                    sparse_vector = sparse_vecs[0] if sparse_vecs else None
            except Exception as e:
                logger.warning(f"BGE3 encoding failed: {e}")
        else:
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
                filter_db_name=filter_db_name,
                use_rrf=self.use_rrf,
            )
        except Exception as e:
            logger.error(f"Hybrid search failed: {e.with_traceback()}")
            return []

        recall_results = []
        for doc in results:
            recall_results.append(
                RecallResult(
                    db_name=doc.get("db_name", ""),
                    table_name=doc.get("table_name", ""),
                    all_names=doc.get("all_names", ""),
                    score=doc.get("_score", 0.0),
                    rerank_score=-float("inf"),
                    rrf_score=doc.get("rrf_score", 0.0),
                    match_type="hybrid",
                )
            )

        if self.use_rerank and recall_results:
            recall_results = self._rerank(query, recall_results)
            recall_results.sort(key=lambda x: x.rerank_score, reverse=True)
        else:
            recall_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Hybrid retrieval complete: {len(recall_results)} results")
        return recall_results[:self.top_k]

    def _rerank(self, query: str, results: List[RecallResult]) -> List[RecallResult]:
        if not results:
            return results

        logger.info(f"Rerank enabled, processing {len(results)} results")

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

            for score, rank in zip(scores, rankings):
                if 1 <= rank <= len(results):
                    results[rank - 1].rerank_score = score

            logger.debug(f"rerank threshold: {self.rerank_threshold}")
            results = [r for r in results if r.rerank_score and r.rerank_score > -float("inf") and r.rerank_score >= self.rerank_threshold]
            results = results[:self.rerank_top_k]
            logger.debug(f"Rerank complete: {results[0]}")

        except Exception as e:
            logger.warning(f"Rerank failed: {e}")

        return results
