import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from nl2sql_v3.client.api_client import sparse_vector_client
from nl2sql_v3.client.es_client import es_client
from nl2sql_v3.recall.base import RecallResult
from nl2sql_v3.config import config

logger = logging.getLogger(__name__)


class SparseRecaller:
    def __init__(self, top_k: Optional[int] = None):
        self.top_k = top_k if top_k is not None else config.recall.top_k

    def recall(self, query: str) -> List[RecallResult]:
        try:
            sparse_result = sparse_vector_client.encode(query, query_mode=True)
            sparse_vector = sparse_result.get("sparse_id_vector", {})

            if not sparse_vector:
                logger.warning("Empty sparse vector returned")
                return []

            results = es_client.sparse_vector_search(
                field="sparse_vector",
                sparse_vector=sparse_vector,
                k=self.top_k,
            )

            recall_results = []
            for doc in results:
                recall_results.append(
                    RecallResult(
                        db_name=doc.get("db_name", ""),
                        table_name=doc.get("table_name", ""),
                        score=doc.get("_score", 0.0),
                        match_type="sparse",
                    )
                )

            if recall_results:
                scores = [r.score for r in recall_results if r.score]
                if scores:
                    max_score = max(scores)
                    if max_score > 0:
                        for r in recall_results:
                            r.score = r.score / max_score

            return recall_results

        except Exception as e:
            logger.error(f"Sparse recall failed: {e}")
            return []


def sparse_recall(query: str, top_k: Optional[int] = None) -> List[RecallResult]:
    recaller = SparseRecaller(top_k=top_k)
    return recaller.recall(query)
