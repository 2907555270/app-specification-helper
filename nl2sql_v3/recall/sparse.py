import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from nl2sql_v3.client.api_client import sparse_vector_client, bge3_client
from nl2sql_v3.client.es_client import es_client
from nl2sql_v3.recall.base import RecallResult
from nl2sql_v3.config import config

logger = logging.getLogger(__name__)


class SparseRecaller:
    def __init__(self, top_k: Optional[int] = None, use_bge3: bool = True):
        self.top_k = top_k if top_k is not None else config.recall.top_k
        self.use_bge3 = use_bge3

    def recall(self, query: str) -> List[RecallResult]:
        try:
            if self.use_bge3:
                bge_result = bge3_client.encode(query, dense_output=False, sparse_output=True)
                sparse_vecs = bge_result.get("sparse_vecs", [])
                sparse_vector = sparse_vecs[0] if sparse_vecs else {}
            else:
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

            return recall_results

        except Exception as e:
            logger.error(f"Sparse recall failed: {e}")
            return []


def sparse_recall(query: str, top_k: Optional[int] = None, use_bge3: bool = True) -> List[RecallResult]:
    recaller = SparseRecaller(top_k=top_k, use_bge3=use_bge3)
    return recaller.recall(query)
