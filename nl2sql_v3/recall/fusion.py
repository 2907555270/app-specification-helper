import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from nl2sql_v3.config import config
from nl2sql_v3.recall.base import RecallResult, TableInfo
from nl2sql_v3.recall.keyword import keyword_recall
from nl2sql_v3.recall.sparse import sparse_recall
from nl2sql_v3.recall.dense import dense_recall

logger = logging.getLogger(__name__)


class Fusion:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 5,
    ):
        self.weights = weights or {
            "keyword": config.recall.weights.keyword,
            "sparse": config.recall.weights.sparse,
            "dense": config.recall.weights.dense,
        }
        self.top_k = top_k

    def fuse(
        self,
        keyword_results: List[RecallResult],
        sparse_results: List[RecallResult],
        dense_results: List[RecallResult],
    ) -> List[RecallResult]:
        all_tables: Dict[str, Dict[str, float]] = {}

        for r in keyword_results:
            key = f"{r.db_name}.{r.table_name}"
            if key not in all_tables:
                all_tables[key] = {"keyword": 0, "sparse": 0, "dense": 0, "db_name": r.db_name, "table_name": r.table_name}
            all_tables[key]["keyword"] = r.score

        for r in sparse_results:
            key = f"{r.db_name}.{r.table_name}"
            if key not in all_tables:
                all_tables[key] = {"keyword": 0, "sparse": 0, "dense": 0, "db_name": r.db_name, "table_name": r.table_name}
            all_tables[key]["sparse"] = r.score

        for r in dense_results:
            key = f"{r.db_name}.{r.table_name}"
            if key not in all_tables:
                all_tables[key] = {"keyword": 0, "sparse": 0, "dense": 0, "db_name": r.db_name, "table_name": r.table_name}
            all_tables[key]["dense"] = r.score

        final_results = []
        for key, scores in all_tables.items():
            final_score = (
                self.weights.get("keyword", 0) * scores.get("keyword", 0) +
                self.weights.get("sparse", 0) * scores.get("sparse", 0) +
                self.weights.get("dense", 0) * scores.get("dense", 0)
            )
            match_types = []
            if scores.get("keyword", 0) > 0:
                match_types.append("keyword")
            if scores.get("sparse", 0) > 0:
                match_types.append("sparse")
            if scores.get("dense", 0) > 0:
                match_types.append("dense")

            final_results.append(
                RecallResult(
                    db_name=scores["db_name"],
                    table_name=scores["table_name"],
                    score=final_score,
                    match_type="hybrid" if len(match_types) > 1 else match_types[0] if match_types else "none",
                )
            )

        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:self.top_k]


class HybridRetriever:
    def __init__(
        self,
        tables: List[TableInfo],
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 5,
        use_keyword: bool = True,
        use_sparse: bool = True,
        use_dense: bool = True,
    ):
        self.tables = tables
        self.weights = weights
        self.top_k = top_k
        self.use_keyword = use_keyword
        self.use_sparse = use_sparse
        self.use_dense = use_dense
        self.fusion = Fusion(weights=weights, top_k=top_k)

    def retrieve(self, query: str) -> List[RecallResult]:
        keyword_results: List[RecallResult] = []
        sparse_results: List[RecallResult] = []
        dense_results: List[RecallResult] = []

        if self.use_keyword:
            keyword_results = keyword_recall(query, self.tables)
            logger.debug(f"Keyword recall: {len(keyword_results)} results")

        if self.use_sparse:
            try:
                sparse_results = sparse_recall(query, top_k=10)
                logger.debug(f"Sparse recall: {len(sparse_results)} results")
            except Exception as e:
                logger.warning(f"Sparse recall failed: {e}")

        if self.use_dense:
            try:
                dense_results = dense_recall(query, top_k=10)
                logger.debug(f"Dense recall: {len(dense_results)} results")
            except Exception as e:
                logger.warning(f"Dense recall failed: {e}")

        fused_results = self.fusion.fuse(keyword_results, sparse_results, dense_results)
        logger.info(f"Fusion complete: {len(fused_results)} results")

        return fused_results
