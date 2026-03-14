import logging
import sys
import os
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from nl2sql_v3.recall.base import EvaluationResult, QueryRecord, RecallResult
from nl2sql_v3.recall.fusion import HybridRetriever
from nl2sql_v3.recall.keyword import keyword_recall
from nl2sql_v3.data.loader import QueryLoader, metadata_loader

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        use_keyword: bool = True,
        use_sparse: bool = True,
        use_dense: bool = True,
        weights: Optional[dict] = None,
    ):
        self.use_keyword = use_keyword
        self.use_sparse = use_sparse
        self.use_dense = use_dense
        self.weights = weights
        self.query_loader = QueryLoader()
        self.tables = metadata_loader.load()

    def evaluate(
        self,
        db_name: Optional[str] = None,
        top_k_values: Optional[List[int]] = None,
        filter_db: bool = False,
    ) -> EvaluationResult:
        if top_k_values is None:
            top_k_values = [1, 3, 5]

        queries = self.query_loader.load()
        start_time = time.time()
        if db_name:
            queries = [q for q in queries if q.db_name == db_name]

        if not queries:
            logger.warning("No queries to evaluate")
            return EvaluationResult(
                total_queries=0,
                hit_rate_at_1=0.0,
                hit_rate_at_3=0.0,
                hit_rate_at_5=0.0,
                mrr=0.0,
            )

        details = []
        hit_counts = {k: 0 for k in top_k_values}
        reciprocal_ranks = []

        for query in queries:
            result = self._evaluate_single_query(query, top_k_values, filter_db)
            details.append(result["detail"])
            for k in top_k_values:
                hit_counts[k] += result[f"hit@{k}"]
            reciprocal_ranks.append(result["mrr"])

        total = len(queries)
        total_time = time.time() - start_time
        avg_time = total_time / total if total > 0 else 0.0

        hit_rates = {k: hit_counts[k] / total for k in top_k_values}
        mrr = sum(reciprocal_ranks) / total if reciprocal_ranks else 0.0

        eval_result = EvaluationResult(
            total_queries=total,
            hit_rate_at_1=hit_rates.get(1, 0.0),
            hit_rate_at_3=hit_rates.get(3, 0.0),
            hit_rate_at_5=hit_rates.get(5, 0.0),
            mrr=mrr,
            details=details,
            total_time=total_time,
            avg_time=avg_time,
        )

        logger.info(f"Evaluation complete: Hit@1={eval_result.hit_rate_at_1:.2%}, Hit@3={eval_result.hit_rate_at_3:.2%}, Hit@5={eval_result.hit_rate_at_5:.2%}, MRR={eval_result.mrr:.2%}")
        return eval_result

    def _evaluate_single_query(
        self,
        query: QueryRecord,
        top_k_values: List[int],
        filter_db: bool = False,
    ) -> dict:
        if filter_db:
            db_tables = [t for t in self.tables if t.db_name == query.db_name]
        else:
            db_tables = self.tables
        if not db_tables:
            return {
                "question": query.question,
                "expected": query.tables,
                "actual": [],
                "detail": {},
                **{f"hit@{k}": 0 for k in top_k_values},
                "mrr": 0.0,
            }

        retriever = HybridRetriever(
            tables=db_tables,
            weights=self.weights,
            top_k=max(top_k_values),
            use_keyword=self.use_keyword,
            use_sparse=self.use_sparse,
            use_dense=self.use_dense,
        )

        results = retriever.retrieve(query.question)
        recalled_tables = [r.table_name for r in results]

        expected_set = set(query.tables)
        hit_result = {}
        mrr = 0.0

        for k in top_k_values:
            top_k_tables = set(recalled_tables[:k])
            hit_result[f"hit@{k}"] = 1 if expected_set & top_k_tables else 0

        for i, table in enumerate(recalled_tables):
            if table in expected_set:
                mrr = 1.0 / (i + 1)
                break

        return {
            "question": query.question,
            "expected": query.tables,
            "actual": recalled_tables,
            "detail": {
                "question": query.question,
                "expected": query.tables,
                "recalled": recalled_tables,
            },
            **hit_result,
            "mrr": mrr,
        }


def evaluate_keyword_only() -> EvaluationResult:
    queries = QueryLoader().load()
    tables = metadata_loader.load()

    hit_counts = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []

    for query in queries:
        db_tables = [t for t in tables if t.db_name == query.db_name]
        results = keyword_recall(query.question, db_tables)
        recalled_tables = [r.table_name for r in results]

        expected_set = set(query.tables)
        for k in [1, 3, 5]:
            top_k_tables = set(recalled_tables[:k])
            if expected_set & top_k_tables:
                hit_counts[k] += 1

        for i, table in enumerate(recalled_tables):
            if table in expected_set:
                reciprocal_ranks.append(1.0 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0.0)

    total = len(queries)
    return EvaluationResult(
        total_queries=total,
        hit_rate_at_1=hit_counts[1] / total,
        hit_rate_at_3=hit_counts[3] / total,
        hit_rate_at_5=hit_counts[5] / total,
        mrr=sum(reciprocal_ranks) / total,
    )
