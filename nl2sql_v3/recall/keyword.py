import logging
import sys
import os
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from nl2sql_v3.recall.base import RecallResult, TableInfo
from nl2sql_v3.client.es_client import es_client
from nl2sql_v3.config import config

logger = logging.getLogger(__name__)


def keyword_recall(
    query: str,
    tables: List[TableInfo],
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
) -> List[RecallResult]:
    if not query or not tables:
        return []

    threshold = threshold if threshold is not None else config.recall.keyword_threshold
    top_k = top_k if top_k is not None else config.recall.top_k

    table_set = {(t.db_name, t.table_name) for t in tables}
    
    hits = es_client.bm25_search(
        query=query,
        size=top_k,
    )
    
    results = []
    for hit in hits:
        db_name = hit.get("db_name", "")
        table_name = hit.get("table_name", "")
        
        if (db_name, table_name) in table_set:
            score = hit.get("_score", 0) or 0.0
            if score > threshold:
                results.append(
                    RecallResult(
                        db_name=db_name,
                        table_name=table_name,
                        score=score,
                        match_type="keyword",
                    )
                )

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]


class KeywordRecaller:
    def __init__(self, threshold: Optional[float] = None, top_k: Optional[int] = None):
        self.threshold = threshold if threshold is not None else config.recall.keyword_threshold
        self.top_k = top_k if top_k is not None else config.recall.top_k

    def recall(
        self,
        query: str,
        tables: List[TableInfo],
    ) -> List[RecallResult]:
        return keyword_recall(query, tables, self.threshold, self.top_k)
