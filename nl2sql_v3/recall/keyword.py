import logging
import sys
import os
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from nl2sql_v3.recall.base import RecallResult, TableInfo
from nl2sql_v3.client.es_client import es_client

logger = logging.getLogger(__name__)


def keyword_recall(
    query: str,
    tables: List[TableInfo],
    threshold: float = 0.0,
    top_k: int = 10,
) -> List[RecallResult]:
    if not query or not tables:
        return []

    db_names = set(t.db_name for t in tables)
    
    results = []
    for db_name in db_names:
        filter_query = {"term": {"db_name": db_name}}
        hits = es_client.search(
            query=query,
            fields=["all_names", "table_name"],
            filter_query=filter_query,
            size=top_k,
        )
        
        for hit in hits:
            table_name = hit.get("table_name", "")
            if any(t.db_name == db_name and t.table_name == table_name for t in tables):
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
    def __init__(self, threshold: float = 0.0, top_k: int = 10):
        self.threshold = threshold
        self.top_k = top_k

    def recall(
        self,
        query: str,
        tables: List[TableInfo],
    ) -> List[RecallResult]:
        return keyword_recall(query, tables, self.threshold, self.top_k)
