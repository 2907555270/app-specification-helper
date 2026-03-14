import re
import logging
import sys
import os
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from nl2sql_v3.recall.base import RecallResult, TableInfo

logger = logging.getLogger(__name__)


def extract_keywords(text: str) -> List[str]:
    text = text.lower()
    words = re.findall(r"[a-zA-Z]+", text)
    keywords = [w for w in words if len(w) > 2]
    return keywords


def keyword_recall(
    query: str,
    tables: List[TableInfo],
    threshold: float = 0.5,
) -> List[RecallResult]:
    keywords = extract_keywords(query)
    query_lower = query.lower()

    results = []
    for table in tables:
        score = 0.0

        table_name_lower = table.table_name.lower()
        if table_name_lower in query_lower:
            score += 1.0
        elif any(kw in table_name_lower for kw in keywords):
            score += 0.5

        for col in table.columns:
            col_lower = col.lower()
            if col_lower in query_lower:
                score += 0.3
            elif any(kw in col_lower for kw in keywords):
                score += 0.1

        for kw in keywords:
            if kw in table_name_lower or any(kw in col.lower() for col in table.columns):
                score += 0.2

        if score >= threshold:
            results.append(
                RecallResult(
                    db_name=table.db_name,
                    table_name=table.table_name,
                    score=score,
                    match_type="keyword",
                )
            )

    results.sort(key=lambda x: x.score, reverse=True)
    return results


class KeywordRecaller:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def recall(
        self,
        query: str,
        tables: List[TableInfo],
    ) -> List[RecallResult]:
        return keyword_recall(query, tables, self.threshold)
