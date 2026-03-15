from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RecallResult(BaseModel):
    db_name: str
    table_name: str
    all_names: str
    score: float
    match_type: str
    rrf_score: float
    rerank_score: float

class TableMatch(BaseModel):
    db_name: str
    table_name: str
    score: float
    match_type: str


class EvaluationResult(BaseModel):
    total_queries: int
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    mrr: float
    total_time: float = 0.0
    avg_time: float = 0.0
    details: List[dict] = []
    min_rerank_threshold: Optional[float] = None


class QueryRecord(BaseModel):
    db_name: str
    question: str
    tables: List[str]
