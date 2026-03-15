import logging
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from elasticsearch import Elasticsearch
from ranx import Run

from nl2sql_v3.config import config

logger = logging.getLogger(__name__)


class ESClient:
    def __init__(self):
        self.hosts = config.services.elasticsearch.hosts
        self.index = config.services.elasticsearch.index
        self.client = Elasticsearch(
            self.hosts,
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3,
        )
        logger.info(f"ES client initialized with hosts: {self.hosts}")

    def create_index(
        self,
        index_name: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        target_index = index_name or self.index
        if self.client.indices.exists(index=target_index):
            if force:
                self.client.indices.delete(index=target_index)
                logger.info(f"Deleted existing index: {target_index}")
            else:
                logger.info(f"Index {target_index} already exists")
                return False

        mapping = {
            "mappings": {
                "properties": {
                    "db_name": {"type": "keyword"},
                    "table_name": {"type": "keyword"},
                    "all_names": {
                        "type": "text",
                        "analyzer": "standard",
                    },
                    "sparse_vector": {
                        "type": "sparse_vector",
                    },
                    "dense_vector": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        }

        self.client.indices.create(index=target_index, body=mapping)
        logger.info(f"Created index: {target_index}")
        return True

    def bulk_index(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        if not documents:
            return 0

        total_indexed = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            actions = []
            for doc in batch:
                actions.append({"index": {"_index": self.index}})
                actions.append(doc)

            response = self.client.bulk(operations=actions, refresh=True)
            
            if response.get("errors"):
                for item in response["items"]:
                    if "error" in item.get("index", {}):
                        logger.error(f"Index error: {item['index']['error']}")
            
            total_indexed += len(batch)
            logger.info(f"Indexed {total_indexed}/{len(documents)} documents")

        return total_indexed

    def search(
        self,
        query: Optional[Dict[str, Any]] = None,
        knn: Optional[Dict[str, Any]] = None,
        size: int = 10,
    ) -> List[Dict[str, Any]]:
        if not query and not knn:
            raise ValueError("Either query or knn must be provided")

        body: Dict[str, Any] = {"size": size}
        if query:
            body["query"] = query
        if knn:
            body["knn"] = knn

        response = self.client.search(index=self.index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        return [hit["_source"] for hit in hits]

    def bm25_search(
        self,
        query: str,
        size: int = 10,
    ) -> List[Dict]:
        body = {
            "size": size,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["all_names^2", "table_name^3", "columns_name"],
                }
            },
        }
        response = self.client.search(index=self.index, body=body)
        hits = response["hits"]["hits"]
        return [{"_score": hit["_score"], **hit["_source"]} for hit in hits]

    def knn_search(
        self,
        field: str,
        vector: List[float],
        k: int = 10,
        filter_query: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        knn_body = {
            "field": field,
            "query_vector": vector,
            "k": k,
            "similarity": 0.7,
        }

        query: Dict[str, Any] = {"knn": knn_body, "size": k}
        if filter_query:
            query["query"] = filter_query

        response = self.client.search(index=self.index, body=query)
        hits = response.get("hits", {}).get("hits", [])
        return [hit["_source"] for hit in hits]

    def sparse_vector_search(
        self,
        field: str,
        sparse_vector: Dict[str, float],
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        query = {
            "query": {
                "sparse_vector": {
                    "field": field,
                    "query_vector": sparse_vector,
                }
            },
            "size": k,
        }
        response = self.client.search(index=self.index, body=query)
        hits = response.get("hits", {}).get("hits", [])
        return [hit["_source"] for hit in hits]

    def hybrid_search(
        self,
        query: str,
        sparse_vector: Optional[Dict[str, float]] = None,
        dense_vector: Optional[List[float]] = None,
        keyword_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        dense_weight: Optional[float] = None,
        size: int = 10,
        filter_db_name: Optional[str] = None,
        use_rrf: bool = False,
    ) -> List[Dict[str, Any]]:
        keyword_w = keyword_weight if keyword_weight is not None else config.recall.weights.keyword
        sparse_w = sparse_weight if sparse_weight is not None else config.recall.weights.sparse
        dense_w = dense_weight if dense_weight is not None else config.recall.weights.dense

        base_filter = []
        if filter_db_name:
            base_filter.append({"term": {"db_name": filter_db_name}})

        if use_rrf:
            runs_list: List[Run] = []

            start_time = time.time()
            if query:
                keyword_results = self.keyword_search(query, size=size, filter_db_name=filter_db_name)
                if keyword_results:
                    keyword_run_dict = {"q1": {}}
                    for i, doc in enumerate(keyword_results):
                        doc_id = f"{doc.get('db_name', '')}_{doc.get('table_name', '')}"
                        keyword_run_dict["q1"][doc_id] = doc.get("_score", 0.0) * keyword_w
                    runs_list.append(Run(keyword_run_dict))

            if sparse_vector:
                sparse_results = self.sparse_search(sparse_vector, size=size, filter_db_name=filter_db_name)
                if sparse_results:
                    sparse_run_dict = {"q1": {}}
                    for i, doc in enumerate(sparse_results):
                        doc_id = f"{doc.get('db_name', '')}_{doc.get('table_name', '')}"
                        sparse_run_dict["q1"][doc_id] = doc.get("_score", 0.0) * sparse_w
                    runs_list.append(Run(sparse_run_dict))

            if dense_vector:
                dense_results = self.dense_search(dense_vector, size=size, filter_db_name=filter_db_name)
                if dense_results:
                    dense_run_dict = {"q1": {}}
                    for i, doc in enumerate(dense_results):
                        doc_id = f"{doc.get('db_name', '')}_{doc.get('table_name', '')}"
                        dense_run_dict["q1"][doc_id] = doc.get("rrf_score", 0.0) * dense_w
                    runs_list.append(Run(dense_run_dict))

            if not runs_list:
                raise ValueError("At least one of query, sparse_vector, or dense_vector must be provided")
            logger.info(f"RRF runs list prepare, include es search time: {time.time() - start_time}")

            start_time = time.time()
            raw_runs = [run.run["q1"] for run in runs_list]
            fused_results = self.manual_rrf(raw_runs, k=60)
            logger.info(f"RRF fused search time: {time.time() - start_time}")

            doc_id_to_data = {}
            all_results = []
            if query and keyword_results:
                all_results.extend(keyword_results)
            if sparse_vector and sparse_results:
                all_results.extend(sparse_results)
            if dense_vector and dense_results:
                all_results.extend(dense_results)

            for doc in all_results:
                doc_id = f"{doc.get('db_name', '')}_{doc.get('table_name', '')}"
                doc_id_to_data[doc_id] = doc

            final_results = []
            for doc_id, score in fused_results:
                if doc_id in doc_id_to_data:
                    doc = doc_id_to_data[doc_id]
                    final_results.append({"rrf_score": score, **doc})

            return final_results

        should: List[Dict[str, Any]] = []
        knn_list: List[Dict[str, Any]] = []

        if query:
            should.append({
                "multi_match": {
                    "query": query,
                    "fields": ["all_names"],
                    "type": "best_fields",
                    "boost": keyword_w,
                }
            })

        if sparse_vector:
            should.append({
                "sparse_vector": {
                    "field": "sparse_vector",
                    "query_vector": sparse_vector,
                    "boost": sparse_w,
                }
            })

        if dense_vector:
            knn_list.append({
                "field": "dense_vector",
                "query_vector": dense_vector,
                "k": size,
                "similarity": 0.7,
                "boost": dense_w,
            })

        if not should and not knn_list:
            raise ValueError("At least one of query, sparse_vector, or dense_vector must be provided")

        body: Dict[str, Any] = {"size": size}
        
        bool_query: Dict[str, Any] = {}
        
        if should:
            bool_query["should"] = should
            bool_query["minimum_should_match"] = 1 if len(should) > 1 else 0
        
        if base_filter:
            bool_query["filter"] = base_filter
        
        if bool_query:
            body["query"] = {"bool": bool_query}
        
        if knn_list:
            body["knn"] = knn_list[0] if len(knn_list) == 1 else knn_list

        response = self.client.search(index=self.index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        return [{"_score": hit["_score"], **hit["_source"]} for hit in hits]

    def keyword_search(
        self,
        query: str,
        size: int = 100,
        filter_db_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {
            "size": size,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["all_names"],
                    "type": "best_fields",
                }
            }
        }

        if filter_db_name:
            body["query"] = {
                "bool": {
                    "must": {"multi_match": {"query": query, "fields": ["all_names"], "type": "best_fields"}},
                    "filter": [{"term": {"db_name": filter_db_name}}]
                }
            }

        response = self.client.search(index=self.index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        return [{"_score": hit["_score"], **hit["_source"]} for hit in hits]

    def sparse_search(
        self,
        sparse_vector: Dict[str, float],
        size: int = 100,
        filter_db_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {
            "size": size,
            "query": {
                "sparse_vector": {
                    "field": "sparse_vector",
                    "query_vector": sparse_vector,
                }
            }
        }

        if filter_db_name:
            body["query"] = {
                "bool": {
                    "must": {"sparse_vector": {"field": "sparse_vector", "query_vector": sparse_vector}},
                    "filter": [{"term": {"db_name": filter_db_name}}]
                }
            }

        response = self.client.search(index=self.index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        return [{"_score": hit["_score"], **hit["_source"]} for hit in hits]

    def dense_search(
        self,
        dense_vector: List[float],
        size: int = 100,
        filter_db_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {
            "size": size,
            "knn": {
                "field": "dense_vector",
                "query_vector": dense_vector,
                "k": size,
                "num_candidates": size * 2,
            }
        }

        if filter_db_name:
            body["query"] = {
                "bool": {
                    "filter": [{"term": {"db_name": filter_db_name}}]
                }
            }

        response = self.client.search(index=self.index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        return [{"_score": hit["_score"], **hit["_source"]} for hit in hits]

    def delete_index(self, index_name: Optional[str] = None) -> bool:
        target_index = index_name or self.index
        if self.client.indices.exists(index=target_index):
            self.client.indices.delete(index=target_index)
            logger.info(f"Deleted index: {target_index}")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        return self.client.indices.stats(index=self.index)
    
    def manual_rrf(all_runs: List[Dict[str, float]], k: int = 60) -> List[tuple]:
        doc_scores: Dict[str, float] = {}
        for run_dict in all_runs:
            if not run_dict:
                continue
            sorted_items = sorted(run_dict.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(sorted_items, 1):
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += 1.0 / (k + rank)
        
        if not doc_scores:
            return []
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs


es_client = ESClient()
