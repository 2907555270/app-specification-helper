import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from elasticsearch import Elasticsearch

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
        keyword_weight: float = 1.0,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        size: int = 10,
        filter_db_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        should: List[Dict[str, Any]] = []
        knn_list: List[Dict[str, Any]] = []

        if query:
            should.append({
                "multi_match": {
                    "query": query,
                    "fields": ["all_names", "table_name"],
                    "type": "best_fields",
                    "boost": keyword_weight,
                }
            })

        if sparse_vector:
            should.append({
                "sparse_vector": {
                    "field": "sparse_vector",
                    "query_vector": sparse_vector,
                    "boost": sparse_weight,
                }
            })

        if dense_vector:
            knn_list.append({
                "field": "dense_vector",
                "query_vector": dense_vector,
                "k": size,
                "similarity": 0.7,
                "boost": dense_weight,
            })

        if not should and not knn_list:
            raise ValueError("At least one of query, sparse_vector, or dense_vector must be provided")

        body: Dict[str, Any] = {"size": size}
        
        bool_query: Dict[str, Any] = {}
        
        if should:
            bool_query["should"] = should
            bool_query["minimum_should_match"] = 1 if len(should) > 1 else 0
        
        if filter_db_name:
            bool_query["filter"] = [{"term": {"db_name": filter_db_name}}]
        
        if bool_query:
            body["query"] = {"bool": bool_query}
        
        if knn_list:
            body["knn"] = knn_list[0] if len(knn_list) == 1 else knn_list

        logger.debug(f"Hybrid search body: {body}")
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


es_client = ESClient()
