import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nl2sql_v3.config import config

logger = logging.getLogger(__name__)


class APIClient:
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _post(
        self,
        url: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        try:
            response = self.session.post(
                url, json=json_data, params=params, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout: {url}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {url}, error: {e}")
            raise


class SparseVectorClient(APIClient):
    def __init__(self):
        super().__init__()
        self.url = config.services.sparse_vector.url

    def encode(self, text: str, query_mode: bool = True) -> Dict[str, Any]:
        result = self._post(
            self.url, json_data={"text": text, "query_mode": query_mode}
        )
        return result


class DenseVectorClient(APIClient):
    def __init__(self):
        super().__init__()
        self.url = config.services.dense_vector.url

    def encode(self, text: str) -> List[float]:
        result = self._post(self.url, json_data={"text": text})
        return result.get("dense_vector", [])


class TranslateClient(APIClient):
    def __init__(self):
        super().__init__()
        self.url = config.services.translate.url

    def translate(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        result = self._post(self.url, json_data={"text": text})
        return result.get("translated_text", "")


class LLMClient(APIClient):
    def __init__(self):
        super().__init__()
        self.base_url = config.services.llm.base_url
        self.api_key = config.services.llm.api_key
        self.model = config.services.llm.model
        if not self.api_key:
            logger.warning("LLM API key not configured")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        if not self.api_key:
            raise ValueError("LLM API key not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = self._post(
            f"{self.base_url}/chat/completions",
            json_data=payload,
            headers=headers,
        )

        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        return ""


class RerankClient(APIClient):
    def __init__(self):
        super().__init__()
        self.url = config.services.rerank.url

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        json_data = {
            "query": query,
            "documents": documents,
        }
        if top_k is not None:
            json_data["top_k"] = top_k

        result = self._post(self.url, json_data=json_data)
        return result


sparse_vector_client = SparseVectorClient()
dense_vector_client = DenseVectorClient()
translate_client = TranslateClient()
llm_client = LLMClient()
rerank_client = RerankClient()
