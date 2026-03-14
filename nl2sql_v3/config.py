import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SparseVectorConfig(BaseModel):
    url: str = "http://127.0.0.1:8000/api/v1/sparse_vector"


class DenseVectorConfig(BaseModel):
    url: str = "http://127.0.0.1:8000/api/v1/dense_vector"


class TranslateConfig(BaseModel):
    url: str = "http://127.0.0.1:8000/api/v1/translate"


class ElasticsearchConfig(BaseModel):
    hosts: List[str] = ["http://192.168.116.5:9200"]
    index: str = "tables-metadata"


class LLMConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = os.environ.get("OPENROUTER_API_KEY")
    model: str = "x-ai/grok-4.1-fast"


class ServicesConfig(BaseModel):
    sparse_vector: SparseVectorConfig = SparseVectorConfig()
    dense_vector: DenseVectorConfig = DenseVectorConfig()
    translate: TranslateConfig = TranslateConfig()
    elasticsearch: ElasticsearchConfig = ElasticsearchConfig()
    llm: LLMConfig = LLMConfig()


class WeightsConfig(BaseModel):
    keyword: float = 0.3
    sparse: float = 0.3
    dense: float = 0.4


class RecallConfig(BaseModel):
    weights: WeightsConfig = WeightsConfig()
    top_k: int = 5
    keyword_threshold: float = 0.5


class DataConfig(BaseModel):
    metadata: str = "data/metadatas.json"
    queries: str = "data/query_and_tables.json"

    def get_metadata_path(self, config_file: Path = None) -> Path:
        if config_file is None:
            config_file = Path(__file__).parent / "config.yaml"
        config_dir = config_file.parent
        path = Path(self.metadata)
        if path.is_absolute():
            return path
        return config_dir / path

    def get_queries_path(self, config_file: Path = None) -> Path:
        if config_file is None:
            config_file = Path(__file__).parent / "config.yaml"
        config_dir = config_file.parent
        path = Path(self.queries)
        if path.is_absolute():
            return path
        return config_dir / path


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Config(BaseModel):
    services: ServicesConfig = ServicesConfig()
    recall: RecallConfig = RecallConfig()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        if config_path is None:
            config_path = os.environ.get("NL2SQL_CONFIG", "config.yaml")

        config_file = Path(config_path)
        if not config_file.exists():
            config_file = Path(__file__).parent / "config.yaml"

        if not config_file.exists():
            logger.warning("No config file found, using defaults")
            return cls()

        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        config_dict = cls._resolve_env_vars(config_dict)
        return cls(**config_dict)

    @staticmethod
    def _resolve_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in config_dict.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config_dict[key] = os.environ.get(env_var, "")
            elif isinstance(value, dict):
                config_dict[key] = Config._resolve_env_vars(value)
        return config_dict


config = Config.load()
