import json
import logging
from typing import Any, Dict, List, Optional, Literal

from nl2sql_v3.config import config

logger = logging.getLogger(__name__)


class LLMResponse:
    def __init__(self, content: str, raw_response: Optional[Any] = None):
        self.content = content
        self.raw_response = raw_response or {}


def get_llm_client(
    provider: str = "openrouter",
    type: Literal["analyse", "coding"] = "analyse",
    **kwargs,
) -> "LangChainLLMClient":
    if provider == "openrouter":
        return LangChainLLMClient(
            api_key=kwargs.get("api_key") or config.services.llm.api_key,
            base_url=kwargs.get("base_url") or config.services.llm.base_url,
            analyse_model=kwargs.get("analyse_model") or config.services.llm.analyse_model,
            coding_model=kwargs.get("coding_model") or config.services.llm.coding_model,
            temperature=kwargs.get("temperature", 0.0),
            type=type,
        )
    elif provider == "openai":
        return LangChainLLMClient(
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
            analyse_model=kwargs.get("analyse_model") or config.services.llm.analyse_model,
            coding_model=kwargs.get("coding_model") or config.services.llm.coding_model,
            temperature=kwargs.get("temperature", 0.0),
            type=type,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Available: openrouter, openai")


class LangChainLLMClient:
    def __init__(
        self,
        api_key: Optional[str] = config.services.llm.api_key,
        base_url: Optional[str] = config.services.llm.base_url,
        analyse_model: Optional[str] = config.services.llm.analyse_model,
        coding_model: Optional[str] = config.services.llm.coding_model,
        temperature: float = 0.3,
        type: Literal["analyse", "coding"] = "analyse",
    ):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("langchain-openai is required. Install with: pip install langchain-openai")

        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        model = analyse_model if type == "analyse" else coding_model

        self.client = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        from langchain_core.messages import HumanMessage, SystemMessage

        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))

        response = self.client.invoke(langchain_messages)
        return LLMResponse(content=response.content, raw_response=response)

    def chat_with_json_output(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.output_parsers import JsonOutputParser
        from pydantic import BaseModel

        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))

        try:
            class SQLGenerationResult(BaseModel):
                sql: str
                confidence: float
                explanation: str
                selected_tables: List[str]
                used_columns: List[str]

            structured_llm = self.client.with_structured_output(SQLGenerationResult)
            logger.info(f"Invoking LLM with structured output")
            result = structured_llm.invoke(langchain_messages)
            logger.info(f"LLM response: {result}")
            return result.model_dump() if hasattr(result, 'model_dump') else result.dict()
        except Exception as e:
            logger.warning(f"Structured output failed, trying JsonOutputParser: {e}")
            parser = JsonOutputParser()
            chain = self.client | parser
            result = chain.invoke(langchain_messages)
            return result if isinstance(result, dict) else json.loads(result)
