"""LLM client abstraction supporting Azure OpenAI and OpenAI."""

from abc import ABC, abstractmethod
from typing import Any, List, Type, Optional
from pydantic import BaseModel
from openai import AzureOpenAI, OpenAI


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def chat_completions_parse(
        self,
        model: str,
        messages: List[dict],
        response_format: Type[BaseModel],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Call LLM with structured output parsing.

        Returns completion object with .choices[0].message.parsed
        """
        pass


class AzureLLMClient(BaseLLMClient):
    """Azure OpenAI client wrapper."""

    def __init__(self, azure_endpoint: str, api_key: str, api_version: str):
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )

    def chat_completions_parse(
        self,
        model: str,
        messages: List[dict],
        response_format: Type[BaseModel],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        return self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


class OpenAIClient(BaseLLMClient):
    """OpenAI (direct) client wrapper."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def chat_completions_parse(
        self,
        model: str,
        messages: List[dict],
        response_format: Type[BaseModel],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        return self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
