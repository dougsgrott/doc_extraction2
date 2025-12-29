"""Factory for creating LLM clients based on configuration."""

from .client import BaseLLMClient, AzureLLMClient, OpenAIClient


def create_llm_client(config) -> BaseLLMClient:
    """ Create LLM client based on config.llm_provider. """
    provider = config.llm_provider.lower()

    if provider == "azure":
        if not all([config.llm_endpoint, config.llm_api_key, config.llm_model]):
            raise ValueError(
                "Azure provider requires: llm_endpoint, llm_api_key, llm_model"
            )
        return AzureLLMClient(
            azure_endpoint=config.llm_endpoint,
            api_key=config.llm_api_key,
            api_version=config.llm_api_version
        )

    elif provider == "openai":
        if not all([config.llm_api_key, config.llm_model]):
            raise ValueError("OpenAI provider requires: llm_api_key, llm_model")
        return OpenAIClient(api_key=config.llm_api_key)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. Use 'azure' or 'openai'"
        )
