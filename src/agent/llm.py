"""
LLM Factory - Supports OpenAI, Google Gemini, and DeepSeek.
"""
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from src.config import Config


@lru_cache(maxsize=2)
def get_llm(provider: str = None) -> BaseChatModel:
    """
    Get a cached LLM instance.
    
    Args:
        provider: LLM provider ("openai", "gemini", or "deepseek"). 
                  Defaults to Config.LLM_PROVIDER.
    
    Returns:
        BaseChatModel: Configured LLM instance.
        
    Raises:
        ValueError: If provider is not supported.
    """
    provider = provider or Config.LLM_PROVIDER
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=0,
            api_key=Config.OPENAI_API_KEY,
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            temperature=0,
            google_api_key=Config.GOOGLE_API_KEY,
        )
    elif provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            model=Config.DEEPSEEK_MODEL,
            temperature=0,
            api_key=Config.DEEPSEEK_API_KEY,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'openai', 'gemini', or 'deepseek'.")


def get_available_providers() -> list[str]:
    """
    Get list of available (configured) LLM providers.
    
    Returns:
        list[str]: List of provider names that have API keys configured.
    """
    providers = []
    
    if Config.OPENAI_API_KEY:
        providers.append("openai")
    if Config.GOOGLE_API_KEY:
        providers.append("gemini")
    if Config.DEEPSEEK_API_KEY:
        providers.append("deepseek")
    
    return providers
