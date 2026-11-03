"""Factory helpers for chat and embedding models."""
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from agentic.config import CHAT_MODEL, EMBEDDING_MODEL
from agentic.rate_limits import TokenCounter


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    # Every call's tokens count toward the daily budget
    return ChatOpenAI(model=CHAT_MODEL, temperature=temperature, callbacks=[TokenCounter()])


@lru_cache(maxsize=None)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)
