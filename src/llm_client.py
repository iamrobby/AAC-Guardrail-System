"""Thin, fault-tolerant wrapper around the Groq (OpenAI-compatible) chat API."""
import logging
import random
import time
from typing import Dict, List, Optional

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from src import config

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM cannot return a usable response."""


_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Create the client lazily so that importing this module never needs a key."""
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise LLMError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
            timeout=config.LLM_TIMEOUT,
            max_retries=0,  # we do our own retries below
        )
    return _client


def call_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> str:
    """Return the assistant text. Retries transient errors and empty replies."""
    retries = max(1, config.LLM_RETRIES)
    last_err: Optional[Exception] = None

    for attempt in range(retries):
        try:
            kwargs = {
                "model": model or config.MODEL,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            response = get_client().chat.completions.create(**kwargs)
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
            last_err = LLMError("model returned empty content")
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
            last_err = exc
            log.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, retries, exc)

        if attempt < retries - 1:
            time.sleep(min(2 ** attempt, 30) + random.random())

    raise LLMError(f"LLM call failed after {retries} attempts: {last_err}") from last_err
