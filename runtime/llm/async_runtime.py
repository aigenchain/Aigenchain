import asyncio
import json
import logging

import httpx

from fastapi import HTTPException
from typing import Optional, Dict, List

from runtime.llm.config import LLMConfig

logger = logging.getLogger(__name__)

from runtime.llm.cache import (
    _get_cache_key,
    _get_cached_response,
    _set_cached_response,
    note_model_activity,
)

from runtime.llm.providers import (
    _detect_provider,
    _provider_headers,
)

from runtime.llm.ollama import (
    _build_ollama_payload,
)

from runtime.llm.anthropic import (
    _build_anthropic_payload,
    _parse_anthropic_response,
    _normalize_anthropic_url,
)

from runtime.llm.chatgpt import (
    _build_chatgpt_responses_payload,
    _normalize_chatgpt_subscription_url,
    _format_chatgpt_subscription_error,
    _format_upstream_error,
)

from runtime.llm.helpers import (
    _sanitize_llm_messages,
    _restricts_temperature,
    _uses_max_completion_tokens,
)

from runtime.llm.sync import (
    llm_call,
    _dedupe_candidates,
)
async def llm_call_async_with_fallback(candidates, messages, **kwargs) -> str:
    """Async variant of llm_call_with_fallback."""
    cands = _dedupe_candidates(candidates)
    if not cands:
        raise HTTPException(503, "No model endpoint configured")

    last_err = None

    for i, (url, model, headers) in enumerate(cands):
        try:
            return await llm_call_async(
                url,
                model,
                messages,
                headers=headers,
                **kwargs,
            )
        except Exception as e:
            last_err = e
            tag = "primary" if i == 0 else "candidate"
            logger.warning(
                f"[fallback] {tag} {model} failed ({type(e).__name__}); trying next"
            )

    raise last_err if last_err else HTTPException(
        503,
        "All fallback candidates failed",
    )
