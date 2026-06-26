import json
import logging

from fastapi import HTTPException
from typing import Optional, Dict, List
from runtime.llm.config import LLMConfig

from runtime.llm.cache import (
    _get_cache_key,
    _get_cached_response,
    _set_cached_response,
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
)

from runtime.llm.chatgpt import (
    _build_chatgpt_responses_payload,
)
from runtime.llm.helpers import (
    _as_content_blocks,
)
logger = logging.getLogger(__name__)
def llm_call(url: str, model: str, messages: List[Dict], temperature: float = LLMConfig.DEFAULT_TEMPERATURE,
             max_tokens: int = LLMConfig.DEFAULT_MAX_TOKENS, headers: Optional[Dict] = None, 
             timeout: int = LLMConfig.DEFAULT_TIMEOUT, prompt_type: Optional[str] = None) -> str:
    """Synchronous LLM call with optional prompt type enhancement."""
    h = _provider_headers(_detect_provider(url))
    # Tolerate headers that arrive as a JSON string (some sessions stored them
    # double-encoded) — otherwise h.update() throws "dictionary update sequence
    # element #0 has length 1; 2 is required".
    if isinstance(headers, str):
        try:
            headers = json.loads(headers)
        except Exception:
            headers = None
    if isinstance(headers, dict):
        h.update(headers)

    messages_copy = _sanitize_llm_messages(messages)

    # Consolidate multiple system messages into one at the start.
    sys_parts = []
    non_sys = []
    for m in messages_copy:
        if m.get("role") == "system":
            sys_parts.append(m.get('content') or '')
        else:
            non_sys.append(m)
    if sys_parts:
        messages_copy = [{"role": "system", "content": "\n\n".join(sys_parts)}] + non_sys
    else:
        messages_copy = non_sys

    provider = _detect_provider(url)
    cache_key = _get_cache_key(url, model, messages_copy, temperature, max_tokens)
    cached_response = _get_cached_response(cache_key)
    if cached_response:
        logger.debug(f"Returning cached response for key: {cache_key}")
        return cached_response

    if provider == "anthropic":
        target_url = _normalize_anthropic_url(url)
        h = _build_anthropic_headers(headers)
        payload = _build_anthropic_payload(model, messages_copy, temperature, max_tokens)
    elif provider == "ollama":
        target_url = _normalize_ollama_url(url)
        payload = _build_ollama_payload(
            model, messages_copy, temperature, max_tokens,
            stream=False, num_ctx=get_context_length(url, model),
        )
    else:
        target_url = url
        if provider == "copilot":
            from src.copilot import apply_request_headers
            apply_request_headers(h, messages_copy)
        payload = {
            "model": model,
            "messages": messages_copy,
            "temperature": temperature,
        }
        if _restricts_temperature(model):
            payload.pop("temperature", None)
        if max_tokens and max_tokens > 0:
            tok_key = "max_completion_tokens" if _uses_max_completion_tokens(model) else "max_tokens"
            payload[tok_key] = max_tokens
    try:
        note_model_activity(target_url, model)
        r = httpx.post(target_url, headers=h, json=payload, timeout=timeout)
    except Exception as e:
        raise HTTPException(502, f"POST {target_url} failed: {e}")
    if not r.is_success:
        raise HTTPException(502, f"Upstream {target_url} -> {r.status_code}: {r.text}")
    data = r.json()
    try:
        if provider == "anthropic":
            response = _parse_anthropic_response(data)
        elif provider == "ollama":
            response = _parse_ollama_response(data)
        else:
            msg = data["choices"][0]["message"]
            response = msg.get("content") or msg.get("reasoning_content") or ""
        _set_cached_response(cache_key, response)
        return response
    except Exception:
        raise HTTPException(502, f"Unexpected schema from {target_url}: {str(data)[:400]}")


def _dedupe_candidates(candidates):
    """Filter malformed entries and drop a later repeat of an already-seen
    ``(url, model)`` route, preserving order (first occurrence wins).

    The chain is the primary target followed by the configured fallbacks, so a
    fallback that repeats the session's current model — a common misconfiguration,
    since callers prepend the live ``(url, model)`` to ``default_model_fallbacks``
    — would otherwise make the chain re-attempt the very route that just failed:
    a wasted round-trip plus a spurious ``fallback`` notice for a switch that did
    not happen. Headers are not part of the key; the first tuple (with its
    headers) is the one kept.
    """
    seen = set()
    out = []
    for c in candidates or []:
        if not c or not c[0] or not c[1]:
            continue
        key = (c[0], c[1])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


