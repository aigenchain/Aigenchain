import asyncio
import json
import logging
import httpx

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

from runtime.llm.helpers import (
    _format_upstream_error,
    _restricts_temperature,
    _anthropic_rejects_temperature,
    _supports_thinking,
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

logger = logging.getLogger(__name__)
async def llm_call_async(
    url: str,
    model: str,
    messages: List[Dict],
    temperature: float = LLMConfig.DEFAULT_TEMPERATURE,
    max_tokens: int = LLMConfig.DEFAULT_MAX_TOKENS,
    headers: Optional[Dict] = None,
    timeout: int = LLMConfig.STREAM_TIMEOUT,
    max_retries: int = LLMConfig.MAX_RETRIES,
    prompt_type: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Asynchronous LLM call using httpx with connection pooling, timeout, retry logic, and performance logging."""
    provider = _detect_provider(url)
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

    cache_key = _get_cache_key(url, model, messages_copy, temperature, max_tokens)
    cached_response = _get_cached_response(cache_key)
    if cached_response:
        logger.debug(f"Returning cached response for key: {cache_key}")
        return cached_response

    if provider == "chatgpt-subscription":
        # ChatGPT/Codex requires streamed Responses requests even for callers
        # that want a plain string (auto-title, memory extraction, etc.).
        # Reuse stream_llm's validated Codex SSE path and collect deltas.
        parts: List[str] = []
        async for chunk in stream_llm(
            url,
            model,
            messages_copy,
            temperature=temperature,
            max_tokens=max_tokens,
            headers=headers,
            timeout=timeout,
        ):
            event_is_error = False
            for line in str(chunk).splitlines():
                if line.startswith("event:"):
                    event_is_error = line[6:].strip() == "error"
                    continue
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                if raw == "[DONE]":
                    response = "".join(parts)
                    _set_cached_response(cache_key, response)
                    return response
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if event_is_error or data.get("error") or (data.get("status") and data.get("text")):
                    status = int(data.get("status") or 502)
                    text = data.get("text") or data.get("error") or "ChatGPT Subscription request failed"
                    raise HTTPException(status, text)
                delta = data.get("delta")
                if isinstance(delta, str):
                    parts.append(delta)
        response = "".join(parts)
        _set_cached_response(cache_key, response)
        return response

    if provider == "anthropic":
        target_url = _normalize_anthropic_url(url)
        h = _build_anthropic_headers(headers)
        payload = _build_anthropic_payload(model, messages_copy, temperature, max_tokens)
    elif provider == "ollama":
        target_url = _normalize_ollama_url(url)
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        payload = _build_ollama_payload(
            model, messages_copy, temperature, max_tokens,
            stream=False, num_ctx=get_context_length(url, model),
        )
    else:
        target_url = url
        h = _provider_headers(provider, headers)
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
        # Suppress thinking for qwen3/gemma4 on Ollama /v1 — same as stream_llm.
        if _is_ollama_openai_compat_url(url) and _supports_thinking(model):
            payload["think"] = False
        _apply_local_cache_affinity(payload, url, session_id)

    if _is_host_dead(target_url):
        raise HTTPException(503, f"Upstream {_host_key(target_url)} marked unreachable (cooldown active)")

    call_timeout = httpx.Timeout(connect=3.0, read=float(timeout), write=10.0, pool=5.0)
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        start = time.time()
        try:
            note_model_activity(target_url, model)
            client = _get_http_client()
            r = await client.post(target_url, headers=h, json=payload, timeout=call_timeout)
            duration = time.time() - start
            if not r.is_success:
                friendly = _format_upstream_error(r.status_code, r.text, target_url)
                logger.warning(
                    f"LLM async call to {target_url} failed in {duration:.2f}s "
                    f"(attempt {attempt}): HTTP {r.status_code} {friendly}"
                )
                if r.status_code in (429, 502, 503, 504) and attempt < max_retries:
                    await asyncio.sleep(LLMConfig.RETRY_DELAY)
                    continue
                raise HTTPException(r.status_code, friendly)
            logger.info(f"LLM async call to {target_url} succeeded in {duration:.2f}s (attempt {attempt})")
            _clear_host_dead(target_url)
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
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            _cooled = _mark_host_dead(target_url)
            duration = time.time() - start
            _tail = f" — host cooled for {DEAD_HOST_COOLDOWN:.0f}s" if _cooled else " — transient, will retry"
            logger.warning(f"LLM async connect to {target_url} failed after {duration:.2f}s: {e}{_tail}")
            if _cooled or attempt >= max_retries:
                raise HTTPException(503, f"Cannot reach {_host_key(target_url)}: {e}")
            await asyncio.sleep(LLMConfig.RETRY_DELAY)
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            duration = time.time() - start
            logger.warning(f"LLM async call attempt {attempt} failed after {duration:.2f}s: {e}")
            if attempt >= max_retries:
                raise HTTPException(502, f"POST {target_url} failed after {max_retries} attempts: {e}")
            await asyncio.sleep(LLMConfig.RETRY_DELAY)

