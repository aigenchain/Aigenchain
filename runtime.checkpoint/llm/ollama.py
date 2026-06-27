import json
from typing import Optional, List, Dict
from urllib.parse import urlparse
from runtime.llm.providers import _host_match
def _is_ollama_native_url(url: str) -> bool:
    """Return True for native Ollama API URLs, including Ollama Cloud."""
    try:
        parsed = urlparse(url or "")
    except Exception:
        return False
    host = parsed.hostname or ""
    path = (parsed.path or "").rstrip("/")
    if _host_match(url, "ollama.com"):
        return True
    if path.startswith("/v1"):
        return False
    local_ollama_host = host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or parsed.port == 11434
    return local_ollama_host and (path == "" or path == "/api" or path.startswith("/api/"))


def _is_ollama_openai_compat_url(url: str) -> bool:
    """Return True for local Ollama's OpenAI-compatible /v1 surface.

    Mirrors the host detection used by ``_is_ollama_native_url`` so that the
    two helpers stay in lockstep: a localhost Ollama on a non-default port
    (custom ``OLLAMA_HOST``, reverse proxy, container port remap) is treated
    the same way here as it is on the native ``/api`` path.
    """
    try:
        parsed = urlparse(url or "")
    except Exception:
        return False
    host = parsed.hostname or ""
    path = (parsed.path or "").rstrip("/")
    local_ollama_host = host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or parsed.port == 11434
    return local_ollama_host and (path == "/v1" or path.startswith("/v1/"))


def _ollama_api_root(url: str) -> str:
    """Return a native Ollama API root such as https://ollama.com/api."""
    url = (url or "").strip().rstrip("/")
    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/api/chat"):
        return url[: -len("/chat")]
    if path.endswith("/api/tags"):
        return url[: -len("/tags")]
    if path.endswith("/api/generate"):
        return url[: -len("/generate")]
    if path.endswith("/api"):
        return url
    if path == "":
        return url + "/api"
    if _host_match(url, "ollama.com"):
        root = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else "https://ollama.com"
        return root.rstrip("/") + "/api"
    return url


def _normalize_ollama_url(url: str) -> str:
    """Ensure a native Ollama URL points at /api/chat."""
    base = _ollama_api_root(url)
    return base.rstrip("/") + "/chat"


def _ollama_normalize_tool_messages(messages: List[Dict]) -> List[Dict]:
    """Adapt Odysseus' canonical OpenAI-style messages to native Ollama /api/chat.

    Odysseus carries assistant tool calls in the OpenAI shape, where
    `function.arguments` is a JSON *string*. Native Ollama expects it to be a
    JSON *object*; given the string it fails the whole request with HTTP 400
    "Value looks like object, but can't find closing '}' symbol", which aborts
    every follow-up (tool-result) round. Parse the arguments back into an object
    here, on a shallow copy, leaving non-tool messages untouched. The opaque
    Gemini `extra_content` (thought_signature) is dropped — it is meaningless to
    Ollama and only matters when the conversation is replayed to Gemini.
    """
    out: List[Dict] = []
    for m in messages or []:
        tcs = m.get("tool_calls") if isinstance(m, dict) else None
        if not tcs:
            out.append(m)
            continue
        new_calls = []
        for tc in tcs:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
            call: Dict = {"function": {"name": fn.get("name", ""), "arguments": args or {}}}
            if tc.get("id"):
                call["id"] = tc["id"]
            new_calls.append(call)
        nm = dict(m)
        nm["tool_calls"] = new_calls
        out.append(nm)
    return out


def _build_ollama_payload(
    model: str,
    messages: List[Dict],
    temperature: float,
    max_tokens: int,
    stream: bool = False,
    tools: Optional[List[Dict]] = None,
    num_ctx: Optional[int] = None,
) -> Dict:
    """Build the JSON payload for Ollama's /api/chat endpoint.

    ``num_ctx`` sets the input context window. Ollama defaults to 2048
    when the option is omitted, so a model with a larger advertised
    window is silently truncated there, and a model with a smaller one
    gets an oversized window it can't service. Pass the discovered
    context length through ``num_ctx``; this builder only emits it when
    the value is trusted (not the ``DEFAULT_CONTEXT`` fallback), so we
    don't guess for unknown models but do tell Ollama the real window
    when we know it — even if it's smaller than 2048.
    """
    payload: Dict = {
        "model": model,
        "messages": _ollama_normalize_tool_messages(messages),
        "stream": stream,
    }
    options: Dict = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens and max_tokens > 0:
        options["num_predict"] = max_tokens
    if num_ctx is not None and num_ctx > 0 and num_ctx != DEFAULT_CONTEXT:
        options["num_ctx"] = num_ctx
    if options:
        payload["options"] = options
    if tools:
        payload["tools"] = tools
    return payload


def _parse_ollama_response(data: dict) -> str:
    message = data.get("message") or {}
    return message.get("content") or data.get("response") or ""


