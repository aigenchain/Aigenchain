import json
from typing import Optional, Dict, List
def _message_content_as_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                if part:
                    parts.append(str(part))
                continue
            if isinstance(part.get("text"), str):
                parts.append(part["text"])
                continue
            if isinstance(part.get("content"), str):
                parts.append(part["content"])
        return "\n".join(parts)
    return "" if content is None else str(content)


def _chatgpt_subscription_instructions(messages: List[Dict]) -> str:
    instructions = [
        _message_content_as_text(msg.get("content")).strip()
        for msg in messages or []
        if (msg.get("role") or "") == "system"
    ]
    instructions = [part for part in instructions if part]
    if instructions:
        return "\n\n".join(instructions)
    return "You are a helpful AI assistant."


def _build_chatgpt_responses_payload(
    model: str,
    messages: List[Dict],
    temperature: float,
    max_tokens: int,
    *,
    stream: bool = False,
) -> Dict:
    from src.chatgpt_subscription import build_responses_input

    conversation = [msg for msg in (messages or []) if (msg.get("role") or "") != "system"]
    payload: Dict = {
        "model": model,
        "instructions": _chatgpt_subscription_instructions(messages),
        "input": build_responses_input(conversation),
        "stream": stream,
        "store": False,
    }
    if not _restricts_temperature(model):
        payload["temperature"] = temperature
    # ChatGPT Subscription Codex API does not support max_output_tokens —
    # passing it returns HTTP 400 "Unsupported parameter: max_output_tokens".
    # Do not include it in the payload.
    return payload


def _format_chatgpt_subscription_error(status_code: int, text: str) -> str:
    if status_code in (401, 403):
        return "ChatGPT Subscription credentials expired or were rejected. Reconnect the provider."
    if status_code == 429:
        return "ChatGPT Subscription quota or rate limit was reached. Retry after the upstream limit resets."
    return _format_upstream_error(status_code, text, "https://chatgpt.com/backend-api/codex")


def _format_upstream_error(status: int, body: bytes | str, url: str) -> str:
    """Turn an upstream HTTP error into a user-readable sentence.

    Auth failures (401/403) become 'xAI rejected the API key' etc., so the UI
    stops showing raw JSON like '{"error":{"message":"User not found."}}'.
    """
    if isinstance(body, bytes):
        try:
            body = body.decode("utf-8", errors="replace")
        except Exception:
            body = str(body)
    provider = _provider_label(url)
    # Try to pull a message out of the body
    detail = ""
    try:
        j = json.loads(body) if body else {}
        if isinstance(j, dict):
            err = j.get("error") or j
            if isinstance(err, dict):
                detail = (err.get("message") or err.get("detail") or "").strip()
            elif isinstance(err, str):
                detail = err.strip()
    except Exception:
        detail = (body or "").strip()[:240]

    if status in (401, 403):
        msg = f"{provider} rejected the API key"
        if status == 403:
            msg = f"{provider} denied access (403)"
        if detail:
            msg += f" — {detail}"
        msg += ". Check Model Endpoints → {} and re-paste the key.".format(provider)
        return msg
    if status == 404:
        return f"{provider} returned 404 — check the base URL and model name." + (f" ({detail})" if detail else "")
    if status == 429:
        return f"{provider} rate-limited the request (429)." + (f" {detail}" if detail else "")
    if status >= 500:
        return f"{provider} is having an outage (HTTP {status})." + (f" {detail}" if detail else "")
    return f"{provider} returned HTTP {status}" + (f": {detail}" if detail else "")

# Models that require max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_MODELS = {"o1", "o3", "o4", "gpt-4.5", "gpt-5"}

def _uses_max_completion_tokens(model: str) -> bool:
    """Check if a model requires max_completion_tokens instead of max_tokens."""
    if not model:
        return False
    m = model.lower()
    return any(m.startswith(p) or f"/{p}" in m for p in _MAX_COMPLETION_TOKENS_MODELS)

# OpenAI reasoning models (o1, o3, o4, gpt-5 families) only accept the default
# temperature. Sending any explicit value — even 0.0 — returns HTTP 400
# ("Only the default (1) value is supported"). That otherwise breaks chat when a
# preset sets a non-default temperature, and makes endpoint probing report a
# perfectly good model as failing. For these models we omit the field and let
# the API use its required default. (gpt-4.5 is intentionally excluded — it is
# not a reasoning model and accepts temperature normally.)
_FIXED_TEMPERATURE_MODELS = ("o1", "o3", "o4", "gpt-5")

def _restricts_temperature(model: str) -> bool:
    """Check if a model rejects any non-default temperature."""
    if not model:
        return False
    m = model.lower()
    return any(m.startswith(p) or f"/{p}" in m for p in _FIXED_TEMPERATURE_MODELS)

# Anthropic removed the sampling parameters (temperature, top_p, top_k) starting
# with Claude Opus 4.7. On Opus 4.7 and later, sending `temperature` at all —
# even 0.0 — returns HTTP 400. Earlier Claude models (Opus 4.6 and below, every
# Sonnet/Haiku) still accept temperature in [0.0, 1.0], so the omission must be
# version-gated rather than applied to all `claude-*` models.
def _anthropic_rejects_temperature(model: str) -> bool:
    """Check if a native-Anthropic model rejects the temperature field (Opus 4.7+)."""
    if not isinstance(model, str) or not model:
        return False
    # `(?<![a-z])` anchors "opus" to a word boundary so a substring match like
    # `oct-opus`/`octopus-4-8` can't be read as Opus (it would otherwise strip
    # temperature). Cap the minor at 1-2 digits and forbid a trailing digit so a
    # dated id like `claude-opus-4-20250514` (Opus 4.0) parses as major-only (no
    # minor match, kept) instead of reading the date `20250514` as a giant minor
    # that would falsely test >= 4.7. Dated 4.7+ snapshots (`claude-opus-4-7-
    # 20260201`) keep their explicit minor and are still matched.
    match = re.search(r"(?<![a-z])opus[-_]?(\d+)[-_.](\d{1,2})(?!\d)", model.lower())
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2))) >= (4, 7)

# Models that support structured thinking — may output </think> without opening tag
_THINKING_MODEL_PATTERNS = ("qwen3", "qwq", "deepseek-r1", "deepseek-reasoner", "minimax", "m2-reap", "gemma")

def _supports_thinking(model: str) -> bool:
    """Check if model supports structured thinking output."""
    if not model:
        return False
    m = model.lower()
    return any(p in m for p in _THINKING_MODEL_PATTERNS)

def _as_content_blocks(content) -> List[Dict]:
    """Coerce a message `content` into a list of content blocks.

    A list (multimodal: text + image parts) passes through; a non-empty string
    becomes a single text block; None/empty yields no blocks.
    """
    if isinstance(content, list):
        return content
    if content:
        return [{"type": "text", "text": str(content)}]
    return []
def _sanitize_llm_messages(messages: List[Dict]) -> List[Dict]:
    """Strip Odysseus-only metadata before sending messages to providers.

    Per the OpenAI chat format: user/system messages must have content; a tool
    message needs content + tool_call_id; an assistant message may carry content,
    tool_calls, or both. The old guard required content on every message, which
    dropped a valid assistant message that has only tool_calls — e.g. the
    follow-up message _append_tool_results builds for a no-prose native tool call
    (content=None, since Gemini/Ollama reject tool_calls alongside ""). Dropping
    it leaves the tool result dangling and breaks the next round.
    """
    allowed = {"role", "content", "name", "tool_call_id", "tool_calls", "function_call", "reasoning_content"}
    cleaned = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        item = {k: v for k, v in msg.items() if k in allowed and v is not None}
        role = item.get("role")
        if not role:
            continue
        if role == "assistant":
            # Re-add an explicit content=None when the message is tool-calls-only
            # (the None was stripped above) so the provider gets the spec-correct
            # `content: null`, not an omitted key.
            if "content" not in item and item.get("tool_calls"):
                item["content"] = None
            if "content" in item or item.get("tool_calls"):
                cleaned.append(item)
        elif role == "tool":
            if "content" in item and "tool_call_id" in item:
                cleaned.append(item)
        elif "content" in item:
            cleaned.append(item)

    # Repair tool-call adjacency before sending to any OpenAI-compatible
    # provider. Trimming/compaction/retries can leave `role:"tool"` messages
    # without their immediately-preceding assistant `tool_calls` parent, which
    # DeepSeek rejects with:
    # "Messages with role 'tool' must be a response to a preceding message with
    # 'tool_calls'". Also strip unanswered assistant tool_calls; some providers
    # reject those as incomplete conversations.
    repaired: List[Dict] = []
    i = 0
    while i < len(cleaned):
        msg = cleaned[i]
        role = msg.get("role")

        if role == "tool":
            # Orphan tool result. There is no valid assistant tool_calls parent
            # immediately before this batch, so it cannot be sent.
            logger.debug("Dropping orphan tool message before provider request")
            i += 1
            continue

        tool_calls = msg.get("tool_calls") if role == "assistant" else None
        if not tool_calls:
            repaired.append(msg)
            i += 1
            continue

        call_ids = [
            str(tc.get("id"))
            for tc in tool_calls
            if isinstance(tc, dict) and tc.get("id")
        ]
        expected = set(call_ids)
        answered_ids = []
        tool_batch = []
        j = i + 1
        while j < len(cleaned) and cleaned[j].get("role") == "tool":
            tid = str(cleaned[j].get("tool_call_id") or "")
            if tid in expected and tid not in answered_ids:
                answered_ids.append(tid)
                tool_batch.append(cleaned[j])
            else:
                logger.debug("Dropping unmatched/duplicate tool message before provider request")
            j += 1

        if not tool_batch:
            plain = {k: v for k, v in msg.items() if k != "tool_calls"}
            if (plain.get("content") or "").strip():
                repaired.append(plain)
            else:
                logger.debug("Dropping unanswered assistant tool_calls before provider request")
            i = j
            continue

        answered = set(answered_ids)
        pruned_calls = [
            tc for tc in tool_calls
            if isinstance(tc, dict) and str(tc.get("id")) in answered
        ]
        fixed = dict(msg)
        fixed["tool_calls"] = pruned_calls
        if "content" not in fixed:
            fixed["content"] = None
        repaired.append(fixed)
        repaired.extend(tool_batch)
        if len(pruned_calls) != len(tool_calls):
            logger.debug("Pruned unanswered assistant tool_calls before provider request")
        i = j

    # Merge consecutive user messages to satisfy strict role alternation
    # requirements after invalid tool-call fragments have been removed.
    merged: List[Dict] = []
    for item in repaired:
        if not merged:
            merged.append(item)
            continue

        last = merged[-1]
        if last.get("role") == "user" and item.get("role") == "user":
            last_copy = dict(last)
            lc = last_copy.get("content")
            ic = item.get("content")
            if isinstance(lc, list) or isinstance(ic, list):
                # Preserve multimodal content blocks (e.g. an image part) by
                # concatenating the block lists. str()-ing a list turned an
                # image message into its Python repr and dropped the image.
                merged_blocks = _as_content_blocks(lc) + _as_content_blocks(ic)
                if merged_blocks:
                    last_copy["content"] = merged_blocks
                else:
                    last_copy.pop("content", None)
            else:
                last_str = str(lc) if lc is not None else ""
                item_str = str(ic) if ic is not None else ""
                new_content = "\n\n".join(part for part in (last_str, item_str) if part)
                if new_content:
                    last_copy["content"] = new_content
                else:
                    last_copy.pop("content", None)
            merged[-1] = last_copy
        else:
            merged.append(item)

    return merged

