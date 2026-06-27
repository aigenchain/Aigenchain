"""
Agent Intent Classifier.
"""

from typing import Dict, List


def _detect_admin_intent(messages: List[Dict]) -> bool:
    """Check if the last user message suggests admin/management tool usage."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            content_lower = content.lower()
            return any(kw in content_lower for kw in _ADMIN_KEYWORDS)
    return False


def _extract_last_user_message(messages: List[Dict]) -> str:
    """Return the most recent user message as plain text."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            return content
    return ""


_LOW_SIGNAL_RE = re.compile(r"^[\W_]*$", re.UNICODE)
_EXPLICIT_CONTINUATION_RE = re.compile(
    r"^\s*(?:"
    r"yes|y|yeah|yep|ok|okay|sure|do it|go ahead|continue|carry on|"
    r"run it|launch it|start it|use that|that one|same|the same|"
    r"first|second|third|the first one|the second one|the third one|"
    r"[123]|[abc]"
    r")\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def _is_explicit_continuation(text: str) -> bool:
    """Only these terse replies may inherit older user turns for tool retrieval."""
    return bool(_EXPLICIT_CONTINUATION_RE.match(str(text or "").strip()))


def _assistant_requested_followup(messages: List[Dict]) -> bool:
    """True when the previous assistant turn asked for missing task details.

    This allows natural replies like "buy milk" after "What would you like on
    your to-do list?" to inherit the prior domain, without letting random
    greetings inherit stale Cookbook/email/document context.
    """
    seen_latest_user = False
    for msg in reversed(messages):
        role = msg.get("role")
        if role == "user" and not seen_latest_user:
            seen_latest_user = True
            continue
        if not seen_latest_user:
            continue
        if role != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        text = str(content or "").lower()
        if "?" not in text:
            return False
        return bool(re.search(
            r"\b(what would you like|what should|what do you want|which one|which model|"
            r"what.+(?:todo|to-do|list|document|email|model|server|item)|"
            r"any specific|give me|tell me)\b",
            text,
        ))
    return False


def _classify_agent_request(messages: List[Dict], last_user: str) -> Dict[str, object]:
    """Classify only whether this turn deserves domain tool retrieval.

    Normal chat should not inherit old Cookbook/email/document context. Recent
    context is used only for explicit continuations ("yes", "do it", "1").
    This function does not inject tools directly; selected tools later decide
    which domain rule packs get appended to the system prompt.
    """
    text = str(last_user or "").strip()
    continuation = _is_explicit_continuation(text) or _assistant_requested_followup(messages)
    retrieval_query = _recent_context_for_retrieval(messages) if continuation else text
    q = retrieval_query.lower()

    if not text or bool(_LOW_SIGNAL_RE.match(text)):
        return {
            "low_signal": True,
            "continuation": False,
            "domains": set(),
            "retrieval_query": text,
        }

    domains: Set[str] = set()

    def has(*patterns: str) -> bool:
        return any(re.search(p, q) for p in patterns)

    if has(r"\b(cookbook|serve|serving|served|launch|start|preset|vllm|sglang|llama\.?cpp|ollama|download|downloading|pull|cached models?|running models?|model servers?|models? (?:are )?running|what models?|model picker|gpu box|kierkegaard|odysseus|ajax|qwen|gemma|llama|mistral|minimax)\b"):
        domains.add("cookbook")
    if has(r"\b(emails?|mails?|gmail|inbox|reply|forward|cc|bcc|send email|compose email|draft email|message chris|message him|message her)\b"):
        domains.add("email")
    if has(r"\b(note|todo|to-do|checklist|task list|remind me|reminder|buy|pickup|pick up)\b"):
        domains.add("notes_calendar_tasks")
    if has(r"\b(every day|every morning|every evening|recurring|automatically|cron|scheduled task|background task)\b"):
        domains.add("notes_calendar_tasks")
    if has(r"\b(calendar|event|meeting|appointment|schedule)\b"):
        domains.add("notes_calendar_tasks")
    if has(r"\b(documents?|docs?|draft|compose|poem|story|essay|outline|letter|edit|rewrite|proofread|suggest|feedback|review this|make a file)\b"):
        domains.add("documents")
    if "notes_calendar_tasks" not in domains and has(r"\bwrite\b"):
        domains.add("documents")
    if has(r"\b(search|web|google|look up|latest|news|current|weather|forecast|stock price|price of|website|url|https?://|www\.)\b"):
        domains.add("web")
    if has(r"\b(research|deep dive|investigate|look into)\b"):
        domains.add("web")
    if has(r"\b(open|show|toggle|turn on|turn off|disable|enable|switch model|change model|settings|theme|panel)\b"):
        domains.add("ui")
    if has(r"\b(session|chat history|rename chat|delete chat|archive chat|fork chat|list chats)\b"):
        domains.add("sessions")
    if has(r"\b(file|folder|directory|repo|git|grep|find in files|read file|edit file|shell|terminal|bash|python)\b"):
        domains.add("files")
    if has(r"\b(endpoint|api token|mcp|webhook|preference|configure|config|setting)\b"):
        domains.add("settings")

    low_signal = not continuation and not domains
    return {
        "low_signal": low_signal,
        "continuation": continuation,
        "domains": domains,
        "retrieval_query": retrieval_query,
    }


