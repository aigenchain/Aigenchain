"""
Agent Helpers.

Extracted from src.agent_loop
"""

from typing import Dict, List


async def _run_verifier_subagent(
    instruction: str, actions_snapshot: str,
    *, endpoint_url: str, model: str, headers: dict,
) -> list:
    """Fresh-context completion verifier. A second model instance with NO
    shared history reads the user's request + a record of what the agent did
    and judges whether the task is genuinely complete. The independent context
    is the whole point: a model checking its own work rationalizes; one that
    didn't do the work reads it cold. Returns a list of failure reasons
    (empty = pass, or silently empty on any error so it can't block a valid
    completion)."""
    from src.llm_core import llm_call_async
    prompt = (
        "You are an independent verifier. Another assistant just claimed the "
        "following task is complete. Using ONLY the request and the record of "
        "what it actually did, decide whether that claim is correct. Be strict: "
        "only say SUCCESS if the work genuinely satisfies the request.\n\n"
        f"<user_request>\n{(instruction or '')[:4000]}\n</user_request>\n\n"
        f"<actions_taken>\n{actions_snapshot[:8000]}\n</actions_taken>\n\n"
        "<checklist>\n"
        "1. Every concrete deliverable the request asked for was actually produced\n"
        "2. Outputs/edits match what was asked — nothing missing, no extra or unrequested changes\n"
        "3. Tool results show success, not errors or empty output that got ignored\n"
        "4. Anything the request said to leave alone was left unchanged\n"
        "</checklist>\n\n"
        "Reason briefly (2-3 sentences max). Then output EXACTLY one of:\n"
        "  VERIFICATION: SUCCESS\n"
        "  VERIFICATION: FAIL: <one short sentence per issue, semicolon-separated>\n"
        "Output nothing after the VERIFICATION line."
    )
    try:
        raw = await llm_call_async(
            url=endpoint_url, model=model,
            messages=[{"role": "user", "content": prompt}],
            headers=headers, temperature=0.0, max_tokens=600, timeout=60,
        )
    except Exception as e:
        logger.warning(f"[agent] verifier subagent failed: {e}")
        return []
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL | re.IGNORECASE)
    last_v = None
    for line in raw.splitlines():
        if "VERIFICATION:" in line:
            last_v = line.strip()
    if not last_v or "VERIFICATION: FAIL:" not in last_v:
        return []
    reasons = last_v.split("VERIFICATION: FAIL:", 1)[1].strip()
    return [r.strip() for r in reasons.split(";") if r.strip()]


def _empty_response_fallback(
    full_response: str,
    round_reasoning: str,
    tool_events: list,
) -> tuple:
    """Return (final_response, sse_chunk_or_none) for the end-of-loop empty-response guard.

    When a thinking model routes all tokens to reasoning_content (leaving
    content=""), full_response is empty but round_reasoning has content.
    The reasoning was already streamed as {thinking:true} chunks — do not
    re-emit it as a normal delta.  Just persist it and yield nothing.

    Returns:
        (final_response: str, chunk: str | None)
            chunk is the SSE string to yield, or None if nothing should be emitted.
    """
    if full_response.strip() or tool_events:
        return full_response, None
    if round_reasoning.strip():
        return round_reasoning, None
    _error_msg = "The model returned an empty response. Please try again or switch to a different model."
    return _error_msg, f'data: {json.dumps({"delta": _error_msg})}\n\n'


PLAN_MODE_DIRECTIVE = (
    "## PLAN MODE — OVERRIDES EVERYTHING ELSE BELOW\n"
    "You are in PLAN MODE. Your ONLY job this turn is to PROPOSE a plan. You have "
    "NOT done anything yet. Do NOT claim you created, wrote, ran, sent, or changed "
    "anything — that would be a lie.\n"
    "\n"
    "ABSOLUTE RULE — DO NOT MUTATE ANYTHING. Every write/state-changing tool, "
    "including the shell (`bash`/`python`), is disabled this turn and will be "
    "rejected — only read-only tools remain available. Use the read-only tools "
    "listed below (read files, search code, browse the project, web lookups) to "
    "ground the plan. If the task is 'write a file', your plan is to DESCRIBE "
    "writing it — you do NOT write it now.\n"
    "\n"
    "OUTPUT: present the plan as a GitHub-style checklist, one concrete step per line:\n"
    "- [ ] first action you will take once approved\n"
    "- [ ] next action\n"
    "Each item = one concrete action (file to create/edit, command to run, side "
    "effect). Do not execute. Do not end with 'Done' or anything implying the work "
    "is finished. End your turn with the checklist."
)


def build_active_plan_note(approved_plan: str) -> str:
    """System note that pins an approved plan during execution.

    Sent back by the frontend each turn so a long plan on a weak model survives
    history truncation — the agent can always re-read it. Returns "" for empty
    input.
    """
    if not approved_plan or not approved_plan.strip():
        return ""
    return (
        "## ACTIVE PLAN (approved — execute this)\n"
        "You are executing a plan the user already approved. THE FULL PLAN IS "
        "BELOW — it is always provided here every turn. Do NOT say you lost it, "
        "and do NOT look for it in tasks, notes, memory, files, or the API; just "
        "read it below. Work through it IN ORDER. After finishing each step, call "
        "the `update_plan` tool with the full checklist and that step marked "
        "`- [x]` so progress stays visible in the user's plan window. If the user "
        "asks to change the plan, call `update_plan` with the revised checklist. "
        "Do the next unchecked item until all are done. Do not skip, reorder, or "
        "invent steps; if a step is genuinely impossible, say so and stop.\n\n"
        "Current plan:\n"
        + approved_plan.strip()
    )


def _detect_runaway_call(call_freq, threshold=15):
    """Tool name of a call signature repeated >= ``threshold`` times — a real
    runaway loop. Counts IDENTICAL repeated calls (same tool AND args), so a
    legitimate batch of distinct calls to one tool (e.g. creating 18 calendar
    events at once) is NOT flagged. Returns ``None`` when nothing is runaway.

    ``call_freq`` is a Counter keyed by ``"{tool_type}:{content[:120]}"``.
    """
    sig = next((s for s, n in call_freq.items() if n >= threshold), None)
    return sig.split(":", 1)[0] if sig else None


