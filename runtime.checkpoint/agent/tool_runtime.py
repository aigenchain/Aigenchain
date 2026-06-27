"""
Agent Tool Runtime.

Extracted from src.agent_loop
"""

from typing import Dict, List


def _resolve_tool_blocks(round_response: str, native_tool_calls: list, round_num: int, is_api_model: bool = False):
    """Choose native function calls or fenced code block parsing. Returns (tool_blocks, used_native)."""
    used_native = False
    if native_tool_calls:
        tool_blocks = []
        for tc in native_tool_calls:
            tc_name = tc.get("name", "")
            tc_args = tc.get("arguments", "{}")
            block = function_call_to_tool_block(tc_name, tc_args)
            if block:
                tool_blocks.append(block)
                logger.info(f"  -> converted: {tc_name} -> {block.tool_type}")
            else:
                logger.warning(f"  -> FAILED to convert native call: {tc_name} args={tc_args[:200]}")
        if tool_blocks:
            used_native = True
    if not used_native:
        # Native function-calling models (GPT/Claude/Grok/Qwen3/DeepSeek-V, etc.)
        # have a reliable structured channel for real tool invocations. When such
        # a model emits no native tool_calls, any ```bash/```python/```json fence
        # in its prose is virtually always an illustrative example for the user
        # (e.g. "here's the command you'd run"), not an attempted tool call —
        # executing it causes accidental runs and clarification loops (#3222).
        #
        # Gate ONLY that fenced-block pattern for native models, not the whole
        # parser: explicit [TOOL_CALL]/<invoke>/<tool_code>/DSML markup that
        # leaks into content as text is never illustrative — it's a real call
        # the model couldn't emit on its structured channel (e.g. DeepSeek-V
        # falling back to DSML). Dropping the whole parser would silently lose
        # those too. Non-native / textual-only models keep every pattern,
        # fenced blocks included, since that's their *only* tool channel.
        tool_blocks = parse_tool_blocks(round_response, skip_fenced=is_api_model)
        if tool_blocks:
            logger.info(f"Agent round {round_num}: {len(tool_blocks)} fenced tool block(s) detected")

    resp_preview = round_response[:200].replace('\n', '\\n') if round_response else "(empty)"
    logger.info(f"Agent round {round_num} summary: {len(round_response)} chars, "
                f"{len(native_tool_calls)} native calls, "
                f"{len(tool_blocks)} tool blocks. Preview: {resp_preview}")

    return tool_blocks, used_native


def _append_tool_results(
    messages: List[Dict],
    round_response: str,
    native_tool_calls: list,
    tool_results: list,
    tool_result_texts: list,
    used_native: bool,
    round_num: int,
    round_reasoning: str = "",
):
    """Append tool execution results back into the message history for the next LLM round.

    `round_reasoning` (DeepSeek / vLLM reasoning-parser deltas) is echoed
    back via `reasoning_content` on the assistant message — DeepSeek's API
    rejects follow-up requests in thinking mode that don't include the
    prior reasoning.

    NOTE: it is NOT universally ignored. Nemotron's chat template re-injects
    EVERY prior `reasoning_content` as a <think> block, and this agent loop is
    trimmed only once (before the loop), so across rounds the reasoning piles
    up unbounded — bloating context and feeding the model its own prior
    reasoning, which reinforces repetition/looping. So keep reasoning_content
    on the MOST RECENT assistant turn only: enough for DeepSeek continuity,
    without the per-round accumulation.
    """
    # Strip reasoning_content from earlier assistant turns; only the newest keeps it.
    for _m in messages:
        if _m.get("role") == "assistant":
            _m.pop("reasoning_content", None)
    if used_native and native_tool_calls:
        assistant_msg = {"role": "assistant"}
        # When the model emitted ONLY tool calls (no prose), content must be
        # null, NOT an empty string. Google Gemini's OpenAI-compatible endpoint
        # and Ollama both reject an assistant message that carries tool_calls
        # alongside empty-string content with HTTP 400 ("contents is not
        # specified" / a JSON parse error), which aborts every tool-using turn
        # at the follow-up round. null (i.e. omitted text) is the spec-correct
        # form the OpenAI SDK itself emits, and OpenAI/Anthropic accept it too.
        assistant_msg["content"] = round_response if round_response.strip() else None
        if round_reasoning:
            assistant_msg["reasoning_content"] = round_reasoning
        assistant_msg["tool_calls"] = [
            {
                "id": tc.get("id", f"call_{round_num}_{j}"),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", "{}"),
                },
                # Gemini 3 requires the opaque thought_signature it returned with
                # each function call to be echoed back on the follow-up turn, or
                # the next request 400s. Replay it when present; other providers
                # never emit it (their payload builders just ignore the field).
                **({"extra_content": tc["extra_content"]} if tc.get("extra_content") else {}),
            }
            for j, tc in enumerate(native_tool_calls)
        ]
        messages.append(assistant_msg)
        for j, tc in enumerate(native_tool_calls):
            result_text = tool_result_texts[j] if j < len(tool_result_texts) else ""
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{round_num}_{j}"),
                "content": result_text,
            })
    else:
        tool_output_text = "\n\n".join(tool_results)
        msg = {"role": "assistant", "content": round_response}
        if round_reasoning:
            msg["reasoning_content"] = round_reasoning
        messages.append(msg)
        messages.append(
            {"role": "user", "content": f"[Tool execution results]\n\n{tool_output_text}"}
        )


def _compute_final_metrics(
    messages: List[Dict],
    full_response: str,
    total_duration: float,
    time_to_first_token,
    context_length: int,
    real_input_tokens: int,
    real_output_tokens: int,
    has_real_usage: bool,
    tool_events: list,
    round_texts: list,
    model: str = "",
    last_round_input_tokens: int = 0,
    prep_timings: Optional[Dict[str, float]] = None,
    backend_gen_tps: float = 0,
    backend_prefill_tps: float = 0,
) -> dict:
    """Compute token counts, TPS, and build the final metrics dict."""
    if has_real_usage:
        input_tokens = real_input_tokens
        output_tokens = real_output_tokens
    else:
        input_content = ""
        for msg in messages:
            if isinstance(msg.get("content"), str):
                input_content += msg["content"] + "\n"
        input_tokens = len(input_content) // 4
        output_tokens = len(full_response) // 4
    # Prefer the backend's true generation speed (llama.cpp
    # timings.predicted_per_second) — pure decode, no prefill/tool/network time.
    # Fall back to tokens/wall-clock only when the backend didn't report it
    # (e.g. cloud APIs without timings); that figure reads low because
    # total_duration includes prefill + agent overhead.
    if backend_gen_tps and backend_gen_tps > 0:
        tps = backend_gen_tps
    else:
        tps = output_tokens / total_duration if total_duration > 0 else 0
    # Use last round's input tokens for context % (peak usage) when available
    ctx_tokens = last_round_input_tokens if last_round_input_tokens > 0 else input_tokens
    ctx_pct = min(round((ctx_tokens / context_length) * 100, 1), 100.0) if context_length else 0

    metrics = {
        "response_time": round(total_duration, 2),
        "time_to_first_token": round(time_to_first_token, 2) if time_to_first_token else 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": round(tps, 2),
        # True decode speed when the backend reported it; "computed" = the
        # tokens/wall-clock fallback (reads low — includes prefill/overhead).
        "tps_source": "backend" if (backend_gen_tps and backend_gen_tps > 0) else "computed",
        "total_tokens": input_tokens + output_tokens,
        "context_length": context_length,
        "context_percent": ctx_pct,
        "usage_source": "real" if has_real_usage else "estimated",
        "model": model,
    }
    if backend_prefill_tps and backend_prefill_tps > 0:
        metrics["prefill_tps"] = round(backend_prefill_tps, 2)
    if prep_timings:
        prep_total = round(sum(prep_timings.values()), 3)
        metrics["agent_prep_time"] = prep_total
        metrics["agent_model_wait_time"] = round(max((time_to_first_token or 0) - prep_total, 0), 3)
        metrics["agent_prep_breakdown"] = {
            key: round(value, 3) for key, value in prep_timings.items()
        }
    if tool_events:
        metrics["tool_events"] = tool_events
        metrics["round_texts"] = round_texts
    return metrics


# ── Completion verifier ──
# Tools whose effects produce a checkable artifact. A turn that used one of
# these is "effectful" and worth an independent completion check; pure
# read-only / Q&A turns are not.
_VERIFIER_EFFECTFUL_TOOLS = {
    "create_document", "update_document", "edit_document",
    "bash", "python", "write_file",
}
_VERIFIER_MAX_ROUNDS = 2  # cap re-verify cycles per turn — never loop forever


def _build_actions_snapshot(tool_events: list, limit: int = 8000) -> str:
    """Compact record of what the agent actually did this turn, for the
    verifier to judge against. One block per tool execution: the command and
    a head of its output."""
    parts = []
    for ev in tool_events:
        tool = ev.get("tool", "?")
        cmd = (ev.get("command") or "").strip()
        out = (ev.get("output") or "").strip()
        rc = ev.get("exit_code")
        head = f"[{tool}] {cmd}" if cmd else f"[{tool}]"
        rc_s = f" (exit {rc})" if rc not in (None, 0) else ""
        body = (out[:1200] + " …") if len(out) > 1200 else (out or "(no output)")
        parts.append(f"{head}{rc_s}\n-> {body}")
    snap = "\n\n".join(parts)
    return snap[:limit] if len(snap) > limit else snap


