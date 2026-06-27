"""
Cookbook Actions.

Extracted from src.builtin_actions
"""

from typing import Tuple


async def action_cookbook_serve(
    owner: str,
    task_name: str = "",
    progress_cb=None,
    command: str = "",
    **kwargs,
) -> Tuple[str, bool]:
    """Launch a Cookbook model serve as a scheduled task.

    `command` is the JSON config string the task carries in `prompt`,
    of shape: {"preset": "name"} OR {"repo_id": "...", "cmd": "...", "host": "..."}.
    Optional `end_after_min: N` schedules a hard-stop N minutes after launch
    (handled by cookbook_serve_lifecycle_loop in src/cookbook_serve_lifecycle.py).
    """
    import json
    import time as _time
    import httpx
    from pathlib import Path
    from core.middleware import INTERNAL_TOOL_HEADER, INTERNAL_TOOL_TOKEN
    from core.atomic_io import atomic_write_json

    headers = {INTERNAL_TOOL_HEADER: INTERNAL_TOOL_TOKEN}
    try:
        cfg = json.loads(command or "{}")
    except Exception:
        return f"Invalid JSON config: {command!r}", False
    if not isinstance(cfg, dict):
        return "Config must be a JSON object", False

    # Resolve the preset (if named) OR fall through with explicit fields.
    preset_name = (cfg.get("preset") or "").strip()
    repo_id = (cfg.get("repo_id") or "").strip()
    cmd = (cfg.get("cmd") or "").strip()
    host = (cfg.get("host") or cfg.get("remote_host") or "").strip()
    try:
        end_after_min = int(cfg.get("end_after_min") or 0)
    except Exception:
        end_after_min = 0

    state_path = Path(COOKBOOK_STATE_FILE)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    except Exception:
        state = {}

    # Preset lookup. Try three matching strategies in order so the
    # schedule still works even when the user's preset is named
    # differently from the model's short name:
    #
    #   1. Exact preset.name == preset_name (case-insensitive)
    #   2. preset.model / preset.modelId == repo_id  (caller knows the repo)
    #   3. preset.model's short name (after final /) == preset_name
    #
    # Without #2 and #3, scheduling "Qwen3.5-397B-A17B-AWQ" failed when
    # the saved preset was named "vllm-qwen-397b" or had the model field
    # populated with the full HF repo path. Either should resolve.
    def _short(name: str) -> str:
        return (name or "").rsplit("/", 1)[-1].lower()

    if not cmd or not repo_id:
        presets = state.get("presets") or []
        chosen = None
        # Strategy 1: exact name match.
        if preset_name:
            chosen = next(
                (p for p in presets if isinstance(p, dict)
                 and (p.get("name") or "").lower() == preset_name.lower()),
                None,
            )
        # Strategy 2: repo_id matches the preset's model field.
        if chosen is None and repo_id:
            chosen = next(
                (p for p in presets if isinstance(p, dict)
                 and (p.get("model") or p.get("modelId") or "").lower() == repo_id.lower()),
                None,
            )
        # Strategy 3: model's short name matches the preset_name.
        if chosen is None and preset_name:
            chosen = next(
                (p for p in presets if isinstance(p, dict)
                 and _short(p.get("model") or p.get("modelId") or "") == preset_name.lower()),
                None,
            )
        if chosen is not None:
            repo_id = repo_id or chosen.get("model") or chosen.get("modelId") or ""
            cmd = cmd or (chosen.get("cmd") or "").strip()
            host = host or chosen.get("host") or chosen.get("remoteHost") or ""
    if not repo_id or not cmd or cmd.startswith("(adopted"):
        # Surface what we tried so the user can name their preset to match.
        preset_names = [(p.get("name") or "") for p in (state.get("presets") or []) if isinstance(p, dict)]
        hint = f" Saved presets: {preset_names!r}" if preset_names else ""
        return (f"No launchable config for {preset_name!r} (repo_id={repo_id!r}). "
                f"Check Cookbook → Presets has a real cmd, not 'adopted'.{hint}", False)

    # Resolve env_prefix etc. from the host's saved cookbook server entry,
    # matching the chat agent's serve_model path.
    body = {"repo_id": repo_id, "cmd": cmd}
    if host:
        body["remote_host"] = host
    env = (state.get("env") or {})
    srv = next(
        (s for s in (env.get("servers") or [])
         if isinstance(s, dict) and (s.get("host") == host or s.get("name") == host)),
        {},
    )
    if srv.get("env") == "venv" and srv.get("envPath"):
        body["env_prefix"] = f"source {srv['envPath']}/bin/activate"
    elif srv.get("env") == "conda" and srv.get("envPath"):
        body["env_prefix"] = f"conda activate {srv['envPath']}"
    if srv.get("hfToken"): body["hf_token"] = srv["hfToken"]
    if srv.get("port"): body["ssh_port"] = str(srv["port"])
    if srv.get("platform"): body["platform"] = srv["platform"]

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{internal_api_base()}/api/model/serve",
                                  json=body, headers=headers)
            data = r.json() if r.content else {}
    except Exception as e:
        return f"Launch HTTP failed: {e}", False
    if not data.get("ok"):
        return f"Launch rejected: {data.get('error') or data.get('detail') or 'unknown'}", False

    sid = data.get("session_id") or ""
    # Register the new task in cookbook_state.json + stamp it with our
    # scheduler-owner markers. /api/model/serve spawns the tmux session
    # but leaves the state-write to the UI — when a scheduled action
    # launches a serve from server-side, NOBODY writes the task into
    # state, so the Cookbook tab never shows it. We do the write here.
    if sid:
        try:
            # Re-read fresh (the route may have updated state already).
            try:
                fresh = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                fresh = {}
            if not isinstance(fresh, dict):
                fresh = {}
            tasks = fresh.get("tasks") if isinstance(fresh.get("tasks"), list) else []
            existing = next(
                (t for t in tasks if isinstance(t, dict) and t.get("sessionId") == sid),
                None,
            )
            if existing is None:
                display_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
                placeholder = (
                    f"Launched by scheduled task {task_name!r} — waiting for tmux output…\n"
                    f"  session: {sid}\n"
                    f"  target:  {host or 'local'}\n"
                    f"  cmd:     {cmd[:200]}{'…' if len(cmd) > 200 else ''}"
                )
                existing = {
                    "id": sid,
                    "sessionId": sid,
                    "name": display_name,
                    "modelId": repo_id,
                    "type": "serve",
                    "status": "running",
                    "output": placeholder,
                    "ts": int(_time.time() * 1000),
                    "payload": {"repo_id": repo_id, "remote_host": host or "", "_cmd": cmd},
                    "remoteHost": host or "",
                    "sshPort": "",
                    "platform": "linux",
                    "_serveReady": False,
                    "_endpointAdded": False,
                }
                tasks.append(existing)
            # Stamp ownership + end-at on the task entry.
            existing["_scheduledByTask"] = task_name or ""
            existing["_scheduledByOwner"] = owner or ""
            if end_after_min > 0:
                existing["_scheduledStopAtMs"] = int(_time.time() * 1000) + end_after_min * 60 * 1000
            fresh["tasks"] = tasks
            atomic_write_json(state_path, fresh)
        except Exception as e:
            logger.warning(f"cookbook_serve: state register/stamp failed: {e}")
    # Don't try to render absolute clock time in the message — the
    # server runs in UTC (Docker default), the user reads it as local,
    # and the offset depends on the user's TZ which the action doesn't
    # have a reliable handle on. The Tasks UI already shows the RUN
    # timestamp in the user's local time right above this message, so
    # "stops 8 min after that" gives the user everything they need.
    if end_after_min:
        return (
            f"Launched {repo_id} (session {sid}); stops {end_after_min} min after this ran",
            True,
        )
    return f"Launched {repo_id} (session {sid})", True


BUILTIN_ACTIONS = {
    "tidy_sessions": action_tidy_sessions,
    "tidy_documents": action_tidy_documents,
    "consolidate_memory": action_consolidate_memory,
    "tidy_research": action_tidy_research,
    "summarize_emails": action_summarize_emails,
    "draft_email_replies": action_draft_email_replies,
    "extract_email_events": action_extract_email_events,
    "classify_events": action_classify_events,
    # ping_events removed from the user-facing registry. Calendar reminders
    # are represented as Notes, so note pings are the single dispatch path.
    "daily_brief": action_daily_brief,
    "learn_sender_signatures": action_learn_sender_signatures,
    "ssh_command": action_ssh_command,
    "run_script": action_run_script,
    "run_local": action_run_local,
    "test_skills": action_test_skills,
    "audit_skills": action_audit_skills,
    "check_email_urgency": action_check_email_urgency,
    "cookbook_serve": action_cookbook_serve,
    # ping_notes removed from the registry — runs only inside `_note_pings_loop`.
}

# Descriptions for the UI/API
BUILTIN_ACTION_INFO = {
    "tidy_sessions": "Clean up empty chat sessions and auto-sort into folders",
    "tidy_documents": "Remove junk/empty documents",
    "consolidate_memory": "Remove duplicate memories",
    "tidy_research": "Remove orphaned research files (sessions that were deleted)",
    "summarize_emails": "Pre-generate AI summaries for new inbox emails",
    "draft_email_replies": "Pre-draft AI reply suggestions for new inbox emails",
    "extract_email_events": "Scan emails for booking/meeting confirmations and auto-add to calendar",
    "classify_events": "Tag upcoming events with importance (low/normal/high/critical) and type (work/health/travel/etc.); colors them too",
    "daily_brief": "Build a morning digest: today's calendar, unread email count + top senders, active todos",
    "learn_sender_signatures": "LLM learns each sender's signature from 3+ of their recent emails; cached per address so future renders fold sigs reliably without heuristics",
    "ssh_command": "Run a shell command on a local or remote host",
    "run_script": "Run a script locally or on ODYSSEUS_SCRIPT_HOST",
    "test_skills": "Run the per-skill Test on every skill: agent run + LLM judge → records verdict on the skill (pass/needs_work/fail/inconclusive). Advisory only — never rewrites or demotes anything.",
    "audit_skills": "Audit unaudited skills after enough new skills are added: test, narrow metadata, self-edit/retry, optional teacher rewrite, tag duplicates/trivial skills, and publish/draft using the auto-approve threshold.",
    "check_email_urgency": "Scan unread emails hourly, tag urgent/reply-soon/newsletter/marketing/spam, and send a reminder when a new email needs a fast reply.",
}
