"""
Scheduler delivery.

Extracted from src.task_scheduler
"""


def _task_needs_model_slot(self, task_id: str) -> bool:
    """Only LLM/research/model-backed actions should wait in the model
    queue. Pure housekeeping actions can run immediately."""
    from core.database import SessionLocal, ScheduledTask

    db = SessionLocal()
    try:
        task = db.query(ScheduledTask).filter(ScheduledTask.id == task_id).first()
        if not task:
            return True
        task_type = getattr(task, "task_type", "") or "llm"
        if task_type != "action":
            return True
        return (getattr(task, "action", "") or "") in self._MODEL_BACKED_ACTIONS
    finally:
        db.close()

def _log_to_assistant(self, db, task, result_text: str):
    """Log a task result to the assistant's chat session."""
    # Don't double-log check-ins (they already save directly)
    if "check-in" in (task.name or "").lower():
        return
    # Built-in housekeeping noise stays out of the chat.
    if (getattr(task, "action", "") or "") in self._SILENT_ACTIONS:
        return
    from src.assistant_log import log_to_assistant
    log_to_assistant(
        task.owner,
        result_text[:1000],
        category=(task.name or "Task"),
    )

async def _execute_action(self, task, run_id: str | None = None) -> tuple:
    """Execute a built-in action (no LLM needed)."""
    from src.builtin_actions import BUILTIN_ACTIONS

    action_fn = BUILTIN_ACTIONS.get(task.action)
    if not action_fn:
        return f"Unknown action: {task.action}", False

    from src.builtin_actions import TaskNoop
    try:
        # Pass task prompt as script/command for ssh_command/run_script actions.
        def _progress(message: str):
            self._set_run_progress(run_id, message)

        kwargs = {"owner": task.owner, "task_name": task.name, "progress_cb": _progress}
        if task.action in ("run_script", "run_local", "ssh_command") and task.prompt:
            kwargs["script" if task.action in ("run_script", "run_local") else "command"] = task.prompt
        # cookbook_serve carries its JSON config in task.prompt — feed it
        # through as `command` so action_cookbook_serve can json.loads it.
        elif task.action == "cookbook_serve" and task.prompt:
            kwargs["command"] = task.prompt
        result, success = await action_fn(**kwargs)
        return result, success
    except TaskNoop:
        # Bubble up so _execute_task_locked can drop the run row silently.
        raise
    except Exception as e:
        logger.error(f"Action '{task.action}' failed: {e}")
        return str(e), False

# ── Check-in source discovery ──
# Pattern-based: if an MCP server has a tool matching a pattern, it becomes
# a check-in source. Add new patterns here to support new integrations —
# no code changes needed elsewhere.
CHECKIN_MCP_PATTERNS = [
    {"detect": "list_emails",   "section": "Email",    "tool": "list_emails",
     "args": {"mailbox": "INBOX", "limit": 10, "unread_only": True},
     "label_from_identity": True,
     "formatter": "_format_email_output"},
    {"detect": "search_emails", "section": "Email",    "tool": "search_emails",
     "args": {"query": "is:unread", "limit": 10},
     "label_from_identity": True,
     "formatter": "_format_email_output"},
    {"detect": "get_feed",      "section": "RSS",      "tool": "get_feed",
     "args": {},
     "label_from_identity": False},
    {"detect": "list_feeds",    "section": "RSS",      "tool": "list_feeds",
     "args": {},
     "label_from_identity": False},
    {"detect": "list_messages", "section": "Messages", "tool": "list_messages",
     "args": {"limit": 10},
     "label_from_identity": True},
]

@staticmethod
def _format_email_output(raw: str) -> str:
    """Clean up raw MCP email list output into readable format."""
    import re as _re
    lines = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip header lines like "📬 [INBOX] 856 emails..."
        if line.startswith(("\U0001f4ec", "📬", "No emails", "---", "Page ")):
            continue
        # Skip "more pages available" etc
        if "page" in line.lower() and "/" in line:
            continue
        # Parse: [1778] Re: Subject From: Name | Date
        m = _re.match(r'\[?\d+\]?\s*(?:↩️\s*|📎\s*|🔵\s*|⭐\s*)?(.+?)(?:\s*From:\s*(.+?))?(?:\s*\|\s*(\S+))?$', line)
        if m:
            subject = m.group(1).strip().rstrip('|').strip()
            sender = (m.group(2) or "").strip().rstrip('|').strip()
            if sender:
                lines.append(f"- {sender} — {subject}")
            else:
                lines.append(f"- {subject}")
        elif line.startswith("[") or line.startswith("-"):
            # Generic cleanup
            cleaned = _re.sub(r'^\[?\d+\]?\s*(?:↩️\s*|📎\s*)?', '', line.lstrip('- '))
            if cleaned.strip():
                lines.append(f"- {cleaned.strip()}")
    if not lines:
        return "No unread emails"
    return "\n".join(lines[:10])

async def _execute_checkin(self, task, crew, db, session_id: str,
                           endpoint_url: str, model: str) -> str:
    """Gather raw data from all integrations, hand it to the LLM to write the check-in."""
    from runtime.tools.notes import do_manage_notes
    from src.tool_utils import get_mcp_manager

    tz_name = _resolve_task_timezone(db, task)
    try:
        if tz_name:
            from zoneinfo import ZoneInfo
            from datetime import timezone, timedelta
            now = _utcnow().replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz_name))
        else:
            from datetime import timedelta
            now = _utcnow()
        time_str = now.strftime("%A, %B %d %Y, %H:%M")
    except Exception:
        from datetime import timedelta
        now = _utcnow()
        time_str = now.strftime("%H:%M UTC")

    raw = {}

    # Calendar: today+tomorrow, this week, month ahead
    # Pull directly from DB so we can include event_type and importance.
    try:
        from core.database import SessionLocal as _SL, CalendarEvent as _CE
        _db = _SL()
        try:
            for label, start, end in _digest_windows(now):
                # Strip timezone for naive DB comparison
                _s = start.replace(tzinfo=None) if start.tzinfo else start
                _e = end.replace(tzinfo=None) if end.tzinfo else end
                evs = _db.query(_CE).filter(
                    _CE.dtstart >= _s,
                    _CE.dtstart <= _e,
                    _CE.status != "cancelled",
                ).order_by(_CE.dtstart).all()
                if not evs:
                    continue
                # Group by importance for richer output
                by_imp = {"critical": [], "high": [], "normal": [], "low": []}
                for ev in evs:
                    imp = (ev.importance or "normal").lower()
                    by_imp.setdefault(imp, []).append(ev)
                lines = []
                for tier in ("critical", "high", "normal", "low"):
                    items = by_imp.get(tier, [])
                    if not items:
                        continue
                    marker = {"critical": "[!!]", "high": "[!]", "normal": "  ", "low": " ·"}[tier]
                    for ev in items:
                        t = ev.dtstart.strftime("%a %b %d %H:%M")
                        tag = f" ({ev.event_type})" if ev.event_type else ""
                        loc = f" @ {ev.location}" if ev.location else ""
                        lines.append(f"{marker} {t} — {ev.summary}{tag}{loc}")
                if lines:
                    raw[f"calendar_{label}"] = "\n".join(lines)
        finally:
            _db.close()
    except Exception as e:
        raw["calendar"] = f"Error: {e}"

    # Notes/Tasks
    try:
        r = await do_manage_notes(json.dumps({"action": "list"}), owner=task.owner)
        raw["notes_tasks"] = r.get("results") or r.get("response") or "No notes"
    except Exception as e:
        raw["notes_tasks"] = f"Error: {e}"

    # Auto-discover API integrations (Miniflux RSS, etc.).
    try:
        import httpx
        from src.integrations import load_integrations
        for integ in load_integrations():
            if not integ.get("enabled"):
                continue
            preset = integ.get("preset", "")
            base_url = integ.get("base_url", "").rstrip("/")
            api_key = integ.get("api_key", "")
            if not base_url:
                continue

            # Build auth headers
            headers = {}
            if integ.get("auth_type") == "header" and api_key:
                headers[integ.get("auth_header", "X-Auth-Token")] = api_key
            elif integ.get("auth_type") == "bearer" and api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            # Miniflux: fetch unread entries (cached 3 min across tasks)
            if preset == "miniflux":
                async def _fetch_miniflux(_base=base_url, _headers=dict(headers)):
                    async with httpx.AsyncClient(timeout=10) as client:
                        resp = await client.get(
                            f"{_base}/v1/entries",
                            params={"status": "unread", "limit": 15, "order": "published_at", "direction": "desc"},
                            headers=_headers,
                        )
                        if resp.status_code != 200:
                            return None
                        entries = resp.json().get("entries", []) or []
                        if not entries:
                            return None
                        lines = []
                        for e in entries[:15]:
                            title = e.get("title", "?")
                            feed = (e.get("feed") or {}).get("title", "?")
                            url = e.get("url", "")
                            lines.append(f"- [{feed}] {title} — {url}")
                        return "\n".join(lines)
                try:
                    val = await _cached(("miniflux_unread", base_url), 180, _fetch_miniflux)
                    if val:
                        raw["rss_miniflux_unread"] = val
                except Exception as e:
                    logger.warning(f"Miniflux fetch failed: {e}")
    except Exception as e:
        logger.warning(f"Integrations discovery failed: {e}")

    # Auto-discover MCP sources
    mcp = get_mcp_manager()
    if mcp:
        discovered = set()
        for server_id, tools in mcp._tools.items():
            if mcp.is_builtin(server_id):
                continue
            conn = mcp._connections.get(server_id, {})
            if conn.get("status") != "connected":
                continue
            identity = conn.get("identity", "")
            tool_names = {t["name"] for t in tools}
            for pattern in self.CHECKIN_MCP_PATTERNS:
                if pattern["detect"] not in tool_names:
                    continue
                key = f"{pattern['section']}_{server_id}"
                if key in discovered:
                    continue
                discovered.add(key)
                label = f"{pattern['section']} ({identity})" if identity else pattern["section"]
                qualified = f"mcp__{server_id}__{pattern['tool']}"
                args = dict(pattern.get("args", {}))
                args["account"] = "default"
                try:
                    # Cache 3 min: different scheduled tasks firing at the
                    # same minute share the same MCP snapshot.
                    async def _call_mcp(_q=qualified, _args=args):
                        return await mcp.call_tool(_q, _args)
                    cache_key = ("mcp_snapshot", qualified, json.dumps(args, sort_keys=True))
                    result = await _cached(cache_key, 180, _call_mcp)
                    if result.get("exit_code", 0) != 0:
                        continue
                    content = result.get("stdout") or result.get("output") or ""
                    if content.strip():
                        raw[label] = content[:3000]
                except Exception:
                    pass

    # Build the data dump and hand it to the LLM
    data_dump = f"Current time: {time_str}\n\n"
    for key, val in raw.items():
        data_dump += f"--- {key} ---\n{val}\n\n"

    context = (
        data_dump +
        f"---\n\n{task.prompt}\n\n"
        "Write the check-in. YOU decide what matters, what to skip, how to format. "
        "Only show future events. Calendar events are pre-tagged with importance: "
        "[!!] critical, [!] high, plain = normal, ' ·' = low. "
        "GROUP your output by importance — lead with critical/high, then normal, "
        "skip low entirely unless explicitly relevant. Mention event type (work/health/travel/etc) "
        "where it adds context (e.g. 'leave 1h early for travel'). "
        "Flag anything coming up that needs prep (birthdays, deadlines, holidays). "
        "Use tools to take action if needed. Keep it concise — no raw data dumps."
    )

    return await self._run_agent_loop(
        endpoint_url, model, task, session_id,
        system_prompt=(crew.personality or "").strip() if crew else None,
        disabled_tools=None, relevant_tools=None,
        override_user_message=context,
    )

async def _execute_llm_task(self, task, db) -> str:
    """Execute an LLM task with full tool access via the agent loop."""
    from core.database import Session as DbSession, ChatMessage, CrewMember

    # If this task is wired to a CrewMember (personal assistant, custom
    # crew), prefer the crew member's persona/model/endpoint as overrides.
    crew = None
    if getattr(task, "crew_member_id", None):
        try:
            crew = db.query(CrewMember).filter(CrewMember.id == task.crew_member_id).first()
        except Exception:
            crew = None

    # Determine endpoint + model
    endpoint_url = task.endpoint_url
    model = task.model
    if (not endpoint_url or not model) and crew:
        endpoint_url = endpoint_url or crew.endpoint_url
        model = model or crew.model
    if not endpoint_url or not model:
        endpoint_url, model = self._resolve_defaults(db, task.owner)
    if not endpoint_url or not model:
        raise RuntimeError("No model/endpoint configured")
    # Record the resolved model so _execute_task_locked can persist it on
    # the run (tasks rarely pin a model, so this is the only record of
    # which model actually produced the output).
    self._last_run_model = model

    # Ensure a session exists for output
    session_id = task.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        sess = DbSession(
            id=session_id,
            name=f"[Task] {task.name}",
            endpoint_url=endpoint_url,
            model=model,
            owner=task.owner,
            folder="Tasks",
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
        db.add(sess)
        task.session_id = session_id
        db.commit()
        if self._session_manager:
            try:
                self._session_manager.ensure_task_session(
                    session_id, f"[Task] {task.name}", endpoint_url, model,
                    owner=task.owner, task=task
                )
            except Exception:
                pass

    # For assistant check-ins: call each tool directly and post results
    # as separate messages. More reliable than hoping the model calls tools.
    is_checkin = crew and crew.is_default_assistant and "check-in" in (task.name or "").lower()
    if is_checkin:
        return await self._execute_checkin(task, crew, db, session_id, endpoint_url, model)

    # Build system prompt: crew member persona overrides the default.
    system_prompt = (
        (crew.personality or "").strip()
        if crew and crew.personality
        else "You are a helpful assistant executing a scheduled task. Use available tools to complete the task thoroughly."
    )
    # Inject current time so the model knows what's past vs upcoming
    tz_name = _resolve_task_timezone(db, task)
    try:
        if tz_name:
            from zoneinfo import ZoneInfo
            from datetime import timezone
            now_local = _utcnow().replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz_name))
            time_str = now_local.strftime("%A, %B %d %Y, %H:%M %Z")
        else:
            time_str = _utcnow().strftime("%A, %B %d %Y, %H:%M UTC")
    except Exception:
        time_str = _utcnow().strftime("%A, %B %d %Y, %H:%M UTC")
    system_prompt = f"Current time: {time_str}\n\n{system_prompt}"

    # Compute tool filter from CrewMember.enabled_tools if set
    disabled_tools = None
    if crew and crew.enabled_tools:
        try:
            enabled = json.loads(crew.enabled_tools)
            if isinstance(enabled, list) and enabled:
                from src.tool_index import BUILTIN_TOOL_DESCRIPTIONS
                all_tools = set(BUILTIN_TOOL_DESCRIPTIONS.keys())
                disabled_tools = all_tools - set(enabled)
        except Exception:
            pass

    # RAG-select relevant tools for this prompt + always-available assistant tools.
    # Without this, all 40+ tools get sent and models hit their tool limit.
    relevant_tools = None
    try:
        from src.tool_index import get_tool_index, ASSISTANT_ALWAYS_AVAILABLE
        tool_idx = get_tool_index()
        if tool_idx:
            rag_tools = tool_idx.get_tools_for_query(task.prompt or "", k=8)
            relevant_tools = (rag_tools | ASSISTANT_ALWAYS_AVAILABLE)
            if disabled_tools:
                relevant_tools -= disabled_tools
            logger.info(f"[assistant] RAG selected {len(rag_tools)} tools + {len(ASSISTANT_ALWAYS_AVAILABLE)} always-available = {len(relevant_tools)} total for '{task.name}'")
    except Exception as e:
        logger.warning(f"[assistant] RAG tool selection failed, using all: {e}")

    # Try using the agent loop for full tool access
    try:
        result = await self._run_agent_loop(
            endpoint_url, model, task, session_id,
            system_prompt=system_prompt, disabled_tools=disabled_tools,
            relevant_tools=relevant_tools,
        )
    except Exception as e:
        logger.warning(f"Agent loop failed for task '{task.name}', falling back to simple call: {e}")
        from src.llm_core import llm_call_async
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.prompt},
        ]
        result = await llm_call_async(url=endpoint_url, model=model, messages=messages, timeout=120)

    # Strip the model's chain-of-thought before saving/delivering. Task
    # output is LLM-only, so prose=True (which also removes untagged
    # "The user wants me to…" reasoning) is safe here — without this the
    # thinking leaked into the saved result.
    try:
        from src.text_helpers import strip_think
        result = strip_think(result or "", prose=True, prompt_echo=True).strip() or result
    except Exception:
        pass

    return result

async def _deliver_task_result(self, task, result: str, db, model: str = None):
    """Deliver a completed task result according to output_target.

    This is intentionally shared by LLM/research/action tasks so built-in
    actions cannot drift into hidden delivery paths that disagree with the
    task's visible output target.
    """
    from core.database import Session as DbSession, ChatMessage, CrewMember
    from core.models import ChatMessage as MemChatMessage

    output = task.output_target or "session"
    if (
        output == "session"
        and (getattr(task, "task_type", "") or "") == "action"
        and (getattr(task, "action", "") or "") in self._SILENT_ACTIONS
    ):
        return
    if output.startswith("mcp__"):
        await self._deliver_via_mcp(output, task, result)
        return

    if self._is_email_output_target(output):
        await self._deliver_via_email(output, task, result)
        return

    if output != "session":
        return

    endpoint_url = task.endpoint_url
    model_name = model or task.model
    crew = None
    if getattr(task, "crew_member_id", None):
        try:
            crew = db.query(CrewMember).filter(CrewMember.id == task.crew_member_id).first()
        except Exception:
            crew = None
    if (not endpoint_url or not model_name) and crew:
        endpoint_url = endpoint_url or crew.endpoint_url
        model_name = model_name or crew.model
    if not endpoint_url or not model_name:
        try:
            resolved_url, resolved_model = self._resolve_defaults(db, task.owner)
            endpoint_url = endpoint_url or resolved_url
            model_name = model_name or resolved_model
        except Exception:
            pass

    session_id = task.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        sess = DbSession(
            id=session_id,
            name=f"[Task] {task.name}",
            endpoint_url=endpoint_url or "",
            model=model_name or "",
            owner=task.owner,
            folder="Tasks",
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
        db.add(sess)
        task.session_id = session_id
        db.commit()
        if self._session_manager:
            try:
                self._session_manager.ensure_task_session(
                    session_id, f"[Task] {task.name}", endpoint_url, model_name,
                    owner=task.owner, task=task
                )
            except Exception:
                pass

    meta = {}
    if model_name:
        meta["model"] = model_name
    if crew and crew.is_default_assistant:
        meta.update({"source": "cron", "task_id": task.id, "task_name": task.name})

    # Use SessionManager for persistence so in-memory cache stays in sync
    if self._session_manager and session_id:
        try:
            self._session_manager.add_message(
                session_id,
                MemChatMessage(
                    "user",
                    task.prompt or f"[Task] {task.name}",
                    metadata=dict(meta),
                ),
            )
            self._session_manager.add_message(
                session_id,
                MemChatMessage(
                    "assistant",
                    result or "",
                    metadata=dict(meta),
                ),
            )
        except Exception:
            logger.exception("Failed to deliver task %s through SessionManager", task.id)
    else:
        # Fallback: raw DB write (no session manager available)
        msg_meta = json.dumps(meta)
        user_msg = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=task.prompt or f"[Task] {task.name}",
            timestamp=_utcnow(),
            meta_data=msg_meta,
        )
        assistant_msg = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=result or "",
            timestamp=_utcnow(),
            meta_data=msg_meta,
        )
        db.add(user_msg)
        db.add(assistant_msg)
        db.commit()

@staticmethod
def _is_email_output_target(output: str) -> bool:
    target = (output or "").strip()
    if target in {"email", "email:self"}:
        return True
    if target.startswith("email:"):
        return True
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", target))

async def _deliver_via_email(self, output: str, task, result: str):
    """Send task output through the app's configured SMTP account.

    Supported output_target values:
    - email / email:self: send to the account's From address
    - email:name@example.com or raw name@example.com: send there
    """
    from email.message import EmailMessage

    target = (output or "").strip()
    explicit = ""
    if target.startswith("email:"):
        explicit = target.split(":", 1)[1].strip()
    elif "@" in target:
        explicit = target

    try:
        from routes.email_routes import _resolve_send_config
        from routes.email_helpers import _send_smtp_message

        cfg = _resolve_send_config(owner=task.owner or "")
        to_addr = explicit or cfg.get("from_address") or cfg.get("smtp_user") or ""
        if not to_addr:
            raise RuntimeError("No email recipient resolved for task output")

        from_addr = cfg.get("from_address") or cfg.get("smtp_user") or to_addr
        msg = EmailMessage()
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg["Subject"] = f"[Task] {task.name}"
        msg["X-Odysseus-Origin"] = "odysseus-ui"
        msg["X-Odysseus-Kind"] = "task"
        msg["X-Odysseus-Ref"] = str(task.id)
        msg.set_content(result or "")
        _send_smtp_message(cfg, from_addr, [to_addr], msg.as_string(), timeout=30)
        logger.info("Task %s emailed result to %s (%sb)", task.id, to_addr, len(result or ""))
    except Exception as e:
        logger.error("Task %s email delivery failed: %s", task.id, e, exc_info=True)
        raise

async def _run_agent_loop(self, endpoint_url: str, model: str, task, session_id: str,
                          system_prompt: str | None = None,
                          disabled_tools: set | None = None,
                          relevant_tools: set | None = None,
                          override_user_message: str | None = None) -> str:
    """Run the full agent loop with tool access, collecting the final text."""
    from src.agent_loop import stream_agent_loop

    system_content = system_prompt or "You are a helpful assistant executing a scheduled task. Use available tools to complete the task thoroughly."
    user_content = override_user_message or task.prompt
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # Resolve headers from the endpoint's API key
    headers = {}
    try:
        from core.database import SessionLocal, ModelEndpoint
        from src.endpoint_resolver import normalize_base, build_headers
        from src.auth_helpers import owner_filter
        db2 = SessionLocal()
        try:
            ep_q = db2.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True)
            ep_q = owner_filter(ep_q, ModelEndpoint, task.owner or None)
            eps = ep_q.all()
            for ep in eps:
                if normalize_base(ep.base_url) in endpoint_url or endpoint_url in normalize_base(ep.base_url):
                    headers = build_headers(ep.api_key, normalize_base(ep.base_url))
                    break
        finally:
            db2.close()
    except Exception:
        pass
    full_text = ""
    tool_results = []

    # Honor per-task max_steps (defense against runaway agent loops).
    # Falls back to 20 if not set — the historical default.
    _task_max_rounds = task.max_steps if task.max_steps and task.max_steps > 0 else 20
    # Tasks are background workloads — they share the Utility model's
    # fallback chain (Settings → Utility Model → Fallbacks). A downed
    # primary endpoint won't silently yield `(no output)` — same recipe
    # chat uses but with the utility list (`utility_model_fallbacks`).
    try:
        from src.endpoint_resolver import resolve_utility_fallback_candidates
        _task_fallbacks = resolve_utility_fallback_candidates(owner=task.owner or None)
    except Exception:
        _task_fallbacks = []
    async for event_str in stream_agent_loop(
        endpoint_url=endpoint_url,
        model=model,
        messages=messages,
        max_rounds=_task_max_rounds,
        session_id=session_id,
        owner=task.owner,
        headers=headers,
        disabled_tools=disabled_tools,
        relevant_tools=relevant_tools,
        fallbacks=_task_fallbacks,
    ):
        if event_str.startswith("data: ") and not event_str.startswith("data: [DONE]"):
            try:
                data = json.loads(event_str[6:])
                # Capture text from all event types, not just delta
                if "delta" in data:
                    full_text += data["delta"]
                elif data.get("type") == "tool_output":
                    # Tool results — capture summary so we have SOMETHING even
                    # if the model never produces a final text response
                    tool_summary = data.get("stdout") or data.get("output") or data.get("result") or ""
                    if isinstance(tool_summary, str) and tool_summary.strip():
                        tool_results.append(f"[{data.get('tool', '?')}] {tool_summary[:500]}")
            except (json.JSONDecodeError, KeyError):
                pass

    # Grace summarization — if the model exhausted rounds on tool calls
    # without producing a final text response, do one last LLM call
    # asking it to summarize what it did. Guarantees output.
    if not full_text.strip():
        try:
            from src.llm_core import llm_call_async_with_fallback
            from src.endpoint_resolver import resolve_utility_fallback_candidates
            grace_context = "You ran out of steps. "
            if tool_results:
                grace_context += "Here's what your tools returned:\n" + "\n".join(tool_results[-5:])
            else:
                grace_context += "No tool results were captured."
            grace_context += "\n\nSummarize what you accomplished and what's still pending. Be concise."
            _grace_candidates = [(endpoint_url, model, headers)] + resolve_utility_fallback_candidates(owner=task.owner or None)
            full_text = await llm_call_async_with_fallback(
                _grace_candidates,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": grace_context},
                ],
                timeout=30,
            )
            full_text = (full_text or "").strip()
        except Exception as e:
            logger.warning(f"Grace summarization failed: {e}")
            if tool_results:
                full_text = "\n".join(tool_results[-5:])

    return full_text or "(no output)"

async def _execute_research_task(self, task, db) -> str:
    """Execute a deep research task using DeepResearcher."""
    from core.database import Session as DbSession, ChatMessage
    from src.deep_research import DeepResearcher
    from src.research_handler import RESEARCH_DATA_DIR, ResearchHandler
    from src.research_utils import strip_thinking
    from src.settings import get_setting

    # Resolve endpoint/model: research settings > task settings > session defaults
    endpoint_url = task.endpoint_url
    model = task.model
    headers = {}
    headers_from_resolver = False

    if not endpoint_url or not model:
        try:
            from src.endpoint_resolver import resolve_endpoint
            ep_url, ep_model, ep_headers = resolve_endpoint(
                "research",
                endpoint_url or None,
                model or None,
                None,
                owner=task.owner or None,
            )
            endpoint_url = ep_url or endpoint_url
            model = ep_model or model
            if ep_headers is not None:
                headers = ep_headers
                headers_from_resolver = True
        except Exception:
            pass

    if not endpoint_url or not model:
        endpoint_url, model = self._resolve_defaults(db, task.owner)
    if not endpoint_url or not model:
        raise RuntimeError("No model/endpoint configured for research")
    # Record the resolved model for the run record (see _execute_task_locked).
    self._last_run_model = model

    # Resolve headers
    try:
        from core.database import ModelEndpoint
        from src.endpoint_resolver import normalize_base, build_headers
        from src.auth_helpers import owner_filter
        db2 = db
        if not headers_from_resolver:
            ep_q = db2.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True)
            ep_q = owner_filter(ep_q, ModelEndpoint, task.owner or None)
            eps = ep_q.all()
            for ep in eps:
                if normalize_base(ep.base_url) in endpoint_url or endpoint_url in normalize_base(ep.base_url):
                    headers = build_headers(ep.api_key, normalize_base(ep.base_url))
                    break
    except Exception:
        pass

    max_tokens = int(get_setting("research_max_tokens", 8192))
    extraction_timeout = int(get_setting("research_extraction_timeout_seconds", 90) or 90)
    extraction_concurrency = int(get_setting("research_extraction_concurrency", 3) or 3)

    researcher = DeepResearcher(
        llm_endpoint=endpoint_url,
        llm_model=model,
        llm_headers=headers,
        max_rounds=8,
        max_time=600,  # 10 min for scheduled research
        max_report_tokens=max_tokens,
        extraction_timeout=extraction_timeout,
        extraction_concurrency=extraction_concurrency,
    )

    started_ts = time.time()
    report = await researcher.research(task.prompt)
    completed_ts = time.time()
    try:
        stats = researcher.get_stats() or {}
    except Exception:
        stats = {}

    # Ensure a session exists for output
    session_id = task.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        sess = DbSession(
            id=session_id,
            name=f"[Research] {task.name}",
            endpoint_url=endpoint_url,
            model=model,
            owner=task.owner,
            folder="Tasks",
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
        db.add(sess)
        task.session_id = session_id
        db.commit()
        if self._session_manager:
            try:
                self._session_manager.sessions[session_id] = self._session_manager._db_to_session(sess)
            except Exception:
                pass

    # Persist scheduled research in the same on-disk shape used by the
    # Research panel. Without this, task research had Markdown output but
    # no Library entry and no visual report route to open.
    try:
        RESEARCH_DATA_DIR.mkdir(parents=True, exist_ok=True)
        findings = getattr(researcher, "findings", []) or []
        payload = {
            "query": task.prompt or task.name or "Scheduled research",
            "status": "done",
            "result": report,
            "raw_report": strip_thinking(report or ""),
            "sources": ResearchHandler._extract_sources(findings),
            "raw_findings": ResearchHandler._extract_raw_findings(findings),
            "stats": stats,
            "category": "scheduled",
            "started_at": started_ts,
            "completed_at": completed_ts,
            "owner": task.owner or "",
            "task_id": task.id,
            "task_name": task.name,
        }
        (RESEARCH_DATA_DIR / f"{session_id}.json").write_text(json.dumps(payload), encoding="utf-8")
        try:
            from src.event_bus import fire_event
            fire_event("research_completed", task.owner or None)
        except Exception:
            logger.debug("research_completed event dispatch failed", exc_info=True)
    except Exception as e:
        logger.warning("Failed to persist task research report %s: %s", session_id, e)

    return report

async def _run_chained(self, task_id: str):
    """Run a chained task. Acquires _executing membership the same way
    run_task_now does so an overlapping scheduler tick can't double-dispatch
    the same task while the chain run is in flight."""
    async with self._executing_lock:
        if task_id in self._executing:
            return  # already in flight (manual trigger, scheduler tick, or another chain)
        self._executing.add(task_id)
    await self._execute_task(task_id)

def _has_chain_cycle(self, db, start_id: str, max_depth: int = 10, owner: str | None = None) -> bool:
    """Detect cycles in task chains."""
    from core.database import ScheduledTask
    visited = set()
    current = start_id
    for _ in range(max_depth):
        if current in visited:
            return True
        visited.add(current)
        task = db.query(ScheduledTask).filter(ScheduledTask.id == current).first()
        if owner is not None and task and task.owner != owner:
            return True
        if not task or not task.then_task_id:
            return False
        current = task.then_task_id
    return True  # too deep, treat as cycle

def _resolve_defaults(self, db, owner):
    """Find the first available endpoint + model from an existing session."""
    from core.database import Session as DbSession
    try:
        recent = db.query(DbSession).filter(
            DbSession.endpoint_url.isnot(None),
            DbSession.model.isnot(None),
            *([DbSession.owner == owner] if owner else []),
        ).order_by(DbSession.created_at.desc()).first()
        if recent:
            return recent.endpoint_url, recent.model
    except Exception:
        pass
    return None, None

async def _deliver_via_mcp(self, tool_name: str, task, result: str):
    """Send the task result via an MCP tool (e.g. Gmail send).

    Resolves a recipient (so email-style tools have a 'to') by trying the
    configured From address first (the `daily_brief` pattern — email
    yourself) then falling back to the task owner. Common recipient field
    names (to / recipient / email / address) are all populated so we don't
    have to special-case each tool's schema; the MCP tool ignores keys it
    doesn't recognise.
    """
    from src.tool_utils import get_mcp_manager
    mcp = get_mcp_manager()
    if not mcp:
        logger.warning(f"Task {task.id}: MCP manager not available for delivery")
        return

    # Resolve recipient — prefer the configured email From (the established
    # "email yourself" pattern from daily_brief), fall back to task.owner.
    # `_get_email_config()` is the single source of truth that handles both
    # the legacy `email_from` setting and the per-account DB rows.
    recipient = None
    try:
        from routes.email_helpers import _get_email_config
        cfg = _get_email_config() or {}
        recipient = cfg.get("from_address") or None
    except Exception as _e:
        logger.debug(f"_deliver_via_mcp: email config lookup failed: {_e}")
    if not recipient and task.owner and "@" in str(task.owner):
        recipient = task.owner

    args = {
        "subject": f"[Task] {task.name}",
        "body": result,
        "headers": {
            "X-Odysseus-Origin": "odysseus-ui",
            "X-Odysseus-Kind": "task",
            "X-Odysseus-Ref": str(task.id),
        },
    }
    if recipient:
        # Cover the common field names so we work across MCP servers (Gmail,
        # generic SMTP, Slack DMs, etc.) without having to hard-code each.
        args["to"] = recipient
        args["recipient"] = recipient
        args["email"] = recipient
        args["address"] = recipient
    else:
        logger.warning(
            f"Task {task.id}: no recipient resolved for MCP delivery via {tool_name} — "
            "set an email From address in Settings or give the task an owner email."
        )
    try:
        mcp_result = await mcp.call_tool(tool_name, args)
        stderr = mcp_result.get("stderr", "")
        stdout = mcp_result.get("stdout", "")
        body_len = len(result or "")
        exit_code = mcp_result.get("exit_code", 0)
        if exit_code != 0:
            logger.warning(
                f"Task {task.id} MCP delivery FAILED via {tool_name}: "
                f"exit={exit_code} stderr={stderr[:400]!r} stdout={stdout[:400]!r}"
            )
        else:
            # Include the MCP tool's own stdout (e.g. email_server returns
            # "Sent email to ... with subject ...") + the body size so a
            # silent SMTP failure is easier to spot in the logs.
            logger.info(
                f"Task {task.id} delivered via MCP tool {tool_name} "
                f"(to={recipient or '<unset>'}, body={body_len}b, reply={stdout[:200]!r})"
            )
    except Exception as e:
        logger.error(f"Task {task.id} MCP delivery failed: {e}")

