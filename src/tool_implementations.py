"""
tool_implementations.py

Extracted tool implementation functions (do_* and helpers) from agent_tools.py.
These handle the actual execution logic for each tool type.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from src.constants import MAX_READ_CHARS, DEEP_RESEARCH_DIR, VAULT_FILE
from src.tool_utils import get_mcp_manager
from core.constants import internal_api_base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_tool_args(content):
    """Parse a tool-call argument blob.

    Accepts either a JSON string or an already-decoded dict. Unwraps the
    common `{"body": {...}}` envelope that smaller models emit when they
    read tool descriptions like "Body is JSON: {...}" literally — they
    pass `body` as a field name rather than treating it as a noun.

    Returns a dict on success, raises ValueError on bad JSON.
    """
    if isinstance(content, str):
        try:
            args = json.loads(content) if content.strip() else {}
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(str(e))
    elif isinstance(content, dict):
        args = content
    else:
        args = {}
    # Unwrap {"body": {...}} envelope — but only if `body` is the sole key
    # and points at a dict. We don't want to clobber a legitimate `body`
    # field on tools where it's a real arg (e.g. send_email body text).
    if (
        isinstance(args, dict)
        and len(args) == 1
        and "body" in args
        and isinstance(args["body"], dict)
        and "action" in args["body"]  # extra safety: only unwrap if the inner dict looks like a tool call
    ):
        args = args["body"]
    return args

# ---------------------------------------------------------------------------
# Search chats
# ---------------------------------------------------------------------------

async def do_manage_calendar(content: str, owner: Optional[str] = None) -> Dict:
    """Handle manage_calendar tool calls: list/create/update/delete calendar events (local SQLite)."""
    from datetime import datetime, timedelta
    from core.database import SessionLocal, CalendarCal, CalendarEvent, Note
    from routes.calendar_routes import _ensure_default_calendar, _parse_dt, _parse_dt_pair, parse_due_for_user, _resolve_base_uid
    import uuid as _uuid

    try:
        args = _parse_tool_args(content)
    except ValueError:
        return {"error": "Invalid JSON arguments", "exit_code": 1}

    # ── Batch normalization ──
    # Some models (e.g. deepseek-v4-flash) emit {"events": [{...}, ...]}
    # instead of individual create_event calls. Iterate and create each.
    if isinstance(args.get("events"), list) and not args.get("action"):
        results = []
        for ev in args["events"]:
            if not isinstance(ev, dict):
                continue
            # Normalize start/end from {dateTime: "..."} object to flat string
            for field, target in [("start", "dtstart"), ("end", "dtend")]:
                val = ev.pop(field, None)
                if val and target not in ev:
                    ev[target] = val.get("dateTime", val) if isinstance(val, dict) else val
            ev.setdefault("action", "create_event")
            r = await do_manage_calendar(json.dumps(ev), owner=owner)
            results.append(r)
        created = [r for r in results if r.get("exit_code") == 0 and not r.get("error")]
        failed = [r for r in results if r.get("error")]

        if not results:
            return {"error": "No events to create", "exit_code": 1}

        # Surface both successes and failures
        parts = []
        if created:
            summaries = [r.get("response", "") for r in created]
            parts.append(f"Created {len(created)} event(s):\n" + "\n".join(summaries))
        if failed:
            first_error = failed[0].get("error", "Unknown error")
            parts.append(f"Failed to create {len(failed)} event(s). First error: {first_error}")

        response = "\n\n".join(parts)
        # Non-zero exit code for partial or total failure
        exit_code = 0 if not failed else 1
        return {"response": response, "exit_code": exit_code, "created_count": len(created), "failed_count": len(failed)}

    # Normalize action — some models emit hyphens ("list-calendars") instead
    # of underscores. Treat them as equivalent so we don't bounce a
    # cosmetic typo back to the model and waste a round-trip. Also accept
    # short forms (`create`, `update`, `delete`) as aliases for the
    # full `<verb>_event` names — models keep emitting the short forms.
    action = (args.get("action") or "list_events").replace("-", "_").strip().lower()
    _ACTION_ALIASES = {
        "create": "create_event",
        "update": "update_event",
        "delete": "delete_event",
        "list": "list_events",
    }
    action = _ACTION_ALIASES.get(action, action)
    db = SessionLocal()

    def _calendar_query():
        q = db.query(CalendarCal)
        if owner is not None:
            q = q.filter(CalendarCal.owner == owner)
        return q

    def _event_query():
        q = db.query(CalendarEvent).join(CalendarCal)
        if owner is not None:
            q = q.filter(CalendarCal.owner == owner)
        return q

    def _reminder_minutes(raw_args) -> Optional[int]:
        raw = (
            raw_args.get("reminder_minutes")
            or raw_args.get("remind_before_minutes")
            or raw_args.get("alarm_minutes")
            or raw_args.get("reminder")
            or raw_args.get("alarm")
        )
        if raw in (None, ""):
            desc = str(raw_args.get("description") or "")
            if re.search(r"\b(remind|reminder|alarm)\b", desc, re.I):
                raw = desc
        if raw in (None, "", False):
            return None
        if raw is True:
            return 10
        if isinstance(raw, (int, float)):
            return max(0, int(raw))
        text = str(raw).strip().lower()
        if text in {"none", "no", "off", "false"}:
            return None
        m = re.search(r"(\d+)\s*(?:m|min|minute|minutes)\b", text)
        if m:
            return max(0, int(m.group(1)))
        m = re.search(r"(\d+)\s*(?:h|hr|hour|hours)\b", text)
        if m:
            return max(0, int(m.group(1)) * 60)
        if text.isdigit():
            return max(0, int(text))
        return None

    def _event_description(raw_args, minutes_before: Optional[int]) -> str:
        desc = str(raw_args.get("description", "") or "")
        if minutes_before is None:
            return desc
        reminder_only = re.compile(
            r"^\s*(?:remind(?:er)?|alarm)\s*:?\s*\d+\s*"
            r"(?:m|min|minute|minutes|h|hr|hour|hours)\b.*$",
            re.I,
        )
        return "" if reminder_only.match(desc) else desc

    def _parse_event_dt(raw: str) -> tuple[datetime, bool]:
        """Parse agent event datetimes in the user's timezone when available."""
        return _parse_dt_pair(parse_due_for_user(raw))

    def _first_nonempty_arg(*names: str):
        for name in names:
            value = args.get(name)
            if value not in (None, ""):
                return value
        return None

    def _create_calendar_reminder(summary: str, location: str, dtstart: datetime,
                                  all_day: bool, minutes_before: int,
                                  is_utc: bool = False) -> tuple[Optional[str], Optional[str]]:
        remind_at = dtstart - timedelta(minutes=minutes_before)
        now = datetime.utcnow() if is_utc else datetime.now()
        if dtstart <= now:
            return None, "event already passed"
        if remind_at <= now:
            # If the requested "before" time already passed but the event is
            # still upcoming, create an immediate Note reminder instead of
            # silently dropping it.
            remind_at = now
        start_fmt = dtstart.strftime("%a %b %d") if all_day else dtstart.strftime("%a %b %d %H:%M")
        loc = f" @ {location}" if location else ""
        text = f"{summary}{loc} — {start_fmt}"
        due_date = remind_at.isoformat() + ("Z" if is_utc else "")
        expected_title = f"Reminder: {summary}"
        existing_q = db.query(Note).filter(
            Note.archived == False,  # noqa: E712
            Note.due_date == due_date,
        )
        if owner is not None:
            existing_q = existing_q.filter(Note.owner == owner)
        target_title = re.sub(r"^\s*reminder\s*:\s*", "", expected_title.strip().lower())
        for existing in existing_q.limit(25).all():
            existing_title = re.sub(r"^\s*reminder\s*:\s*", "", (existing.title or "").strip().lower())
            if existing_title == target_title:
                return existing.id, "duplicate reminder already exists"
        note = Note(
            id=str(_uuid.uuid4()),
            owner=owner,
            title=expected_title,
            items=json.dumps([{"text": text, "done": False, "checked": False}]),
            note_type="todo",
            label="calendar",
            due_date=due_date,
            source="calendar",
        )
        db.add(note)
        return note.id, None

    try:
        if action == "list_calendars":
            _ensure_default_calendar(db, owner)
            cals = _calendar_query().all()
            result = [{"name": c.name, "href": c.id} for c in cals]
            if result:
                lines = [f"Found {len(result)} calendar(s):"]
                for c in result:
                    lines.append(f"- {c['name']} ({c['href'][:8]})")
                response_text = "\n".join(lines)
            else:
                response_text = "No calendars found."
            return {"response": response_text, "calendars": result, "exit_code": 0}

        elif action == "list_events":
            try:
                start_raw = _first_nonempty_arg(
                    "start", "start_date", "range_start", "from", "dtstart", "since"
                )
                end_raw = _first_nonempty_arg(
                    "end", "end_date", "range_end", "to", "dtend", "until"
                )
                if start_raw:
                    start_dt = _parse_dt(start_raw)
                else:
                    start_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                if end_raw:
                    end_dt = _parse_dt(end_raw)
                else:
                    end_dt = start_dt + timedelta(days=14)
            except ValueError as e:
                return {"error": f"Invalid date format: {e}", "exit_code": 1}

            q = _event_query().filter(
                CalendarEvent.dtstart < end_dt,
                CalendarEvent.dtend > start_dt,
                CalendarEvent.status != "cancelled",
            )
            calendar_filter = args.get("calendar")
            if calendar_filter:
                q = q.filter(
                    (CalendarEvent.calendar_id == calendar_filter) |
                    (CalendarCal.name == calendar_filter)
                )
            rows = q.order_by(CalendarEvent.dtstart).all()
            events = []
            for ev in rows:
                if ev.all_day:
                    s, e = ev.dtstart.strftime("%Y-%m-%d"), ev.dtend.strftime("%Y-%m-%d")
                else:
                    suffix = "Z" if getattr(ev, "is_utc", False) else ""
                    s, e = ev.dtstart.isoformat() + suffix, ev.dtend.isoformat() + suffix
                events.append({
                    "uid": ev.uid, "summary": ev.summary or "", "dtstart": s, "dtend": e,
                    "all_day": ev.all_day, "description": ev.description or "",
                    "location": ev.location or "",
                    "calendar": ev.calendar.name if ev.calendar else "",
                    "calendar_href": ev.calendar_id,
                    "event_type": ev.event_type or "",
                    "importance": ev.importance or "normal",
                })
            if not events:
                response_text = f"No events between {start_dt.date().isoformat()} and {end_dt.date().isoformat()}."
            else:
                lines = [f"Found {len(events)} event(s) between {start_dt.date().isoformat()} and {end_dt.date().isoformat()}:"]
                for ev in events:
                    when = ev["dtstart"]
                    when_str = f"{when} (all day)" if ev.get("all_day") else f"{when} -> {ev.get('dtend', '')}"
                    # Clickable anchor — opens the calendar on the event's day.
                    line = f"- {when_str}: [{ev['summary']}](#event-{ev['uid']})"
                    if ev.get("event_type"):
                        line += f" #{ev['event_type']}"
                    if ev.get("importance") and ev["importance"] != "normal":
                        line += f" !{ev['importance']}"
                    if ev.get("location"):
                        line += f" @ {ev['location']}"
                    if ev.get("calendar"):
                        line += f" ({ev['calendar']})"
                    if ev.get("description"):
                        desc = ev["description"].strip().replace("\n", " ")
                        if len(desc) > 120:
                            desc = desc[:117] + "..."
                        line += f"\n    {desc}"
                    lines.append(line)
                response_text = "\n".join(lines)
            return {"response": response_text, "events": events, "exit_code": 0}

        elif action == "create_event":
            summary = args.get("summary")
            # Accept the various names models like to use for the start
            # field: dtstart (canonical), start, start_time, when.
            dtstart_str = (args.get("dtstart") or args.get("start")
                           or args.get("start_time") or args.get("when"))
            if not summary or not dtstart_str:
                return {"error": "summary and dtstart are required", "exit_code": 1}

            # Accept either an href OR a calendar name/short-id like "Main"
            # or "62e545d8" — saves the model from having to memorize hrefs
            # after a `list_calendars` call returned short prefixes.
            cal_href = args.get("calendar_href") or args.get("calendar")
            cal = None
            if cal_href:
                cal = (_calendar_query()
                       .filter(CalendarCal.id == cal_href)
                       .first())
                if not cal:
                    # Try by name (case-insensitive) or by short-id prefix
                    cal = (_calendar_query()
                           .filter(CalendarCal.name.ilike(cal_href))
                           .first())
                if not cal:
                    cal = (_calendar_query()
                           .filter(CalendarCal.id.like(f"{cal_href}%"))
                           .first())
            if not cal:
                cal = _ensure_default_calendar(db, owner)

            all_day = bool(args.get("all_day", False))
            try:
                dtstart, dtstart_is_utc = _parse_event_dt(dtstart_str)
            except ValueError as e:
                return {"error": f"Could not parse dtstart {dtstart_str!r}: {e}", "exit_code": 1}
            dtend_raw = args.get("dtend") or args.get("end") or args.get("end_time")
            if dtend_raw:
                try:
                    dtend, dtend_is_utc = _parse_event_dt(dtend_raw)
                    dtstart_is_utc = dtstart_is_utc or dtend_is_utc
                except ValueError as e:
                    return {"error": f"Could not parse dtend {dtend_raw!r}: {e}", "exit_code": 1}
            else:
                # Support duration: "1h", "30m", "90min", "1hr30m"
                dur = (args.get("duration") or "").strip().lower()
                delta = None
                if dur:
                    import re as _re_d
                    h = _re_d.search(r'(\d+)\s*(?:h|hr|hours?)', dur)
                    m = _re_d.search(r'(\d+)\s*(?:m|min|minutes?)', dur)
                    secs = (int(h.group(1)) * 3600 if h else 0) + (int(m.group(1)) * 60 if m else 0)
                    if secs > 0:
                        delta = timedelta(seconds=secs)
                if delta is not None:
                    dtend = dtstart + delta
                elif all_day:
                    dtend = dtstart + timedelta(days=1)
                else:
                    dtend = dtstart + timedelta(hours=1)

            # Dedup: if a non-cancelled event with the same title + start time already
            # exists, return its UID instead of creating a fresh copy. Prevents the
            # email triage from multiplying events when several emails reference the
            # same meeting. Compare case-insensitively since LLM-extracted titles
            # can vary in capitalisation.
            from sqlalchemy import func as _func
            existing = (
                _event_query()
                .filter(
                    CalendarEvent.dtstart == dtstart,
                    CalendarEvent.status != "cancelled",
                    _func.lower(CalendarEvent.summary) == summary.lower(),
                )
                .first()
            )
            if existing is not None:
                reminder_note_id = None
                reminder_skipped_reason = None
                minutes_before = _reminder_minutes(args)
                if minutes_before is not None:
                    reminder_note_id, reminder_skipped_reason = _create_calendar_reminder(
                        existing.summary or summary,
                        existing.location or "",
                        existing.dtstart,
                        existing.all_day,
                        minutes_before,
                        bool(existing.is_utc),
                    )
                    if reminder_note_id:
                        db.commit()
                reminder_text = ""
                if minutes_before is not None:
                    reminder_text = (
                        f"; reminder set {minutes_before} min before"
                        if reminder_note_id
                        else f"; reminder not set ({reminder_skipped_reason or 'reminder time already passed'})"
                    )
                return {
                    "response": (
                        f"Event already exists: '{summary}' on {dtstart_str}"
                        + reminder_text
                    ),
                    "uid": existing.uid,
                    "reminder_note_id": reminder_note_id,
                    "reminder_skipped_reason": reminder_skipped_reason,
                    "duplicate": True,
                    "exit_code": 0,
                }

            # Optional tag/category and importance — friendly aliases.
            event_type = (args.get("event_type") or args.get("tag")
                          or args.get("category") or args.get("type") or "") or None
            importance = args.get("importance") or "normal"
            minutes_before = _reminder_minutes(args)

            uid = str(_uuid.uuid4())
            ev = CalendarEvent(
                uid=uid, calendar_id=cal.id, summary=summary,
                description=_event_description(args, minutes_before),
                location=args.get("location", "") or "",
                dtstart=dtstart, dtend=dtend, all_day=all_day,
                is_utc=dtstart_is_utc and not all_day,
                rrule=args.get("rrule", "") or "",
                event_type=event_type,
                importance=importance,
            )
            db.add(ev)
            reminder_note_id = None
            reminder_skipped_reason = None
            if minutes_before is not None:
                reminder_note_id, reminder_skipped_reason = _create_calendar_reminder(
                    summary,
                    args.get("location", "") or "",
                    dtstart,
                    all_day,
                    minutes_before,
                    dtstart_is_utc and not all_day,
                )
            db.commit()
            tag_blurb = f" [{event_type}]" if event_type else ""
            if minutes_before is None:
                reminder_blurb = ""
            elif reminder_note_id:
                reminder_blurb = f" with reminder {minutes_before} min before"
            else:
                reminder_blurb = f" without reminder ({reminder_skipped_reason or 'reminder time already passed'})"
            # Return a clickable anchor so the agent can surface a link
            # that opens the calendar on that day. See the markdown
            # anchor convention ([Name](#event-<uid>)).
            return {
                "response": f"Created event [{summary}](#event-{uid}){tag_blurb} on {dtstart_str}{reminder_blurb}",
                "uid": uid,
                "anchor": f"[{summary}](#event-{uid})",
                "reminder_note_id": reminder_note_id,
                "reminder_skipped_reason": reminder_skipped_reason,
                "exit_code": 0,
            }

        elif action == "update_event":
            uid = args.get("uid")
            if not uid:
                return {"error": "uid is required", "exit_code": 1}
            try:
                base_uid = _resolve_base_uid(uid)
            except ValueError as e:
                return {"error": str(e), "exit_code": 1}
            ev = _event_query().filter(CalendarEvent.uid == base_uid).first()
            if not ev:
                return {"error": f"Event {uid} not found", "exit_code": 1}
            if args.get("summary") is not None:
                ev.summary = args["summary"]
            if args.get("description") is not None:
                ev.description = args["description"]
            if args.get("location") is not None:
                ev.location = args["location"]
            if args.get("dtstart") is not None:
                # Anchor naive/natural-language input to the USER's timezone and
                # refresh is_utc, exactly like create_event. Parsing with the
                # raw server-local _parse_dt here (and never touching is_utc)
                # silently shifted an updated event by the user's UTC offset.
                _eff_all_day = (
                    args["all_day"] if args.get("all_day") is not None else ev.all_day
                )
                ev.dtstart, _su = _parse_event_dt(args["dtstart"])
                ev.is_utc = bool(_su and not _eff_all_day)
            if args.get("dtend") is not None:
                ev.dtend, _eu = _parse_event_dt(args["dtend"])
            if args.get("all_day") is not None:
                ev.all_day = args["all_day"]
            # Tag/category + importance updates (any of these aliases).
            _tag = (args.get("event_type") or args.get("tag")
                    or args.get("category") or args.get("type"))
            if _tag is not None:
                ev.event_type = _tag or None
            if args.get("importance") is not None:
                ev.importance = args["importance"]
            db.commit()
            return {"response": f"Updated event {uid}", "exit_code": 0}

        elif action == "delete_event":
            uid = args.get("uid")
            if not uid:
                return {"error": "uid is required", "exit_code": 1}
            try:
                base_uid = _resolve_base_uid(uid)
            except ValueError as e:
                return {"error": str(e), "exit_code": 1}
            ev = _event_query().filter(CalendarEvent.uid == base_uid).first()
            if not ev:
                return {"error": f"Event {uid} not found", "exit_code": 1}
            db.delete(ev)
            db.commit()
            return {"response": f"Deleted event {uid}", "exit_code": 0}

        else:
            return {
                "error": f"Unknown action: {action}. Use list_events, create_event, update_event, delete_event, list_calendars",
                "exit_code": 1,
            }

    except Exception as e:
        db.rollback()
        logger.error(f"manage_calendar error: {e}")
        return {"error": str(e), "exit_code": 1}
    finally:
        db.close()


# ── Cookbook tools ──

# In-process loopback base for agent tools that call Odysseus's own API
# (cookbook state, model serve, gallery, email, calendar). We ride the
# per-process internal token so require_admin lets us through. See
# core/middleware.py. Resolution (override / APP_PORT / 7000) lives in
# core.constants.internal_api_base().
_INTERNAL_BASE = internal_api_base()


def _internal_headers(owner: Optional[str] = None) -> Dict[str, str]:
    from core.middleware import INTERNAL_TOOL_HEADER, INTERNAL_TOOL_TOKEN
    headers = {INTERNAL_TOOL_HEADER: INTERNAL_TOOL_TOKEN}
    if owner:
        headers["X-Odysseus-Owner"] = owner
    return headers


async def _cookbook_servers() -> Dict[str, Any]:
    """Return the cookbook's configured servers + the currently-selected
    default host. Shape: {default_host, hosts: [{host, platform, env, envPath}]}.
    The agent uses this to route downloads/serves to the right machine
    instead of silently defaulting to localhost."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{_INTERNAL_BASE}/api/cookbook/state", headers=_internal_headers())
            state = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    except Exception:
        return {"default_host": "", "hosts": []}
    env = (state or {}).get("env") or {}
    if not isinstance(env, dict):
        return {"default_host": "", "hosts": []}
    hosts = []
    for s in (env.get("servers") or []):
        if isinstance(s, dict):
            hosts.append({
                "name": s.get("name") or "",
                "host": s.get("host") or "",   # "" = Local
                "platform": s.get("platform") or "",
                "env": s.get("env") or "",
                "envPath": s.get("envPath") or "",
                "port": s.get("port") or "",
            })
    return {"default_host": env.get("remoteHost") or "", "hosts": hosts}


async def _resolve_cookbook_host(name_or_host: str) -> str:
    """Map a friendly server NAME ('gpu-box', 'workstation') to its ssh host
    string ('user@192.0.2.10'). If the input already looks like an
    ssh host (contains '@' or matches a known host), or matches nothing,
    it's returned unchanged. 'local'/'localhost' → '' (this machine)."""
    if not name_or_host:
        return ""
    val = name_or_host.strip()
    low = val.lower()
    if low in ("local", "localhost", "this machine", "here"):
        return ""
    servers = await _cookbook_servers()
    # Exact host match → already an ssh host
    for h in servers.get("hosts") or []:
        if h.get("host") and h["host"] == val:
            return val
    # Name match (case-insensitive)
    for h in servers.get("hosts") or []:
        if (h.get("name") or "").lower() == low:
            return h.get("host") or ""   # "" for the Local entry
    # Substring name match as a fallback
    for h in servers.get("hosts") or []:
        if low and low in (h.get("name") or "").lower():
            return h.get("host") or ""
    # No match — assume the caller passed a raw host/alias; return as-is
    # (ssh can resolve aliases from ~/.ssh/config).
    return val


async def _cookbook_env_for_host(host: str) -> Dict[str, Any]:
    """Resolve env_prefix / gpus / platform / hf_token / ssh_port for a
    given host by looking it up in cookbook_state.env. The user
    configures these per-host in the Cookbook UI; without them, raw
    `vllm serve …` fails with 'command not found' because vLLM lives
    inside a venv that has to be sourced first.

    Returns a dict with keys ready to drop into the /api/model/serve
    payload: env_prefix, gpus, platform, hf_token, ssh_port.
    Falls back to the top-level env settings if no per-host entry exists.
    """
    import httpx
    headers = _internal_headers()
    state: Dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{_INTERNAL_BASE}/api/cookbook/state", headers=headers)
            state = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    except Exception as e:
        logger.debug(f"cookbook env lookup failed for host={host!r}: {e}")
        return {}
    if not isinstance(state, dict):
        return {}
    env_root = state.get("env") or {}
    if not isinstance(env_root, dict):
        return {}

    # Per-host entry takes precedence over top-level.
    per_host: Dict[str, Any] = {}
    for s in (env_root.get("servers") or []):
        if isinstance(s, dict) and (s.get("host") or "") == (host or ""):
            per_host = s
            break

    env_kind = per_host.get("env") or env_root.get("env") or "none"
    env_path = per_host.get("envPath") or env_root.get("envPath") or ""
    platform = per_host.get("platform") or env_root.get("platform") or "linux"
    ssh_port = per_host.get("sshPort") or env_root.get("sshPort") or ""

    env_prefix = ""
    if env_kind == "venv" and env_path:
        if platform == "windows":
            activate = env_path if env_path.endswith("\\Scripts\\Activate.ps1") else env_path.rstrip("\\") + "\\Scripts\\Activate.ps1"
            env_prefix = f"& {activate}"
        else:
            activate = env_path if env_path.endswith("/bin/activate") else env_path.rstrip("/") + "/bin/activate"
            env_prefix = f"source {activate}"
    elif env_kind == "conda" and env_path:
        if platform == "windows":
            env_prefix = f"conda activate {env_path}"
        else:
            env_prefix = f'eval "$(conda shell.bash hook)" && conda activate {env_path}'

    from routes.cookbook_helpers import load_stored_hf_token
    return {
        "env_prefix": env_prefix,
        "env_type": env_kind,
        "env_path": env_path,
        "gpus": env_root.get("gpus") or "",
        "platform": platform,
        "hf_token": load_stored_hf_token(),
        "ssh_port": ssh_port,
    }


def _infer_serve_port(cmd: str) -> int:
    """Infer likely listen port from a serve command."""
    if not cmd:
        return 8080
    m = re.search(r"--port\\s+(\\d+)", cmd)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    m = re.search(r"OLLAMA_HOST=[^\\s]*?:(\\d+)", cmd)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    if "ollama" in cmd:
        return 11434
    return 8080


def _infer_serve_host(host: str | None) -> tuple[str, bool]:
    """Return (host, container_local) for registering a served endpoint."""
    if not (host or "").strip():
        return "localhost", True
    base_host = host.split("@", 1)[-1] if "@" in host else host
    return base_host, False


async def _ensure_served_endpoint(
    *,
    model: str,
    cmd: str,
    host: str | None,
) -> Dict[str, Any]:
    """Register/fetch a model endpoint for a running serve session."""
    import httpx
    endpoint_host, container_local = _infer_serve_host(host)
    port = _infer_serve_port(cmd)
    base_url = f"http://{endpoint_host}:{port}/v1"
    short_name = model.split("/")[-1] if "/" in model else model
    is_image = "diffusion_server.py" in (cmd or "")
    payload = {
        "name": short_name if not is_image else f"{short_name} (image)",
        "base_url": base_url,
        "skip_probe": "true",
        "model_type": "image" if is_image else "llm",
        "container_local": "true" if container_local else "false",
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{_INTERNAL_BASE}/api/model-endpoints",
                data=payload,
                headers=_internal_headers(),
            )
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        if resp.status_code >= 400:
            logger.debug(
                f"ensure endpoint failed for {model!r}: status={resp.status_code} data={data}"
            )
            return {"added": False, "endpoint_id": "", "base_url": base_url, "error": data}
        ep_id = data.get("id") if isinstance(data, dict) else None
        return {
            "added": bool(ep_id),
            "endpoint_id": ep_id or "",
            "base_url": base_url,
            "data": data,
        }
    except Exception as e:
        logger.debug(f"ensure endpoint exception for {model!r}: {e}")
        return {"added": False, "endpoint_id": "", "base_url": base_url, "error": str(e)}


async def _cookbook_register_task(
    session_id: str,
    model: str,
    host: str,
    cmd: str,
    task_type: str = "serve",
    *,
    endpoint_added: bool = False,
    endpoint_id: str = "",
) -> bool:
    """Append a task entry to cookbook_state.json after the agent
    launches via /api/model/serve or /api/model/download. The route
    spawns tmux but leaves state-writing to the UI; the agent needs to
    do that here so the task shows up in the Cookbook tab.
    Returns True on success, False if the write failed (best-effort)."""
    import httpx
    import time as _time
    headers = _internal_headers()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{_INTERNAL_BASE}/api/cookbook/state", headers=headers)
            state = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    except Exception as e:
        logger.debug(f"cookbook state read failed: {e}")
        return False
    if not isinstance(state, dict):
        state = {}
    tasks = state.get("tasks") if isinstance(state.get("tasks"), list) else []
    # Skip duplicate (same session_id) entries
    if any(isinstance(t, dict) and t.get("sessionId") == session_id for t in tasks):
        return True
    display_name = model.split("/")[-1] if "/" in model else model
    # Placeholder output — the cookbook UI's CSS hides empty <pre>
    # via `.cookbook-output-pre:empty { display: none }`, so an
    # empty-string output makes the expansion appear broken until the
    # frontend's reconnect-polling loop captures tmux output. A short
    # placeholder gives the user something to see immediately; it gets
    # replaced by real tmux output within a few seconds.
    target = f"{host}:" if host else "local:"
    placeholder = (
        f"Launched via agent — waiting for tmux output…\n"
        f"  session: {session_id}\n"
        f"  target:  {target}{(cmd.split() or [''])[0] if cmd else ''}\n"
        f"  cmd:     {cmd[:200]}{'…' if len(cmd) > 200 else ''}"
    )
    tasks.append({
        "id": session_id,
        "sessionId": session_id,
        "name": display_name,
        "modelId": model,
        "type": task_type,
        "status": "running",
        "output": placeholder,
        "ts": int(_time.time() * 1000),
        "payload": {"repo_id": model, "remote_host": host or "", "_cmd": cmd},
        "remoteHost": host or "",
        "sshPort": "",
        "platform": "linux",
        "_serveReady": False,
        "_endpointAdded": bool(endpoint_added),
        "_endpointId": endpoint_id or "",
    })
    state["tasks"] = tasks
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(f"{_INTERNAL_BASE}/api/cookbook/state",
                                  json=state, headers=headers)
        return r.status_code < 400
    except Exception as e:
        logger.debug(f"cookbook state write failed: {e}")
        return False


# Paths the generic `app_api` tool will refuse to call. Auth/token/user
# administration and host shell execution are too risky to route through an
# agent surface even when the agent is admin-context; accidental account or
# command mistakes have permanent blast radius.
_APP_API_BLOCKLIST_PREFIXES = (
    "/api/auth",           # login/logout/password
    "/api/users",          # user CRUD (bare /api/users list+create+delete must also block)
    "/api/tokens",         # api token mgmt (bare /api/tokens list+create must also block)
    "/api/admin",          # admin one-shots (wipe etc.)
    "/api/shell",          # host shell execution must stay behind named command tooling
    "/api/backup/restore", # destructive restore
)

# (method, prefix) pairs to refuse specifically. Used for endpoints
# where GET is fine but writes are destructive or host-control shaped.
# Saw the agent wipe cookbook_state.json (presets + tasks) by POSTing
# {"tasks": []} to /api/cookbook/state, which overwrote the whole file.
# Use dedicated tools or UI flows instead.
_APP_API_BLOCKLIST_METHOD_PATH = (
    ("GET",    "/api/email/accounts"),  # owner-filtered in tool context; use list_email_accounts MCP tool
    ("POST",   "/api/cookbook/state"),   # whole-file overwrite — agent must use serve_preset/serve_model instead
    ("DELETE", "/api/cookbook/state"),
    # Host-control routes: package install, engine rebuild, and process
    # signalling should not be reachable through the generic API bridge.
    ("POST",   "/api/cookbook/packages/install"),
    ("POST",   "/api/cookbook/rebuild-engine"),
    ("POST",   "/api/cookbook/kill-pid"),
    # Use the named tools (download_model / serve_model) — they handle
    # host-name resolution, per-host env_prefix, AND register the task
    # in cookbook state so it shows in the UI + list_downloads. Hitting
    # the raw endpoint via app_api skips all of that → orphan task.
    ("POST",   "/api/model/download"),
    ("POST",   "/api/model/serve"),
    # Use trigger_research — it returns a UI hint so the Deep Research
    # sidebar surfaces the session. Raw start works but the agent
    # fumbles the payload + the session doesn't reliably show up.
    ("POST",   "/api/research/start"),
    # Use the named tools — they handle owner attribution, natural-
    # language due_date parsing, timezone, dedup, and tag/category
    # normalization. Hitting the raw endpoint via app_api saves a
    # note/event with the wrong fields, no reminder, or the wrong tz.
    ("POST",   "/api/notes"),
    ("PUT",    "/api/notes"),
    ("DELETE", "/api/notes"),
    ("POST",   "/api/calendar/events"),
    ("PUT",    "/api/calendar/events"),
    ("DELETE", "/api/calendar/events"),
)


async def do_app_api(content: str, owner: Optional[str] = None) -> Dict:
    """Generic loopback to allowed internal Odysseus API endpoints. Lets the
    agent reach the full UI-button surface (cookbook, email, notes,
    calendar, skills, sessions, gallery, research, etc.) without us
    landing a named tool wrapper for every one.

    Args (JSON):
      action: "call" (default) | "endpoints"
      path:   "/api/cookbook/gpus"     # required for call
      method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE" (default GET)
      body:   <object>                 # JSON body for POST/PUT/PATCH
      query:  <object>                 # querystring params

    The `endpoints` action returns the OpenAPI surface (method + path +
    summary) so the agent can discover what's reachable. A blocklist
    refuses sensitive auth/user/admin/shell paths and method-specific
    host-control routes to keep blast radius bounded.
    """
    import httpx
    try:
        args = _parse_tool_args(content) if content.strip() else {}
    except ValueError:
        return {"error": "Invalid JSON arguments", "exit_code": 1}

    action = (args.get("action") or "call").lower()
    base = _INTERNAL_BASE

    if action == "endpoints":
        # Fetch FastAPI's OpenAPI schema so the agent can discover any
        # endpoint without us pre-listing them. Filter by an optional
        # `filter` keyword (substring match on path or summary).
        kw = (args.get("filter") or "").lower()
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{base}/openapi.json",
                                        headers=_internal_headers())
                data = resp.json()
        except Exception as e:
            return {"error": f"OpenAPI fetch failed: {e}", "exit_code": 1}
        rows: List[Dict[str, Any]] = []
        for path, methods in (data.get("paths") or {}).items():
            if not isinstance(methods, dict):
                continue
            if any(path.startswith(p) for p in _APP_API_BLOCKLIST_PREFIXES):
                continue
            for method, op in methods.items():
                if method.lower() not in ("get", "post", "put", "patch", "delete"):
                    continue
                if any(method.upper() == m and path.startswith(p) for m, p in _APP_API_BLOCKLIST_METHOD_PATH):
                    continue
                summary = (op or {}).get("summary") or (op or {}).get("description") or ""
                if isinstance(summary, str):
                    summary = summary.strip().split("\n")[0][:140]
                if kw and kw not in path.lower() and kw not in (summary or "").lower():
                    continue
                rows.append({"method": method.upper(), "path": path, "summary": summary})
        rows.sort(key=lambda r: (r["path"], r["method"]))
        if not rows:
            return {"output": f"No endpoints match filter {kw!r}." if kw else "No endpoints found.", "exit_code": 0}
        lines = [f"{len(rows)} endpoint(s)" + (f" matching {kw!r}" if kw else "") + ":"]
        for r in rows[:200]:
            line = f"  {r['method']:6s} {r['path']}"
            if r["summary"]:
                line += f"  — {r['summary']}"
            lines.append(line)
        if len(rows) > 200:
            lines.append(f"  ...({len(rows) - 200} more — filter to narrow)")
        return {"output": "\n".join(lines), "endpoints": rows, "exit_code": 0}

    # action == "call"
    path = args.get("path") or ""
    if not path:
        return {"error": "path is required (e.g. '/api/cookbook/gpus')", "exit_code": 1}
    if not path.startswith("/"):
        path = "/" + path
    if any(path.startswith(p) for p in _APP_API_BLOCKLIST_PREFIXES):
        return {"error": f"Path blocked for safety: {path}. Sensitive endpoints are off-limits via app_api.", "exit_code": 1}

    method = (args.get("method") or "GET").upper()
    if method not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
        return {"error": f"Unsupported method: {method}", "exit_code": 1}
    if any(method == m and path.startswith(p) for m, p in _APP_API_BLOCKLIST_METHOD_PATH):
        if "/api/email/accounts" in path:
            return {"error": "Don't use /api/email/accounts via app_api — it is owner-filtered in tool context and may return empty. Use the `list_email_accounts` email tool, then pass `account` to list_emails/read_email.", "exit_code": 1}
        if "/api/cookbook/packages/install" in path:
            return {"error": "Don't POST /api/cookbook/packages/install via app_api — package installation is host code execution. Use the dedicated Cookbook dependency UI/flow instead.", "exit_code": 1}
        if "/api/cookbook/rebuild-engine" in path:
            return {"error": "Don't POST /api/cookbook/rebuild-engine via app_api — engine rebuild mutates local or remote host state. Use the dedicated Cookbook UI/flow instead.", "exit_code": 1}
        if "/api/cookbook/kill-pid" in path:
            return {"error": "Don't POST /api/cookbook/kill-pid via app_api — process signalling is host control. Use the dedicated Cookbook stop/diagnostic flow instead.", "exit_code": 1}
        if "/api/model/download" in path:
            return {"error": "Don't POST /api/model/download directly — use the `download_model` tool (it resolves the server name, sets the venv env_prefix, and registers the task so it shows in the UI).", "exit_code": 1}
        if "/api/model/serve" in path:
            return {"error": "Don't POST /api/model/serve directly — use the `serve_model` or `serve_preset` tool (handles host resolution, env_prefix, and cookbook tracking).", "exit_code": 1}
        if "/api/research/start" in path:
            return {"error": "Don't POST /api/research/start directly — use the `trigger_research` tool (it surfaces the session in the Deep Research sidebar).", "exit_code": 1}
        if "/api/notes" in path:
            return {"error": "Don't hit /api/notes via app_api — use the `manage_notes` tool. It accepts natural-language due_date ('11pm today', 'tomorrow at 9am'), fires reminders from the due_date itself (no separate calendar event), and uses the caller's timezone. The raw endpoint requires ISO-UTC + a separate calendar event, both of which the agent tends to get wrong.", "exit_code": 1}
        if "/api/calendar/events" in path:
            return {"error": "Don't hit /api/calendar/events via app_api — use the `manage_calendar` tool. It handles tz-aware natural-language datetimes and reminder_minutes correctly. If the user wants a note + reminder, prefer `manage_notes` with due_date — it bundles both.", "exit_code": 1}
        return {"error": f"{method} {path} is blocked — it overwrites the whole cookbook state file. Use list_serve_presets / serve_preset / serve_model instead.", "exit_code": 1}

    body = args.get("body")
    query = args.get("query") or None
    # Pass owner so the backend impersonates the user — without this,
    # POSTs (notes, calendar, todos, ...) get owner="internal-tool"
    # and the user that asked for them can't see the result.
    headers = {**_internal_headers(owner=owner), "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.request(
                method, f"{base}{path}",
                json=body if body is not None and method in ("POST", "PUT", "PATCH") else None,
                params=query,
                headers=headers,
            )
        # Try to parse JSON; fall back to raw text.
        try:
            payload = resp.json()
            preview = json.dumps(payload, indent=2, default=str)
            if len(preview) > 4000:
                preview = preview[:4000] + "\n... (truncated)"
        except Exception:
            payload = None
            preview = (resp.text or "")[:4000]
        if resp.status_code >= 400:
            return {
                "error": f"{method} {path} -> HTTP {resp.status_code}",
                "status_code": resp.status_code,
                "body": preview,
                "exit_code": 1,
            }
        return {
            "output": f"{method} {path} -> {resp.status_code}\n{preview}",
            "status_code": resp.status_code,
            "json": payload,
            "exit_code": 0,
        }
    except Exception as e:
        return {"error": f"{method} {path} failed: {e}", "exit_code": 1}


# Patterns for detecting running LLM/diffusion model servers outside
# the cookbook's task tracker. Each entry: (label, substring-list).
# Match is case-insensitive against the FULL cmdline. First-match wins.
_MODEL_PROCESS_PATTERNS = [
    ("vLLM",            ["vllm.entrypoints", "vllm serve", "/vllm/", "vllm-openai"]),
    ("SGLang",          ["sglang.launch_server", "sglang/launch_server"]),
    ("llama.cpp",       ["llama-server", "llama_cpp_server", "llamacppserver"]),
    ("Ollama",          ["ollama serve", "ollama runner", "/ollama "]),
    ("ComfyUI",         ["comfyui/main.py", "/ComfyUI/main.py", "ComfyUI"]),
    ("A1111 WebUI",     ["stable-diffusion-webui/webui", "stable-diffusion-webui/launch", "webui.sh"]),
    ("Fooocus",         ["Fooocus/entry_with_update", "Fooocus/launch"]),
    ("InvokeAI",        ["invokeai-web", "invokeai.app", "invokeai/api_app"]),
    ("Forge WebUI",     ["stable-diffusion-webui-forge", "forge/webui"]),
    ("SD.Next",         ["automatic/webui", "sd.next"]),
    ("TGI",             ["text-generation-launcher", "text_generation_launcher"]),
    ("Aphrodite",       ["aphrodite.endpoints", "aphrodite-engine"]),
    ("Triton",          ["tritonserver", "triton/main"]),
    ("Diffusers",       ["diffusers.pipelines", "StableDiffusionInpaintPipeline", "DiffusionPipeline"]),
]


def _cookbook_apply_retry_suggestion(cmd: str, suggestion: Dict[str, Any]) -> str:
    """Apply a structured Cookbook diagnosis suggestion to a serve command."""
    if not cmd or not suggestion:
        return cmd
    op = suggestion.get("op")
    if op == "append":
        arg = (suggestion.get("arg") or "").strip()
        if not arg or arg in cmd:
            return cmd
        return f"{cmd.rstrip()} {arg}"
    if op == "remove":
        flag = (suggestion.get("flag") or "").strip()
        if not flag:
            return cmd
        return re.sub(rf"\s*{re.escape(flag)}(?:\s+\S+)?", "", cmd).strip()
    if op == "replace":
        flag = (suggestion.get("flag") or "").strip()
        value = str(suggestion.get("value") or "").strip()
        if not flag or not value:
            return cmd
        repl = f"{flag} {value}"
        if re.search(rf"(^|\s){re.escape(flag)}(\s+\S+)?", cmd):
            return re.sub(rf"(^|\s){re.escape(flag)}(?:\s+\S+)?", lambda m: (m.group(1) or " ") + repl, cmd).strip()
        return f"{cmd.rstrip()} {repl}"
    return cmd


def _scan_running_model_processes() -> List[Dict[str, Any]]:
    """Scan /proc for running model server processes. Linux-only; returns
    [] on other platforms or if /proc isn't accessible. Each match returns
    a dict shaped like a cookbook task so the caller can merge cleanly.
    """
    import os
    if not os.path.isdir("/proc"):
        return []
    out: List[Dict[str, Any]] = []
    seen_keys = set()
    try:
        for pid_dir in os.listdir("/proc"):
            if not pid_dir.isdigit():
                continue
            try:
                with open(f"/proc/{pid_dir}/cmdline", "rb") as f:
                    raw = f.read()
            except (OSError, PermissionError):
                continue
            if not raw:
                continue
            # cmdline is NUL-separated; join with spaces for matching/display
            cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
            if not cmdline:
                continue
            lower = cmdline.lower()
            for label, needles in _MODEL_PROCESS_PATTERNS:
                if any(n.lower() in lower for n in needles):
                    # Dedupe by (label, first-arg) — multi-worker servers
                    # spawn N processes; only show one row per server.
                    key = (label, cmdline.split(" ")[0])
                    if key in seen_keys:
                        break
                    seen_keys.add(key)
                    # Try to pluck a model name out of the cmdline.
                    model = ""
                    for tok in cmdline.split():
                        if "/" in tok and any(s in tok.lower() for s in (
                            "model", "checkpoint", ".safetensors", ".gguf", ".bin", "huggingface"
                        )):
                            model = tok
                            break
                    out.append({
                        "session_id": f"pid-{pid_dir}",
                        "model": model or label,
                        "phase": "running (external)",
                        "type": "serve",
                        "remote": "local",
                        "pid": int(pid_dir),
                        "label": label,
                        "cmdline_preview": cmdline[:140] + ("…" if len(cmdline) > 140 else ""),
                        "external": True,
                    })
                    break
    except Exception as e:
        logger.debug(f"_scan_running_model_processes failed: {e}")
    return out


async def do_edit_image(content: str, owner: Optional[str] = None) -> Dict:
    """Edit a gallery image (upscale, rembg, inpaint, harmonize)."""
    import httpx
    try:
        args = _parse_tool_args(content)
    except ValueError:
        return {"error": "Invalid JSON arguments", "exit_code": 1}
    image_id = args.get("image_id", "")
    action = args.get("action", "")
    if not image_id or not action:
        return {"error": "image_id and action are required", "exit_code": 1}
    payload = {"image_id": image_id}
    if args.get("prompt"):
        payload["prompt"] = args["prompt"]
    if args.get("scale"):
        payload["scale"] = args["scale"]
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{_INTERNAL_BASE}/api/gallery/{action}", json=payload)
            data = resp.json()
        if data.get("success") or data.get("id"):
            return {"output": f"Image edited ({action}). New image ID: {data.get('id', '?')}", "exit_code": 0}
        return {"error": data.get("error", f"{action} failed"), "exit_code": 1}
    except Exception as e:
        return {"error": str(e), "exit_code": 1}


# ── Research tools ──

