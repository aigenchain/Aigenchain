"""
builtin_actions.py

Registry of built-in automation actions that can be executed by the task
scheduler without needing an LLM call.
"""

import logging
import os
from datetime import datetime
from typing import Tuple

from src.auth_helpers import owner_filter
from core.platform_compat import IS_WINDOWS, find_bash
from core.constants import internal_api_base
from src.constants import DATA_DIR, DEEP_RESEARCH_DIR, TIDY_CALENDAR_STATE_FILE, EMAIL_URGENCY_CACHE_DIR, COOKBOOK_STATE_FILE

logger = logging.getLogger(__name__)


class TaskNoop(BaseException):
    """Raised by an action when it determined there's nothing to do.

    Inherits from BaseException (not Exception) so the standard
    `except Exception` wrappers each action uses for real error handling
    don't accidentally catch it. The scheduler explicitly catches TaskNoop,
    drops the queued TaskRun row, advances last_run / next_run, and exits
    silently. Nothing appears in the Activity log; the message is logged
    server-side only.
    """


class TaskDeferred(BaseException):
    """Raised when a task should run later without recording a skipped run."""

    def __init__(self, reason: str, delay_seconds: int = 20 * 60):
        super().__init__(reason)
        self.reason = reason
        self.delay_seconds = delay_seconds



from runtime.actions.maintenance import (
    action_tidy_sessions,
    action_tidy_documents,
    action_consolidate_memory,
    _run_subprocess,
    action_ssh_command,
    action_run_script,
    action_run_local,
    action_tidy_research,
    action_tidy_calendar,
    _result_has_work,
)



from runtime.actions.email_actions import (
    action_summarize_emails,
    action_draft_email_replies,
    _classify_event_heuristic,
    _memory_context_lines,
    action_classify_events,
    action_ping_events,
    action_extract_email_events,
    action_learn_sender_signatures,
)



from runtime.actions.briefing import (
    action_daily_brief,
)


async def action_test_skills(owner: str, **kwargs) -> Tuple[str, bool]:
    """Run the per-skill Test on every skill: agent runs the procedure in a
    sandbox, LLM judges the transcript, verdict is recorded on the skill.
    ADVISORY ONLY — only writes set_audit (never rewrites SKILL.md, never
    demotes status, never overrides confidence)."""
    try:
        from services.memory.skills import SkillsManager
        from src.constants import DATA_DIR
        from routes.skills_routes import _run_skill_test_once, _skill_test_task
        from src.endpoint_resolver import resolve_endpoint

        # #3 SCOPE GUARD: refuse to run on a None/empty owner — otherwise
        # `sm.load(owner=None)` returns every user's skills and we'd cross-
        # test (and write audit verdicts to) other users' data in a
        # multi-user deployment.
        if not owner:
            return "test_skills requires an owner on the task — refusing to run without scope.", False

        sm = SkillsManager(DATA_DIR)
        skills = sm.load(owner=owner)
        names = [s.get("name") for s in skills if s.get("name")]
        if not names:
            raise TaskNoop("no skills to test")

        url, model, headers = resolve_endpoint("default", owner=owner)
        if not url or not model:
            return "No Default/Utility model configured — set one in Settings.", False

        # #2 NO SILENT MODEL SWAP: if the configured model isn't served by the
        # endpoint, try a basename match — but fail loudly instead of grabbing
        # `avail[0]` which could be an embedding-only model and produce 36
        # garbage transcripts → 36 'unknown' verdicts with no hint why.
        try:
            from src.llm_core import list_model_ids
            avail = list_model_ids(url, headers=headers)
            if avail and model not in avail:
                import os as _os
                base = _os.path.basename((model or "").rstrip("/"))
                m = next((a for a in avail if _os.path.basename(a.rstrip("/")) == base), None)
                if m:
                    model = m
                else:
                    return (f"Default model '{model}' not served by endpoint {url}. "
                            f"Available: {', '.join(avail[:8])}{'…' if len(avail) > 8 else ''}. "
                            "Set a valid Default model in Settings."), False
        except Exception as _e:
            logger.warning(f"test_skills model resolve check failed (continuing): {_e}")

        logger.info(f"test_skills: starting on {len(names)} skills, model={model}, owner={owner!r}")

        from collections import Counter
        tally = Counter()
        per_skill_log = []
        for skill in skills:
            name = skill.get("name")
            if not name:
                continue
            md = sm.read_skill_md(name, owner=owner) or ""
            if not md:
                tally["skipped"] += 1
                per_skill_log.append(f"{name}: skipped (no SKILL.md)")
                continue
            task = _skill_test_task(skill)
            try:
                transcript, verdict = await _run_skill_test_once(md, task, url, model, headers, owner)
                v = (verdict or {}).get("verdict") or "unknown"
                tally[v] += 1
                summary = (verdict or {}).get("summary") or ""
                tlen = len(transcript or "")
                detail = ""
                if v in ("unknown", "inconclusive", "fail", "needs_work"):
                    bits = []
                    if summary: bits.append(summary[:160])
                    if tlen < 200: bits.append(f"transcript {tlen}b")
                    if bits: detail = " — " + "; ".join(bits)
                per_skill_log.append(f"{name}: {v}{detail}")
                # #4 + #8 + #12: ONLY persist a real verdict (pass / needs_work /
                # fail / inconclusive). Skip 'unknown' — that's the judge's
                # "couldn't parse" sentinel, not a real result, and persisting
                # it pollutes the verified-badge UI. Also skip the confidence
                # rewrite entirely — update_skill() re-serialises SKILL.md
                # (contradicts "advisory only" docstring) and overwriting a
                # user-set value (e.g. 1.0 → 0.95) is destructive.
                if v in ("pass", "needs_work", "fail", "inconclusive"):
                    try:
                        sm.set_audit(name, v, by_teacher=False, worker_model=model, owner=owner)
                    except Exception as _e:
                        logger.warning(f"test_skills set_audit({name}) failed: {_e}")
                if v == "unknown":
                    logger.warning(f"test_skills: {name} → unknown — {summary[:200]}; transcript_len={tlen}")
            except Exception as e:
                logger.exception(f"test_skills: {name} errored")
                tally["error"] += 1
                per_skill_log.append(f"{name}: error — {str(e)[:200]}")

        parts = []
        for k in ("pass", "needs_work", "fail", "inconclusive", "unknown", "skipped", "error"):
            if tally.get(k):
                parts.append(f"{tally[k]} {k}")
        header = f"Tested {len(names)} skill(s): " + (" · ".join(parts) or "0")
        # Multi-line result: summary first, then per-skill detail. The Tasks
        # Activity feed renders this verbatim, so the user can see per-skill
        # outcomes + the judge's "why" without checking uvicorn stdout.
        body = "\n".join(per_skill_log)
        return f"{header}\nmodel={model}\n\n{body}", True
    except TaskNoop:
        raise
    except Exception as e:
        logger.error(f"test_skills action failed: {e}")
        return str(e), False


async def action_audit_skills(owner: str, **kwargs) -> Tuple[str, bool]:
    """Run the real skills audit pipeline for skills that have not been audited.

    Unlike test_skills, this uses the same audit logic as the UI Audit all flow:
    metadata narrowing, self-edit/retry, optional teacher rewrite, necessity
    tagging, and publish/draft finalization from the user's confidence threshold.
    """
    try:
        from services.memory.skills import SkillsManager
        from src.constants import DATA_DIR
        from routes.skills_routes import (
            _resolve_audit_models, _run_audit_all_job, _skill_audit_jobs,
        )

        if not owner:
            return "audit_skills requires an owner — refusing to run without scope.", False

        key = (owner or "",)
        existing = _skill_audit_jobs.get(key)
        if existing and existing.get("status") == "running":
            raise TaskNoop("skill audit already running")

        sm = SkillsManager(DATA_DIR)
        skills = sm.load(owner=owner)
        names = [
            s.get("name") for s in skills
            if s.get("name") and not s.get("audit_verdict")
        ]
        if not names:
            raise TaskNoop("no unaudited skills")

        url, model, headers, teacher = _resolve_audit_models()
        try:
            from src.llm_core import seconds_since_model_activity
            recent = seconds_since_model_activity(url, model)
        except Exception:
            recent = None
        if recent is not None and recent < (20 * 60):
            raise TaskDeferred(
                f"audit model {model} was used {int(recent)}s ago; waiting for quiet window",
                delay_seconds=20 * 60,
            )

        import time as _time
        _skill_audit_jobs[key] = {
            "status": "running", "scope": "scheduled-unchecked", "model": model,
            "teacher": teacher[1] if teacher else None,
            "total": len(names), "done": 0, "current": None,
            "results": [], "log": [
                f"Scheduled audit of {len(names)} unaudited skill(s) with {model}"
                + (f"; teacher {teacher[1]}" if teacher else "")
            ],
            "started": _time.time(), "cancel": False,
        }
        await _run_audit_all_job(key, sm, names, url, model, headers, teacher, owner)
        job = _skill_audit_jobs.get(key, {})
        counts = {}
        for r in job.get("results", []):
            k = r.get("result") or "unknown"
            counts[k] = counts.get(k, 0) + 1
        summary = " · ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "0 results"
        return f"Audited {job.get('done', 0)}/{len(names)} unaudited skill(s): {summary}", True
    except TaskNoop:
        raise
    except Exception as e:
        logger.error(f"audit_skills action failed: {e}")
        return str(e), False


async def action_ping_notes(owner: str, **kwargs) -> Tuple[str, bool]:
    """Background note-due scanner. Fires a reminder for any note whose
    `due_date` falls in the current ±5-minute window and hasn't been pinged
    within the last 25 minutes. Mirrors `action_ping_events` for calendar.

    State (`data/note_pings.json`): {note_id: iso_ts_of_last_ping}. Pruned
    on each run by dropping entries for notes that are gone/archived/replied.
    """
    try:
        import json as _json
        import time as _time
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        from pathlib import Path as _P
        from core.database import SessionLocal as _SL, Note as _N

        # Per-owner state file so cache-pruning doesn't cross-delete other
        # users' entries (review C4). Legacy path kept as fallback so a
        # single-user install (empty owner) doesn't lose its history.
        _owner_slug = "".join(c if (c.isalnum() or c in "-_.@") else "_" for c in (owner or "default"))
        STATE = _P(DATA_DIR) / f"note_pings_{_owner_slug}.json"
        STATE.parent.mkdir(parents=True, exist_ok=True)
        # One-time migration: if legacy global file exists and per-owner file
        # doesn't, seed from global (entries for OTHER owners still get pruned
        # on their first run — acceptable, prevents silent loss).
        _legacy = _P(DATA_DIR) / "note_pings.json"
        if _legacy.exists() and not STATE.exists():
            try:
                STATE.write_text(_legacy.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass
        # Scanner ticks every 60s in _note_pings_loop. 90s window guarantees
        # every note's due time lands inside at least one tick's window.
        WINDOW_SEC = 90
        REPING_MIN = 25     # don't re-ping same note more often than this

        def _parse_due(s: str):
            """Accept '2026-05-29T16:31' (local) or '...Z' (UTC). Returns UTC datetime."""
            if not s:
                return None
            try:
                # Handle the JS-style 'Z' suffix.
                if s.endswith("Z"):
                    return _dt.fromisoformat(s[:-1]).replace(tzinfo=_tz.utc)
                # Naive → assume local server time.
                d = _dt.fromisoformat(s)
                if d.tzinfo is None:
                    d = d.astimezone().astimezone(_tz.utc)
                return d.astimezone(_tz.utc)
            except Exception:
                return None

        try:
            cache = _json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
        except Exception:
            cache = {}

        db = _SL()
        try:
            q = db.query(_N).filter(_N.archived == False)  # noqa: E712
            q = q.filter(_N.due_date.isnot(None), _N.due_date != "")
            if owner:
                # Match owner OR legacy null-owner notes (single-user installs).
                q = owner_filter(q, _N, owner)
            notes = q.all()
            if not notes:
                raise TaskNoop("no notes with due dates")

            now = _dt.now(_tz.utc)
            window = _td(seconds=WINDOW_SEC)
            reping_cutoff = now - _td(minutes=REPING_MIN)
            seen_ids = set()
            sent = []

            for n in notes:
                seen_ids.add(n.id)
                due = _parse_due(n.due_date)
                if not due:
                    continue
                # Inside the ±5min window?
                if abs((due - now).total_seconds()) > window.total_seconds():
                    continue
                # Recently pinged? Skip.
                last = cache.get(n.id)
                if last:
                    try:
                        if isinstance(last, dict):
                            last = last.get("at")
                        last_dt = _dt.fromisoformat(str(last))
                        if last_dt.tzinfo is None:
                            last_dt = last_dt.replace(tzinfo=_tz.utc)
                        if last_dt >= reping_cutoff:
                            continue
                    except Exception:
                        pass
                # Compose + dispatch.
                title = (n.title or "Reminder").strip() or "Reminder"
                body_parts = []
                if n.content:
                    body_parts.append(n.content[:400])
                # Items: list pending checklist entries inline.
                if n.items:
                    try:
                        items = _json.loads(n.items)
                        pending = [
                            it.get("text", "")
                            for it in items
                            if not it.get("done") and not it.get("checked")
                        ]
                        if pending:
                            body_parts.append("Pending:\n" + "\n".join(f"- {t}" for t in pending[:8]))
                    except Exception:
                        pass
                body = "\n\n".join(p for p in body_parts if p) or title
                try:
                    from routes.note_routes import dispatch_reminder
                    await dispatch_reminder(
                        title=title, note_body=body, note_id=n.id,
                        owner=n.owner or owner or "",
                    )
                    cache[n.id] = now.isoformat()
                    sent.append(title)
                except Exception as e:
                    logger.warning(f"ping_notes: dispatch failed for {n.id}: {e}")

            # Prune cache entries for notes that no longer exist.
            for stale in [k for k in cache if k not in seen_ids]:
                cache.pop(stale, None)

            try:
                STATE.write_text(_json.dumps(cache), encoding="utf-8")
            except Exception as e:
                logger.warning(f"ping_notes: cache write failed: {e}")

            if not sent:
                raise TaskNoop(f"scanned {len(notes)} note(s), none due in ±{WINDOW_SEC}s")
            preview = "; ".join(sent[:3])
            extra = f" (+{len(sent) - 3} more)" if len(sent) > 3 else ""
            return f"Pinged {len(sent)} note(s): {preview}{extra}", True
        finally:
            db.close()
    except TaskNoop:
        raise
    except Exception as e:
        logger.exception("ping_notes action failed")
        return str(e), False


async def action_check_email_urgency(owner: str, **kwargs) -> Tuple[str, bool]:
    """Scan unread emails across all accounts, LLM-triage new ones, cache
    per-UID verdicts, tag the inbox, and fire a reminder when a previously
    unseen UID scores reply-soon/urgent (>=2). State persists under
    data/email_urgency_state_* so the UI can color the unread dot by tier.

    Design notes:
    - Only classifies emails newer than 7 days (first-run scale guard).
    - Cache key = `<account_id>:<uid>` so the same UID across accounts doesn't collide.
    - Re-notify gate: only when at least one UID NEW to `notified_uids` scores ≥2.
      Repeat scans where the set is unchanged stay silent.
    """
    from src.settings import load_settings

    try:
        settings = load_settings()
        import json as _json
        import email as _email_mod
        import asyncio as _aio
        import os as _os
        import re as _re
        import time as _time
        import httpx
        from datetime import datetime as _dt, timedelta as _td
        from pathlib import Path as _P
        from core.database import SessionLocal as _SL, EmailAccount as _EA
        from routes.email_helpers import _imap_connect, _decode_header
        from src.endpoint_resolver import resolve_endpoint, resolve_utility_fallback_candidates
        from src.llm_core import llm_call_async_with_fallback

        # Per-owner state file so multi-user runs don't clobber each other's
        # notified_uids / urgency counts. Empty owner falls back to a generic
        # filename for single-user installs (matches prior behaviour).
        _owner_slug = "".join(c if (c.isalnum() or c in "-_.@") else "_" for c in (owner or "default"))
        STATE_PATH = _P(DATA_DIR) / f"email_urgency_state_{_owner_slug}.json"
        CACHE_DIR = _P(EMAIL_URGENCY_CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        AGE_CUTOFF = _dt.utcnow() - _td(days=7)
        TRIAGE_VERSION = 3
        CATEGORY_TAGS = {
            "newsletter", "marketing", "notification", "finance", "bills",
            "receipt", "travel", "security", "shopping", "social", "work",
            "personal", "calendar",
        }
        MANAGED_TAGS = CATEGORY_TAGS | {"urgent", "reply-soon", "promo"}

        # ── 1. Resolve LLM candidates (utility primary + utility fallbacks; fall
        # through to default chat as a last resort).
        url, model, headers = resolve_endpoint("utility", owner=owner)
        if not url or not model:
            url, model, headers = resolve_endpoint("default", owner=owner)
        if not url or not model:
            return "No LLM endpoint available", False
        candidates = [(url, model, headers)] + resolve_utility_fallback_candidates(owner=owner)

        # ── 2. Enumerate enabled accounts. Match this task's owner AND fall
        # back to the legacy "unowned account whose imap_user / from_address
        # == this owner" pattern — same rule `_get_email_config` uses, so a
        # pre-multi-user account row still gets picked up for the seeded task.
        db = _SL()
        try:
            from sqlalchemy import and_ as _and, or_ as _or
            q = db.query(_EA).filter(_EA.enabled == True)  # noqa: E712
            if owner:
                unowned = _or(_EA.owner == None, _EA.owner == "")  # noqa: E711
                same_mailbox = _or(_EA.imap_user == owner, _EA.from_address == owner)
                q = q.filter(_or(_EA.owner == owner, _and(unowned, same_mailbox)))
            accounts = q.all()
        finally:
            db.close()
        if not accounts:
            raise TaskNoop("no email accounts configured")

        urgency_prompt = settings.get("urgent_email_prompt", "")
        per_uid_scores = {}   # key = "<acc_id>:<uid>" → {"score": 0-3, "reason": "..."}
        all_unread_keys = set()  # for cache pruning
        llm_attempts = 0
        saved_classifications = 0
        failed_classifications = []
        scanned = 0

        # ── 3. Per-account scan: pull headers + lightweight body for new UIDs
        # since 7 days ago, score via LLM, cache the verdict.
        for acc in accounts:
            cache_file = CACHE_DIR / f"{acc.id}.json"
            try:
                cache = _json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {"uids": {}}
            except Exception:
                cache = {"uids": {}}

            def _scan_one(account=acc, cache_uids=cache.get("uids", {})):
                """Sync IMAP work runs in a thread."""
                results = []
                conn = _imap_connect(account.id)
                try:
                    conn.select("INBOX", readonly=True)
                    # IMAP date is the only practical pre-filter — UNSEEN AND
                    # SINCE 7-days-ago. Date format is DD-Mon-YYYY.
                    since_str = AGE_CUTOFF.strftime("%d-%b-%Y")
                    status, data = conn.search(None, f'(UNSEEN SINCE {since_str})')
                    if status != "OK" or not data or not data[0]:
                        return results
                    uids = data[0].split()
                    for uid_b in uids:
                        uid = uid_b.decode() if isinstance(uid_b, bytes) else str(uid_b)
                        key = f"{account.id}:{uid}"
                        cached = cache_uids.get(uid)
                        cached_ok = isinstance(cached, dict) and cached.get("triage_version") == TRIAGE_VERSION
                        results.append({"key": key, "uid": uid, "cached": cached if cached_ok else None})
                        if cached_ok:
                            # Already classified — skip the fetch.
                            continue
                        # Pull headers + first ~800 chars of plaintext body.
                        try:
                            st, msg_data = conn.fetch(uid_b, "(RFC822.HEADER BODY.PEEK[TEXT]<0.800>)")
                            if st != "OK" or not msg_data:
                                continue
                            # Headers + body land in different tuples in the
                            # response — concatenate the bytes for parsing.
                            raw = b""
                            for part in msg_data:
                                if isinstance(part, tuple) and part[1]:
                                    raw += part[1] + b"\n\n"
                            if not raw:
                                continue
                            msg = _email_mod.message_from_bytes(raw)
                            # Skip Odysseus-generated reminders so the scanner
                            # doesn't classify its own emails as urgent and
                            # trigger a feedback loop. Match on either the
                            # stamped headers OR the subject prefix.
                            _ody_origin = (msg.get("X-Odysseus-Origin") or "").strip().lower()
                            _ody_kind = (msg.get("X-Odysseus-Kind") or "").strip().lower()
                            _raw_subj = (msg.get("Subject") or "").lower()
                            # MCP path drops custom headers (email_server's
                            # schema doesn't accept them), so we ALSO match the
                            # `[Task]` subject prefix that `_deliver_via_mcp`
                            # always stamps. Anything that looks self-generated
                            # is dropped before classification to prevent the
                            # scanner from labelling its own emails "urgent".
                            if (_ody_origin == "odysseus-ui" or _ody_kind == "reminder"
                                    or _raw_subj.startswith("reminder (odysseus):")
                                    or _raw_subj.startswith("reminder:")
                                    or _raw_subj.startswith("[task]")):
                                # Drop this candidate entirely — don't list it
                                # in results so its UID never enters the cache
                                # nor counts toward `scanned`.
                                results.pop()
                                continue
                            subject = _decode_header(msg.get("Subject") or "")
                            from_raw = _decode_header(msg.get("From") or "")
                            header_blob = "\n".join(
                                f"{name}: {msg.get(name, '')}"
                                for name in (
                                    "From", "Subject", "List-Unsubscribe", "List-ID",
                                    "Precedence", "X-Mailchimp-Campaign-Id",
                                    "X-Campaign", "X-MC-User",
                                )
                                if msg.get(name)
                            )
                            body_snippet = ""
                            try:
                                if msg.is_multipart():
                                    for part in msg.walk():
                                        if part.get_content_type() == "text/plain":
                                            body_snippet = part.get_payload(decode=True).decode("utf-8", errors="ignore")[:1600]
                                            break
                                else:
                                    body_snippet = (msg.get_payload(decode=True) or b"").decode("utf-8", errors="ignore")[:1600]
                            except Exception:
                                body_snippet = ""
                            results[-1].update({
                                "subject": subject,
                                "from": from_raw,
                                "headers": header_blob,
                                "body": body_snippet.strip(),
                                "message_id": (msg.get("Message-ID") or "").strip(),
                            })
                        except Exception as _fe:
                            logger.debug(f"urgency: header fetch for uid {uid} failed: {_fe}")
                finally:
                    try: conn.logout()
                    except Exception: pass
                return results

            try:
                items = await _aio.to_thread(_scan_one)
            except Exception as e:
                logger.warning(f"urgency: IMAP scan failed for account {acc.id}: {e}")
                continue

            for item in items:
                scanned += 1
                key = item["key"]
                all_unread_keys.add(key)
                if item.get("cached"):
                    per_uid_scores[key] = item["cached"]
                    continue
                # Skip uids we couldn't fetch (no subject/from/body).
                if not item.get("subject") and not item.get("from"):
                    continue
                # ── LLM-classify. JSON-only response; bullet-proof parse.
                llm_attempts += 1
                prompt = (
                    "You are triaging ONE unread email. Return ONLY JSON: "
                    "{\"score\":0|1|2|3,\"tags\":[\"...\"],\"spam\":false,"
                    "\"reason\":\"one short phrase\"}.\n"
                    "0 = trivial / promotional · 1 = informational, no reply needed · "
                    "2 = should reply within a day · 3 = urgent, reply now (deadline, blocker).\n\n"
                    "Allowed tags: newsletter, marketing, notification, finance, bills, receipt, "
                    "travel, security, shopping, social, work, personal, calendar.\n"
                    "Use marketing for ads, promos, sales, offers, and cold sales. Use newsletter "
                    "for newsletters, digests, and recurring content. spam=true for scams, phishing, "
                    "junk, cold sales, generic ads, or no-personal-action bulk mail.\n"
                    "Important: 'I'm outside', 'I am outside', 'waiting outside', 'at the door', "
                    "'locked out', or 'can't get in' means score 3 unless clearly historical.\n\n"
                    f"User's rules:\n{urgency_prompt}\n\n"
                    f"Email:\nFrom: {item.get('from','')}\nSubject: {item.get('subject','')}\n"
                    f"Snippet:\n{item.get('body','')}\n"
                )
                try:
                    raw = await llm_call_async_with_fallback(
                        candidates,
                        [{"role": "user", "content": prompt}],
                        temperature=0.1, max_tokens=220, timeout=30,
                    )
                    # Tolerant JSON-parse: strip code fences if present.
                    txt = (raw or "").strip()
                    if txt.startswith("```"):
                        txt = txt.strip("`")
                        # Drop a leading "json\n" or any tag.
                        nl = txt.find("\n")
                        if nl >= 0:
                            txt = txt[nl + 1:]
                    # Find first { ... } in the response.
                    s = txt.find("{")
                    e = txt.rfind("}")
                    if s < 0 or e <= s:
                        failed_classifications.append({
                            "subject": item.get("subject") or "(no subject)",
                            "from": item.get("from") or "",
                            "reason": "model returned no JSON",
                        })
                        continue
                    obj = _json.loads(txt[s:e + 1])
                    score = int(obj.get("score", 0))
                    reason = str(obj.get("reason", ""))[:200]
                    raw_tags = obj.get("tags") or []
                    if isinstance(raw_tags, str):
                        raw_tags = [raw_tags]
                    tags = []
                    for t in raw_tags:
                        if not isinstance(t, str):
                            continue
                        tag = t.strip().lower().replace("_", "-")
                        if tag == "promo":
                            tag = "marketing"
                        if tag in CATEGORY_TAGS and tag not in tags:
                            tags.append(tag)
                    _spam_raw = obj.get("spam")
                    if isinstance(_spam_raw, bool):
                        spam = _spam_raw
                    elif isinstance(_spam_raw, (int, float)):
                        spam = bool(_spam_raw)
                    else:
                        spam = str(_spam_raw or "").strip().lower() in {"1", "true", "yes", "y"}
                    _blob = f"{item.get('headers','')}\n{item.get('subject','')}\n{item.get('body','')}".lower()
                    if _re.search(r"\b(i'?m|i am|im|we'?re|we are)\s+outside\b", _blob) or _re.search(
                        r"\b(waiting outside|at the door|locked out|can'?t get in|cannot get in)\b", _blob
                    ):
                        if score < 3:
                            reason = "person is waiting outside"
                        score = max(score, 3)
                    bulkish = bool(_re.search(
                        r"\b(list-unsubscribe|list-id|mailchimp|mailchimpapp|view this email in your browser|unsubscribe|newsletter|digest|precedence:\s*bulk)\b",
                        _blob,
                    ))
                    marketingish = bool(_re.search(
                        r"\b(advertisement|sponsored|promo|promotion|sale|discount|offer|limited time|deal|tickets?|tour|merch|stream|purchase|sold out|low tickets|coupon|shop now|buy now)\b",
                        _blob,
                    ))
                    if "newsletter" not in tags and bulkish:
                        tags.append("newsletter")
                    if "marketing" not in tags and marketingish:
                        tags.append("marketing")
                    if (bulkish or marketingish) and score < 2:
                        score = 0
                        if not reason or "urgent" in reason.lower():
                            reason = "Bulk marketing/newsletter; no personal reply needed"
                    # Strip "Name <addr>" to bare display name for compact summary.
                    _from_raw = item.get("from", "") or ""
                    if "<" in _from_raw:
                        _from_short = _from_raw.split("<", 1)[0].strip().strip('"') or _from_raw
                    else:
                        _from_short = _from_raw
                    verdict = {
                        "score": max(0, min(3, score)),
                        "tags": tags[:4],
                        "spam": spam,
                        "reason": reason,
                        "subject": (item.get("subject") or "")[:200],
                        "from": _from_short[:120],
                        "triage_version": TRIAGE_VERSION,
                        # Cache the message_id too so re-scans of already-cached
                        # UIDs can still write the inbox tag without re-LLM'ing.
                        "message_id": (item.get("message_id") or "").strip(),
                        "ts": _time.time(),
                    }
                    cache.setdefault("uids", {})[item["uid"]] = verdict
                    per_uid_scores[key] = verdict
                    saved_classifications += 1
                except Exception as e:
                    failed_classifications.append({
                        "subject": item.get("subject") or "(no subject)",
                        "from": item.get("from") or "",
                        "reason": str(e)[:120] or "classification failed",
                    })
                    logger.debug(f"urgency: LLM classify failed for {key}: {e}")
                    continue

            # ── Prune cache entries for UIDs that are no longer unread (replied
            # / archived / deleted). Compare against `items` (everything UNSEEN
            # in this scan window).
            seen_uids = {it["uid"] for it in items}
            cache_uids = cache.get("uids", {})
            for stale in [u for u in cache_uids if u not in seen_uids]:
                cache_uids.pop(stale, None)

            try:
                cache_file.write_text(_json.dumps(cache), encoding="utf-8")
            except Exception as e:
                logger.warning(f"urgency: cache write failed for {acc.id}: {e}")

        # ── 3.5  Mirror triage verdicts into email_tags so inbox filters and
        # pills show urgency + category tags. Runs for BOTH cached and freshly
        # classified items; message_id lives on the cached verdict so this is cheap.
        try:
            import sqlite3 as _sql3
            from routes.email_helpers import SCHEDULED_DB, _init_scheduled_db
            from datetime import datetime as _dt2
            _init_scheduled_db()
            _conn = _sql3.connect(SCHEDULED_DB)
            try:
                for _key, _v in per_uid_scores.items():
                    _msg_id = (_v.get("message_id") or "").strip()
                    _score = _v.get("score", 0)
                    if not _msg_id:
                        continue
                    _new_tags = []
                    if _score >= 3:
                        _new_tags.append("urgent")
                    elif _score >= 2:
                        _new_tags.append("reply-soon")
                    for _tag in (_v.get("tags") or []):
                        _tag = str(_tag).strip().lower().replace("_", "-")
                        if _tag == "promo":
                            _tag = "marketing"
                        if _tag in CATEGORY_TAGS and _tag not in _new_tags:
                            _new_tags.append(_tag)
                    _spam = 1 if _v.get("spam") else 0
                    # _key is "<account_id>:<uid>" — extract uid for the row.
                    _uid_only = _key.split(":", 1)[-1]
                    _owner_key = owner or ""
                    _row = _conn.execute(
                        "SELECT tags FROM email_tags WHERE message_id=? AND owner=?",
                        (_msg_id, _owner_key),
                    ).fetchone()
                    if _row:
                        try:
                            _existing = _json.loads(_row[0] or "[]")
                            if not isinstance(_existing, list):
                                _existing = []
                        except Exception:
                            _existing = []
                        # Drop previous triage-owned tags so re-classification
                        # can upgrade/downgrade/clear without touching manual tags.
                        _existing = [
                            str(t).strip().lower().replace("_", "-")
                            for t in _existing
                            if str(t).strip().lower().replace("_", "-") not in MANAGED_TAGS
                        ]
                        for _tag in _new_tags:
                            if _tag not in _existing:
                                _existing.append(_tag)
                        _conn.execute(
                            "UPDATE email_tags SET tags=?, spam_verdict=?, spam_reason=?, uid=?, folder=?, subject=?, sender=? "
                            "WHERE message_id=? AND owner=?",
                            (_json.dumps(_existing), _spam, _v.get("reason", ""), _uid_only, "INBOX",
                             _v.get("subject", ""), _v.get("from", ""), _msg_id, _owner_key),
                        )
                    else:
                        if not _new_tags and not _spam:
                            continue
                        _conn.execute(
                            "INSERT INTO email_tags "
                            "(message_id, owner, uid, folder, subject, sender, tags, spam_verdict, spam_reason, created_at) "
                            "VALUES (?, ?, ?, 'INBOX', ?, ?, ?, ?, ?, ?)",
                            (_msg_id, _owner_key, _uid_only, _v.get("subject", ""),
                             _v.get("from", ""), _json.dumps(_new_tags), _spam, _v.get("reason", ""),
                             _dt2.utcnow().isoformat()),
                        )
                _conn.commit()
            finally:
                _conn.close()
        except Exception as _te:
            logger.warning(f"urgency: bulk tag write failed: {_te}")

        # ── 4. Aggregate state. urgent = score ≥ 2.
        urgent_keys = [k for k, v in per_uid_scores.items() if v.get("score", 0) >= 2]
        max_score = max((v.get("score", 0) for v in per_uid_scores.values()), default=0)
        total_urgent = len(urgent_keys)

        # Load prior state to know which urgent UIDs we've already notified.
        try:
            prior = _json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}
        except Exception:
            prior = {}
        notified_uids = set(prior.get("notified_uids", []))

        # ── 5. Fire reminder ONLY when a previously-unnotified UID scores urgent.
        new_urgent = [k for k in urgent_keys if k not in notified_uids]
        newly_notified = set()
        notify_failed = set()
        if new_urgent:
            title = "Urgent email" if total_urgent == 1 else f"{total_urgent} urgent emails"
            # Build a real listing — subject · sender · reason for each urgent
            # one — so the reminder email tells you which messages to act on,
            # not just "4 needing reply". Optional deep-link when the user has
            # `app_public_url` configured in Settings (so the email row links
            # straight into the Odysseus Email tab).
            # Sort: highest-scored UIDs first; cap at 10 to keep the email tidy.
            sorted_urgent = sorted(
                ((k, per_uid_scores[k]) for k in urgent_keys),
                key=lambda kv: kv[1].get("score", 0), reverse=True,
            )[:10]
            _pub = (settings.get("app_public_url") or "").strip().rstrip("/")
            from urllib.parse import quote as _quote
            lines = [f"{total_urgent} email" + ("" if total_urgent == 1 else "s") + " need an urgent reply:", ""]
            for i, (k, v) in enumerate(sorted_urgent, 1):
                subj = (v.get("subject") or "(no subject)")[:160]
                frm = v.get("from") or ""
                why = v.get("reason") or ""
                uid_for_link = str(k).split(":", 1)[-1]
                hash_link = f"#email={_quote('INBOX', safe='')}:{uid_for_link}"
                open_link = f"{_pub}/{hash_link}" if _pub else hash_link
                line = f"{i}. {subj}"
                if frm:
                    line += f"  —  {frm}"
                if why:
                    line += f"  ·  {why}"
                lines.append(line)
                lines.append(f"   Open email: {open_link}")
            if total_urgent > len(sorted_urgent):
                lines.append("")
                lines.append(f"…and {total_urgent - len(sorted_urgent)} more.")
            body = "\n".join(lines)
            try:
                # Call dispatch_reminder DIRECTLY (no HTTP/auth roundtrip — the
                # endpoint version 401's the background scheduler because it
                # has no session cookie).
                from routes.note_routes import dispatch_reminder
                dispatch_result = await dispatch_reminder(
                    title=title, note_body=body, note_id="urgent-email",
                    owner=owner or "",
                )
                channel = (settings.get("reminder_channel") or "browser").strip().lower()
                delivered = bool(dispatch_result.get("browser_sent"))
                if channel == "email":
                    delivered = bool(dispatch_result.get("email_sent"))
                elif channel == "ntfy":
                    delivered = bool(dispatch_result.get("ntfy_sent"))
                elif channel == "webhook":
                    delivered = bool(dispatch_result.get("webhook_sent"))
                if delivered:
                    newly_notified.update(new_urgent)
                else:
                    notify_failed.update(new_urgent)
                    logger.warning(f"urgency: reminder dispatch returned no successful delivery path: {dispatch_result}")
            except Exception as e:
                logger.warning(f"urgency: reminder dispatch failed: {e}")
                notify_failed.update(new_urgent)
            # Mark only successfully delivered UIDs as notified so a transient
            # SMTP/ntfy/browser failure retries instead of lying forever.
            notified_uids.update(newly_notified)

        # Prune notified_uids that aren't unread anymore (so a future re-urgent
        # message with the same UID — rare but possible after archive→unarchive
        # — can re-notify). Keep only UIDs still in `all_unread_keys`.
        notified_uids = {u for u in notified_uids if u in all_unread_keys}

        state = {
            "ts": _time.time(),
            "owner": owner or "",
            "total_unread": len(all_unread_keys),
            "total_urgent": total_urgent,
            "max_score": max_score,
            "per_uid": per_uid_scores,
            "notified_uids": sorted(notified_uids),
        }
        try:
            STATE_PATH.write_text(_json.dumps(state), encoding="utf-8")
        except Exception as e:
            logger.warning(f"urgency: state write failed: {e}")

        # ── 6. Activity-log summary — counts line on top, then per-tier
        # bulleted breakdown so the user can see WHICH emails ranked where
        # (subject · sender · reason) and which ones triggered notifications.
        tier_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for v in per_uid_scores.values():
            tier_counts[v.get("score", 0)] = tier_counts.get(v.get("score", 0), 0) + 1
        if scanned == 0:
            raise TaskNoop("no unread emails in last 7 days")
        head = (
            f"scanned {scanned} · urgent {tier_counts[3]} · "
            f"reply-soon {tier_counts[2]} · info {tier_counts[1]} · trivial {tier_counts[0]} · "
            f"{saved_classifications} saved classifications"
        )
        if llm_attempts != saved_classifications:
            head += f" · {llm_attempts - saved_classifications} failed"
        if newly_notified:
            head += f" · notified {len(newly_notified)}"
        if notify_failed:
            head += f" · notify failed {len(notify_failed)}"

        def _fmt_one(v, newly_notified_set, failed_set, key):
            subj = (v.get("subject") or "(no subject)")[:80]
            frm = v.get("from") or ""
            why = v.get("reason") or ""
            tag = " · *notified now*" if key in newly_notified_set else (" · *notify failed*" if key in failed_set else "")
            line = f"- **{subj}**" + (f" — _{frm}_" if frm else "")
            if why:
                line += f" — {why}"
            return line + tag

        # Sort each tier by reason length (longest reason first → most info).
        by_tier = {3: [], 2: [], 1: [], 0: []}
        for k, v in per_uid_scores.items():
            by_tier.setdefault(v.get("score", 0), []).append((k, v))
        lines = [head]
        tier_labels = {3: "Urgent", 2: "Reply soon", 1: "Informational", 0: "Trivial"}
        for tier in (3, 2, 1, 0):
            items_t = by_tier.get(tier, [])
            if not items_t:
                continue
            lines.append("")
            lines.append(f"**{tier_labels[tier]} ({len(items_t)}):**")
            # Cap each tier at 8 rows to keep the activity entry readable.
            for k, v in items_t[:8]:
                lines.append(_fmt_one(v, newly_notified, notify_failed, k))
            if len(items_t) > 8:
                lines.append(f"…and {len(items_t) - 8} more")
        if failed_classifications:
            lines.append("")
            lines.append(f"**Unclassified ({len(failed_classifications)}):**")
            for v in failed_classifications[:8]:
                subj = (v.get("subject") or "(no subject)")[:80]
                frm = v.get("from") or ""
                why = v.get("reason") or ""
                line = f"- **{subj}**" + (f" — _{frm}_" if frm else "")
                if why:
                    line += f" — {why}"
                lines.append(line)
            if len(failed_classifications) > 8:
                lines.append(f"…and {len(failed_classifications) - 8} more")
        return "\n".join(lines), True
    except TaskNoop:
        raise
    except Exception as e:
        logger.exception("check_email_urgency action failed")
        return str(e), False


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
