"""
Briefing Actions.

Extracted from src.builtin_actions
"""

from typing import Tuple


async def action_daily_brief(owner: str, **kwargs) -> Tuple[str, bool]:
    """Build a short morning digest: today's calendar events, unread email count
    + top-N senders/subjects, active todos."""
    try:
        from datetime import datetime as _dt, timedelta as _td
        import json as _json

        from core.database import SessionLocal, CalendarEvent, CalendarCal, Note
        from routes.email_helpers import _imap_connect, _decode_header

        # ----- Calendar: today's events -----
        today = _dt.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + _td(days=1)
        # v2 review HIGH-12: gate the OR-null branch on single-user
        # (unconfigured) deploys only. In a multi-user deploy, one
        # user's daily brief must not include another user's notes or
        # events that happen to be stored with owner=None.
        try:
            from core.auth import AuthManager
            _allow_null = not AuthManager().is_configured
        except Exception:
            _allow_null = False
        db = SessionLocal()
        try:
            ev_q = db.query(CalendarEvent).join(CalendarCal).filter(
                CalendarEvent.dtstart < tomorrow,
                CalendarEvent.dtend > today,
                CalendarEvent.status != "cancelled",
            )
            if owner:
                ev_q = owner_filter(ev_q, CalendarCal, owner, include_shared=_allow_null)
            events = ev_q.order_by(CalendarEvent.dtstart).all()
            # ----- Notes: pinned + non-archived todos with at least one undone item -----
            n_q = db.query(Note).filter(Note.archived == False)  # noqa: E712
            if owner:
                n_q = owner_filter(n_q, Note, owner, include_shared=_allow_null)
            notes = n_q.all()
        finally:
            db.close()

        # ----- Email: unread count + top 5 inbox subjects (best-effort) -----
        # Direct IMAP: cheaper than the full _list_emails_sync helper and
        # avoids the module/import coupling that broke this once already.
        unread_count = 0
        recent_subjects: list[tuple[str, str]] = []
        try:
            import email as _email
            conn = _imap_connect(None)
            try:
                conn.select("INBOX", readonly=True)
                status, data = conn.search(None, "UNSEEN")
                uids = (data[0].split() if status == "OK" and data and data[0] else [])
                unread_count = len(uids)
                # Grab headers for the most recent 5 unread (UIDs increase with arrival)
                for uid in uids[-5:][::-1]:
                    try:
                        _, msg_data = conn.fetch(uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT)])")
                        if not msg_data or not msg_data[0]:
                            continue
                        hdr = msg_data[0][1] if isinstance(msg_data[0], tuple) else msg_data[0]
                        parsed = _email.message_from_bytes(hdr)
                        subject = _decode_header(parsed.get("Subject") or "") or "(no subject)"
                        from_raw = _decode_header(parsed.get("From") or "") or "?"
                        # Extract just the display name if "Name <addr>" form
                        if "<" in from_raw:
                            name = from_raw.split("<", 1)[0].strip().strip('"') or from_raw
                        else:
                            name = from_raw
                        recent_subjects.append((name, subject))
                    except Exception as fe:
                        logger.debug(f"daily_brief: header fetch for uid {uid} failed: {fe}")
            finally:
                try: conn.logout()
                except Exception: pass
        except Exception as ee:
            logger.debug(f"daily_brief: email fetch failed: {ee}")

        # Pull active todo items from notes
        todo_lines: list[str] = []
        for n in notes:
            if n.note_type == "checklist" and n.items:
                try:
                    items = _json.loads(n.items)
                    pending = [it.get("text", "") for it in items if not it.get("done")]
                    for t in pending[:3]:
                        if t:
                            todo_lines.append(f"{n.title or 'Checklist'}: {t}")
                except Exception:
                    continue
            elif n.pinned and n.title:
                todo_lines.append(n.title)

        # ----- Compose -----
        # %-d is GNU-only; format the day with str() so the brief works on
        # Windows / non-glibc Python builds too.
        date_label = today.strftime(f"%A, %B {today.day}, %Y")

        plain = [f"Daily brief — {date_label}", ""]
        if events:
            plain.append("Calendar:")
            for e in events:
                t = e.dtstart.strftime("%H:%M") if not e.all_day else "all day"
                loc = f" @ {e.location}" if e.location else ""
                plain.append(f"  {t}  {e.summary}{loc}")
            plain.append("")
        else:
            plain.append("Calendar: nothing scheduled.")
            plain.append("")

        plain.append(f"Email: {unread_count} unread")
        for sender, subj in recent_subjects:
            plain.append(f"  · {sender} — {subj}")
        plain.append("")

        if todo_lines:
            plain.append("Todos:")
            for t in todo_lines[:10]:
                plain.append(f"  · {t}")
        else:
            plain.append("Todos: none active.")

        plain_body = "\n".join(plain)

        return plain_body, True
    except Exception as e:
        logger.error(f"daily_brief action failed: {e}")
        return str(e), False


