"""
Scheduler lifecycle.

Extracted from src.task_scheduler
"""


async def start(self):
    # On startup, mark any leftover "running" task_runs as errored. Without
    # this, a server crash leaves rows stuck running indefinitely and the
    # _executing in-memory set forgets them, so the UI shows phantoms.
    try:
        from core.database import SessionLocal, TaskRun
        db = SessionLocal()
        try:
            # Zombies from a prior server crash. Tagged "aborted" (not
            # "error") so the Activity view + error-rate stats don't
            # falsely blame the task for what was an infrastructure event.
            stale = db.query(TaskRun).filter(
                TaskRun.status.in_(("running", "queued"))
            ).all()
            if stale:
                now = _utcnow()
                for r in stale:
                    old_status = r.status or "running"
                    r.status = "aborted"
                    r.error = "Server restarted while task was " + old_status
                    r.finished_at = now
                db.commit()
                logger.info(f"Cleared {len(stale)} stale task_runs from previous run")
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Could not clear stale task_runs on startup: {e}")

    # Advance next_run for active tasks whose next_run is already in the
    # past. Without this, a restart hits _check_due_tasks() with an empty
    # in-process _executing set, and the same overdue task fires once per
    # poll until it completes.
    try:
        from core.database import SessionLocal as _SL, ScheduledTask as _ST
        db = _SL()
        try:
            now = _utcnow()
            overdue = db.query(_ST).filter(
                _ST.status == "active",
                _ST.next_run.isnot(None),
                _ST.next_run < now,
            ).all()
            if overdue:
                for t in overdue:
                    t.next_run = now + timedelta(seconds=60)
                db.commit()
                logger.info(
                    "Pushed next_run forward by 60s for %d overdue active tasks on startup",
                    len(overdue),
                )
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Could not advance overdue next_run on startup: {e}")

    # Defense-in-depth dedupe sweep: for any owner with >1 rows where
    # is_default_assistant=True, keep the oldest and demote the rest +
    # delete their orphaned check-in tasks. This is the safety net for
    # the synthetic-owner seeding bug (we cleaned a manual instance of
    # it, but a stale code path or DB import could recreate it).
    try:
        from core.database import SessionLocal, CrewMember, ScheduledTask
        db = SessionLocal()
        try:
            from sqlalchemy import func
            groups = db.query(CrewMember.owner, func.count(CrewMember.id).label("n")).filter(
                CrewMember.is_default_assistant == True,  # noqa: E712
            ).group_by(CrewMember.owner).having(func.count(CrewMember.id) > 1).all()
            for owner, n in groups:
                rows = db.query(CrewMember).filter(
                    CrewMember.owner == owner,
                    CrewMember.is_default_assistant == True,  # noqa: E712
                ).order_by(CrewMember.created_at.asc()).all()
                keep = rows[0]
                losers = rows[1:]
                loser_ids = [r.id for r in losers]
                # Delete the orphaned tasks tied to the loser crews — they
                # are duplicates of the keeper's check-ins.
                n_tasks = db.query(ScheduledTask).filter(
                    ScheduledTask.crew_member_id.in_(loser_ids)
                ).delete(synchronize_session=False)
                for r in losers:
                    db.delete(r)
                db.commit()
                logger.warning(
                    "Default-assistant dedupe: owner=%r had %d rows, kept %s, "
                    "dropped %d crew + %d orphan tasks",
                    owner, n, keep.id, len(losers), n_tasks,
                )
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Could not dedupe default-assistant rows on startup: {e}")

    self._running = True
    self._task = asyncio.create_task(self._loop())
    # Internal background scanner that isn't a user-facing "task" — pure
    # infra (no LLM), shouldn't clutter the Tasks UI, fires on its own
    # cadence inside the scheduler process.
    #
    # Calendar event reminders are represented as Notes by the calendar UI,
    # so the Notes scanner is the single reminder dispatch path. Running the
    # old event scanner too caused duplicate emails/notifications for the
    # same calendar event.
    self._note_pings_task = asyncio.create_task(self._note_pings_loop())
    logger.info(f"Task scheduler started (concurrency cap: {self._concurrency_cap})")
    # Audit clusters: show any minute-of-day where >1 active scheduled
    # tasks land. Helps spot "all my tasks fire at 9am" patterns the user
    # may want to spread out.
    try:
        from core.database import SessionLocal, ScheduledTask
        db = SessionLocal()
        try:
            rows = db.query(ScheduledTask).filter(
                ScheduledTask.status == "active",
                ScheduledTask.trigger_type == "schedule",
                ScheduledTask.next_run.isnot(None),
            ).all()
            buckets: Dict[str, list] = {}
            for r in rows:
                if not r.next_run:
                    continue
                key = r.next_run.strftime("%H:%M")
                buckets.setdefault(key, []).append(r.name or r.id)
            clusters = {k: v for k, v in buckets.items() if len(v) > 1}
            if clusters:
                summary = ", ".join(f"{k} ({len(v)})" for k, v in sorted(clusters.items()))
                logger.info(f"Task scheduling clusters (>1 task/minute): {summary}")
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"Cluster audit skipped: {e}")

async def stop(self):
    self._running = False
    if self._task:
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
    for attr in ("_note_pings_task", "_event_pings_task"):
        t = getattr(self, attr, None)
        if t:
            t.cancel()
            try: await t
            except asyncio.CancelledError: pass
    logger.info("Task scheduler stopped")

async def _note_pings_loop(self):
    """Built-in note-due scanner — ticks every 60s inside the scheduler.
    Pure infra (no LLM), doesn't surface in the Tasks UI. Iterates
    per-owner so cache pruning in `action_ping_notes` (which removes
    cache entries for notes not in the current scan's seen_ids) doesn't
    cross-delete other users' entries (review C4).
    """
    await asyncio.sleep(30)
    from src.builtin_actions import action_ping_notes, TaskNoop
    while self._running:
        owners = self._known_task_owners()
        for ow in (owners or [""]):
            try:
                await action_ping_notes(owner=ow)
            except TaskNoop:
                pass
            except Exception as e:
                logger.warning(f"ping_notes background scanner errored for owner={ow!r}: {e}")
        await asyncio.sleep(60)  # 1 min

async def _event_pings_loop(self):
    """Built-in calendar-event scanner — same recipe as note pings. Runs
    every 10 min, fires reminders via dispatch_reminder. Not a user task.
    Iterates per-owner so each user only gets their own calendar pings
    (passing owner="" globally would email User B's events to User A's
    configured SMTP "from" address — see review C3).
    """
    await asyncio.sleep(90)
    from src.builtin_actions import action_ping_events, TaskNoop
    while self._running:
        owners = self._known_task_owners()
        for ow in (owners or [""]):
            try:
                await action_ping_events(owner=ow)
            except TaskNoop:
                pass
            except Exception as e:
                logger.warning(f"ping_events background scanner errored for owner={ow!r}: {e}")
        await asyncio.sleep(600)  # 10 min

def _known_task_owners(self) -> list:
    """Distinct non-empty owners that background scanners should visit.

    Scheduled tasks used to be the only owner source. Calendar reminders
    are stored as Notes, though, so an account with due notes but no task
    rows could get the browser reminder while the backend email/ntfy
    scanner never ran for that owner.
    """
    from core.database import SessionLocal, ScheduledTask, Note
    db = SessionLocal()
    try:
        owners = set()
        for r in db.query(ScheduledTask.owner).distinct().all():
            if r[0]:
                owners.add(r[0])
        note_q = db.query(Note.owner).filter(
            Note.due_date.isnot(None),
            Note.due_date != "",
            Note.archived == False,  # noqa: E712
        ).distinct()
        for r in note_q.all():
            if r[0]:
                owners.add(r[0])
        return sorted(owners)
    except Exception:
        return []
    finally:
        db.close()

async def _loop(self):
    await asyncio.sleep(10)
    while self._running:
        try:
            await self._check_due_tasks()
        except Exception:
            logger.exception("Error in task scheduler loop")
        # Sleep until the next scheduled run, capped at 60s. A `* * * * *`
        # cron task previously fired up to ~60s late because we always
        # slept the full minute; now the loop wakes near the boundary.
        sleep_for = 60.0
        try:
            from core.database import SessionLocal as _SL, ScheduledTask as _ST
            _db = _SL()
            try:
                next_run = _db.query(_ST.next_run).filter(
                    _ST.status == "active",
                    _ST.next_run.isnot(None),
                ).order_by(_ST.next_run.asc()).first()
                if next_run and next_run[0]:
                    delta = (next_run[0] - _utcnow()).total_seconds()
                    sleep_for = max(1.0, min(60.0, delta))
            finally:
                _db.close()
        except Exception:
            pass
        await asyncio.sleep(sleep_for)

