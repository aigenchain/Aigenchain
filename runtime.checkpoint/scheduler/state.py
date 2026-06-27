"""
Scheduler state helpers.

Extracted from src.task_scheduler
"""




def _set_run_progress(self, run_id: str, message: str):
    """Persist short live progress text for Activity while a run is active."""
    if not run_id:
        return
    try:
        from core.database import SessionLocal, TaskRun
        db = SessionLocal()
        try:
            run = db.query(TaskRun).filter(TaskRun.id == run_id).first()
            if run and run.status in ("queued", "running"):
                run.result = (message or "")[:4000]
                db.commit()
        finally:
            db.close()
    except Exception:
        logger.debug("Task progress update failed", exc_info=True)

def _mark_run_aborted(self, task_id: str, run_id: str | None = None, message: str = "Stopped by user") -> bool:
    """Mark an active run as aborted. Used by stop/cancel paths."""
    try:
        from core.database import SessionLocal, TaskRun
        db = SessionLocal()
        try:
            q = db.query(TaskRun)
            if run_id:
                q = q.filter(TaskRun.id == run_id)
            else:
                q = q.filter(
                    TaskRun.task_id == task_id,
                    TaskRun.status.in_(("queued", "running")),
                ).order_by(TaskRun.started_at.desc())
            run = q.first()
            if not run or run.status not in ("queued", "running"):
                return False
            run.status = "aborted"
            run.error = message
            run.result = run.result or message
            run.finished_at = _utcnow()
            db.commit()
            return True
        finally:
            db.close()
    except Exception:
        logger.debug("Task abort marker failed for %s", task_id, exc_info=True)
        return False

def add_notification(self, task_name: str, status: str, task_id: str = None, owner: str = None, body: str = None):
    """Store a notification about a completed task run. Tagged with the
    task's owner so `pop_notifications` can return only that user's
    notifications and prevent cross-tenant drain. `body` is the result
    text — populated when output_target='notification' so the client can
    show a rich browser Notification, not just a toast."""
    self._pending_notifications.append({
        "task_name": task_name,
        "status": status,
        "task_id": task_id,
        "owner": owner,
        "body": (body[:500] + "…") if body and len(body) > 500 else body,
        "timestamp": _utcnow().isoformat() + "Z",
    })
    # Cap at 50 to avoid unbounded growth
    if len(self._pending_notifications) > 50:
        self._pending_notifications = self._pending_notifications[-50:]

def pop_notifications(self, owner: str = None) -> list:
    """Return and clear pending notifications.

    When `owner` is set, only matching notifications are returned (and
    cleared). Notifications stored before owner-tagging existed (or
    from owner-less tasks) are included when the caller is anonymous
    or when no owner filter is given — preserves backward behaviour
    for the legacy single-user deploy.
    """
    if owner is None:
        notes = self._pending_notifications[:]
        self._pending_notifications.clear()
        return notes
    # Strict owner scope — used to OR-in null-owner notifications for
    # "legacy single-user" compat but that leaked notification bodies to
    # any authenticated user once a second account existed.
    keep, take = [], []
    for n in self._pending_notifications:
        if n.get("owner") == owner:
            take.append(n)
        else:
            keep.append(n)
    self._pending_notifications = keep
    return take

