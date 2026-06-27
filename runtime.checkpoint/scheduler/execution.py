"""
Scheduler execution.

Extracted from src.task_scheduler
"""


async def _check_due_tasks(self):
    from core.database import SessionLocal, ScheduledTask
    db = SessionLocal()
    try:
        now = _utcnow()
        async with self._executing_lock:
            # Snapshot under the lock so we don't race with mid-iteration adds.
            executing_snapshot = set(self._executing)
            # Scheduled tasks and deferred event tasks both use next_run.
            due = db.query(ScheduledTask).filter(
                ScheduledTask.status == "active",
                ScheduledTask.next_run <= now,
                ScheduledTask.id.notin_(executing_snapshot) if executing_snapshot else True,
            ).all()
            to_dispatch = []
            for task in due:
                if task.id in self._executing:
                    continue
                self._executing.add(task.id)
                to_dispatch.append(task.id)
        for task_id in to_dispatch:
            asyncio.create_task(self._execute_task(task_id))
    finally:
        db.close()

async def _execute_task(self, task_id: str, *, bypass_model_slot: bool = False, release_executing: bool = True):
    # Create the run record with status="queued" BEFORE waiting on the
    # semaphore so the UI can show that a manually-triggered task is in
    # line behind another. Once we acquire the slot, flip to "running"
    # and hand off to _execute_task_locked.
    from core.database import SessionLocal, TaskRun
    current = asyncio.current_task()
    if current:
        self._task_handles[task_id] = current
    run_id = str(uuid.uuid4())
    _q_db = SessionLocal()
    try:
        run = TaskRun(
            id=run_id,
            task_id=task_id,
            started_at=_utcnow(),
            status="queued",
            result="Queued — waiting for a free slot…",
        )
        _q_db.add(run)
        _q_db.commit()
    except Exception:
        logger.exception(f"Failed to create queued run row for task {task_id}")
    finally:
        _q_db.close()

    try:
        if bypass_model_slot or not self._task_needs_model_slot(task_id):
            await self._execute_task_locked(task_id, run_id, release_executing=release_executing)
            return

        async with self._run_semaphore:
            await self._execute_task_locked(task_id, run_id, release_executing=release_executing)
    except asyncio.CancelledError:
        # If cancellation happens while queued behind the semaphore,
        # _execute_task_locked never runs and cannot update the Activity row.
        self._mark_run_aborted(task_id, run_id)
        raise
    finally:
        handle = self._task_handles.get(task_id)
        if handle is current:
            self._task_handles.pop(task_id, None)
        if release_executing:
            async with self._executing_lock:
                self._executing.discard(task_id)

async def _execute_task_locked(self, task_id: str, run_id: str, *, release_executing: bool = True):
    from core.database import SessionLocal, ScheduledTask, TaskRun

    db = SessionLocal()
    try:
        task = db.query(ScheduledTask).filter(ScheduledTask.id == task_id).first()
        if not task or task.status != "active":
            # Task was paused/deleted while queued — record that outcome
            # so the run row doesn't sit as "queued" forever.
            stale = db.query(TaskRun).filter(TaskRun.id == run_id).first()
            if stale and stale.status == "queued":
                stale.status = "skipped"
                stale.finished_at = _utcnow()
                stale.error = f"Task no longer active (status={task.status if task else 'deleted'})"
                db.commit()
            return

        # Flip the run from queued → running. Reset started_at to the
        # actual execution start so queue wait time is visible from
        # created_at vs started_at if we ever surface that.
        run = db.query(TaskRun).filter(TaskRun.id == run_id).first()
        if run:
            run.status = "running"
            run.started_at = _utcnow()
            run.result = "Starting…"
            db.commit()
        else:
            # Defensive: row may have been wiped; recreate so the rest of
            # the code can look it up by run_id without crashing.
            run = TaskRun(
                id=run_id,
                task_id=task.id,
                started_at=_utcnow(),
                status="running",
                result="Starting…",
            )
            db.add(run)
            db.commit()

        task_type = task.task_type or "llm"

        from src.builtin_actions import TaskDeferred, TaskNoop

        # Cleared each run so an action task (no model) doesn't inherit a
        # previous llm/research run's model. The executors set it once the
        # model is resolved.
        self._last_run_model = None
        try:
            if task_type == "action":
                result, success = await self._execute_action(task, run_id=run_id)
                run.status = "success" if success else "error"
                run.result = result
                if not success:
                    run.error = result
            elif task_type == "research":
                result = await self._execute_research_task(task, db)
                run.status = "success"
                run.result = result
            else:
                # LLM task — use agent loop for tool access
                result = await self._execute_llm_task(task, db)
                run.status = "success"
                run.result = result
            # Record which model actually ran (resolved inside the executor).
            if getattr(self, "_last_run_model", None):
                run.model = self._last_run_model
            if run.status == "success":
                await self._deliver_task_result(task, result, db, model=getattr(self, "_last_run_model", None))
        except TaskDeferred as defer:
            count = self._task_defer_counts.get(task_id, 0) + 1
            self._task_defer_counts[task_id] = count
            delay_seconds = int(getattr(defer, "delay_seconds", 20 * 60) or (20 * 60))
            if count > 2:
                delay_seconds = max(delay_seconds, 40 * 60)
            when = _utcnow() + timedelta(seconds=delay_seconds)
            logger.info(
                "Task '%s' deferred for %ss after %s quiet-window hit(s): %s",
                task.name, delay_seconds, count, defer,
            )
            run_obj = db.query(TaskRun).filter(TaskRun.id == run_id).first()
            if run_obj:
                db.delete(run_obj)
            task.next_run = when
            db.commit()
            return
        except asyncio.CancelledError:
            logger.info("Task '%s' stopped by user", task.name)
            run_obj = db.query(TaskRun).filter(TaskRun.id == run_id).first()
            if run_obj:
                run_obj.status = "aborted"
                run_obj.error = "Stopped by user"
                run_obj.result = run_obj.result or "Stopped by user"
                run_obj.finished_at = _utcnow()
            task.last_run = _utcnow()
            if (task.trigger_type or "schedule") == "schedule":
                task.next_run = compute_next_run(
                    task.schedule, task.scheduled_time,
                    task.scheduled_day, task.scheduled_date,
                    after=_utcnow(),
                    cron_expression=task.cron_expression,
                    tz_name=_resolve_task_timezone(db, task),
                )
            else:
                task.next_run = None
            db.commit()
            return
        except TaskNoop as noop:
            # Action reported "nothing to do". Mark the run as `skipped`
            # with the reason in `result` so it surfaces in Activity as a
            # slim "skipped — <reason>" row instead of vanishing silently.
            # (Previous behavior was `db.delete(run)`, which made the user
            # think queued tasks had been dropped on the floor.)
            logger.info(f"Task '{task.name}' no-op: {noop}")
            run.status = "skipped"
            run.result = str(noop)
            run.finished_at = _utcnow()
            task.last_run = _utcnow()
            if (task.trigger_type or "schedule") == "schedule":
                task.next_run = compute_next_run(
                    task.schedule, task.scheduled_time,
                    task.scheduled_day, task.scheduled_date,
                    after=_utcnow(),
                    cron_expression=task.cron_expression,
                    tz_name=_resolve_task_timezone(db, task),
                )
            else:
                task.next_run = None
            db.commit()
            return

        run.finished_at = _utcnow()

        # Update task
        task.last_run = _utcnow()
        task.run_count = (task.run_count or 0) + 1
        self._task_defer_counts.pop(task_id, None)

        # Compute next run only for schedule-triggered tasks
        if (task.trigger_type or "schedule") == "schedule":
            task.next_run = compute_next_run(
                task.schedule, task.scheduled_time,
                task.scheduled_day, task.scheduled_date,
                after=_utcnow(),
                cron_expression=task.cron_expression,
                tz_name=_resolve_task_timezone(db, task),
            )
            if task.next_run is None and task.schedule == "once":
                task.status = "completed"
        else:
            task.next_run = None

        db.commit()
        logger.info(f"Task '{task.name}' completed (run {run_id})")
        output = task.output_target or "session"
        # Per-task notification gate. Default True (notifications_enabled
        # defaults to True at column level), but skip when the user has
        # explicitly turned them off for this task — quiets chatty
        # housekeeping cron tasks without disabling them entirely.
        should_notify = (
            (task.task_type or "llm") in {"llm", "research"}
            and getattr(task, "notifications_enabled", True)
        )
        if should_notify:
            self.add_notification(
                task.name,
                run.status,
                task_id,
                owner=task.owner,
                body=run.result if output == "notification" else None,
            )

        # Log result to the assistant chat so all task activity is visible.
        # Skip skipped/error rows — user shouldn't see "skipped: …" noise
        # for cron tasks that no-op'd, or duplicate error spam for tasks
        # that already fired an error notification above.
        if run.status == "success":
            self._log_to_assistant(db, task, run.result or "[success]")

        # Task chaining — trigger the next task on success
        if run.status == "success" and task.then_task_id:
            chain_id = task.then_task_id
            chain_task = db.query(ScheduledTask).filter(ScheduledTask.id == chain_id).first()
            if not chain_task or chain_task.owner != task.owner:
                logger.warning(
                    "Skipping chain from %r: target task %s is missing or not owned by %r",
                    task.name, chain_id, task.owner,
                )
            elif not self._has_chain_cycle(db, chain_id, owner=task.owner):
                logger.info(f"Chaining: '{task.name}' → task {chain_id}")
                asyncio.create_task(self._run_chained(chain_id))
            else:
                logger.warning(f"Skipping chain from '{task.name}': cycle detected")

    except Exception as exec_exc:
        logger.exception(f"Task {task_id} execution error")
        # Fetch the task's owner so the error notification reaches
        # the same user the success notification would have.
        _owner = None
        try:
            _t = db.query(ScheduledTask).filter(ScheduledTask.id == task_id).first()
            _owner = _t.owner if _t else None
        except Exception:
            pass
        _should_notify_error = False
        try:
            _t_for_notify = db.query(ScheduledTask).filter(ScheduledTask.id == task_id).first()
            _should_notify_error = (
                bool(_t_for_notify)
                and (_t_for_notify.task_type or "llm") in {"llm", "research"}
                and getattr(_t_for_notify, "notifications_enabled", True)
            )
        except Exception:
            _should_notify_error = False
        if _should_notify_error:
            self.add_notification(f"Task {task_id}", "error", task_id, owner=_owner)
        try:
            # Persist the actual exception message so the UI can show it
            err_text = f"{type(exec_exc).__name__}: {exec_exc}"
            run_obj = db.query(TaskRun).filter(TaskRun.id == run_id).first()
            if run_obj and run_obj.status in ("running", "success"):
                run_obj.status = "error"
                run_obj.error = err_text[:2000]
                run_obj.finished_at = _utcnow()
            # Advance next_run even on failure so a broken task doesn't
            # busy-loop the scheduler every tick with a stale past date.
            task_obj = db.query(ScheduledTask).filter(ScheduledTask.id == task_id).first()
            if task_obj and (task_obj.trigger_type or "schedule") == "schedule":
                task_obj.last_run = _utcnow()
                try:
                    task_obj.next_run = compute_next_run(
                        task_obj.schedule, task_obj.scheduled_time,
                        task_obj.scheduled_day, task_obj.scheduled_date,
                        after=_utcnow(),
                        cron_expression=task_obj.cron_expression,
                        tz_name=_resolve_task_timezone(db, task_obj),
                    )
                except Exception:
                    pass
            try:
                db.commit()
            except Exception as commit_err:
                # Commit failed — without a fallback the run row stays
                # "running" forever AND next_run stays in the past, so the
                # scheduler busy-loops dispatching the same task every tick
                # until restart. Force the recovery in a fresh session.
                logger.warning("Task %s error-path commit failed: %s — falling back", task_id, commit_err)
                try:
                    db.rollback()
                except Exception:
                    pass
                from datetime import timedelta as _td
                _recover_db = SessionLocal()
                try:
                    _r = _recover_db.query(TaskRun).filter(TaskRun.id == run_id).first()
                    if _r and _r.status in ("running", "queued"):
                        _r.status = "aborted"
                        _r.error = f"commit_failed: {type(commit_err).__name__}: {commit_err}"[:2000]
                        _r.finished_at = _utcnow()
                    _t = _recover_db.query(ScheduledTask).filter(ScheduledTask.id == task_id).first()
                    if _t and (_t.trigger_type or "schedule") == "schedule":
                        # Push next_run forward 5min as a safe stall so the
                        # scheduler doesn't immediately re-dispatch.
                        _t.next_run = _utcnow() + _td(minutes=5)
                        _t.last_run = _utcnow()
                    _recover_db.commit()
                except Exception as recover_err:
                    logger.error("Task %s recovery commit ALSO failed: %s", task_id, recover_err)
                finally:
                    _recover_db.close()
        except Exception:
            logger.exception("Task %s error-path failed unexpectedly", task_id)
    finally:
        db.close()
        handle = self._task_handles.get(task_id)
        if handle is asyncio.current_task():
            self._task_handles.pop(task_id, None)
        if release_executing:
            async with self._executing_lock:
                self._executing.discard(task_id)



# Built-in housekeeping actions whose output is pure infra (no user-facing
# content) — don't pollute the assistant chat session with their summaries.
# Activity log + reminder email already carry everything the user needs.
_SILENT_ACTIONS = frozenset({
    "check_email_urgency",
    "learn_sender_signatures",
    "summarize_emails",
    "draft_email_replies",
    "extract_email_events",
    "classify_events",
    "tidy_sessions",
    "tidy_documents",
    "consolidate_memory",
    "tidy_research",
    "test_skills",
    "audit_skills",
})

_MODEL_BACKED_ACTIONS = frozenset({
    "summarize_emails",
    "draft_email_replies",
    "extract_email_events",
    "classify_events",
    "learn_sender_signatures",
    "check_email_urgency",
    "test_skills",
    "audit_skills",
    "consolidate_memory",
})

