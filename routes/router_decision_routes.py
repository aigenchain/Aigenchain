"""Admin observability for router/orchestration decisions (Tahap 2).

Exposes the `router_decisions` table so orchestration behaviour can be analysed
without grepping logs:

  GET /api/admin/router-decisions            — recent rows (filterable)
  GET /api/admin/router-decisions/summary    — aggregate metrics
  GET /api/admin/router-decisions/timeseries — bucketed trend (hour|day)
  GET /api/admin/router-decisions/export     — raw rows as CSV or JSON download

All are admin-only. Read-only; no side effects.
"""

import csv
import io
import json
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from core.middleware import require_admin
from core.database import (
    get_router_decisions,
    get_router_decisions_summary,
    get_router_decisions_timeseries,
    iter_router_decisions_for_export,
    _ROUTER_DECISION_EXPORT_COLUMNS,
)

logger = logging.getLogger(__name__)


def _parse_since(hours: int = None, since: str = None):
    """Resolve an optional lower time bound (naive UTC) from query params."""
    if since:
        try:
            dt = datetime.fromisoformat(since)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except ValueError:
            raise HTTPException(400, f"Invalid 'since' timestamp: {since!r}")
    if hours and hours > 0:
        return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
    return None


def setup_router_decision_routes():
    router = APIRouter(prefix="/api/admin")

    @router.get("/router-decisions")
    def list_router_decisions(
        request: Request,
        limit: int = 100,
        session_id: str = None,
        category: str = None,
        knowledge_source: str = None,
        hours: int = None,
        since: str = None,
    ):
        require_admin(request)
        _since = _parse_since(hours=hours, since=since)
        rows = get_router_decisions(
            limit=limit,
            session_id=(session_id or None),
            category=(category or None),
            knowledge_source=(knowledge_source or None),
            since=_since,
        )
        return {"count": len(rows), "decisions": rows}

    @router.get("/router-decisions/summary")
    def router_decisions_summary(
        request: Request,
        hours: int = None,
        since: str = None,
    ):
        require_admin(request)
        _since = _parse_since(hours=hours, since=since)
        return get_router_decisions_summary(since=_since)

    @router.get("/router-decisions/timeseries")
    def router_decisions_timeseries(
        request: Request,
        bucket: str = "hour",
        hours: int = None,
        since: str = None,
        until: str = None,
    ):
        require_admin(request)
        if bucket not in ("hour", "day"):
            raise HTTPException(400, "bucket must be 'hour' or 'day'")
        _since = _parse_since(hours=hours, since=since)
        _until = _parse_since(since=until) if until else None
        return get_router_decisions_timeseries(
            bucket=bucket, since=_since, until=_until,
        )

    @router.get("/router-decisions/export")
    def router_decisions_export(
        request: Request,
        format: str = "csv",
        hours: int = None,
        since: str = None,
        until: str = None,
        limit: int = 10000,
    ):
        require_admin(request)
        fmt = (format or "csv").lower()
        if fmt not in ("csv", "json"):
            raise HTTPException(400, "format must be 'csv' or 'json'")
        _since = _parse_since(hours=hours, since=since)
        _until = _parse_since(since=until) if until else None
        rows = iter_router_decisions_for_export(
            since=_since, until=_until, limit=limit,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if fmt == "json":
            payload = json.dumps(
                {"count": len(rows), "decisions": rows}, ensure_ascii=False,
            )
            return StreamingResponse(
                iter([payload]),
                media_type="application/json",
                headers={
                    "Content-Disposition":
                        f'attachment; filename="router_decisions_{stamp}.json"',
                },
            )
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf, fieldnames=list(_ROUTER_DECISION_EXPORT_COLUMNS),
            extrasaction="ignore",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    f'attachment; filename="router_decisions_{stamp}.csv"',
            },
        )

    return router
