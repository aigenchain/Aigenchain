"""Admin observability for router/orchestration decisions (Tahap 2).

Exposes the `router_decisions` table so orchestration behaviour can be analysed
without grepping logs:

  GET /api/admin/router-decisions          — recent rows (filterable)
  GET /api/admin/router-decisions/summary  — aggregate metrics

Both are admin-only. Read-only; no side effects.
"""

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request

from core.middleware import require_admin
from core.database import get_router_decisions, get_router_decisions_summary

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

    return router
