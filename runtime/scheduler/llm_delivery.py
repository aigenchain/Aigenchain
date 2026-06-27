from __future__ import annotations

from typing import Any


async def _execute_llm_task(self, task, db) -> str:
    raise NotImplementedError


async def _run_agent_loop(
    self,
    endpoint_url: str,
    model: str,
    task,
    session_id: str,
    run_id: str | None = None,
):
    raise NotImplementedError


async def _execute_research_task(self, task, db) -> str:
    raise NotImplementedError
