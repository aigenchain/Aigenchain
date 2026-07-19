"""Fase C1: past_chat knowledge-source retrieval injection.

`_maybe_past_chat_context` returns an untrusted-context message with snippets
from the user's OTHER sessions when the turn routes to ``past_chat`` and the
``router_past_chat_retrieval`` setting is on. It excludes the current session,
respects the snippet limit, and is a no-op when disabled / non-past_chat.
"""

from dataclasses import dataclass

import pytest

from routes.chat_helpers import _maybe_past_chat_context


@dataclass
class _FakeResult:
    session_id: str
    session_name: str
    role: str
    content_snippet: str


def _patch(monkeypatch, *, enabled=True, limit=3, results=None, route_source="past_chat"):
    import src.settings as settings
    import src.action_intents as ai
    import src.session_search as ss

    def _get_setting(key, default=None):
        if key == "router_past_chat_retrieval":
            return enabled
        if key == "router_past_chat_limit":
            return limit
        return default

    monkeypatch.setattr(settings, "get_setting", _get_setting)

    class _Route:
        source = route_source
        needs_retrieval = route_source in ("personal_recall", "past_chat", "personal_docs")

    monkeypatch.setattr(ai, "resolve_knowledge_route", lambda text: _Route())
    monkeypatch.setattr(
        ss, "search_session_messages",
        lambda *a, **k: list(results or []),
    )


PAST_MSG = "what did we discuss about the migration yesterday?"


def test_disabled_setting_returns_none(monkeypatch):
    _patch(monkeypatch, enabled=False, results=[_FakeResult("s2", "Migration", "user", "we agreed on blue-green")])
    assert _maybe_past_chat_context(PAST_MSG, "alice", "s1") is None


def test_non_past_chat_route_returns_none(monkeypatch):
    _patch(monkeypatch, route_source="none", results=[_FakeResult("s2", "Migration", "user", "hi")])
    assert _maybe_past_chat_context(PAST_MSG, "alice", "s1") is None


def test_injects_snippets_from_other_sessions(monkeypatch):
    _patch(monkeypatch, results=[
        _FakeResult("s2", "Migration", "user", "we agreed on blue-green deploys"),
        _FakeResult("s3", "Planning", "assistant", "the cutover is Friday"),
    ])
    msg = _maybe_past_chat_context(PAST_MSG, "alice", "s1")
    assert msg is not None
    content = msg["content"]
    assert "blue-green deploys" in content
    assert "cutover is Friday" in content
    assert "[Migration]" in content


def test_excludes_current_session(monkeypatch):
    _patch(monkeypatch, results=[
        _FakeResult("s1", "Current", "user", "SHOULD_NOT_APPEAR"),
        _FakeResult("s2", "Other", "assistant", "keep me"),
    ])
    msg = _maybe_past_chat_context(PAST_MSG, "alice", "s1")
    assert msg is not None
    assert "SHOULD_NOT_APPEAR" not in msg["content"]
    assert "keep me" in msg["content"]


def test_respects_limit(monkeypatch):
    results = [_FakeResult(f"s{i}", f"S{i}", "user", f"snippet {i}") for i in range(2, 12)]
    _patch(monkeypatch, limit=2, results=results)
    msg = _maybe_past_chat_context(PAST_MSG, "alice", "s1")
    assert msg is not None
    bullets = [ln for ln in msg["content"].splitlines() if ln.startswith("- [")]
    assert len(bullets) == 2


def test_no_usable_snippets_returns_none(monkeypatch):
    _patch(monkeypatch, results=[_FakeResult("s1", "Current", "user", "only current session")])
    assert _maybe_past_chat_context(PAST_MSG, "alice", "s1") is None


def test_empty_message_returns_none(monkeypatch):
    _patch(monkeypatch, results=[_FakeResult("s2", "x", "user", "y")])
    assert _maybe_past_chat_context("   ", "alice", "s1") is None
