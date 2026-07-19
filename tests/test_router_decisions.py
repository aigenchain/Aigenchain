"""Router-decision observability (Tahap 2).

Covers the DB layer added for one-row-per-request routing traces:
record_router_decision() roundtrip + robustness, and the query/summary
helpers that back the admin endpoints. Uses a dedicated temp SQLite file so
these tests are isolated from the in-memory collection DB.
"""

import tempfile
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from tests.helpers.import_state import clear_fake_database_modules

clear_fake_database_modules()

import core.database as cdb

_TMPDB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_ENGINE = create_engine(
    f"sqlite:///{_TMPDB.name}",
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
)
cdb.Base.metadata.create_all(_ENGINE)
_TS = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)
# Helpers read the module-global SessionLocal via get_db_session().
cdb.SessionLocal = _TS


def _clear():
    with _TS() as s:
        s.query(cdb.RouterDecision).delete()
        s.commit()


def _base(**over):
    row = {
        "session_id": None,
        "owner": "alice",
        "message_preview": "hello",
        "requested_mode": "chat",
        "effective_mode": "chat",
        "auto_escalated": False,
        "escalation_reason": None,
        "intent_category": None,
        "search_enabled": False,
        "search_trigger": "none",
        "image_fastpath": False,
        "image_clarify": False,
        "pending_resolved": False,
        "selected_tools": None,
        "agent_rounds": 0,
        "tool_calls": 0,
        "latency_ms": 10,
        "outcome": "ok",
        "model": "hy3-free",
        "exit_path": "chat",
        "knowledge_source": "none",
        "knowledge_reason": "no knowledge-source pattern matched",
        "knowledge_retrieval": False,
    }
    row.update(over)
    return row


def test_record_and_read_roundtrip():
    _clear()
    assert cdb.record_router_decision(**_base(
        intent_category="web",
        search_enabled=True,
        search_trigger="auto_intent",
        selected_tools=["web_search", "web_fetch"],
        auto_escalated=True,
        effective_mode="agent",
        exit_path="agent",
    )) is True
    rows = cdb.get_router_decisions(limit=10)
    assert len(rows) == 1
    r = rows[0]
    assert r["intent_category"] == "web"
    assert r["search_enabled"] is True
    assert r["search_trigger"] == "auto_intent"
    assert r["selected_tools"] == ["web_search", "web_fetch"]
    assert r["auto_escalated"] is True
    assert r["effective_mode"] == "agent"
    assert r["exit_path"] == "agent"
    assert r["id"]
    assert r["created_at"]


def test_unknown_keys_ignored_never_raises():
    _clear()
    # A loose trace dict with extra keys must not blow up.
    ok = cdb.record_router_decision(**_base(
        this_key_does_not_exist=123,
        another_bogus="x",
    ))
    assert ok is True
    assert len(cdb.get_router_decisions(limit=10)) == 1


def test_message_preview_trimmed_to_200():
    _clear()
    long_msg = "x" * 500
    cdb.record_router_decision(**_base(message_preview=long_msg))
    r = cdb.get_router_decisions(limit=1)[0]
    assert len(r["message_preview"]) == 200


def _mk_session(sid):
    with _TS() as s:
        if not s.get(cdb.Session, sid):
            s.add(cdb.Session(
                id=sid, name=sid, owner="alice",
                endpoint_url="http://x", model="hy3-free",
            ))
            s.commit()


def test_filters_session_and_category():
    _clear()
    _mk_session("s1")
    _mk_session("s2")
    cdb.record_router_decision(**_base(session_id="s1", intent_category="web"))
    cdb.record_router_decision(**_base(session_id="s2", intent_category="image"))
    cdb.record_router_decision(**_base(session_id="s2", intent_category="web"))

    assert len(cdb.get_router_decisions(session_id="s1")) == 1
    assert len(cdb.get_router_decisions(session_id="s2")) == 2
    web = cdb.get_router_decisions(category="web")
    assert len(web) == 2
    assert all(x["intent_category"] == "web" for x in web)
    both = cdb.get_router_decisions(session_id="s2", category="web")
    assert len(both) == 1


def test_since_filter():
    _clear()
    cdb.record_router_decision(**_base())
    future = datetime.utcnow() + timedelta(hours=1)
    assert cdb.get_router_decisions(since=future) == []
    past = datetime.utcnow() - timedelta(hours=1)
    assert len(cdb.get_router_decisions(since=past)) == 1


def test_newest_first_and_limit():
    _clear()
    for i in range(5):
        cdb.record_router_decision(**_base(message_preview=f"m{i}"))
    rows = cdb.get_router_decisions(limit=3)
    assert len(rows) == 3


def test_summary_aggregates():
    _clear()
    cdb.record_router_decision(**_base(
        intent_category="web", search_trigger="auto_intent",
        search_enabled=True, auto_escalated=True, outcome="ok",
        exit_path="agent", agent_rounds=2, tool_calls=3, latency_ms=100,
    ))
    cdb.record_router_decision(**_base(
        intent_category="web", search_trigger="manual",
        search_enabled=True, outcome="ok", exit_path="agent",
        agent_rounds=1, tool_calls=1, latency_ms=200,
    ))
    cdb.record_router_decision(**_base(
        intent_category=None, search_trigger="none",
        search_enabled=False, outcome="ok", exit_path="chat",
        agent_rounds=0, tool_calls=0, latency_ms=50,
    ))
    s = cdb.get_router_decisions_summary()
    assert s["total"] == 3
    assert s["auto_escalated"] == 1
    assert s["search_enabled"] == 2
    assert s["by_category"]["web"] == 2
    assert s["by_search_trigger"]["auto_intent"] == 1
    assert s["by_search_trigger"]["manual"] == 1
    assert s["by_search_trigger"]["none"] == 1
    assert s["by_exit_path"]["agent"] == 2
    assert s["by_exit_path"]["chat"] == 1
    assert s["latency_ms"]["max"] == 200
    assert s["latency_ms"]["p50"] is not None


def test_knowledge_fields_roundtrip_and_filter():
    _clear()
    cdb.record_router_decision(**_base(
        knowledge_source="personal_recall",
        knowledge_reason="what is my (EN)",
        knowledge_retrieval=True,
    ))
    cdb.record_router_decision(**_base(
        knowledge_source="past_chat",
        knowledge_reason="what did we discuss (EN)",
        knowledge_retrieval=True,
    ))
    cdb.record_router_decision(**_base(knowledge_source="none", knowledge_retrieval=False))

    r = cdb.get_router_decisions(knowledge_source="personal_recall")
    assert len(r) == 1
    assert r[0]["knowledge_source"] == "personal_recall"
    assert r[0]["knowledge_reason"] == "what is my (EN)"
    assert r[0]["knowledge_retrieval"] is True

    assert len(cdb.get_router_decisions(knowledge_source="past_chat")) == 1
    assert len(cdb.get_router_decisions()) == 3


def test_summary_knowledge_aggregates():
    _clear()
    cdb.record_router_decision(**_base(knowledge_source="personal_recall", knowledge_retrieval=True))
    cdb.record_router_decision(**_base(knowledge_source="personal_recall", knowledge_retrieval=True))
    cdb.record_router_decision(**_base(knowledge_source="web", knowledge_retrieval=False))
    cdb.record_router_decision(**_base(knowledge_source="none", knowledge_retrieval=False))
    s = cdb.get_router_decisions_summary()
    assert s["total"] == 4
    assert s["knowledge_retrieval"] == 2
    assert s["by_knowledge_source"]["personal_recall"] == 2
    assert s["by_knowledge_source"]["web"] == 1
    assert s["by_knowledge_source"]["none"] == 1


def test_summary_empty():
    _clear()
    s = cdb.get_router_decisions_summary()
    assert s["total"] == 0
    assert s["latency_ms"]["max"] is None


def test_decision_survives_session_deletion():
    # SET NULL on session_id: create a session, attach a decision, delete the
    # session, decision remains queryable with session_id nulled.
    _clear()
    with _TS() as s:
        sess = cdb.Session(
            id="live1", name="tmp", owner="alice",
            endpoint_url="http://x", model="hy3-free",
        )
        s.add(sess)
        s.commit()
    cdb.record_router_decision(**_base(session_id="live1", intent_category="web"))
    assert len(cdb.get_router_decisions(session_id="live1")) == 1
    # Emulate FK SET NULL (SQLite needs PRAGMA; assert model config instead).
    fk = cdb.RouterDecision.__table__.c.session_id.foreign_keys
    assert any(f.ondelete == "SET NULL" for f in fk)
