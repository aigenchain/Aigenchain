"""Fase C2: anaphoric contextual web follow-up detection.

`_is_contextual_web_followup` must inherit the previous turn's web intent for
short anaphoric follow-ups ("and what about in Europe?", "kalau yang di Amerika
bagaimana?") when the recent session text shows a web lookup — closing the two
contextual false-negatives left after Fase B. The behavior is gated by the
``router_contextual_web_followup`` setting (default on).
"""

import pytest

from routes.chat_routes import (
    _ANAPHORIC_FOLLOWUP_RE,
    _RECENT_WEB_CONTEXT_RE,
    _WEB_FOLLOWUP_RE,
    _is_contextual_web_followup,
)


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeSession:
    def __init__(self, texts):
        self.history = [_FakeMsg(t) for t in texts]


# ── regex-level guards ────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "and what about in Europe?",
        "what about in Japan?",
        "how about in Germany",
        "and in France?",
        "kalau yang di Amerika bagaimana?",
        "kalau di Bandung gimana?",
        "bagaimana dengan Jakarta?",
    ],
)
def test_anaphoric_re_matches_topic_followups(text):
    assert _ANAPHORIC_FOLLOWUP_RE.search(text)


@pytest.mark.parametrize(
    "text",
    [
        "write me a poem about Europe",
        "explain how photosynthesis works",
        "apa ibu kota Indonesia",
    ],
)
def test_anaphoric_re_ignores_non_followups(text):
    assert not _ANAPHORIC_FOLLOWUP_RE.search(text)


def test_recent_web_context_re_matches_indonesian_freshness():
    assert _RECENT_WEB_CONTEXT_RE.search("berita terbaru soal harga emas")
    assert _RECENT_WEB_CONTEXT_RE.search("cuaca sekarang")


# ── the two Fase B contextual false-negatives ─────────────────────


def test_id_anaphoric_followup_inherits_web(monkeypatch):
    import src.settings as settings

    monkeypatch.setattr(settings, "get_setting", lambda k, d=None: True)
    sess = _FakeSession(["berita terbaru soal harga emas"])
    assert _is_contextual_web_followup("kalau yang di Amerika bagaimana?", sess)


def test_en_anaphoric_followup_inherits_web(monkeypatch):
    import src.settings as settings

    monkeypatch.setattr(settings, "get_setting", lambda k, d=None: True)
    sess = _FakeSession(["what is the latest news on AI regulation"])
    assert _is_contextual_web_followup("and what about in Europe?", sess)


# ── gating + guards ───────────────────────────────────────────────


def test_anaphoric_followup_disabled_by_setting(monkeypatch):
    import src.settings as settings

    monkeypatch.setattr(settings, "get_setting", lambda k, d=None: False)
    sess = _FakeSession(["berita terbaru soal harga emas"])
    assert not _is_contextual_web_followup("kalau yang di Amerika bagaimana?", sess)


def test_anaphoric_followup_requires_web_context(monkeypatch):
    import src.settings as settings

    monkeypatch.setattr(settings, "get_setting", lambda k, d=None: True)
    sess = _FakeSession(["ceritakan tentang sejarah Romawi kuno"])
    assert not _is_contextual_web_followup("kalau yang di Amerika bagaimana?", sess)


def test_bare_retry_followup_not_gated_by_setting(monkeypatch):
    import src.settings as settings

    monkeypatch.setattr(settings, "get_setting", lambda k, d=None: False)
    sess = _FakeSession(["what's the weather now"])
    assert _is_contextual_web_followup("search now", sess)


def test_non_followup_message_ignored(monkeypatch):
    import src.settings as settings

    monkeypatch.setattr(settings, "get_setting", lambda k, d=None: True)
    sess = _FakeSession(["what's the weather now"])
    assert not _is_contextual_web_followup("write me a poem", sess)
