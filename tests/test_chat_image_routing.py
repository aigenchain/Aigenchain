import sys
for mod_name in ["src.endpoint_resolver", "src.database", "core.database"]:
    _mod = sys.modules.get(mod_name)
    if _mod is not None and not getattr(_mod, "__file__", None):
        sys.modules.pop(mod_name, None)

from types import SimpleNamespace

from tests.helpers.import_state import clear_fake_endpoint_resolver_modules

clear_fake_endpoint_resolver_modules("routes.chat_routes")

from routes import chat_routes
from src.action_intents import classify_tool_intent


def _session(model="qwen3.5:latest", endpoint_url="http://localhost:11434/v1/chat/completions"):
    return SimpleNamespace(model=model, endpoint_url=endpoint_url, is_image=True)


def test_image_generation_is_never_a_sticky_session_mode(monkeypatch):
    # Images are produced only when the LLM calls the generate_image tool. The
    # legacy per-session routing (image model prefixes, endpoint matching, the
    # sticky is_image flag) must no longer force a session into image mode.
    def fail_if_called():
        raise AssertionError("image routing must not touch the DB anymore")

    monkeypatch.setattr(chat_routes, "SessionLocal", fail_if_called)

    assert not chat_routes._is_image_generation_session(_session(model="dall-e-3"))
    assert not chat_routes._is_image_generation_session(_session(model="sdxl-local"))
    assert not chat_routes._is_image_generation_session(_session())


def test_explicit_image_requests_are_detected_as_image_intent():
    for text in (
        "tolong buatkan gambar kucing astronot",
        "gambarkan proses fotosintesis",
        "generate an image of a sunset",
        "draw me a robot",
        "bikin ilustrasi naga",
        "create a poster for my event",
    ):
        intent = classify_tool_intent(text)
        assert intent.needs_tools and intent.category == "image", text


def test_plain_chat_and_questions_do_not_trigger_image_intent():
    for text in (
        "apa itu fotosintesis?",
        "apa itu image generation?",
        "how does image generation work?",
        "jelaskan cara membuat gambar dengan AI",
        "halo apa kabar",
    ):
        intent = classify_tool_intent(text)
        assert not (intent.needs_tools and intent.category == "image"), text
