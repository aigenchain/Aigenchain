from src.action_intents import (
    classify_knowledge_intent,
    classify_tool_intent,
    message_needs_tools,
    resolve_knowledge_route,
)


def test_calendar_entry_request_promotes_to_agent():
    assert message_needs_tools("Can you add an entry to my calendar?")
    intent = classify_tool_intent("Can you add an entry to my calendar?")
    assert intent.needs_tools
    assert intent.category == "calendar"


def test_calendar_imperative_variants_promote_to_agent():
    assert message_needs_tools("add lunch with Sam to my calendar tomorrow at noon")
    assert message_needs_tools("schedule a call with Mina next Friday")
    assert message_needs_tools("put dentist appointment on my calendar")
    assert message_needs_tools("Alright. Recreate that same appointment")
    assert message_needs_tools("Okay delete that doctor appointment from the calendar")
    assert message_needs_tools("have another go at adding a test entry to the calendar")
    assert message_needs_tools(
        "Okay so you should be able to create that calendar event for tomorrow at 1:30 p.m. right for me to go to the hardware store"
    )
    assert message_needs_tools(
        "make it an appointment at 12pm for me to visit the doctor it's tomorrow the 2nd of June 2026"
    )


def test_calendar_read_requests_promote_to_agent():
    assert message_needs_tools("What upcoming events do I have?")
    assert message_needs_tools("Can you show my next appointments?")
    assert message_needs_tools("Do I have upcoming Taekwondo classes this week?")
    assert message_needs_tools("What's on my calendar tomorrow?")
    assert message_needs_tools("When is my next meeting?")


def test_note_todo_and_reminder_actions_promote_to_agent():
    assert message_needs_tools("add milk to my todo list")
    assert message_needs_tools("take a note that the server needs checking")
    assert message_needs_tools("set a reminder to call Pat at 4pm")


def test_email_and_ui_actions_promote_to_agent():
    assert message_needs_tools("reply to that email")
    assert message_needs_tools("mark those emails as read")
    assert message_needs_tools("open my calendar")
    assert message_needs_tools("turn off web search")


def test_research_action_promotes_to_agent():
    assert message_needs_tools("research cost effective local models")
    assert message_needs_tools("can you look into GPU hosting options")


def test_explicit_web_search_promotes_to_agent():
    assert message_needs_tools("use web search and find a recipe for chocolate chip cookies")
    assert message_needs_tools("do a web search for the best chocolate chip cookies")
    assert message_needs_tools("search the web for current RTX 3090 prices")
    assert classify_tool_intent("use web search and find a recipe").category == "web"


def test_indonesian_fresh_lookups_promote_to_web():
    fresh = [
        "berita terbaru gempa hari ini",
        "harga bitcoin sekarang berapa",
        "cuaca Jakarta besok gimana",
        "siapa yang menang pertandingan tadi malam",
        "kurs dolar ke rupiah hari ini",
        "berita terbaru soal harga emas",
        "kabar terbaru tentang AI",
        "info terkini tentang startup teknologi",
        "trending topic di media sosial hari ini",
    ]
    for text in fresh:
        intent = classify_tool_intent(text)
        assert intent.category == "web", f"expected web for: {text!r} (got {intent.category})"


def test_indonesian_stable_knowledge_stays_plain_chat():
    stable = [
        "apa itu fotosintesis",
        "jelaskan cara kerja rekursi dalam pemrograman",
        "bagaimana cara membuat kopi yang enak",
        "ceritakan sejarah Kerajaan Majapahit",
        "tips belajar bahasa Inggris",
        "resep nasi goreng sederhana",
        "halo apa kabar",
        "apa kabar hari ini",
        "jelaskan fenomena cuaca ekstrem",
        "bagaimana cuaca terbentuk",
        "apa itu harga pokok penjualan",
        "tips agar tidak viral di medsos",
        "jelaskan sejarah trending topic",
    ]
    for text in stable:
        intent = classify_tool_intent(text)
        assert intent.category != "web", f"unexpected web for: {text!r}"


def test_english_present_moment_lookups_promote_to_web():
    fresh = [
        "who is the president of Argentina right now",
        "who is the prime minister of the UK currently",
        "what is the price of gold today",
        "how much is one euro in dollars right now",
    ]
    for text in fresh:
        intent = classify_tool_intent(text)
        assert intent.category == "web", f"expected web for: {text!r} (got {intent.category})"


def test_english_stable_who_what_stays_plain_chat():
    stable = [
        "who wrote Hamlet",
        "what is the capital of Japan",
        "who is Albert Einstein",
        "what is the capital of France",
        "who is the president of the United States",
    ]
    for text in stable:
        intent = classify_tool_intent(text)
        assert intent.category != "web", f"unexpected web for: {text!r}"


def test_explanatory_calendar_questions_stay_plain_chat():
    assert not message_needs_tools("How do I add an entry to my calendar?")
    assert not message_needs_tools("What about the built-in Odysseus calendar, is that linked to email?")
    assert not message_needs_tools("Can you explain how calendar reminders work?")
    intent = classify_tool_intent("How do I add an entry to my calendar?")
    assert not intent.needs_tools
    assert intent.reason == "explanatory feature question"


def test_router_reports_non_calendar_categories():
    assert classify_tool_intent("reply to that email").category == "email"
    assert classify_tool_intent("open my calendar").category == "ui"
    assert classify_tool_intent("research cost effective local models").category == "research"


def test_image_generation_multilingual_requests_route_to_image():
    # The keyword fast-path covers the world's most-spoken languages. Anything
    # it misses is still handled because generate_image is offered to the LLM
    # once escalated; these assertions lock in the fast-path coverage.
    cases = [
        "generate an image of a sunset",
        "draw me a dragon",
        "buat gambar pantai",
        "gambar pemandangan gunung",
        "crea una imagen de un gato",       # ES
        "dibuja un perro rojo",             # ES
        "crie uma imagem de praia",         # PT
        "desenhe um gato",                  # PT
        "crée une image de montagne",       # FR
        "dessine un chat",                  # FR
        "erstelle ein bild von einem hund", # DE
        "zeichne mir einen drachen",        # DE
        "disegna un cane",                  # IT
        "maak een afbeelding van een kat",  # NL
        "нарисуй картинку заката",          # RU
        "создай изображение кота",          # RU
        "bir kedi resmi çiz",               # TR
        "gün batımı görseli oluştur",       # TR
        "narysuj obraz psa",                # PL
        "vẽ một con mèo",                   # VI
        "ارسم صورة قطة",                    # AR
        "猫の画像を作って",                  # JA
        "絵を描いて",                        # JA
        "고양이 이미지를 만들어",            # KO
        "生成一张猫的图片",                  # ZH
        "画一只狗",                          # ZH
        "请画一只猫",                        # ZH
    ]
    for text in cases:
        intent = classify_tool_intent(text)
        assert intent.category == "image", f"expected image for {text!r}, got {intent.category!r}"


def test_image_informational_and_references_stay_plain_chat():
    negatives = [
        "how does image generation work?",
        "what is an image?",
        "apa itu gambar vektor?",
        "gambar itu bagus banget",
        "que es una imagen?",       # ES
        "was ist ein bild?",        # DE
        "что такое изображение?",   # RU
        "这幅画很美",                # ZH: "this painting is beautiful"
        "这张照片不错",              # ZH: "this photo is nice"
    ]
    for text in negatives:
        intent = classify_tool_intent(text)
        assert intent.category != "image", f"unexpected image match for {text!r}"


# --------------------------------------------------------------------------- #
# Knowledge routing (Tahap 3)
# --------------------------------------------------------------------------- #

def test_knowledge_personal_recall():
    cases = [
        "what did I tell you about my sister?",
        "do you remember my favorite color?",
        "what do you know about me?",
        "what is my birthday?",
        "apa yang pernah aku bilang ke kamu soal pekerjaanku?",
        "kamu masih inget alamat aku?",
        "apa nama saya?",
    ]
    for text in cases:
        ki = classify_knowledge_intent(text)
        assert ki.source == "personal_recall", f"{text!r} -> {ki.source} ({ki.reason})"
        assert ki.needs_retrieval


def test_knowledge_past_chat():
    cases = [
        "what did we discuss yesterday?",
        "in our previous conversation you mentioned a plan",
        "you told me earlier about the config",
        "obrolan kita kemarin soal apa?",
        "apa yang kita bahas waktu itu?",
        "kamu bilang tadi soal harga",
    ]
    for text in cases:
        ki = classify_knowledge_intent(text)
        assert ki.source == "past_chat", f"{text!r} -> {ki.source} ({ki.reason})"
        assert ki.needs_retrieval


def test_knowledge_personal_docs():
    cases = [
        "according to the document I uploaded, what is the deadline?",
        "in my report, summarize section 2",
        "based on the file I shared earlier",
        "menurut dokumen yang aku upload, berapa totalnya?",
        "di file laporan itu ada angka berapa?",
    ]
    for text in cases:
        ki = classify_knowledge_intent(text)
        assert ki.source == "personal_docs", f"{text!r} -> {ki.source} ({ki.reason})"
        assert ki.needs_retrieval


def test_knowledge_web_source_phrasing():
    for text in ["search the web for python 3.14 release", "cari di internet harga emas"]:
        ki = classify_knowledge_intent(text)
        assert ki.source == "web", f"{text!r} -> {ki.source} ({ki.reason})"
        assert not ki.needs_retrieval


def test_knowledge_none_for_general_and_howto():
    negatives = [
        "what is the capital of France?",
        "explain how recursion works",
        "how do I upload a document?",
        "write a haiku about autumn",
        "apa itu fotosintesis?",
        "halo apa kabar",
    ]
    for text in negatives:
        ki = classify_knowledge_intent(text)
        assert ki.source == "none", f"{text!r} -> {ki.source} ({ki.reason})"
        assert not ki.needs_retrieval


def test_resolve_knowledge_route_backing_tools():
    r = resolve_knowledge_route("what is my birthday?")
    assert r.source == "personal_recall"
    assert r.backing_tool == "manage_memory"
    assert r.needs_retrieval

    r = resolve_knowledge_route("what did we discuss yesterday?")
    assert r.source == "past_chat"
    assert r.backing_tool == "search_chats"

    r = resolve_knowledge_route("in the document I uploaded, what's the deadline?")
    assert r.source == "personal_docs"
    assert r.backing_tool == "manage_documents"

    r = resolve_knowledge_route("what is the capital of France?")
    assert r.source == "none"
    assert r.backing_tool is None
    assert not r.needs_retrieval


def test_resolve_knowledge_route_web_supersedes():
    # Web active (manual toggle or tool classifier) supersedes model knowledge.
    r = resolve_knowledge_route("what is the capital of France?", web_active=True)
    assert r.source == "web"
    assert r.backing_tool == "web_search"
    assert not r.needs_retrieval

    r = resolve_knowledge_route("latest news", tool_category="web")
    assert r.source == "web"

    # But web must NOT override an explicit internal-recall request.
    r = resolve_knowledge_route("what is my birthday?", web_active=True)
    assert r.source == "personal_recall"
