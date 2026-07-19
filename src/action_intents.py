"""Lightweight routing hints for chat requests that need tools.

These patterns are intentionally conservative. They only promote plain chat
to agent mode when the user asks the assistant to take an action, not when the
user asks how a feature works.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Pattern


@dataclass(frozen=True)
class ToolIntent:
    """A cheap, deterministic chat-to-agent routing decision."""

    needs_tools: bool
    category: str = ""
    reason: str = ""


_ACTION_QUESTION = r"\b(?:can|could|would|will)\s+you\s+"
_ACTION_FOLLOWUP = (
    r"\b(?:you\s+should\s+be\s+able\s+to|"
    r"(?:can|could|would|will|should)\s+you|"
    r"you\s+(?:can|could|would|will|should|need\s+to|have\s+to))\s+"
)
_PLEASE = r"^\s*(?:(?:please|ok(?:ay)?|alright|right|sure|cool|great|thanks|tolong|coba|mohon|minta|bisakah|bisa|tolongin|dong|ayo|yuk)[\s,.!-]+)*"

_CALENDAR_ACTION = (
    r"(?:add|adding|create|creating|recreate|recreating|schedule|scheduling|"
    r"reschedule|rescheduling|book|booking|put|set\s+up|make|making|"
    r"delete|deleting|remove|removing|cancel|cancelling|canceling)"
)
_CALENDAR_THING = r"(?:calendar|calendar\s+(?:entry|item)|event|meeting|appointment|entry|call)"
_CALENDAR_READ_THING = r"(?:calendar|schedule|events?|meetings?|appointments?|classes?)"
_EXPLANATORY_PREFIX = re.compile(
    r"^\s*(?:how\s+(?:do|can)\s+i|can\s+you\s+explain|what\s+about|tell\s+me\s+how|show\s+me\s+how)\b",
    re.I,
)

_PANEL = (
    r"(?:calendar|notes?|inbox|email|mail|documents?|docs|library|gallery|"
    r"settings|cookbook|sessions?|chats?|skills|memories|memory|brain)"
)

# Image generation — multilingual. The primary safety net is the LLM itself:
# generate_image is offered to the model, which understands every language and
# decides when to call it. These keyword patterns are only a fast path so the
# request is escalated to the tool-capable agent loop without waiting on the
# model. They cover the world's most-spoken languages; anything they miss still
# works because the LLM can call the tool once escalated / when tools are shown.
#
# Visual-object nouns across major languages (EN, ID/MS, ES, PT, FR, DE, IT, NL,
# RU, TR, PL, VI, plus CJK/AR ideographs handled separately below).
_IMAGE_THING = (
    r"(?:image|images|picture|pictures|photo|photos|drawing|drawings|"
    r"illustration|illustrations|artwork|art|logo|logos|poster|posters|"
    r"wallpaper|icon|sketch|painting|"
    r"gambar|foto|ilustrasi|lukisan|sketsa|desain|"          # ID/MS
    r"imagen|imagenes|imágenes|dibujo|dibujos|foto|fotos|ilustración|ilustracion|cuadro|"  # ES
    r"imagem|imagens|desenho|desenhos|ilustração|ilustracao|figura|"  # PT
    r"image|images|dessin|dessins|photo|illustration|tableau|"  # FR
    r"bild|bilder|zeichnung|zeichnungen|foto|abbildung|grafik|"  # DE
    r"immagine|immagini|disegno|disegni|foto|illustrazione|"  # IT
    r"afbeelding|afbeeldingen|tekening|foto|illustratie|"  # NL
    r"картинку|картинка|изображение|изображению|рисунок|фото|иллюстрацию|"  # RU
    r"resim|resmi|resmini|görsel|görseli|gorsel|çizim|cizim|fotoğraf|fotograf|"  # TR
    r"obraz|obrazek|rysunek|zdjęcie|zdjecie|ilustracja|"  # PL
    r"hình|ảnh|tranh|hình\s*ảnh|"  # VI
    r"صورة|صوره|رسم|"  # AR
    r"画像|絵|イラスト|"  # JA
    r"이미지|그림|"  # KO
    r"图片|图像|画|插图)"  # ZH
)
# Creation verbs across major languages.
_IMAGE_EN_VERB = (
    r"(?:generate|create|make|draw|render|paint|design|sketch|produce|"
    r"genera|generar|crea|crear|dibuja|dibujar|haz|hacer|pinta|diseña|disena|diseñar|"  # ES
    r"gera|gerar|cria|crie|criar|desenha|desenhar|faz|fazer|"  # PT
    r"génère|genere|crée|cree|créer|creer|dessine|dessiner|fais|faire|"  # FR
    r"erstelle|erstellen|erzeuge|zeichne|zeichnen|male|malen|generiere|"  # DE
    r"crea|creare|genera|generare|disegna|disegnare|dipingi|"  # IT
    r"maak|teken|genereer|"  # NL
    r"нарисуй|создай|сгенерируй|сделай|нарисовать|создать|"  # RU
    r"çiz|ciz|oluştur|olustur|yap|yarat|"  # TR
    r"narysuj|stwórz|stworz|wygeneruj|zrób|zrob|"  # PL
    r"vẽ|tạo|"  # VI
    r"ارسم|أنشئ|انشئ|"  # AR
    r"描いて|作って|生成して|"  # JA
    r"그려|만들어|생성해|"  # KO
    r"画|生成|绘制)"  # ZH
)
_IMAGE_ID_VERB = (
    r"(?:buat|buatkan|buatin|bikin|bikinkan|bikinin|gambar|gambarkan|gambarin|"
    r"lukis|lukiskan|lukisin|desain|desainkan|desainin|rancang|rancangkan)"
)

_ROUTING_PATTERNS: tuple[tuple[str, str, Pattern[str]], ...] = tuple(
    (category, reason, re.compile(pattern, re.I))
    for category, reason, pattern in (
        # Image generation (English + Indonesian). Checked first so requests
        # like "create a poster for my event" resolve to image, not calendar.
        # Only when the user asks the assistant to actually produce a visual.
        ("image", "assistant image generation request", rf"{_ACTION_QUESTION}{_IMAGE_EN_VERB}\b.{{0,60}}\b{_IMAGE_THING}\b"),
        ("image", "image generation imperative request", rf"{_PLEASE}{_IMAGE_EN_VERB}\s+(?:me\s+)?(?:a\s+|an\s+|some\s+|the\s+)?(?:\w+\s+){{0,4}}?{_IMAGE_THING}\b"),
        ("image", "draw imperative request", rf"{_PLEASE}draw\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?.+"),
        # Draw/paint verbs whose meaning is inherently "make a picture", so no
        # accompanying image noun is required (mirrors English "draw ..."):
        # ES dibuja/pinta, PT desenha/pinta, FR dessine, DE zeichne/male,
        # IT disegna/dipingi, VI vẽ, RU нарисуй.
        ("image", "multilingual draw-verb request", r"^(?:\s*(?:por favor|please|s'?il vous plaît|bitte|por gentileza|пожалуйста|làm ơn)[,\s]+)?(?:dibuja|dibújame|pinta|píntame|desenha|desenhe|dessine|dessine-moi|zeichne|zeichne\s+mir|male|male\s+mir|disegna|disegnami|dipingi|vẽ|нарисуй)\b.+"),
        # CJK single-char draw/paint verb directly followed by subject:
        # ZH 画一只狗 / 绘制…, no spaces. 画 is anchored to the start (verb
        # "draw"); mid-sentence 画 is usually the noun "painting" (这幅画很美)
        # and must not match. 绘制 is unambiguously the verb, allowed anywhere.
        ("image", "cjk draw-verb request", r"^(?:请|帮我|给我)?画\S+|绘制\S+|生成.{0,6}(?:图片|图像|图)"),
        ("image", "indonesian image generation request", rf"{_PLEASE}{_IMAGE_ID_VERB}\s+(?:kan\s+|in\s+|aku\s+|saya\s+|sebuah\s+|satu\s+)?(?:\w+\s+){{0,4}}?{_IMAGE_THING}\b"),
        ("image", "indonesian gambarkan request", rf"{_PLEASE}(?:gambarkan|gambarin)\b.+"),
        # Bare visual-noun command: "gambar pemandangan gunung", "foto kucing
        # lucu", "ilustrasi naga". A visual noun at the very start followed by a
        # subject reads as "produce this image". Excludes single bare words
        # ("gambar?", "foto") by requiring at least one following subject word.
        ("image", "indonesian bare visual-noun request", rf"{_PLEASE}(?:gambar|foto|lukisan|ilustrasi|sketsa|desain|poster|wallpaper)\s+(?!itu\b|ini\b|tersebut\b|tadi\b|kamu\b|kamu\?|nya\b|apa\b|mana\b|yang\b)\w+.*"),
        # Universal multilingual: a creation verb anywhere near a visual noun,
        # in either word order (covers SVO and SOV/OV languages, e.g. RU
        # "нарисуй картинку", TR "resim çiz", JA "絵を描いて", ES "crea una
        # imagen"). The verb/noun classes already exclude bare informational use.
        ("image", "multilingual verb+noun image request", rf"\b{_IMAGE_EN_VERB}\b.{{0,40}}{_IMAGE_THING}"),
        ("image", "multilingual noun+verb image request", rf"{_IMAGE_THING}.{{0,40}}\b{_IMAGE_EN_VERB}\b"),
        # CJK/scriptio-continua: no spaces between verb and noun.
        ("image", "cjk image request", r"(?:画像|絵|イラスト|图片|图像|插图|이미지|그림).{0,8}(?:描いて|作って|生成して|画|生成|绘制|그려|만들어|생성해)|(?:描いて|作って|生成して|绘制|그려|만들어|생성해).{0,8}(?:画像|絵|イラスト|图片|图像|插图|이미지|그림)"),

        # Calendar/event creation. Covers "Can you add an entry to my
        # calendar?", imperatives like "add lunch to my calendar", and
        # follow-ups such as "you should be able to create that event now".
        ("calendar", "assistant calendar action request", rf"{_ACTION_QUESTION}{_CALENDAR_ACTION}\b.{{0,120}}\b{_CALENDAR_THING}\b"),
        ("calendar", "calendar follow-up action request", rf"{_ACTION_FOLLOWUP}{_CALENDAR_ACTION}\b.{{0,120}}\b{_CALENDAR_THING}\b"),
        ("calendar", "calendar imperative action request", rf"{_PLEASE}{_CALENDAR_ACTION}\b.{{0,120}}\b{_CALENDAR_THING}\b"),
        ("calendar", "calendar target action request", rf"{_PLEASE}{_CALENDAR_ACTION}\b.{{0,120}}\b(?:to|on|in|into|for)\s+(?:my\s+|the\s+|this\s+)?calendar\b"),
        ("calendar", "calendar item action request", rf"{_PLEASE}{_CALENDAR_ACTION}\s+(?:it\s+)?(?:a\s+|an\s+)?(?:calendar\s+)?(?:event|meeting|appointment|entry|item|call)\b"),
        ("calendar", "calendar target action request", rf"\b{_CALENDAR_ACTION}\b.{{0,120}}\b(?:to|on|in|into|for)\s+(?:my\s+|the\s+|this\s+)?calendar\b"),
        ("calendar", "put item on calendar request", r"\bput\s+.+\bon\s+(?:my\s+)?calendar\b"),

        # Calendar/event lookup. A question such as "Do I have Taekwondo
        # classes this week?" needs the calendar tool; plain chat cannot know.
        ("calendar", "calendar lookup request", rf"\b(?:list|show|check|find)\b.{{0,120}}\b(?:my\s+|the\s+)?(?:upcoming|next|today'?s?|tomorrow'?s?|this\s+week'?s?)\b.{{0,120}}\b{_CALENDAR_READ_THING}\b"),
        ("calendar", "calendar lookup question", rf"\b(?:what|which)\b.{{0,120}}\b(?:upcoming|next|today'?s?|tomorrow'?s?|this\s+week'?s?)\b.{{0,120}}\b{_CALENDAR_READ_THING}\b"),
        ("calendar", "calendar availability question", rf"\bdo\s+i\s+have\b.{{0,120}}\b(?:upcoming|next|today|tomorrow|this\s+week)\b.{{0,120}}\b{_CALENDAR_READ_THING}\b"),
        ("calendar", "calendar agenda question", r"\bwhat(?:'s| is)\s+on\s+(?:my\s+)?calendar\b"),
        ("calendar", "next calendar item question", r"\bwhen\s+(?:is|are)\s+(?:my\s+)?next\s+(?:event|meeting|appointment|class)\b"),

        # Notes, todos, checklists, and reminders.
        ("notes", "reminder request", r"\bremind\s+me\b"),
        ("notes", "assistant note/todo action request", rf"{_ACTION_QUESTION}(?:add|create|make|take|jot|write\s+down|set)\b.{{0,120}}\b(?:note|todo|task|checklist|reminder)\b"),
        ("notes", "note/todo imperative request", rf"{_PLEASE}(?:add|create|make)\s+(?:a\s+|an\s+)?(?:todo|task|reminder|note|checklist)\b"),
        ("notes", "take note request", rf"{_PLEASE}(?:take|jot|write\s+down)\s+(?:a\s+|an\s+)?note\b"),
        ("notes", "add item to notes/todo request", rf"{_PLEASE}(?:add|jot|write\s+down)\b.{{0,120}}\b(?:to|in|into)\s+(?:my\s+|the\s+)?(?:todo(?:\s+list)?|task\s+list|notes?|checklist)\b"),
        ("notes", "set reminder request", rf"{_PLEASE}set\s+(?:a\s+)?reminder\b"),
        ("notes", "assistant reminder request", rf"{_ACTION_QUESTION}set\s+(?:a\s+)?reminder\b"),

        # Email actions.
        ("email", "assistant email action request", rf"{_ACTION_QUESTION}(?:send|write|reply|email|message|archive|delete|mark)\b.{{0,120}}\b(?:emails?|mail|messages?|inbox|unread|read)\b"),
        ("email", "send/write/reply email request", rf"{_PLEASE}(?:send|write|reply)\b.{{0,120}}\b(?:emails?|mail|messages?)\b"),
        ("email", "archive/delete/mark email request", rf"{_PLEASE}(?:archive|delete|mark)\b.{{0,120}}\b(?:emails?|mail|messages?|inbox)\b"),
        ("email", "email composition request", r"\b(?:send|write|reply)\s+(?:an?\s+)?(?:email|message|mail)\b"),
        ("email", "email contact request", r"\bemail\s+\w+\b"),
        ("email", "check inbox request", r"\bcheck\s+(?:my\s+)?(?:email|inbox|mail)\b"),
        ("email", "unread email request", r"\bunread\s+(?:email|mail)s?\b"),

        # UI/control-plane actions that should open panels or flip toggles.
        ("ui", "open/show panel request", rf"{_PLEASE}(?:open|show|bring\s+up)\s+(?:me\s+)?(?:my\s+|the\s+)?{_PANEL}\b"),
        ("ui", "tool or feature toggle request", r"\b(?:disable|enable|turn\s+(?:on|off))\s+(?:the\s+)?(?:shell|search|web|browser|documents?|memory|skills|images?|calendar|email|mail|research|incognito)\b"),

        # Deep research jobs, not quick conceptual mentions of research.
        ("web", "explicit web search request", rf"{_PLEASE}(?:do|run|use|perform|make)\s+(?:a\s+)?(?:web\s+search|search\s+the\s+web)\b.+"),
        ("web", "generic search request", rf"{_PLEASE}search\s+(?!(?:my\s+)?(?:chats?|history|sessions?|notes?|todos?|emails?|mail|inbox|documents?|docs|gallery|images?|files?)\b).+"),
        ("web", "web lookup imperative request", rf"{_PLEASE}(?:web\s+search|search\s+the\s+web|search\s+online|look\s+up|google(?:\s+it)?)\b.*"),
        ("web", "short web lookup follow-up", rf"{_PLEASE}(?:just\s+)?(?:look\s+it\s+up|look\s+up|search\s+(?:online|web|now)|search\s+it)\b\s*$"),
        ("web", "assistant short web lookup request", rf"{_ACTION_QUESTION}(?:search|look\s+up|google)(?:\s+(?:online|web|now|it))?\b.*"),
        ("web", "assistant web lookup request", rf"{_ACTION_QUESTION}(?:web\s+search|search\s+the\s+web|search\s+online|look\s+up|google(?:\s+it)?)\b.*"),
        ("web", "assistant weather check request", rf"{_ACTION_QUESTION}(?:check|find|get|look\s+up)\b.{{0,100}}\b(?:weather|forecast)\b.*"),
        ("web", "news lookup request", r"\b(?:news|headlines)\s+(?:in|from|about|for)\s+[\w\s.-]{2,80}\??\s*$"),
        ("web", "forecast lookup request", r"\b(?:hourly|daily|weekly|local)\s+(?:weather\s+)?forecast\b|\b(?:weather\s+)?forecast\s+(?:for|today|tomorrow|now|hourly)\b"),
        ("web", "weather lookup request", r"\bweather\b.{0,80}\b(?:hourly|rain|raining|rin|today|tomorrow|update|current|now)\b|\b(?:hourly|rain|raining|rin)\b.{0,80}\bweather\b"),
        ("web", "rain lookup request", r"\b(?:hourly|daily|weekly|local|today|tomorrow|current|now|update)\b.{0,100}\b(?:rain|raining|rainy|precipitation|showers?)\b|\b(?:rain|raining|rainy|precipitation|showers?)\b.{0,100}\b(?:hourly|daily|weekly|local|today|tomorrow|current|now|update|in|for|at)\b"),
        ("web", "bare weather lookup request", r"\b(?:weather|forecast)\s+(?:in|for|at)?\s*[\w\s.-]{2,80}\??\s*$|\b[\w\s.-]{2,80}\s+(?:weather|forecast)\??\s*$"),
        ("web", "latest info lookup request", r"\b(?:latest|current|newest|recent|up(?: |-)?to(?: |-)?date)\s+(?:info|information|updates?|details?|developments?)\s+(?:on|about|for|in)\s+[\w\s.,:'\"/-]{2,120}\??\s*$"),
        ("web", "current/latest lookup request", r"\b(?:current|latest|today'?s?|right\s+now|live|online)\b.{0,120}\b(?:rate|price|news|weather|forecast|score|exchange|market|status)\b"),
        ("web", "rate/price/news lookup request", r"\b(?:rate|rates|price|prices|news|weather|forecast|score|exchange|currency|market)\b.{0,120}\b(?:now|today|current|latest|online|live|search|look\s+up|find)\b"),
        ("web", "conversion-rate lookup request", r"\b(?:convert|conversion|exchange)\b.{0,120}\b(?:rate|rates|currency|currencies|price|prices)\b"),
        # "who is the president of X right now" — a factual question pinned to the
        # present moment. The trailing freshness marker distinguishes it from a
        # stable-knowledge question ("who wrote Hamlet", "what is the capital ..").
        ("web", "who/what is ... right now lookup", r"\b(?:who|what|where|how\s+much|how\s+many)\s+(?:is|are|was|were)\b.{0,100}\b(?:right\s+now|currently|today|these\s+days|at\s+the\s+moment|as\s+of\s+(?:today|now))\b"),

        # ---- Indonesian web-lookup phrasings ----
        # Freshness markers pair with news/finance/weather subjects so that
        # stable-knowledge questions ("apa itu ...", "jelaskan cara kerja ...",
        # "ceritakan sejarah ...") never match. "halo apa kabar" is safe because
        # it carries no freshness marker alongside "kabar".
        ("web", "cari/browsing di internet (ID)", rf"{_PLEASE}(?:cari(?:kan)?|carikan|browsing|googling|google)\b.{{0,40}}\b(?:di\s+)?(?:internet|web|online|google)\b"),
        ("web", "berita/info terbaru lookup (ID)", r"\b(?:berita|info|informasi|update)\b.{0,60}\b(?:terbaru|terkini|terupdate|hari\s+ini|sekarang|saat\s+ini|barusan)\b|\b(?:terbaru|terkini|terupdate)\b.{0,60}\b(?:berita|info|informasi|update)\b"),
        # "kabar" needs a freshness marker AND must not be the greeting
        # "apa kabar" (with or without a trailing time word).
        ("web", "kabar terbaru lookup (ID)", r"(?<!apa\s)\bkabar\b\s+(?:terbaru|terkini|terupdate)\b"),
        ("web", "harga/kurs terkini lookup (ID)", r"\b(?:harga|kurs|nilai\s+tukar|saham|kripto)\b.{0,60}\b(?:sekarang|saat\s+ini|hari\s+ini|terkini|terbaru|live|berapa)\b|\bberapa\b.{0,40}\b(?:harga|kurs)\b"),
        ("web", "cuaca lookup (ID)", r"\bcuaca\b.{0,60}\b(?:hari\s+ini|besok|sekarang|nanti\s+(?:sore|malam|pagi)|malam\s+ini|pagi\s+ini|sore\s+ini)\b|\b(?:ramalan|prakiraan)\s+cuaca\b"),
        ("web", "skor/pertandingan lookup (ID)", r"\b(?:skor|hasil|pemenang|jadwal|siapa\s+(?:yang\s+)?menang)\b.{0,60}\b(?:pertandingan|laga|bola|piala|liga|tadi(?:\s+malam)?|semalam|barusan|hari\s+ini|malam\s+ini|nanti)\b"),
        ("web", "trending/viral topic lookup (ID)", r"\b(?:trending|viral|lagi\s+ramai|lagi\s+heboh)\b.{0,60}\b(?:hari\s+ini|sekarang|saat\s+ini)\b|\b(?:apa\s+(?:yang\s+)?(?:lagi\s+)?|yang\s+lagi\s+)(?:trending|viral|ramai|heboh)\b"),

        ("research", "deep research imperative request", rf"{_PLEASE}(?:research|deep\s+dive|look\s+into|investigate)\s+.+"),
        ("research", "assistant deep research request", rf"{_ACTION_QUESTION}(?:research|do\s+research|deep\s+dive|look\s+into|investigate)\s+.+"),

        # Shell / remote-host intent.
        ("shell", "ssh request", r"\bssh\s+(?:in)?to\b"),
        ("shell", "ssh target request", r"\bssh\s+\w+"),
        ("shell", "remote command request", r"\b(run|execute)\s+.{1,40}\bon\s+\w+"),
        ("shell", "assistant command execution request", r"\b(can|could|please|would)\s+you\s+(run|execute|exec)\b"),
        # Shell verbs only count in imperative position (start of message,
        # optionally after "please") or as a "can you ..." request. A bare
        # word match promoted informational questions ("What does the grep
        # command do?") and incidental uses ("My cat ate my homework").
        ("shell", "imperative shell command request", rf"{_PLEASE}(deploy|build|install|restart|reboot|kill|tail|grep|cat|ls|cd|cp|mv|rm)\b\s+\S+"),
        ("shell", "assistant shell command request", rf"{_ACTION_QUESTION}(deploy|build|install|restart|reboot|kill|tail|grep|cat|ls|cd|cp|mv|rm)\b\s+\S+"),
        ("shell", "system/file check request", r"\b(check|see)\s+(if|whether|what)\s+.{1,40}\b(running|process|service|port|file|exists?)\b"),
    )
)

_TOOL_INTENT_PATTERNS: tuple[Pattern[str], ...] = tuple(
    pattern for _, _, pattern in _ROUTING_PATTERNS
)


def classify_tool_intent(text: str) -> ToolIntent:
    """Classify whether a chat message should be promoted to agent mode."""
    if not text:
        return ToolIntent(False, reason="empty message")
    if _EXPLANATORY_PREFIX.search(text):
        return ToolIntent(False, reason="explanatory feature question")
    for category, reason, pattern in _ROUTING_PATTERNS:
        if pattern.search(text):
            return ToolIntent(True, category=category, reason=reason)
    return ToolIntent(False, reason="no tool-action pattern matched")


def message_needs_tools(text: str, patterns: Iterable[Pattern[str]] = _TOOL_INTENT_PATTERNS) -> bool:
    """Return True when a plain chat message should be promoted to agent mode."""
    if not text:
        return False
    if _EXPLANATORY_PREFIX.search(text):
        return False
    if patterns is _TOOL_INTENT_PATTERNS:
        return classify_tool_intent(text).needs_tools
    return any(pattern.search(text) for pattern in patterns)


# ---------------------------------------------------------------------------
# Follow-up intents that operate on the LAST generated image in a conversation.
#
# These are only meaningful when a previous assistant turn produced an image.
# The caller checks that first, then uses these to decide:
#   * EDIT   → regenerate a new image from (previous prompt + the requested
#              change), because the text-to-image model has no img2img.
#   * ANALYZE→ run a vision model on the actual image file so the assistant can
#              answer questions about what is really in the picture (faces,
#              colours, defects) instead of guessing from the prompt text.
# ---------------------------------------------------------------------------

# A pronoun/determiner that refers back to the image just made:
#   EN it/that/this/the (image/picture/photo)  ID itu/ini/nya/tersebut/tadi
_IMAGE_REF = (
    r"(?:it|that|this|the\s+(?:image|picture|photo|pic)|"
    r"(?:gambar|foto|fotonya|gambarnya|hasilnya)?\s*(?:itu|ini|tadi|tersebut|nya))"
)

# Change/fix verbs, multilingual (EN + ID first-class, plus common others).
_IMAGE_EDIT_VERB = (
    r"(?:edit|change|modify|adjust|alter|fix|improve|enhance|redo|regenerate|"
    r"remake|redraw|repaint|update|revise|tweak|refine|correct|replace|add|"
    r"remove|make\s+it|turn\s+it\s+into|"
    r"ubah|ganti|perbaiki|perbaikin|betulkan|betulin|edit|revisi|"
    r"sesuaikan|sempurnakan| perbagus|perbagus|percantik|percantikkan|"
    r"tambah|tambahkan|tambahin|hapus|hilangkan|hilangin|buat\s+lebih|"
    r"jadikan|ubahlah|gantikan)"
)

_IMAGE_EDIT_PATTERNS: tuple[Pattern[str], ...] = tuple(
    re.compile(p, re.I) for p in (
        # "perbaiki gambarnya", "ubah gambar itu", "fix the image", "edit that photo"
        rf"{_PLEASE}{_IMAGE_EDIT_VERB}\b.{{0,40}}\b{_IMAGE_THING}\b",
        rf"{_PLEASE}{_IMAGE_EDIT_VERB}\b.{{0,20}}{_IMAGE_REF}\b",
        # "make it more realistic", "buat lebih realistis", "buat wajahnya lebih
        # jelas", "bikin gambarnya lebih terang" — allow a short noun between the
        # verb and "lebih/more".
        rf"{_PLEASE}(?:make\s+(?:it|the\s+\w+|\w+)\s+more|make\s+it|"
        rf"buat(?:\s+\w+)?\s+lebih|jadikan(?:\s+\w+)?\s+lebih|bikin(?:\s+\w+)?\s+lebih)\b.+",
        # "perbaiki mukanya", "fix the face/hands/eyes" — body-part fixes imply
        # editing the just-made image.
        rf"{_PLEASE}(?:fix|perbaiki|perbaikin|betulkan|betulin|ubah|ganti)\b.{{0,20}}"
        rf"(?:face|faces|hand|hands|eye|eyes|mouth|nose|hair|smile|"
        rf"muka|mukanya|wajah|wajahnya|tangan|tangannya|mata|matanya|"
        rf"mulut|hidung|rambut|senyum)\b",
        # bare "perbaiki"/"perbagus"/"buat lebih bagus" as a short follow-up
        rf"{_PLEASE}(?:perbaiki|perbaikin|perbagus|percantik|buat\s+lebih\s+bagus|"
        rf"make\s+it\s+better|improve\s+it|fix\s+it)\s*$",
        # add/remove an element to the just-made image: "tambahkan matahari",
        # "add a sunset", "hapus orangnya", "remove the tree", "ganti warnanya"
        rf"{_PLEASE}(?:add|remove|delete|tambah|tambahkan|tambahin|hapus|"
        rf"hilangkan|hilangin|ganti|gantikan|change)\s+\w+",
    )
)

# Questions about the image content: "what is this image?", "gambar apa ini?",
# "kok mukanya aneh?", "why does the face look weird?", "describe it".
_IMAGE_QUESTION_PATTERNS: tuple[Pattern[str], ...] = tuple(
    re.compile(p, re.I) for p in (
        rf"\b(?:what|which|who|whose|why|how|describe|explain|tell\s+me\s+about)\b.{{0,40}}{_IMAGE_REF}\b",
        rf"{_IMAGE_REF}\b.{{0,30}}\b(?:show|showing|depict|about|mean|means|look|looks?\s+like)\b",
        # ID: "gambar apa ini/itu?", "ini gambar apa?"
        r"(?:gambar|foto)\s+apa\s+(?:ini|itu|nya|tadi|tersebut)\b|"
        r"(?:ini|itu)\s+(?:gambar|foto)\s+apa\b",
        # ID/EN "kok mukanya aneh?", "why is the face weird", "kenapa ... aneh"
        r"\b(?:kok|kenapa|mengapa|napa)\b.{0,40}\b(?:aneh|jelek|buruk|salah|"
        r"weird|strange|ugly|bad|wrong|off|distorted)\b",
        r"\b(?:aneh|jelek|buruk|salah|weird|strange|ugly|bad|wrong|distorted)\b.{0,20}\?",
        # "describe/analyze the image", "jelaskan gambarnya", "gambar ini menceritakan apa"
        rf"{_PLEASE}(?:describe|analyz[es]?|jelaskan|ceritakan|deskripsikan)\b.{{0,20}}{_IMAGE_REF}\b",
        r"(?:gambar|foto)(?:nya|\s+(?:ini|itu|tadi|tersebut))?\s+menceritakan\s+apa\b",
    )
)


def is_image_edit_intent(text: str) -> bool:
    """True when the user asks to change/fix/improve the last generated image.

    Meaningful only when a previous image exists (caller must verify). The
    handler regenerates a fresh image from the prior prompt plus this change.
    """
    if not text:
        return False
    return any(p.search(text) for p in _IMAGE_EDIT_PATTERNS)


def is_image_question_intent(text: str) -> bool:
    """True when the user asks about the content/quality of the last image.

    Meaningful only when a previous image exists (caller must verify). The
    handler feeds the actual image file to a vision model to answer.
    """
    if not text:
        return False
    return any(p.search(text) for p in _IMAGE_QUESTION_PATTERNS)


# ---------------------------------------------------------------------------
# Ambiguous image-generation intent.
#
# Principle: detect the *intent* to create a visual even without an explicit
# image noun ("gambar"/"image"), and — when unsure — ask a friendly clarifying
# question instead of either silently generating or flatly refusing.
#
# Example: "tolong buat pantai" → no image noun, but a creation verb + a
# concrete visual subject → likely wants a picture → classify as "maybe" so the
# handler can ask "Maksudnya kamu mau aku bikinin gambar pantainya?".
# ---------------------------------------------------------------------------

# Creation verb (EN + ID) as a standalone leading verb, without requiring an
# accompanying image noun. Reuses the same verb vocabulary as explicit routing.
_CREATE_VERB_LEAD = re.compile(
    rf"{_PLEASE}(?:{_IMAGE_EN_VERB[3:-1]}|{_IMAGE_ID_VERB[3:-1]})\s+"
    r"(?:me\s+|us\s+|aku\s+|saya\s+|kan\s+|in\s+|kamu\s+)?"
    r"(?:a\s+|an\s+|the\s+|some\s+|sebuah\s+|satu\s+|se\w*\s+)?"
    r"(?P<subject>.+)$",
    re.I,
)

# Subjects that make a creation verb clearly NOT an image request. If the object
# is code/text/plan/document/etc. it is a different task, so never treat as image.
_NON_VISUAL_SUBJECT = re.compile(
    r"^\s*(?:"
    r"code|program|script|function|app|application|website|web|api|"
    r"list|plan|schedule|table|document|doc|file|folder|report|essay|article|"
    r"story|poem|song|email|e-mail|message|summary|note|notes|account|"
    r"reminder|event|meeting|appointment|budget|recipe|"
    r"kode|program|aplikasi|situs|daftar|rencana|jadwal|tabel|dokumen|file|"
    r"folder|laporan|esai|artikel|cerita|puisi|lagu|email|surel|pesan|"
    r"ringkasan|catatan|akun|pengingat|acara|rapat|jadwal|anggaran|resep|"
    # Emotions / abstract states — "buat aku senang", "make me happy" are not
    # image requests.
    r"senang|bahagia|sedih|marah|tenang|nyaman|happy|sad|angry|calm|relax"
    r")\b",
    re.I,
)

# Words that clearly are already an image noun — those are handled by explicit
# routing, so exclude them from the "maybe" bucket.
_HAS_IMAGE_NOUN = re.compile(_IMAGE_THING, re.I)


def is_ambiguous_image_intent(text: str) -> bool:
    """True when a message *might* be an image request but isn't explicit.

    Returns True only for the ambiguous middle ground: a creation verb followed
    by a concrete (visual-plausible) subject, WITHOUT any image noun and WITHOUT
    a non-visual subject (code/doc/list/etc.). The handler should respond with a
    friendly clarifying question rather than generating or refusing.
    """
    if not text:
        return False
    # Already explicit → not "maybe" (handled elsewhere).
    if classify_tool_intent(text).category == "image":
        return False
    if _HAS_IMAGE_NOUN.search(text):
        return False
    m = _CREATE_VERB_LEAD.search(text)
    if not m:
        return False
    subject = (m.group("subject") or "").strip()
    if len(subject) < 2:
        return False
    if _NON_VISUAL_SUBJECT.search(subject):
        return False
    # A question about how to do something isn't a creation request.
    if subject.endswith("?") and re.search(r"\b(how|cara|bagaimana|gimana)\b", text, re.I):
        return False
    return True


# Short affirmative / negative replies, used to resolve a pending clarification.
_AFFIRMATIVE = re.compile(
    r"^\s*(?:"
    r"y|ya|yes|yep|yup|yeah|sure|ok(?:ay)?|iya|iyaa+|iyah|iyo|betul|"
    r"benar|bener|boleh|mau|gas|gaskeun|lanjut|lanjutkan|ayo|yuk|silakan|"
    r"silahkan|tolong|please|go|do\s+it|sip|oke|okey|setuju|bikin|buat"
    r")\b",
    re.I,
)
_NEGATIVE = re.compile(
    r"^\s*(?:"
    r"no|nope|nah|tidak|gak|nggak|enggak|engga|jangan|bukan|ga\b|"
    r"stop|batal|cancel|skip|nanti"
    r")\b",
    re.I,
)


def extract_image_subject(text: str) -> str:
    """Strip leading politeness + creation verb, returning the bare subject.

    "tolong buat pantai" → "pantai"; "buatkan aku pemandangan gunung" →
    "pemandangan gunung". Falls back to the trimmed original when nothing
    matches, so the caller always has something to echo.
    """
    if not text:
        return ""
    m = _CREATE_VERB_LEAD.search(text)
    if m:
        subject = (m.group("subject") or "").strip(" .!,?")
        if subject:
            return subject
    return text.strip()


def is_affirmative(text: str) -> bool:
    """True for short 'yes'-style replies confirming a pending clarification."""
    if not text:
        return False
    return bool(_AFFIRMATIVE.search(text.strip()))


def is_negative(text: str) -> bool:
    """True for short 'no'-style replies declining a pending clarification."""
    if not text:
        return False
    return bool(_NEGATIVE.search(text.strip()))


# ---------------------------------------------------------------------------
# Knowledge routing (Tahap 3)
#
# A separate axis from the tool-action classifier above. Where classify_tool_
# intent answers "should this chat be promoted to agent mode?", the knowledge
# classifier answers "which KNOWLEDGE SOURCE should ground the answer?":
#
#   personal_recall → the user's persistent memory ("what did I tell you about
#                     my sister", "apa yang pernah aku ceritakan soal ...")
#   past_chat       → earlier conversations ("what did we discuss yesterday",
#                     "obrolan kita kemarin soal ...")
#   personal_docs   → files the user uploaded / the RAG store ("in the document
#                     I uploaded", "menurut file yang aku upload")
#   web             → clearly external/fresh info (deferred to the tool
#                     classifier's `web` category; kept here only for phrasings
#                     that name the web explicitly as the source)
#   none            → answer from the model's own knowledge (the default)
#
# This is intentionally conservative and OBSERVABILITY-FIRST: it does not force
# retrieval. The route records the decision and may lightly ensure the relevant
# tool stays available, but existing retrieval behaviour is unchanged.
# ---------------------------------------------------------------------------

KNOWLEDGE_SOURCES = ("personal_recall", "past_chat", "personal_docs", "web", "none")


@dataclass(frozen=True)
class KnowledgeIntent:
    """Which knowledge source should ground the answer to a chat message."""

    source: str = "none"
    reason: str = "no knowledge-source pattern matched"

    @property
    def needs_retrieval(self) -> bool:
        """True when an internal knowledge store (not the model itself) applies."""
        return self.source in ("personal_recall", "past_chat", "personal_docs")


# First-person possessive / self-reference across EN + ID, used to disambiguate
# "personal" recall from generic questions.
_KN_SELF = r"(?:my|mine|i|i'?ve|i\s+have|we|we'?ve|our|aku|saya|ku|kita|kami)"

_KNOWLEDGE_PATTERNS: tuple[tuple[str, str, Pattern[str]], ...] = tuple(
    (source, reason, re.compile(pattern, re.I))
    for source, reason, pattern in (
        # ---- personal_docs: files the user uploaded / RAG store ----
        ("personal_docs", "reference to uploaded document (EN)",
         r"\b(?:in|from|according\s+to|based\s+on|per|inside|within)\b.{0,40}\b(?:my\s+|the\s+|that\s+|this\s+|uploaded\s+)?(?:document|doc|docs|file|files|pdf|paper|report|attachment|spreadsheet|slides?)\b"),
        ("personal_docs", "reference to uploaded document (ID)",
         r"\b(?:di|dalam|pada|menurut|berdasar(?:kan)?)\b.{0,40}\b(?:dokumen|dokumenku|file|filenya|berkas|pdf|laporan|lampiran)\b"),
        ("personal_docs", "document I uploaded (EN)",
         r"\b(?:the\s+)?(?:document|doc|file|pdf|paper|report)\b.{0,30}\bi\s+(?:uploaded|shared|attached|sent|gave)\b"),
        ("personal_docs", "document I uploaded (ID)",
         r"\b(?:dokumen|file|berkas|pdf|laporan)\b.{0,30}\b(?:yang\s+)?(?:aku|saya|ku)\s+(?:upload|kirim|lampir(?:kan)?|bagikan)\b"),

        # ---- past_chat: earlier conversations ----
        ("past_chat", "what did we discuss (EN)",
         rf"\bwhat\s+(?:did|had)\s+we\b.{{0,40}}\b(?:discuss|talk(?:ed)?\s+about|say|said|decide[d]?|agree[d]?|cover(?:ed)?)\b"),
        ("past_chat", "our previous conversation (EN)",
         r"\b(?:in|from|during|our|the)\b.{0,20}\b(?:previous|last|earlier|prior|past|recent)\b.{0,20}\b(?:chat|conversation|session|discussion|talk|thread|message[s]?)\b"),
        ("past_chat", "you told me earlier (EN)",
         r"\byou\s+(?:told|said|mentioned|explained)\b.{0,30}\b(?:earlier|before|previously|last\s+time|yesterday|ago)\b"),
        ("past_chat", "did we talk (ID)",
         r"\b(?:obrolan|percakapan|chat|diskusi|pembicaraan)\b.{0,30}\b(?:kita|kemarin|sebelumnya|tadi|lalu|yang\s+lalu|waktu\s+itu)\b"),
        ("past_chat", "what did we talk about (ID)",
         r"\b(?:apa|kapan)\b.{0,30}\bkita\b.{0,20}\b(?:bahas|bicara(?:kan)?|omongin|diskusi(?:kan)?|sepakati|putuskan)\b"),
        ("past_chat", "earlier you said (ID)",
         r"\bkamu\b.{0,20}\b(?:bilang|sebut(?:kan)?|jelas(?:kan|in)?|kata(?:kan)?)\b.{0,25}\b(?:tadi|kemarin|sebelumnya|waktu\s+itu|barusan)\b"),

        # ---- personal_recall: user's persistent memory ----
        ("personal_recall", "what did I tell you (EN)",
         rf"\bwhat\s+(?:did|have)\s+i\b.{{0,25}}\b(?:tell|told|say|said|mention(?:ed)?|share[d]?)\b.{{0,15}}\byou\b"),
        ("personal_recall", "do you remember about me (EN)",
         rf"\b(?:do\s+you\s+remember|you\s+remember|recall|remind\s+me)\b.{{0,40}}\b(?:about\s+)?{_KN_SELF}\b"),
        ("personal_recall", "what do you know about me (EN)",
         rf"\bwhat\s+do\s+you\s+know\b.{{0,20}}\babout\s+(?:me|my|us|our)\b"),
        ("personal_recall", "what is my (EN)",
         r"\bwhat(?:'s| is| are)\s+my\b.{0,60}\b(?:name|birthday|preference[s]?|favorite|favourite|address|job|goal[s]?|plan[s]?|allerg(?:y|ies)|number)\b"),
        ("personal_recall", "what did I tell you (ID)",
         r"\b(?:apa\s+(?:yang\s+)?(?:pernah\s+|udah\s+|sudah\s+)?)?(?:aku|saya|ku)\b.{0,20}\b(?:bilang|cerita(?:kan|in)?|kasih\s+tau|sebut(?:kan)?|sampaikan)\b.{0,15}\b(?:ke\s+)?(?:kamu|kmu)\b"),
        ("personal_recall", "do you remember mine (ID)",
         r"\b(?:kamu\s+)?(?:masih\s+)?(?:inget|ingat)\b.{0,30}\b(?:gak|ga|nggak|tentang|soal|punya)?\b.{0,15}\b(?:aku|saya|ku|punyaku)\b"),
        ("personal_recall", "what is my (ID)",
         r"\bapa\b.{0,15}\b(?:nama|ulang\s+tahun|preferensi|kesukaan|alamat|pekerjaan|tujuan|rencana|nomor)\s*(?:ku|saya)\b"),

        # ---- web: user explicitly names the web as the source ----
        ("web", "search the web phrasing (EN)",
         r"\b(?:on|from|search)\s+the\s+(?:web|internet|google)\b"),
        ("web", "cari di internet (ID)",
         r"\b(?:cari|search|browsing)\b.{0,15}\b(?:di\s+)?(?:web|internet|google|online)\b"),
    )
)


def classify_knowledge_intent(text: str) -> KnowledgeIntent:
    """Classify which knowledge source should ground the answer.

    Conservative and deterministic. Returns source="none" (answer from the
    model's own knowledge) when nothing matches. Feature/how-to questions are
    treated as none so we don't route "how do I upload a document?" to RAG.
    """
    if not text:
        return KnowledgeIntent("none", "empty message")
    if _EXPLANATORY_PREFIX.search(text):
        return KnowledgeIntent("none", "explanatory feature question")
    for source, reason, pattern in _KNOWLEDGE_PATTERNS:
        if pattern.search(text):
            return KnowledgeIntent(source, reason)
    return KnowledgeIntent("none", "no knowledge-source pattern matched")


# Which agent tool backs each internal knowledge source. Used only to record
# the intended retrieval path and (observability-first) to make sure that tool
# is not accidentally withheld — never to force a call.
KNOWLEDGE_SOURCE_TOOL = {
    "personal_recall": "manage_memory",
    "past_chat": "search_chats",
    "personal_docs": "manage_documents",
    "web": "web_search",
    "none": None,
}


@dataclass(frozen=True)
class KnowledgeRoute:
    """Resolved knowledge-routing decision for one chat turn."""

    source: str
    reason: str
    backing_tool: str | None
    needs_retrieval: bool


def resolve_knowledge_route(
    text: str,
    *,
    tool_category: str | None = None,
    web_active: bool = False,
) -> KnowledgeRoute:
    """Resolve the knowledge source for a turn, reconciling with the tool axis.

    Observability-first: this only DESCRIBES where the answer should come from;
    it does not trigger retrieval. Reconciliation rules:

    * If the tool classifier already flagged the turn as ``web`` (or the manual
      web toggle is on), the effective source is ``web`` — an external lookup
      supersedes internal recall for freshness-sensitive queries.
    * Otherwise use the knowledge classifier's source.

    ``backing_tool`` names the tool that would serve the source, so the route
    layer can avoid withholding it. ``needs_retrieval`` is True only for the
    internal stores (memory / past chats / personal docs).
    """
    ki = classify_knowledge_intent(text)
    source = ki.source
    reason = ki.reason
    if tool_category == "web" or web_active:
        if source in ("none", "web"):
            source = "web"
            reason = (
                "web active supersedes model knowledge"
                if reason.startswith("no knowledge") or ki.source == "web"
                else reason
            )
    return KnowledgeRoute(
        source=source,
        reason=reason,
        backing_tool=KNOWLEDGE_SOURCE_TOOL.get(source),
        needs_retrieval=source in ("personal_recall", "past_chat", "personal_docs"),
    )
