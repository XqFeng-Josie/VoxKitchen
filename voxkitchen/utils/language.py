"""Language code normalization.

Single source of truth for converting language codes and names from any
operator or model output into a canonical lowercase full name.

Examples::

    normalize_language("zh")       -> "chinese"
    normalize_language("Zh")       -> "chinese"
    normalize_language("ZH-CN")    -> "chinese"
    normalize_language("Chinese")  -> "chinese"
    normalize_language("en")       -> "english"
    normalize_language(None)       -> None
    normalize_language("unknown")  -> None

Unknown inputs (not in the alias table) are returned as ``None`` so callers
always get a clean, intentional value or nothing.
"""

from __future__ import annotations

# Keys are pre-normalized (lowercased, hyphens/underscores/spaces removed).
# Values are canonical lowercase full language names.
_ALIASES: dict[str, str] = {
    # ── Chinese ───────────────────────────────────────────────────────────
    "zh": "chinese",
    "zhcn": "chinese",
    "zhhans": "chinese",
    "cmn": "chinese",
    "mandarin": "chinese",
    "chinese": "chinese",
    # ── Cantonese ─────────────────────────────────────────────────────────
    "yue": "cantonese",
    "zhyue": "cantonese",
    "zhhant": "cantonese",
    "zhhk": "cantonese",
    "zhtw": "cantonese",
    "cantonese": "cantonese",
    # ── English ───────────────────────────────────────────────────────────
    "en": "english",
    "enus": "english",
    "engb": "english",
    "english": "english",
    # ── Arabic ────────────────────────────────────────────────────────────
    "ar": "arabic",
    "arabic": "arabic",
    # ── German ────────────────────────────────────────────────────────────
    "de": "german",
    "german": "german",
    # ── French ────────────────────────────────────────────────────────────
    "fr": "french",
    "french": "french",
    # ── Spanish ───────────────────────────────────────────────────────────
    "es": "spanish",
    "spanish": "spanish",
    # ── Portuguese ────────────────────────────────────────────────────────
    "pt": "portuguese",
    "ptbr": "portuguese",
    "portuguese": "portuguese",
    # ── Indonesian ────────────────────────────────────────────────────────
    "id": "indonesian",
    "indonesian": "indonesian",
    # ── Italian ───────────────────────────────────────────────────────────
    "it": "italian",
    "italian": "italian",
    # ── Japanese ──────────────────────────────────────────────────────────
    "ja": "japanese",
    "japanese": "japanese",
    # ── Korean ────────────────────────────────────────────────────────────
    "ko": "korean",
    "korean": "korean",
    # ── Russian ───────────────────────────────────────────────────────────
    "ru": "russian",
    "russian": "russian",
    # ── Thai ──────────────────────────────────────────────────────────────
    "th": "thai",
    "thai": "thai",
    # ── Vietnamese ────────────────────────────────────────────────────────
    "vi": "vietnamese",
    "vietnamese": "vietnamese",
    # ── Turkish ───────────────────────────────────────────────────────────
    "tr": "turkish",
    "turkish": "turkish",
    # ── Hindi ─────────────────────────────────────────────────────────────
    "hi": "hindi",
    "hindi": "hindi",
    # ── Malay ─────────────────────────────────────────────────────────────
    "ms": "malay",
    "malay": "malay",
    # ── Dutch ─────────────────────────────────────────────────────────────
    "nl": "dutch",
    "dutch": "dutch",
    # ── Swedish ───────────────────────────────────────────────────────────
    "sv": "swedish",
    "swedish": "swedish",
    # ── Danish ────────────────────────────────────────────────────────────
    "da": "danish",
    "danish": "danish",
    # ── Finnish ───────────────────────────────────────────────────────────
    "fi": "finnish",
    "finnish": "finnish",
    # ── Polish ────────────────────────────────────────────────────────────
    "pl": "polish",
    "polish": "polish",
    # ── Czech ─────────────────────────────────────────────────────────────
    "cs": "czech",
    "czech": "czech",
    # ── Filipino / Tagalog ────────────────────────────────────────────────
    "fil": "filipino",
    "tl": "filipino",
    "filipino": "filipino",
    "tagalog": "filipino",
    # ── Persian / Farsi ───────────────────────────────────────────────────
    "fa": "persian",
    "persian": "persian",
    "farsi": "persian",
    # ── Greek ─────────────────────────────────────────────────────────────
    "el": "greek",
    "greek": "greek",
    # ── Romanian ──────────────────────────────────────────────────────────
    "ro": "romanian",
    "romanian": "romanian",
    # ── Hungarian ─────────────────────────────────────────────────────────
    "hu": "hungarian",
    "hungarian": "hungarian",
    # ── Macedonian ────────────────────────────────────────────────────────
    "mk": "macedonian",
    "macedonian": "macedonian",
    # ── Ukrainian ─────────────────────────────────────────────────────────
    "uk": "ukrainian",
    "ukrainian": "ukrainian",
    # ── Norwegian ─────────────────────────────────────────────────────────
    "no": "norwegian",
    "nb": "norwegian",
    "nn": "norwegian",
    "norwegian": "norwegian",
    # ── Catalan ───────────────────────────────────────────────────────────
    "ca": "catalan",
    "catalan": "catalan",
    # ── Croatian ──────────────────────────────────────────────────────────
    "hr": "croatian",
    "croatian": "croatian",
    # ── Slovak ────────────────────────────────────────────────────────────
    "sk": "slovak",
    "slovak": "slovak",
    # ── Bulgarian ─────────────────────────────────────────────────────────
    "bg": "bulgarian",
    "bulgarian": "bulgarian",
    # ── Serbian ───────────────────────────────────────────────────────────
    "sr": "serbian",
    "serbian": "serbian",
    # ── Hebrew ────────────────────────────────────────────────────────────
    "he": "hebrew",
    "iw": "hebrew",
    "hebrew": "hebrew",
    # ── Swahili ───────────────────────────────────────────────────────────
    "sw": "swahili",
    "swahili": "swahili",
    # ── Welsh ─────────────────────────────────────────────────────────────
    "cy": "welsh",
    "welsh": "welsh",
    # ── Afrikaans ─────────────────────────────────────────────────────────
    "af": "afrikaans",
    "afrikaans": "afrikaans",
}


def normalize_language(lang: str | None) -> str | None:
    """Return the canonical lowercase full language name, or ``None``.

    Accepts ISO 639-1 codes, BCP-47 tags, and full English names in any
    capitalisation. Returns ``None`` for ``None`` input and for any value
    not in the alias table (e.g. ``"unknown"``, ``"auto"``).

    The canonical form is the lowercase full English name (``"chinese"``,
    ``"english"``, etc.) — consistent with how the field is stored in
    ``Supervision.language``.
    """
    if not lang:
        return None
    key = lang.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    return _ALIASES.get(key)
