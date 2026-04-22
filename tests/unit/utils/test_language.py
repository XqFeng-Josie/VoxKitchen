"""Unit tests for voxkitchen.utils.language."""

from __future__ import annotations

import pytest
from voxkitchen.utils.language import normalize_language


@pytest.mark.parametrize(
    "inp, expected",
    [
        # None / empty
        (None, None),
        ("", None),
        ("  ", None),
        # Unknown / sentinel values → None
        ("unknown", None),
        ("auto", None),
        ("und", None),
        # Chinese — ISO codes
        ("zh", "chinese"),
        ("ZH", "chinese"),
        ("Zh", "chinese"),
        ("zh-CN", "chinese"),
        ("zh_cn", "chinese"),
        ("cmn", "chinese"),
        # Chinese — full names
        ("Chinese", "chinese"),
        ("CHINESE", "chinese"),
        ("chinese", "chinese"),
        ("mandarin", "chinese"),
        # Cantonese
        ("yue", "cantonese"),
        ("zh-yue", "cantonese"),
        ("Cantonese", "cantonese"),
        # English
        ("en", "english"),
        ("EN", "english"),
        ("English", "english"),
        ("english", "english"),
        # Arabic
        ("ar", "arabic"),
        ("Arabic", "arabic"),
        # Japanese
        ("ja", "japanese"),
        ("Japanese", "japanese"),
        # Korean
        ("ko", "korean"),
        # Filipino / Tagalog
        ("fil", "filipino"),
        ("tl", "filipino"),
        ("Filipino", "filipino"),
        ("tagalog", "filipino"),
        # Persian / Farsi
        ("fa", "persian"),
        ("farsi", "persian"),
        # Hebrew legacy code
        ("iw", "hebrew"),
        # Norwegian variants
        ("no", "norwegian"),
        ("nb", "norwegian"),
        ("nn", "norwegian"),
        # Portuguese with region
        ("pt-BR", "portuguese"),
    ],
)
def test_normalize_language(inp: str | None, expected: str | None) -> None:
    assert normalize_language(inp) == expected
