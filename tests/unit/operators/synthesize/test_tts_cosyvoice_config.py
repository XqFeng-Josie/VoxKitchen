"""Config-only tests for tts_cosyvoice that run without the cosyvoice package.

The main test_tts_cosyvoice.py skips the whole module when cosyvoice isn't
importable (it's a Docker-only dep), so it can't guard the config defaults in
CI. These tests import only the pydantic Config (cosyvoice is lazy-imported
inside the operator's setup()), so they run everywhere.
"""

from __future__ import annotations

from voxkitchen.operators.synthesize.tts_cosyvoice import TtsCosyVoiceConfig


def test_default_model_id_is_the_live_modelscope_repo() -> None:
    """The default model_id must point at a modelscope repo that exists.

    Regression guard for 2026-06-05: the old default `FunAudioLLM/CosyVoice2-0.5B`
    was removed from modelscope (404 on both .cn and .ai endpoints), making the
    operator fail at setup() for anyone relying on the default. The weights moved
    under the `iic` org. Verified end-to-end against the rebuilt tts image
    (24 kHz / 11.2 s WAV produced).
    """
    assert TtsCosyVoiceConfig().model_id == "iic/CosyVoice2-0.5B", (
        "CosyVoice2 weights live under the `iic` modelscope org; if this default "
        "changes, confirm the new repo actually exists on modelscope (the old "
        "FunAudioLLM/... repo 404s)."
    )


def test_default_mode_is_sft() -> None:
    """Sanity-pin the other defaults so an accidental edit is visible."""
    cfg = TtsCosyVoiceConfig()
    assert cfg.mode == "sft"
    assert cfg.spk_id == "default"
    assert cfg.reference_audio is None
