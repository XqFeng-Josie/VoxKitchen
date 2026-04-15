#!/usr/bin/env python3
"""Verify that the new operators (speaker_embed, speech_enhance, forced_align) work correctly.

Usage:
    # 1. Install the required extras
    pip install -e ".[speaker]"     # WeSpeaker for speaker_embed
    pip install -e ".[enhance]"     # DeepFilterNet for speech_enhance
    pip install -e ".[align]"       # ctc-forced-aligner for forced_align

    # 2. Run this script with a test wav file
    python examples/scripts/verify_new_operators.py path/to/audio.wav

    # 3. Or run the slow tests directly
    pytest -v -m slow --tb=short

Each section is independent — if a package is missing, that section is skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def verify_speaker_embed(audio_path: str) -> None:
    _separator("speaker_embed (WeSpeaker)")
    try:
        from voxkitchen.tools import extract_speaker_embedding

        emb = extract_speaker_embedding(audio_path, method="wespeaker")
        print(f"  Embedding dim: {len(emb)}")
        print(f"  First 5 values: {emb[:5]}")
        print(f"  OK")
    except ImportError as e:
        print(f"  SKIP: {e}")
        print(f"  Install: pip install -e '.[speaker]'")
    except Exception as e:
        print(f"  FAIL: {e}")


def verify_speaker_embed_speechbrain(audio_path: str) -> None:
    _separator("speaker_embed (SpeechBrain)")
    try:
        from voxkitchen.tools import extract_speaker_embedding

        emb = extract_speaker_embedding(
            audio_path, method="speechbrain", model="speechbrain/spkrec-ecapa-voxceleb"
        )
        print(f"  Embedding dim: {len(emb)}")
        print(f"  First 5 values: {emb[:5]}")
        print(f"  OK")
    except ImportError as e:
        print(f"  SKIP: {e}")
        print(f"  Install: pip install -e '.[classify]'")
    except Exception as e:
        print(f"  FAIL: {e}")


def verify_speech_enhance(audio_path: str) -> None:
    _separator("speech_enhance (DeepFilterNet)")
    try:
        from voxkitchen.tools import enhance_speech

        out = Path("/tmp/vkit_verify_enhanced.wav")
        enhance_speech(audio_path, str(out), aggressiveness=0.5)
        print(f"  Output: {out} ({out.stat().st_size} bytes)")
        print(f"  OK")
    except ImportError as e:
        print(f"  SKIP: {e}")
        print(f"  Install: pip install -e '.[enhance]'")
    except Exception as e:
        print(f"  FAIL: {e}")


def verify_forced_align(audio_path: str) -> None:
    _separator("forced_align (ctc-forced-aligner)")
    try:
        from voxkitchen.tools import align_words, transcribe

        # First get text via ASR
        print("  Step 1: Transcribing...")
        segments = transcribe(audio_path, model="tiny")
        text = " ".join(s.text for s in segments if s.text)
        print(f"  ASR text: {text!r}")

        if not text.strip():
            print("  SKIP: no text from ASR (try a file with speech)")
            return

        # Then align
        print("  Step 2: Aligning words...")
        words = align_words(audio_path, text)
        print(f"  Word count: {len(words)}")
        for w in words[:5]:
            print(f"    {w['text']:15s} {w['start']:.3f}s - {w['end']:.3f}s")
        if len(words) > 5:
            print(f"    ... ({len(words) - 5} more)")
        print(f"  OK")
    except ImportError as e:
        print(f"  SKIP: {e}")
        print(f"  Install: pip install -e '.[asr,align]'")
    except Exception as e:
        print(f"  FAIL: {e}")


def verify_reverb_augment(audio_path: str) -> None:
    _separator("reverb_augment (scipy)")
    try:
        import tempfile

        import numpy as np
        import soundfile as sf
        from scipy.signal import fftconvolve  # noqa: F401

        from voxkitchen.tools import resample_audio

        # Create a synthetic RIR
        rir_dir = Path(tempfile.mkdtemp(prefix="vkit-rir-"))
        rir = np.zeros(8000, dtype=np.float32)
        rir[0] = 1.0
        rir[160] = 0.4
        sf.write(str(rir_dir / "test_rir.wav"), rir, 16000)

        # Resample input to 16k first
        tmp_16k = Path(tempfile.mktemp(suffix=".wav"))
        resample_audio(audio_path, str(tmp_16k), target_sr=16000)

        # Run reverb via operator directly
        from voxkitchen.operators.augment.reverb_augment import (
            ReverbAugmentConfig,
            ReverbAugmentOperator,
        )
        from voxkitchen.tools import _make_ctx, _make_cut
        from voxkitchen.schema.cutset import CutSet

        cut = _make_cut(tmp_16k)
        ctx = _make_ctx()
        config = ReverbAugmentConfig(rir_dir=str(rir_dir))
        op = ReverbAugmentOperator(config, ctx)
        op.setup()
        result = op.process(CutSet([cut]))
        op.teardown()
        out_cut = next(iter(result))
        print(f"  Output duration: {out_cut.duration:.2f}s")
        print(f"  RIR file: {out_cut.custom.get('rir_file')}")
        print(f"  OK")
    except ImportError as e:
        print(f"  SKIP: {e}")
    except Exception as e:
        print(f"  FAIL: {e}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python verify_new_operators.py <audio.wav>")
        print("       Provide any speech .wav file to test with.")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not Path(audio_path).exists():
        print(f"Error: file not found: {audio_path}")
        sys.exit(1)

    print(f"Testing with: {audio_path}")

    verify_reverb_augment(audio_path)
    verify_speaker_embed(audio_path)
    verify_speaker_embed_speechbrain(audio_path)
    verify_speech_enhance(audio_path)
    verify_forced_align(audio_path)

    print(f"\n{'=' * 60}")
    print("  Done. Fix any FAIL items above.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
