#!/usr/bin/env python3
"""Pre-download models for a specific VoxKitchen image group.

Each published Docker image (``core`` / ``asr`` / ``tts``) warms a different
subset of models so users don't pay a cold-cache download the first time
they run an operator.

Usage during image build::

    python scripts/warmup_models.py --group core
    python scripts/warmup_models.py --group asr
    python scripts/warmup_models.py --group tts

The ``HF_TOKEN`` env var is read for gated models (currently: pyannote in
the ``asr`` group). If not set, gated models are skipped — the image still
builds, and users can supply the token at ``docker run`` time via
``-e HF_TOKEN=hf_xxx``.

Each download is wrapped in try/except: one failure does not abort the
build. A summary is written to ``/app/warmup_status.json`` so ``vkit doctor``
can report which models are actually cached at runtime.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("warmup")


class WarmupReport:
    """Collects per-model outcomes so downstream tooling can surface them."""

    def __init__(self) -> None:
        self.ok: list[str] = []
        self.skipped: list[tuple[str, str]] = []
        self.failed: list[tuple[str, str]] = []

    def record_ok(self, name: str) -> None:
        log.info("  [OK] %s", name)
        self.ok.append(name)

    def record_skip(self, name: str, reason: str) -> None:
        log.warning("  [SKIP] %s — %s", name, reason)
        self.skipped.append((name, reason))

    def record_fail(self, name: str, exc: BaseException) -> None:
        log.error("  [FAIL] %s — %s", name, exc)
        self.failed.append((name, repr(exc)))

    def dump(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "ok": self.ok,
                    "skipped": [{"name": n, "reason": r} for n, r in self.skipped],
                    "failed": [{"name": n, "error": e} for n, e in self.failed],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Individual warmup functions. Each MUST catch ImportError (optional extras
# may be absent for a given image) and generic Exception (network / auth /
# model-format surprises) so one failure does not cascade.
# ---------------------------------------------------------------------------


def warmup_silero_vad(r: WarmupReport) -> None:
    try:
        import torch

        torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        r.record_ok("silero_vad")
    except ImportError:
        r.record_skip("silero_vad", "torch not installed")
    except Exception as e:
        r.record_fail("silero_vad", e)


def warmup_dnsmos(r: WarmupReport) -> None:
    try:
        import numpy as np
        from speechmos import dnsmos

        dnsmos.run(np.zeros(16000, dtype=np.float32), sr=16000)
        r.record_ok("dnsmos_score")
    except ImportError:
        r.record_skip("dnsmos_score", "speechmos not installed")
    except Exception as e:
        r.record_fail("dnsmos_score", e)


def warmup_utmos(r: WarmupReport) -> None:
    try:
        import numpy as np
        from speechmos import utmos

        utmos.run(np.zeros(16000, dtype=np.float32), sr=16000)
        r.record_ok("utmos_score")
    except ImportError:
        r.record_skip("utmos_score", "speechmos not installed")
    except Exception as e:
        r.record_fail("utmos_score", e)


def warmup_speechbrain_langid(r: WarmupReport) -> None:
    try:
        from speechbrain.inference.classifiers import EncoderClassifier

        EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            run_opts={"device": "cpu"},
        )
        r.record_ok("speechbrain_langid")
    except ImportError:
        r.record_skip("speechbrain_langid", "speechbrain not installed")
    except Exception as e:
        r.record_fail("speechbrain_langid", e)


def warmup_speaker_embed(r: WarmupReport) -> None:
    """Warm wespeaker's English model.

    Known issue: wespeaker -> s3prl ships a dataclass with a mutable
    default that Python 3.11+ rejects. Until s3prl ships a fix (see
    https://github.com/s3prl/s3prl/issues), this warmup will record
    a FAIL and the speaker_embed operator will be unusable at runtime.
    Keep the call so doctor surfaces the gap; don't let it fail the build.
    """
    try:
        import wespeaker

        wespeaker.load_model("english")
        r.record_ok("speaker_embed")
    except ImportError:
        r.record_skip("speaker_embed", "wespeaker not installed")
    except Exception as e:
        r.record_fail("speaker_embed", e)


def warmup_deepfilternet(r: WarmupReport) -> None:
    try:
        from df.enhance import init_df

        init_df()
        r.record_ok("speech_enhance")
    except ImportError:
        r.record_skip("speech_enhance", "deepfilternet not installed")
    except Exception as e:
        r.record_fail("speech_enhance", e)


def warmup_encodec(r: WarmupReport) -> None:
    try:
        from encodec import EncodecModel

        EncodecModel.encodec_model_24khz()
        r.record_ok("codec_tokenize")
    except ImportError:
        r.record_skip("codec_tokenize", "encodec not installed")
    except Exception as e:
        r.record_fail("codec_tokenize", e)


def warmup_faster_whisper(r: WarmupReport) -> None:
    try:
        from faster_whisper import WhisperModel

        for size in ("tiny", "base"):
            WhisperModel(size, device="cpu", compute_type="int8")
            r.record_ok(f"faster_whisper_asr:{size}")
    except ImportError:
        r.record_skip("faster_whisper_asr", "faster-whisper not installed")
    except Exception as e:
        r.record_fail("faster_whisper_asr", e)


def warmup_whisper_openai(r: WarmupReport) -> None:
    try:
        import whisper

        whisper.load_model("tiny", device="cpu")
        r.record_ok("whisper_openai_asr:tiny")
    except ImportError:
        r.record_skip("whisper_openai_asr", "openai-whisper not installed")
    except Exception as e:
        r.record_fail("whisper_openai_asr", e)


def warmup_paraformer(r: WarmupReport) -> None:
    try:
        from funasr import AutoModel

        AutoModel(
            model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            device="cpu",
            disable_update=True,
        )
        r.record_ok("paraformer_asr")
    except ImportError:
        r.record_skip("paraformer_asr", "funasr not installed")
    except Exception as e:
        r.record_fail("paraformer_asr", e)


def warmup_sensevoice(r: WarmupReport) -> None:
    try:
        from funasr import AutoModel

        AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)
        r.record_ok("sensevoice_asr")
    except ImportError:
        r.record_skip("sensevoice_asr", "funasr not installed")
    except Exception as e:
        r.record_fail("sensevoice_asr", e)


def warmup_emotion(r: WarmupReport) -> None:
    try:
        from funasr import AutoModel

        AutoModel(model="iic/emotion2vec_plus_large", device="cpu", disable_update=True)
        r.record_ok("emotion_recognize")
    except ImportError:
        r.record_skip("emotion_recognize", "funasr not installed")
    except Exception as e:
        r.record_fail("emotion_recognize", e)


def warmup_wenet(r: WarmupReport) -> None:
    try:
        import wenet

        wenet.load_model("chinese")
        r.record_ok("wenet_asr")
    except ImportError:
        r.record_skip("wenet_asr", "wenet not installed")
    except Exception as e:
        r.record_fail("wenet_asr", e)


def warmup_qwen3_asr(r: WarmupReport) -> None:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download("Qwen/Qwen3-ASR-0.6B")
        r.record_ok("qwen3_asr")
        snapshot_download("Qwen/Qwen3-ForcedAligner-0.6B")
        r.record_ok("forced_align")
    except ImportError:
        r.record_skip("qwen3_asr", "huggingface_hub not installed")
    except Exception as e:
        r.record_fail("qwen3_asr", e)


def warmup_pyannote(r: WarmupReport, hf_token: str | None) -> None:
    """Download pyannote speaker-diarization-3.1 AND its two sub-model repos.

    speaker-diarization-3.1 is a pipeline config (few KB) that references
    two separate gated repos for its segmentation and embedding weights.
    ``Pipeline.from_pretrained`` in recent pyannote + huggingface_hub
    combos does not reliably pre-fetch those (observed: it returns an
    initialized Pipeline object while the weight download silently errors
    out via the HF Xet backend). Using ``snapshot_download`` against each
    repo explicitly is the most deterministic way to get everything on
    disk at build time.
    """
    if not hf_token:
        r.record_skip(
            "pyannote_diarize",
            "HF_TOKEN not provided at build time — user must pass it at runtime",
        )
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        r.record_skip("pyannote_diarize", "huggingface_hub not installed")
        return

    repos = [
        "pyannote/speaker-diarization-3.1",
        "pyannote/segmentation-3.0",
        "pyannote/wespeaker-voxceleb-resnet34-LM",
    ]
    for repo in repos:
        try:
            snapshot_download(repo, token=hf_token)
        except Exception as e:  # noqa: BLE001
            r.record_fail(f"pyannote_diarize:{repo}", e)
            return
    r.record_ok("pyannote_diarize")


def warmup_kokoro(r: WarmupReport) -> None:
    try:
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code="a")
        # Drive one short synthesis to pull any lazy-loaded NLP resources
        # (spacy en_core_web_sm, phonemizer dictionaries, etc.) at build
        # time — not at the user's first invocation.
        for _ in pipeline("warm up", voice="af_heart"):
            pass
        r.record_ok("tts_kokoro")
    except ImportError:
        r.record_skip("tts_kokoro", "kokoro not installed")
    except Exception as e:
        r.record_fail("tts_kokoro", e)


def warmup_chattts(r: WarmupReport) -> None:
    try:
        import ChatTTS

        chat = ChatTTS.Chat()
        chat.load(compile=False)
        r.record_ok("tts_chattts")
    except ImportError:
        r.record_skip("tts_chattts", "ChatTTS not installed")
    except Exception as e:
        r.record_fail("tts_chattts", e)


def warmup_cosyvoice(r: WarmupReport) -> None:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download("FunAudioLLM/CosyVoice2-0.5B")
        r.record_ok("tts_cosyvoice")
    except ImportError:
        r.record_skip("tts_cosyvoice", "huggingface_hub not installed")
    except Exception as e:
        r.record_fail("tts_cosyvoice", e)


def warmup_fish_speech(r: WarmupReport) -> None:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download("fishaudio/fish-speech-1.5")
        r.record_ok("tts_fish_speech")
    except ImportError:
        r.record_skip("tts_fish_speech", "huggingface_hub not installed")
    except Exception as e:
        r.record_fail("tts_fish_speech", e)


# ---------------------------------------------------------------------------
# Image groups. Must stay in sync with the expected-operator set used by
# ``vkit doctor --expect <group>``.
# ---------------------------------------------------------------------------


def run_core(r: WarmupReport) -> None:
    log.info("[core] Segmentation")
    warmup_silero_vad(r)
    log.info("[core] Quality")
    warmup_dnsmos(r)
    warmup_utmos(r)
    log.info("[core] Classify / Enhance / Codec")
    warmup_speechbrain_langid(r)
    warmup_speaker_embed(r)
    warmup_deepfilternet(r)
    warmup_encodec(r)


def run_asr(r: WarmupReport, hf_token: str | None) -> None:  # noqa: ARG001
    # ASR image ships core extras too, so start from core.
    run_core(r)
    log.info("[asr] ASR")
    warmup_faster_whisper(r)
    warmup_whisper_openai(r)
    warmup_paraformer(r)
    warmup_sensevoice(r)
    warmup_emotion(r)
    warmup_wenet(r)
    warmup_qwen3_asr(r)


def run_diarize(r: WarmupReport, hf_token: str | None) -> None:
    # Diarize image inherits core extras; also warm core models so
    # VAD / quality ops can be used as pre/post stages here.
    run_core(r)
    log.info("[diarize] Diarization")
    warmup_pyannote(r, hf_token)


def run_tts(r: WarmupReport) -> None:
    run_core(r)
    log.info("[tts] TTS (kokoro / ChatTTS / CosyVoice)")
    warmup_kokoro(r)
    warmup_chattts(r)
    warmup_cosyvoice(r)


def run_fish_speech(r: WarmupReport) -> None:
    # Isolated env: no core warmup here, just fish-speech itself.
    log.info("[fish-speech] TTS")
    warmup_fish_speech(r)


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        required=True,
        choices=("core", "asr", "diarize", "tts", "fish-speech"),
        help="Which image group to warm. Must match the Dockerfile building this image.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for gated models (pyannote). Also reads HF_TOKEN env var.",
    )
    parser.add_argument(
        "--status-file",
        default=None,
        help="Where to write per-model outcomes as JSON. "
        "Defaults to /opt/voxkitchen/warmup_<group>.json so multi-env builds "
        "don't clobber each other.",
    )
    args = parser.parse_args()
    if args.status_file is None:
        args.status_file = f"/opt/voxkitchen/warmup_{args.group}.json"

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    r = WarmupReport()

    log.info("=== VoxKitchen warmup: group=%s ===", args.group)
    if args.group == "core":
        run_core(r)
    elif args.group == "asr":
        run_asr(r, hf_token)
    elif args.group == "diarize":
        run_diarize(r, hf_token)
    elif args.group == "tts":
        run_tts(r)
    elif args.group == "fish-speech":
        run_fish_speech(r)

    status_path = Path(args.status_file)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    r.dump(status_path)

    log.info(
        "=== warmup complete: %d ok, %d skipped, %d failed (status → %s) ===",
        len(r.ok),
        len(r.skipped),
        len(r.failed),
        status_path,
    )

    # We intentionally do NOT exit non-zero on failures here — model downloads
    # can fail for transient network reasons and that should not block the
    # image build. `vkit doctor` at the end of the Dockerfile is the real gate.
    return 0


if __name__ == "__main__":
    sys.exit(main())
