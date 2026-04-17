#!/usr/bin/env python3
"""Pre-download all default models for VoxKitchen operators.

This script is run during Docker image build so that users get a
zero-download experience at runtime.

Usage:
    python scripts/warmup_models.py          # download all
    python scripts/warmup_models.py --skip-gpu   # skip large GPU-only models

Each model download is wrapped in try/except so one failure does not
block the rest.
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("warmup")


def _ok(name: str) -> None:
    log.info("  [OK] %s", name)


def _skip(name: str, reason: str) -> None:
    log.warning("  [SKIP] %s — %s", name, reason)


def _fail(name: str, e: Exception) -> None:
    log.error("  [FAIL] %s — %s", name, e)


# ---- Segmentation ----


def warmup_silero_vad() -> None:
    """Download Silero VAD via torch.hub (~2 MB)."""
    try:
        import torch

        torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        _ok("silero_vad")
    except Exception as e:
        _fail("silero_vad", e)


# ---- ASR ----


def warmup_faster_whisper() -> None:
    """Download faster-whisper tiny + base models (~150 MB total)."""
    try:
        from faster_whisper import WhisperModel

        for size in ("tiny", "base"):
            WhisperModel(size, device="cpu", compute_type="int8")
            _ok(f"faster_whisper ({size})")
    except ImportError:
        _skip("faster_whisper", "faster-whisper not installed")
    except Exception as e:
        _fail("faster_whisper", e)


def warmup_whisper_openai() -> None:
    """Download openai-whisper tiny model (~75 MB)."""
    try:
        import whisper

        whisper.load_model("tiny", device="cpu")
        _ok("whisper_openai (tiny)")
    except ImportError:
        _skip("whisper_openai", "openai-whisper not installed")
    except Exception as e:
        _fail("whisper_openai", e)


def warmup_paraformer() -> None:
    """Download Paraformer-large model (~1 GB)."""
    try:
        from funasr import AutoModel

        AutoModel(
            model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            device="cpu",
            disable_update=True,
        )
        _ok("paraformer_asr")
    except ImportError:
        _skip("paraformer_asr", "funasr not installed")
    except Exception as e:
        _fail("paraformer_asr", e)


def warmup_sensevoice() -> None:
    """Download SenseVoiceSmall model (~500 MB)."""
    try:
        from funasr import AutoModel

        AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)
        _ok("sensevoice_asr")
    except ImportError:
        _skip("sensevoice_asr", "funasr not installed")
    except Exception as e:
        _fail("sensevoice_asr", e)


def warmup_emotion() -> None:
    """Download emotion2vec_plus_large model (~300 MB)."""
    try:
        from funasr import AutoModel

        AutoModel(model="iic/emotion2vec_plus_large", device="cpu", disable_update=True)
        _ok("emotion_recognize")
    except ImportError:
        _skip("emotion_recognize", "funasr not installed")
    except Exception as e:
        _fail("emotion_recognize", e)


def warmup_wenet() -> None:
    """Download WeNet Chinese model."""
    try:
        import wenet

        wenet.load_model("chinese")
        _ok("wenet_asr (chinese)")
    except ImportError:
        _skip("wenet_asr", "wenet not installed")
    except Exception as e:
        _fail("wenet_asr", e)


def warmup_qwen3_asr() -> None:
    """Download Qwen3-ASR + ForcedAligner models (~1.2 GB each)."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download("Qwen/Qwen3-ASR-0.6B")
        _ok("qwen3_asr")
        snapshot_download("Qwen/Qwen3-ForcedAligner-0.6B")
        _ok("forced_align")
    except ImportError:
        _skip("qwen3_asr", "huggingface_hub not installed")
    except Exception as e:
        _fail("qwen3_asr", e)


# ---- Quality ----


def warmup_dnsmos() -> None:
    """Trigger speechmos ONNX model download (~20 MB)."""
    try:
        import numpy as np
        from speechmos import dnsmos

        # Run on a short silence to trigger model download
        dnsmos.run(np.zeros(16000, dtype=np.float32), sr=16000)
        _ok("dnsmos_score")
    except ImportError:
        _skip("dnsmos_score", "speechmos not installed")
    except Exception as e:
        _fail("dnsmos_score", e)


def warmup_utmos() -> None:
    """Trigger UTMOS model download (~80 MB)."""
    try:
        import numpy as np
        from speechmos import utmos

        utmos.run(np.zeros(16000, dtype=np.float32), sr=16000)
        _ok("utmos_score")
    except ImportError:
        _skip("utmos_score", "speechmos not installed")
    except Exception as e:
        _fail("utmos_score", e)


# ---- Speaker & Language ----


def warmup_speechbrain_langid() -> None:
    """Download SpeechBrain language ID model (~200 MB)."""
    try:
        from speechbrain.inference.classifiers import EncoderClassifier

        EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            run_opts={"device": "cpu"},
        )
        _ok("speechbrain_langid")
    except ImportError:
        _skip("speechbrain_langid", "speechbrain not installed")
    except Exception as e:
        _fail("speechbrain_langid", e)


def warmup_speaker_embed() -> None:
    """Download WeSpeaker model."""
    try:
        import wespeaker

        wespeaker.load_model("english")
        _ok("speaker_embed (english)")
    except ImportError:
        _skip("speaker_embed", "wespeaker not installed")
    except Exception as e:
        _fail("speaker_embed", e)


def warmup_pyannote(hf_token: str | None) -> None:
    """Download pyannote speaker-diarization-3.1 (~100 MB, needs HF_TOKEN)."""
    if not hf_token:
        _skip("pyannote_diarize", "HF_TOKEN not set")
        return
    try:
        import os

        os.environ["HF_TOKEN"] = hf_token
        from pyannote.audio import Pipeline

        Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        _ok("pyannote_diarize")
    except ImportError:
        _skip("pyannote_diarize", "pyannote.audio not installed")
    except Exception as e:
        _fail("pyannote_diarize", e)


# ---- Enhancement ----


def warmup_deepfilternet() -> None:
    """Download DeepFilterNet model (~20 MB)."""
    try:
        from df.enhance import init_df

        init_df()
        _ok("speech_enhance (deepfilternet)")
    except ImportError:
        _skip("speech_enhance", "deepfilternet not installed")
    except Exception as e:
        _fail("speech_enhance", e)


# ---- Codec ----


def warmup_encodec() -> None:
    """Download Encodec 24kHz model (~50 MB)."""
    try:
        from encodec import EncodecModel

        EncodecModel.encodec_model_24khz()
        _ok("codec_tokenize (encodec_24khz)")
    except ImportError:
        _skip("codec_tokenize", "encodec not installed")
    except Exception as e:
        _fail("codec_tokenize", e)


# ---- TTS ----


def warmup_kokoro() -> None:
    """Download Kokoro TTS model (~200 MB)."""
    try:
        from kokoro import KPipeline

        KPipeline(lang_code="a")
        _ok("tts_kokoro")
    except ImportError:
        _skip("tts_kokoro", "kokoro not installed")
    except Exception as e:
        _fail("tts_kokoro", e)


def warmup_chattts() -> None:
    """Download ChatTTS model (~800 MB)."""
    try:
        import ChatTTS

        chat = ChatTTS.Chat()
        chat.load(compile=False)
        _ok("tts_chattts")
    except ImportError:
        _skip("tts_chattts", "ChatTTS not installed")
    except Exception as e:
        _fail("tts_chattts", e)


def warmup_cosyvoice() -> None:
    """Download CosyVoice2 model via modelscope (~2 GB)."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download("FunAudioLLM/CosyVoice2-0.5B")
        _ok("tts_cosyvoice")
    except ImportError:
        _skip("tts_cosyvoice", "huggingface_hub not installed")
    except Exception as e:
        _fail("tts_cosyvoice", e)


def warmup_fish_speech() -> None:
    """Download Fish-Speech 1.5 model via huggingface (~2 GB)."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download("fishaudio/fish-speech-1.5")
        _ok("tts_fish_speech")
    except ImportError:
        _skip("tts_fish_speech", "huggingface_hub not installed")
    except Exception as e:
        _fail("tts_fish_speech", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download VoxKitchen operator models.")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for gated models (pyannote). "
        "Also reads from HF_TOKEN env var.",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip large GPU-only models (CosyVoice, Fish-Speech, etc.)",
    )
    args = parser.parse_args()

    import os

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    log.info("=== VoxKitchen model warmup ===")

    # Segmentation
    log.info("[Segmentation]")
    warmup_silero_vad()

    # ASR
    log.info("[ASR]")
    warmup_faster_whisper()
    warmup_whisper_openai()
    warmup_paraformer()
    warmup_sensevoice()
    warmup_wenet()
    if not args.skip_gpu:
        warmup_qwen3_asr()

    # Quality
    log.info("[Quality]")
    warmup_dnsmos()
    warmup_utmos()

    # Speaker & Language
    log.info("[Speaker & Language]")
    warmup_speechbrain_langid()
    warmup_speaker_embed()
    warmup_pyannote(hf_token)

    # Enhancement
    log.info("[Enhancement]")
    warmup_deepfilternet()

    # Codec
    log.info("[Codec]")
    warmup_encodec()

    # Emotion
    log.info("[Emotion]")
    warmup_emotion()

    # TTS
    log.info("[TTS]")
    warmup_kokoro()
    warmup_chattts()
    if not args.skip_gpu:
        warmup_cosyvoice()
        warmup_fish_speech()

    log.info("=== warmup complete ===")


if __name__ == "__main__":
    main()
