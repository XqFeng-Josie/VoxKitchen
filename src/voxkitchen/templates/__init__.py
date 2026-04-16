"""Built-in pipeline templates for common speech processing scenarios."""

from __future__ import annotations

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent

TEMPLATES: dict[str, dict[str, str]] = {
    "tts": {
        "file": "tts-data-prep.yaml",
        "description": "TTS data preparation: denoise → VAD → quality filter → ASR → alignment",
    },
    "asr": {
        "file": "asr-training-data.yaml",
        "description": "ASR training data: VAD → augmentation (speed+volume) → ASR → filter → pack",
    },
    "cleaning": {
        "file": "data-cleaning.yaml",
        "description": "Data cleaning: quality metrics → dedup → filter out bad audio",
    },
    "speaker": {
        "file": "speaker-analysis.yaml",
        "description": "Speaker analysis: VAD → diarization → speaker embedding → gender → pack",
    },
}


def get_template_content(name: str) -> str:
    """Return the YAML content of a built-in template."""
    info = TEMPLATES.get(name)
    if info is None:
        available = ", ".join(sorted(TEMPLATES.keys()))
        raise KeyError(f"unknown template: {name!r}. Available: {available}")
    path = TEMPLATES_DIR / info["file"]
    return path.read_text(encoding="utf-8")
