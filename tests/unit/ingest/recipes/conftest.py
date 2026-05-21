"""Shared fixtures for recipe unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def mock_librispeech(tmp_path: Path) -> Path:
    """Create a tiny LibriSpeech-like directory with 2 utterances."""
    subset = tmp_path / "train-clean-100" / "1089" / "134686"
    subset.mkdir(parents=True)
    audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32) * 0.5
    for utt_id in ["0001", "0002"]:
        sf.write(subset / f"1089-134686-{utt_id}.flac", audio, 16000)
    (subset / "1089-134686.trans.txt").write_text(
        "1089-134686-0001 HELLO WORLD\n1089-134686-0002 GOODBYE WORLD\n"
    )
    return tmp_path


@pytest.fixture
def mock_commonvoice(tmp_path: Path) -> Path:
    """Create a tiny CommonVoice-like directory with 2 utterances."""
    clips = tmp_path / "clips"
    clips.mkdir()
    audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32) * 0.5
    for name in ["cv_en_001.wav", "cv_en_002.wav"]:
        sf.write(clips / name, audio, 16000)
    (tmp_path / "train.tsv").write_text(
        "client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccent\tlocale\tsegment\n"
        "client1\tcv_en_001.wav\thello world\t5\t0\t\tmale_masculine\t\ten\t\n"
        "client2\tcv_en_002.wav\tgoodbye world\t3\t1\t\tfemale_feminine\t\ten\t\n"
    )
    return tmp_path


@pytest.fixture
def mock_aishell(tmp_path: Path) -> Path:
    """Create a tiny AISHELL-1-like directory with 2 utterances."""
    wav_dir = tmp_path / "data_aishell" / "wav" / "train" / "S0001"
    wav_dir.mkdir(parents=True)
    audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32) * 0.5
    for name in ["BAC001.wav", "BAC002.wav"]:
        sf.write(wav_dir / name, audio, 16000)
    trans_dir = tmp_path / "data_aishell" / "transcript"
    trans_dir.mkdir(parents=True)
    (trans_dir / "aishell_transcript_v0.8.txt").write_text(
        "BAC001 你 好 世 界\nBAC002 再 见 世 界\n"
    )
    return tmp_path


@pytest.fixture
def mock_ljspeech(tmp_path: Path) -> Path:
    """Create a tiny LJSpeech-1.1-like directory with 2 utterances.

    Mirrors the real layout: a single ``metadata.csv`` (pipe-separated, no
    header, three columns per row) plus a flat ``wavs/`` directory.
    """
    ls_root = tmp_path / "LJSpeech-1.1"
    wavs = ls_root / "wavs"
    wavs.mkdir(parents=True)
    audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32) * 0.5
    for utt in ["LJ001-0001", "LJ001-0002"]:
        sf.write(wavs / f"{utt}.wav", audio, 16000)
    # Row 1 has identical raw/normalized; row 2 differs (normalization
    # expanded "Mr." to "Mister") so the recipe's preference for the
    # normalized column shows up in tests.
    (ls_root / "metadata.csv").write_text(
        "LJ001-0001|Hello world.|Hello world.\nLJ001-0002|Hi Mr. Smith.|Hi Mister Smith.\n",
        encoding="utf-8",
    )
    return tmp_path
