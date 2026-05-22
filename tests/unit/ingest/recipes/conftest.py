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


@pytest.fixture
def mock_aishell3(tmp_path: Path) -> Path:
    """Create a tiny AISHELL-3-like directory.

    Two speakers, three utterances total. ``content.txt`` uses the
    interleaved character + pinyin format of the real corpus, and
    ``spk-info.txt`` carries gender for one of the two speakers (the
    other is intentionally missing so we can verify graceful fallback).
    """
    ds_root = tmp_path / "data_aishell3"

    # spk-info.txt — tab-separated speaker metadata
    (ds_root).mkdir(parents=True)
    (ds_root / "spk-info.txt").write_text(
        "SSB0005\t25\tmale\tNorth\nSSB0009\t30\tfemale\tSouth\n",
        encoding="utf-8",
    )

    # train subset — two speakers, three utterances total
    train_dir = ds_root / "train"
    wav_root = train_dir / "wav"
    audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32) * 0.5
    for spk, utt in [("SSB0005", "0001"), ("SSB0005", "0002"), ("SSB0009", "0001")]:
        spk_dir = wav_root / spk
        spk_dir.mkdir(parents=True, exist_ok=True)
        sf.write(spk_dir / f"{spk}{utt}.wav", audio, 16000)

    (train_dir / "content.txt").write_text(
        "SSB00050001.wav\t你 ni3 好 hao3 世 shi4 界 jie4\n"
        "SSB00050002.wav\t再 zai4 见 jian4\n"
        "SSB00090001.wav\t早 zao3 上 shang4 好 hao3\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def mock_libritts(tmp_path: Path) -> Path:
    """Create a tiny LibriTTS-like directory under <tmp>/LibriTTS/.

    Two speakers, two utterances in train-clean-100. One utterance has
    both a normalized and an original transcript (normalized must win);
    the other has only the original (so we exercise the fallback path).
    The optional speakers.tsv is included to verify gender enrichment
    when the file is present.
    """
    lt_root = tmp_path / "LibriTTS"
    subset_dir = lt_root / "train-clean-100"
    audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32) * 0.5

    # Speaker 1089, chapter 134686 — has both normalized and original text
    ch1 = subset_dir / "1089" / "134686"
    ch1.mkdir(parents=True)
    sf.write(ch1 / "1089_134686_000001_000000.wav", audio, 16000)
    (ch1 / "1089_134686_000001_000000.normalized.txt").write_text("Hello, world.", encoding="utf-8")
    (ch1 / "1089_134686_000001_000000.original.txt").write_text("HELLO WORLD", encoding="utf-8")

    # Speaker 2289, chapter 200000 — only original text (no normalized file)
    ch2 = subset_dir / "2289" / "200000"
    ch2.mkdir(parents=True)
    sf.write(ch2 / "2289_200000_000001_000000.wav", audio, 16000)
    (ch2 / "2289_200000_000001_000000.original.txt").write_text("Goodbye world.", encoding="utf-8")

    # Header + two readers (1089 female, 2289 male) to verify gender lookup
    (lt_root / "speakers.tsv").write_text(
        "READER\tGENDER\tSUBSET\tNAME\n"
        "1089\tF\ttrain-clean-100\tFirst Reader\n"
        "2289\tM\ttrain-clean-100\tSecond Reader\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def mock_musan(tmp_path: Path) -> Path:
    """Create a tiny MUSAN-like directory with one file per major category.

    Mirrors the real layout: ``musan/{noise,music,speech}/<subcategory>/*.wav``.
    Two files in `noise/` (different subcategories) so we can verify the
    subcategory tag survives; one each in music and speech to confirm
    other categories produce cuts too.
    """
    ds = tmp_path / "musan"
    audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32) * 0.5
    fixtures = [
        ("noise", "free-sound", "noise-free-sound-0000.wav"),
        ("noise", "sound-bible", "noise-sound-bible-0001.wav"),
        ("music", "fma", "music-fma-0000.wav"),
        ("speech", "librivox", "speech-librivox-0000.wav"),
    ]
    for cat, subcat, name in fixtures:
        d = ds / cat / subcat
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / name, audio, 16000)
    return tmp_path


@pytest.fixture
def mock_tedlium3(tmp_path: Path) -> Path:
    """Create a tiny TED-LIUM-3-like tree under <tmp>/TEDLIUM_release-3/.

    libsndfile (which ``soundfile`` wraps) detects audio format from the
    file *content*, not the extension, so writing standard WAV data into
    files named ``*.sph`` lets the recipe's ``recording_from_file``
    succeed without depending on the host's libsndfile being built with
    SPHERE support. The recipe only relies on sample-rate / duration
    metadata, which is produced identically for both formats.

    Two talks: one in ``train/`` whose STM exercises every padding /
    valid branch of the STM parser, and one in ``dev/`` so multi-subset
    iteration also has coverage.
    """
    ds = tmp_path / "TEDLIUM_release-3"
    legacy = ds / "legacy"
    audio = np.sin(np.linspace(0, 1, 16000 * 30)).astype(np.float32) * 0.5  # 30s "talk"

    for subset, talk in [("train", "AaronHuey_2010X"), ("dev", "BenSaunders_2014")]:
        sph_dir = legacy / subset / "sph"
        stm_dir = legacy / subset / "stm"
        sph_dir.mkdir(parents=True, exist_ok=True)
        stm_dir.mkdir(parents=True, exist_ok=True)
        # WAV bytes named .sph — libsndfile sniffs by content on read, but
        # sf.write needs the format spelled out explicitly because .sph
        # isn't a known WAV extension. recording_from_file reads via
        # sf.info() which infers from content, so the test path matches
        # what production does on real TED-LIUM SPHERE files.
        sf.write(sph_dir / f"{talk}.sph", audio, 16000, format="WAV")

    # train STM exercises: padding row → skipped, two valid utterances,
    # and a row whose transcript is the "ignore" sentinel → skipped.
    (legacy / "train" / "stm" / "AaronHuey_2010X.stm").write_text(
        "AaronHuey_2010X 1 inter_segment_gap 0.00 5.00 unknown ignore_time_segment_in_scoring\n"
        "AaronHuey_2010X 1 AaronHuey_2010X 5.00 8.40 <o,f0,male> hello world this is a talk\n"
        "AaronHuey_2010X 1 inter_segment_gap 8.40 10.00 unknown ignore_time_segment_in_scoring\n"
        "AaronHuey_2010X 1 AaronHuey_2010X 10.00 14.20 <o,f0,male> {NOISE} more talk content here\n",
        encoding="utf-8",
    )

    # dev STM: one valid utterance, used to verify multi-subset traversal
    # and default-subset auto-discovery.
    (legacy / "dev" / "stm" / "BenSaunders_2014.stm").write_text(
        "BenSaunders_2014 1 BenSaunders_2014 0.50 3.75 <o,f0,male> a different talk\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def mock_cnceleb(tmp_path: Path) -> Path:
    """Create a tiny CN-Celeb-1-like tree under <tmp>/CN-Celeb_flac/.

    Mirrors the layout the real cn-celeb_v2.tar.gz produces — verified
    against the actual ~21 GB tarball:

    - ``data/<spk>/<utt>.flac`` — training audio (here 3 utterances
      across 2 speakers).
    - ``dev/dev.lst`` — speaker IDs only (one per line), not paths.
      Here, id00000 is in dev; id00001 is not.
    - ``eval/enroll/<spk>-enroll.flac`` — flat directory of enrolment
      recordings. Speaker id is the dash-prefix of the filename.
    - ``eval/test/<spk>-<utt>.flac`` — flat directory of test trial
      recordings.

    The fixture also creates ``eval/lists/{enroll,test}.lst`` files
    that the recipe does NOT consume (it walks the directories
    directly because .lst lines spell paths with ``.wav`` while the
    on-disk files are ``.flac``). They are present only to mirror
    the real release; tests don't reference them.
    """
    ds_root = tmp_path / "CN-Celeb_flac"
    data_root = ds_root / "data"
    audio = np.sin(np.linspace(0, 1, 16000)).astype(np.float32) * 0.5

    # data/ — two speakers, three utterances total
    for speaker, utts in [
        ("id00000", ["interview-01-001", "vlog-01-001"]),
        ("id00001", ["interview-01-001"]),
    ]:
        spk_dir = data_root / speaker
        spk_dir.mkdir(parents=True, exist_ok=True)
        for utt in utts:
            sf.write(spk_dir / f"{utt}.flac", audio, 16000, format="FLAC")

    # dev/dev.lst — one speaker id. id00000 is in dev; id00001 is not.
    dev_dir = ds_root / "dev"
    dev_dir.mkdir(parents=True, exist_ok=True)
    (dev_dir / "dev.lst").write_text("id00000\n", encoding="utf-8")

    # eval/enroll/ and eval/test/ — flat directories with speaker-prefixed
    # filenames. Two enrol files (one per eval speaker) + two test files.
    enroll_dir = ds_root / "eval" / "enroll"
    test_dir = ds_root / "eval" / "test"
    enroll_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    sf.write(enroll_dir / "id00800-enroll.flac", audio, 16000, format="FLAC")
    sf.write(enroll_dir / "id00801-enroll.flac", audio, 16000, format="FLAC")
    sf.write(test_dir / "id00800-singing-01-001.flac", audio, 16000, format="FLAC")
    sf.write(test_dir / "id00801-vlog-01-001.flac", audio, 16000, format="FLAC")

    # The real corpus also has eval/lists/*.lst alongside; ship empty
    # placeholders so the directory tree matches reality (the recipe
    # does not read them).
    lists_dir = ds_root / "eval" / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    (lists_dir / "enroll.lst").write_text(
        "id00800-enroll enroll/id00800-enroll.wav\nid00801-enroll enroll/id00801-enroll.wav\n",
        encoding="utf-8",
    )
    (lists_dir / "test.lst").write_text(
        "test/id00800-singing-01-001.wav\ntest/id00801-vlog-01-001.wav\n", encoding="utf-8"
    )

    return tmp_path
