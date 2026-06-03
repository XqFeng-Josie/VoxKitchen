"""Tests for the fixture-generation step of the operator sweep."""

from pathlib import Path


def test_setup_creates_all_generated_fixtures(tmp_path: Path) -> None:
    """--setup must produce all derived fixtures deterministically."""
    from scripts.sweep.setup_fixtures import generate_fixtures

    repo_root = Path(__file__).resolve().parents[3]
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()

    generate_fixtures(
        repo_root=repo_root,
        fixtures_dir=fixtures_dir,
    )

    assert (fixtures_dir / "audio" / "tiny-english.wav").is_file()
    link = fixtures_dir / "audio" / "demo1.opus"
    # On Linux/macOS this should always be a symlink. is_symlink() does not
    # follow the link; is_file() does, so the conjunction proves both that
    # the link exists AND that the target is reachable.
    assert link.is_symlink(), "expected a symlink to demo1.opus (copy fallback fired silently)"
    assert link.is_file(), "symlink target unreachable"
    assert (fixtures_dir / "noise" / "white-5s.wav").is_file()
    assert (fixtures_dir / "rir" / "synthetic-rir.wav").is_file()
    assert (fixtures_dir / "manifests" / "text-en-1cut.jsonl.gz").is_file()
    assert (fixtures_dir / "manifests" / "text-zh-1cut.jsonl.gz").is_file()
    # Added in Task 7 (cer_wer fixture-based approach + speaker_similarity .npy)
    assert (fixtures_dir / "manifests" / "cer-wer-1cut.jsonl.gz").is_file()
    assert (fixtures_dir / "embeddings" / "ref-speaker.npy").is_file()


def test_setup_is_idempotent(tmp_path: Path) -> None:
    """Running --setup twice must produce byte-identical files."""
    from scripts.sweep.setup_fixtures import generate_fixtures

    repo_root = Path(__file__).resolve().parents[3]
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()

    generate_fixtures(repo_root=repo_root, fixtures_dir=fixtures_dir)
    content_first = {
        p.relative_to(fixtures_dir): p.read_bytes() for p in fixtures_dir.rglob("*") if p.is_file()
    }

    generate_fixtures(repo_root=repo_root, fixtures_dir=fixtures_dir)
    content_second = {
        p.relative_to(fixtures_dir): p.read_bytes() for p in fixtures_dir.rglob("*") if p.is_file()
    }

    assert content_first == content_second, "fixture bytes drifted between two --setup runs"


def test_tiny_english_is_5s_16khz_mono(tmp_path: Path) -> None:
    """tiny-english.wav must be 5s @ 16 kHz mono — deterministic from demo1.opus."""
    import soundfile as sf
    from scripts.sweep.setup_fixtures import generate_fixtures

    repo_root = Path(__file__).resolve().parents[3]
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    for sub in ("audio", "noise", "rir", "manifests", "embeddings"):
        (fixtures_dir / sub).mkdir()

    generate_fixtures(repo_root=repo_root, fixtures_dir=fixtures_dir)

    audio, sr = sf.read(fixtures_dir / "audio" / "tiny-english.wav")
    assert sr == 16000, f"expected 16000 Hz, got {sr}"
    assert audio.ndim == 1, f"expected mono, got shape {audio.shape}"
    assert 4.9 <= len(audio) / sr <= 5.1, f"expected ~5s, got {len(audio) / sr:.2f}s"
