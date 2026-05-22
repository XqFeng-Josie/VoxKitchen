"""Unit tests for download utility."""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

import pytest
from voxkitchen.utils.download import extract_tar, format_bytes


def test_format_bytes_renders_human_friendly_units() -> None:
    """``format_bytes`` picks the largest sensible unit and formats it.

    GB is the only unit we render with a decimal (speech datasets are
    almost always at GB scale; .1 GB precision is what users want).
    Below GB we use integer division — "734 MB" not "734.0 MB" — so the
    output stays compact in the `vkit recipes` Size column.
    """
    assert format_bytes(0) == "0 B"
    assert format_bytes(512) == "512 B"
    assert format_bytes(1024) == "1 KB"
    assert format_bytes(2 * 1024**2) == "2 MB"
    assert format_bytes(int(1.5 * 1024**3)) == "1.5 GB"
    assert format_bytes(15_582_913_665) == "14.5 GB"  # AISHELL-1 data
    assert format_bytes(44_565_031_479) == "41.5 GB"  # LibriTTS train-other-500


def test_download_file_skips_existing(tmp_path: Path) -> None:
    from voxkitchen.utils.download import download_file

    dest = tmp_path / "existing.txt"
    dest.write_text("already here")
    original_content = dest.read_text()

    # Should return immediately without overwriting
    result = download_file("http://fake-url/file.txt", dest)
    assert result == dest
    assert dest.read_text() == original_content


def _make_fake_urlretrieve(
    behaviours: list[str], payload: bytes = b"payload"
) -> tuple[Any, list[Path]]:
    """Build an urlretrieve stub that follows a per-attempt script.

    ``behaviours`` is a list of strings: ``"ok"`` writes ``payload`` to
    the destination path; ``"reset"`` raises ConnectionResetError without
    writing anything; ``"partial"`` writes half the payload then raises
    (mimics a mid-stream RST). Returns the fake and a list it appends to
    each call so the test can inspect call history.
    """
    history: list[Path] = []
    iterator = iter(behaviours)

    def _fake(url: str, filename: str, reporthook: Any = None) -> None:
        history.append(Path(filename))
        action = next(iterator)
        if action == "ok":
            Path(filename).write_bytes(payload)
            return
        if action == "partial":
            # Mimic a half-written file followed by a reset.
            Path(filename).write_bytes(payload[: len(payload) // 2])
            raise ConnectionResetError("simulated mid-stream RST")
        if action == "reset":
            raise ConnectionResetError("simulated reset before any bytes")
        raise AssertionError(f"unknown behaviour: {action!r}")

    return _fake, history


def test_download_file_writes_via_partial_and_renames(tmp_path: Path, monkeypatch) -> None:
    """A successful download lands at the final path; no .partial sticks around."""
    import voxkitchen.utils.download as dl

    fake, history = _make_fake_urlretrieve(["ok"], payload=b"hello")
    monkeypatch.setattr(dl, "urlretrieve", fake)

    dest = tmp_path / "blob.bin"
    out = dl.download_file("http://example/blob.bin", dest)
    assert out == dest
    assert dest.read_bytes() == b"hello"
    # The first attempt wrote to .partial, then we renamed to dest.
    assert history[0].name.endswith(".partial")
    assert not (tmp_path / "blob.bin.partial").exists()


def test_download_file_retries_on_transient_reset(tmp_path: Path, monkeypatch) -> None:
    """A mid-stream RST on attempt 1 is retried; attempt 2 succeeds."""
    import voxkitchen.utils.download as dl

    fake, history = _make_fake_urlretrieve(["partial", "ok"], payload=b"hello world")
    monkeypatch.setattr(dl, "urlretrieve", fake)
    # Don't actually sleep in tests — keep them snappy.
    monkeypatch.setattr(dl.time, "sleep", lambda _s: None)

    dest = tmp_path / "blob.bin"
    out = dl.download_file("http://example/blob.bin", dest, max_attempts=3)
    assert out == dest
    assert dest.read_bytes() == b"hello world"
    # Two attempts ran; both targeted the same .partial path.
    assert len(history) == 2
    assert all(p.name.endswith(".partial") for p in history)
    # No stale .partial after success.
    assert not (tmp_path / "blob.bin.partial").exists()


def test_download_file_gives_up_after_max_attempts(tmp_path: Path, monkeypatch) -> None:
    """After max_attempts repeated failures the last exception bubbles up."""
    import voxkitchen.utils.download as dl

    fake, _ = _make_fake_urlretrieve(["reset", "reset", "reset"])
    monkeypatch.setattr(dl, "urlretrieve", fake)
    monkeypatch.setattr(dl.time, "sleep", lambda _s: None)

    dest = tmp_path / "blob.bin"
    with pytest.raises(ConnectionResetError):
        dl.download_file("http://example/blob.bin", dest, max_attempts=3)

    # No stale .partial file — the final attempt cleans up.
    assert not (tmp_path / "blob.bin.partial").exists()
    # Final destination was never created.
    assert not dest.exists()


def test_download_file_drops_stale_partial_before_retrying(tmp_path: Path, monkeypatch) -> None:
    """A leftover .partial from a previous run is removed before attempt 1.

    urlretrieve cannot resume, so a leftover .partial is dead weight that
    would otherwise mislead size-based checks on subsequent retries.
    """
    import voxkitchen.utils.download as dl

    stale = tmp_path / "blob.bin.partial"
    stale.write_bytes(b"left over from a previous run")

    fake, _ = _make_fake_urlretrieve(["ok"], payload=b"fresh")
    monkeypatch.setattr(dl, "urlretrieve", fake)

    dest = tmp_path / "blob.bin"
    dl.download_file("http://example/blob.bin", dest)

    assert dest.read_bytes() == b"fresh"
    # The .partial was rotated through and removed, not left in place.
    assert not stale.exists()


def test_extract_tar_gz(tmp_path: Path) -> None:
    # Create a tar.gz with a test file inside
    archive_dir = tmp_path / "archive_src"
    archive_dir.mkdir()
    (archive_dir / "hello.txt").write_text("world")

    archive_path = tmp_path / "test.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(archive_dir / "hello.txt", arcname="hello.txt")

    # Extract
    extract_dest = tmp_path / "extracted"
    extract_tar(archive_path, extract_dest)

    assert (extract_dest / "hello.txt").exists()
    assert (extract_dest / "hello.txt").read_text() == "world"


def test_extract_tar_gz_with_subdirectory(tmp_path: Path) -> None:
    # Create a tar.gz with a directory structure
    archive_dir = tmp_path / "archive_src" / "LibriSpeech" / "dev-clean"
    archive_dir.mkdir(parents=True)
    (archive_dir / "data.txt").write_text("test data")

    archive_path = tmp_path / "dev-clean.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(
            tmp_path / "archive_src" / "LibriSpeech",
            arcname="LibriSpeech",
        )

    # Extract
    extract_dest = tmp_path / "extracted"
    extract_tar(archive_path, extract_dest)

    assert (extract_dest / "LibriSpeech" / "dev-clean" / "data.txt").exists()
