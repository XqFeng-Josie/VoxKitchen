"""Unit tests for download utility."""

from __future__ import annotations

import gzip
import tarfile
from pathlib import Path

from voxkitchen.utils.download import extract_tar


def test_download_file_skips_existing(tmp_path: Path) -> None:
    from voxkitchen.utils.download import download_file

    dest = tmp_path / "existing.txt"
    dest.write_text("already here")
    original_content = dest.read_text()

    # Should return immediately without overwriting
    result = download_file("http://fake-url/file.txt", dest)
    assert result == dest
    assert dest.read_text() == original_content


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
