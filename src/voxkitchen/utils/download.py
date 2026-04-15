"""HTTP download and archive extraction utilities."""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


def _progress_hook(desc: str):  # type: ignore[no-untyped-def]
    """Return a urlretrieve reporthook that updates a tqdm bar."""
    from tqdm import tqdm

    bar: tqdm[None] | None = None

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        nonlocal bar
        if bar is None and total_size > 0:
            bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=desc)
        if bar is not None:
            bar.update(block_size)
            if block_num * block_size >= total_size:
                bar.close()

    return hook


def download_file(url: str, dest: Path, *, desc: str = "") -> Path:
    """Download url to dest. Skip if dest already exists and is non-empty."""
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("already exists, skipping: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s → %s", url, dest)
    urlretrieve(url, str(dest), reporthook=_progress_hook(desc or dest.name))
    return dest


def extract_tar(archive: Path, dest: Path) -> None:
    """Extract a .tar.gz, .tgz, or .tar.bz2 archive to dest."""
    dest.mkdir(parents=True, exist_ok=True)
    suffix = archive.name.lower()
    if suffix.endswith(".tar.gz") or suffix.endswith(".tgz"):
        mode = "r:gz"
    elif suffix.endswith(".tar.bz2"):
        mode = "r:bz2"
    else:
        mode = "r:*"
    logger.info("extracting %s → %s", archive.name, dest)
    with tarfile.open(archive, mode) as tar:
        tar.extractall(path=dest, filter="data")
