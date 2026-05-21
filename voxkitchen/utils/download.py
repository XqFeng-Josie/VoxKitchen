"""HTTP download and archive extraction utilities."""

from __future__ import annotations

import logging
import tarfile
import time
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

# Big OpenSLR / HuggingFace tarballs occasionally trigger a mid-stream RST
# from upstream. urlretrieve has no built-in retry, so we wrap it with a
# small exponential backoff. Three attempts (0s, 2s, 4s) is enough to clear
# transient resets without making genuine failures slow to surface.
_DEFAULT_MAX_ATTEMPTS = 3


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


def download_file(
    url: str,
    dest: Path,
    *,
    desc: str = "",
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> Path:
    """Download ``url`` to ``dest`` atomically with retry.

    Behaviour:

    - If ``dest`` already exists and is non-empty, the file is treated as
      a previously completed download and returned unmodified.
    - Otherwise the body is streamed into ``dest.<suffix>.partial`` and
      moved into place only after the transfer finishes cleanly. A
      previous ``.partial`` left behind by an aborted attempt is removed
      first — urlretrieve cannot resume, so a partial body is useless and
      keeping it would mask the failure on the next call.
    - Up to ``max_attempts`` attempts are made on transient failures
      (``ConnectionResetError``, ``OSError``, etc.) with exponential
      backoff (2s, 4s, …). Successful download exits immediately. The
      original exception is re-raised after the last attempt.

    Returns the path to the completed file (always ``dest``).
    """
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("already exists, skipping: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")

    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        # Always start a clean .partial — urlretrieve cannot pick up where
        # a previous attempt left off, so any leftover bytes are noise.
        if partial.exists():
            partial.unlink()

        logger.info("downloading %s → %s (attempt %d/%d)", url, dest, attempt, max_attempts)
        try:
            urlretrieve(url, str(partial), reporthook=_progress_hook(desc or dest.name))
            partial.rename(dest)
            return dest
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                wait_seconds = 2**attempt
                logger.warning(
                    "download attempt %d failed (%s); retrying in %ds",
                    attempt,
                    exc,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
            else:
                # Final attempt failed — make sure no stale .partial is left
                # behind to fool the next call into thinking the file is OK.
                if partial.exists():
                    partial.unlink()

    assert last_exc is not None  # max_attempts >= 1 is enforced by the caller
    raise last_exc


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
    with tarfile.open(str(archive), mode) as tar:  # type: ignore[call-overload]
        tar.extractall(path=dest, filter="data")
