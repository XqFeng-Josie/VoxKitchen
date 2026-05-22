"""Tests for the Recipe base class behaviour shared across recipes."""

from __future__ import annotations

import logging
from pathlib import Path

from voxkitchen.ingest.recipes.base import Recipe
from voxkitchen.schema.cutset import CutSet


class _DummyRecipe(Recipe):
    """Minimal Recipe stub that exercises the download() path without I/O."""

    name = "dummy"
    download_urls = {
        "small": ["https://example.invalid/small.tar.gz"],
        "big": ["https://example.invalid/big.tar.gz"],
    }
    download_sizes = {
        "small": 500 * 1024 * 1024,  # 500 MB
        "big": 5 * 1024**3,  # 5 GB
    }

    def prepare(self, root, subsets, ctx):
        return CutSet([])


def test_download_announces_total_size_when_known(monkeypatch, caplog) -> None:
    """`Recipe.download()` must log the total compressed size *before* any fetch.

    The whole point of `download_sizes` is letting users see "you're about
    to fetch N GB" before the network starts moving. We assert that log
    line appears with the correctly formatted total.
    """
    calls: list[str] = []

    def fake_download_file(url, dest, *, desc=""):  # type: ignore[no-untyped-def]
        calls.append(url)
        # Create an empty placeholder so extract_tar isn't tempted; we
        # also stub extract_tar below so this never gets opened.
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).touch()
        return dest

    def fake_extract_tar(archive, dest):  # type: ignore[no-untyped-def]
        pass

    import voxkitchen.utils.download as dl

    monkeypatch.setattr(dl, "download_file", fake_download_file)
    monkeypatch.setattr(dl, "extract_tar", fake_extract_tar)

    recipe = _DummyRecipe()
    with caplog.at_level(logging.INFO, logger="voxkitchen.ingest.recipes.base"):
        recipe.download(Path("/tmp/dummy-test-noop"), ["small", "big"])

    log = "\n".join(r.getMessage() for r in caplog.records)
    # 500 MB + 5 GB = 5.49 GB total; format_bytes rounds GB to 1 decimal.
    assert "5.5 GB" in log
    assert "dummy" in log
    assert "small" in log and "big" in log

    # Both URLs were fetched (recipe didn't short-circuit).
    assert len(calls) == 2


def test_download_skips_total_announcement_when_sizes_missing(monkeypatch, caplog) -> None:
    """No statically-known sizes → no misleading "0 B" total in the log.

    Some future recipe may have download_urls but leave download_sizes
    empty (e.g. while size data is being collected). The base class must
    not advertise a fabricated size in that case.
    """

    class _NoSize(_DummyRecipe):
        download_sizes: dict[str, int] = {}

    def fake_download_file(url, dest, *, desc=""):  # type: ignore[no-untyped-def]
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).touch()
        return dest

    def fake_extract_tar(archive, dest):  # type: ignore[no-untyped-def]
        pass

    import voxkitchen.utils.download as dl

    monkeypatch.setattr(dl, "download_file", fake_download_file)
    monkeypatch.setattr(dl, "extract_tar", fake_extract_tar)

    with caplog.at_level(logging.INFO, logger="voxkitchen.ingest.recipes.base"):
        _NoSize().download(Path("/tmp/dummy-test-noop2"), ["small"])

    log = "\n".join(r.getMessage() for r in caplog.records)
    assert "about to download" not in log
