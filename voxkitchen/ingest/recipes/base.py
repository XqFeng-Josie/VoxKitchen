"""Recipe base class: parse (and optionally download) a dataset into a CutSet."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from voxkitchen.schema.cutset import CutSet

if TYPE_CHECKING:
    from voxkitchen.pipeline.context import RunContext

logger = logging.getLogger(__name__)


class Recipe(ABC):
    name: str = ""

    # Override in subclasses that support download.
    # Mapping of subset name → list of URLs to download.
    download_urls: dict[str, list[str]] = {}

    # Compressed download size per subset, in bytes. Keys mirror
    # ``download_urls``. Values are pulled from HEAD-probed
    # Content-Length on the canonical mirror. Empty when sizes aren't
    # statically known (e.g. recipes that fetch from HuggingFace at
    # runtime, or manual-download recipes).
    download_sizes: dict[str, int] = {}

    @abstractmethod
    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        """Parse a locally-present dataset and return a CutSet."""

    def download(self, root: Path, subsets: list[str] | None) -> None:
        """Download dataset files to *root*.

        The default implementation downloads from ``download_urls`` (if
        defined) and extracts tar archives.  Subclasses may override for
        custom download logic (e.g. HuggingFace ``datasets`` library).
        """
        if not self.download_urls:
            raise NotImplementedError(
                f"recipe {self.name!r} does not support automatic download. "
                f"Please download the dataset manually and provide the root path."
            )
        from voxkitchen.utils.download import download_file, extract_tar, format_bytes

        target_subsets = subsets or list(self.download_urls.keys())
        # Print the total volume the user is about to fetch BEFORE any
        # network call. For multi-GB datasets this is the difference
        # between "I started a 50 GB download and didn't know" and
        # "I started a 50 GB download knowing the cost". One log line
        # per recipe-invocation, not per URL.
        total = sum(
            self.download_sizes.get(s, 0) for s in target_subsets if s in self.download_sizes
        )
        if total > 0:
            logger.info(
                "recipe %s: about to download %s across %d subset(s) — %s",
                self.name,
                format_bytes(total),
                len(target_subsets),
                ", ".join(target_subsets),
            )

        for subset in target_subsets:
            urls = self.download_urls.get(subset)
            if urls is None:
                available = list(self.download_urls.keys())
                raise ValueError(
                    f"unknown subset {subset!r} for {self.name}. Available: {available}"
                )
            size = self.download_sizes.get(subset)
            if size:
                logger.info(
                    "recipe %s subset %r: downloading %s",
                    self.name,
                    subset,
                    format_bytes(size),
                )
            for url in urls:
                filename = url.rsplit("/", 1)[-1]
                archive_path = root / filename
                download_file(url, archive_path, desc=f"{self.name}/{subset}")
                if filename.endswith((".tar.gz", ".tgz", ".tar.bz2")):
                    extract_tar(archive_path, root)
