"""Recipe base class: parse (and optionally download) a dataset into a CutSet."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from voxkitchen.schema.cutset import CutSet

if TYPE_CHECKING:
    from voxkitchen.pipeline.context import RunContext


class Recipe(ABC):
    name: str = ""

    # Override in subclasses that support download.
    # Mapping of subset name → list of URLs to download.
    download_urls: dict[str, list[str]] = {}

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
        from voxkitchen.utils.download import download_file, extract_tar

        target_subsets = subsets or list(self.download_urls.keys())
        for subset in target_subsets:
            urls = self.download_urls.get(subset)
            if urls is None:
                available = list(self.download_urls.keys())
                raise ValueError(
                    f"unknown subset {subset!r} for {self.name}. Available: {available}"
                )
            for url in urls:
                filename = url.rsplit("/", 1)[-1]
                archive_path = root / filename
                download_file(url, archive_path, desc=f"{self.name}/{subset}")
                if filename.endswith((".tar.gz", ".tgz", ".tar.bz2")):
                    extract_tar(archive_path, root)
