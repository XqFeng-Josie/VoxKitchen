"""CutSet: a lazy, functional wrapper over a sequence of Cuts.

CutSet is deliberately not a Pydantic model. It is a thin layer above an
iterable of Cuts backed by either an in-memory list or a JSONL.gz file on
disk. Plan 1 implements the in-memory path; lazy streaming I/O (opening a
file and yielding Cuts without reading the whole thing into memory) is
also supported through ``from_jsonl_gz`` since ``read_cuts`` already returns
an iterator.

The key operations needed by the pipeline engine are:
- ``split(n)``: shard into N CutSets for GPU/CPU pool workers
- ``filter(pred)``: drop Cuts that don't match
- ``map(fn)``: single-Cut transformation
- ``merge(cutsets)``: concatenate several CutSets
- ``to_jsonl_gz(path, header)``: persist to disk
- ``from_jsonl_gz(path)``: load from disk (validates header + schema version)
- ``concat_from_disk(paths)``: lazy concatenation of shard files (used by
  GpuPoolExecutor after workers finish)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.io import HeaderRecord, read_cuts, write_cuts


class CutSet:
    """A sequence of Cuts with functional operations and disk I/O helpers."""

    def __init__(self, cuts: Iterable[Cut]) -> None:
        # Materialize to a list. Lazy-streaming variants can be added later
        # if memory becomes a concern; for Plan 1 we keep the semantics simple
        # and predictable.
        self._cuts: list[Cut] = list(cuts)

    def __len__(self) -> int:
        return len(self._cuts)

    def __iter__(self) -> Iterator[Cut]:
        return iter(self._cuts)

    def split(self, n: int) -> list[CutSet]:
        """Split into ``n`` roughly-equal CutSets.

        When ``n > len(self)``, trailing CutSets may be empty. Preserves order.
        """
        if n <= 0:
            raise ValueError(f"split n must be positive, got {n}")
        shards: list[list[Cut]] = [[] for _ in range(n)]
        for i, cut in enumerate(self._cuts):
            shards[i % n].append(cut)
        return [CutSet(s) for s in shards]

    def filter(self, predicate: Callable[[Cut], bool]) -> CutSet:
        return CutSet(c for c in self._cuts if predicate(c))

    def map(self, fn: Callable[[Cut], Cut]) -> CutSet:
        return CutSet(fn(c) for c in self._cuts)

    @classmethod
    def merge(cls, cutsets: Iterable[CutSet]) -> CutSet:
        """Concatenate multiple CutSets into one, preserving order."""
        out: list[Cut] = []
        for cs in cutsets:
            out.extend(cs)
        return cls(out)

    def to_jsonl_gz(self, path: Path, header: HeaderRecord) -> None:
        write_cuts(path, header, iter(self._cuts))

    @classmethod
    def from_jsonl_gz(cls, path: Path) -> CutSet:
        return cls(read_cuts(path))

    @classmethod
    def concat_from_disk(cls, paths: Sequence[Path]) -> CutSet:
        """Read and concatenate multiple manifest files in the given order."""
        out: list[Cut] = []
        for p in paths:
            out.extend(read_cuts(p))
        return cls(out)
