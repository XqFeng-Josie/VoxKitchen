"""CutSet: an eager or lazy wrapper over a sequence of Cuts.

CutSet is deliberately not a Pydantic model. It is a thin layer above an
iterable of Cuts backed by either an in-memory list or a JSONL.gz file on
disk.

Lazy mode (``CutSet.from_jsonl_gz(path, lazy=True)``) defers reading until
iteration, and auto-materializes when ``len()`` or ``split()`` is called.
Iterating a lazy CutSet streams from disk without building a list in memory;
iterating twice re-reads the file (trading I/O for memory savings).

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
from voxkitchen.schema.io import HeaderRecord, read_cuts, read_header, write_cuts


class CutSet:
    """A sequence of Cuts with functional operations and disk I/O helpers."""

    def __init__(self, cuts: Iterable[Cut]) -> None:
        self._cuts: list[Cut] | None = list(cuts)
        self._source_path: Path | None = None

    # -- lazy support ---------------------------------------------------------

    @property
    def is_lazy(self) -> bool:
        """True when this CutSet has not been materialized into memory."""
        return self._cuts is None

    def _materialize(self) -> None:
        """Load all cuts into memory. No-op if already materialized."""
        if self._cuts is None:
            assert self._source_path is not None
            self._cuts = list(read_cuts(self._source_path))

    # -- core protocol --------------------------------------------------------

    def __len__(self) -> int:
        self._materialize()
        assert self._cuts is not None
        return len(self._cuts)

    def __iter__(self) -> Iterator[Cut]:
        if self._cuts is not None:
            return iter(self._cuts)
        assert self._source_path is not None
        return read_cuts(self._source_path)

    def with_progress(self, desc: str = "cuts") -> CutSet:
        """Return a view whose ``__iter__`` wraps cuts with a tqdm bar."""
        self._materialize()
        assert self._cuts is not None
        return _ProgressCutSet(self._cuts, desc)

    # -- functional operations ------------------------------------------------

    def split(self, n: int) -> list[CutSet]:
        """Split into ``n`` roughly-equal CutSets.

        When ``n > len(self)``, trailing CutSets may be empty. Preserves order.
        """
        if n <= 0:
            raise ValueError(f"split n must be positive, got {n}")
        self._materialize()
        assert self._cuts is not None
        shards: list[list[Cut]] = [[] for _ in range(n)]
        for i, cut in enumerate(self._cuts):
            shards[i % n].append(cut)
        return [CutSet(s) for s in shards]

    def filter(self, predicate: Callable[[Cut], bool]) -> CutSet:
        return CutSet(c for c in self if predicate(c))

    def map(self, fn: Callable[[Cut], Cut]) -> CutSet:
        return CutSet(fn(c) for c in self)

    @classmethod
    def merge(cls, cutsets: Iterable[CutSet]) -> CutSet:
        """Concatenate multiple CutSets into one, preserving order."""
        out: list[Cut] = []
        for cs in cutsets:
            out.extend(cs)
        return cls(out)

    def to_jsonl_gz(self, path: Path, header: HeaderRecord) -> None:
        write_cuts(path, header, iter(self))

    @classmethod
    def from_jsonl_gz(cls, path: Path, *, lazy: bool = False) -> CutSet:
        """Load a CutSet from a manifest file.

        Args:
            lazy: If True, defer reading cuts until iteration. The header
                is validated eagerly (schema version check), but individual
                cut records are streamed on demand. This saves memory for
                large manifests when you only need a single iteration pass.
        """
        if lazy:
            read_header(path)  # validate header eagerly
            cs = cls.__new__(cls)
            cs._cuts = None
            cs._source_path = path
            return cs
        return cls(read_cuts(path))

    @classmethod
    def concat_from_disk(cls, paths: Sequence[Path]) -> CutSet:
        """Read and concatenate multiple manifest files in the given order."""
        out: list[Cut] = []
        for p in paths:
            out.extend(read_cuts(p))
        return cls(out)


class _ProgressCutSet(CutSet):
    """CutSet whose __iter__ shows a tqdm progress bar."""

    def __init__(self, cuts: list[Cut], desc: str) -> None:
        super().__init__(cuts)
        self._desc = desc

    def __iter__(self) -> Iterator[Cut]:
        from tqdm import tqdm  # type: ignore[import-untyped]

        return iter(tqdm(self._cuts, desc=self._desc, unit="cut"))
