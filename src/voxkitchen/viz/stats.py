"""Shared statistics computation for CutSet visualization."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from voxkitchen.schema.cutset import CutSet


def compute_cutset_stats(cuts: CutSet) -> dict[str, Any]:
    """Compute summary statistics from a CutSet."""
    durations = [c.duration for c in cuts]
    languages: Counter[str] = Counter()
    speakers: set[str] = set()
    metrics_keys: set[str] = set()

    for c in cuts:
        for s in c.supervisions:
            if s.language:
                languages[s.language] += 1
            if s.speaker:
                speakers.add(s.speaker)
        metrics_keys.update(c.metrics.keys())

    return {
        "count": len(durations),
        "total_duration_s": round(sum(durations), 2),
        "duration_stats": _percentile_stats(durations) if durations else {},
        "languages": dict(languages.most_common()),
        "speaker_count": len(speakers),
        "metrics_summary": {
            k: _percentile_stats([c.metrics[k] for c in cuts if k in c.metrics])
            for k in sorted(metrics_keys)
        },
    }


def _percentile_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(values)
    return {
        "min": round(float(np.min(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "max": round(float(np.max(arr)), 4),
    }
