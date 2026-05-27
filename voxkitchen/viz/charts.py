"""Plotly chart helpers for the dataset card.

Each chart function returns an HTML ``<div>`` string with NO embedded plotly.js
(call :func:`plotly_script` once per page to load it). Returns ``""`` when there
is no data or plotly is not installed, so callers degrade gracefully to tables.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from voxkitchen.schema.cutset import CutSet


def _plotly() -> tuple[Any, Any] | None:
    """Return (graph_objects, io) or None if plotly is not installed."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        return None
    return go, pio


def plotly_script() -> str:
    """Embed plotly.js once for a self-contained page; '' if plotly is absent."""
    if _plotly() is None:
        return ""
    # Public API for the bundled plotly.js source (plotly.io.get_plotlyjs was
    # removed in plotly 6.x; plotly.offline.get_plotlyjs is the stable accessor).
    from plotly.offline import get_plotlyjs

    return f"<script>{get_plotlyjs()}</script>"


def _histogram(values: list[float], x_title: str) -> str:
    mods = _plotly()
    if mods is None or not values:
        return ""
    go, pio = mods
    fig = go.Figure(go.Histogram(x=values, nbinsx=30))
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Count",
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return str(pio.to_html(fig, full_html=False, include_plotlyjs=False))


def duration_histogram(cuts: CutSet) -> str:
    return _histogram([c.duration for c in cuts], "Duration (s)")


def metric_histogram(cuts: CutSet, key: str) -> str:
    return _histogram([c.metrics[key] for c in cuts if key in c.metrics], key)


def category_bar(counts: Mapping[str, int], title: str) -> str:
    mods = _plotly()
    if mods is None or not counts:
        return ""
    go, pio = mods
    keys = list(counts.keys())
    fig = go.Figure(go.Bar(x=keys, y=[counts[k] for k in keys]))
    fig.update_layout(
        xaxis_title=title,
        yaxis_title="Count",
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return str(pio.to_html(fig, full_html=False, include_plotlyjs=False))
