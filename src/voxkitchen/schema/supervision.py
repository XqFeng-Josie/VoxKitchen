"""Supervision: a labeled time interval over a Recording."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Supervision(BaseModel):
    """A human-interpretable annotation over an interval of a Recording.

    Multiple Supervisions can cover the same Recording (e.g. multiple speakers
    in a conversation) and their time intervals may overlap (e.g. two people
    speaking simultaneously).

    Every annotation field except ``id``, ``recording_id``, ``start``, and
    ``duration`` is optional — operators fill them in progressively as the
    pipeline advances. A Supervision emitted by VAD has no ``text``; after ASR
    it does.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    recording_id: str
    start: float
    duration: float

    channel: int | list[int] | None = None
    text: str | None = None
    language: str | None = None
    speaker: str | None = None
    gender: Literal["m", "f", "o"] | None = None
    age_range: str | None = None
    custom: dict[str, Any] = {}
