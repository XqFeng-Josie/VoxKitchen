"""Recording and AudioSource models.

A Recording describes a physical audio resource — the thing whose bytes live
on disk (or at a URL). A Recording is immutable after ingest: operators that
materialize new audio (format conversion, resampling) create new Recordings
with new ids rather than mutating existing ones.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class AudioSource(BaseModel):
    """One physical source contributing a set of channels to a Recording.

    A Recording can have multiple AudioSources when its audio is split across
    files (e.g. multi-microphone setups where each mic is its own .wav).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["file", "url", "command"]
    channels: list[int]
    source: str


class Recording(BaseModel):
    """A physical audio resource with metadata."""

    model_config = ConfigDict(extra="forbid")

    id: str
    sources: list[AudioSource]
    sampling_rate: int
    num_samples: int
    duration: float
    num_channels: int
    checksum: str | None = None
    custom: dict[str, Any] = {}
