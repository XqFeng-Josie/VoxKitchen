"""Unit tests for Docker image recommendation hints."""

from __future__ import annotations

from voxkitchen.cli.hints import docker_tag_for_extras, recommend_docker_tag


def test_fish_speech_extra_maps_to_fish_speech_image() -> None:
    assert docker_tag_for_extras("tts-fish-speech") == "fish-speech"


def test_recommend_fish_speech_for_fish_speech_only_pipeline() -> None:
    assert recommend_docker_tag([[], ["tts-fish-speech"]]) == "fish-speech"


def test_recommend_latest_when_fish_speech_mixes_with_other_specialized_env() -> None:
    assert recommend_docker_tag([["tts-fish-speech"], ["asr"]]) == "latest"
