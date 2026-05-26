from voxkitchen.datasets.catalog import DatasetEntry
from voxkitchen.datasets.catalog_gen import build_download_info, check_docs, write_docs


def _entries():
    return [DatasetEntry(
        id="librispeech", name="LibriSpeech", task=["asr"], languages=["en"],
        license="CC BY 4.0", summary="s", homepage="h",
        recommendation="r", recipe="librispeech")]


def test_build_download_info_pulls_from_registry():
    info = build_download_info(_entries())
    assert "librispeech" in info
    assert info["librispeech"].source == "openslr"  # real recipe URLs are openslr


def test_write_then_check_roundtrip(tmp_path):
    entries = _entries()
    info = build_download_info(entries)
    write_docs(entries, info, tmp_path)
    assert (tmp_path / "index.md").is_file()
    assert (tmp_path / "librispeech.md").is_file()
    assert check_docs(entries, info, tmp_path) == []


def test_check_detects_drift(tmp_path):
    entries = _entries()
    info = build_download_info(entries)
    write_docs(entries, info, tmp_path)
    (tmp_path / "index.md").write_text("tampered", encoding="utf-8")
    drift = check_docs(entries, info, tmp_path)
    assert "index.md" in drift


def test_check_detects_missing_files(tmp_path):
    # nothing written yet → every rendered file is reported missing, sorted
    entries = _entries()
    drift = check_docs(entries, build_download_info(entries), tmp_path)
    assert drift == sorted(drift)
    assert "index.md" in drift and "librispeech.md" in drift
