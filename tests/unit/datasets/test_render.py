from voxkitchen.datasets.catalog import DatasetEntry
from voxkitchen.datasets.catalog_gen import RecipeDownloadInfo, render_all

GENERATED_HEADER = "AUTO-GENERATED"


def _recipe_entry():
    return DatasetEntry(
        id="librispeech", name="LibriSpeech", task=["asr"], languages=["en"],
        license="CC BY 4.0", summary="Read English audiobooks.",
        homepage="https://www.openslr.org/12",
        recommendation="Standard English ASR benchmark.",
        hours=960.0, recipe="librispeech",
        recommended_pipeline="examples/pipelines/librispeech-asr.yaml",
    )


def _info_entry():
    return DatasetEntry(
        id="gigaspeech", name="GigaSpeech", task=["asr"], languages=["en"],
        license="see source terms", summary="10k h English ASR corpus.",
        homepage="https://github.com/SpeechColab/GigaSpeech",
        recommendation="Large diverse English ASR; needs access request.",
        notes="Request access on the project page.",
    )


def _render():
    entries = [_recipe_entry(), _info_entry()]
    dl = {"librispeech": RecipeDownloadInfo(source="openslr", size_range="337 MB - 28.5 GB",
                                            subsets=["dev-clean", "train-clean-100"])}
    return render_all(entries, dl)


def test_renders_index_and_two_pages():
    out = _render()
    assert "datasets/index.md" in out
    assert "datasets/librispeech.md" in out
    assert "datasets/gigaspeech.md" in out


def test_all_files_have_generated_header():
    for md in _render().values():
        assert md.lstrip().startswith("<!--") and GENERATED_HEADER in md.splitlines()[0]


def test_index_has_disclaimer_and_both_entries():
    idx = _render()["datasets/index.md"]
    assert "license" in idx.lower()
    assert "LibriSpeech" in idx and "GigaSpeech" in idx


def test_recipe_page_shows_download_command_and_recommendation():
    page = _render()["datasets/librispeech.md"]
    assert "Standard English ASR benchmark." in page
    assert "vkit docker download" in page
    assert "librispeech" in page
    assert "examples/pipelines/librispeech-asr.yaml" in page


def test_info_page_has_access_link_no_download_command():
    page = _render()["datasets/gigaspeech.md"]
    assert "https://github.com/SpeechColab/GigaSpeech" in page
    assert "Request access on the project page." in page
    assert "vkit docker download" not in page


def test_render_is_deterministic():
    assert _render() == _render()
