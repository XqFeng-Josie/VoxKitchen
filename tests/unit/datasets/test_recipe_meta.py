from voxkitchen.datasets.recipe_meta import download_source_label, format_size_range


def test_format_size_range_empty():
    assert format_size_range({}) == "—"


def test_format_size_range_single():
    # 1 MiB = 1_048_576 bytes; format_bytes uses binary units (1024^2),
    # so this yields "1 MB".  1_000_000 bytes would yield "976 KB" instead.
    assert "MB" in format_size_range({"a": 1_048_576})


def test_format_size_range_spread():
    out = format_size_range({"a": 1_048_576, "b": 2_000_000_000})
    # original _format_size_column uses a hyphen-minus separator, not an en-dash
    assert " - " in out


def test_source_label_openslr():
    assert download_source_label({"s": ["https://www.openslr.org/resources/12/x.tar.gz"]}) == "openslr"


def test_source_label_huggingface():
    assert download_source_label({"s": ["https://huggingface.co/datasets/x"]}) == "HuggingFace"


def test_source_label_empty_is_url():
    # original _download_source_label returns "url" for no entries, not "manual"
    assert download_source_label({}) == "url"
