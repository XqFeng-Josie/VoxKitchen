from voxkitchen.pipeline.preflight import apply_clears, apply_writes, is_satisfied


def test_exact_match():
    assert is_satisfied("metrics.snr", {"metrics.snr", "audio"})
    assert not is_satisfied("metrics.snr", {"audio"})


def test_namespace_wildcard_in_available_satisfies_specific():
    assert is_satisfied("metrics.snr", {"metrics.*"})
    assert is_satisfied("custom.word_alignments", {"custom.*"})


def test_apply_writes_adds_tokens():
    avail = {"audio"}
    assert apply_writes(avail, ["supervisions.text"]) == {"audio", "supervisions.text"}


def test_apply_clears_removes_namespace_and_specific():
    avail = {"audio", "supervisions.text", "metrics.snr", "metrics.dnsmos"}
    out = apply_clears(avail, ["supervisions.text", "metrics.*"])
    assert out == {"audio"}
