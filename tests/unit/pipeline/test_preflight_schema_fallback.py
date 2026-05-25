from voxkitchen.pipeline.preflight import contract_from_schemas, make_contract_lookup


def test_contract_from_schemas_reads_block():
    schemas = {
        "faster_whisper_asr": {
            "contract": {"reads": ["audio"], "writes": ["supervisions.text"],
                          "optional_reads": [], "clears": []}
        }
    }
    c = contract_from_schemas("faster_whisper_asr", {}, schemas)
    assert c is not None
    assert c.writes == ["supervisions.text"]


def test_contract_from_schemas_unknown_returns_none():
    assert contract_from_schemas("nope", {}, {}) is None


def test_make_contract_lookup_uses_schema_when_registry_misses():
    # an operator name not in the registry, only in schemas
    schemas = {
        "imaginary_op": {
            "contract": {"reads": ["audio"], "writes": [], "optional_reads": [], "clears": []}
        }
    }
    lookup = make_contract_lookup(schemas)
    c = lookup("imaginary_op", {})
    assert c is not None and c.reads == ["audio"]


def test_make_contract_lookup_none_schemas_returns_none_for_unknown():
    lookup = make_contract_lookup(None)
    assert lookup("imaginary_op", {}) is None
