import json

from voxkitchen.runtime.dump_schemas import dump_current_env


def test_dump_includes_contract(tmp_path):
    out = tmp_path / "schemas.json"
    dump_current_env("core", out)
    data = json.loads(out.read_text())
    ops = data["operators"]
    fw = ops["faster_whisper_asr"]
    assert "contract" in fw
    assert fw["contract"]["writes"] == ["supervisions.text", "supervisions.language"]
    assert ops["silero_vad"]["contract"]["clears"]  # non-empty
