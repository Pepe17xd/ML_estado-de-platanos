import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_metadata_contract():
    metadata = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8-sig"))
    assert metadata["classes"] == ["verde", "maduro", "sobre_maduro"]
    assert metadata["input_size"] == [224, 224]
    assert metadata["model_file"] == "production_model_v4.keras"
    assert metadata["checkpoint_sha256"]
    assert metadata["status"] == "production_local"
    assert metadata["days_remaining_available"] is False
