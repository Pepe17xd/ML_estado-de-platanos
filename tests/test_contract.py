import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_metadata_contract():
    metadata = json.loads((ROOT / "models" / "metadata.json").read_text())
    assert metadata["classes"]
    assert metadata["input_size"] == [224, 224]
    assert metadata["model_file"] == "production_model.keras"
    assert "days_remaining" in metadata["outputs"]
