from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import tensorflow as tf

from inference.predictor import BananaPredictor
from training.three_class.model import build_model
from training.v4_remastered.transfer import initialize_from_v2


ROOT = Path(__file__).resolve().parents[1]


def test_v2_transfer_reuses_compatible_backbone_but_resets_four_class_head() -> None:
    model = build_model(weights=None)
    report = initialize_from_v2(
        model,
        ROOT / "models/production_model_v2.keras",
        ROOT / "models/production_model_v2_metadata.json",
    )
    assert model.output_shape == (None, 3)
    assert report["classification_head_reset"] is True
    assert len(report["copied_weighted_layers"]) >= 80
    assert "dense" in report["copied_weighted_layers"]
    assert all(item["layer"] == "state" for item in report["skipped"] if "head" in item["reason"])


def _tiny_model(path: Path) -> str:
    inputs = tf.keras.Input((224, 224, 3), name="image")
    pooled = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="state")(pooled)
    tf.keras.Model(inputs, outputs).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(model_name: str, digest: str) -> dict:
    return {
        "model_version": "4.0.0-candidate",
        "model_name": model_name,
        "model_file": "tiny.keras",
        "classes": ["verde", "maduro", "sobre_maduro"],
        "input_size": [224, 224],
        "color_mode": "RGB",
        "resize_backend": "pillow_bilinear",
        "preprocessing": "keras MobileNetV3 include_preprocessing; RGB float [0,255]",
        "checkpoint_sha256": digest,
        "temperature": 1.0,
        "uncertainty_threshold": 0.5,
    }


def test_checkpoint_hash_and_metadata_contract_are_enforced(tmp_path: Path) -> None:
    path = tmp_path / "tiny.keras"
    digest = _tiny_model(path)
    predictor = BananaPredictor(path, _metadata("test_v4", digest))
    assert predictor.classes == ["verde", "maduro", "sobre_maduro"]
    with pytest.raises(ValueError, match="SHA-256"):
        BananaPredictor(path, _metadata("test_v4", "0" * 64))


def test_three_class_contract_rejects_class_order_and_double_preprocessing_declaration(tmp_path: Path) -> None:
    path = tmp_path / "tiny.keras"
    digest = _tiny_model(path)
    metadata = _metadata("test_v4", digest)
    metadata["classes"] = ["verde", "sobre_maduro", "maduro"]
    with pytest.raises(ValueError, match="exact three-class"):
        BananaPredictor(path, metadata)
    metadata = _metadata("test_v4", digest)
    metadata["preprocessing"] = "external_0_1_scaling"
    with pytest.raises(ValueError, match="embedded MobileNetV3"):
        BananaPredictor(path, metadata)


def test_v4_config_defines_exact_contract_and_excludes_rotten() -> None:
    config = json.loads((ROOT / "configs/v4_remastered.json").read_text(encoding="utf-8"))
    assert config["classes"] == ["verde", "maduro", "sobre_maduro"]
    assert config["v2_checkpoint"] == "models/production_model.h5"
    assert "rotten" not in config["classes"]
