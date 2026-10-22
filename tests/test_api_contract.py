from __future__ import annotations

import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from api import main


ROOT = Path(__file__).resolve().parents[1]


def test_default_api_selection_is_v4_production_artifact() -> None:
    assert main.MODEL_PATH == ROOT / "models/production_model_v4.keras"
    assert main.METADATA_PATH == ROOT / "models/metadata.json"
    assert main.app.version == "4.0.0"


def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color=(140, 180, 70)).save(buffer, format="JPEG")
    return buffer.getvalue()


def assert_prediction_contract(response) -> None:
    assert response.status_code == 200, response.text
    result = response.json()
    assert set(result) == {"estado", "confianza", "incierto"}
    assert result["estado"] in ["verde", "maduro", "sobre_maduro", "incierto"]
    assert 0 <= result["confianza"] <= 1
    assert isinstance(result["incierto"], bool)


def test_v2_http_contract_and_controlled_image_errors_for_rollback(monkeypatch) -> None:
    monkeypatch.setattr(main, "MODEL_PATH", ROOT / "models/production_model_v2.keras")
    monkeypatch.setattr(main, "METADATA_PATH", ROOT / "models/production_model_v2_metadata.json")
    with TestClient(main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert set(health.json()) == {"status", "model", "version"}
        assert health.json()["model"] == "production_model_v2"
        assert_prediction_contract(client.post("/predict", files={"image": ("banana.jpg", jpeg_bytes(), "image/jpeg")}))
        assert client.post("/predict", files={"image": ("broken.jpg", b"not an image", "image/jpeg")}).status_code == 400
        assert client.post("/predict", files={"image": ("large.jpg", b"x" * (10 * 1024 * 1024 + 1), "image/jpeg")}).status_code == 413
        predictor = main.router.predictor
        main.router.predictor = None
        try:
            assert client.post("/predict", files={"image": ("banana.jpg", jpeg_bytes(), "image/jpeg")}).status_code == 503
        finally:
            main.router.predictor = predictor


def test_v4_production_api_contract_uses_three_classes(monkeypatch) -> None:
    class V4Predictor:
        def __init__(self, model_path, metadata):
            assert Path(model_path) == ROOT / "models/production_model_v4.keras"
            assert metadata["classes"] == ["verde", "maduro", "sobre_maduro"]

        def predict(self, image, encoded_image=None):
            return {"estado": "maduro", "confianza": 0.8, "incierto": False}

    monkeypatch.setattr(main, "MODEL_PATH", ROOT / "models/production_model_v4.keras")
    monkeypatch.setattr(main, "METADATA_PATH", ROOT / "models/metadata.json")
    monkeypatch.setattr(main, "BananaPredictor", V4Predictor)
    with TestClient(main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "model": "banana_ai_v4_remastered", "version": "4.0.0"}
        response = client.post("/predict", files={"image": ("banana.jpg", jpeg_bytes(), "image/jpeg")})
        assert response.status_code == 200
        result = response.json()
        assert result["estado"] in {"verde", "maduro", "sobre_maduro", "incierto"}
        assert 0 <= result["confianza"] <= 1
        assert set(result) == {"estado", "confianza", "incierto"}
        esp32 = client.post("/predict/esp32", files={"image": ("banana.jpg", jpeg_bytes(), "image/jpeg")})
        assert esp32.status_code == 200
        assert set(esp32.json()) == {"estado"}


def test_uncertain_prediction_is_not_a_fourth_ripeness_class(monkeypatch) -> None:
    class UncertainPredictor:
        def predict(self, image, encoded_image=None):
            return {
                "estado": "incierto",
                "confianza": 0.42,
                "incierto": True,
            }

    monkeypatch.setattr(main, "MODEL_PATH", ROOT / "models/production_model_v2.keras")
    monkeypatch.setattr(main, "METADATA_PATH", ROOT / "models/production_model_v2_metadata.json")
    with TestClient(main.app) as client:
        original = main.router.predictor
        main.router.predictor = UncertainPredictor()
        try:
            response = client.post("/predict", files={"image": ("banana.jpg", jpeg_bytes(), "image/jpeg")})
            esp32 = client.post("/predict/esp32", files={"image": ("banana.jpg", jpeg_bytes(), "image/jpeg")})
        finally:
            main.router.predictor = original
    assert response.status_code == 200
    assert response.json() == {
        "estado": "incierto",
        "confianza": 0.42,
        "incierto": True,
    }
    assert esp32.status_code == 200
    assert esp32.json() == {"estado": "incierto"}


def test_three_class_candidate_model_load_and_abstain_contract(monkeypatch, tmp_path) -> None:
    import json
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    model_path = tmp_path / "tiny_three_class.keras"
    inputs = tf.keras.Input((224, 224, 3), name="image")
    pooled = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(3, activation="softmax", kernel_initializer="zeros", bias_initializer="zeros", name="state")(pooled)
    tf.keras.Model(inputs, outputs).save(model_path)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "model_version": "3.0.0-three-class-candidate",
        "model_name": "test_three_class",
        "model_file": model_path.name,
        "classes": ["verde", "maduro", "sobre_maduro"],
        "input_size": [224, 224],
        "resize_backend": "pillow_bilinear",
        "preprocessing": "keras MobileNetV3 include_preprocessing; RGB float [0,255]",
        "temperature": 1.0,
        "uncertainty_threshold": 0.5,
    }), encoding="utf-8")
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    monkeypatch.setattr(main, "METADATA_PATH", metadata_path)
    with TestClient(main.app) as client:
        pixels = np.indices((64, 64)).sum(axis=0) % 2 * 255
        image = Image.fromarray(np.stack([pixels] * 3, axis=-1).astype("uint8"))
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        response = client.post("/predict", files={"image": ("checker.jpg", buffer.getvalue(), "image/jpeg")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["estado"] == "incierto"
    assert body["incierto"] is True
    assert set(body) == {"estado", "confianza", "incierto"}


def test_default_v4_http_contract_and_controlled_image_errors(monkeypatch) -> None:
    class V4PredictorStub:
        classes = ["verde", "maduro", "sobre_maduro"]

        def predict(self, image, encoded_image=None):
            return {"estado": "maduro", "confianza": 0.91, "incierto": False}

    monkeypatch.setattr(main, "MODEL_PATH", ROOT / "models/production_model_v4.keras")
    monkeypatch.setattr(main, "METADATA_PATH", ROOT / "models/metadata.json")
    monkeypatch.setattr(main, "BananaPredictor", lambda model_path, metadata: V4PredictorStub())
    with TestClient(main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model"] == "banana_ai_v4_remastered"
        assert health.json()["version"].startswith("4.")
        response = client.post("/predict", files={"image": ("banana.jpg", jpeg_bytes(), "image/jpeg")})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["estado"] in {"verde", "maduro", "sobre_maduro", "incierto"}
        assert set(body) == {"estado", "confianza", "incierto"}
        assert client.post("/predict", files={"image": ("broken.jpg", b"not an image", "image/jpeg")}).status_code == 400
        assert client.post("/predict/esp32", files={"image": ("broken.jpg", b"not an image", "image/jpeg")}).status_code == 400
        predictor = main.router.predictor
        assert predictor.classes == ["verde", "maduro", "sobre_maduro"]
        main.router.predictor = None
        try:
            assert client.post("/predict", files={"image": ("banana.jpg", jpeg_bytes(), "image/jpeg")}).status_code == 503
        finally:
            main.router.predictor = predictor
