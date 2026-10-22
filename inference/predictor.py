from pathlib import Path
import hashlib

import numpy as np
from PIL import Image

from inference.model_loader import load_model
from inference.image_quality import quality_flags
from inference.preprocessing import preprocess


class BananaPredictor:
    def __init__(self, model_path: Path, metadata: dict):
        self.metadata = metadata
        model_path = Path(model_path).resolve()
        expected_hash = metadata.get("checkpoint_sha256")
        if expected_hash:
            digest = hashlib.sha256()
            with model_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_hash:
                raise ValueError("Checkpoint SHA-256 does not match model metadata")
        declared_file = metadata.get("model_file")
        if declared_file and Path(declared_file).name != model_path.name:
            raise ValueError("Model filename does not match model metadata")
        self.model = load_model(model_path)
        selected = metadata.get("selected_checkpoint")
        if selected and model_path != (Path(metadata.get("_metadata_path", model_path.parent)) / selected).resolve():
            raise ValueError("Selected checkpoint in metadata does not match MODEL_PATH")
        self.input_size = tuple(metadata.get("input_size", metadata.get("input_shape", [])[:2]))
        self.classes = metadata.get("classes")
        self.is_three_class_candidate = self.classes == ["verde", "maduro", "sobre_maduro"]
        self.is_legacy_four_class = self.classes == ["verde", "maduro", "sobre_maduro", "podrido"]
        self.is_production_v4 = (
            self.is_three_class_candidate
            and str(metadata.get("model_version", "")).startswith("4.")
            and metadata.get("model_name") == "banana_ai_v4_remastered"
            and metadata.get("status") == "production_local"
        )
        if not (self.is_three_class_candidate or self.is_legacy_four_class):
            raise ValueError("Model metadata classes must use the exact three-class production or legacy four-class order")
        if self.is_three_class_candidate and metadata.get("preprocessing") not in {
            "keras MobileNetV3 include_preprocessing; RGB float [0,255]",
            "keras MobileNetV3 include_preprocessing; RGB float32 [0,255]",
            "embedded_mobilenet_v3",
        }:
            raise ValueError("Three-class metadata must declare embedded MobileNetV3 preprocessing")
        version = str(metadata.get("model_version", ""))
        self.resize_backend = metadata.get(
            "resize_backend", "tensorflow_bilinear" if version.startswith("3.") else "pillow_bilinear"
        )
        if self.input_size != (224, 224):
            raise ValueError(f"Model metadata must declare a 224x224 input, got {self.input_size}")
        expected_outputs = len(self.classes)
        if tuple(self.model.input_shape[1:]) != (224, 224, 3) or tuple(self.model.output_shape[1:]) != (expected_outputs,):
            raise ValueError(f"Model checkpoint must accept RGB 224x224 images and return {expected_outputs} class scores")
        self.temperature = float(metadata.get("temperature", 1.0))
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.uncertainty_threshold = float(metadata.get("uncertainty_threshold", 0.0))
        if not 0.0 <= self.uncertainty_threshold <= 1.0:
            raise ValueError("uncertainty_threshold must be in [0,1]")

    def predict(self, image: Image.Image, encoded_image: bytes | None = None) -> dict:
        if self.is_three_class_candidate:
            flags = quality_flags(image)
            if flags:
                return {
                    "estado": "incierto",
                    "confianza": 0.0,
                    "dias_restantes": None,
                    "motivo_incertidumbre": ",".join(flags),
                }
        probabilities = np.asarray(self.model.predict(
            preprocess(image, self.input_size, self.resize_backend, encoded_image), verbose=0
        ))[0]
        if probabilities.shape != (len(self.classes),) or not np.all(np.isfinite(probabilities)):
            raise ValueError("Model returned invalid class scores")
        if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1.0, atol=1e-4):
            raise ValueError("Model output must be normalized class probabilities")
        if self.is_three_class_candidate and self.temperature != 1.0:
            logits = np.log(np.clip(probabilities, 1e-8, 1.0)) / self.temperature
            logits -= logits.max()
            probabilities = np.exp(logits) / np.exp(logits).sum()
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        if self.is_three_class_candidate and confidence < self.uncertainty_threshold:
            return {
                "estado": "incierto",
                "confianza": round(confidence, 4),
                "dias_restantes": None,
                "motivo_incertidumbre": "confianza_inferior_al_umbral_validado",
            }
        return {
            "estado": self.classes[index],
            "confianza": round(confidence, 4),
            "dias_restantes": None,
        }

if __name__ == "__main__":
    import sys
    import json

    from pathlib import Path

    if len(sys.argv) < 2:
        print("Uso: python -m inference.predictor <imagen>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    # cargar metadata
    metadata_path = Path("models/metadata.json")
    model_path = Path("models/production_model_v4.keras")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    predictor = BananaPredictor(
        model_path=model_path,
        metadata=metadata
    )

    image = Image.open(image_path)

    result = predictor.predict(image)

    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ))
