"""Load and safely adapt the saved four-class V2 model without rebuilding it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from training.v3_e1.augmentations import LightingAugmentation

CLASSES = ["verde", "maduro", "sobre_maduro", "podrido"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_four_class_v2(model: tf.keras.Model, metadata: dict[str, Any] | None = None) -> None:
    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise ValueError(f"V2 input shape must be (224, 224, 3), got {model.input_shape}")
    if tuple(model.output_shape[1:]) != (4,):
        raise ValueError(f"V2 output shape must be four class probabilities, got {model.output_shape}")
    if list(model.output_names) != ["state"]:
        raise ValueError(f"V2 must have one output named 'state', got {model.output_names}")
    try:
        augmentation = model.get_layer("augmentation")
        rescaling = model.get_layer("rescaling")
    except ValueError as exc:
        raise ValueError("V2 checkpoint is missing the expected augmentation or embedded MobileNetV3 rescaling layer") from exc
    if not isinstance(augmentation, (tf.keras.Sequential, LightingAugmentation)):
        raise ValueError("model augmentation layer is neither the saved V2 Sequential nor the V3-E1 layer")
    if not isinstance(rescaling, tf.keras.layers.Rescaling):
        raise ValueError("V2 checkpoint does not expose the expected embedded Rescaling layer")
    scale = np.asarray(rescaling.scale, dtype=np.float32).reshape(-1)
    offset = np.asarray(rescaling.offset, dtype=np.float32).reshape(-1)
    if not np.allclose(scale, 1.0 / 127.5, rtol=1e-5) or not np.allclose(offset, -1.0, atol=1e-6):
        raise ValueError("V2 embedded Rescaling does not match RGB [0,255] to MobileNetV3 [-1,1]")
    if metadata is not None:
        classes = metadata.get("classes")
        if classes != CLASSES:
            raise ValueError(f"V2 class order mismatch: expected {CLASSES}, got {classes}")
        if metadata.get("preprocessing") != "embedded_mobilenet_v3":
            raise ValueError("V2 metadata does not declare embedded_mobilenet_v3 preprocessing")


def load_v2_checkpoint(path: Path, metadata_path: Path | None = None) -> tuple[tf.keras.Model, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"V2 checkpoint not found: {path}")
    metadata: dict[str, Any] = {}
    if metadata_path is not None:
        import json

        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(path, compile=False)
    validate_four_class_v2(model, metadata or None)
    return model, {"path": str(path.resolve()), "sha256": sha256_file(path), "metadata": metadata}


def load_warm_start_weights(model: tf.keras.Model, path: Path) -> dict[str, str]:
    """Load a prior E1 checkpoint only after validating its sidecar and full layout."""
    path = Path(path)
    if not path.is_file() or path.suffix != ".keras":
        raise ValueError("warm-start path must be an existing .keras model")
    metadata_path = path.parent / "run_metadata.json"
    if not metadata_path.is_file():
        raise ValueError("warm-start requires the run_metadata.json saved beside the V3-E1 checkpoint")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("classes") != CLASSES or metadata.get("input_shape") != [224, 224, 3]:
        raise ValueError("warm-start metadata does not match the V2 input shape and class order")
    candidate = tf.keras.models.load_model(path, compile=False)
    validate_four_class_v2(candidate)
    if [layer.name for layer in model.layers] != [layer.name for layer in candidate.layers]:
        raise ValueError("warm-start layer names differ from V2")
    current_shapes = [weight.shape for weight in model.get_weights()]
    candidate_weights = candidate.get_weights()
    if current_shapes != [weight.shape for weight in candidate_weights]:
        raise ValueError("warm-start weight layout differs from V2")
    model.set_weights(candidate_weights)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def set_head_trainable(model: tf.keras.Model) -> None:
    for layer in model.layers:
        layer.trainable = layer.name in {"augmentation", "global_average_pooling2d", "dropout", "dense", "state"}
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def set_finetune_trainable(model: tf.keras.Model, last_layers: int = 30) -> None:
    names = [layer.name for layer in model.layers]
    try:
        start = names.index("rescaling") + 1
        stop = names.index("global_average_pooling2d")
    except ValueError as exc:
        raise ValueError("Could not locate the saved V2 backbone boundary") from exc
    backbone = model.layers[start:stop]
    if not backbone or last_layers < 1:
        raise ValueError("Invalid MobileNetV3 fine-tuning layer selection")
    trainable_backbone = set(id(layer) for layer in backbone[-last_layers:])
    for layer in model.layers:
        is_head = layer.name in {"augmentation", "global_average_pooling2d", "dropout", "dense", "state"}
        layer.trainable = is_head or id(layer) in trainable_backbone
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
