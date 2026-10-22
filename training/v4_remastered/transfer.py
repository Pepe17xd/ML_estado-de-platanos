"""Strict V2 backbone transfer into a three-class MobileNetV3 model."""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from training.v3_e1.model import load_v2_checkpoint


def initialize_from_v2(model: tf.keras.Model, checkpoint: Path, metadata: Path) -> dict:
    """Copy every shape-compatible weighted layer except the old 4-way head.

    The source model is validated as the repository's V2 contract first. A
    missing or shape-mismatched backbone layer aborts instead of partially
    mixing incompatible architectures.
    """
    source, source_info = load_v2_checkpoint(checkpoint, metadata)
    source_layers = {layer.name: layer for layer in source.layers}
    copied, copied_tensors, skipped = [], 0, []
    for target_layer in model.layers:
        if target_layer.name == "state":
            skipped.append({"layer": target_layer.name, "reason": "three-class output head reset"})
            continue
        target_weights = target_layer.get_weights()
        if not target_weights:
            continue
        source_layer = source_layers.get(target_layer.name)
        if source_layer is None:
            skipped.append({"layer": target_layer.name, "reason": "no source layer"})
            continue
        source_weights = source_layer.get_weights()
        if [tuple(x.shape) for x in source_weights] != [tuple(x.shape) for x in target_weights]:
            raise ValueError(
                f"V2 layer {target_layer.name!r} shape mismatch: "
                f"{[x.shape for x in source_weights]} != {[x.shape for x in target_weights]}"
            )
        target_layer.set_weights(source_weights)
        copied.append(target_layer.name)
        copied_tensors += len(source_weights)

    if not copied or "Conv" not in copied or "dense" not in copied:
        raise ValueError("V2 transfer did not copy the expected MobileNetV3 backbone and dense projection")
    return {
        "source": source_info,
        "copied_weighted_layers": copied,
        "copied_tensor_count": copied_tensors,
        "skipped": skipped,
        "classification_head_reset": True,
    }
