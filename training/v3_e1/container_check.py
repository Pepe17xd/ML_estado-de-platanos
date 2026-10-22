"""Read-only container smoke check: load V2 and evaluate one dataset batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from training.v3_e1.data import as_model_inputs, class_counts, directory_dataset, read_manifest
from training.v3_e1.model import CLASSES, load_v2_checkpoint
from training.v3_e1.train import DEFAULT_CONFIG, ROOT, load_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(config_path: Path, batch_size: int) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    config = load_config(config_path)
    checkpoint = ROOT / config["v2_checkpoint"]
    metadata = ROOT / config["v2_metadata"]
    manifest = ROOT / config["manifest"]
    dataset_dir = ROOT / config["dataset_dir"]
    model, checkpoint_info = load_v2_checkpoint(checkpoint, metadata)
    rows = read_manifest(manifest)
    counts = {split: class_counts(rows, split) for split in ("train", "validation", "test")}
    dataset = as_model_inputs(
        directory_dataset(dataset_dir / "validation", tuple(config["input_size"]), batch_size, False, int(config["seed"]))
    )
    images, labels = next(iter(dataset.take(1)))
    probabilities = np.asarray(model(images, training=False))
    loss = tf.keras.losses.SparseCategoricalCrossentropy()(labels, probabilities)
    if probabilities.shape != (int(images.shape[0]), len(CLASSES)):
        raise RuntimeError(f"unexpected model output shape: {probabilities.shape}")
    if not np.isfinite(probabilities).all() or not np.isfinite(float(loss)):
        raise RuntimeError("forward pass produced a non-finite value")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise RuntimeError("model output does not sum to class probabilities")
    return {
        "status": "passed",
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__ if hasattr(tf.keras, "__version__") else "bundled",
        "devices": [device.name for device in tf.config.list_physical_devices()],
        "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "classes": CLASSES,
        "model_input_shape": list(model.input_shape),
        "model_output_shape": list(model.output_shape),
        "checkpoint_sha256": checkpoint_info["sha256"],
        "manifest_sha256": sha256_file(manifest),
        "manifest_counts": counts,
        "batch_shape": list(images.shape),
        "batch_dtype": images.dtype.name,
        "batch_range": [float(tf.reduce_min(images)), float(tf.reduce_max(images))],
        "batch_labels": [int(value) for value in labels.numpy()],
        "forward_loss": float(loss),
        "probability_shape": list(probabilities.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
