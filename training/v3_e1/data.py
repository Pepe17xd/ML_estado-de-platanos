"""Deterministic directory datasets and manifest checks for the historical V2 split."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import tensorflow as tf

from training.v3_e1.model import CLASSES


def read_manifest(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"split", "class_name", "source_dataset", "image_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest must contain {sorted(required)}")
    if any(row["class_name"] not in CLASSES for row in rows):
        raise ValueError("manifest contains labels outside the V2 four-class order")
    return rows


def class_counts(rows: list[dict[str, str]], split: str) -> dict[str, int]:
    counts = Counter(row["class_name"] for row in rows if row["split"] == split)
    missing = [name for name in CLASSES if counts[name] == 0]
    if missing:
        raise ValueError(f"split {split!r} is missing classes: {missing}")
    return {name: int(counts[name]) for name in CLASSES}


def class_weights(rows: list[dict[str, str]]) -> dict[int, float]:
    counts = class_counts(rows, "train")
    maximum = max(counts.values())
    return {index: maximum / counts[name] for index, name in enumerate(CLASSES)}


def directory_dataset(
    directory: Path,
    image_size: tuple[int, int],
    batch_size: int,
    training: bool,
    seed: int,
    photometric_augmentation: tf.keras.layers.Layer | None = None,
) -> tf.data.Dataset:
    if not Path(directory).is_dir():
        raise FileNotFoundError(f"image directory not found: {directory}")
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        class_names=CLASSES,
        label_mode="int",
        color_mode="rgb",
        image_size=tuple(image_size),
        batch_size=batch_size,
        shuffle=training,
        seed=seed,
    )
    if training and photometric_augmentation is not None:
        dataset = dataset.map(
            lambda images, labels: (photometric_augmentation(images, training=True), labels),
            num_parallel_calls=1,
            deterministic=True,
        )
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options).prefetch(tf.data.AUTOTUNE)


def as_model_inputs(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Convert inferred labels to named state targets without pixel rescaling."""
    def convert(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return tf.cast(images, tf.float32), tf.cast(labels, tf.int32)

    return dataset.map(convert, num_parallel_calls=1, deterministic=True).prefetch(tf.data.AUTOTUNE)
