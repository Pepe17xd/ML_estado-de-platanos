"""Input/output sanity checks for E1 previews and training batches."""
from __future__ import annotations

from typing import Any

import tensorflow as tf


def inspect_image_tensor(images: Any, expected_size: tuple[int, int] | None = None) -> dict[str, Any]:
    tensor = tf.convert_to_tensor(images)
    if tensor.dtype != tf.float32:
        raise TypeError(f"expected float32 RGB pixels, got {tensor.dtype.name}")
    if tensor.shape.rank not in (3, 4):
        raise ValueError(f"expected HWC or NHWC image tensor, got rank {tensor.shape.rank}")
    if tensor.shape[-1] != 3:
        raise ValueError(f"expected 3 RGB channels, got {tensor.shape[-1]}")
    tf.debugging.assert_all_finite(tensor, "image tensor contains NaN or infinite values")
    if expected_size is not None:
        actual = tuple(int(v) for v in tensor.shape[-3:-1])
        if actual != tuple(expected_size):
            raise ValueError(f"expected spatial size {expected_size}, got {actual}")
    min_value = float(tf.reduce_min(tensor).numpy())
    max_value = float(tf.reduce_max(tensor).numpy())
    if min_value < 0.0 or max_value > 255.0:
        raise ValueError(f"expected values in [0, 255], observed [{min_value}, {max_value}]")
    pixels = tensor if tensor.shape.rank == 4 else tensor[None, ...]
    per_pixel_mean = tf.reduce_mean(pixels, axis=-1)
    black_fraction = float(tf.reduce_mean(tf.cast(per_pixel_mean <= 4.0, tf.float32)).numpy())
    white_clipped_fraction = float(tf.reduce_mean(tf.cast(tf.reduce_min(pixels, axis=-1) >= 250.0, tf.float32)).numpy())
    flags = []
    if black_fraction >= 0.95:
        flags.append("practically_black")
    if white_clipped_fraction >= 0.95:
        flags.append("practically_saturated")
    return {
        "shape": tuple(int(v) for v in tensor.shape),
        "dtype": tensor.dtype.name,
        "min": min_value,
        "max": max_value,
        "black_pixel_fraction": black_fraction,
        "white_clipped_pixel_fraction": white_clipped_fraction,
        "flags": flags,
    }


def validate_augmentation_pair(original: Any, augmented: Any, expected_size: tuple[int, int] | None = None) -> dict[str, Any]:
    before = inspect_image_tensor(original, expected_size)
    after = inspect_image_tensor(augmented, expected_size)
    if before["shape"] != after["shape"]:
        raise ValueError(f"augmentation changed shape from {before['shape']} to {after['shape']}")
    return {"original": before, "augmented": after}
