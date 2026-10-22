"""Serializable, configurable E1 image augmentations for TensorFlow/Keras 2.15.

Inputs and outputs use RGB float32 pixels in [0, 255], matching the saved V2
model's embedded MobileNetV3 Rescaling layer. Photometric operations are
sampled independently, then limited to ``max_photometric_ops`` per image.
"""
from __future__ import annotations

from typing import Any

import tensorflow as tf


PHOTO_FAMILIES = (
    "brightness",
    "contrast",
    "gamma",
    "saturation",
    "white_balance",
    "shadow",
)


def default_augmentation_config() -> dict[str, Any]:
    """Return a new copy of the conservative defaults used by V3-E1."""
    return {
        "seed": 2026,
        "max_photometric_ops": 2,
        "geometric": {
            "horizontal_flip": True,
            "rotation_factor": 0.08,
            "zoom_factor": 0.12,
        },
        "photometric": {
            "brightness": {"enabled": True, "probability": 0.25, "delta": 14.0},
            # V2 already applies RandomContrast(0.15) while training. Keep the
            # extra E1 contrast available for its isolated ablation only.
            "contrast": {"enabled": False, "probability": 0.25, "factor_range": [0.9, 1.1]},
            "gamma": {"enabled": True, "probability": 0.20, "range": [0.9, 1.1]},
            "saturation": {"enabled": True, "probability": 0.20, "range": [0.9, 1.1]},
            "white_balance": {"enabled": True, "probability": 0.15, "gain_range": [0.96, 1.04]},
            "shadow": {
                "enabled": True,
                "probability": 0.15,
                "strength_range": [0.10, 0.22],
                "radius_range": [0.18, 0.38],
                "softness": 0.10,
            },
        },
    }


def validate_augmentation_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise TypeError("augmentation config must be a dictionary")
    max_ops = config.get("max_photometric_ops", 2)
    if not isinstance(max_ops, int) or max_ops < 0 or max_ops > len(PHOTO_FAMILIES):
        raise ValueError("max_photometric_ops must be an integer from 0 to 6")
    geometric = config.get("geometric", {})
    for name in ("rotation_factor", "zoom_factor"):
        value = float(geometric.get(name, 0.0))
        if value < 0.0 or value > 0.25:
            raise ValueError(f"geometric.{name} must be in [0, 0.25]")
    photo = config.get("photometric", {})
    for family in PHOTO_FAMILIES:
        item = photo.get(family, {"enabled": False, "probability": 0.0})
        probability = float(item.get("probability", 0.0))
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{family}.probability must be in [0, 1]")
        if family == "brightness" and float(item.get("delta", 0.0)) < 0.0:
            raise ValueError("brightness.delta must be non-negative")
        range_key = {
            "contrast": "factor_range",
            "gamma": "range",
            "saturation": "range",
            "white_balance": "gain_range",
            "shadow": "strength_range",
        }.get(family)
        if range_key and range_key in item:
            low, high = map(float, item[range_key])
            if low <= 0.0 or high < low:
                raise ValueError(f"{family}.{range_key} must be positive and ordered")
        if family == "shadow":
            low, high = map(float, item.get("radius_range", [0.18, 0.38]))
            if low <= 0.0 or high < low or high > 1.0:
                raise ValueError("shadow.radius_range must be ordered within (0, 1]")
            if not 0.0 < float(item.get("softness", 0.1)) <= 1.0:
                raise ValueError("shadow.softness must be in (0, 1]")


def ablation_config(base: dict[str, Any], ablation: str) -> dict[str, Any]:
    """Copy config and enable either one photometric family or the E1 set."""
    import copy

    result = copy.deepcopy(base)
    choices = {"none", "combined", *PHOTO_FAMILIES}
    if ablation not in choices:
        raise ValueError(f"unknown ablation {ablation!r}; choose from {sorted(choices)}")
    for family in PHOTO_FAMILIES:
        item = result.setdefault("photometric", {}).setdefault(family, {})
        if ablation == "combined":
            # Respect the explicit default enable/disable state in the config.
            item["enabled"] = bool(item.get("enabled", False))
        else:
            # A named ablation intentionally enables exactly that family.
            item["enabled"] = ablation == family
    return result


@tf.keras.utils.register_keras_serializable(package="BananaV3E1")
class LightingAugmentation(tf.keras.layers.Layer):
    """Apply moderate, seeded geometric and photometric transforms at train time."""

    def __init__(self, augmentation_config: dict[str, Any] | None = None, seed: int = 2026, **kwargs: Any):
        super().__init__(**kwargs)
        self.augmentation_config = augmentation_config or default_augmentation_config()
        self.seed = int(seed)
        validate_augmentation_config(self.augmentation_config)
        geo = self.augmentation_config.get("geometric", {})
        self._flip = tf.keras.layers.RandomFlip("horizontal", seed=self.seed + 11) if geo.get("horizontal_flip", True) else None
        self._rotation = tf.keras.layers.RandomRotation(
            float(geo.get("rotation_factor", 0.08)), seed=self.seed + 13, fill_mode="reflect"
        )
        zoom = float(geo.get("zoom_factor", 0.12))
        self._zoom = tf.keras.layers.RandomZoom((-zoom, zoom), seed=self.seed + 17, fill_mode="reflect")

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        images = tf.cast(inputs, tf.float32)
        if training is None or training is False:
            return images
        if isinstance(training, bool):
            return self._augment(images)
        return tf.cond(tf.cast(training, tf.bool), lambda: self._augment(images), lambda: images)

    def _augment(self, images: tf.Tensor) -> tf.Tensor:
        if self._flip is not None:
            images = self._flip(images, training=True)
        images = self._rotation(images, training=True)
        images = self._zoom(images, training=True)
        base_seed = tf.random.uniform([2], minval=0, maxval=2**31 - 1, dtype=tf.int32, seed=self.seed)
        indices = tf.range(tf.shape(images)[0])
        signature = tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32)
        result = tf.map_fn(
            lambda pair: self._augment_one(pair[0], tf.random.experimental.stateless_fold_in(base_seed, pair[1])),
            (images, indices),
            fn_output_signature=signature,
            parallel_iterations=1,
        )
        return tf.clip_by_value(result, 0.0, 255.0)

    def _augment_one(self, image: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        photo = self.augmentation_config.get("photometric", {})
        enabled = []
        priorities = []
        for index, family in enumerate(PHOTO_FAMILIES):
            item = photo.get(family, {})
            p = float(item.get("probability", 0.0)) if item.get("enabled", False) else 0.0
            family_seed = tf.random.experimental.stateless_fold_in(seed, index * 7 + 1)
            draw = tf.random.stateless_uniform([], family_seed)
            priority_seed = tf.random.experimental.stateless_fold_in(seed, index * 7 + 2)
            priority = tf.random.stateless_uniform([], priority_seed)
            enabled.append(draw < p)
            priorities.append(tf.where(draw < p, priority, tf.constant(2.0, tf.float32)))
        priority_tensor = tf.stack(priorities)
        ranks = tf.argsort(tf.argsort(priority_tensor, stable=True), stable=True)
        active = tf.logical_and(tf.stack(enabled), ranks < self.augmentation_config.get("max_photometric_ops", 2))

        x = image
        for index, family in enumerate(PHOTO_FAMILIES):
            family_seed = tf.random.experimental.stateless_fold_in(seed, index * 7 + 3)
            x = tf.cond(active[index], lambda x=x, family=family, s=family_seed: self._apply_family(x, family, photo.get(family, {}), s), lambda x=x: x)
        return tf.clip_by_value(x, 0.0, 255.0)

    @staticmethod
    def _sample(seed: tf.Tensor, value_range: list[float] | tuple[float, float]) -> tf.Tensor:
        low, high = map(float, value_range)
        return tf.random.stateless_uniform([], seed, minval=low, maxval=high, dtype=tf.float32)

    def _apply_family(self, image: tf.Tensor, family: str, item: dict[str, Any], seed: tf.Tensor) -> tf.Tensor:
        unit = tf.clip_by_value(image / 255.0, 0.0, 1.0)
        if family == "brightness":
            delta = self._sample(seed, [-float(item.get("delta", 14.0)), float(item.get("delta", 14.0))])
            return image + delta
        if family == "contrast":
            factor = self._sample(seed, item.get("factor_range", [0.9, 1.1]))
            return tf.image.adjust_contrast(unit, factor) * 255.0
        if family == "gamma":
            gamma = self._sample(seed, item.get("range", [0.9, 1.1]))
            return tf.pow(unit, gamma) * 255.0
        if family == "saturation":
            factor = self._sample(seed, item.get("range", [0.9, 1.1]))
            return tf.image.adjust_saturation(unit, factor) * 255.0
        if family == "white_balance":
            low, high = item.get("gain_range", [0.96, 1.04])
            red = self._sample(tf.random.experimental.stateless_fold_in(seed, 1), [low, high])
            blue = self._sample(tf.random.experimental.stateless_fold_in(seed, 2), [low, high])
            gains = tf.stack([red, 1.0, blue])[None, None, :]
            return image * gains
        if family == "shadow":
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            yy = tf.linspace(-1.0, 1.0, height)[:, None]
            xx = tf.linspace(-1.0, 1.0, width)[None, :]
            cy = self._sample(tf.random.experimental.stateless_fold_in(seed, 1), [-0.7, 0.7])
            cx = self._sample(tf.random.experimental.stateless_fold_in(seed, 2), [-0.7, 0.7])
            radius = self._sample(tf.random.experimental.stateless_fold_in(seed, 3), item.get("radius_range", [0.18, 0.38]))
            aspect = self._sample(tf.random.experimental.stateless_fold_in(seed, 4), [0.65, 1.5])
            strength = self._sample(tf.random.experimental.stateless_fold_in(seed, 5), item.get("strength_range", [0.10, 0.22]))
            softness = float(item.get("softness", 0.10))
            distance = tf.sqrt(tf.square((yy - cy) / aspect) + tf.square(xx - cx))
            mask = tf.sigmoid((radius - distance) / softness)[..., None]
            return image * (1.0 - strength * mask)
        raise ValueError(f"unsupported photometric family: {family}")

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"augmentation_config": self.augmentation_config, "seed": self.seed})
        return config
