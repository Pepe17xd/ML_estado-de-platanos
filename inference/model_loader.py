from pathlib import Path

import tensorflow as tf

# Register the custom augmentation layer embedded in experimental V3 .keras
# files before deserialization. V2 models do not depend on this module.
from training.v3_e1.augmentations import LightingAugmentation  # noqa: F401


def load_model(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"BananaV3E1>LightingAugmentation": LightingAugmentation},
    )
