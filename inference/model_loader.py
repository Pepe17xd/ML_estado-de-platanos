from pathlib import Path

import tensorflow as tf


def load_model(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)
