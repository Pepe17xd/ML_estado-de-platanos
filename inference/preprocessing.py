from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image


def preprocess(
    image: Image.Image,
    input_size: tuple[int, int],
    resize_backend: str = "pillow_bilinear",
    encoded_image: bytes | None = None,
) -> np.ndarray:
    if image.width < 1 or image.height < 1:
        raise ValueError("Image has invalid dimensions")
    if resize_backend == "tensorflow_bilinear":
        import tensorflow as tf

        encoded_is_webp = encoded_image is not None and encoded_image[8:12] == b"WEBP"
        if encoded_image is not None and not encoded_is_webp:
            pixels = tf.io.decode_image(encoded_image, channels=3, expand_animations=False)
            pixels.set_shape([None, None, 3])
        else:
            pixels = tf.convert_to_tensor(np.asarray(image.convert("RGB"), dtype=np.uint8))
        resized = tf.image.resize(pixels, input_size, method="bilinear", antialias=False)
        return np.asarray(resized, dtype=np.float32)[None, ...]
    if resize_backend != "pillow_bilinear":
        raise ValueError(f"unsupported resize backend: {resize_backend}")
    resized = image.convert("RGB").resize(input_size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)[None, ...]


def preprocess_encoded(encoded_image: bytes, input_size: tuple[int, int], resize_backend: str) -> np.ndarray:
    """Decode an encoded image and apply the selected training/inference resize path."""
    if resize_backend == "tensorflow_bilinear":
        import tensorflow as tf

        pixels = tf.io.decode_image(encoded_image, channels=3, expand_animations=False)
        pixels.set_shape([None, None, 3])
        resized = tf.image.resize(pixels, input_size, method="bilinear", antialias=False)
        return np.asarray(resized, dtype=np.float32)[None, ...]
    with Image.open(BytesIO(encoded_image)) as image:
        return preprocess(image, input_size, resize_backend=resize_backend)
