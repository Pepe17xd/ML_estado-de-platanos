from pathlib import Path

import numpy as np
from PIL import Image


def preprocess(image: Image.Image, input_size: tuple[int, int]) -> np.ndarray:
    if image.width < 1 or image.height < 1:
        raise ValueError("Image has invalid dimensions")
    resized = image.convert("RGB").resize(input_size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)[None, ...]
