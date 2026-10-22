"""Conservative image-quality checks for candidate abstention.

Thresholds here are engineering guardrails, not learned/validated banana
detection rules. They can flag only obvious image-quality failures.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def quality_flags(image: Image.Image) -> list[str]:
    rgb = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    gray_image = rgb.convert("L")
    gray = np.asarray(gray_image, dtype=np.float32)
    flags: list[str] = []
    mean_luminance = float(gray.mean())
    clipped_fraction = float((gray >= 250).mean())
    if mean_luminance < 12.0:
        flags.append("imagen_demasiado_oscura")
    if mean_luminance > 245.0 or clipped_fraction > 0.70:
        flags.append("imagen_sobreexpuesta")
    edge_strength = np.asarray(gray_image.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    if float(edge_strength.var()) < 1.5:
        flags.append("imagen_posiblemente_borrosa")
    return flags
