"""Conservative image-quality checks for candidate abstention.

Thresholds here are engineering guardrails, not learned/validated banana
detection rules. They can flag only obvious image-quality failures.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def quality_flags(image: Image.Image) -> list[str]:
    rgb = np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)
    gray_image = Image.fromarray(rgb).convert("L")
    gray = np.asarray(gray_image, dtype=np.float32)
    flags: list[str] = []
    mean_luminance = float(gray.mean())
    clipped_fraction = float((gray >= 250).mean())
    if mean_luminance < 12.0:
        flags.append("imagen_demasiado_oscura")

    # Foreground is not inferred from chroma: green bananas on green backgrounds
    # and low-chroma overripe fruit defeat that assumption. Instead detect broad
    # luminance clipping across the whole frame and require clipping to dominate
    # the high-luminance tail. A white/black/colored background alone therefore
    # cannot trigger this, while subject-heavy extreme exposure still can.
    # Distinguish a clipped background mode from broad clipping throughout the
    # image by checking whether the non-white portion is itself bright. This
    # uses a robust quantile rather than attempting color-dependent foreground
    # segmentation.
    non_clipped = gray[gray < 250.0]
    non_clipped_fraction = float(non_clipped.size / gray.size)
    overexposed_subject = (
        mean_luminance > 245.0
        or (
            clipped_fraction > 0.35
            and non_clipped_fraction > 0.08
            and float(np.percentile(non_clipped, 90)) > 235.0
        )
    )
    if overexposed_subject:
        flags.append("imagen_sobreexpuesta")
    edge_strength = np.asarray(gray_image.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    if float(edge_strength.var()) < 1.5:
        flags.append("imagen_posiblemente_borrosa")
    return flags
