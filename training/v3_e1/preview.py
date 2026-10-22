"""Explicitly create a visual contact sheet for E1 augmentations."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from training.v3_e1.augmentations import PHOTO_FAMILIES, LightingAugmentation, default_augmentation_config
from training.v3_e1.quality import validate_augmentation_pair

TITLES = ["original", "brillo", "gamma", "balance color", "saturación", "sombra", "combinación"]


def render_preview(source: Path, destination: Path, seed: int = 2026) -> dict[str, object]:
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite preview: {destination}")
    with Image.open(source) as opened:
        original_pil = opened.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    original = np.asarray(original_pil, dtype=np.float32)
    variants = [original]
    family_for_title = [None, "brightness", "gamma", "white_balance", "saturation", "shadow", "combined"]
    for index, family in enumerate(family_for_title[1:], start=1):
        config = default_augmentation_config()
        config["seed"] = seed + index
        config["geometric"] = {"horizontal_flip": False, "rotation_factor": 0.0, "zoom_factor": 0.0}
        config["max_photometric_ops"] = 1 if family != "combined" else 2
        if family == "combined":
            for name in PHOTO_FAMILIES:
                config["photometric"][name]["probability"] = 1.0
        else:
            for name in PHOTO_FAMILIES:
                config["photometric"][name]["enabled"] = name == family
                config["photometric"][name]["probability"] = 1.0 if name == family else 0.0
        layer = LightingAugmentation(config, seed=seed + index)
        changed = layer(original[None, ...], training=True)[0].numpy()
        validate_augmentation_pair(original, changed, (224, 224))
        variants.append(changed)

    sheet = Image.new("RGB", (224 * len(variants), 250), color="white")
    draw = ImageDraw.Draw(sheet)
    for index, (title, values) in enumerate(zip(TITLES, variants)):
        array = np.clip(np.rint(values), 0, 255).astype(np.uint8)
        sheet.paste(Image.fromarray(array, mode="RGB"), (index * 224, 24))
        draw.text((index * 224 + 5, 5), title, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG")
    return {"source": str(source.resolve()), "output": str(destination.resolve()), "shape": [224, 224, 3], "variants": TITLES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(render_preview(args.image, args.output, args.seed))


if __name__ == "__main__":
    main()
