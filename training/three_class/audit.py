"""Read-only audit of the V1/V2 image inventories and three-class policy."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image, ImageFilter, UnidentifiedImageError

from training.three_class.contract import (
    AMBIGUOUS_CROSS_LABEL_LINKS,
    CLASSES,
    SAME_LABEL_VISUAL_LINKS,
    SOURCE_TO_CLASS,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def d_hash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def audit(manifest: Path, root: Path) -> dict:
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "split", "class_name", "original_state", "group_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest must contain {sorted(required)}")

    raw_by_label = Counter(row["original_state"] for row in rows)
    by_split_label = {split: dict(Counter(row["original_state"] for row in rows if row["split"] == split))
                      for split in sorted({row["split"] for row in rows})}
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    mixed_groups = {
        group_id: {"rows": len(members), "labels": sorted({member["original_state"] for member in members}),
                   "splits": sorted({member["split"] for member in members})}
        for group_id, members in groups.items() if len({member["original_state"] for member in members}) > 1
    }

    exact: dict[str, list[dict[str, str]]] = defaultdict(list)
    phashes: list[tuple[int, dict[str, str]]] = []
    dimensions = Counter()
    exif_cameras = Counter()
    exposure = defaultdict(list)
    blur = defaultdict(list)
    unreadable = []
    for index, row in enumerate(rows):
        path = Path(row["image_path"])
        if not path.is_file():
            parts = path.parts
            rebased = next((root.joinpath(*parts[i:]) for i, part in enumerate(parts)
                            if part.lower() == "dataset" and root.joinpath(*parts[i:]).is_file()), None)
            if rebased is None:
                unreadable.append({"row": index, "path": row["image_path"], "error": "not_found"})
                continue
            path = rebased
        try:
            encoded = path.read_bytes()
            digest = hashlib.sha256(encoded).hexdigest()
            exact[digest].append(row)
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                dimensions[f"{image.width}x{image.height}"] += 1
                exif = opened.getexif()
                if exif:
                    mapping = {name: tag for tag, name in ExifTags.TAGS.items()}
                    camera = tuple(str(exif.get(mapping.get(key, -1), "")) for key in ("Make", "Model"))
                    if any(camera): exif_cameras[camera] += 1
                gray = np.asarray(image.convert("L"), dtype=np.uint8)
                exposure[row["original_state"]].append({
                    "mean_luminance": float(gray.mean()),
                    "dark_pixel_fraction": float((gray < 12).mean()),
                    "clipped_pixel_fraction": float((gray > 245).mean()),
                })
                edge = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
                blur[row["original_state"]].append(float(edge.var()))
                phashes.append((d_hash(image), row))
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            unreadable.append({"row": index, "path": str(path), "error": type(exc).__name__})

    duplicate_groups = [members for members in exact.values() if len(members) > 1]
    cross_split_duplicates = [members for members in duplicate_groups if len({m["split"] for m in members}) > 1]
    near_pairs = []
    near_counts = Counter()
    near_split_cross = 0
    near_label_cross = 0
    near_different_group = 0
    # Perceptual hash is a candidate finder, not proof of duplicate identity.
    for i, (hash_a, row_a) in enumerate(phashes):
        for hash_b, row_b in phashes[i + 1:]:
            distance = (hash_a ^ hash_b).bit_count()
            if distance <= 4 and row_a["image_path"] != row_b["image_path"]:
                near_counts[distance] += 1
                near_split_cross += row_a["split"] != row_b["split"]
                near_label_cross += row_a["original_state"] != row_b["original_state"]
                near_different_group += row_a["group_id"] != row_b["group_id"]
                if len(near_pairs) < 100:
                    near_pairs.append({"distance": distance, "path_a": row_a["image_path"],
                                       "path_b": row_b["image_path"], "label_a": row_a["original_state"],
                                       "label_b": row_b["original_state"], "split_a": row_a["split"],
                                       "split_b": row_b["split"], "group_a": row_a["group_id"],
                                       "group_b": row_b["group_id"],
                                       "same_visual_group": row_a["group_id"] == row_b["group_id"]})

    def summarize(records: list[dict]) -> dict:
        if not records: return {"n": 0}
        return {key: float(np.median([record[key] for record in records])) for key in records[0]} | {"n": len(records)}

    return {
        "manifest": str(manifest),
        "rows": len(rows),
        "source_label_counts": dict(raw_by_label),
        "split_by_source_label": by_split_label,
        "target_policy": {
            "mapping": SOURCE_TO_CLASS,
            "excluded_source_label": "rotten (excluded; never merged into sobre_maduro)",
            "excluded_conflicting_visual_groups": len(mixed_groups),
            "ambiguous_group_ids": mixed_groups,
            "manually_reviewed_cross_label_links": sorted([list(pair) for pair in AMBIGUOUS_CROSS_LABEL_LINKS]),
            "manually_reviewed_same_label_links": sorted([list(pair) for pair in SAME_LABEL_VISUAL_LINKS]),
            "target_classes": CLASSES,
        },
        "exact_duplicates": {
            "groups": len(duplicate_groups),
            "files_in_groups": sum(len(group) for group in duplicate_groups),
            "cross_split_groups": len(cross_split_duplicates),
        },
        "near_duplicate_candidates_dhash_distance_le_4": {
            "algorithm": "64-bit difference hash; permissive candidate threshold only, not a duplicate decision",
            "candidate_pair_count": int(sum(near_counts.values())),
            "distance_histogram": {str(k): v for k, v in sorted(near_counts.items())},
            "pairs_crossing_existing_split": int(near_split_cross),
            "pairs_with_different_source_label": int(near_label_cross),
            "pairs_in_different_visual_groups": int(near_different_group),
            "first_100_candidates_for_manual_review": near_pairs,
        },
        "dimensions": dict(dimensions),
        "exif_camera_make_model": {" | ".join(k): v for k, v in exif_cameras.items()},
        "luminance_median_by_source_label": {key: summarize(value) for key, value in exposure.items()},
        "edge_variance_median_by_source_label": {key: float(np.median(value)) for key, value in blur.items()},
        "unreadable_or_missing": unreadable,
        "unverifiable_from_manifest": [
            "physical banana/hand identity", "camera when EXIF is absent", "session and lot identity",
            "controlled light condition", "whether multiple fruits are present", "visual correctness of every label",
        ],
        "manual_review_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_v1/manifest.csv"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.manifest, args.root)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        if args.output.exists(): raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote read-only audit report to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
