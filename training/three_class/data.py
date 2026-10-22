"""Build a leakage-aware three-class view from the legacy image manifest.

Only source labels unripe, ripe and overripe are eligible. Rotten images are
excluded, never relabeled. Entire visual-family groups with conflicting source
labels are excluded. Existing validation/test rows are deliberately not used:
the experiment creates fresh grouped validation/test partitions from legacy
train rows, whose physical identity guarantees remain unknown.
"""
from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import tensorflow as tf
from training.three_class.contract import (
    ALLOWED_SUFFIXES,
    AMBIGUOUS_CROSS_LABEL_LINKS,
    CLASSES,
    SAME_LABEL_VISUAL_LINKS,
    SOURCE_TO_CLASS,
)


def read_rows(manifest: Path) -> list[dict[str, str]]:
    with Path(manifest).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "split", "original_state", "group_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest must contain {sorted(required)}")
    return rows


def resolve_image_path(value: str, root: Path) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    # Legacy manifests embed the original workstation path. Rebase from the
    # first dataset/ segment so the manifest remains usable after relocation.
    parts = candidate.parts
    for index, part in enumerate(parts):
        if part.lower() == "dataset":
            rebased = Path(root).resolve().joinpath(*parts[index:])
            if rebased.is_file():
                return rebased
    raise FileNotFoundError(f"image_path does not resolve on this workspace: {value}")


def prepare_rows(manifest: Path, root: Path, seed: int = 2026) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows = read_rows(manifest)
    raw_counts = Counter(row["original_state"] for row in rows)
    training_rows = [row for row in rows if row["split"] == "train"]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in training_rows:
        groups[row["group_id"]].append(row)

    conflicting_groups = {
        group_id for group_id, members in groups.items()
        if len({member["original_state"] for member in members}) != 1
    }
    manually_ambiguous_groups = {group for pair in AMBIGUOUS_CROSS_LABEL_LINKS for group in pair} & set(groups)
    # Exclude the full connected visual family if an ambiguous cross-label pair
    # is linked to additional same-label groups.
    changed = True
    while changed:
        changed = False
        for first, second in SAME_LABEL_VISUAL_LINKS:
            if first in manually_ambiguous_groups and second in groups and second not in manually_ambiguous_groups:
                manually_ambiguous_groups.add(second)
                changed = True
            if second in manually_ambiguous_groups and first in groups and first not in manually_ambiguous_groups:
                manually_ambiguous_groups.add(first)
                changed = True
    excluded_groups = conflicting_groups | manually_ambiguous_groups
    eligible = [
        row for group_id, members in groups.items() if group_id not in excluded_groups
        for row in members if row["original_state"] in SOURCE_TO_CLASS
    ]

    # Exact duplicate bytes are retained once. If identical bytes have
    # conflicting labels, exclude the entire hash group.
    hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        path = resolve_image_path(row["image_path"], root)
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        prepared = {**row, "_path": str(path), "_sha256": digest,
                    "class_name": SOURCE_TO_CLASS[row["original_state"]]}
        hashes[digest].append(prepared)

    kept: list[dict[str, Any]] = []
    duplicate_hashes = 0
    conflicting_hashes = 0
    for same_hash in hashes.values():
        labels = {row["class_name"] for row in same_hash}
        if len(labels) > 1:
            conflicting_hashes += 1
            continue
        if len(same_hash) > 1:
            duplicate_hashes += len(same_hash) - 1
        kept.append(same_hash[0])

    by_class: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASSES}
    for row in kept:
        by_class[row["class_name"]].append(row)

    # Split by visual-family group, not by image. Existing legacy validation
    # and test sets are untouched and are not used by this candidate run.
    partitions: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for class_name, class_rows in by_class.items():
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in class_rows:
            by_group[row["group_id"]].append(row)
        parent = {group_id: group_id for group_id in by_group}

        def find(group_id: str) -> str:
            while parent[group_id] != group_id:
                parent[group_id] = parent[parent[group_id]]
                group_id = parent[group_id]
            return group_id

        for first, second in SAME_LABEL_VISUAL_LINKS:
            if first in parent and second in parent:
                parent[find(second)] = find(first)
        visual_families: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for group_id, members in by_group.items():
            visual_families[find(group_id)].extend(members)
        family_ids = sorted(visual_families)
        rng.shuffle(family_ids)
        n = len(family_ids)
        if n < 6:
            raise ValueError(f"not enough independent visual families for {class_name}: {n}")
        n_test = max(1, round(n * 0.15))
        n_validation = max(1, round(n * 0.15))
        test_families = set(family_ids[:n_test])
        validation_families = set(family_ids[n_test:n_test + n_validation])
        for family_id, members in visual_families.items():
            split = "test" if family_id in test_families else "validation" if family_id in validation_families else "train"
            partitions[split].extend(members)

    for split_rows in partitions.values():
        split_rows.sort(key=lambda row: (row["class_name"], row["group_id"], row["_path"]))
    group_splits: dict[str, set[str]] = defaultdict(set)
    for split, split_rows in partitions.items():
        for row in split_rows:
            group_splits[row["group_id"]].add(split)
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise AssertionError("visual groups crossed the newly generated split")
    linked_splits = []
    for first, second in SAME_LABEL_VISUAL_LINKS:
        if first in group_splits and second in group_splits:
            linked_splits.append((group_splits[first], group_splits[second]))
    if any(first != second for first, second in linked_splits):
        raise AssertionError("manually linked near-duplicate families crossed the new split")

    summary = {
        "source_manifest": str(Path(manifest)),
        "seed": seed,
        "raw_source_label_counts": dict(raw_counts),
        "legacy_train_rows": len(training_rows),
        "excluded_rotten_rows": sum(row["original_state"] == "rotten" for row in training_rows),
        "excluded_conflicting_visual_groups": sorted(conflicting_groups),
        "excluded_manually_reviewed_ambiguous_groups": sorted(manually_ambiguous_groups),
        "manually_reviewed_ambiguous_links": sorted([list(pair) for pair in AMBIGUOUS_CROSS_LABEL_LINKS]),
        "manually_reviewed_same_label_visual_links": sorted([list(pair) for pair in SAME_LABEL_VISUAL_LINKS]),
        "excluded_conflicting_group_rows": sum(len(groups[group]) for group in conflicting_groups),
        "excluded_manually_ambiguous_group_rows": sum(len(groups[group]) for group in manually_ambiguous_groups),
        "excluded_exact_duplicate_rows": duplicate_hashes,
        "excluded_conflicting_hash_groups": conflicting_hashes,
        "eligible_rows_after_group_and_hash_filter": len(kept),
        "class_counts_by_split": {
            split: dict(Counter(row["class_name"] for row in split_rows))
            for split, split_rows in partitions.items()
        },
        "group_counts_by_split": {
            split: len({row["group_id"] for row in split_rows})
            for split, split_rows in partitions.items()
        },
        "identity_caveat": "group_id is a visual-family cluster, not a verified fruit/hand/session identity",
    }
    return partitions, summary


def make_dataset(rows: list[dict[str, Any]], image_size: tuple[int, int], batch_size: int,
                 training: bool, seed: int, augmentation: tf.keras.layers.Layer | None = None) -> tf.data.Dataset:
    paths = [row["_path"] for row in rows]
    labels = [CLASSES.index(row["class_name"]) for row in rows]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        encoded = tf.io.read_file(path)
        image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size, method="bilinear", antialias=False)
        image = tf.cast(image, tf.float32)
        if training and augmentation is not None:
            image = augmentation(image[None, ...], training=True)[0]
        return image, tf.cast(label, tf.int32)

    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.map(load, num_parallel_calls=1, deterministic=True).batch(batch_size).with_options(options).prefetch(tf.data.AUTOTUNE)
