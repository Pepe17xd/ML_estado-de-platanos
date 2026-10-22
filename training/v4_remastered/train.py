"""Train a V4 three-class candidate from V2 weights on the V2 training split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from PIL import Image

from training.three_class.contract import CLASSES
from training.three_class.model import build_model, compile_model, set_finetune_trainable
from training.v3_e1.augmentations import LightingAugmentation
from training.v4_remastered.transfer import initialize_from_v2

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/v4_remastered.json"
RUN_ROOT = (ROOT / "artifacts/v4_remastered").resolve()


def safe_run_dir(path: Path) -> Path:
    result = path.resolve()
    if os.path.commonpath([str(RUN_ROOT), str(result)]) != str(RUN_ROOT) or result == RUN_ROOT:
        raise ValueError(f"run-dir must be a new child of {RUN_ROOT}")
    if result.exists():
        raise FileExistsError(f"refusing to overwrite {result}")
    return result


def rows_for_split(manifest: Path, image_root: Path, split: str) -> list[dict]:
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"split", "class_name", "image_path", "sha256"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"V2 manifest must contain {sorted(required)}")
    selected = []
    for row in rows:
        if row["split"] != split or row["class_name"] == "podrido":
            continue
        if row["class_name"] not in CLASSES:
            raise ValueError(f"unexpected source class {row['class_name']!r}")
        path = image_root / row["image_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.append({**row, "path": str(path.resolve()), "label": CLASSES.index(row["class_name"])})
    counts = Counter(CLASSES[row["label"]] for row in selected)
    if len(selected) == 0 or any(counts[name] == 0 for name in CLASSES):
        raise ValueError(f"split {split!r} lacks one or more target classes: {dict(counts)}")
    return selected


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.uint8)
    bits = pixels[:, :-1] > pixels[:, 1:]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def filter_near_duplicate_validation(manifest: Path, image_root: Path,
                                     validation_rows: list[dict], max_distance: int = 4) -> tuple[list[dict], dict]:
    """Conservatively drop target validation images resembling any V2 train image.

    dHash is only a candidate detector, not proof of identity. Matching rows are
    excluded from selection/holdout (never relabeled) to reduce teacher leakage.
    """
    with Path(manifest).open(encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    train_paths = [image_root / row["image_path"] for row in all_rows if row["split"] == "train"]
    train_hashes = np.fromiter((dhash(path) for path in train_paths), dtype=np.uint64, count=len(train_paths))
    lookup = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    retained, excluded, distances = [], [], []
    for row in validation_rows:
        value = np.uint64(dhash(Path(row["path"])))
        differences = np.bitwise_xor(train_hashes, value).view(np.uint8).reshape(-1, 8)
        minimum = int(lookup[differences].sum(axis=1).min())
        distances.append(minimum)
        (excluded if minimum <= max_distance else retained).append(row)
    counts_before = Counter(CLASSES[row["label"]] for row in validation_rows)
    counts_after = Counter(CLASSES[row["label"]] for row in retained)
    if any(counts_after[name] < 25 for name in CLASSES):
        raise ValueError(f"too few validation images remain after near-duplicate exclusion: {dict(counts_after)}")
    return retained, {
        "method": "64-bit dHash, exclude validation candidates with minimum distance <=4 against all V2 train images",
        "threshold": max_distance,
        "train_images_compared_including_rotten": len(train_hashes),
        "validation_target_images_before": len(validation_rows),
        "validation_target_images_excluded_as_near_duplicate_candidates": len(excluded),
        "validation_target_images_retained": len(retained),
        "excluded_by_class": {name: counts_before[name] - counts_after[name] for name in CLASSES},
        "retained_by_class": {name: counts_after[name] for name in CLASSES},
        "minimum_distance_histogram": {str(distance): distances.count(distance) for distance in sorted(set(distances))},
        "caveat": "dHash matches are not identity proof; false positives may have been excluded. No physical group IDs are available.",
    }


def split_validation(rows: list[dict], seed: int, holdout_fraction: float = 0.25) -> tuple[list[dict], list[dict]]:
    """Make a deterministic stratified dev/holdout split; never touches V2 test."""
    rng = random.Random(seed)
    development, heldout = [], []
    for class_index in range(len(CLASSES)):
        group = [row for row in rows if row["label"] == class_index]
        rng.shuffle(group)
        count = max(1, round(len(group) * holdout_fraction))
        heldout.extend(group[:count])
        development.extend(group[count:])
    return development, heldout


def isolate_validation_holdout(rows: list[dict], development: list[dict], heldout: list[dict],
                               max_distance: int = 4) -> tuple[list[dict], dict]:
    """Keep one representative per dHash family wholly outside V4 selection.

    Connected components touching development are removed from the heldout;
    cross-label components are excluded as ambiguous. This cannot prove fruit
    identity, but prevents known near-image families crossing this split.
    """
    all_rows = development + heldout
    values = [dhash(Path(row["path"])) for row in all_rows]
    parent = list(range(len(all_rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        left, right = find(first), find(second)
        if left != right:
            parent[right] = left

    for first in range(len(all_rows)):
        for second in range(first + 1, len(all_rows)):
            if (values[first] ^ values[second]).bit_count() <= max_distance:
                union(first, second)

    dev_paths = {row["path"] for row in development}
    components: dict[int, list[dict]] = {}
    for index, row in enumerate(all_rows):
        components.setdefault(find(index), []).append(row)
    kept, cross_dev, ambiguous, duplicate_members = [], 0, 0, 0
    for members in components.values():
        member_paths = {row["path"] for row in members}
        if not member_paths.intersection({row["path"] for row in heldout}):
            continue
        if member_paths & dev_paths:
            cross_dev += len(member_paths - dev_paths)
            continue
        if len({row["label"] for row in members}) > 1:
            ambiguous += len(members)
            continue
        chosen = min(members, key=lambda row: (row.get("sha256", ""), row["path"]))
        kept.append(chosen)
        duplicate_members += len(members) - 1
    kept.sort(key=lambda row: (row["label"], row.get("sha256", "")))
    counts = Counter(CLASSES[row["label"]] for row in kept)
    return kept, {
        "method": f"connected components of validation dHash <= {max_distance}; keep one image per component",
        "development_images": len(development),
        "raw_holdout_images": len(heldout),
        "holdout_images_removed_for_link_to_development": cross_dev,
        "holdout_images_removed_for_cross_label_ambiguity": ambiguous,
        "same_label_duplicate_members_collapsed": duplicate_members,
        "final_holdout_independent_dhash_groups": len(kept),
        "final_holdout_by_class": {name: counts[name] for name in CLASSES},
        "caveat": "dHash family separation is approximate; no verified physical fruit/session IDs exist.",
    }


def dataset(rows: list[dict], batch: int, training: bool, seed: int, augmentation=None) -> tf.data.Dataset:
    paths = [r["path"] for r in rows]
    labels = [r["label"] for r in rows]
    data = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        data = data.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [224, 224], method="bilinear", antialias=False)
        image = tf.cast(image, tf.float32)
        if training and augmentation is not None:
            image = augmentation(image[None, ...], training=True)[0]
        return image, tf.cast(label, tf.int32)

    options = tf.data.Options()
    options.experimental_deterministic = True
    return data.map(load, num_parallel_calls=1, deterministic=True).batch(batch).with_options(options).prefetch(tf.data.AUTOTUNE)


class ValidationMacroF1(tf.keras.callbacks.Callback):
    def __init__(self, validation, path: Path):
        super().__init__()
        self.validation, self.path = validation, path
        self.best = -1.0
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        truth, pred = [], []
        for images, labels in self.validation:
            truth.extend(labels.numpy().tolist())
            pred.extend(np.argmax(self.model(images, training=False).numpy(), axis=1).tolist())
        f1s = []
        for cls in range(3):
            tp = sum(y == cls and p == cls for y, p in zip(truth, pred))
            fp = sum(y != cls and p == cls for y, p in zip(truth, pred))
            fn = sum(y == cls and p != cls for y, p in zip(truth, pred))
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1s.append(2 * precision * recall / max(1e-12, precision + recall))
        score = float(np.mean(f1s))
        if logs is not None:
            logs["val_macro_f1"] = score
        self.records.append({"epoch": epoch + 1, "val_macro_f1": score, **{
            key: float(value) for key, value in (logs or {}).items() if key.startswith("val_") and key != "val_macro_f1"
        }})
        if score > self.best:
            self.best = score
            self.model.save(self.path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/dataset_v2/manifest.csv")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--epochs-head", type=int)
    parser.add_argument("--epochs-finetune", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("classes") != CLASSES or config.get("input_size") != [224, 224]:
        parser.error("V4 config class order/input contract mismatch")
    seed = int(config["seed"])
    batch = args.batch_size or int(config["batch_size"])
    head_epochs = args.epochs_head if args.epochs_head is not None else int(config["epochs_head"])
    ft_epochs = args.epochs_finetune if args.epochs_finetune is not None else int(config["epochs_finetune"])
    if batch < 1 or head_epochs < 1 or ft_epochs < 0:
        parser.error("invalid batch or epoch count")

    run = safe_run_dir(args.run_dir)
    train_rows = rows_for_split(args.manifest, ROOT, "train")
    all_val_rows = rows_for_split(args.manifest, ROOT, "validation")
    filtered_val_rows, leakage_audit = filter_near_duplicate_validation(args.manifest, ROOT, all_val_rows)
    val_rows, heldout_rows = split_validation(filtered_val_rows, seed)
    # This pre-existing validation split has historical uses and is dev-only,
    # never reported as an independent final test.
    seed_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    run.mkdir(parents=True, exist_ok=False)
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    tf.config.threading.set_intra_op_parallelism_threads(int(config.get("intra_op_threads", 8)))
    tf.config.threading.set_inter_op_parallelism_threads(int(config.get("inter_op_threads", 2)))

    model = build_model(weights=None)
    transfer = initialize_from_v2(
        model, ROOT / config["v2_checkpoint"], ROOT / config["v2_metadata"]
    )
    augmentation_config = dict(config["augmentation"])
    augmentation_config["seed"] = seed
    augmentation = LightingAugmentation(augmentation_config, seed=seed, name="v4_lighting_augmentation")
    train_ds = dataset(train_rows, batch, True, seed, augmentation)
    val_ds = dataset(val_rows, batch, False, seed)
    counts = Counter(CLASSES[row["label"]] for row in train_rows)
    weights = {CLASSES.index(name): max(counts.values()) / counts[name] for name in CLASSES}
    metadata = {
        "status": "training_started",
        "model_version": "4.0.0-candidate",
        "seed": seed,
        "tensorflow_version": tf.__version__,
        "classes": CLASSES,
        "input_size": [224, 224],
        "resize_backend": "tensorflow_bilinear",
        "preprocessing": "embedded_mobilenet_v3; RGB float32 [0,255]",
        "source_manifest_sha256": seed_hash,
        "split_policy": "V2 train for fitting; 75% of historical V2 validation for selection; 25% reserved from V4 selection; V2 test unused",
        "train_rows": len(train_rows), "validation_dev_rows": len(val_rows),
        "reserved_holdout_rows": len(heldout_rows),
        "class_counts_train": dict(counts),
        "class_counts_validation_dev": dict(Counter(CLASSES[r["label"]] for r in val_rows)),
        "class_counts_reserved_holdout": dict(Counter(CLASSES[r["label"]] for r in heldout_rows)),
        "validation_near_duplicate_audit": leakage_audit,
        "class_weights": {CLASSES[index]: value for index, value in weights.items()},
        "augmentation": augmentation_config,
        "transfer": transfer,
        "epochs": {"head": head_epochs, "finetune": ft_epochs},
        "config": config,
        "selection": "validation macro F1, dev-only, no independent test",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    best_path = run / "best.keras"
    selector = ValidationMacroF1(val_ds, best_path)
    callbacks = [
        selector,
        tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1, min_lr=1e-7),
    ]
    histories = []

    def fit_phase(name, epochs, lr):
        if epochs == 0:
            return
        compile_model(model, lr, float(config["weight_decay"]))
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=weights,
                            callbacks=callbacks, verbose=2)
        histories.append({"phase": name, "history": {k: [float(v) for v in values] for k, values in history.history.items()}})

    fit_phase("head", head_epochs, float(config["learning_rate_head"]))
    if ft_epochs:
        set_finetune_trainable(model, last_layers=int(config["fine_tune_last_layers"]))
        fit_phase("finetune", ft_epochs, float(config["learning_rate_finetune"]))
    if not best_path.is_file():
        raise RuntimeError("no V4 checkpoint was selected")
    metadata.update({"status": "candidate_trained", "model_file": best_path.name,
                     "best_validation_macro_f1": selector.best, "validation_macro_f1_by_epoch": selector.records,
                     "phase_histories": histories, "completed_utc": datetime.now(timezone.utc).isoformat()})
    (run / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run), "status": metadata["status"],
                      "best_validation_macro_f1": selector.best, "epochs": selector.records}, indent=2))


if __name__ == "__main__":
    main()
