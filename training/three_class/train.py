"""Train an isolated, group-split three-class candidate from ImageNet weights."""
from __future__ import annotations

import argparse
import copy
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

from training.three_class.data import CLASSES, make_dataset, prepare_rows
from training.three_class.model import build_model, compile_model, set_finetune_trainable
from training.v3_e1.augmentations import LightingAugmentation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data/dataset_v1/manifest.csv"
DEFAULT_CONFIG = ROOT / "configs/three_class.json"
ALLOWED_RUN_ROOT = (ROOT / "artifacts/three_class").resolve()


def safe_run_dir(path: Path) -> Path:
    result = Path(path).resolve()
    if os.path.commonpath([str(ALLOWED_RUN_ROOT), str(result)]) != str(ALLOWED_RUN_ROOT) or result == ALLOWED_RUN_ROOT:
        raise ValueError(f"run directory must be a new child of {ALLOWED_RUN_ROOT}")
    if result.exists():
        raise FileExistsError(f"refusing to overwrite run directory {result}")
    return result


class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, validation: tf.data.Dataset, path: Path):
        super().__init__()
        self.validation = validation
        self.path = path
        self.best = -1.0
        self.history: list[dict[str, float]] = []

    def on_epoch_end(self, epoch: int, logs=None):
        truths, predictions = [], []
        for images, labels in self.validation:
            truths.extend(labels.numpy().tolist())
            predictions.extend(np.argmax(self.model(images, training=False).numpy(), axis=1).tolist())
        per_class = []
        for index in range(len(CLASSES)):
            tp = sum(y == index and p == index for y, p in zip(truths, predictions))
            fp = sum(y != index and p == index for y, p in zip(truths, predictions))
            fn = sum(y == index and p != index for y, p in zip(truths, predictions))
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            per_class.append(2 * precision * recall / max(1e-12, precision + recall))
        score = float(np.mean(per_class))
        record = {"epoch": float(epoch + 1), "val_macro_f1": score}
        if logs:
            record.update({k: float(v) for k, v in logs.items() if k.startswith("val_")})
        self.history.append(record)
        if score > self.best:
            self.best = score
            self.model.save(self.path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs-head", type=int)
    parser.add_argument("--epochs-finetune", type=int)
    parser.add_argument("--imagenet-weights", choices=["imagenet", "none"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("classes") != CLASSES or config.get("input_size") != [224, 224]:
        parser.error("config must preserve the exact three-class order and 224x224 input contract")
    seed = args.seed if args.seed is not None else int(config["seed"])
    batch_size = args.batch_size if args.batch_size is not None else int(config["batch_size"])
    epochs_head = args.epochs_head if args.epochs_head is not None else int(config["epochs_head"])
    epochs_finetune = args.epochs_finetune if args.epochs_finetune is not None else int(config["epochs_finetune"])
    imagenet_weights = args.imagenet_weights or config["imagenet_initialization"]
    if batch_size < 1 or epochs_head < 1 or epochs_finetune < 0:
        parser.error("batch size must be positive, head epochs >=1 and fine-tuning epochs >=0")

    run_dir = safe_run_dir(args.run_dir)
    partitions, audit = prepare_rows(args.manifest, ROOT, seed=seed)
    # Create only this fresh run directory after input validation has completed.
    run_dir.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    train_rows, val_rows = partitions["train"], partitions["validation"]
    aug_config = copy.deepcopy(config["augmentation"])
    aug_config["seed"] = seed
    augmentation = LightingAugmentation(aug_config, seed=seed, name="three_class_lighting_augmentation")
    train_ds = make_dataset(train_rows, (224, 224), batch_size, True, seed, augmentation)
    val_ds = make_dataset(val_rows, (224, 224), batch_size, False, seed)
    class_counts = Counter(row["class_name"] for row in train_rows)
    largest = max(class_counts.values())
    class_weights = {CLASSES.index(name): largest / class_counts[name] for name in CLASSES}

    model = build_model(weights=None if imagenet_weights == "none" else "imagenet")
    model_info = {
        "architecture": "MobileNetV3Small",
        "initialization": imagenet_weights,
        "classes": CLASSES,
        "input_shape": [224, 224, 3],
        "preprocessing": "keras MobileNetV3 include_preprocessing; RGB float [0,255]",
    }
    selected_path = run_dir / "best.keras"
    selector = MacroF1Checkpoint(val_ds, selected_path)
    callbacks = [
        selector,
        tf.keras.callbacks.EarlyStopping(
            monitor=str(config["early_stopping_monitor"]),
            patience=int(config["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(config["lr_scheduler"]["factor"]),
            patience=int(config["lr_scheduler"]["patience"]),
            min_lr=float(config["lr_scheduler"]["min_lr"]),
        ),
    ]
    metadata = {
        "experiment": "banana_three_class_grouped_candidate",
        "status": "training_started",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "bundled"),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "architecture": model_info,
        "classes": CLASSES,
        "input_size": [224, 224],
        "resize_backend": "tensorflow_bilinear",
        "augmentation": aug_config,
        "class_weights": {CLASSES[index]: float(value) for index, value in class_weights.items()},
        "class_counts_train": dict(class_counts),
        "split_audit": audit,
        "selection": "validation macro F1; early stopping on val_loss; grouped visual-family split",
        "test_policy": "fresh group holdout from legacy train only; test untouched until final evaluation",
        "epochs": {"head": epochs_head, "finetune": epochs_finetune},
        "config": config,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "selected_checkpoint": "best.keras",
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def phase(phase_name: str, epochs: int, learning_rate: float):
        nonlocal model
        if epochs == 0:
            return []
        compile_model(model, learning_rate, weight_decay=float(config["weight_decay"]))
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=2,
        )
        return [{"phase": phase_name, "history": {k: [float(x) for x in v] for k, v in history.history.items()}}]

    histories = phase("head", epochs_head, float(config["learning_rate_head"]))
    if epochs_finetune:
        set_finetune_trainable(model, last_layers=int(config.get("fine_tune_last_layers", 30)))
        histories.extend(phase("finetune", epochs_finetune, float(config["learning_rate_finetune"])))
    if not selected_path.exists():
        raise RuntimeError("no validation checkpoint was selected")
    metadata.update({
        "status": "candidate_trained",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "phase_histories": histories,
        "validation_macro_f1_by_epoch": selector.history,
        "best_validation_macro_f1": selector.best,
        "model_file": selected_path.name,
    })
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "status": metadata["status"],
                      "best_validation_macro_f1": selector.best,
                      "validation_epochs": selector.history}, indent=2))


if __name__ == "__main__":
    main()
