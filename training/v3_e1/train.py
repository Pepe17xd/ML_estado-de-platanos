"""Train an isolated V3-E1 run. This script is intentionally never run by setup."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tensorflow as tf

from training.v3_e1.augmentations import PHOTO_FAMILIES, LightingAugmentation, ablation_config
from training.v3_e1.data import as_model_inputs, class_counts, class_weights, directory_dataset, read_manifest
from training.v3_e1.model import CLASSES, load_v2_checkpoint, load_warm_start_weights, set_finetune_trainable, set_head_trainable
from training.v3_e1.selection import ValidationCheckpointSelector, validate_selection_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/v3_e1.json"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("classes") != CLASSES or config.get("input_size") != [224, 224]:
        raise ValueError("V3-E1 config must preserve the V2 four classes and 224x224 RGB input")
    validate_selection_config(config.get("selection", {}))
    return config


def safe_new_run_dir(path: Path) -> Path:
    allowed = (ROOT / "artifacts" / "v3_e1").resolve()
    resolved = Path(path).resolve()
    if os.path.commonpath([str(allowed), str(resolved)]) != str(allowed) or resolved == allowed:
        raise ValueError(f"run directory must be a new child of {allowed}")
    if resolved.exists():
        raise FileExistsError(f"refusing to reuse existing run directory: {resolved}")
    return resolved


def make_callbacks(
    run_dir: Path,
    phase: str,
    patience: int,
    validation_data: tf.data.Dataset,
    selection_config: dict[str, Any],
) -> tuple[list[tf.keras.callbacks.Callback], ValidationCheckpointSelector]:
    selector = ValidationCheckpointSelector(validation_data, run_dir, phase, selection_config)
    primary_metric = selection_config.get("primary_metric", "macro_f1")
    callbacks: list[tf.keras.callbacks.Callback] = [
        # This callback runs first and adds validation metrics to the shared logs.
        selector,
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "last.keras"),
            save_best_only=False,
            save_freq="epoch",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=f"val_{primary_metric}", mode="max", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7
        ),
    ]
    return callbacks, selector


def compile_for_phase(model: tf.keras.Model, learning_rate: float, weight_decay: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def history_json(history: tf.keras.callbacks.History) -> dict[str, Any]:
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ablation", choices=["none", "combined", *PHOTO_FAMILIES], default="combined")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs-head", type=int)
    parser.add_argument("--epochs-finetune", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume-from", type=Path, help="Warm-start from a compatible .keras checkpoint; optimizer state resets.")
    parser.add_argument("--resume-phase", choices=["head", "finetune"], default="head")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    config = load_config(args.config)
    training = dict(config["training"])
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.epochs_head is not None:
        training["epochs_head"] = args.epochs_head
    if args.epochs_finetune is not None:
        training["epochs_finetune"] = args.epochs_finetune
    seed = args.seed if args.seed is not None else int(config["seed"])
    if training["epochs_head"] < 0 or training["epochs_finetune"] < 0:
        parser.error("epoch counts must be non-negative")
    if (args.resume_phase == "head" and training["epochs_head"] == 0 and training["epochs_finetune"] == 0) or (
        args.resume_phase == "finetune" and training["epochs_finetune"] == 0
    ):
        parser.error("at least one epoch must be configured for the requested phase")
    if args.resume_phase == "finetune" and args.resume_from is None:
        parser.error("--resume-phase finetune requires --resume-from")

    tf.keras.utils.set_random_seed(seed)
    augmentation = ablation_config(config["augmentation"], args.ablation)
    augmentation["seed"] = seed
    # Geometry and V2 contrast stay in the saved V2 graph. This external layer
    # contributes only the selected E1 photometric family during input loading.
    augmentation["geometric"] = {"horizontal_flip": False, "rotation_factor": 0.0, "zoom_factor": 0.0}
    photometric_layer = LightingAugmentation(augmentation, seed=seed, name="v3_e1_photometric")
    checkpoint = ROOT / config["v2_checkpoint"]
    metadata_path = ROOT / config["v2_metadata"]
    model, checkpoint_info = load_v2_checkpoint(checkpoint, metadata_path)
    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        if not resume_path.is_file() or resume_path.suffix != ".keras":
            raise ValueError("--resume-from must be an existing .keras checkpoint")
        checkpoint_info["warm_start"] = load_warm_start_weights(model, resume_path)
    run_dir = safe_new_run_dir(args.run_dir)

    manifest_path = ROOT / config["manifest"]
    rows = read_manifest(manifest_path)
    counts_train = class_counts(rows, "train")
    class_weight_map = class_weights(rows)
    batch_size = int(training["batch_size"])
    size = tuple(config["input_size"])
    train_ds = as_model_inputs(directory_dataset(
        ROOT / config["dataset_dir"] / "train", size, batch_size, True, seed, photometric_layer
    ))
    validation_ds = as_model_inputs(directory_dataset(ROOT / config["dataset_dir"] / "validation", size, batch_size, False, seed))

    run_dir.mkdir(parents=True, exist_ok=False)
    run_record: dict[str, Any] = {
        "experiment": config["experiment"],
        "model_version": "3.0.0-lighting-e1",
        "classes": CLASSES,
        "input_shape": [224, 224, 3],
        "preprocessing": "saved_v2_embedded_mobilenet_v3_rescaling; input RGB float32 [0,255]",
        "resize_backend": "tensorflow_bilinear",
        "resize_method": "tf.io.decode_image RGB; tf.image.resize bilinear, antialias=false",
        "ablation": args.ablation,
        "augmentation": augmentation,
        "training": training,
        "selection_criteria": config["selection"],
        "seed": seed,
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "bundled with TensorFlow"),
        "v2_checkpoint": checkpoint_info,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "train_counts": counts_train,
        "class_weights": {CLASSES[index]: float(value) for index, value in class_weight_map.items()},
        "resume_phase": args.resume_phase if args.resume_from else None,
        "resume_note": "Weights-only warm-start; optimizer state and epoch schedule are reset." if args.resume_from else None,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "training_started",
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")

    phase_histories: dict[str, Any] = {}
    phase_validation_metrics: dict[str, Any] = {}
    best_candidates: list[tuple[tuple[float, ...], Path, str, dict[str, Any]]] = []
    if args.resume_phase == "head":
        set_head_trainable(model)
        compile_for_phase(model, float(training["learning_rate_head"]), float(training["weight_decay"]))
        if int(training["epochs_head"]) > 0:
            callbacks, selector = make_callbacks(
                run_dir, "head", int(training["early_stopping_patience"]), validation_ds, config["selection"]
            )
            history = model.fit(
                train_ds,
                validation_data=validation_ds,
                epochs=int(training["epochs_head"]),
                class_weight=class_weight_map,
                callbacks=callbacks,
                verbose=2,
            )
            phase_histories["head"] = history_json(history)
            phase_validation_metrics["head"] = selector.epoch_metrics
            if selector.best_checkpoint is not None and selector.best_score is not None and selector.best_metrics is not None:
                best_candidates.append((selector.best_score, selector.best_checkpoint, "head", selector.best_metrics))
            run_record["phase_histories"] = phase_histories
            run_record["validation_metrics_by_epoch"] = phase_validation_metrics
            (run_dir / "run_metadata.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")

    if int(training["epochs_finetune"]) > 0:
        set_finetune_trainable(model, int(training["fine_tune_last_layers"]))
        compile_for_phase(model, float(training["learning_rate_finetune"]), float(training["weight_decay"]))
        callbacks, selector = make_callbacks(
            run_dir, "finetune", int(training["early_stopping_patience"]), validation_ds, config["selection"]
        )
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=int(training["epochs_finetune"]),
            class_weight=class_weight_map,
            callbacks=callbacks,
            verbose=2,
        )
        phase_histories["finetune"] = history_json(history)
        phase_validation_metrics["finetune"] = selector.epoch_metrics
        if selector.best_checkpoint is not None and selector.best_score is not None and selector.best_metrics is not None:
            best_candidates.append((selector.best_score, selector.best_checkpoint, "finetune", selector.best_metrics))
        run_record["phase_histories"] = phase_histories
        run_record["validation_metrics_by_epoch"] = phase_validation_metrics
        (run_dir / "run_metadata.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    if not best_candidates:
        run_record.update({
            "status": "no_checkpoint_met_selection_criteria",
            "selection_warning": "No validation epoch met the configured rotten-recall guardrail; no best.keras was selected.",
            "phase_histories": phase_histories,
            "validation_metrics_by_epoch": phase_validation_metrics,
        })
        (run_dir / "run_metadata.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        raise RuntimeError("no validation checkpoint met the configured selection criteria; inspect run_metadata.json")

    best_score, best_checkpoint, best_phase, best_metrics = max(best_candidates, key=lambda item: item[0])
    shutil.copy2(best_checkpoint, run_dir / "best.keras")
    run_record.update({
        "status": "completed",
        "selection_criteria": config["selection"],
        "selected_validation_score": list(best_score),
        "selected_validation_metrics": best_metrics,
        "selected_phase_by_validation_selection": best_phase,
        "selected_checkpoint": "best.keras",
        "phase_histories": phase_histories,
        "validation_metrics_by_epoch": phase_validation_metrics,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    (run_dir / "run_metadata.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(json.dumps({
        "run_dir": str(run_dir), "best_checkpoint": str(run_dir / "best.keras"),
        "selected_validation_score": list(best_score), "selected_phase": best_phase, "status": "completed",
    }, indent=2))


if __name__ == "__main__":
    main()
