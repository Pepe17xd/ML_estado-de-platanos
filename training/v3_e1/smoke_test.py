"""Run a tiny, isolated two-phase V3-E1 training smoke test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from training.v3_e1.augmentations import LightingAugmentation, ablation_config
from training.v3_e1.data import read_manifest
from training.v3_e1.model import CLASSES, load_v2_checkpoint, set_finetune_trainable, set_head_trainable
from training.v3_e1.train import DEFAULT_CONFIG, ROOT, compile_for_phase, load_config, make_callbacks, safe_new_run_dir


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def production_hashes() -> dict[str, str]:
    paths = [ROOT / "models/production_model.keras", ROOT / "models/production_model_v2.keras", ROOT / "models/metadata.json"]
    h5 = ROOT / "models/production_model.h5"
    if h5.is_file():
        paths.append(h5)
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def small_manifest_dataset(
    rows: list[dict[str, str]],
    split: str,
    samples_per_class: int,
    batch_size: int,
    training: bool,
    augmentation: tf.keras.layers.Layer | None = None,
) -> tuple[tf.data.Dataset, int]:
    selected: list[dict[str, str]] = []
    for class_name in CLASSES:
        candidates = sorted(
            (row for row in rows if row["split"] == split and row["class_name"] == class_name),
            key=lambda row: row["image_path"],
        )
        if len(candidates) < samples_per_class:
            raise ValueError(f"split {split!r} has fewer than {samples_per_class} records for {class_name}")
        selected.extend(candidates[:samples_per_class])
    paths = [str((ROOT / row["image_path"]).resolve()) for row in selected]
    labels = [CLASSES.index(row["class_name"]) for row in selected]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(tf.cast(image, tf.float32), [224, 224], method="bilinear")
        return image, tf.cast(label, tf.int32)

    dataset = dataset.map(decode, num_parallel_calls=1, deterministic=True).batch(batch_size)
    if training and augmentation is not None:
        dataset = dataset.map(
            lambda images, class_ids: (augmentation(images, training=True), class_ids),
            num_parallel_calls=1,
            deterministic=True,
        )
        dataset = dataset.repeat()
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options).prefetch(1), len(selected)


def changed_trainable_weights(before: list[np.ndarray], model: tf.keras.Model) -> bool:
    after = model.trainable_weights
    if len(before) != len(after):
        raise RuntimeError("trainable weight layout changed inside a phase")
    return any(not np.array_equal(old, new.numpy()) for old, new in zip(before, after))


def history_values(history: tf.keras.callbacks.History) -> dict[str, list[float]]:
    return {name: [float(value) for value in values] for name, values in history.history.items()}


def run_smoke(
    config_path: Path,
    run_dir: Path,
    batch_size: int,
    samples_per_class: int,
    steps_per_phase: int,
    seed: int,
) -> dict[str, Any]:
    config = load_config(config_path)
    if min(batch_size, samples_per_class, steps_per_phase) < 1:
        raise ValueError("batch size, samples per class, and steps per phase must be positive")
    tf.keras.utils.set_random_seed(seed)
    hashes_before = production_hashes()
    model, checkpoint_info = load_v2_checkpoint(ROOT / config["v2_checkpoint"], ROOT / config["v2_metadata"])
    rows = read_manifest(ROOT / config["manifest"])
    augmentation_config = ablation_config(config["augmentation"], "combined")
    augmentation_config["seed"] = seed
    augmentation_config["geometric"] = {"horizontal_flip": False, "rotation_factor": 0.0, "zoom_factor": 0.0}
    augmentation = LightingAugmentation(augmentation_config, seed=seed, name="v3_e1_photometric")
    train_ds, train_count = small_manifest_dataset(
        rows, "train", samples_per_class, batch_size, True, augmentation
    )
    validation_ds, validation_count = small_manifest_dataset(
        rows, "validation", samples_per_class, batch_size, False
    )

    destination = safe_new_run_dir(run_dir)
    destination.mkdir(parents=True, exist_ok=False)
    phase_results: dict[str, Any] = {}
    for phase, set_trainable, learning_rate in (
        ("head", set_head_trainable, float(config["training"]["learning_rate_head"])),
        (
            "finetune",
            lambda target: set_finetune_trainable(target, int(config["training"]["fine_tune_last_layers"])),
            float(config["training"]["learning_rate_finetune"]),
        ),
    ):
        set_trainable(model)
        compile_for_phase(model, learning_rate, float(config["training"]["weight_decay"]))
        before = [weight.numpy().copy() for weight in model.trainable_weights]
        callbacks, selector = make_callbacks(
            destination, f"smoke_{phase}", 1, validation_ds, config["selection"]
        )
        history = model.fit(
            train_ds,
            steps_per_epoch=steps_per_phase,
            validation_data=validation_ds,
            epochs=1,
            callbacks=callbacks,
            verbose=2,
        )
        updated = changed_trainable_weights(before, model)
        if not updated:
            raise RuntimeError(f"smoke test observed no trainable-weight update in phase {phase}")
        if not selector.epoch_metrics:
            raise RuntimeError(f"smoke test produced no validation metrics in phase {phase}")
        phase_results[phase] = {
            "learning_rate": learning_rate,
            "trainable_weight_arrays": len(model.trainable_weights),
            "weights_updated": updated,
            "history": history_values(history),
            "validation_metrics": selector.epoch_metrics,
            "selection_checkpoint_written": selector.best_checkpoint is not None,
        }

    checkpoint_path = destination / "smoke_checkpoint.keras"
    model.save(checkpoint_path, overwrite=False)
    restored = tf.keras.models.load_model(checkpoint_path, compile=False)
    sample_images, _ = next(iter(validation_ds))
    probabilities = np.asarray(restored(sample_images, training=False))
    if probabilities.shape != (min(batch_size, validation_count), len(CLASSES)):
        raise RuntimeError(f"reloaded checkpoint returned unexpected output shape {probabilities.shape}")
    if not np.all(np.isfinite(probabilities)) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise RuntimeError("reloaded checkpoint returned invalid class probabilities")
    hashes_after = production_hashes()
    if hashes_before != hashes_after:
        raise RuntimeError("a production model or metadata file changed during the smoke test")

    result = {
        "status": "passed",
        "experiment": config["experiment"],
        "seed": seed,
        "ablation": "combined",
        "smoke_subset": {
            "train_images": train_count,
            "validation_images": validation_count,
            "samples_per_class_per_split": samples_per_class,
            "batch_size": batch_size,
            "steps_per_phase": steps_per_phase,
            "phases": ["head", "finetune"],
        },
        "v2_checkpoint": checkpoint_info,
        "phase_results": phase_results,
        "checkpoint": str(checkpoint_path.resolve()),
        "reloaded_output_shape": list(probabilities.shape),
        "production_hashes_before": hashes_before,
        "production_hashes_after": hashes_after,
        "production_hashes_unchanged": True,
    }
    (destination / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--steps-per-phase", type=int, default=1)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["seed"])
    result = run_smoke(args.config, args.run_dir, args.batch_size, args.samples_per_class, args.steps_per_phase, seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
