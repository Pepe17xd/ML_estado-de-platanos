"""Train experimental V2.1: four-class state plus continuous maturity score."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLASSES = ["verde", "maduro", "sobre_maduro", "podrido"]
MATURITY_TARGETS = {"verde": 0.125, "maduro": 0.40, "sobre_maduro": 0.70, "podrido": 0.925}


def manifest_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def make_dataset(directory: Path, image_size: int, batch_size: int, training: bool, weights: dict[int, float]):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.15),
    ]) if training else None
    raw = tf.keras.utils.image_dataset_from_directory(
        directory, labels="inferred", class_names=CLASSES, label_mode="int",
        image_size=(image_size, image_size), batch_size=batch_size,
        shuffle=training, seed=42,
    )

    def convert(images, labels):
        labels = tf.cast(labels, tf.int32)
        if augmentation is not None:
            images = augmentation(images, training=True)
        score = tf.gather(tf.constant([MATURITY_TARGETS[name] for name in CLASSES], tf.float32), labels)[:, None]
        state = tf.one_hot(labels, len(CLASSES))
        sample_weight = tf.gather(tf.constant([weights[i] for i in range(len(CLASSES))], tf.float32), labels)
        return images, {"state": state, "maturity_score": score}, {"state": sample_weight, "maturity_score": tf.ones_like(sample_weight)}

    return raw.map(convert, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def build_model(image_size: int):
    inputs = tf.keras.Input((image_size, image_size, 3), name="image")
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV3Small(include_top=False, weights="imagenet", input_tensor=x)
    backbone.trainable = False
    shared = tf.keras.layers.Dropout(0.25)(tf.keras.layers.GlobalAveragePooling2D()(backbone.output))
    shared = tf.keras.layers.Dense(128, activation="relu", name="shared_features")(shared)
    state = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="state")(shared)
    maturity = tf.keras.layers.Dense(1, activation="sigmoid", name="maturity_score")(shared)
    return tf.keras.Model(inputs, {"state": state, "maturity_score": maturity}, name="production_model_v2_1"), backbone


def compile_model(model, learning_rate: float):
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate, weight_decay=1e-5),
        loss={"state": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05), "maturity_score": tf.keras.losses.Huber()},
        loss_weights={"state": 1.0, "maturity_score": 0.25},
        metrics={"state": [tf.keras.metrics.CategoricalAccuracy(name="accuracy")], "maturity_score": [tf.keras.metrics.MeanAbsoluteError(name="mae")]},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/dataset_v2/manifest.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dataset_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "models/production_model_v2_1.keras")
    parser.add_argument("--epochs-head", type=int, default=3)
    parser.add_argument("--epochs-finetune", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(42)
    rows = manifest_rows(args.manifest)
    counts = Counter(row["class_name"] for row in rows if row["split"] == "train")
    maximum = max(counts.values())
    class_weights = {index: maximum / counts[name] for index, name in enumerate(CLASSES)}
    train = make_dataset(args.data_dir / "train", args.image_size, args.batch_size, True, class_weights)
    validation = make_dataset(args.data_dir / "validation", args.image_size, args.batch_size, False, {i: 1.0 for i in range(4)})
    model, backbone = build_model(args.image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(args.output), monitor="val_state_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_state_accuracy", mode="max", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]
    compile_model(model, 1e-3)
    first = model.fit(train, validation_data=validation, epochs=args.epochs_head, callbacks=callbacks)
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 1e-5)
    second = model.fit(train, validation_data=validation, initial_epoch=args.epochs_head, epochs=args.epochs_head + args.epochs_finetune, callbacks=callbacks)
    model.save(args.output)
    metadata = json.loads((ROOT / "models/metadata_v2.json").read_text(encoding="utf-8"))
    metadata.update({
        "model_version": "2.1.0-candidate",
        "model_name": "production_model_v2_1",
        "model_file": args.output.name,
        "outputs": ["estado", "confianza", "madurez_score", "days_remaining"],
        "maturity_score_scale": "0-100",
        "maturity_targets": {name: value * 100 for name, value in MATURITY_TARGETS.items()},
        "train_counts": dict(counts),
        "class_weights": {str(index): value for index, value in class_weights.items()},
        "best_validation_accuracy": max(first.history["val_state_accuracy"] + second.history["val_state_accuracy"]),
        "status": "candidate",
    })
    (ROOT / "models/production_model_v2_1_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "best_validation_accuracy": metadata["best_validation_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
