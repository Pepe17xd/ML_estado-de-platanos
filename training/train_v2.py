"""Train the four-class production model candidate on dataset-v2."""
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


def read_manifest(path: Path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def dataset(directory: Path, image_size: int, batch_size: int, class_names: list[str], training: bool):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        class_names=class_names,
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=training,
        seed=42,
    ).prefetch(tf.data.AUTOTUNE)


def build_model(image_size: int):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augmentation")
    inputs = tf.keras.Input((image_size, image_size, 3), name="image")
    x = tf.keras.applications.mobilenet_v3.preprocess_input(augmentation(inputs))
    backbone = tf.keras.applications.MobileNetV3Small(
        include_top=False, weights="imagenet", input_tensor=x
    )
    backbone.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    output = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="state")(x)
    return tf.keras.Model(inputs, output, name="production_model_v2"), backbone


def compile_model(model, learning_rate: float):
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate, weight_decay=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/dataset_v2/manifest.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dataset_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "models/production_model_v2.keras")
    parser.add_argument("--epochs-head", type=int, default=8)
    parser.add_argument("--epochs-finetune", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(42)

    rows = read_manifest(args.manifest)
    train_counts = Counter(row["class_name"] for row in rows if row["split"] == "train")
    maximum = max(train_counts.values())
    class_weights = {CLASSES.index(name): maximum / train_counts[name] for name in CLASSES}
    train = dataset(args.data_dir / "train", args.image_size, args.batch_size, CLASSES, True)
    validation = dataset(args.data_dir / "validation", args.image_size, args.batch_size, CLASSES, False)
    model, backbone = build_model(args.image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(args.output), monitor="val_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]
    compile_model(model, 1e-3)
    first = model.fit(train, validation_data=validation, epochs=args.epochs_head, class_weight=class_weights, callbacks=callbacks)
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 1e-5)
    second = model.fit(
        train, validation_data=validation, initial_epoch=args.epochs_head,
        epochs=args.epochs_head + args.epochs_finetune, class_weight=class_weights, callbacks=callbacks,
    )
    model.save(args.output)
    metadata = json.loads((ROOT / "models/metadata_v2.json").read_text(encoding="utf-8"))
    metadata.update({
        "model_file": args.output.name,
        "train_counts": dict(train_counts),
        "class_weights": {str(key): value for key, value in class_weights.items()},
        "best_validation_accuracy": max(first.history["val_accuracy"] + second.history["val_accuracy"]),
    })
    (ROOT / "models/production_model_v2_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "class_weights": class_weights, "best_val_accuracy": metadata["best_validation_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
