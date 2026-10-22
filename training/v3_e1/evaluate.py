"""Paired V2/V3-E1 evaluation on one frozen image manifest; no tuning is performed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from inference.preprocessing import preprocess_encoded
from training.v3_e1.model import CLASSES

ROOT = Path(__file__).resolve().parents[2]


def read_evaluation_rows(manifest: Path, split: str) -> list[dict[str, str]]:
    with Path(manifest).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image_path"}.issubset(reader.fieldnames) or not ({"class_name"} <= set(reader.fieldnames) or {"label"} <= set(reader.fieldnames)):
            raise ValueError("evaluation manifest needs image_path and class_name (or label) columns")
        rows = list(reader)
    if "split" in (reader.fieldnames or []):
        rows = [row for row in rows if row.get("split") == split]
    elif split not in {"esp32_cam", "custom"}:
        raise ValueError("manifest has no split column; use --split esp32_cam or custom for an independent benchmark")
    for row in rows:
        label = row.get("class_name") or row.get("label")
        if label not in CLASSES:
            raise ValueError(f"unknown class label {label!r}; expected class order {CLASSES}")
        raw_path = Path(row["image_path"])
        path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        if not path.is_file():
            raise FileNotFoundError(f"evaluation image missing: {path}")
        row["_resolved_path"] = str(path.resolve())
        row["_label"] = label
    if not rows:
        raise ValueError(f"no evaluation records for split {split!r}")
    return rows


def image_dataset(rows: list[dict[str, str]], batch_size: int) -> tf.data.Dataset:
    paths = [row["_resolved_path"] for row in rows]
    labels = [CLASSES.index(row["_label"]) for row in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        def tensorflow_resize(path_value: bytes) -> np.ndarray:
            if hasattr(path_value, "numpy"):
                path_value = path_value.numpy()
            raw_path = path_value.decode("utf-8")
            return preprocess_encoded(Path(raw_path).read_bytes(), (224, 224), "tensorflow_bilinear")[0]

        image = tf.py_function(tensorflow_resize, [path], Tout=tf.float32)
        image.set_shape([224, 224, 3])
        return image, tf.cast(label, tf.int32)

    options = tf.data.Options()
    options.experimental_deterministic = True
    return ds.map(load, num_parallel_calls=1, deterministic=True).batch(batch_size).with_options(options).prefetch(tf.data.AUTOTUNE)


def classification_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for actual, predicted in zip(truth, prediction):
        matrix[int(actual), int(predicted)] += 1
    supports = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    recall = np.divide(matrix.diagonal(), supports, out=np.zeros(len(CLASSES), dtype=float), where=supports > 0)
    precision = np.divide(matrix.diagonal(), predicted_counts, out=np.zeros(len(CLASSES), dtype=float), where=predicted_counts > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(CLASSES), dtype=float), where=(precision + recall) > 0)
    return {
        "n": int(len(truth)),
        "accuracy": float(np.mean(truth == prediction)),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "per_class": {
            name: {"support": int(supports[i]), "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i, name in enumerate(CLASSES)
        },
        "confusion_matrix": matrix.tolist(),
    }


def calibration_metrics(probabilities: np.ndarray, truth: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Return descriptive ECE and multiclass Brier score; no threshold is applied."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape != (len(truth), len(CLASSES)) or len(truth) == 0:
        raise ValueError("probabilities must have shape (n, 4) matching non-empty truth")
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        in_bin = (confidence >= edges[index]) & (confidence < edges[index + 1] if index < bins - 1 else confidence <= edges[index + 1])
        if np.any(in_bin):
            ece += float(np.mean(in_bin)) * abs(float(np.mean(correct[in_bin])) - float(np.mean(confidence[in_bin])))
    one_hot = np.eye(len(CLASSES), dtype=np.float64)[truth]
    return {"expected_calibration_error_10_bins": ece, "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))}


def _predictions(model_path: Path, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray, float]:
    model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise ValueError(f"model {model_path} does not accept 224x224 RGB input")
    if tuple(model.output_shape[1:]) != (4,):
        raise ValueError(f"model {model_path} does not have four class outputs")
    first = next(iter(dataset), None)
    if first is None:
        raise ValueError("evaluation dataset is empty")
    model(first[0], training=False).numpy()  # warm-up excluded from the reported model-call time
    batches = []
    probability_batches = []
    elapsed = 0.0
    for images, _ in dataset:
        started = time.perf_counter()
        output = model(images, training=False)
        if isinstance(output, dict):
            output = output.get("state")
        elif isinstance(output, (tuple, list)):
            output = output[0]
        probabilities = np.asarray(output)
        if probabilities.ndim != 2 or probabilities.shape[1] != 4:
            raise ValueError(f"model {model_path} returned incompatible output shape {probabilities.shape}")
        elapsed += time.perf_counter() - started
        probability_batches.append(probabilities)
        batches.append(probabilities.argmax(axis=1))
    return (
        np.concatenate(batches).astype(np.int64),
        np.concatenate(probability_batches),
        elapsed * 1000.0 / sum(len(batch) for batch in batches),
    )


def grouped_metrics(rows: list[dict[str, str]], truth: np.ndarray, prediction: np.ndarray, key: str) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        value = row.get(key, "").strip()
        if value:
            groups[value].append(index)
    return {
        value: classification_metrics(truth[indices], prediction[indices])
        for value, indices in sorted(groups.items())
    }


def evaluate_pair(rows: list[dict[str, str]], dataset: tf.data.Dataset, v2_path: Path, v3_path: Path) -> dict[str, Any]:
    truth = np.asarray([CLASSES.index(row["_label"]) for row in rows], dtype=np.int64)
    v2, v2_probabilities, v2_ms = _predictions(v2_path, dataset)
    v3, v3_probabilities, v3_ms = _predictions(v3_path, dataset)
    if len(v2) != len(rows) or len(v3) != len(rows):
        raise ValueError("model prediction count differs from manifest row count")
    v2_metrics = classification_metrics(truth, v2)
    v3_metrics = classification_metrics(truth, v3)
    keys = ("accuracy", "macro_f1", "balanced_accuracy")
    return {
        "class_order": CLASSES,
        "n": len(rows),
        "v2": v2_metrics,
        "v3_e1": v3_metrics,
        "calibration": {
            "v2": calibration_metrics(v2_probabilities, truth),
            "v3_e1": calibration_metrics(v3_probabilities, truth),
            "note": "descriptive ECE/Brier on the requested split; no confidence threshold is selected",
        },
        "runtime": {
            "v2_model_size_bytes": int(Path(v2_path).stat().st_size),
            "v3_model_size_bytes": int(Path(v3_path).stat().st_size),
            "v2_model_ms_per_image": v2_ms,
            "v3_model_ms_per_image": v3_ms,
            "timing_note": "model calls on decoded batches; excludes model loading, decoding, and HTTP overhead",
        },
        "delta_v3_minus_v2": {key: v3_metrics[key] - v2_metrics[key] for key in keys},
        "delta_per_class_v3_minus_v2": {
            name: {
                metric: v3_metrics["per_class"][name][metric] - v2_metrics["per_class"][name][metric]
                for metric in ("precision", "recall", "f1")
            }
            for name in CLASSES
        },
        "confusion_matrix_delta_v3_minus_v2": (
            np.asarray(v3_metrics["confusion_matrix"], dtype=np.int64)
            - np.asarray(v2_metrics["confusion_matrix"], dtype=np.int64)
        ).tolist(),
        "by_source_dataset": {"v2": grouped_metrics(rows, truth, v2, "source_dataset"), "v3_e1": grouped_metrics(rows, truth, v3, "source_dataset")},
        "by_lighting_condition": {"v2": grouped_metrics(rows, truth, v2, "lighting_condition"), "v3_e1": grouped_metrics(rows, truth, v3, "lighting_condition")},
    }


def validate_model_class_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    if metadata.get("classes") != CLASSES:
        raise ValueError(f"model metadata class order must be {CLASSES}")
    return metadata


def evaluation_markdown(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    lines = [
        "# Comparación V2/V3-E1",
        "",
        f"- Split: `{result['split']}`; imágenes: {metrics['n']}.",
        f"- SHA-256 manifiesto: `{result['manifest_sha256']}`.",
        "- El test se reserva para informe final; no usarlo para seleccionar hiperparámetros.",
        "",
        "| Modelo | Accuracy | Macro F1 | Balanced accuracy | Tamaño (bytes) | Inferencia (ms/imagen) | Recall podrido |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, key, runtime_key in (("V2", "v2", "v2"), ("V3-E1", "v3_e1", "v3")):
        model_metrics = metrics[key]
        lines.append(
            f"| {name} | {model_metrics['accuracy']:.4f} | {model_metrics['macro_f1']:.4f} | "
            f"{model_metrics['balanced_accuracy']:.4f} | {metrics['runtime'][runtime_key + '_model_size_bytes']} | "
            f"{metrics['runtime'][runtime_key + '_model_ms_per_image']:.3f} | "
            f"{model_metrics['per_class']['podrido']['recall']:.4f} |"
        )
    lines += [
        "",
        "## Calibración descriptiva",
        "",
        "| Modelo | ECE (10 bins) | Brier multiclase |",
        "|---|---:|---:|",
    ]
    for name, key in (("V2", "v2"), ("V3-E1", "v3_e1")):
        calibration = metrics["calibration"][key]
        lines.append(
            f"| {name} | {calibration['expected_calibration_error_10_bins']:.4f} | "
            f"{calibration['multiclass_brier_score']:.4f} |"
        )
    lines += ["", "## Matrices de confusión", "", "Filas = clase real; columnas = clase predicha en el orden declarado por metadata.", ""]
    for name, key in (("V2", "v2"), ("V3-E1", "v3_e1")):
        lines += [f"### {name}", "", "```text", json.dumps(metrics[key]["confusion_matrix"], indent=2), "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test", help="historical split name, or esp32_cam/custom for a manifest without split")
    parser.add_argument("--v2-model", type=Path, default=ROOT / "models/production_model_v2.keras")
    parser.add_argument("--v3-model", type=Path, required=True)
    parser.add_argument("--v2-metadata", type=Path, default=ROOT / "models/production_model_v2_metadata.json")
    parser.add_argument("--v3-metadata", type=Path, help="defaults to run_metadata.json beside --v3-model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, help="optional new JSON report path; existing files are never overwritten")
    args = parser.parse_args()
    validate_model_class_metadata(args.v2_metadata)
    v3_metadata_path = args.v3_metadata or args.v3_model.resolve().parent / "run_metadata.json"
    v3_metadata = validate_model_class_metadata(v3_metadata_path)
    rows = read_evaluation_rows(args.manifest, args.split)
    ds = image_dataset(rows, args.batch_size)
    result = {
        "experiment": "banana_ai_system_v3_lighting_e1",
        "split": args.split,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "v3_model_version": v3_metadata.get("model_version"),
        "selection_warning": "Use validation for choices. Historical test is final reporting only; ESP32-CAM benchmark must be independently collected.",
        "fruit_lot_independence_warning": "Historical manifest does not prove independence by physical fruit or lot.",
        "input_preprocessing": "same tf.io.decode_image RGB and tf.image.resize bilinear path used by V3 training/API",
        "metrics": evaluate_pair(rows, ds, args.v2_model, args.v3_model),
    }
    serialized = json.dumps(result, indent=2)
    if args.output:
        output = args.output.resolve()
        markdown_path = output.with_suffix(".md")
        if output.exists() or markdown_path.exists():
            raise FileExistsError(f"refusing to overwrite evaluation report: {output} or {markdown_path}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        markdown_path.write_text(evaluation_markdown(result), encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
