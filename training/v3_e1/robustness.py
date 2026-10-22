"""Compare V2/V3 under deterministic, synthetic illumination perturbations.

This is a stress test, not a substitute for photographs collected under real
cameras, lighting, fruit identities, or storage conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from training.v3_e1.evaluate import classification_metrics, image_dataset, read_evaluation_rows, validate_model_class_metadata
from training.v3_e1.model import CLASSES

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = (
    "clean",
    "dim",
    "bright",
    "low_contrast",
    "high_contrast",
    "cool",
    "warm",
    "gamma_bright",
    "gamma_dark",
    "soft_shadow",
)


def perturb_batch(images: tf.Tensor, condition: str) -> tf.Tensor:
    """Apply one fixed, moderate synthetic condition to RGB [0,255] batches."""
    x = tf.cast(images, tf.float32)
    if condition == "clean":
        return x
    unit = tf.clip_by_value(x / 255.0, 0.0, 1.0)
    if condition == "dim":
        result = x * 0.78
    elif condition == "bright":
        result = x * 1.12 + 8.0
    elif condition == "low_contrast":
        result = tf.image.adjust_contrast(unit, 0.82) * 255.0
    elif condition == "high_contrast":
        result = tf.image.adjust_contrast(unit, 1.18) * 255.0
    elif condition == "cool":
        result = x * tf.constant([0.97, 1.0, 1.04], tf.float32)
    elif condition == "warm":
        result = x * tf.constant([1.04, 1.0, 0.97], tf.float32)
    elif condition == "gamma_bright":
        result = tf.image.adjust_gamma(unit, gamma=0.85) * 255.0
    elif condition == "gamma_dark":
        result = tf.image.adjust_gamma(unit, gamma=1.15) * 255.0
    elif condition == "soft_shadow":
        height, width = tf.shape(x)[1], tf.shape(x)[2]
        yy = tf.linspace(-1.0, 1.0, height)[:, None]
        xx = tf.linspace(-1.0, 1.0, width)[None, :]
        distance = tf.sqrt(tf.square((yy + 0.15) / 0.8) + tf.square((xx - 0.15) / 1.1))
        mask = tf.sigmoid((0.9 - distance) / 0.22)[None, ..., None]
        result = x * (1.0 - 0.18 * mask)
    else:
        raise ValueError(f"unknown perturbation {condition!r}; choose from {CONDITIONS}")
    return tf.clip_by_value(result, 0.0, 255.0)


def _load_model(path: Path) -> tf.keras.Model:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    model = tf.keras.models.load_model(path, compile=False)
    if tuple(model.input_shape[1:]) != (224, 224, 3) or tuple(model.output_shape[1:]) != (4,):
        raise ValueError(f"model must accept 224x224 RGB and return 4 classes: {path}")
    return model


def _evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    truth: np.ndarray,
    rows: list[dict[str, str]],
    model_size_bytes: int,
) -> dict[str, Any]:
    predictions_by_condition: dict[str, np.ndarray] = {}
    model_seconds = 0.0
    first = next(iter(dataset), None)
    if first is None:
        raise ValueError("robustness evaluation dataset is empty")
    model(perturb_batch(first[0], "clean"), training=False).numpy()  # exclude one-time graph warm-up
    for condition in CONDITIONS:
        batches: list[np.ndarray] = []
        for images, _ in dataset:
            values = perturb_batch(images, condition)
            started = time.perf_counter()
            output = model(values, training=False)
            probabilities = np.asarray(output)
            model_seconds += time.perf_counter() - started
            if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASSES):
                raise ValueError(f"incompatible model output shape {probabilities.shape}")
            batches.append(probabilities.argmax(axis=1).astype(np.int64))
        predictions_by_condition[condition] = np.concatenate(batches)

    clean = predictions_by_condition["clean"]
    clean_accuracy = float(np.mean(clean == truth))
    conditions: dict[str, Any] = {}
    for condition, predictions in predictions_by_condition.items():
        correct_to_incorrect = np.flatnonzero((clean == truth) & (predictions != truth))
        changed = predictions != clean
        metrics = classification_metrics(truth, predictions)
        conditions[condition] = {
            **metrics,
            "accuracy_drop_vs_clean_percentage_points": clean_accuracy - metrics["accuracy"],
            "prediction_changed_count_vs_clean": int(changed.sum()),
            "prediction_changed_percent_vs_clean": float(changed.mean() * 100.0),
            "clean_correct_became_wrong_count": int(len(correct_to_incorrect)),
            "clean_correct_became_wrong_percent_of_clean_correct": (
                float(len(correct_to_incorrect) / max(1, int((clean == truth).sum())) * 100.0)
            ),
            "regressed_examples": [
                {
                    "image_path": rows[index]["image_path"],
                    "true_class": CLASSES[int(truth[index])],
                    "clean_prediction": CLASSES[int(clean[index])],
                    "condition_prediction": CLASSES[int(predictions[index])],
                }
                for index in correct_to_incorrect[:50]
            ],
        }
    return {
        "model_seconds_across_all_conditions": model_seconds,
        "model_ms_per_image_condition_pass": model_seconds / (len(truth) * len(CONDITIONS)) * 1000.0,
        "model_size_bytes": int(model_size_bytes),
        "conditions": conditions,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Robustez sintética V2/V3-E1",
        "",
        "> Perturbaciones sintéticas moderadas sobre imágenes existentes; no representan una cámara real ni garantizan conservación semántica de la etiqueta.",
        "",
        f"- Split evaluado: `{result['split']}` ({result['n']} imágenes).",
        f"- SHA-256 del manifiesto: `{result['manifest_sha256']}`.",
        f"- Orden de clases: `{', '.join(result['class_order'])}`.",
        "- No usar este reporte para ajustar hiperparámetros si el split es test.",
        "",
        "| Modelo | Tamaño checkpoint (bytes) | Tiempo modelo (ms/imagen/condición) |",
        "|---|---:|---:|",
    ]
    for model_name, model_result in result["models"].items():
        lines.append(
            f"| {model_name} | {model_result['model_size_bytes']} | "
            f"{model_result['model_ms_per_image_condition_pass']:.3f} |"
        )
    lines += [
        "",
        "| Modelo | Condición | Accuracy | Macro F1 | Recall podrido | Cambio de predicción vs limpia | Correctas que pasan a incorrectas | Caída accuracy (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, model_result in result["models"].items():
        for condition, metrics in model_result["conditions"].items():
            rotten_recall = metrics["per_class"]["podrido"]["recall"]
            lines.append(
                f"| {model_name} | {condition} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
                f"{rotten_recall:.4f} | {metrics['prediction_changed_percent_vs_clean']:.2f}% | "
                f"{metrics['clean_correct_became_wrong_count']} | {metrics['accuracy_drop_vs_clean_percentage_points']:.4f} |"
            )
    lines += [
        "",
        "## Interpretación y límites",
        "",
        "Las variantes se calculan sobre las imágenes del split y usan las mismas etiquetas para comparar sensibilidad. La transformación sintética puede cambiar señales visuales relevantes; las regresiones deben revisarse antes de considerarlas errores inequívocos. Este análisis no sustituye fotos reales con iluminación, cámara, exposición y fruta identificadas.",
        "",
        "El tiempo indicado cubre llamadas del modelo sobre batches ya decodificados y transformados, excluye carga del modelo y no es una medición de latencia de servicio. El tamaño corresponde al archivo del checkpoint.",
        "",
    ]
    return "\n".join(lines)


def run(
    manifest: Path,
    split: str,
    v2_path: Path,
    v3_path: Path,
    output: Path,
    batch_size: int,
    v2_metadata_path: Path = ROOT / "models/production_model_v2_metadata.json",
    v3_metadata_path: Path | None = None,
) -> dict[str, Any]:
    v2_metadata = validate_model_class_metadata(v2_metadata_path)
    v3_metadata_path = v3_metadata_path or Path(v3_path).resolve().parent / "run_metadata.json"
    v3_metadata = validate_model_class_metadata(v3_metadata_path)
    rows = read_evaluation_rows(manifest, split)
    dataset = image_dataset(rows, batch_size)
    truth = np.asarray([CLASSES.index(row["_label"]) for row in rows], dtype=np.int64)
    models = {}
    for name, path in (("v2", v2_path), ("v3_e1", v3_path)):
        model = _load_model(path)
        models[name] = _evaluate_model(model, dataset, truth, rows, Path(path).stat().st_size)
    manifest_hash = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    result = {
        "experiment": "banana_ai_system_v3_e1_synthetic_robustness",
        "split": split,
        "n": len(rows),
        "class_order": CLASSES,
        "input_preprocessing": "same tf.io.decode_image RGB and tf.image.resize bilinear path used by V3 training/API",
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": manifest_hash,
        "synthetic_conditions": list(CONDITIONS),
        "real_world_robustness_claim": False,
        "model_versions": {"v2": v2_metadata.get("model_version"), "v3_e1": v3_metadata.get("model_version")},
        "models": models,
    }
    output = Path(output).resolve()
    markdown_path = output.with_suffix(".md")
    if output.exists() or markdown_path.exists():
        raise FileExistsError(f"refusing to overwrite robustness output: {output} or {markdown_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/dataset_v2/manifest.csv")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--v2-model", type=Path, default=ROOT / "models/production_model_v2.keras")
    parser.add_argument("--v2-metadata", type=Path, default=ROOT / "models/production_model_v2_metadata.json")
    parser.add_argument("--v3-model", type=Path, required=True)
    parser.add_argument("--v3-metadata", type=Path, help="defaults to run_metadata.json beside --v3-model")
    parser.add_argument("--output", type=Path, required=True, help="nuevo path JSON; también crea un Markdown con el mismo basename")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    result = run(
        args.manifest,
        args.split,
        args.v2_model,
        args.v3_model,
        args.output,
        args.batch_size,
        args.v2_metadata,
        args.v3_metadata,
    )
    print(json.dumps({"n": result["n"], "conditions": result["synthetic_conditions"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
