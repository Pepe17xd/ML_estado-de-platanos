"""Evaluate V2 and V2.1 on exactly data/dataset_v2/test."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["verde", "maduro", "sobre_maduro", "podrido"]


def metric_report(truth, prediction):
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision_macro": float(precision_score(truth, prediction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(truth, prediction, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, prediction).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, default=ROOT / "models/production_model.keras")
    parser.add_argument("--v2_1", type=Path, default=ROOT / "models/production_model_v2_1.keras")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dataset_v2/test")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    data = tf.keras.utils.image_dataset_from_directory(args.data_dir, class_names=CLASSES, label_mode="int", image_size=(224, 224), batch_size=args.batch_size, shuffle=False)
    truth = np.concatenate([labels.numpy() for _, labels in data])
    maturity_targets = np.array([0.125, 0.40, 0.70, 0.925], dtype=np.float32)[truth] * 100.0
    v1 = tf.keras.models.load_model(args.v1, compile=False)
    v2_1 = tf.keras.models.load_model(args.v2_1, compile=False)
    v1_start = time.perf_counter(); v1_pred = v1.predict(data, verbose=0).argmax(1); v1_seconds = time.perf_counter() - v1_start
    v2_start = time.perf_counter(); raw = v2_1.predict(data, verbose=0); v2_seconds = time.perf_counter() - v2_start
    v2_state = raw["state"] if isinstance(raw, dict) else raw[0]
    v2_pred = np.asarray(v2_state).argmax(1)
    maturity_output = raw["maturity_score"] if isinstance(raw, dict) else raw[1]
    maturity = np.asarray(maturity_output)[:, 0] * 100.0
    result = {
        "test_images": len(truth),
        "v2": metric_report(truth, v1_pred),
        "v2_1": metric_report(truth, v2_pred),
        "v2_1_maduro_recall": float(metric_report(truth, v2_pred)["confusion_matrix"][1][1] / max(1, sum(truth == 1))),
        "v2_1_sobre_maduro_recall": float(metric_report(truth, v2_pred)["confusion_matrix"][2][2] / max(1, sum(truth == 2))),
        "v2_1_maturity_score_mae_weak_pct": float(mean_absolute_error(maturity_targets, maturity)),
        "size_bytes": {"v2": args.v1.stat().st_size, "v2_1": args.v2_1.stat().st_size},
        "inference_ms_per_image": {"v2": v1_seconds / len(truth) * 1000, "v2_1": v2_seconds / len(truth) * 1000},
    }
    output = ROOT / "reports/model_comparison_v2_vs_v2_1.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    result["recommendation"] = "promote" if result["v2_1"]["f1_macro"] > result["v2"]["f1_macro"] and result["v2_1_maduro_recall"] >= result["v2"]["confusion_matrix"][1][1] / max(1, sum(truth == 1)) and result["v2_1_sobre_maduro_recall"] >= result["v2"]["confusion_matrix"][2][2] / max(1, sum(truth == 2)) else "keep_v2"
    output.write_text(f"""# Comparación V2 vs V2.1\n\nEvaluación exacta sobre `data/dataset_v2/test`: **{len(truth)} imágenes**.\n\n| Modelo | Accuracy | Precision macro | Recall macro | F1 macro |\n|---|---:|---:|---:|---:|\n| V2 | {result['v2']['accuracy']:.4f} | {result['v2']['precision_macro']:.4f} | {result['v2']['recall_macro']:.4f} | {result['v2']['f1_macro']:.4f} |\n| V2.1 | {result['v2_1']['accuracy']:.4f} | {result['v2_1']['precision_macro']:.4f} | {result['v2_1']['recall_macro']:.4f} | {result['v2_1']['f1_macro']:.4f} |\n\n## Matrices de confusión\n\nV2:\n```text\n{result['v2']['confusion_matrix']}\n```\n\nV2.1 (`verde`, `maduro`, `sobre_maduro`, `podrido`):\n```text\n{result['v2_1']['confusion_matrix']}\n```\n\n## Clases críticas\n\n- Recall `maduro` V2: {result['v2']['confusion_matrix'][1][1] / max(1, sum(truth == 1)):.4f}; V2.1: {result['v2_1_maduro_recall']:.4f}.\n- Recall `sobre_maduro` V2: {result['v2']['confusion_matrix'][2][2] / max(1, sum(truth == 2)):.4f}; V2.1: {result['v2_1_sobre_maduro_recall']:.4f}.\n\n## Tamaño y latencia\n\n| Modelo | Tamaño | Inferencia media |\n|---|---:|---:|\n| V2 | {result['size_bytes']['v2'] / 1024 / 1024:.2f} MiB | {result['inference_ms_per_image']['v2']:.2f} ms/imagen |\n| V2.1 | {result['size_bytes']['v2_1'] / 1024 / 1024:.2f} MiB | {result['inference_ms_per_image']['v2_1']:.2f} ms/imagen |\n\n## Recomendación automática\n\n`{result['recommendation']}`. La promoción definitiva requiere revisar también calibración, coste de latencia y comportamiento fuera de distribución.\n""", encoding="utf-8")
    (ROOT / "reports/model_comparison_v2_vs_v2_1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
