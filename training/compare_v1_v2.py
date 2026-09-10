"""Compare V1 and V2 on the V2 held-out test distribution."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
V2_CLASSES = ["verde", "maduro", "sobre_maduro", "podrido"]
V1_CLASSES = ["verde", "maduro", "pasado"]


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/dataset_v2/manifest.csv")
    parser.add_argument("--v1", type=Path, default=ROOT / "models/production_model.keras")
    parser.add_argument("--v2", type=Path, default=ROOT / "models/production_model_v2.keras")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == "test"]
    paths = [row["image_path"] for row in rows]
    truth4 = np.array([V2_CLASSES.index(row["class_name"]) for row in rows])
    truth3 = np.array([0 if x == 0 else 1 if x == 1 else 2 for x in truth4])

    def load(path):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(tf.cast(image, tf.float32), (224, 224))

    data = tf.data.Dataset.from_tensor_slices(paths).map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size)
    v1 = tf.keras.models.load_model(args.v1, compile=False)
    v2 = tf.keras.models.load_model(args.v2, compile=False)
    v1_time_start = time.perf_counter(); v1_pred = v1.predict(data, verbose=0).argmax(1); v1_time = time.perf_counter() - v1_time_start
    v2_time_start = time.perf_counter(); v2_pred = v2.predict(data, verbose=0).argmax(1); v2_time = time.perf_counter() - v2_time_start
    v2_pred3 = np.array([0 if x == 0 else 1 if x == 1 else 2 for x in v2_pred])
    result = {
        "test_images": len(rows),
        "distribution": {name: int(sum(truth4 == i)) for i, name in enumerate(V2_CLASSES)},
        "v1_comparable_three_class": metrics(truth3, v1_pred),
        "v2_native_four_class": metrics(truth4, v2_pred),
        "v2_comparable_three_class": metrics(truth3, v2_pred3),
        "model_sizes_bytes": {"v1": args.v1.stat().st_size, "v2": args.v2.stat().st_size},
        "inference_seconds_total": {"v1": v1_time, "v2": v2_time},
        "inference_ms_per_image": {"v1": v1_time / len(rows) * 1000, "v2": v2_time / len(rows) * 1000},
        "note": "V1 se compara en tres clases; V2 también reporta su resultado nativo de cuatro clases. La distribución V2 es externa+actual y no es el test histórico de V1.",
    }
    output = ROOT / "reports/model_comparison_v1_vs_v2.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"""# Comparación production_model V1 vs V2\n\n## Dataset\n\n- Imágenes de test V2: **{len(rows)}**\n- V1 se evalúa en el mismo conjunto, colapsando `sobre_maduro` y `podrido` en `pasado`.\n\n## Métricas comparables de tres clases\n\n| Modelo | Accuracy | Precision macro | Recall macro | F1 macro |\n|---|---:|---:|---:|---:|\n| V1 | {result['v1_comparable_three_class']['accuracy']:.4f} | {result['v1_comparable_three_class']['precision_macro']:.4f} | {result['v1_comparable_three_class']['recall_macro']:.4f} | {result['v1_comparable_three_class']['f1_macro']:.4f} |\n| V2 colapsado | {result['v2_comparable_three_class']['accuracy']:.4f} | {result['v2_comparable_three_class']['precision_macro']:.4f} | {result['v2_comparable_three_class']['recall_macro']:.4f} | {result['v2_comparable_three_class']['f1_macro']:.4f} |\n\n## Métricas nativas V2 (cuatro clases)\n\n| Métrica | Resultado |\n|---|---:|\n| Accuracy | {result['v2_native_four_class']['accuracy']:.4f} |\n| Precision macro | {result['v2_native_four_class']['precision_macro']:.4f} |\n| Recall macro | {result['v2_native_four_class']['recall_macro']:.4f} |\n| F1 macro | {result['v2_native_four_class']['f1_macro']:.4f} |\n\nMatriz V2, filas reales y columnas predichas (`verde`, `maduro`, `sobre_maduro`, `podrido`):\n\n```text\n{result['v2_native_four_class']['confusion_matrix']}\n```\n\n## Tamaño y tiempo\n\n| Modelo | Tamaño | Tiempo medio por imagen |\n|---|---:|---:|\n| V1 | {result['model_sizes_bytes']['v1'] / 1024 / 1024:.2f} MiB | {result['inference_ms_per_image']['v1']:.2f} ms |\n| V2 | {result['model_sizes_bytes']['v2'] / 1024 / 1024:.2f} MiB | {result['inference_ms_per_image']['v2']:.2f} ms |\n\n## Interpretación\n\nLa promoción requiere que V2 supere a V1 en F1 macro comparable y no degrade de forma relevante la latencia o el tamaño. No se debe comparar directamente el accuracy nativo de tres y cuatro clases.\n""", encoding="utf-8")
    (ROOT / "reports/model_comparison_v1_vs_v2.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
