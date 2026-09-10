"""Temperature scaling and threshold analysis for V2.1 state confidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["verde", "maduro", "sobre_maduro", "podrido"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models/production_model_v2_1.keras")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dataset_v2/validation")
    args = parser.parse_args()
    data = tf.keras.utils.image_dataset_from_directory(args.data_dir, class_names=CLASSES, label_mode="int", image_size=(224, 224), batch_size=64, shuffle=False)
    truth = np.concatenate([y.numpy() for _, y in data])
    raw = tf.keras.models.load_model(args.model, compile=False).predict(data, verbose=0)
    probs = np.asarray(raw["state"] if isinstance(raw, dict) else raw[0])
    logits = np.log(np.clip(probs, 1e-7, 1.0))

    def nll(temperature):
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        p = np.exp(scaled); p /= p.sum(axis=1, keepdims=True)
        return -np.mean(np.log(np.clip(p[np.arange(len(truth)), truth], 1e-7, 1.0)))

    candidates = np.linspace(0.05, 10.0, 200)
    fitted = float(candidates[int(np.argmin([nll(value) for value in candidates]))])
    scaled = logits / fitted; scaled -= scaled.max(axis=1, keepdims=True)
    calibrated = np.exp(scaled); calibrated /= calibrated.sum(axis=1, keepdims=True)
    confidence = calibrated.max(axis=1)
    prediction = calibrated.argmax(1)
    thresholds = []
    for threshold in np.arange(0.50, 0.96, 0.05):
        accepted = confidence >= threshold
        thresholds.append({"threshold": round(float(threshold), 2), "coverage": float(accepted.mean()), "accuracy_accepted": float(accuracy_score(truth[accepted], prediction[accepted])) if accepted.any() else None, "unknown_rate": float((~accepted).mean())})
    result = {"model": str(args.model), "validation_images": len(truth), "temperature": fitted, "raw_accuracy": float(accuracy_score(truth, probs.argmax(1))), "calibrated_accuracy": float(accuracy_score(truth, prediction)), "calibrated_f1_macro": float(f1_score(truth, prediction, average="macro")), "thresholds": thresholds, "recommended_threshold": 0.60, "policy": "confidence < 0.60 returns estado=desconocido", "note": "El umbral se debe volver a validar en un test externo y por cámara antes de producción."}
    (ROOT / "reports/confidence_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (ROOT / "reports/confidence_calibration.md").write_text(f"""# Calibración de confianza V2.1\n\n- Imágenes de validation: **{len(truth)}**\n- Temperature scaling óptimo: **{fitted:.4f}**\n- Accuracy sin calibrar: **{result['raw_accuracy']:.4f}**\n- Accuracy calibrada: **{result['calibrated_accuracy']:.4f}**\n- F1 macro calibrado: **{result['calibrated_f1_macro']:.4f}**\n\n## Política propuesta\n\nSi la confianza calibrada es menor que **0.60**, responder `estado: "desconocido"`. La predicción se conserva en logs para revisión, pero no debe accionar decisiones automáticas.\n\nEl valor 0.60 es un umbral inicial conservador; debe validarse por cámara, iluminación y dominio antes de desplegarlo.\n""", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
