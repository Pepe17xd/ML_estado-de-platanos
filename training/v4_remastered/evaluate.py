"""Evaluate V4 and historical baselines on a reserved validation development holdout.

This is not the independent final test requested for promotion: the containing
legacy validation set has historical use and the project contains no new capture set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance

from training.three_class.contract import CLASSES
from training.three_class.evaluate import choose_abstention_threshold, choose_temperature, temperature_scale
from training.v4_remastered.train import (
    ROOT, filter_near_duplicate_validation, isolate_validation_holdout, rows_for_split, split_validation,
)


def model_probs(model, rows: list[dict], batch_size: int = 32) -> np.ndarray:
    paths = [row["path"] for row in rows]
    dataset = tf.data.Dataset.from_tensor_slices(paths)

    def load(path):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        return tf.image.resize(image, [224, 224], method="bilinear", antialias=False)

    dataset = dataset.map(load, num_parallel_calls=1, deterministic=True).batch(batch_size)
    outputs = []
    for batch in dataset:
        outputs.extend(model(batch, training=False).numpy().tolist())
    return np.asarray(outputs, dtype=np.float64)


def confusion_and_metrics(y: np.ndarray, probs: np.ndarray, four_class: bool = False) -> dict:
    pred_raw = probs.argmax(axis=1)
    pred = pred_raw.copy()
    if four_class:
        pred[pred_raw == 3] = 3
    matrix = np.zeros((3, 4 if four_class else 3), dtype=int)
    for truth, guess in zip(y, pred):
        matrix[int(truth), int(guess)] += 1
    per_class = {}
    f1s, recalls = [], []
    for cls, name in enumerate(CLASSES):
        tp = matrix[cls, cls]
        precision = tp / max(1, int(matrix[:, cls].sum()))
        recall = tp / max(1, int(matrix[cls, :].sum()))
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        f1s.append(f1); recalls.append(recall)
        per_class[name] = {"precision": float(precision), "recall": float(recall),
                           "f1": float(f1), "support": int(matrix[cls, :].sum())}
    correct = pred == y
    confidence = probs.max(axis=1)
    high = confidence >= 0.9
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        selected = (confidence >= left) & (confidence < right if right < 1 else confidence <= right)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return {"n": int(len(y)), "accuracy": float(correct.mean()), "macro_f1": float(np.mean(f1s)),
            "per_class": per_class, "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
            "prediction_columns": [*CLASSES, *( ["podrido_no_admitido"] if four_class else [])],
            "ece_10_bins": float(ece), "high_confidence_threshold": 0.9,
            "brier_multiclass": float(np.mean(np.sum((probs[:, :3] - np.eye(3)[y]) ** 2, axis=1))) if probs.shape[1] == 3 else None,
            "high_confidence_n": int(high.sum()),
            "high_confidence_error_rate": float((~correct[high]).mean()) if high.any() else None}


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n <= 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [float(center - half), float(center + half)]


def confidence_intervals(y: np.ndarray, probs: np.ndarray, seed: int = 4072026, reps: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    prediction = probs.argmax(axis=1)
    n = len(y)
    f1_boot = []
    for _ in range(reps):
        indices = rng.integers(0, n, size=n)
        mat = np.zeros((3, 3), dtype=int)
        for truth, guess in zip(y[indices], prediction[indices]):
            mat[int(truth), int(guess)] += 1
        f1 = []
        for cls in range(3):
            precision = mat[cls, cls] / max(1, int(mat[:, cls].sum()))
            recall = mat[cls, cls] / max(1, int(mat[cls, :].sum()))
            f1.append(2 * precision * recall / max(1e-12, precision + recall))
        f1_boot.append(float(np.mean(f1)))
    accuracy = int(np.sum(prediction == y))
    return {"method": "Wilson 95% for accuracy/recall; percentile bootstrap 2000 resamples for macro-F1",
            "accuracy_95_ci": wilson(accuracy, n),
            "recall_95_ci_by_class": {
                name: wilson(int(np.sum((prediction == cls) & (y == cls))), int(np.sum(y == cls)))
                for cls, name in enumerate(CLASSES)},
            "macro_f1_95_ci": [float(np.quantile(f1_boot, 0.025)), float(np.quantile(f1_boot, 0.975))]}


def photo_variants(image: Image.Image) -> dict[str, Image.Image]:
    image = image.convert("RGB")
    red, green, blue = image.split()
    warm = Image.merge("RGB", (red.point(lambda v: min(255, round(v * 1.06))), green,
                                blue.point(lambda v: min(255, round(v * 0.94)))))
    cool = Image.merge("RGB", (red.point(lambda v: min(255, round(v * 0.94))), green,
                                blue.point(lambda v: min(255, round(v * 1.06)))))
    array = np.asarray(image, dtype=np.float32)
    yy, xx = np.mgrid[0:image.height, 0:image.width]
    mask = np.exp(-((xx-image.width*0.68)**2 + (yy-image.height*0.48)**2) /
                  (2 * max(1., min(image.size)*0.38)**2))[..., None]
    shadow = Image.fromarray(np.uint8(np.clip(array * (1 - 0.35 * mask), 0, 255)))
    highlight = Image.fromarray(np.uint8(np.clip(array + 35 * mask, 0, 255)))
    return {
        "gamma_1_25": image.point(lambda v: round(255 * (v / 255) ** 1.25)),
        "gamma_0_75": image.point(lambda v: round(255 * (v / 255) ** 0.75)),
        "exposure_0_70": ImageEnhance.Brightness(image).enhance(0.70),
        "exposure_1_20": ImageEnhance.Brightness(image).enhance(1.20),
        "warm_white_balance": warm, "cool_white_balance": cool,
        "contrast_0_85": ImageEnhance.Contrast(image).enhance(0.85),
        "contrast_1_20": ImageEnhance.Contrast(image).enhance(1.20),
        "saturation_0_85": ImageEnhance.Color(image).enhance(0.85),
        "soft_shadow": shadow, "uneven_highlight": highlight,
    }


def predict_pil(model, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
    batches = []
    for start in range(0, len(images), batch_size):
        pixels = [tf.image.resize(np.asarray(im.convert("RGB"), np.uint8), [224, 224],
                                  method="bilinear", antialias=False).numpy()
                  for im in images[start:start + batch_size]]
        batches.extend(model(np.asarray(pixels, dtype=np.float32), training=False).numpy().tolist())
    return np.asarray(batches, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/dataset_v2/manifest.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--previous-candidate", type=Path,
                        default=ROOT / "artifacts/three_class/mobilenetv3_seed2026_cpu_reviewed/best.keras")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") not in {"candidate_trained", "candidate_early_stopped"}:
        raise ValueError("run is not a saved V4 candidate")
    raw_val_rows = rows_for_split(args.manifest, ROOT, "validation")
    val_rows, leakage_audit = filter_near_duplicate_validation(args.manifest, ROOT, raw_val_rows)
    development, raw_holdout = split_validation(val_rows, int(metadata["seed"]))
    heldout, holdout_audit = isolate_validation_holdout(val_rows, development, raw_holdout)
    if not heldout:
        raise ValueError("no V4 holdout images remain after near-duplicate isolation")
    y = np.asarray([row["label"] for row in heldout], dtype=np.int64)
    v4 = tf.keras.models.load_model(run / "best.keras", compile=False)
    dev_probs = model_probs(v4, development)
    raw = model_probs(v4, heldout)
    temperature, _ = choose_temperature(np.asarray([r["label"] for r in development]), dev_probs)
    scaled_dev = temperature_scale(dev_probs, temperature)
    development_truth = np.asarray([r["label"] for r in development], dtype=np.int64)
    development_metrics = confusion_and_metrics(development_truth, scaled_dev)
    abstention = choose_abstention_threshold(np.asarray([r["label"] for r in development]), scaled_dev, 0.95)
    calibrated = temperature_scale(raw, temperature)
    v4_metrics = confusion_and_metrics(y, calibrated)
    v4_metrics["confidence_intervals"] = confidence_intervals(y, calibrated)
    threshold = float(abstention["threshold"])
    accepted = calibrated.max(axis=1) >= threshold
    v4_metrics["selective"] = {"threshold_from_dev_split": threshold,
                               "coverage": float(accepted.mean()),
                               "selective_accuracy": float((calibrated.argmax(axis=1)[accepted] == y[accepted]).mean()) if accepted.any() else None,
                               "dev_coverage": abstention["validation_coverage"],
                               "dev_selective_accuracy": abstention["validation_selective_accuracy"]}

    # Same exact reserved rows, three target labels. V2's rotten prediction is
    # retained as a non-admitted fourth output and counts as an error.
    baseline_paths = {
        "V2_4_class": ROOT / "models/production_model.h5",
        "previous_three_class_candidate": args.previous_candidate.resolve(),
    }
    comparison = {"V4": v4_metrics}
    v2 = tf.keras.models.load_model(baseline_paths["V2_4_class"], compile=False)
    v2_probs = model_probs(v2, heldout)
    comparison["V2_4_class"] = confusion_and_metrics(y, v2_probs, four_class=True)
    previous = tf.keras.models.load_model(baseline_paths["previous_three_class_candidate"], compile=False)
    previous_probs = model_probs(previous, heldout)
    comparison["previous_three_class_candidate"] = confusion_and_metrics(y, previous_probs)

    clean_prediction = calibrated.argmax(axis=1)
    perturbations = {}
    variant_images: dict[str, list[Image.Image]] = {}
    for row in heldout:
        with Image.open(row["path"]) as image:
            for name, variant in photo_variants(image).items():
                variant_images.setdefault(name, []).append(variant)
    base_accuracy = v4_metrics["accuracy"]
    for name, images in variant_images.items():
        probabilities = temperature_scale(predict_pil(v4, images), temperature)
        prediction = probabilities.argmax(axis=1)
        perturb_metrics = confusion_and_metrics(y, probabilities)
        perturbations[name] = {"accuracy": perturb_metrics["accuracy"],
                               "macro_f1": perturb_metrics["macro_f1"],
                               "accuracy_drop_percentage_points": (base_accuracy - perturb_metrics["accuracy"]) * 100,
                               "prediction_flip_rate": float(np.mean(prediction != clean_prediction)),
                               "n": len(y)}

    result = {
        "status": "experimental_candidate_only",
        "model_version": "4.0.0-candidate",
        "classes": CLASSES,
        "checkpoint_sha256": __import__("hashlib").sha256((run / "best.keras").read_bytes()).hexdigest(),
        "evaluation_population": "one representative per dHash family in a 25% holdout from historical data/dataset_v2 validation; families crossing V4 development are excluded",
        "near_duplicate_filter": leakage_audit,
        "holdout_group_audit": holdout_audit,
        "independent_final_test": False,
        "metrics_v4_calibrated": v4_metrics,
        "selection_development_metrics_resubstitution": development_metrics,
        "calibration": {"temperature": temperature, "fitted_on_v4_dev_subset": True,
                        "abstention_target_selective_accuracy": 0.95},
        "same_holdout_comparison": comparison,
        "synthetic_lighting": {"base_accuracy": base_accuracy, "conditions": perturbations,
                               "worst_accuracy_drop_pp": max(x["accuracy_drop_percentage_points"] for x in perturbations.values())},
        "caveats": [
            "No new capture set exists. This holdout is drawn from a historical validation split, not an independent production test.",
            "The parent validation split has been used by earlier model work; results are development evidence only.",
            "No verified fruit/hand/session IDs are present, so near-duplicate and physical independence are unproven.",
            "Synthetic photometric transforms do not establish real illumination or ESP32-CAM robustness.",
            "Metrics are image-level; individual-fruit and hand-level labels are absent.",
        ],
    }
    output = args.output.resolve() if args.output else run / "evaluation_final.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    inference_metadata = {
        "model_version": "4.0.0-candidate",
        "model_name": "banana_ai_v4_remastered_candidate",
        "model_file": "best.keras",
        "framework": "tensorflow_keras",
        "classes": CLASSES,
        "input_size": [224, 224],
        "color_mode": "RGB",
        "resize_backend": "tensorflow_bilinear",
        "preprocessing": "keras MobileNetV3 include_preprocessing; RGB float32 [0,255]",
        "temperature": float(temperature),
        "uncertainty_threshold": threshold,
        "uncertainty_threshold_source": "95% selective accuracy target on V4 development subset; experimental, not independently calibrated",
        "days_remaining_available": False,
        "status": "experimental_candidate_only",
        "checkpoint_sha256": result["checkpoint_sha256"],
        "evaluation_report": output.name,
    }
    sidecar = args.metadata_output.resolve() if args.metadata_output else run / "inference_metadata_final.json"
    if sidecar.exists():
        existing = json.loads(sidecar.read_text(encoding="utf-8"))
        if existing != inference_metadata:
            raise FileExistsError(f"existing inference metadata differs; refusing to overwrite {sidecar}")
    else:
        sidecar.write_text(json.dumps(inference_metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "status": result["status"],
                      "accuracy": v4_metrics["accuracy"], "macro_f1": v4_metrics["macro_f1"],
                      "worst_synthetic_lighting_drop_pp": result["synthetic_lighting"]["worst_accuracy_drop_pp"]}, indent=2))


if __name__ == "__main__":
    main()
