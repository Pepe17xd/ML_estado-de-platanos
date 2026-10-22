"""Final grouped-holdout and paired synthetic-lighting evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance

from training.three_class.data import CLASSES, make_dataset, prepare_rows


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_pred = probabilities.argmax(axis=1)
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth), int(prediction)] += 1
    per_class = {}
    f1_values = []
    for index, name in enumerate(CLASSES):
        tp = matrix[index, index]
        precision = tp / max(1, int(matrix[:, index].sum()))
        recall = tp / max(1, int(matrix[index, :].sum()))
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[index, :].sum())}
        f1_values.append(f1)
    confidence = probabilities.max(axis=1)
    correct = y_pred == y_true
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        selected = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    high_conf = confidence >= 0.9
    return {
        "n": int(len(y_true)),
        "accuracy": float(correct.mean()) if len(correct) else None,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
        "ece_10_bins": float(ece),
        "brier_multiclass": float(np.mean(np.sum((probabilities - np.eye(len(CLASSES))[y_true]) ** 2, axis=1))),
        "high_confidence_threshold": 0.9,
        "high_confidence_n": int(high_conf.sum()),
        "high_confidence_error_rate": float((~correct[high_conf]).mean()) if high_conf.any() else None,
    }


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-8, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exponents = np.exp(logits)
    return exponents / exponents.sum(axis=1, keepdims=True)


def choose_temperature(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.geomspace(0.25, 4.0, 151)
    nlls = []
    for temperature in candidates:
        calibrated = temperature_scale(probabilities, float(temperature))
        nlls.append(float(-np.log(np.clip(calibrated[np.arange(len(y_true)), y_true], 1e-8, 1.0)).mean()))
    best = int(np.argmin(nlls))
    return float(candidates[best]), float(nlls[best])


def choose_abstention_threshold(y_true: np.ndarray, probabilities: np.ndarray,
                                target_selective_accuracy: float = 0.95) -> dict:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    candidates = sorted(set(float(x) for x in confidence), reverse=True)
    eligible = []
    for threshold in candidates:
        selected = confidence >= threshold
        if selected.any() and float((prediction[selected] == y_true[selected]).mean()) >= target_selective_accuracy:
            eligible.append((int(selected.sum()), threshold, float((prediction[selected] == y_true[selected]).mean())))
    if not eligible:
        return {"threshold": 1.0, "target_selective_accuracy": target_selective_accuracy,
                "validation_coverage": 0.0, "validation_selective_accuracy": None}
    coverage_count, threshold, selective_accuracy = max(eligible)
    return {"threshold": threshold, "target_selective_accuracy": target_selective_accuracy,
            "validation_coverage": coverage_count / len(y_true),
            "validation_selective_accuracy": selective_accuracy}


def photo_variants(path: Path) -> dict[str, Image.Image]:
    image = Image.open(path).convert("RGB")
    red, green, blue = image.split()
    warm = Image.merge("RGB", (red.point(lambda x: min(255, round(x * 1.06))),
                                green.point(lambda x: min(255, round(x * 1.00))),
                                blue.point(lambda x: min(255, round(x * 0.94)))))
    cool = Image.merge("RGB", (red.point(lambda x: min(255, round(x * 0.94))),
                                green.point(lambda x: min(255, round(x * 1.00))),
                                blue.point(lambda x: min(255, round(x * 1.06)))))
    shadow_array = np.asarray(image, dtype=np.float32).copy()
    yy, xx = np.mgrid[0:image.height, 0:image.width]
    cx, cy = image.width * 0.68, image.height * 0.48
    radius = max(1.0, min(image.width, image.height) * 0.38)
    mask = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius ** 2))
    shadow_array *= (1.0 - 0.35 * mask[..., None])
    shadow = Image.fromarray(np.clip(shadow_array, 0, 255).astype(np.uint8))
    highlight_array = np.asarray(image, dtype=np.float32).copy()
    highlight_array += (35.0 * mask[..., None])
    highlight = Image.fromarray(np.clip(highlight_array, 0, 255).astype(np.uint8))
    return {
        "clean": image,
        "dark_gamma_exponent_1_25": image.point(lambda x: round(255 * (x / 255) ** 1.25)),
        "gamma_exponent_0_75": image.point(lambda x: round(255 * (x / 255) ** 0.75)),
        "exposure_0_70": ImageEnhance.Brightness(image).enhance(0.70),
        "bright_exposure_1_2": ImageEnhance.Brightness(image).enhance(1.2),
        "warm_white_balance": warm,
        "cool_white_balance": cool,
        "low_contrast_0_85": ImageEnhance.Contrast(image).enhance(0.85),
        "high_contrast_1_20": ImageEnhance.Contrast(image).enhance(1.2),
        "low_saturation_0_85": ImageEnhance.Color(image).enhance(0.85),
        "soft_shadow": shadow,
        "uneven_highlight": highlight,
    }


def predict_images(model: tf.keras.Model, images: list[Image.Image]) -> np.ndarray:
    tensors = []
    for image in images:
        pixels = tf.convert_to_tensor(np.asarray(image.convert("RGB"), dtype=np.uint8))
        tensors.append(tf.image.resize(pixels, (224, 224), method="bilinear", antialias=False).numpy())
    return model.predict(np.asarray(tensors, dtype=np.float32), verbose=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_v1/manifest.csv"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    metadata_path = run_dir / "run_metadata.json"
    model_path = run_dir / "best.keras"
    if not metadata_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"expected candidate and metadata in {run_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("classes") != CLASSES or metadata.get("status") != "candidate_trained":
        raise ValueError("candidate metadata does not match the three-class evaluation contract")
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if manifest_hash != metadata.get("manifest_sha256"):
        raise ValueError("evaluation manifest differs from the manifest used for training")
    partitions, audit = prepare_rows(args.manifest, Path.cwd(), seed=int(metadata["seed"]))
    test_rows = partitions["test"]
    test_ds = make_dataset(test_rows, (224, 224), args.batch_size, False, int(metadata["seed"]))
    model = tf.keras.models.load_model(model_path, compile=False)
    all_probs, all_truth = [], []
    for images, labels in test_ds:
        all_probs.extend(model.predict(images, verbose=0).tolist())
        all_truth.extend(labels.numpy().tolist())
    val_ds = make_dataset(partitions["validation"], (224, 224), args.batch_size, False, int(metadata["seed"]))
    val_probs, val_truth = [], []
    for images, labels in val_ds:
        val_probs.extend(model.predict(images, verbose=0).tolist())
        val_truth.extend(labels.numpy().tolist())
    val_probs = np.asarray(val_probs)
    val_truth = np.asarray(val_truth)
    test_probs = np.asarray(all_probs)
    test_truth = np.asarray(all_truth)
    temperature, validation_nll = choose_temperature(val_truth, val_probs)
    abstention = choose_abstention_threshold(val_truth, temperature_scale(val_probs, temperature))
    base_metrics = metrics(test_truth, test_probs)
    calibrated_metrics = metrics(test_truth, temperature_scale(test_probs, temperature))

    transformed_probabilities: dict[str, list[list[float]]] = {}
    condition_truths: dict[str, list[int]] = {}
    for row in test_rows:
        variants = photo_variants(Path(row["_path"]))
        names = list(variants)
        probs = predict_images(model, list(variants.values()))
        true_index = CLASSES.index(row["class_name"])
        for name, probability in zip(names, probs):
            transformed_probabilities.setdefault(name, []).append(probability.tolist())
            condition_truths.setdefault(name, []).append(true_index)
    clean_preds = np.asarray(transformed_probabilities["clean"]).argmax(axis=1)
    robustness: dict[str, dict] = {}
    for name, probabilities in transformed_probabilities.items():
        probabilities_array = np.asarray(probabilities)
        prediction_array = probabilities_array.argmax(axis=1)
        condition_metrics = metrics(np.asarray(condition_truths[name]), probabilities_array)
        robustness[name] = {
            "accuracy": condition_metrics["accuracy"],
            "macro_f1": condition_metrics["macro_f1"],
            "prediction_flip_rate_vs_clean": float(np.mean(prediction_array != clean_preds)),
            "accuracy_drop_percentage_points_vs_clean": float((base_metrics["accuracy"] - condition_metrics["accuracy"]) * 100),
        }
    result = {
        "status": "experimental_candidate_only",
        "model": "MobileNetV3Small",
        "classes": CLASSES,
        "test_split_audit": audit,
        "test_metrics": base_metrics,
        "calibration": {
            "temperature_fitted_on_validation": temperature,
            "validation_nll_after_temperature_scaling": validation_nll,
            "test_metrics_before_temperature_scaling": base_metrics,
            "test_metrics_after_temperature_scaling": calibrated_metrics,
            "abstention_threshold_selected_on_validation": abstention,
            "test_selective_metrics_at_threshold": {
                "threshold": abstention["threshold"],
                "coverage": float((temperature_scale(test_probs, temperature).max(axis=1) >= abstention["threshold"]).mean()),
                "selective_accuracy": (
                    float((test_probs[temperature_scale(test_probs, temperature).max(axis=1) >= abstention["threshold"]].argmax(axis=1)
                          == test_truth[temperature_scale(test_probs, temperature).max(axis=1) >= abstention["threshold"]]).mean())
                    if np.any(temperature_scale(test_probs, temperature).max(axis=1) >= abstention["threshold"]) else None
                ),
            },
        },
        "synthetic_lighting": robustness,
        "caveats": [
            "This holdout was made from legacy train rows to avoid legacy test reuse; physical fruit independence is unverified.",
            "No boxes or tags distinguish individual bananas from hands, so metrics are image-level only.",
            "Photometric changes are synthetic and do not establish real-camera invariance.",
            "All photo perturbations use fixed synthetic settings and are not calibrated to real cameras.",
        ],
    }
    destination = args.output or run_dir / "evaluation.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {destination}")
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Export a candidate sidecar for an explicitly configured experimental API.
    # This does not alter the production model or its metadata.
    inference_metadata_path = run_dir / "inference_metadata.json"
    if inference_metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite {inference_metadata_path}")
    inference_metadata = {
        "model_version": "3.0.0-three-class-candidate",
        "model_name": "banana_mobilenetv3small_three_class_candidate",
        "model_file": "best.keras",
        "classes": CLASSES,
        "input_size": [224, 224],
        "color_mode": "RGB",
        "resize_backend": "tensorflow_bilinear",
        "preprocessing": "keras MobileNetV3 include_preprocessing; RGB float [0,255]",
        "temperature": temperature,
        "uncertainty_threshold": abstention["threshold"],
        "uncertainty_threshold_source": "validation selective-accuracy target; experimental, small validation set",
        "days_remaining_available": False,
        "status": "experimental_candidate_only",
    }
    inference_metadata_path.write_text(json.dumps(inference_metadata, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
