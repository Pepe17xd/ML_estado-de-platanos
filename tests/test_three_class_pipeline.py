from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from training.three_class.data import CLASSES, prepare_rows
from training.three_class.evaluate import metrics
from training.three_class.model import build_model
from inference.image_quality import quality_flags


def test_three_class_label_policy_excludes_rotten_conflicts_and_keeps_groups(tmp_path: Path) -> None:
    rows = []
    for offset, label in enumerate(("unripe", "ripe", "overripe")):
        for index in range(8):
            path = tmp_path / f"{label}_{index}.png"
            Image.new("RGB", (20, 15), (index + 20 + offset * 70, 60, 90)).save(path)
            rows.append({"image_path": str(path), "split": "train", "original_state": label,
                         "class_name": "pasado" if label in {"overripe", "rotten"} else {"unripe": "verde", "ripe": "maduro"}[label],
                         "group_id": f"{label}-{index}"})
    rotten = tmp_path / "rotten.png"
    Image.new("RGB", (20, 15), (140, 40, 30)).save(rotten)
    rows.append({"image_path": str(rotten), "split": "train", "original_state": "rotten", "class_name": "pasado", "group_id": "rotten-1"})
    # Conflicting visual family: neither member should enter a target class.
    rows.extend([
        {"image_path": str(tmp_path / "unripe_0.png"), "split": "train", "original_state": "unripe", "class_name": "verde", "group_id": "ambiguous"},
        {"image_path": str(rotten), "split": "train", "original_state": "rotten", "class_name": "pasado", "group_id": "ambiguous"},
    ])
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    splits, summary = prepare_rows(manifest, tmp_path, seed=7)
    assert CLASSES == ["verde", "maduro", "sobre_maduro"]
    assert summary["excluded_rotten_rows"] == 2
    assert summary["excluded_conflicting_visual_groups"] == ["ambiguous"]
    seen_groups: dict[str, set[str]] = defaultdict(set)
    for split, members in splits.items():
        for row in members:
            assert row["class_name"] in CLASSES
            seen_groups[row["group_id"]].add(split)
    assert all(len(split_names) == 1 for split_names in seen_groups.values())
    assert "ambiguous" not in seen_groups


def test_three_class_model_contract() -> None:
    model = build_model(weights=None)
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 3)
    assert model.output_names == ["state"]


def test_evaluation_metrics_report_three_classes_and_calibration() -> None:
    y_true = np.asarray([0, 0, 1, 1, 2, 2])
    probabilities = np.eye(3, dtype=np.float32)[y_true] * 0.9 + 0.1 / 3
    result = metrics(y_true, probabilities)
    assert result["n"] == 6
    assert result["accuracy"] == 1.0
    assert set(result["per_class"]) == set(CLASSES)
    assert result["confusion_matrix_rows_true_columns_predicted"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert "ece_10_bins" in result and "high_confidence_error_rate" in result


def test_quality_gate_flags_extreme_dark_and_overexposure_inputs() -> None:
    dark = Image.new("RGB", (64, 64), (0, 0, 0))
    bright = Image.new("RGB", (64, 64), (255, 255, 255))
    assert "imagen_demasiado_oscura" in quality_flags(dark)
    assert "imagen_sobreexpuesta" in quality_flags(bright)


def test_quality_gate_ignores_white_background_but_checks_chromatic_subject() -> None:
    # Saturated green/yellow and muted overripe-like foregrounds on varied
    # bright, dark, green, and magenta backgrounds.
    for background in ((255, 255, 255), (8, 8, 8), (60, 130, 35), (170, 80, 140)):
        for color in ((45, 150, 35), (225, 190, 20), (155, 105, 75)):
            image = Image.new("RGB", (120, 90), background)
            image.paste(Image.new("RGB", (50, 60), color), (35, 15))
            assert "imagen_sobreexpuesta" not in quality_flags(image)

    # A genuinely clipped colored subject remains a quality failure.
    extreme = Image.new("RGB", (120, 90), (255, 255, 255))
    extreme.paste(Image.new("RGB", (60, 65), (255, 255, 190)), (30, 12))
    assert "imagen_sobreexpuesta" in quality_flags(extreme)


def test_banana9_has_no_false_overexposure_flag() -> None:
    image = Image.open(Path("test_images/banana9.jpg"))
    assert "imagen_sobreexpuesta" not in quality_flags(image)


def test_three_class_training_config_has_exact_contract_and_photo_controls() -> None:
    config = json.loads(Path("configs/three_class.json").read_text(encoding="utf-8"))
    assert config["classes"] == CLASSES
    assert config["source_label_mapping"]["rotten"] is None
    assert config["augmentation"]["photometric"]["contrast"]["enabled"] is False
    assert config["augmentation"]["photometric"]["white_balance"]["enabled"] is True
    assert config["augmentation"]["geometric"]["rotation_factor"] == 0.0
