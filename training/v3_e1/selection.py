"""Validation-only checkpoint criteria and reproducible metric recording."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from training.v3_e1.evaluate import classification_metrics
from training.v3_e1.model import CLASSES


def validate_selection_config(config: dict[str, Any]) -> None:
    primary = config.get("primary_metric", "macro_f1")
    if primary not in {"macro_f1", "accuracy", "balanced_accuracy"}:
        raise ValueError("selection.primary_metric must be macro_f1, accuracy, or balanced_accuracy")
    minimum = float(config.get("min_rotten_recall", 0.0))
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("selection.min_rotten_recall must be in [0, 1]")
    alert = float(config.get("rotten_as_maduro_alert_rate", 1.0))
    if not 0.0 <= alert <= 1.0:
        raise ValueError("selection.rotten_as_maduro_alert_rate must be in [0, 1]")
    ties = config.get("tie_breakers", ["rotten_recall", "balanced_accuracy"])
    allowed = {"macro_f1", "accuracy", "balanced_accuracy", "rotten_recall"}
    if not isinstance(ties, list) or any(value not in allowed for value in ties):
        raise ValueError(f"selection.tie_breakers must be a list drawn from {sorted(allowed)}")


def selection_values(metrics: dict[str, Any]) -> dict[str, float | int]:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    rotten_support = int(matrix[3].sum())
    rotten_as_maduro = int(matrix[3, 1])
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "rotten_recall": float(metrics["per_class"]["podrido"]["recall"]),
        "rotten_as_maduro_count": rotten_as_maduro,
        "rotten_as_maduro_rate": float(rotten_as_maduro / rotten_support) if rotten_support else 0.0,
    }


def eligible_for_selection(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    validate_selection_config(config)
    values = selection_values(metrics)
    return values["rotten_recall"] >= float(config.get("min_rotten_recall", 0.0))


def selection_score(metrics: dict[str, Any], config: dict[str, Any]) -> tuple[float, ...]:
    validate_selection_config(config)
    values = selection_values(metrics)
    keys = [config.get("primary_metric", "macro_f1"), *config.get("tie_breakers", ["rotten_recall", "balanced_accuracy"])]
    return tuple(values[key] for key in keys)


class ValidationCheckpointSelector(tf.keras.callbacks.Callback):
    """Compute validation metrics each epoch and save eligible macro-F1 winners."""

    def __init__(
        self,
        validation_data: tf.data.Dataset,
        run_dir: Path,
        phase: str,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        validate_selection_config(config)
        self.validation_data = validation_data
        self.run_dir = Path(run_dir)
        self.phase = phase
        self.config = dict(config)
        self.best_checkpoint: Path | None = None
        self.best_score: tuple[float, ...] | None = None
        self.best_metrics: dict[str, Any] | None = None
        self.epoch_metrics: list[dict[str, Any]] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        truths: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        for images, labels in self.validation_data:
            output = self.model(images, training=False)
            if isinstance(output, dict):
                output = output.get("state")
            elif isinstance(output, (tuple, list)):
                output = output[0]
            probabilities = np.asarray(output)
            if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASSES):
                raise ValueError(f"validation output has incompatible shape {probabilities.shape}")
            truths.append(np.asarray(labels).reshape(-1).astype(np.int64))
            predictions.append(probabilities.argmax(axis=1).astype(np.int64))
        if not truths:
            raise ValueError("validation dataset produced no batches")

        metrics = classification_metrics(np.concatenate(truths), np.concatenate(predictions))
        values = selection_values(metrics)
        metrics.update(values)
        metrics["eligible_for_checkpoint"] = eligible_for_selection(metrics, self.config)
        alert_rate = float(self.config.get("rotten_as_maduro_alert_rate", 1.0))
        metrics["rotten_as_maduro_alert"] = values["rotten_as_maduro_rate"] > alert_rate
        self.epoch_metrics.append({"epoch": int(epoch + 1), **metrics})

        if logs is not None:
            logs["val_macro_f1"] = metrics["macro_f1"]
            logs["val_balanced_accuracy"] = metrics["balanced_accuracy"]
            logs["val_rotten_recall"] = metrics["rotten_recall"]
            logs["val_rotten_as_maduro_rate"] = metrics["rotten_as_maduro_rate"]

        if not metrics["eligible_for_checkpoint"]:
            return
        score = selection_score(metrics, self.config)
        if self.best_score is None or score > self.best_score:
            checkpoint = self.run_dir / f"{self.phase}_best.keras"
            self.model.save(checkpoint, overwrite=True)
            self.best_checkpoint = checkpoint
            self.best_score = score
            self.best_metrics = metrics
