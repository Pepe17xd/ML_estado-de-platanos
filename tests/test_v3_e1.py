from __future__ import annotations

import hashlib
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from training.v3_e1.augmentations import LightingAugmentation, ablation_config, default_augmentation_config
from training.v3_e1.data import as_model_inputs, directory_dataset
from training.v3_e1.evaluate import calibration_metrics, classification_metrics, image_dataset
from training.v3_e1.model import CLASSES, load_v2_checkpoint, set_finetune_trainable, set_head_trainable
from training.v3_e1.quality import inspect_image_tensor, validate_augmentation_pair
from training.v3_e1.selection import eligible_for_selection, selection_score, selection_values, validate_selection_config
from training.v3_e1.train import load_config, safe_new_run_dir
from training.v3_e1.robustness import CONDITIONS, perturb_batch, run as run_robustness
from inference.predict_v3 import load_experimental_model, predict_image
from inference.preprocessing import preprocess_encoded

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LightingAugmentationTests(unittest.TestCase):
    def test_pinned_tensorflow_version_and_experiment_config(self) -> None:
        self.assertTrue(tf.__version__.startswith("2.15."), tf.__version__)
        config = load_config(ROOT / "configs/v3_e1.json")
        self.assertEqual(config["classes"], CLASSES)
        self.assertEqual(config["input_size"], [224, 224])

    def test_default_combination_disables_e1_contrast_but_ablation_enables_it(self) -> None:
        config = default_augmentation_config()
        combined = ablation_config(config, "combined")
        contrast_only = ablation_config(config, "contrast")
        self.assertFalse(combined["photometric"]["contrast"]["enabled"])
        self.assertTrue(contrast_only["photometric"]["contrast"]["enabled"])
        self.assertEqual(contrast_only["photometric"]["contrast"]["probability"], 0.25)

    def test_selection_uses_macro_f1_and_rotten_recall_guardrail(self) -> None:
        truth = np.asarray([0, 1, 2, 3, 3, 3, 3, 3, 3])
        prediction = np.asarray([0, 1, 2, 3, 3, 3, 1, 3, 3])
        metrics = classification_metrics(truth, prediction)
        values = selection_values(metrics)
        policy = {
            "primary_metric": "macro_f1",
            "min_rotten_recall": 0.80,
            "rotten_as_maduro_alert_rate": 0.15,
            "tie_breakers": ["rotten_recall", "balanced_accuracy"],
        }
        self.assertEqual(values["rotten_as_maduro_count"], 1.0)
        self.assertAlmostEqual(values["rotten_as_maduro_rate"], 1 / 6)
        self.assertTrue(eligible_for_selection(metrics, policy))
        self.assertEqual(selection_score(metrics, policy)[0], metrics["macro_f1"])
        policy["min_rotten_recall"] = 0.90
        self.assertFalse(eligible_for_selection(metrics, policy))
        with self.assertRaises(ValueError):
            validate_selection_config({"primary_metric": "val_accuracy"})

    def test_shape_dtype_range_and_semantic_label_count(self) -> None:
        config = default_augmentation_config()
        config["max_photometric_ops"] = 6
        for item in config["photometric"].values():
            item["probability"] = 1.0
        layer = LightingAugmentation(config, seed=91)
        images = tf.fill([2, 224, 224, 3], 128.0)
        augmented = layer(images, training=True)
        self.assertEqual(tuple(augmented.shape), (2, 224, 224, 3))
        self.assertEqual(augmented.dtype, tf.float32)
        self.assertEqual(len(CLASSES), 4)
        self.assertTrue(bool(tf.reduce_all(augmented >= 0.0)))
        self.assertTrue(bool(tf.reduce_all(augmented <= 255.0)))
        validate_augmentation_pair(images, augmented, (224, 224))

    def test_validation_and_inference_are_identity(self) -> None:
        layer = LightingAugmentation(default_augmentation_config(), seed=17)
        images = tf.reshape(tf.range(24, dtype=tf.float32), [2, 2, 2, 3])
        np.testing.assert_array_equal(layer(images, training=False).numpy(), images.numpy())
        np.testing.assert_array_equal(layer(images, training=None).numpy(), images.numpy())

    def test_seed_repeats_augmentation(self) -> None:
        config = default_augmentation_config()
        config["max_photometric_ops"] = 6
        for item in config["photometric"].values():
            item["probability"] = 1.0
        images = tf.random.stateless_uniform([1, 32, 32, 3], [8, 9], maxval=255.0)
        tf.keras.utils.set_random_seed(1234)
        first = LightingAugmentation(config, seed=1234)(images, training=True).numpy()
        tf.keras.utils.set_random_seed(1234)
        second = LightingAugmentation(config, seed=1234)(images, training=True).numpy()
        np.testing.assert_array_equal(first, second)

    def test_serialization_round_trip(self) -> None:
        inputs = tf.keras.Input((16, 16, 3), name="image")
        aug = LightingAugmentation(default_augmentation_config(), seed=3, name="augmentation")
        x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(aug(inputs))
        outputs = tf.keras.layers.Dense(4, activation="softmax", name="state")(
            tf.keras.layers.GlobalAveragePooling2D()(x)
        )
        model = tf.keras.Model(inputs, outputs)
        values = tf.ones([1, 16, 16, 3], dtype=tf.float32) * 127.0
        expected = model(values, training=False).numpy()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "round_trip.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path, compile=False)
            actual = restored(values, training=False).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_invalid_inputs_are_reported(self) -> None:
        with self.assertRaises(TypeError):
            inspect_image_tensor(tf.zeros([8, 8, 3], dtype=tf.uint8))
        with self.assertRaises(ValueError):
            inspect_image_tensor(tf.zeros([8, 8], dtype=tf.float32))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            inspect_image_tensor(tf.fill([8, 8, 3], float("nan")))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            inspect_image_tensor(tf.fill([8, 8, 3], float("inf")))
        with self.assertRaises(ValueError):
            inspect_image_tensor(tf.ones([8, 8, 3], dtype=tf.float32) * 300.0)

    def test_nearly_black_and_saturated_images_are_flagged(self) -> None:
        black = inspect_image_tensor(tf.zeros([8, 8, 3], dtype=tf.float32))
        white = inspect_image_tensor(tf.ones([8, 8, 3], dtype=tf.float32) * 255.0)
        self.assertIn("practically_black", black["flags"])
        self.assertIn("practically_saturated", white["flags"])

    def test_validation_dataset_does_not_apply_photometric_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected_by_class = {}
            for index, name in enumerate(CLASSES):
                folder = root / name
                folder.mkdir()
                value = 40 + index * 30
                Image.new("RGB", (12, 12), color=(value, value, value)).save(folder / "sample.png")
                expected_by_class[name] = value
            layer_config = default_augmentation_config()
            layer_config["geometric"] = {"horizontal_flip": False, "rotation_factor": 0.0, "zoom_factor": 0.0}
            layer_config["max_photometric_ops"] = 6
            for item in layer_config["photometric"].values():
                item["probability"] = 1.0
            layer = LightingAugmentation(layer_config, seed=73)
            ds = directory_dataset(root, (224, 224), 4, training=False, seed=11, photometric_augmentation=layer)
            images, labels = next(iter(ds))
            label_values = labels.numpy().tolist()
            self.assertEqual(sorted(label_values), [0, 1, 2, 3])
            means = tf.reduce_mean(images, axis=[1, 2, 3]).numpy()
            expected = [expected_by_class[CLASSES[label]] for label in label_values]
            np.testing.assert_allclose(means, expected, atol=1.0)
            train_ds = directory_dataset(root, (224, 224), 4, training=True, seed=11, photometric_augmentation=layer)
            train_images, train_labels = next(iter(train_ds))
            self.assertEqual(tuple(train_images.shape), (4, 224, 224, 3))
            self.assertEqual(sorted(train_labels.numpy().tolist()), [0, 1, 2, 3])
            self.assertGreaterEqual(float(tf.reduce_min(train_images)), 0.0)
            self.assertLessEqual(float(tf.reduce_max(train_images)), 255.0)


class EvaluationMetricTests(unittest.TestCase):
    def test_evaluation_and_training_input_match_tensorflow_inference_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, class_name in enumerate(CLASSES):
                folder = root / class_name
                folder.mkdir()
                Image.new("RGB", (37 + index, 21), color=(31, 129, 247)).save(folder / "input.png")
            image_path = root / CLASSES[0] / "input.png"
            eval_rows = [{"_resolved_path": str(image_path), "_label": CLASSES[0]}]
            eval_images, eval_labels = next(iter(image_dataset(eval_rows, 1)))
            train_images, train_labels = next(iter(as_model_inputs(directory_dataset(
                root, (224, 224), 4, training=False, seed=11
            ))))
            verde_index = train_labels.numpy().tolist().index(0)
            expected = preprocess_encoded(image_path.read_bytes(), (224, 224), "tensorflow_bilinear")[0]
            np.testing.assert_array_equal(eval_images.numpy()[0], expected)
            np.testing.assert_array_equal(train_images.numpy()[verde_index], expected)
            self.assertEqual(eval_labels.numpy().tolist(), [0])
            self.assertIn(0, train_labels.numpy().tolist())

    def test_metric_contract_and_class_order(self) -> None:
        truth = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
        prediction = np.asarray([0, 1, 1, 1, 2, 3, 3, 3])
        metrics = classification_metrics(truth, prediction)
        self.assertEqual(metrics["n"], 8)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 1], [0, 0, 0, 2]])
        self.assertEqual(list(metrics["per_class"]), CLASSES)
        self.assertAlmostEqual(metrics["accuracy"], 6 / 8)

    def test_calibration_metrics_and_robustness_perturbations(self) -> None:
        truth = np.asarray([0, 1, 2, 3])
        probabilities = np.eye(4, dtype=np.float32) * 0.8 + 0.05
        calibration = calibration_metrics(probabilities, truth)
        self.assertEqual(set(calibration), {"expected_calibration_error_10_bins", "multiclass_brier_score"})
        self.assertAlmostEqual(calibration["expected_calibration_error_10_bins"], 0.15, places=6)

        images = tf.ones([2, 224, 224, 3], dtype=tf.float32) * 128.0
        original = images.numpy().copy()
        self.assertEqual(tuple(perturb_batch(images, "clean").shape), (2, 224, 224, 3))
        for condition in CONDITIONS[1:]:
            changed = perturb_batch(images, condition)
            self.assertEqual(tuple(changed.shape), (2, 224, 224, 3))
            self.assertGreaterEqual(float(tf.reduce_min(changed)), 0.0)
            self.assertLessEqual(float(tf.reduce_max(changed)), 255.0)
        np.testing.assert_array_equal(images.numpy(), original)

    def test_robustness_report_writes_json_and_markdown_to_new_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            for index, class_name in enumerate(CLASSES):
                image_path = root / f"{class_name}.png"
                Image.new("RGB", (16, 16), color=(40 + index * 30, 100, 150)).save(image_path)
                rows.append({"image_path": str(image_path.resolve()), "class_name": class_name})
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "class_name"])
                writer.writeheader()
                writer.writerows(rows)

            inputs = tf.keras.Input((224, 224, 3))
            outputs = tf.keras.layers.Dense(4, activation="softmax")(
                tf.keras.layers.GlobalAveragePooling2D()(inputs)
            )
            model_path = root / "tiny.keras"
            tf.keras.Model(inputs, outputs).save(model_path)
            (root / "run_metadata.json").write_text(
                json.dumps({"classes": CLASSES, "model_version": "tiny-test"}), encoding="utf-8"
            )
            report_path = root / "robustness.json"
            result = run_robustness(manifest, "custom", model_path, model_path, report_path, 4)
            self.assertEqual(result["n"], 4)
            self.assertEqual(len(result["models"]["v2"]["conditions"]), len(CONDITIONS))
            self.assertTrue(report_path.is_file())
            self.assertTrue(report_path.with_suffix(".md").is_file())
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["n"], 4)


class V3PredictionCliTests(unittest.TestCase):
    def test_explicit_checkpoint_metadata_and_jpeg_png_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = tf.keras.Input((224, 224, 3))
            outputs = tf.keras.layers.Dense(4, activation="softmax")(
                tf.keras.layers.GlobalAveragePooling2D()(inputs)
            )
            model_path = root / "candidate.keras"
            tf.keras.Model(inputs, outputs).save(model_path)
            metadata_path = root / "run_metadata.json"
            metadata_path.write_text(
                __import__("json").dumps({
                    "model_version": "3-test",
                    "classes": CLASSES,
                    "input_shape": [224, 224, 3],
                    "preprocessing": "saved_v2_embedded_mobilenet_v3_rescaling; input RGB float32 [0,255]",
                }),
                encoding="utf-8",
            )
            loaded, metadata = load_experimental_model(model_path, metadata_path)
            for extension, format_name in ((".jpg", "JPEG"), (".png", "PNG")):
                image_path = root / f"image{extension}"
                Image.new("RGB", (32, 24), color=(80, 130, 40)).save(image_path, format=format_name)
                result = predict_image(loaded, metadata, image_path)
                self.assertIn(result["estado"], CLASSES)
                self.assertGreaterEqual(result["score"], 0.0)
                self.assertLessEqual(result["score"], 1.0)

    def test_invalid_metadata_and_image_fail_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path = root / "metadata.json"
            metadata_path.write_text('{"classes": ["bad"]}', encoding="utf-8")
            model_path = root / "candidate.keras"
            model_path.write_bytes(b"placeholder")
            with self.assertRaises(ValueError):
                load_experimental_model(model_path, metadata_path)


class V2CompatibilityTests(unittest.TestCase):
    def test_saved_v2_graph_weights_and_output_are_preserved(self) -> None:
        production_paths = [
            ROOT / "models/production_model.keras",
            ROOT / "models/production_model_v2.keras",
            ROOT / "models/production_model_v2_metadata.json",
        ]
        production_h5 = ROOT / "models/production_model.h5"
        if production_h5.exists():
            production_paths.append(production_h5)
        hashes_before = [_sha256(path) for path in production_paths]
        v2, _ = load_v2_checkpoint(ROOT / "models/production_model_v2.keras", ROOT / "models/production_model_v2_metadata.json")
        self.assertEqual(tuple(v2.input_shape), (None, 224, 224, 3))
        self.assertEqual(tuple(v2.output_shape), (None, 4))
        self.assertEqual(v2.output_names, ["state"])
        x = tf.random.stateless_uniform([1, 224, 224, 3], [4, 5], maxval=255.0)
        prediction = v2(x, training=False).numpy()
        repeated_prediction = v2(x, training=False).numpy()
        self.assertEqual(prediction.shape, (1, 4))
        np.testing.assert_allclose(prediction.sum(axis=1), [1.0], rtol=1e-5)
        np.testing.assert_array_equal(prediction, repeated_prediction)
        self.assertAlmostEqual(float(v2.get_layer("augmentation").get_layer("random_contrast").factor), 0.15)
        self.assertFalse(ablation_config(default_augmentation_config(), "combined")["photometric"]["contrast"]["enabled"])
        set_finetune_trainable(v2, last_layers=30)
        self.assertTrue(all(not layer.trainable for layer in v2.layers if isinstance(layer, tf.keras.layers.BatchNormalization)))
        self.assertEqual(hashes_before, [_sha256(path) for path in production_paths])

    def test_head_and_finetune_phases_keep_batch_normalization_frozen(self) -> None:
        v2, _ = load_v2_checkpoint(ROOT / "models/production_model_v2.keras", ROOT / "models/production_model_v2_metadata.json")
        set_head_trainable(v2)
        self.assertTrue(v2.get_layer("dense").trainable)
        self.assertTrue(v2.get_layer("state").trainable)
        self.assertFalse(any(layer.trainable for layer in v2.layers if isinstance(layer, tf.keras.layers.BatchNormalization)))
        set_finetune_trainable(v2, last_layers=30)
        self.assertTrue(v2.get_layer("expanded_conv_10/expand").trainable)
        self.assertFalse(any(layer.trainable for layer in v2.layers if isinstance(layer, tf.keras.layers.BatchNormalization)))

    def test_training_output_directory_is_isolated(self) -> None:
        with self.assertRaises(ValueError):
            safe_new_run_dir(ROOT / "models" / "production_model.keras")


if __name__ == "__main__":
    unittest.main()
