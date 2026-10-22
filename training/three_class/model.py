"""MobileNetV3Small three-class candidate architecture."""
from __future__ import annotations

import tensorflow as tf

from training.three_class.data import CLASSES


def build_model(image_size: tuple[int, int] = (224, 224), weights: str | None = "imagenet") -> tf.keras.Model:
    """Build a fresh three-class transfer-learning model.

    Starting from ImageNet weights avoids reusing V2/V3 weights that may have
    seen the legacy banana images, which could contaminate a fresh grouped test.
    """
    inputs = tf.keras.Input((*image_size, 3), name="image")
    backbone = tf.keras.applications.MobileNetV3Small(
        include_top=False, weights=weights, input_tensor=inputs, include_preprocessing=True,
    )
    backbone.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(backbone.output)
    x = tf.keras.layers.Dropout(0.25, name="dropout")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense")(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="state")(x)
    return tf.keras.Model(inputs, outputs, name="banana_mobilenetv3small_three_class")


def set_finetune_trainable(model: tf.keras.Model, last_layers: int = 30) -> None:
    backbone_layers = [layer for layer in model.layers if layer.name not in {
        "global_average_pooling2d", "dropout", "dense", "state"
    }]
    trainable_ids = {id(layer) for layer in backbone_layers[-last_layers:]}
    for layer in model.layers:
        layer.trainable = layer.name in {"global_average_pooling2d", "dropout", "dense", "state"} or id(layer) in trainable_ids
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def compile_model(model: tf.keras.Model, learning_rate: float, weight_decay: float = 1e-5) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
