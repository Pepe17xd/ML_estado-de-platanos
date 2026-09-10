from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import tensorflow as tf
import yaml

def load_config(path):
    with open(path, encoding="utf-8") as f: return yaml.safe_load(f)
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
def split_train_validation(directory, cfg):
    common = dict(labels="inferred", class_names=cfg["states"], label_mode="int", validation_split=cfg["validation_split"], seed=cfg["seed"], image_size=(cfg["image_size"], cfg["image_size"]), batch_size=cfg["batch_size"])
    return (tf.keras.utils.image_dataset_from_directory(directory, subset="training", shuffle=True, **common), tf.keras.utils.image_dataset_from_directory(directory, subset="validation", shuffle=False, **common))
def add_targets(ds, cfg):
    t = cfg["weak_targets"]
    maturity = tf.constant([t[s]["maturity_pct"] / 100 for s in cfg["states"]], tf.float32)
    days = tf.constant([t[s]["days_remaining"] for s in cfg["states"]], tf.float32)
    def convert(images, labels):
        labels = tf.cast(labels, tf.int32)
        return images, {"state": tf.one_hot(labels, len(cfg["states"])), "maturity": tf.gather(maturity, labels)[:, None], "days_remaining": tf.gather(days, labels)[:, None]}
    return ds.map(convert, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
def save_metadata(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: cfg[k] for k in ("states", "image_size", "backbone", "weak_targets")}, indent=2), encoding="utf-8")
