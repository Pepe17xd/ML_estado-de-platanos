"""Find visually similar images with ImageNet MobileNetV3 embeddings.

Run after installing TensorFlow. Review pairs manually: similarity is evidence, not proof.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from data_audit.common import image_records, write_csv

def load_image(path, size):
    raw = tf.io.read_file(path); image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    return tf.image.resize(tf.cast(image, tf.float32), (size, size))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset-dir", default="dataset"); p.add_argument("--output-dir", default="reports/data_quality"); p.add_argument("--threshold", type=float, default=.985); p.add_argument("--batch-size", type=int, default=64); args = p.parse_args()
    records = list(image_records(Path(args.dataset_dir))); paths = [r["path"] for r in records]
    ds = tf.data.Dataset.from_tensor_slices(paths).map(lambda x: load_image(x, 224), num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    base = tf.keras.applications.MobileNetV3Small(include_top=False, pooling="avg", weights="imagenet")
    embeddings = base.predict(ds, verbose=1); embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
    scores = embeddings @ embeddings.T; np.fill_diagonal(scores, -1)
    left, right = np.where(np.triu(scores >= args.threshold, 1))
    rows = []
    for a, b in zip(left, right):
        rows.append({"path_a": paths[a], "split_a": records[a]["original_split"], "state_a": records[a]["state"], "path_b": paths[b], "split_b": records[b]["original_split"], "state_b": records[b]["state"], "cosine_similarity": round(float(scores[a,b]), 6), "cross_split": records[a]["original_split"] != records[b]["original_split"]})
    out = Path(args.output_dir); write_csv(out / "embedding_similar_pairs.csv", rows, list(rows[0]) if rows else ["path_a","split_a","state_a","path_b","split_b","state_b","cosine_similarity","cross_split"])
    np.savez_compressed(out / "embeddings.npz", paths=np.array(paths), embeddings=embeddings)
    print(f"Scanned {len(records)} images; found {len(rows)} pairs >= {args.threshold}.")
if __name__ == "__main__": main()
