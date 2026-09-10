"""Find byte-identical images, including duplicates across existing splits."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from data_audit.common import image_records, sha256, write_csv

def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset-dir", default="dataset"); p.add_argument("--output-dir", default="reports/data_quality"); args = p.parse_args()
    groups = defaultdict(list)
    for record in image_records(Path(args.dataset_dir)):
        record["sha256"] = sha256(record["path"]); groups[record["sha256"]].append(record)
    duplicates = [g for g in groups.values() if len(g) > 1]
    rows = []
    for cluster_id, group in enumerate(duplicates, 1):
        leakage = len({x["original_split"] for x in group}) > 1
        for record in group:
            rows.append({**record, "cluster_id": cluster_id, "cross_split": leakage})
    out = Path(args.output_dir); write_csv(out / "exact_duplicates.csv", rows, ["cluster_id", "path", "original_split", "state", "sha256", "cross_split"])
    summary = {"images_scanned": sum(len(v) for v in groups.values()), "duplicate_clusters": len(duplicates), "duplicate_images": len(rows), "cross_split_duplicate_clusters": sum(len({x['original_split'] for x in g}) > 1 for g in duplicates)}
    (out / "exact_duplicates_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
if __name__ == "__main__": main()
