"""Create train/validation/test manifests without moving images.

Requires a real lot_id column. It refuses simulated IDs by default.
"""
from __future__ import annotations
import argparse, csv, random
from collections import Counter, defaultdict
from pathlib import Path
from data_audit.common import write_csv

def main():
    p=argparse.ArgumentParser();p.add_argument("--metadata",default="data/metadata.csv");p.add_argument("--output-dir",default="data/splits");p.add_argument("--seed",type=int,default=42);p.add_argument("--allow-simulated",action="store_true");a=p.parse_args()
    with open(a.metadata,encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    required={"image_path","state_label"}; lot_column="lot_id" if rows and "lot_id" in rows[0] else "lot_id_simulated"
    if not rows or not required.issubset(rows[0]) or lot_column not in rows[0]: raise ValueError("metadata needs image_path, state_label and real lot_id")
    if lot_column == "lot_id_simulated" and not a.allow_simulated: raise ValueError("Refusing simulated lots. Replace with real lot_id, or use --allow-simulated for a dry run.")
    groups=defaultdict(list)
    for row in rows: groups[row[lot_column]].append(row)
    rng=random.Random(a.seed); lots=list(groups);rng.shuffle(lots)
    # Greedy group assignment targets 70/15/15 while keeping every lot intact.
    targets={"train":.70*len(rows),"validation":.15*len(rows),"test":.15*len(rows)}; assigned=Counter(); manifests=[]
    for lot in lots:
        split=min(targets,key=lambda s: assigned[s]/max(targets[s],1))
        for row in groups[lot]: manifests.append({"image_path":row["image_path"],"state_label":row["state_label"],"lot_id":lot,"split":split})
        assigned[split]+=len(groups[lot])
    out=Path(a.output_dir);write_csv(out/"split_manifest.csv",manifests,["image_path","state_label","lot_id","split"])
    print(dict(assigned));print(Counter((r['split'],r['state_label']) for r in manifests))
if __name__=="__main__":main()
