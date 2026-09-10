"""Create an explicitly synthetic schema/template for shelf-life data collection."""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
from data_audit.common import image_records, write_csv

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset-dir",default="dataset"); p.add_argument("--output",default="data/metadata_simulated.csv"); args=p.parse_args()
    rows=[]
    defaults={"unripe":(15,6),"ripe":(60,3),"overripe":(85,1),"rotten":(100,0)}
    for r in image_records(Path(args.dataset_dir)):
        # Deterministic pseudo-lot is only a template. Replace it with physical receiving lot/camera session IDs.
        token=hashlib.sha1(Path(r['path']).stem.encode()).hexdigest()[:2]
        maturity, days=defaults[r['state']]
        rows.append({"image_path":r['path'],"split_legacy":r['original_split'],"state_label":r['state'],"lot_id_simulated":f"SIM-{r['state']}-{token}","capture_timestamp_utc":"", "cultivar":"unknown", "storage_temperature_c":"", "relative_humidity_pct":"", "maturity_pct_weak":maturity, "days_remaining_weak":days, "days_remaining_measured":"", "label_source":"SIMULATED_WEAK_LABEL_DO_NOT_TRAIN_PRODUCTION"})
    fields=["image_path","split_legacy","state_label","lot_id_simulated","capture_timestamp_utc","cultivar","storage_temperature_c","relative_humidity_pct","maturity_pct_weak","days_remaining_weak","days_remaining_measured","label_source"]
    write_csv(Path(args.output),rows,fields); print(f"Wrote {len(rows)} simulated rows to {args.output}")
if __name__=="__main__": main()
