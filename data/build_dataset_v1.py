"""Build the traceable three-class dataset-v1 manifest; source images untouched."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
from data_audit.common import write_csv

MAP={'unripe':'verde','ripe':'maduro','overripe':'pasado','rotten':'pasado'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default='data/splits_grouped/split_manifest.csv');p.add_argument('--metadata',default='data/metadata_grouped.csv');p.add_argument('--output',default='data/dataset_v1/manifest.csv');a=p.parse_args()
 with open(a.manifest,encoding='utf-8',newline='') as f: split_rows=list(csv.DictReader(f))
 with open(a.metadata,encoding='utf-8',newline='') as f: metadata={r['image_path']:r for r in csv.DictReader(f)}
 rows=[]
 for r in split_rows:
  m=metadata.get(r['image_path'])
  if not m or m['group_id']!=r['group_id']: raise ValueError(f'Metadata mismatch: {r["image_path"]}')
  rows.append({'image_path':r['image_path'],'split':r['split'],'class_name':MAP[r['state_label']],'original_state':r['state_label'],'group_id':r['group_id']})
 out=Path(a.output);write_csv(out,rows,['image_path','split','class_name','original_state','group_id']);print(f'Wrote dataset v1: {len(rows)} rows -> {out}')
if __name__=='__main__':main()
