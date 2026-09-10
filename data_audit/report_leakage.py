"""Summarise exact and embedding-based evidence of train/test leakage."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def rows(path):
    return list(csv.DictReader(open(path,encoding="utf-8",newline=""))) if path.exists() else []
def main():
 p=argparse.ArgumentParser();p.add_argument('--reports-dir',default='reports/data_quality');a=p.parse_args();root=Path(a.reports_dir)
 exact=rows(root/'exact_duplicates.csv');similar=rows(root/'embedding_similar_pairs.csv')
 embedding_file = root / 'embedding_similar_pairs.csv'
 has_cross_split = any(r.get('cross_split') == 'True' for r in exact + similar)
 status = 'BLOCK_TRAINING' if has_cross_split else ('INCOMPLETE_AUDIT' if not embedding_file.exists() else 'NO_CROSS_SPLIT_EVIDENCE_FOUND')
 result={'status':status,'exact_duplicate_rows':len(exact),'exact_cross_split_rows':sum(r.get('cross_split')=='True' for r in exact),'embedding_pairs':len(similar),'embedding_cross_split_pairs':sum(r.get('cross_split')=='True' for r in similar),'notes':['Embedding scan was not run.' if not embedding_file.exists() else 'Embedding matches require manual review; do not delete automatically.','A clean report is not proof of no leakage: partition by real lot, camera session or source video.']}
 (root/'leakage_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
