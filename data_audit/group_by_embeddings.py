"""Create visual-family group IDs from precomputed normalized embeddings.

Connected components are used intentionally: any similarity edge at or above the
threshold joins a family, preventing indirect A-B-C leakage across splits.
"""
from __future__ import annotations
import argparse, csv, json, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from data_audit.common import image_records, write_csv

class UnionFind:
    def __init__(self, n): self.parent=list(range(n)); self.size=[1]*n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x]=self.parent[self.parent[x]]; x=self.parent[x]
        return x
    def union(self, a, b):
        a,b=self.find(a),self.find(b)
        if a==b:return
        if self.size[a]<self.size[b]:a,b=b,a
        self.parent[b]=a;self.size[a]+=self.size[b]

def assign_splits(groups, seed):
    """Greedy, deterministic group-level 70/15/15 class balancing."""
    labels=sorted({label for rows in groups.values() for label in rows['labels']})
    total=Counter(label for rows in groups.values() for label in rows['labels'])
    ratios={"train":.70,"validation":.15,"test":.15}
    targets={split:{label: ratio*total[label] for label in labels} for split,ratio in ratios.items()}
    total_targets={split:ratio*sum(total.values()) for split,ratio in ratios.items()}
    current={split:Counter() for split in targets}; assignment={}
    ids=list(groups); random.Random(seed).shuffle(ids)
    # Large/mixed groups first makes the allocation more stable.
    ids.sort(key=lambda gid: len(groups[gid]['labels']), reverse=True)
    for gid in ids:
        counts=Counter(groups[gid]['labels'])
        def error(split, proposed):
            class_error=sum(((proposed[label]-targets[split][label]) / max(targets[split][label],1))**2 for label in labels)
            total_error=((sum(proposed.values())-total_targets[split]) / max(total_targets[split],1))**2
            return class_error + .25 * total_error
        # Compare the *increase* in error. Absolute candidate error incorrectly
        # favors an already overloaded split when a large group is processed.
        def cost(split):
            proposed=current[split].copy(); proposed.update(counts)
            return error(split, proposed)-error(split, current[split])
        split=min(targets,key=cost);assignment[gid]=split;current[split].update(counts)
    return assignment, current

def main():
    p=argparse.ArgumentParser();p.add_argument('--embeddings',default='reports/data_quality/embeddings.npz');p.add_argument('--dataset-dir',default='dataset');p.add_argument('--threshold',type=float,default=.985);p.add_argument('--output-metadata',default='data/metadata_grouped.csv');p.add_argument('--output-dir',default='data/splits_grouped');p.add_argument('--report',default='reports/data_quality/group_split_leakage_report.json');p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    archive=np.load(a.embeddings);paths=[str(x) for x in archive['paths']];vectors=archive['embeddings'];records={r['path']:r for r in image_records(Path(a.dataset_dir))}
    if set(paths) != set(records): raise ValueError('Embedding paths do not match the current dataset. Recompute embeddings.')
    uf=UnionFind(len(paths));similarities=vectors @ vectors.T
    left,right=np.where(np.triu(similarities >= a.threshold,1))
    for i,j in zip(left,right):uf.union(int(i),int(j))
    components=defaultdict(list)
    for i,path in enumerate(paths):components[uf.find(i)].append(path)
    # Stable IDs based on the lexicographically first path in each component.
    ordered=sorted(components.values(),key=lambda xs: min(xs));path_to_group={path:f'VF-{number:05d}' for number,paths_in_group in enumerate(ordered,1) for path in paths_in_group}
    groups=defaultdict(lambda:{'paths':[],'labels':[],'legacy_splits':[]})
    for path in paths:
        rec=records[path];gid=path_to_group[path];groups[gid]['paths'].append(path);groups[gid]['labels'].append(rec['state']);groups[gid]['legacy_splits'].append(rec['original_split'])
    assignment, distribution=assign_splits(groups,a.seed)
    metadata=[];manifest=[]
    for path in sorted(paths):
        rec=records[path];gid=path_to_group[path];new_split=assignment[gid]
        metadata.append({'image_path':path,'state_label':rec['state'],'legacy_split':rec['original_split'],'group_id':gid,'group_size':len(groups[gid]['paths']),'new_split':new_split})
        manifest.append({'image_path':path,'state_label':rec['state'],'group_id':gid,'split':new_split})
    write_csv(Path(a.output_metadata),metadata,['image_path','state_label','legacy_split','group_id','group_size','new_split'])
    out=Path(a.output_dir);write_csv(out/'split_manifest.csv',manifest,['image_path','state_label','group_id','split'])
    before_cross=[gid for gid,g in groups.items() if len(set(g['legacy_splits']))>1]
    after_cross=[gid for gid in groups if len({assignment[gid]})>1]  # Invariant; should always be empty.
    report={'threshold':a.threshold,'images':len(paths),'similarity_edges':int(len(left)),'visual_families':len(groups),'multi_image_families':sum(len(g['paths'])>1 for g in groups.values()),'largest_family':max(len(g['paths']) for g in groups.values()),'before':{'groups_crossing_legacy_splits':len(before_cross),'images_in_crossing_groups':sum(len(groups[g]['paths']) for g in before_cross)},'after':{'groups_crossing_new_splits':len(after_cross),'images_in_crossing_groups':sum(len(groups[g]['paths']) for g in after_cross),'invariant_passed':not after_cross},'new_split_distribution':{split:dict(sorted(counts.items())) for split,counts in distribution.items()},'status':'GROUP_LEAKAGE_FIXED_REQUIRES_MANUAL_REVIEW' if not after_cross else 'BLOCK_TRAINING'}
    report_path=Path(a.report);report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
