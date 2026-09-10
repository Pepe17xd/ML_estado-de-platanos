"""Analyze one visual family: approximate visual trajectory and label conflicts."""
from __future__ import annotations
import argparse, base64, csv, html, io, json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from data_audit.common import write_csv

def thumbnail(path):
    with Image.open(path) as im:
        im=im.convert('RGB');im.thumbnail((220,165));b=io.BytesIO();im.save(b,'JPEG',quality=78)
    return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--group-id',default='VF-00004');p.add_argument('--metadata',default='data/metadata_grouped.csv');p.add_argument('--embeddings',default='reports/data_quality/embeddings.npz');p.add_argument('--output-dir',default='reports/data_quality');p.add_argument('--threshold',type=float,default=.985);a=p.parse_args()
    with open(a.metadata,encoding='utf-8',newline='') as f: rows=[r for r in csv.DictReader(f) if r['group_id']==a.group_id]
    if not rows: raise ValueError(f'Group not found: {a.group_id}')
    archive=np.load(a.embeddings);vec={str(path):v for path,v in zip(archive['paths'],archive['embeddings'])}; paths=[r['image_path'] for r in rows]; X=np.stack([vec[x] for x in paths]); X=X/(np.linalg.norm(X,axis=1,keepdims=True).clip(1e-12))
    # PCA axis is an ordering of visual appearance, not a date claim.
    centered=X-X.mean(0); _,_,vh=np.linalg.svd(centered,full_matrices=False); axis=centered@vh[0]; order=np.argsort(axis)
    ordered=[rows[i] for i in order]; projections=[float(axis[i]) for i in order]
    sim=X@X.T; np.fill_diagonal(sim,-1); edge_i,edge_j=np.where(np.triu(sim>=a.threshold,1)); contradictory=[(int(i),int(j),float(sim[i,j])) for i,j in zip(edge_i,edge_j) if rows[i]['state_label']!=rows[j]['state_label']]
    labels=Counter(r['state_label'] for r in rows); split=Counter(r['new_split'] for r in rows)
    by_label=Counter(r['state_label'] for r in ordered); transitions=sum(ordered[i]['state_label']!=ordered[i-1]['state_label'] for i in range(1,len(ordered)))
    runs=[]
    for row in ordered:
        if not runs or runs[-1]['class'] != row['state_label']: runs.append({'class':row['state_label'],'count':1})
        else: runs[-1]['count'] += 1
    report={'group_id':a.group_id,'images':len(rows),'threshold':a.threshold,'label_distribution':dict(labels),'split_distribution':dict(split),'pca_order_note':'Approximate visual ordering only; PCA direction has no guaranteed chronological meaning.','similarity_edges':int(len(edge_i)),'high_similarity_cross_label_edges':len(contradictory),'possible_label_contradictions':len(contradictory),'visual_order_label_transitions':transitions,'visual_order_label_runs':runs,'ordered_images':[{'order':i+1,'path':r['image_path'],'class':r['state_label'],'legacy_split':r['legacy_split'],'new_split':r['new_split'],'visual_axis':round(projections[i],6)} for i,r in enumerate(ordered)]}
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);(out/f'{a.group_id}_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    write_csv(out/f'{a.group_id}_timeline.csv',report['ordered_images'],['order','path','class','legacy_split','new_split','visual_axis'])
    cards=[]
    for item in report['ordered_images']:
        cards.append(f'<figure><img src="{thumbnail(item["path"])}"><figcaption>#{item["order"]} · {html.escape(item["class"])}<br><code>{html.escape(item["path"])}</code></figcaption></figure>')
    html_doc='''<!doctype html><html lang="es"><meta charset="utf-8"><title>''' + a.group_id + ''' · análisis</title><style>body{font:15px system-ui;margin:20px;background:#f6f7f9;color:#18212b}section,article{background:white;border:1px solid #d9dde3;border-radius:9px;padding:14px;margin:14px 0}.stats{display:flex;gap:12px;flex-wrap:wrap}.stat{background:#edf2f7;padding:8px;border-radius:6px} .timeline{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}figure{margin:0}img{width:220px;height:165px;object-fit:contain;background:#eef1f4}figcaption{font-size:12px}code{word-break:break-all;font-size:10px;color:#475569}@media(max-width:600px){body{margin:10px}}</style><body><h1>''' + a.group_id + ''' · evolución visual aproximada</h1><section><p>Orden PCA sobre embeddings normalizados. Es un eje de similitud visual para revisión; no demuestra fechas ni causalidad temporal.</p><div class="stats"><div class="stat"><b>Imágenes</b><br>''' + str(len(rows)) + '''</div><div class="stat"><b>Clases</b><br>''' + html.escape(', '.join(f'{k}: {v}' for k,v in labels.items())) + '''</div><div class="stat"><b>Aristas ≥ ''' + str(a.threshold) + '''</b><br>''' + str(len(edge_i)) + '''</div><div class="stat"><b>Contradicciones potenciales</b><br>''' + str(len(contradictory)) + ''' aristas entre clases</div><div class="stat"><b>Cambios de clase en orden visual</b><br>''' + str(transitions) + '''</div></div></section><section><h2>Distribución y contradicciones</h2><p>Una familia con clases distintas puede ser una misma banana en distintos días, una etiqueta errónea o una agrupación demasiado permisiva. Revisar especialmente vecinos de similitud alta que cambian de clase.</p><p>Splits nuevos: ''' + html.escape(', '.join(f'{k}: {v}' for k,v in split.items())) + '''</p></section><section><h2>Orden de evolución visual (izquierda → derecha)</h2><div class="timeline">''' + ''.join(cards) + '''</div></section></body></html>'''
    (out/f'{a.group_id}_analysis.html').write_text(html_doc,encoding='utf-8');print(json.dumps({k:report[k] for k in ('group_id','images','label_distribution','split_distribution','similarity_edges','high_similarity_cross_label_edges','visual_order_label_transitions')},indent=2))
if __name__=='__main__':main()
