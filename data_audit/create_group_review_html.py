"""Create a manual-review report for the largest embedding-connected families."""
from __future__ import annotations
import argparse, base64, csv, hashlib, html, io
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from data_audit.common import write_csv

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def thumbnail(path):
    with Image.open(path) as im:
        im=im.convert('RGB');im.thumbnail((260,190));buf=io.BytesIO();im.save(buf,'JPEG',quality=82)
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()
def has_frame_name(paths):
    return sum(('frame_' in Path(x).stem.lower() or 'image_' in Path(x).stem.lower()) for x in paths) / len(paths) >= .5
def hypothesis(paths, classes, embeddings_by_path):
    hashes=[digest(path) for path in paths]
    if len(set(hashes)) < len(hashes): return 'b) duplicados', 'Hay al menos dos archivos byte-a-byte idénticos.'
    vectors=np.stack([embeddings_by_path[p] for p in paths]); sims=vectors@vectors.T; np.fill_diagonal(sims,-1)
    max_sim=float(sims.max()) if len(paths)>1 else 0.
    if has_frame_name(paths): return 'a) mismo video', f'Predominan nombres de fotogramas; similitud máxima {max_sim:.4f}.'
    if len(classes)>1: return 'c) misma banana en distintos días', f'Mezcla de clases visualmente conectadas; similitud máxima {max_sim:.4f}. Confirmar etiquetas y fechas.'
    return 'd) imágenes independientes', f'No hay hash exacto, patrón de vídeo ni mezcla de clases concluyente; similitud máxima {max_sim:.4f}.'
def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata',default='data/metadata_grouped.csv');p.add_argument('--embeddings',default='reports/data_quality/embeddings.npz');p.add_argument('--output-dir',default='reports/data_quality');p.add_argument('--top-k',type=int,default=20);p.add_argument('--examples',type=int,default=6);a=p.parse_args()
    with open(a.metadata,encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    grouped=defaultdict(list)
    for row in rows: grouped[row['group_id']].append(row)
    archive=np.load(a.embeddings);embedding_map={str(path):vec for path,vec in zip(archive['paths'],archive['embeddings'])}
    top=sorted(grouped.items(),key=lambda x:(-len(x[1]),x[0]))[:a.top_k];summary=[];cards=[]
    for rank,(gid,group) in enumerate(top,1):
        paths=[r['image_path'] for r in group];classes=sorted({r['state_label'] for r in group});splits=sorted({r['new_split'] for r in group})
        label,reason=hypothesis(paths,classes,embedding_map)
        summary.append({'rank':rank,'group_id':gid,'images':len(group),'classes':', '.join(classes),'assigned_split':', '.join(splits),'hypothesis':label,'rationale':reason})
        examples=''.join(f'<figure><img src="{thumbnail(row["image_path"])}"><figcaption>{html.escape(Path(row["image_path"]).name)}<br>clase: {html.escape(row["state_label"])}<br><code>{html.escape(row["image_path"])}</code></figcaption></figure>' for row in group[:a.examples])
        cards.append(f'<article data-type="{html.escape(label[3:])}"><header><b>#{rank} · {html.escape(gid)}</b><span>{len(group)} imágenes</span><span>clases: {html.escape(", ".join(classes))}</span><span>split asignado: {html.escape(", ".join(splits))}</span></header><p><strong>Hipótesis: {html.escape(label)}</strong><br>{html.escape(reason)}</p><div class="examples">{examples}</div></article>')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);write_csv(out/'top_20_visual_families.csv',summary,['rank','group_id','images','classes','assigned_split','hypothesis','rationale'])
    page='''<!doctype html><html lang="es"><meta charset="utf-8"><title>Revisión de familias visuales</title><style>body{font:16px system-ui;margin:24px;background:#f6f7f9;color:#17202a;max-width:1400px}article{background:#fff;border:1px solid #d9dde3;border-radius:10px;margin:16px 0;padding:16px}header{display:flex;gap:14px;flex-wrap:wrap;align-items:center}header span{background:#edf2f7;padding:3px 7px;border-radius:5px}.examples{display:flex;gap:12px;overflow-x:auto}figure{min-width:260px;max-width:260px;margin:8px 0}img{width:260px;height:190px;object-fit:contain;background:#eef1f4}figcaption{font-size:12px}code{display:block;word-break:break-all;color:#475569;font-size:10px;margin-top:4px}button{padding:8px;margin-right:7px}@media(max-width:600px){body{margin:12px}}</style><body><h1>Top 20 familias visuales</h1><p>Los grupos fueron creados con componentes conexas de embeddings MobileNetV3Small y umbral coseno ≥ 0,985. La hipótesis es automática; validar con los ejemplos antes de cambiar datos.</p><p><button onclick="f('')">Todas</button><button onclick="f('a) mismo video')">Video</button><button onclick="f('b) duplicados')">Duplicados</button><button onclick="f('c) misma banana en distintos días')">Distintos días</button><button onclick="f('d) imágenes independientes')">Independientes</button></p>''' + ''.join(cards) + '''<script>function f(t){document.querySelectorAll('article').forEach(x=>x.hidden=t&&!x.dataset.type.startsWith(t.slice(3)))}</script></body></html>'''
    (out/'top_20_visual_families_review.html').write_text(page,encoding='utf-8');print(f'Wrote {len(top)} groups to {out / "top_20_visual_families_review.html"}')
if __name__=='__main__':main()
