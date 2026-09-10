"""Build a self-contained, filterable manual-review page from embedding results."""
from __future__ import annotations
import argparse, base64, html
from pathlib import Path
import numpy as np
from PIL import Image
from data_audit.common import image_records, write_csv

def thumbnail_data_url(path: str, size: int = 260) -> str:
    import io
    with Image.open(path) as image:
        image = image.convert("RGB"); image.thumbnail((size, size))
        buffer = io.BytesIO(); image.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

def main():
    p=argparse.ArgumentParser();p.add_argument("--embeddings",default="reports/data_quality/embeddings.npz");p.add_argument("--output-dir",default="reports/data_quality");p.add_argument("--top-k",type=int,default=100);a=p.parse_args()
    archive=np.load(a.embeddings); paths=[str(x) for x in archive["paths"]]; vectors=archive["embeddings"]
    metadata={r["path"]:r for r in image_records(Path("dataset"))}
    sim=vectors@vectors.T; np.fill_diagonal(sim,-np.inf)
    # Similarity is symmetric, so collect extra candidates before de-duplicating (A,B)/(B,A).
    pool=min(sim.size, a.top_k * 4)
    candidate=np.argpartition(sim.ravel(), -pool)[-pool:]; candidate=candidate[np.argsort(sim.ravel()[candidate])[::-1]]
    pairs=[]; seen=set()
    for flat in candidate:
        i,j=divmod(int(flat),sim.shape[1]); key=tuple(sorted((i,j)))
        if key in seen: continue
        seen.add(key); left,right=metadata[paths[i]],metadata[paths[j]]
        pairs.append({"rank":len(pairs)+1,"score":round(float(sim[i,j]),6),"path_a":paths[i],"split_a":left["original_split"],"class_a":left["state"],"image_a":thumbnail_data_url(paths[i]),"path_b":paths[j],"split_b":right["original_split"],"class_b":right["state"],"image_b":thumbnail_data_url(paths[j]),"cross_split":left["original_split"] != right["original_split"]})
        if len(pairs)==a.top_k: break
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"top_100_embedding_pairs.csv",[{k:v for k,v in pair.items() if not k.startswith('image_')} for pair in pairs],["rank","score","path_a","split_a","class_a","path_b","split_b","class_b","cross_split"])
    cards=[]
    for pair in pairs:
        def block(side):
            return f'<section><img src="{pair["image_"+side]}" alt="image {side.upper()}"><div><b>Imagen {side.upper()}</b><br><span>split: {html.escape(pair["split_"+side])}</span> · <span>clase: {html.escape(pair["class_"+side])}</span><code>{html.escape(pair["path_"+side])}</code></div></section>'
        warning='<span class="warning">CRUCE TRAIN/TEST</span>' if pair['cross_split'] else ''
        cards.append(f'<article data-cross="{str(pair["cross_split"]).lower()}"><header>#{pair["rank"]} · similitud coseno: <strong>{pair["score"]:.6f}</strong> {warning}</header><div class="pair">{block("a")}{block("b")}</div></article>')
    page='''<!doctype html><html lang="es"><meta charset="utf-8"><title>Revisión manual: embeddings</title><style>body{font:16px system-ui;margin:24px;background:#f6f7f9;color:#17202a}article{background:white;border:1px solid #d9dde3;border-radius:10px;margin:16px 0;padding:14px}header{margin-bottom:10px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}img{width:100%;max-width:360px;height:240px;object-fit:contain;background:#eef1f4}code{display:block;word-break:break-all;font-size:11px;color:#475569;margin-top:5px}.warning{background:#fee2e2;color:#991b1b;padding:3px 6px;border-radius:4px;font-size:12px}button{padding:8px;margin-right:8px}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style><body><h1>Top 100 pares más similares por embeddings</h1><p>MobileNetV3Small/ImageNet · similitud coseno. Revisar manualmente antes de borrar o reasignar. <button onclick="filter(false)">Todos</button><button onclick="filter(true)">Sólo cruces train/test</button></p><div id="cards">''' + ''.join(cards) + '''</div><script>function filter(cross){document.querySelectorAll('article').forEach(x=>x.hidden=cross&&x.dataset.cross!=='true')}</script></body></html>'''
    (out/"top_100_embedding_review.html").write_text(page,encoding="utf-8")
    print(f"Wrote {len(pairs)} pairs to {out / 'top_100_embedding_review.html'}")
if __name__=="__main__":main()
