"""Exploratory V4 stability benchmark on local images; no ground-truth labels."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]

def variants(image):
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    def temperature(r, g, b):
        x = arr * np.array([r, g, b], dtype=np.float32)
        return Image.fromarray(np.uint8(np.clip(x, 0, 255)))
    shadow = arr.copy()
    h, w = shadow.shape[:2]
    shadow[:, :w // 2] *= 0.48
    return {
        "brightness_0.85": ImageEnhance.Brightness(rgb).enhance(0.85),
        "brightness_1.15": ImageEnhance.Brightness(rgb).enhance(1.15),
        "contrast_0.8": ImageEnhance.Contrast(rgb).enhance(0.8),
        "contrast_1.2": ImageEnhance.Contrast(rgb).enhance(1.2),
        "warm": temperature(1.10, 1.0, 0.88),
        "cool": temperature(0.88, 1.0, 1.10),
        "gamma_0.8": Image.fromarray(np.uint8(np.clip((arr / 255.0) ** 0.8 * 255, 0, 255))),
        "gamma_1.2": Image.fromarray(np.uint8(np.clip((arr / 255.0) ** 1.2 * 255, 0, 255))),
        "shadow_left": Image.fromarray(np.uint8(np.clip(shadow, 0, 255))),
        "exposure_0.75": ImageEnhance.Brightness(rgb).enhance(0.75),
        "exposure_1.25": ImageEnhance.Brightness(rgb).enhance(1.25),
    }

def predict(image):
    from io import BytesIO
    buffer=BytesIO(); image.save(buffer, format="JPEG", quality=95)
    boundary="banana-bench-"+uuid.uuid4().hex
    body=(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"variant.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode()+buffer.getvalue()+f"\r\n--{boundary}--\r\n".encode()
    request=urllib.request.Request("http://127.0.0.1:8000/predict",data=body,headers={"Content-Type":f"multipart/form-data; boundary={boundary}"},method="POST")
    with urllib.request.urlopen(request,timeout=90) as response: return json.loads(response.read())

def main():
    rows=[]
    for path in sorted((ROOT / "test_images").glob("*")):
        try:
            original = Image.open(path).convert("RGB")
        except Exception:
            continue
        base=predict(original)
        variants_result={}
        for name, image in variants(original).items():
            result=predict(image)
            variants_result[name]={"estado": result["estado"], "confianza": result["confianza"], "incierto": result["incierto"]}
        rows.append({"image":path.name,"original":{"estado":base["estado"],"confianza":base["confianza"],"incierto":base["incierto"]},"variants":variants_result})
    n=sum(len(row["variants"]) for row in rows)
    retained=sum(v["estado"]==row["original"]["estado"] for row in rows for v in row["variants"].values())
    mature={"verde","maduro","sobre_maduro"}
    changed=sum(v["estado"]!=row["original"]["estado"] for row in rows for v in row["variants"].values())
    mature_flips=sum(row["original"]["estado"] in mature and v["estado"] in mature and v["estado"]!=row["original"]["estado"] for row in rows for v in row["variants"].values())
    abstentions=sum(v["incierto"] for row in rows for v in row["variants"].values())
    confidence_deltas=[v["confianza"]-row["original"]["confianza"] for row in rows for v in row["variants"].values()]
    states=["verde","maduro","sobre_maduro","incierto"]
    confusion={source:{target:0 for target in states} for source in states}
    changed_cases=[]
    for row in rows:
        for name, variant in row["variants"].items():
            source=row["original"]["estado"]; target=variant["estado"]
            confusion[source][target]+=1
            if source != target:
                changed_cases.append({"image":row["image"],"transformation":name,"from":source,"to":target,"confidence_from":row["original"]["confianza"],"confidence_to":variant["confianza"]})
    output={"interpretation":"Exploratory synthetic sensitivity only; no real labels and not an independent test set.","images":len(rows),"variants_per_image":11,"variant_predictions":n,"same_state_percent":round(100*retained/n,2) if n else None,"state_change_count":changed,"mature_class_flip_count":mature_flips,"abstention_count":abstentions,"mean_confidence_delta":round(float(np.mean(confidence_deltas)),4) if n else None,"mean_absolute_confidence_delta":round(float(np.mean(np.abs(confidence_deltas))),4) if n else None,"confusion_original_rows_variant_columns":confusion,"changed_cases":changed_cases,"per_image":rows}
    print(json.dumps(output,indent=2,ensure_ascii=False))
if __name__ == "__main__": main()
