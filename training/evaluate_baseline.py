"""Evaluate baseline-v1 on the group-held-out test split."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np, tensorflow as tf
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));CLASSES=['verde','maduro','pasado']
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default=ROOT/'data/dataset_v1/manifest.csv');p.add_argument('--model',default=ROOT/'models/banana_baseline_v1.keras');p.add_argument('--image-size',type=int,default=224);p.add_argument('--batch-size',type=int,default=32);a=p.parse_args()
 with open(a.manifest,encoding='utf-8',newline='') as f:rows=[r for r in csv.DictReader(f) if r['split']=='test']
 paths=[r['image_path'] for r in rows];truth=np.array([CLASSES.index(r['class_name']) for r in rows])
 def load(path):
  image=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False);return tf.image.resize(tf.cast(image,tf.float32),(a.image_size,a.image_size))
 ds=tf.data.Dataset.from_tensor_slices(paths).map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(a.batch_size);probs=tf.keras.models.load_model(a.model,compile=False).predict(ds,verbose=0);pred=probs.argmax(1)
 result={'model':str(a.model),'images':len(rows),'classes':CLASSES,'accuracy':float(accuracy_score(truth,pred)),'precision_macro':float(precision_score(truth,pred,average='macro',zero_division=0)),'recall_macro':float(recall_score(truth,pred,average='macro',zero_division=0)),'confusion_matrix':confusion_matrix(truth,pred).tolist(),'classification_report':classification_report(truth,pred,target_names=CLASSES,output_dict=True,zero_division=0)}
 out=ROOT/'models/baseline_v1_evaluation.json';out.write_text(json.dumps(result,indent=2),encoding='utf-8')
 matrix='\n'.join('| '+' | '.join(map(str,row))+' |' for row in result['confusion_matrix'])
 markdown=f'''# Baseline v1 evaluation\n\nModel: `{a.model}`  \nTest images: **{result["images"]}**\n\n| Metric | Value |\n|---|---:|\n| Accuracy | {result["accuracy"]:.4f} |\n| Precision (macro) | {result["precision_macro"]:.4f} |\n| Recall (macro) | {result["recall_macro"]:.4f} |\n\nClasses: `{", ".join(CLASSES)}`. Confusion matrix rows are actual classes and columns predicted classes.\n\n| | verde | maduro | pasado |\n|---|---:|---:|---:|\n{matrix}\n\nThis is a baseline result on the group-based test manifest; no architecture tuning was performed.\n'''
 (ROOT/'models/baseline_v1_evaluation.md').write_text(markdown,encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
