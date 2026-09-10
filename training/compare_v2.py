"""Evaluate baseline and fine-tuned candidate on the unchanged grouped test set."""
from __future__ import annotations
import argparse,csv,json,shutil,sys
from pathlib import Path
import numpy as np,tensorflow as tf
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
ROOT=Path(__file__).resolve().parents[1];CLASSES=['verde','maduro','pasado']
def evaluate(model_path,items,size,batch):
 paths=[r['image_path'] for r in items];truth=np.array([CLASSES.index(r['class_name']) for r in items])
 def load(path):
  image=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False);return tf.image.resize(tf.cast(image,tf.float32),(size,size))
 ds=tf.data.Dataset.from_tensor_slices(paths).map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(batch);pred=tf.keras.models.load_model(model_path,compile=False).predict(ds,verbose=0).argmax(1)
 return {'accuracy':float(accuracy_score(truth,pred)),'precision_macro':float(precision_score(truth,pred,average='macro',zero_division=0)),'recall_macro':float(recall_score(truth,pred,average='macro',zero_division=0)),'f1_macro':float(f1_score(truth,pred,average='macro',zero_division=0)),'confusion_matrix':confusion_matrix(truth,pred).tolist(),'classification_report':classification_report(truth,pred,target_names=CLASSES,output_dict=True,zero_division=0)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default=ROOT/'data/dataset_v1/manifest.csv');p.add_argument('--baseline',default=ROOT/'models/banana_baseline_v1.keras');p.add_argument('--candidate',default=ROOT/'models/banana_model_v2_candidate.keras');p.add_argument('--output',default=ROOT/'models/baseline_vs_v2_evaluation.json');p.add_argument('--image-size',type=int,default=224);p.add_argument('--batch-size',type=int,default=32);a=p.parse_args()
 with open(a.manifest,encoding='utf-8',newline='') as f:test=[r for r in csv.DictReader(f) if r['split']=='test']
 baseline=evaluate(a.baseline,test,a.image_size,a.batch_size);candidate=evaluate(a.candidate,test,a.image_size,a.batch_size);improved=candidate['accuracy']>baseline['accuracy']
 if improved:shutil.copy2(a.candidate,ROOT/'models/banana_model_v2.keras')
 result={'test_images':len(test),'classes':CLASSES,'baseline_v1':baseline,'candidate_v2':candidate,'improved_accuracy':improved,'promotion_metric':'accuracy','promoted_model':'models/banana_model_v2.keras' if improved else None,'note':'Same dataset-v1 grouped test manifest; no additional audit or data change.'}
 Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8');md=f'''# Baseline v1 vs fine-tuned v2\n\nTest images: **{len(test)}**. Promotion criterion: v2 accuracy > v1 accuracy.\n\n| Model | Accuracy | Precision macro | Recall macro | F1 macro |\n|---|---:|---:|---:|---:|\n| baseline_v1 | {baseline["accuracy"]:.4f} | {baseline["precision_macro"]:.4f} | {baseline["recall_macro"]:.4f} | {baseline["f1_macro"]:.4f} |\n| candidate_v2 | {candidate["accuracy"]:.4f} | {candidate["precision_macro"]:.4f} | {candidate["recall_macro"]:.4f} | {candidate["f1_macro"]:.4f} |\n\n## Confusion matrix v2\nRows actual, columns predicted (`verde`, `maduro`, `pasado`):\n\n```text\n{candidate["confusion_matrix"]}\n```\n\n**Promoted:** {improved}.\n''';Path(ROOT/'models/baseline_vs_v2_evaluation.md').write_text(md,encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
