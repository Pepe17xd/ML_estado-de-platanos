"""Evaluate official baseline classifier plus independent maturity regressor."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,tensorflow as tf
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,mean_absolute_error,mean_squared_error
ROOT=Path(__file__).resolve().parents[1];CLASSES=['verde','maduro','pasado'];MAT={'unripe':15.,'ripe':60.,'overripe':85.,'rotten':100.}
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default=ROOT/'data/dataset_v1/manifest.csv');p.add_argument('--classifier',default=ROOT/'models/banana_baseline_v1.keras');p.add_argument('--regressor',default=ROOT/'models/banana_maturity_regressor_v3.keras');p.add_argument('--image-size',type=int,default=224);p.add_argument('--batch-size',type=int,default=32);a=p.parse_args()
 with open(a.manifest,encoding='utf-8',newline='') as f:rows=[r for r in csv.DictReader(f) if r['split']=='test']
 paths=[r['image_path'] for r in rows];truth=np.array([CLASSES.index(r['class_name']) for r in rows]);target=np.array([MAT[r['original_state']] for r in rows])
 def load(path):
  image=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False);return tf.image.resize(tf.cast(image,tf.float32),(a.image_size,a.image_size))
 ds=tf.data.Dataset.from_tensor_slices(paths).map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(a.batch_size);classifier=tf.keras.models.load_model(a.classifier,compile=False);regressor=tf.keras.models.load_model(a.regressor,compile=False);pred=classifier.predict(ds,verbose=0).argmax(1);maturity=regressor.predict(ds,verbose=0)[:,0]*100
 classification={'accuracy':float(accuracy_score(truth,pred)),'precision_macro':float(precision_score(truth,pred,average='macro',zero_division=0)),'recall_macro':float(recall_score(truth,pred,average='macro',zero_division=0)),'f1_macro':float(f1_score(truth,pred,average='macro',zero_division=0)),'confusion_matrix':confusion_matrix(truth,pred).tolist(),'classification_report':classification_report(truth,pred,target_names=CLASSES,output_dict=True,zero_division=0)};regression={'maturity_mae_pct':float(mean_absolute_error(target,maturity)),'maturity_rmse_pct':float(mean_squared_error(target,maturity)**.5),'target_warning':'Weak labels derived from original state; replace with measured maturity_pct for production.'};result={'classifier':str(a.classifier),'regressor':str(a.regressor),'test_images':len(rows),'classification':classification,'regression':regression,'days_remaining':'removed from Quality v3','note':'Dataset v1 and current grouped test split unchanged.'};(ROOT/'models/banana_quality_v3_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
