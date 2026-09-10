"""Evaluate held-out test data and write models/evaluation.json."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.pipeline import add_targets,load_config
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",default=ROOT/"models/banana_ripeness.keras");p.add_argument("--config",default=ROOT/"configs/model.yaml");a=p.parse_args();cfg=load_config(a.config)
 raw=tf.keras.utils.image_dataset_from_directory(ROOT/cfg["data_dir"]/'test',class_names=cfg['states'],label_mode='int',image_size=(cfg['image_size'],cfg['image_size']),batch_size=cfg['batch_size'],shuffle=False)
 truth=np.concatenate([labels.numpy() for _,labels in raw]); pred=tf.keras.models.load_model(a.model).predict(add_targets(raw,cfg),verbose=0)[0].argmax(1)
 result={'confusion_matrix':confusion_matrix(truth,pred).tolist(),'classification_report':classification_report(truth,pred,target_names=cfg['states'],output_dict=True,zero_division=0)}
 (ROOT/'models/evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
