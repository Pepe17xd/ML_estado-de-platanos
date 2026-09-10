"""Fine-tune baseline-v1 without changing dataset or split manifests."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
CLASSES=['verde','maduro','pasado']
def rows(path,split):
 with open(path,encoding='utf-8',newline='') as f:return [r for r in csv.DictReader(f) if r['split']==split]
def dataset(items,size,batch,weights=None,augment=False):
 paths=[r['image_path'] for r in items]; labels=[CLASSES.index(r['class_name']) for r in items]
 ds=tf.data.Dataset.from_tensor_slices((paths,labels))
 def load(path,label):
  image=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False);image=tf.image.resize(tf.cast(image,tf.float32),(size,size))
  if augment:
   image=tf.image.random_flip_left_right(image);image=tf.image.random_brightness(image,.1);image=tf.image.random_contrast(image,.9,1.1)
  if weights is None:return image,label
  return image,label,tf.gather(tf.constant(weights,tf.float32),label)
 ds=ds.shuffle(len(items),seed=42,reshuffle_each_iteration=True).map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)
 return ds
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default=ROOT/'data/dataset_v1/manifest.csv');p.add_argument('--baseline',default=ROOT/'models/banana_baseline_v1.keras');p.add_argument('--output',default=ROOT/'models/banana_model_v2_candidate.keras');p.add_argument('--epochs',type=int,default=15);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--image-size',type=int,default=224);a=p.parse_args()
 tf.keras.utils.set_random_seed(42);train_rows=rows(a.manifest,'train');val_rows=rows(a.manifest,'validation');counts=[sum(r['class_name']==c for r in train_rows) for c in CLASSES];total=sum(counts);weights=[total/(len(CLASSES)*n) for n in counts];print('class_weights',dict(zip(CLASSES,weights)))
 train=dataset(train_rows,a.image_size,a.batch_size,weights,True);val=dataset(val_rows,a.image_size,a.batch_size,None,False);model=tf.keras.models.load_model(a.baseline,compile=False)
 # Fine-tune only the final 30 backbone layers; keep BatchNorm frozen for small data.
 backbone=next(layer for layer in model.layers if isinstance(layer,tf.keras.Model) and 'mobilenetv3' in layer.name.lower());backbone.trainable=True
 for layer in backbone.layers[:-30]:layer.trainable=False
 for layer in backbone.layers:
  if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
 model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
 out=str(a.output);callbacks=[tf.keras.callbacks.ModelCheckpoint(out,monitor='val_accuracy',mode='max',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',patience=4,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.3,patience=2,min_lr=1e-7)]
 history=model.fit(train,validation_data=val,epochs=a.epochs,callbacks=callbacks);model.save(out);print('candidate_saved',out);print({k:round(float(v[-1]),4) for k,v in history.history.items()})
if __name__=='__main__':main()
