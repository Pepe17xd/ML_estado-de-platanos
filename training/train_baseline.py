"""Train a three-class MobileNetV3Small baseline from dataset-v1 manifest."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
CLASSES=['verde','maduro','pasado']
def read_rows(path,split):
 with open(path,encoding='utf-8',newline='') as f:return [r for r in csv.DictReader(f) if r['split']==split]
def make_ds(rows,size,batch,augment=False):
 paths=[r['image_path'] for r in rows]; labels=[CLASSES.index(r['class_name']) for r in rows]
 ds=tf.data.Dataset.from_tensor_slices((paths,labels))
 def load(path,label):
  image=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False);image=tf.image.resize(tf.cast(image,tf.float32),(size,size));
  if augment:
   image=tf.image.random_flip_left_right(image);image=tf.image.random_brightness(image,.12);image=tf.image.random_contrast(image,.85,1.15)
  return image,label
 return ds.shuffle(len(rows),seed=42,reshuffle_each_iteration=True).map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default=ROOT/'data/dataset_v1/manifest.csv');p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--image-size',type=int,default=224);a=p.parse_args()
 tf.keras.utils.set_random_seed(42);train_rows=read_rows(a.manifest,'train');val_rows=read_rows(a.manifest,'validation')
 train=make_ds(train_rows,a.image_size,a.batch_size,True);val=make_ds(val_rows,a.image_size,a.batch_size,False)
 base=tf.keras.applications.MobileNetV3Small(include_top=False,weights='imagenet',input_shape=(a.image_size,a.image_size,3));base.trainable=False
 inputs=tf.keras.Input((a.image_size,a.image_size,3));x=tf.keras.applications.mobilenet_v3.preprocess_input(inputs);x=base(x,training=False);x=tf.keras.layers.GlobalAveragePooling2D()(x);x=tf.keras.layers.Dropout(.25)(x);outputs=tf.keras.layers.Dense(3,activation='softmax',name='state')(x);model=tf.keras.Model(inputs,outputs)
 model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
 out=ROOT/'models/banana_baseline_v1.keras';out.parent.mkdir(exist_ok=True)
 callbacks=[tf.keras.callbacks.ModelCheckpoint(str(out),monitor='val_accuracy',mode='max',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',patience=5,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2,factor=.3)]
 history=model.fit(train,validation_data=val,epochs=a.epochs,callbacks=callbacks);model.save(out);print(f'Best baseline saved to {out}');print({k:round(float(v[-1]),4) for k,v in history.history.items()})
if __name__=='__main__':main()
