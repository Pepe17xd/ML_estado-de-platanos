"""Train an independent 0-100 maturity regressor; no days head."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
MAT={'unripe':15.,'ripe':60.,'overripe':85.,'rotten':100.}
def read(path,split):
 with open(path,encoding='utf-8',newline='') as f:return [r for r in csv.DictReader(f) if r['split']==split]
def dataset(items,size,batch,augment=False):
 paths=[r['image_path'] for r in items]; targets=[MAT[r['original_state']]/100 for r in items];ds=tf.data.Dataset.from_tensor_slices((paths,targets))
 def load(path,target):
  image=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False);image=tf.image.resize(tf.cast(image,tf.float32),(size,size))
  if augment:image=tf.image.random_flip_left_right(image);image=tf.image.random_brightness(image,.1);image=tf.image.random_contrast(image,.9,1.1)
  return image,target
 return ds.shuffle(len(items),seed=42,reshuffle_each_iteration=True).map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default=ROOT/'data/dataset_v1/manifest.csv');p.add_argument('--output',default=ROOT/'models/banana_maturity_regressor_v3.keras');p.add_argument('--epochs',type=int,default=15);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--image-size',type=int,default=224);a=p.parse_args();tf.keras.utils.set_random_seed(42)
 train=dataset(read(a.manifest,'train'),a.image_size,a.batch_size,True);val=dataset(read(a.manifest,'validation'),a.image_size,a.batch_size,False);inp=tf.keras.Input((a.image_size,a.image_size,3),name='image');x=tf.keras.applications.mobilenet_v3.preprocess_input(inp);backbone=tf.keras.applications.MobileNetV3Small(include_top=False,weights='imagenet',input_tensor=x);backbone.trainable=False;x=tf.keras.layers.Dropout(.25)(tf.keras.layers.GlobalAveragePooling2D()(backbone.output));x=tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x);out=tf.keras.layers.Dense(1,activation='sigmoid',name='maturity')(x);model=tf.keras.Model(inp,out,name='banana_maturity_regressor_v3');model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3,weight_decay=1e-5),loss=tf.keras.losses.Huber(),metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]);path=str(a.output);callbacks=[tf.keras.callbacks.ModelCheckpoint(path,monitor='val_mae',mode='min',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_mae',mode='min',patience=5,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.3,patience=2,min_lr=1e-6)];h=model.fit(train,validation_data=val,epochs=a.epochs,callbacks=callbacks);model.save(path);print('saved',path);print({k:round(float(v[-1]),4) for k,v in h.history.items()})
if __name__=='__main__':main()
