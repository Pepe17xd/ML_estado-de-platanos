"""Train multitask banana model on unchanged dataset-v1 grouped splits."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
CLASSES=['verde','maduro','pasado'];MAT={'unripe':15.,'ripe':60.,'overripe':85.,'rotten':100.};DAYS={'unripe':6.,'ripe':3.,'overripe':1.,'rotten':0.}
def read(path,split):
 with open(path,encoding='utf-8',newline='') as f:return [r for r in csv.DictReader(f) if r['split']==split]
def dataset(items,size,batch,augment=False):
 paths=[r['image_path'] for r in items]; state=[CLASSES.index(r['class_name']) for r in items]; maturity=[MAT[r['original_state']]/100. for r in items]; days=[DAYS[r['original_state']] for r in items]
 ds=tf.data.Dataset.from_tensor_slices((paths,state,maturity,days))
 def load(path,label,mat,day):
  image=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False);image=tf.image.resize(tf.cast(image,tf.float32),(size,size))
  if augment:image=tf.image.random_flip_left_right(image);image=tf.image.random_brightness(image,.1);image=tf.image.random_contrast(image,.9,1.1)
  return image,{'state':tf.one_hot(label,3),'maturity':mat,'days_remaining':day}
 return ds.shuffle(len(items),seed=42,reshuffle_each_iteration=True).map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default=ROOT/'data/dataset_v1/manifest.csv');p.add_argument('--output',default=ROOT/'models/banana_multitask_v1.keras');p.add_argument('--epochs',type=int,default=15);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--image-size',type=int,default=224);p.add_argument('--days-head-experimental',action='store_true',default=True);a=p.parse_args()
 tf.keras.utils.set_random_seed(42);train=dataset(read(a.manifest,'train'),a.image_size,a.batch_size,True);val=dataset(read(a.manifest,'validation'),a.image_size,a.batch_size,False)
 inp=tf.keras.Input((a.image_size,a.image_size,3),name='image');x=tf.keras.applications.mobilenet_v3.preprocess_input(inp);backbone=tf.keras.applications.MobileNetV3Small(include_top=False,weights='imagenet',input_tensor=x);backbone.trainable=False;x=tf.keras.layers.Dropout(.25)(tf.keras.layers.GlobalAveragePooling2D()(backbone.output));shared=tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
 model=tf.keras.Model(inp,[tf.keras.layers.Dense(3,activation='softmax',name='state')(shared),tf.keras.layers.Dense(1,activation='sigmoid',name='maturity')(shared),tf.keras.layers.Dense(1,activation='relu',name='days_remaining')(shared)],name='banana_multitask_v1')
 model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3,weight_decay=1e-5),loss={'state':tf.keras.losses.CategoricalCrossentropy(label_smoothing=.05),'maturity':tf.keras.losses.Huber(),'days_remaining':tf.keras.losses.Huber()},loss_weights={'state':1.,'maturity':.25,'days_remaining':.15},metrics={'state':[tf.keras.metrics.CategoricalAccuracy(name='accuracy')],'maturity':[tf.keras.metrics.MeanAbsoluteError(name='mae')],'days_remaining':[tf.keras.metrics.MeanAbsoluteError(name='mae')]})
 out=str(a.output);callbacks=[tf.keras.callbacks.ModelCheckpoint(out,monitor='val_state_accuracy',mode='max',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_state_accuracy',mode='max',patience=5,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.3,patience=2,min_lr=1e-6)]
 h=model.fit(train,validation_data=val,epochs=a.epochs,callbacks=callbacks);model.save(out);print('saved',out);print({k:round(float(v[-1]),4) for k,v in h.history.items()})
if __name__=='__main__':main()
