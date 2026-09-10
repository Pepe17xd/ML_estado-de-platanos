"""Two-stage multitask training: classification first, auxiliary heads later."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
CLASSES=['verde','maduro','pasado'];MAT={'unripe':15.,'ripe':60.,'overripe':85.,'rotten':100.};DAYS={'unripe':6.,'ripe':3.,'overripe':1.,'rotten':0.}
def read(path,split):
 with open(path,encoding='utf-8',newline='') as f:return [r for r in csv.DictReader(f) if r['split']==split]
def ds(items,size,batch,augment=False):
 paths=[r['image_path'] for r in items];states=[CLASSES.index(r['class_name']) for r in items];mats=[MAT[r['original_state']]/100 for r in items];days=[DAYS[r['original_state']] for r in items]
 data=tf.data.Dataset.from_tensor_slices((paths,states,mats,days))
 def load(path,state,mat,day):
  image=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False);image=tf.image.resize(tf.cast(image,tf.float32),(size,size))
  if augment:image=tf.image.random_flip_left_right(image);image=tf.image.random_brightness(image,.1);image=tf.image.random_contrast(image,.9,1.1)
  return image,{'state':tf.one_hot(state,3),'maturity':mat,'days_remaining':day}
 return data.shuffle(len(items),seed=42,reshuffle_each_iteration=True).map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)
def compile_for(model,lr,weights):
 model.compile(optimizer=tf.keras.optimizers.AdamW(lr,weight_decay=1e-5),loss={'state':tf.keras.losses.CategoricalCrossentropy(label_smoothing=.05),'maturity':tf.keras.losses.Huber(),'days_remaining':tf.keras.losses.Huber()},loss_weights=weights,metrics={'state':[tf.keras.metrics.CategoricalAccuracy(name='accuracy')],'maturity':[tf.keras.metrics.MeanAbsoluteError(name='mae')],'days_remaining':[tf.keras.metrics.MeanAbsoluteError(name='mae')]})
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default=ROOT/'data/dataset_v1/manifest.csv');p.add_argument('--output',default=ROOT/'models/banana_multitask_v2.keras');p.add_argument('--epochs-classification',type=int,default=8);p.add_argument('--epochs-finetune',type=int,default=10);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--image-size',type=int,default=224);a=p.parse_args();tf.keras.utils.set_random_seed(42)
 train=ds(read(a.manifest,'train'),a.image_size,a.batch_size,True);val=ds(read(a.manifest,'validation'),a.image_size,a.batch_size,False)
 inp=tf.keras.Input((a.image_size,a.image_size,3),name='image');x=tf.keras.applications.mobilenet_v3.preprocess_input(inp);backbone=tf.keras.applications.MobileNetV3Small(include_top=False,weights='imagenet',input_tensor=x);backbone.trainable=False;x=tf.keras.layers.Dropout(.25)(tf.keras.layers.GlobalAveragePooling2D()(backbone.output));shared=tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
 model=tf.keras.Model(inp,[tf.keras.layers.Dense(3,activation='softmax',name='state')(shared),tf.keras.layers.Dense(1,activation='sigmoid',name='maturity')(shared),tf.keras.layers.Dense(1,activation='relu',name='days_remaining')(shared)],name='banana_multitask_v2')
 out=str(a.output);callbacks=[tf.keras.callbacks.ModelCheckpoint(out,monitor='val_state_accuracy',mode='max',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_state_accuracy',mode='max',patience=4,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.3,patience=2,min_lr=1e-7)]
 compile_for(model,1e-3,{'state':1.,'maturity':.1,'days_remaining':0.});h1=model.fit(train,validation_data=val,epochs=a.epochs_classification,callbacks=callbacks)
 backbone.trainable=True
 for layer in backbone.layers[:-30]:layer.trainable=False
 for layer in backbone.layers:
  if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
 compile_for(model,1e-5,{'state':1.,'maturity':.25,'days_remaining':.05});h2=model.fit(train,validation_data=val,initial_epoch=a.epochs_classification,epochs=a.epochs_classification+a.epochs_finetune,callbacks=callbacks);model.save(out)
 print('saved',out);print('stage1_best_val_state_accuracy',max(h1.history['val_state_accuracy']));print('stage2_best_val_state_accuracy',max(h2.history['val_state_accuracy']))
if __name__=='__main__':main()
