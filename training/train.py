"""Two-stage transfer-learning training entry point."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import tensorflow as tf
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from training.model import build_model, compile_model
from training.pipeline import add_targets, load_config, save_metadata, set_seed, split_train_validation
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config", default=ROOT/"configs/model.yaml"); p.add_argument("--data-dir", default=None); args=p.parse_args()
    cfg=load_config(args.config); set_seed(cfg["seed"]); data=ROOT/(args.data_dir or cfg["data_dir"])
    train_raw,val_raw=split_train_validation(data/"train",cfg); train,val=add_targets(train_raw,cfg),add_targets(val_raw,cfg)
    model,backbone=build_model(cfg); out=ROOT/"models/banana_ripeness.keras"
    callbacks=[tf.keras.callbacks.ModelCheckpoint(out,monitor="val_state_accuracy",mode="max",save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor="val_state_accuracy",mode="max",patience=5,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.3,patience=2)]
    compile_model(model,cfg["learning_rate_head"]); model.fit(train,validation_data=val,epochs=cfg["epochs_head"],callbacks=callbacks)
    backbone.trainable=True
    for layer in backbone.layers[:-30]: layer.trainable=False
    compile_model(model,cfg["learning_rate_finetune"]); model.fit(train,validation_data=val,initial_epoch=cfg["epochs_head"],epochs=cfg["epochs_head"]+cfg["epochs_finetune"],callbacks=callbacks)
    model.save(out); save_metadata(ROOT/"models/metadata.json",cfg); print(f"Saved model: {out}")
if __name__ == "__main__": main()
