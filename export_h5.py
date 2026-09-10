import tensorflow as tf

model = tf.keras.models.load_model(
    "models/production_model.keras",
    compile=False
)

model.save(
    "models/production_model.h5"
)

print("Modelo exportado correctamente")