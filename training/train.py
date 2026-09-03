import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os


IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20


TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"


# Preparación de datos

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)


test_datagen = ImageDataGenerator(
    rescale=1./255
)


train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)


val_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)


test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


print("\nClases detectadas:")
print(train_data.class_indices)


# Modelo base

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)


base_model.trainable = False


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(
    4,
    activation="softmax"
)(x)


model = Model(
    inputs=base_model.input,
    outputs=output
)


model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# Guardar mejor modelo

os.makedirs(
    "models",
    exist_ok=True
)


callbacks = [

EarlyStopping(
    patience=5,
    restore_best_weights=True
),

ModelCheckpoint(
    "models/best_banana_model.h5",
    save_best_only=True
)

]


# Entrenamiento

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)


# Evaluación

loss, accuracy = model.evaluate(test_data)

print("\nAccuracy prueba:")
print(accuracy)


# Guardar modelo final

model.save(
    "models/banana_final_model.h5"
)


# Gráficas

plt.figure()

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Accuracy")
plt.legend(
    ["train","validation"]
)

plt.savefig(
    "models/accuracy.png"
)


plt.figure()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Loss")

plt.legend(
    ["train","validation"]
)

plt.savefig(
    "models/loss.png"
)


print("\nEntrenamiento terminado")