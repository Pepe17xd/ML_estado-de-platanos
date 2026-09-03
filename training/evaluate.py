import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# Configuración

IMG_SIZE = 224
BATCH_SIZE = 32

TEST_DIR = "dataset/test"

MODEL_PATH = "models/banana_final_model.h5"


# Cargar modelo

model = tf.keras.models.load_model(MODEL_PATH)


# Cargar dataset de prueba

test_datagen = ImageDataGenerator(
    rescale=1./255
)


test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# Predicciones

predictions = model.predict(test_data)


y_pred = np.argmax(
    predictions,
    axis=1
)


y_true = test_data.classes


classes = list(
    test_data.class_indices.keys()
)


print("\nClases:")
print(classes)



# Matriz de confusión

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\nMatriz de confusión:")
print(cm)



plt.figure(figsize=(8,6))


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes
)


plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Banana")


os.makedirs(
    "models",
    exist_ok=True
)


plt.savefig(
    "models/confusion_matrix.png"
)


# Reporte

report = classification_report(
    y_true,
    y_pred,
    target_names=classes
)


print(report)


with open(
    "models/classification_report.txt",
    "w"
) as f:
    f.write(report)


print("\nEvaluación terminada")