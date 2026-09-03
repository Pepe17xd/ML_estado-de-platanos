import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

classes = [
    "unripe",
    "ripe",
    "overripe",
    "rotten"
]

days = {
    "unripe":"5-7 days",
    "ripe":"2-4 days",
    "overripe":"1-2 days",
    "rotten":"0 days"
}

model = tf.keras.models.load_model("models/banana_final_model.h5")

img = image.load_img(
    sys.argv[1],
    target_size=(224,224)
)

x = image.img_to_array(img)
x = np.expand_dims(x,0)/255

prediction = model.predict(x)

idx = np.argmax(prediction)

print({
    "class": classes[idx],
    "confidence": float(prediction[0][idx]),
    "estimated_time": days[classes[idx]]
})