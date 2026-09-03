from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

model = tf.keras.models.load_model("../models/banana_mobilenetv2.h5")

classes = [
    "unripe",
    "ripe",
    "overripe",
    "rotten"
]

life = {
    "unripe":"5-7 days",
    "ripe":"2-4 days",
    "overripe":"1-2 days",
    "rotten":"0 days"
}

@app.route("/predict", methods=["POST"])
def predict():

    img = Image.open(
        io.BytesIO(request.files["image"].read())
    ).convert("RGB")

    img = img.resize((224,224))

    x = np.array(img)/255
    x = np.expand_dims(x,0)

    result = model.predict(x)

    idx = np.argmax(result)

    return jsonify({
        "class":classes[idx],
        "confidence":float(result[0][idx]),
        "days_remaining":life[classes[idx]]
    })


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)