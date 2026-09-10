import tensorflow as tf
def build_model(cfg):
    size = cfg["image_size"]
    aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(.08), tf.keras.layers.RandomZoom(.12), tf.keras.layers.RandomContrast(.15), tf.keras.layers.RandomTranslation(.08, .08)], name="augmentation")
    inputs = tf.keras.Input((size, size, 3), name="image")
    x = tf.keras.applications.mobilenet_v3.preprocess_input(aug(inputs))
    backbone = tf.keras.applications.MobileNetV3Small(include_top=False, weights="imagenet", input_tensor=x); backbone.trainable = False
    x = tf.keras.layers.Dropout(.25)(tf.keras.layers.GlobalAveragePooling2D()(backbone.output))
    shared = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    model = tf.keras.Model(inputs, [tf.keras.layers.Dense(len(cfg["states"]), activation="softmax", name="state")(shared), tf.keras.layers.Dense(1, activation="sigmoid", name="maturity")(shared), tf.keras.layers.Dense(1, activation="relu", name="days_remaining")(shared)], name="banana_ripeness")
    return model, backbone
def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.AdamW(lr, weight_decay=1e-5), loss={"state": tf.keras.losses.CategoricalCrossentropy(label_smoothing=.05), "maturity": tf.keras.losses.Huber(), "days_remaining": tf.keras.losses.Huber()}, loss_weights={"state": 1., "maturity": .25, "days_remaining": .15}, metrics={"state": [tf.keras.metrics.CategoricalAccuracy(name="accuracy")], "maturity": [tf.keras.metrics.MeanAbsoluteError(name="mae")], "days_remaining": [tf.keras.metrics.MeanAbsoluteError(name="mae")]})
