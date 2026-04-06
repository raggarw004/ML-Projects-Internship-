import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from pathlib import Path

IMG_SIZE = (160, 160)
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label

def load_data():
    (train_ds, val_ds), info = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:90%]"],
        as_supervised=True,
        with_info=True
    )

    train_ds = train_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model

def plot_history(history, out_path):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training Curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

def main():
    Path("models").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = load_data()

    model, base_model = build_model()

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5
    )

    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5
    )

    model.save("models/cat_dog_transfer.h5")

    for k in history2.history:
        history1.history[k].extend(history2.history[k])

    plot_history(history1, "reports/figures/training_curve.png")

if __name__ == "__main__":
    main()