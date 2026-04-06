import tensorflow as tf
import numpy as np
from pathlib import Path

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def load_images(path):
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels=None,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    ds = ds.map(lambda x: (x / 255.0, x / 255.0))
    return ds

def build_autoencoder():
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=IMG_SIZE + (3,)),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
    ])

    decoder = tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same"),
        tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(3, 3, activation="sigmoid", padding="same"),
    ])

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    encoded = encoder(inputs)
    decoded = decoder(encoded)

    autoencoder = tf.keras.Model(inputs, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder

def main():
    Path("models").mkdir(exist_ok=True)

    train_ds = load_images("data/normal")
    model = build_autoencoder()

    model.fit(train_ds, epochs=20)
    model.save("models/autoencoder_defect.h5")

if __name__ == "__main__":
    main()