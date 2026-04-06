import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models
from pathlib import Path

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    x_train, y_train, x_test, y_test = load_data()

    model = build_model()
    model.summary()

    model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test accuracy:", test_acc)

    Path("models").mkdir(exist_ok=True)
    model.save("models/cifar10_cnn.h5")

if __name__ == "__main__":
    main()