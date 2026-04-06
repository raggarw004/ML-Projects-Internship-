import tensorflow as tf
from pathlib import Path

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
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
    model.fit(
        x_train, y_train,
        epochs=5,
        validation_split=0.1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test accuracy:", test_acc)

    Path("models").mkdir(exist_ok=True)
    model.save("models/mnist_digit_model.h5")

if __name__ == "__main__":
    main()