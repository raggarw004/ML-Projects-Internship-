import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def main():
    model = tf.keras.models.load_model("models/cifar10_cnn.h5")
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test / 255.0

    preds = model.predict(x_test[:9])
    pred_labels = preds.argmax(axis=1)

    plt.figure(figsize=(6,6))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(x_test[i])
        plt.title(
            f"Pred: {CLASS_NAMES[pred_labels[i]]}\nTrue: {CLASS_NAMES[y_test[i][0]]}",
            fontsize=8
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("reports/figures/sample_predictions.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()