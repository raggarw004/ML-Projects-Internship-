import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)
    return img_array

def main():
    model = tf.keras.models.load_model("models/mnist_digit_model.h5")
    image_dir = Path("data/custom_digits")

    for img_path in image_dir.glob("*.png"):
        img = preprocess_image(img_path)
        pred = model.predict(img)
        digit = pred.argmax()

        plt.imshow(img.reshape(28, 28), cmap="gray")
        plt.title(f"Predicted Digit: {digit}")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()