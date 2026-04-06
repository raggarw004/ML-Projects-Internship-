import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

IMG_SIZE = (160, 160)
CLASS_NAMES = ["Mask", "No Mask"]

def predict_image(model, img_path):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]
    label = CLASS_NAMES[int(prob >= 0.5)]
    confidence = prob if prob >= 0.5 else 1 - prob

    return label, confidence

def main():
    model = tf.keras.models.load_model("models/face_mask_detector.keras")

    sample_dir = Path("reports/figures")
    sample_dir.mkdir(parents=True, exist_ok=True)

    test_images = list(Path("data").rglob("*.jpg"))[:4]

    plt.figure(figsize=(8, 8))
    for i, img_path in enumerate(test_images):
        label, conf = predict_image(model, img_path)
        plt.subplot(2, 2, i+1)
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.title(f"{label} ({conf:.2f})")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("reports/figures/inference_demo.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()