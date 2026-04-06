import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

IMG_SIZE = (160, 160)
CLASS_NAMES = ["Mask", "No Mask"]

def predict_image(model, img_path):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, axis=0)

    prob = model.predict(img_array, verbose=0)[0][0]

    if prob < 0.5:
        label = CLASS_NAMES[0]
        confidence = 1 - prob
    else:
        label = CLASS_NAMES[1]
        confidence = prob

    return label, confidence

def main():
    model = tf.keras.models.load_model("models/face_mask_detector.keras")

    test_images = list(Path("data").rglob("*.jpg"))[:4]

    plt.figure(figsize=(8, 8))
    for i, img_path in enumerate(test_images):
        label, conf = predict_image(model, img_path)
        plt.subplot(2, 2, i + 1)
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.title(f"{label} ({conf:.2f})")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("reports/figures/inference_demo.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()