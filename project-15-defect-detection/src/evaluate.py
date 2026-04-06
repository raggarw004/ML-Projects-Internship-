import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMG_SIZE = (128, 128)

def load_image(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    img = tf.keras.utils.img_to_array(img) / 255.0
    return img

def anomaly_score(model, img):
    recon = model.predict(img[None, ...])
    return np.mean((img - recon[0]) ** 2)

def main():
    model = tf.keras.models.load_model("models/autoencoder_defect.h5", compile=False)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    normal_imgs = list(Path("data/normal").glob("*.png"))[:3]
    defect_imgs = list(Path("data/defect").glob("*.png"))[:3]

    all_imgs = normal_imgs + defect_imgs

    plt.figure(figsize=(9, 6))
    for i, img_path in enumerate(all_imgs):
        img = load_image(img_path)
        score = anomaly_score(model, img)

        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(f"Score: {score:.4f}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("reports/figures/anomaly_examples.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()