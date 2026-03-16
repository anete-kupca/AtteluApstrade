import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def logarithmic_correction(img):
    img_float = img.astype(float)
    c = 255 / np.log(1 + np.max(img_float))
    log_img = c * (np.log(1 + np.max(img_float) * 0.0001 + img_float))
    log_img = c * (np.log(1 + img_float))
    return np.array(log_img, dtype=np.uint8)

def linear_stretching(img):
    img_float = img.astype(float)
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    if max_val == min_val:
        return img
    stretch_img = (img_float - min_val) * (255 / (max_val - min_val))
    return np.array(stretch_img, dtype=np.uint8)

def apply_to_channels(img, method):
    channels = cv2.split(img)
    corrected_channels = [method(ch) for ch in channels]
    return cv2.merge(corrected_channels)

def show_results(original, log_corr, lin_stretch, title):
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    imgs = [original, log_corr, lin_stretch]
    labels = ['Original', 'Logarithmic', 'Linear Stretching']
    for i in range(3):
        axes[i, 0].imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"{title}: {labels[i]}")
        axes[i, 0].axis('off')
        colors = ('r', 'g', 'b')
        for j, col in enumerate(colors):
            hist = cv2.calcHist([imgs[i]], [j], None, [256], [0, 256])
            axes[i, 1].plot(hist, color=col)
        axes[i, 1].set_title(f"Histogram: {labels[i]}")
        axes[i, 1].set_xlim([0, 256])
    plt.tight_layout()
    plt.show()

def process_and_display(file_path, label):
    if not os.path.exists(file_path):
        return
    img = cv2.imread(file_path)
    if img is None:
        return
    log_img = apply_to_channels(img, logarithmic_correction)
    lin_img = apply_to_channels(img, linear_stretching)
    show_results(img, log_img, lin_img, label)

if __name__ == "__main__":
    tasks = [
        ("piemers1.jpg", "Partumsots"),
        ("piemers2.jpg", "Pargaismots"),
        ("piemers3.jpg", "Pelecigs")
    ]
    for file_name, description in tasks:
        process_and_display(file_name, description)
