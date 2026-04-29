import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def adaptive_thresholding(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def region_growing(img, seed=None, threshold=15):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if seed is None:
        seed = (img.shape[0] // 2, img.shape[1] // 2)

    h, w = img.shape
    segmented = np.zeros_like(img)
    visited = np.zeros_like(img, dtype=bool)

    list_pixels = [seed]
    visited[seed] = True
    start_val = int(img[seed])

    while list_pixels:
        x, y = list_pixels.pop(0)
        segmented[x, y] = 255

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                if abs(int(img[nx, ny]) - start_val) <= threshold:
                    visited[nx, ny] = True
                    list_pixels.append((nx, ny))

    return segmented

def run_experiment(image_names):
    existing_images = [img for img in image_names if os.path.exists(img)]

    if not existing_images:
        print("Kļūda: Neviens no norādītajiem failiem netika atrasts!")
        print("Lūdzu, augšupielādējiet failus: peleks.jpg, dzivnieks.jpg, random.jpg")
        return

    fig, axes = plt.subplots(len(existing_images), 3, figsize=(15, 5 * len(existing_images)))

    if len(existing_images) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, name in enumerate(existing_images):
        img = cv2.imread(name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = adaptive_thresholding(gray)
        grown = region_growing(gray)

        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Oriģināls: {name}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(thresh, cmap='gray')
        axes[i, 1].set_title("Adaptīvā sliekšņošana")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(grown, cmap='gray')
        axes[i, 2].set_title("Apgabalu audzēšana")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

image_files = ['peleks.jpg', 'dzivnieks.jpg', 'random.jpg']
run_experiment(image_files)
