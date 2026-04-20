import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

def display_images(images, titles):
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        if len(img.shape) == 3:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()

def apply_canny(image, low_threshold, high_threshold):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return cv2.Canny(gray, low_threshold, high_threshold)

def apply_roberts(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    gray = np.float32(gray)
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    padded = np.pad(gray, ((0, 1), (0, 1)), mode='edge')
    rx = np.zeros_like(gray)
    ry = np.zeros_like(gray)
    for r in range(gray.shape[0]):
        for c in range(gray.shape[1]):
            region = padded[r:r+2, c:c+2]
            rx[r, c] = np.sum(region * roberts_x)
            ry[r, c] = np.sum(region * roberts_y)
    magnitude = np.sqrt(rx**2 + ry**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(magnitude)

if __name__ == '__main__':
    try:
        img1 = cv2.imread('att1.jpg')
        if img1 is None: raise FileNotFoundError
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    except FileNotFoundError:
        img1 = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img1, (50, 50), (250, 250), (255, 100, 100), -1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    try:
        img2 = cv2.imread('att2.jpg')
        if img2 is None: raise FileNotFoundError
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    except FileNotFoundError:
        img2 = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(img2, (150, 150), 100, (100, 255, 100), -1)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    noise_img1 = np.array(255*random_noise(img1, mode='s&p', amount=0.05), dtype=np.uint8)

    display_images([img1, noise_img1, img2], 
                   ['Originals', 'Ar Salt & Pepper troksni', 'Brivi izvelets'])

    c_low, c_high = 50, 150
    c1 = apply_canny(img1, c_low, c_high)
    cn = apply_canny(noise_img1, c_low, c_high)
    c2 = apply_canny(img2, c_low, c_high)

    display_images([c1, cn, c2], 
                   ['Canny: att1', 'Canny: Troksnis', 'Canny: att2'])

    r1 = apply_roberts(img1)
    rn = apply_roberts(noise_img1)
    r2 = apply_roberts(img2)

    display_images([r1, rn, r2], 
                   ['Roberts: att1', 'Roberts: Troksnis', 'Roberts: att2'])
