import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def add_salt_and_pepper(image, amount=0.02):
    row, col, ch = image.shape
    out = np.copy(image)
    num_salt = np.ceil(amount * image.size * 0.5 / ch)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    out[tuple(coords)] = 255
    num_pepper = np.ceil(amount * image.size * 0.5 / ch)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    out[tuple(coords)] = 0
    return out

def add_gaussian_noise(image, sigma=25):
    gauss = np.random.normal(0, sigma, image.shape)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_combined_filter(image, m_k=3, g_k=(3, 3)):
    median = cv2.medianBlur(image, m_k)
    combined = cv2.GaussianBlur(median, g_k, 0)
    return combined

image_dir = "/content/"
image_names = [f"piemers{i}.jpg" for i in range(1, 6)]
image_paths = [os.path.join(image_dir, name) for name in image_names]

noises_config = [
    ("Sāls & Pipari", lambda img: add_salt_and_pepper(img, 0.05)),
    ("Intensīvs Sāls & Pipari", lambda img: add_salt_and_pepper(img, 0.15)),
    ("Viegls Troksnis", lambda img: add_gaussian_noise(img, 15)),
    ("Vidējs Troksnis", lambda img: add_gaussian_noise(img, 30)),
    ("Vidējs + Sāls & Pipari", lambda img: add_salt_and_pepper(add_gaussian_noise(img, 25), 0.05))
]

params = [
    {"m_k": 3, "g_k": (3, 3)},
    {"m_k": 5, "g_k": (3, 3)},
    {"m_k": 3, "g_k": (5, 5)},
    {"m_k": 3, "g_k": (5, 5)},
    {"m_k": 5, "g_k": (5, 5)}
]

fig, axes = plt.subplots(5, 3, figsize=(15, 20))
plt.subplots_adjust(hspace=0.4)

for i in range(5):
    path = image_paths[i]
    noise_name, noise_func = noises_config[i]
    p = params[i]
    
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        noisy = img
        filtered = img
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        noisy = noise_func(img)
        filtered = apply_combined_filter(noisy, p["m_k"], p["g_k"])
    
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Oriģināls {i+1}")
    axes[i, 1].imshow(noisy)
    axes[i, 1].set_title(f"Trokšņains: {noise_name}")
    axes[i, 2].imshow(filtered)
    axes[i, 2].set_title(f"Attīrīts (m={p['m_k']}, g={p['g_k']})")
    
    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()
plt.show()
