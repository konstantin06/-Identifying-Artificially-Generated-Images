import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import seaborn as sns

def compute_entropy(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    entropies = []
    for i in [1, 2]:  # Cr, Cb
        hist = cv2.calcHist([image_ycrcb], [i], None, [256], [0, 256])
        hist = hist.ravel()
        hist /= hist.sum()
        ent = entropy(hist, base=2)
        entropies.append(ent)
    return np.mean(entropies)

# Папки с изображениями
folders = {
    'real_phone': 'real_phone',
    'real_net': 'real_net',
    'fake': 'fake'
}

data = []

for label, path in folders.items():
    for filename in os.listdir(path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path_full = os.path.join(path, filename)
            ent = compute_entropy(path_full)
            if ent:
                data.append({
                    'filename': filename,
                    'label': label,
                    'entropy': ent
                })

df = pd.DataFrame(data)

# 📈 График
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='entropy', hue='label', bins=30, kde=True, stat='density', common_norm=False, palette='Set1')
plt.title("Распределение средней энтропии каналов Cr и Cb")
plt.xlabel("Entropy")
plt.ylabel("Плотность")
plt.grid(True)
plt.tight_layout()
plt.show()
