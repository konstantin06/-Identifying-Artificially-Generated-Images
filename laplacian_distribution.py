import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_mean_laplacian(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return np.mean(np.abs(laplacian))

folders = {
    'real_phone': 'real_phone',
    'real_net': 'real_net',
    'fake': 'fake'
}

data = []

for label, path in folders.items():
    for filename in os.listdir(path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(path, filename)
            mval = compute_mean_laplacian(img_path)
            if mval:
                data.append({'filename': filename, 'label': label, 'mean_laplacian': mval})

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='mean_laplacian', hue='label', bins=30, kde=True, stat='density', palette='Set2')
plt.title("Распределение среднего значения Лапласиана")
plt.xlabel("mean(abs(Laplacian))")
plt.grid(True)
plt.tight_layout()
plt.show()
