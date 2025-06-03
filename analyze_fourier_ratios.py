import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Функция: вычисление ratio без нормализации
def compute_fourier_ratio(image_path, center_size=20):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    log_spectrum = np.log1p(np.abs(f_shift))

    h, w = log_spectrum.shape
    center = log_spectrum[h//2 - center_size//2:h//2 + center_size//2,
                          w//2 - center_size//2:w//2 + center_size//2]
    central_energy = np.sum(center)
    total_energy = np.sum(log_spectrum)
    ratio = central_energy / total_energy
    return ratio

# Сканирование всех папок
folders = {
    'real_phone': 'real_phone',
    'real_net': 'real_net',
    'fake': 'fake'
}

data = []

for label, path in folders.items():
    for filename in os.listdir(path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(path, filename)
            ratio = compute_fourier_ratio(image_path)
            if ratio is not None:
                data.append({
                    'filename': filename,
                    'label': label,
                    'ratio': ratio
                })

# Создаём DataFrame
df = pd.DataFrame(data)

# 📈 Строим гистограмму
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='ratio', hue='label', bins=30, kde=True, palette='Set1', stat='density', common_norm=False)
plt.title("Распределение отношения центральной энергии к общей (Fourier ratio)")
plt.xlabel("Central energy / Total energy")
plt.ylabel("Плотность")
plt.grid(True)
plt.tight_layout()
plt.show()
