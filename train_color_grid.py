import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from Color_Distribution_method import analyze_color_raw
from tqdm import tqdm

folders = {
    "real_phone": "real_phone",
    "real_net": "real_net",
    "fake": "fake"
}

data = []

print("📥 Извлекаем признаки цветовой энтропии и std...")
for label, folder in folders.items():
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, fname)
            try:
                ent, std = analyze_color_raw(path)
                data.append({
                    "filename": fname,
                    "label": label,
                    "binary_label": 1 if label == "fake" else 0,
                    "entropy": ent,
                    "std_cbcr": std
                })
            except:
                continue

df = pd.DataFrame(data)

# === Подбор min/max для двух признаков
results = []

print("⚙️ Подбор параметров нормализации...")
for e_min in np.linspace(3.0, 5.0, 10):
    for e_max in np.linspace(5.1, 7.0, 10):
        if e_max <= e_min:
            continue
        for s_min in np.linspace(10, 40, 10):
            for s_max in np.linspace(41, 90, 10):
                if s_max <= s_min:
                    continue

                # нормализация
                e_score = (df["entropy"] - e_min) / (e_max - e_min)
                s_score = 1 - (df["std_cbcr"] - s_min) / (s_max - s_min)  # инвертируем: низкая std = подозрительно

                e_score = np.clip(e_score, 0, 1)
                s_score = np.clip(s_score, 0, 1)

                combined = 0.5 * e_score + 0.5 * s_score  # можно потом подобрать веса

                auc = roc_auc_score(df["binary_label"], combined)
                results.append({
                    "entropy_min": e_min,
                    "entropy_max": e_max,
                    "std_min": s_min,
                    "std_max": s_max,
                    "AUC": auc
                })

df_results = pd.DataFrame(results)
best = df_results.loc[df_results["AUC"].idxmax()]
print("\n🏆 Лучшая конфигурация:")
print(best)

# Сохраняем
import json
with open("best_color_params.json", "w") as f:
    json.dump(best.to_dict(), f, indent=4)

df_results.to_csv("color_grid_results.csv", index=False)
