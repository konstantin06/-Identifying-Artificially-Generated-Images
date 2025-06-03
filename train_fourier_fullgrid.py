import os
import numpy as np
import pandas as pd
from Fourier_Transform_method import analyze_fourier_raw
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

folders = {
    "real_phone": "real_phone",
    "real_net": "real_net",
    "fake": "fake"
}

data = []

print("📥 Извлекаем признаки...")
for label, folder in folders.items():
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, fname)
            for logmode in [True, False]:
                for low_cut in [10, 15, 20, 25]:
                    for high_cut in [40, 60, 80]:
                        try:
                            ratio = analyze_fourier_raw(path, low_cut, high_cut, use_log=logmode)
                            data.append({
                                "filename": fname,
                                "label": label,
                                "binary_label": 1 if label == "fake" else 0,
                                "low_cut": low_cut,
                                "high_cut": high_cut,
                                "logmode": logmode,
                                "ratio": ratio
                            })
                        except:
                            continue

df = pd.DataFrame(data)

# === Подбор нормализации для каждой комбинации
results = []

print("⚙️ Подбираем нормализацию...")
for (lc, hc, logmode), group in tqdm(df.groupby(["low_cut", "high_cut", "logmode"])):
    for min_v in np.linspace(0.1, 0.6, 10):
        for max_v in np.linspace(0.7, 1.5, 10):
            if max_v <= min_v:
                continue
            scaled = (group["ratio"] - min_v) / (max_v - min_v)
            scaled = np.clip(scaled, 0, 1)
            auc = roc_auc_score(group["binary_label"], scaled)
            results.append({
                "low_cut": lc,
                "high_cut": hc,
                "logmode": logmode,
                "min_val": min_v,
                "max_val": max_v,
                "AUC": auc
            })

df_results = pd.DataFrame(results)
best = df_results.loc[df_results["AUC"].idxmax()]
print("\n🏆 Лучшая конфигурация:")
print(best)

# Сохраняем в JSON
import json
with open("best_fourier_params.json", "w") as f:
    json.dump(best.to_dict(), f, indent=4)

df_results.to_csv("fourier_grid_results.csv", index=False)
