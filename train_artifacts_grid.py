import os
import numpy as np
import pandas as pd
from Generation_artifacts_method import analyze_artifacts_raw
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

folders = {
    "real_phone": "real_phone",
    "real_net": "real_net",
    "fake": "fake"
}

data = []

print("📥 Сбор mean_laplacian...")
for label, folder in folders.items():
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, fname)
            for blur in [0, 3, 5]:
                try:
                    val = analyze_artifacts_raw(path, blur_kernel=blur)
                    data.append({
                        "filename": fname,
                        "label": label,
                        "binary_label": 1 if label == "fake" else 0,
                        "mean_lap": val,
                        "blur_kernel": blur
                    })
                except:
                    continue

df = pd.DataFrame(data)

# === Подбор min/max
results = []

print("🔧 Подбор нормализации...")
for blur in df["blur_kernel"].unique():
    subset = df[df["blur_kernel"] == blur]
    for min_v in np.linspace(3.0, 6.0, 10):
        for max_v in np.linspace(6.1, 12.0, 10):
            if max_v <= min_v:
                continue
            scaled = (subset["mean_lap"] - min_v) / (max_v - min_v)
            scaled = np.clip(scaled, 0, 1)
            auc = roc_auc_score(subset["binary_label"], scaled)
            results.append({
                "blur_kernel": blur,
                "min_val": min_v,
                "max_val": max_v,
                "AUC": auc
            })

df_results = pd.DataFrame(results)
best = df_results.loc[df_results["AUC"].idxmax()]
print("\n🏆 Лучшая конфигурация:")
print(best)

# Сохраняем
import json
with open("best_artifacts_params.json", "w") as f:
    json.dump(best.to_dict(), f, indent=4)

df_results.to_csv("artifacts_grid_results.csv", index=False)
