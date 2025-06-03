import os
import numpy as np
import pandas as pd
from The_entropy_complexity_method import analyze_entropy_complexity_raw
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

folders = {
    'real_phone': 'real_phone',
    'real_net': 'real_net',
    'fake': 'fake'
}

raw_data = []

# === Сбор сырых данных по каждому изображению ===
print("📥 Сбор значений...")
for label, path in folders.items():
    for fname in os.listdir(path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            fpath = os.path.join(path, fname)
            try:
                for tolerance in [0.10, 0.15, 0.20]:
                    for radius in [5, 7, 9]:
                        raw = analyze_entropy_complexity_raw(fpath, radius=radius, tolerance=tolerance)
                        raw_data.append({
                            "filename": fname,
                            "label": label,
                            "binary_label": 1 if label == "fake" else 0,
                            "tolerance": tolerance,
                            "radius": radius,
                            "mean_cluster_size": raw
                        })
            except Exception as e:
                print(f"❌ {fname}: {e}")

df_raw = pd.DataFrame(raw_data)

# === Обучаем min_size и max_size для каждого сочетания radius + tolerance ===
print("⚙️ Подбор параметров...")
results = []

for (tol, rad), group in tqdm(df_raw.groupby(["tolerance", "radius"])):
    for min_s in range(10, 100, 5):
        for max_s in range(min_s + 50, 400, 10):
            scaled = 1 - np.clip((group["mean_cluster_size"] - min_s) / (max_s - min_s), 0, 1)
            auc = roc_auc_score(group["binary_label"], scaled)
            results.append({
                "tolerance": tol,
                "radius": rad,
                "min_size": min_s,
                "max_size": max_s,
                "AUC": auc
            })

df_results = pd.DataFrame(results)
best_row = df_results.loc[df_results["AUC"].idxmax()]

print("\n🏆 Лучшие параметры:")
print(best_row)

# === Сохраним всё в таблицу
df_results.to_csv("entropy_grid_full_results.csv", index=False)

# === Сохраняем лучшие параметры в JSON
import json
with open("best_entropy_params.json", "w") as f:
    json.dump({
        "tolerance": float(best_row["tolerance"]),
        "radius": int(best_row["radius"]),
        "min_size": int(best_row["min_size"]),
        "max_size": int(best_row["max_size"])
    }, f, indent=4)

print("✅ Параметры сохранены в best_entropy_params.json")

