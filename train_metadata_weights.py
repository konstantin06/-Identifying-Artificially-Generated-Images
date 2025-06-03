import os
import json
import pandas as pd
from sklearn.metrics import roc_auc_score
from itertools import product
from Metadata_analysis_method import analyze_metadata_features

# Папки
folders = {
    "real_phone": "real_phone",
    "real_net": "real_net",
    "fake": "fake"
}

# Сбор признаков
data = []
for label, folder in folders.items():
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, fname)
            try:
                feats = analyze_metadata_features(path)
                feats["filename"] = fname
                feats["label"] = label
                feats["binary_label"] = 1 if label == "fake" else 0
                data.append(feats)
            except:
                continue

df = pd.DataFrame(data)

# Все признаки
features = ["has_exif", "has_camera_info", "suspicious_software",
            "low_dpi", "weird_size", "has_icc", "suspicious_quant"]

weight_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
grid = list(product(weight_values, repeat=len(features)))

best_auc = 0
best_weights = None

print("🔁 Перебор весов...")
for weights in grid:
    score = sum(df[f] * w for f, w in zip(features, weights))
    prob_fake = score.clip(0, 1)
    auc = roc_auc_score(df["binary_label"], prob_fake)
    if auc > best_auc:
        best_auc = auc
        best_weights = weights

weights_dict = dict(zip(features, best_weights))
with open("best_metadata_weights.json", "w") as f:
    json.dump(weights_dict, f, indent=4)

print("🏆 Лучшие веса:", weights_dict)
print(f"AUC = {best_auc:.3f}")
