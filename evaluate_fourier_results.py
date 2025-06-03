import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    classification_report,
    confusion_matrix
)

# === 1. Загрузка данных ===
df = pd.read_csv("results_fourier.csv")

# === 2. Бинарная метка: fake = 1, остальное = 0 ===
df["binary_label"] = df["label"].apply(lambda x: 1 if x == "fake" else 0)

# === 3. Оптимальный порог по Youden J ===
fpr, tpr, thresholds = roc_curve(df["binary_label"], df["prob_fake"])
j_scores = tpr - fpr
j_best_idx = np.argmax(j_scores)
best_threshold = thresholds[j_best_idx]
acc_best = accuracy_score(df["binary_label"], df["prob_fake"] >= best_threshold)

print(f"🔍 Лучший порог по Youden J: {best_threshold:.3f}")
print(f"🎯 Accuracy @ best threshold: {acc_best:.3f}\n")

# === 4. ROC-кривые по каждому классу ===
classes = ['real_phone', 'real_net', 'fake']
colors = ['green', 'orange', 'red']

plt.figure(figsize=(10, 6))
for cls, color in zip(classes, colors):
    y_true = (df["label"] == cls).astype(int)
    if cls == "fake":
        y_score = df["prob_fake"]
    else:
        y_score = 1 - df["prob_fake"]
    fpr_c, tpr_c, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr_c, tpr_c)
    plt.plot(fpr_c, tpr_c, label=f"{cls} (AUC = {roc_auc:.2f})", color=color)

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.title("ROC-кривые по классам (Fourier)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# === 5. Мультиклассовая классификация ===

# Преобразуем метки в числа
label_map = {"real_phone": 0, "real_net": 1, "fake": 2}
df["true_multi"] = df["label"].map(label_map)

# Предсказание: fake если prob_fake >= threshold
df["pred_binary"] = (df["prob_fake"] >= best_threshold).astype(int)

# Мультикласс: делим real на phone/net по под-порогам
df["pred_multi"] = 0  # real_phone по умолчанию
df.loc[(df["pred_binary"] == 0) & (df["prob_fake"] > 0.004), "pred_multi"] = 1  # real_net
df.loc[df["pred_binary"] == 1, "pred_multi"] = 2  # fake

print("📋 Classification Report (Мультикласс):")
print(classification_report(df["true_multi"], df["pred_multi"],
                            target_names=["real_phone", "real_net", "fake"]))

print("🔁 Confusion Matrix:")
print(confusion_matrix(df["true_multi"], df["pred_multi"]))
