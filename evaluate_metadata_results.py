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

# === Загрузка ===
df = pd.read_csv("results_metadata.csv")
df["binary_label"] = df["label"].apply(lambda x: 1 if x == "fake" else 0)

# === Оптимальный порог ===
fpr, tpr, thresholds = roc_curve(df["binary_label"], df["prob_fake"])
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
acc_best = accuracy_score(df["binary_label"], df["prob_fake"] >= best_threshold)

print(f"🔍 Best threshold (Youden): {best_threshold:.3f}")
print(f"🎯 Accuracy @ best threshold: {acc_best:.3f}")

# === ROC по классам ===
classes = ["real_phone", "real_net", "fake"]
colors = ["green", "orange", "red"]

plt.figure(figsize=(10, 6))
for cls, color in zip(classes, colors):
    y_true = (df["label"] == cls).astype(int)
    y_score = 1 - df["prob_fake"] if cls != "fake" else df["prob_fake"]
    fpr_c, tpr_c, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr_c, tpr_c)
    plt.plot(fpr_c, tpr_c, label=f"{cls} (AUC = {auc_score:.2f})", color=color)

plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC-кривые по классам (Metadata)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# === Мультиклассовая классификация ===
label_map = {"real_phone": 0, "real_net": 1, "fake": 2}
df["true_multi"] = df["label"].map(label_map)
df["pred_binary"] = (df["prob_fake"] >= best_threshold).astype(int)

df["pred_multi"] = 0
df.loc[(df["pred_binary"] == 0) & (df["prob_fake"] > 0.6), "pred_multi"] = 1
df.loc[df["pred_binary"] == 1, "pred_multi"] = 2

print("\n📋 Classification report:")
print(classification_report(df["true_multi"], df["pred_multi"], target_names=["real_phone", "real_net", "fake"]))
print("🔁 Confusion matrix:")
print(confusion_matrix(df["true_multi"], df["pred_multi"]))
