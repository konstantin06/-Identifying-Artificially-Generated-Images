import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("combined_results.csv")
df["binary_label"] = df["label"].apply(lambda x: 1 if x == "fake" else 0)

methods = ["entropy", "fourier", "color", "metadata", "artifacts"]

print("\n📊 Оценка качества отдельных методов:")
for method in methods:
    auc = roc_auc_score(df["binary_label"], df[method])
    threshold = 0.5  # можно отдельно искать best threshold
    pred = (df[method] >= threshold).astype(int)
    acc = accuracy_score(df["binary_label"], pred)
    f1 = f1_score(df["binary_label"], pred)
    print(f"🔹 {method.upper()}: AUC={auc:.3f}, Accuracy={acc:.3f}, F1={f1:.3f}")

# ROC-график
from sklearn.metrics import roc_curve

plt.figure(figsize=(8, 6))
for method in methods:
    fpr, tpr, _ = roc_curve(df["binary_label"], df[method])
    auc = roc_auc_score(df["binary_label"], df[method])
    plt.plot(fpr, tpr, label=f"{method} (AUC={auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривые отдельных методов")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
