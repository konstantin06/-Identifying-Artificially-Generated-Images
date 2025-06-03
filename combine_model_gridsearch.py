import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from itertools import product

df = pd.read_csv("combined_results.csv")
df["binary_label"] = df["label"].apply(lambda x: 1 if x == "fake" else 0)

features = ["entropy", "fourier", "color", "metadata", "artifacts"]
X = df[features]
y = df["binary_label"]

# Сетка весов (на каждую фичу)
grid = list(product([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0], repeat=len(features)))
grid = [g for g in grid if sum(g) > 0]  # исключаем нулевую сумму

best_auc = 0
best_weights = None
for weights in grid:
    combo = np.dot(X.values, np.array(weights))
    combo = combo / np.max(combo)  # нормализация
    auc = roc_auc_score(y, combo)
    if auc > best_auc:
        best_auc = auc
        best_weights = weights

print("=== Grid Search Best Weights ===")
print(dict(zip(features, best_weights)))
print(f"AUC: {best_auc:.3f}")

# ROC-кривые
plt.figure(figsize=(8, 6))
for method in features:
    fpr, tpr, _ = roc_curve(y, df[method])
    auc = roc_auc_score(y, df[method])
    plt.plot(fpr, tpr, label=f"{method} (AUC={auc:.2f})")

combo = np.dot(X.values, np.array(best_weights))
combo = combo / np.max(combo)
fpr, tpr, _ = roc_curve(y, combo)
plt.plot(fpr, tpr, label=f"Combined (AUC={best_auc:.3f})", linewidth=2, color='black')
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Grid Search)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_gridsearch_combined.png")
plt.show()
