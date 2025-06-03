import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

df = pd.read_csv("combined_results.csv")
df["binary_label"] = df["label"].apply(lambda x: 1 if x == "fake" else 0)

X = df[["entropy", "fourier", "color", "metadata", "artifacts"]]
y = df["binary_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

models = {
    "SVM (RBF)": SVC(probability=True, kernel="rbf"),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier()
}

plt.figure(figsize=(8, 6))

# Отдельные методы
for method in X.columns:
    fpr, tpr, _ = roc_curve(df["binary_label"], df[method])
    auc = roc_auc_score(df["binary_label"], df[method])
    plt.plot(fpr, tpr, label=f"{method} (AUC={auc:.2f})")


# ML модели
for name, model in models.items():
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})", linewidth=2)
    print(f"AUC({name}): {auc:.2f}")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (ML Models)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_ml_combined.png")
plt.show()
