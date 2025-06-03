import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv("combined_results.csv")
df["binary_label"] = df["label"].apply(lambda x: 1 if x == "fake" else 0)

X = df[["entropy", "fourier", "color", "metadata", "artifacts"]]
y = df["binary_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_score = model.predict_proba(X_test)[:, 1]

print("=== Logistic Regression Report ===")
print(classification_report(y_test, model.predict(X_test)))
print(f"AUC: {roc_auc_score(y_test, y_score):.3f}")

# ROC curves
plt.figure(figsize=(8, 6))
for method in X.columns:
    fpr, tpr, _ = roc_curve(df["binary_label"], df[method])
    auc = roc_auc_score(df["binary_label"], df[method])
    plt.plot(fpr, tpr, label=f"{method} (AUC={auc:.2f})")

fpr, tpr, _ = roc_curve(y_test, y_score)
plt.plot(fpr, tpr, label=f"Combined (AUC={roc_auc_score(y_test, y_score):.3f})", linewidth=2, color='black')
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (LogReg)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_logreg_combined.png")
plt.show()
