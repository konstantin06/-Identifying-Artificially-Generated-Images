import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("combined_results.csv")
df["binary_label"] = df["label"].apply(lambda x: 1 if x == "fake" else 0)

X = df[["entropy", "fourier", "color", "metadata", "artifacts"]]
y = df["binary_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]

print("\n🧠 Комбинированная модель:")
print(classification_report(y_test, y_pred))
print(f"AUC = {roc_auc_score(y_test, y_score):.3f}")
print("Коэффициенты признаков:")
for feat, coef in zip(X.columns, model.coef_[0]):
    print(f"  {feat}: {coef:.3f}")
