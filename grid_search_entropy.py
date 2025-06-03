import os
import csv
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from The_entropy_complexity_method import analyze_entropy_complexity

# Загрузка изображений
def process_folder(folder_path, label, tolerance, radius):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            try:
                prob = analyze_entropy_complexity(image_path, tolerance=tolerance, radius=radius, show_images=False)
                results.append((filename, label, prob))
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
    return results

# Маппинг меток в числа
label_map = {'real_phone': 0, 'real_net': 0, 'fake': 1}

# Параметры для перебора
tolerances = [0.05, 0.1, 0.15, 0.2]
radii = [3, 5, 7, 9]

# Пути к данным
folders = {
    'real_phone': 'real_phone',
    'real_net': 'real_net',
    'fake': 'fake'
}

results = []

# Грид-серч
for tol in tqdm(tolerances, desc="Tolerance grid"):
    for rad in radii:
        all_data = []
        for label, path in folders.items():
            all_data.extend(process_folder(path, label, tol, rad))

        # Преобразуем в датафрейм
        df = pd.DataFrame(all_data, columns=["filename", "label", "prob_fake"])
        df["true_label"] = df["label"].map(label_map)
        df["pred_label"] = df["prob_fake"].apply(lambda x: 1 if x >= 0.5 else 0)

        # Метрики
        try:
            auc_score = roc_auc_score(df["true_label"], df["prob_fake"])
            f1 = f1_score(df["true_label"], df["pred_label"])
            acc = accuracy_score(df["true_label"], df["pred_label"])
        except:
            auc_score = f1 = acc = 0

        results.append({
            "tolerance": tol,
            "radius": rad,
            "AUC": round(auc_score, 4),
            "F1": round(f1, 4),
            "Accuracy": round(acc, 4)
        })

# Сохраняем результаты
df_out = pd.DataFrame(results)
df_out.to_csv("grid_search_results.csv", index=False)
print("\n✅ Grid search results saved to grid_search_results.csv")
print(f"t={tol} r={rad} ✓ {len(all_data)} images processed")

