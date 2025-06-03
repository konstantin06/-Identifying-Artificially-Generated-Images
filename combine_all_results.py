import pandas as pd

# Загрузим CSV-файлы
entropy = pd.read_csv("results_entropy.csv").rename(columns={"prob_fake": "entropy"})
fourier = pd.read_csv("results_fourier.csv").rename(columns={"prob_fake": "fourier"})
color = pd.read_csv("results_color.csv").rename(columns={"prob_fake": "color"})
metadata = pd.read_csv("results_metadata.csv").rename(columns={"prob_fake": "metadata"})
artifacts = pd.read_csv("results_artifacts.csv").rename(columns={"prob_fake": "artifacts"})

# Объединяем по filename + label
df = entropy \
    .merge(fourier, on=["filename", "label"]) \
    .merge(color, on=["filename", "label"]) \
    .merge(metadata, on=["filename", "label"]) \
    .merge(artifacts, on=["filename", "label"])

# Сохраняем
df.to_csv("combined_results.csv", index=False)
print("✅ Объединённая таблица сохранена: combined_results.csv")
