import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("grid_search_results.csv")

# Строим тепловую карту по AUC
heatmap_data = df.pivot(index="radius", columns="tolerance", values="AUC")

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("AUC по параметрам (tolerance × radius)")
plt.ylabel("Radius")
plt.xlabel("Tolerance")
plt.tight_layout()
plt.show()
