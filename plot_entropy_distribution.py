import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка результатов
df = pd.read_csv('results_entropy.csv')

# Преобразование label в читаемый вид
df['label'] = df['label'].map({
    'real_net': 'Real (Internet)',
    'real_phone': 'Real (Phone)',
    'fake': 'Fake'
})

# Построение гистограммы
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='prob_fake', hue='label', bins=20, kde=True, palette='Set1', stat='density', common_norm=False)
plt.title("Распределение вероятностей подделки (prob_fake)")
plt.xlabel("prob_fake")
plt.ylabel("Плотность")
plt.grid(True)
plt.tight_layout()
plt.show()
