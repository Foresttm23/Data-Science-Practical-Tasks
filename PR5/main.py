import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_data(df, columns):
    """
    Нормалізація: приведення значень до діапазону [0, 1].
    """
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[columns] = scaler.fit_transform(df[columns])
    return df_norm

def standardize_data(df, columns):
    """
    Стандартизація: приведення до середнього 0 та стандартного відхилення 1.
    """
    scaler = StandardScaler()
    df_std = df.copy()
    df_std[columns] = scaler.fit_transform(df[columns])
    return df_std

def create_sample_data():
    np.random.seed(42)
    data = {
        'Вік': np.random.randint(18, 65, 100),              # Діапазон [18, 65]
        'Дохід': np.random.normal(50000, 15000, 100),      # Діапазон [~10k, ~90k]
        'Рейтинг': np.random.uniform(1, 5, 100)            # Діапазон [1, 5]
    }
    return pd.DataFrame(data)

df = create_sample_data()
cols_to_scale = ['Вік', 'Дохід', 'Рейтинг']

df_normalized = normalize_data(df, cols_to_scale)
df_standardized = standardize_data(df, cols_to_scale)

print("Оригінальні дані (перші 5 рядків):")
print(df.head())
print("\nСтатистика оригінальних даних (Mean, Std):")
print(df.agg(['mean', 'std']).round(2))

print("\n" + "="*50)
print("Статистика після Нормалізації (Min, Max):")
print(df_normalized[cols_to_scale].agg(['min', 'max']).round(2))

print("\n" + "="*50)
print("Статистика після Стандартизації (Mean, Std):")
print(df_standardized[cols_to_scale].agg(['mean', 'std']).round(2))

# 4. Візуалізація для звіту
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.kdeplot(df['Дохід'], ax=axes[0], fill=True, color='blue').set_title('Оригінальний Дохід')
sns.kdeplot(df_normalized['Дохід'], ax=axes[1], fill=True, color='green').set_title('Нормалізований (Min-Max)')
sns.kdeplot(df_standardized['Дохід'], ax=axes[2], fill=True, color='red').set_title('Стандартизований (Z-score)')

plt.tight_layout()
plt.show()