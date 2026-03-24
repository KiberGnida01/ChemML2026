import numpy as np
import pandas as pd

np.random.seed(42)
n_molecules = 500

# Молекулярные дескрипторы (типичные для QSPR-моделей)
data = {
    "MolWeight": np.random.uniform(100, 600, n_molecules),       # Молекулярная масса
    "LogP": np.random.normal(2.5, 1.5, n_molecules),             # Коэф. распределения октанол/вода
    "HBD": np.random.randint(0, 6, n_molecules),                 # Доноры водородных связей
    "HBA": np.random.randint(0, 10, n_molecules),                # Акцепторы водородных связей
    "TPSA": np.random.uniform(20, 150, n_molecules),             # Топологическая полярная площадь
    "RotBonds": np.random.randint(0, 12, n_molecules),           # Вращаемые связи
    "AromaticRings": np.random.randint(0, 5, n_molecules),       # Ароматические кольца
    "HeavyAtoms": np.random.randint(7, 45, n_molecules),         # Тяжёлые атомы
    "FormalCharge": np.random.choice([-1, 0, 0, 0, 1], n_molecules),  # Формальный заряд
}

df = pd.DataFrame(data)

# Целевая переменная: логарифм растворимости (logS)
# Упрощённая зависимость, вдохновлённая уравнением Ясуды-Шинкая
df["LogS"] = (
    -0.01 * df["MolWeight"]
    - 0.5 * df["LogP"]
    + 0.3 * df["HBD"]
    + 0.2 * df["HBA"]
    - 0.005 * df["TPSA"]
    - 0.1 * df["RotBonds"]
    + np.random.normal(0, 0.5, n_molecules)  # шум
    + 0.4 * df["FormalCharge"]
)

df.to_csv("data/molecules.csv", index=False)
print(f"Датасет сохранён: {len(df)} молекул, {len(df.columns) - 1} дескрипторов")
print(f"\nПервые 5 строк:\n{df.head()}")
print(f"\nСтатистика LogS:\n{df['LogS'].describe()}")