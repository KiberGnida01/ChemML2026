"""Обучение модели предсказания растворимости молекул."""

import json
import os

import matplotlib
matplotlib.use("Agg")  # неинтерактивный бэкенд (для работы в Docker без дисплея)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# --- Загрузка данных ---

df = pd.read_csv("data/molecules.csv")
print(f"Загружено {len(df)} молекул")

feature_cols = [c for c in df.columns if c != "LogS"]
X = df[feature_cols].values
y = df["LogS"].values

# --- Разделение на train/test ---

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# --- Масштабирование признаков ---

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Обучение нескольких моделей ---

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel="rbf", C=1.0),
}

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
)
grid_search.fit(X_train_scaled, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучший R² (CV): {grid_search.best_score_:.4f}")

models["RandomForest"] = grid_search.best_estimator_

best_params = grid_search.best_params_
with open("results/best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)

results = {}

for name, model in models.items():
    # Кросс-валидация на train
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2")

    # Обучение на полном train и предсказание на test
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    metrics = {
        "R2_test": round(r2_score(y_test, y_pred), 4),
        "MAE_test": round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE_test": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "R2_cv_mean": round(cv_scores.mean(), 4),
        "R2_cv_std": round(cv_scores.std(), 4),
    }
    results[name] = metrics
    print(f"\n{name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

# --- Сохранение результатов ---

os.makedirs("results", exist_ok=True)

# Метрики в JSON
with open("results/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nМетрики сохранены в results/metrics.json")

# --- Визуализация ---

# 1. Сравнение моделей (bar chart)
fig, ax = plt.subplots(figsize=(8, 5))
model_names = list(results.keys())
r2_values = [results[m]["R2_test"] for m in model_names]
mae_values = [results[m]["MAE_test"] for m in model_names]

x = np.arange(len(model_names))
width = 0.35
ax.bar(x - width / 2, r2_values, width, label="R² (test)", color="#2196F3")
ax.bar(x + width / 2, mae_values, width, label="MAE (test)", color="#FF9800")
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15)
ax.set_ylabel("Значение метрики")
ax.set_title("Сравнение моделей предсказания LogS")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150)
print("График сравнения сохранён в results/model_comparison.png")

# 2. Predicted vs Actual для лучшей модели
best_model_name = max(results, key=lambda m: results[m]["R2_test"])
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred_best, alpha=0.6, edgecolors="k", linewidth=0.5)
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    linewidth=2,
    label="Идеальное предсказание",
)
ax.set_xlabel("Экспериментальный LogS")
ax.set_ylabel("Предсказанный LogS")
ax.set_title(f"{best_model_name}: Predicted vs Actual (R²={results[best_model_name]['R2_test']})")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/pred_vs_actual.png", dpi=150)
print("График pred vs actual сохранён в results/pred_vs_actual.png")

# 3. Важность признаков (для Random Forest)
if hasattr(models["RandomForest"],"feature_importances_"):
    importances = models["RandomForest"].feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        range(len(feature_cols)),
        importances[indices],
        color="#4CAF50",
    )
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([feature_cols[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest: важность дескрипторов")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150)
    print("График важности признаков сохранён в results/feature_importance.png")

print("\nГотово! Все результаты в папке results/") 
