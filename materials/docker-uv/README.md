# Практикум: Docker + uv для ML-проектов

## Цель

К концу практикума вы:
- Создадите ML-проект с помощью `uv` (управление зависимостями, запуск скриптов)
- Напишете пайплайн обучения модели на химических данных (предсказание растворимости молекул)
- Упакуете проект в Docker-контейнер для воспроизводимого запуска

```
Итоговая схема:

┌──────────────────────────────────────────────────────┐
│  Docker-контейнер                                    │
│  ┌────────────────────────────────────────────────┐  │
│  │  uv-окружение (Python 3.12 + зависимости)     │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │  train.py → модель + метрики + графики   │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────┘  │
│                        ↕ volume                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  /app/results/ ←→ ./results/ (хост-машина)    │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

---

## Предварительные требования

### Установка Docker

```bash
# Проверьте, установлен ли Docker
docker --version

# Если не установлен:
# macOS — скачайте Docker Desktop: https://www.docker.com/products/docker-desktop/
# Linux (Ubuntu/Debian):
sudo apt update && sudo apt install docker.io
sudo usermod -aG docker $USER   # чтобы не писать sudo перед каждой командой
# После этого перелогиньтесь

# Проверьте, что Docker работает
docker run hello-world
```

### Установка uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Проверьте
uv --version
```

> Если uv уже установлен (см. [материалы по Python-окружениям](../python-env/)), переходите к Части 1.

---

## Часть 1: Создание ML-проекта с uv

### Шаг 1.1 — Инициализация проекта

```bash
# Создайте и перейдите в директорию проекта
mkdir solubility-predictor
cd solubility-predictor

# Инициализируйте uv-проект (Python 3.12 — совпадает с Docker-образом)
uv init --python 3.12
```

Посмотрите, что создалось:

```
solubility-predictor/
├── .python-version    # Версия Python (3.12)
├── pyproject.toml     # Описание проекта и зависимости
├── hello.py           # Пример скрипта (удалим)
└── README.md          # Описание проекта
```

Удалите файл-пример:

```bash
rm hello.py
```

### Шаг 1.2 — Добавление зависимостей

Установим библиотеки для ML и работы с данными:

```bash
uv add numpy pandas scikit-learn matplotlib
```

> `uv add` автоматически:
> 1. Добавляет зависимости в `pyproject.toml`
> 2. Создаёт виртуальное окружение `.venv/`
> 3. Устанавливает все пакеты
> 4. Генерирует `uv.lock` — lock-файл для воспроизводимости

Проверьте `pyproject.toml`:

```bash
cat pyproject.toml
```

Вы увидите что-то вроде:

```toml
[project]
name = "solubility-predictor"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "matplotlib>=3.10",
    "numpy>=2.4",
    "pandas>=3.0",
    "scikit-learn>=1.8",
]
```

### Шаг 1.3 — Генерация датасета

Создайте скрипт `generate_data.py`, который создаёт синтетический датасет молекулярных дескрипторов:

```bash
touch generate_data.py
```

Откройте `generate_data.py` в редакторе и вставьте:

```python
"""Генерация синтетического датасета молекулярных дескрипторов."""

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
)

df.to_csv("data/molecules.csv", index=False)
print(f"Датасет сохранён: {len(df)} молекул, {len(df.columns) - 1} дескрипторов")
print(f"\nПервые 5 строк:\n{df.head()}")
print(f"\nСтатистика LogS:\n{df['LogS'].describe()}")
```

Создайте директорию для данных и запустите:

```bash
mkdir -p data
uv run python generate_data.py
```

### Шаг 1.4 — Скрипт обучения модели

Создайте `train.py`:

```python
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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

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
}

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
if hasattr(models["RandomForest"], "feature_importances_"):
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
```

Запустите обучение:

```bash
uv run python train.py
```

Проверьте результаты:

```bash
ls results/
cat results/metrics.json
```

Вы должны увидеть:
- `metrics.json` — метрики всех моделей
- `model_comparison.png` — сравнение моделей
- `pred_vs_actual.png` — график предсказаний лучшей модели
- `feature_importance.png` — важность дескрипторов

### Шаг 1.5 — Проверка структуры проекта

К этому моменту у вас должна быть такая структура:

```
solubility-predictor/
├── .python-version
├── .venv/                   # Виртуальное окружение (НЕ коммитить)
├── pyproject.toml           # Описание проекта
├── uv.lock                  # Lock-файл зависимостей
├── README.md
├── generate_data.py         # Генерация датасета
├── train.py                 # Обучение моделей
├── data/
│   └── molecules.csv        # Датасет
└── results/
    ├── metrics.json
    ├── model_comparison.png
    ├── pred_vs_actual.png
    └── feature_importance.png
```

---

## Часть 2: Контейнеризация с Docker

### Шаг 2.1 — Зачем нужен Docker?

```
Без Docker:                              С Docker:

"У меня работает!" 🤷                    Работает одинаково везде 🎯

Студент A: Python 3.10, numpy 1.24      ┌─────────────────────┐
Студент B: Python 3.12, numpy 2.1       │  Docker-контейнер   │
Преподаватель: Python 3.11, numpy 1.26  │  Python 3.12        │
                                         │  numpy 2.1.3        │
Три окружения → три результата           │  sklearn 1.5.2      │
                                         │  ...                │
                                         │  Один результат ✓   │
                                         └─────────────────────┘
```

### Шаг 2.2 — Основные понятия Docker

```
┌──────────────┐    docker build    ┌──────────────┐    docker run    ┌──────────────┐
│  Dockerfile  │ ────────────────→  │    Image      │ ──────────────→ │  Container   │
│  (рецепт)    │                    │  (снимок)     │                 │  (процесс)   │
└──────────────┘                    └──────────────┘                  └──────────────┘

Dockerfile — текстовый файл с инструкциями, как собрать образ
Image      — неизменяемый снимок файловой системы + метаданные
Container  — запущенный экземпляр образа (изолированный процесс)
```

| Понятие | Аналогия |
|---------|----------|
| Dockerfile | Рецепт блюда |
| Image | Замороженная заготовка |
| Container | Готовое блюдо на тарелке |
| Volume | Судочек, в который забираете еду с собой |
| Registry (Docker Hub) | Кулинарная книга с готовыми рецептами |

### Шаг 2.3 — Пишем Dockerfile

Создайте файл `Dockerfile` (без расширения) в корне проекта:

```dockerfile
# Базовый образ с Python и uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Копируем файлы зависимостей (для кеширования слоёв)
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости
RUN uv sync --no-dev --no-install-project

# Копируем остальные файлы проекта
COPY generate_data.py train.py ./

# Создаём директории для данных и результатов
RUN mkdir -p data results

# Команда по умолчанию: генерация данных + обучение
CMD ["uv", "run", "sh", "-c", "python generate_data.py && python train.py"]
```

> **Почему COPY в два этапа?**
> Docker кеширует каждый слой (шаг). Если зависимости не менялись, при пересборке Docker не будет заново устанавливать пакеты — только скопирует обновлённый код. Это экономит время.

```
Без кеширования:                    С кешированием слоёв:

Изменил train.py → пересборка       Изменил train.py → пересборка

1. FROM python ✓ (кеш)              1. FROM python ✓ (кеш)
2. COPY всё → ИНВАЛИДАЦИЯ           2. COPY pyproject.toml ✓ (кеш)
3. RUN uv sync → ~30 сек            3. RUN uv sync ✓ (кеш!) ← вот тут экономия
4. ...                               4. COPY train.py → ИНВАЛИДАЦИЯ
                                     5. ...
Итого: ~40 сек                       Итого: ~3 сек
```

### Шаг 2.4 — Создаём .dockerignore

Создайте файл `.dockerignore`, чтобы не копировать лишнее в контейнер:

```
.venv/
.git/
__pycache__/
results/
data/
*.pyc
.python-version
```

### Шаг 2.5 — Собираем образ

```bash
docker build -t solubility-predictor .
```

Разберём команду:
| Часть | Значение |
|-------|----------|
| `docker build` | Собрать образ |
| `-t solubility-predictor` | Имя (тег) образа |
| `.` | Контекст сборки — текущая директория |

Вы увидите пошаговый вывод сборки. Каждый `Step` — это слой из Dockerfile.

Проверьте, что образ создался:

```bash
docker images | head -5
```

### Шаг 2.6 — Запускаем контейнер

```bash
# Простой запуск
docker run solubility-predictor
```

Вы увидите вывод `generate_data.py` и `train.py`. Но результаты остались внутри контейнера — вы не можете их увидеть на хост-машине.

### Шаг 2.7 — Монтирование volume (связь контейнера с хостом)

Чтобы результаты сохранялись на вашем компьютере, используйте **volume** — общую папку между хостом и контейнером:

```bash
# Создайте папку для результатов
mkdir -p output

# Запуск с монтированием volume
docker run -v $(pwd)/output:/app/results solubility-predictor
```

Разберём флаг `-v`:

```
-v $(pwd)/output:/app/results
    ↑                ↑
    Папка на         Папка внутри
    вашем            контейнера
    компьютере
```

Проверьте результаты:

```bash
ls output/
cat output/metrics.json
```

Графики (`model_comparison.png`, `pred_vs_actual.png`, `feature_importance.png`) теперь доступны в папке `output/` — откройте их!

### Шаг 2.8 — Полезные команды Docker

| Команда | Описание |
|---------|----------|
| `docker images` | Список образов |
| `docker ps` | Запущенные контейнеры |
| `docker ps -a` | Все контейнеры (включая остановленные) |
| `docker rm <id>` | Удалить контейнер |
| `docker rmi <image>` | Удалить образ |
| `docker logs <id>` | Логи контейнера |
| `docker exec -it <id> bash` | Войти в работающий контейнер |
| `docker system prune` | Очистить неиспользуемые ресурсы |

### Шаг 2.9 — Интерактивная отладка

Если нужно зайти внутрь контейнера и посмотреть, что там происходит:

```bash
# Запустить контейнер в интерактивном режиме
docker run -it solubility-predictor bash

# Внутри контейнера:
ls                    # посмотреть файлы
uv run python -c "import sklearn; print(sklearn.__version__)"
cat pyproject.toml
exit                  # выйти
```

---

## Часть 3: Задания 

### Задание 1 (обязательное): Добавьте новую модель

1. Добавьте в `train.py` ещё одну модель — **Support Vector Regression (SVR)**:

```python
from sklearn.svm import SVR

# Добавьте в словарь models:
"SVR": SVR(kernel="rbf", C=1.0)
```

2. Пересоберите Docker-образ и запустите:

```bash
docker build -t solubility-predictor .
docker run -v $(pwd)/output:/app/results solubility-predictor
```

3. Сравните результаты в `output/metrics.json` — какая модель лучше?

### Задание 2 (обязательное): Добавьте новый дескриптор

1. Добавьте в `generate_data.py` новый молекулярный дескриптор — **FormalCharge** (формальный заряд):

```python
"FormalCharge": np.random.choice([-1, 0, 0, 0, 1], n_molecules),  # Формальный заряд
```

2. Включите его влияние на LogS (добавьте слагаемое в формулу):

```python
+ 0.4 * df["FormalCharge"]
```

3. Пересоберите и запустите. Изменился ли рейтинг важности признаков?

### Задание 3 (дополнительное): Оптимизация гиперпараметров

Добавьте в `train.py` подбор гиперпараметров для `RandomForest` с помощью `GridSearchCV`:

```python
from sklearn.model_selection import GridSearchCV

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
```

Сохраните лучшие параметры в `results/best_params.json`.

---

## Сдача работы

### Структура сдачи

Работу необходимо сдать через Pull Request в основной репозиторий. Ваши файлы должны лежать в `dir_1/Фамилия_Имя/`:

```
dir_1/
└── Ivanov_Ivan/
    ├── Dockerfile
    ├── .dockerignore
    ├── pyproject.toml
    ├── uv.lock
    ├── generate_data.py
    └── train.py
```

> **Не коммитьте** `.venv/`, `data/`, `results/`, `__pycache__/` — они генерируются автоматически.

### Порядок сдачи

```bash
# 1. Форкните репозиторий (если ещё не сделали в Stage 0)

# 2. Клонируйте свой форк
git clone https://github.com/<ваш-username>/ChemML2026.git
cd ChemML2026

# 3. Создайте директорию и скопируйте файлы проекта
mkdir -p dir_1/Фамилия_Имя
cp path/to/your/{Dockerfile,.dockerignore,pyproject.toml,uv.lock,generate_data.py,train.py} dir_1/Фамилия_Имя/

# 4. Закоммитьте и запушьте
git add dir_1/Фамилия_Имя/
git commit -m "Stage 1: Docker + uv практикум"
git push origin main

# 5. Создайте Pull Request на GitHub
```

### Автопроверка

При создании Pull Request автоматически запустится проверка, которая:

1. Проверит наличие обязательных файлов
2. Проверит зависимости в `pyproject.toml`
3. Соберёт Docker-образ из вашего `Dockerfile`
4. Запустит контейнер и проверит результаты
5. Проверит выполнение заданий 1 и 2
6. Опубликует отчёт в комментарии к PR

### Система баллов

| Проверка | Баллы |
|----------|-------|
| Обязательные файлы | 1 |
| pyproject.toml (зависимости) | 1 |
| Docker build (образ собирается) | 2 |
| Docker run (контейнер работает) | 2 |
| metrics.json (корректная структура) | 1 |
| Графики (>= 3 PNG) | 1 |
| Задание 1: SVR модель | 1 |
| Задание 2: FormalCharge дескриптор | 1 |
| Бонус: GridSearchCV + best_params.json | 1 |
| **Итого** | **11** |

**Оценка:** >= 9 — отлично, >= 7 — хорошо, >= 5 — удовлетворительно, < 5 — доработка.

---

## Шпаргалка: полный рабочий цикл

```bash
# 1. Создать проект
uv init --python 3.12 my-project && cd my-project

# 2. Добавить зависимости
uv add numpy pandas scikit-learn matplotlib

# 3. Написать код (generate_data.py, train.py)

# 4. Запустить локально
uv run python generate_data.py
uv run python train.py

# 5. Создать Dockerfile + .dockerignore

# 6. Собрать образ
docker build -t my-project .

# 7. Запустить контейнер с volume
docker run -v $(pwd)/output:/app/results my-project

# 8. Изменили код? Пересобрать и запустить
docker build -t my-project . && docker run -v $(pwd)/output:/app/results my-project
```

---

## Частые ошибки и решения

### «docker: command not found»

Docker не установлен или не запущен. На macOS убедитесь, что Docker Desktop запущен (иконка в трее).

### «permission denied» при запуске docker

```bash
# Linux: добавьте себя в группу docker
sudo usermod -aG docker $USER
# Перелогиньтесь после этого
```

### «no space left on device» при сборке

```bash
# Очистите неиспользуемые образы и контейнеры
docker system prune -a
```

### «uv: command not found» внутри контейнера

Убедитесь, что используете базовый образ `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`. Этот образ уже содержит uv.

### Изменения в коде не отражаются в контейнере

Вы забыли пересобрать образ. После каждого изменения кода нужно заново запустить `docker build`.

---

## Разбор Dockerfile по слоям

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
```

| Инструкция | Что делает |
|------------|------------|
| `FROM` | Базовый образ. Всё начинается с него. Можно взять чистую ОС (`ubuntu`), Python (`python:3.12-slim`) или специализированный образ (как здесь — с uv) |

```dockerfile
WORKDIR /app
```

| Инструкция | Что делает |
|------------|------------|
| `WORKDIR` | Устанавливает рабочую директорию. Все последующие команды выполняются из этой папки. Если папки нет — она создаётся |

```dockerfile
COPY pyproject.toml uv.lock ./
```

| Инструкция | Что делает |
|------------|------------|
| `COPY` | Копирует файлы из контекста сборки (вашего компьютера) в образ. `./` — текущая `WORKDIR` |

```dockerfile
RUN uv sync --no-dev --no-install-project
```

| Инструкция | Что делает |
|------------|------------|
| `RUN` | Выполняет команду при **сборке** образа. Результат сохраняется в слой. `--no-dev` — без dev-зависимостей, `--no-install-project` — только зависимости, не сам проект |

```dockerfile
CMD ["uv", "run", "sh", "-c", "python generate_data.py && python train.py"]
```

| Инструкция | Что делает |
|------------|------------|
| `CMD` | Команда по умолчанию при **запуске** контейнера. Выполняется только при `docker run`, не при `docker build` |

### Разница RUN vs CMD

```
docker build:   FROM → COPY → RUN → RUN → ...   (формируют образ)
docker run:     CMD                                (запускает контейнер)
```

---

## Бонус: Docker Compose для сложных проектов

Если в проекте несколько сервисов (например, ML-модель + веб-интерфейс + база данных), используют **Docker Compose**:

```yaml
# docker-compose.yml
services:
  train:
    build: .
    volumes:
      - ./output:/app/results
      - ./data:/app/data
```

```bash
# Запуск
docker compose up

# Пересборка + запуск
docker compose up --build
```

Это удобнее, чем запоминать длинные команды `docker run -v ...`.

---

## 📚 Полезные материалы

### Docker
- [Docker — официальная документация](https://docs.docker.com/)
- [Docker для начинающих (Habr)](https://habr.com/ru/articles/353238/)
- [Play with Docker](https://labs.play-with-docker.com/) — песочница Docker в браузере
- [Docker Curriculum](https://docker-curriculum.com/) — пошаговый курс

### uv
- [uv — официальная документация](https://docs.astral.sh/uv/)
- [uv: Getting Started](https://docs.astral.sh/uv/getting-started/)
- [Сравнение uv с другими менеджерами пакетов](https://docs.astral.sh/uv/concepts/resolution/)

### ML в химии
- [scikit-learn — документация](https://scikit-learn.org/stable/)
- [QSPR/QSAR: обзор подходов](https://en.wikipedia.org/wiki/Quantitative_structure%E2%80%93activity_relationship)
- [Молекулярные дескрипторы](https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors) (RDKit)
