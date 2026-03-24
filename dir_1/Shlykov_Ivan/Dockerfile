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
