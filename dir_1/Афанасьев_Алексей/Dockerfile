FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project
COPY generate_data.py train.py ./
RUN mkdir -p data results
CMD ["uv", "run", "sh", "-c", "python generate_data.py && python train.py"]
