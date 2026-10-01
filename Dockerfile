FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/
WORKDIR /code
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY app ./app
COPY main.py ./
EXPOSE 80
CMD ["uv", "run", "--frozen", "--no-dev", "fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "80"]
