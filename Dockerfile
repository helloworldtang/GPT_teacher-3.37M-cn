FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY config/ config/
COPY data/ data/
COPY tokenizer/ tokenizer/
COPY checkpoints/best.pt checkpoints/best.pt
COPY scripts/docker_entrypoint.py docker_entrypoint.py

# Install dependencies, pin starlette<1.0 for Gradio 4.x compatibility
RUN uv sync --no-dev --frozen && \
    uv pip install "starlette<1.0.0"

# Expose Gradio port
EXPOSE 7860

# Gradio in Docker: bind all interfaces
ENV SERVER_NAME=0.0.0.0
ENV GRADIO_ANALYTICS_ENABLED=false

CMD ["/app/.venv/bin/python", "docker_entrypoint.py"]
