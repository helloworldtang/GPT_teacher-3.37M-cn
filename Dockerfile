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
COPY run.py ./

# Install dependencies
RUN uv sync --no-dev --frozen

# Expose Gradio port
EXPOSE 7860

# Run web demo with pre-trained model
CMD ["uv", "run", "python", "run.py", "--skip-train"]
