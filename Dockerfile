# Use a base Python image
FROM python:3.12-slim AS builder

# ENV vars
ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PATH="/app/.venv/bin:$PATH"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy the project configuration
COPY pyproject.toml .

# Install dependencies
RUN uv sync --only-group prod

# Copy the project
COPY . .

# Expose the port for Gradio
EXPOSE 7860

# Run the Gradio app by default
CMD ["python", "-m", "src.ui"]
