# Use a base Python image
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy the project configuration
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies
RUN uv sync --only-group prod

# Create a new stage for the final image
FROM python:3.12-slim

# Copy the rest of the project
COPY . /app

# Copy the virtual environment from the previous stage
COPY --from=builder /app/.venv /app/.venv

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"

# Expose the port for Gradio
EXPOSE 7860

WORKDIR /app

# Run the Gradio app by default
CMD ["python", "-m", "src.ui"]
