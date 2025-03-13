# Use a base Python image
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy the project configuration
COPY pyproject.toml .

# Install dependencies
RUN uv sync

# Copy the rest of the project
COPY . .

# Create a new stage for the final image
FROM python:3.12-slim

# Copy the virtual environment from the previous stage
COPY --from=builder /app/.venv /app/.venv

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"

# Expose the port for Gradio
EXPOSE 7860

# Run the Gradio app by default
CMD ["uv", "run", "gradio_app.py"]
