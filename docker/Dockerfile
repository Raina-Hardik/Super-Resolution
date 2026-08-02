# Use an NVIDIA CUDA base image with Python support (via ubuntu or a python image)
# We'll use a python slim image as base, but it can be run with --gpus all for CUDA.
# The actual PyTorch version installed by uv will have CUDA support.
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy pyproject.toml and lockfile (if exists)
COPY pyproject.toml .

# Install dependencies using uv (this will install PyTorch with CUDA support if requested)
RUN uv pip install --system -e .

# Copy application code
COPY api/ api/
COPY core/ core/
COPY models/ models/
COPY utils/ utils/
COPY train.py .

# Create necessary directories
RUN mkdir -p uploads output weights

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
