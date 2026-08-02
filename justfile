# justfile

set shell := ["bash", "-c"]

# Format and lint code
lint:
    uv run ruff check .
    uv run ruff format .

# Run the FastAPI development server
run:
    uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Train the super resolution model
train:
    uv run python train.py

# Build the Docker image
build:
    docker build -t super-resolution -f docker/Dockerfile .

# Deploy CPU only stack
deploy-cpu:
    docker compose up -d

# Deploy GPU stack
deploy-gpu:
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Deploy auto (checks for nvidia-smi)
deploy:
    @if command -v nvidia-smi > /dev/null 2>&1; then \
        echo "NVIDIA GPU detected. Deploying with GPU support..."; \
        just deploy-gpu; \
    else \
        echo "No NVIDIA GPU detected. Deploying CPU only..."; \
        just deploy-cpu; \
    fi

# Run tests
test:
    uv run pytest tests/

# Install dependencies using uv
install:
    uv sync
