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

# Deploy the stack (Docker Compose or similar)
deploy:
    docker-compose up -d

# Install dependencies using uv
install:
    uv sync
