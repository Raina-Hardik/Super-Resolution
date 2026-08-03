# Image Super-Resolution with EDSR

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.5-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project implements a scalable image super-resolution system based on the Enhanced Deep Residual Networks (EDSR) architecture. It enables users to enhance the resolution of their images using a pre-trained EDSR model, optimized for production workloads with asynchronous task processing and memory-efficient image tiling.

## Features

- **FastAPI Backend**: High-performance RESTful API with automatic documentation.
- **Asynchronous Processing**: Celery workers backed by Redis/Dragonfly for handling heavy image processing tasks in the background.
- **Memory-Efficient Tiling**: Automatically splits large images into tiles for super-resolution, preventing Out-of-Memory (OOM) errors on large inputs, and stitches them back together.
- **High-Speed Caching**: Utilizes Dragonfly for fast transient tile caching and rate limiting.
- **Rate Limiting**: IP-based rate limiting to protect the system from abuse.
- **Secure Administration**: JWT-based authentication for administrative actions, such as cache management.
- **Observability**: Structured JSON logging (via `structlog`) and Prometheus metrics for comprehensive monitoring.
- **Containerized Deployment**: Ready-to-use Docker Compose configuration for one-click infrastructure provisioning.

## Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)
- [just](https://github.com/casey/just) command runner
- Docker and Docker Compose (recommended for deployment)
- NVIDIA GPU with CUDA (optional, but highly recommended for inference speed)

## Quick Start

### Docker Deployment (Recommended)

To launch the complete infrastructure (FastAPI, Celery Worker, Dragonfly Cache):

```bash
# Using standard docker compose
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# Or using the just recipe
just deploy-gpu
```

For environments without an NVIDIA GPU (CPU-only mode):

```bash
just deploy-cpu
```

### Manual Installation (Development)

1. Clone the repository:
   ```bash
   git clone https://github.com/Raina-Hardik/Super-Resolution.git
   cd Super-Resolution
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Start the cache backend (Dragonfly or Redis):
   ```bash
   docker run -d -p 6379:6379 docker.dragonflydb.io/dragonflydb/dragonfly:latest
   ```

4. Start the Celery Worker:
   ```bash
   uv run celery -A core.tasks.celery_app worker --loglevel=info
   ```

5. Start the FastAPI application:
   ```bash
   uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Usage

### API Endpoints

The system exposes a RESTful API for interaction. Full interactive documentation is available locally via Swagger UI (`http://localhost:8000/docs`) and ReDoc (`http://localhost:8000/redoc`).

#### Submitting a Job
```bash
curl -X POST "http://localhost:8000/api/v1/jobs" \
  -F "file=@your-image.jpg"

# Expected Response:
# {"job_id":"abc123def456","status":"PENDING"}
```

#### Checking Job Status
```bash
curl "http://localhost:8000/api/v1/jobs/abc123def456"

# Expected Response:
# {"job_id":"abc123def456","status":"PROCESSING","progress":50}
```

#### Retrieving Results
Once the status returns `SUCCESS`, you can download the enhanced image:
```bash
curl -o enhanced-image.png "http://localhost:8000/api/v1/jobs/abc123def456/result"
```

### Administrative Endpoints

To manually clear the tile cache in Dragonfly, an Admin JWT token is required.

```bash
# Generate a token locally
just generate-admin-token

# Clear the cache using the token
curl -X POST "http://localhost:8000/api/v1/cache/bust" \
  -H "Authorization: Bearer <YOUR_TOKEN>"
```

### Metrics and Monitoring

Prometheus metrics are exposed at the `/metrics` endpoint for scraping:
```bash
curl "http://localhost:8000/metrics"
```

## Configuration

System behavior can be configured via environment variables. Create a `.env` file in the project root:

```env
# Server Configuration
DOMAIN_NAME=super-res.local
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=your_secure_secret_key

# App Settings
LOG_LEVEL=INFO
```

## Architecture

The project is structured as follows:

```text
Super-Resolution/
├── api/                   # FastAPI routes and dependencies
├── core/                  # Core config, logging, auth, and Celery tasks
├── models/                # EDSR PyTorch model definitions
├── utils/                 # Tiling, caching, and image processing utilities
├── tests/                 # Pytest test suite
├── docker-compose*.yml    # Docker configurations
├── justfile               # Build and deployment recipes
└── pyproject.toml         # Dependency management
```

## Development

### Running Tests
Execute the test suite using pytest:
```bash
uv run pytest tests/ -v
```

### Code Formatting and Linting
The project uses Ruff for linting and formatting:
```bash
uvx ruff check --fix .
uvx ruff format .
```

## Acknowledgments

- This project is based on the Enhanced Deep Residual Networks (EDSR) architecture, developed by Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee. Further details can be found in their [paper](https://arxiv.org/abs/1707.02921).
- Built leveraging [FastAPI](https://fastapi.tiangolo.com/), [Celery](https://docs.celeryq.dev/), and [Dragonfly](https://www.dragonflydb.io/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
