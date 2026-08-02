# Image Super-Resolution with EDSR

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.5-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project implements a super-resolution model based on the Enhanced Deep Residual Networks (EDSR) architecture. It allows users to enhance the resolution of their images using a pre-trained EDSR model.

## 🎉 What's New in v2.0

This is a **modernized and highly scalable version** of the application:
- ✨ **FastAPI** instead of Flask for better performance and automatic API documentation
- 🚀 **Asynchronous Processing** with Celery and Redis/Dragonfly backend to handle heavy loads
- 🧠 **Memory Efficient Tiling** allowing super-resolution of arbitrarily large images without Out of Memory (OOM) errors
- ⚡ **Dragonfly Cache** for extremely fast transient tile caching and rate limiting
- 🛡️ **Guest Rate Limiting** to prevent abuse (e.g. 10 requests per minute per IP)
- 🔑 **JWT Admin Authentication** for cache busting and administration
- 📝 **Structured JSON Logging** with `structlog`, complete with 5MB rotating file handlers
- 📊 **Prometheus Metrics** available for scraping and monitoring (`/metrics`)
- 🐳 **Docker Compose** support for one-click full pipeline deployment (API, Celery, Dragonfly)
- 🛠️ **Justfile** task runner for simplified development commands

### Breaking Changes

The application now operates entirely on an asynchronous worker model.
- `POST /api/v1/jobs` - Submits an image and returns a `job_id`
- `GET /api/v1/jobs/{job_id}` - Polls the status and progress of a task
- `GET /api/v1/jobs/{job_id}/result` - Retrieves the final enhanced image once the status is `SUCCESS`

## 📋 Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) for faster dependency management
- [just](https://github.com/casey/just) command runner
- Docker and Docker Compose (highly recommended for deployment)
- NVIDIA GPU with CUDA (optional but highly recommended for speed)

## 🚀 Quick Start

### Docker Deployment (Recommended)

To spin up the entire system (FastAPI, Celery Worker, Dragonfly Cache):

```bash
# Using standard docker compose
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# Or simply using our just recipe
just deploy-gpu
```

If you do not have an NVIDIA GPU, you can run in CPU-only mode:

```bash
just deploy-cpu
```

### Manual Installation for Development

1. Clone the repository:
   ```bash
   git clone https://github.com/Raina-Hardik/Super-Resolution.git
   cd Super-Resolution
   ```

2. Sync dependencies:
   ```bash
   uv sync
   ```

3. Spin up the cache backend (Dragonfly or Redis):
   ```bash
   docker run -d -p 6379:6379 docker.dragonflydb.io/dragonflydb/dragonfly:latest
   ```

4. Start the Celery Worker (in a new terminal):
   ```bash
   uv run celery -A core.tasks.celery_app worker --loglevel=info
   ```

5. Start the FastAPI application:
   ```bash
   uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

## 📖 Usage

### API Endpoints

#### Interactive Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

#### Submitting a Job
```bash
# Submit an image
curl -X POST "http://localhost:8000/api/v1/jobs" \
  -F "file=@your-image.jpg"

# Response
# {"job_id":"abc123def456","status":"PENDING"}
```

#### Checking Job Status
```bash
curl "http://localhost:8000/api/v1/jobs/abc123def456"

# Response
# {"job_id":"abc123def456","status":"PROCESSING","progress":50}
```

#### Retrieving Results
```bash
curl -o enhanced-image.png "http://localhost:8000/api/v1/jobs/abc123def456/result"
```

### Administrative Endpoints

To manually clear the tile cache in Dragonfly, you need an Admin JWT token.

```bash
# Generate a token locally
just generate-admin-token

# Use the token to clear cache
curl -X POST "http://localhost:8000/api/v1/cache/bust" \
  -H "Authorization: Bearer <YOUR_TOKEN>"
```

### Metrics & Monitoring
Prometheus metrics are available at `/metrics`:
```bash
curl "http://localhost:8000/metrics"
```

## 🔧 Configuration

### Environment Variables

You can supply configuration in a `.env` file:

```env
# Server Configuration
DOMAIN_NAME=super-res.local
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=your_super_secret_key_here

# App Settings
LOG_LEVEL=INFO
```

## 🏗️ Architecture

```
Super-Resolution/
├── api/                   # FastAPI routes and dependencies
├── core/                  # Core config, logging, auth, and Celery tasks
├── models/                # EDSR PyTorch model definitions
├── utils/                 # Tiling, caching, and image processing utilities
├── tests/                 # Pytest test suite
├── docker-compose*.yml    # Docker configurations
├── justfile               # Build and deployment recipes
└── pyproject.toml         # Dependencies managed via uv
```

## 🛠️ Development

### Running Tests
```bash
uv run pytest tests/ -v
```

### Code Formatting and Linting
We use Ruff for rapid linting and formatting.
```bash
uvx ruff check --fix .
uvx ruff format .
```

## 📚 Acknowledgments

- This project is based on the Enhanced Deep Residual Networks (EDSR) architecture, developed by Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee. More information about EDSR can be found in their [paper](https://arxiv.org/abs/1707.02921).
- Built with [FastAPI](https://fastapi.tiangolo.com/), [Celery](https://docs.celeryq.dev/), and [Dragonfly](https://www.dragonflydb.io/).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
