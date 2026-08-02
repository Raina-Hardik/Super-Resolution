# Super-Resolution Quick Reference Guide

## Quick Start Commands

### With uv (Recommended - 10-100x faster)
```bash
# Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Setup and run
uv venv
source .venv/bin/activate  # Linux/Mac: .venv\Scripts\activate on Windows
uv pip install -e .
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Local Development (Traditional)
```bash
# Linux/Mac
./deploy.sh

# Windows
.\deploy.ps1 -Mode dev
```

### Docker Deployment
```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## API Endpoints

### Web Interface
- **URL**: `http://localhost:8000/`
- **Method**: GET/POST
- **Description**: HTML form interface

### REST API (Recommended)
- **URL**: `http://localhost:8000/upload`
- **Method**: POST
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (image file)
- **Response**: Enhanced image (PNG)

### Health Check
- **URL**: `http://localhost:8000/health`
- **Method**: GET
- **Response**: `{"status": "healthy", "version": "2.0.0"}`

### API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## cURL Examples

### Upload Image
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@image.jpg" \
  --output enhanced.png
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Python Client

```python
import requests

# Upload image
with open("image.jpg", "rb") as f:
    response = requests.post("http://localhost:8000/upload", files={"file": f})

    # Save result
    with open("enhanced.png", "wb") as out:
        out.write(response.content)
```

## JavaScript/Node.js Client

```javascript
const FormData = require('form-data');
const fs = require('fs');
const fetch = require('node-fetch');

const form = new FormData();
form.append('file', fs.createReadStream('image.jpg'));

fetch('http://localhost:8000/upload', {
    method: 'POST',
    body: form
})
.then(res => res.buffer())
.then(buffer => {
    fs.writeFileSync('enhanced.png', buffer);
});
```

## Common Tasks

### Change Port
```bash
# Linux/Mac
uvicorn app:app --host 0.0.0.0 --port 3000

# Windows
uvicorn app:app --host 0.0.0.0 --port 3000
```

### Production with Workers
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Run with SSL (Self-Signed)
```bash
uvicorn app:app --host 0.0.0.0 --port 8443 \
  --ssl-keyfile=./key.pem \
  --ssl-certfile=./cert.pem
```

## Docker Commands

### Build Custom Image
```bash
docker build -t my-super-resolution:latest .
```

### Run with Custom Volume
```bash
docker run -d -p 8000:8000 \
  -v $(pwd)/my-uploads:/app/uploads \
  -v $(pwd)/my-output:/app/output \
  super-resolution
```

### View Container Logs
```bash
docker logs -f super-resolution-app
```

### Execute Commands in Container
```bash
docker exec -it super-resolution-app /bin/bash
```

## Caddy Reverse Proxy

### Quick Start
```bash
# Copy example file
cp Caddyfile.example Caddyfile

# Edit your domain
nano Caddyfile

# Run Caddy
caddy run
```

### Docker with Caddy
```yaml
# Uncomment Caddy service in docker-compose.yml
docker-compose up -d
```

## Troubleshooting

### Port Already in Use
```bash
# Linux/Mac
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Check Application Status
```bash
curl http://localhost:8000/health
```

### View Application Logs
```bash
# Docker
docker-compose logs -f super-resolution

# Systemd (Linux)
sudo journalctl -u super-resolution -f
```

### Restart Application
```bash
# Docker
docker-compose restart

# Systemd (Linux)
sudo systemctl restart super-resolution
```

## Environment Variables

Create `.env` file:
```env
HOST=0.0.0.0
PORT=8000
WORKERS=4
LOG_LEVEL=info
MAX_UPLOAD_SIZE=10485760
```

Then run:
```bash
uvicorn app:app --env-file .env
```

## Performance Tips

1. **Use multiple workers** for production:
   ```bash
   uvicorn app:app --workers 4
   ```

2. **Enable Caddy caching** for static content

3. **Use Docker** for consistent environments

4. **Monitor with health checks**:
   ```bash
   watch -n 5 curl -s http://localhost:8000/health
   ```

## File Size Limits

- Default max upload: 10MB
- Modify in `.env` or app configuration
- For larger files, adjust nginx/Caddy limits too

## Supported Image Formats

- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- GIF (`.gif`)
- WebP (`.webp`)

## URLs

- Web Interface: `http://localhost:8000/`
- API Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- API Info: `http://localhost:8000/api/v1/info`
