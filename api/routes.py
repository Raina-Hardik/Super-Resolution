import os
import uuid

import redis
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from api.dependencies import get_current_admin, rate_limit_guest
from core.config import settings
from core.logging import get_logger
from core.tasks import process_job

logger = get_logger("api_routes")
router = APIRouter()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

redis_client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://dragonfly:6379/0"))

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Duration', ['endpoint'])

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@router.post("/api/v1/jobs", dependencies=[Depends(rate_limit_guest)])
async def submit_job(file: UploadFile = File(...)):
    """
    Submit an image for super-resolution processing. (Guest mode - rate limited)
    """
    if not file.filename or not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file format.")

    job_id = uuid.uuid4().hex

    # Read bytes and dispatch to Celery
    file_bytes = await file.read()
    process_job.apply_async(args=[job_id, file_bytes], task_id=job_id)

    logger.info("Job submitted", job_id=job_id)
    return {"job_id": job_id, "status": "PENDING"}


@router.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Check the status of a super-resolution job.
    """
    task = AsyncResult(job_id)
    response = {
        "job_id": job_id,
        "status": task.status,
    }

    if task.status == 'PROCESSING':
        response["progress"] = task.info.get("progress", 0) if task.info else 0
    elif task.status == 'FAILURE':
        response["error"] = str(task.info)

    return response


@router.get("/api/v1/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Download the final enhanced image if the job is complete.
    """
    task = AsyncResult(job_id)
    if task.status != 'SUCCESS':
        raise HTTPException(status_code=400, detail=f"Job is not complete. Current status: {task.status}")

    output_path = task.result.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Result file not found on disk.")

    return FileResponse(output_path, media_type="image/png", filename=f"{job_id}_enhanced.png")


@router.post("/api/v1/cache/bust", dependencies=[Depends(get_current_admin)])
async def bust_cache():
    """
    Clear all cached tiles in Dragonfly. Requires Admin JWT.
    """
    try:
        # Delete all keys matching tile_cache:*
        keys = redis_client.keys("tile_cache:*")
        if keys:
            redis_client.delete(*keys)
        logger.info("Cache busted successfully", cleared_keys=len(keys))
        return {"status": "success", "cleared_keys": len(keys)}
    except Exception as e:
        logger.error("Cache bust failed", error=str(e))
        raise HTTPException(status_code=500, detail="Cache bust failed") from e


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.version}
