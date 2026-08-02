import time

from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference

from api.routes import REQUEST_COUNT, REQUEST_LATENCY, router
from core.config import settings
from core.logging import setup_logging

setup_logging()

app = FastAPI(
    title=settings.app_name,
    description="Enhance image resolution using deep learning",
    version=settings.version,
    docs_url=None,  # Disable default swagger
    redoc_url=None,  # Disable default redoc
)

@app.middleware("http")
async def prometheus_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    endpoint = request.url.path
    if endpoint not in ["/metrics", "/health"]:
        REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, http_status=response.status_code).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

    return response

app.include_router(router)


@app.get("/docs", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )
