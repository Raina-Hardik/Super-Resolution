from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference

from api.routes import router
from core.config import settings

app = FastAPI(
    title=settings.app_name,
    description="Enhance image resolution using deep learning",
    version=settings.version,
    docs_url=None, # Disable default swagger
    redoc_url=None # Disable default redoc
)

app.include_router(router)

@app.get("/docs", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )
