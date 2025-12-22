"""
Super-Resolution FastAPI Application

This is a modernized version of the Super-Resolution web application.
The Flask implementation is deprecated but maintained for backward compatibility.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional
import warnings

# Create FastAPI app
app = FastAPI(
    title="Super-Resolution API",
    description="Enhance image resolution using deep learning",
    version="2.0.0"
)

# Setup directories
UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("output")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}


def superres(input_image: Image.Image) -> Image.Image:
    """
    Apply super-resolution to the input image.
    TODO: Replace with actual EDSR model after training.
    
    Args:
        input_image: PIL Image to enhance
        
    Returns:
        Enhanced PIL Image
    """
    output_image = input_image.resize(
        (input_image.width * 2, input_image.height * 2),
        Image.Resampling.LANCZOS
    )
    return output_image


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, error: Optional[str] = None):
    """Render the main page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "error": error}
    )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Modern API endpoint for image super-resolution.
    
    Accepts an image file and returns the enhanced version.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate unique filename to avoid conflicts
    file_id = uuid.uuid4().hex
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    input_path = UPLOAD_FOLDER / f"{file_id}_input.{file_ext}"
    output_path = OUTPUT_FOLDER / f"{file_id}_output.png"
    
    try:
        # Save uploaded file
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image
        input_image = Image.open(input_path)
        output_image = superres(input_image)
        output_image.save(output_path)
        
        # Clean up input file
        input_path.unlink(missing_ok=True)
        
        return FileResponse(
            output_path,
            media_type="image/png",
            filename="enhanced_image.png"
        )
    
    except Exception as e:
        # Clean up on error
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/", response_class=HTMLResponse)
@app.post("/process", response_class=HTMLResponse, deprecated=True)
async def process_image_form(request: Request, file: UploadFile = File(...)):
    """
    DEPRECATED: Legacy form-based endpoint for backward compatibility.
    Use POST /upload instead for new implementations.
    
    This endpoint maintains compatibility with the old Flask implementation.
    """
    warnings.warn(
        "Form-based upload endpoint is deprecated. Use POST /upload instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if not file.filename:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "No selected file"}
        )
    
    if not allowed_file(file.filename):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Invalid file format"}
        )
    
    file_id = uuid.uuid4().hex
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    input_path = UPLOAD_FOLDER / f"{file_id}_input.{file_ext}"
    output_path = OUTPUT_FOLDER / f"{file_id}_output.png"
    
    try:
        # Save uploaded file
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image
        input_image = Image.open(input_path)
        output_image = superres(input_image)
        output_image.save(output_path)
        
        # Clean up input file
        input_path.unlink(missing_ok=True)
        
        return FileResponse(
            output_path,
            media_type="image/png",
            filename="enhanced_image.png"
        )
    
    except Exception as e:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Processing error: {str(e)}"}
        )


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/api/v1/info")
async def api_info():
    """Get API information."""
    return {
        "name": "Super-Resolution API",
        "version": "2.0.0",
        "endpoints": {
            "upload": "/upload (POST) - Modern API endpoint",
            "process": "/process (POST) - Deprecated form endpoint",
            "health": "/health (GET) - Health check"
        }
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
