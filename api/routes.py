import uuid
import shutil
import io
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image
import torch
from torchvision.utils import save_image

from api.dependencies import GeneratorDep
from utils.image import to_tensor
from core.config import settings

router = APIRouter()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Setup directories
UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("output")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@router.post("/upload")
async def upload_file(model: GeneratorDep, file: UploadFile = File(...)):
    """
    Enhance image resolution using the EDSR model.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    file_id = uuid.uuid4().hex
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    input_path = UPLOAD_FOLDER / f"{file_id}_input.{file_ext}"
    output_path = OUTPUT_FOLDER / f"{file_id}_output.png"
    
    try:
        # Save uploaded file
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Open and process image
        img = Image.open(input_path).convert("RGB")
        
        # Convert to tensor and add batch dimension
        img_tensor = to_tensor(img).unsqueeze(0).to(settings.device)
        
        # Inference
        with torch.no_grad():
            output_tensor = model(img_tensor)
            
        # Denormalize (output is tanh [-1, 1], convert to [0, 1])
        output_tensor = (output_tensor + 1.0) / 2.0
        
        # Save output
        save_image(output_tensor, output_path)
        
        # Clean up input
        input_path.unlink(missing_ok=True)
        
        return FileResponse(
            output_path,
            media_type="image/png",
            filename="enhanced_image.png"
        )
    except Exception as e:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@router.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.version}

@router.get("/api/v1/info")
async def api_info():
    return {
        "name": settings.app_name,
        "version": settings.version,
        "endpoints": {
            "upload": "/upload (POST) - Image Super-Resolution endpoint",
            "health": "/health (GET) - Health check"
        }
    }
