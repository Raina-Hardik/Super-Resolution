import hashlib
import io
import os

import redis
import torch
from celery import Celery
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from core.config import settings
from core.logging import get_logger
from core.model_loader import get_generator
from utils.image import stitch_tiles, tile_image, to_tensor

logger = get_logger("celery_tasks")

celery_app = Celery(
    "super-res-tasks",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://dragonfly:6379/0"),
    backend=os.environ.get("REDIS_URL", "redis://dragonfly:6379/0")
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # High throughput config
    worker_concurrency=1, # 1 GPU task at a time per worker to avoid OOM
    task_track_started=True,
)

redis_client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://dragonfly:6379/0"))

@celery_app.task(bind=True, name="tasks.process_job")
def process_job(self, job_id: str, image_bytes: bytes):
    """
    Main orchestration task.
    1. Loads image from bytes.
    2. Tiles the image.
    3. Processes each tile (with caching).
    4. Stitches the tiles.
    5. Saves the final result to disk or redis.
    """
    logger.info(f"Starting job {job_id}")
    self.update_state(state="PROCESSING", meta={"progress": 0})

    # Load model
    model = get_generator()
    model.eval()

    # Process image
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = to_tensor(img).unsqueeze(0).to(settings.device)

        # Tile image (128x128 patches)
        patch_size = 128
        scale_factor = 4
        tiles, original_dims = tile_image(img_tensor, patch_size)

        total_tiles = len(tiles)
        upscaled_tiles = []

        for i, tile in enumerate(tiles):
            # Compute cache key based on tile tensor content
            tile_bytes = tile.cpu().numpy().tobytes()
            tile_hash = hashlib.sha256(tile_bytes).hexdigest()
            cache_key = f"tile_cache:{tile_hash}"

            # Check cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for tile {i} in job {job_id}")
                # Deserialize from bytes
                buffer = io.BytesIO(cached_result)
                up_tile = torch.load(buffer, map_location=settings.device, weights_only=True)
            else:
                # Process
                with torch.no_grad():
                    up_tile = model(tile)
                # Cache result (TTL 24 hours)
                buffer = io.BytesIO()
                torch.save(up_tile.cpu(), buffer)
                redis_client.setex(cache_key, 86400, buffer.getvalue())

            upscaled_tiles.append(up_tile)
            self.update_state(state="PROCESSING", meta={
                "progress": int(((i + 1) / total_tiles) * 100),
                "total": total_tiles,
                "current": i + 1
            })

        # Stitch
        final_tensor = stitch_tiles(upscaled_tiles, original_dims, patch_size, scale_factor)

        # Denormalize (output is tanh [-1, 1], convert to [0, 1])
        final_tensor = (final_tensor + 1.0) / 2.0
        final_tensor = final_tensor.clamp(0, 1)

        # Save to output folder
        final_img = to_pil_image(final_tensor.squeeze(0).cpu())
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{job_id}_output.png")
        final_img.save(output_path)

        logger.info(f"Completed job {job_id}")
        return {"status": "success", "output_path": output_path}

    except Exception as e:
        logger.error(f"Job {job_id} failed", exc_info=True)
        raise e
