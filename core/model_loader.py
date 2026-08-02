from pathlib import Path

import torch

from core.config import settings
from models.edsr import Generator


class ModelLoader:
    _instance = None
    _model = None

    @classmethod
    def get_model(cls) -> Generator:
        if cls._model is None:
            cls._model = Generator(
                in_channels=settings.in_channels, num_channels=settings.num_channels, num_blocks=settings.num_blocks
            ).to(settings.device)

            # Try to load weights
            weight_path = Path(settings.weights_dir) / settings.checkpoint_gen
            if weight_path.exists():
                checkpoint = torch.load(weight_path, map_location=settings.device)
                if "state_dict" in checkpoint:
                    cls._model.load_state_dict(checkpoint["state_dict"])
                else:
                    cls._model.load_state_dict(checkpoint)
                cls._model.eval()
            else:
                print(f"Warning: No weights found at {weight_path}. Model will output noise.")

        return cls._model


def get_generator() -> Generator:
    return ModelLoader.get_model()
