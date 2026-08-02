from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import torch

class Settings(BaseSettings):
    app_name: str = "Super-Resolution API"
    version: str = "2.0.0"
    
    # Model parameters
    weights_dir: str = "weights"
    checkpoint_gen: str = "gen.pth.tar"
    
    # Server configs
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model architecture params
    in_channels: int = 3
    num_channels: int = 64
    num_blocks: int = 16
    
    # Training
    high_res: int = 128
    low_res: int = 32
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 100
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

settings = Settings()
