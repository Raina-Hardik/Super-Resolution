from typing import Annotated
from fastapi import Depends
from models.edsr import Generator
from core.model_loader import get_generator

GeneratorDep = Annotated[Generator, Depends(get_generator)]
