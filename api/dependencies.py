import time
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.auth import validate_token
from core.model_loader import get_generator
from models.edsr import Generator

GeneratorDep = Annotated[Generator, Depends(get_generator)]

security = HTTPBearer()

def get_current_admin(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    payload = validate_token(token)

    if not payload or payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Admin access required")
    return payload

# In-memory store for rate limiting
# Structure: { "ip_address": [timestamp1, timestamp2, ...] }
_rate_limit_store = {}

def rate_limit_guest(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()

    if client_ip not in _rate_limit_store:
        _rate_limit_store[client_ip] = []

    # Retain only requests within the last 60 seconds
    _rate_limit_store[client_ip] = [
        ts for ts in _rate_limit_store[client_ip]
        if current_time - ts < 60
    ]

    # Check limit (10 requests per minute)
    if len(_rate_limit_store[client_ip]) >= 10:
        raise HTTPException(status_code=429, detail="Too Many Requests")

    _rate_limit_store[client_ip].append(current_time)
