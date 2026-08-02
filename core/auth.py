import os
from datetime import datetime, timedelta

import jwt

SECRET_KEY = os.getenv("SECRET_KEY", "dummy_secret_key")
ALGORITHM = "HS256"

def generate_token(payload: dict = None) -> str:
    if payload is None:
        payload = {"role": "admin"}

    # Add expiration if not present
    if "exp" not in payload:
        payload["exp"] = datetime.utcnow() + timedelta(days=1)

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def validate_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
