"""Stub: auth/security not needed for evaluation pipeline."""
import os

SECRET_KEY = os.getenv("SECRET_KEY", "eval-pipeline-placeholder-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7


def hash_password(password: str) -> str:
    raise NotImplementedError("Auth disabled in eval pipeline")


def hash_token(token: str) -> str:
    raise NotImplementedError("Auth disabled in eval pipeline")
