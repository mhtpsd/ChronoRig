from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from fastapi import Header, HTTPException, status

from src.core.config import settings
from src.core.errors import ChronoRigDomainError


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(encoded: str) -> bytes:
    padding = "=" * ((4 - (len(encoded) % 4)) % 4)
    return base64.urlsafe_b64decode((encoded + padding).encode("ascii"))


def create_jwt_token(
    subject: str, ttl_seconds: int = 86400, extra_claims: dict[str, Any] | None = None
) -> str:
    """Issues an RFC 7519 HS256 signed JWT using settings.JWT_SECRET_KEY."""
    header = {"alg": settings.JWT_ALGORITHM, "typ": "JWT"}
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + ttl_seconds,
        **(extra_claims or {}),
    }
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

    secret = settings.JWT_SECRET_KEY.get_secret_value().encode("utf-8")
    signature = hmac.new(secret, signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url_encode(signature)}"


def verify_jwt_token(token: str) -> dict[str, Any]:
    """Cryptographically verifies an HS256 JWT signature and expiration timestamp."""
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed JWT token structure",
        )
    header_b64, payload_b64, signature_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    secret = settings.JWT_SECRET_KEY.get_secret_value().encode("utf-8")
    expected_sig = hmac.new(secret, signing_input, hashlib.sha256).digest()

    try:
        actual_sig = _b64url_decode(signature_b64)
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid JWT encoding",
        ) from exc

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid JWT signature",
        )

    exp = int(payload.get("exp", 0))
    if exp < int(time.time()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT token has expired",
        )

    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT token missing valid 'sub' claim",
        )
    return payload


def authenticate_caller(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> str:
    """
    FastAPI dependency that verifies Bearer JWT tokens.
    In non-production environments, if no Authorization header is sent, issues
    a default 'studio-default-tenant' identity so local UI demos work seamlessly.
    """
    if authorization:
        if not authorization.lower().startswith("bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header must use Bearer scheme",
            )
        raw_token = authorization[7:].strip()
        claims = verify_jwt_token(raw_token)
        return str(claims["sub"])

    if settings.ENVIRONMENT == "production":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer JWT Authorization header",
        )
    return "studio-default-tenant"


def validate_video_magic_bytes(header_bytes: bytes) -> None:
    """
    Verifies container magic bytes so non-video files renamed to .mp4 are rejected
    before hitting OpenCV or downstream storage.
    """
    if len(header_bytes) < 12:
        raise ChronoRigDomainError(
            code="INVALID_VIDEO_CONTAINER",
            message="Uploaded file is too small to be a valid MP4/WebM container.",
            status_code=415,
        )
    is_mp4_mov = header_bytes[4:8] in (b"ftyp", b"moov", b"wide", b"mdat")
    is_webm_mkv = header_bytes[:4] == b"\x1a\x45\xdf\xa3"
    if not (is_mp4_mov or is_webm_mkv):
        raise ChronoRigDomainError(
            code="UNSUPPORTED_MEDIA_CONTAINER",
            message="File magic bytes do not match a valid ISO MP4 or WebM container.",
            status_code=415,
        )
