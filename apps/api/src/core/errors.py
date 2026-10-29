import json
import logging
import re
import time
import traceback
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

SENSITIVE_KEYS_PATTERN = re.compile(
    r"(password|secret|token|authorization|api_key|cookie)", re.IGNORECASE
)


def sanitize_payload(data: Any) -> Any:
    """Recursively redact sensitive fields before structured logging."""
    if isinstance(data, dict):
        return {
            k: ("[REDACTED]" if SENSITIVE_KEYS_PATTERN.search(k) else sanitize_payload(v))
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [sanitize_payload(item) for item in data]
    return data


class StructuredJsonFormatter(logging.Formatter):
    """Formats logs as single-line JSON for Datadog / ELK ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "untracked"),
            "path": getattr(record, "path", None),
            "latency_ms": getattr(record, "latency_ms", None),
        }
        if record.exc_info:
            payload["exception"] = "".join(traceback.format_exception(*record.exc_info))
        return json.dumps(sanitize_payload(payload))


logger = logging.getLogger("chronorig.api")
handler = logging.StreamHandler()
handler.setFormatter(StructuredJsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ChronoRigDomainError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        trace_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.trace_id = trace_id
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            response.headers["X-Request-ID"] = trace_id
            logger.info(
                f"{request.method} {request.url.path} -> {response.status_code}",
                extra={
                    "trace_id": trace_id,
                    "path": request.url.path,
                    "latency_ms": latency_ms,
                },
            )
            return response
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                f"Unhandled exception on {request.method} {request.url.path}",
                exc_info=exc,
                extra={
                    "trace_id": trace_id,
                    "path": request.url.path,
                    "latency_ms": latency_ms,
                },
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                headers={"X-Request-ID": trace_id},
                content={
                    "type": "https://chronorig.dev/errors/internal-server-error",
                    "title": "Internal Server Error",
                    "status": 500,
                    "code": "INTERNAL_ERROR",
                    "detail": "An unexpected internal error occurred. Reference trace_id for support.",
                    "trace_id": trace_id,
                },
            )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_middleware(ObservabilityMiddleware)

    @app.exception_handler(ChronoRigDomainError)
    async def domain_error_handler(request: Request, exc: ChronoRigDomainError) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=exc.status_code,
            headers={"X-Request-ID": trace_id},
            content={
                "type": f"https://chronorig.dev/errors/{exc.code.lower()}",
                "title": exc.code,
                "status": exc.status_code,
                "code": exc.code,
                "detail": exc.message,
                "details": sanitize_payload(exc.details),
                "trace_id": trace_id,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            headers={"X-Request-ID": trace_id},
            content={
                "type": "https://chronorig.dev/errors/validation-failed",
                "title": "Request Validation Failed",
                "status": 422,
                "code": "VALIDATION_ERROR",
                "detail": "One or more request parameters failed schema validation.",
                "errors": sanitize_payload(exc.errors()),
                "trace_id": trace_id,
            },
        )
