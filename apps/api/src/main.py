from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.errors import register_exception_handlers
from src.modules.perception.sam_service import SamPerceptionService

redis_client: aioredis.Redis | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    global redis_client
    redis_client = aioredis.from_url(str(settings.REDIS_URL), decode_responses=True)
    yield
    if redis_client:
        await redis_client.aclose()


app = FastAPI(
    title="ChronoRig Autonomous Video-to-Physics API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    return {
        "status": "ok",
        "sam_model": settings.SAM_MODEL_ID,
        "spark_model": settings.MUSE_SPARK_MODEL_ID,
        "open_weight_model": settings.MUSE_SPARK_OPEN_MODEL_ID,
    }


@app.post("/api/v1/jobs/{job_id}/voice-command")
async def transcribe_voice_part_prompts(
    job_id: uuid.UUID, audio: UploadFile = File(...)
) -> dict[str, Any]:
    """
    Uses Meta's Muse Voice Transcribe + meta-models/muse-spark-open-1026
    to let users speak mechanical part prompts hands-free.
    """
    raw_bytes = await audio.read()
    if len(raw_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Voice command audio exceeds 10MB limit",
        )
    service = SamPerceptionService()
    prompts = await service.transcribe_voice_Part_prompts(raw_bytes)
    return {"jobId": str(job_id), "extractedPartPrompts": prompts}


@app.websocket("/api/v1/jobs/{job_id}/stream")
async def job_telemetry_websocket(websocket: WebSocket, job_id: uuid.UUID) -> None:
    await websocket.accept()
    if not redis_client:
        await websocket.close(code=1011)
        return

    pubsub = redis_client.pubsub()
    channel_name = f"chronorig:job:{job_id}:events"
    await pubsub.subscribe(channel_name)

    try:
        # Send immediate initial handshake event matching WsTelemetryEventSchema
        await websocket.send_text(
            json.dumps(
                {
                    "event": "STATUS_CHANGED",
                    "jobId": str(job_id),
                    "status": "SEGMENTING_SAM",
                    "progressPct": 20,
                    "message": "Connected to ChronoRig Redis Pub/Sub telemetry stream",
                }
            )
        )
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message.get("data"):
                await websocket.send_text(str(message["data"]))
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(channel_name)
        await pubsub.aclose()
