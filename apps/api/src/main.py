from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.errors import ChronoRigDomainError, register_exception_handlers
from src.core.repository import DurableJobRepository
from src.core.security import (
    authenticate_caller,
    create_jwt_token,
    validate_video_magic_bytes,
    verify_jwt_token,
)
from src.modules.perception.sam_service import (
    SamPerceptionService,
    generate_controlled_hinge_mp4,
    probe_and_validate_video,
)
from src.modules.synthesis.synthesizer import (
    PhysicsRigSynthesizer,
    generate_deterministic_blender_script,
)
from src.modules.verification.iou_verifier import ClosedLoopIouVerifier
from src.worker.tasks import DurableWorkerEngine

ARTIFACTS_DIR = settings.ARTIFACTS_DIR
UPLOADS_DIR = ARTIFACTS_DIR / "uploads"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_ID_PATTERN = re.compile(r"^[0-9a-fA-F-]{36}$")
S3_KEY_PATTERN = re.compile(r"^uploads/([0-9a-fA-F-]{36})\.mp4$")

job_repo = DurableJobRepository()


class AuthTokenRequest(BaseModel):
    tenantId: str = Field(default="studio-default-tenant", min_length=3, max_length=64)


class PresignUploadRequest(BaseModel):
    filename: str
    contentType: str = Field(pattern=r"^video/(mp4|webm)$")
    fileSizeBytes: int = Field(gt=0, le=52428800)


class CreateRigJobJsonRequest(BaseModel):
    videoS3Key: str = Field(min_length=5, max_length=512)
    partPrompts: list[str] = Field(min_length=2, max_length=12)
    targetIouThreshold: float = Field(default=0.85, ge=0.5, le=0.98)
    maxSelfDebugIterations: int = Field(default=3, ge=1, le=5)
    useLiveSam: bool | None = None


async def broadcast_job_event(job_id: str, payload: dict[str, Any]) -> None:
    """Writes telemetry events to the durable SQLite WAL outbox shared across all workers."""
    await asyncio.to_thread(job_repo.append_event, job_id, payload)


def build_base_url(request: Request | None = None) -> str:
    if request is not None:
        base = str(request.base_url).rstrip("/")
        if not base.startswith("http://testserver"):
            return base
    return settings.PUBLIC_API_BASE_URL.rstrip("/")


def _create_initial_queued_payload(
    job_id: str, part_prompts: list[str], base_url: str
) -> dict[str, Any]:
    now_iso = datetime.now(UTC).isoformat()
    return {
        "id": job_id,
        "status": "QUEUED",
        "videoUrl": f"{base_url}/api/v1/jobs/{job_id}/video.mp4",
        "videoFps": 30.0,
        "videoFrameCount": 0,
        "partPrompts": part_prompts,
        "parts": [],
        "joints": [],
        "iouIterations": [],
        "finalIouScore": None,
        "generatedR3fCode": None,
        "blenderScriptDownloadUrl": None,
        "errorCode": None,
        "errorMessage": None,
        "createdAt": now_iso,
        "updatedAt": now_iso,
    }


async def _execute_pipeline_on_mp4(
    job_id: str,
    owner_id: str,
    mp4_path: Path,
    part_prompts: list[str],
    base_url: str,
    use_live_sam_api: bool | None = None,
    perturb_initial_anchor_for_demo: bool = False,
    target_iou: float = 0.85,
    max_iterations: int = 3,
) -> dict[str, Any]:
    sam_service = SamPerceptionService()
    should_use_live_sam = (
        use_live_sam_api if use_live_sam_api is not None else settings.has_live_meta_key()
    )

    try:
        # Always probe actual video dimensions, FPS, frame count, and duration first
        fps, frame_count, width, height = await asyncio.to_thread(
            probe_and_validate_video, mp4_path
        )

        await broadcast_job_event(
            job_id,
            {
                "event": "STATUS_CHANGED",
                "jobId": job_id,
                "status": "SEGMENTING_SAM",
                "progressPct": 25,
                "message": f"Extracting masks across {frame_count} frames ({width}x{height} @ {fps:.1f}fps)...",
            },
        )

        if should_use_live_sam:
            raw_bytes = await asyncio.to_thread(mp4_path.read_bytes)
            data_uri = f"data:video/mp4;base64,{base64.b64encode(raw_bytes).decode('utf-8')}"
            concept_masks = []
            for prompt in part_prompts:
                masks = await sam_service.segment_single_concept_video(
                    video_data_uri=data_uri,
                    concept_prompt=prompt,
                    frame_width=width,
                    frame_height=height,
                )
                concept_masks.append((prompt, masks))
        else:
            concept_masks, fps, frame_count, width, height = await asyncio.to_thread(
                sam_service.extract_local_cv_masks_from_mp4, mp4_path, part_prompts
            )

        await broadcast_job_event(
            job_id,
            {
                "event": "STATUS_CHANGED",
                "jobId": job_id,
                "status": "SOLVING_KINEMATICS",
                "progressPct": 50,
                "message": "Solving PCA orientations and parent-relative Kasa hinge pivots...",
            },
        )

        parts, joints = await asyncio.to_thread(
            sam_service.build_kinematic_graph_from_concept_masks,
            concept_masks,
            fps,
            width,
            height,
        )

        if perturb_initial_anchor_for_demo and joints:
            joints[0].anchor_world_xyz = [
                round(joints[0].anchor_world_xyz[0] + 0.45, 3),
                round(joints[0].anchor_world_xyz[1] - 0.35, 3),
                0.0,
            ]

        synthesizer = PhysicsRigSynthesizer(parts_lookup=parts)
        verifier = ClosedLoopIouVerifier(synthesizer=synthesizer)

        await broadcast_job_event(
            job_id,
            {
                "event": "STATUS_CHANGED",
                "jobId": job_id,
                "status": "VERIFYING_IOU",
                "progressPct": 75,
                "message": "Running forward-kinematic silhouette IoU verification loop...",
            },
        )

        verified_joints, iou_iterations, best_iou = await verifier.run_verification_loop(
            job_id=job_id,
            parts=parts,
            joints=joints,
            target_iou=target_iou,
            max_iterations=max_iterations,
            broadcast_callback=lambda evt: broadcast_job_event(job_id, evt),
        )

        bpy_script = generate_deterministic_blender_script(
            parts=parts, joints=verified_joints, video_fps=fps
        )

        now_iso = datetime.now(UTC).isoformat()
        existing = await asyncio.to_thread(job_repo.get_job_row, job_id)
        created_iso = existing["payload"].get("createdAt", now_iso) if existing else now_iso

        job_payload: dict[str, Any] = {
            "id": job_id,
            "status": "COMPLETED",
            "videoUrl": f"{base_url}/api/v1/jobs/{job_id}/video.mp4",
            "videoFps": fps,
            "videoFrameCount": frame_count,
            "partPrompts": part_prompts,
            "parts": [
                {
                    "id": p.part_id,
                    "samObjectId": p.sam_object_id,
                    "label": p.label,
                    "isStaticRoot": p.is_static_root,
                    "estimatedMassKg": p.estimated_mass_kg,
                    "boundingBoxExtents": p.bounding_box_extents,
                    "trajectoryFrames": p.trajectory_frames,
                    "textureUrl": None,
                }
                for p in parts
            ],
            "joints": [
                {
                    "id": j.joint_id,
                    "parentPartId": j.parent_part_id,
                    "childPartId": j.child_part_id,
                    "jointType": j.joint_type,
                    "anchorWorldXyz": j.anchor_world_xyz,
                    "axisXyz": j.axis_xyz,
                    "limits": j.limits,
                }
                for j in verified_joints
            ],
            "iouIterations": iou_iterations,
            "finalIouScore": best_iou,
            "generatedR3fCode": "// Parameterized RigSpec rendered deterministically by PhysicsRigViewport",
            "blenderScriptDownloadUrl": f"{base_url}/api/v1/jobs/{job_id}/blender.py",
            "errorCode": None,
            "errorMessage": None,
            "createdAt": created_iso,
            "updatedAt": now_iso,
        }

        await asyncio.to_thread(
            job_repo.save_job,
            job_id,
            owner_id,
            job_payload,
            bpy_script,
            mp4_path,
            should_use_live_sam,
            target_iou,
            max_iterations,
        )
        await broadcast_job_event(
            job_id,
            {
                "event": "STATUS_CHANGED",
                "jobId": job_id,
                "status": "COMPLETED",
                "progressPct": 100,
                "message": f"Reconstruction verified (Final Mean IoU: {best_iou * 100:.1f}%)",
            },
        )
        return job_payload
    except Exception as exc:
        now_iso = datetime.now(UTC).isoformat()
        err_code = getattr(exc, "code", "PIPELINE_EXECUTION_ERROR")
        failed_payload = _create_initial_queued_payload(job_id, part_prompts, base_url)
        failed_payload["status"] = "FAILED"
        failed_payload["errorCode"] = err_code
        failed_payload["errorMessage"] = str(exc)
        failed_payload["updatedAt"] = now_iso
        await asyncio.to_thread(job_repo.save_job, job_id, owner_id, failed_payload, None, mp4_path)
        await broadcast_job_event(
            job_id,
            {
                "event": "JOB_FAILED",
                "jobId": job_id,
                "errorCode": err_code,
                "errorMessage": str(exc),
            },
        )
        raise


worker_engine = DurableWorkerEngine(repo=job_repo, pipeline_runner=_execute_pipeline_on_mp4)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    worker_task = None
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        worker_task = asyncio.create_task(worker_engine.run_forever(poll_interval_sec=0.15))
    yield
    if worker_task:
        worker_engine.stop()
        worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await worker_task


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
async def healthcheck() -> dict[str, Any]:
    return {
        "status": "ok",
        "meta_api_connected": settings.has_live_meta_key(),
        "baseten_api_connected": settings.has_live_baseten_key(),
        "modal_api_connected": settings.has_live_modal_key(),
        "open_weight_api_connected": settings.has_live_open_weight_key(),
        "sam_model": settings.SAM_MODEL_ID,
        "spark_model": settings.MUSE_SPARK_MODEL_ID,
        "open_weight_model": settings.MUSE_SPARK_OPEN_MODEL_ID,
        "storage": "durable_sqlite_wal",
    }


@app.post("/api/v1/auth/token")
async def issue_session_jwt(body: AuthTokenRequest) -> dict[str, str]:
    """Issues a cryptographically signed HS256 Bearer JWT for HTTP and WebSocket auth."""
    token = create_jwt_token(subject=body.tenantId)
    return {"accessToken": token, "tokenType": "Bearer", "tenantId": body.tenantId}


@app.post("/api/v1/uploads/presign")
async def create_presigned_upload(
    body: PresignUploadRequest,
    request: Request,
    _owner_id: str = Depends(authenticate_caller),
) -> dict[str, str]:
    upload_id = str(uuid.uuid4())
    base_url = build_base_url(request)
    s3_key = f"uploads/{upload_id}.mp4"
    return {
        "uploadUrl": f"{base_url}/api/v1/uploads/direct/{upload_id}",
        "videoS3Key": s3_key,
    }


async def _stream_bytes_to_disk(async_chunks: AsyncGenerator[bytes, None], dest_path: Path) -> int:
    """
    Streams incoming video chunks directly to disk without buffering the full file
    in RAM, validating magic bytes on the first chunk and enforcing MAX_VIDEO_SIZE_BYTES.
    """
    total_bytes = 0
    first_chunk_checked = False

    def _write_chunk(fp: Any, data: bytes) -> None:
        fp.write(data)

    fp = await asyncio.to_thread(open, dest_path, "wb")
    try:
        async for chunk in async_chunks:
            if not chunk:
                continue
            if not first_chunk_checked:
                validate_video_magic_bytes(chunk)
                first_chunk_checked = True
            total_bytes += len(chunk)
            if total_bytes > settings.MAX_VIDEO_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Video exceeds 50MB limit",
                )
            await asyncio.to_thread(_write_chunk, fp, chunk)
    except Exception:
        await asyncio.to_thread(fp.close)
        if await asyncio.to_thread(dest_path.exists):
            await asyncio.to_thread(dest_path.unlink)
        raise
    else:
        await asyncio.to_thread(fp.close)

    if not first_chunk_checked:
        raise ChronoRigDomainError(
            code="EMPTY_VIDEO_UPLOAD",
            message="Uploaded video stream contained zero bytes.",
            status_code=400,
        )
    return total_bytes


@app.put("/api/v1/uploads/direct/{upload_id}", status_code=status.HTTP_201_CREATED)
@app.post("/api/v1/uploads/direct/{upload_id}", status_code=status.HTTP_201_CREATED)
async def upload_direct_binary(
    upload_id: str,
    request: Request,
    _owner_id: str = Depends(authenticate_caller),
) -> dict[str, Any]:
    """Receives raw binary stream at the presigned uploadUrl, validates magic bytes & probes video."""
    if not UPLOAD_ID_PATTERN.match(upload_id):
        raise HTTPException(status_code=400, detail="Invalid upload_id format")
    dest_path = UPLOADS_DIR / f"{upload_id}.mp4"
    written = await _stream_bytes_to_disk(request.stream(), dest_path)
    fps, frame_count, width, height = await asyncio.to_thread(probe_and_validate_video, dest_path)
    return {
        "videoS3Key": f"uploads/{upload_id}.mp4",
        "bytesWritten": written,
        "videoFps": fps,
        "videoFrameCount": frame_count,
        "dimensions": [width, height],
    }


@app.post("/api/v1/jobs/demo-hinge", status_code=status.HTTP_201_CREATED)
async def create_verified_demo_hinge_job(
    request: Request,
    owner_id: str = Depends(authenticate_caller),
) -> dict[str, Any]:
    base_url = build_base_url(request)
    job_id = str(uuid.uuid4())
    mp4_path = ARTIFACTS_DIR / f"{job_id}.mp4"
    prompts = ["static_anchor_base", "rotating_pendulum_arm"]

    queued = _create_initial_queued_payload(job_id, prompts, base_url)
    await asyncio.to_thread(job_repo.save_job, job_id, owner_id, queued, None, mp4_path)
    await asyncio.to_thread(generate_controlled_hinge_mp4, mp4_path, 36, 30.0)

    return await _execute_pipeline_on_mp4(
        job_id=job_id,
        owner_id=owner_id,
        mp4_path=mp4_path,
        part_prompts=prompts,
        base_url=base_url,
        use_live_sam_api=False,
        perturb_initial_anchor_for_demo=True,
    )


@app.post("/api/v1/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_rig_job(
    request: Request,
    response: Response,
    wait: bool = Query(default=False),
    owner_id: str = Depends(authenticate_caller),
) -> dict[str, Any]:
    """
    Supports BOTH the OpenAPI JSON contract (`CreateRigJobRequest` with `videoS3Key`)
    AND direct `multipart/form-data` uploads, enqueuing work into the durable SQLite WAL queue.
    """
    base_url = build_base_url(request)
    content_type = request.headers.get("content-type", "").lower()
    job_id = str(uuid.uuid4())
    target_iou = 0.85
    max_iterations = 3
    use_live_sam: bool | None = None

    if "application/json" in content_type:
        raw_body = await request.json()
        parsed = CreateRigJobJsonRequest.model_validate(raw_body)
        match = S3_KEY_PATTERN.match(parsed.videoS3Key)
        if not match:
            raise ChronoRigDomainError(
                code="INVALID_VIDEO_S3_KEY",
                message="videoS3Key must match 'uploads/<uuid>.mp4' from /api/v1/uploads/presign",
                status_code=422,
            )
        upload_id = match.group(1)
        mp4_path = UPLOADS_DIR / f"{upload_id}.mp4"
        if not await asyncio.to_thread(mp4_path.exists):
            raise HTTPException(
                status_code=404,
                detail=f"Uploaded video artifact '{parsed.videoS3Key}' not found",
            )
        prompts = [p.strip() for p in parsed.partPrompts if p.strip()]
        target_iou = parsed.targetIouThreshold
        max_iterations = parsed.maxSelfDebugIterations
        use_live_sam = parsed.useLiveSam
    elif "multipart/form-data" in content_type:
        form = await request.form()
        video_field = form.get("video")
        if video_field is None or not hasattr(video_field, "read"):
            raise ChronoRigDomainError(
                code="MISSING_VIDEO_FILE",
                message="Multipart request missing 'video' file field.",
                status_code=422,
            )
        mp4_path = ARTIFACTS_DIR / f"{job_id}.mp4"

        async def _form_file_chunks() -> AsyncGenerator[bytes, None]:
            while True:
                chunk = await video_field.read(1024 * 1024)  # type: ignore[union-attr]
                if not chunk:
                    break
                yield chunk

        await _stream_bytes_to_disk(_form_file_chunks(), mp4_path)
        await asyncio.to_thread(probe_and_validate_video, mp4_path)

        prompts_csv = str(form.get("part_prompts_csv", "static_base,rotating_arm"))
        prompts = [p.strip() for p in prompts_csv.split(",") if p.strip()]
        raw_live = form.get("use_live_sam")
        if raw_live is not None:
            use_live_sam = str(raw_live).lower() in ("true", "1", "yes")
    else:
        raise ChronoRigDomainError(
            code="UNSUPPORTED_CONTENT_TYPE",
            message="POST /api/v1/jobs requires application/json or multipart/form-data",
            status_code=415,
        )

    if len(prompts) < 2:
        raise ChronoRigDomainError(
            code="INVALID_PART_PROMPTS",
            message="At least 2 part prompts are required to infer a kinematic joint.",
            status_code=422,
        )

    should_use_live = use_live_sam if use_live_sam is not None else settings.has_live_meta_key()
    queued_payload = _create_initial_queued_payload(job_id, prompts, base_url)
    await asyncio.to_thread(
        job_repo.save_job,
        job_id,
        owner_id,
        queued_payload,
        None,
        mp4_path,
        should_use_live,
        target_iou,
        max_iterations,
    )
    await broadcast_job_event(
        job_id,
        {
            "event": "STATUS_CHANGED",
            "jobId": job_id,
            "status": "QUEUED",
            "progressPct": 10,
            "message": "Job persisted to durable queue",
        },
    )

    if wait:
        response.status_code = status.HTTP_201_CREATED
        await worker_engine.process_next_available_job()
        row = await asyncio.to_thread(job_repo.get_job_row, job_id)
        return row["payload"] if row else queued_payload

    # Trigger durable worker processing asynchronously (in production/dev)
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        asyncio.create_task(worker_engine.process_next_available_job())
    return queued_payload


def _get_authorized_job_row(job_id: str, caller_owner_id: str) -> dict[str, Any]:
    row = job_repo.get_job_row(job_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if caller_owner_id != "studio-default-tenant" and row["owner_id"] != caller_owner_id:
        raise HTTPException(status_code=403, detail="Forbidden: caller does not own this job")
    return row


@app.get("/api/v1/jobs/{job_id}")
async def get_rig_job(
    job_id: str,
    owner_id: str = Depends(authenticate_caller),
) -> dict[str, Any]:
    row = await asyncio.to_thread(_get_authorized_job_row, job_id, owner_id)
    return row["payload"]


@app.get("/api/v1/jobs/{job_id}/blender.py", response_class=PlainTextResponse)
async def download_blender_script(
    job_id: str,
    owner_id: str = Depends(authenticate_caller),
) -> str:
    row = await asyncio.to_thread(_get_authorized_job_row, job_id, owner_id)
    if not row["blender_script"]:
        raise HTTPException(status_code=404, detail=f"Blender script for job {job_id} not ready")
    return str(row["blender_script"])


@app.get("/api/v1/jobs/{job_id}/video.mp4")
async def stream_job_video(
    job_id: str,
    owner_id: str = Depends(authenticate_caller),
) -> FileResponse:
    row = await asyncio.to_thread(_get_authorized_job_row, job_id, owner_id)
    path: Path | None = row["video_path"]
    if (not path or not await asyncio.to_thread(path.exists)) and job_id == "11111111-2222-4333-8444-555555555555":
        demo_path = ARTIFACTS_DIR / f"{job_id}.mp4"
        if not demo_path.exists():
            await asyncio.to_thread(generate_controlled_hinge_mp4, demo_path, 36, 30.0)
        path = demo_path
    if not path or not await asyncio.to_thread(path.exists):
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path, media_type="video/mp4")


@app.websocket("/api/v1/jobs/{job_id}/stream")
async def job_telemetry_websocket(
    websocket: WebSocket,
    job_id: str,
    token: str | None = Query(default=None),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    """
    Authenticates the caller and verifies job ownership BEFORE accepting the WebSocket,
    then streams cross-worker telemetry events from the durable SQLite WAL event outbox.
    """
    raw_jwt: str | None = token
    if not raw_jwt and authorization and authorization.lower().startswith("bearer "):
        raw_jwt = authorization[7:].strip()

    if raw_jwt:
        try:
            claims = verify_jwt_token(raw_jwt)
            caller_owner = str(claims["sub"])
        except HTTPException:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    elif settings.ENVIRONMENT == "production":
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    else:
        caller_owner = "studio-default-tenant"

    row = await asyncio.to_thread(job_repo.get_job_row, job_id)
    if caller_owner != "studio-default-tenant" and row and row["owner_id"] != caller_owner:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    last_event_id = 0

    try:
        current_status = row["status"] if row else "QUEUED"
        await websocket.send_text(
            json.dumps(
                {
                    "event": "STATUS_CHANGED",
                    "jobId": job_id,
                    "status": current_status,
                    "progressPct": 100 if current_status == "COMPLETED" else 10,
                    "message": "Connected to ChronoRig durable WAL telemetry stream",
                }
            )
        )
        while True:
            events = await asyncio.to_thread(job_repo.fetch_events_since, job_id, last_event_id)
            for ev_id, ev_json in events:
                await websocket.send_text(ev_json)
                last_event_id = ev_id
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
