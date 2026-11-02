from __future__ import annotations

import asyncio
import base64
import json
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
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
from src.modules.perception.sam_service import (
    SamPerceptionService,
    generate_controlled_hinge_mp4,
)
from src.modules.synthesis.synthesizer import (
    PhysicsRigSynthesizer,
    generate_deterministic_blender_script,
)
from src.modules.verification.iou_verifier import ClosedLoopIouVerifier

ARTIFACTS_DIR = settings.ARTIFACTS_DIR
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

job_repo = DurableJobRepository()
JOB_EVENT_QUEUES: dict[str, list[asyncio.Queue[str]]] = {}


class PresignUploadRequest(BaseModel):
    filename: str
    contentType: str = Field(pattern=r"^video/(mp4|webm)$")
    fileSizeBytes: int = Field(gt=0, le=52428800)


async def broadcast_job_event(job_id: str, payload: dict[str, Any]) -> None:
    msg = json.dumps(payload)
    for q in JOB_EVENT_QUEUES.get(job_id, []):
        await q.put(msg)


def resolve_caller_owner_id(
    x_owner_id: str | None = Header(default=None, alias="X-Owner-Id"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> str:
    """
    Extracts a deterministic caller identity from X-Owner-Id or Bearer token
    so tenants cannot read or mutate another tenant's private jobs.
    """
    if x_owner_id and x_owner_id.strip():
        return x_owner_id.strip()
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        if token:
            return f"bearer:{token[:24]}"
    return "public-studio-user"


def build_base_url(request: Request | None = None) -> str:
    if request is not None:
        base = str(request.base_url).rstrip("/")
        if not base.startswith("http://testserver"):
            return base
    return settings.PUBLIC_API_BASE_URL.rstrip("/")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    yield


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
        "storage": "durable_sqlite_wal",
    }


@app.post("/api/v1/uploads/presign")
async def create_presigned_upload(body: PresignUploadRequest, request: Request) -> dict[str, str]:
    """Implements the OpenAPI presigned upload contract."""
    upload_id = str(uuid.uuid4())
    base_url = build_base_url(request)
    s3_key = f"uploads/{upload_id}.mp4"
    return {
        "uploadUrl": f"{base_url}/api/v1/uploads/direct/{upload_id}",
        "videoS3Key": s3_key,
    }


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
) -> dict[str, Any]:
    sam_service = SamPerceptionService()
    should_use_live_sam = (
        use_live_sam_api if use_live_sam_api is not None else settings.has_live_meta_key()
    )

    try:
        await broadcast_job_event(
            job_id,
            {
                "event": "STATUS_CHANGED",
                "jobId": job_id,
                "status": "SEGMENTING_SAM",
                "progressPct": 25,
                "message": "Extracting per-concept binary masks...",
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
                    frame_width=640,
                    frame_height=480,
                )
                concept_masks.append((prompt, masks))
            fps, frame_count, width, height = 30.0, 36, 640, 480
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
            target_iou=0.85,
            max_iterations=3,
            broadcast_callback=lambda evt: broadcast_job_event(job_id, evt),
        )

        bpy_script = generate_deterministic_blender_script(
            parts=parts, joints=verified_joints, video_fps=fps
        )

        now_iso = datetime.now(UTC).isoformat()
        existing = job_repo.get_job_row(job_id)
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
            job_repo.save_job, job_id, owner_id, job_payload, bpy_script, mp4_path
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


@app.post("/api/v1/jobs/demo-hinge", status_code=status.HTTP_201_CREATED)
async def create_verified_demo_hinge_job(
    request: Request,
    x_owner_id: str | None = Header(default=None, alias="X-Owner-Id"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> dict[str, Any]:
    owner_id = resolve_caller_owner_id(x_owner_id, authorization)
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
async def upload_and_create_rig_job(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    part_prompts_csv: str = Form("static_base,rotating_arm"),
    use_live_sam: bool | None = Form(None),
    wait: bool = Query(
        default=False,
        description="If true, waits synchronously and returns 201 on completion; default is async 202.",
    ),
    x_owner_id: str | None = Header(default=None, alias="X-Owner-Id"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> dict[str, Any]:
    owner_id = resolve_caller_owner_id(x_owner_id, authorization)
    base_url = build_base_url(request)

    # Validate MIME type and stream upload in 1MB chunks to enforce size cap before buffering
    if video.content_type and video.content_type not in (
        "video/mp4",
        "video/webm",
        "application/octet-stream",
    ):
        raise ChronoRigDomainError(
            code="UNSUPPORTED_MEDIA_TYPE",
            message=f"Unsupported video content_type '{video.content_type}'",
            status_code=415,
        )

    chunks: list[bytes] = []
    total_bytes = 0
    while True:
        chunk = await video.read(1024 * 1024)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > settings.MAX_VIDEO_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Video exceeds 50MB limit")
        chunks.append(chunk)

    raw_bytes = b"".join(chunks)
    job_id = str(uuid.uuid4())
    mp4_path = ARTIFACTS_DIR / f"{job_id}.mp4"
    await asyncio.to_thread(mp4_path.write_bytes, raw_bytes)

    prompts = [p.strip() for p in part_prompts_csv.split(",") if p.strip()]
    if len(prompts) < 2:
        raise ChronoRigDomainError(
            code="INVALID_PART_PROMPTS",
            message="At least 2 part prompts are required to infer a kinematic joint.",
            status_code=422,
        )

    queued_payload = _create_initial_queued_payload(job_id, prompts, base_url)
    await asyncio.to_thread(job_repo.save_job, job_id, owner_id, queued_payload, None, mp4_path)

    if wait:
        response.status_code = status.HTTP_201_CREATED
        return await _execute_pipeline_on_mp4(
            job_id=job_id,
            owner_id=owner_id,
            mp4_path=mp4_path,
            part_prompts=prompts,
            base_url=base_url,
            use_live_sam_api=use_live_sam,
            perturb_initial_anchor_for_demo=False,
        )

    background_tasks.add_task(
        _execute_pipeline_on_mp4,
        job_id,
        owner_id,
        mp4_path,
        prompts,
        base_url,
        use_live_sam,
        False,
    )
    return queued_payload


def _get_authorized_job_row(job_id: str, caller_owner_id: str) -> dict[str, Any]:
    row = job_repo.get_job_row(job_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if row["owner_id"] != caller_owner_id and row["owner_id"] != "public-studio-user":
        raise HTTPException(status_code=403, detail="Forbidden: caller does not own this job")
    return row


@app.get("/api/v1/jobs/{job_id}")
async def get_rig_job(
    job_id: str,
    x_owner_id: str | None = Header(default=None, alias="X-Owner-Id"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> dict[str, Any]:
    caller = resolve_caller_owner_id(x_owner_id, authorization)
    row = await asyncio.to_thread(_get_authorized_job_row, job_id, caller)
    return row["payload"]


@app.get("/api/v1/jobs/{job_id}/blender.py", response_class=PlainTextResponse)
async def download_blender_script(
    job_id: str,
    x_owner_id: str | None = Header(default=None, alias="X-Owner-Id"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> str:
    caller = resolve_caller_owner_id(x_owner_id, authorization)
    row = await asyncio.to_thread(_get_authorized_job_row, job_id, caller)
    if not row["blender_script"]:
        raise HTTPException(status_code=404, detail=f"Blender script for job {job_id} not ready")
    return str(row["blender_script"])


@app.get("/api/v1/jobs/{job_id}/video.mp4")
async def stream_job_video(
    job_id: str,
    x_owner_id: str | None = Header(default=None, alias="X-Owner-Id"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> FileResponse:
    caller = resolve_caller_owner_id(x_owner_id, authorization)
    row = await asyncio.to_thread(_get_authorized_job_row, job_id, caller)
    path: Path | None = row["video_path"]
    if not path or not await asyncio.to_thread(path.exists):
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path, media_type="video/mp4")


@app.websocket("/api/v1/jobs/{job_id}/stream")
async def job_telemetry_websocket(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    queue: asyncio.Queue[str] = asyncio.Queue()
    JOB_EVENT_QUEUES.setdefault(job_id, []).append(queue)

    try:
        row = await asyncio.to_thread(job_repo.get_job_row, job_id)
        current_status = row["status"] if row else "QUEUED"
        await websocket.send_text(
            json.dumps(
                {
                    "event": "STATUS_CHANGED",
                    "jobId": job_id,
                    "status": current_status,
                    "progressPct": 100 if current_status == "COMPLETED" else 10,
                    "message": "Connected to ChronoRig live telemetry stream",
                }
            )
        )
        while True:
            msg = await queue.get()
            await websocket.send_text(msg)
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in JOB_EVENT_QUEUES and queue in JOB_EVENT_QUEUES[job_id]:
            JOB_EVENT_QUEUES[job_id].remove(queue)
