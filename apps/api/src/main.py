from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

from src.core.config import settings
from src.core.errors import register_exception_handlers
from src.modules.perception.sam_service import (
    SamPerceptionService,
    generate_controlled_hinge_mp4,
)
from src.modules.synthesis.synthesizer import (
    PhysicsRigSynthesizer,
    generate_deterministic_blender_script,
)
from src.modules.verification.iou_verifier import ClosedLoopIouVerifier

ARTIFACTS_DIR = Path(tempfile.gettempdir()) / "chronorig_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

JOBS_STORE: dict[str, dict[str, Any]] = {}
JOB_BLENDER_SCRIPTS: dict[str, str] = {}
JOB_VIDEO_PATHS: dict[str, Path] = {}
JOB_EVENT_QUEUES: dict[str, list[asyncio.Queue[str]]] = {}


async def broadcast_job_event(job_id: str, payload: dict[str, Any]) -> None:
    msg = json.dumps(payload)
    for q in JOB_EVENT_QUEUES.get(job_id, []):
        await q.put(msg)


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
    }


async def _execute_pipeline_on_mp4(
    job_id: str,
    mp4_path: Path,
    part_prompts: list[str],
    use_live_sam_api: bool = False,
    perturb_initial_anchor_for_demo: bool = False,
) -> dict[str, Any]:
    sam_service = SamPerceptionService()

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

    if use_live_sam_api:
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
    JOB_BLENDER_SCRIPTS[job_id] = bpy_script
    JOB_VIDEO_PATHS[job_id] = mp4_path

    now_iso = datetime.now(UTC).isoformat()
    job_payload: dict[str, Any] = {
        "id": job_id,
        "status": "COMPLETED",
        "videoUrl": f"http://localhost:8000/api/v1/jobs/{job_id}/video.mp4",
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
        "blenderScriptDownloadUrl": f"http://localhost:8000/api/v1/jobs/{job_id}/blender.py",
        "errorCode": None,
        "errorMessage": None,
        "createdAt": now_iso,
        "updatedAt": now_iso,
    }

    JOBS_STORE[job_id] = job_payload
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


@app.post("/api/v1/jobs/demo-hinge", status_code=status.HTTP_201_CREATED)
async def create_verified_demo_hinge_job() -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    mp4_path = ARTIFACTS_DIR / f"{job_id}.mp4"
    await asyncio.to_thread(generate_controlled_hinge_mp4, mp4_path, 36, 30.0)
    return await _execute_pipeline_on_mp4(
        job_id=job_id,
        mp4_path=mp4_path,
        part_prompts=["static_anchor_base", "rotating_pendulum_arm"],
        use_live_sam_api=False,
        perturb_initial_anchor_for_demo=True,
    )


@app.post("/api/v1/jobs", status_code=status.HTTP_201_CREATED)
async def upload_and_create_rig_job(
    video: UploadFile = File(...),
    part_prompts_csv: str = Form("static_base,rotating_arm"),
    use_live_sam: bool = Form(False),
) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    mp4_path = ARTIFACTS_DIR / f"{job_id}.mp4"
    raw_bytes = await video.read()
    if len(raw_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Video exceeds 50MB limit")
    await asyncio.to_thread(mp4_path.write_bytes, raw_bytes)

    prompts = [p.strip() for p in part_prompts_csv.split(",") if p.strip()]
    return await _execute_pipeline_on_mp4(
        job_id=job_id,
        mp4_path=mp4_path,
        part_prompts=prompts,
        use_live_sam_api=use_live_sam,
        perturb_initial_anchor_for_demo=False,
    )


@app.get("/api/v1/jobs/{job_id}")
async def get_rig_job(job_id: str) -> dict[str, Any]:
    if job_id not in JOBS_STORE:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JOBS_STORE[job_id]


@app.get("/api/v1/jobs/{job_id}/blender.py", response_class=PlainTextResponse)
async def download_blender_script(job_id: str) -> str:
    if job_id not in JOB_BLENDER_SCRIPTS:
        raise HTTPException(status_code=404, detail=f"Blender script for job {job_id} not found")
    return JOB_BLENDER_SCRIPTS[job_id]


@app.get("/api/v1/jobs/{job_id}/video.mp4")
async def stream_job_video(job_id: str) -> FileResponse:
    path = JOB_VIDEO_PATHS.get(job_id)
    if not path or not await asyncio.to_thread(path.exists):
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path, media_type="video/mp4")


@app.websocket("/api/v1/jobs/{job_id}/stream")
async def job_telemetry_websocket(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    queue: asyncio.Queue[str] = asyncio.Queue()
    JOB_EVENT_QUEUES.setdefault(job_id, []).append(queue)

    try:
        current_job = JOBS_STORE.get(job_id)
        await websocket.send_text(
            json.dumps(
                {
                    "event": "STATUS_CHANGED",
                    "jobId": job_id,
                    "status": current_job["status"] if current_job else "QUEUED",
                    "progressPct": 100 if current_job else 10,
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
