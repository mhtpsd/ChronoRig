from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.core.security import create_jwt_token
from src.main import app, worker_engine
from src.modules.perception.sam_service import (
    SamPerceptionService,
    generate_controlled_hinge_mp4,
)
from src.modules.synthesis.synthesizer import validate_blender_script_ast


def test_jwt_authenticated_demo_hinge_and_cross_tenant_isolation() -> None:
    client = TestClient(app)

    # 1. Issue signed HS256 JWTs for two distinct tenants
    alpha_resp = client.post("/api/v1/auth/token", json={"tenantId": "tenant-alpha"})
    assert alpha_resp.status_code == 200
    alpha_jwt = alpha_resp.json()["accessToken"]
    beta_jwt = create_jwt_token(subject="tenant-beta")

    # 2. Tenant Alpha runs the demo-hinge reconstruction
    resp = client.post(
        "/api/v1/jobs/demo-hinge",
        headers={"Authorization": f"Bearer {alpha_jwt}"},
    )
    assert resp.status_code == 201
    job = resp.json()
    job_id = job["id"]
    assert job["status"] == "COMPLETED"
    assert job["finalIouScore"] >= 0.85

    # 3. Tenant Alpha can read; Tenant Beta gets 403; forged JWT gets 401
    assert (
        client.get(
            f"/api/v1/jobs/{job_id}",
            headers={"Authorization": f"Bearer {alpha_jwt}"},
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"/api/v1/jobs/{job_id}",
            headers={"Authorization": f"Bearer {beta_jwt}"},
        ).status_code
        == 403
    )
    assert (
        client.get(
            f"/api/v1/jobs/{job_id}",
            headers={"Authorization": f"Bearer {alpha_jwt}tampered"},
        ).status_code
        == 401
    )

    # 4. WebSocket enforces tenant ownership before accepting connection
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/api/v1/jobs/{job_id}/stream?token={beta_jwt}"):
            pass
    assert exc_info.value.code == 1008

    # Tenant Alpha connects to WebSocket and reads from the durable SQLite WAL event outbox
    with client.websocket_connect(f"/api/v1/jobs/{job_id}/stream?token={alpha_jwt}") as ws:
        handshake = ws.receive_json()
        assert handshake["jobId"] == job_id


@pytest.mark.asyncio
async def test_openapi_presign_direct_upload_and_json_job_creation_with_3_parts() -> None:
    client = TestClient(app)
    jwt_token = create_jwt_token(subject="openapi-client")
    headers = {"Authorization": f"Bearer {jwt_token}"}

    # 1. Request presigned upload URL
    presign = client.post(
        "/api/v1/uploads/presign",
        json={
            "filename": "three_part_mech.mp4",
            "contentType": "video/mp4",
            "fileSizeBytes": 65536,
        },
        headers=headers,
    )
    assert presign.status_code == 200
    s3_key = presign.json()["videoS3Key"]
    upload_id = s3_key.split("/")[1].replace(".mp4", "")

    # 2. Reject non-video magic bytes (spoofed text file) at direct upload endpoint
    bad_upload = client.put(
        f"/api/v1/uploads/direct/{upload_id}",
        content=b"NOT_A_VALID_MP4_HEADER_1234567890",
        headers=headers,
    )
    assert bad_upload.status_code == 415

    # 3. Upload valid MP4 binary to PUT /api/v1/uploads/direct/{upload_id}
    with tempfile.TemporaryDirectory() as tmpdir:
        mp4_file = Path(tmpdir) / "valid.mp4"
        generate_controlled_hinge_mp4(mp4_file, frame_count=15, fps=30.0)
        raw_mp4 = mp4_file.read_bytes()

    good_upload = client.put(
        f"/api/v1/uploads/direct/{upload_id}",
        content=raw_mp4,
        headers=headers,
    )
    assert good_upload.status_code == 201
    assert good_upload.json()["videoFrameCount"] == 15

    # 4. Create job via OpenAPI JSON contract with 3 partPrompts (verifying N>2 support)
    create_resp = client.post(
        "/api/v1/jobs",
        json={
            "videoS3Key": s3_key,
            "partPrompts": ["static_base", "lower_arm", "upper_tip"],
            "targetIouThreshold": 0.80,
            "maxSelfDebugIterations": 2,
            "useLiveSam": False,
        },
        headers=headers,
    )
    assert create_resp.status_code == 202
    job_id = create_resp.json()["id"]

    # Process via durable worker engine
    await worker_engine.process_next_available_job()

    for _ in range(40):
        poll_resp = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        assert poll_resp.status_code == 200
        completed = poll_resp.json()
        if completed.get("status") == "COMPLETED":
            break
        await asyncio.sleep(0.05)

    assert completed["status"] == "COMPLETED"
    assert len(completed["parts"]) == 3


def test_native_sam_rle_stream_decoder_and_ast_guard() -> None:
    raw_stream = "<frame=0><rle=10,5,20><frame=1><rle=8,7,20>"
    masks = SamPerceptionService._decode_sam_stream_to_masks(
        raw_stream, frame_width=7, frame_height=5
    )
    assert int(masks[0].sum()) == 5
    assert int(masks[1].sum()) == 7

    with pytest.raises(ValueError, match="Forbidden import 'subprocess'"):
        validate_blender_script_ast("import bpy\nimport subprocess\n")
