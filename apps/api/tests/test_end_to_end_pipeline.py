from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.modules.perception.sam_service import (
    SamPerceptionService,
    generate_controlled_hinge_mp4,
)
from src.modules.synthesis.synthesizer import validate_blender_script_ast


def test_end_to_end_demo_hinge_pipeline_and_iou_convergence() -> None:
    client = TestClient(app)

    resp = client.post("/api/v1/jobs/demo-hinge", headers={"X-Owner-Id": "tenant-alpha"})
    assert resp.status_code == 201
    job = resp.json()

    job_id = job["id"]
    assert job["status"] == "COMPLETED"
    assert len(job["parts"]) == 2
    assert len(job["joints"]) == 1

    iterations = job["iouIterations"]
    assert len(iterations) >= 2
    assert iterations[0]["meanIou"] < 0.75
    assert job["finalIouScore"] >= 0.85

    # Verify owner authorization: tenant-alpha succeeds (200), tenant-beta is blocked (403)
    ok_resp = client.get(f"/api/v1/jobs/{job_id}", headers={"X-Owner-Id": "tenant-alpha"})
    assert ok_resp.status_code == 200

    forbidden_resp = client.get(f"/api/v1/jobs/{job_id}", headers={"X-Owner-Id": "tenant-beta"})
    assert forbidden_resp.status_code == 403

    bpy_resp = client.get(
        f"/api/v1/jobs/{job_id}/blender.py", headers={"X-Owner-Id": "tenant-alpha"}
    )
    assert bpy_resp.status_code == 200
    validate_blender_script_ast(bpy_resp.text)


def test_async_202_upload_lifecycle_and_presign() -> None:
    client = TestClient(app)

    # 1. Verify presign endpoint matches OpenAPI contract
    presign_resp = client.post(
        "/api/v1/uploads/presign",
        json={
            "filename": "mechanism.mp4",
            "contentType": "video/mp4",
            "fileSizeBytes": 102400,
        },
    )
    assert presign_resp.status_code == 200
    assert "videoS3Key" in presign_resp.json()

    # 2. Upload a real generated MP4 to POST /api/v1/jobs and verify 202 Accepted
    with tempfile.TemporaryDirectory() as tmpdir:
        mp4_file = Path(tmpdir) / "upload_test.mp4"
        generate_controlled_hinge_mp4(mp4_file, frame_count=12, fps=30.0)
        raw_mp4 = mp4_file.read_bytes()

    upload_resp = client.post(
        "/api/v1/jobs",
        files={"video": ("upload_test.mp4", raw_mp4, "video/mp4")},
        data={"part_prompts_csv": "base,arm", "use_live_sam": "false"},
        headers={"X-Owner-Id": "uploader-1"},
    )
    assert upload_resp.status_code == 202
    job_id = upload_resp.json()["id"]

    # BackgroundTask runs after response in TestClient; polling GET should now be COMPLETED
    poll_resp = client.get(f"/api/v1/jobs/{job_id}", headers={"X-Owner-Id": "uploader-1"})
    assert poll_resp.status_code == 200
    assert poll_resp.json()["status"] == "COMPLETED"


def test_native_sam_rle_stream_decoder() -> None:
    raw_stream = "<frame=0><rle=10,5,20><frame=1><rle=8,7,20>"
    masks = SamPerceptionService._decode_sam_stream_to_masks(
        raw_stream, frame_width=7, frame_height=5
    )
    assert 0 in masks and 1 in masks
    assert int(masks[0].sum()) == 5
    assert int(masks[1].sum()) == 7


def test_blender_ast_guard_blocks_malicious_imports() -> None:
    malicious_code = "import bpy\nimport subprocess\nsubprocess.Popen(['calc.exe'])\n"
    with pytest.raises(ValueError, match="Forbidden import 'subprocess'"):
        validate_blender_script_ast(malicious_code)
