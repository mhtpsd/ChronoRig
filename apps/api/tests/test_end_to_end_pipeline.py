from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.modules.synthesis.synthesizer import validate_blender_script_ast


def test_end_to_end_demo_hinge_pipeline_and_iou_convergence() -> None:
    client = TestClient(app)

    # 1. Trigger end-to-end reconstruction on a real generated 36-frame MP4 video
    resp = client.post("/api/v1/jobs/demo-hinge")
    assert resp.status_code == 201
    job = resp.json()

    job_id = job["id"]
    assert job["status"] == "COMPLETED"
    assert len(job["parts"]) == 2
    assert len(job["joints"]) == 1

    # Verify static root vs dynamic rotating arm detection
    root_parts = [p for p in job["parts"] if p["isStaticRoot"]]
    dynamic_parts = [p for p in job["parts"] if not p["isStaticRoot"]]
    assert len(root_parts) == 1
    assert len(dynamic_parts) == 1
    assert root_parts[0]["label"] == "static_anchor_base"
    assert dynamic_parts[0]["label"] == "rotating_pendulum_arm"

    # Verify each frame includes real extracted contour polygons
    first_frame = dynamic_parts[0]["trajectoryFrames"][0]
    assert len(first_frame["contourPolygonPx"]) >= 4

    # Verify the joint was classified as revolute
    joint = job["joints"][0]
    assert joint["jointType"] == "revolute"

    # Verify ClosedLoopIouVerifier self-debugged the perturbed Iteration 1 anchor
    iterations = job["iouIterations"]
    assert len(iterations) >= 2
    iter1_iou = iterations[0]["meanIou"]
    final_iou = job["finalIouScore"]
    assert iter1_iou < 0.75  # Perturbed initial anchor has lower IoU
    assert final_iou >= 0.85  # Self-corrected anchor exceeds 0.85 threshold
    assert final_iou > iter1_iou

    # 2. Verify GET /api/v1/jobs/{job_id} returns identical contract
    get_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["finalIouScore"] == final_iou

    # 3. Verify GET /api/v1/jobs/{job_id}/blender.py returns AST-safe Blender 4.x Python code
    bpy_resp = client.get(f"/api/v1/jobs/{job_id}/blender.py")
    assert bpy_resp.status_code == 200
    bpy_code = bpy_resp.text
    assert "import bpy" in bpy_code
    assert "rbc.type = 'HINGE'" in bpy_code
    assert "keyframe_insert(data_path='rotation_euler'" in bpy_code
    validate_blender_script_ast(bpy_code)


def test_blender_ast_guard_blocks_malicious_imports() -> None:
    malicious_code = "import bpy\nimport subprocess\nsubprocess.Popen(['calc.exe'])\n"
    with pytest.raises(ValueError, match="Forbidden import 'subprocess'"):
        validate_blender_script_ast(malicious_code)
