from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.config import settings
from src.core.repository import DurableJobRepository
from src.modules.perception.sam_service import ExtractedJointConstraint, ExtractedPartTrack
from src.modules.synthesis.synthesizer import generate_deterministic_blender_script


def create_rotated_rect_points(center_x: float, center_y: float, length: float, width: float, angle_rad: float) -> np.ndarray:
    """Returns the 4 corner points of an oriented rectangle."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    half_l = length / 2.0
    half_w = width / 2.0

    # Corners relative to center
    dx1, dy1 = half_l * cos_a - half_w * sin_a, half_l * sin_a + half_w * cos_a
    dx2, dy2 = -half_l * cos_a - half_w * sin_a, -half_l * sin_a + half_w * cos_a
    dx3, dy3 = -half_l * cos_a + half_w * sin_a, -half_l * sin_a - half_w * cos_a
    dx4, dy4 = half_l * cos_a + half_w * sin_a, half_l * sin_a - half_w * cos_a

    pts = np.array([
        [center_x + dx1, center_y + dy1],
        [center_x + dx2, center_y + dy2],
        [center_x + dx3, center_y + dy3],
        [center_x + dx4, center_y + dy4],
    ], dtype=np.float32)
    return pts


def generate_robot_arm_demo(
    output_mp4_path: Path,
    frame_count: int = 90,
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Generates a high-quality 90-frame MP4 video of the 3-part articulated robot arm
    matching the exact kinematics and joints of job 11111111-2222-4333-8444-555555555555:
    - Base: heavy base at (320, 360)
    - Lower Arm: swinging around Joint 1 (320, 331)
    - Upper Actuator: swinging around Joint 2 (elbow)
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4_path), fourcc, fps, (width, height))

    j1_px = (320.0, 331.0)
    arm1_len = 154.0
    arm1_w = 40.0
    arm2_len = 135.0
    arm2_w = 34.0

    base_pts = np.array([
        [240, 335],
        [400, 335],
        [400, 385],
        [240, 385],
    ], dtype=np.int32)
    base_centroid = [320.0, 360.0]
    base_bbox = [240, 335, 160, 50]
    base_poly = [[float(p[0]), float(p[1])] for p in base_pts]

    base_trajectory: list[dict[str, Any]] = []
    arm1_trajectory: list[dict[str, Any]] = []
    arm2_trajectory: list[dict[str, Any]] = []

    for f_idx in range(frame_count):
        t_sec = round(f_idx / fps, 4)
        phase = 2.0 * math.pi * f_idx / frame_count

        # Joint 1 angular position (harmonic oscillation around vertical)
        theta1 = 0.35 * math.sin(phase)
        arm1_dir_angle = -math.pi / 2.0 + theta1

        # Center and elbow position
        c1_x = j1_px[0] + (arm1_len / 2.0) * math.sin(theta1)
        c1_y = j1_px[1] - (arm1_len / 2.0) * math.cos(theta1)
        j2_x = j1_px[0] + arm1_len * math.sin(theta1)
        j2_y = j1_px[1] - arm1_len * math.cos(theta1)

        arm1_pts = create_rotated_rect_points(c1_x, c1_y, arm1_len, arm1_w, arm1_dir_angle)
        arm1_int_pts = np.int32(arm1_pts)
        bx1, by1, bw1, bh1 = cv2.boundingRect(arm1_int_pts)

        # Upper actuator arm angle (articulating from elbow)
        theta2 = -0.22 + 0.38 * math.sin(phase - 0.45)
        c2_x = j2_x + (arm2_len / 2.0) * math.cos(theta2)
        c2_y = j2_y + (arm2_len / 2.0) * math.sin(theta2)

        arm2_pts = create_rotated_rect_points(c2_x, c2_y, arm2_len, arm2_w, theta2)
        arm2_int_pts = np.int32(arm2_pts)
        bx2, by2, bw2, bh2 = cv2.boundingRect(arm2_int_pts)

        # Draw Frame into canvas
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background subtle studio dark slate
        frame[:] = (15, 12, 10)

        # Draw Heavy Base (cyan/blue metallic)
        cv2.fillPoly(frame, [base_pts], (90, 70, 40))
        cv2.polylines(frame, [base_pts], True, (212, 245, 0), 2)

        # Draw Lower Arm (magenta/slate metallic)
        cv2.fillPoly(frame, [arm1_int_pts], (75, 45, 110))
        cv2.polylines(frame, [arm1_int_pts], True, (133, 42, 255), 2)

        # Draw Upper Actuator Arm (yellow/gold metallic)
        cv2.fillPoly(frame, [arm2_int_pts], (40, 95, 115))
        cv2.polylines(frame, [arm2_int_pts], True, (64, 228, 254), 2)

        # Draw Joint pins
        cv2.circle(frame, (int(j1_px[0]), int(j1_px[1])), 8, (64, 228, 254), -1)
        cv2.circle(frame, (int(j1_px[0]), int(j1_px[1])), 9, (255, 255, 255), 1)

        cv2.circle(frame, (int(j2_x), int(j2_y)), 7, (64, 228, 254), -1)
        cv2.circle(frame, (int(j2_x), int(j2_y)), 8, (255, 255, 255), 1)

        writer.write(frame)

        # Save trajectory frame snapshots
        base_trajectory.append({
            "frameIndex": f_idx,
            "timestampSec": t_sec,
            "centroid2dPx": [round(base_centroid[0], 2), round(base_centroid[1], 2)],
            "angleRad": 0.0,
            "bboxPx": base_bbox,
            "contourPolygonPx": base_poly,
        })

        arm1_trajectory.append({
            "frameIndex": f_idx,
            "timestampSec": t_sec,
            "centroid2dPx": [round(c1_x, 2), round(c1_y, 2)],
            "angleRad": round(arm1_dir_angle, 4),
            "bboxPx": [int(bx1), int(by1), int(bw1), int(bh1)],
            "contourPolygonPx": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in arm1_pts],
        })

        arm2_trajectory.append({
            "frameIndex": f_idx,
            "timestampSec": t_sec,
            "centroid2dPx": [round(c2_x, 2), round(c2_y, 2)],
            "angleRad": round(theta2, 4),
            "bboxPx": [int(bx2), int(by2), int(bw2), int(bh2)],
            "contourPolygonPx": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in arm2_pts],
        })

    writer.release()

    parts_payload = [
        {
            "id": "aaaa1111-1111-4111-8111-111111111111",
            "samObjectId": 1,
            "label": "heavy base",
            "isStaticRoot": True,
            "estimatedMassKg": 5.0,
            "boundingBoxExtents": [1.4, 0.35, 1.0],
            "trajectoryFrames": base_trajectory,
            "textureUrl": None,
        },
        {
            "id": "bbbb2222-2222-4222-8222-222222222222",
            "samObjectId": 2,
            "label": "lower arm",
            "isStaticRoot": False,
            "estimatedMassKg": 1.5,
            "boundingBoxExtents": [0.3, 1.4, 0.3],
            "trajectoryFrames": arm1_trajectory,
            "textureUrl": None,
        },
        {
            "id": "cccc3333-3333-4333-8333-333333333333",
            "samObjectId": 3,
            "label": "upper actuator arm",
            "isStaticRoot": False,
            "estimatedMassKg": 1.0,
            "boundingBoxExtents": [1.2, 0.28, 0.28],
            "trajectoryFrames": arm2_trajectory,
            "textureUrl": None,
        },
    ]

    joints_payload = [
        {
            "id": "dddd4444-4444-4444-8444-444444444444",
            "parentPartId": "aaaa1111-1111-4111-8111-111111111111",
            "childPartId": "bbbb2222-2222-4222-8222-222222222222",
            "jointType": "revolute",
            "anchorWorldXyz": [0.0, 0.55, 0.0],
            "axisXyz": [0.0, 0.0, 1.0],
            "limits": {"min": -1.2, "max": 1.2, "stiffness": 0.0, "damping": 0.5},
        },
        {
            "id": "eeee5555-5555-4555-8555-555555555555",
            "parentPartId": "bbbb2222-2222-4222-8222-222222222222",
            "childPartId": "cccc3333-3333-4333-8333-333333333333",
            "jointType": "revolute",
            "anchorWorldXyz": [0.1, 1.65, 0.0],
            "axisXyz": [0.0, 0.0, 1.0],
            "limits": {"min": -1.5, "max": 1.5, "stiffness": 0.0, "damping": 0.4},
        },
    ]

    return parts_payload, joints_payload


if __name__ == "__main__":
    job_id = "11111111-2222-4333-8444-555555555555"
    artifacts_dir = settings.ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    mp4_target = artifacts_dir / f"{job_id}.mp4"

    print(f"Generating 90-frame articulated robot arm video to {mp4_target}...")
    parts_data, joints_data = generate_robot_arm_demo(mp4_target, frame_count=90, fps=30.0)

    # Convert to domain tracks for blender synthesizer
    tracks = [
        ExtractedPartTrack(
            part_id=p["id"],
            sam_object_id=p["samObjectId"],
            label=p["label"],
            is_static_root=p["isStaticRoot"],
            estimated_mass_kg=p["estimatedMassKg"],
            bounding_box_extents=p["boundingBoxExtents"],
            trajectory_frames=p["trajectoryFrames"],
            keyframes_binary_masks={},
        )
        for p in parts_data
    ]

    joints = [
        ExtractedJointConstraint(
            joint_id=j["id"],
            parent_part_id=j["parentPartId"],
            child_part_id=j["childPartId"],
            joint_type=j["jointType"],
            anchor_world_xyz=j["anchorWorldXyz"],
            axis_xyz=j["axisXyz"],
            limits=j["limits"],
        )
        for j in joints_data
    ]

    bpy_script = generate_deterministic_blender_script(parts=tracks, joints=joints, video_fps=30.0)

    repo = DurableJobRepository()
    row = repo.get_job_row(job_id)
    if row:
        payload = row["payload"]
        payload["parts"] = parts_data
        payload["joints"] = joints_data
        payload["videoFrameCount"] = 90
        payload["videoFps"] = 30.0
        payload["videoUrl"] = f"http://127.0.0.1:8000/api/v1/jobs/{job_id}/video.mp4"
        payload["blenderScriptDownloadUrl"] = f"http://127.0.0.1:8000/api/v1/jobs/{job_id}/blender.py"
        repo.save_job(
            job_id=job_id,
            owner_id=row["owner_id"],
            payload=payload,
            blender_script=bpy_script,
            video_path=mp4_target,
        )
        print("Updated job 11111111-2222-4333-8444-555555555555 in SQLite database successfully!")
    else:
        print("Job row not found in repo.")
