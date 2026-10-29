from __future__ import annotations

import copy
import math
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np

from src.modules.perception.kinematics import compute_binary_mask_iou
from src.modules.perception.sam_service import (
    ExtractedJointConstraint,
    ExtractedPartTrack,
)
from src.modules.synthesis.synthesizer import PhysicsRigSynthesizer


def world_xyz_to_pixel_xy(
    anchor_world_xyz: list[float], frame_width: int = 640, frame_height: int = 480
) -> tuple[float, float]:
    """Inverse of the pixel-to-normalized-world projection used in SamPerceptionService."""
    px = anchor_world_xyz[0] * 140.0 + (frame_width / 2.0)
    py = (frame_height / 2.0) - (anchor_world_xyz[1] - 1.2) * 140.0
    return (px, py)


def rasterize_articulated_rig_part_mask(
    part: ExtractedPartTrack,
    parent_joint: ExtractedJointConstraint | None,
    frame_idx: int,
    frame_shape: tuple[int, int] = (480, 640),
) -> np.ndarray:
    """
    True Forward-Kinematics Silhouette Rasterizer:
    Takes the part's Frame-0 contour polygon and transforms it to `frame_idx`
    strictly using the candidate joint's anchor pivot and constraint type.
    """
    h, w = frame_shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    if not part.trajectory_frames:
        return canvas

    frame0 = part.trajectory_frames[0]
    poly0 = np.asarray(frame0.get("contourPolygonPx", []), dtype=np.float64)
    if poly0.ndim != 2 or len(poly0) < 3:
        bx, by, bw, bh = frame0["bboxPx"]
        poly0 = np.array(
            [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]],
            dtype=np.float64,
        )

    # Static root does not move
    if part.is_static_root or parent_joint is None:
        pts_i32 = np.round(poly0).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts_i32], 1)
        return canvas

    target_pose = next(
        (f for f in part.trajectory_frames if f["frameIndex"] == frame_idx),
        frame0,
    )

    if parent_joint.joint_type == "revolute":
        # Rotate Frame-0 polygon strictly around the candidate joint's hinge pivot
        pivot_px = np.array(
            world_xyz_to_pixel_xy(parent_joint.anchor_world_xyz, w, h),
            dtype=np.float64,
        )
        theta0 = float(frame0.get("angleRad") or 0.0)
        theta_t = float(target_pose.get("angleRad") or theta0)
        d_theta = theta_t - theta0

        cos_t = math.cos(d_theta)
        sin_t = math.sin(d_theta)
        rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)

        transformed = (poly0 - pivot_px) @ rot_mat.T + pivot_px
    elif parent_joint.joint_type == "prismatic":
        # Translate Frame-0 polygon strictly along the joint's unit axis vector
        axis_2d = np.array([parent_joint.axis_xyz[0], -parent_joint.axis_xyz[1]], dtype=np.float64)
        norm = float(np.linalg.norm(axis_2d))
        if norm > 1e-6:
            axis_2d /= norm
        c0 = np.array(frame0["centroid2dPx"], dtype=np.float64)
        ct = np.array(target_pose["centroid2dPx"], dtype=np.float64)
        proj_dist = float(np.dot(ct - c0, axis_2d))
        transformed = poly0 + axis_2d * proj_dist
    else:
        transformed = poly0

    pts_i32 = np.round(transformed).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [pts_i32], 1)
    return canvas


class ClosedLoopIouVerifier:
    """
    Verifiable Agent controller that compares forward-kinematic simulated part
    silhouettes against per-part SAM 3.1 binary masks at each keyframe.
    """

    def __init__(self, synthesizer: PhysicsRigSynthesizer) -> None:
        self._synthesizer = synthesizer

    async def run_verification_loop(
        self,
        job_id: str,
        parts: list[ExtractedPartTrack],
        joints: list[ExtractedJointConstraint],
        target_iou: float,
        max_iterations: int,
        broadcast_callback: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> tuple[list[ExtractedJointConstraint], list[dict[str, Any]], float]:
        iterations_history: list[dict[str, Any]] = []
        best_iou = -1.0
        best_joints = copy.deepcopy(joints)
        current_joints = copy.deepcopy(joints)

        for iteration in range(1, max_iterations + 1):
            joint_by_child = {j.child_part_id: j for j in current_joints}
            frame_ious: dict[str, float] = {}

            # Evaluate per-part, per-frame IoU across keyframes
            sample_frames = sorted(
                {f_idx for p in parts for f_idx in p.keyframes_binary_masks.keys()}
            ) or [0, 30, 60]

            for frame_idx in sample_frames:
                part_scores: list[float] = []
                for part in parts:
                    gt_mask = part.keyframes_binary_masks.get(frame_idx)
                    if gt_mask is None:
                        continue
                    sim_mask = rasterize_articulated_rig_part_mask(
                        part=part,
                        parent_joint=joint_by_child.get(part.part_id),
                        frame_idx=frame_idx,
                        frame_shape=gt_mask.shape,
                    )
                    part_scores.append(compute_binary_mask_iou(gt_mask, sim_mask))

                frame_ious[str(frame_idx)] = (
                    round(float(np.mean(part_scores)), 4) if part_scores else 0.0
                )

            mean_iou = round(float(np.mean(list(frame_ious.values()))), 4) if frame_ious else 0.0

            # Record best_joints BEFORE any mutation so score and joints strictly match
            if mean_iou > best_iou:
                best_iou = mean_iou
                best_joints = copy.deepcopy(current_joints)

            if mean_iou >= target_iou:
                status_note = (
                    f"+ Verified meanIoU = {mean_iou:.3f} (>= {target_iou:.2f} target). "
                    "Locking verified joint anchors."
                )
                next_joints = current_joints
            elif iteration < max_iterations:
                next_joints, status_note = await self._synthesizer.propose_iou_self_debug_patch(
                    joints=copy.deepcopy(current_joints),
                    frame_iou_breakdown=frame_ious,
                    iteration_number=iteration,
                )
            else:
                status_note = (
                    f"! Reached max_iterations ({max_iterations}) with meanIoU = {mean_iou:.3f} "
                    f"(below {target_iou:.2f} target). Retaining highest-scoring candidate ({best_iou:.3f})."
                )
                next_joints = current_joints

            iter_record = {
                "iterationNumber": iteration,
                "meanIou": mean_iou,
                "frameIouBreakdown": frame_ious,
                "agentDiffPatch": status_note,
                "createdAt": datetime.now(UTC).isoformat(),
            }
            iterations_history.append(iter_record)

            await broadcast_callback(
                {
                    "event": "IOU_ITERATION_COMPLETED",
                    "jobId": job_id,
                    "iteration": iter_record,
                }
            )

            if mean_iou >= target_iou:
                break

            current_joints = next_joints

        return best_joints, iterations_history, max(0.0, best_iou)
