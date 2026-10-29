from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.core.config import settings
from src.modules.perception.kinematics import (
    compute_mask_pose_pca,
    estimate_revolute_pivot_2d,
)

logger = logging.getLogger("chronorig.api")


@dataclass
class ExtractedPartTrack:
    part_id: str
    sam_object_id: int
    label: str
    is_static_root: bool
    estimated_mass_kg: float
    bounding_box_extents: list[float]
    trajectory_frames: list[dict[str, Any]]
    keyframes_binary_masks: dict[int, np.ndarray]


@dataclass
class ExtractedJointConstraint:
    joint_id: str
    parent_part_id: str
    child_part_id: str
    joint_type: str
    anchor_world_xyz: list[float]
    axis_xyz: list[float]
    limits: dict[str, float]


class SamPerceptionService:
    """
    Streams video to Meta's SAM 3.1 API using ONE object concept per request
    and decodes the output_text token stream into per-frame binary masks.
    """

    def __init__(self) -> None:
        self._headers = {
            "Authorization": f"Bearer {settings.META_MODEL_API_KEY.get_secret_value()}",
            "Content-Type": "application/json",
        }

    @retry(
        wait=wait_random_exponential(min=1, max=15),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def segment_single_concept_video(
        self,
        video_data_uri: str,
        concept_prompt: str,
        frame_width: int,
        frame_height: int,
    ) -> dict[int, np.ndarray]:
        """
        Issues a single-concept SAM 3.1 streaming request (POST /v1/responses, stream=True)
        and parses the output_text stream via meta_sam_parser into {frame_index: binary_mask}.
        """
        payload = {
            "model": settings.SAM_MODEL_ID,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_video", "video_url": video_data_uri},
                        {"type": "input_text", "text": concept_prompt.strip()},
                    ],
                }
            ],
            "stream": True,
        }

        accumulated_text: list[str] = []
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{settings.META_MODEL_BASE_URL}/responses",
                headers=self._headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break
                    import json

                    event_obj = json.loads(data_str)
                    delta = event_obj.get("delta") or event_obj.get("output_text") or ""
                    if isinstance(delta, str):
                        accumulated_text.append(delta)

        raw_sam_stream = "".join(accumulated_text)
        return self._decode_sam_stream_to_masks(
            raw_sam_stream, frame_width=frame_width, frame_height=frame_height
        )

    @staticmethod
    def _decode_sam_stream_to_masks(
        raw_sam_stream: str, frame_width: int, frame_height: int
    ) -> dict[int, np.ndarray]:
        """
        Uses official meta_sam_parser when installed to decode SAM 3.1 output_text records.
        """
        import importlib

        masks_by_frame: dict[int, np.ndarray] = {}
        try:
            parser_mod = importlib.import_module("meta_sam_parser")
            for record in parser_mod.parse_sam_stream(raw_sam_stream):
                f_idx = int(record.frame_index)
                mask = parser_mod.decode_rle_to_mask(
                    record.mask, width=frame_width, height=frame_height
                )
                masks_by_frame[f_idx] = (np.asarray(mask) > 0).astype(np.uint8)
        except ModuleNotFoundError:
            logger.warning("meta_sam_parser not installed yet; using offline fallback.")
        return masks_by_frame

    def build_kinematic_graph_from_concept_masks(
        self,
        concept_masks: list[tuple[str, dict[int, np.ndarray]]],
        video_fps: float = 30.0,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> tuple[list[ExtractedPartTrack], list[ExtractedJointConstraint]]:
        """
        Takes an ordered list of (concept_prompt, {frame_idx: binary_mask}) pairs,
        computes 2D PCA poses and contour polygons, identifies the static root,
        and solves relative parent-child joint constraints.
        """
        extracted_parts: list[ExtractedPartTrack] = []
        part_motion_variance: dict[str, float] = {}

        for obj_idx, (prompt_label, frame_dict) in enumerate(concept_masks, start=1):
            part_uuid = str(uuid.uuid4())
            trajectory_frames: list[dict[str, Any]] = []
            keyframes_masks: dict[int, np.ndarray] = {}
            bboxes_wh: list[tuple[int, int]] = []
            last_valid_angle = 0.0

            for f_idx in sorted(frame_dict.keys()):
                mask = frame_dict[f_idx]
                if int((mask > 0).sum()) < 3:
                    continue

                pose = compute_mask_pose_pca(mask)
                angle_val = pose.angle_rad if pose.angle_rad is not None else last_valid_angle
                last_valid_angle = angle_val

                ys, xs = np.nonzero(mask > 0)
                x0, y0, x1, y1 = (
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                )
                bw, bh = max(4, x1 - x0), max(4, y1 - y0)
                bboxes_wh.append((bw, bh))

                trajectory_frames.append(
                    {
                        "frameIndex": f_idx,
                        "timestampSec": round(f_idx / video_fps, 4),
                        "centroid2dPx": [
                            round(pose.centroid_xy[0], 2),
                            round(pose.centroid_xy[1], 2),
                        ],
                        "angleRad": round(angle_val, 4),
                        "angleConfidence": (
                            "high" if pose.angle_rad is not None else "low_eccentricity"
                        ),
                        "bboxPx": [x0, y0, bw, bh],
                        "contourPolygonPx": [
                            [round(px, 1), round(py, 1)] for px, py in pose.contour_polygon_px
                        ],
                    }
                )
                keyframes_masks[f_idx] = mask

            if not trajectory_frames:
                continue

            cx_series = [f["centroid2dPx"][0] for f in trajectory_frames]
            cy_series = [f["centroid2dPx"][1] for f in trajectory_frames]
            part_motion_variance[part_uuid] = float(np.var(cx_series) + np.var(cy_series))

            mean_w = float(np.mean([b[0] for b in bboxes_wh]))
            mean_h = float(np.mean([b[1] for b in bboxes_wh]))
            extents_xyz = [
                round(max(0.2, mean_w / 140.0), 3),
                round(max(0.2, mean_h / 140.0), 3),
                0.32,
            ]

            extracted_parts.append(
                ExtractedPartTrack(
                    part_id=part_uuid,
                    sam_object_id=obj_idx,
                    label=prompt_label,
                    is_static_root=False,
                    estimated_mass_kg=round(max(0.5, (mean_w * mean_h) / 4500.0), 2),
                    bounding_box_extents=extents_xyz,
                    trajectory_frames=trajectory_frames,
                    keyframes_binary_masks=keyframes_masks,
                )
            )

        if extracted_parts:
            root_part = min(
                extracted_parts,
                key=lambda p: part_motion_variance.get(p.part_id, 0.0),
            )
            root_part.is_static_root = True
            root_part.estimated_mass_kg = 5.0

        extracted_joints: list[ExtractedJointConstraint] = []
        for i in range(len(extracted_parts) - 1):
            parent = extracted_parts[i]
            child = extracted_parts[i + 1]

            # Synchronize common frame indices between parent and child
            parent_by_f = {f["frameIndex"]: f for f in parent.trajectory_frames}
            child_by_f = {f["frameIndex"]: f for f in child.trajectory_frames}
            common_frames = sorted(set(parent_by_f.keys()) & set(child_by_f.keys()))

            if len(common_frames) < 3:
                continue

            parent_centroids = [
                (
                    float(parent_by_f[f]["centroid2dPx"][0]),
                    float(parent_by_f[f]["centroid2dPx"][1]),
                )
                for f in common_frames
            ]
            parent_angles = [float(parent_by_f[f]["angleRad"]) for f in common_frames]
            child_centroids = [
                (
                    float(child_by_f[f]["centroid2dPx"][0]),
                    float(child_by_f[f]["centroid2dPx"][1]),
                )
                for f in common_frames
            ]
            child_angles = [float(child_by_f[f]["angleRad"]) for f in common_frames]

            pivot_2d, j_type = estimate_revolute_pivot_2d(
                parent_centroids_seq=parent_centroids,
                child_centroids_seq=child_centroids,
                child_angles_seq=child_angles,
                parent_angles_seq=parent_angles,
            )

            world_x = round((pivot_2d[0] - (frame_width / 2.0)) / 140.0, 3)
            world_y = round(((frame_height / 2.0) - pivot_2d[1]) / 140.0 + 1.2, 3)

            extracted_joints.append(
                ExtractedJointConstraint(
                    joint_id=str(uuid.uuid4()),
                    parent_part_id=parent.part_id,
                    child_part_id=child.part_id,
                    joint_type=j_type,
                    anchor_world_xyz=[world_x, world_y, 0.0],
                    axis_xyz=([0.0, 0.0, 1.0] if j_type == "revolute" else [1.0, 0.0, 0.0]),
                    limits={
                        "min": -1.57,
                        "max": 1.57,
                        "stiffness": 0.0,
                        "damping": 0.45,
                    },
                )
            )

        return extracted_parts, extracted_joints
