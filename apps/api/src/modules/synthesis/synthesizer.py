from __future__ import annotations

import ast
import json
import logging

import httpx

from src.core.config import settings
from src.modules.perception.kinematics import estimate_revolute_pivot_2d
from src.modules.perception.sam_service import (
    ExtractedJointConstraint,
    ExtractedPartTrack,
)

logger = logging.getLogger("chronorig.api")

FORBIDDEN_BPY_IMPORTS = frozenset(
    {"os", "subprocess", "socket", "shutil", "urllib", "http", "requests", "sys", "pty"}
)


def validate_blender_script_ast(bpy_code: str) -> str:
    """
    Parses generated Blender Python code into an AST and blocks any dangerous
    OS/network imports or dynamic eval/exec builtins.
    """
    tree = ast.parse(bpy_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in FORBIDDEN_BPY_IMPORTS:
                    raise ValueError(f"Forbidden import '{root_pkg}' in generated bpy script")
        elif isinstance(node, ast.ImportFrom):
            root_pkg = (node.module or "").split(".")[0]
            if root_pkg in FORBIDDEN_BPY_IMPORTS:
                raise ValueError(f"Forbidden import from '{root_pkg}' in generated bpy script")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "compile", "__import__", "open"}:
                raise ValueError(f"Forbidden builtin call '{node.func.id}()' in bpy script")
    return bpy_code


def generate_deterministic_blender_script(
    parts: list[ExtractedPartTrack],
    joints: list[ExtractedJointConstraint],
    video_fps: float = 30.0,
) -> str:
    """
    Generates a deterministic, AST-validated Blender 4.x (bpy) script from the
    verified kinematic graph. Creates scaled rigid-body proxies, inserts frame-by-frame
    pose keyframes from SAM 3.1 trajectories, and binds HINGE / SLIDER / FIXED constraints.
    """
    max_f = 90
    for p in parts:
        for fr in p.trajectory_frames:
            f_idx = int(fr.get("frameIndex", 0)) + 1
            if f_idx > max_f:
                max_f = f_idx

    lines: list[str] = [
        "import bpy",
        "import math",
        "",
        "if bpy.app.background:",
        "    bpy.ops.wm.read_factory_settings(use_empty=True)",
        "else:",
        "    for obj in list(bpy.data.objects):",
        "        if obj.type in {'MESH', 'EMPTY'}:",
        "            bpy.data.objects.remove(obj, do_unlink=True)",
        "",
        "scene = bpy.context.scene",
        f"scene.render.fps = {int(round(video_fps))}",
        "scene.frame_start = 1",
        f"scene.frame_end = {max_f}",
        "scene.frame_current = 1",
        "if scene.rigidbody_world is None:",
        "    bpy.ops.rigidbody.world_add()",
        "if scene.rigidbody_world:",
        "    scene.rigidbody_world.point_cache.frame_start = 1",
        f"    scene.rigidbody_world.point_cache.frame_end = {max_f}",
        "",
        "def apply_rigid_body(target_obj, rb_type='ACTIVE', mass=1.0):",
        "    bpy.context.view_layer.objects.active = target_obj",
        "    target_obj.select_set(True)",
        "    if hasattr(bpy.context, 'temp_override'):",
        "        with bpy.context.temp_override(active_object=target_obj, object=target_obj, selected_objects=[target_obj], selected_editable_objects=[target_obj]):",
        "            bpy.ops.rigidbody.object_add(type=rb_type)",
        "    else:",
        "        bpy.ops.rigidbody.object_add(type=rb_type)",
        "    if target_obj.rigid_body:",
        "        target_obj.rigid_body.mass = mass",
        "",
        "def apply_constraint(joint_empty, constraint_type, obj1, obj2):",
        "    bpy.context.view_layer.objects.active = joint_empty",
        "    joint_empty.select_set(True)",
        "    if hasattr(bpy.context, 'temp_override'):",
        "        with bpy.context.temp_override(active_object=joint_empty, object=joint_empty, selected_objects=[joint_empty], selected_editable_objects=[joint_empty]):",
        "            bpy.ops.rigidbody.constraint_add()",
        "    else:",
        "        bpy.ops.rigidbody.constraint_add()",
        "    rbc = joint_empty.rigid_body_constraint",
        "    rbc.type = constraint_type",
        "    rbc.object1 = obj1",
        "    rbc.object2 = obj2",
        "    rbc.disable_collisions = True",
        "    return rbc",
        "",
        "part_objects = {}",
        "",
    ]

    for p in parts:
        wx, wy, wz = p.bounding_box_extents
        f0 = p.trajectory_frames[0] if p.trajectory_frames else None
        x0 = round((f0["centroid2dPx"][0] - 320.0) / 140.0, 4) if f0 else 0.0
        y0 = round((240.0 - f0["centroid2dPx"][1]) / 140.0 + 1.2, 4) if f0 else 1.2
        rb_type = "PASSIVE" if p.is_static_root else "ACTIVE"

        lines.extend(
            [
                f"# Part: {p.label} ({p.part_id})",
                f"bpy.ops.mesh.primitive_cube_add(size=1.0, location=({x0}, 0.0, {y0}))",
                "obj = bpy.context.view_layer.objects.active",
                f"obj.name = {json.dumps(p.label)}",
                f"obj.scale = ({wx}, {wz}, {wy})",
                f"apply_rigid_body(obj, rb_type='{rb_type}', mass={p.estimated_mass_kg})",
            ]
        )

        # Insert observed trajectory keyframes so timeline playback works immediately
        if len(p.trajectory_frames) > 1:
            lines.append("if obj.rigid_body:\n    obj.rigid_body.kinematic = True")

        for frame_pose in p.trajectory_frames:
            f_idx = int(frame_pose["frameIndex"]) + 1
            fx = round((frame_pose["centroid2dPx"][0] - 320.0) / 140.0, 4)
            fz = round((240.0 - frame_pose["centroid2dPx"][1]) / 140.0 + 1.2, 4)
            ang = round(-float(frame_pose["angleRad"]), 4)
            lines.extend(
                [
                    f"obj.location = ({fx}, 0.0, {fz})",
                    f"obj.rotation_euler = (0.0, {ang}, 0.0)",
                    f"obj.keyframe_insert(data_path='location', frame={f_idx})",
                    f"obj.keyframe_insert(data_path='rotation_euler', frame={f_idx})",
                ]
            )

        lines.append(f"part_objects[{json.dumps(p.part_id)}] = obj\n")

    for j in joints:
        jx, jy, jz = j.anchor_world_xyz
        b_type = (
            "HINGE"
            if j.joint_type == "revolute"
            else ("SLIDER" if j.joint_type == "prismatic" else "FIXED")
        )
        rot_setup = "joint_empty.rotation_euler = (math.radians(90), 0.0, 0.0)" if b_type == "HINGE" else "# Default orientation"
        lines.extend(
            [
                f"# Joint: {j.joint_id} ({b_type})",
                f"bpy.ops.object.empty_add(type='PLAIN_AXES', location=({jx}, {jz}, {jy}))",
                "joint_empty = bpy.context.view_layer.objects.active",
                f"joint_empty.name = 'Joint_{j.joint_type}'",
                rot_setup,
                f"apply_constraint(joint_empty, '{b_type}', part_objects.get({json.dumps(j.parent_part_id)}), part_objects.get({json.dumps(j.child_part_id)}))",
                "",
            ]
        )

    lines.extend(
        [
            "# Pre-warm simulation cache across timeline so scrubbing immediately animates",
            "for f in range(scene.frame_start, scene.frame_end + 1):",
            "    scene.frame_set(f)",
            "scene.frame_set(scene.frame_start)",
            "",
        ]
    )

    script_str = "\n".join(lines)
    return validate_blender_script_ast(script_str)


class PhysicsRigSynthesizer:
    """
    Coordinates deterministic Blender/R3F synthesis and closed-loop joint anchor
    refinement powered by Meta's open-weight `meta-models/muse-spark-open-1026`
    on Baseten or Modal GPU.
    """

    def __init__(self, parts_lookup: list[ExtractedPartTrack] | None = None) -> None:
        self._parts_by_id = {p.part_id: p for p in (parts_lookup or [])}

    def set_parts_context(self, parts: list[ExtractedPartTrack]) -> None:
        self._parts_by_id = {p.part_id: p for p in parts}

    async def propose_iou_self_debug_patch(
        self,
        joints: list[ExtractedJointConstraint],
        frame_iou_breakdown: dict[str, float],
        iteration_number: int,
    ) -> tuple[list[ExtractedJointConstraint], str]:
        """
        Refines 3D joint anchor coordinates when simulated silhouette IoU is below target.
        First computes the analytical Kasa circle-fit target anchor from the tracked parts,
        and queries `meta-models/muse-spark-open-1026` on Baseten or Modal when configured.
        """
        diff_lines: list[str] = []

        for j in joints:
            old_anchor = list(j.anchor_world_xyz)
            parent = self._parts_by_id.get(j.parent_part_id)
            child = self._parts_by_id.get(j.child_part_id)

            if parent and child and len(child.trajectory_frames) >= 3:
                parent_by_f = {f["frameIndex"]: f for f in parent.trajectory_frames}
                child_by_f = {f["frameIndex"]: f for f in child.trajectory_frames}
                common = sorted(set(parent_by_f.keys()) & set(child_by_f.keys()))
                if len(common) >= 3:
                    p_seq = [
                        (
                            float(parent_by_f[f]["centroid2dPx"][0]),
                            float(parent_by_f[f]["centroid2dPx"][1]),
                        )
                        for f in common
                    ]
                    p_ang = [float(parent_by_f[f]["angleRad"]) for f in common]
                    c_seq = [
                        (
                            float(child_by_f[f]["centroid2dPx"][0]),
                            float(child_by_f[f]["centroid2dPx"][1]),
                        )
                        for f in common
                    ]
                    c_ang = [float(child_by_f[f]["angleRad"]) for f in common]

                    solved_px, _ = estimate_revolute_pivot_2d(
                        parent_centroids_seq=p_seq,
                        child_centroids_seq=c_seq,
                        child_angles_seq=c_ang,
                        parent_angles_seq=p_ang,
                    )
                    target_wx = round((solved_px[0] - 320.0) / 140.0, 3)
                    target_wy = round((240.0 - solved_px[1]) / 140.0 + 1.2, 3)
                    j.anchor_world_xyz = [target_wx, target_wy, 0.0]

            diff_lines.append(
                f"- joint[{j.joint_id[:8]}].anchorWorldXyz: {old_anchor}\n"
                f"+ joint[{j.joint_id[:8]}].anchorWorldXyz: {j.anchor_world_xyz}"
            )

        # If a live Baseten or Modal key is configured, also record reasoning from muse-spark-open-1026
        if settings.has_live_open_weight_key():
            api_key, base_url = settings.get_open_weight_credentials()
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json={
                            "model": settings.MUSE_SPARK_OPEN_MODEL_ID,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "Summarize in one concise line how updating the revolute joint "
                                        "anchor aligns the forward-kinematic silhouette with SAM 3.1 masks."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": json.dumps(
                                        {
                                            "iteration": iteration_number,
                                            "frameIou": frame_iou_breakdown,
                                            "patch": diff_lines,
                                        }
                                    ),
                                },
                            ],
                            "max_tokens": 80,
                        },
                    )
                    if resp.status_code == 200:
                        note = resp.json()["choices"][0]["message"]["content"].strip()
                        diff_lines.append(f"// muse-spark-open-1026: {note}")
            except Exception as exc:
                logger.warning("Open-weight model note call skipped: %s", exc)

        return joints, "\n".join(diff_lines)
