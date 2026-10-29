from __future__ import annotations

import ast
import json
import logging

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.core.config import settings
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
    OWASP A03/A08 Security Guardrail: Parses generated Blender Python code into an AST
    and blocks any dangerous OS/network imports or eval/exec calls before saving or running.
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


class PhysicsRigSynthesizer:
    """
    Orchestrates Muse Image, Muse Wonder, Muse Spark 1.3, and Muse Spark Open (1026)
    to synthesize textures, PBR materials, React Three Fiber code, and Blender (bpy) rigs.
    """

    def __init__(self) -> None:
        self._meta_headers = {
            "Authorization": f"Bearer {settings.META_MODEL_API_KEY.get_secret_value()}",
            "Content-Type": "application/json",
        }
        self._baseten_headers = {
            "Authorization": f"Bearer {settings.BASETEN_API_KEY.get_secret_value()}",
            "Content-Type": "application/json",
        }

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    async def synthesize_part_texture_and_material(
        self, part_label: str, crop_png_b64: str
    ) -> dict[str, str]:
        """
        Uses Muse Image for orthographic inpainting of occluded mechanical regions
        and Muse Wonder for creative PBR material styling ($5K Best Creative Build).
        """
        async with httpx.AsyncClient(timeout=45.0) as client:
            img_resp = await client.post(
                f"{settings.META_MODEL_BASE_URL}/images/edits",
                headers=self._meta_headers,
                json={
                    "model": settings.MUSE_IMAGE_MODEL_ID,
                    "image": crop_png_b64,
                    "prompt": (
                        f"Clean orthographic product texture of '{part_label}', "
                        "inpaint any occluded regions, studio neutral lighting, seamless albedo."
                    ),
                },
            )
            img_resp.raise_for_status()
            texture_url = img_resp.json().get("data", [{}])[0].get("url", "")

            wonder_resp = await client.post(
                f"{settings.META_MODEL_BASE_URL}/responses",
                headers=self._meta_headers,
                json={
                    "model": settings.MUSE_WONDER_MODEL_ID,
                    "input": (
                        f"Generate a physically based rendering (PBR) surface shader descriptor "
                        f"for mechanical component '{part_label}'."
                    ),
                },
            )
            wonder_resp.raise_for_status()
            return {
                "textureUrl": texture_url,
                "wonderMaterialDescriptor": wonder_resp.json().get("output_text", "metallic"),
            }

    @retry(wait=wait_random_exponential(min=1, max=15), stop=stop_after_attempt(3))
    async def synthesize_physics_rig_code(
        self,
        parts: list[ExtractedPartTrack],
        joints: list[ExtractedJointConstraint],
    ) -> tuple[str, str]:
        """
        Uses Muse Spark 1.3 (reasoning_effort: 'max') to synthesize both the React Three Fiber
        (@react-three/rapier) component code and a self-contained Blender 4.x (bpy) script.
        """
        kinematic_payload = {
            "parts": [
                {
                    "id": p.part_id,
                    "label": p.label,
                    "isStaticRoot": p.is_static_root,
                    "massKg": p.estimated_mass_kg,
                    "extents": p.bounding_box_extents,
                }
                for p in parts
            ],
            "joints": [
                {
                    "id": j.joint_id,
                    "parentPartId": j.parent_part_id,
                    "childPartId": j.child_part_id,
                    "type": j.joint_type,
                    "anchorWorldXyz": j.anchor_world_xyz,
                    "axisXyz": j.axis_xyz,
                    "limits": j.limits,
                }
                for j in joints
            ],
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.META_MODEL_BASE_URL}/chat/completions",
                headers=self._meta_headers,
                json={
                    "model": settings.MUSE_SPARK_MODEL_ID,
                    "reasoning_effort": "max",
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are ChronoRig's Physics Synthesis Engine. Given a kinematic graph "
                                "extracted via SAM 3.1, output a JSON object with keys 'r3f_code' "
                                "(React Three Fiber + @react-three/rapier TSX) and 'bpy_code' "
                                "(pure Blender 4.x Python script importing ONLY 'bpy' and 'math' to "
                                "create rigid bodies and rigid_body_constraint hinges)."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(kinematic_payload),
                        },
                    ],
                },
            )
            resp.raise_for_status()
            content = json.loads(resp.json()["choices"][0]["message"]["content"])
            r3f_code = str(content.get("r3f_code", ""))
            bpy_code = validate_blender_script_ast(str(content.get("bpy_code", "import bpy\n")))
            return r3f_code, bpy_code

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    async def propose_iou_self_debug_patch(
        self,
        joints: list[ExtractedJointConstraint],
        frame_iou_breakdown: dict[str, float],
        iteration_number: int,
    ) -> tuple[list[ExtractedJointConstraint], str]:
        """
        Uses Meta's Open-Weight `meta-models/muse-spark-open-1026` on Baseten/Modal
        to analyze why the simulated 3D silhouette diverged from SAM 3.1's mask
        and compute updated 3D joint anchor coordinates ($10K Open-Weight + $5K Verifiable Agent).
        """
        prompt_payload = {
            "iteration": iteration_number,
            "frameIouBreakdown": frame_iou_breakdown,
            "currentJoints": [
                {
                    "jointId": j.joint_id,
                    "type": j.joint_type,
                    "anchorWorldXyz": j.anchor_world_xyz,
                    "limits": j.limits,
                }
                for j in joints
            ],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.BASETEN_BASE_URL}/chat/completions",
                headers=self._baseten_headers,
                json={
                    "model": settings.MUSE_SPARK_OPEN_MODEL_ID,
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a Verifiable Physics Agent running on meta-models/muse-spark-open-1026. "
                                "Adjust joint 'anchorWorldXyz' and 'limits' to maximize pixel Intersection-over-Union "
                                "(IoU) against SAM 3.1 ground truth masks. Return JSON with keys "
                                "'updatedJoints' (list of {jointId, anchorWorldXyz, limits}) and 'diffSummary' (string)."
                            ),
                        },
                        {"role": "user", "content": json.dumps(prompt_payload)},
                    ],
                },
            )
            resp.raise_for_status()
            parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
            updates_by_id = {
                u["jointId"]: u for u in parsed.get("updatedJoints", []) if "jointId" in u
            }

            for j in joints:
                if j.joint_id in updates_by_id:
                    u = updates_by_id[j.joint_id]
                    if "anchorWorldXyz" in u and len(u["anchorWorldXyz"]) == 3:
                        j.anchor_world_xyz = [round(float(x), 3) for x in u["anchorWorldXyz"]]

            diff_summary = str(
                parsed.get(
                    "diffSummary",
                    f"// Iteration #{iteration_number}: Refined joint anchors via muse-spark-open-1026",
                )
            )
            return joints, diff_summary
