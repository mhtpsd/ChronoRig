from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine

from src.core.config import settings
from src.core.repository import DurableJobRepository

logger = logging.getLogger("chronorig.worker")


class DurableWorkerEngine:
    """
    Polls DurableJobRepository for QUEUED jobs and executes the perception,
    kinematic solver, and IoU verification pipeline with bounded retries.
    """

    def __init__(
        self,
        repo: DurableJobRepository,
        pipeline_runner: Callable[..., Coroutine[Any, Any, dict[str, Any]]],
        max_attempts: int = 3,
    ) -> None:
        self._repo = repo
        self._pipeline_runner = pipeline_runner
        self._max_attempts = max_attempts
        self._running = False

    async def process_next_available_job(self) -> bool:
        claimed = await asyncio.to_thread(
            self._repo.claim_next_queued_job, self._max_attempts
        )
        if not claimed:
            return False

        job_id = claimed["job_id"]
        owner_id = claimed["owner_id"]
        video_path: Path | None = claimed["video_path"]
        payload = claimed["payload"]
        part_prompts = payload.get("partPrompts", ["static_base", "rotating_arm"])
        use_live_sam = claimed["use_live_sam"]
        target_iou = claimed["target_iou"]
        max_iterations = claimed["max_iterations"]
        attempts = claimed["attempts"]

        if not video_path or not video_path.exists():
            payload["status"] = "FAILED"
            payload["errorCode"] = "MISSING_VIDEO_ARTIFACT"
            payload["errorMessage"] = f"Video artifact for job {job_id} does not exist."
            await asyncio.to_thread(self._repo.save_job, job_id, owner_id, payload)
            return True

        try:
            await self._pipeline_runner(
                job_id=job_id,
                owner_id=owner_id,
                mp4_path=video_path,
                part_prompts=part_prompts,
                base_url=settings.PUBLIC_API_BASE_URL.rstrip("/"),
                use_live_sam_api=use_live_sam,
                perturb_initial_anchor_for_demo=False,
                target_iou=target_iou,
                max_iterations=max_iterations,
            )
        except Exception as exc:
            logger.error(
                "Worker attempt %d/%d failed for job %s: %s",
                attempts,
                self._max_attempts,
                job_id,
                exc,
            )
            if attempts < self._max_attempts:
                payload["status"] = "QUEUED"
                await asyncio.to_thread(
                    self._repo.save_job, job_id, owner_id, payload, None, video_path
                )
        return True

    async def run_forever(self, poll_interval_sec: float = 0.25) -> None:
        self._running = True
        while self._running:
            processed = await self.process_next_available_job()
            if not processed:
                await asyncio.sleep(poll_interval_sec)

    def stop(self) -> None:
        self._running = False