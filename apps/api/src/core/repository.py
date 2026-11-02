from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.config import settings


class DurableJobRepository:
    """
    Multi-worker durable SQLite WAL repository providing:
    1. Persistent job, artifact, and Blender script storage across Uvicorn workers.
    2. Atomic durable queue claiming with retry attempts and crash recovery.
    3. Cross-worker WebSocket event outbox (`durable_job_events`) so any Uvicorn
       worker replica can stream real-time telemetry to subscribed clients.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        base_dir = settings.ARTIFACTS_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path or (base_dir / "chronorig_durable_state.sqlite3")
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=20.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS durable_rig_jobs (
                    job_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    blender_script TEXT,
                    video_path TEXT,
                    use_live_sam INTEGER NOT NULL DEFAULT 0,
                    target_iou REAL NOT NULL DEFAULT 0.85,
                    max_iterations INTEGER NOT NULL DEFAULT 3,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS durable_job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_job_events_job_id ON durable_job_events (job_id, id);"
            )
            conn.commit()

    def save_job(
        self,
        job_id: str,
        owner_id: str,
        payload: dict[str, Any],
        blender_script: str | None = None,
        video_path: Path | None = None,
        use_live_sam: bool = False,
        target_iou: float = 0.85,
        max_iterations: int = 3,
    ) -> None:
        now_iso = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO durable_rig_jobs (
                    job_id, owner_id, status, payload_json, blender_script,
                    video_path, use_live_sam, target_iou, max_iterations, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    blender_script = COALESCE(excluded.blender_script, durable_rig_jobs.blender_script),
                    video_path = COALESCE(excluded.video_path, durable_rig_jobs.video_path),
                    updated_at = excluded.updated_at
                """,
                (
                    job_id,
                    owner_id,
                    str(payload.get("status", "QUEUED")),
                    json.dumps(payload),
                    blender_script,
                    str(video_path) if video_path else None,
                    1 if use_live_sam else 0,
                    target_iou,
                    max_iterations,
                    str(payload.get("updatedAt", now_iso)),
                ),
            )
            conn.commit()

    def get_job_row(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM durable_rig_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if not row:
                return None
            return {
                "job_id": row["job_id"],
                "owner_id": row["owner_id"],
                "status": row["status"],
                "payload": json.loads(row["payload_json"]),
                "blender_script": row["blender_script"],
                "video_path": Path(row["video_path"]) if row["video_path"] else None,
                "use_live_sam": bool(row["use_live_sam"]),
                "target_iou": float(row["target_iou"]),
                "max_iterations": int(row["max_iterations"]),
                "attempts": int(row["attempts"]),
            }

    def claim_next_queued_job(self, max_attempts: int = 3) -> dict[str, Any] | None:
        """Atomically claims the oldest QUEUED job for durable worker execution."""
        now_iso = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT job_id FROM durable_rig_jobs
                WHERE status = 'QUEUED' AND attempts < ?
                ORDER BY updated_at ASC LIMIT 1
                """,
                (max_attempts,),
            ).fetchone()
            if not row:
                return None
            job_id = str(row["job_id"])
            cursor = conn.execute(
                """
                UPDATE durable_rig_jobs
                SET status = 'SEGMENTING_SAM', attempts = attempts + 1, updated_at = ?
                WHERE job_id = ? AND status = 'QUEUED'
                """,
                (now_iso, job_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                return None
        return self.get_job_row(job_id)

    def append_event(self, job_id: str, event_payload: dict[str, Any]) -> int:
        """Persists a telemetry event to the cross-worker SQLite WAL outbox."""
        now_iso = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO durable_job_events (job_id, event_json, created_at)
                VALUES (?, ?, ?)
                """,
                (job_id, json.dumps(event_payload), now_iso),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    def fetch_events_since(self, job_id: str, last_event_id: int = 0) -> list[tuple[int, str]]:
        """Reads all telemetry events for `job_id` newer than `last_event_id`."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, event_json FROM durable_job_events
                WHERE job_id = ? AND id > ?
                ORDER BY id ASC
                """,
                (job_id, last_event_id),
            ).fetchall()
            return [(int(r["id"]), str(r["event_json"])) for r in rows]
