from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.core.config import settings


class DurableJobRepository:
    """
    Process-safe, multi-worker durable SQLite repository (with WAL mode enabled)
    so job state, owner authorization, video paths, and Blender scripts persist
    across Uvicorn workers and server restarts without requiring external daemons.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        base_dir = settings.ARTIFACTS_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path or (base_dir / "chronorig_durable_state.sqlite3")
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=15.0)
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
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_job(
        self,
        job_id: str,
        owner_id: str,
        payload: dict[str, Any],
        blender_script: str | None = None,
        video_path: Path | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO durable_rig_jobs (
                    job_id, owner_id, status, payload_json, blender_script, video_path, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
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
                    str(payload.get("updatedAt", "")),
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
            }
