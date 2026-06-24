"""Job queue endpoints."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Support both Docker (/app/packages) and local runs from apps/api.
_repo_root = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "packages").is_dir()
    ),
    None,
)
if _repo_root is None:
    _repo_root = Path("/app")
for _package_dir in ("queue", "types", "config", "shared", "cache"):
    _path = str(_repo_root / "packages" / _package_dir)
    if _path not in sys.path:
        sys.path.append(_path)

try:
    from tiwmai_config import get_queue_settings
    from tiwmai_queue import QueueClient
except Exception:  # pragma: no cover - local workspace can omit queue packages.
    get_queue_settings = None
    QueueClient = None

router = APIRouter(prefix="/jobs", tags=["jobs"])
ALLOWED_QUEUES = {"ai", "grading", "notification"}


class EnqueueJobRequest(BaseModel):
    type: str = Field(default="ping", min_length=1)
    payload: Dict[str, Any] = Field(default_factory=dict)


class JobResponse(BaseModel):
    id: str
    queue: str
    type: str
    payload: Dict[str, Any]
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


async def _get_client():
    if get_queue_settings is None or QueueClient is None:
        raise HTTPException(
            status_code=503,
            detail="Job queue packages are not installed in this workspace",
        )
    settings = get_queue_settings()
    return QueueClient(settings.redis_url)


@router.post("/{queue_name}", response_model=JobResponse)
async def enqueue_job(queue_name: str, request: EnqueueJobRequest) -> Dict[str, Any]:
    if queue_name not in ALLOWED_QUEUES:
        raise HTTPException(status_code=404, detail="queue not found")
    client = await _get_client()
    try:
        job = await client.enqueue(queue_name, request.type, request.payload)
        return job.to_dict()
    finally:
        await client.close()


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> Dict[str, Any]:
    client = await _get_client()
    try:
        job = await client.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return job.to_dict()
    finally:
        await client.close()
