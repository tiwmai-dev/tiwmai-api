"""Supabase Storage service for student uploads."""

import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional
from urllib.parse import quote

from app.core.config import get_settings
from app.core.exceptions import StorageError
from app.core.logging import app_logger
from app.services.supabase_service import get_supabase_service


class SupabaseStorageService:
    """Thin wrapper around Supabase Storage."""

    avatar_prefix = "avatars"

    def __init__(self):
        self.settings = get_settings()
        self.supabase = get_supabase_service()
        self.default_bucket = self.settings.supabase_storage_bucket
        self.avatar_bucket = self.settings.supabase_avatar_bucket or self.default_bucket

    @property
    def client(self):
        return self.supabase.client

    def _get_bucket(self, bucket_name: str):
        return self.client.storage.from_(bucket_name)

    def _generate_path(
        self,
        prefix: str,
        user_id: str,
        filename: str,
        timestamp: Optional[datetime] = None,
    ) -> str:
        timestamp = timestamp or datetime.now(timezone.utc)
        return f"{prefix}/{timestamp:%Y/%m/%d}/{user_id}/{filename}"

    def _generate_filename(self, original_filename: str, purpose: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        stem = "".join(
            c for c in Path(original_filename).stem if c.isalnum() or c in ("_", "-")
        )
        suffix = Path(original_filename).suffix
        return f"{stem or purpose}_{timestamp}{suffix}"

    def _read_bytes(self, file_data: BinaryIO) -> bytes:
        position = None
        try:
            position = file_data.tell()
        except Exception:
            position = None
        data = file_data.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        if position is not None:
            try:
                file_data.seek(position)
            except Exception:
                pass
        return data

    async def _record_file_metadata(self, item: Dict[str, Any]) -> None:
        try:
            payload = {
                "file_id": item.get("file_id") or str(uuid.uuid4()),
                "user_id": item.get("user_id"),
                "storage_key": item.get("s3_key") or item.get("storage_key"),
                "content_type": item.get("content_type"),
                "created_at": item.get("created_at") or datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "data": item,
            }
            await self.supabase.run(
                lambda: self.client.table("files")
                .upsert(payload, on_conflict="file_id")
                .execute()
            )
        except Exception as e:
            app_logger.warning(f"Unable to record file metadata in Supabase: {e}")

    async def upload_object(
        self,
        file_data: BinaryIO,
        original_filename: str,
        user_id: str,
        purpose: str,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None,
        bucket_name: Optional[str] = None,
        key_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            target_bucket = (
                str(bucket_name or self.default_bucket).strip() or self.default_bucket
            )
            target_prefix = str(key_prefix or purpose).strip() or purpose
            storage_filename = self._generate_filename(original_filename, purpose)
            storage_key = self._generate_path(target_prefix, user_id, storage_filename)
            bucket_client = self._get_bucket(target_bucket)
            await self.supabase.run(
                bucket_client.upload,
                storage_key,
                self._read_bytes(file_data),
                {"content-type": content_type, "upsert": "true"},
            )
            result = {
                "s3_key": storage_key,
                "s3_filename": storage_filename,
                "bucket_name": target_bucket,
                "original_filename": original_filename,
                "user_id": user_id,
                "document_type": purpose,
                "content_type": content_type,
                "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                "s3_url": f"supabase://{target_bucket}/{storage_key}",
                "metadata": metadata or {},
            }
            await self._record_file_metadata(result)
            return result
        except Exception as e:
            app_logger.error(f"Supabase Storage upload failed: {e}")
            raise StorageError(f"Failed to upload to Supabase Storage: {e}")

    def _extract_url_from_response(self, result: Any) -> str:
        def _normalize(url: str) -> str:
            text = str(url or "").strip()
            return text[:-1] if text.endswith("?") else text

        if isinstance(result, str):
            return _normalize(result)
        if isinstance(result, dict):
            data = result.get("data")
            if isinstance(data, dict):
                return _normalize(data.get("publicUrl") or data.get("publicURL") or "")
            return _normalize(
                result.get("publicUrl")
                or result.get("publicURL")
                or result.get("url")
                or ""
            )
        data = getattr(result, "data", None)
        if isinstance(data, dict):
            return _normalize(data.get("publicUrl") or data.get("publicURL") or "")
        return _normalize(getattr(result, "public_url", "") or "")

    def build_public_url(self, storage_key: str, bucket_name: Optional[str] = None) -> str:
        target_bucket = str(bucket_name or self.default_bucket).strip() or self.default_bucket
        normalized_key = quote(str(storage_key).lstrip("/"), safe="/")
        base_url = str(self.settings.supabase_url or "").rstrip("/")
        return f"{base_url}/storage/v1/object/public/{target_bucket}/{normalized_key}"

    def get_public_url(self, storage_key: str, bucket_name: Optional[str] = None) -> str:
        target_bucket = str(bucket_name or self.default_bucket).strip() or self.default_bucket
        try:
            raw = self._get_bucket(target_bucket).get_public_url(storage_key)
            extracted = self._extract_url_from_response(raw)
            if extracted:
                return extracted
        except Exception as e:
            app_logger.warning(
                f"Failed to build public URL from SDK for {target_bucket}/{storage_key}: {e}"
            )
        return self.build_public_url(storage_key, target_bucket)

    def get_avatar_bucket_name(self) -> str:
        return self.avatar_bucket


@lru_cache(maxsize=1)
def get_storage_service() -> SupabaseStorageService:
    return SupabaseStorageService()
