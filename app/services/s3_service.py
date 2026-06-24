"""Supabase Storage service with the legacy S3Service API."""

import json
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional
from urllib.parse import quote

from app.core.config import get_settings
from app.core.exceptions import S3StorageError
from app.core.logging import app_logger
from app.services.supabase_service import get_supabase_service


class S3Service:
    """Compatibility wrapper that stores documents in Supabase Storage."""

    def __init__(self):
        self.settings = get_settings()
        self.supabase = get_supabase_service()
        self.bucket_name = self.settings.supabase_storage_bucket
        self.course_images_bucket = (
            self.settings.supabase_course_images_bucket or self.bucket_name
        )
        self.avatar_bucket = (
            self.settings.supabase_avatar_bucket
            or self.settings.supabase_course_images_bucket
            or self.bucket_name
        )
        self.raw_documents_prefix = "raw-documents"
        self.course_images_prefix = "course-images"
        self.avatar_prefix = "avatars"
        self.extracted_text_prefix = "extracted-text"

    @property
    def client(self):
        return self.supabase.client

    @property
    def bucket(self):
        return self._get_bucket(self.bucket_name)

    def _get_bucket(self, bucket_name: str):
        return self.client.storage.from_(bucket_name)

    def _generate_hierarchical_path(
        self,
        prefix: str,
        user_id: str,
        filename: str,
        timestamp: Optional[datetime] = None,
    ) -> str:
        timestamp = timestamp or datetime.now(timezone.utc)
        return f"{prefix}/{timestamp:%Y/%m/%d}/{user_id}/{filename}"

    def _generate_filename(
        self, original_filename: str, document_type: str = "document"
    ) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        stem = "".join(
            c for c in Path(original_filename).stem if c.isalnum() or c in ("_", "-")
        )
        suffix = Path(original_filename).suffix
        return f"{stem or document_type}_{timestamp}{suffix}"

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

    async def upload_raw_document(
        self,
        file_data: BinaryIO,
        original_filename: str,
        user_id: str,
        document_type: str = "document",
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
        bucket_name: Optional[str] = None,
        key_prefix: Optional[str] = None,
        cache_control: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            target_bucket = (
                str(bucket_name or self.bucket_name).strip() or self.bucket_name
            )
            target_prefix = (
                str(key_prefix or self.raw_documents_prefix).strip()
                or self.raw_documents_prefix
            )
            storage_filename = self._generate_filename(original_filename, document_type)
            storage_key = self._generate_hierarchical_path(
                target_prefix,
                user_id,
                storage_filename,
            )
            body = self._read_bytes(file_data)
            bucket_client = self._get_bucket(target_bucket)
            upload_options = {"content-type": content_type, "upsert": "true"}
            if cache_control:
                upload_options["cache-control"] = cache_control
            await self.supabase.run(
                bucket_client.upload,
                storage_key,
                body,
                upload_options,
            )
            result = {
                "s3_key": storage_key,
                "s3_filename": storage_filename,
                "bucket_name": target_bucket,
                "original_filename": original_filename,
                "user_id": user_id,
                "document_type": document_type,
                "content_type": content_type,
                "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                "s3_url": f"supabase://{target_bucket}/{storage_key}",
                "metadata": metadata or {},
            }
            await self._record_file_metadata(result)
            return result
        except Exception as e:
            app_logger.error(f"Supabase Storage upload failed: {e}")
            raise S3StorageError(f"Failed to upload to Supabase Storage: {e}")

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

    def build_public_url(self, s3_key: str, bucket_name: Optional[str] = None) -> str:
        target_bucket = str(bucket_name or self.bucket_name).strip() or self.bucket_name
        normalized_key = quote(str(s3_key).lstrip("/"), safe="/")
        base_url = str(self.settings.supabase_url or "").rstrip("/")
        return f"{base_url}/storage/v1/object/public/{target_bucket}/{normalized_key}"

    def get_public_url(self, s3_key: str, bucket_name: Optional[str] = None) -> str:
        target_bucket = str(bucket_name or self.bucket_name).strip() or self.bucket_name
        try:
            raw = self._get_bucket(target_bucket).get_public_url(s3_key)
            extracted = self._extract_url_from_response(raw)
            if extracted:
                return extracted
        except Exception as e:
            app_logger.warning(
                f"Failed to build public URL from SDK for {target_bucket}/{s3_key}: {e}"
            )
        return self.build_public_url(s3_key, target_bucket)

    def get_course_images_bucket_name(self) -> str:
        return self.course_images_bucket

    def get_avatar_bucket_name(self) -> str:
        return self.avatar_bucket

    async def save_extracted_text(
        self,
        content: str,
        original_s3_key: str,
        user_id: str,
        document_id: str,
        confidence_score: float,
        processing_time_ms: int,
        metadata: Optional[Dict[str, Any]] = None,
        content_format: str = "json",
    ) -> Dict[str, Any]:
        try:
            original_filename = Path(original_s3_key).name
            suffix = ".json" if content_format == "json" else ".md"
            output_filename = f"{Path(original_filename).stem}{suffix}"
            content_type = (
                "application/json; charset=utf-8"
                if content_format == "json"
                else "text/markdown; charset=utf-8"
            )
            storage_key = self._generate_hierarchical_path(
                self.extracted_text_prefix,
                user_id,
                output_filename,
            )
            await self.supabase.run(
                self.bucket.upload,
                storage_key,
                content.encode("utf-8"),
                {"content-type": content_type, "upsert": "true"},
            )
            result = {
                "s3_key": storage_key,
                "bucket_name": self.bucket_name,
                "original_s3_key": original_s3_key,
                "user_id": user_id,
                "document_id": document_id,
                "content_type": content_type,
                "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                "s3_url": f"supabase://{self.bucket_name}/{storage_key}",
                "confidence_score": confidence_score,
                "processing_time_ms": processing_time_ms,
                "content_format": content_format,
                "metadata": metadata or {},
            }
            await self._record_file_metadata(result)
            return result
        except Exception as e:
            app_logger.error(f"Supabase Storage content save failed: {e}")
            raise S3StorageError(f"Failed to save content to Supabase Storage: {e}")

    async def append_extracted_text_to_user_file(
        self,
        content: str,
        original_s3_key: str,
        user_id: str,
        document_id: str,
        confidence_score: float,
        processing_time_ms: int,
        page_num: int = 1,
        original_filename: str = None,
        metadata: Optional[Dict[str, Any]] = None,
        content_format: str = "json",
    ) -> Dict[str, Any]:
        today = datetime.now(timezone.utc)
        date_str = today.strftime("%Y%m%d")
        filename = f"{user_id}_ocr_results_{date_str}.{'json' if content_format == 'json' else 'md'}"
        storage_key = self._generate_hierarchical_path(
            self.extracted_text_prefix,
            user_id,
            filename,
            timestamp=today,
        )
        existing = None
        try:
            existing = await self.download_document(storage_key)
        except Exception:
            existing = None

        if content_format == "json":
            rows = []
            if existing:
                try:
                    rows = json.loads(existing.decode("utf-8"))
                    if not isinstance(rows, list):
                        rows = [rows]
                except Exception:
                    rows = []
            try:
                new_data = json.loads(content)
            except Exception:
                new_data = {"raw_content": content}
            new_data["extraction_metadata"] = {
                "document_id": document_id,
                "original_filename": original_filename,
                "confidence_score": confidence_score,
                "processing_time_ms": processing_time_ms,
                "extraction_timestamp": today.isoformat(),
                "page_num": page_num,
                "metadata": metadata or {},
            }
            rows.insert(0, new_data)
            updated_content = json.dumps(rows, ensure_ascii=False, indent=2)
        else:
            previous = existing.decode("utf-8") if existing else ""
            updated_content = f"{content}\n\n{previous}" if previous else content

        return await self.save_extracted_text(
            updated_content,
            original_s3_key,
            user_id,
            document_id,
            confidence_score,
            processing_time_ms,
            metadata,
            content_format,
        )

    async def download_document(self, s3_key: str) -> bytes:
        try:
            result = await self.supabase.run(self.bucket.download, s3_key)
            if isinstance(result, bytes):
                return result
            if isinstance(result, str):
                return result.encode("utf-8")
            return bytes(result)
        except Exception as e:
            app_logger.error(f"Supabase Storage download failed for {s3_key}: {e}")
            raise S3StorageError(f"Failed to download document: {e}")

    async def delete_document(self, s3_key: str) -> bool:
        try:
            await self.supabase.run(self.bucket.remove, [s3_key])
            return True
        except Exception as e:
            app_logger.error(f"Supabase Storage delete failed for {s3_key}: {e}")
            return False

    async def list_user_documents(
        self, user_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        try:
            rows: List[Dict[str, Any]] = []
            for prefix in (self.raw_documents_prefix, self.extracted_text_prefix):
                folder = f"{prefix}"
                try:
                    objects = await self.supabase.run(
                        self.bucket.list, folder, {"limit": limit}
                    )
                except Exception:
                    objects = []
                for obj in objects or []:
                    name = obj.get("name") if isinstance(obj, dict) else str(obj)
                    if not name or user_id not in name:
                        continue
                    rows.append(
                        {
                            "s3_key": f"{folder}/{name}",
                            "filename": Path(name).name,
                            "size": obj.get("metadata", {}).get("size")
                            if isinstance(obj, dict)
                            else None,
                            "last_modified": obj.get("updated_at")
                            if isinstance(obj, dict)
                            else None,
                            "user_id": user_id,
                            "s3_url": f"supabase://{self.bucket_name}/{folder}/{name}",
                        }
                    )
            if not rows:
                result = await self.supabase.run(
                    lambda: self.client.table("files")
                    .select("*")
                    .eq("user_id", user_id)
                    .limit(limit)
                    .execute()
                )
                for row in getattr(result, "data", None) or []:
                    data = row.get("data") or {}
                    rows.append(
                        {**data, "s3_key": row.get("storage_key") or data.get("s3_key")}
                    )
            return rows[:limit]
        except Exception as e:
            app_logger.error(f"Supabase Storage list failed for user {user_id}: {e}")
            raise S3StorageError(f"Failed to list documents: {e}")

    async def check_bucket_access(self) -> bool:
        try:
            await self.supabase.run(self.bucket.list, "", {"limit": 1})
            return True
        except Exception as e:
            app_logger.error(f"Supabase Storage bucket access failed: {e}")
            return False

    async def generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        try:
            result = await self.supabase.run(
                self.bucket.create_signed_url, s3_key, expiration
            )
            if isinstance(result, dict):
                return (
                    result.get("signedURL")
                    or result.get("signed_url")
                    or result.get("url")
                    or ""
                )
            return str(result)
        except Exception as e:
            app_logger.error(f"Supabase signed URL failed for {s3_key}: {e}")
            raise S3StorageError(f"Failed to generate signed URL: {e}")


@lru_cache()
def get_s3_service() -> S3Service:
    return S3Service()
