"""Student file handling service."""

from io import BytesIO
from pathlib import Path

from fastapi import UploadFile

from app.core.config import get_settings
from app.core.exceptions import (
    FileTooLargeError,
    FileUploadError,
    StorageError,
    UnsupportedFileTypeError,
)
from app.services.storage_service import get_storage_service


class FileService:
    """Service for student-facing file operations."""

    AVATAR_MAX_SIZE_BYTES = 1024 * 1024

    def __init__(self):
        self.settings = get_settings()
        self.storage_service = (
            get_storage_service() if self.settings.use_supabase_storage else None
        )

    def _validate_file_type(self, filename: str) -> str:
        file_extension = Path(filename).suffix.lower().lstrip(".")
        if file_extension not in self.settings.allowed_extensions_list:
            raise UnsupportedFileTypeError(
                file_type=file_extension,
                allowed_types=self.settings.allowed_extensions_list,
            )
        return file_extension

    def _get_mime_type(self, file_extension: str) -> str:
        mime_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "tiff": "image/tiff",
            "webp": "image/webp",
        }
        return mime_types.get(file_extension, "application/octet-stream")

    async def upload_profile_avatar_to_supabase(
        self, upload_file: UploadFile, user_id: str
    ) -> dict:
        """Upload a student profile avatar to Supabase Storage."""
        if not self.storage_service:
            raise StorageError("Supabase storage not enabled")
        if not upload_file.filename:
            raise FileUploadError("Filename is required")

        content_type = str(getattr(upload_file, "content_type", "") or "").lower()
        if not content_type.startswith("image/"):
            raise UnsupportedFileTypeError(
                file_type=content_type or "unknown",
                allowed_types=["image/*"],
            )

        file_extension = self._validate_file_type(upload_file.filename)
        file_content = await upload_file.read()
        if len(file_content) > self.AVATAR_MAX_SIZE_BYTES:
            raise FileTooLargeError(max_size=self.AVATAR_MAX_SIZE_BYTES)

        bucket_name = self.storage_service.get_avatar_bucket_name()
        upload_result = await self.storage_service.upload_object(
            file_data=BytesIO(file_content),
            original_filename=upload_file.filename,
            user_id=user_id,
            purpose="avatar",
            content_type=content_type or self._get_mime_type(file_extension),
            bucket_name=bucket_name,
            key_prefix=self.storage_service.avatar_prefix,
            metadata={"purpose": "profile_avatar"},
        )
        avatar_url = self.storage_service.get_public_url(
            upload_result["s3_key"], bucket_name=bucket_name
        )
        return {
            **upload_result,
            "avatar_url": avatar_url,
        }
