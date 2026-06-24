"""File handling service."""

import uuid
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import aiofiles
from fastapi import UploadFile
from PIL import Image, ImageOps

from app.core.config import get_settings
from app.core.exceptions import (
    FileNotFoundError,
    FileTooLargeError,
    FileUploadError,
    S3StorageError,
    UnsupportedFileTypeError,
)
from app.core.logging import app_logger
from app.services.s3_service import get_s3_service


class FileService:
    """Service for handling file operations."""

    COURSE_IMAGE_MAX_DIMENSION = 1600
    COURSE_IMAGE_WEBP_QUALITY = 80
    # Supabase storage-py expects cache-control as a max-age value in seconds.
    COURSE_IMAGE_CACHE_CONTROL = "31536000"

    def __init__(self):
        self.settings = get_settings()
        self.upload_dir = Path(self.settings.upload_folder)
        self.upload_dir.mkdir(exist_ok=True)
        self.s3_service = get_s3_service() if self.settings.use_s3_storage else None
        self.storage_service = self.s3_service

    def _generate_unique_filename(self, original_filename: str) -> str:
        """Generate a unique filename while preserving extension."""
        file_extension = Path(original_filename).suffix.lower()
        unique_id = str(uuid.uuid4())
        return f"{unique_id}{file_extension}"

    def _validate_file_type(self, filename: str) -> str:
        """Validate file type and return the extension."""
        file_extension = Path(filename).suffix.lower().lstrip(".")

        if file_extension not in self.settings.allowed_extensions_list:
            raise UnsupportedFileTypeError(
                file_type=file_extension,
                allowed_types=self.settings.allowed_extensions_list,
            )

        return file_extension

    def _validate_file_size(self, file_size: int, file_extension: str = "") -> None:
        """Validate file size."""
        max_size = (
            self.settings.max_pdf_file_size
            if file_extension == "pdf"
            else self.settings.max_file_size
        )
        if file_size > max_size:
            raise FileTooLargeError(max_size=max_size)

    def _get_mime_type(self, file_extension: str) -> str:
        """Get MIME type from file extension."""
        mime_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "tiff": "image/tiff",
            "webp": "image/webp",
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "doc": "application/msword",
        }
        return mime_types.get(file_extension, "application/octet-stream")

    async def validate_image_file(self, file_path: Path) -> Tuple[int, int]:
        """
        Validate image file and return dimensions.

        Args:
            file_path: Path to the image file

        Returns:
            Tuple of (width, height)

        Raises:
            FileUploadError: If file is corrupted or not a valid image
        """
        try:
            with Image.open(file_path) as img:
                return img.size
        except Exception as e:
            app_logger.error(f"Failed to validate image {file_path}: {e}")
            raise FileUploadError(f"Invalid image file: {e}")

    async def save_upload_file_with_s3(
        self,
        upload_file: UploadFile,
        user_id: str,
        document_type: str = "document",
        document_id: Optional[str] = None,
    ) -> Tuple[str, Optional[Path], dict]:
        """
        Save uploaded file to S3 and optionally local storage.

        Args:
            upload_file: FastAPI UploadFile object
            user_id: User identifier for S3 path structure
            document_type: Type of document
            document_id: Optional document ID

        Returns:
            Tuple of (document_id, file_path, metadata)

        Raises:
            FileUploadError: If file upload fails
            S3StorageError: If S3 upload fails
        """
        if not upload_file.filename:
            raise FileUploadError("Filename is required")

        # Validate file type
        file_extension = self._validate_file_type(upload_file.filename)

        # Get file content
        content = await upload_file.read()
        file_size = len(content)

        # Validate file size
        self._validate_file_size(file_size, file_extension)

        # Generate document ID if not provided
        if not document_id:
            document_id = str(uuid.uuid4())

        metadata = {
            "original_filename": upload_file.filename,
            "file_extension": file_extension,
            "file_size": file_size,
            "mime_type": self._get_mime_type(file_extension),
            "document_type": document_type,
            "user_id": user_id,
        }

        # Handle S3 storage if enabled
        s3_result = None
        if self.s3_service:
            try:
                # Reset file pointer for S3 upload
                from io import BytesIO

                file_data = BytesIO(content)

                s3_result = await self.s3_service.upload_raw_document(
                    file_data=file_data,
                    original_filename=upload_file.filename,
                    user_id=user_id,
                    document_type=document_type,
                    content_type=self._get_mime_type(file_extension),
                    metadata={"document_id": document_id, "file_size": str(file_size)},
                )

                metadata.update(
                    {
                        "s3_key": s3_result["s3_key"],
                        "s3_url": s3_result["s3_url"],
                        "s3_bucket": s3_result["bucket_name"],
                        "s3_filename": s3_result["s3_filename"],
                    }
                )

                app_logger.info(f"File uploaded to S3: {s3_result['s3_key']}")

            except Exception as e:
                app_logger.error(f"S3 upload failed for {upload_file.filename}: {e}")
                if not self.settings.debug:  # In production, fail if S3 upload fails
                    raise S3StorageError(f"Failed to upload to S3: {e}")

        # Always save locally for processing (can be cleaned up later)
        local_path = None
        try:
            unique_filename = self._generate_unique_filename(upload_file.filename)
            local_path = self.upload_dir / unique_filename

            # Save file to local disk
            async with aiofiles.open(local_path, "wb") as f:
                await f.write(content)

            # Validate image files (only for image formats)
            if file_extension in ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"]:
                image_dimensions = await self.validate_image_file(local_path)
                metadata["image_dimensions"] = image_dimensions

            metadata["local_filename"] = unique_filename
            metadata["local_path"] = str(local_path)

            app_logger.info(f"File saved locally: {local_path}")

        except Exception as e:
            # Clean up local file if save failed
            if local_path and local_path.exists():
                local_path.unlink()

            app_logger.error(f"Failed to save file locally {upload_file.filename}: {e}")
            # If S3 upload succeeded but local save failed, that's still acceptable
            if not s3_result:
                raise FileUploadError(f"Failed to save file: {e}")

        return document_id, local_path, metadata

    async def save_upload_file(
        self, upload_file: UploadFile, document_id: Optional[str] = None
    ) -> Tuple[str, Path, dict]:
        """
        Save uploaded file to disk.

        Args:
            upload_file: FastAPI UploadFile object
            document_id: Optional document ID to use as filename prefix

        Returns:
            Tuple of (document_id, file_path, metadata)

        Raises:
            FileUploadError: If file upload fails
            FileTooLargeError: If file exceeds size limit
            UnsupportedFileTypeError: If file type is not allowed
        """
        if not upload_file.filename:
            raise FileUploadError("Filename is required")

        # Validate file type
        file_extension = self._validate_file_type(upload_file.filename)

        # Get file size
        content = await upload_file.read()
        file_size = len(content)

        # Validate file size
        self._validate_file_size(file_size, file_extension)

        # Generate unique filename
        if not document_id:
            document_id = str(uuid.uuid4())

        unique_filename = self._generate_unique_filename(upload_file.filename)
        file_path = self.upload_dir / unique_filename

        try:
            # Save file to disk
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)

            # Validate image files (only for image formats)
            image_dimensions = None
            if file_extension in ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"]:
                image_dimensions = await self.validate_image_file(file_path)

            metadata = {
                "original_filename": upload_file.filename,
                "file_extension": file_extension,
                "file_size": file_size,
                "mime_type": self._get_mime_type(file_extension),
                "saved_filename": unique_filename,
                "image_dimensions": image_dimensions,
            }

            app_logger.info(f"File saved successfully: {file_path}")
            return document_id, file_path, metadata

        except Exception as e:
            # Clean up file if save failed
            if file_path.exists():
                file_path.unlink()

            app_logger.error(f"Failed to save file {upload_file.filename}: {e}")
            raise FileUploadError(f"Failed to save file: {e}")

    async def delete_file(self, file_path: Path) -> bool:
        """
        Delete file from disk.

        Args:
            file_path: Path to file to delete

        Returns:
            True if file was deleted, False if file didn't exist
        """
        try:
            if file_path.exists():
                file_path.unlink()
                app_logger.info(f"File deleted: {file_path}")
                return True
            else:
                app_logger.warning(f"File not found for deletion: {file_path}")
                return False
        except Exception as e:
            app_logger.error(f"Failed to delete file {file_path}: {e}")
            raise FileUploadError(f"Failed to delete file: {e}")

    async def get_file_info(self, file_path: Path) -> dict:
        """
        Get file information.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file information

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stat = file_path.stat()
        file_extension = file_path.suffix.lower().lstrip(".")

        return {
            "file_path": str(file_path),
            "file_size": stat.st_size,
            "file_extension": file_extension,
            "mime_type": self._get_mime_type(file_extension),
            "created_at": stat.st_ctime,
            "modified_at": stat.st_mtime,
        }

    def get_file_url(self, filename: str) -> str:
        """
        Generate URL for accessing uploaded file.

        Args:
            filename: Filename to generate URL for

        Returns:
            URL string
        """
        return f"/api/v1/files/{filename}"

    async def save_extracted_content_to_s3(
        self,
        content: str,
        original_s3_key: str,
        user_id: str,
        document_id: str,
        confidence_score: float,
        processing_time_ms: int,
        metadata: Optional[dict] = None,
        content_format: str = "json",
    ) -> Optional[dict]:
        """
        Save extracted content to S3 storage.

        Args:
            content: Extracted content (JSON or markdown)
            original_s3_key: S3 key of original document
            user_id: User identifier
            document_id: Document processing ID
            confidence_score: OCR confidence score
            processing_time_ms: Processing time
            metadata: Additional metadata
            content_format: Format of content ("json" or "markdown")

        Returns:
            S3 save result or None if S3 not enabled
        """
        if not self.s3_service:
            app_logger.info("S3 storage not enabled, skipping content save to S3")
            return None

        try:
            result = await self.s3_service.save_extracted_text(
                content=content,
                original_s3_key=original_s3_key,
                user_id=user_id,
                document_id=document_id,
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                metadata=metadata,
                content_format=content_format,
            )

            app_logger.info(
                f"Extracted content saved to S3: {result['s3_key']} (format: {content_format})"
            )
            return result

        except Exception as e:
            app_logger.error(f"Failed to save extracted content to S3: {e}")
            if not self.settings.debug:  # In production, this might be critical
                raise S3StorageError(f"Failed to save extracted content to S3: {e}")
            return None

    async def append_extracted_text_to_user_s3_file(
        self,
        content: str,
        original_s3_key: str,
        user_id: str,
        document_id: str,
        confidence_score: float,
        processing_time_ms: int,
        page_num: int = 1,
        original_filename: str = None,
        metadata: Optional[dict] = None,
        content_format: str = "json",
    ) -> Optional[dict]:
        """
        Append extracted content to user's consolidated S3 file.

        Args:
            content: Extracted content (JSON or markdown)
            original_s3_key: S3 key of original document
            user_id: User identifier
            document_id: Document processing ID
            confidence_score: OCR confidence score
            processing_time_ms: Processing time
            page_num: Page number being processed
            original_filename: Original filename for display
            metadata: Additional metadata
            content_format: Format of content ("json" or "markdown")

        Returns:
            S3 save result or None if S3 not enabled
        """
        if not self.s3_service:
            app_logger.info(
                "S3 storage not enabled, skipping consolidated content save to S3"
            )
            return None

        try:
            result = await self.s3_service.append_extracted_text_to_user_file(
                content=content,
                original_s3_key=original_s3_key,
                user_id=user_id,
                document_id=document_id,
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                page_num=page_num,
                original_filename=original_filename,
                metadata=metadata,
                content_format=content_format,
            )

            app_logger.info(
                f"Extracted content appended to consolidated S3 file: {result['s3_key']} (format: {content_format})"
            )
            return result

        except Exception as e:
            app_logger.error(
                f"Failed to append extracted content to consolidated S3 file: {e}"
            )
            if not self.settings.debug:  # In production, this might be critical
                raise S3StorageError(f"Failed to append extracted content to S3: {e}")
            return None

    async def cleanup_local_file(self, file_path: Optional[Path]) -> None:
        """
        Clean up local temporary file after processing.

        Args:
            file_path: Path to local file to clean up
        """
        if not file_path or not file_path.exists():
            return

        try:
            file_path.unlink()
            app_logger.info(f"Cleaned up local file: {file_path}")
        except Exception as e:
            app_logger.warning(f"Failed to clean up local file {file_path}: {e}")

    async def download_from_s3(self, s3_key: str) -> bytes:
        """
        Download file from S3.

        Args:
            s3_key: S3 object key

        Returns:
            File content as bytes

        Raises:
            S3StorageError: If S3 not enabled or download fails
        """
        if not self.s3_service:
            raise S3StorageError("S3 storage not enabled")

        return await self.s3_service.download_document(s3_key)

    async def upload_file_to_s3(
        self, upload_file: UploadFile, user_id: str, document_type: str = "document"
    ) -> dict:
        """
        Upload a file directly to S3 without local storage.

        Args:
            upload_file: File to upload (can be a local file wrapper)
            user_id: User identifier for S3 organization
            document_type: Type of document (document/book/exam)

        Returns:
            Dictionary with S3 upload metadata

        Raises:
            S3StorageError: If S3 not enabled or upload fails
        """
        if not self.s3_service:
            raise S3StorageError("S3 storage not enabled")

        try:
            # Validate file type
            file_extension = self._validate_file_type(upload_file.filename)

            # Read file content
            file_content = await upload_file.read()

            # Validate file size
            self._validate_file_size(len(file_content), file_extension)

            # Upload to S3
            from io import BytesIO

            file_data = BytesIO(file_content)

            s3_result = await self.s3_service.upload_raw_document(
                file_data=file_data,
                original_filename=upload_file.filename,
                user_id=user_id,
                document_type=document_type,
                content_type=getattr(
                    upload_file, "content_type", self._get_mime_type(file_extension)
                ),
            )

            return s3_result

        except Exception as e:
            app_logger.error(f"Failed to upload file to S3: {e}")
            raise S3StorageError(f"S3 upload failed: {e}")

    async def upload_course_image_to_s3(
        self, upload_file: UploadFile, user_id: str
    ) -> dict:
        """
        Upload a course cover image to Supabase Storage and return public URL metadata.

        Args:
            upload_file: Image file to upload
            user_id: User identifier for folder organization

        Returns:
            Dictionary with storage key, bucket name, and image_url
        """
        if not self.s3_service:
            raise S3StorageError("S3 storage not enabled")
        if not upload_file.filename:
            raise FileUploadError("Filename is required")

        content_type = str(getattr(upload_file, "content_type", "") or "").lower()
        if not content_type.startswith("image/"):
            raise UnsupportedFileTypeError(
                file_type=content_type or "unknown",
                allowed_types=["image/*"],
            )

        # Validate file type/size using existing upload constraints.
        file_extension = self._validate_file_type(upload_file.filename)
        file_content = await upload_file.read()
        self._validate_file_size(len(file_content), file_extension)

        try:
            with Image.open(BytesIO(file_content)) as source_image:
                image = ImageOps.exif_transpose(source_image)
                image.thumbnail(
                    (
                        self.COURSE_IMAGE_MAX_DIMENSION,
                        self.COURSE_IMAGE_MAX_DIMENSION,
                    ),
                    Image.Resampling.LANCZOS,
                )
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert(
                        "RGBA" if "transparency" in image.info else "RGB"
                    )

                optimized_content = BytesIO()
                image.save(
                    optimized_content,
                    format="WEBP",
                    quality=self.COURSE_IMAGE_WEBP_QUALITY,
                    method=6,
                )
        except Exception as e:
            raise FileUploadError(f"Invalid course image: {e}") from e

        optimized_bytes = optimized_content.getvalue()
        source_stem = Path(upload_file.filename).stem or "course-image"
        optimized_filename = f"{source_stem}-{uuid.uuid4().hex[:12]}.webp"
        file_data = BytesIO(optimized_bytes)
        bucket_name = self.s3_service.get_course_images_bucket_name()
        s3_result = await self.s3_service.upload_raw_document(
            file_data=file_data,
            original_filename=optimized_filename,
            user_id=user_id,
            document_type="course_image",
            content_type="image/webp",
            bucket_name=bucket_name,
            key_prefix=self.s3_service.course_images_prefix,
            metadata={
                "purpose": "course_cover",
                "source_filename": upload_file.filename,
                "source_content_type": content_type,
            },
            cache_control=self.COURSE_IMAGE_CACHE_CONTROL,
        )
        image_url = self.s3_service.get_public_url(
            s3_result["s3_key"], bucket_name=bucket_name
        )
        return {
            **s3_result,
            "image_url": image_url,
        }

    async def upload_profile_avatar_to_supabase(
        self, upload_file: UploadFile, user_id: str
    ) -> dict:
        """Upload a student profile avatar to Supabase Storage."""
        avatar_service = self.storage_service or self.s3_service
        if not avatar_service:
            raise S3StorageError("Supabase storage not enabled")
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
        if len(file_content) > 1024 * 1024:
            raise FileTooLargeError(max_size=1024 * 1024)

        from io import BytesIO

        bucket_name = avatar_service.get_avatar_bucket_name()
        upload_kwargs = {
            "file_data": BytesIO(file_content),
            "original_filename": upload_file.filename,
            "user_id": user_id,
            "content_type": content_type or self._get_mime_type(file_extension),
            "bucket_name": bucket_name,
            "key_prefix": avatar_service.avatar_prefix,
            "metadata": {"purpose": "profile_avatar"},
        }
        if hasattr(avatar_service, "upload_object"):
            s3_result = await avatar_service.upload_object(
                **upload_kwargs,
                purpose="avatar",
            )
        else:
            s3_result = await avatar_service.upload_raw_document(
                **upload_kwargs,
                document_type="avatar",
            )
        avatar_url = avatar_service.get_public_url(
            s3_result["s3_key"], bucket_name=bucket_name
        )
        return {
            **s3_result,
            "avatar_url": avatar_url,
        }

    async def list_user_documents_from_s3(
        self, user_id: str, limit: int = 100
    ) -> List[dict]:
        """
        List user documents from S3.

        Args:
            user_id: User identifier
            limit: Maximum number of documents

        Returns:
            List of document information
        """
        if not self.s3_service:
            app_logger.info("S3 storage not enabled, returning empty list")
            return []

        return await self.s3_service.list_user_documents(user_id, limit)
