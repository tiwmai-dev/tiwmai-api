"""Tests for student-facing services."""

from io import BytesIO

import pytest
from fastapi import UploadFile
from PIL import Image

from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.services.file_service import FileService


class FakeAvatarStorage:
    def __init__(self):
        self.upload = None
        self.avatar_prefix = "avatars"

    def get_avatar_bucket_name(self):
        return "avatars"

    async def upload_object(self, **kwargs):
        self.upload = kwargs
        return {
            "s3_key": "avatars/test/avatar.png",
            "bucket_name": "avatars",
            "original_filename": kwargs["original_filename"],
        }

    def get_public_url(self, s3_key, bucket_name=None):
        return f"https://storage.example/{bucket_name}/{s3_key}"


def _image_upload(filename="avatar.png", size=(64, 64), content_type="image/png"):
    image = Image.new("RGB", size, color="blue")
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    upload = UploadFile(filename=filename, file=image_bytes)
    upload.headers = {"content-type": content_type}
    return upload


@pytest.fixture
def file_service():
    service = FileService()
    service.storage_service = FakeAvatarStorage()
    return service


@pytest.mark.asyncio
async def test_upload_profile_avatar_to_supabase(file_service):
    result = await file_service.upload_profile_avatar_to_supabase(
        _image_upload(), "student-1"
    )

    storage = file_service.storage_service
    assert result["avatar_url"] == "https://storage.example/avatars/avatars/test/avatar.png"
    assert storage.upload["purpose"] == "avatar"
    assert storage.upload["content_type"] == "image/png"
    assert storage.upload["key_prefix"] == "avatars"
    assert storage.upload["metadata"] == {"purpose": "profile_avatar"}


@pytest.mark.asyncio
async def test_upload_profile_avatar_rejects_non_image(file_service):
    upload = UploadFile(filename="avatar.txt", file=BytesIO(b"text"))
    upload.headers = {"content-type": "text/plain"}

    with pytest.raises(UnsupportedFileTypeError):
        await file_service.upload_profile_avatar_to_supabase(upload, "student-1")


@pytest.mark.asyncio
async def test_upload_profile_avatar_rejects_large_file(file_service):
    upload = UploadFile(filename="avatar.png", file=BytesIO(b"x" * (1024 * 1024 + 1)))
    upload.headers = {"content-type": "image/png"}

    with pytest.raises(FileTooLargeError):
        await file_service.upload_profile_avatar_to_supabase(upload, "student-1")


def test_validate_file_type_valid():
    service = FileService()

    assert service._validate_file_type("test.jpg") == "jpg"
    assert service._validate_file_type("test.PNG") == "png"
    assert service._validate_file_type("test.webp") == "webp"


def test_validate_file_type_invalid():
    service = FileService()

    with pytest.raises(UnsupportedFileTypeError):
        service._validate_file_type("test.txt")
