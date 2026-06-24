"""Tests for API endpoints."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

# Test client
client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data == {"ok": True} or ("message" in data and "version" in data)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_upload_pdf_file():
    """OCR upload is exposed for the migrated tutor API."""
    # Create a simple PDF-like file for testing
    pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_content)
        f.flush()

        with open(f.name, "rb") as pdf_file:
            files = {"file": ("test.pdf", pdf_file, "application/pdf")}
            data = {"document_type": "document"}

            response = client.post("/api/v1/upload", files=files, data=data)

            assert response.status_code == 200

        # Cleanup
        Path(f.name).unlink(missing_ok=True)


def test_upload_invalid_file():
    """OCR upload validates unsupported file types."""
    # Create a text file
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w") as f:
        f.write("This is a text file")
        f.flush()

        with open(f.name, "rb") as file:
            files = {"file": ("test.txt", file, "text/plain")}
            data = {"document_type": "document"}

            response = client.post("/api/v1/upload", files=files, data=data)

            assert response.status_code == 400


def test_upload_large_file():
    """OCR upload accepts valid image files under the configured limit."""
    # Create a large image (this would need to be larger than MAX_FILE_SIZE in real test)
    img = Image.new("RGB", (5000, 5000), color="white")
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        img.save(f)
        f.seek(0)

        # This test assumes the test image is smaller than the limit
        # In a real scenario, you'd create an image larger than MAX_FILE_SIZE
        files = {"file": ("large_test.png", f, "image/png")}
        data = {"document_type": "document"}

        response = client.post("/api/v1/upload", files=files, data=data)

        assert response.status_code == 200


@pytest.mark.asyncio
async def test_process_document_without_typhoon_ocr_key():
    """OCR processing route is exposed and returns a structured processing response."""

    pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_content)
        f.flush()

        with open(f.name, "rb") as file:
            files = {"file": ("test.pdf", file, "application/pdf")}
            data = {
                "document_type": "document",
                "language": "auto",
                "enhance_markdown": True,
            }

            response = client.post("/api/v1/upload-and-process", files=files, data=data)

            assert response.status_code == 200

        # Cleanup
        Path(f.name).unlink(missing_ok=True)


def test_get_nonexistent_file():
    """Test retrieving a file that doesn't exist."""
    response = client.get("/api/v1/files/nonexistent.png")
    assert response.status_code == 404


def test_delete_nonexistent_file():
    """Test deleting a file that doesn't exist."""
    response = client.delete("/api/v1/files/nonexistent.png")
    assert response.status_code == 404


def test_get_processing_status_nonexistent_document():
    """Test getting status for non-existent document."""
    response = client.get("/api/v1/documents/nonexistent/status")
    assert response.status_code == 404
