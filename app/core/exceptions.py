"""Custom exception classes."""

from typing import Any, Dict, Optional


class BaseAPIException(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(BaseAPIException):
    """Validation error exception."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)


class FileUploadError(BaseAPIException):
    """File upload error exception."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)


class FileNotFoundError(BaseAPIException):
    """File not found error exception."""

    def __init__(self, message: str = "File not found"):
        super().__init__(message, status_code=404)


class FileTooLargeError(FileUploadError):
    """File too large error exception."""

    def __init__(self, max_size: int):
        message = f"File size exceeds maximum allowed size of {max_size} bytes"
        details = {"max_size": max_size}
        super().__init__(message, details=details)


class UnsupportedFileTypeError(FileUploadError):
    """Unsupported file type error exception."""

    def __init__(self, file_type: str, allowed_types: list):
        message = f"Unsupported file type: {file_type}"
        details = {"file_type": file_type, "allowed_types": allowed_types}
        super().__init__(message, details=details)


class LLMProcessingError(BaseAPIException):
    """LLM processing error exception."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class OCRProcessingError(LLMProcessingError):
    """OCR/LLM processing error exception for legacy tutor endpoints."""


class TyphoonOCRError(OCRProcessingError):
    """TyphoonOCR API error exception."""

    def __init__(self, message: str, api_response: Optional[Dict[str, Any]] = None):
        details = {"api_response": api_response} if api_response else None
        super().__init__(f"TyphoonOCR API error: {message}", details=details)


class RateLimitError(BaseAPIException):
    """Rate limit exceeded error exception."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)


class AuthenticationError(BaseAPIException):
    """Authentication error exception."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, status_code=401)


class AuthorizationError(BaseAPIException):
    """Authorization error exception."""

    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, status_code=403)


class StorageError(BaseAPIException):
    """Storage error exception."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class S3StorageError(StorageError):
    """S3-compatible storage error exception for legacy tutor endpoints."""


class ProcessingTimeoutError(OCRProcessingError):
    """Processing timeout error exception."""

    def __init__(self, timeout_seconds: int):
        message = f"Processing timeout after {timeout_seconds} seconds"
        details = {"timeout_seconds": timeout_seconds}
        super().__init__(message, details=details)
