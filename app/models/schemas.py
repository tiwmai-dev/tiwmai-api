"""Pydantic models for request/response schemas."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


def serialize_utc_timestamp(value: datetime) -> str:
    """Serialize timestamps with an explicit UTC timezone for clients."""
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


class ErrorResponse(BaseModel):
    """Standard error response."""

    model_config = ConfigDict(from_attributes=True)

    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )
    request_id: Optional[str] = Field(
        default=None, description="Request identifier for tracking"
    )

    @field_serializer("timestamp")
    def serialize_timestamp(self, v: datetime) -> str:
        """Serialize datetime to ISO format string."""
        return v.isoformat()


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(description="API version")
    uptime_seconds: float = Field(description="Service uptime in seconds")
    llm_status: str = Field(description="LLM provider status")

    @field_serializer("timestamp")
    def serialize_timestamp(self, v: datetime) -> str:
        """Serialize datetime to ISO format string."""
        return v.isoformat()


# Authentication Models


class UserRegisterRequest(BaseModel):
    """User registration request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    username: str = Field(min_length=3, max_length=50, description="Username")
    email: str = Field(description="Email address")
    password: str = Field(min_length=8, description="Password")
    given_name: Optional[str] = Field(
        default=None, max_length=100, description="First name"
    )
    family_name: Optional[str] = Field(
        default=None, max_length=100, description="Last name"
    )


class UserLoginRequest(BaseModel):
    """User login request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    username: str = Field(description="Username or email")
    password: str = Field(description="Password")


class TokenRefreshRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str = Field(description="Refresh token")


class AuthCallbackRequest(BaseModel):
    """OAuth callback request."""

    code: str = Field(description="Authorization code")
    state: Optional[str] = Field(default=None, description="State parameter")


class OAuthSessionRequest(BaseModel):
    """OAuth session tokens returned to the browser callback."""

    access_token: str = Field(description="OAuth access token")
    refresh_token: Optional[str] = Field(
        default=None, description="OAuth refresh token"
    )
    provider_token: Optional[str] = Field(
        default=None, description="Provider access token"
    )
    provider_refresh_token: Optional[str] = Field(
        default=None, description="Provider refresh token"
    )
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: Optional[int] = Field(default=None, description="Token expiration")


class UserInfo(BaseModel):
    """User information."""

    model_config = ConfigDict(from_attributes=True)

    user_id: str = Field(description="User ID")
    username: str = Field(description="Username")
    email: str = Field(description="Email address")
    email_verified: bool = Field(description="Whether email is verified")
    given_name: Optional[str] = Field(default=None, description="First name")
    family_name: Optional[str] = Field(default=None, description="Last name")
    phone_number: Optional[str] = Field(default=None, description="Phone number")
    groups: Optional[List[str]] = Field(default=None, description="User groups")


class StudentOnboardingProfile(BaseModel):
    """Student mandatory onboarding profile."""

    model_config = ConfigDict(str_strip_whitespace=True)

    nickname: str = Field(min_length=1, max_length=50, description="Student nickname")
    grade_level: str = Field(min_length=1, max_length=60, description="Grade level")
    age: int = Field(ge=5, le=100, description="Student age")
    school: Optional[str] = Field(
        default=None, max_length=120, description="School name"
    )
    interested_subjects: List[str] = Field(
        min_length=1, max_length=8, description="Interested subjects"
    )
    primary_goal: Literal[
        "daily_practice",
        "exam_preparation",
        "learn_ahead",
    ] = Field(description="Primary learning goal")
    avatar_url: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="Public URL for the student's profile avatar",
    )
    avatar_storage_path: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Supabase Storage object path for the profile avatar",
    )
    avatar_bucket: Optional[str] = Field(
        default=None,
        max_length=120,
        description="Supabase Storage bucket containing the profile avatar",
    )
    avatar_data_url: Optional[str] = Field(
        default=None,
        max_length=1500000,
        description="Legacy local profile image encoded as a browser data URL",
    )

    @field_validator("avatar_data_url")
    @classmethod
    def validate_avatar_data_url(cls, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        if not value.startswith("data:image/") or ";base64," not in value:
            raise ValueError("avatar_data_url must be an image data URL")
        return value


class StudentOnboardingResponse(BaseModel):
    """Student onboarding status response."""

    onboarding_completed: bool = Field(
        default=False, description="Onboarding completion status"
    )
    onboarding_profile: Optional[StudentOnboardingProfile] = Field(
        default=None, description="Student onboarding profile"
    )


class AuthResponse(BaseModel):
    """Authentication response."""

    access_token: str = Field(description="Access token")
    id_token: Optional[str] = Field(default=None, description="ID token")
    refresh_token: Optional[str] = Field(default=None, description="Refresh token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(description="Token expiration in seconds")
    user: UserInfo = Field(description="User information")


class TokenResponse(BaseModel):
    """Token response."""

    access_token: str = Field(description="Access token")
    id_token: Optional[str] = Field(default=None, description="ID token")
    refresh_token: Optional[str] = Field(default=None, description="Refresh token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(description="Token expiration in seconds")


class RegisterResponse(BaseModel):
    """Registration response."""

    success: bool = Field(default=True, description="Registration success status")
    user_id: str = Field(description="User ID")
    email: str = Field(description="Email address")
    message: str = Field(description="Registration message")


class LogoutResponse(BaseModel):
    """Logout response."""

    message: str = Field(description="Logout message")
    success: bool = Field(description="Logout success status")


class OAuthUrlResponse(BaseModel):
    """OAuth URL response."""

    authorization_url: str = Field(description="OAuth authorization URL")
    state: Optional[str] = Field(default=None, description="State parameter")


# Chat Models


class ChatMessageRequest(BaseModel):
    """Chat message request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=2000, description="User message")
    course_id: Optional[str] = Field(default=None, description="Course ID for context")
    user_id: str = Field(description="User ID for personalization")
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID for context"
    )
    question_context: Optional[str] = Field(
        default=None, description="Current question text/context"
    )
    chat_mode: Optional[Literal["study_solver", "learning_advisor"]] = Field(
        default="study_solver", description="Chat behavior mode"
    )


class ChatMessage(BaseModel):
    """Chat message structure."""

    id: str = Field(description="Message ID")
    type: str = Field(description="Message type (user or ai)")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional message metadata"
    )

    @field_serializer("timestamp")
    def serialize_timestamp(self, v: datetime) -> str:
        """Serialize datetime to ISO format string."""
        return serialize_utc_timestamp(v)


class ChatResponse(BaseModel):
    """Chat AI response."""

    message_id: str = Field(description="Response message ID")
    content: str = Field(description="AI response content")
    timestamp: datetime = Field(description="Response timestamp")
    confidence: Optional[float] = Field(
        default=None, description="Response confidence score"
    )
    course_context: Optional[str] = Field(
        default=None, description="Course context used"
    )
    conversation_id: str = Field(description="Conversation ID")
    processing_time_ms: Optional[int] = Field(
        default=None, description="Processing time in milliseconds"
    )
    chat_energy_limit_thb: Optional[float] = Field(
        default=None,
        description="Effective daily chat energy limit in THB",
    )
    chat_energy_used_thb: Optional[float] = Field(
        default=None,
        description="Used chat energy for current day in THB",
    )
    chat_energy_remaining_thb: Optional[float] = Field(
        default=None,
        description="Remaining chat energy for current day in THB",
    )
    chat_energy_percent: Optional[float] = Field(
        default=None,
        description="Remaining energy percent (0-100)",
    )
    chat_energy_exhausted: Optional[bool] = Field(
        default=None,
        description="Whether chat energy is exhausted",
    )

    @field_serializer("timestamp")
    def serialize_timestamp(self, v: datetime) -> str:
        """Serialize datetime to ISO format string."""
        return serialize_utc_timestamp(v)


class ConversationHistory(BaseModel):
    """Conversation history."""

    conversation_id: str = Field(description="Conversation ID")
    user_id: str = Field(description="User ID")
    course_id: Optional[str] = Field(default=None, description="Course ID")
    messages: List[ChatMessage] = Field(description="List of messages in conversation")
    created_at: datetime = Field(description="Conversation creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")


# Lesson-related models
class LessonDocument(BaseModel):
    """Lesson document reference."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Document ID")
    title: Optional[str] = Field(default=None, description="Document title")
    type: Optional[str] = Field(default=None, description="Document type")


class LessonQuiz(BaseModel):
    """Lesson quiz reference."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Quiz ID")
    title: Optional[str] = Field(default=None, description="Quiz title")
    questions: Optional[int] = Field(default=None, description="Number of questions")


class Lesson(BaseModel):
    """Lesson model."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Lesson ID")
    title: str = Field(description="Lesson title")
    description: Optional[str] = Field(default="", description="Lesson description")
    order: int = Field(description="Lesson order in course")
    courseId: str = Field(description="Course ID")
    userId: str = Field(description="Creator user ID")
    documents: List[LessonDocument] = Field(default=[], description="Lesson documents")
    quizzes: List[LessonQuiz] = Field(default=[], description="Lesson quizzes")
    isPublished: bool = Field(default=False, description="Lesson publish status")
    createdAt: str = Field(description="Creation timestamp")
    updatedAt: str = Field(description="Update timestamp")


class LessonResponse(BaseModel):
    """Lesson response."""

    success: bool = Field(description="Operation success status")
    message: str = Field(description="Response message")
    lesson: Optional[Lesson] = Field(default=None, description="Lesson data")
    lesson_id: Optional[str] = Field(default=None, description="Lesson ID")


class LessonListResponse(BaseModel):
    """List of lessons response."""

    lessons: List[Lesson] = Field(description="List of lessons")
    total: int = Field(description="Total number of lessons")
