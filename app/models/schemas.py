"""Pydantic models for request/response schemas."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


def serialize_utc_timestamp(value: datetime) -> str:
    """Serialize timestamps with an explicit UTC timezone for clients."""
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


class DocumentTypeEnum(str, Enum):
    """Document type enumeration."""

    DOCUMENT = "document"
    BOOK = "book"
    EXAM = "exam"


class ProcessingStatusEnum(str, Enum):
    """Processing status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class OCRRequest(BaseModel):
    """OCR processing request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    document_type: DocumentTypeEnum = Field(
        default=DocumentTypeEnum.DOCUMENT, description="Type of document to process"
    )
    language: str = Field(
        default="th",
        description="Primary language of the document",
        min_length=2,
        max_length=5,
    )
    enhance_markdown: bool = Field(
        default=True,
        description="Whether to enhance the output with markdown formatting",
    )
    course_id: Optional[str] = Field(
        default=None, description="Course ID to associate this document with"
    )


class QuestionChoice(BaseModel):
    """Question and choice data structure."""

    question: str = Field(description="The question text")
    context: Optional[str] = Field(
        default=None,
        description="Shared instructions, passage, or supporting context needed to answer the question",
    )
    choices: List[str] = Field(description="List of answer choices")
    correct_answer: Optional[int] = Field(
        default=None, description="Correct answer index if available"
    )
    explanation: Optional[str] = Field(
        default=None, description="Answer explanation if available"
    )
    difficulty: Optional[int] = Field(
        default=None, description="Difficulty score if available"
    )
    topic_tag: Optional[str] = Field(default=None, description="Topic tag if available")
    subject_tag: Optional[str] = Field(
        default=None, description="Subject tag if available"
    )
    image_url: Optional[str] = Field(
        default=None, description="Question image URL if available"
    )


class DocumentContent(BaseModel):
    """Document content structure for JSON storage."""

    document_type: str = Field(description="Type of document (document, book, exam)")
    title: Optional[str] = Field(default=None, description="Document title")
    content: Dict[str, Any] = Field(description="Document content in structured format")
    questions: Optional[List[QuestionChoice]] = Field(
        default=None, description="Extracted questions for exams"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional content metadata"
    )


class OCRResponse(BaseModel):
    """OCR processing response."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(description="Unique document identifier")
    status: ProcessingStatusEnum = Field(description="Processing status")
    original_text: Optional[str] = Field(default=None, description="Raw extracted text")
    structured_content: Optional[DocumentContent] = Field(
        default=None, description="Structured document content as JSON"
    )
    questions_data: Optional[List[QuestionChoice]] = Field(
        default=None, description="Extracted questions and choices"
    )
    confidence_score: Optional[float] = Field(
        default=None, description="OCR confidence score (0-1)", ge=0.0, le=1.0
    )
    processing_time_ms: Optional[int] = Field(
        default=None, description="Processing time in milliseconds", ge=0
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class QuizAugmentRequest(BaseModel):
    """Quiz augmentation request."""

    questions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Questions to augment"
    )
    language: str = Field(default="th", description="Language hint for augmentation")
    num_questions: Optional[int] = Field(
        default=None, description="Number of questions to return"
    )
    num_sets: Optional[int] = Field(
        default=None, description="Number of question sets to return"
    )
    classify_topic_tag: bool = Field(
        default=False,
        description="Whether to classify and return topic_tag for each question",
    )
    course_topics: Optional[List[str]] = Field(
        default=None, description="Allowed topic_tag values for classification"
    )
    mode: Literal["transform", "solve", "image_filter", "deconstruct"] = Field(
        default="transform",
        description="transform=create variants, solve=keep original question and infer correct_answer/explanation, image_filter=classify whether each question requires an image/diagram, deconstruct=extract copyright-safe academic skeleton",
    )


class QuizAugmentResponse(BaseModel):
    """Quiz augmentation response."""

    questions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Augmented questions"
    )
    sets: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Augmented question sets"
    )
    model: Optional[str] = Field(
        default=None, description="Model used for augmentation"
    )


class CourseAIGenerateRequest(BaseModel):
    """Course detail AI generation request."""

    prompt: str = Field(
        min_length=1, max_length=1200, description="Prompt for course details"
    )
    name: Optional[str] = Field(default=None, description="Course name")
    category: Optional[str] = Field(default=None, description="Course category")
    topics: List[str] = Field(
        default_factory=list, description="Course topic names for context"
    )
    content_items: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Course content items to generate descriptions for",
    )


class CourseAIGenerateResponse(BaseModel):
    """Course detail AI generation response."""

    description: str = Field(default="", description="Generated course description")
    target_profile: str = Field(
        default="", description="Generated target profile summary"
    )
    structure_summary: str = Field(
        default="", description="Generated structure summary"
    )
    content_items: List[Dict[str, str]] = Field(
        default_factory=list, description="Generated course content item descriptions"
    )


class DocumentInfo(BaseModel):
    """Document information."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(description="Unique document identifier")
    filename: str = Field(description="Original filename")
    file_size: int = Field(description="File size in bytes", gt=0)
    mime_type: str = Field(description="MIME type of the file")
    document_type: DocumentTypeEnum = Field(description="Type of document")
    upload_timestamp: datetime = Field(description="Upload timestamp")
    status: ProcessingStatusEnum = Field(description="Processing status")
    course_id: Optional[str] = Field(
        default=None, description="Course ID associated with this document"
    )


class ProcessingJob(BaseModel):
    """Processing job information."""

    model_config = ConfigDict(from_attributes=True)

    job_id: str = Field(description="Unique job identifier")
    document_id: str = Field(description="Associated document identifier")
    status: ProcessingStatusEnum = Field(description="Current job status")
    created_at: datetime = Field(description="Job creation timestamp")
    started_at: Optional[datetime] = Field(
        default=None, description="Job start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="Job completion timestamp"
    )
    progress_percentage: int = Field(
        default=0, description="Progress percentage", ge=0, le=100
    )
    result: Optional[OCRResponse] = Field(
        default=None, description="Job result if completed"
    )


class UploadResponse(BaseModel):
    """File upload response."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(description="Unique document identifier")
    filename: str = Field(description="Uploaded filename")
    file_size: int = Field(description="File size in bytes")
    mime_type: str = Field(description="MIME type")
    upload_url: str = Field(description="URL to access the uploaded file")
    message: str = Field(description="Success message")


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
    gemini_status: str = Field(description="Gemini API status")

    @field_serializer("timestamp")
    def serialize_timestamp(self, v: datetime) -> str:
        """Serialize datetime to ISO format string."""
        return v.isoformat()


class DocumentListResponse(BaseModel):
    """Document list response."""

    documents: List[DocumentInfo] = Field(description="List of documents")
    total_count: int = Field(description="Total number of documents", ge=0)
    page: int = Field(default=1, description="Current page number", ge=1)
    page_size: int = Field(default=20, description="Items per page", ge=1, le=100)
    total_pages: int = Field(description="Total number of pages", ge=0)


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
    email_verification_required: bool = Field(
        default=False, description="Whether the user must verify email before login"
    )


class ResendVerificationEmailRequest(BaseModel):
    """Request to resend a signup verification email."""

    model_config = ConfigDict(str_strip_whitespace=True)

    email: str = Field(description="Email address used during registration")


class ResendVerificationEmailResponse(BaseModel):
    """Response after requesting a verification email resend."""

    success: bool = Field(default=True, description="Request accepted")
    message: str = Field(description="User-facing status message")


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


# Invitation Models


class InvitationStatusEnum(str, Enum):
    """Invitation status enumeration."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"


class CreateInvitationRequest(BaseModel):
    """Create course invitation request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    student_id: str = Field(description="Student ID to invite")
    course_id: str = Field(description="Course ID")
    message: Optional[str] = Field(
        default=None, description="Optional invitation message"
    )


class CourseInvitation(BaseModel):
    """Course invitation model."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Invitation ID")
    course_id: str = Field(description="Course ID")
    course_name: str = Field(description="Course name")
    instructor_id: str = Field(description="Instructor ID")
    instructor_name: str = Field(description="Instructor name")
    student_id: str = Field(description="Student ID")
    student_email: Optional[str] = Field(default=None, description="Student email")
    status: InvitationStatusEnum = Field(description="Invitation status")
    message: Optional[str] = Field(default=None, description="Invitation message")
    created_at: datetime = Field(description="Creation timestamp")
    expires_at: datetime = Field(description="Expiration timestamp")

    @field_serializer("created_at", "expires_at")
    def serialize_datetime(self, v: datetime) -> str:
        """Serialize datetime to ISO format string."""
        return v.isoformat()


class InvitationResponse(BaseModel):
    """Invitation response."""

    success: bool = Field(description="Operation success status")
    message: str = Field(description="Response message")
    invitation_id: Optional[str] = Field(default=None, description="Invitation ID")


class AcceptInvitationRequest(BaseModel):
    """Accept invitation request."""

    invitation_id: str = Field(description="Invitation ID to accept")


class InvitationListResponse(BaseModel):
    """List of invitations response."""

    invitations: List[CourseInvitation] = Field(description="List of invitations")
    total: int = Field(description="Total number of invitations")


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


class CreateLessonRequest(BaseModel):
    """Create lesson request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(description="Lesson title")
    description: Optional[str] = Field(default="", description="Lesson description")
    order: int = Field(default=1, description="Lesson order in course")
    selectedDocuments: List[LessonDocument] = Field(
        default=[], description="Selected documents for lesson"
    )
    selectedQuizzes: List[LessonQuiz] = Field(
        default=[], description="Selected quizzes for lesson"
    )


class UpdateLessonRequest(BaseModel):
    """Update lesson request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: Optional[str] = Field(default=None, description="Lesson title")
    description: Optional[str] = Field(default=None, description="Lesson description")
    order: Optional[int] = Field(default=None, description="Lesson order in course")
    selectedDocuments: Optional[List[LessonDocument]] = Field(
        default=None, description="Selected documents for lesson"
    )
    selectedQuizzes: Optional[List[LessonQuiz]] = Field(
        default=None, description="Selected quizzes for lesson"
    )
    isPublished: Optional[bool] = Field(
        default=None, description="Lesson publish status"
    )


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
