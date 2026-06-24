"""Application configuration settings."""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    reload: bool = Field(default=False, env="RELOAD")

    # Security
    secret_key: str = Field(env="SECRET_KEY")
    access_token_expire_minutes: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )

    # Gemini Configuration (used by OCR service)
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash-lite", env="GEMINI_MODEL")
    gemini_timeout: int = Field(default=60, env="GEMINI_TIMEOUT")
    gemini_max_retries: int = Field(default=3, env="GEMINI_MAX_RETRIES")
    ocr_pdf_max_concurrency: int = Field(default=4, env="OCR_PDF_MAX_CONCURRENCY")
    quiz_gen_max_concurrency: int = Field(default=20, env="QUIZ_GEN_MAX_CONCURRENCY")

    # OpenRouter Chat Configuration (used by chat service)
    openrouter_api_key: str = Field(default="", env="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", env="OPENROUTER_BASE_URL"
    )
    openrouter_model: str = Field(default="openai/gpt-5-chat", env="OPENROUTER_MODEL")
    openrouter_chat_model: Optional[str] = Field(default=None, env="OPENROUTER_CHAT_MODEL")
    openrouter_site_url: Optional[str] = Field(default=None, env="OPENROUTER_SITE_URL")
    openrouter_site_name: Optional[str] = Field(default=None, env="OPENROUTER_SITE_NAME")
    openrouter_reasoning_effort: Optional[str] = Field(
        default=None, env="OPENROUTER_REASONING_EFFORT"
    )
    openrouter_cost_usd_to_thb: float = Field(default=36.0, env="OPENROUTER_COST_USD_TO_THB")
    chat_context_classifier_enabled: bool = Field(
        default=True, env="CHAT_CONTEXT_CLASSIFIER_ENABLED"
    )
    chat_context_classifier_confidence_threshold: float = Field(
        default=0.65, env="CHAT_CONTEXT_CLASSIFIER_CONFIDENCE_THRESHOLD"
    )
    chat_context_classifier_model: Optional[str] = Field(
        default=None, env="CHAT_CONTEXT_CLASSIFIER_MODEL"
    )
    chat_context_classifier_max_tokens: int = Field(
        default=140, env="CHAT_CONTEXT_CLASSIFIER_MAX_TOKENS"
    )

    # Legacy LiteLLM-compatible names used by migrated tutor/admin endpoints.
    litellm_api_key: str = Field(default="", env="LITELLM_API_KEY")
    litellm_base_url: str = Field(
        default="http://localhost:4000", env="LITELLM_BASE_URL"
    )
    litellm_model: str = Field(default="openai/gpt-5-chat", env="LITELLM_MODEL")
    litellm_generate_quiz_model: Optional[str] = Field(
        default=None, env="LITELLM_GENERATE_QUIZ_MODEL"
    )
    litellm_quiz_verify_model: Optional[str] = Field(
        default=None, env="LITELLM_QUIZ_VERIFY_MODEL"
    )
    litellm_chat_model: Optional[str] = Field(default=None, env="LITELLM_CHAT_MODEL")
    litellm_site_url: Optional[str] = Field(default=None, env="LITELLM_SITE_URL")
    litellm_site_name: Optional[str] = Field(default=None, env="LITELLM_SITE_NAME")
    litellm_reasoning_effort: Optional[str] = Field(
        default=None, env="LITELLM_REASONING_EFFORT"
    )
    litellm_cost_usd_to_thb: float = Field(default=36.0, env="LITELLM_COST_USD_TO_THB")
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # Payment configuration (optional)
    stripe_public_key: Optional[str] = Field(default=None, env="STRIPE_PUBLIC_KEY")
    stripe_private_key: Optional[str] = Field(default=None, env="STRIPE_PRIVATE_KEY")
    stripe_webhook_secret: Optional[str] = Field(
        default=None, env="STRIPE_WEBHOOK_SECRET"
    )
    resend_api_key: Optional[str] = Field(default=None, env="RESEND_API_KEY")
    payment_email_from: str = Field(
        default="Tiwmai <payments@tewmai.com>", env="PAYMENT_EMAIL_FROM"
    )
    student_web_app_url: Optional[str] = Field(default=None, env="STUDENT_WEB_APP_URL")

    # Supabase Configuration
    supabase_url: str = Field(default="", env="SUPABASE_URL")
    supabase_anon_key: str = Field(default="", env="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(default="", env="SUPABASE_SERVICE_ROLE_KEY")
    supabase_jwt_secret: str = Field(default="", env="SUPABASE_JWT_SECRET")
    supabase_storage_bucket: str = Field(
        default="tanaijarn-documents", env="SUPABASE_STORAGE_BUCKET"
    )
    supabase_course_images_bucket: str = Field(
        default="tiwmai-course-images", env="SUPABASE_COURSE_IMAGES_BUCKET"
    )
    supabase_avatar_bucket: str = Field(default="", env="SUPABASE_AVATAR_BUCKET")
    supabase_oauth_redirect_uri: Optional[str] = Field(
        default=None, env="SUPABASE_OAUTH_REDIRECT_URI"
    )
    use_supabase: bool = Field(default=True, env="USE_SUPABASE")
    use_supabase_storage: bool = Field(default=True, env="USE_SUPABASE_STORAGE")

    # Compatibility aliases for older endpoint names and frontend previews.
    use_dynamodb: bool = Field(default=False, env="USE_DYNAMODB")
    use_s3_storage: bool = Field(default=True, env="USE_S3_STORAGE")
    s3_bucket_name: str = Field(default="", env="S3_BUCKET_NAME")
    s3_region: str = Field(default="", env="S3_REGION")

    # File Upload Settings
    max_file_size: int = Field(default=10485760, env="MAX_FILE_SIZE")  # 10MB
    max_pdf_file_size: int = Field(
        default=52428800, env="MAX_PDF_FILE_SIZE"
    )  # 50MB
    allowed_extensions: str = Field(
        default="png,jpg,jpeg,tiff,bmp,gif,webp,pdf,docx,doc",
        env="ALLOWED_EXTENSIONS",
    )
    upload_folder: str = Field(default="./uploads", env="UPLOAD_FOLDER")

    # Database Configuration
    database_url: str = Field(default="sqlite:///./sql_app.db", env="DATABASE_URL")

    # JWT Configuration
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_audience: Optional[str] = Field(default=None, env="JWT_AUDIENCE")

    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        env="LOG_FORMAT",
    )

    # Error monitoring
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    sentry_send_default_pii: bool = Field(default=True, env="SENTRY_SEND_DEFAULT_PII")
    sentry_environment: str = Field(default="development", env="SENTRY_ENVIRONMENT")

    # CORS Settings
    allowed_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://localhost:3005,http://127.0.0.1:3005",
        env="ALLOWED_ORIGINS",
    )
    allowed_methods: str = Field(
        default="GET,POST,PUT,DELETE,OPTIONS", env="ALLOWED_METHODS"
    )
    allowed_headers: str = Field(default="*", env="ALLOWED_HEADERS")

    @property
    def allowed_extensions_list(self) -> List[str]:
        """Get allowed extensions as a list."""
        if not self.allowed_extensions.strip():
            return [
                "png",
                "jpg",
                "jpeg",
                "tiff",
                "bmp",
                "gif",
                "webp",
                "pdf",
                "docx",
                "doc",
            ]
        return [
            ext.strip().lower()
            for ext in self.allowed_extensions.split(",")
            if ext.strip()
        ]

    @property
    def allowed_origins_list(self) -> List[str]:
        """Get allowed origins as a list."""
        if not self.allowed_origins.strip():
            return [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:3001",
            ]
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]

    @property
    def allowed_methods_list(self) -> List[str]:
        """Get allowed methods as a list."""
        if not self.allowed_methods.strip():
            return ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        return [
            method.strip().upper()
            for method in self.allowed_methods.split(",")
            if method.strip()
        ]

    @property
    def allowed_headers_list(self) -> List[str]:
        """Get allowed headers as a list."""
        if self.allowed_headers.strip() == "*":
            return ["*"]
        return [
            header.strip()
            for header in self.allowed_headers.split(",")
            if header.strip()
        ]

    @field_validator(
        "openrouter_reasoning_effort", "litellm_reasoning_effort", mode="before"
    )
    @classmethod
    def normalize_reasoning_effort(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized or None

    @field_validator("chat_context_classifier_confidence_threshold", mode="before")
    @classmethod
    def normalize_classifier_confidence_threshold(cls, value: Optional[float]) -> float:
        try:
            threshold = float(value if value is not None else 0.65)
        except (TypeError, ValueError):
            threshold = 0.65
        return max(0.0, min(1.0, threshold))

    @field_validator("chat_context_classifier_max_tokens", mode="before")
    @classmethod
    def normalize_classifier_max_tokens(cls, value: Optional[int]) -> int:
        try:
            max_tokens = int(value if value is not None else 140)
        except (TypeError, ValueError):
            max_tokens = 140
        return max(64, min(512, max_tokens))

    @field_validator(
        "debug",
        "reload",
        "chat_context_classifier_enabled",
        "sentry_send_default_pii",
        mode="before",
    )
    @classmethod
    def normalize_bool_flags(cls, value):
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        if text in {"1", "true", "yes", "on", "debug", "dev", "development"}:
            return True
        if text in {"0", "false", "no", "off", "release", "prod", "production", ""}:
            return False
        return False

    @model_validator(mode="after")
    def fill_legacy_llm_defaults(self):
        """Let migrated LiteLLM endpoints run against the existing OpenRouter config."""
        if not self.litellm_api_key and self.openrouter_api_key:
            self.litellm_api_key = self.openrouter_api_key
        if (
            self.litellm_base_url == "http://localhost:4000"
            and self.openrouter_base_url
        ):
            self.litellm_base_url = self.openrouter_base_url
        if self.litellm_model == "openai/gpt-5-chat" and self.openrouter_model:
            self.litellm_model = self.openrouter_model
        if not self.litellm_chat_model and self.openrouter_chat_model:
            self.litellm_chat_model = self.openrouter_chat_model
        if not self.litellm_site_url and self.openrouter_site_url:
            self.litellm_site_url = self.openrouter_site_url
        if not self.litellm_site_name and self.openrouter_site_name:
            self.litellm_site_name = self.openrouter_site_name
        if not self.litellm_reasoning_effort and self.openrouter_reasoning_effort:
            self.litellm_reasoning_effort = self.openrouter_reasoning_effort
        return self

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
