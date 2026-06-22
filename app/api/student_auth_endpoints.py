"""Student Authentication API endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings
from app.core.exceptions import BaseAPIException
from app.core.logging import app_logger
from app.models.schemas import (
    AuthCallbackRequest,
    AuthResponse,
    LogoutResponse,
    OAuthSessionRequest,
    OAuthUrlResponse,
    RegisterResponse,
    StudentOnboardingProfile,
    StudentOnboardingResponse,
    TokenRefreshRequest,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
)
from app.services.data_service import get_db_service
from app.services.file_service import FileService
from app.services.student_auth_service import StudentAuthService, StudentInfo

# Create router
router = APIRouter(prefix="/student/auth", tags=["student-authentication"])

# Security scheme
security = HTTPBearer()


# Dependency to get student auth service
async def get_student_auth_service() -> StudentAuthService:
    return StudentAuthService()


async def get_data_service():
    return get_db_service()


async def get_file_service() -> FileService:
    return FileService()


# Dependency to get current student user
async def get_current_student(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: StudentAuthService = Depends(get_student_auth_service),
) -> StudentInfo:
    """Get current authenticated student."""
    try:
        # get_student_info validates the access token and loads its profile.
        student_info = await auth_service.get_student_info(credentials.credentials)

        return student_info

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error getting current student: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )


# Optional authentication dependency
async def get_current_student_optional(
    authorization: Optional[str] = Header(None),
    auth_service: StudentAuthService = Depends(get_student_auth_service),
) -> Optional[StudentInfo]:
    """Get current authenticated student (optional)."""
    if not authorization:
        return None

    try:
        # Extract token from Bearer header
        if not authorization.startswith("Bearer "):
            return None

        token = authorization.split(" ")[1]

        # get_student_info validates the access token and loads its profile.
        student_info = await auth_service.get_student_info(token)

        return student_info

    except Exception as e:
        app_logger.warning(f"Optional student auth failed: {e}")
        return None


class StudentRegisterRequest(UserRegisterRequest):
    """Student registration request with additional fields."""

    student_id: Optional[str] = None


@router.post("/register", response_model=RegisterResponse)
async def register_student(
    user_data: StudentRegisterRequest,
    auth_service: StudentAuthService = Depends(get_student_auth_service),
):
    """
    Register a new student.

    - **username**: Username (3-50 characters)
    - **email**: Valid email address
    - **password**: Password (minimum 8 characters)
    - **given_name**: First name (optional)
    - **family_name**: Last name (optional)
    - **student_id**: Student ID (optional)
    """
    try:
        app_logger.info(
            f"Student registration attempt for username: {user_data.username}"
        )

        result = await auth_service.register_student(
            username=user_data.username,
            password=user_data.password,
            email=user_data.email,
            given_name=user_data.given_name,
            family_name=user_data.family_name,
            student_id=user_data.student_id,
        )

        app_logger.info(f"Student {user_data.username} registered successfully")

        return RegisterResponse(
            success=True,
            message=result["message"],
            user_id=result["user_id"],
            email=user_data.email,
        )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Student registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Student registration failed",
        )


@router.post("/login", response_model=AuthResponse)
async def login_student(
    user_credentials: UserLoginRequest,
    auth_service: StudentAuthService = Depends(get_student_auth_service),
):
    """
    Student login with username/email and password.

    - **username**: Username or email address
    - **password**: User password

    Returns access token, refresh token, and student information.
    """
    try:
        app_logger.info(f"Student login attempt for: {user_credentials.username}")

        result = await auth_service.authenticate_student(
            username=user_credentials.username, password=user_credentials.password
        )

        app_logger.info(f"Student {user_credentials.username} logged in successfully")

        return AuthResponse(
            access_token=result["access_token"],
            refresh_token=result.get("refresh_token"),
            id_token=result.get("id_token"),
            token_type=result["token_type"],
            expires_in=result["expires_in"],
            user=result["user"],
        )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Student login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Student login failed",
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_student_token(
    refresh_data: TokenRefreshRequest,
    auth_service: StudentAuthService = Depends(get_student_auth_service),
):
    """
    Refresh student access token using refresh token.

    - **refresh_token**: Valid refresh token
    """
    try:
        app_logger.info("Student token refresh attempt")

        result = await auth_service.refresh_token(refresh_data.refresh_token)

        app_logger.info("Student token refreshed successfully")

        return TokenResponse(
            access_token=result["access_token"],
            id_token=result.get("id_token"),
            refresh_token=result.get("refresh_token"),
            token_type=result["token_type"],
            expires_in=result["expires_in"],
        )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Student token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Student token refresh failed",
        )


@router.post("/logout", response_model=LogoutResponse)
async def logout_student(
    current_student: StudentInfo = Depends(get_current_student),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: StudentAuthService = Depends(get_student_auth_service),
):
    """
    Student logout - invalidates all tokens for the student.

    Requires valid access token in Authorization header.
    """
    try:
        app_logger.info(f"Student logout attempt for: {current_student.username}")

        success = await auth_service.logout_student(credentials.credentials)

        if success:
            app_logger.info(
                f"Student {current_student.username} logged out successfully"
            )
            return LogoutResponse(
                success=True, message="Student logged out successfully"
            )
        else:
            app_logger.warning(f"Student logout failed for: {current_student.username}")
            return LogoutResponse(success=False, message="Student logout failed")

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Student logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Student logout failed",
        )


@router.get("/me", response_model=StudentInfo)
async def get_student_profile(
    current_student: StudentInfo = Depends(get_current_student),
):
    """
    Get current student profile information.

    Requires valid access token in Authorization header.
    """
    try:
        app_logger.info(f"Profile request for student: {current_student.username}")
        return current_student

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Student profile error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve student profile",
        )


@router.get("/onboarding-profile", response_model=StudentOnboardingResponse)
async def get_student_onboarding_profile(
    current_student: StudentInfo = Depends(get_current_student),
    data_service=Depends(get_data_service),
):
    """Get current student's onboarding completion status and profile."""
    try:
        onboarding_data = await data_service.get_student_onboarding(
            current_student.user_id,
            email=current_student.email,
            username=current_student.username,
            student_id=current_student.student_id,
        )
        return StudentOnboardingResponse(
            onboarding_completed=bool(onboarding_data.get("onboarding_completed")),
            onboarding_profile=onboarding_data.get("onboarding_profile"),
        )
    except Exception as e:
        app_logger.error(f"Failed to retrieve student onboarding profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve onboarding profile",
        )


@router.put("/onboarding-profile", response_model=StudentOnboardingResponse)
async def upsert_student_onboarding_profile(
    profile: StudentOnboardingProfile,
    current_student: StudentInfo = Depends(get_current_student),
    data_service=Depends(get_data_service),
):
    """Create or update mandatory onboarding profile for current student."""
    try:
        saved = await data_service.save_student_onboarding(
            user_id=current_student.user_id,
            onboarding_profile=profile.model_dump(),
            base_user={
                "email": current_student.email,
                "name": current_student.given_name or current_student.username,
                "username": current_student.username,
            },
        )
        return StudentOnboardingResponse(
            onboarding_completed=bool(saved.get("onboarding_completed")),
            onboarding_profile=saved.get("onboarding_profile"),
        )
    except Exception as e:
        app_logger.error(f"Failed to save student onboarding profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save onboarding profile",
        )


@router.post("/avatar")
async def upload_student_avatar(
    file: UploadFile = File(...),
    current_student: StudentInfo = Depends(get_current_student),
    file_service: FileService = Depends(get_file_service),
):
    """Upload current student's profile avatar to Supabase Storage."""
    try:
        settings = get_settings()
        if not settings.use_supabase_storage:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Supabase storage is disabled",
            )

        upload_metadata = await file_service.upload_profile_avatar_to_supabase(
            file,
            current_student.user_id,
        )
        storage_path = str(upload_metadata.get("s3_key") or "").strip()
        bucket_name = str(upload_metadata.get("bucket_name") or "").strip()
        avatar_url = str(upload_metadata.get("avatar_url") or "").strip()
        if not storage_path or not avatar_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to resolve uploaded avatar URL",
            )

        app_logger.info(
            f"Student avatar uploaded for {current_student.user_id}: "
            f"{bucket_name}/{storage_path}"
        )
        return {
            "filename": upload_metadata.get("original_filename") or file.filename,
            "avatar_url": avatar_url,
            "avatar_storage_path": storage_path,
            "avatar_bucket": bucket_name,
            "message": "Avatar uploaded successfully",
        }
    except HTTPException:
        raise
    except BaseAPIException as e:
        app_logger.error(f"Student avatar upload failed: {e}")
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        app_logger.error(f"Unexpected error during student avatar upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload avatar",
        )


@router.get("/oauth/authorize", response_model=OAuthUrlResponse)
async def get_student_oauth_url(
    state: Optional[str] = None,
    provider: Optional[str] = None,
    auth_service: StudentAuthService = Depends(get_student_auth_service),
):
    """
    Get OAuth authorization URL for student Cognito hosted UI.

    - **state**: Optional state parameter for CSRF protection
    - **provider**: Optional identity provider (e.g. Google)
    """
    try:
        authorization_url = auth_service.get_oauth_authorization_url(state, provider)

        return OAuthUrlResponse(authorization_url=authorization_url, state=state)
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Student OAuth URL generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Student OAuth URL generation failed",
        )


@router.post("/oauth/callback", response_model=AuthResponse)
async def student_oauth_callback(
    callback_data: AuthCallbackRequest,
    auth_service: StudentAuthService = Depends(get_student_auth_service),
):
    """
    Handle student OAuth callback and exchange code for tokens.

    - **code**: Authorization code from OAuth provider
    - **state**: State parameter for CSRF protection
    """
    try:
        app_logger.info("Processing student OAuth callback")

        token_response = await auth_service.exchange_code_for_tokens(callback_data.code)

        student_info = await auth_service.get_student_info(
            token_response["access_token"]
        )

        return AuthResponse(
            access_token=token_response["access_token"],
            id_token=token_response.get("id_token"),
            refresh_token=token_response.get("refresh_token"),
            token_type=token_response.get("token_type", "Bearer"),
            expires_in=token_response.get("expires_in", 3600),
            user=student_info,
        )
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Student OAuth callback error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Student OAuth callback processing failed",
        )


@router.post("/oauth/session", response_model=AuthResponse)
async def student_oauth_session(
    session_data: OAuthSessionRequest,
    auth_service: StudentAuthService = Depends(get_student_auth_service),
):
    """Normalize OAuth callback tokens into a backend-validated student session."""
    try:
        app_logger.info("Processing student OAuth session tokens")

        result = await auth_service.normalize_oauth_session(
            access_token=session_data.access_token,
            refresh_token=session_data.refresh_token,
            provider_token=session_data.provider_token,
        )

        return AuthResponse(
            access_token=result["access_token"],
            refresh_token=result.get("refresh_token"),
            id_token=result.get("id_token"),
            token_type=result["token_type"],
            expires_in=result["expires_in"],
            user=result["user"],
        )
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Student OAuth session error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Student OAuth session processing failed",
        )
