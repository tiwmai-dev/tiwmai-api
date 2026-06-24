"""Authentication API endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.logging import app_logger
from app.models.schemas import (
    AuthCallbackRequest,
    AuthResponse,
    LogoutResponse,
    OAuthUrlResponse,
    RegisterResponse,
    TokenRefreshRequest,
    TokenResponse,
    UserInfo,
    UserLoginRequest,
    UserRegisterRequest,
)
from app.services.auth_service import AuthService

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Security scheme
security = HTTPBearer()


# Dependency to get auth service
async def get_auth_service() -> AuthService:
    return AuthService()


# Dependency to get current user
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserInfo:
    """Get current authenticated user."""
    try:
        app_logger.info(f"=== AUTHENTICATION CHECK ===")
        app_logger.info(
            f"Token present: {bool(credentials and credentials.credentials)}"
        )
        if credentials and credentials.credentials:
            token_preview = (
                credentials.credentials[:50] + "..."
                if len(credentials.credentials) > 50
                else credentials.credentials
            )
            app_logger.info(f"Token preview: {token_preview}")

        # Verify JWT token
        payload = await auth_service.verify_jwt_token(credentials.credentials)
        app_logger.info(f"Token verification successful, payload: {payload}")

        # Get user info from access token
        user_info = await auth_service.get_user_info(credentials.credentials)
        app_logger.info(f"User info retrieved: {user_info}")

        return user_info

    except HTTPException as e:
        app_logger.error(
            f"HTTPException in authentication: {e.status_code} - {e.detail}"
        )
        raise
    except Exception as e:
        app_logger.error(f"Unexpected error getting current user: {e}")
        app_logger.error(f"Exception type: {type(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )


# Optional authentication dependency
async def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    auth_service: AuthService = Depends(get_auth_service),
) -> Optional[UserInfo]:
    """Get current authenticated user (optional)."""
    if not authorization:
        return None

    try:
        # Extract token from Bearer header
        if not authorization.startswith("Bearer "):
            return None

        token = authorization.split(" ")[1]

        # Verify JWT token
        payload = await auth_service.verify_jwt_token(token)

        # Get user info from access token
        user_info = await auth_service.get_user_info(token)

        return user_info

    except Exception as e:
        app_logger.warning(f"Optional auth failed: {e}")
        return None


@router.post("/register", response_model=RegisterResponse)
async def register(
    user_data: UserRegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Register a new user.

    - **username**: Username (3-50 characters)
    - **email**: Valid email address
    - **password**: Password (minimum 8 characters)
    - **given_name**: First name (optional)
    - **family_name**: Last name (optional)
    """
    try:
        app_logger.info(f"Registration attempt for username: {user_data.username}")

        result = await auth_service.register_user(
            username=user_data.username,
            password=user_data.password,
            email=user_data.email,
            given_name=user_data.given_name,
            family_name=user_data.family_name,
        )

        return RegisterResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed",
        )


@router.post("/login", response_model=AuthResponse)
async def login(
    login_data: UserLoginRequest, auth_service: AuthService = Depends(get_auth_service)
):
    """
    Authenticate user and return tokens.

    - **username**: Username or email
    - **password**: User password
    """
    try:
        app_logger.info(f"Login attempt for: {login_data.username}")

        result = await auth_service.authenticate_user(
            username=login_data.username, password=login_data.password
        )

        return AuthResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed",
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_data: TokenRefreshRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Refresh access token using refresh token.

    - **refresh_token**: Valid refresh token
    """
    try:
        result = await auth_service.refresh_token(refresh_data.refresh_token)

        return TokenResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed",
        )


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    current_user: UserInfo = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Logout current user and invalidate tokens.
    """
    try:
        success = await auth_service.logout_user(credentials.credentials)

        return LogoutResponse(
            message="Successfully logged out" if success else "Logout completed",
            success=success,
        )

    except Exception as e:
        app_logger.error(f"Logout error: {e}")
        # Even if logout fails, we consider it successful on client side
        return LogoutResponse(message="Logout completed", success=False)


@router.get("/me", response_model=UserInfo)
async def get_current_user_info(current_user: UserInfo = Depends(get_current_user)):
    """
    Get current authenticated user information.
    """
    return current_user


@router.get("/oauth/authorize", response_model=OAuthUrlResponse)
async def get_oauth_url(
    state: Optional[str] = None, auth_service: AuthService = Depends(get_auth_service)
):
    """
    Get OAuth authorization URL for Cognito hosted UI.

    - **state**: Optional state parameter for CSRF protection
    """
    try:
        authorization_url = auth_service.get_oauth_authorization_url(state)

        return OAuthUrlResponse(authorization_url=authorization_url, state=state)

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"OAuth URL generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth URL generation failed",
        )


@router.post("/oauth/callback", response_model=AuthResponse)
async def oauth_callback(
    callback_data: AuthCallbackRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Handle OAuth callback and exchange code for tokens.

    - **code**: Authorization code from OAuth provider
    - **state**: State parameter for CSRF protection
    """
    try:
        app_logger.info("Processing OAuth callback")

        # Exchange code for tokens
        token_response = await auth_service.exchange_code_for_tokens(callback_data.code)

        # Get user info from access token
        user_info = await auth_service.get_user_info(token_response["access_token"])

        return AuthResponse(
            access_token=token_response["access_token"],
            id_token=token_response.get("id_token"),
            refresh_token=token_response.get("refresh_token"),
            token_type=token_response.get("token_type", "Bearer"),
            expires_in=token_response.get("expires_in", 3600),
            user=user_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"OAuth callback error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth callback processing failed",
        )


@router.get("/health")
async def auth_health_check():
    """
    Authentication service health check.
    """
    try:
        # Test basic service initialization
        auth_service = AuthService()

        return {
            "status": "healthy",
            "service": "authentication",
            "provider": "supabase",
            "supabase_configured": bool(
                auth_service.settings.supabase_url
                and auth_service.settings.supabase_service_role_key
            ),
        }

    except Exception as e:
        app_logger.error(f"Auth health check failed: {e}")
        return {"status": "unhealthy", "service": "authentication", "error": str(e)}
