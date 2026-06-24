"""Authentication middleware for JWT token verification."""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.logging import app_logger
from app.services.auth_service import AuthService, UserInfo


class AuthMiddleware:
    """Authentication middleware for protecting endpoints."""

    def __init__(self):
        self.security = HTTPBearer(auto_error=False)

    async def get_auth_service(self) -> AuthService:
        """Get authentication service instance."""
        return AuthService()

    async def verify_token(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(
            HTTPBearer(auto_error=False)
        ),
        auth_service: AuthService = Depends(lambda: AuthService()),
    ) -> Optional[UserInfo]:
        """Verify JWT token and return user info."""
        if not credentials:
            return None

        try:
            await auth_service.verify_jwt_token(credentials.credentials)
            return await auth_service.get_user_info(credentials.credentials)
        except HTTPException:
            raise
        except Exception as e:
            app_logger.error(f"Token verification error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def require_auth(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
        auth_service: AuthService = Depends(lambda: AuthService()),
    ) -> UserInfo:
        """Require authentication and return user info."""
        try:
            await auth_service.verify_jwt_token(credentials.credentials)
            return await auth_service.get_user_info(credentials.credentials)
        except HTTPException:
            raise
        except Exception as e:
            app_logger.error(f"Required authentication failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def require_admin(self, current_user: UserInfo) -> UserInfo:
        """Require admin privileges."""
        if not current_user.groups or "admin" not in current_user.groups:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required",
            )
        return current_user

    async def optional_auth(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(
            HTTPBearer(auto_error=False)
        ),
        auth_service: AuthService = Depends(lambda: AuthService()),
    ) -> Optional[UserInfo]:
        """Optional authentication - returns None if no valid token."""
        if not credentials:
            return None

        try:
            await auth_service.verify_jwt_token(credentials.credentials)
            return await auth_service.get_user_info(credentials.credentials)
        except Exception as e:
            app_logger.warning(f"Optional authentication failed: {e}")
            return None


auth_middleware = AuthMiddleware()

get_current_user = auth_middleware.require_auth
get_current_user_optional = auth_middleware.optional_auth


async def require_admin(
    current_user: UserInfo = Depends(get_current_user),
) -> UserInfo:
    return await auth_middleware.require_admin(current_user)
