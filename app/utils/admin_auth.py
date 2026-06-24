"""Admin authorization helpers."""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.services.auth_service import AuthService, UserInfo

ADMIN_BEARER_OPTIONAL = HTTPBearer(auto_error=False)


async def _get_auth_service() -> AuthService:
    return AuthService()


async def validate_admin_actor(
    admin_user_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        ADMIN_BEARER_OPTIONAL
    ),
    auth_service: AuthService = Depends(_get_auth_service),
) -> str:
    """Validate admin mutation requests.

    When a bearer token is supplied, require an admin role and a matching actor id.
    When no token is supplied, preserve legacy body-only admin checks for migration.
    """
    normalized_admin_id = str(admin_user_id or "").strip()
    if not normalized_admin_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="admin_user_id is required",
        )

    if not credentials or not credentials.credentials:
        return normalized_admin_id

    await auth_service.verify_jwt_token(credentials.credentials)
    user_info: UserInfo = await auth_service.get_user_info(credentials.credentials)
    groups = user_info.groups or []
    if "admin" not in groups:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )

    actor_ids = {
        str(user_info.user_id or "").strip(),
        str(user_info.username or "").strip(),
        str(getattr(user_info, "email", "") or "").strip(),
    }
    if normalized_admin_id not in actor_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="admin_user_id must match authenticated admin",
        )

    return normalized_admin_id
