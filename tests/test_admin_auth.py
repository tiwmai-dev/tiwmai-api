"""Tests for admin actor validation during migration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.utils.admin_auth import validate_admin_actor


@pytest.mark.asyncio
async def test_validate_admin_actor_allows_legacy_body_only_requests():
    admin_id = await validate_admin_actor("admin-1", None)
    assert admin_id == "admin-1"


@pytest.mark.asyncio
async def test_validate_admin_actor_requires_admin_group_when_token_present():
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="token-admin"
    )
    auth_service = SimpleNamespace(
        verify_jwt_token=AsyncMock(return_value={"sub": "admin-1"}),
        get_user_info=AsyncMock(
            return_value=SimpleNamespace(
                user_id="admin-1",
                username="admin-1",
                email="admin@example.com",
                groups=["student"],
            )
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await validate_admin_actor("admin-1", credentials, auth_service=auth_service)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_validate_admin_actor_matches_token_actor():
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="token-admin"
    )
    auth_service = SimpleNamespace(
        verify_jwt_token=AsyncMock(return_value={"sub": "admin-1"}),
        get_user_info=AsyncMock(
            return_value=SimpleNamespace(
                user_id="admin-1",
                username="admin-1",
                email="admin@example.com",
                groups=["admin"],
            )
        ),
    )

    admin_id = await validate_admin_actor(
        "admin-1", credentials, auth_service=auth_service
    )
    assert admin_id == "admin-1"
