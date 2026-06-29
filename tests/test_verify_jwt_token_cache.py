import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import jwt
import pytest

from app.services.auth_service import AuthService


def _make_auth_service(*, secret_key: str = "test-secret", supabase_jwt_secret: str = ""):
    service = object.__new__(AuthService)
    service.settings = SimpleNamespace(
        jwt_algorithm="HS256",
        jwt_audience=None,
        supabase_jwt_secret=supabase_jwt_secret,
        secret_key=secret_key,
    )
    service._verified_auth_users = {}
    service._jwt_payload_cache = {}
    service._fetch_auth_user = AsyncMock()
    return service


def _encode_token(secret: str, *, sub: str = "student-1", exp_offset: int = 3600) -> str:
    return jwt.encode(
        {
            "sub": sub,
            "email": "student@example.com",
            "exp": int(time.time()) + exp_offset,
        },
        secret,
        algorithm="HS256",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_verify_jwt_token_uses_local_decode_before_auth_api():
    service = _make_auth_service(secret_key="local-secret")
    token = _encode_token("local-secret")

    payload = await service.verify_jwt_token(token)

    assert payload["sub"] == "student-1"
    service._fetch_auth_user.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_verify_jwt_token_caches_payload_and_skips_repeat_work():
    service = _make_auth_service(secret_key="local-secret")
    token = _encode_token("local-secret")

    first = await service.verify_jwt_token(token)
    second = await service.verify_jwt_token(token)

    assert first["sub"] == "student-1"
    assert second["sub"] == "student-1"
    service._fetch_auth_user.assert_not_awaited()
    assert token in service._jwt_payload_cache


@pytest.mark.unit
@pytest.mark.asyncio
async def test_verify_jwt_token_falls_back_to_auth_api_when_local_decode_fails():
    service = _make_auth_service(secret_key="local-secret")
    token = _encode_token("other-secret")
    service._fetch_auth_user = AsyncMock(
        return_value={"id": "student-2", "email": "student2@example.com"}
    )

    payload = await service.verify_jwt_token(token)

    assert payload["sub"] == "student-2"
    service._fetch_auth_user.assert_awaited_once_with(token)
