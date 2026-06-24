from types import SimpleNamespace
from unittest.mock import AsyncMock

import jwt
import pytest
from fastapi import HTTPException

from app.api.student_auth_endpoints import get_current_student
from app.services.auth_service import AuthService, _registration_error_detail
from app.services.student_auth_service import StudentAuthService, StudentInfo


class _UserInfoStub:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return dict(self._payload)

    def __getattr__(self, name):
        return self._payload.get(name)


class _ProfileQueryStub:
    def __init__(self, rows):
        self.rows = rows
        self.filters = []

    def select(self, *_args):
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def limit(self, *_args):
        return self

    def execute(self):
        rows = self.rows
        for column, value in self.filters:
            rows = [row for row in rows if row.get(column) == value]
        return SimpleNamespace(data=rows)


def _make_auth_service(run_side_effect):
    service = object.__new__(AuthService)
    set_session = object()
    refresh_session = object()

    anon_auth = SimpleNamespace(
        set_session=set_session,
        refresh_session=refresh_session,
    )
    anon_client = SimpleNamespace(auth=anon_auth)

    service.supabase = SimpleNamespace(
        run=AsyncMock(side_effect=run_side_effect),
        anon_client=anon_client,
    )
    service.settings = SimpleNamespace(
        supabase_url="",
        supabase_anon_key="",
        supabase_service_role_key="",
        access_token_expire_minutes=30,
        secret_key="test-secret",
    )
    return service, set_session, refresh_session


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_user_info_reuses_verified_supabase_user():
    service = object.__new__(AuthService)
    service._verified_auth_users = {}
    service._profile_cache = {}
    service._fetch_auth_user = AsyncMock(
        return_value={
            "id": "student-1",
            "email": "student@example.com",
            "email_confirmed_at": "2026-06-12T00:00:00Z",
            "metadata": {"username": "student-1"},
            "app_metadata": {"role": "student"},
        }
    )
    service._get_profile = AsyncMock(
        return_value={
            "user_id": "student-1",
            "email": "student@example.com",
            "username": "student-1",
            "role": "student",
        }
    )

    user = await AuthService.get_user_info(service, "access-token")

    assert user.user_id == "student-1"
    service._fetch_auth_user.assert_awaited_once_with("access-token")
    service._get_profile.assert_awaited_once_with("student-1", "student@example.com")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_current_student_does_not_verify_token_twice():
    student = StudentInfo(
        user_id="student-1",
        username="student-1",
        email="student@example.com",
        email_verified=True,
    )
    auth_service = SimpleNamespace(
        get_student_info=AsyncMock(return_value=student),
        verify_jwt_token=AsyncMock(),
    )

    result = await get_current_student(
        credentials=SimpleNamespace(credentials="access-token"),
        auth_service=auth_service,
    )

    assert result == student
    auth_service.get_student_info.assert_awaited_once_with("access-token")
    auth_service.verify_jwt_token.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_instructor_login_email_resolution_prefers_instructor_profile():
    rows = [
        {
            "username": "drive135",
            "role": "student",
            "email": "d.thus_sk135@hotmail.com",
        },
        {
            "username": "drive135",
            "role": "instructor",
            "email": "evirdz5@gmail.com",
        },
    ]
    service = object.__new__(AuthService)
    service.supabase = SimpleNamespace(
        client=SimpleNamespace(table=lambda table_name: _ProfileQueryStub(rows)),
        run=AsyncMock(side_effect=lambda fn: fn()),
    )

    email = await AuthService._resolve_login_email(service, "drive135")

    assert email == "evirdz5@gmail.com"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_normalize_oauth_session_survives_set_and_refresh_failures():
    async def _run(method, *args):
        raise RuntimeError("supabase session call failed")

    service, _, _ = _make_auth_service(_run)
    service.get_user_info = AsyncMock(
        return_value=_UserInfoStub(
            {
                "user_id": "student-1",
                "username": "student@example.com",
                "email": "student@example.com",
                "email_verified": True,
            }
        )
    )
    service._create_local_google_session = AsyncMock()

    result = await AuthService.normalize_oauth_session(
        service,
        access_token="access-token-1",
        refresh_token="refresh-token-1",
        provider_token="provider-token-1",
    )

    assert result["access_token"] != "access-token-1"
    assert result["refresh_token"] == "refresh-token-1"
    assert result["token_type"] == "Bearer"
    assert result["user"]["user_id"] == "student-1"
    assert result["id_token"] == result["access_token"]
    payload = jwt.decode(
        result["access_token"],
        "test-secret",
        algorithms=["HS256"],
        options={"verify_aud": False},
    )
    assert payload["tiwmai_auth"] is True
    assert payload["sub"] == "student-1"
    service.get_user_info.assert_awaited_once_with("access-token-1")
    service._create_local_google_session.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_normalize_oauth_session_falls_back_to_provider_token_when_supabase_verification_fails():
    async def _run(method, *args):
        raise RuntimeError("supabase session call failed")

    service, _, _ = _make_auth_service(_run)
    service.get_user_info = AsyncMock(
        side_effect=HTTPException(status_code=401, detail="Invalid access token")
    )
    service._create_local_google_session = AsyncMock(
        return_value={
            "access_token": "local-access-token",
            "refresh_token": None,
            "id_token": "local-access-token",
            "token_type": "Bearer",
            "expires_in": 1800,
            "user": {
                "user_id": "student-local",
                "username": "google_1067",
                "email": "student@example.com",
                "email_verified": True,
            },
        }
    )

    result = await AuthService.normalize_oauth_session(
        service,
        access_token="access-token-2",
        refresh_token="refresh-token-2",
        provider_token="provider-token-2",
    )

    assert result["access_token"] == "local-access-token"
    assert result["refresh_token"] == "refresh-token-2"
    service._create_local_google_session.assert_awaited_once_with("provider-token-2")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_normalize_oauth_session_does_not_use_supabase_token_for_google_fallback_without_provider_token():
    async def _run(method, *args):
        raise RuntimeError("supabase session call failed")

    service, _, _ = _make_auth_service(_run)
    service.get_user_info = AsyncMock(
        side_effect=HTTPException(status_code=401, detail="Invalid access token")
    )
    service._create_local_google_session = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        await AuthService.normalize_oauth_session(
            service,
            access_token="access-token-3",
            refresh_token="refresh-token-3",
            provider_token=None,
        )

    assert exc_info.value.status_code == 401
    service._create_local_google_session.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_normalize_oauth_session_uses_rest_refresh_fallback_when_sdk_refresh_fails():
    async def _run(method, *args):
        raise RuntimeError("supabase session call failed")

    service, _, _ = _make_auth_service(_run)
    service.get_user_info = AsyncMock(
        return_value=_UserInfoStub(
            {
                "user_id": "student-rest",
                "username": "student@example.com",
                "email": "student@example.com",
                "email_verified": True,
            }
        )
    )
    service._refresh_session_via_rest = AsyncMock(
        return_value={
            "access_token": "rest-access-token",
            "refresh_token": "rest-refresh-token",
            "expires_in": 3600,
        }
    )
    service._create_local_google_session = AsyncMock()

    result = await AuthService.normalize_oauth_session(
        service,
        access_token="expired-access-token",
        refresh_token="refresh-token-4",
        provider_token=None,
    )

    assert result["access_token"] != "rest-access-token"
    assert result["refresh_token"] == "rest-refresh-token"
    assert result["expires_in"] == 1800
    payload = jwt.decode(
        result["access_token"],
        "test-secret",
        algorithms=["HS256"],
        options={"verify_aud": False},
    )
    assert payload["sub"] == "student-rest"
    service.get_user_info.assert_awaited_once_with("rest-access-token")
    service._create_local_google_session.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_refresh_token_returns_local_jwt_and_latest_refresh_token():
    refresh_session = object()

    async def _run(method, *args):
        if method is refresh_session:
            return {
                "session": {
                    "access_token": "supabase-access-token",
                    "refresh_token": "supabase-refresh-token-new",
                    "expires_in": 3600,
                }
            }
        raise RuntimeError("unexpected call")

    service = object.__new__(AuthService)
    service.supabase = SimpleNamespace(
        run=AsyncMock(side_effect=_run),
        anon_client=SimpleNamespace(
            auth=SimpleNamespace(refresh_session=refresh_session)
        ),
    )
    service.settings = SimpleNamespace(
        supabase_url="",
        supabase_anon_key="",
        supabase_service_role_key="",
        access_token_expire_minutes=30,
        secret_key="test-secret",
    )
    service.get_user_info = AsyncMock(
        return_value=_UserInfoStub(
            {
                "user_id": "student-refresh",
                "username": "student@example.com",
                "email": "student@example.com",
                "email_verified": True,
            }
        )
    )
    service._refresh_session_via_rest = AsyncMock(return_value=None)

    result = await AuthService.refresh_token(service, "supabase-refresh-token-old")

    assert result["access_token"] != "supabase-access-token"
    assert result["refresh_token"] == "supabase-refresh-token-new"
    assert result["id_token"] == result["access_token"]
    assert result["expires_in"] == 1800
    payload = jwt.decode(
        result["access_token"],
        "test-secret",
        algorithms=["HS256"],
        options={"verify_aud": False},
    )
    assert payload["sub"] == "student-refresh"
    service.get_user_info.assert_awaited_once_with("supabase-access-token")


@pytest.mark.unit
def test_registration_error_detail_maps_duplicate_email():
    assert (
        _registration_error_detail(RuntimeError("User already registered"))
        == "อีเมลนี้ถูกใช้สมัครบัญชีแล้ว"
    )


@pytest.mark.unit
def test_registration_error_detail_maps_duplicate_username():
    assert (
        _registration_error_detail(
            RuntimeError('duplicate key value violates unique constraint "profiles_username_key"')
        )
        == "ชื่อผู้ใช้นี้ถูกใช้แล้ว"
    )


@pytest.mark.unit
def test_registration_error_detail_maps_weak_password():
    assert (
        _registration_error_detail(
            RuntimeError("Password should be at least 8 characters")
        )
        == "Password does not meet the minimum security requirements"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_student_register_skips_profile_rewrite_without_student_id():
    service = object.__new__(StudentAuthService)
    service.register_user = AsyncMock(
        return_value={
            "user_id": "student-1",
            "email": "student@example.com",
            "message": "User registered successfully",
        }
    )
    service._upsert_profile = AsyncMock()

    result = await StudentAuthService.register_student(
        service,
        username="student",
        password="password123",
        email="student@example.com",
    )

    assert result["student_id"] is None
    assert result["message"] == "Student registered successfully"
    service._upsert_profile.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_student_register_updates_profile_with_student_id():
    service = object.__new__(StudentAuthService)
    service.register_user = AsyncMock(
        return_value={
            "user_id": "student-1",
            "email": "student@example.com",
            "message": "User registered successfully",
        }
    )
    service._upsert_profile = AsyncMock()

    result = await StudentAuthService.register_student(
        service,
        username="student",
        password="password123",
        email="student@example.com",
        student_id="S001",
    )

    assert result["student_id"] == "S001"
    service._upsert_profile.assert_awaited_once_with(
        "student-1",
        "student@example.com",
        "student",
        "student",
        None,
        None,
        {"student_id": "S001"},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_user_rejects_duplicate_username():
    service = object.__new__(AuthService)
    service._username_is_taken = AsyncMock(return_value=True)
    service.supabase = SimpleNamespace(run=AsyncMock())

    with pytest.raises(HTTPException) as exc_info:
        await AuthService.register_user(
            service,
            username="existing-user",
            password="password123",
            email="new@example.com",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "ชื่อผู้ใช้นี้ถูกใช้แล้ว"
    service.supabase.run.assert_not_awaited()
