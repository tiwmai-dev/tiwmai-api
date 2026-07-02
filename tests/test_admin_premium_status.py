from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.api import endpoints


USER_ID = "student-premium-1"
ADMIN_ID = "admin-local"


class FakeDynamoService:
    def __init__(self, user=None):
        self.user = user or {"user_id": USER_ID, "email": "u@example.com"}
        self.saved_subscription = None

    async def get_user(self, user_id):
        if user_id != USER_ID:
            return None
        return self.user

    async def save_premium_subscription(self, user_id, subscription):
        if user_id != USER_ID:
            raise ValueError(f"User {user_id} not found")
        self.saved_subscription = dict(subscription)
        self.user = {**self.user, "premium_subscription": dict(subscription)}
        return self.user


def _premium_request(**overrides):
    payload = {
        "admin_user_id": ADMIN_ID,
        "tier": "premium",
        "reason": "support grant",
        "duration_months": 1,
    }
    payload.update(overrides)
    return endpoints.AdminUserPremiumStatusRequest(**payload)


@pytest.mark.asyncio
async def test_admin_grant_premium_sets_active_subscription():
    service = FakeDynamoService()
    payload = _premium_request()

    result = await endpoints.admin_override_user_premium_status(
        USER_ID,
        payload,
        credentials=None,
        dynamodb_service=service,
    )

    assert result["tier"] == "premium"
    assert result["is_active"] is True
    assert service.saved_subscription["status"] == "active"
    assert service.saved_subscription["payment_provider"] == "admin"
    assert service.saved_subscription["admin_override"]["updated_by"] == ADMIN_ID


@pytest.mark.asyncio
async def test_admin_revoke_premium_sets_expired():
    future = (datetime.utcnow() + timedelta(days=30)).isoformat()
    service = FakeDynamoService(
        user={
            "user_id": USER_ID,
            "premium_subscription": {
                "status": "active",
                "expires_at": future,
                "started_at": datetime.utcnow().isoformat(),
            },
        }
    )
    payload = _premium_request(tier="free", reason="downgrade test")

    result = await endpoints.admin_override_user_premium_status(
        USER_ID,
        payload,
        credentials=None,
        dynamodb_service=service,
    )

    assert result["tier"] == "free"
    assert result["is_active"] is False
    assert service.saved_subscription["status"] == "expired"


@pytest.mark.asyncio
async def test_admin_premium_requires_reason():
    service = FakeDynamoService()
    payload = _premium_request(reason="")

    with pytest.raises(HTTPException) as exc:
        await endpoints.admin_override_user_premium_status(
            USER_ID,
            payload,
            credentials=None,
            dynamodb_service=service,
        )

    assert exc.value.status_code == 400


def test_admin_premium_route_is_registered():
    from app.main import app

    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/api/v1/admin/users/{user_id}/premium-status" in paths
