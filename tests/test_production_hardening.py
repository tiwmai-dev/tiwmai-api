from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from app.api import student_handlers as endpoints
from app.core.config import Settings
from app.main import create_app


def test_production_app_hides_docs_and_minimizes_root():
    app = create_app(Settings(secret_key="test-secret", debug=False))
    client = TestClient(app)

    assert client.get("/").json() == {"ok": True}
    assert client.get("/api/docs").status_code == 404
    assert client.get("/api/redoc").status_code == 404
    assert client.get("/api/openapi.json").status_code == 404


def test_private_user_endpoint_requires_bearer_token():
    app = create_app(Settings(secret_key="test-secret", debug=True))
    client = TestClient(app)

    response = client.get("/api/v1/student/users/student-1/enrolled-courses")

    assert response.status_code == 401
    assert response.json()["message"] == "UNAUTHORIZED: missing bearer token"


@pytest.mark.asyncio
async def test_required_user_token_rejects_mismatched_user():
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="token-for-someone-else",
    )
    auth_service = SimpleNamespace(
        verify_jwt_token=AsyncMock(return_value={
            "sub": "student-2",
            "username": "student-2",
        })
    )

    with pytest.raises(Exception) as exc_info:
        await endpoints._require_user_matches_token(
            user_id="student-1",
            credentials=credentials,
            auth_service=auth_service,
        )

    assert getattr(exc_info.value, "status_code", None) == 403
    assert getattr(exc_info.value, "detail", None) == "USER_ID_TOKEN_MISMATCH"


def test_promptpay_create_intent_requires_bearer_token_before_payment_work():
    app = create_app(Settings(secret_key="test-secret", debug=True))
    client = TestClient(app)

    response = client.post(
        "/api/v1/student/payments/promptpay/create-intent",
        json={
            "user_id": "student-1",
            "course_id": "course-1",
            "amount_thb": 199,
        },
    )

    assert response.status_code == 401
    assert response.json()["message"] == "UNAUTHORIZED: missing bearer token"


def test_stripe_webhook_does_not_require_bearer_token(monkeypatch):
    app = create_app(Settings(secret_key="test-secret", debug=True))
    client = TestClient(app)

    monkeypatch.setattr(
        endpoints,
        "get_settings",
        lambda: SimpleNamespace(stripe_webhook_secret="whsec_test"),
    )
    monkeypatch.setattr(
        endpoints,
        "_verify_stripe_webhook_signature",
        lambda payload, signature_header, webhook_secret: None,
    )

    response = client.post(
        "/api/v1/student/payments/stripe/webhook",
        content=b'{"type":"payment_intent.created","data":{"object":{}}}',
        headers={"Stripe-Signature": "test-signature"},
    )

    assert response.status_code == 200
    assert response.json() == {"received": True}
