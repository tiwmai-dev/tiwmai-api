import hashlib
import hmac
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.api import student_handlers as endpoints

USER_ID = "student-1"
COURSE_ID = "course-1"
ENROLLMENT_ID = "enr-1"
PAYMENT_INTENT_ID = "pi_test_123"


@pytest.mark.unit
def test_paid_at_from_intent_serializes_stripe_timestamp_as_explicit_utc():
    paid_at = endpoints._paid_at_from_intent({"created": 1778887800})

    assert paid_at == "2026-05-15T23:30:00+00:00"


@pytest.mark.unit
def test_payment_order_id_uses_bangkok_calendar_date():
    intent_id = "pi_bangkok_date"
    expected_suffix = hashlib.sha1(intent_id.encode("utf-8")).hexdigest()[:4].upper()

    order_id = endpoints._build_payment_order_id(
        "2026-05-15T18:30:00+00:00",
        intent_id,
    )

    assert order_id == f"TM20260516-{expected_suffix}"


@pytest.mark.unit
def test_payment_order_id_treats_legacy_naive_timestamp_as_utc():
    intent_id = "pi_legacy_utc"
    expected_suffix = hashlib.sha1(intent_id.encode("utf-8")).hexdigest()[:4].upper()

    order_id = endpoints._build_payment_order_id("2026-05-15T18:30:00", intent_id)

    assert order_id == f"TM20260516-{expected_suffix}"


class FakeDataService:
    def __init__(self, expires_at: str):
        self.expires_at = expires_at
        self.updated = None
        self.enrolled = None

    async def get_course(self, course_id):
        if course_id != COURSE_ID:
            return None
        return {
            "course_id": COURSE_ID,
            "name": "Course",
            "price": 199,
            "pricing_plans": [{"duration_months": 1, "price": 199, "label": "1 เดือน"}],
        }

    async def get_user(self, user_id):
        return {"user_id": user_id, "email": "u@example.com"}

    async def get_user_enrollments(self, user_id):
        if user_id != USER_ID:
            return []
        return [
            {
                "enrollment_id": ENROLLMENT_ID,
                "user_id": USER_ID,
                "course_id": COURSE_ID,
                "status": "active",
                "started_at": (datetime.utcnow() - timedelta(days=60)).isoformat(),
                "expires_at": self.expires_at,
                "duration_months": 1,
                "payment_provider": "stripe",
                "payment_type": "promptpay",
                "payment_intent_id": "pi_old",
                "payment_status": "succeeded",
                "paid_amount_thb": 199.0,
                "paid_currency": "THB",
                "plan_label": "1 เดือน",
                "paid_at": (datetime.utcnow() - timedelta(days=60)).isoformat(),
            }
        ]

    async def update_enrollment(self, enrollment_id, updates):
        if enrollment_id != ENROLLMENT_ID:
            return False
        self.updated = updates
        return True

    async def enroll_user_in_course(self, user_id, course_id, enrollment_data):
        self.enrolled = {
            "user_id": user_id,
            "course_id": course_id,
            "data": enrollment_data,
        }
        return "new-enrollment-id"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_intent_allows_when_enrollment_is_active(monkeypatch):
    db = FakeDataService(
        expires_at=(datetime.utcnow() + timedelta(days=3)).isoformat()
    )
    monkeypatch.setattr(
        endpoints,
        "get_settings",
        lambda: SimpleNamespace(
            stripe_private_key="sk_test", stripe_public_key="pk_test"
        ),
    )
    stripe_request_mock = AsyncMock(
        return_value={
            "id": PAYMENT_INTENT_ID,
            "client_secret": "cs_test",
            "currency": "thb",
            "status": "requires_payment_method",
        }
    )
    monkeypatch.setattr(endpoints, "_stripe_request", stripe_request_mock)

    body = endpoints.PromptPayCreateIntentRequest(
        user_id=USER_ID,
        course_id=COURSE_ID,
        billing_email="u@example.com",
        amount_thb=199,
        duration_months=1,
        plan_label="1 เดือน",
    )
    response = await endpoints.create_promptpay_payment_intent(
        body=body, data_service=db
    )
    assert response.payment_intent_id == PAYMENT_INTENT_ID
    assert response.already_enrolled is True
    stripe_payload = stripe_request_mock.await_args.kwargs["data"]
    assert "receipt_email" not in stripe_payload


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_intent_allows_when_enrollment_is_expired(monkeypatch):
    db = FakeDataService(
        expires_at=(datetime.utcnow() - timedelta(days=1)).isoformat()
    )
    monkeypatch.setattr(
        endpoints,
        "get_settings",
        lambda: SimpleNamespace(
            stripe_private_key="sk_test", stripe_public_key="pk_test"
        ),
    )
    monkeypatch.setattr(
        endpoints,
        "_stripe_request",
        AsyncMock(
            return_value={
                "id": PAYMENT_INTENT_ID,
                "client_secret": "cs_test",
                "currency": "thb",
                "status": "requires_payment_method",
            }
        ),
    )

    body = endpoints.PromptPayCreateIntentRequest(
        user_id=USER_ID,
        course_id=COURSE_ID,
        amount_thb=199,
        duration_months=1,
        plan_label="1 เดือน",
    )
    response = await endpoints.create_promptpay_payment_intent(
        body=body, data_service=db
    )
    assert response.payment_intent_id == PAYMENT_INTENT_ID
    assert response.already_enrolled is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_confirm_payment_renews_expired_enrollment(monkeypatch):
    db = FakeDataService(
        expires_at=(datetime.utcnow() - timedelta(days=2)).isoformat()
    )
    monkeypatch.setattr(
        endpoints,
        "get_settings",
        lambda: SimpleNamespace(
            stripe_private_key="sk_test", stripe_public_key="pk_test"
        ),
    )
    monkeypatch.setattr(
        endpoints,
        "_stripe_request",
        AsyncMock(
            return_value={
                "id": PAYMENT_INTENT_ID,
                "status": "succeeded",
                "currency": "thb",
                "amount_received": 19900,
                "receipt_email": "u@example.com",
                "created": int(datetime.utcnow().timestamp()),
                "latest_charge": {
                    "id": "ch_test_123",
                    "receipt_number": "1234-5678",
                    "receipt_url": "https://pay.stripe.com/receipts/test",
                },
                "metadata": {
                    "user_id": USER_ID,
                    "course_id": COURSE_ID,
                    "duration_months": "1",
                    "plan_label": "1 เดือน",
                },
            }
        ),
    )

    payload = endpoints.PromptPayConfirmRequest(
        user_id=USER_ID,
        course_id=COURSE_ID,
        payment_intent_id=PAYMENT_INTENT_ID,
    )
    response = await endpoints.confirm_promptpay_payment_and_enroll(
        body=payload,
        data_service=db,
    )

    assert response["enrolled"] is True
    assert response["enrollment_id"] == ENROLLMENT_ID
    assert "renewed" in response["message"].lower()
    assert db.updated is not None
    assert isinstance(db.updated.get("payment_history"), list)
    assert len(db.updated["payment_history"]) == 2
    assert db.updated["payment_history"][0]["payment_intent_id"] == "pi_old"
    new_cycle = db.updated["payment_history"][1]
    assert new_cycle["payment_intent_id"] == PAYMENT_INTENT_ID
    assert new_cycle["order_id"] == endpoints._build_payment_order_id(
        new_cycle["paid_at"], PAYMENT_INTENT_ID
    )
    assert new_cycle["stripe_charge_id"] == "ch_test_123"
    assert new_cycle["receipt_number"] == "1234-5678"
    assert new_cycle["receipt_url"] == "https://pay.stripe.com/receipts/test"
    assert response["order_id"] == new_cycle["order_id"]
    assert response["receipt_url"] == "https://pay.stripe.com/receipts/test"
    assert db.enrolled is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_confirm_payment_does_not_enqueue_payment_email(monkeypatch):
    db = FakeDataService(
        expires_at=(datetime.utcnow() - timedelta(days=2)).isoformat()
    )
    monkeypatch.setattr(
        endpoints,
        "get_settings",
        lambda: SimpleNamespace(
            stripe_private_key="sk_test", stripe_public_key="pk_test"
        ),
    )
    monkeypatch.setattr(
        endpoints,
        "_stripe_request",
        AsyncMock(
            return_value={
                "id": PAYMENT_INTENT_ID,
                "status": "succeeded",
                "currency": "thb",
                "amount_received": 19900,
                "receipt_email": "u@example.com",
                "created": int(datetime.utcnow().timestamp()),
                "latest_charge": {
                    "id": "ch_test_123",
                    "receipt_url": "https://pay.stripe.com/receipts/test",
                },
                "metadata": {
                    "user_id": USER_ID,
                    "course_id": COURSE_ID,
                    "duration_months": "1",
                    "plan_label": "1 เดือน",
                },
            }
        ),
    )
    payload = endpoints.PromptPayConfirmRequest(
        user_id=USER_ID,
        course_id=COURSE_ID,
        payment_intent_id=PAYMENT_INTENT_ID,
    )
    response = await endpoints.confirm_promptpay_payment_and_enroll(
        body=payload,
        data_service=db,
    )

    assert response["enrolled"] is True
    new_cycle = db.updated["payment_history"][1]
    assert "payment_success_email_status" not in new_cycle
    assert "payment_success_email_job_id" not in new_cycle
    assert new_cycle["order_id"] == response["order_id"]
    assert new_cycle["receipt_url"] == "https://pay.stripe.com/receipts/test"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_confirm_payment_does_not_enqueue_duplicate_email(monkeypatch):
    db = FakeDataService(
        expires_at=(datetime.utcnow() + timedelta(days=30)).isoformat()
    )

    async def _get_user_enrollments(_user_id):
        return [
            {
                "enrollment_id": ENROLLMENT_ID,
                "user_id": USER_ID,
                "course_id": COURSE_ID,
                "status": "active",
                "payment_history": [
                    {
                        "payment_intent_id": PAYMENT_INTENT_ID,
                        "payment_status": "succeeded",
                        "paid_amount_thb": 199.0,
                        "paid_at": datetime.utcnow().isoformat(),
                    }
                ],
            }
        ]

    db.get_user_enrollments = _get_user_enrollments
    monkeypatch.setattr(
        endpoints,
        "get_settings",
        lambda: SimpleNamespace(
            stripe_private_key="sk_test", stripe_public_key="pk_test"
        ),
    )
    monkeypatch.setattr(
        endpoints,
        "_stripe_request",
        AsyncMock(
            return_value={
                "id": PAYMENT_INTENT_ID,
                "status": "succeeded",
                "currency": "thb",
                "amount_received": 19900,
                "receipt_email": "u@example.com",
                "created": int(datetime.utcnow().timestamp()),
                "latest_charge": {
                    "id": "ch_test_123",
                    "receipt_url": "https://pay.stripe.com/receipts/test",
                },
                "metadata": {
                    "user_id": USER_ID,
                    "course_id": COURSE_ID,
                },
            }
        ),
    )
    response = await endpoints._complete_promptpay_payment(
        payment_intent_id=PAYMENT_INTENT_ID,
        data_service=db,
        expected_user_id=USER_ID,
        expected_course_id=COURSE_ID,
    )

    assert response["enrolled"] is True
    assert "already confirmed" in response["message"].lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_confirm_payment_renews_active_enrollment_and_appends_history(
    monkeypatch,
):
    active_expires_at = (datetime.utcnow() + timedelta(days=5)).isoformat()
    db = FakeDataService(expires_at=active_expires_at)
    monkeypatch.setattr(
        endpoints,
        "get_settings",
        lambda: SimpleNamespace(
            stripe_private_key="sk_test", stripe_public_key="pk_test"
        ),
    )
    monkeypatch.setattr(
        endpoints,
        "_stripe_request",
        AsyncMock(
            return_value={
                "id": PAYMENT_INTENT_ID,
                "status": "succeeded",
                "currency": "thb",
                "amount_received": 19900,
                "receipt_email": "u@example.com",
                "created": int(datetime.utcnow().timestamp()),
                "latest_charge": {
                    "id": "ch_test_123",
                    "receipt_url": "https://pay.stripe.com/receipts/test",
                },
                "metadata": {
                    "user_id": USER_ID,
                    "course_id": COURSE_ID,
                    "duration_months": "1",
                    "plan_label": "1 เดือน",
                },
            }
        ),
    )

    payload = endpoints.PromptPayConfirmRequest(
        user_id=USER_ID,
        course_id=COURSE_ID,
        payment_intent_id=PAYMENT_INTENT_ID,
    )
    response = await endpoints.confirm_promptpay_payment_and_enroll(
        body=payload,
        data_service=db,
    )

    assert response["enrolled"] is True
    assert response["enrollment_id"] == ENROLLMENT_ID
    assert "renewed" in response["message"].lower()
    assert db.updated is not None
    assert isinstance(db.updated.get("payment_history"), list)
    assert len(db.updated["payment_history"]) == 2
    old_cycle = db.updated["payment_history"][0]
    new_cycle = db.updated["payment_history"][1]
    assert old_cycle["payment_intent_id"] == "pi_old"
    assert new_cycle["payment_intent_id"] == PAYMENT_INTENT_ID
    assert new_cycle["order_id"] == endpoints._build_payment_order_id(
        new_cycle["paid_at"], PAYMENT_INTENT_ID
    )
    assert new_cycle["stripe_charge_id"] == "ch_test_123"
    assert new_cycle["receipt_url"] == "https://pay.stripe.com/receipts/test"
    assert endpoints._parse_iso_datetime(
        new_cycle["started_at"]
    ) == endpoints._parse_iso_datetime(active_expires_at)
    assert endpoints._parse_iso_datetime(
        new_cycle["expires_at"]
    ) > endpoints._parse_iso_datetime(active_expires_at)
    assert db.enrolled is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_payment_history_returns_all_payment_cycles():
    db = FakeDataService(
        expires_at=(datetime.utcnow() + timedelta(days=30)).isoformat()
    )
    paid_at_old = (datetime.utcnow() - timedelta(days=30)).isoformat()
    paid_at_new = (datetime.utcnow() - timedelta(days=2)).isoformat()

    async def _get_user_enrollments(_user_id):
        return [
            {
                "enrollment_id": ENROLLMENT_ID,
                "user_id": USER_ID,
                "course_id": COURSE_ID,
                "status": "active",
                "enrolled_at": paid_at_old,
                "payment_history": [
                    {
                        "order_id": "TM20260501-OLD1",
                        "payment_intent_id": "pi_old",
                        "stripe_charge_id": "ch_old",
                        "receipt_url": "https://pay.stripe.com/receipts/old",
                        "payment_status": "succeeded",
                        "paid_amount_thb": 199.0,
                        "duration_months": 1,
                        "paid_at": paid_at_old,
                        "started_at": paid_at_old,
                        "expires_at": (
                            datetime.utcnow() - timedelta(days=1)
                        ).isoformat(),
                    },
                    {
                        "payment_intent_id": "pi_new",
                        "stripe_charge_id": "ch_new",
                        "receipt_url": "https://pay.stripe.com/receipts/new",
                        "payment_status": "succeeded",
                        "paid_amount_thb": 299.0,
                        "duration_months": 3,
                        "paid_at": paid_at_new,
                        "started_at": paid_at_new,
                        "expires_at": (
                            datetime.utcnow() + timedelta(days=88)
                        ).isoformat(),
                    },
                ],
            }
        ]

    db.get_user_enrollments = _get_user_enrollments
    response = await endpoints.get_user_payment_history(
        user_id=USER_ID, data_service=db
    )

    assert response["total"] == 2
    assert response["rows"][0]["payment_intent_id"] == "pi_new"
    assert response["rows"][0]["order_id"] == endpoints._build_payment_order_id(
        paid_at_new, "pi_new"
    )
    assert response["rows"][0]["stripe_charge_id"] == "ch_new"
    assert response["rows"][0]["receipt_url"] == "https://pay.stripe.com/receipts/new"
    assert response["rows"][1]["payment_intent_id"] == "pi_old"
    assert response["rows"][1]["order_id"] == "TM20260501-OLD1"
    assert response["rows"][1]["receipt_url"] == "https://pay.stripe.com/receipts/old"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_payment_history_hydrates_missing_receipt_url(monkeypatch):
    db = FakeDataService(
        expires_at=(datetime.utcnow() + timedelta(days=30)).isoformat()
    )
    paid_at = (datetime.utcnow() - timedelta(days=2)).isoformat()

    async def _get_user_enrollments(_user_id):
        return [
            {
                "enrollment_id": ENROLLMENT_ID,
                "user_id": USER_ID,
                "course_id": COURSE_ID,
                "status": "active",
                "enrolled_at": paid_at,
                "payment_history": [
                    {
                        "payment_intent_id": "pi_missing_receipt",
                        "payment_status": "succeeded",
                        "paid_amount_thb": 299.0,
                        "duration_months": 3,
                        "paid_at": paid_at,
                        "started_at": paid_at,
                        "expires_at": (
                            datetime.utcnow() + timedelta(days=88)
                        ).isoformat(),
                    },
                ],
            }
        ]

    db.get_user_enrollments = _get_user_enrollments
    monkeypatch.setattr(
        endpoints,
        "get_settings",
        lambda: SimpleNamespace(stripe_private_key="sk_test"),
    )
    stripe_request_mock = AsyncMock(
        return_value={
            "id": "pi_missing_receipt",
            "latest_charge": {
                "id": "ch_missing_receipt",
                "receipt_number": "9876-5432",
                "receipt_url": "https://pay.stripe.com/receipts/hydrated",
            },
        }
    )
    monkeypatch.setattr(endpoints, "_stripe_request", stripe_request_mock)

    response = await endpoints.get_user_payment_history(
        user_id=USER_ID, data_service=db
    )

    assert response["rows"][0]["payment_intent_id"] == "pi_missing_receipt"
    assert response["rows"][0]["stripe_charge_id"] == "ch_missing_receipt"
    assert response["rows"][0]["receipt_number"] == "9876-5432"
    assert (
        response["rows"][0]["receipt_url"] == "https://pay.stripe.com/receipts/hydrated"
    )
    stripe_request_mock.assert_awaited_once()


@pytest.mark.unit
def test_verify_stripe_webhook_signature_accepts_valid_signature():
    payload = b'{"type":"payment_intent.succeeded"}'
    timestamp = "1700000000"
    secret = "whsec_test"
    signature = hmac.new(
        secret.encode("utf-8"),
        b".".join([timestamp.encode("utf-8"), payload]),
        hashlib.sha256,
    ).hexdigest()

    endpoints._verify_stripe_webhook_signature(
        payload=payload,
        signature_header=f"t={timestamp},v1={signature}",
        webhook_secret=secret,
    )


@pytest.mark.unit
def test_verify_stripe_webhook_signature_rejects_invalid_signature():
    with pytest.raises(HTTPException):
        endpoints._verify_stripe_webhook_signature(
            payload=b"{}",
            signature_header="t=1700000000,v1=bad",
            webhook_secret="whsec_test",
        )
