from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api import chat_endpoints
from app.models.schemas import ChatResponse

USER_ID = "student-1"


class FakeAuthService:
    def __init__(self, payload):
        self.payload = payload

    async def verify_jwt_token(self, token):
        return self.payload


class FakeDataService:
    def __init__(self, premium: bool):
        self.premium = premium

    async def get_user(self, user_id):
        if not self.premium:
            return {"user_id": USER_ID}
        return {
            "user_id": USER_ID,
            "premium_subscription": {
                "status": "active",
                "started_at": (datetime.utcnow() - timedelta(days=10)).isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(days=5)).isoformat(),
            },
        }


class FakeChatService:
    def normalize_chat_mode(self, chat_mode):
        return chat_mode

    async def get_chat_response(self, **kwargs):
        return ChatResponse(
            message_id="msg-1",
            content="ok",
            conversation_id="conv-1",
            timestamp=datetime.utcnow(),
        )


def _credentials():
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials="dummy-token")


def _auth_with_matching_user():
    return FakeAuthService({"sub": USER_ID, "cognito:username": USER_ID})


async def _send(db, chat_mode):
    return await chat_endpoints.send_chat_message(
        message="ช่วยหน่อย",
        user_id=USER_ID,
        course_id="course-1",
        conversation_id=None,
        question_context=None,
        chat_mode=chat_mode,
        image=None,
        credentials=_credentials(),
        student_auth_service=_auth_with_matching_user(),
        chat_service=FakeChatService(),
        data_service=db,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_study_solver_blocked_for_free_user():
    db = FakeDataService(premium=False)
    with pytest.raises(HTTPException) as exc:
        await _send(db, "study_solver")
    assert exc.value.status_code == 403
    assert "PREMIUM_REQUIRED" in str(exc.value.detail)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_study_solver_allowed_for_premium_user():
    db = FakeDataService(premium=True)
    response = await _send(db, "study_solver")
    assert response is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_learning_advisor_allowed_for_free_user():
    db = FakeDataService(premium=False)
    response = await _send(db, "learning_advisor")
    assert response is not None
