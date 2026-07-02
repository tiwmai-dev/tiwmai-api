from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api import student_handlers as endpoints

USER_ID = "student-1"
COURSE_ID = "course-1"
QUIZ_ID = "quiz-1"
NOW = datetime.utcnow().isoformat()


class FakeAuthService:
    def __init__(self, payload):
        self.payload = payload

    async def verify_jwt_token(self, token):
        return self.payload


class FakeDataService:
    def __init__(self, mode: str):
        self.mode = mode

    async def get_user(self, user_id):
        if user_id != USER_ID:
            return None
        if self.mode == "active":
            return {
                "user_id": USER_ID,
                "premium_subscription": {
                    "status": "active",
                    "started_at": (datetime.utcnow() - timedelta(days=10)).isoformat(),
                    "expires_at": (datetime.utcnow() + timedelta(days=5)).isoformat(),
                },
            }
        return {"user_id": USER_ID}

    async def get_user_enrollments(self, user_id):
        if user_id != USER_ID:
            return []
        if self.mode == "not_enrolled":
            return []
        if self.mode == "active":
            return [
                {
                    "enrollment_id": "enr-1",
                    "user_id": USER_ID,
                    "course_id": COURSE_ID,
                    "status": "active",
                    "enrollment_source": "premium",
                    "started_at": (datetime.utcnow() - timedelta(days=10)).isoformat(),
                }
            ]
        # "expired" mode: a stale non-free enrollment row with no active Premium
        # (e.g. a lapsed Premium subscription) must still be denied access.
        return [
            {
                "enrollment_id": "enr-1",
                "user_id": USER_ID,
                "course_id": COURSE_ID,
                "status": "active",
                "enrollment_source": "premium",
                "started_at": (datetime.utcnow() - timedelta(days=10)).isoformat(),
            }
        ]

    async def enroll_user_in_course(self, user_id, course_id, enrollment_data):
        return "enr-new-1"

    async def get_course_lessons(self, course_id, user_id=None):
        return [
            {
                "id": "lesson-1",
                "title": "บทที่ 1",
                "description": "desc",
                "order": 1,
                "courseId": COURSE_ID,
                "userId": USER_ID,
                "documents": [],
                "quizzes": [],
                "isPublished": True,
                "createdAt": NOW,
                "updatedAt": NOW,
            }
        ]

    async def get_quizzes_by_course(self, course_id):
        return [{"quiz_id": QUIZ_ID, "course_id": course_id, "title": "Quiz"}]

    async def get_quiz(self, quiz_id):
        return {
            "quiz_id": quiz_id,
            "course_id": COURSE_ID,
            "questions": [
                {
                    "id": "q1",
                    "question": "1+1=?",
                    "choices": ["2", "3"],
                    "correct_answer": 0,
                }
            ],
        }

    async def create_quiz_result(self, user_id, quiz_id, result):
        return "result-1"


class FakeAliasOnlyEnrollmentService(FakeDataService):
    async def get_user_enrollments(self, user_id):
        return []

    async def get_user_enrollments_with_aliases(self, user_id):
        if user_id != USER_ID:
            return []
        return await super().get_user_enrollments(USER_ID)


def _credentials():
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials="dummy-token")


def _auth_with_matching_user():
    return FakeAuthService({"sub": USER_ID, "cognito:username": USER_ID})


async def _run_get_lessons(db, credentials, auth):
    return await endpoints.get_course_lessons(
        course_id=COURSE_ID,
        user_id=USER_ID,
        credentials=credentials,
        student_auth_service=auth,
        data_service=db,
    )


async def _run_list_course_quizzes(db, credentials, auth):
    return await endpoints.list_course_quizzes(
        course_id=COURSE_ID,
        user_id=USER_ID,
        credentials=credentials,
        student_auth_service=auth,
        data_service=db,
    )


async def _run_get_quiz(db, credentials, auth):
    return await endpoints.get_quiz(
        quiz_id=QUIZ_ID,
        user_id=USER_ID,
        course_id=COURSE_ID,
        credentials=credentials,
        student_auth_service=auth,
        data_service=db,
    )


async def _run_submit(db, credentials, auth):
    payload = endpoints.QuizSubmitPayload(
        answers=[0],
        course_id=COURSE_ID,
        lesson_id="lesson-1",
    )
    return await endpoints.submit_quiz_answers(
        user_id=USER_ID,
        quiz_id=QUIZ_ID,
        payload=payload,
        credentials=credentials,
        student_auth_service=auth,
        data_service=db,
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner",
    [_run_get_lessons, _run_list_course_quizzes, _run_get_quiz, _run_submit],
)
async def test_active_enrollment_can_access(runner):
    db = FakeDataService("active")
    response = await runner(db, _credentials(), _auth_with_matching_user())
    assert response is not None


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner",
    [_run_get_lessons, _run_list_course_quizzes, _run_get_quiz, _run_submit],
)
async def test_active_enrollment_alias_lookup_can_access(runner):
    db = FakeAliasOnlyEnrollmentService("active")
    response = await runner(db, _credentials(), _auth_with_matching_user())
    assert response is not None


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner",
    [_run_get_lessons, _run_list_course_quizzes, _run_get_quiz, _run_submit],
)
async def test_expired_enrollment_is_blocked(runner):
    db = FakeDataService("expired")
    with pytest.raises(HTTPException) as exc:
        await runner(db, _credentials(), _auth_with_matching_user())
    assert exc.value.status_code == 403
    assert "PREMIUM_REQUIRED" in str(exc.value.detail)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner",
    [_run_get_lessons, _run_list_course_quizzes, _run_get_quiz, _run_submit],
)
async def test_not_enrolled_is_blocked(runner):
    db = FakeDataService("not_enrolled")
    with pytest.raises(HTTPException) as exc:
        await runner(db, _credentials(), _auth_with_matching_user())
    assert exc.value.status_code == 403
    assert "COURSE_ACCESS_DENIED" in str(exc.value.detail)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_user_id_must_match_token_principal():
    db = FakeDataService("active")
    auth = FakeAuthService(
        {"sub": "different-user", "cognito:username": "another-user"}
    )
    with pytest.raises(HTTPException) as exc:
        await _run_get_lessons(db, _credentials(), auth)
    assert exc.value.status_code == 403
    assert "USER_ID_TOKEN_MISMATCH" in str(exc.value.detail)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_submit_requires_bearer_token_when_user_bound():
    db = FakeDataService("active")
    with pytest.raises(HTTPException) as exc:
        await _run_submit(db, None, _auth_with_matching_user())
    assert exc.value.status_code == 401
    assert "missing bearer token" in str(exc.value.detail)
