from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException

from app.api import student_handlers as endpoints

USER_ID = "student-1"
COURSE_ID = "course-1"


class FakeDataService:
    def __init__(self, enrollments=None):
        self.enrollments = enrollments or []
        self.saved_enrollment = None

    async def get_course(self, course_id):
        if course_id != COURSE_ID:
            return None
        return {"course_id": COURSE_ID, "name": "Course 1"}

    async def get_user_enrollments(self, user_id):
        if user_id != USER_ID:
            return []
        return self.enrollments

    async def enroll_user_in_course(self, user_id, course_id, enrollment_data):
        self.saved_enrollment = {
            "user_id": user_id,
            "course_id": course_id,
            "enrollment_data": enrollment_data,
        }
        return "enr-new-1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_trial_enrollment_success_and_expires_in_one_day():
    db = FakeDataService(enrollments=[])
    response = await endpoints.enroll_user_in_course(
        user_id=USER_ID,
        course_id=COURSE_ID,
        enrollment_mode="trial",
        progress=0,
        completed_quizzes=0,
        total_quizzes=0,
        completed_questions=0,
        total_questions=0,
        data_service=db,
    )

    assert response["is_trial"] is True
    assert response["enrollment_mode"] == "trial"
    assert response["enrollment_id"] == "enr-new-1"
    assert db.saved_enrollment is not None

    saved = db.saved_enrollment["enrollment_data"]
    assert saved["enrollment_source"] == "trial"
    assert saved["enrollment_type"] == "trial"
    assert saved["trial_consumed_at"]
    assert saved["trial_expires_at"]
    assert saved["expires_at"] == saved["trial_expires_at"]

    started_at = datetime.fromisoformat(saved["started_at"])
    expires_at = datetime.fromisoformat(saved["expires_at"])
    duration = expires_at - started_at
    assert timedelta(hours=23, minutes=59) <= duration <= timedelta(days=1, minutes=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_trial_enrollment_is_blocked_when_user_already_used_trial():
    db = FakeDataService(
        enrollments=[
            {
                "enrollment_id": "enr-old-trial",
                "user_id": USER_ID,
                "course_id": "course-old",
                "status": "active",
                "enrollment_source": "trial",
                "trial_consumed_at": datetime.utcnow().isoformat(),
            }
        ]
    )

    with pytest.raises(HTTPException) as exc:
        await endpoints.enroll_user_in_course(
            user_id=USER_ID,
            course_id=COURSE_ID,
            enrollment_mode="trial",
            progress=0,
            completed_quizzes=0,
            total_quizzes=0,
            completed_questions=0,
            total_questions=0,
            data_service=db,
        )

    assert exc.value.status_code == 400
    assert "TRIAL_ALREADY_USED" in str(exc.value.detail)
    assert db.saved_enrollment is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_trial_enrollment_is_blocked_when_course_already_enrolled():
    db = FakeDataService(
        enrollments=[
            {
                "enrollment_id": "enr-existing",
                "user_id": USER_ID,
                "course_id": COURSE_ID,
                "status": "active",
                "started_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat(),
            }
        ]
    )

    with pytest.raises(HTTPException) as exc:
        await endpoints.enroll_user_in_course(
            user_id=USER_ID,
            course_id=COURSE_ID,
            enrollment_mode="trial",
            progress=0,
            completed_quizzes=0,
            total_quizzes=0,
            completed_questions=0,
            total_questions=0,
            data_service=db,
        )

    assert exc.value.status_code == 400
    assert "TRIAL_NOT_ALLOWED" in str(exc.value.detail)
    assert db.saved_enrollment is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_standard_enrollment_still_allows_without_trial_rules():
    db = FakeDataService(
        enrollments=[
            {
                "enrollment_id": "enr-old-trial",
                "user_id": USER_ID,
                "course_id": "course-old",
                "status": "active",
                "enrollment_source": "trial",
                "trial_consumed_at": datetime.utcnow().isoformat(),
            }
        ]
    )

    response = await endpoints.enroll_user_in_course(
        user_id=USER_ID,
        course_id=COURSE_ID,
        enrollment_mode="standard",
        progress=0,
        completed_quizzes=0,
        total_quizzes=0,
        completed_questions=0,
        total_questions=0,
        data_service=db,
    )

    assert response["is_trial"] is False
    assert response["enrollment_mode"] == "standard"
    assert db.saved_enrollment is not None
    saved = db.saved_enrollment["enrollment_data"]
    assert saved["enrollment_source"] == "manual"
    assert saved["expires_at"] is None
