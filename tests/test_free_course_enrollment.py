from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException

from app.api import student_handlers as endpoints

USER_ID = "student-1"
COURSE_ID = "course-1"
OTHER_COURSE_ID = "course-2"


class FakeDataService:
    def __init__(self, enrollments=None, user=None):
        self.enrollments = enrollments or []
        self.user = user or {"user_id": USER_ID, "email": "u@example.com"}
        self.saved_enrollment = None

    async def get_course(self, course_id):
        if course_id not in {COURSE_ID, OTHER_COURSE_ID}:
            return None
        return {"course_id": course_id, "name": "Course"}

    async def get_user(self, user_id):
        if user_id != USER_ID:
            return None
        return self.user

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
        enrollment = {
            "enrollment_id": "enr-new-1",
            "user_id": user_id,
            "course_id": course_id,
            "status": "active",
            **enrollment_data,
        }
        self.enrollments = [*self.enrollments, enrollment]
        return "enr-new-1"

    async def update_enrollment(self, enrollment_id, updates):
        updated = None
        next_enrollments = []
        for enrollment in self.enrollments:
            if str(enrollment.get("enrollment_id")) == str(enrollment_id):
                updated = {**enrollment, **updates}
                next_enrollments.append(updated)
            else:
                next_enrollments.append(enrollment)
        if updated is None:
            return False
        self.enrollments = next_enrollments
        self.saved_enrollment = {
            "enrollment_id": enrollment_id,
            "updates": updates,
            "enrollment": updated,
        }
        return True


def _active_premium_subscription():
    return {
        "status": "active",
        "plan_id": "1m",
        "plan_label": "1 เดือน",
        "duration_months": 1,
        "started_at": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(days=20)).isoformat(),
    }


def _expired_premium_subscription():
    return {
        "status": "active",
        "plan_id": "1m",
        "plan_label": "1 เดือน",
        "duration_months": 1,
        "started_at": (datetime.utcnow() - timedelta(days=60)).isoformat(),
        "expires_at": (datetime.utcnow() - timedelta(days=30)).isoformat(),
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_first_free_course_claim_succeeds():
    db = FakeDataService(enrollments=[])
    response = await endpoints.enroll_user_in_course(
        user_id=USER_ID,
        course_id=COURSE_ID,
        enrollment_mode="free",
        progress=0,
        completed_quizzes=0,
        total_quizzes=0,
        completed_questions=0,
        total_questions=0,
        data_service=db,
    )

    assert response["is_free_course"] is True
    assert response["enrollment_mode"] == "free"
    assert response["enrollment_id"] == "enr-new-1"
    assert response["expires_at"] is None
    assert db.saved_enrollment is not None

    saved = db.saved_enrollment["enrollment_data"]
    assert saved["enrollment_source"] == "free"
    assert saved["enrollment_type"] == "free"
    assert saved["free_course_claimed_at"]
    assert "expires_at" not in saved


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reclaiming_same_free_course_is_idempotent():
    db = FakeDataService(
        enrollments=[
            {
                "enrollment_id": "enr-existing-free",
                "user_id": USER_ID,
                "course_id": COURSE_ID,
                "status": "active",
                "enrollment_source": "free",
                "free_course_claimed_at": datetime.utcnow().isoformat(),
            }
        ]
    )

    response = await endpoints.enroll_user_in_course(
        user_id=USER_ID,
        course_id=COURSE_ID,
        enrollment_mode="free",
        progress=0,
        completed_quizzes=0,
        total_quizzes=0,
        completed_questions=0,
        total_questions=0,
        data_service=db,
    )

    assert response["is_free_course"] is True
    assert response["enrollment_id"] == "enr-existing-free"
    assert db.saved_enrollment is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_claiming_a_different_free_course_is_rejected():
    db = FakeDataService(
        enrollments=[
            {
                "enrollment_id": "enr-existing-free",
                "user_id": USER_ID,
                "course_id": OTHER_COURSE_ID,
                "status": "active",
                "enrollment_source": "free",
                "free_course_claimed_at": datetime.utcnow().isoformat(),
            }
        ]
    )

    with pytest.raises(HTTPException) as exc:
        await endpoints.enroll_user_in_course(
            user_id=USER_ID,
            course_id=COURSE_ID,
            enrollment_mode="free",
            progress=0,
            completed_quizzes=0,
            total_quizzes=0,
            completed_questions=0,
            total_questions=0,
            data_service=db,
        )

    assert exc.value.status_code == 400
    assert "FREE_COURSE_ALREADY_CLAIMED" in str(exc.value.detail)
    assert db.saved_enrollment is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_free_course_converts_existing_premium_enrollment_after_premium_expires():
    db = FakeDataService(
        enrollments=[
            {
                "enrollment_id": "enr-premium-1",
                "user_id": USER_ID,
                "course_id": COURSE_ID,
                "status": "active",
                "enrollment_source": "premium",
                "enrollment_type": "premium",
                "last_activity": "เพิ่งเข้าร่วมด้วยสิทธิ์ Premium",
            }
        ],
        user={
            "user_id": USER_ID,
            "email": "u@example.com",
            "premium_subscription": _expired_premium_subscription(),
        },
    )

    response = await endpoints.enroll_user_in_course(
        user_id=USER_ID,
        course_id=COURSE_ID,
        enrollment_mode="free",
        progress=0,
        completed_quizzes=0,
        total_quizzes=0,
        completed_questions=0,
        total_questions=0,
        data_service=db,
    )

    assert response["is_free_course"] is True
    assert response["enrollment_id"] == "enr-premium-1"
    assert db.saved_enrollment["updates"]["enrollment_source"] == "free"
    assert db.saved_enrollment["updates"]["free_course_claimed_at"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_standard_enrollment_still_allowed_regardless_of_free_course_state():
    db = FakeDataService(
        enrollments=[
            {
                "enrollment_id": "enr-existing-free",
                "user_id": USER_ID,
                "course_id": OTHER_COURSE_ID,
                "status": "active",
                "enrollment_source": "free",
                "free_course_claimed_at": datetime.utcnow().isoformat(),
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

    assert response["is_free_course"] is False
    assert response["enrollment_mode"] == "standard"
    assert db.saved_enrollment is not None
    saved = db.saved_enrollment["enrollment_data"]
    assert saved["enrollment_source"] == "manual"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_premium_user_bypasses_access_check_and_lazily_creates_enrollment():
    db = FakeDataService(
        enrollments=[],
        user={
            "user_id": USER_ID,
            "email": "u@example.com",
            "premium_subscription": _active_premium_subscription(),
        },
    )

    result = await endpoints._ensure_active_course_access(
        data_service=db, user_id=USER_ID, course_id=COURSE_ID
    )

    assert result["enrollment"]["enrollment_source"] == "premium"
    assert db.saved_enrollment is not None
    assert db.saved_enrollment["course_id"] == COURSE_ID


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lapsed_premium_user_with_stale_enrollment_is_denied():
    db = FakeDataService(
        enrollments=[
            {
                "enrollment_id": "enr-stale-premium",
                "user_id": USER_ID,
                "course_id": COURSE_ID,
                "status": "active",
                "enrollment_source": "premium",
            }
        ],
        user={
            "user_id": USER_ID,
            "email": "u@example.com",
            "premium_subscription": _expired_premium_subscription(),
        },
    )

    with pytest.raises(HTTPException) as exc:
        await endpoints._ensure_active_course_access(
            data_service=db, user_id=USER_ID, course_id=COURSE_ID
        )

    assert exc.value.status_code == 403
    assert "PREMIUM_REQUIRED" in str(exc.value.detail)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_free_course_grants_access_to_matching_course_only():
    db = FakeDataService(
        enrollments=[
            {
                "enrollment_id": "enr-free",
                "user_id": USER_ID,
                "course_id": COURSE_ID,
                "status": "active",
                "enrollment_source": "free",
                "free_course_claimed_at": datetime.utcnow().isoformat(),
            }
        ]
    )

    result = await endpoints._ensure_active_course_access(
        data_service=db, user_id=USER_ID, course_id=COURSE_ID
    )
    assert result["enrollment"]["enrollment_id"] == "enr-free"

    with pytest.raises(HTTPException) as exc:
        await endpoints._ensure_active_course_access(
            data_service=db, user_id=USER_ID, course_id=OTHER_COURSE_ID
        )
    assert exc.value.status_code == 403
    assert "FREE_COURSE_LIMIT" in str(exc.value.detail)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_never_enrolled_user_is_denied_access():
    db = FakeDataService(enrollments=[])

    with pytest.raises(HTTPException) as exc:
        await endpoints._ensure_active_course_access(
            data_service=db, user_id=USER_ID, course_id=COURSE_ID
        )

    assert exc.value.status_code == 403
    assert "COURSE_ACCESS_DENIED" in str(exc.value.detail)
