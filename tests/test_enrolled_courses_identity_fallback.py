import asyncio
from unittest.mock import AsyncMock

import pytest
from postgrest.exceptions import APIError

from app.services.data_service import SupabaseDataService


def _make_service():
    SupabaseDataService._billing_email_column_missing = None
    return object.__new__(SupabaseDataService)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_enrolled_courses_loads_course_details_concurrently():
    service = _make_service()
    active_requests = 0
    max_active_requests = 0
    enrollments = {
        f"course-{index}": {
            "enrollment_id": f"enrollment-{index}",
            "user_id": "student-1",
            "course_id": f"course-{index}",
            "status": "active",
        }
        for index in range(3)
    }

    async def _resolve_enrollments(user_id, limit=50):
        return [user_id], enrollments

    async def _get_course(course_id):
        nonlocal active_requests, max_active_requests
        active_requests += 1
        max_active_requests = max(max_active_requests, active_requests)
        await asyncio.sleep(0)
        active_requests -= 1
        return {"course_id": course_id, "name": course_id}

    service._resolve_enrollment_identity_candidates = AsyncMock(
        side_effect=_resolve_enrollments
    )
    service.get_course = AsyncMock(side_effect=_get_course)

    courses = await SupabaseDataService.get_enrolled_courses_for_user(
        service, "student-1"
    )

    assert len(courses) == 3
    assert max_active_requests == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_enrolled_courses_prefers_canonical_user_and_keeps_enrollment_fields():
    service = _make_service()

    canonical_user_id = "user-uuid-1"
    username_alias = "student_demo"
    student_id_alias = "S001"
    course_id = "course-1"

    enrollment = {
        "enrollment_id": "enr-1",
        "user_id": canonical_user_id,
        "course_id": course_id,
        "status": "active",
        "enrolled_at": "2026-05-01T10:00:00",
        "started_at": "2026-05-01T10:00:00",
        "expires_at": "2026-06-01T10:00:00",
        "duration_months": 1,
        "enrollment_source": "payment",
        "enrollment_type": "standard",
        "payment_provider": "stripe",
        "payment_type": "promptpay",
        "payment_intent_id": "pi_1",
        "payment_status": "succeeded",
        "paid_amount_thb": 1290.0,
        "paid_currency": "THB",
        "billing_email": "student@example.com",
        "plan_label": "1 month",
        "paid_at": "2026-05-01T10:01:00",
        "payment_history": [{"payment_intent_id": "pi_1"}],
        "trial_consumed_at": None,
        "trial_expires_at": None,
        "progress": 12,
        "completed_quizzes": 3,
        "total_quizzes": 10,
        "completed_questions": 40,
        "total_questions": 100,
        "last_activity": "completed lesson quiz 1",
    }

    seen_lookup_ids = []

    async def _get_user(user_id):
        if user_id != canonical_user_id:
            return None
        return {
            "user_id": canonical_user_id,
            "username": username_alias,
            "student_id": student_id_alias,
        }

    async def _get_user_enrollments(user_id, limit=50):
        seen_lookup_ids.append(user_id)
        if user_id == canonical_user_id:
            return [enrollment]
        return []

    async def _get_course(requested_course_id):
        if requested_course_id != course_id:
            return None
        return {
            "course_id": course_id,
            "name": "Sample Course",
            "progress": 999,  # should be overridden by enrollment progress
        }

    service.get_user = AsyncMock(side_effect=_get_user)
    service.get_user_enrollments = AsyncMock(side_effect=_get_user_enrollments)
    service.get_course = AsyncMock(side_effect=_get_course)

    courses = await SupabaseDataService.get_enrolled_courses_for_user(
        service, canonical_user_id
    )

    assert seen_lookup_ids == [canonical_user_id, username_alias, student_id_alias]
    assert len(courses) == 1
    row = courses[0]
    assert row["course_id"] == course_id
    assert row["enrollment_id"] == "enr-1"
    assert row["enrollment_status"] == "active"
    assert row["started_at"] == enrollment["started_at"]
    assert row["expires_at"] == enrollment["expires_at"]
    assert row["duration_months"] == 1
    assert row["enrollment_source"] == "payment"
    assert row["enrollment_type"] == "standard"
    assert row["payment_intent_id"] == "pi_1"
    assert row["paid_amount_thb"] == 1290.0
    assert row["plan_label"] == "1 month"
    assert row["progress"] == 12
    assert row["completed_questions"] == 40
    assert row["last_activity"] == "completed lesson quiz 1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_enrolled_courses_falls_back_to_username_alias():
    service = _make_service()

    canonical_user_id = "user-uuid-2"
    username_alias = "legacy_username"
    course_id = "course-2"

    async def _get_user(user_id):
        if user_id != canonical_user_id:
            return None
        return {
            "user_id": canonical_user_id,
            "username": username_alias,
            "student_id": None,
        }

    async def _get_user_enrollments(user_id, limit=50):
        if user_id == canonical_user_id:
            return []
        if user_id == username_alias:
            return [
                {
                    "enrollment_id": "enr-legacy",
                    "user_id": username_alias,
                    "course_id": course_id,
                    "status": "active",
                    "enrolled_at": "2026-04-01T10:00:00",
                }
            ]
        return []

    async def _get_course(requested_course_id):
        if requested_course_id != course_id:
            return None
        return {"course_id": course_id, "name": "Legacy Course"}

    service.get_user = AsyncMock(side_effect=_get_user)
    service.get_user_enrollments = AsyncMock(side_effect=_get_user_enrollments)
    service.get_course = AsyncMock(side_effect=_get_course)

    courses = await SupabaseDataService.get_enrolled_courses_for_user(
        service, canonical_user_id
    )

    assert len(courses) == 1
    assert courses[0]["enrollment_id"] == "enr-legacy"
    assert courses[0]["course_id"] == course_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_enrolled_courses_falls_back_to_billing_email_alias():
    service = _make_service()

    canonical_user_id = "supabase-user-1"
    legacy_user_id = "09ba954c-10b1-7039-20ab-a3e50a9d77fd"
    course_id = "course-email"

    async def _get_user(user_id):
        if user_id != canonical_user_id:
            return None
        return {
            "user_id": canonical_user_id,
            "username": "student@example.com",
            "student_id": None,
            "email": "Student@Example.com",
        }

    async def _filter(table, where, include_deleted=False, limit=50):
        assert table == "enrollments"
        key, value = where
        assert key == "billing_email"
        assert value == "student@example.com"
        return [{"enrollment_id": "enr-email", "user_id": legacy_user_id}]

    async def _get_user_enrollments(user_id, limit=50):
        if user_id == legacy_user_id:
            return [
                {
                    "enrollment_id": "enr-email",
                    "user_id": legacy_user_id,
                    "course_id": course_id,
                    "status": "active",
                    "billing_email": "student@example.com",
                    "enrolled_at": "2026-05-01T10:00:00",
                }
            ]
        return []

    async def _get_course(requested_course_id):
        if requested_course_id != course_id:
            return None
        return {"course_id": course_id, "name": "Email Linked Course"}

    service.get_user = AsyncMock(side_effect=_get_user)
    service._filter = AsyncMock(side_effect=_filter)
    service.get_user_enrollments = AsyncMock(side_effect=_get_user_enrollments)
    service.get_course = AsyncMock(side_effect=_get_course)

    courses = await SupabaseDataService.get_enrolled_courses_for_user(
        service, canonical_user_id
    )

    assert len(courses) == 1
    assert courses[0]["enrollment_id"] == "enr-email"
    assert courses[0]["course_id"] == course_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_student_onboarding_treats_enrolled_legacy_user_as_complete():
    service = _make_service()

    canonical_user_id = "supabase-user-onboarding"
    username_alias = "legacy_student"
    course_id = "course-onboarding"

    user = {
        "user_id": canonical_user_id,
        "username": username_alias,
        "student_id": None,
        "email": "student@example.com",
        "onboarding_completed": False,
        "onboarding_profile": None,
    }

    async def _get_user(user_id):
        if user_id == canonical_user_id:
            return user
        return None

    async def _get_user_enrollments(user_id, limit=50):
        if user_id == canonical_user_id:
            return [
                {
                    "enrollment_id": "enr-onboarding",
                    "user_id": canonical_user_id,
                    "course_id": course_id,
                    "status": "active",
                    "enrolled_at": "2026-05-01T10:00:00",
                }
            ]
        return []

    service.find_user_by_identity = AsyncMock(return_value=user)
    service.get_user = AsyncMock(side_effect=_get_user)
    service.get_enrollment_user_ids_by_billing_email = AsyncMock(return_value=[])
    service._filter = AsyncMock(return_value=[])
    service.get_user_enrollments = AsyncMock(side_effect=_get_user_enrollments)

    result = await SupabaseDataService.get_student_onboarding(
        service,
        canonical_user_id,
        email="student@example.com",
        username=username_alias,
    )

    assert result["onboarding_completed"] is True
    assert result["onboarding_profile"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_enrolled_courses_does_not_mix_email_alias_when_primary_exists():
    service = _make_service()

    canonical_user_id = "supabase-user-primary"
    username_alias = "student_primary"
    billing_alias_user_id = "legacy-email-user"
    primary_course_id = "course-primary"
    alias_course_id = "course-alias"

    async def _get_user(user_id):
        if user_id != canonical_user_id:
            return None
        return {
            "user_id": canonical_user_id,
            "username": username_alias,
            "student_id": None,
            "email": "student@example.com",
        }

    async def _get_user_enrollments(user_id, limit=50):
        if user_id == canonical_user_id:
            return [
                {
                    "enrollment_id": "enr-primary",
                    "user_id": canonical_user_id,
                    "course_id": primary_course_id,
                    "status": "active",
                    "enrolled_at": "2026-05-01T10:00:00",
                }
            ]
        if user_id == username_alias:
            return []
        if user_id == billing_alias_user_id:
            return [
                {
                    "enrollment_id": "enr-alias",
                    "user_id": billing_alias_user_id,
                    "course_id": alias_course_id,
                    "status": "active",
                    "enrolled_at": "2026-05-02T10:00:00",
                }
            ]
        return []

    async def _get_course(requested_course_id):
        if requested_course_id == primary_course_id:
            return {"course_id": primary_course_id, "name": "Primary Course"}
        if requested_course_id == alias_course_id:
            return {"course_id": alias_course_id, "name": "Alias Course"}
        return None

    service.get_user = AsyncMock(side_effect=_get_user)
    service.get_user_enrollments = AsyncMock(side_effect=_get_user_enrollments)
    service.get_course = AsyncMock(side_effect=_get_course)
    service.get_enrollment_user_ids_by_billing_email = AsyncMock(
        return_value=[billing_alias_user_id]
    )

    courses = await SupabaseDataService.get_enrolled_courses_for_user(
        service, canonical_user_id
    )

    assert len(courses) == 1
    assert courses[0]["enrollment_id"] == "enr-primary"
    assert courses[0]["course_id"] == primary_course_id
    # We should not reach billing-email fallback once primary identity has enrollments.
    service.get_enrollment_user_ids_by_billing_email.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_enrollment_user_ids_by_billing_email_falls_back_when_column_missing():
    service = _make_service()

    async def _filter(table, where, include_deleted=False, limit=50):
        key, value = where
        if table == "enrollments" and key == "billing_email":
            raise APIError(
                {
                    "message": "column enrollments.billing_email does not exist",
                    "code": "42703",
                    "hint": None,
                    "details": None,
                }
            )
        if table == "profiles" and key == "email" and value == "student@example.com":
            return [
                {
                    "user_id": "legacy-profile-user",
                    "username": "google_113576115499500102728",
                    "student_id": None,
                }
            ]
        if table == "enrollments" and key == "user_id":
            if value == "google_113576115499500102728":
                return [
                    {
                        "enrollment_id": "enr-fallback",
                        "user_id": "google_113576115499500102728",
                    }
                ]
            return []
        return []

    service._filter = AsyncMock(side_effect=_filter)

    user_ids = await SupabaseDataService.get_enrollment_user_ids_by_billing_email(
        service, "student@example.com"
    )

    assert user_ids == ["google_113576115499500102728"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_enrollment_user_ids_by_billing_email_skips_missing_column_query_after_first_failure():
    service = _make_service()
    call_count = 0

    async def _filter(table, where, include_deleted=False, limit=50):
        nonlocal call_count
        key, value = where
        if table == "enrollments" and key == "billing_email":
            call_count += 1
            raise APIError(
                {
                    "message": "column enrollments.billing_email does not exist",
                    "code": "42703",
                    "hint": None,
                    "details": None,
                }
            )
        if table == "profiles" and key == "email" and value == "student@example.com":
            return [{"user_id": "legacy-profile-user"}]
        if table == "enrollments" and key == "user_id":
            return [{"enrollment_id": "enr-fallback", "user_id": value}]
        return []

    service._filter = AsyncMock(side_effect=_filter)

    first = await SupabaseDataService.get_enrollment_user_ids_by_billing_email(
        service, "student@example.com"
    )
    second = await SupabaseDataService.get_enrollment_user_ids_by_billing_email(
        service, "student@example.com"
    )

    assert first == ["legacy-profile-user"]
    assert second == ["legacy-profile-user"]
    assert call_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_enrolled_courses_falls_back_when_billing_email_column_missing():
    service = _make_service()

    canonical_user_id = "supabase-user-column-missing"
    legacy_user_id = "google_113576115499500102728"
    course_id = "course-column-missing"

    async def _get_user(user_id):
        if user_id != canonical_user_id:
            return None
        return {
            "user_id": canonical_user_id,
            "username": "student@example.com",
            "student_id": None,
            "email": "student@example.com",
        }

    async def _filter(table, where, include_deleted=False, limit=50):
        key, value = where
        if table == "enrollments" and key == "billing_email":
            raise APIError(
                {
                    "message": "column enrollments.billing_email does not exist",
                    "code": "42703",
                    "hint": None,
                    "details": None,
                }
            )
        if table == "profiles" and key == "email" and value == "student@example.com":
            return [
                {
                    "user_id": legacy_user_id,
                    "email": "student@example.com",
                    "username": legacy_user_id,
                }
            ]
        if table == "enrollments" and key == "user_id":
            if value == legacy_user_id:
                return [{"enrollment_id": "enr-fallback", "user_id": legacy_user_id}]
            return []
        return []

    async def _get_user_enrollments(user_id, limit=50):
        if user_id == legacy_user_id:
            return [
                {
                    "enrollment_id": "enr-fallback",
                    "user_id": legacy_user_id,
                    "course_id": course_id,
                    "status": "active",
                    "enrolled_at": "2026-05-02T10:00:00",
                }
            ]
        return []

    async def _get_course(requested_course_id):
        if requested_course_id != course_id:
            return None
        return {"course_id": course_id, "name": "Recovered by Profile Email"}

    service.get_user = AsyncMock(side_effect=_get_user)
    service._filter = AsyncMock(side_effect=_filter)
    service.get_user_enrollments = AsyncMock(side_effect=_get_user_enrollments)
    service.get_course = AsyncMock(side_effect=_get_course)

    courses = await SupabaseDataService.get_enrolled_courses_for_user(
        service, canonical_user_id
    )

    assert len(courses) == 1
    assert courses[0]["enrollment_id"] == "enr-fallback"
    assert courses[0]["course_id"] == course_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_student_onboarding_completed_canonical_profile_skips_legacy_fallback():
    service = _make_service()
    user_id = "supabase-user-fast-path"
    profile = {
        "user_id": user_id,
        "email": "student@example.com",
        "username": "student@example.com",
        "onboarding_completed": True,
        "onboarding_profile": {"nickname": "Mali"},
    }

    service.get_user = AsyncMock(return_value=profile)
    service._filter = AsyncMock(
        side_effect=AssertionError("completed canonical profile must not use fallback")
    )

    onboarding = await SupabaseDataService.get_student_onboarding(
        service,
        user_id=user_id,
        email=profile["email"],
        username=profile["username"],
    )

    assert onboarding == {
        "onboarding_completed": True,
        "onboarding_profile": {"nickname": "Mali"},
    }
    service.get_user.assert_awaited_once_with(user_id)
    service._filter.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_student_onboarding_is_complete_for_existing_billing_email_user():
    service = _make_service()
    user_id = "supabase-user-2"

    async def _get_user(requested_user_id):
        if requested_user_id != user_id:
            return None
        return {
            "user_id": user_id,
            "email": "student@example.com",
            "username": "student@example.com",
            "role": "student",
        }

    async def _filter(table, where, include_deleted=False, limit=50):
        if table == "enrollments":
            return [{"enrollment_id": "enr-existing", "user_id": "legacy-user"}]
        return []

    service.get_user = AsyncMock(side_effect=_get_user)
    service._filter = AsyncMock(side_effect=_filter)

    onboarding = await SupabaseDataService.get_student_onboarding(
        service,
        user_id=user_id,
        email="student@example.com",
        username="student@example.com",
    )

    assert onboarding == {
        "onboarding_completed": True,
        "onboarding_profile": None,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_student_onboarding_stays_complete_when_billing_email_column_missing():
    service = _make_service()
    user_id = "supabase-user-column-missing-onboarding"
    legacy_user_id = "google_113576115499500102728"

    async def _get_user(requested_user_id):
        if requested_user_id != user_id:
            return None
        return {
            "user_id": user_id,
            "email": "student@example.com",
            "username": "student@example.com",
            "role": "student",
            "onboarding_completed": False,
            "onboarding_profile": None,
        }

    async def _filter(table, where, include_deleted=False, limit=50):
        key, value = where
        if table == "enrollments" and key == "billing_email":
            raise APIError(
                {
                    "message": "column enrollments.billing_email does not exist",
                    "code": "42703",
                    "hint": None,
                    "details": None,
                }
            )
        if table == "profiles" and key == "email" and value == "student@example.com":
            return [
                {
                    "user_id": legacy_user_id,
                    "email": "student@example.com",
                    "username": legacy_user_id,
                    "role": "student",
                }
            ]
        if table == "enrollments" and key == "user_id":
            if value == legacy_user_id:
                return [
                    {
                        "enrollment_id": "enr-existing",
                        "user_id": legacy_user_id,
                        "status": "active",
                    }
                ]
            return []
        return []

    service.get_user = AsyncMock(side_effect=_get_user)
    service._filter = AsyncMock(side_effect=_filter)

    onboarding = await SupabaseDataService.get_student_onboarding(
        service,
        user_id=user_id,
        email="student@example.com",
        username="student@example.com",
    )

    assert onboarding == {
        "onboarding_completed": True,
        "onboarding_profile": None,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_student_onboarding_copies_completed_legacy_student_id_profile():
    service = _make_service()
    canonical_user_id = "supabase-user-3"
    legacy_user_id = "google_113576115499500102728"
    legacy_profile = {
        "nickname": "น้ำผึ้ง",
        "grade_level": "ประถมปลาย",
        "age": 15,
        "interested_subjects": ["คณิตศาสตร์"],
        "primary_goal": "learn_ahead",
    }
    saved = []

    async def _get_user(requested_user_id):
        if requested_user_id == canonical_user_id:
            return {
                "user_id": canonical_user_id,
                "email": "student@example.com",
                "username": "student@example.com",
                "student_id": legacy_user_id,
                "role": "student",
            }
        if requested_user_id == legacy_user_id:
            return {
                "user_id": legacy_user_id,
                "onboarding_completed": True,
                "onboarding_profile": legacy_profile,
                "email": f"{legacy_user_id}@example.com",
                "name": legacy_user_id,
            }
        return None

    async def _save_student_onboarding(user_id, onboarding_profile, base_user=None):
        saved.append(
            {
                "user_id": user_id,
                "onboarding_profile": onboarding_profile,
                "base_user": base_user,
            }
        )
        return {
            "user_id": user_id,
            "onboarding_completed": True,
            "onboarding_profile": onboarding_profile,
        }

    async def _filter(table, where, include_deleted=False, limit=50):
        return []

    service.get_user = AsyncMock(side_effect=_get_user)
    service._filter = AsyncMock(side_effect=_filter)
    service.save_student_onboarding = AsyncMock(side_effect=_save_student_onboarding)

    onboarding = await SupabaseDataService.get_student_onboarding(
        service,
        user_id=canonical_user_id,
        email="student@example.com",
        username="student@example.com",
        student_id=legacy_user_id,
    )

    assert onboarding == {
        "onboarding_completed": True,
        "onboarding_profile": legacy_profile,
    }
    assert saved[0]["user_id"] == canonical_user_id
    assert saved[0]["onboarding_profile"] == legacy_profile


@pytest.mark.unit
@pytest.mark.asyncio
async def test_student_onboarding_copies_completed_profile_from_matching_email():
    service = _make_service()
    canonical_user_id = "supabase-user-4"
    legacy_user_id = "google_999111222333444555666"
    legacy_profile = {
        "nickname": "หนูนา",
        "grade_level": "ม.ต้น",
        "age": 14,
        "interested_subjects": ["วิทยาศาสตร์"],
        "primary_goal": "exam_preparation",
    }
    saved = []

    async def _get_user(requested_user_id):
        if requested_user_id == canonical_user_id:
            return {
                "user_id": canonical_user_id,
                "email": "student@example.com",
                "username": "student@example.com",
                "role": "student",
                "onboarding_completed": False,
            }
        return None

    async def _filter(table, where, include_deleted=False, limit=50):
        key, value = where
        if table == "enrollments":
            return []
        if table == "profiles" and key == "email" and value == "student@example.com":
            return [
                {
                    "user_id": legacy_user_id,
                    "email": "student@example.com",
                    "username": legacy_user_id,
                    "role": "student",
                    "onboarding_completed": True,
                    "onboarding_profile": legacy_profile,
                }
            ]
        return []

    async def _save_student_onboarding(user_id, onboarding_profile, base_user=None):
        saved.append(
            {
                "user_id": user_id,
                "onboarding_profile": onboarding_profile,
                "base_user": base_user,
            }
        )
        return {
            "user_id": user_id,
            "onboarding_completed": True,
            "onboarding_profile": onboarding_profile,
        }

    service.get_user = AsyncMock(side_effect=_get_user)
    service._filter = AsyncMock(side_effect=_filter)
    service.save_student_onboarding = AsyncMock(side_effect=_save_student_onboarding)

    onboarding = await SupabaseDataService.get_student_onboarding(
        service,
        user_id=canonical_user_id,
        email="student@example.com",
        username="student@example.com",
    )

    assert onboarding == {
        "onboarding_completed": True,
        "onboarding_profile": legacy_profile,
    }
    assert saved[0]["user_id"] == canonical_user_id
    assert saved[0]["onboarding_profile"] == legacy_profile


@pytest.mark.unit
@pytest.mark.asyncio
async def test_student_onboarding_remains_incomplete_for_true_new_user():
    service = _make_service()
    canonical_user_id = "supabase-user-new"

    async def _get_user(requested_user_id):
        return None

    async def _filter(table, where, include_deleted=False, limit=50):
        return []

    service.get_user = AsyncMock(side_effect=_get_user)
    service._filter = AsyncMock(side_effect=_filter)

    onboarding = await SupabaseDataService.get_student_onboarding(
        service,
        user_id=canonical_user_id,
        email="new_student@example.com",
        username="new_student@example.com",
    )

    assert onboarding == {
        "onboarding_completed": False,
        "onboarding_profile": None,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_enrolled_courses_deduplicates_by_course_and_prefers_canonical():
    service = _make_service()

    canonical_user_id = "user-uuid-3"
    username_alias = "legacy_u3"
    course_id = "course-3"

    canonical_enrollment = {
        "enrollment_id": "enr-canonical",
        "user_id": canonical_user_id,
        "course_id": course_id,
        "status": "active",
        "enrolled_at": "2026-05-03T10:00:00",
        "expires_at": "2026-06-03T10:00:00",
    }
    alias_enrollment = {
        "enrollment_id": "enr-alias",
        "user_id": username_alias,
        "course_id": course_id,
        "status": "active",
        "enrolled_at": "2026-05-04T10:00:00",
        "expires_at": "2026-07-04T10:00:00",
    }

    async def _get_user(user_id):
        if user_id != canonical_user_id:
            return None
        return {
            "user_id": canonical_user_id,
            "username": username_alias,
            "student_id": None,
        }

    async def _get_user_enrollments(user_id, limit=50):
        if user_id == canonical_user_id:
            return [canonical_enrollment]
        if user_id == username_alias:
            return [alias_enrollment]
        return []

    async def _get_course(requested_course_id):
        if requested_course_id != course_id:
            return None
        return {"course_id": course_id, "name": "Dedup Course"}

    service.get_user = AsyncMock(side_effect=_get_user)
    service.get_user_enrollments = AsyncMock(side_effect=_get_user_enrollments)
    service.get_course = AsyncMock(side_effect=_get_course)

    courses = await SupabaseDataService.get_enrolled_courses_for_user(
        service, canonical_user_id
    )

    assert len(courses) == 1
    # canonical enrollment wins even though alias enrollment is newer
    assert courses[0]["enrollment_id"] == "enr-canonical"
    assert courses[0]["expires_at"] == "2026-06-03T10:00:00"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_user_quiz_results_falls_back_to_username_alias():
    service = _make_service()

    canonical_user_id = "user-uuid-4"
    username_alias = "legacy_u4"
    course_id = "course-4"

    async def _get_user(user_id):
        if user_id != canonical_user_id:
            return None
        return {
            "user_id": canonical_user_id,
            "username": username_alias,
            "student_id": None,
        }

    async def _filter(table, where, include_deleted=True, limit=None):
        assert table == "quiz_results"
        key, value = where
        assert key == "user_id"
        if value == canonical_user_id:
            return []
        if value == username_alias:
            return [
                {
                    "result_id": "result-legacy",
                    "user_id": username_alias,
                    "quiz_id": "quiz-1",
                    "course_id": course_id,
                    "submitted_at": "2026-05-05T09:00:00",
                }
            ]
        return []

    service.get_user = AsyncMock(side_effect=_get_user)
    service._filter = AsyncMock(side_effect=_filter)

    results = await SupabaseDataService.get_user_quiz_results(
        service, canonical_user_id
    )

    assert len(results) == 1
    assert results[0]["result_id"] == "result-legacy"
    assert results[0]["course_id"] == course_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_user_quiz_results_course_filter_includes_legacy_rows_by_quiz_id():
    service = _make_service()

    user_id = "user-uuid-legacy-course"
    course_id = "course-legacy"
    matching_quiz_id = "quiz-in-course"

    async def _get_user(requested_user_id):
        if requested_user_id != user_id:
            return None
        return {
            "user_id": user_id,
            "username": None,
            "student_id": None,
        }

    async def _filter(table, where, include_deleted=True, limit=None):
        assert table == "quiz_results"
        key, value = where
        assert key == "user_id"
        if value != user_id:
            return []
        return [
            {
                "result_id": "result-missing-course",
                "user_id": user_id,
                "quiz_id": matching_quiz_id,
                "course_id": "",
                "submitted_at": "2026-05-08T09:00:00",
            },
            {
                "result_id": "result-default-course",
                "user_id": user_id,
                "quiz_id": matching_quiz_id,
                "course_id": "default-course",
                "submitted_at": "2026-05-08T10:00:00",
            },
            {
                "result_id": "result-other-quiz",
                "user_id": user_id,
                "quiz_id": "quiz-other-course",
                "course_id": "",
                "submitted_at": "2026-05-08T11:00:00",
            },
        ]

    service.get_user = AsyncMock(side_effect=_get_user)
    service._filter = AsyncMock(side_effect=_filter)
    service.get_quizzes_by_course = AsyncMock(
        return_value=[
            {
                "quiz_id": matching_quiz_id,
                "course_id": course_id,
            }
        ]
    )

    results = await SupabaseDataService.get_user_quiz_results(
        service, user_id, course_id=course_id
    )

    assert [row["result_id"] for row in results] == [
        "result-default-course",
        "result-missing-course",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_user_quiz_results_course_filter_keeps_explicit_course_match_only():
    service = _make_service()

    user_id = "user-uuid-explicit-course"
    course_id = "course-target"

    async def _get_user(requested_user_id):
        if requested_user_id != user_id:
            return None
        return {
            "user_id": user_id,
            "username": None,
            "student_id": None,
        }

    async def _filter(table, where, include_deleted=True, limit=None):
        assert table == "quiz_results"
        key, value = where
        assert key == "user_id"
        if value != user_id:
            return []
        return [
            {
                "result_id": "result-target",
                "user_id": user_id,
                "quiz_id": "quiz-target",
                "course_id": course_id,
                "submitted_at": "2026-05-08T09:00:00",
            },
            {
                "result_id": "result-other-course",
                "user_id": user_id,
                "quiz_id": "quiz-target",
                "course_id": "course-other",
                "submitted_at": "2026-05-08T10:00:00",
            },
        ]

    service.get_user = AsyncMock(side_effect=_get_user)
    service._filter = AsyncMock(side_effect=_filter)
    service.get_quizzes_by_course = AsyncMock(
        return_value=[
            {
                "quiz_id": "quiz-target",
                "course_id": course_id,
            }
        ]
    )

    results = await SupabaseDataService.get_user_quiz_results(
        service, user_id, course_id=course_id
    )

    assert [row["result_id"] for row in results] == ["result-target"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_user_quiz_results_deduplicates_result_id_across_aliases():
    service = _make_service()

    canonical_user_id = "user-uuid-5"
    username_alias = "legacy_u5"

    async def _get_user(user_id):
        if user_id != canonical_user_id:
            return None
        return {
            "user_id": canonical_user_id,
            "username": username_alias,
            "student_id": None,
        }

    canonical_row = {
        "result_id": "result-shared",
        "user_id": canonical_user_id,
        "quiz_id": "quiz-2",
        "course_id": "course-5",
        "submitted_at": "2026-05-07T12:00:00",
    }
    alias_row = {
        "result_id": "result-shared",
        "user_id": username_alias,
        "quiz_id": "quiz-2",
        "course_id": "course-5",
        "submitted_at": "2026-05-07T12:00:00",
    }

    async def _filter(table, where, include_deleted=True, limit=None):
        assert table == "quiz_results"
        key, value = where
        assert key == "user_id"
        if value == canonical_user_id:
            return [canonical_row]
        if value == username_alias:
            return [alias_row]
        return []

    service.get_user = AsyncMock(side_effect=_get_user)
    service._filter = AsyncMock(side_effect=_filter)

    results = await SupabaseDataService.get_user_quiz_results(
        service, canonical_user_id
    )

    assert len(results) == 1
    assert results[0]["result_id"] == "result-shared"
    assert results[0]["user_id"] == canonical_user_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_collect_preferred_enrollments_batches_supabase_alias_lookup():
    service = _make_service()
    service.supabase = object()
    service._query = AsyncMock(
        return_value={
            "rows": [
                {
                    "enrollment_id": "enr-1",
                    "user_id": "student-1",
                    "course_id": "course-1",
                    "status": "active",
                },
                {
                    "enrollment_id": "enr-2",
                    "user_id": "legacy-student-1",
                    "course_id": "course-2",
                    "status": "active",
                },
            ]
        }
    )
    service.get_user_enrollments = AsyncMock(
        side_effect=AssertionError("Supabase path should batch enrollment lookup")
    )

    preferred = await SupabaseDataService._collect_preferred_enrollments_by_course(
        service,
        ["student-1", "legacy-student-1"],
        limit=50,
    )

    assert set(preferred) == {"course-1", "course-2"}
    service._query.assert_awaited_once()
    _, kwargs = service._query.await_args
    assert kwargs["in_filters"] == {"user_id": ["student-1", "legacy-student-1"]}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_user_quiz_results_batches_supabase_alias_lookup():
    service = _make_service()
    service.supabase = object()
    service.get_user = AsyncMock(
        return_value={
            "user_id": "student-1",
            "username": "legacy-student-1",
            "student_id": None,
        }
    )
    service._query = AsyncMock(
        return_value={
            "rows": [
                {
                    "result_id": "result-1",
                    "user_id": "student-1",
                    "quiz_id": "quiz-1",
                    "submitted_at": "2026-05-01T10:00:00",
                }
            ]
        }
    )

    results = await SupabaseDataService.get_user_quiz_results(service, "student-1")

    assert [row["result_id"] for row in results] == ["result-1"]
    service._query.assert_awaited_once()
    _, kwargs = service._query.await_args
    assert kwargs["in_filters"] == {"user_id": ["student-1", "legacy-student-1"]}
