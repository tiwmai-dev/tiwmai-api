"""Query-count regression tests for optimized Supabase access paths."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.data_service import SupabaseDataService


@pytest.fixture
def data_service():
    service = SupabaseDataService.__new__(SupabaseDataService)
    service._read_cache = {}
    service._normalize_identity = SupabaseDataService._normalize_identity
    service._append_unique_identity = SupabaseDataService._append_unique_identity
    service._is_uuid_like = SupabaseDataService._is_uuid_like
    service._cache_get = lambda key: None
    service._cache_set = lambda key, value, ttl=45: value
    service.find_user_by_identity = AsyncMock(return_value=None)
    service._resolve_enrollment_identity_candidates = AsyncMock(
        return_value=(
            ["student-1"],
            {"course-1": {"enrollment_id": "en-1", "course_id": "course-1"}},
        )
    )
    service.get_user_quiz_results = AsyncMock(return_value=[])
    service.get_course_lessons = AsyncMock(return_value=[])
    service.get_quizzes_by_course = AsyncMock(return_value=[])
    service._query = AsyncMock(
        return_value={"rows": [{"course_id": "course-1", "name": "Math"}]}
    )
    service._filter_in = AsyncMock(return_value=[])
    service.supabase = MagicMock()
    return service


@pytest.mark.asyncio
async def test_get_user_courses_uses_two_batch_queries(data_service):
    data_service._filter_in = AsyncMock(
        side_effect=[
            [{"course_id": "course-1", "user_id": "tutor-1"}],
            [],
        ]
    )

    courses = await data_service.get_user_courses("tutor-1", summary=True)

    assert len(courses) == 1
    assert data_service._filter_in.await_count == 2


@pytest.mark.asyncio
async def test_get_course_learning_overview_uses_direct_enrollment_lookup(data_service):
    data_service.get_user_enrollment_for_course = AsyncMock(
        return_value={"enrollment_id": "en-1", "course_id": "course-1"}
    )

    overview = await data_service.get_course_learning_overview(
        "course-1", user_id="student-1"
    )

    assert overview["course_id"] == "course-1"
    data_service.get_user_enrollment_for_course.assert_awaited_once_with(
        "student-1", "course-1"
    )


@pytest.mark.asyncio
async def test_get_dashboard_learning_inputs_batches_related_tables(data_service):
    data_service._filter_in = AsyncMock(
        side_effect=[
            [{"course_id": "course-1"}],
            [{"lesson_id": "lesson-1", "course_id": "course-1"}],
        ]
    )
    data_service.get_quizzes_for_courses = AsyncMock(
        return_value=[{"quiz_id": "quiz-1", "course_id": "course-1"}]
    )
    data_service._query = AsyncMock(
        return_value={
            "rows": [
                {
                    "result_id": "result-1",
                    "course_id": "course-1",
                    "user_id": "student-1",
                }
            ]
        }
    )
    data_service.supabase = MagicMock()

    inputs = await data_service.get_dashboard_learning_inputs("student-1", limit=10)

    assert inputs["candidate_user_ids"] == ["student-1"]
    assert data_service._filter_in.await_count == 2
    assert data_service.get_quizzes_for_courses.await_count == 1
    assert data_service._query.await_count == 1
