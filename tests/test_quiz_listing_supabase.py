from unittest.mock import AsyncMock

import pytest

from app.services.data_service import SupabaseDataService


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_user_quizzes_keeps_legacy_null_status_and_filters_deleted():
    service = object.__new__(SupabaseDataService)
    service._query = AsyncMock(
        return_value={
            "rows": [
                {"quiz_id": "active", "status": "active", "created_at": "2026-01-03"},
                {"quiz_id": "legacy", "status": None, "created_at": "2026-01-02"},
                {"quiz_id": "deleted", "status": "deleted", "created_at": "2026-01-01"},
            ]
        }
    )

    rows = await SupabaseDataService.get_user_quizzes(
        service, "user-1", course_id="course-1"
    )

    assert [row["quiz_id"] for row in rows] == ["active", "legacy"]
    service._query.assert_awaited_once_with(
        "quizzes",
        eq={"user_id": "user-1", "course_id": "course-1"},
        order_by="created_at",
        desc=True,
        limit=5000,
    )
