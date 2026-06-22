import os

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret")

from app.services.data_service import SupabaseDataService


class _MemorySupabaseDataService(SupabaseDataService):
    def __init__(self):
        self.user = {"user_id": "user-1"}

    async def get_user(self, user_id):
        return self.user if user_id == self.user["user_id"] else None

    async def _upsert(self, table_name, item):
        assert table_name == "profiles"
        self.user = item
        return item


@pytest.mark.asyncio
async def test_student_analysis_summary_cache_supports_payload_key():
    service = _MemorySupabaseDataService()

    await service.set_student_analysis_summary_cache(
        user_id="user-1",
        course_id="course-1",
        cache_key="payload-a",
        score_version="score-v1",
        response={"summary_paragraph": "first"},
    )

    cached = await service.get_student_analysis_summary_cache(
        user_id="user-1",
        course_id="course-1",
        cache_key="payload-a",
        score_version="score-v1",
    )

    assert cached["summary_paragraph"] == "first"
    assert cached["cache_key"] == "payload-a"
    assert cached["score_version"] == "score-v1"


@pytest.mark.asyncio
async def test_student_analysis_summary_cache_misses_different_payload_key():
    service = _MemorySupabaseDataService()

    await service.set_student_analysis_summary_cache(
        user_id="user-1",
        course_id="course-1",
        cache_key="payload-a",
        score_version="score-v1",
        response={"summary_paragraph": "first"},
    )

    cached = await service.get_student_analysis_summary_cache(
        user_id="user-1",
        course_id="course-1",
        cache_key="payload-b",
        score_version="score-v1",
    )

    assert cached is None


@pytest.mark.asyncio
async def test_student_analysis_summary_cache_keeps_legacy_course_cache():
    service = _MemorySupabaseDataService()

    await service.set_student_analysis_summary_cache(
        user_id="user-1",
        course_id="course-1",
        summary={"summary_paragraph": "legacy"},
    )

    cached = await service.get_student_analysis_summary_cache(
        user_id="user-1",
        course_id="course-1",
    )

    assert cached["summary_paragraph"] == "legacy"
