import pytest

from app.api import student_handlers as endpoints


class FakeQuizSortingService:
    async def get_quizzes_by_course(self, course_id):
        return [
            {
                "quiz_id": "easy-5",
                "course_id": course_id,
                "title": "แบบทดสอบ Vocabulary - ชุดที่ 5",
                "difficulty_avg": 2,
                "created_at": "2026-05-05T00:00:00",
            },
            {
                "quiz_id": "hard-1",
                "course_id": course_id,
                "title": "แบบทดสอบ Vocabulary - ชุดที่ 1",
                "difficulty_avg": 4,
                "created_at": "2026-05-01T00:00:00",
            },
            {
                "quiz_id": "easy-10",
                "course_id": course_id,
                "title": "แบบทดสอบ Vocabulary - ชุดที่ 10",
                "difficulty_avg": 2,
                "created_at": "2026-05-10T00:00:00",
            },
            {
                "quiz_id": "easy-1",
                "course_id": course_id,
                "title": "แบบทดสอบ Vocabulary - ชุดที่ 1",
                "difficulty_avg": 2,
                "created_at": "2026-05-01T00:00:00",
            },
        ]


class FakePaginatedQuizService(FakeQuizSortingService):
    def __init__(self):
        self.called = False

    async def get_course_quizzes_page(
        self,
        course_id,
        *,
        page=1,
        page_size=20,
        q=None,
        sort="latest",
        quiz_ids=None,
        summary=False,
    ):
        self.called = True
        assert course_id == "course-1"
        assert page == 2
        assert page_size == 2
        assert q == "vocab"
        assert sort == "latest"
        assert quiz_ids == ["quiz-1", "quiz-2"]
        assert summary is False
        return {
            "rows": [
                {"quiz_id": "quiz-3", "course_id": course_id},
                {"quiz_id": "quiz-4", "course_id": course_id},
            ],
            "total": 5,
            "page": 2,
            "page_size": 2,
            "total_pages": 3,
        }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_course_quizzes_difficulty_sort_uses_natural_title_tiebreaker():
    response = await endpoints.list_course_quizzes(
        course_id="course-1",
        sort="difficulty_asc",
        data_service=FakeQuizSortingService(),
    )

    assert [quiz["quiz_id"] for quiz in response["quizzes"]] == [
        "easy-1",
        "easy-5",
        "easy-10",
        "hard-1",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_course_quizzes_uses_db_pagination_for_latest_listing():
    service = FakePaginatedQuizService()

    response = await endpoints.list_course_quizzes(
        course_id="course-1",
        q="vocab",
        sort="latest",
        page=2,
        page_size=2,
        quiz_ids="quiz-1,quiz-2",
        data_service=service,
    )

    assert service.called
    assert response["total_filtered"] == 5
    assert response["total_pages"] == 3
    assert response["has_next"] is True
    assert response["has_prev"] is True
    assert [quiz["quiz_id"] for quiz in response["quizzes"]] == ["quiz-3", "quiz-4"]
