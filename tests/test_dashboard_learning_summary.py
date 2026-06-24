import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api import student_handlers as endpoints
from app.services.data_service import SupabaseDataService


@pytest.mark.unit
def test_dashboard_course_stats_excludes_mock_exam_from_progress_but_counts_split():
    courses = [
        {
            "course_id": "course-1",
            "progress": 0,
            "learning_activity_days": ["2026-05-06"],
        }
    ]
    lessons = [
        {
            "lesson_id": "lesson-1",
            "course_id": "course-1",
            "title": "Lesson 1",
            "order": 1,
            "quizzes": [{"id": "quiz-1"}, {"id": "quiz-mock"}],
        }
    ]
    quizzes = [
        {
            "quiz_id": "quiz-1",
            "course_id": "course-1",
            "document_type": "manual",
            "difficulty": 1,
            "topic": "พีชคณิต",
        },
        {
            "quiz_id": "quiz-mock",
            "course_id": "course-1",
            "document_type": "mock_exam",
            "difficulty": 3,
        },
    ]
    results = [
        {
            "result_id": "result-1",
            "quiz_id": "quiz-1",
            "course_id": "course-1",
            "lesson_id": "lesson-1",
            "total_questions": 10,
            "correct_count": 8,
            "question_insights": [
                {"topic": "สมการ", "is_correct": True},
                {"topic": "สมการ", "is_correct": False},
                {"topic": "กราฟ", "is_correct": True},
            ],
            "submitted_at": "2026-05-05T10:00:00",
        },
        {
            "result_id": "result-2",
            "quiz_id": "quiz-mock",
            "course_id": "course-1",
            "lesson_id": "lesson-1",
            "total_questions": 20,
            "correct_count": 10,
            "submitted_at": "2026-05-06T10:00:00",
        },
    ]

    stats = endpoints._build_dashboard_course_stats(courses, lessons, quizzes, results)

    row = stats["course-1"]
    assert row["totalQuizzes"] == 1
    assert row["completedQuizzes"] == 2
    assert row["progress"] == 100
    assert row["scoreSplit"] == {"lesson": 80, "mockExam": 50}
    assert row["difficultyScore"]["easy"] == 80
    assert row["lessonRows"][0]["scoreSplit"] == {"lesson": 80, "mockExam": 50}
    assert row["topicRows"] == [
        {
            "id": "course-1-ไม่ระบุหัวข้อ",
            "topic": "ไม่ระบุหัวข้อ",
            "total": 20,
            "correct": 10,
            "accuracy": 50,
        },
        {
            "id": "course-1-สมการ",
            "topic": "สมการ",
            "total": 2,
            "correct": 1,
            "accuracy": 50,
        },
        {
            "id": "course-1-กราฟ",
            "topic": "กราฟ",
            "total": 1,
            "correct": 1,
            "accuracy": 100,
        },
    ]
    assert row["topicRowsByLesson"] == [
        {
            "lessonId": "lesson-1",
            "lessonName": "Lesson 1",
            "lessonOrder": 1,
            "topics": [
                {
                    "id": "course-1-lesson-1-สมการ",
                    "topic": "สมการ",
                    "total": 2,
                    "correct": 1,
                    "accuracy": 50,
                },
                {
                    "id": "course-1-lesson-1-กราฟ",
                    "topic": "กราฟ",
                    "total": 1,
                    "correct": 1,
                    "accuracy": 100,
                },
            ],
        },
    ]
    assert row["learningActivityDays"] == ["2026-05-06"]


@pytest.mark.unit
def test_dashboard_course_stats_prefers_quiz_lesson_mapping_when_result_lesson_is_stale():
    courses = [{"course_id": "course-1"}]
    lessons = [
        {
            "lesson_id": "lesson-current",
            "course_id": "course-1",
            "title": "ร้อยละ และ อัตราส่วน",
            "order": 1,
            "quizzes": [{"id": "quiz-1"}],
        }
    ]
    quizzes = [
        {
            "quiz_id": "quiz-1",
            "course_id": "course-1",
            "document_type": "manual",
        }
    ]
    results = [
        {
            "result_id": "result-1",
            "quiz_id": "quiz-1",
            "course_id": "course-1",
            "lesson_id": "lesson-stale",
            "total_questions": 10,
            "correct_count": 2,
            "submitted_at": "2026-05-05T10:00:00",
        }
    ]

    stats = endpoints._build_dashboard_course_stats(courses, lessons, quizzes, results)

    assert stats["course-1"]["lessonRows"] == [
        {
            "id": "lesson-current",
            "name": "ร้อยละ และ อัตราส่วน",
            "scoreSplit": {"lesson": 20, "mockExam": None},
            "minutes": 0,
        }
    ]


@pytest.mark.unit
def test_dashboard_course_stats_does_not_create_fallback_lesson_rows():
    courses = [{"course_id": "course-1"}]
    lessons = [
        {
            "lesson_id": "lesson-current",
            "course_id": "course-1",
            "title": "สมการ",
            "order": 1,
            "quizzes": [],
        }
    ]
    quizzes = [
        {
            "quiz_id": "quiz-unmapped",
            "course_id": "course-1",
            "document_type": "manual",
        }
    ]
    results = [
        {
            "result_id": "result-1",
            "quiz_id": "quiz-unmapped",
            "course_id": "course-1",
            "lesson_id": "lesson-stale",
            "total_questions": 10,
            "correct_count": 2,
            "submitted_at": "2026-05-05T10:00:00",
        }
    ]

    stats = endpoints._build_dashboard_course_stats(courses, lessons, quizzes, results)

    assert stats["course-1"]["lessonRows"] == [
        {
            "id": "lesson-current",
            "name": "สมการ",
            "scoreSplit": {"lesson": None, "mockExam": None},
            "minutes": 0,
        }
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_record_learning_activity_updates_enrollment_day_list():
    service = object.__new__(SupabaseDataService)

    async def _resolve(user_id, limit=200):
        return [user_id], {
            "course-1": {
                "enrollment_id": "enr-1",
                "course_id": "course-1",
                "learning_activity_days": ["2026-05-05"],
            }
        }

    updated = {}

    async def _update(enrollment_id, updates):
        updated["enrollment_id"] = enrollment_id
        updated["updates"] = updates
        return True

    service._resolve_enrollment_identity_candidates = _resolve
    service.update_enrollment = _update
    service._normalize_identity = lambda value: str(value or "").strip()

    result = await SupabaseDataService.record_learning_activity(
        service,
        user_id="student-1",
        course_id="course-1",
        lesson_id="lesson-1",
        activity_day="2026-05-06",
        activity_days=["2026-05-04", "not-a-day"],
    )

    assert result["activity_day"] == "2026-05-06"
    assert result["activity_days"] == ["2026-05-04", "2026-05-05", "2026-05-06"]
    assert updated["enrollment_id"] == "enr-1"
    assert updated["updates"]["learning_activity_days"] == [
        "2026-05-04",
        "2026-05-05",
        "2026-05-06",
    ]
    assert updated["updates"]["last_lesson_id"] == "lesson-1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_record_learning_activity_uses_verified_enrollment_without_lookup():
    service = object.__new__(SupabaseDataService)

    async def _resolve(*args, **kwargs):
        raise AssertionError("verified enrollment should avoid a second lookup")

    updated = {}

    async def _update(enrollment_id, updates):
        updated["enrollment_id"] = enrollment_id
        return True

    service._resolve_enrollment_identity_candidates = _resolve
    service.update_enrollment = _update
    service._normalize_identity = lambda value: str(value or "").strip()

    result = await SupabaseDataService.record_learning_activity(
        service,
        user_id="student-1",
        course_id="course-1",
        lesson_id="lesson-1",
        activity_day="2026-05-06",
        enrollment={
            "enrollment_id": "enr-verified",
            "course_id": "course-1",
            "learning_activity_days": [],
        },
    )

    assert result["course_id"] == "course-1"
    assert updated["enrollment_id"] == "enr-verified"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dashboard_learning_summary_empty_enrollment_returns_empty_payload():
    class FakeDb:
        async def get_dashboard_learning_inputs(self, user_id, limit=50):
            return {
                "candidate_user_ids": [user_id],
                "enrollments": [],
                "courses": [],
                "lessons": [],
                "quizzes": [],
                "quiz_results": [],
            }

    response = await endpoints.get_dashboard_learning_summary(
        user_id="student-1",
        credentials=None,
        student_auth_service=None,
        data_service=FakeDb(),
    )

    assert response["user_id"] == "student-1"
    assert response["courses"] == []
    assert response["course_stats"] == {}
    assert response["generated_at"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dashboard_learning_summary_uses_alias_results_and_dedupes():
    class FakeDb:
        async def get_dashboard_learning_inputs(self, user_id, limit=50):
            return {
                "candidate_user_ids": [user_id, "legacy-user"],
                "enrollments": [
                    {
                        "enrollment_id": "enr-1",
                        "user_id": user_id,
                        "course_id": "course-1",
                        "status": "active",
                        "enrolled_at": "2026-05-01T10:00:00",
                    }
                ],
                "courses": [{"course_id": "course-1", "name": "Course 1"}],
                "lessons": [],
                "quizzes": [
                    {
                        "quiz_id": "quiz-1",
                        "course_id": "course-1",
                        "document_type": "manual",
                    }
                ],
                "quiz_results": [
                    {
                        "result_id": "shared",
                        "user_id": user_id,
                        "quiz_id": "quiz-1",
                        "course_id": "course-1",
                        "score": 80,
                        "submitted_at": "2026-05-05T10:00:00",
                    },
                    {
                        "result_id": "shared",
                        "user_id": "legacy-user",
                        "quiz_id": "quiz-1",
                        "course_id": "course-1",
                        "score": 20,
                        "submitted_at": "2026-05-05T10:00:00",
                    },
                    {
                        "result_id": "legacy-only",
                        "user_id": "legacy-user",
                        "quiz_id": "quiz-1",
                        "course_id": "course-1",
                        "score": 60,
                        "submitted_at": "2026-05-06T10:00:00",
                    },
                ],
            }

    response = await endpoints.get_dashboard_learning_summary(
        user_id="student-1",
        credentials=None,
        student_auth_service=None,
        data_service=FakeDb(),
    )

    stats = response["course_stats"]["course-1"]
    assert len(response["courses"]) == 1
    assert stats["completedQuizzes"] == 1
    assert [row["score"] for row in stats["attemptRows"]] == [80, 60]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dashboard_learning_inputs_collects_alias_rows_in_batches():
    service = object.__new__(SupabaseDataService)

    async def _get_user(user_id):
        if user_id == "student-1":
            return {"user_id": "student-1", "username": "legacy-user"}
        return None

    async def _get_user_enrollments(user_id, limit=50):
        if user_id == "legacy-user":
            return [
                {
                    "enrollment_id": "enr-legacy",
                    "user_id": "legacy-user",
                    "course_id": "course-1",
                    "status": "active",
                    "enrolled_at": "2026-05-01T10:00:00",
                }
            ]
        return []

    async def _filter_in(table, column, values, **kwargs):
        if table == "courses":
            return [{"course_id": "course-1", "name": "Course 1"}]
        if table == "quiz_results":
            assert values == ["student-1", "legacy-user"]
            return [{"result_id": "result-1", "user_id": "legacy-user"}]
        return []

    async def _get_quizzes_for_courses(course_ids, summary=True):
        assert course_ids == ["course-1"]
        return [{"quiz_id": "quiz-1", "course_id": "course-1"}]

    service.get_user = _get_user
    service.get_user_enrollments = _get_user_enrollments
    service._filter_in = _filter_in
    service.get_quizzes_for_courses = _get_quizzes_for_courses

    rows = await SupabaseDataService.get_dashboard_learning_inputs(service, "student-1")

    assert rows["candidate_user_ids"] == ["student-1", "legacy-user"]
    assert rows["enrollments"][0]["enrollment_id"] == "enr-legacy"
    assert rows["courses"][0]["course_id"] == "course-1"
    assert rows["quiz_results"][0]["user_id"] == "legacy-user"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dashboard_learning_summary_preserves_401_when_token_verification_fails():
    class FakeAuthService:
        async def verify_jwt_token(self, token):
            raise HTTPException(status_code=401, detail="Invalid access token")

    class FakeDb:
        async def get_dashboard_learning_inputs(self, user_id, limit=50):
            raise AssertionError("db access should not be reached for invalid token")

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="invalid-token"
    )

    with pytest.raises(HTTPException) as exc_info:
        await endpoints.get_dashboard_learning_summary(
            user_id="student-1",
            credentials=credentials,
            student_auth_service=FakeAuthService(),
            data_service=FakeDb(),
        )

    assert exc_info.value.status_code == 401


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dashboard_learning_summary_preserves_403_when_user_id_mismatches_token():
    class FakeAuthService:
        async def verify_jwt_token(self, token):
            return {"sub": "another-user"}

    class FakeDb:
        async def get_dashboard_learning_inputs(self, user_id, limit=50):
            raise AssertionError("db access should not be reached for mismatched token")

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="valid-token"
    )

    with pytest.raises(HTTPException) as exc_info:
        await endpoints.get_dashboard_learning_summary(
            user_id="student-1",
            credentials=credentials,
            student_auth_service=FakeAuthService(),
            data_service=FakeDb(),
        )

    assert exc_info.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enrolled_courses_preserves_401_when_token_verification_fails():
    class FakeAuthService:
        async def verify_jwt_token(self, token):
            raise HTTPException(status_code=401, detail="Invalid access token")

    class FakeDb:
        async def get_enrolled_courses_for_user(self, user_id, limit=50):
            raise AssertionError("db access should not be reached for invalid token")

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="invalid-token"
    )

    with pytest.raises(HTTPException) as exc_info:
        await endpoints.get_user_enrolled_courses(
            user_id="student-1",
            credentials=credentials,
            student_auth_service=FakeAuthService(),
            data_service=FakeDb(),
        )

    assert exc_info.value.status_code == 401


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enrolled_courses_preserves_403_when_user_id_mismatches_token():
    class FakeAuthService:
        async def verify_jwt_token(self, token):
            return {"sub": "another-user"}

    class FakeDb:
        async def get_enrolled_courses_for_user(self, user_id, limit=50):
            raise AssertionError("db access should not be reached for mismatched token")

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="valid-token"
    )

    with pytest.raises(HTTPException) as exc_info:
        await endpoints.get_user_enrolled_courses(
            user_id="student-1",
            credentials=credentials,
            student_auth_service=FakeAuthService(),
            data_service=FakeDb(),
        )

    assert exc_info.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enrolled_courses_awaits_service_directly():
    class FakeAuthService:
        async def verify_jwt_token(self, token):
            return {"sub": "student-1"}

    class FakeDb:
        async def get_enrolled_courses_for_user(self, user_id, limit=50):
            assert user_id == "student-1"
            assert limit == 50
            return [
                {
                    "course_id": "course-1",
                    "name": "Course 1",
                    "enrollment_id": "enrollment-1",
                    "status": "active",
                }
            ]

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="valid-token"
    )

    response = await endpoints.get_user_enrolled_courses(
        user_id="student-1",
        credentials=credentials,
        student_auth_service=FakeAuthService(),
        data_service=FakeDb(),
    )

    assert response[0]["id"] == "course-1"
    assert response[0]["enrollment_id"] == "enrollment-1"
