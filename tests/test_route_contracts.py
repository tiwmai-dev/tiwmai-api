"""Lock frontend-facing API route contracts for student, tutor, and admin apps."""

from app.api.admin_endpoints import ADMIN_FRONTEND_PATHS
from app.main import app

STUDENT_FRONTEND_PATHS = {
    "/api/v1/health",
    "/api/v1/student/auth/login",
    "/api/v1/student/auth/register",
    "/api/v1/student/auth/resend-verification-email",
    "/api/v1/student/auth/refresh",
    "/api/v1/student/auth/logout",
    "/api/v1/student/auth/me",
    "/api/v1/student/auth/onboarding-profile",
    "/api/v1/student/auth/avatar",
    "/api/v1/student/auth/oauth/authorize",
    "/api/v1/student/auth/oauth/callback",
    "/api/v1/student/auth/oauth/session",
    "/api/v1/student/legal/terms-of-use",
    "/api/v1/student/legal/privacy-policy",
    "/api/v1/student/courses",
    "/api/v1/student/courses/{course_id}/learning-overview",
    "/api/v1/student/courses/{course_id}/lessons",
    "/api/v1/student/courses/{course_id}/quizzes",
    "/api/v1/student/courses/{course_id}/mock-exam-leaderboard",
    "/api/v1/student/lessons/{lesson_id}",
    "/api/v1/student/quizzes/{quiz_id}",
    "/api/v1/student/users/{user_id}/enrolled-courses",
    "/api/v1/student/users/{user_id}/dashboard-learning-summary",
    "/api/v1/student/users/{user_id}/learning-activity",
    "/api/v1/student/users/{user_id}/quiz-results",
    "/api/v1/student/users/{user_id}/quizzes",
    "/api/v1/student/users/{user_id}/quizzes/{quiz_id}/submit",
    "/api/v1/student/users/{user_id}/quizzes/{quiz_id}/results",
    "/api/v1/student/users/{user_id}/payment-history",
    "/api/v1/student/enrollments",
    "/api/v1/student/payments/premium/promptpay/create-intent",
    "/api/v1/student/payments/premium/promptpay/confirm",
    "/api/v1/student/payments/stripe/webhook",
    "/api/v1/student/chat",
    "/api/v1/student/chat/energy",
}

TUTOR_LEGACY_PATHS = {
    "/api/v1/health",
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/auth/refresh",
    "/api/v1/auth/logout",
    "/api/v1/auth/me",
    "/api/v1/upload",
    "/api/v1/upload-and-process",
    "/api/v1/submit-to-s3/{document_id}",
    "/api/v1/files/{filename}",
    "/api/v1/documents/{document_id}/status",
    "/api/v1/s3/presigned-url",
    "/api/v1/s3/health",
    "/api/v1/users/{user_id}/documents",
    "/api/v1/users/{user_id}/quizzes",
    "/api/v1/users/{user_id}/courses",
    "/api/v1/quiz/{quiz_id}",
    "/api/v1/quiz/augment",
    "/api/v1/quiz/generate",
    "/api/v1/quiz/generate/progress/{job_id}",
    "/api/v1/courses",
    "/api/v1/courses/{course_id}",
    "/api/v1/courses/upload-image",
    "/api/v1/courses/ai-generate",
    "/api/v1/courses/{course_id}/lessons",
    "/api/v1/courses/{course_id}/question-bank",
    "/api/v1/courses/{course_id}/students",
    "/api/v1/courses/{course_id}/tutor-overview",
    "/api/v1/lessons/{lesson_id}",
    "/api/v1/chat/json",
    "/api/v1/invitations/create",
}

TUTOR_CANONICAL_PATHS = {
    "/api/v1/tutor/health",
    "/api/v1/tutor/auth/login",
    "/api/v1/tutor/auth/register",
    "/api/v1/tutor/auth/refresh",
    "/api/v1/tutor/auth/logout",
    "/api/v1/tutor/auth/me",
    "/api/v1/tutor/upload",
    "/api/v1/tutor/upload-and-process",
    "/api/v1/tutor/submit-to-s3/{document_id}",
    "/api/v1/tutor/files/{filename}",
    "/api/v1/tutor/documents/{document_id}/status",
    "/api/v1/tutor/s3/presigned-url",
    "/api/v1/tutor/s3/health",
    "/api/v1/tutor/users/{user_id}/documents",
    "/api/v1/tutor/users/{user_id}/quizzes",
    "/api/v1/tutor/users/{user_id}/courses",
    "/api/v1/tutor/quiz/{quiz_id}",
    "/api/v1/tutor/quiz/augment",
    "/api/v1/tutor/quiz/generate",
    "/api/v1/tutor/quiz/generate/progress/{job_id}",
    "/api/v1/tutor/courses",
    "/api/v1/tutor/courses/{course_id}",
    "/api/v1/tutor/courses/upload-image",
    "/api/v1/tutor/courses/ai-generate",
    "/api/v1/tutor/courses/{course_id}/lessons",
    "/api/v1/tutor/courses/{course_id}/question-bank",
    "/api/v1/tutor/courses/{course_id}/students",
    "/api/v1/tutor/courses/{course_id}/tutor-overview",
    "/api/v1/tutor/lessons/{lesson_id}",
    "/api/v1/tutor/chat/json",
    "/api/v1/tutor/invitations/create",
}

ADMIN_PATHS = {f"/api/v1{path}" for path in ADMIN_FRONTEND_PATHS}


def _paths():
    return {getattr(route, "path", "") for route in app.routes}


def test_student_frontend_route_contract():
    assert STUDENT_FRONTEND_PATHS.issubset(_paths())


def test_tutor_legacy_route_contract():
    assert TUTOR_LEGACY_PATHS.issubset(_paths())


def test_tutor_canonical_route_contract():
    assert TUTOR_CANONICAL_PATHS.issubset(_paths())


def test_admin_frontend_route_contract():
    assert ADMIN_PATHS.issubset(_paths())
