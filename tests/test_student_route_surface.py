from app.main import app


def _paths():
    return {getattr(route, "path", "") for route in app.routes}


def test_student_route_surface_is_registered():
    paths = _paths()

    assert "/api/v1/health" in paths
    assert "/api/v1/student/auth/login" in paths
    assert "/api/v1/student/courses" in paths
    assert "/api/v1/student/courses/{course_id}/learning-overview" in paths
    assert "/api/v1/student/users/{user_id}/enrolled-courses" in paths
    assert "/api/v1/student/users/{user_id}/quizzes/{quiz_id}/submit" in paths
    assert "/api/v1/student/payments/promptpay/create-intent" in paths
    assert "/api/v1/student/chat" in paths


def test_tutor_and_admin_routes_are_registered():
    paths = _paths()

    expected_paths = {
        "/api/v1/upload",
        "/api/v1/upload-and-process",
        "/api/v1/quiz/generate",
        "/api/v1/quiz/augment",
        "/api/v1/users/{user_id}/quizzes",
        "/api/v1/courses",
        "/api/v1/courses/{course_id}",
        "/api/v1/courses/{course_id}/lessons",
        "/api/v1/auth/login",
        "/api/v1/invitations/create",
        "/api/v1/admin/students",
        "/api/v1/admin/transactions",
        "/api/v1/admin/token-usage/daily",
    }

    assert expected_paths.issubset(paths)
