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


def test_non_student_routes_are_not_registered():
    paths = _paths()
    blocked_paths = {
        "/metrics",
        "/api/v1/upload",
        "/api/v1/quiz/generate",
        "/api/v1/admin/students",
        "/api/v1/admin/transactions",
        "/api/v1/courses",
        "/api/v1/auth/login",
        "/api/v1/auth/student/login",
    }

    assert paths.isdisjoint(blocked_paths)
    assert not any(path.startswith("/api/v1/tutor") for path in paths)
    assert not any(path.startswith("/api/v1/admin") for path in paths)
