"""Canonical admin API route contract.

Admin routes are served at ``/api/v1/admin/*`` and shared course/enrollment helpers
at legacy flat paths used by the admin dashboard.
"""

ADMIN_ROUTE_PATHS = (
    "/admin/token-usage/daily",
    "/admin/students",
    "/admin/transactions",
    "/admin/chat-energy/settings",
    "/admin/users/{user_id}/chat-energy",
    "/admin/users/{user_id}/trial-status",
    "/admin/users/{user_id}/premium-status",
    "/admin/enrollments/{enrollment_id}/expiry",
)

ADMIN_SHARED_LEGACY_PATHS = (
    "/courses",
    "/courses/{course_id}",
    "/courses/{course_id}/students",
    "/enroll",
    "/enrollments/{enrollment_id}",
)

ADMIN_FRONTEND_PATHS = ADMIN_ROUTE_PATHS + ADMIN_SHARED_LEGACY_PATHS
