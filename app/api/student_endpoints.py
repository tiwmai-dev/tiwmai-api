"""Student-facing API route map.

This module exposes only the endpoints used by the student web app.
"""

from fastapi import APIRouter

from app.api.chat_endpoints import send_chat_message
from app.api.student_handlers import (
    confirm_promptpay_payment_and_enroll,
    create_promptpay_payment_intent,
    enroll_user_in_course,
    get_course_learning_overview,
    get_course_lessons,
    get_course_mock_exam_leaderboard,
    get_lesson,
    get_privacy_policy,
    get_quiz,
    get_terms_of_use,
    get_user_enrolled_courses,
    get_user_payment_history,
    get_user_quiz_results,
    get_dashboard_learning_summary,
    get_student_chat_energy_status,
    health_check,
    handle_stripe_payment_webhook,
    list_all_courses,
    list_course_quizzes,
    list_user_quiz_results,
    list_user_quizzes,
    record_user_learning_activity,
    submit_quiz_answers,
)

router = APIRouter(tags=["student"])

router.add_api_route("/health", health_check, methods=["GET"])

router.add_api_route("/student/legal/terms-of-use", get_terms_of_use, methods=["GET"])
router.add_api_route(
    "/student/legal/privacy-policy", get_privacy_policy, methods=["GET"]
)

router.add_api_route("/student/courses", list_all_courses, methods=["GET"])
router.add_api_route(
    "/student/courses/{course_id}/learning-overview",
    get_course_learning_overview,
    methods=["GET"],
)
router.add_api_route(
    "/student/courses/{course_id}/lessons", get_course_lessons, methods=["GET"]
)
router.add_api_route(
    "/student/courses/{course_id}/quizzes", list_course_quizzes, methods=["GET"]
)
router.add_api_route(
    "/student/courses/{course_id}/mock-exam-leaderboard",
    get_course_mock_exam_leaderboard,
    methods=["GET"],
)
router.add_api_route("/student/lessons/{lesson_id}", get_lesson, methods=["GET"])
router.add_api_route("/student/quizzes/{quiz_id}", get_quiz, methods=["GET"])

router.add_api_route(
    "/student/users/{user_id}/enrolled-courses",
    get_user_enrolled_courses,
    methods=["GET"],
)
router.add_api_route(
    "/student/users/{user_id}/dashboard-learning-summary",
    get_dashboard_learning_summary,
    methods=["GET"],
)
router.add_api_route(
    "/student/users/{user_id}/learning-activity",
    record_user_learning_activity,
    methods=["POST"],
)
router.add_api_route(
    "/student/users/{user_id}/quiz-results",
    list_user_quiz_results,
    methods=["GET"],
)
router.add_api_route(
    "/student/users/{user_id}/quizzes", list_user_quizzes, methods=["GET"]
)
router.add_api_route(
    "/student/users/{user_id}/quizzes/{quiz_id}/submit",
    submit_quiz_answers,
    methods=["POST"],
)
router.add_api_route(
    "/student/users/{user_id}/quizzes/{quiz_id}/results",
    get_user_quiz_results,
    methods=["GET"],
)
router.add_api_route(
    "/student/users/{user_id}/payment-history",
    get_user_payment_history,
    methods=["GET"],
)

router.add_api_route(
    "/student/enrollments", enroll_user_in_course, methods=["POST"]
)
router.add_api_route(
    "/student/payments/promptpay/create-intent",
    create_promptpay_payment_intent,
    methods=["POST"],
)
router.add_api_route(
    "/student/payments/promptpay/confirm",
    confirm_promptpay_payment_and_enroll,
    methods=["POST"],
)
router.add_api_route(
    "/student/payments/stripe/webhook",
    handle_stripe_payment_webhook,
    methods=["POST"],
)
router.add_api_route("/student/chat", send_chat_message, methods=["POST"])
router.add_api_route(
    "/student/chat/energy", get_student_chat_energy_status, methods=["GET"]
)
