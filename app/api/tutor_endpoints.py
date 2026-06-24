"""Canonical tutor API routes under ``/api/v1/tutor/*``.

Legacy flat routes under ``/api/v1/*`` remain registered for backward compatibility.
"""

from fastapi import APIRouter

from app.api import auth_endpoints, chat_endpoints, endpoints, invitation_endpoints
from app.api.route_aliases import register_auth_aliases, register_route_aliases

router = APIRouter(tags=["tutor"])

TUTOR_ROUTE_SPECS = [
    ("GET", "/health", endpoints.health_check),
    ("POST", "/upload", endpoints.upload_file),
    ("POST", "/courses/upload-image", endpoints.upload_course_image),
    ("POST", "/process-document/{document_id}", endpoints.process_document),
    ("POST", "/upload-and-process", endpoints.upload_and_process),
    ("POST", "/submit-to-s3/{document_id}", endpoints.submit_document_to_s3),
    ("GET", "/users/{user_id}/documents", endpoints.list_user_documents),
    ("GET", "/s3/health", endpoints.s3_health_check),
    ("GET", "/files/{filename}", endpoints.get_file),
    ("DELETE", "/files/{filename}", endpoints.delete_file),
    ("GET", "/documents/{document_id}/status", endpoints.get_processing_status),
    ("GET", "/quiz/{quiz_id}", endpoints.get_quiz),
    ("PUT", "/quiz/{quiz_id}", endpoints.update_quiz),
    ("DELETE", "/quiz/{quiz_id}", endpoints.delete_quiz),
    ("GET", "/users/{user_id}/quizzes", endpoints.list_user_quizzes),
    ("GET", "/users/{user_id}/quiz-results", endpoints.list_user_quiz_results),
    ("GET", "/courses/{course_id}/quizzes", endpoints.list_course_quizzes),
    ("POST", "/users/{user_id}/quizzes", endpoints.create_quiz_manual),
    ("GET", "/s3/presigned-url", endpoints.get_presigned_url),
    ("POST", "/quiz/skeletons/upsample", endpoints.upsample_quiz_skeletons),
    (
        "GET",
        "/quiz/skeletons/upsample/progress/{job_id}",
        endpoints.get_skeleton_upsample_progress,
    ),
    ("POST", "/quiz/generate", endpoints.generate_quiz_with_ai),
    ("GET", "/quiz/generate/progress/{job_id}", endpoints.get_quiz_generate_progress),
    ("POST", "/courses/ai-generate", endpoints.generate_course_details_with_ai),
    ("POST", "/quiz/augment", endpoints.augment_quiz_questions),
    ("POST", "/courses", endpoints.create_course),
    ("GET", "/courses/{course_id}", endpoints.get_course),
    ("GET", "/courses/{course_id}/learning-overview", endpoints.get_course_learning_overview),
    ("GET", "/courses/{course_id}/question-bank", endpoints.get_course_question_bank),
    ("PUT", "/courses/{course_id}/question-bank", endpoints.replace_course_question_bank),
    ("GET", "/users/{user_id}/courses", endpoints.list_user_courses),
    ("GET", "/courses", endpoints.list_all_courses),
    ("PUT", "/courses/{course_id}", endpoints.update_course),
    ("DELETE", "/courses/{course_id}", endpoints.delete_course),
    ("GET", "/courses/{course_id}/students", endpoints.get_course_students),
    ("GET", "/courses/{course_id}/tutor-overview", endpoints.get_course_tutor_overview),
    ("POST", "/courses/{course_id}/lessons", endpoints.create_lesson),
    ("GET", "/courses/{course_id}/lessons", endpoints.get_course_lessons),
    ("PUT", "/lessons/{lesson_id}", endpoints.update_lesson),
    ("DELETE", "/lessons/{lesson_id}", endpoints.delete_lesson),
    ("GET", "/lessons/{lesson_id}", endpoints.get_lesson),
    ("POST", "/chat/json", chat_endpoints.send_chat_message_json),
    ("POST", "/invitations/create", invitation_endpoints.create_course_invitation),
]

register_route_aliases(router, "/tutor", TUTOR_ROUTE_SPECS)
register_auth_aliases(router, "/tutor", auth_endpoints.router)
