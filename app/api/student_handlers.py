"""Student-only endpoint handlers used by the student web app."""

import asyncio
import hashlib
import hmac
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import httpx
from fastapi import Body, Depends, Form, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.exceptions import BaseAPIException
from app.core.logging import app_logger
from app.models.schemas import HealthCheckResponse, LessonListResponse, LessonResponse
from app.services.chat_service import ChatService
from app.services.data_service import get_db_service
from app.services.student_auth_service import StudentAuthService

STRIPE_API_BASE = "https://api.stripe.com/v1"
PAYMENT_TIME_ZONE = ZoneInfo("Asia/Bangkok")
STUDENT_BEARER_OPTIONAL = HTTPBearer(auto_error=False)
LEGAL_DOC_VERSION = "1.0"
LEGAL_DOC_LAST_UPDATED = "2026-04-13"
TRIAL_OVERRIDE_MODES = {"auto", "available", "used"}


async def health_check():
    """Health check endpoint for the student API."""
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=0.0,
        llm_status="configured" if get_settings().openrouter_api_key else "not_configured",
    )

class PromptPayCreateIntentRequest(BaseModel):
    user_id: str
    course_id: str
    billing_email: Optional[str] = None
    amount_thb: Optional[float] = None
    plan_label: Optional[str] = None
    duration_months: Optional[int] = None


class PromptPayCreateIntentResponse(BaseModel):
    payment_intent_id: str
    client_secret: str
    publishable_key: str
    amount: int
    currency: str
    payment_status: str
    already_enrolled: bool = False


class PromptPayConfirmRequest(BaseModel):
    user_id: str
    course_id: str
    payment_intent_id: str


class LegalSection(BaseModel):
    heading: str
    details: List[str]


class LegalDocumentResponse(BaseModel):
    document_name: str
    version: str
    last_updated: str
    summary: str
    sections: List[LegalSection]
    contact_email: Optional[str] = None


class QuizSubmitPayload(BaseModel):
    answers: Any
    course_id: Optional[str] = None
    lesson_id: Optional[str] = None
    time_spent_seconds: Optional[int] = 0
    per_question_time_seconds: Optional[Dict[str, int]] = None
    confidence_by_question: Optional[Dict[str, Optional[str]]] = None


class LearningActivityPayload(BaseModel):
    course_id: str
    lesson_id: Optional[str] = None
    activity_day: Optional[str] = None
    activity_days: List[str] = Field(default_factory=list)


class StudentAnalysisSummaryRequest(BaseModel):
    recommendation_schema_version: Optional[str] = None
    analysis_plan: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    recent_trend: Dict[str, Any] = Field(default_factory=dict)
    focus_area: Dict[str, Any] = Field(default_factory=dict)
    priority_patterns: Dict[str, Any] = Field(default_factory=dict)
    available_actions: List[Dict[str, Any]] = Field(default_factory=list)
    client_recommendation_cards: List[Dict[str, Any]] = Field(default_factory=list)


async def _get_student_auth_service() -> StudentAuthService:
    return StudentAuthService()


async def _stripe_request(method: str, path: str, secret_key: str, data: Optional[Dict[str, Any]]=None, params: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    url = f'{STRIPE_API_BASE}{path}'
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.request(method=method.upper(), url=url, auth=(secret_key, ''), params=params, data=data, headers={'Content-Type': 'application/x-www-form-urlencoded'})
    payload = response.json() if response.content else {}
    if response.status_code >= 400:
        error_obj = payload.get('error') if isinstance(payload, dict) else {}
        message = error_obj.get('message') if isinstance(error_obj, dict) else None
        raise HTTPException(status_code=400, detail=message or 'Stripe request failed')
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail='Invalid Stripe response')
    return payload


async def _get_existing_enrollment_with_schedule(data_service, user_id: str, course_id: str) -> Optional[Dict[str, Any]]:
    enrollment = await _get_user_course_enrollment(data_service=data_service, user_id=user_id, course_id=course_id)
    if not enrollment:
        return None
    schedule = _build_enrollment_schedule(started_at_raw=enrollment.get('started_at') or enrollment.get('enrolled_at'), expires_at_raw=enrollment.get('expires_at'), duration_months_raw=enrollment.get('duration_months'))
    return {'enrollment': enrollment, 'schedule': schedule}


async def _get_user_course_enrollment(data_service, user_id: str, course_id: str) -> Optional[Dict[str, Any]]:
    get_with_aliases = getattr(data_service, 'get_user_enrollments_with_aliases', None)
    if callable(get_with_aliases):
        enrollments = await get_with_aliases(user_id)
    else:
        enrollments = await data_service.get_user_enrollments(user_id)
    target_course_id = str(course_id or '').strip()
    for row in enrollments:
        enrolled_course_id = str(row.get('course_id') or row.get('id') or row.get('_id') or '').strip()
        if enrolled_course_id != target_course_id:
            continue
        status = str(row.get('status') or 'active').strip().lower()
        if status == 'cancelled':
            continue
        return row
    return None


async def _ensure_active_course_access(data_service, user_id: str, course_id: str) -> Dict[str, Any]:
    normalized_user_id = str(user_id or '').strip()
    normalized_course_id = str(course_id or '').strip()
    if not normalized_user_id or not normalized_course_id:
        raise HTTPException(status_code=400, detail='Missing user_id or course_id for access check')
    enrollment = await _get_user_course_enrollment(data_service=data_service, user_id=normalized_user_id, course_id=normalized_course_id)
    if not enrollment:
        raise HTTPException(status_code=403, detail='COURSE_ACCESS_DENIED: not enrolled in this course')
    schedule = _build_enrollment_schedule(started_at_raw=enrollment.get('started_at') or enrollment.get('enrolled_at'), expires_at_raw=enrollment.get('expires_at'), duration_months_raw=enrollment.get('duration_months'))
    if schedule['is_expired']:
        expired_at = schedule.get('expires_at') or 'unknown'
        raise HTTPException(status_code=403, detail=f'COURSE_EXPIRED: enrollment expired on {expired_at}')
    return {'enrollment': enrollment, 'schedule': schedule}


async def _ensure_user_matches_token(user_id: Optional[str], credentials: Optional[HTTPAuthorizationCredentials], auth_service: StudentAuthService) -> None:
    requested_user_id = str(user_id or '').strip()
    if not requested_user_id:
        return
    if not credentials or not str(credentials.credentials or '').strip():
        raise HTTPException(status_code=401, detail='UNAUTHORIZED: missing bearer token')
    payload = await auth_service.verify_jwt_token(credentials.credentials)
    principals = {str(value).strip() for value in [payload.get('sub'), payload.get('username'), payload.get('cognito:username'), payload.get('user_id'), payload.get('custom:student_id')] if str(value or '').strip()}
    if not principals:
        raise HTTPException(status_code=401, detail='UNAUTHORIZED: token has no principal')
    if requested_user_id in principals:
        return
    requested_lower = requested_user_id.lower()
    principals_lower = {value.lower() for value in principals}
    if requested_lower in principals_lower:
        return
    raise HTTPException(status_code=403, detail='USER_ID_TOKEN_MISMATCH')


def _parse_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(str(value).strip())
        return parsed if parsed > 0 else None
    except Exception:
        return None


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    raw = str(value or '').strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace('Z', '+00:00'))
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except Exception:
        return None


def _format_utc_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.replace(tzinfo=timezone.utc).isoformat()


def _build_enrollment_schedule(started_at_raw: Any, expires_at_raw: Any, duration_months_raw: Any) -> Dict[str, Any]:
    started_dt = _parse_iso_datetime(started_at_raw) or datetime.utcnow()
    duration_months = _parse_positive_int(duration_months_raw)
    expires_dt = _parse_iso_datetime(expires_at_raw)
    if expires_dt is None and duration_months:
        expires_dt = started_dt + timedelta(days=duration_months * 30)
    now = datetime.utcnow()
    is_expired = bool(expires_dt and expires_dt < now)
    days_remaining = None
    if expires_dt:
        days_remaining = max(0, int((expires_dt - now).total_seconds() // 86400))
    return {'started_at': _format_utc_iso(started_dt), 'expires_at': _format_utc_iso(expires_dt), 'duration_months': duration_months, 'is_expired': is_expired, 'days_remaining': days_remaining}


def _is_trial_enrollment(enrollment: Dict[str, Any]) -> bool:
    source = str(enrollment.get('enrollment_source') or enrollment.get('enrollment_type') or '').strip().lower()
    if source == 'trial':
        return True
    return any((bool(enrollment.get(field)) for field in ('trial_consumed_at', 'trial_expires_at')))


def _coerce_number(*values: Any) -> float:
    for value in values:
        try:
            parsed = float(value)
            if parsed == parsed and parsed not in (float('inf'), float('-inf')):
                return parsed
        except Exception:
            continue
    return 0.0


def _normalize_learning_text(value: Any) -> str:
    return str(value or '').strip().lower()


def _normalize_lesson_id(value: Any) -> str:
    return str(value or '').strip()


def _get_lesson_name(lesson: Dict[str, Any], fallback_index: int=0) -> str:
    raw = lesson.get('title') or lesson.get('name') or lesson.get('lesson_name') or lesson.get('topic') or ''
    text = str(raw or '').strip()
    return text or f'บทเรียน {fallback_index + 1}'


def _to_topic_label(*values: Any) -> str:
    for value in values:
        text = str(value or '').strip()
        if text:
            return text
    return ''


def _resolve_question_topic_label(question: Dict[str, Any], fallback: str='ไม่ระบุหัวข้อ') -> str:
    return _to_topic_label(question.get('topic_tag'), question.get('topicTag'), question.get('topic'), question.get('subject_tag'), question.get('subject'), question.get('category')) or fallback


def _get_attempt_stats(item: Dict[str, Any]) -> Optional[Dict[str, float]]:
    total_questions = max(0.0, _coerce_number(item.get('total_questions')))
    correct_count = max(0.0, _coerce_number(item.get('correct_count')))
    if total_questions > 0:
        bounded_correct = min(correct_count, total_questions)
        return {'total': total_questions, 'correct': bounded_correct, 'accuracy': bounded_correct / total_questions * 100}
    score_raw = item.get('score')
    try:
        score = float(score_raw)
    except Exception:
        return None
    if score != score:
        return None
    bounded_score = max(0.0, min(100.0, score))
    return {'total': 1.0, 'correct': bounded_score / 100.0, 'accuracy': bounded_score}


def _to_difficulty_label(value: Any) -> Optional[str]:
    if value is None or value == '':
        return None
    if isinstance(value, (int, float)):
        if value <= 1:
            return 'ง่าย'
        if value == 2:
            return 'กลาง'
        return 'ยาก'
    text = _normalize_learning_text(value)
    if 'easy' in text or 'ง่าย' in text:
        return 'ง่าย'
    if 'hard' in text or 'ยาก' in text or 'advanced' in text:
        return 'ยาก'
    if 'medium' in text or 'กลาง' in text or 'intermediate' in text:
        return 'กลาง'
    return None


def _to_difficulty_label_from_quiz(quiz: Dict[str, Any]) -> Optional[str]:
    direct = _to_difficulty_label(quiz.get('difficulty_avg') if quiz.get('difficulty_avg') is not None else quiz.get('difficulty') if quiz.get('difficulty') is not None else quiz.get('level_difficulty') if quiz.get('level_difficulty') is not None else quiz.get('difficulty_level') if quiz.get('difficulty_level') is not None else quiz.get('level'))
    if direct:
        return direct
    questions = quiz.get('questions')
    if not isinstance(questions, list):
        return None
    numeric_difficulties = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        try:
            parsed = float(question.get('difficulty'))
        except Exception:
            continue
        if parsed > 0:
            numeric_difficulties.append(parsed)
    if not numeric_difficulties:
        return None
    avg = sum(numeric_difficulties) / len(numeric_difficulties)
    if avg <= 2:
        return 'ง่าย'
    if avg >= 4:
        return 'ยาก'
    return 'กลาง'


def _detect_quiz_kind(payload: Dict[str, Any]) -> str:
    tags = payload.get('tags')
    if isinstance(tags, list):
        tags_text = ' '.join((str(item) for item in tags))
    else:
        tags_text = str(tags or '')
    text = ' '.join((str(value or '') for value in (payload.get('title'), payload.get('name'), payload.get('quiz_type'), payload.get('type'), payload.get('purpose'), payload.get('description'), payload.get('document_type'), tags_text) if value)).lower()
    if 'mock_exam' in text or 'mock exam' in text or 'แบบทดสอบจำลอง' in text:
        return 'mock_exam'
    return 'lesson'


def _parse_timestamp_ms(value: Any) -> float:
    dt = _parse_iso_datetime(value)
    if not dt:
        return 0.0
    return dt.timestamp() * 1000


def _format_attempt_label(submitted_at_ms: float, fallback_index: int) -> str:
    if submitted_at_ms > 0:
        dt = datetime.fromtimestamp(submitted_at_ms / 1000)
        return dt.strftime('%d/%m')
    return f'ครั้งที่ {fallback_index + 1}'


def _merge_course_with_enrollment(course: Dict[str, Any], enrollment: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(course)
    row['enrollment'] = enrollment
    row['enrollment_id'] = enrollment.get('enrollment_id')
    row['enrollment_status'] = enrollment.get('status')
    row['enrolled_at'] = enrollment.get('enrolled_at')
    row['started_at'] = enrollment.get('started_at')
    row['expires_at'] = enrollment.get('expires_at')
    row['duration_months'] = enrollment.get('duration_months')
    row['enrollment_source'] = enrollment.get('enrollment_source')
    row['enrollment_type'] = enrollment.get('enrollment_type')
    row['payment_provider'] = enrollment.get('payment_provider')
    row['payment_type'] = enrollment.get('payment_type')
    row['payment_intent_id'] = enrollment.get('payment_intent_id')
    row['payment_status'] = enrollment.get('payment_status')
    row['paid_amount_thb'] = enrollment.get('paid_amount_thb')
    row['paid_currency'] = enrollment.get('paid_currency')
    row['billing_email'] = enrollment.get('billing_email')
    row['plan_label'] = enrollment.get('plan_label')
    row['paid_at'] = enrollment.get('paid_at')
    row['payment_history'] = enrollment.get('payment_history')
    row['trial_consumed_at'] = enrollment.get('trial_consumed_at')
    row['trial_expires_at'] = enrollment.get('trial_expires_at')
    row['progress'] = enrollment.get('progress', row.get('progress', 0))
    row['completed_quizzes'] = enrollment.get('completed_quizzes', row.get('completed_quizzes', 0))
    row['total_quizzes'] = enrollment.get('total_quizzes', row.get('total_quizzes', 0))
    row['completed_questions'] = enrollment.get('completed_questions', row.get('completed_questions', 0))
    row['total_questions'] = enrollment.get('total_questions', row.get('total_questions', 0))
    row['last_activity'] = enrollment.get('last_activity', row.get('last_activity'))
    return row


def _format_student_course(course: Dict[str, Any]) -> Dict[str, Any]:
    schedule = _build_enrollment_schedule(started_at_raw=course.get('started_at') or course.get('enrolled_at'), expires_at_raw=course.get('expires_at'), duration_months_raw=course.get('duration_months'))
    return {'id': course.get('course_id'), 'name': course.get('name'), 'description': course.get('description'), 'detail': course.get('detail'), 'target_profile': course.get('target_profile'), 'structure_summary': course.get('structure_summary'), 'topics': course.get('topics', []), 'tags': course.get('tags', []), 'course_tags': course.get('tags', []), 'benefits': course.get('benefits', []), 'content_items': course.get('content_items', []), 'instructor': course.get('instructor') or course.get('teacher_name') or 'อาจารย์ระบบ', 'teacher_name': course.get('teacher_name') or course.get('instructor') or 'อาจารย์ระบบ', 'category': course.get('category', 'ทั่วไป'), 'progress': course.get('progress', 0), 'totalQuizzes': course.get('total_quizzes', 0), 'completedQuizzes': course.get('completed_quizzes', 0), 'totalQuestions': course.get('total_questions', 0), 'completedQuestions': course.get('completed_questions', 0), 'lastActivity': course.get('last_activity', 'เพิ่งเข้าร่วม'), 'color': '#4ecdc4', 'image': '📚', 'image_url': course.get('image_url'), 'thumbnail_url': course.get('thumbnail_url'), 'preview_image_url': course.get('preview_image_url'), 'purchase_preview_image_url': course.get('purchase_preview_image_url'), 'price': course.get('price'), 'enrollment_id': course.get('enrollment_id'), 'enrolled_at': course.get('enrolled_at'), 'started_at': schedule['started_at'], 'expires_at': schedule['expires_at'], 'duration_months': schedule['duration_months'], 'is_expired': schedule['is_expired'], 'days_remaining': schedule['days_remaining'], 'enrollment_source': course.get('enrollment_source'), 'enrollment_type': course.get('enrollment_type'), 'trial_consumed_at': course.get('trial_consumed_at'), 'trial_expires_at': course.get('trial_expires_at'), 'is_trial': _is_trial_enrollment(course)}


def _build_dashboard_course_stats(courses: List[Dict[str, Any]], lessons: List[Dict[str, Any]], quizzes: List[Dict[str, Any]], quiz_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lessons_by_course: Dict[str, List[Dict[str, Any]]] = {}
    for lesson in lessons:
        course_id = str(lesson.get('course_id') or lesson.get('courseId') or '')
        lessons_by_course.setdefault(course_id, []).append(lesson)
    quizzes_by_course: Dict[str, List[Dict[str, Any]]] = {}
    for quiz in quizzes:
        course_id = str(quiz.get('course_id') or quiz.get('courseId') or '')
        quizzes_by_course.setdefault(course_id, []).append(quiz)
    stats: Dict[str, Dict[str, Any]] = {}
    week_start_ms = (datetime.utcnow() - timedelta(days=7)).timestamp() * 1000
    for course in courses:
        course_id = str(course.get('course_id') or course.get('id') or '')
        if not course_id:
            continue
        course_lessons = lessons_by_course.get(course_id, [])
        course_quizzes = quizzes_by_course.get(course_id, [])
        regular_quizzes = [quiz for quiz in course_quizzes if str(quiz.get('document_type') or '').lower() != 'mock_exam']
        lesson_rows = []
        for lesson_index, lesson in enumerate(sorted(course_lessons, key=lambda row: _coerce_number(row.get('order'), 0) if row.get('order') is not None else 0)):
            lesson_id = _normalize_lesson_id(lesson.get('id') or lesson.get('lesson_id'))
            if not lesson_id:
                continue
            lesson_rows.append({'id': lesson_id, 'name': _get_lesson_name(lesson, lesson_index), 'order': int(_coerce_number(lesson.get('order'), lesson_index + 1))})
        lesson_name_by_id = {lesson['id']: lesson['name'] for lesson in lesson_rows}
        quiz_to_lesson_id: Dict[str, str] = {}
        for lesson in course_lessons:
            lesson_id = _normalize_lesson_id(lesson.get('id') or lesson.get('lesson_id'))
            if not lesson_id:
                continue
            quiz_refs = []
            for key in ('quizzes', 'selected_quizzes', 'selectedQuizzes'):
                value = lesson.get(key)
                if isinstance(value, list):
                    quiz_refs.extend(value)
            for quiz_ref in quiz_refs:
                if not isinstance(quiz_ref, dict):
                    quiz_id = str(quiz_ref or '').strip()
                    if quiz_id:
                        quiz_to_lesson_id[quiz_id] = lesson_id
                    continue
                for raw_id in (quiz_ref.get('quiz_id'), quiz_ref.get('id'), quiz_ref.get('document_id')):
                    quiz_id = str(raw_id or '').strip()
                    if quiz_id:
                        quiz_to_lesson_id[quiz_id] = lesson_id
        quiz_ids = {str(quiz.get('quiz_id') or quiz.get('id') or quiz.get('document_id') or '') for quiz in course_quizzes if str(quiz.get('quiz_id') or quiz.get('id') or quiz.get('document_id') or '')}
        quiz_difficulty_by_id = {str(quiz.get('quiz_id') or quiz.get('id') or quiz.get('document_id')): _to_difficulty_label_from_quiz(quiz) for quiz in regular_quizzes if str(quiz.get('quiz_id') or quiz.get('id') or quiz.get('document_id') or '')}
        quiz_topic_by_id = {str(quiz.get('quiz_id') or quiz.get('id') or quiz.get('document_id')): _to_topic_label(quiz.get('topic_tag'), quiz.get('topicTag'), quiz.get('topic'), quiz.get('category'), quiz.get('subject')) for quiz in course_quizzes if str(quiz.get('quiz_id') or quiz.get('id') or quiz.get('document_id') or '') and _to_topic_label(quiz.get('topic_tag'), quiz.get('topicTag'), quiz.get('topic'), quiz.get('category'), quiz.get('subject'))}
        all_quiz_kind_by_id = {str(quiz.get('quiz_id') or quiz.get('id') or quiz.get('document_id')): _detect_quiz_kind(quiz) for quiz in course_quizzes if str(quiz.get('quiz_id') or quiz.get('id') or quiz.get('document_id') or '')}
        course_results = []
        for item in quiz_results:
            result_course_id = str(item.get('course_id') or '')
            result_quiz_id = str(item.get('quiz_id') or '')
            if result_course_id and result_course_id == course_id:
                course_results.append(item)
            elif result_quiz_id and result_quiz_id in quiz_ids:
                course_results.append(item)
        time_spent_seconds_this_week = 0.0
        for item in course_results:
            seconds = max(0.0, _coerce_number(item.get('time_spent_seconds'), item.get('total_time_spent_seconds')))
            if seconds <= 0:
                continue
            timestamp_ms = _parse_timestamp_ms(item.get('submitted_at') or item.get('updated_at') or item.get('created_at'))
            if timestamp_ms <= 0 or timestamp_ms >= week_start_ms:
                time_spent_seconds_this_week += seconds
        attempted_quiz_ids = {str(item.get('quiz_id')) for item in course_results if str(item.get('quiz_id') or '')}
        question_attempts = sum((_coerce_number(item.get('total_questions')) for item in course_results))
        correct_answers = sum((_coerce_number(item.get('correct_count')) for item in course_results))
        attempted_lesson_ids = set()
        difficulty_buckets = {'easy': {'correct': 0.0, 'total': 0.0}, 'medium': {'correct': 0.0, 'total': 0.0}, 'hard': {'correct': 0.0, 'total': 0.0}}
        score_buckets = {'lesson': {'correct': 0.0, 'total': 0.0}, 'mockExam': {'correct': 0.0, 'total': 0.0}}
        topic_buckets: Dict[str, Dict[str, Any]] = {}
        lesson_topic_buckets: Dict[str, Dict[str, Any]] = {}
        scored_attempt_count = 0
        lesson_buckets: Dict[str, Dict[str, Any]] = {}
        for lesson in lesson_rows:
            lesson_buckets[lesson['id']] = {'id': lesson['id'], 'name': lesson['name'], 'order': lesson['order'], 'minutes': 0.0, 'lesson': {'correct': 0.0, 'total': 0.0}, 'mockExam': {'correct': 0.0, 'total': 0.0}}

        def _add_topic_stat(bucket: Dict[str, Dict[str, Any]], topic_label: str, total: float, correct: float) -> None:
            if topic_label not in bucket:
                bucket[topic_label] = {'topic': topic_label, 'total': 0.0, 'correct': 0.0}
            bucket[topic_label]['total'] += total
            bucket[topic_label]['correct'] += correct

        def _add_lesson_topic_stat(lesson_id: Optional[str], topic_label: str, total: float, correct: float) -> None:
            group_key = lesson_id or '__unassigned__'
            lesson_meta = next((lesson for lesson in lesson_rows if lesson['id'] == lesson_id), None)
            if group_key not in lesson_topic_buckets:
                lesson_topic_buckets[group_key] = {'lessonId': lesson_id, 'lessonName': (lesson_name_by_id.get(lesson_id) if lesson_id else None) or 'ไม่ระบุบท', 'lessonOrder': lesson_meta.get('order') if lesson_meta else 9007199254740991, 'topics': {}}
            _add_topic_stat(lesson_topic_buckets[group_key]['topics'], topic_label, total, correct)
        for item in course_results:
            attempt_stats = _get_attempt_stats(item)
            if not attempt_stats:
                continue
            scored_attempt_count += 1
            total_questions = attempt_stats['total']
            correct_count = attempt_stats['correct']
            mapped_difficulty = quiz_difficulty_by_id.get(str(item.get('quiz_id'))) or _to_difficulty_label(item.get('difficulty') or item.get('level_difficulty') or item.get('difficulty_level'))
            bucket_key = 'easy' if mapped_difficulty == 'ง่าย' else 'hard' if mapped_difficulty == 'ยาก' else 'medium' if mapped_difficulty == 'กลาง' else None
            if bucket_key:
                difficulty_buckets[bucket_key]['total'] += total_questions
                difficulty_buckets[bucket_key]['correct'] += correct_count
            kind = all_quiz_kind_by_id.get(str(item.get('quiz_id'))) or _detect_quiz_kind(item)
            kind_key = 'mockExam' if kind == 'mock_exam' else 'lesson'
            is_lesson_practice = kind_key == 'lesson'
            score_buckets[kind_key]['total'] += total_questions
            score_buckets[kind_key]['correct'] += correct_count
            explicit_lesson_id = _normalize_lesson_id(item.get('lesson_id') or item.get('lessonId'))
            quiz_mapped_lesson_id = _normalize_lesson_id(quiz_to_lesson_id.get(str(item.get('quiz_id') or '')))
            mapped_lesson_id = explicit_lesson_id if explicit_lesson_id and explicit_lesson_id in lesson_name_by_id else quiz_mapped_lesson_id or explicit_lesson_id
            has_known_lesson = bool(mapped_lesson_id and mapped_lesson_id in lesson_buckets)
            fallback_topic_label = _to_topic_label(item.get('topic_tag'), item.get('topicTag'), item.get('topic'), item.get('subject_tag'), quiz_topic_by_id.get(str(item.get('quiz_id') or ''))) or 'ไม่ระบุหัวข้อ'
            question_insights = item.get('question_insights')
            answered_question_insights = []
            if isinstance(question_insights, list):
                answered_question_insights = [question for question in question_insights if isinstance(question, dict) and question.get('is_correct') in (True, False)]
            if answered_question_insights:
                for question in answered_question_insights:
                    topic_label = _resolve_question_topic_label(question, fallback_topic_label)
                    correct_value = 1.0 if question.get('is_correct') is True else 0.0
                    _add_topic_stat(topic_buckets, topic_label, 1.0, correct_value)
                    if is_lesson_practice and has_known_lesson:
                        _add_lesson_topic_stat(mapped_lesson_id, topic_label, 1.0, correct_value)
            elif total_questions > 0:
                _add_topic_stat(topic_buckets, fallback_topic_label, total_questions, correct_count)
                if is_lesson_practice and has_known_lesson:
                    _add_lesson_topic_stat(mapped_lesson_id, fallback_topic_label, total_questions, correct_count)
            if not has_known_lesson:
                continue
            attempted_lesson_ids.add(mapped_lesson_id)
            lesson_buckets[mapped_lesson_id][kind_key]['total'] += total_questions
            lesson_buckets[mapped_lesson_id][kind_key]['correct'] += correct_count
            seconds = max(0.0, _coerce_number(item.get('time_spent_seconds'), item.get('total_time_spent_seconds')))
            if seconds > 0:
                lesson_buckets[mapped_lesson_id]['minutes'] += seconds / 60
        computed_lesson_rows = []
        for lesson in sorted(lesson_buckets.values(), key=lambda row: (row.get('order') or 0, str(row.get('name') or ''))):
            lesson_total = lesson['lesson']['total']
            mock_total = lesson['mockExam']['total']
            computed_lesson_rows.append({'id': lesson['id'], 'name': lesson['name'], 'scoreSplit': {'lesson': round(lesson['lesson']['correct'] / lesson_total * 100) if lesson_total > 0 else None, 'mockExam': round(lesson['mockExam']['correct'] / mock_total * 100) if mock_total > 0 else None}, 'minutes': round(lesson['minutes']) if lesson['minutes'] > 0 else 0})
        attempt_rows = []
        for index, item in enumerate(course_results):
            attempt_stats = _get_attempt_stats(item)
            if not attempt_stats or attempt_stats['total'] <= 0:
                continue
            submitted_at_raw = item.get('submitted_at') or item.get('updated_at') or item.get('created_at')
            submitted_at_ms = _parse_timestamp_ms(submitted_at_raw)
            safe_score = max(0, min(100, round(attempt_stats['correct'] / attempt_stats['total'] * 100)))
            attempt_rows.append({'id': f"{course_id}-{item.get('result_id') or item.get('id') or item.get('quiz_id') or index}", 'score': safe_score, 'submittedAt': submitted_at_raw or None, 'submittedAtMs': submitted_at_ms if submitted_at_ms > 0 else 0, 'quizTitle': str(item.get('quiz_title') or item.get('quiz_name') or item.get('title') or '').strip(), 'sequence': index + 1})
        attempt_rows.sort(key=lambda row: (row['submittedAtMs'] <= 0, row['submittedAtMs'] if row['submittedAtMs'] > 0 else row['sequence']))
        for index, row in enumerate(attempt_rows):
            row['label'] = _format_attempt_label(row['submittedAtMs'], index)
            row['attemptIndex'] = index + 1
        topic_rows = [{'id': f"{course_id}-{topic['topic']}", 'topic': topic['topic'], 'total': int(round(topic['total'])), 'correct': int(round(topic['correct'])), 'accuracy': round(topic['correct'] / topic['total'] * 100)} for topic in topic_buckets.values() if topic['total'] > 0]
        topic_rows.sort(key=lambda topic: (-topic['total'], topic['topic'] == 'ไม่ระบุหัวข้อ', str(topic['topic'] or '')))
        topic_rows_by_lesson = []
        for group in lesson_topic_buckets.values():
            group_topics = [{'id': f"{course_id}-{group['lessonId'] or 'unassigned'}-{topic['topic']}", 'topic': topic['topic'], 'total': int(round(topic['total'])), 'correct': int(round(topic['correct'])), 'accuracy': round(topic['correct'] / topic['total'] * 100)} for topic in group['topics'].values() if topic['total'] > 0]
            group_topics.sort(key=lambda topic: (-topic['total'], topic['topic'] == 'ไม่ระบุหัวข้อ', str(topic['topic'] or '')))
            if group_topics:
                topic_rows_by_lesson.append({'lessonId': group['lessonId'], 'lessonName': group['lessonName'], 'lessonOrder': group['lessonOrder'], 'topics': group_topics})
        topic_rows_by_lesson.sort(key=lambda group: (group.get('lessonOrder') or 9007199254740991, str(group.get('lessonName') or '')))
        difficulty_score = {key: round(bucket['correct'] / bucket['total'] * 100) if bucket['total'] > 0 else None for key, bucket in difficulty_buckets.items()}
        total_difficulty_questions = sum((bucket['total'] for bucket in difficulty_buckets.values()))
        total_difficulty_correct = sum((bucket['correct'] for bucket in difficulty_buckets.values()))
        score_split = {'lesson': round(score_buckets['lesson']['correct'] / score_buckets['lesson']['total'] * 100) if score_buckets['lesson']['total'] > 0 else None, 'mockExam': round(score_buckets['mockExam']['correct'] / score_buckets['mockExam']['total'] * 100) if score_buckets['mockExam']['total'] > 0 else None}
        completed_quizzes = len(attempted_quiz_ids)
        total_quizzes = len(regular_quizzes)
        progress = round(completed_quizzes / total_quizzes * 100) if total_quizzes > 0 else _coerce_number(course.get('progress'))
        submitted_values = [str(item.get('submitted_at') or '') for item in course_results if str(item.get('submitted_at') or '')]
        last_submitted_at = sorted(submitted_values)[-1] if submitted_values else None
        stats[course_id] = {'totalLessons': len(course_lessons), 'completedLessons': len(attempted_lesson_ids), 'totalQuizzes': total_quizzes, 'completedQuizzes': completed_quizzes, 'totalQuestions': question_attempts if question_attempts > 0 else scored_attempt_count, 'completedQuestions': correct_answers if question_attempts > 0 else scored_attempt_count, 'lessonRows': computed_lesson_rows, 'attemptRows': attempt_rows, 'topicRows': topic_rows, 'topicRowsByLesson': topic_rows_by_lesson, 'minutesThisWeek': int((time_spent_seconds_this_week + 59) // 60) if time_spent_seconds_this_week > 0 else 0, 'progress': max(0, min(100, round(progress))), 'averageScore': round(total_difficulty_correct / total_difficulty_questions * 100) if total_difficulty_questions > 0 else round(correct_answers / question_attempts * 100) if question_attempts > 0 else 0, 'difficultyScore': difficulty_score, 'scoreSplit': score_split, 'lastActivity': last_submitted_at, 'learningActivityDays': [str(day) for day in course.get('learning_activity_days', []) if str(day or '').strip()] if isinstance(course.get('learning_activity_days'), list) else []}
    return stats


def _normalize_trial_override_mode(value: Any) -> str:
    mode = str(value or '').strip().lower()
    if mode in TRIAL_OVERRIDE_MODES:
        return mode
    return 'auto'


def _extract_trial_override(user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(user, dict):
        return {'mode': 'auto', 'updated_at': None, 'updated_by': None, 'reason': None}
    override = user.get('admin_trial_override')
    if not isinstance(override, dict):
        return {'mode': _normalize_trial_override_mode(user.get('trial_override_mode')), 'updated_at': None, 'updated_by': None, 'reason': None}
    return {'mode': _normalize_trial_override_mode(override.get('mode')), 'updated_at': str(override.get('updated_at') or '').strip() or None, 'updated_by': str(override.get('updated_by') or '').strip() or None, 'reason': str(override.get('reason') or '').strip() or None}


def _resolve_effective_trial_used(trial_used_from_enrollments: bool, override_mode: Any) -> Dict[str, Any]:
    normalized_mode = _normalize_trial_override_mode(override_mode)
    if normalized_mode == 'used':
        return {'trial_used': True, 'trial_status_source': 'admin_override'}
    if normalized_mode == 'available':
        return {'trial_used': False, 'trial_status_source': 'admin_override'}
    return {'trial_used': bool(trial_used_from_enrollments), 'trial_status_source': 'enrollment'}


async def _set_user_trial_override(data_service, user_id: str, mode: str, updated_by: str, reason: Optional[str]=None) -> Dict[str, Any]:
    normalized_user_id = str(user_id or '').strip()
    normalized_mode = _normalize_trial_override_mode(mode)
    normalized_updated_by = str(updated_by or '').strip()
    normalized_reason = str(reason or '').strip() or None
    if not normalized_user_id:
        raise HTTPException(status_code=400, detail='user_id is required')
    if not normalized_updated_by:
        raise HTTPException(status_code=400, detail='admin_user_id is required')
    now = datetime.utcnow().isoformat()
    user = await data_service.get_user(normalized_user_id)
    item = dict(user or {})
    if not str(item.get('user_id') or '').strip():
        item['user_id'] = normalized_user_id
    if not str(item.get('email') or '').strip():
        item['email'] = f'{normalized_user_id}@example.com'
    if not str(item.get('name') or '').strip():
        item['name'] = f'User {normalized_user_id}'
    if not str(item.get('role') or '').strip():
        item['role'] = 'student'
    if not str(item.get('status') or '').strip():
        item['status'] = 'active'
    if not str(item.get('created_at') or '').strip():
        item['created_at'] = now
    trial_override = {'mode': normalized_mode, 'updated_at': now, 'updated_by': normalized_updated_by, 'reason': normalized_reason}
    item['admin_trial_override'] = trial_override
    item['updated_at'] = now
    data_service.users_table.put_item(Item=data_service._convert_floats_to_decimal(item))
    return trial_override


def _safe_float(value: Any, default: float=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_chat_energy_response(status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(status, dict):
        return {'daily_limit_thb': 0.0, 'used_thb': 0.0, 'remaining_thb': 0.0, 'remaining_percent': 0.0, 'is_exhausted': True, 'daily_limit_override_thb': None, 'daily_adjustment_thb': 0.0, 'limit_source': 'global_default', 'usage_date': datetime.utcnow().date().isoformat(), 'request_count': 0, 'policy_updated_at': None, 'policy_updated_by': None, 'policy_reason': None, 'platform_updated_at': None, 'platform_updated_by': None, 'platform_reason': None, 'default_daily_limit_thb': 0.0}
    return {'daily_limit_thb': _safe_float(status.get('daily_limit_thb'), 0.0), 'used_thb': _safe_float(status.get('used_thb'), 0.0), 'remaining_thb': _safe_float(status.get('remaining_thb'), 0.0), 'remaining_percent': _safe_float(status.get('remaining_percent'), 0.0), 'is_exhausted': bool(status.get('is_exhausted')), 'daily_limit_override_thb': _safe_float(status.get('daily_limit_override_thb'), 0.0) if status.get('daily_limit_override_thb') is not None else None, 'daily_adjustment_thb': _safe_float(status.get('daily_adjustment_thb'), 0.0), 'limit_source': str(status.get('limit_source') or 'global_default'), 'usage_date': str(status.get('usage_date') or datetime.utcnow().date().isoformat()), 'request_count': int(_safe_float(status.get('request_count'), 0)), 'policy_updated_at': str(status.get('policy_updated_at') or '').strip() or None, 'policy_updated_by': str(status.get('policy_updated_by') or '').strip() or None, 'policy_reason': str(status.get('policy_reason') or '').strip() or None, 'platform_updated_at': str(status.get('platform_updated_at') or '').strip() or None, 'platform_updated_by': str(status.get('platform_updated_by') or '').strip() or None, 'platform_reason': str(status.get('platform_reason') or '').strip() or None, 'default_daily_limit_thb': _safe_float(status.get('default_daily_limit_thb'), 0.0)}


def _build_payment_order_id(paid_at: Any, payment_intent_id: Any) -> Optional[str]:
    intent_id = str(payment_intent_id or '').strip()
    if not intent_id:
        return None
    paid_at_dt = _parse_iso_datetime(paid_at) or datetime.utcnow()
    paid_at_dt = paid_at_dt.replace(tzinfo=timezone.utc).astimezone(PAYMENT_TIME_ZONE)
    suffix = hashlib.sha1(intent_id.encode('utf-8')).hexdigest()[:4].upper()
    return f"TM{paid_at_dt.strftime('%Y%m%d')}-{suffix}"


def _latest_charge_from_intent(intent: Dict[str, Any]) -> Dict[str, Any]:
    latest_charge = intent.get('latest_charge')
    if isinstance(latest_charge, dict):
        return latest_charge
    if isinstance(latest_charge, str) and latest_charge.strip():
        return {'id': latest_charge.strip()}
    return {}


async def _stripe_receipt_fields_from_payment_intent(payment_intent_id: Any, secret_key: str) -> Dict[str, Optional[str]]:
    intent_id = str(payment_intent_id or '').strip()
    if not intent_id:
        return {'stripe_charge_id': None, 'receipt_number': None, 'receipt_url': None}
    intent = await _stripe_request(method='GET', path=f'/payment_intents/{intent_id}', secret_key=secret_key, params={'expand[]': 'latest_charge'})
    latest_charge = _latest_charge_from_intent(intent)
    stripe_charge_id = str(latest_charge.get('id') or '').strip() or None
    receipt_number = str(latest_charge.get('receipt_number') or '').strip() or None
    receipt_url = str(latest_charge.get('receipt_url') or '').strip() or None
    if stripe_charge_id and (not receipt_number or not receipt_url):
        charge = await _stripe_request(method='GET', path=f'/charges/{stripe_charge_id}', secret_key=secret_key)
        receipt_number = receipt_number or str(charge.get('receipt_number') or '').strip() or None
        receipt_url = receipt_url or str(charge.get('receipt_url') or '').strip() or None
    return {'stripe_charge_id': stripe_charge_id, 'receipt_number': receipt_number, 'receipt_url': receipt_url}


async def _hydrate_payment_history_receipts(rows: List[Dict[str, Any]]) -> None:
    try:
        settings = get_settings()
    except Exception:
        return
    secret_key = str(getattr(settings, 'stripe_private_key', '') or '').strip()
    if not secret_key:
        return
    receipt_cache: Dict[str, Dict[str, Optional[str]]] = {}
    for row in rows:
        if str(row.get('receipt_url') or '').strip() and str(row.get('receipt_number') or '').strip():
            continue
        payment_intent_id = str(row.get('payment_intent_id') or '').strip()
        if not payment_intent_id:
            continue
        if str(row.get('payment_provider') or '').strip().lower() != 'stripe':
            continue
        if str(row.get('payment_status') or '').strip().lower() != 'succeeded':
            continue
        try:
            if payment_intent_id not in receipt_cache:
                receipt_cache[payment_intent_id] = await _stripe_receipt_fields_from_payment_intent(payment_intent_id, secret_key)
            fields = receipt_cache[payment_intent_id]
        except Exception as exc:
            app_logger.warning(f'Unable to hydrate payment receipt for payment_intent_id={payment_intent_id}: {exc}')
            continue
        if fields.get('stripe_charge_id') and (not row.get('stripe_charge_id')):
            row['stripe_charge_id'] = fields['stripe_charge_id']
        if fields.get('receipt_number') and (not row.get('receipt_number')):
            row['receipt_number'] = fields['receipt_number']
        if fields.get('receipt_url'):
            row['receipt_url'] = fields['receipt_url']


def _to_payment_event(data: Dict[str, Any]) -> Dict[str, Any]:
    schedule = _build_enrollment_schedule(started_at_raw=data.get('started_at') or data.get('enrolled_at') or data.get('paid_at'), expires_at_raw=data.get('expires_at'), duration_months_raw=data.get('duration_months'))
    paid_amount = data.get('paid_amount_thb')
    try:
        paid_amount = float(paid_amount) if paid_amount is not None else None
    except Exception:
        paid_amount = None
    payment_status = str(data.get('payment_status') or '').strip()
    event = {'order_id': data.get('order_id') or _build_payment_order_id(data.get('paid_at') or data.get('enrolled_at'), data.get('payment_intent_id')), 'payment_provider': data.get('payment_provider') or 'stripe', 'payment_type': data.get('payment_type') or 'promptpay', 'payment_intent_id': data.get('payment_intent_id'), 'stripe_charge_id': data.get('stripe_charge_id'), 'receipt_number': data.get('receipt_number'), 'receipt_url': data.get('receipt_url'), 'payment_status': payment_status or ('succeeded' if paid_amount is not None else 'active'), 'paid_amount_thb': paid_amount, 'paid_currency': data.get('paid_currency') or 'THB', 'billing_email': data.get('billing_email'), 'plan_label': data.get('plan_label'), 'duration_months': schedule['duration_months'], 'paid_at': data.get('paid_at') or data.get('enrolled_at'), 'started_at': schedule['started_at'], 'expires_at': schedule['expires_at']}
    for field_name in ('payment_success_email_status', 'payment_success_email_job_id', 'payment_success_email_sent_at'):
        value = data.get(field_name)
        if value:
            event[field_name] = value
    return event


def _normalize_payment_history(enrollment: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = enrollment.get('payment_history')
    normalized_events: List[Dict[str, Any]] = []
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                normalized_events.append(_to_payment_event(event))
    if not normalized_events:
        payment_intent_id = str(enrollment.get('payment_intent_id') or '').strip()
        paid_amount = enrollment.get('paid_amount_thb')
        payment_status = str(enrollment.get('payment_status') or '').strip().lower()
        if payment_intent_id or paid_amount is not None or payment_status == 'succeeded':
            normalized_events.append(_to_payment_event(enrollment))
    return normalized_events


def _find_payment_event_by_intent(payment_events: List[Dict[str, Any]], payment_intent_id: Any) -> Optional[Dict[str, Any]]:
    target_intent_id = str(payment_intent_id or '').strip()
    if not target_intent_id:
        return None
    for event in payment_events:
        if not isinstance(event, dict):
            continue
        event_intent_id = str(event.get('payment_intent_id') or '').strip()
        if event_intent_id == target_intent_id:
            return event
    return None


def _extract_stripe_signature_parts(signature_header: str) -> Dict[str, List[str]]:
    parts: Dict[str, List[str]] = {}
    for item in str(signature_header or '').split(','):
        key, separator, value = item.partition('=')
        if not separator:
            continue
        parts.setdefault(key.strip(), []).append(value.strip())
    return parts


def _verify_stripe_webhook_signature(payload: bytes, signature_header: str, webhook_secret: str) -> None:
    signature_parts = _extract_stripe_signature_parts(signature_header)
    timestamps = signature_parts.get('t') or []
    signatures = signature_parts.get('v1') or []
    if not timestamps or not signatures:
        raise HTTPException(status_code=400, detail='Invalid Stripe signature')
    signed_payload = b'.'.join([timestamps[0].encode('utf-8'), payload])
    expected_signature = hmac.new(webhook_secret.encode('utf-8'), signed_payload, hashlib.sha256).hexdigest()
    if not any((hmac.compare_digest(expected_signature, signature) for signature in signatures)):
        raise HTTPException(status_code=400, detail='Invalid Stripe signature')


def _payment_amount_thb_from_intent(intent: Dict[str, Any]) -> Optional[float]:
    amount_received_raw = intent.get('amount_received')
    amount_raw = intent.get('amount')
    try:
        if amount_received_raw is not None:
            return round(float(amount_received_raw) / 100.0, 2)
        if amount_raw is not None:
            return round(float(amount_raw) / 100.0, 2)
    except Exception:
        return None
    return None


def _paid_at_from_intent(intent: Dict[str, Any]) -> str:
    paid_at = datetime.now(timezone.utc).isoformat()
    try:
        created_ts = int(intent.get('created'))
        if created_ts > 0:
            paid_at = datetime.fromtimestamp(created_ts, timezone.utc).isoformat()
    except Exception:
        pass
    return paid_at


async def _complete_promptpay_payment(*, payment_intent_id: str, data_service, expected_user_id: Optional[str]=None, expected_course_id: Optional[str]=None) -> Dict[str, Any]:
    settings = get_settings()
    if not settings.stripe_private_key:
        raise HTTPException(status_code=500, detail='Stripe private key is not configured')
    payment_intent_id = str(payment_intent_id or '').strip()
    if not payment_intent_id:
        raise HTTPException(status_code=400, detail='payment_intent_id is required')
    intent = await _stripe_request(method='GET', path=f'/payment_intents/{payment_intent_id}', secret_key=settings.stripe_private_key, params={'expand[]': 'latest_charge'})
    metadata = intent.get('metadata') if isinstance(intent.get('metadata'), dict) else {}
    user_id = str(metadata.get('user_id') or expected_user_id or '').strip()
    course_id = str(metadata.get('course_id') or expected_course_id or '').strip()
    if not user_id or not course_id:
        raise HTTPException(status_code=400, detail='Payment is missing required user or course metadata')
    if expected_user_id and user_id != str(expected_user_id).strip():
        raise HTTPException(status_code=400, detail='Payment does not belong to this user')
    if expected_course_id and course_id != str(expected_course_id).strip():
        raise HTTPException(status_code=400, detail='Payment does not belong to this course')
    payment_status = str(intent.get('status') or '').strip()
    if payment_status != 'succeeded':
        return {'payment_intent_id': payment_intent_id, 'payment_status': payment_status or 'unknown', 'enrolled': False, 'message': 'Payment is not completed yet'}
    existing_enrollment_with_schedule = await _get_existing_enrollment_with_schedule(data_service=data_service, user_id=user_id, course_id=course_id)
    paid_at = _paid_at_from_intent(intent)
    latest_charge = _latest_charge_from_intent(intent)
    stripe_charge_id = str(latest_charge.get('id') or '').strip() or None
    receipt_number = str(latest_charge.get('receipt_number') or '').strip() or None
    receipt_url = str(latest_charge.get('receipt_url') or '').strip() or None
    duration_months = _parse_positive_int(metadata.get('duration_months'))
    schedule = _build_enrollment_schedule(started_at_raw=paid_at, expires_at_raw=None, duration_months_raw=duration_months)
    enrollment_data = {'progress': 0, 'completed_quizzes': 0, 'total_quizzes': 0, 'completed_questions': 0, 'total_questions': 0, 'last_activity': 'เพิ่งชำระเงินและเข้าร่วม', 'enrollment_source': 'payment', 'order_id': _build_payment_order_id(paid_at, payment_intent_id), 'payment_provider': 'stripe', 'payment_type': 'promptpay', 'payment_intent_id': payment_intent_id, 'stripe_charge_id': stripe_charge_id, 'receipt_number': receipt_number, 'receipt_url': receipt_url, 'payment_status': payment_status, 'paid_amount_thb': _payment_amount_thb_from_intent(intent), 'paid_currency': str(intent.get('currency') or 'THB').upper(), 'billing_email': str(intent.get('receipt_email') or '').strip(), 'plan_label': str(metadata.get('plan_label') or '').strip(), 'duration_months': duration_months, 'paid_at': paid_at, 'started_at': schedule['started_at'], 'expires_at': schedule['expires_at']}
    if existing_enrollment_with_schedule:
        existing_enrollment = existing_enrollment_with_schedule['enrollment']
        enrollment_id = str(existing_enrollment.get('enrollment_id') or '').strip()
        if not enrollment_id:
            raise HTTPException(status_code=500, detail='Existing enrollment is missing enrollment_id')
        payment_history = _normalize_payment_history(existing_enrollment)
        existing_payment_event = _find_payment_event_by_intent(payment_events=payment_history, payment_intent_id=payment_intent_id)
        if existing_payment_event:
            return {'payment_intent_id': payment_intent_id, 'order_id': existing_payment_event.get('order_id') or _build_payment_order_id(existing_payment_event.get('paid_at'), payment_intent_id), 'receipt_url': existing_payment_event.get('receipt_url'), 'payment_status': payment_status, 'enrolled': True, 'enrollment_id': enrollment_id, 'message': 'Payment already confirmed for this enrollment'}
        current_schedule = existing_enrollment_with_schedule.get('schedule') or {}
        current_expires_at = _parse_iso_datetime(current_schedule.get('expires_at'))
        paid_at_dt = _parse_iso_datetime(paid_at) or datetime.utcnow()
        renewal_start_dt = paid_at_dt
        if current_expires_at and current_expires_at > paid_at_dt:
            renewal_start_dt = current_expires_at
        renewal_schedule = _build_enrollment_schedule(started_at_raw=renewal_start_dt.isoformat(), expires_at_raw=None, duration_months_raw=duration_months)
        enrollment_data['started_at'] = renewal_schedule['started_at']
        enrollment_data['expires_at'] = renewal_schedule['expires_at']
        payment_event = _to_payment_event(enrollment_data)
        payment_history.append(payment_event)
        renewal_updates = {'status': 'active', 'order_id': enrollment_data['order_id'], 'payment_provider': enrollment_data['payment_provider'], 'payment_type': enrollment_data['payment_type'], 'payment_intent_id': enrollment_data['payment_intent_id'], 'stripe_charge_id': enrollment_data['stripe_charge_id'], 'receipt_number': enrollment_data['receipt_number'], 'receipt_url': enrollment_data['receipt_url'], 'payment_status': enrollment_data['payment_status'], 'paid_amount_thb': enrollment_data['paid_amount_thb'], 'paid_currency': enrollment_data['paid_currency'], 'billing_email': enrollment_data['billing_email'], 'plan_label': enrollment_data['plan_label'], 'duration_months': enrollment_data['duration_months'], 'paid_at': enrollment_data['paid_at'], 'started_at': enrollment_data['started_at'], 'expires_at': enrollment_data['expires_at'], 'payment_history': payment_history, 'last_activity': 'ต่ออายุคอร์สแล้ว'}
        success = await data_service.update_enrollment(enrollment_id, renewal_updates)
        if not success:
            raise HTTPException(status_code=500, detail='Failed to renew existing enrollment')
        return {'payment_intent_id': payment_intent_id, 'order_id': payment_event.get('order_id'), 'receipt_url': payment_event.get('receipt_url'), 'payment_status': payment_status, 'enrolled': True, 'enrollment_id': enrollment_id, 'message': 'Payment verified and enrollment renewed'}
    payment_event = _to_payment_event(enrollment_data)
    enrollment_id = await data_service.enroll_user_in_course(user_id=user_id, course_id=course_id, enrollment_data={**enrollment_data, 'payment_history': [payment_event]})
    return {'payment_intent_id': payment_intent_id, 'order_id': payment_event.get('order_id'), 'receipt_url': payment_event.get('receipt_url'), 'payment_status': payment_status, 'enrolled': True, 'enrollment_id': enrollment_id, 'message': 'Payment verified and enrollment completed'}


async def get_data_service():
    return get_db_service()


async def get_chat_service() -> ChatService:
    return ChatService()


async def get_terms_of_use():
    """Get Terms of Use (Thai summary)."""
    return LegalDocumentResponse(document_name='เงื่อนไขการใช้งาน (Terms of Use)', version=LEGAL_DOC_VERSION, last_updated=LEGAL_DOC_LAST_UPDATED, summary='เอกสารนี้กำหนดเงื่อนไขการใช้งาน TEWMai แพลตฟอร์มฝึกโจทย์ ข้อสอบจำลอง วิเคราะห์ผล และผู้ช่วย AI สำหรับการเรียนรู้', sections=[LegalSection(heading='การยอมรับเงื่อนไข', details=['เมื่อผู้ใช้เข้าเว็บไซต์ ลงทะเบียน ชำระเงิน หรือใช้ฟีเจอร์ใด ๆ ของ TEWMai ถือว่าผู้ใช้ได้อ่าน เข้าใจ และยอมรับเงื่อนไขการใช้งานฉบับนี้แล้ว', 'หากผู้ใช้ไม่ยอมรับเงื่อนไขข้อใดข้อหนึ่ง ควรงดใช้บริการหรือหยุดใช้งานบัญชีจนกว่าจะเข้าใจรายละเอียดครบถ้วน', 'ในกรณีที่ผู้ใช้เป็นผู้เยาว์ ผู้ปกครองหรือผู้แทนโดยชอบธรรมควรรับทราบและยินยอมต่อการใช้งานบริการ']), LegalSection(heading='รายละเอียดบริการ', details=['TEWMai ให้บริการแบบฝึกหัด ข้อสอบจำลอง บทเรียน รายงานวิเคราะห์คะแนน และผู้ช่วย AI เพื่ออธิบายแนวคิดในการทำโจทย์', 'ระบบอาจมีฟีเจอร์สำหรับอัปโหลดภาพโจทย์หรือวิธีทำ เพื่อให้ AI ช่วยอ่าน วิเคราะห์ และแนะนำแนวทางการเรียนรู้', 'บางฟีเจอร์ คอร์ส หรือรายงานเชิงลึกอาจเปิดให้ใช้เฉพาะผู้ใช้ที่สมัครคอร์ส ชำระเงิน หรือมีสิทธิ์เข้าถึงตามแพ็กเกจที่กำหนด', 'บริการมีเป้าหมายเพื่อสนับสนุนการเรียนรู้ ไม่ใช่การรับประกันผลสอบ คะแนนสอบ หรือการเข้าเรียนในสถาบันใดสถาบันหนึ่ง']), LegalSection(heading='บัญชีผู้ใช้และสิทธิ์การเข้าใช้งาน', details=['ผู้ใช้ต้องให้ข้อมูลบัญชีที่ถูกต้อง เป็นปัจจุบัน และไม่แอบอ้างเป็นบุคคลอื่น', 'ผู้ใช้ต้องรักษารหัสผ่าน ลิงก์เข้าสู่ระบบ และอุปกรณ์ที่ใช้เข้าใช้งานให้ปลอดภัย หากพบการใช้งานผิดปกติควรแจ้งผู้ให้บริการโดยเร็ว', 'สิทธิ์การเข้าถึงคอร์สหรือฟีเจอร์เป็นสิทธิ์เฉพาะบัญชี ไม่ควรขาย ให้เช่า โอน หรือแบ่งปันบัญชีให้ผู้อื่นใช้งานโดยไม่ได้รับอนุญาต']), LegalSection(heading='หน้าที่ของผู้ใช้งาน', details=['ผู้ใช้ต้องใช้บริการอย่างสุจริต ถูกกฎหมาย และไม่กระทำการที่รบกวนการทำงานของระบบหรือผู้ใช้อื่น', 'ผู้ใช้รับผิดชอบต่อความถูกต้อง ความเหมาะสม และสิทธิ์ในการใช้ข้อมูล รูปภาพ ไฟล์ หรือข้อความที่อัปโหลดเข้าสู่ระบบ', 'ผู้ใช้ไม่ควรอัปโหลดข้อมูลส่วนบุคคลที่ไม่จำเป็น ข้อมูลของผู้อื่นโดยไม่ได้รับอนุญาต หรือเนื้อหาที่ละเมิดกฎหมายและสิทธิของบุคคลภายนอก']), LegalSection(heading='การใช้ผู้ช่วย AI และผลลัพธ์การเรียน', details=['คำตอบ คำอธิบาย และคำแนะนำจาก AI เป็นเครื่องมือช่วยเรียนรู้ ผู้ใช้ควรใช้วิจารณญาณ ตรวจสอบความถูกต้อง และปรึกษาครูหรือผู้ปกครองเมื่อจำเป็น', 'AI อาจตีความโจทย์ ภาพลายมือ หรือบริบทผิดพลาดได้ โดยเฉพาะภาพที่ไม่ชัดเจน ข้อมูลไม่ครบ หรือโจทย์ที่มีหลายวิธีคิด', 'ผู้ให้บริการอาจนำข้อมูลการใช้งานที่เหมาะสมไปใช้ปรับปรุงคุณภาพระบบ การตรวจจับข้อผิดพลาด และประสบการณ์การเรียน โดยเป็นไปตามนโยบายความเป็นส่วนตัว']), LegalSection(heading='ข้อมูล การอัปโหลด และความเป็นส่วนตัว', details=['ผู้ใช้ยังคงเป็นเจ้าของเนื้อหา ไฟล์ ภาพโจทย์ และข้อมูลการเรียนที่ตนเองอัปโหลดหรือสร้างขึ้นผ่านระบบ', 'TEWMai จะเข้าถึงและประมวลผลข้อมูลเท่าที่จำเป็นเพื่อให้บริการ เช่น การตรวจคำตอบ การวิเคราะห์คะแนน การแสดงประวัติการเรียน และการช่วยเหลือผ่าน AI', 'รายละเอียดการเก็บ ใช้ เปิดเผย เก็บรักษา และลบข้อมูลส่วนบุคคลระบุไว้ในนโยบายความเป็นส่วนตัวของเรา']), LegalSection(heading='เงื่อนไขการชำระเงิน คอร์ส และแพ็กเกจ', details=['ราคา ระยะเวลาเข้าถึง เนื้อหาคอร์ส ฟีเจอร์ และสิทธิ์การใช้งานเป็นไปตามข้อมูลที่แสดงในหน้าคอร์สหรือหน้าชำระเงิน ณ เวลาที่ผู้ใช้สมัคร', 'ผู้ให้บริการอาจปรับราคา แพ็กเกจ โปรโมชัน หรือโครงสร้างฟีเจอร์ในอนาคต โดยการเปลี่ยนแปลงจะไม่มีผลย้อนหลังต่อสิทธิ์ที่ผู้ใช้ชำระเงินและได้รับยืนยันแล้ว เว้นแต่ระบุไว้เป็นอย่างอื่น', 'หากมีการต่ออายุ อัปเกรด หรือเปลี่ยนแพ็กเกจ ระบบอาจคำนวณสิทธิ์หรือระยะเวลาใช้งานตามเงื่อนไขที่ประกาศในขณะทำรายการ']), LegalSection(heading='การคืนเงินและการยกเลิกสิทธิ์', details=['นโยบายการคืนเงินหรือยกเลิกสิทธิ์เป็นไปตามเงื่อนไขที่แสดงในหน้าชำระเงิน ประกาศของระบบ และข้อกำหนดของผู้ให้บริการชำระเงินที่เกี่ยวข้อง', 'ผู้ใช้ควรตรวจสอบชื่อคอร์ส ราคา ระยะเวลา และรายละเอียดสิทธิ์ก่อนยืนยันการชำระเงิน', 'หากพบการเรียกเก็บเงินผิดพลาดหรือเข้าถึงคอร์สไม่ได้หลังชำระเงิน ผู้ใช้ควรติดต่อทีมสนับสนุนพร้อมหลักฐานการชำระเงินเพื่อให้ตรวจสอบ']), LegalSection(heading='การใช้งานที่ห้าม', details=['ห้ามใช้ระบบเพื่อกระทำการผิดกฎหมาย ฉ้อโกง คุกคาม ละเมิดสิทธิผู้อื่น หรือเผยแพร่เนื้อหาที่ไม่เหมาะสม', 'ห้ามใช้บอท สคริปต์ การขูดข้อมูล หรือวิธีอัตโนมัติอื่นใดเพื่อดึงข้อมูลจำนวนมาก หลีกเลี่ยงข้อจำกัด หรือสร้างภาระเกินสมควรต่อระบบ', 'ห้ามพยายามเจาะระบบ ข้ามมาตรการรักษาความปลอดภัย แก้ไข ดัดแปลง ทำ reverse engineer หรือเข้าถึงข้อมูลที่ไม่ได้รับอนุญาต', 'การละเมิดเงื่อนไขอาจทำให้ถูกจำกัด ระงับ หรือยกเลิกบัญชี รวมถึงอาจดำเนินการตามกฎหมายหากมีความจำเป็น']), LegalSection(heading='ความพร้อมใช้งานและการเปลี่ยนแปลงบริการ', details=['ผู้ให้บริการพยายามดูแลให้ระบบใช้งานได้ต่อเนื่อง แต่ไม่รับประกันว่าบริการจะไม่มีข้อผิดพลาด หยุดชะงัก หรือพร้อมใช้งานตลอดเวลา', 'ระบบอาจหยุดให้บริการชั่วคราวเนื่องจากการบำรุงรักษา การอัปเดต เหตุขัดข้องทางเทคนิค หรือปัจจัยภายนอก เช่น ผู้ให้บริการคลาวด์หรือระบบชำระเงิน', 'ผู้ให้บริการอาจปรับปรุง แก้ไข เพิ่ม ลด หรือยุติบางฟีเจอร์เมื่อจำเป็น โดยจะพยายามสื่อสารการเปลี่ยนแปลงที่มีผลสำคัญต่อผู้ใช้']), LegalSection(heading='ข้อจำกัดความรับผิด', details=['TEWMai ให้บริการตามสภาพที่มีอยู่และตามขอบเขตที่ระบบรองรับ ผู้ใช้ยอมรับความเสี่ยงในการใช้ข้อมูล คำแนะนำ และผลวิเคราะห์เพื่อประกอบการเรียนรู้', 'ผู้ให้บริการไม่รับผิดชอบต่อความเสียหายทางอ้อม การสูญเสียโอกาส ผลสอบไม่เป็นไปตามคาด หรือความเสียหายที่เกิดจากการใช้งานผิดวัตถุประสงค์', 'ในขอบเขตสูงสุดที่กฎหมายอนุญาต ความรับผิดของผู้ให้บริการจะจำกัดตามมูลค่าบริการที่เกี่ยวข้องกับเหตุการณ์นั้น']), LegalSection(heading='การเปลี่ยนแปลงเงื่อนไข', details=['ผู้ให้บริการอาจปรับปรุงเงื่อนไขการใช้งานเป็นครั้งคราว เพื่อให้สอดคล้องกับฟีเจอร์ใหม่ ข้อกำหนดทางกฎหมาย หรือวิธีดำเนินงานของระบบ', 'เมื่อมีการเปลี่ยนแปลงที่สำคัญ ระบบอาจแจ้งผ่านหน้าเว็บไซต์ แอป อีเมล หรือช่องทางที่เหมาะสม พร้อมปรับวันที่อัปเดตของเอกสาร', 'การใช้งานบริการต่อหลังจากเงื่อนไขใหม่มีผล ถือว่าผู้ใช้ยอมรับเงื่อนไขฉบับที่ปรับปรุงแล้ว']), LegalSection(heading='ติดต่อเรา', details=['หากมีคำถามเกี่ยวกับเงื่อนไขการใช้งาน การชำระเงิน สิทธิ์การเข้าถึง หรือปัญหาเกี่ยวกับบัญชี สามารถติดต่อทีมสนับสนุนได้ที่ support@tewmai.com, LINE @tewmai หรือ Facebook: TEWMai - ติวอัจฉริยะด้วย AI', 'เพื่อให้ตรวจสอบได้รวดเร็ว โปรดระบุอีเมลบัญชี ชื่อคอร์ส หลักฐานการชำระเงิน หรือรายละเอียดปัญหาที่เกี่ยวข้องเมื่อส่งคำขอ'])], contact_email='support@tewmai.com')


async def get_privacy_policy():
    """Get Privacy Policy (Thai summary)."""
    return LegalDocumentResponse(document_name='นโยบายความเป็นส่วนตัว (Privacy Policy)', version=LEGAL_DOC_VERSION, last_updated=LEGAL_DOC_LAST_UPDATED, summary='เอกสารนี้อธิบายวิธีที่ TEWMai เก็บ ใช้ เปิดเผย เก็บรักษา และคุ้มครองข้อมูลส่วนบุคคลของผู้เรียน ผู้ปกครอง และผู้ใช้งานระบบ', sections=[LegalSection(heading='ข้อมูลที่เราเก็บ', details=['ข้อมูลบัญชี เช่น ชื่อ อีเมล เบอร์โทรศัพท์ (ถ้ามี) รูปโปรไฟล์ ช่องทางเข้าสู่ระบบ และข้อมูลที่ใช้ระบุตัวตนของบัญชี', 'ข้อมูลการสมัครและการชำระเงิน เช่น คอร์สที่สมัคร สถานะการชำระเงิน ประวัติการต่ออายุ ใบเสร็จหรือหลักฐานที่เกี่ยวข้อง โดยอาจประมวลผลผ่านผู้ให้บริการชำระเงินภายนอก', 'ข้อมูลการเรียน เช่น คะแนน คำตอบ เวลาใช้งาน จำนวนครั้งที่ทำโจทย์ ความแม่นยำรายหัวข้อ ประวัติการเรียน และรายงานวิเคราะห์ผล', 'ข้อมูลทางเทคนิค เช่น ประเภทอุปกรณ์ เบราว์เซอร์ หมายเลข IP บันทึกการใช้งาน เหตุขัดข้อง และข้อมูลคุกกี้หรือเทคโนโลยีที่คล้ายกัน']), LegalSection(heading='ข้อมูลจากภาพโจทย์ ไฟล์ และผู้ช่วย AI', details=['เมื่อผู้ใช้อัปโหลดภาพโจทย์ วิธีทำ ไฟล์ หรือข้อความ ระบบอาจประมวลผลข้อมูลดังกล่าวเพื่ออ่านโจทย์ ตรวจคำตอบ วิเคราะห์แนวคิด และสร้างคำแนะนำจาก AI', 'ข้อมูลที่ส่งให้ผู้ช่วย AI อาจประกอบด้วยข้อความที่ผู้ใช้พิมพ์ รูปภาพ คำตอบ คะแนน และบริบทการเรียนที่จำเป็นต่อการตอบคำถาม', 'ผู้ใช้ควรหลีกเลี่ยงการอัปโหลดข้อมูลส่วนบุคคลที่ไม่จำเป็น เช่น เลขบัตรประชาชน ที่อยู่ ข้อมูลสุขภาพ หรือข้อมูลของบุคคลอื่นที่ไม่ได้รับอนุญาต', 'เราใช้ข้อมูลดังกล่าวเพื่อให้บริการและปรับปรุงคุณภาพระบบ ไม่ได้ออกแบบมาเพื่อใช้เป็นเครื่องมือจัดเก็บข้อมูลอ่อนไหวหรือเอกสารสำคัญส่วนตัว']), LegalSection(heading='วัตถุประสงค์การใช้ข้อมูล', details=['ให้บริการบัญชีผู้ใช้ คอร์ส แบบฝึกหัด ข้อสอบจำลอง ผู้ช่วย AI รายงานผล และฟีเจอร์ที่ผู้ใช้ร้องขอ', 'วิเคราะห์พัฒนาการ จุดแข็ง จุดที่ควรฝึกเพิ่ม และปรับคำแนะนำการเรียนให้เหมาะสมกับผู้ใช้แต่ละคน', 'ตรวจสอบการชำระเงิน จัดการสิทธิ์เข้าถึงคอร์ส ให้การสนับสนุนลูกค้า และแก้ไขปัญหาทางเทคนิค', 'รักษาความปลอดภัย ป้องกันการใช้งานผิดปกติ ตรวจจับการละเมิดเงื่อนไข และปรับปรุงเสถียรภาพของระบบ', 'พัฒนาคุณภาพโมเดล กระบวนการตรวจโจทย์ และประสบการณ์ใช้งาน โดยใช้ข้อมูลเท่าที่จำเป็นและลดการระบุตัวตนเมื่อเหมาะสม']), LegalSection(heading='การเปิดเผยข้อมูล', details=['เราไม่ขายหรือให้เช่าข้อมูลส่วนบุคคลของผู้ใช้แก่บุคคลที่สาม', 'เราอาจเปิดเผยข้อมูลเท่าที่จำเป็นต่อผู้ให้บริการโครงสร้างพื้นฐาน ระบบยืนยันตัวตน ระบบชำระเงิน ระบบวิเคราะห์ข้อผิดพลาด หรือบริการ AI ที่ช่วยให้ระบบทำงานได้', 'ผู้ให้บริการภายนอกที่เกี่ยวข้องจะได้รับข้อมูลเฉพาะส่วนที่จำเป็นต่อการให้บริการ และต้องปฏิบัติตามมาตรการรักษาความลับและความปลอดภัยที่เหมาะสม', 'เราอาจเปิดเผยข้อมูลเมื่อกฎหมาย คำสั่งหน่วยงานรัฐ หรือกระบวนการทางกฎหมายกำหนด หรือเมื่อจำเป็นเพื่อปกป้องสิทธิ ความปลอดภัย และความมั่นคงของระบบ']), LegalSection(heading='ความปลอดภัยของข้อมูล', details=['เราใช้มาตรการทางเทคนิคและองค์กรที่เหมาะสม เช่น การเชื่อมต่อที่ปลอดภัย การจำกัดสิทธิ์เข้าถึง และการตรวจสอบระบบ เพื่อลดความเสี่ยงจากการเข้าถึงโดยไม่ได้รับอนุญาต', 'ข้อมูลสำคัญบางประเภท เช่น token หรือ credential ที่ใช้เชื่อมต่อบริการ จะถูกจัดเก็บและควบคุมตามแนวทางความปลอดภัยของระบบ', 'แม้เราจะพยายามปกป้องข้อมูลอย่างเหมาะสม แต่ไม่มีระบบออนไลน์ใดปลอดภัยสมบูรณ์ ผู้ใช้ควรรักษารหัสผ่าน อุปกรณ์ และบัญชีอีเมลของตนเองให้ปลอดภัยด้วย']), LegalSection(heading='การเก็บรักษาและการลบข้อมูล', details=['เราจะเก็บข้อมูลส่วนบุคคลเท่าที่จำเป็นต่อการให้บริการ การปฏิบัติตามกฎหมาย การบัญชี การตรวจสอบข้อพิพาท และการรักษาความปลอดภัยของระบบ', 'ข้อมูลการเรียนและประวัติการใช้งานอาจถูกเก็บไว้ตลอดระยะเวลาที่บัญชียังใช้งาน เพื่อให้ผู้ใช้ดูพัฒนาการย้อนหลังและใช้รายงานวิเคราะห์ได้ต่อเนื่อง', 'ข้อมูลชั่วคราวจากการประมวลผล เช่น ไฟล์หรือข้อมูลที่ใช้สำหรับอ่านภาพและวิเคราะห์โจทย์ อาจถูกลบหรือทำให้ลดการระบุตัวตนเมื่อหมดความจำเป็น', 'ผู้ใช้สามารถติดต่อเพื่อขอลบหรือปิดบัญชีได้ โดยบางข้อมูลอาจยังต้องเก็บไว้ตามที่กฎหมายกำหนดหรือเพื่อป้องกันการทุจริตและข้อพิพาท']), LegalSection(heading='สิทธิของเจ้าของข้อมูล', details=['ผู้ใช้สามารถขอเข้าถึง สำเนา แก้ไข ลบ หรือจำกัดการประมวลผลข้อมูลส่วนบุคคลของตนเองได้ตามขอบเขตที่กฎหมายคุ้มครองข้อมูลส่วนบุคคลกำหนด', 'ผู้ใช้สามารถถอนความยินยอมหรือคัดค้านการประมวลผลบางประเภทได้ หากการดำเนินการนั้นอยู่บนฐานความยินยอมหรือเข้าข่ายที่กฎหมายอนุญาต', 'การลบหรือจำกัดข้อมูลบางอย่างอาจส่งผลให้ไม่สามารถใช้ฟีเจอร์บางส่วนได้ เช่น รายงานย้อนหลัง การวิเคราะห์ผล หรือการยืนยันสิทธิ์คอร์ส', 'เราจะพิจารณาคำขอตามขั้นตอนที่เหมาะสม และอาจขอข้อมูลเพิ่มเติมเพื่อยืนยันตัวตนก่อนดำเนินการ']), LegalSection(heading='ข้อมูลของผู้เรียนและผู้เยาว์', details=['บริการของเราออกแบบเพื่อสนับสนุนการเรียนรู้ของผู้เรียน ซึ่งอาจรวมถึงผู้เยาว์ จึงควรมีผู้ปกครองหรือผู้ดูแลรับทราบการสมัครและการใช้งาน', 'หากผู้ปกครองพบว่าผู้เยาว์ให้ข้อมูลส่วนบุคคลโดยไม่ได้รับอนุญาต สามารถติดต่อเราเพื่อขอให้ตรวจสอบ แก้ไข หรือลบข้อมูลที่เกี่ยวข้อง', 'เราพยายามจำกัดการเก็บข้อมูลของผู้เรียนเท่าที่จำเป็นต่อการให้บริการด้านการเรียน การวิเคราะห์ผล และการดูแลความปลอดภัยของบัญชี']), LegalSection(heading='คุกกี้ บันทึกการใช้งาน และการวิเคราะห์ระบบ', details=['เว็บไซต์หรือแอปอาจใช้คุกกี้ local storage หรือเทคโนโลยีที่คล้ายกันเพื่อจดจำสถานะการเข้าสู่ระบบ การตั้งค่า และปรับปรุงประสบการณ์ใช้งาน', 'เราอาจเก็บบันทึกการใช้งาน เหตุขัดข้อง และข้อมูลประสิทธิภาพของระบบ เพื่อแก้ปัญหา ป้องกันการใช้งานผิดปกติ และพัฒนาคุณภาพบริการ', 'ข้อมูลเชิงวิเคราะห์ที่ใช้เพื่อปรับปรุงระบบจะถูกใช้ในขอบเขตที่เหมาะสม และเมื่อเป็นไปได้จะลดการระบุตัวตนของผู้ใช้']), LegalSection(heading='การโอนหรือประมวลผลข้อมูลโดยผู้ให้บริการภายนอก', details=['บางบริการ เช่น คลาวด์ โครงสร้างพื้นฐาน ระบบ AI ระบบอีเมล หรือระบบชำระเงิน อาจตั้งอยู่หรือประมวลผลข้อมูลในต่างประเทศ', 'เมื่อมีการใช้ผู้ให้บริการภายนอก เราจะพิจารณามาตรการคุ้มครองข้อมูลที่เหมาะสมกับลักษณะข้อมูลและความเสี่ยงของการประมวลผล', 'การใช้งานบริการต่อถือว่าผู้ใช้รับทราบว่าอาจมีการประมวลผลข้อมูลผ่านระบบหรือผู้ให้บริการที่จำเป็นต่อการให้บริการของ TEWMai']), LegalSection(heading='การเปลี่ยนแปลงนโยบายนี้', details=['เราอาจปรับปรุงนโยบายความเป็นส่วนตัวเป็นครั้งคราว เพื่อให้สอดคล้องกับฟีเจอร์ใหม่ วิธีดำเนินงาน หรือข้อกำหนดทางกฎหมาย', 'เมื่อมีการเปลี่ยนแปลงสำคัญ เราอาจแจ้งผ่านเว็บไซต์ แอป อีเมล หรือช่องทางที่เหมาะสม พร้อมปรับวันที่อัปเดตของเอกสาร', 'การใช้งานบริการต่อหลังจากนโยบายฉบับใหม่มีผล ถือว่าผู้ใช้รับทราบการเปลี่ยนแปลงตามขอบเขตที่กฎหมายอนุญาต']), LegalSection(heading='ติดต่อเรา', details=['หากมีคำถามเกี่ยวกับนโยบายความเป็นส่วนตัว การใช้ข้อมูล หรือการใช้สิทธิของเจ้าของข้อมูล สามารถติดต่อได้ที่ support@tewmai.com, LINE @tewmai หรือ Facebook: TEWMai - ติวอัจฉริยะด้วย AI', 'เพื่อความปลอดภัย เราอาจขอข้อมูลเพื่อยืนยันตัวตนก่อนเปิดเผย แก้ไข หรือลบข้อมูลตามคำขอ'])], contact_email='support@tewmai.com')


async def get_quiz(quiz_id: str, user_id: Optional[str]=None, course_id: Optional[str]=None, credentials: Optional[HTTPAuthorizationCredentials]=Depends(STUDENT_BEARER_OPTIONAL), student_auth_service: StudentAuthService=Depends(_get_student_auth_service), data_service=Depends(get_db_service)):
    """Get quiz questions by quiz ID using the configured data service."""
    try:
        quiz = await data_service.get_quiz(quiz_id)
        if not quiz:
            raise HTTPException(status_code=404, detail=f'Quiz {quiz_id} not found')
        if user_id:
            await _ensure_user_matches_token(user_id=user_id, credentials=credentials, auth_service=student_auth_service)
        effective_course_id = str(course_id or quiz.get('course_id') or '').strip()
        if user_id and effective_course_id and (effective_course_id != 'default-course'):
            await _ensure_active_course_access(data_service=data_service, user_id=user_id, course_id=effective_course_id)
        return quiz
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f'Error retrieving quiz {quiz_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to retrieve quiz')


async def list_user_quizzes(user_id: str, limit: int=50, course_id: Optional[str]=None, data_service=Depends(get_data_service)):
    """List all quizzes for a specific user, optionally filtered by course."""
    try:
        quizzes = await data_service.get_user_quizzes(user_id, course_id)
        return {'user_id': user_id, 'total_quizzes': len(quizzes), 'quizzes': quizzes}
    except Exception as e:
        app_logger.error(f'Error listing quizzes for user {user_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to list user quizzes')


async def list_user_quiz_results(user_id: str, course_id: Optional[str]=None, quiz_id: Optional[str]=None, data_service=Depends(get_data_service)):
    """List quiz submission results for a user, optionally filtered by course_id/quiz_id."""
    try:
        results = await data_service.get_user_quiz_results(user_id=user_id, quiz_id=quiz_id, course_id=course_id)
        return {'user_id': user_id, 'total_results': len(results), 'results': results}
    except Exception as e:
        app_logger.error(f'Error listing quiz results for user {user_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to list user quiz results')


async def list_course_quizzes(course_id: str, user_id: Optional[str]=None, q: Optional[str]=None, difficulty: Optional[str]=None, sort: str='latest', view: str='full', page: int=1, page_size: int=20, quiz_ids: Optional[str]=None, credentials: Optional[HTTPAuthorizationCredentials]=Depends(STUDENT_BEARER_OPTIONAL), student_auth_service: StudentAuthService=Depends(_get_student_auth_service), data_service=Depends(get_data_service)):
    """List all quizzes associated with a course (any instructor)."""
    try:
        if user_id and str(course_id or '').strip():
            await _ensure_user_matches_token(user_id=user_id, credentials=credentials, auth_service=student_auth_service)
            await _ensure_active_course_access(data_service=data_service, user_id=user_id, course_id=course_id)

        def normalize_text(value: Any) -> str:
            return re.sub('\\s+', ' ', str(value or '').strip().lower())
        page = max(1, int(page or 1))
        page_size = max(1, min(100, int(page_size or 20)))
        sort_key = normalize_text(sort)
        summary_view = normalize_text(view) == 'summary'
        difficulty_filter = normalize_text(difficulty)
        allowed_ids = []
        if quiz_ids and quiz_ids.strip():
            allowed_ids = [token.strip() for token in quiz_ids.split(',') if token and token.strip()]
        get_quizzes_page = getattr(data_service, 'get_course_quizzes_page', None)
        db_page_supported = callable(get_quizzes_page) and sort_key in {'', 'latest', 'oldest'} and (difficulty_filter not in {'easy', 'ง่าย', 'medium', 'ปานกลาง', 'hard', 'ยาก'})
        if db_page_supported:
            page_result = await get_quizzes_page(course_id, page=page, page_size=page_size, q=q, sort=sort_key or 'latest', quiz_ids=allowed_ids or None, summary=summary_view)
            total_filtered = int(page_result.get('total') or 0)
            total_pages = int(page_result.get('total_pages') or 1)
            current_page = int(page_result.get('page') or page)
            return {'course_id': course_id, 'total_quizzes': total_filtered, 'total_filtered': total_filtered, 'page': current_page, 'page_size': int(page_result.get('page_size') or page_size), 'total_pages': total_pages, 'has_next': current_page < total_pages, 'has_prev': current_page > 1, 'quizzes': page_result.get('rows') or []}
        try:
            quizzes = await data_service.get_quizzes_by_course(course_id, summary=summary_view)
        except TypeError:
            quizzes = await data_service.get_quizzes_by_course(course_id)

        def natural_text_key(value: Any) -> tuple:
            text = normalize_text(value)
            if not text:
                return tuple()
            thai_digit_map = str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789')
            parts = []
            cursor = 0
            for match in re.finditer('[0-9๐-๙]+', text):
                if match.start() > cursor:
                    parts.append((0, text[cursor:match.start()]))
                number_text = match.group(0).translate(thai_digit_map)
                try:
                    parts.append((1, int(number_text)))
                except Exception:
                    parts.append((0, number_text))
                cursor = match.end()
            if cursor < len(text):
                parts.append((0, text[cursor:]))
            return tuple(parts)

        def quiz_identity_key(item: Dict[str, Any]) -> str:
            return normalize_text(item.get('quiz_id') or item.get('id') or item.get('document_id') or '')

        def quiz_title_key(item: Dict[str, Any]) -> tuple:
            return natural_text_key(item.get('title') or item.get('name'))

        def to_question_count(item: Dict[str, Any]) -> int:
            if isinstance(item.get('total_questions'), (int, float)):
                return int(item.get('total_questions') or 0)
            qs = item.get('questions')
            if isinstance(qs, list):
                return len(qs)
            return 0

        def try_parse_difficulty_score(raw: Any) -> Optional[float]:
            if raw is None:
                return None
            if isinstance(raw, (int, float)):
                try:
                    return float(raw)
                except Exception:
                    return None
            value = normalize_text(raw)
            if not value:
                return None
            try:
                return float(value)
            except Exception:
                pass
            if value in {'easy', 'ง่าย'}:
                return 2.0
            if value in {'medium', 'ปานกลาง'}:
                return 3.0
            if value in {'hard', 'ยาก'}:
                return 4.0
            return None

        def to_difficulty_score(raw: Any) -> float:
            parsed = try_parse_difficulty_score(raw)
            if parsed is not None:
                return parsed
            return 3.0

        def pick_difficulty_value(item: Dict[str, Any]) -> Any:
            for key in ('difficulty_avg', 'difficulty', 'difficulty_level', 'level_difficulty', 'level'):
                value = item.get(key)
                if isinstance(value, str) and (not normalize_text(value)):
                    continue
                if value is not None:
                    return value
            questions = item.get('questions')
            if isinstance(questions, list) and questions:
                scores: List[float] = []
                for question in questions:
                    if not isinstance(question, dict):
                        continue
                    parsed = try_parse_difficulty_score(question.get('difficulty', question.get('level')))
                    if parsed is not None:
                        scores.append(parsed)
                if scores:
                    return sum(scores) / len(scores)
            return None

        def to_difficulty_bucket(score: float) -> str:
            stars = max(1, min(5, int(round(score))))
            if stars <= 2:
                return 'easy'
            if stars == 3:
                return 'medium'
            return 'hard'
        total_before_filter = len(quizzes)
        if allowed_ids:
            allowed_ids_set = set(allowed_ids)
            quizzes = [item for item in quizzes if str(item.get('quiz_id') or item.get('id') or item.get('document_id') or '') in allowed_ids_set]
        q_text = normalize_text(q)
        if q_text:
            quizzes = [item for item in quizzes if q_text in normalize_text(item.get('title')) or q_text in normalize_text(item.get('description'))]
        if difficulty_filter in {'easy', 'ง่าย', 'medium', 'ปานกลาง', 'hard', 'ยาก'}:
            target_bucket = 'easy' if difficulty_filter in {'easy', 'ง่าย'} else 'medium' if difficulty_filter in {'medium', 'ปานกลาง'} else 'hard'

            def in_bucket(score: float) -> bool:
                return to_difficulty_bucket(score) == target_bucket
            quizzes = [item for item in quizzes if in_bucket(to_difficulty_score(pick_difficulty_value(item)))]
        if sort_key in {'oldest'}:
            quizzes.sort(key=lambda x: x.get('created_at', ''))
        elif sort_key in {'title_asc'}:
            quizzes.sort(key=lambda x: (quiz_title_key(x), quiz_identity_key(x)))
        elif sort_key in {'title_desc'}:
            quizzes.sort(key=lambda x: (quiz_title_key(x), quiz_identity_key(x)), reverse=True)
        elif sort_key in {'difficulty_asc'}:
            quizzes.sort(key=lambda x: (to_difficulty_score(pick_difficulty_value(x)), quiz_title_key(x), quiz_identity_key(x)))
        elif sort_key in {'difficulty_desc'}:
            quizzes.sort(key=lambda x: (-to_difficulty_score(pick_difficulty_value(x)), quiz_title_key(x), quiz_identity_key(x)))
        elif sort_key in {'questions_asc'}:
            quizzes.sort(key=to_question_count)
        elif sort_key in {'questions_desc'}:
            quizzes.sort(key=to_question_count, reverse=True)
        else:
            quizzes.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        total_filtered = len(quizzes)
        total_pages = max(1, (total_filtered + page_size - 1) // page_size)
        if page > total_pages:
            page = total_pages
        start = (page - 1) * page_size
        end = start + page_size
        quizzes_page = quizzes[start:end]
        return {'course_id': course_id, 'total_quizzes': total_before_filter, 'total_filtered': total_filtered, 'page': page, 'page_size': page_size, 'total_pages': total_pages, 'has_next': page < total_pages, 'has_prev': page > 1, 'quizzes': quizzes_page}
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f'Error listing quizzes for course {course_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to list course quizzes')


def _coerce_int(value: Any, default: int=0) -> int:
    try:
        if value is None:
            return default
        return int(round(float(value)))
    except Exception:
        return default


def _normalize_short_text_list(value: Any, max_items: int=5, max_chars: int=140) -> List[str]:
    rows: List[str] = []
    if isinstance(value, list):
        candidates = value
    elif isinstance(value, str):
        candidates = re.split('\\s*(?:\\||•|\\n|;)\\s*', value)
    else:
        candidates = []
    for candidate in candidates:
        text = str(candidate or '').strip()
        if not text:
            continue
        rows.append(text[:max_chars])
        if len(rows) >= max_items:
            break
    return rows


def _normalize_recommendation_cards(rows: Any, max_items: int=3) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(rows, list):
        return normalized
    for item in rows:
        if not isinstance(item, dict):
            continue
        title = str(item.get('title') or item.get('headline') or '').strip()
        evidence = str(item.get('evidence') or item.get('why') or '').strip()
        action = str(item.get('action') or item.get('next_action') or '').strip()
        note = str(item.get('note') or '').strip()
        target = str(item.get('target') or '').strip()
        evidence_items = _normalize_short_text_list(item.get('evidence_items') or item.get('evidenceItems') or evidence, max_items=3, max_chars=100)
        action_steps = _normalize_short_text_list(item.get('action_steps') or item.get('actionSteps') or action, max_items=2, max_chars=120)
        focus_preview = str(item.get('focus_preview') or item.get('focusPreview') or note or '').strip()
        target_metric = str(item.get('target_metric') or item.get('targetMetric') or target or '').strip()
        cta_type = str(item.get('cta_type') or item.get('ctaType') or '').strip()
        if not (title or evidence or action or note or target or evidence_items or action_steps or focus_preview or target_metric):
            continue
        normalized.append({'title': title[:120], 'evidence': evidence[:220], 'action': action[:220], 'note': note[:220], 'target': target[:160], 'evidence_items': evidence_items, 'action_steps': action_steps, 'focus_preview': focus_preview[:220], 'target_metric': target_metric[:180], 'cta_type': cta_type[:80]})
        if len(normalized) >= max_items:
            break
    return normalized


def _build_student_analysis_fallback(payload: StudentAnalysisSummaryRequest) -> Dict[str, Any]:
    metrics = payload.metrics if isinstance(payload.metrics, dict) else {}
    trend = payload.recent_trend if isinstance(payload.recent_trend, dict) else {}
    focus_area = payload.focus_area if isinstance(payload.focus_area, dict) else {}
    patterns = payload.priority_patterns if isinstance(payload.priority_patterns, dict) else {}
    available_actions = [item for item in payload.available_actions or [] if isinstance(item, dict) and str(item.get('type') or '').strip()]
    client_cards = _normalize_recommendation_cards(payload.client_recommendation_cards)

    def pick_action(*preferred_types: str) -> Dict[str, Any]:
        for preferred_type in preferred_types:
            for item in available_actions:
                if str(item.get('type') or '').strip() == preferred_type:
                    return item
        return available_actions[0] if available_actions else {}

    def action_label(item: Dict[str, Any], default: str) -> str:
        return str(item.get('label') or default).strip() or default
    attempts = max(0, _coerce_int(metrics.get('attempts_total'), 0))
    avg_score = max(0, min(100, _coerce_int(metrics.get('average_score_pct'), 0)))
    last_score = max(0, min(100, _coerce_int(metrics.get('latest_score_pct'), 0)))
    accuracy7d = max(0, min(100, _coerce_int(metrics.get('accuracy_7d_pct'), 0)))
    avg_sec = max(0, _coerce_int(metrics.get('average_time_per_question_sec'), 0))
    confident_wrong_rate = max(0, min(100, _coerce_int(metrics.get('confident_wrong_rate_pct'), 0)))
    trend_label = str(trend.get('direction_label') or 'ยังทรงตัว').strip() or 'ยังทรงตัว'
    trend_delta = _coerce_int(trend.get('score_delta_pct'), 0)
    focus_label = str(focus_area.get('label') or '').strip()
    focus_accuracy = max(0, min(100, _coerce_int(focus_area.get('accuracy_pct'), 0)))
    focus_type = str(focus_area.get('type') or '').strip().lower()
    focus_note = str(focus_area.get('note') or '').strip()
    if focus_label:
        focus_prefix = 'หัวข้อ' if focus_type == 'topic' else 'ระดับโจทย์'
        if focus_accuracy > 0:
            improvement_line = f'ควรปรับปรุง{focus_prefix} {focus_label} เพราะ Accuracy อยู่ที่ {focus_accuracy}%'
        else:
            improvement_line = f'ควรปรับปรุง{focus_prefix} {focus_label} เป็นลำดับแรก'
        if focus_note:
            improvement_line += f' ({focus_note})'
    else:
        improvement_line = 'ควรปรับปรุงความแม่นยำในข้อที่ผิดซ้ำจากชุดล่าสุด'
    if confident_wrong_rate >= 30:
        improvement_line += f' และมีอัตรามั่นใจผิดสูง {confident_wrong_rate}%'
    sampled_questions = max(0, _coerce_int(patterns.get('sampled_questions'), 0))
    wrong_questions = max(0, _coerce_int(patterns.get('wrong_questions'), 0))
    latest_wrong_questions = max(0, _coerce_int(patterns.get('latest_wrong_questions'), 0))
    confident_wrong_questions = max(0, _coerce_int(patterns.get('confident_wrong_questions'), 0))
    top_questions = [item for item in patterns.get('top_questions') or [] if isinstance(item, dict)]
    weak_topics = [item for item in patterns.get('weak_topics') or [] if isinstance(item, dict)]
    top_question = top_questions[0] if top_questions else {}
    top_question_label = str(top_question.get('question_label') or '').strip()
    top_question_topic = str(top_question.get('topic') or '').strip()
    top_question_preview = str(top_question.get('question_preview') or '').strip()
    top_question_wrong = max(0, _coerce_int(top_question.get('wrong_count'), 0))
    top_question_recent_wrong = max(0, _coerce_int(top_question.get('recent_wrong_count'), 0))
    top_question_avg_sec = max(0, _coerce_int(top_question.get('avg_time_sec'), 0))
    top_weak_topic = weak_topics[0] if weak_topics else {}
    weak_topic_label = str(top_weak_topic.get('topic') or focus_label).strip()
    weak_topic_wrong = max(0, _coerce_int(top_weak_topic.get('wrong_count'), 0))
    weak_topic_question_count = max(0, _coerce_int(top_weak_topic.get('question_count'), 0))
    trend_delta_text = f" ({('+' if trend_delta > 0 else '')}{trend_delta}%)" if trend_delta != 0 else ''
    trend_line = f'คะแนนช่วงล่าสุด{trend_label}{trend_delta_text} โดยค่าเฉลี่ยอยู่ที่ {avg_score}% และคะแนนล่าสุด {last_score}% (Accuracy 7 วัน {accuracy7d}%)'
    trend_title = 'แนวโน้มกำลังดีขึ้น' if trend_delta > 0 else 'แนวโน้มช่วงล่าสุดลดลง' if trend_delta < 0 else 'คะแนนยังทรงตัว'
    trend_action_item = pick_action('practice_recent_mistakes', 'practice_weak_topic')
    trend_action = f"ภายใน 24 ชั่วโมง ทำชุด {action_label(trend_action_item, 'ฝึกซ้ำ 10 ข้อที่พลาดล่าสุด')} แล้วทบทวนข้อที่ยังผิดซ้ำ" if trend_delta < 0 else f"ทำชุด {action_label(trend_action_item, 'ฝึกซ้ำ 10 ข้อที่พลาดล่าสุด')} 1 รอบ เพื่อรักษาจังหวะและดัน Accuracy 7 วันให้ถึง {max(70, min(95, accuracy7d + 3))}%" if trend_delta > 0 else f"ทำชุด {action_label(trend_action_item, 'ฝึกซ้ำ 10 ข้อที่พลาดล่าสุด')} 1 รอบ แล้วเช็กเฉลยเฉพาะข้อที่ยังพลาด"
    speed_target = max(20, int(round(avg_sec * 0.9))) if avg_sec > 0 else 30
    advice_line = f'รอบถัดไปตั้งเป้า Accuracy 7 วันอย่างน้อย {max(70, accuracy7d)}% และคุมเวลาเฉลี่ยไม่เกิน {speed_target} วินาที/ข้อ' if avg_sec > 0 else f'รอบถัดไปตั้งเป้า Accuracy 7 วันอย่างน้อย {max(70, accuracy7d)}%'
    if sampled_questions > 0:
        advice_line += f' พร้อมทบทวนข้อที่พลาด {wrong_questions}/{sampled_questions} ข้อ'
    if confident_wrong_questions > 0:
        advice_line += f' และแก้พฤติกรรมมั่นใจผิด {confident_wrong_questions} ข้อ'
    if top_question_label:
        summary_paragraph = f"{top_question_label}{(f' หัวข้อ {top_question_topic}' if top_question_topic else '')} เป็นจุดที่ควรแก้ก่อน เพราะผิด {top_question_wrong} ครั้ง จากข้อมูล {attempts} ครั้งล่าสุด"
    else:
        summary_paragraph = f'จากข้อมูล {attempts} ครั้งล่าสุด {trend_line}'
    focus_prefix = 'หัวข้อ' if focus_type == 'topic' else 'ระดับโจทย์'
    focus_target_label = focus_label or 'ข้อที่พลาดซ้ำจากรอบล่าสุด'
    focus_action_item = pick_action('practice_weak_topic', 'practice_recent_mistakes')
    focus_action = f"วันนี้ทำชุด {action_label(focus_action_item, 'เก็บหัวข้ออ่อน 10 ข้อ')} แล้วจดสาเหตุทุกข้อที่ผิด" if focus_label else f"วันนี้ทำชุด {action_label(focus_action_item, 'ฝึกซ้ำ 10 ข้อที่พลาดล่าสุด')} แล้วทำซ้ำให้ถูกครบ"
    confidence_note = f'พบมั่นใจแต่ตอบผิด {confident_wrong_questions} ข้อ ควรเช็กวิธีคิดก่อนกดตอบ' if confident_wrong_questions > 0 else ''
    level_up_action_item = pick_action('practice_level_up', 'practice_weak_topic', 'practice_recent_mistakes')
    if top_question_label:
        question_evidence_parts = [f'ผิดสะสม {top_question_wrong} ครั้ง', f'พลาดใน 5 ครั้งล่าสุด {top_question_recent_wrong} ครั้ง' if top_question_recent_wrong > 0 else '', f'เฉลี่ย {top_question_avg_sec} วิ/ข้อ' if top_question_avg_sec > 0 else '', f'"{top_question_preview[:90]}"' if top_question_preview else '']
        question_evidence = ' | '.join((part for part in question_evidence_parts if part))
        recommendation_cards = [{'title': f'แก้ข้อผิดซ้ำ: {top_question_label}', 'evidence': question_evidence, 'action': f"ทำชุด {action_label(trend_action_item, 'ฝึกซ้ำ 10 ข้อที่พลาดล่าสุด')} แล้วอ่านเฉลยของ {top_question_label} ก่อนทำซ้ำ", 'note': f'หัวข้อ {top_question_topic} ต้องตอบถูกติดกันก่อนข้าม' if top_question_topic else 'เช็กวิธีคิดก่อนดูเฉลย', 'target': f'ตอบ {top_question_label} และข้อแนวเดียวกันให้ถูก 2 รอบติด', 'evidence_items': [part for part in question_evidence_parts if part][:4], 'action_steps': [f"ทำชุด {action_label(trend_action_item, 'ฝึกซ้ำ 10 ข้อที่พลาดล่าสุด')}", f'อ่านเฉลยของ {top_question_label} แล้วสรุปว่าพลาดตรงไหน', f'ทำ {top_question_label} ซ้ำจนตอบถูกติดกัน'], 'focus_preview': f'{top_question_label}: {top_question_preview}' if top_question_preview else top_question_label, 'target_metric': f'ถูก 2 รอบติดใน {top_question_label} และข้อแนวเดียวกัน', 'cta_type': str(trend_action_item.get('type') or '')}, {'title': f'เก็บหัวข้อ {weak_topic_label}' if weak_topic_label else f'เก็บข้อที่พลาดล่าสุด {latest_wrong_questions} ข้อ', 'evidence': f'หัวข้อนี้ผิดรวม {weak_topic_wrong} ครั้ง จาก {weak_topic_question_count} ข้อที่ระบบจับได้' if weak_topic_wrong > 0 else improvement_line, 'action': focus_action, 'note': focus_note or confidence_note, 'target': f'ลดข้อผิดหัวข้อนี้ให้เหลือไม่เกิน {max(1, int(weak_topic_wrong * 0.7))} ครั้ง' if weak_topic_wrong > 0 else f'ลดข้อผิดใน{focus_prefix}นี้ลงอย่างน้อย 30%', 'evidence_items': [f'ผิดรวม {weak_topic_wrong} ครั้ง' if weak_topic_wrong > 0 else improvement_line, f'{weak_topic_question_count} ข้อที่ระบบจับได้' if weak_topic_question_count > 0 else '', f'หัวข้อ {weak_topic_label}' if weak_topic_label else ''], 'action_steps': [f"ทำชุด {action_label(focus_action_item, 'เก็บหัวข้ออ่อน 10 ข้อ')}", 'จดสาเหตุข้อที่ผิดเป็นคำสั้นๆ หลังดูเฉลย', 'ทำซ้ำเฉพาะข้อที่สาเหตุซ้ำกัน'], 'focus_preview': f'{weak_topic_label}: เริ่มจาก {top_question_label}' if weak_topic_label and top_question_label else weak_topic_label, 'target_metric': f'ข้อผิดหัวข้อนี้เหลือไม่เกิน {max(1, int(weak_topic_wrong * 0.7))} ครั้ง' if weak_topic_wrong > 0 else f'ลดข้อผิดใน{focus_prefix}นี้ลงอย่างน้อย 30%', 'cta_type': str(focus_action_item.get('type') or '')}, {'title': 'วัดผลรอบถัดไป', 'evidence': f'ล่าสุด {last_score}% | Accuracy 7 วัน {accuracy7d}% | พลาดล่าสุด {latest_wrong_questions} ข้อ', 'action': f"ทำชุด {action_label(level_up_action_item, 'รวมโจทย์ท้าทาย')} หลังทบทวนข้อผิดซ้ำ", 'note': f'คุมเวลาเฉลี่ยต่อข้อไม่เกิน {speed_target} วินาที' if avg_sec > 0 else 'ถ้าผิดซ้ำให้กลับไปอ่านเฉลยข้อแรกก่อน', 'target': advice_line, 'evidence_items': [f'ล่าสุด {last_score}%', f'Accuracy 7 วัน {accuracy7d}%', f'พลาดล่าสุด {latest_wrong_questions} ข้อ'], 'action_steps': ['ทบทวนข้อผิดซ้ำก่อนเริ่มจับเวลา', f"ทำชุด {action_label(level_up_action_item, 'รวมโจทย์ท้าทาย')}", 'หลังทำเสร็จ ให้เทียบว่าข้อผิดซ้ำลดลงหรือไม่'], 'focus_preview': 'เริ่มจากข้อผิดซ้ำก่อน แล้วค่อยทำโจทย์ท้าทาย', 'target_metric': advice_line, 'cta_type': str(level_up_action_item.get('type') or '')}]
    else:
        recommendation_cards = [{'title': trend_title, 'evidence': trend_line, 'action': trend_action, 'note': f'คะแนนเฉลี่ย {avg_score}% | ล่าสุด {last_score}% | Accuracy 7 วัน {accuracy7d}%', 'target': f'Accuracy 7 วันอย่างน้อย {max(70, accuracy7d)}%', 'evidence_items': [f'เฉลี่ย {avg_score}%', f'ล่าสุด {last_score}%', f'Accuracy 7 วัน {accuracy7d}%'], 'action_steps': [f"ทำชุด {action_label(trend_action_item, 'ฝึกซ้ำ 10 ข้อที่พลาดล่าสุด')}", 'เช็กเฉลยเฉพาะข้อที่ยังพลาด', 'กลับมาดูแนวโน้มหลังจบรอบ'], 'focus_preview': trend_line, 'target_metric': f'Accuracy 7 วันอย่างน้อย {max(70, accuracy7d)}%', 'cta_type': str(trend_action_item.get('type') or '')}, {'title': f'จุดอ่อนหลัก: {focus_target_label}', 'evidence': improvement_line, 'action': focus_action, 'note': confidence_note or focus_note, 'target': f'ลดข้อผิดใน{focus_prefix}นี้ลงอย่างน้อย 30%', 'evidence_items': [improvement_line, confidence_note or focus_note], 'action_steps': [f"ทำชุด {action_label(focus_action_item, 'เก็บหัวข้ออ่อน 10 ข้อ')}", 'จดสาเหตุทุกข้อที่ผิด', 'ทำซ้ำข้อที่สาเหตุซ้ำกัน'], 'focus_preview': focus_target_label, 'target_metric': f'ลดข้อผิดใน{focus_prefix}นี้ลงอย่างน้อย 30%', 'cta_type': str(focus_action_item.get('type') or '')}, {'title': 'เป้าหมายรอบถัดไป', 'evidence': f'พลาด {wrong_questions}/{sampled_questions} ข้อจากข้อมูลที่วิเคราะห์' if sampled_questions > 0 else 'ยังมีข้อมูลรายข้อไม่มาก แนะนำเก็บรอบฝึกเพิ่มเพื่อวิเคราะห์ให้แม่นขึ้น', 'action': f"ทำชุด {action_label(level_up_action_item, 'รวมโจทย์ท้าทาย')} แล้วตั้งเป้าตามผลรอบถัดไป", 'note': f'คุมเวลาเฉลี่ยต่อข้อไม่เกิน {speed_target} วินาที' if avg_sec > 0 else 'รักษาความสม่ำเสมอในการฝึกต่อเนื่อง', 'target': advice_line, 'evidence_items': [f'พลาด {wrong_questions}/{sampled_questions} ข้อ' if sampled_questions > 0 else 'ยังมีข้อมูลรายข้อไม่มาก', f'เวลาเป้า {speed_target} วิ/ข้อ' if avg_sec > 0 else ''], 'action_steps': [f"ทำชุด {action_label(level_up_action_item, 'รวมโจทย์ท้าทาย')}", 'ตรวจข้อผิดทันทีหลังทำ', 'ตั้งเป้าตามผลรอบถัดไป'], 'focus_preview': 'ใช้รอบนี้เพื่อแยกสาเหตุข้อผิดให้ชัดขึ้น', 'target_metric': advice_line, 'cta_type': str(level_up_action_item.get('type') or '')}]
    if client_cards:
        recommendation_cards = client_cards
    recommendation_cards = _normalize_recommendation_cards(recommendation_cards)
    return {'summary_paragraph': summary_paragraph, 'recommendations': [trend_line, improvement_line, advice_line], 'recommendation_cards': recommendation_cards}


def _build_analysis_summary_cache_key(payload: StudentAnalysisSummaryRequest) -> str:
    normalized_payload = {'recommendation_schema_version': str(payload.recommendation_schema_version or '').strip(), 'analysis_plan': payload.analysis_plan if isinstance(payload.analysis_plan, list) else [], 'context': payload.context if isinstance(payload.context, dict) else {}, 'metrics': payload.metrics if isinstance(payload.metrics, dict) else {}, 'recent_trend': payload.recent_trend if isinstance(payload.recent_trend, dict) else {}, 'focus_area': payload.focus_area if isinstance(payload.focus_area, dict) else {}, 'priority_patterns': payload.priority_patterns if isinstance(payload.priority_patterns, dict) else {}, 'available_actions': payload.available_actions if isinstance(payload.available_actions, list) else [], 'client_recommendation_cards': payload.client_recommendation_cards if isinstance(payload.client_recommendation_cards, list) else []}
    serialized = json.dumps(normalized_payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


async def generate_student_analysis_summary(user_id: str, course_id: str, payload: StudentAnalysisSummaryRequest=Body(...), credentials: Optional[HTTPAuthorizationCredentials]=Depends(STUDENT_BEARER_OPTIONAL), student_auth_service: StudentAuthService=Depends(_get_student_auth_service), chat_service: ChatService=Depends(get_chat_service), data_service=Depends(get_data_service)):
    """Generate AI summary for student analysis using OpenRouter."""
    try:
        await _ensure_user_matches_token(user_id=user_id, credentials=credentials, auth_service=student_auth_service)
        await _ensure_active_course_access(data_service=data_service, user_id=user_id, course_id=course_id)
        settings = get_settings()
        resolved_model = str(settings.openrouter_model).strip() or settings.openrouter_model
        fallback = _build_student_analysis_fallback(payload)
        cache_key = _build_analysis_summary_cache_key(payload)
        score_version = await data_service.get_student_course_score_version(user_id, course_id)
        cached_response = await data_service.get_student_analysis_summary_cache(user_id=user_id, course_id=course_id, cache_key=cache_key, score_version=score_version)
        if isinstance(cached_response, dict):
            cached_cards = _normalize_recommendation_cards(cached_response.get('recommendation_cards'))
            if not cached_cards:
                cached_cards = _normalize_recommendation_cards(fallback.get('recommendation_cards'))
            return {**cached_response, 'recommendation_cards': cached_cards[:3], 'cached': True}
        default_plan = ['วิเคราะห์แนวโน้มคะแนน', 'จุดที่ต้องปรับปรุง', 'คำแนะนำ']
        normalized_plan = [str(item).strip() for item in payload.analysis_plan or [] if str(item or '').strip()]
        if not normalized_plan:
            normalized_plan = default_plan
        input_data = {'recommendation_schema_version': str(payload.recommendation_schema_version or '').strip(), 'analysis_plan': normalized_plan, 'context': payload.context if isinstance(payload.context, dict) else {}, 'metrics': payload.metrics if isinstance(payload.metrics, dict) else {}, 'recent_trend': payload.recent_trend if isinstance(payload.recent_trend, dict) else {}, 'focus_area': payload.focus_area if isinstance(payload.focus_area, dict) else {}, 'priority_patterns': payload.priority_patterns if isinstance(payload.priority_patterns, dict) else {}, 'available_actions': payload.available_actions if isinstance(payload.available_actions, list) else [], 'client_recommendation_cards': payload.client_recommendation_cards if isinstance(payload.client_recommendation_cards, list) else []}
        plan_block = '\n'.join([f'- {line}' for line in normalized_plan])
        glossary_block = '- context.analysis_source_type: ประเภทการวิเคราะห์ (lesson หรือ mock_exam)\n- context.scope_label: ชื่อบทเรียน/ชุดข้อสอบที่กำลังดู\n- context.topic_filter: หัวข้อที่เลือกกรอง (ถ้าไม่มีให้เป็น null)\n- metrics.attempts_total: จำนวนครั้งที่นำมาวิเคราะห์\n- metrics.average_score_pct: คะแนนเฉลี่ยทั้งหมด (% 0-100)\n- metrics.latest_score_pct: คะแนนครั้งล่าสุด (% 0-100)\n- metrics.accuracy_7d_pct: ค่าเฉลี่ยความแม่นยำ 7 วันล่าสุด\n- metrics.average_time_per_question_sec: เวลาเฉลี่ยต่อข้อ (วินาที)\n- metrics.confident_wrong_rate_pct: สัดส่วนมั่นใจแต่ตอบผิด (% 0-100)\n- recent_trend.direction_label: ฉลากแนวโน้มโดยรวม\n- recent_trend.score_delta_pct: ความต่างคะแนนล่าสุดเทียบรอบก่อนหน้า\n- recent_trend.attempts: ลิสต์ผลแต่ละครั้งล่าสุด (attempt_label/date_label/score_pct)\n- focus_area: จุดที่ควรโฟกัสหลัก (type, label, accuracy_pct, attempts_count, note)\n- priority_patterns: สถิติจากข้อที่ควรทบทวน รวม top_questions และ weak_topics ถ้ามี\n- priority_patterns.top_questions: ข้อที่ควรแก้ก่อน มี question_label/topic/wrong_count/recent_wrong_count/question_preview/evidence\n- priority_patterns.weak_topics: หัวข้อที่ผิดบ่อย มี topic/wrong_count/question_count/sample_questions\n- available_actions: ภารกิจ/ปุ่มที่ระบบทำได้จริง แต่ละรายการมี type, label, question_count, topic\n- client_recommendation_cards: คำแนะนำจาก rule ฝั่งระบบที่อ้างอิงข้อจริง ใช้เป็น baseline ถ้าข้อมูลเหมาะสม'
        prompt = f"คุณเป็นผู้ช่วยวิเคราะห์ผลการเรียนของนักเรียน\nใช้น้ำเสียงโค้ชที่ชัด สั้น และให้คำแนะนำที่นักเรียนทำต่อได้ทันที\nสรุปจากข้อมูลจริงที่ให้เท่านั้น ห้ามเดาข้อมูลเพิ่ม\nต้องทำตามแผนสรุปทุกข้อ ห้ามหลุดประเด็นนอกแผน\nห้ามใช้ข้อความกว้างๆ เช่น 'พยายามต่อไป' โดยไม่มีขั้นตอน\nให้เสนอเป็น Next Best Action: ทำอะไร เพราะอะไร และเป้ารอบถัดไปคืออะไร\nnext action ต้องอ้างอิงภารกิจจาก available_actions เท่านั้น และ cta_type ต้องตรงกับ type ใน available_actions\nถ้ามี priority_patterns.top_questions ให้ระบุเลขข้อหรือ preview ของข้อใน evidence หรือ note อย่างน้อย 1 card\nถ้ามี priority_patterns.weak_topics ให้ระบุจำนวนผิดรวมและชื่อหัวข้อใน evidence อย่างน้อย 1 card\nถ้า available_actions ว่าง ให้ cta_type เป็นค่าว่างและแนะนำการทบทวนจากข้อมูลจริงเท่านั้น\nตอบกลับเป็น JSON object เท่านั้น โดยมีคีย์ต่อไปนี้:\n- summary_paragraph: สรุปภาพรวม 1 ประโยค (ไม่เกิน 220 ตัวอักษร)\n- recommendations: array 3 ข้อ เรียงตามแผนสรุป (แต่ละข้อไม่เกิน 180 ตัวอักษร)\n- recommendation_cards: array 3 objects (เรียงตามแผนสรุป) แต่ละ object ต้องมีคีย์ title/evidence/action/note/target/evidence_items/action_steps/focus_preview/target_metric/cta_type\n  - title: หัวข้อสั้นในรูปแบบภารกิจ ไม่เกิน 70 ตัวอักษร\n  - evidence: เหตุผลสั้นๆ อ้างอิงตัวเลขจริงจาก input อย่างน้อย 1 ค่า\n  - action: สิ่งที่ให้ทำทันที ต้องระบุ label หรือประเภทภารกิจจาก available_actions\n  - note: ข้อควรระวังหรือจุดโฟกัสเพิ่มเติมแบบสั้น\n  - target: เป้ารอบถัดไปที่วัดได้ เช่น พลาดไม่เกินกี่ข้อ หรือ Accuracy กี่เปอร์เซ็นต์\n  - evidence_items: array 2-3 ข้อ แต่ละข้อเป็นหลักฐานสั้นๆ เช่นจำนวนผิด ชื่อหัวข้อ คะแนน หรือเวลา ห้ามเป็นประโยคยาว\n  - action_steps: array 2 ขั้นตอนที่นักเรียนทำได้วันนี้ ต้องเริ่มด้วยกริยาและอ้างภารกิจจาก available_actions\n  - focus_preview: ข้อ/หัวข้อเริ่มต้นที่ควรดู ไม่เกิน 160 ตัวอักษร\n  - target_metric: เป้าวัดผลสั้นๆ ไม่เกิน 120 ตัวอักษร\n  - cta_type: type จาก available_actions ที่ตรงกับ action\nห้ามยัด evidence/action/target ทั้งหมดเป็นประโยคเดียว ให้แตกเป็น evidence_items และ action_steps เสมอ\nrecommendations ต้องสอดคล้องกับ recommendation_cards และห้ามเพิ่มหัวข้ออื่น\nห้ามมีคำนำหรือต่อท้ายนอกเหนือจาก JSON\n\nแผนการสรุป:\n{plan_block}\n\nคำอธิบายความหมายของฟิลด์:\n{glossary_block}\n\nข้อมูลนักเรียน (JSON):\n{json.dumps(input_data, ensure_ascii=False)}"
        openrouter_user, openrouter_metadata = await chat_service._get_openrouter_user_context(user_id)
        raw_text = await chat_service._call_gemini_chat(prompt, max_tokens=900, response_format={'type': 'json_object'}, model_name=resolved_model, openrouter_user=openrouter_user, openrouter_metadata=openrouter_metadata)
        parsed = chat_service._safe_extract_json_object(raw_text)
        if not isinstance(parsed, dict):
            parsed = {}
        summary_paragraph = str(parsed.get('summary_paragraph') or parsed.get('summary') or fallback.get('summary_paragraph') or '').strip()
        recommendations_raw = parsed.get('recommendations')
        recommendations: List[str] = []
        if isinstance(recommendations_raw, list):
            recommendations = [str(item).strip() for item in recommendations_raw if str(item or '').strip()]
        elif isinstance(recommendations_raw, str) and recommendations_raw.strip():
            recommendations = [recommendations_raw.strip()]
        fallback_rows = fallback.get('recommendations')
        fallback_recommendations: List[str] = []
        if isinstance(fallback_rows, list):
            fallback_recommendations = [str(item).strip() for item in fallback_rows if str(item or '').strip()]
        fallback_cards = _normalize_recommendation_cards(fallback.get('recommendation_cards'))
        recommendation_cards = _normalize_recommendation_cards(parsed.get('recommendation_cards'))
        allowed_cta_types = {str(item.get('type') or '').strip() for item in payload.available_actions or [] if isinstance(item, dict) and str(item.get('type') or '').strip()}
        for recommendation_card in recommendation_cards:
            cta_type = str(recommendation_card.get('cta_type') or '').strip()
            if cta_type and cta_type not in allowed_cta_types:
                recommendation_card['cta_type'] = ''
        if len(recommendation_cards) < 3:
            seen_card_signatures = {f"{item.get('title', '')}|{item.get('action', '')}" for item in recommendation_cards}
            for fallback_card in fallback_cards:
                if len(recommendation_cards) >= 3:
                    break
                signature = f"{fallback_card.get('title', '')}|{fallback_card.get('action', '')}"
                if signature in seen_card_signatures:
                    continue
                recommendation_cards.append(fallback_card)
                seen_card_signatures.add(signature)
        if not recommendations:
            recommendations = fallback_recommendations
        if len(recommendations) < 3:
            for idx in range(len(recommendations), 3):
                if idx < len(fallback_recommendations):
                    recommendations.append(fallback_recommendations[idx])
        recommendations = recommendations[:3]
        recommendation_cards = recommendation_cards[:3]
        if not summary_paragraph:
            summary_paragraph = str(fallback.get('summary_paragraph') or '').strip()
        response_payload = {'summary_paragraph': summary_paragraph, 'recommendations': recommendations, 'recommendation_cards': recommendation_cards, 'model': resolved_model, 'is_fallback': False, 'generated_at': datetime.utcnow().isoformat() + 'Z', 'cached': False}
        await data_service.set_student_analysis_summary_cache(user_id=user_id, course_id=course_id, cache_key=cache_key, score_version=score_version, response=response_payload)
        return response_payload
    except HTTPException:
        raise
    except Exception as e:
        app_logger.warning(f'Student analysis summary fallback for user {user_id}: {e}')
        fallback = _build_student_analysis_fallback(payload)
        response_payload = {'summary_paragraph': str(fallback.get('summary_paragraph') or '').strip(), 'recommendations': [str(item).strip() for item in fallback.get('recommendations') or [] if str(item or '').strip()][:3], 'recommendation_cards': _normalize_recommendation_cards(fallback.get('recommendation_cards'))[:3], 'model': str(get_settings().openrouter_model).strip() or get_settings().openrouter_model, 'is_fallback': True, 'generated_at': datetime.utcnow().isoformat() + 'Z', 'cached': False}
        return response_payload


async def submit_quiz_answers(user_id: str, quiz_id: str, payload: QuizSubmitPayload, credentials: Optional[HTTPAuthorizationCredentials]=Depends(STUDENT_BEARER_OPTIONAL), student_auth_service: StudentAuthService=Depends(_get_student_auth_service), data_service=Depends(get_data_service)):
    """Submit quiz answers, compute score, and store result history in Data service."""
    try:
        await _ensure_user_matches_token(user_id=user_id, credentials=credentials, auth_service=student_auth_service)
        quiz = await data_service.get_quiz(quiz_id)
        if not quiz:
            raise HTTPException(status_code=404, detail=f'Quiz {quiz_id} not found')
        effective_course_id = str(payload.course_id or quiz.get('course_id') or '').strip()
        if effective_course_id and effective_course_id != 'default-course':
            await _ensure_active_course_access(data_service=data_service, user_id=user_id, course_id=effective_course_id)
        questions = quiz.get('questions') or []
        provided = payload.answers
        ordered_answers: list = []
        if isinstance(provided, list):
            ordered_answers = provided
        elif isinstance(provided, dict):
            for idx, q in enumerate(questions):
                qid = q.get('id') or f'q{idx + 1}'
                ordered_answers.append(provided.get(qid))
        else:
            raise HTTPException(status_code=400, detail='Invalid answers format')

        def _normalize_correct_index(q: dict) -> int:
            try:
                for key in ('correct_answer', 'correct_index', 'answer_index', 'correct'):
                    if q.get(key) is not None:
                        val = q.get(key)
                        if isinstance(val, (int, float)):
                            return int(val)
                        if isinstance(val, str):
                            s = val.strip().lower()
                            mapping = {'a': 0, '1': 0, 'ก': 0, 'b': 1, '2': 1, 'ข': 1, 'c': 2, '3': 2, 'ค': 2, 'd': 3, '4': 3, 'ง': 3}
                            if s in mapping:
                                return mapping[s]
                            import re
                            m = re.search('(\\d+)', s)
                            if m:
                                n = int(m.group(1)) - 1
                                if n >= 0:
                                    return n
                            options = q.get('choices') or q.get('options') or []
                            for idx, opt in enumerate(options):
                                if str(opt).strip().lower() == s:
                                    return idx
                if 'answer' in q and isinstance(q['answer'], (int, float)):
                    return int(q['answer'])
            except Exception:
                pass
            return -1
        correct_indices = []
        for q in questions:
            correct_indices.append(_normalize_correct_index(q))
        total_questions = len(questions)
        correct_count = 0
        for i in range(min(len(ordered_answers), total_questions)):
            try:
                ai = int(ordered_answers[i]) if ordered_answers[i] is not None else None
            except Exception:
                ai = None
            if ai is not None and correct_indices[i] == ai:
                correct_count += 1
        score = int(round(correct_count / total_questions * 100)) if total_questions > 0 else 0
        result = {'answers': ordered_answers, 'correct_count': correct_count, 'total_questions': total_questions, 'score': score, 'time_spent_seconds': payload.time_spent_seconds or 0, 'course_id': payload.course_id or quiz.get('course_id', ''), 'lesson_id': payload.lesson_id or ''}
        if isinstance(payload.per_question_time_seconds, dict):
            clean_question_times: Dict[str, int] = {}
            for qid, spent in payload.per_question_time_seconds.items():
                if qid is None:
                    continue
                try:
                    normalized = max(0, int(spent))
                except Exception:
                    normalized = 0
                clean_question_times[str(qid)] = normalized
            result['per_question_time_seconds'] = clean_question_times
        else:
            result['per_question_time_seconds'] = {}
        if isinstance(payload.confidence_by_question, dict):
            clean_confidence: Dict[str, str] = {}
            for qid, confidence in payload.confidence_by_question.items():
                if qid is None:
                    continue
                value = str(confidence).strip().lower() if confidence is not None else ''
                if value in ('confident', 'not_confident'):
                    clean_confidence[str(qid)] = value
            result['confidence_by_question'] = clean_confidence
        else:
            result['confidence_by_question'] = {}
        per_question_time_list: List[int] = []
        confidence_list: List[Optional[str]] = []
        for idx, q in enumerate(questions):
            qid = str(q.get('id') or f'q{idx + 1}')
            per_question_time_list.append(int(result['per_question_time_seconds'].get(qid, 0)))
            confidence_list.append(result['confidence_by_question'].get(qid))
        result['per_question_time_list'] = per_question_time_list
        result['confidence_list'] = confidence_list
        result_id = await data_service.create_quiz_result(user_id, quiz_id, result)
        invalidate_summary_cache = getattr(data_service, 'invalidate_student_analysis_summary_cache', None)
        if effective_course_id and callable(invalidate_summary_cache):
            await invalidate_summary_cache(user_id=user_id, course_id=effective_course_id)
        return {'message': 'Submission recorded', 'result_id': result_id, 'user_id': user_id, 'quiz_id': quiz_id, 'score': score, 'correct_count': correct_count, 'total_questions': total_questions, 'time_spent_seconds': result.get('time_spent_seconds', 0), 'per_question_time_seconds': result.get('per_question_time_seconds', {}), 'confidence_by_question': result.get('confidence_by_question', {})}
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f'Error submitting quiz {quiz_id} for user {user_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to submit quiz answers')


async def record_user_learning_activity(user_id: str, payload: LearningActivityPayload, credentials: Optional[HTTPAuthorizationCredentials]=Depends(STUDENT_BEARER_OPTIONAL), student_auth_service: StudentAuthService=Depends(_get_student_auth_service), data_service=Depends(get_data_service)):
    """Record a durable lesson-view activity day for dashboard consistency."""
    try:
        await _ensure_user_matches_token(user_id=user_id, credentials=credentials, auth_service=student_auth_service)
        course_id = str(payload.course_id or '').strip()
        if not course_id:
            raise HTTPException(status_code=400, detail='course_id is required')
        active_enrollment = None
        if course_id != 'default-course':
            access = await _ensure_active_course_access(data_service=data_service, user_id=user_id, course_id=course_id)
            active_enrollment = access['enrollment']
        recorder = getattr(data_service, 'record_learning_activity', None)
        if not callable(recorder):
            raise HTTPException(status_code=500, detail='Learning activity storage is unavailable')
        result = await recorder(user_id=user_id, course_id=course_id, lesson_id=payload.lesson_id, activity_day=payload.activity_day, activity_days=payload.activity_days, enrollment=active_enrollment)
        if not result:
            raise HTTPException(status_code=404, detail='Enrollment not found')
        return {'message': 'Learning activity recorded', **result}
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f'Error recording learning activity for {user_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to record learning activity')


async def get_user_quiz_results(user_id: str, quiz_id: str, data_service=Depends(get_data_service)):
    """Return submission history for a user and quiz."""
    try:
        results = await data_service.get_user_quiz_results(user_id, quiz_id)
        return {'user_id': user_id, 'quiz_id': quiz_id, 'total_results': len(results), 'results': results}
    except Exception as e:
        app_logger.error(f'Error getting quiz results for user {user_id}, quiz {quiz_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to get quiz results')


async def get_course(course_id: str, data_service=Depends(get_data_service)):
    """Get course by ID."""
    try:
        course = await data_service.get_course(course_id)
        if not course:
            raise HTTPException(status_code=404, detail=f'Course {course_id} not found')
        return course
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f'Error retrieving course {course_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to retrieve course')


async def get_course_learning_overview(course_id: str, user_id: Optional[str]=None, credentials: Optional[HTTPAuthorizationCredentials]=Depends(STUDENT_BEARER_OPTIONAL), student_auth_service: StudentAuthService=Depends(_get_student_auth_service), data_service=Depends(get_data_service)):
    """Return compact student course detail data in one request."""
    try:
        if user_id:
            await _ensure_user_matches_token(user_id=user_id, credentials=credentials, auth_service=student_auth_service)
            await _ensure_active_course_access(data_service=data_service, user_id=user_id, course_id=course_id)
        get_overview = getattr(data_service, 'get_course_learning_overview', None)
        if callable(get_overview):
            overview = await get_overview(course_id, user_id=user_id)
        else:
            course, lessons, quizzes, quiz_results = await asyncio.gather(data_service.get_course(course_id), data_service.get_course_lessons(course_id), data_service.get_quizzes_by_course(course_id), data_service.get_user_quiz_results(user_id, course_id=course_id) if user_id else asyncio.sleep(0, result=[]))
            overview = {'course_id': course_id, 'user_id': user_id, 'course': course, 'enrollment': None, 'lessons': lessons, 'quizzes': quizzes, 'quiz_results': quiz_results, 'generated_at': datetime.utcnow().isoformat()}
        if not overview.get('course'):
            raise HTTPException(status_code=404, detail=f'Course {course_id} not found')
        return overview
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f'Error building learning overview for {course_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to get learning overview')


async def list_all_courses(data_service=Depends(get_data_service)):
    """List all courses available on the platform (active only)."""
    try:
        courses = await data_service.get_all_courses()
        return {'total_courses': len(courses), 'courses': courses}
    except Exception as e:
        app_logger.error(f'Error listing all courses: {e}')
        raise HTTPException(status_code=500, detail='Failed to list courses')


async def create_promptpay_payment_intent(body: PromptPayCreateIntentRequest, data_service=Depends(get_data_service)):
    """Create a Stripe PromptPay PaymentIntent for a course purchase."""
    settings = get_settings()
    if not settings.stripe_private_key or not settings.stripe_public_key:
        raise HTTPException(status_code=500, detail='Stripe keys are not configured (STRIPE_PRIVATE_KEY / STRIPE_PUBLIC_KEY)')
    user_id = str(body.user_id or '').strip()
    course_id = str(body.course_id or '').strip()
    if not user_id or not course_id:
        raise HTTPException(status_code=400, detail='user_id and course_id are required')
    course = await data_service.get_course(course_id)
    if not course:
        raise HTTPException(status_code=404, detail='Course not found')
    existing_enrollment_with_schedule = await _get_existing_enrollment_with_schedule(data_service=data_service, user_id=user_id, course_id=course_id)
    has_existing_active_enrollment = bool(existing_enrollment_with_schedule and (not existing_enrollment_with_schedule['schedule']['is_expired']))
    allowed_prices: List[float] = []
    base_price_raw = course.get('price')
    try:
        base_price = float(base_price_raw or 0)
    except Exception:
        base_price = 0.0
    if base_price > 0:
        allowed_prices.append(round(base_price, 2))
    pricing_plans = course.get('pricing_plans')
    if isinstance(pricing_plans, list):
        for plan in pricing_plans:
            if not isinstance(plan, dict):
                continue
            try:
                p = float(plan.get('price', 0))
            except Exception:
                p = 0.0
            if p > 0:
                allowed_prices.append(round(p, 2))
    unique_allowed_prices = sorted(set(allowed_prices))
    requested_amount = body.amount_thb
    if requested_amount is None:
        amount_thb = unique_allowed_prices[0] if unique_allowed_prices else 0.0
    else:
        try:
            amount_thb = round(float(requested_amount), 2)
        except Exception:
            raise HTTPException(status_code=400, detail='Invalid amount_thb')
        if unique_allowed_prices and amount_thb not in unique_allowed_prices:
            raise HTTPException(status_code=400, detail='Selected amount does not match course pricing')
    if amount_thb <= 0:
        raise HTTPException(status_code=400, detail='This course is free, please enroll directly without payment')
    amount_satang = int(round(amount_thb * 100))
    if amount_satang < 100:
        raise HTTPException(status_code=400, detail='Amount is too low for payment')
    course_name = str(course.get('name') or course.get('title') or 'คอร์สเรียน').strip()
    stripe_payload = {'amount': str(amount_satang), 'currency': 'thb', 'payment_method_types[]': 'promptpay', 'description': f'Course payment: {course_name}', 'metadata[user_id]': user_id, 'metadata[course_id]': course_id, 'metadata[payment_type]': 'course_enrollment', 'metadata[plan_label]': str(body.plan_label or '').strip(), 'metadata[duration_months]': str(body.duration_months or '')}
    intent = await _stripe_request(method='POST', path='/payment_intents', secret_key=settings.stripe_private_key, data=stripe_payload)
    payment_intent_id = str(intent.get('id') or '').strip()
    client_secret = str(intent.get('client_secret') or '').strip()
    if not payment_intent_id or not client_secret:
        raise HTTPException(status_code=500, detail='Failed to create payment intent')
    return PromptPayCreateIntentResponse(payment_intent_id=payment_intent_id, client_secret=client_secret, publishable_key=settings.stripe_public_key, amount=amount_satang, currency=str(intent.get('currency') or 'thb').upper(), payment_status=str(intent.get('status') or 'requires_payment_method'), already_enrolled=has_existing_active_enrollment)


async def confirm_promptpay_payment_and_enroll(body: PromptPayConfirmRequest, data_service=Depends(get_data_service)):
    """Verify payment status from Stripe and enroll the user when payment succeeds."""
    user_id = str(body.user_id or '').strip()
    course_id = str(body.course_id or '').strip()
    payment_intent_id = str(body.payment_intent_id or '').strip()
    if not user_id or not course_id or (not payment_intent_id):
        raise HTTPException(status_code=400, detail='user_id, course_id, and payment_intent_id are required')
    return await _complete_promptpay_payment(payment_intent_id=payment_intent_id, data_service=data_service, expected_user_id=user_id, expected_course_id=course_id)


async def handle_stripe_payment_webhook(request: Request, stripe_signature: Optional[str]=Header(default=None, alias='Stripe-Signature'), data_service=Depends(get_data_service)):
    """Handle Stripe payment webhooks for payment fulfillment."""
    settings = get_settings()
    if not settings.stripe_webhook_secret:
        raise HTTPException(status_code=500, detail='Stripe webhook secret is not configured')
    payload = await request.body()
    _verify_stripe_webhook_signature(payload=payload, signature_header=stripe_signature or '', webhook_secret=settings.stripe_webhook_secret)
    try:
        event = json.loads(payload.decode('utf-8'))
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid Stripe webhook payload')
    if not isinstance(event, dict):
        raise HTTPException(status_code=400, detail='Invalid Stripe webhook payload')
    event_type = str(event.get('type') or '').strip()
    data = event.get('data') if isinstance(event.get('data'), dict) else {}
    obj = data.get('object') if isinstance(data.get('object'), dict) else {}
    if event_type == 'payment_intent.succeeded':
        payment_intent_id = str(obj.get('id') or '').strip()
        if not payment_intent_id:
            raise HTTPException(status_code=400, detail='Missing payment intent id in webhook')
        await _complete_promptpay_payment(payment_intent_id=payment_intent_id, data_service=data_service)
    return {'received': True}


async def get_user_payment_history(user_id: str, data_service=Depends(get_data_service)):
    """Return student payment history from enrollment records."""
    try:
        get_with_aliases = getattr(data_service, 'get_user_enrollments_with_aliases', None)
        if callable(get_with_aliases):
            enrollments = await get_with_aliases(user_id)
        else:
            enrollments = await data_service.get_user_enrollments(user_id)
        rows: List[Dict[str, Any]] = []
        for enrollment in enrollments:
            course_id = str(enrollment.get('course_id') or '').strip()
            if not course_id:
                continue
            course = await data_service.get_course(course_id)
            course_name = str((course or {}).get('name') or (course or {}).get('title') or '').strip() or 'คอร์ส'
            payment_events = _normalize_payment_history(enrollment)
            if not payment_events:
                schedule = _build_enrollment_schedule(started_at_raw=enrollment.get('started_at') or enrollment.get('enrolled_at'), expires_at_raw=enrollment.get('expires_at'), duration_months_raw=enrollment.get('duration_months'))
                rows.append({'enrollment_id': enrollment.get('enrollment_id'), 'course_id': course_id, 'course_name': course_name, 'order_id': enrollment.get('order_id') or _build_payment_order_id(enrollment.get('paid_at') or enrollment.get('enrolled_at'), enrollment.get('payment_intent_id')), 'payment_provider': enrollment.get('payment_provider') or 'stripe', 'payment_type': enrollment.get('payment_type') or 'manual', 'payment_intent_id': enrollment.get('payment_intent_id'), 'stripe_charge_id': enrollment.get('stripe_charge_id'), 'receipt_number': enrollment.get('receipt_number'), 'receipt_url': enrollment.get('receipt_url'), 'payment_status': enrollment.get('payment_status') or 'active', 'paid_amount_thb': None, 'paid_currency': enrollment.get('paid_currency') or 'THB', 'billing_email': enrollment.get('billing_email'), 'plan_label': enrollment.get('plan_label'), 'duration_months': schedule['duration_months'], 'paid_at': enrollment.get('paid_at') or enrollment.get('enrolled_at'), 'enrolled_at': enrollment.get('enrolled_at'), 'started_at': schedule['started_at'], 'expires_at': schedule['expires_at'], 'is_expired': schedule['is_expired'], 'days_remaining': schedule['days_remaining'], 'in_system': True})
                continue
            for event in payment_events:
                schedule = _build_enrollment_schedule(started_at_raw=event.get('started_at') or enrollment.get('started_at') or enrollment.get('enrolled_at'), expires_at_raw=event.get('expires_at') or enrollment.get('expires_at'), duration_months_raw=event.get('duration_months'))
                rows.append({'enrollment_id': enrollment.get('enrollment_id'), 'course_id': course_id, 'course_name': course_name, 'order_id': event.get('order_id'), 'payment_provider': event.get('payment_provider') or 'stripe', 'payment_type': event.get('payment_type') or 'promptpay', 'payment_intent_id': event.get('payment_intent_id'), 'stripe_charge_id': event.get('stripe_charge_id'), 'receipt_number': event.get('receipt_number'), 'receipt_url': event.get('receipt_url'), 'payment_status': event.get('payment_status') or 'active', 'paid_amount_thb': event.get('paid_amount_thb'), 'paid_currency': event.get('paid_currency') or 'THB', 'billing_email': event.get('billing_email'), 'plan_label': event.get('plan_label'), 'duration_months': schedule['duration_months'], 'paid_at': event.get('paid_at') or enrollment.get('enrolled_at'), 'enrolled_at': enrollment.get('enrolled_at'), 'started_at': schedule['started_at'], 'expires_at': schedule['expires_at'], 'is_expired': schedule['is_expired'], 'days_remaining': schedule['days_remaining'], 'in_system': True})
        rows.sort(key=lambda row: str(row.get('paid_at') or row.get('enrolled_at') or ''), reverse=True)
        await _hydrate_payment_history_receipts(rows)
        return {'user_id': user_id, 'total': len(rows), 'rows': rows}
    except Exception as e:
        app_logger.error(f'Error fetching payment history for {user_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to get payment history')


async def get_student_chat_energy_status(user_id: str, credentials: Optional[HTTPAuthorizationCredentials]=Depends(STUDENT_BEARER_OPTIONAL), student_auth_service: StudentAuthService=Depends(_get_student_auth_service), data_service=Depends(get_data_service)):
    """Get current student's chat energy status (daily THB budget)."""
    try:
        normalized_user_id = str(user_id or '').strip()
        if not normalized_user_id:
            raise HTTPException(status_code=400, detail='user_id is required')
        if credentials and str(credentials.credentials or '').strip():
            await _ensure_user_matches_token(user_id=normalized_user_id, credentials=credentials, auth_service=student_auth_service)
        status = await data_service.get_student_chat_energy_status(normalized_user_id)
        return {'user_id': normalized_user_id, **_to_chat_energy_response(status)}
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f'Failed to get chat energy status for user {user_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to get chat energy status')


async def enroll_user_in_course(user_id: str=Form(...), course_id: str=Form(...), enrollment_mode: str=Form('standard'), progress: int=Form(0), completed_quizzes: int=Form(0), total_quizzes: int=Form(0), completed_questions: int=Form(0), total_questions: int=Form(0), data_service=Depends(get_data_service)):
    """Enroll a user in a course."""
    try:
        app_logger.info(f'Enrolling user {user_id} in course {course_id}')
        enrollment_mode_normalized = str(enrollment_mode or 'standard').strip().lower()
        if enrollment_mode_normalized not in {'standard', 'trial'}:
            raise HTTPException(status_code=400, detail="enrollment_mode must be either 'standard' or 'trial'")
        course = await data_service.get_course(course_id)
        if not course:
            raise HTTPException(status_code=404, detail=f'Course {course_id} not found')
        get_user = getattr(data_service, 'get_user', None)
        user_data = await get_user(user_id) if callable(get_user) else None
        trial_override = _extract_trial_override(user_data or {})
        reset_trial_override_after_enroll = False
        if enrollment_mode_normalized == 'trial':
            existing_enrollment = await _get_user_course_enrollment(data_service=data_service, user_id=user_id, course_id=course_id)
            if existing_enrollment:
                raise HTTPException(status_code=400, detail='TRIAL_NOT_ALLOWED: user already has enrollment for this course')
            user_enrollments = await data_service.get_user_enrollments(user_id)
            trial_used_from_enrollments = any((_is_trial_enrollment(enrollment) for enrollment in user_enrollments))
            effective_trial = _resolve_effective_trial_used(trial_used_from_enrollments=trial_used_from_enrollments, override_mode=trial_override.get('mode'))
            if bool(effective_trial.get('trial_used')):
                if str(effective_trial.get('trial_status_source')) == 'admin_override':
                    raise HTTPException(status_code=400, detail='TRIAL_ALREADY_USED: blocked by admin override mode=used')
                raise HTTPException(status_code=400, detail='TRIAL_ALREADY_USED: each user can only use trial once')
            reset_trial_override_after_enroll = str(trial_override.get('mode') or '').strip() == 'available'
        now = datetime.utcnow()
        trial_started_at = now.isoformat()
        trial_expires_at = None
        if enrollment_mode_normalized == 'trial':
            trial_expires_at = (now + timedelta(days=1)).isoformat()
        schedule = _build_enrollment_schedule(started_at_raw=trial_started_at, expires_at_raw=trial_expires_at, duration_months_raw=None)
        default_last_activity = 'เริ่มทดลองเรียน' if enrollment_mode_normalized == 'trial' else 'เพิ่งเข้าร่วม'
        enrollment_data = {'progress': progress, 'completed_quizzes': completed_quizzes, 'total_quizzes': total_quizzes, 'completed_questions': completed_questions, 'total_questions': total_questions, 'last_activity': f'{completed_questions}/{total_questions} คำถาม' if total_questions > 0 else default_last_activity, 'started_at': schedule['started_at'], 'expires_at': schedule['expires_at'], 'enrollment_source': 'trial' if enrollment_mode_normalized == 'trial' else 'manual', 'enrollment_type': enrollment_mode_normalized}
        if enrollment_mode_normalized == 'trial':
            enrollment_data['trial_consumed_at'] = trial_started_at
            enrollment_data['trial_expires_at'] = schedule['expires_at']
        enrollment_id = await data_service.enroll_user_in_course(user_id, course_id, enrollment_data)
        if enrollment_mode_normalized == 'trial' and reset_trial_override_after_enroll:
            try:
                await _set_user_trial_override(data_service=data_service, user_id=user_id, mode='auto', updated_by='system', reason='Auto reset after successful trial enrollment')
            except Exception as override_exc:
                app_logger.warning(f'Failed to auto reset trial override after trial enrollment for user {user_id}: {override_exc}')
        return {'message': 'Trial enrollment created successfully' if enrollment_mode_normalized == 'trial' else 'User enrolled successfully', 'enrollment_id': enrollment_id, 'user_id': user_id, 'course_id': course_id, 'enrollment_mode': enrollment_mode_normalized, 'is_trial': enrollment_mode_normalized == 'trial', 'expires_at': schedule['expires_at']}
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f'Error enrolling user {user_id} in course {course_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to enroll user')


async def get_user_enrolled_courses(user_id: str, limit: int=50, credentials: Optional[HTTPAuthorizationCredentials]=Depends(STUDENT_BEARER_OPTIONAL), student_auth_service: StudentAuthService=Depends(_get_student_auth_service), data_service=Depends(get_data_service)):
    """Get courses the user is enrolled in."""
    try:
        if credentials:
            await _ensure_user_matches_token(user_id=user_id, credentials=credentials, auth_service=student_auth_service)
        enrolled_courses = await data_service.get_enrolled_courses_for_user(user_id, limit=limit)
        formatted_courses = []
        for course in enrolled_courses:
            formatted_courses.append(_format_student_course(course))
        return formatted_courses
    except HTTPException:
        raise
    except Exception as e:
        app_logger.exception(f'Error getting enrolled courses for user {user_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to get enrolled courses')


async def get_dashboard_learning_summary(user_id: str, include_ai: bool=False, course_limit: int=50, credentials: Optional[HTTPAuthorizationCredentials]=Depends(STUDENT_BEARER_OPTIONAL), student_auth_service: StudentAuthService=Depends(_get_student_auth_service), data_service=Depends(get_data_service)):
    """Return enrolled courses plus computed learning stats for dashboard views."""
    del include_ai
    try:
        if credentials:
            await _ensure_user_matches_token(user_id=user_id, credentials=credentials, auth_service=student_auth_service)
        safe_limit = max(1, min(200, int(course_limit or 50)))
        get_inputs = getattr(data_service, 'get_dashboard_learning_inputs', None)
        if not callable(get_inputs):
            enrolled_courses = await data_service.get_enrolled_courses_for_user(user_id, limit=safe_limit)
            quiz_results = await data_service.get_user_quiz_results(user_id)
            lessons: List[Dict[str, Any]] = []
            quizzes: List[Dict[str, Any]] = []
            for course in enrolled_courses:
                course_id = str(course.get('course_id') or course.get('id') or '')
                if not course_id:
                    continue
                lessons.extend(await data_service.get_course_lessons(course_id))
                quizzes.extend(await data_service.get_quizzes_by_course(course_id))
            formatted_courses = [_format_student_course(course) for course in enrolled_courses]
            return {'user_id': user_id, 'courses': formatted_courses, 'course_stats': _build_dashboard_course_stats(enrolled_courses, lessons, quizzes, quiz_results), 'generated_at': datetime.utcnow().isoformat()}
        inputs = await get_inputs(user_id, limit=safe_limit)
        candidate_user_ids = inputs.get('candidate_user_ids') or []
        candidate_rank = {str(candidate_id): idx for idx, candidate_id in enumerate(candidate_user_ids)}
        courses_by_id = {str(course.get('course_id') or ''): course for course in inputs.get('courses', []) if str(course.get('course_id') or '')}
        merged_courses = []
        for enrollment in inputs.get('enrollments', []):
            course_id = str(enrollment.get('course_id') or '').strip()
            course = courses_by_id.get(course_id)
            if not course:
                continue
            merged_courses.append(_merge_course_with_enrollment(course, enrollment))
        deduped_results = []
        seen_result_ids = set()
        for row in sorted(inputs.get('quiz_results', []), key=lambda item: (candidate_rank.get(str(item.get('user_id') or ''), len(candidate_rank)), str(item.get('submitted_at') or item.get('created_at') or ''))):
            result_id = str(row.get('result_id') or '').strip()
            if result_id and result_id in seen_result_ids:
                continue
            if result_id:
                seen_result_ids.add(result_id)
            deduped_results.append(row)
        deduped_results.sort(key=lambda row: str(row.get('submitted_at') or row.get('created_at') or ''), reverse=True)
        return {'user_id': user_id, 'courses': [_format_student_course(course) for course in merged_courses], 'course_stats': _build_dashboard_course_stats(merged_courses, inputs.get('lessons', []), inputs.get('quizzes', []), deduped_results), 'generated_at': datetime.utcnow().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        app_logger.exception(f'Error building dashboard learning summary for {user_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to get dashboard learning summary')


async def get_course_mock_exam_leaderboard(course_id: str, limit: int=50, data_service=Depends(get_data_service)):
    """Get student ranking by average score from mock exams in a course."""
    try:
        course = await data_service.get_course(course_id)
        if not course:
            raise HTTPException(status_code=404, detail=f'Course {course_id} not found')
        quizzes = await data_service.get_quizzes_by_course(course_id)
        mock_quiz_ids = {str(item.get('quiz_id') or '') for item in quizzes if str(item.get('document_type') or '').strip().lower() == 'mock_exam' and str(item.get('quiz_id') or '').strip()}
        enrollments = await data_service.get_course_enrollments(course_id)
        user_ids = [str(enrollment.get('user_id') or '').strip() for enrollment in enrollments if str(enrollment.get('user_id') or '').strip()]
        get_users_by_ids = getattr(data_service, 'get_users_by_ids', None)
        users_by_id: Dict[str, Dict[str, Any]] = {}
        if callable(get_users_by_ids) and user_ids:
            users_by_id = {str(user.get('user_id') or ''): user for user in await get_users_by_ids(user_ids, limit=len(user_ids)) if str(user.get('user_id') or '')}
        get_results_for_course = getattr(data_service, 'get_quiz_results_for_course', None)
        results_by_user: Dict[str, List[Dict[str, Any]]] = {user_id: [] for user_id in user_ids}
        if callable(get_results_for_course) and user_ids and mock_quiz_ids:
            course_results = await get_results_for_course(course_id, user_ids=user_ids, quiz_ids=list(mock_quiz_ids), limit=max(1000, len(user_ids) * max(1, len(mock_quiz_ids)) * 20))
            for result in course_results:
                result_user_id = str(result.get('user_id') or '').strip()
                if result_user_id in results_by_user:
                    results_by_user[result_user_id].append(result)

        def _display_name(user_id: str, user_data: Optional[Dict[str, Any]]) -> str:
            user_data = user_data or {}
            onboarding_profile = user_data.get('onboarding_profile') if isinstance(user_data, dict) else {}
            nickname = str((onboarding_profile or {}).get('nickname') or '').strip()
            given_name = str((user_data or {}).get('given_name') or '').strip()
            family_name = str((user_data or {}).get('family_name') or '').strip()
            full_name = f'{given_name} {family_name}'.strip()
            fallback_name = str((user_data or {}).get('name') or '').strip()
            fallback_email = str((user_data or {}).get('email') or '').strip()
            email_name = fallback_email.split('@')[0] if '@' in fallback_email else ''
            return nickname or full_name or (fallback_name if fallback_name and fallback_name != user_id else '') or email_name or user_id or 'ผู้เรียน'
        rankings = []
        for enrollment in enrollments:
            user_id = str(enrollment.get('user_id') or '').strip()
            if not user_id:
                continue
            user_data = users_by_id.get(user_id)
            if not user_data and (not callable(get_users_by_ids)):
                user_data = await data_service.get_user(user_id)
            display_name = _display_name(user_id, user_data)
            if callable(get_results_for_course):
                candidate_results = results_by_user.get(user_id, [])
            else:
                candidate_results = await data_service.get_user_quiz_results(user_id, course_id=course_id)
            mock_results = []
            for item in candidate_results:
                result_quiz_id = str(item.get('quiz_id') or '').strip()
                result_course_id = str(item.get('course_id') or '').strip()
                if result_course_id and result_course_id != str(course_id):
                    continue
                if result_quiz_id in mock_quiz_ids:
                    score = item.get('score')
                    if isinstance(score, (int, float)):
                        mock_results.append(item)
            if not mock_results:
                continue
            scores = [float(item.get('score')) for item in mock_results]
            time_values = [float(item.get('time_spent_seconds')) for item in mock_results if isinstance(item.get('time_spent_seconds'), (int, float)) and float(item.get('time_spent_seconds')) >= 0]
            average_score = round(sum(scores) / len(scores), 2)
            best_score = round(max(scores), 2)
            average_time_seconds = round(sum(time_values) / len(time_values), 2) if time_values else None
            latest_result = max(mock_results, key=lambda row: str(row.get('submitted_at') or ''))
            rankings.append({'user_id': user_id, 'display_name': display_name, 'average_score': average_score, 'best_score': best_score, 'average_time_seconds': average_time_seconds, 'attempt_count': len(scores), 'last_submitted_at': latest_result.get('submitted_at'), 'enrolled_at': enrollment.get('enrolled_at')})
        rankings.sort(key=lambda row: (float(row.get('average_score') or 0), int(row.get('attempt_count') or 0), str(row.get('last_submitted_at') or '')), reverse=True)
        for index, row in enumerate(rankings, start=1):
            row['rank'] = index
        if isinstance(limit, int) and limit > 0:
            rankings = rankings[:limit]
        return {'course_id': course_id, 'course_name': course.get('name') or course.get('title') or 'คอร์สเรียน', 'metric': 'average_mock_exam_score', 'total_students': len(rankings), 'mock_exam_count': len(mock_quiz_ids), 'rankings': rankings, 'generated_at': datetime.utcnow().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f'Error getting mock exam leaderboard for course {course_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to get mock exam leaderboard')


async def get_user_enrollments(user_id: str, limit: int=50, data_service=Depends(get_data_service)):
    """Get all enrollments for a specific user."""
    try:
        try:
            enrollments = await data_service.get_user_enrollments(user_id, limit)
        except TypeError:
            enrollments = await data_service.get_user_enrollments(user_id)
        enrollments = list(enrollments or [])
        enrollments.sort(key=lambda row: str(row.get('enrolled_at') or ''), reverse=True)
        if isinstance(limit, int) and limit > 0:
            enrollments = enrollments[:limit]
        return {'user_id': user_id, 'total_enrollments': len(enrollments), 'enrollments': enrollments}
    except Exception as e:
        app_logger.error(f'Error getting enrollments for user {user_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to get user enrollments')


async def update_enrollment(enrollment_id: str, progress: Optional[int]=Form(None), completed_quizzes: Optional[int]=Form(None), total_quizzes: Optional[int]=Form(None), completed_questions: Optional[int]=Form(None), total_questions: Optional[int]=Form(None), data_service=Depends(get_data_service)):
    """Update enrollment progress."""
    try:
        updates = {}
        if progress is not None:
            updates['progress'] = progress
        if completed_quizzes is not None:
            updates['completed_quizzes'] = completed_quizzes
        if total_quizzes is not None:
            updates['total_quizzes'] = total_quizzes
        if completed_questions is not None:
            updates['completed_questions'] = completed_questions
        if total_questions is not None:
            updates['total_questions'] = total_questions
        if completed_questions is not None and total_questions is not None:
            updates['last_activity'] = f'{completed_questions}/{total_questions} คำถาม'
        if not updates:
            raise HTTPException(status_code=400, detail='No updates provided')
        success = await data_service.update_enrollment(enrollment_id, updates)
        if success:
            return {'message': f'Enrollment {enrollment_id} updated successfully'}
        else:
            raise HTTPException(status_code=500, detail='Failed to update enrollment')
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f'Error updating enrollment {enrollment_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to update enrollment')


async def get_course_lessons(course_id: str, user_id: Optional[str]=None, credentials: Optional[HTTPAuthorizationCredentials]=Depends(STUDENT_BEARER_OPTIONAL), student_auth_service: StudentAuthService=Depends(_get_student_auth_service), data_service=Depends(get_data_service)):
    """Get all lessons for a specific course."""
    try:
        if user_id and str(course_id or '').strip():
            await _ensure_user_matches_token(user_id=user_id, credentials=credentials, auth_service=student_auth_service)
            await _ensure_active_course_access(data_service=data_service, user_id=user_id, course_id=course_id)
        try:
            lessons = await data_service.get_course_lessons(
                course_id=course_id, user_id=None, summary=True
            )
        except TypeError:
            lessons = await data_service.get_course_lessons(
                course_id=course_id, user_id=None
            )
        normalized_lessons = []
        for lesson_item in lessons:
            documents = []
            raw_documents = lesson_item.get('documents') or lesson_item.get('selectedDocuments') or lesson_item.get('selected_documents') or []
            for doc in raw_documents:
                if isinstance(doc, str):
                    documents.append({'id': doc, 'title': None, 'type': None})
                    continue
                if isinstance(doc, dict):
                    doc_id = doc.get('id') or doc.get('document_id')
                    if not doc_id:
                        continue
                    documents.append({'id': doc_id, 'title': doc.get('title'), 'type': doc.get('type')})
            quizzes = []
            raw_quizzes = lesson_item.get('quizzes') or lesson_item.get('selectedQuizzes') or lesson_item.get('selected_quizzes') or []
            for q in raw_quizzes:
                if isinstance(q, str):
                    quizzes.append({'id': q, 'title': None, 'questions': 0})
                    continue
                if isinstance(q, dict):
                    qid = q.get('id') or q.get('quiz_id') or q.get('document_id')
                    if not qid:
                        continue
                    questions_val = q.get('questions', 0)
                    if isinstance(questions_val, list):
                        questions_count = len(questions_val)
                    elif isinstance(questions_val, int):
                        questions_count = questions_val
                    else:
                        questions_count = 0
                    quizzes.append({'id': qid, 'title': q.get('title'), 'questions': questions_count})
            normalized_lessons.append({'id': lesson_item.get('lesson_id') or lesson_item.get('id'), 'title': lesson_item.get('title'), 'description': lesson_item.get('description', ''), 'order': lesson_item.get('order', 1), 'courseId': lesson_item.get('course_id') or lesson_item.get('courseId'), 'userId': lesson_item.get('user_id') or lesson_item.get('userId'), 'documents': documents, 'quizzes': quizzes, 'isPublished': lesson_item.get('isPublished', lesson_item.get('is_published', False)), 'createdAt': lesson_item.get('created_at') or lesson_item.get('createdAt'), 'updatedAt': lesson_item.get('updated_at') or lesson_item.get('updatedAt')})
        return LessonListResponse(lessons=normalized_lessons, total=len(normalized_lessons))
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f'Error retrieving lessons for course {course_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to retrieve lessons')


async def get_lesson(lesson_id: str, user_id: Optional[str]=None, data_service=Depends(get_data_service)):
    """Get a specific lesson by ID using the configured data service."""
    try:
        lesson_item = await data_service.get_lesson(lesson_id)
        if not lesson_item:
            raise HTTPException(status_code=404, detail='Lesson not found')
        if lesson_item.get('status') not in (None, 'active'):
            raise HTTPException(status_code=404, detail='Lesson not found')
        documents = []
        for doc in lesson_item.get('selected_documents', []) or []:
            if isinstance(doc, dict):
                if not doc.get('id'):
                    continue
                documents.append({'id': doc.get('id'), 'title': doc.get('title'), 'type': doc.get('type')})
        quizzes = []
        for q in lesson_item.get('selected_quizzes', []) or []:
            if isinstance(q, str):
                quizzes.append({'id': q, 'title': None, 'questions': 0})
                continue
            if isinstance(q, dict):
                qid = q.get('id') or q.get('quiz_id') or q.get('document_id')
                if not qid:
                    continue
                questions_val = q.get('questions', 0)
                if isinstance(questions_val, list):
                    questions_count = len(questions_val)
                elif isinstance(questions_val, int):
                    questions_count = questions_val
                else:
                    questions_count = 0
                quizzes.append({'id': qid, 'title': q.get('title'), 'questions': questions_count})
        lesson = {'id': lesson_item.get('lesson_id'), 'title': lesson_item.get('title'), 'description': lesson_item.get('description'), 'order': lesson_item.get('order', 1), 'courseId': lesson_item.get('course_id'), 'userId': lesson_item.get('user_id'), 'documents': documents, 'quizzes': quizzes, 'isPublished': lesson_item.get('is_published', False), 'createdAt': lesson_item.get('created_at'), 'updatedAt': lesson_item.get('updated_at')}
        return LessonResponse(success=True, message='Lesson retrieved successfully', lesson=lesson)
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f'Error retrieving lesson {lesson_id}: {e}')
        raise HTTPException(status_code=500, detail='Failed to retrieve lesson')
