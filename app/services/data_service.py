"""Supabase-backed data service for student API persistence."""

import asyncio
import hashlib
import math
import re
import time
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from postgrest.exceptions import APIError
from postgrest.types import ReturnMethod

from app.core.config import get_settings
from app.core.logging import app_logger
from app.services.supabase_service import get_supabase_service

CHAT_ENERGY_CONFIG_USER_ID = "__system_chat_energy_config__"
DEFAULT_CHAT_ENERGY_DAILY_LIMIT_THB = 2.0
LEARNING_ACTIVITY_TIME_ZONE = ZoneInfo("Asia/Bangkok")
LEARNING_ACTIVITY_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MAX_LEARNING_ACTIVITY_DAYS = 90
READ_CACHE_TTL_SECONDS = 45.0

COURSE_SUMMARY_SELECT = ",".join(
    (
        "course_id",
        "user_id",
        "instructor_id",
        "name",
        "title",
        "category",
        "status",
        "created_at",
        "updated_at",
        "description:data->>description",
        "detail:data->>detail",
        "instructor:data->>instructor",
        "teacher_name:data->>teacher_name",
        "grade_level:data->>grade_level",
        "course_format:data->>course_format",
        "thumbnail_url:data->>thumbnail_url",
        "image_url:data->>image_url",
        "preview_image_url:data->>preview_image_url",
        "purchase_preview_image_url:data->>purchase_preview_image_url",
        "price:data->price",
        "pricing_plans:data->pricing_plans",
        "topics:data->topics",
        "tags:data->tags",
        "benefits:data->benefits",
        "content_items:data->content_items",
        "target_profile:data->>target_profile",
        "structure_summary:data->>structure_summary",
    )
)

LESSON_SUMMARY_SELECT = ",".join(
    (
        "lesson_id",
        "course_id",
        "user_id",
        "status",
        "created_at",
        "updated_at",
        "title:data->>title",
        "description:data->>description",
        "order:data->order",
        "documents:data->documents",
        "selected_documents:data->selected_documents",
        "selectedDocuments:data->selectedDocuments",
        "quizzes:data->quizzes",
        "selected_quizzes:data->selected_quizzes",
        "selectedQuizzes:data->selectedQuizzes",
        "isPublished:data->isPublished",
        "is_published:data->is_published",
    )
)

QUIZ_SUMMARY_SELECT = ",".join(
    (
        "quiz_id",
        "course_id",
        "user_id",
        "lesson_id",
        "document_id",
        "status",
        "created_at",
        "updated_at",
        "title:data->>title",
        "name:data->>name",
        "description:data->>description",
        "document_type:data->>document_type",
        "difficulty:data->difficulty",
        "difficulty_avg:data->difficulty_avg",
        "duration_minutes:data->duration_minutes",
        "total_questions:data->total_questions",
        "topic:data->>topic",
        "topic_tag:data->>topic_tag",
        "topicTag:data->>topicTag",
        "subject:data->>subject",
        "subject_tag:data->>subject_tag",
        "subjectTag:data->>subjectTag",
        "selection_reasons:data->selection_reasons",
        "reasons:data->reasons",
        "pick_reasons:data->pick_reasons",
    )
)

QUIZ_RESULT_SUMMARY_SELECT = ",".join(
    (
        "result_id",
        "user_id",
        "quiz_id",
        "course_id",
        "submitted_at",
        "created_at",
        "updated_at",
        "score:data->score",
        "total_questions:data->total_questions",
        "correct_count:data->correct_count",
        "time_spent_seconds:data->time_spent_seconds",
        "lesson_id:data->>lesson_id",
    )
)

ENROLLMENT_SUMMARY_SELECT = ",".join(
    (
        "enrollment_id",
        "user_id",
        "course_id",
        "status",
        "enrolled_at",
        "expires_at",
        "created_at",
        "updated_at",
        "started_at:data->>started_at",
        "duration_months:data->duration_months",
        "enrollment_source:data->>enrollment_source",
        "enrollment_type:data->>enrollment_type",
        "payment_provider:data->>payment_provider",
        "payment_type:data->>payment_type",
        "payment_intent_id:data->>payment_intent_id",
        "payment_status:data->>payment_status",
        "paid_amount_thb:data->paid_amount_thb",
        "paid_currency:data->>paid_currency",
        "billing_email:data->>billing_email",
        "plan_label:data->>plan_label",
        "paid_at:data->>paid_at",
        "payment_history:data->payment_history",
        "trial_consumed_at:data->>trial_consumed_at",
        "trial_expires_at:data->>trial_expires_at",
        "progress:data->progress",
        "completed_quizzes:data->completed_quizzes",
        "total_quizzes:data->total_quizzes",
        "completed_questions:data->completed_questions",
        "total_questions:data->total_questions",
        "last_activity:data->>last_activity",
        "learning_activity_days:data->learning_activity_days",
        "last_learning_activity_at:data->>last_learning_activity_at",
        "last_lesson_id:data->>last_lesson_id",
    )
)


TABLE_PK = {
    "profiles": "user_id",
    "courses": "course_id",
    "lessons": "lesson_id",
    "quizzes": "quiz_id",
    "question_bank_items": "item_id",
    "enrollments": "enrollment_id",
    "quiz_results": "result_id",
    "files": "file_id",
    "chat_messages": "message_id",
    "platform_config": "config_key",
    "invitations": "invitation_id",
}

TABLE_COLUMNS = {
    "profiles": {
        "user_id",
        "email",
        "username",
        "name",
        "role",
        "status",
        "given_name",
        "family_name",
        "student_id",
        "created_at",
        "updated_at",
        "data",
    },
    "courses": {
        "course_id",
        "user_id",
        "instructor_id",
        "name",
        "title",
        "category",
        "status",
        "created_at",
        "updated_at",
        "data",
    },
    "lessons": {
        "lesson_id",
        "course_id",
        "user_id",
        "status",
        "created_at",
        "updated_at",
        "data",
    },
    "quizzes": {
        "quiz_id",
        "course_id",
        "user_id",
        "lesson_id",
        "document_id",
        "status",
        "created_at",
        "updated_at",
        "data",
    },
    "question_bank_items": {
        "item_id",
        "course_id",
        "user_id",
        "source",
        "status",
        "created_at",
        "updated_at",
        "data",
    },
    "enrollments": {
        "enrollment_id",
        "user_id",
        "course_id",
        "status",
        "enrolled_at",
        "expires_at",
        "created_at",
        "updated_at",
        "data",
    },
    "quiz_results": {
        "result_id",
        "user_id",
        "quiz_id",
        "course_id",
        "submitted_at",
        "created_at",
        "updated_at",
        "data",
    },
    "files": {
        "file_id",
        "user_id",
        "storage_key",
        "content_type",
        "created_at",
        "updated_at",
        "data",
    },
    "chat_messages": {
        "message_id",
        "conversation_id",
        "user_id",
        "role",
        "created_at",
        "data",
    },
    "platform_config": {"config_key", "updated_at", "data"},
    "invitations": {
        "invitation_id",
        "course_id",
        "instructor_id",
        "student_id",
        "status",
        "created_at",
        "expires_at",
        "data",
    },
}


def _utcnow() -> str:
    return datetime.utcnow().isoformat()


class SupabaseTableAdapter:
    """Small table shim for legacy scan/put_item call sites."""

    def __init__(self, service: "SupabaseDataService", table_name: str):
        self.service = service
        self.table_name = table_name
        self.pk = TABLE_PK[table_name]

    def scan(self, **kwargs: Any) -> Dict[str, Any]:
        limit = int(kwargs.get("Limit") or 10000)
        rows = self.service._select_sync(self.table_name, limit=limit)
        return {"Items": rows}

    def get_item(self, Key: Dict[str, Any]) -> Dict[str, Any]:
        value = Key.get(self.pk)
        row = self.service._get_sync(self.table_name, value)
        return {"Item": row} if row else {}

    def put_item(self, Item: Dict[str, Any]) -> Dict[str, Any]:
        self.service._upsert_sync(self.table_name, Item)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def delete_item(self, Key: Dict[str, Any]) -> Dict[str, Any]:
        value = Key.get(self.pk)
        self.service._delete_sync(self.table_name, value)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def update_item(
        self,
        Key: Dict[str, Any],
        UpdateExpression: Optional[str] = None,
        ExpressionAttributeNames: Optional[Dict[str, str]] = None,
        ExpressionAttributeValues: Optional[Dict[str, Any]] = None,
        ReturnValues: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        current = self.service._get_sync(self.table_name, Key.get(self.pk)) or {
            self.pk: Key.get(self.pk)
        }
        updates: Dict[str, Any] = {}
        if UpdateExpression and UpdateExpression.strip().upper().startswith("SET"):
            assignments = UpdateExpression.strip()[3:].split(",")
            for assignment in assignments:
                if "=" not in assignment:
                    continue
                name_expr, value_expr = [
                    part.strip() for part in assignment.split("=", 1)
                ]
                name = (ExpressionAttributeNames or {}).get(name_expr, name_expr)
                value = (ExpressionAttributeValues or {}).get(value_expr)
                updates[name] = value
        current.update(updates)
        current["updated_at"] = _utcnow()
        saved = self.service._upsert_sync(self.table_name, current)
        if ReturnValues:
            return {"Attributes": saved}
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}


class SupabaseDataService:
    """Data service backed by Supabase Postgres."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.supabase = get_supabase_service()
        self.users_table = SupabaseTableAdapter(self, "profiles")
        self.courses_table = SupabaseTableAdapter(self, "courses")
        self.lessons_table = SupabaseTableAdapter(self, "lessons")
        self.quizzes_table = SupabaseTableAdapter(self, "quizzes")
        self.question_bank_items_table = SupabaseTableAdapter(
            self, "question_bank_items"
        )
        self.enrollments_table = SupabaseTableAdapter(self, "enrollments")
        self.quiz_results_table = SupabaseTableAdapter(self, "quiz_results")
        self.files_table = SupabaseTableAdapter(self, "files")
        self.chat_table = SupabaseTableAdapter(self, "chat_messages")
        self._read_cache: Dict[str, Tuple[float, Any]] = {}

    @property
    def client(self):
        return self.supabase.client

    def _convert_floats_to_decimal(self, item: Any) -> Any:
        if isinstance(item, dict):
            return {k: self._convert_floats_to_decimal(v) for k, v in item.items()}
        if isinstance(item, list):
            return [self._convert_floats_to_decimal(v) for v in item]
        if isinstance(item, float):
            return Decimal(str(item))
        return item

    def _convert_decimals_to_float(self, item: Any) -> Any:
        if isinstance(item, dict):
            return {k: self._convert_decimals_to_float(v) for k, v in item.items()}
        if isinstance(item, list):
            return [self._convert_decimals_to_float(v) for v in item]
        if isinstance(item, Decimal):
            return float(item)
        return item

    def _sanitize_for_postgres_json(self, item: Any) -> Any:
        """Remove NUL characters that PostgreSQL cannot store in jsonb text."""
        if isinstance(item, dict):
            return {
                self._sanitize_for_postgres_json(k): self._sanitize_for_postgres_json(v)
                for k, v in item.items()
            }
        if isinstance(item, list):
            return [self._sanitize_for_postgres_json(v) for v in item]
        if isinstance(item, tuple):
            return [self._sanitize_for_postgres_json(v) for v in item]
        if isinstance(item, str):
            return item.replace("\x00", "")
        return item

    def _cache_get(self, key: str) -> Optional[Any]:
        cache = getattr(self, "_read_cache", None)
        if not isinstance(cache, dict):
            return None
        entry = cache.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            cache.pop(key, None)
            return None
        return value

    def _cache_set(
        self, key: str, value: Any, ttl: float = READ_CACHE_TTL_SECONDS
    ) -> Any:
        cache = getattr(self, "_read_cache", None)
        if not isinstance(cache, dict):
            return value
        cache[key] = (time.monotonic() + ttl, value)
        return value

    def _cache_clear(self, *prefixes: str) -> None:
        cache = getattr(self, "_read_cache", None)
        if not isinstance(cache, dict):
            return
        if not prefixes:
            cache.clear()
            return
        for key in list(cache):
            if any(key.startswith(prefix) for prefix in prefixes):
                cache.pop(key, None)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _normalize_identity(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _normalize_email(value: Any) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _is_uuid_like(value: Any) -> bool:
        try:
            uuid.UUID(str(value or "").strip())
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _append_unique_identity(candidates: List[str], value: Any) -> None:
        normalized = str(value or "").strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    @staticmethod
    def _as_google_legacy_identity(value: Any) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            return ""
        if normalized.startswith("google_"):
            return normalized if len(normalized) > len("google_") else ""
        if normalized.isdigit():
            return f"google_{normalized}"
        return ""

    @staticmethod
    def _is_missing_column_error(error: Exception, *, table: str, column: str) -> bool:
        code = str(getattr(error, "code", "") or "").strip()
        message = str(getattr(error, "message", "") or str(error) or "").lower()
        table_column = f"{table}.{column}".lower()
        if "does not exist" not in message:
            return False
        if table_column in message:
            return True
        return code == "42703" and column.lower() in message

    async def _get_enrollment_user_ids_from_profile_email(
        self, normalized_email: str, limit: int = 50
    ) -> List[str]:
        profile_rows = await self._filter(
            "profiles",
            ("email", normalized_email),
            include_deleted=False,
            limit=limit,
        )
        candidate_user_ids: List[str] = []
        for profile in profile_rows:
            for value in (
                profile.get("user_id"),
                profile.get("username"),
                profile.get("student_id"),
            ):
                self._append_unique_identity(candidate_user_ids, value)
                self._append_unique_identity(
                    candidate_user_ids, self._as_google_legacy_identity(value)
                )

        matched_user_ids: List[str] = []
        max_candidates = max(1, min(len(candidate_user_ids), limit * 3))
        enrollments_by_user = await self._get_user_enrollments_by_user_ids(
            candidate_user_ids[:max_candidates],
            limit_per_user=limit,
        )
        for candidate_user_id, rows in enrollments_by_user.items():
            if not rows:
                continue
            user_id = self._normalize_identity(candidate_user_id)
            if not user_id or user_id in matched_user_ids:
                continue
            matched_user_ids.append(user_id)
            if len(matched_user_ids) >= limit:
                return matched_user_ids

        return matched_user_ids

    def _pack(self, table: str, item: Dict[str, Any]) -> Dict[str, Any]:
        item = self._sanitize_for_postgres_json(
            self._convert_decimals_to_float(dict(item))
        )
        columns = TABLE_COLUMNS[table]
        pk = TABLE_PK[table]
        if pk not in item or item.get(pk) in (None, ""):
            item[pk] = str(uuid.uuid4())
        row = {k: item.get(k) for k in columns if k != "data" and k in item}
        row["data"] = item
        return row

    def _unpack(self, row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not row:
            return None
        data = row.get("data") if isinstance(row.get("data"), dict) else {}
        merged = dict(data)
        for key, value in row.items():
            if key != "data" and value is not None:
                merged[key] = value
        return merged

    def _select_sync(self, table: str, limit: int = 10000) -> List[Dict[str, Any]]:
        result = self.client.table(table).select("*").limit(limit).execute()
        return [self._unpack(row) for row in (getattr(result, "data", None) or [])]

    def _query_sync(
        self,
        table: str,
        *,
        select: str = "*",
        eq: Optional[Dict[str, Any]] = None,
        neq: Optional[Dict[str, Any]] = None,
        in_filters: Optional[Dict[str, List[Any]]] = None,
        order_by: Optional[str] = None,
        desc: bool = False,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        count: bool = False,
        or_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        if count:
            query = self.client.table(table).select(select, count="exact")
        else:
            query = self.client.table(table).select(select)
        for column, value in (eq or {}).items():
            query = query.eq(column, value)
        for column, value in (neq or {}).items():
            query = query.neq(column, value)
        for column, values in (in_filters or {}).items():
            cleaned_values = [
                value
                for value in list(dict.fromkeys(values or []))
                if value not in (None, "")
            ]
            if not cleaned_values:
                return {"rows": [], "count": 0}
            query = query.in_(column, cleaned_values)
        if or_filter:
            query = query.or_(or_filter)
        if order_by:
            query = query.order(order_by, desc=desc)
        if limit is not None:
            safe_limit = max(1, int(limit))
            safe_offset = max(0, int(offset or 0))
            query = query.range(safe_offset, safe_offset + safe_limit - 1)
        result = query.execute()
        rows = [self._unpack(row) for row in (getattr(result, "data", None) or [])]
        total = getattr(result, "count", None)
        return {"rows": rows, "count": int(total if total is not None else len(rows))}

    async def _query(
        self,
        table: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await self.supabase.run(self._query_sync, table, **kwargs)

    def _get_sync(self, table: str, key_value: Any) -> Optional[Dict[str, Any]]:
        if key_value is None:
            return None
        pk = TABLE_PK[table]
        result = (
            self.client.table(table).select("*").eq(pk, key_value).limit(1).execute()
        )
        rows = getattr(result, "data", None) or []
        return self._unpack(rows[0]) if rows else None

    def _upsert_sync(self, table: str, item: Dict[str, Any]) -> Dict[str, Any]:
        pk = TABLE_PK[table]
        row = self._pack(table, item)
        result = self.client.table(table).upsert(row, on_conflict=pk).execute()
        rows = getattr(result, "data", None) or []
        self._cache_clear()
        return self._unpack(rows[0]) if rows else self._unpack(row)

    def _delete_sync(self, table: str, key_value: Any) -> None:
        if key_value is None:
            return
        self.client.table(table).delete().eq(TABLE_PK[table], key_value).execute()
        self._cache_clear()

    async def _select(self, table: str, limit: int = 10000) -> List[Dict[str, Any]]:
        return await self.supabase.run(self._select_sync, table, limit)

    async def _get(self, table: str, key_value: Any) -> Optional[Dict[str, Any]]:
        return await self.supabase.run(self._get_sync, table, key_value)

    async def _upsert(self, table: str, item: Dict[str, Any]) -> Dict[str, Any]:
        return await self.supabase.run(self._upsert_sync, table, item)

    async def _delete(self, table: str, key_value: Any) -> None:
        await self.supabase.run(self._delete_sync, table, key_value)

    async def _filter(
        self,
        table: str,
        *conditions: Any,
        limit: int = 10000,
        include_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        def run_query() -> List[Dict[str, Any]]:
            query = self.client.table(table).select("*").limit(limit)
            for column, value in conditions:
                query = query.eq(column, value)
            if not include_deleted and "status" in TABLE_COLUMNS.get(table, set()):
                query = query.neq("status", "deleted")
            result = query.execute()
            rows = [self._unpack(row) for row in (getattr(result, "data", None) or [])]
            if include_deleted:
                return rows
            return [
                row for row in rows if str(row.get("status", "")).lower() != "deleted"
            ]

        return await self.supabase.run(run_query)

    async def _filter_in(
        self,
        table: str,
        column: str,
        values: List[Any],
        *,
        limit: int = 10000,
        include_deleted: bool = False,
        select: str = "*",
    ) -> List[Dict[str, Any]]:
        normalized_values = [
            self._normalize_identity(value)
            for value in values
            if self._normalize_identity(value)
        ]
        if not normalized_values:
            return []

        def run_query() -> List[Dict[str, Any]]:
            query = (
                self.client.table(table)
                .select(select)
                .in_(column, list(dict.fromkeys(normalized_values)))
                .limit(limit)
            )
            if not include_deleted and "status" in TABLE_COLUMNS.get(table, set()):
                query = query.neq("status", "deleted")
            result = query.execute()
            rows = [self._unpack(row) for row in (getattr(result, "data", None) or [])]
            if include_deleted:
                return rows
            return [
                row for row in rows if str(row.get("status", "")).lower() != "deleted"
            ]

        return await self.supabase.run(run_query)

    async def create_user(self, user_data: Dict[str, Any]) -> str:
        user_id = str(user_data.get("user_id") or uuid.uuid4())
        now = _utcnow()
        item = {
            "user_id": user_id,
            "email": user_data.get("email") or f"{user_id}@example.com",
            "username": user_data.get("username") or user_data.get("email") or user_id,
            "name": user_data.get("name") or f"User {user_id}",
            "role": user_data.get("role") or "student",
            "status": user_data.get("status") or "active",
            "created_at": user_data.get("created_at") or now,
            "updated_at": now,
            **user_data,
        }
        await self._upsert("profiles", item)
        return user_id

    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        return await self._get("profiles", user_id)

    async def get_users_by_ids(
        self, user_ids: List[str], *, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        return (
            await self._query(
                "profiles",
                in_filters={"user_id": user_ids},
                neq={"status": "deleted"},
                limit=limit,
            )
        )["rows"]

    async def find_user_by_identity(
        self,
        *,
        user_id: Optional[str] = None,
        email: Optional[str] = None,
        username: Optional[str] = None,
        student_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        candidate_user_id = self._normalize_identity(user_id)
        if candidate_user_id:
            user = await self.get_user(candidate_user_id)
            if user:
                return user

        for column, value in (
            ("email", email),
            ("username", username),
            ("student_id", student_id),
        ):
            normalized_value = self._normalize_identity(value)
            if not normalized_value:
                continue
            rows = await self._filter(
                "profiles", (column, normalized_value), include_deleted=False, limit=1
            )
            if rows:
                return rows[0]
        return None

    async def get_enrollment_user_ids_by_billing_email(
        self, email: Optional[str], limit: int = 50
    ) -> List[str]:
        normalized_email = self._normalize_email(email)
        if not normalized_email:
            return []

        try:
            rows = await self._filter(
                "enrollments",
                ("billing_email", normalized_email),
                include_deleted=False,
                limit=limit,
            )
        except APIError as error:
            if not self._is_missing_column_error(
                error, table="enrollments", column="billing_email"
            ):
                raise
            app_logger.warning(
                "Enrollment lookup fallback: missing enrollments.billing_email column."
            )
            return await self._get_enrollment_user_ids_from_profile_email(
                normalized_email, limit=limit
            )
        except Exception as error:
            if not self._is_missing_column_error(
                error, table="enrollments", column="billing_email"
            ):
                raise
            app_logger.warning(
                "Enrollment lookup fallback: missing enrollments.billing_email column."
            )
            return await self._get_enrollment_user_ids_from_profile_email(
                normalized_email, limit=limit
            )
        user_ids: List[str] = []
        for row in rows:
            user_id = self._normalize_identity(row.get("user_id"))
            if user_id and user_id not in user_ids:
                user_ids.append(user_id)
        return user_ids

    async def has_existing_student_activity_by_email(
        self, email: Optional[str]
    ) -> bool:
        return bool(await self.get_enrollment_user_ids_by_billing_email(email, limit=1))

    async def has_existing_student_activity(
        self,
        user_id: str,
        email: Optional[str] = None,
        username: Optional[str] = None,
        student_id: Optional[str] = None,
        user: Optional[Dict[str, Any]] = None,
    ) -> bool:
        candidate_ids = await self._collect_student_onboarding_identity_candidates(
            user_id=user_id,
            email=email,
            username=username,
            student_id=student_id,
            user=user,
        )

        for candidate_id in candidate_ids:
            enrollments = await self.get_user_enrollments(candidate_id, limit=50)
            if any(
                str(enrollment.get("status") or "active").lower()
                not in {"cancelled", "canceled", "deleted", "expired"}
                for enrollment in enrollments
            ):
                return True

        return False

    async def _collect_student_onboarding_identity_candidates(
        self,
        user_id: str,
        email: Optional[str] = None,
        username: Optional[str] = None,
        student_id: Optional[str] = None,
        user: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        user = user or {}
        candidates: List[str] = []

        for value in (
            user_id,
            student_id,
            user.get("student_id"),
            username,
            user.get("username"),
            email,
            user.get("email"),
        ):
            self._append_unique_identity(candidates, value)
            self._append_unique_identity(
                candidates, self._as_google_legacy_identity(value)
            )

        lookup_email = email or user.get("email")
        for alias in await self.get_enrollment_user_ids_by_billing_email(
            lookup_email, limit=50
        ):
            self._append_unique_identity(candidates, alias)
            self._append_unique_identity(
                candidates, self._as_google_legacy_identity(alias)
            )

        return candidates

    async def _collect_student_onboarding_candidate_profiles(
        self,
        candidate_ids: List[str],
        email: Optional[str] = None,
        username: Optional[str] = None,
        student_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        candidates_by_user_id: Dict[str, Dict[str, Any]] = {}

        def register_profile(profile: Optional[Dict[str, Any]]) -> None:
            if not profile:
                return
            profile_user_id = self._normalize_identity(profile.get("user_id"))
            if profile_user_id and profile_user_id not in candidates_by_user_id:
                candidates_by_user_id[profile_user_id] = profile

        for candidate_id in candidate_ids:
            profile = await self.get_user(candidate_id)
            register_profile(profile)

        email_candidates = [
            self._normalize_identity(email),
            self._normalize_email(email),
        ]
        for email_candidate in email_candidates:
            if not email_candidate:
                continue
            rows = await self._filter(
                "profiles",
                ("email", email_candidate),
                include_deleted=False,
                limit=10,
            )
            for row in rows:
                register_profile(row)

        for field_name, value in (("username", username), ("student_id", student_id)):
            normalized = self._normalize_identity(value)
            if not normalized:
                continue
            rows = await self._filter(
                "profiles",
                (field_name, normalized),
                include_deleted=False,
                limit=10,
            )
            for row in rows:
                register_profile(row)

        return candidates_by_user_id

    async def save_student_onboarding(
        self,
        user_id: str,
        onboarding_profile: Dict[str, Any],
        base_user: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = _utcnow()
        existing = await self.get_user(user_id)
        base_user = base_user or {}
        item = dict(existing or {})
        item.update(
            {
                "user_id": user_id,
                "email": item.get("email")
                or base_user.get("email")
                or f"{user_id}@example.com",
                "username": item.get("username")
                or base_user.get("username")
                or base_user.get("email")
                or user_id,
                "name": item.get("name")
                or base_user.get("name")
                or base_user.get("username")
                or f"User {user_id}",
                "role": item.get("role") or "student",
                "status": item.get("status") or "active",
                "updated_at": now,
                "onboarding_profile": onboarding_profile,
                "onboarding_completed": True,
                "onboarding_completed_at": item.get("onboarding_completed_at") or now,
            }
        )
        item.setdefault("created_at", now)
        return await self._upsert("profiles", item)

    async def get_student_onboarding(
        self,
        user_id: str,
        email: Optional[str] = None,
        username: Optional[str] = None,
        student_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        canonical_user_id = self._normalize_identity(user_id)
        user = await self.find_user_by_identity(
            user_id=user_id,
            email=email,
            username=username,
            student_id=student_id,
        )
        if (
            user
            and self._normalize_identity(user.get("user_id")) == canonical_user_id
            and bool(user.get("onboarding_completed"))
        ):
            return {
                "onboarding_completed": True,
                "onboarding_profile": user.get("onboarding_profile"),
            }

        candidate_ids = await self._collect_student_onboarding_identity_candidates(
            user_id=user_id,
            email=email,
            username=username,
            student_id=student_id,
            user=user,
        )
        candidate_profiles = await self._collect_student_onboarding_candidate_profiles(
            candidate_ids,
            email=email or (user or {}).get("email"),
            username=username or (user or {}).get("username"),
            student_id=student_id or (user or {}).get("student_id"),
        )
        if user:
            candidate_profiles.setdefault(
                self._normalize_identity(user.get("user_id")), user
            )

        completed_profile = next(
            (
                profile
                for profile in candidate_profiles.values()
                if bool(profile.get("onboarding_completed"))
            ),
            None,
        )

        if completed_profile:
            completed_profile_id = self._normalize_identity(
                completed_profile.get("user_id")
            )
            if canonical_user_id and completed_profile_id != canonical_user_id:
                await self.save_student_onboarding(
                    user_id=canonical_user_id,
                    onboarding_profile=completed_profile.get("onboarding_profile")
                    or {},
                    base_user={
                        "email": email
                        or (user or {}).get("email")
                        or completed_profile.get("email"),
                        "username": username
                        or (user or {}).get("username")
                        or completed_profile.get("username"),
                        "name": (user or {}).get("name")
                        or completed_profile.get("name"),
                    },
                )
            return {
                "onboarding_completed": True,
                "onboarding_profile": completed_profile.get("onboarding_profile"),
            }

        inferred_existing_student = await self.has_existing_student_activity(
            canonical_user_id or user_id,
            email=email or (user or {}).get("email"),
            username=username or (user or {}).get("username"),
            student_id=student_id or (user or {}).get("student_id"),
            user=user,
        )
        return {
            "onboarding_completed": inferred_existing_student,
            "onboarding_profile": (user or {}).get("onboarding_profile"),
        }

    async def create_course(self, user_id: str, course_data: Dict[str, Any]) -> str:
        course_id = str(course_data.get("course_id") or uuid.uuid4())
        now = _utcnow()
        item = {
            "course_id": course_id,
            "user_id": user_id,
            "instructor_id": course_data.get("instructor_id") or user_id,
            "name": course_data.get("name")
            or course_data.get("title")
            or "Untitled Course",
            "title": course_data.get("title")
            or course_data.get("name")
            or "Untitled Course",
            "status": course_data.get("status") or "active",
            "created_at": course_data.get("created_at") or now,
            "updated_at": now,
            **course_data,
        }
        await self._upsert("courses", item)
        return course_id

    async def get_course(self, course_id: str) -> Optional[Dict[str, Any]]:
        return await self._get("courses", course_id)

    async def get_course_by_id(self, course_id: str) -> Optional[Dict[str, Any]]:
        return await self.get_course(course_id)

    async def get_user_courses(
        self,
        user_id: str,
        *,
        aliases: Optional[List[str]] = None,
        summary: bool = False,
    ) -> List[Dict[str, Any]]:
        alias_key = ",".join(
            sorted(
                {
                    self._normalize_identity(alias)
                    for alias in (aliases or [])
                    if self._normalize_identity(alias)
                }
            )
        )
        cache_key = f"user_courses:{user_id}:{alias_key}:{'summary' if summary else 'full'}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        requested_identity = self._normalize_identity(user_id)
        lookup_ids: List[str] = []
        self._append_unique_identity(lookup_ids, requested_identity)
        for alias in aliases or []:
            self._append_unique_identity(lookup_ids, alias)

        if requested_identity and not self._is_uuid_like(requested_identity):
            profile = await self.find_user_by_identity(
                email=requested_identity if "@" in requested_identity else None,
                username=requested_identity,
                student_id=requested_identity,
            )
            if profile:
                self._append_unique_identity(lookup_ids, profile.get("user_id"))

        rows: List[Dict[str, Any]] = []
        seen_course_ids = set()
        select_expr = COURSE_SUMMARY_SELECT if summary else "*"
        owner_rows, instructor_rows = await asyncio.gather(
            self._filter_in(
                "courses",
                "user_id",
                lookup_ids,
                limit=1000,
                include_deleted=False,
                select=select_expr,
            ),
            self._filter_in(
                "courses",
                "instructor_id",
                lookup_ids,
                limit=1000,
                include_deleted=False,
                select=select_expr,
            ),
        )
        for row in owner_rows + instructor_rows:
            course_id = row.get("course_id")
            if course_id and course_id in seen_course_ids:
                continue
            if course_id:
                seen_course_ids.add(course_id)
            rows.append(row)

        return self._cache_set(
            cache_key,
            sorted(
                rows, key=lambda row: str(row.get("created_at") or ""), reverse=True
            ),
        )

    async def get_courses_by_user(
        self, user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        return (await self.get_user_courses(user_id))[:limit]

    async def get_all_courses(self) -> List[Dict[str, Any]]:
        result = await self._query(
            "courses",
            neq={"status": "deleted"},
            order_by="created_at",
            desc=True,
            limit=10000,
        )
        return [
            row
            for row in result["rows"]
            if str(row.get("status", "active")).lower() == "active"
        ]

    async def get_courses_by_ids(
        self, course_ids: List[str], *, limit: int = 1000, summary: bool = False
    ) -> List[Dict[str, Any]]:
        return (
            await self._query(
                "courses",
                select=COURSE_SUMMARY_SELECT if summary else "*",
                in_filters={"course_id": course_ids},
                neq={"status": "deleted"},
                limit=limit,
            )
        )["rows"]

    async def update_course(self, course_id: str, updates: Dict[str, Any]) -> bool:
        item = await self.get_course(course_id) or {"course_id": course_id}
        item.update(updates)
        item["updated_at"] = _utcnow()
        await self._upsert("courses", item)
        return True

    async def delete_course(self, course_id: str) -> bool:
        return await self.update_course(
            course_id, {"status": "deleted", "deleted_at": _utcnow()}
        )

    async def create_lesson(
        self, user_id: str, course_id: str, lesson_data: Dict[str, Any]
    ) -> str:
        lesson_id = str(lesson_data.get("lesson_id") or uuid.uuid4())
        now = _utcnow()
        item = {
            "lesson_id": lesson_id,
            "course_id": course_id,
            "user_id": user_id,
            "status": lesson_data.get("status") or "active",
            "created_at": lesson_data.get("created_at") or now,
            "updated_at": now,
            **lesson_data,
        }
        await self._upsert("lessons", item)
        return lesson_id

    async def store_lesson(
        self, user_id: str, course_id: str, lesson_data: Dict[str, Any]
    ) -> str:
        return await self.create_lesson(user_id, course_id, lesson_data)

    async def get_lesson(self, lesson_id: str) -> Optional[Dict[str, Any]]:
        return await self._get("lessons", lesson_id)

    async def get_course_lessons(
        self, course_id: str, user_id: Optional[str] = None, summary: bool = False
    ) -> List[Dict[str, Any]]:
        cache_key = f"course_lessons:{course_id}:{user_id or ''}:{'summary' if summary else 'full'}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        result = await self._query(
            "lessons",
            select=LESSON_SUMMARY_SELECT if summary else "*",
            eq={"course_id": course_id},
            neq={"status": "deleted"},
            order_by="created_at",
            limit=5000,
        )
        rows = result["rows"]
        if user_id:
            rows = [
                row
                for row in rows
                if not row.get("user_id") or row.get("user_id") == user_id
            ]
        return self._cache_set(
            cache_key,
            sorted(
                rows,
                key=lambda row: str(row.get("created_at") or row.get("order") or ""),
            ),
        )

    async def get_lessons_by_course(
        self, course_id: str, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return await self.get_course_lessons(course_id, user_id)

    async def update_lesson(
        self, lesson_id: str, lesson_data: Dict[str, Any], user_id: Optional[str] = None
    ) -> bool:
        item = await self.get_lesson(lesson_id) or {"lesson_id": lesson_id}
        item.update(lesson_data)
        item["updated_at"] = _utcnow()
        await self._upsert("lessons", item)
        return True

    async def delete_lesson(
        self, lesson_id: str, user_id: Optional[str] = None
    ) -> bool:
        return await self.update_lesson(
            lesson_id, {"status": "deleted", "deleted_at": _utcnow()}, user_id
        )

    async def create_quiz(
        self, user_id: str, course_id: str, quiz_data: Dict[str, Any]
    ) -> str:
        quiz_id = str(quiz_data.get("quiz_id") or uuid.uuid4())
        now = _utcnow()
        item = {
            "quiz_id": quiz_id,
            "course_id": course_id,
            "user_id": user_id,
            "lesson_id": quiz_data.get("lesson_id"),
            "document_id": quiz_data.get("document_id"),
            "status": quiz_data.get("status") or "active",
            "created_at": quiz_data.get("created_at") or now,
            "updated_at": now,
            **quiz_data,
        }
        await self._upsert("quizzes", item)
        return quiz_id

    async def get_quiz(self, quiz_id: str) -> Optional[Dict[str, Any]]:
        return await self._get("quizzes", quiz_id)

    async def get_quiz_by_id(self, quiz_id: str) -> Optional[Dict[str, Any]]:
        return await self.get_quiz(quiz_id)

    async def get_user_quizzes(
        self, user_id: str, course_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        eq = {"user_id": user_id}
        if course_id:
            eq["course_id"] = course_id
        result = await self._query(
            "quizzes",
            eq=eq,
            order_by="created_at",
            desc=True,
            limit=5000,
        )
        rows = [
            row
            for row in result["rows"]
            if str(row.get("status") or "").strip().lower() != "deleted"
        ]
        return sorted(
            rows, key=lambda row: str(row.get("created_at") or ""), reverse=True
        )

    async def get_quizzes_by_course(
        self, course_id: str, summary: bool = False
    ) -> List[Dict[str, Any]]:
        cache_key = f"course_quizzes:{course_id}:{'summary' if summary else 'full'}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        result = await self._query(
            "quizzes",
            select=QUIZ_SUMMARY_SELECT if summary else "*",
            eq={"course_id": course_id},
            order_by="created_at",
            desc=True,
            limit=5000,
        )
        rows = [
            row
            for row in result["rows"]
            if str(row.get("status") or "").strip().lower() != "deleted"
        ]
        return self._cache_set(
            cache_key,
            sorted(
                rows, key=lambda row: str(row.get("created_at") or ""), reverse=True
            ),
        )

    async def get_quizzes_for_courses(
        self, course_ids: List[str], *, summary: bool = True
    ) -> List[Dict[str, Any]]:
        unique_course_ids = list(
            dict.fromkeys(
                self._normalize_identity(course_id)
                for course_id in course_ids
                if self._normalize_identity(course_id)
            )
        )
        if not unique_course_ids:
            return []

        quiz_groups = await asyncio.gather(
            *[
                self.get_quizzes_by_course(course_id, summary=summary)
                for course_id in unique_course_ids
            ]
        )
        quizzes: List[Dict[str, Any]] = []
        for rows in quiz_groups:
            quizzes.extend(rows)
        return quizzes

    async def get_course_quizzes_page(
        self,
        course_id: str,
        *,
        page: int = 1,
        page_size: int = 20,
        q: Optional[str] = None,
        sort: str = "latest",
        quiz_ids: Optional[List[str]] = None,
        summary: bool = False,
    ) -> Dict[str, Any]:
        page = max(1, int(page or 1))
        page_size = max(1, min(100, int(page_size or 20)))
        sort_key = str(sort or "latest").strip().lower()
        order_by = "created_at"
        desc = sort_key not in {"oldest"}
        offset = (page - 1) * page_size
        eq = {"course_id": course_id}
        in_filters = {"quiz_id": quiz_ids or []} if quiz_ids else None
        or_filter = None
        q_text = str(q or "").strip()
        if q_text:
            escaped = q_text.replace("%", "\\%").replace(",", "\\,")
            or_filter = (
                f"data->>title.ilike.%{escaped}%,"
                f"data->>name.ilike.%{escaped}%,"
                f"data->>description.ilike.%{escaped}%"
            )
        result = await self._query(
            "quizzes",
            select=QUIZ_SUMMARY_SELECT if summary else "*",
            eq=eq,
            neq={"status": "deleted"},
            in_filters=in_filters,
            or_filter=or_filter,
            order_by=order_by,
            desc=desc,
            limit=page_size,
            offset=offset,
            count=True,
        )
        total = int(result["count"])
        total_pages = max(1, math.ceil(total / page_size)) if total else 1
        return {
            "rows": result["rows"],
            "total": total,
            "page": min(page, total_pages),
            "page_size": page_size,
            "total_pages": total_pages,
        }

    async def get_quizzes_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        return await self._filter(
            "quizzes", ("document_id", document_id), include_deleted=False
        )

    async def update_quiz(self, quiz_id: str, updates: Dict[str, Any]) -> bool:
        item = await self.get_quiz(quiz_id) or {"quiz_id": quiz_id}
        item.update(updates)
        item["updated_at"] = _utcnow()
        await self._upsert("quizzes", item)
        return True

    async def update_quiz_status(self, quiz_id: str, status: str) -> bool:
        return await self.update_quiz(quiz_id, {"status": status})

    async def delete_quiz(self, quiz_id: str) -> bool:
        return await self.update_quiz(
            quiz_id, {"status": "deleted", "deleted_at": _utcnow()}
        )

    async def get_question_bank_items(
        self,
        course_id: str,
        *,
        limit: int = 10000,
        verified_only: bool = False,
        compact: bool = False,
        page: int = 1,
        page_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        normalized_course_id = self._normalize_identity(course_id)
        safe_limit = max(1, min(int(limit), 10000))
        page = max(1, int(page or 1))
        safe_page_size = (
            max(1, min(1000, int(page_size))) if page_size is not None else None
        )
        cache_key = (
            f"question_bank:{normalized_course_id}:{safe_limit}:"
            f"{verified_only}:{compact}:{page}:{safe_page_size or 0}"
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        compact_select = ",".join(
            (
                "item_id",
                "source",
                "status",
                "id:data->>id",
                "question:data->>question",
                "context:data->>context",
                "choices:data->choices",
                "correct_answer:data->correct_answer",
                "correctAnswer:data->correctAnswer",
                "explanation:data->>explanation",
                "difficulty:data->difficulty",
                "topic_tag:data->>topic_tag",
                "topicTag:data->>topicTag",
                "subject_tag:data->>subject_tag",
                "subjectTag:data->>subjectTag",
                "image_url:data->>image_url",
                "imageUrl:data->>imageUrl",
                "verification_status:data->>verification_status",
                "verificationStatus:data->>verificationStatus",
            )
        )

        def get_all_sync() -> List[Dict[str, Any]]:
            items: List[Dict[str, Any]] = []
            page_size = 1000
            if safe_page_size is not None:
                ranges = [
                    ((page - 1) * safe_page_size, (page - 1) * safe_page_size + safe_page_size)
                ]
            else:
                ranges = [(start, min(start + page_size, safe_limit)) for start in range(0, safe_limit, page_size)]
            for start, stop in ranges:
                if start >= safe_limit:
                    break
                end = min(start + page_size, safe_limit) - 1
                if safe_page_size is not None:
                    end = min(stop, safe_limit) - 1
                query = (
                    self.client.table("question_bank_items")
                    .select(compact_select if compact else "*")
                    .eq("course_id", normalized_course_id)
                    .neq("status", "deleted")
                )
                if verified_only:
                    query = query.or_(
                        "data->>verification_status.eq.verified,"
                        "data->>verificationStatus.eq.verified"
                    )
                result = query.order("updated_at", desc=True).range(start, end).execute()
                rows = getattr(result, "data", None) or []
                items.extend(rows if compact else (self._unpack(row) for row in rows))
                if len(rows) < end - start + 1:
                    break
            return items

        return self._cache_set(cache_key, await self.supabase.run(get_all_sync))

    async def replace_question_bank_items(
        self,
        course_id: str,
        user_id: str,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized_course_id = self._normalize_identity(course_id)
        normalized_user_id = self._normalize_identity(user_id)
        now = _utcnow()
        normalized_items: List[Dict[str, Any]] = []
        seen_ids = set()
        for raw_item in items:
            if not isinstance(raw_item, dict):
                continue
            question_id = self._normalize_identity(
                raw_item.get("id") or raw_item.get("item_id")
            )
            if not question_id:
                continue
            item_id = f"{normalized_course_id}:{question_id}"
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            normalized_items.append(
                {
                    **raw_item,
                    "id": question_id,
                    "item_id": item_id,
                    "course_id": normalized_course_id,
                    "user_id": normalized_user_id,
                    "source": self._normalize_identity(raw_item.get("source")) or "manual",
                    "status": "active",
                    "created_at": raw_item.get("created_at")
                    or raw_item.get("createdAt")
                    or now,
                    "updated_at": raw_item.get("updated_at")
                    or raw_item.get("updatedAt")
                    or now,
                }
            )

        def replace_sync() -> List[Dict[str, Any]]:
            for start in range(0, len(normalized_items), 200):
                rows = [
                    self._pack("question_bank_items", item)
                    for item in normalized_items[start : start + 200]
                ]
                (
                    self.client.table("question_bank_items")
                    .upsert(
                        rows,
                        on_conflict="item_id",
                        returning=ReturnMethod.minimal,
                    )
                    .execute()
                )

            stale_ids: List[str] = []
            page_size = 1000
            for start in range(0, 10000, page_size):
                existing_result = (
                    self.client.table("question_bank_items")
                    .select("item_id")
                    .eq("course_id", normalized_course_id)
                    .range(start, start + page_size - 1)
                    .execute()
                )
                rows = getattr(existing_result, "data", None) or []
                stale_ids.extend(
                    str(row.get("item_id"))
                    for row in rows
                    if str(row.get("item_id") or "") not in seen_ids
                )
                if len(rows) < page_size:
                    break

            for start in range(0, len(stale_ids), 200):
                (
                    self.client.table("question_bank_items")
                    .delete(returning=ReturnMethod.minimal)
                    .eq("course_id", normalized_course_id)
                    .in_("item_id", stale_ids[start : start + 200])
                    .execute()
                )
            return normalized_items

        return await self.supabase.run(replace_sync)

    async def store_quiz_questions(
        self,
        quiz_id: str,
        questions: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return await self.update_quiz(
            quiz_id, {"questions": questions, "metadata": metadata or {}}
        )

    async def create_enrollment(
        self,
        user_id: str,
        course_id: str,
        enrollment_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        enrollment_data = enrollment_data or {}
        enrollment_id = str(enrollment_data.get("enrollment_id") or uuid.uuid4())
        now = _utcnow()
        item = {
            "enrollment_id": enrollment_id,
            "user_id": user_id,
            "course_id": course_id,
            "status": enrollment_data.get("status") or "active",
            "enrolled_at": enrollment_data.get("enrolled_at") or now,
            "created_at": enrollment_data.get("created_at") or now,
            "updated_at": now,
            **enrollment_data,
        }
        await self._upsert("enrollments", item)
        return enrollment_id

    async def enroll_user_in_course(
        self,
        user_id: str,
        course_id: str,
        enrollment_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        return await self.create_enrollment(user_id, course_id, enrollment_data)

    async def get_user_enrollments(
        self, user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        rows = await self._filter(
            "enrollments", ("user_id", user_id), include_deleted=False, limit=limit
        )
        return sorted(
            rows,
            key=lambda row: str(row.get("enrolled_at") or row.get("created_at") or ""),
            reverse=True,
        )

    async def _get_user_enrollments_by_user_ids(
        self, user_ids: List[str], *, limit_per_user: int = 50
    ) -> Dict[str, List[Dict[str, Any]]]:
        normalized_user_ids: List[str] = []
        for user_id in user_ids:
            self._append_unique_identity(normalized_user_ids, user_id)
        if not normalized_user_ids:
            return {}

        if not hasattr(self, "supabase"):
            serial_rows = await asyncio.gather(
                *[
                    self.get_user_enrollments(user_id, limit=limit_per_user)
                    for user_id in normalized_user_ids
                ]
            )
            return {
                user_id: rows
                for user_id, rows in zip(normalized_user_ids, serial_rows)
            }

        max_rows = max(limit_per_user, len(normalized_user_ids) * limit_per_user)
        result = await self._query(
            "enrollments",
            select=ENROLLMENT_SUMMARY_SELECT,
            in_filters={"user_id": normalized_user_ids},
            neq={"status": "deleted"},
            order_by="enrolled_at",
            desc=True,
            limit=max_rows,
        )
        rows_by_user_id: Dict[str, List[Dict[str, Any]]] = {
            user_id: [] for user_id in normalized_user_ids
        }
        for row in result["rows"]:
            user_id = self._normalize_identity(row.get("user_id"))
            if user_id not in rows_by_user_id:
                continue
            if len(rows_by_user_id[user_id]) >= limit_per_user:
                continue
            rows_by_user_id[user_id].append(row)
        return rows_by_user_id

    async def _get_user_identity_candidates(
        self, user_id: str, *, include_email_alias: bool = True
    ) -> List[str]:
        canonical_user_id = self._normalize_identity(user_id)
        if not canonical_user_id:
            return []

        candidates: List[str] = [canonical_user_id]
        user = await self.get_user(canonical_user_id)
        if not user:
            return candidates

        for field_name in ("username", "student_id"):
            alias = self._normalize_identity(user.get(field_name))
            if alias and alias not in candidates:
                candidates.append(alias)
        if include_email_alias:
            for alias in await self.get_enrollment_user_ids_by_billing_email(
                user.get("email")
            ):
                if alias not in candidates:
                    candidates.append(alias)
        return candidates

    async def _collect_preferred_enrollments_by_course(
        self, candidate_ids: List[str], *, limit: int = 50
    ) -> Dict[str, Dict[str, Any]]:
        rank_by_user_id = {
            candidate_id: idx for idx, candidate_id in enumerate(candidate_ids)
        }
        preferred_enrollments: Dict[str, Dict[str, Any]] = {}
        enrollments_by_user_id = await self._get_user_enrollments_by_user_ids(
            candidate_ids, limit_per_user=limit
        )

        for candidate_user_id in candidate_ids:
            enrollments = enrollments_by_user_id.get(candidate_user_id, [])
            for enrollment in enrollments:
                status = str(enrollment.get("status", "")).lower()
                if status not in {"active", "trial", "paid"}:
                    continue

                course_id = self._normalize_identity(enrollment.get("course_id"))
                if not course_id:
                    continue

                existing = preferred_enrollments.get(course_id)
                if not existing:
                    preferred_enrollments[course_id] = enrollment
                    continue

                existing_user_id = self._normalize_identity(existing.get("user_id"))
                incoming_user_id = self._normalize_identity(enrollment.get("user_id"))
                existing_rank = rank_by_user_id.get(
                    existing_user_id, len(candidate_ids)
                )
                incoming_rank = rank_by_user_id.get(
                    incoming_user_id, len(candidate_ids)
                )
                if incoming_rank < existing_rank:
                    preferred_enrollments[course_id] = enrollment
        return preferred_enrollments

    async def _resolve_enrollment_identity_candidates(
        self, user_id: str, *, limit: int = 50
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        primary_candidate_ids = await self._get_user_identity_candidates(
            user_id, include_email_alias=False
        )
        if not primary_candidate_ids:
            return [], {}

        preferred_enrollments = await self._collect_preferred_enrollments_by_course(
            primary_candidate_ids, limit=limit
        )
        if preferred_enrollments:
            return primary_candidate_ids, preferred_enrollments

        all_candidate_ids = await self._get_user_identity_candidates(
            user_id, include_email_alias=True
        )
        fallback_candidate_ids = [
            candidate_id
            for candidate_id in all_candidate_ids
            if candidate_id not in primary_candidate_ids
        ]
        if not fallback_candidate_ids:
            return primary_candidate_ids, {}

        fallback_enrollments = await self._collect_preferred_enrollments_by_course(
            fallback_candidate_ids, limit=limit
        )
        if not fallback_enrollments:
            return primary_candidate_ids, {}

        return primary_candidate_ids + fallback_candidate_ids, fallback_enrollments

    async def get_user_enrollments_with_aliases(
        self, user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        candidate_ids = await self._get_user_identity_candidates(user_id)
        rows: List[Dict[str, Any]] = []
        seen_enrollment_ids = set()
        enrollments_by_user_id = await self._get_user_enrollments_by_user_ids(
            candidate_ids, limit_per_user=limit
        )

        for candidate_user_id in candidate_ids:
            enrollments = enrollments_by_user_id.get(candidate_user_id, [])
            for enrollment in enrollments:
                enrollment_id = self._normalize_identity(
                    enrollment.get("enrollment_id")
                )
                if enrollment_id and enrollment_id in seen_enrollment_ids:
                    continue
                if enrollment_id:
                    seen_enrollment_ids.add(enrollment_id)
                rows.append(enrollment)

        return sorted(
            rows,
            key=lambda row: str(row.get("enrolled_at") or row.get("created_at") or ""),
            reverse=True,
        )

    async def get_course_enrollments(self, course_id: str) -> List[Dict[str, Any]]:
        return (
            await self._query(
                "enrollments",
                eq={"course_id": course_id},
                neq={"status": "deleted"},
                order_by="enrolled_at",
                desc=True,
                limit=10000,
            )
        )["rows"]

    async def get_course_enrollments_page(
        self, course_id: str, *, page: int = 1, page_size: int = 50
    ) -> Dict[str, Any]:
        page = max(1, int(page or 1))
        page_size = max(1, min(200, int(page_size or 50)))
        result = await self._query(
            "enrollments",
            eq={"course_id": course_id},
            neq={"status": "deleted"},
            order_by="enrolled_at",
            desc=True,
            limit=page_size,
            offset=(page - 1) * page_size,
            count=True,
        )
        total = int(result["count"])
        total_pages = max(1, math.ceil(total / page_size)) if total else 1
        return {
            "rows": result["rows"],
            "total": total,
            "page": min(page, total_pages),
            "page_size": page_size,
            "total_pages": total_pages,
        }

    async def get_enrollment_by_id(
        self, enrollment_id: str
    ) -> Optional[Dict[str, Any]]:
        return await self._get("enrollments", enrollment_id)

    async def update_enrollment(
        self, enrollment_id: str, updates: Dict[str, Any]
    ) -> bool:
        item = await self.get_enrollment_by_id(enrollment_id) or {
            "enrollment_id": enrollment_id
        }
        item.update(updates)
        item["updated_at"] = _utcnow()
        await self._upsert("enrollments", item)
        return True

    async def record_learning_activity(
        self,
        user_id: str,
        course_id: str,
        lesson_id: Optional[str] = None,
        activity_day: Optional[str] = None,
        activity_days: Optional[List[str]] = None,
        enrollment: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        normalized_course_id = self._normalize_identity(course_id)
        if not normalized_course_id:
            return None

        if enrollment is None:
            _, preferred_enrollments = await self._resolve_enrollment_identity_candidates(
                user_id, limit=200
            )
            enrollment = preferred_enrollments.get(normalized_course_id)
        if not enrollment:
            return None

        day_key = str(activity_day or "").strip()
        if not LEARNING_ACTIVITY_DAY_RE.match(day_key):
            day_key = datetime.now(LEARNING_ACTIVITY_TIME_ZONE).strftime("%Y-%m-%d")

        requested_days = activity_days if isinstance(activity_days, list) else []
        previous_days = enrollment.get("learning_activity_days")
        if not isinstance(previous_days, list):
            previous_days = []

        activity_days = sorted(
            {
                str(day).strip()
                for day in [*previous_days, *requested_days, day_key]
                if LEARNING_ACTIVITY_DAY_RE.match(str(day).strip())
            }
        )[-MAX_LEARNING_ACTIVITY_DAYS:]

        updates = {
            "learning_activity_days": activity_days,
            "last_learning_activity_at": datetime.now(
                LEARNING_ACTIVITY_TIME_ZONE
            ).isoformat(),
        }
        if lesson_id:
            updates["last_lesson_id"] = str(lesson_id)

        enrollment_id = enrollment.get("enrollment_id")
        if not enrollment_id:
            return None
        await self.update_enrollment(str(enrollment_id), updates)
        return {
            "course_id": normalized_course_id,
            "lesson_id": str(lesson_id or ""),
            "activity_day": day_key,
            "activity_days": activity_days,
        }

    async def cancel_enrollment(self, enrollment_id: str) -> bool:
        return await self.update_enrollment(
            enrollment_id, {"status": "cancelled", "cancelled_at": _utcnow()}
        )

    async def get_enrolled_courses_for_user(
        self, user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        cache_key = f"enrolled_courses:{user_id}:{int(limit or 50)}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        _, preferred_enrollments = await self._resolve_enrollment_identity_candidates(
            user_id, limit=limit
        )
        if not preferred_enrollments:
            return []

        enrollments = list(preferred_enrollments.values())
        course_ids = [
            self._normalize_identity(enrollment.get("course_id"))
            for enrollment in enrollments
            if self._normalize_identity(enrollment.get("course_id"))
        ]
        try:
            fetched_courses = await self.get_courses_by_ids(
                course_ids, limit=max(limit, len(course_ids) or 1), summary=True
            )
            courses_by_id = {
                self._normalize_identity(course.get("course_id")): course
                for course in fetched_courses
            }
            course_rows = [
                courses_by_id.get(self._normalize_identity(enrollment.get("course_id")))
                for enrollment in enrollments
            ]
        except Exception:
            course_rows = await asyncio.gather(
                *[
                    self.get_course(self._normalize_identity(enrollment.get("course_id")))
                    for enrollment in enrollments
                ]
            )

        courses: List[Dict[str, Any]] = []
        for enrollment, course in zip(enrollments, course_rows):
            if course and str(course.get("status", "active")).lower() != "deleted":
                row = dict(course)
                row["enrollment"] = enrollment
                row["enrollment_id"] = enrollment.get("enrollment_id")
                row["enrollment_status"] = enrollment.get("status")
                row["enrolled_at"] = enrollment.get("enrolled_at")
                row["started_at"] = enrollment.get("started_at")
                row["expires_at"] = enrollment.get("expires_at")
                row["duration_months"] = enrollment.get("duration_months")
                row["enrollment_source"] = enrollment.get("enrollment_source")
                row["enrollment_type"] = enrollment.get("enrollment_type")
                row["payment_provider"] = enrollment.get("payment_provider")
                row["payment_type"] = enrollment.get("payment_type")
                row["payment_intent_id"] = enrollment.get("payment_intent_id")
                row["payment_status"] = enrollment.get("payment_status")
                row["paid_amount_thb"] = enrollment.get("paid_amount_thb")
                row["paid_currency"] = enrollment.get("paid_currency")
                row["billing_email"] = enrollment.get("billing_email")
                row["plan_label"] = enrollment.get("plan_label")
                row["paid_at"] = enrollment.get("paid_at")
                row["payment_history"] = enrollment.get("payment_history")
                row["trial_consumed_at"] = enrollment.get("trial_consumed_at")
                row["trial_expires_at"] = enrollment.get("trial_expires_at")
                row["progress"] = enrollment.get("progress", row.get("progress", 0))
                row["completed_quizzes"] = enrollment.get(
                    "completed_quizzes", row.get("completed_quizzes", 0)
                )
                row["total_quizzes"] = enrollment.get(
                    "total_quizzes", row.get("total_quizzes", 0)
                )
                row["completed_questions"] = enrollment.get(
                    "completed_questions", row.get("completed_questions", 0)
                )
                row["total_questions"] = enrollment.get(
                    "total_questions", row.get("total_questions", 0)
                )
                row["last_activity"] = enrollment.get(
                    "last_activity", row.get("last_activity")
                )
                courses.append(row)
        return self._cache_set(
            cache_key,
            sorted(
                courses,
                key=lambda row: str(
                    row.get("enrolled_at") or row.get("created_at") or ""
                ),
                reverse=True,
            ),
        )

    async def get_dashboard_learning_inputs(
        self, user_id: str, limit: int = 50
    ) -> Dict[str, Any]:
        cache_key = f"dashboard_inputs:{user_id}:{int(limit or 50)}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        (
            candidate_ids,
            preferred_enrollments,
        ) = await self._resolve_enrollment_identity_candidates(user_id, limit=limit)
        if not candidate_ids:
            return {
                "candidate_user_ids": [],
                "enrollments": [],
                "courses": [],
                "lessons": [],
                "quizzes": [],
                "quiz_results": [],
            }

        enrollments = sorted(
            preferred_enrollments.values(),
            key=lambda row: str(row.get("enrolled_at") or row.get("created_at") or ""),
            reverse=True,
        )
        course_ids = [
            self._normalize_identity(enrollment.get("course_id"))
            for enrollment in enrollments
            if self._normalize_identity(enrollment.get("course_id"))
        ]
        if not course_ids:
            return {
                "candidate_user_ids": candidate_ids,
                "enrollments": [],
                "courses": [],
                "lessons": [],
                "quizzes": [],
                "quiz_results": [],
            }

        courses_task = self._filter_in(
            "courses",
            "course_id",
            course_ids,
            limit=max(limit, len(course_ids) or 1),
            include_deleted=False,
            select=COURSE_SUMMARY_SELECT,
        )
        lessons_task = self._filter_in(
            "lessons",
            "course_id",
            course_ids,
            limit=max(1000, len(course_ids) * 100),
            select=LESSON_SUMMARY_SELECT,
        )
        quizzes_task = self.get_quizzes_for_courses(course_ids, summary=True)
        if hasattr(self, "supabase"):
            results_task = self._query(
                "quiz_results",
                select=QUIZ_RESULT_SUMMARY_SELECT,
                in_filters={"user_id": candidate_ids, "course_id": course_ids},
                order_by="submitted_at",
                desc=True,
                limit=10000,
            )
            courses, lessons, quizzes, quiz_results_payload = await asyncio.gather(
                courses_task, lessons_task, quizzes_task, results_task
            )
            quiz_results = quiz_results_payload["rows"]
        else:
            courses, lessons, quizzes, quiz_results = await asyncio.gather(
                courses_task,
                lessons_task,
                quizzes_task,
                self._filter_in(
                    "quiz_results",
                    "user_id",
                    candidate_ids,
                    limit=10000,
                    include_deleted=True,
                ),
            )

        return self._cache_set(
            cache_key,
            {
                "candidate_user_ids": candidate_ids,
                "enrollments": enrollments,
                "courses": courses,
                "lessons": lessons,
                "quizzes": quizzes,
                "quiz_results": quiz_results,
            },
        )

    async def get_user_enrollment_for_course(
        self, user_id: str, course_id: str
    ) -> Optional[Dict[str, Any]]:
        normalized_course_id = self._normalize_identity(course_id)
        if not normalized_course_id:
            return None

        candidate_ids, preferred_enrollments = (
            await self._resolve_enrollment_identity_candidates(user_id, limit=50)
        )
        if not candidate_ids:
            return None

        enrollment = preferred_enrollments.get(normalized_course_id)
        if enrollment:
            return enrollment

        rows = (
            await self._query(
                "enrollments",
                in_filters={"user_id": candidate_ids},
                eq={"course_id": normalized_course_id},
                neq={"status": "deleted"},
                limit=1,
            )
        )["rows"]
        return rows[0] if rows else None

    async def get_course_learning_overview(
        self, course_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        normalized_course_id = self._normalize_identity(course_id)
        normalized_user_id = self._normalize_identity(user_id)
        cache_key = f"learning_overview:{normalized_course_id}:{normalized_user_id}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        course_task = self._query(
            "courses",
            select=COURSE_SUMMARY_SELECT,
            eq={"course_id": normalized_course_id},
            neq={"status": "deleted"},
            limit=1,
        )
        lessons_task = self.get_course_lessons(normalized_course_id, summary=True)
        quizzes_task = self.get_quizzes_by_course(normalized_course_id, summary=True)

        if normalized_user_id:
            results_task = self.get_user_quiz_results(
                normalized_user_id,
                course_id=normalized_course_id,
                summary=True,
            )
            enrollment_task = self.get_user_enrollment_for_course(
                normalized_user_id, normalized_course_id
            )
        else:
            results_task = asyncio.sleep(0, result=[])
            enrollment_task = asyncio.sleep(0, result=None)

        course_payload, lessons, quizzes, quiz_results, enrollment_row = (
            await asyncio.gather(
                course_task, lessons_task, quizzes_task, results_task, enrollment_task
            )
        )
        course = (course_payload.get("rows") or [None])[0]
        enrollment = None
        if enrollment_row:
            enrollment = enrollment_row

        return self._cache_set(
            cache_key,
            {
                "course_id": normalized_course_id,
                "user_id": normalized_user_id or None,
                "course": course,
                "enrollment": enrollment,
                "lessons": lessons,
                "quizzes": quizzes,
                "quiz_results": quiz_results,
                "generated_at": _utcnow(),
            },
        )

    async def get_all_active_enrollments(
        self, limit: int = 5000
    ) -> List[Dict[str, Any]]:
        return (
            await self._query(
                "enrollments",
                eq={"status": "active"},
                order_by="enrolled_at",
                desc=True,
                limit=limit,
            )
        )["rows"]

    async def create_quiz_result(
        self, user_id: str, quiz_id: str, result: Dict[str, Any]
    ) -> str:
        result_id = str(result.get("result_id") or uuid.uuid4())
        quiz = await self.get_quiz(quiz_id)
        now = _utcnow()
        item = {
            "result_id": result_id,
            "user_id": user_id,
            "quiz_id": quiz_id,
            "course_id": result.get("course_id") or (quiz or {}).get("course_id"),
            "submitted_at": result.get("submitted_at") or now,
            "created_at": result.get("created_at") or now,
            "updated_at": now,
            **result,
        }
        await self._upsert("quiz_results", item)
        return result_id

    async def get_user_quiz_results(
        self,
        user_id: str,
        quiz_id: Optional[str] = None,
        course_id: Optional[str] = None,
        limit: int = 10000,
        summary: bool = False,
    ) -> List[Dict[str, Any]]:
        primary_candidate_ids = await self._get_user_identity_candidates(
            user_id, include_email_alias=False
        )
        if not primary_candidate_ids:
            return []

        rows: List[Dict[str, Any]] = []
        seen_result_ids = set()
        normalized_course_id = str(course_id or "").strip()
        course_quiz_ids = set()
        if normalized_course_id:
            course_quiz_ids = {
                str(
                    quiz.get("quiz_id")
                    or quiz.get("id")
                    or quiz.get("document_id")
                    or ""
                ).strip()
                for quiz in await self.get_quizzes_by_course(
                    normalized_course_id, summary=True
                )
            }
            course_quiz_ids.discard("")

        async def collect_rows(candidate_ids: List[str]) -> None:
            normalized_candidate_ids: List[str] = []
            for candidate_user_id in candidate_ids:
                self._append_unique_identity(
                    normalized_candidate_ids, candidate_user_id
                )
            if not normalized_candidate_ids:
                return

            if hasattr(self, "supabase"):
                eq = {}
                in_filters: Dict[str, List[Any]] = {
                    "user_id": normalized_candidate_ids
                }
                if quiz_id:
                    eq["quiz_id"] = quiz_id
                elif normalized_course_id and course_quiz_ids:
                    in_filters["quiz_id"] = list(course_quiz_ids)
                query_limit = max(limit, len(normalized_candidate_ids) * limit)
                candidate_rows = (
                    await self._query(
                        "quiz_results",
                        select=QUIZ_RESULT_SUMMARY_SELECT if summary else "*",
                        eq=eq,
                        in_filters=in_filters,
                        order_by="submitted_at",
                        desc=True,
                        limit=query_limit,
                    )
                )["rows"]
            else:
                serial_rows = []
                for candidate_user_id in normalized_candidate_ids:
                    candidate_rows = await self._filter(
                        "quiz_results",
                        ("user_id", candidate_user_id),
                        include_deleted=True,
                        limit=limit,
                    )
                    serial_rows.extend(candidate_rows)
                candidate_rows = serial_rows

            for row in candidate_rows:
                result_id = self._normalize_identity(row.get("result_id"))
                if result_id and result_id in seen_result_ids:
                    continue
                if result_id:
                    seen_result_ids.add(result_id)
                rows.append(row)

        await collect_rows(primary_candidate_ids)
        if not rows:
            all_candidate_ids = await self._get_user_identity_candidates(
                user_id, include_email_alias=True
            )
            fallback_candidate_ids = [
                candidate_id
                for candidate_id in all_candidate_ids
                if candidate_id not in primary_candidate_ids
            ]
            if fallback_candidate_ids:
                await collect_rows(fallback_candidate_ids)

        if quiz_id:
            rows = [row for row in rows if row.get("quiz_id") == quiz_id]
        if normalized_course_id:
            rows = [
                row
                for row in rows
                if str(row.get("course_id") or "").strip() == normalized_course_id
                or (
                    str(row.get("course_id") or "").strip() in {"", "default-course"}
                    and str(row.get("quiz_id") or "").strip() in course_quiz_ids
                )
            ]
        return sorted(
            rows,
            key=lambda row: str(row.get("submitted_at") or row.get("created_at") or ""),
            reverse=True,
        )

    async def get_quiz_results_for_course(
        self,
        course_id: str,
        *,
        user_ids: Optional[List[str]] = None,
        quiz_ids: Optional[List[str]] = None,
        limit: int = 10000,
        summary: bool = False,
    ) -> List[Dict[str, Any]]:
        in_filters: Dict[str, List[Any]] = {}
        if user_ids:
            in_filters["user_id"] = user_ids
        if quiz_ids:
            in_filters["quiz_id"] = quiz_ids
        result = await self._query(
            "quiz_results",
            select=QUIZ_RESULT_SUMMARY_SELECT if summary else "*",
            eq={"course_id": course_id},
            in_filters=in_filters or None,
            order_by="submitted_at",
            desc=True,
            limit=limit,
        )
        return result["rows"]

    async def get_student_course_score_version(
        self, user_id: str, course_id: str
    ) -> str:
        raw = f"{user_id}:{course_id}:{_utcnow()[:10]}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    async def record_student_token_usage(
        self,
        user_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        llm_cost_usd: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        usage_date = str(
            kwargs.get("usage_date") or datetime.utcnow().date().isoformat()
        )
        user = await self.get_user(user_id) or {
            "user_id": user_id,
            "email": f"{user_id}@example.com",
            "name": f"User {user_id}",
            "role": "student",
            "status": "active",
            "created_at": _utcnow(),
        }
        daily = (
            user.get("token_usage_daily")
            if isinstance(user.get("token_usage_daily"), dict)
            else {}
        )
        row = daily.get(usage_date) if isinstance(daily.get(usage_date), dict) else {}
        row["input_tokens"] = int(row.get("input_tokens") or 0) + int(input_tokens or 0)
        row["output_tokens"] = int(row.get("output_tokens") or 0) + int(
            output_tokens or 0
        )
        row["total_tokens"] = int(row.get("total_tokens") or 0) + int(
            total_tokens or input_tokens + output_tokens or 0
        )
        row["request_count"] = int(row.get("request_count") or 0) + 1
        row["llm_cost_usd"] = self._safe_float(
            row.get("llm_cost_usd"), 0.0
        ) + self._safe_float(llm_cost_usd, 0.0)
        daily[usage_date] = row
        user["token_usage_daily"] = daily
        user["updated_at"] = _utcnow()
        await self._upsert("profiles", user)
        return row

    async def get_chat_energy_platform_config(self) -> Dict[str, Any]:
        row = await self._get("platform_config", "chat_energy")
        data = dict((row or {}).get("data") or row or {})
        return {
            "default_daily_limit_thb": self._safe_float(
                data.get("default_daily_limit_thb"),
                DEFAULT_CHAT_ENERGY_DAILY_LIMIT_THB,
            ),
            "updated_at": data.get("updated_at"),
            "updated_by": data.get("updated_by"),
            "reason": data.get("reason"),
        }

    async def set_chat_energy_platform_config(
        self,
        default_daily_limit_thb: float,
        updated_by: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        item = {
            "config_key": "chat_energy",
            "default_daily_limit_thb": float(default_daily_limit_thb),
            "updated_at": _utcnow(),
            "updated_by": updated_by,
            "reason": reason,
        }
        await self._upsert("platform_config", item)
        return await self.get_chat_energy_platform_config()

    async def set_user_chat_energy_policy(
        self,
        user_id: str,
        daily_limit_override_thb: Optional[float] = None,
        daily_adjustment_thb: float = 0.0,
        updated_by: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        user = await self.get_user(user_id) or {
            "user_id": user_id,
            "email": f"{user_id}@example.com",
        }
        policy = {
            "daily_limit_override_thb": daily_limit_override_thb,
            "daily_adjustment_thb": float(daily_adjustment_thb or 0.0),
            "updated_at": _utcnow(),
            "updated_by": updated_by,
            "reason": reason,
        }
        user["chat_energy_policy"] = policy
        user["updated_at"] = _utcnow()
        await self._upsert("profiles", user)
        return policy

    async def get_student_chat_energy_status(self, user_id: str) -> Dict[str, Any]:
        config, user = await asyncio.gather(
            self.get_chat_energy_platform_config(),
            self.get_user(user_id),
        )
        user = user or {}
        policy = (
            user.get("chat_energy_policy")
            if isinstance(user.get("chat_energy_policy"), dict)
            else {}
        )
        usage_date = datetime.utcnow().date().isoformat()
        token_daily = (
            user.get("token_usage_daily")
            if isinstance(user.get("token_usage_daily"), dict)
            else {}
        )
        usage = (
            token_daily.get(usage_date)
            if isinstance(token_daily.get(usage_date), dict)
            else {}
        )
        used_thb = self._safe_float(usage.get("llm_cost_thb"), 0.0)
        if not used_thb:
            used_usd = self._safe_float(usage.get("llm_cost_usd"), 0.0)
            used_thb = used_usd * self._safe_float(
                self.settings.openrouter_cost_usd_to_thb,
                36.0,
            )
        default_limit = self._safe_float(
            config.get("default_daily_limit_thb"), DEFAULT_CHAT_ENERGY_DAILY_LIMIT_THB
        )
        override = policy.get("daily_limit_override_thb")
        adjustment = self._safe_float(policy.get("daily_adjustment_thb"), 0.0)
        daily_limit = (
            self._safe_float(override, default_limit) + adjustment
            if override is not None
            else default_limit + adjustment
        )
        daily_limit = max(0.0, daily_limit)
        remaining = max(0.0, daily_limit - used_thb)
        return {
            "daily_limit_thb": daily_limit,
            "used_thb": used_thb,
            "remaining_thb": remaining,
            "remaining_percent": (remaining / daily_limit * 100.0)
            if daily_limit
            else 0.0,
            "is_exhausted": remaining <= 0,
            "daily_limit_override_thb": override,
            "daily_adjustment_thb": adjustment,
            "limit_source": "user_override"
            if override is not None
            else "global_default",
            "usage_date": usage_date,
            "request_count": int(usage.get("request_count") or 0),
            "policy_updated_at": policy.get("updated_at"),
            "policy_updated_by": policy.get("updated_by"),
            "policy_reason": policy.get("reason"),
            "platform_updated_at": config.get("updated_at"),
            "platform_updated_by": config.get("updated_by"),
            "platform_reason": config.get("reason"),
            "default_daily_limit_thb": default_limit,
        }

    async def check_tables_health(self) -> Dict[str, Any]:
        table_status = {}
        healthy = True
        for table in TABLE_PK:
            try:
                await self._select(table, limit=1)
                table_status[table] = "healthy"
            except Exception as e:
                healthy = False
                table_status[table] = f"error: {e}"
        return {
            "service": "supabase",
            "status": "healthy" if healthy else "error",
            "healthy": healthy,
            "tables": table_status,
        }

    async def check_table_health(self) -> Dict[str, Any]:
        return await self.check_tables_health()


class DataService:
    """Facade for Supabase-backed student data access."""

    def __init__(self) -> None:
        self.service = SupabaseDataService()

    def __getattr__(self, name: str):
        return getattr(self.service, name)


_db_service: Optional[DataService] = None


def get_db_service() -> DataService:
    global _db_service
    if _db_service is None:
        _db_service = DataService()
    return _db_service
