"""API endpoints for OCR processing."""

import asyncio
import copy
import hashlib
import hmac
import json
import os
import random
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import fitz  # PyMuPDF
import httpx
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.exceptions import BaseAPIException, FileUploadError, OCRProcessingError
from app.core.logging import app_logger
from app.models.schemas import (
    CourseAIGenerateRequest,
    CourseAIGenerateResponse,
    CreateLessonRequest,
    DocumentTypeEnum,
    ErrorResponse,
    HealthCheckResponse,
    LessonListResponse,
    LessonResponse,
    OCRRequest,
    OCRResponse,
    ProcessingStatusEnum,
    QuizAugmentRequest,
    QuizAugmentResponse,
    UpdateLessonRequest,
    UploadResponse,
)
from app.services.chat_service import ChatService
from app.services.dynamodb_service import get_db_service
from app.services.file_service import FileService
from app.services.gemini_ocr_service import GeminiOCRService
from app.services.quiz_augment_service import QuizAugmentService
from app.services.student_auth_service import StudentAuthService
from app.utils.admin_auth import ADMIN_BEARER_OPTIONAL, validate_admin_actor

_repo_root = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "packages").is_dir()
    ),
    None,
)
if _repo_root is None:
    _repo_root = Path("/app")
for _package_dir in ("queue", "types", "config", "shared", "cache"):
    _path = str(_repo_root / "packages" / _package_dir)
    if _path not in sys.path:
        sys.path.append(_path)

try:
    from tiwmai_config import get_queue_settings
    from tiwmai_queue import QueueClient
except Exception:  # pragma: no cover - local partial installs can omit shared packages.
    get_queue_settings = None
    QueueClient = None

# Create router
router = APIRouter()
DEFAULT_COURSE_SUBJECT = "วิชาพื้นฐาน"
QUIZ_GEN_PROGRESS: Dict[str, Dict[str, Any]] = {}
QUIZ_GEN_PROGRESS_LOCK = asyncio.Lock()
STRIPE_API_BASE = "https://api.stripe.com/v1"
PAYMENT_TIME_ZONE = ZoneInfo("Asia/Bangkok")
STUDENT_BEARER_OPTIONAL = HTTPBearer(auto_error=False)
LEGAL_DOC_VERSION = "1.0"
LEGAL_DOC_LAST_UPDATED = "2026-04-13"
ADMIN_TOKEN_USAGE_CACHE_TTL_SECONDS = 60
ADMIN_TOKEN_USAGE_CACHE: Dict[str, Dict[str, Any]] = {}


class SkeletonUpsampleRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    skeletons: List[Dict[str, Any]] = Field(default_factory=list)
    target_count: int = Field(ge=1, le=200)
    model: Optional[str] = None
    progress_job_id: Optional[str] = None


class QuestionBankReplaceRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    items: List[Dict[str, Any]] = Field(default_factory=list, max_length=10000)


def _abstract_skeleton_numeric_text(value: Any) -> str:
    """Replace concrete numeric values with stable placeholders within one field."""
    text = str(value or "").strip()
    if not text:
        return ""
    number_pattern = re.compile(
        r"(?<![A-Za-zก-ฮ])[-+]?(?:\d+|[๐-๙]+)(?:\.(?:\d+|[๐-๙]+))?(?:/(?:\d+|[๐-๙]+))?"
    )
    symbol_pool = ["a", "b", "c", "d", "n", "k", "m", "p", "q", "r"]
    number_to_symbol: Dict[str, str] = {}

    def replace_with_symbol(match: re.Match[str]) -> str:
        token = match.group(0)
        if token not in number_to_symbol:
            idx = len(number_to_symbol)
            number_to_symbol[token] = (
                symbol_pool[idx] if idx < len(symbol_pool) else f"v{idx + 1}"
            )
        return number_to_symbol[token]

    return number_pattern.sub(replace_with_symbol, text)


def _first_seed_context_text(*candidates: Any) -> str:
    for candidate in candidates:
        if isinstance(candidate, dict):
            nested = _first_seed_context_text(
                candidate.get("source_context"),
                candidate.get("sourceContext"),
                candidate.get("extracted_context"),
                candidate.get("extractedContext"),
                candidate.get("document_context"),
                candidate.get("documentContext"),
                candidate.get("context"),
                candidate.get("question_context"),
                candidate.get("questionContext"),
                candidate.get("shared_context"),
                candidate.get("sharedContext"),
                candidate.get("passage"),
                candidate.get("reading_passage"),
                candidate.get("readingPassage"),
                candidate.get("instructions"),
                candidate.get("instruction"),
                candidate.get("stimulus"),
                candidate.get("common_stem"),
                candidate.get("commonStem"),
            )
            if nested:
                return nested
            continue
        text = re.sub(r"\s+", " ", str(candidate or "").strip())
        if text:
            return text
    return ""


def _extract_seed_skeleton_source_material(item: Any) -> Dict[str, Any]:
    """Preserve original extracted context/question material for upsample prompting."""
    if not isinstance(item, dict):
        return {}

    source_candidates = [
        item.get("one_shot_source"),
        item.get("oneShotSource"),
        item.get("original_question"),
        item.get("originalQuestion"),
        item.get("example"),
    ]
    source_dict = next(
        (candidate for candidate in source_candidates if isinstance(candidate, dict)),
        {},
    )

    source_context = _first_seed_context_text(
        item.get("source_context"),
        item.get("sourceContext"),
        item.get("extracted_context"),
        item.get("extractedContext"),
        item.get("document_context"),
        item.get("documentContext"),
        item.get("context"),
        item.get("question_context"),
        item.get("questionContext"),
        item.get("shared_context"),
        item.get("sharedContext"),
        item.get("passage"),
        item.get("reading_passage"),
        item.get("readingPassage"),
        item.get("instructions"),
        item.get("instruction"),
        item.get("stimulus"),
        item.get("common_stem"),
        item.get("commonStem"),
        *source_candidates,
    )
    source_question = re.sub(
        r"\s+",
        " ",
        str(
            item.get("source_question")
            or item.get("sourceQuestion")
            or item.get("question")
            or source_dict.get("question")
            or ""
        ).strip(),
    )
    raw_choices = item.get("choices") or source_dict.get("choices") or []
    source_choices = (
        [
            re.sub(r"\s+", " ", str(choice or "").strip())
            for choice in raw_choices
            if str(choice or "").strip()
        ][:4]
        if isinstance(raw_choices, list)
        else []
    )
    source_explanation = re.sub(
        r"\s+",
        " ",
        str(item.get("explanation") or source_dict.get("explanation") or "").strip(),
    )

    material: Dict[str, Any] = {}
    if source_context:
        material["source_context"] = source_context
    if source_question:
        material["source_question"] = source_question
    if source_choices:
        material["source_choices"] = source_choices
    if source_explanation:
        material["source_explanation"] = source_explanation
    return material


def _normalize_seed_skeleton(item: Any) -> Optional[Dict[str, Any]]:
    """Normalize a skeleton payload to the internal safe-generation shape."""
    if not isinstance(item, dict):
        return None

    source_material = _extract_seed_skeleton_source_material(item)
    subject = str(item.get("subject") or "").strip()
    topic_tags_raw = item.get("topic_tags")
    if not isinstance(topic_tags_raw, list):
        topic_tags_raw = []
    topic_tags = [
        str(tag or "").strip() for tag in topic_tags_raw if str(tag or "").strip()
    ]
    learning_objective = str(item.get("learning_objective") or "").strip()
    core_logic_and_formulas = str(item.get("core_logic_and_formulas") or "").strip()
    context_guidance = str(
        item.get("context_guidance")
        or item.get("contextGuidance")
        or item.get("context_requirements")
        or _first_seed_context_text(
            item.get("source_context"),
            item.get("sourceContext"),
            item.get("extracted_context"),
            item.get("extractedContext"),
            item.get("document_context"),
            item.get("documentContext"),
            item.get("source_material"),
            item.get("sourceMaterial"),
            item.get("context"),
            item.get("question_context"),
            item.get("questionContext"),
            item.get("shared_context"),
            item.get("sharedContext"),
            item.get("passage"),
            item.get("reading_passage"),
            item.get("readingPassage"),
            item.get("instructions"),
            item.get("instruction"),
            item.get("stimulus"),
            item.get("common_stem"),
            item.get("commonStem"),
            item.get("one_shot_source"),
            item.get("oneShotSource"),
            item.get("original_question"),
            item.get("originalQuestion"),
            item.get("example"),
        )
        or source_material.get("source_context")
        or ""
    ).strip()
    variables_raw = (
        item.get("variables") if isinstance(item.get("variables"), dict) else {}
    )
    given_raw = (
        variables_raw.get("given")
        if isinstance(variables_raw.get("given"), list)
        else []
    )
    given = [
        str(value or "").strip() for value in given_raw if str(value or "").strip()
    ]
    target = str(variables_raw.get("target") or "").strip()
    constraints_raw = item.get("constraints_and_tricks")
    if not isinstance(constraints_raw, list):
        constraints_raw = []
    constraints_and_tricks = [
        str(value or "").strip()
        for value in constraints_raw
        if str(value or "").strip()
    ]
    distractor_raw = item.get("distractor_logic")
    if not isinstance(distractor_raw, list):
        distractor_raw = []
    distractor_logic = [
        str(value or "").strip() for value in distractor_raw if str(value or "").strip()
    ]
    if not any(
        [
            subject,
            topic_tags,
            learning_objective,
            core_logic_and_formulas,
            context_guidance,
            given,
            target,
        ]
    ):
        return None

    normalized_skeleton = {
        "subject": subject,
        "topic_tags": topic_tags,
        "learning_objective": learning_objective,
        "core_logic_and_formulas": core_logic_and_formulas,
        "variables": {
            "given": [_abstract_skeleton_numeric_text(value) for value in given],
            "target": _abstract_skeleton_numeric_text(target),
        },
        "constraints_and_tricks": [
            _abstract_skeleton_numeric_text(value) for value in constraints_and_tricks
        ],
        "distractor_logic": [
            _abstract_skeleton_numeric_text(value) for value in distractor_logic
        ],
    }
    if context_guidance:
        normalized_skeleton["context_guidance"] = _abstract_skeleton_numeric_text(
            context_guidance
        )
    return normalized_skeleton


def _normalize_one_shot_example(item: Any) -> Optional[Dict[str, Any]]:
    """Normalize a user-selected example question for one-shot style guidance."""
    if not isinstance(item, dict):
        return None

    def compact_text(value: Any, max_len: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        if len(text) <= max_len:
            return text
        return f"{text[:max_len]}..."

    question = compact_text(item.get("question"), 500)
    raw_choices = item.get("choices") if isinstance(item.get("choices"), list) else []
    choices = [
        compact_text(choice, 180) for choice in raw_choices if str(choice or "").strip()
    ]
    choices = choices[:4]
    if not question or len(choices) < 4:
        return None

    raw_correct = item.get("correct_answer", item.get("correctAnswer", 0))
    try:
        correct_answer = int(raw_correct)
    except (TypeError, ValueError):
        correct_answer = 0
    correct_answer = min(3, max(0, correct_answer))

    normalized = {
        "question": question,
        "choices": choices,
        "correct_answer": correct_answer,
    }
    context = compact_text(item.get("context"), 400)
    explanation = compact_text(item.get("explanation"), 360)
    if context:
        normalized["context"] = context
    if explanation:
        normalized["explanation"] = explanation
    return normalized


def _seed_skeleton_key(skeleton: Dict[str, Any]) -> str:
    return json.dumps(skeleton, ensure_ascii=False, sort_keys=True)


def _normalize_one_shot_example_pair(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    normalized_skeleton = _normalize_seed_skeleton(item.get("skeleton"))
    raw_example = item.get("example") if isinstance(item.get("example"), dict) else item
    normalized_example = _normalize_one_shot_example(raw_example)
    if not normalized_skeleton or not normalized_example:
        return None
    return {
        "skeleton": normalized_skeleton,
        "example": normalized_example,
    }


def _compute_skeleton_pool_target(original_count: int, total_target: int) -> int:
    if original_count <= 0 or total_target <= 0:
        return 0
    return min(total_target, max(original_count, min(original_count * 3, 60)))


def _select_skeleton_for_question(
    skeletons: List[Dict[str, Any]],
    set_index: int,
    num_questions: int,
    question_index: int,
) -> Optional[Dict[str, Any]]:
    if not skeletons:
        return None
    global_index = (
        (max(1, set_index) - 1) * max(0, num_questions) + max(1, question_index) - 1
    )
    return skeletons[global_index % len(skeletons)]


def _skeleton_distribution_label(skeleton: Dict[str, Any]) -> str:
    topic_tags = skeleton.get("topic_tags")
    if isinstance(topic_tags, list):
        for value in topic_tags:
            text = str(value or "").strip().lower()
            if text:
                return text
    return (
        str(skeleton.get("subject") or skeleton.get("learning_objective") or "")
        .strip()
        .lower()
    )


def _build_distributed_skeleton_plan(
    skeletons: List[Dict[str, Any]], total_target: int
) -> List[Dict[str, Any]]:
    if not skeletons or total_target <= 0:
        return []

    pool = skeletons[:]
    random.shuffle(pool)

    if total_target >= len(pool):
        plan: List[Dict[str, Any]] = []
        while len(plan) < total_target:
            cycle = pool[:]
            random.shuffle(cycle)
            if plan and len(cycle) > 1:
                previous_label = _skeleton_distribution_label(plan[-1])
                if _skeleton_distribution_label(cycle[0]) == previous_label:
                    swap_index = next(
                        (
                            index
                            for index, skeleton in enumerate(cycle[1:], start=1)
                            if _skeleton_distribution_label(skeleton) != previous_label
                        ),
                        None,
                    )
                    if swap_index is not None:
                        cycle[0], cycle[swap_index] = cycle[swap_index], cycle[0]
            plan.extend(cycle)
        return plan[:total_target]

    pool_size = len(pool)
    offset = random.randrange(pool_size)
    plan = [
        pool[(offset + (index * pool_size) // total_target) % pool_size]
        for index in range(total_target)
    ]

    for index in range(1, len(plan)):
        previous_label = _skeleton_distribution_label(plan[index - 1])
        if _skeleton_distribution_label(plan[index]) != previous_label:
            continue
        swap_index = next(
            (
                candidate_index
                for candidate_index in range(index + 1, len(plan))
                if _skeleton_distribution_label(plan[candidate_index]) != previous_label
            ),
            None,
        )
        if swap_index is not None:
            plan[index], plan[swap_index] = plan[swap_index], plan[index]

    return plan


def _quiz_question_fingerprint(question: Dict[str, Any]) -> str:
    question_text = re.sub(
        r"\s+", " ", str(question.get("question") or "").strip().lower()
    )
    context_text = re.sub(
        r"\s+", " ", str(question.get("context") or "").strip().lower()
    )
    choices = (
        question.get("choices") if isinstance(question.get("choices"), list) else []
    )
    choice_text = "|".join(
        re.sub(r"\s+", " ", str(choice or "").strip().lower()) for choice in choices
    )
    return hashlib.sha256(
        f"{context_text}|{question_text}|{choice_text}".encode()
    ).hexdigest()


def _quiz_question_similarity_text(question: Dict[str, Any]) -> str:
    choices = (
        question.get("choices") if isinstance(question.get("choices"), list) else []
    )
    text = " ".join(
        [
            str(question.get("context") or ""),
            str(question.get("question") or ""),
            *[str(choice or "") for choice in choices],
        ]
    ).lower()
    return re.sub(r"\s+", " ", text).strip()


def _find_external_visual_reference(question: Dict[str, Any]) -> Optional[str]:
    """Return the visual-reference pattern that makes a question incomplete."""
    choices = (
        question.get("choices") if isinstance(question.get("choices"), list) else []
    )
    text = " ".join(
        [
            str(question.get("context") or ""),
            str(question.get("question") or ""),
            *[str(choice or "") for choice in choices],
        ]
    )
    compact = re.sub(r"\s+", " ", text).strip().lower()
    if not compact:
        return None

    patterns = [
        r"!\[[^\]]*\]\([^)]+\)",
        r"<img\b",
        r"(?:ดู|พิจารณา|อาศัย)\s*(?:รูป|ภาพ|แผนภาพ|ไดอะแกรม|กราฟ)(?!แบบ)",
        r"(?:จาก|ตาม)\s*(?:รูป|ภาพ|แผนภาพ|ไดอะแกรม|กราฟ)\s*(?:ด้านล่าง|ด้านบน|ต่อไปนี้|ข้างต้น|ที่กำหนด|ดังกล่าว|ประกอบ|นี้|[\s,:;]\s*(?:จง|ข้อ|ให้|what|which))",
        r"ดัง\s*(?:รูป|ภาพ|แผนภาพ|ไดอะแกรม|กราฟ)(?!แบบ)",
        r"(?:รูป|ภาพ|แผนภาพ|ไดอะแกรม|กราฟ)\s*(?:ด้านล่าง|ด้านบน|ต่อไปนี้|ข้างต้น|ที่กำหนดให้|ดังกล่าว|ประกอบ)",
        r"รูปใด(?:ต่อไปนี้)?",
        r"(?:refer to|look at|consider|according to|based on)\s+(?:the\s+)?(?:figure|image|picture|diagram|graph)\b",
        r"(?:figure|image|picture|diagram|graph)\s+(?:below|above|shown|provided|following)\b",
        r"(?:shown|illustrated|depicted)\s+in\s+(?:the\s+)?(?:figure|image|picture|diagram|graph)\b",
        r"which\s+(?:figure|image|picture|diagram|graph)\b",
    ]
    for pattern in patterns:
        if re.search(pattern, compact, flags=re.IGNORECASE):
            return pattern
    return None


def _has_quiz_task_instruction(text: str) -> bool:
    compact = re.sub(r"\s+", " ", str(text or "").strip()).lower()
    if not compact:
        return False

    task_patterns = [
        r"\?",
        r"\b(?:what|which|who|where|when|why|how|choose|select|pick|find|solve|calculate|complete|fill|synonym|antonym|odd one out)\b",
        r"(?:ข้อใด|คำใด|ประโยคใด|ตัวเลือกใด|จง|ให้หา|หา|คำนวณ|เลือก|เติม|ตอบ|หมายถึง|แปลว่า|ตรงกับ|สัมพันธ์กับ|ไม่ใช่|ถูกต้อง|เท่ากับ|เท่าไร|อย่างไร|เพราะเหตุใด|คืออะไร)",
    ]
    return any(re.search(pattern, compact, flags=re.IGNORECASE) for pattern in task_patterns)


def _normalize_quiz_list_item(text: Any) -> str:
    return re.sub(r"[^a-z0-9ก-๙]+", "", str(text or "").strip().lower())


def _find_incomplete_quiz_question_reason(question: Dict[str, Any]) -> Optional[str]:
    """Return a reason when a generated item is not a complete readable question."""
    question_text = re.sub(
        r"\s+", " ", str(question.get("question") or "").strip()
    )
    context_text = re.sub(r"\s+", " ", str(question.get("context") or "").strip())
    choices = (
        question.get("choices") if isinstance(question.get("choices"), list) else []
    )
    choices = [str(choice or "").strip() for choice in choices if str(choice).strip()]

    if not question_text:
        return "empty_question"

    combined_instruction_text = f"{context_text} {question_text}".strip()
    if _has_quiz_task_instruction(combined_instruction_text):
        return None

    # Reject items that only repeat the answer choices as a slash/comma/newline list,
    # e.g. "smiled / complained / screamed / fainted" with the same four choices.
    separator_count = len(re.findall(r"\s*(?:/|,|;|\n)\s*", question_text))
    if separator_count >= 2 and len(choices) >= 3:
        parts = [
            _normalize_quiz_list_item(part)
            for part in re.split(r"\s*(?:/|,|;|\n)\s*", question_text)
            if _normalize_quiz_list_item(part)
        ]
        normalized_choices = {
            _normalize_quiz_list_item(choice) for choice in choices if choice.strip()
        }
        matched_parts = [part for part in parts if part in normalized_choices]
        if len(parts) >= 3 and len(matched_parts) >= min(3, len(parts)):
            return "question_is_only_choice_list_without_task"

    return None


def _token_set_for_similarity(text: str) -> Set[str]:
    return set(re.findall(r"[A-Za-z0-9ก-๙]{2,}", text.lower()))


def _is_duplicate_quiz_question(
    candidate: Dict[str, Any],
    accepted_questions: List[Dict[str, Any]],
    accepted_fingerprints: Set[str],
) -> bool:
    fingerprint = _quiz_question_fingerprint(candidate)
    if fingerprint in accepted_fingerprints:
        return True

    candidate_text = _quiz_question_similarity_text(candidate)
    candidate_tokens = _token_set_for_similarity(candidate_text)
    if not candidate_text:
        return False

    for accepted in accepted_questions:
        accepted_text = _quiz_question_similarity_text(accepted)
        if not accepted_text:
            continue
        if candidate_text == accepted_text:
            return True
        if SequenceMatcher(None, candidate_text, accepted_text).ratio() >= 0.9:
            return True
        accepted_tokens = _token_set_for_similarity(accepted_text)
        if len(candidate_tokens) < 4 or len(accepted_tokens) < 4:
            continue
        overlap = candidate_tokens & accepted_tokens
        containment = len(overlap) / max(
            1, min(len(candidate_tokens), len(accepted_tokens))
        )
        jaccard = len(overlap) / max(1, len(candidate_tokens | accepted_tokens))
        if containment >= 0.92 or jaccard >= 0.86:
            return True
    return False


async def _get_student_auth_service() -> StudentAuthService:
    return StudentAuthService()


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


class AdminEnrollmentExpiryOverrideRequest(BaseModel):
    admin_user_id: str
    reason: str
    expires_at: Optional[str] = None


class AdminUserTrialStatusOverrideRequest(BaseModel):
    admin_user_id: str
    mode: str = Field(..., description="auto|available|used")
    reason: Optional[str] = None


class AdminUserPremiumStatusRequest(BaseModel):
    admin_user_id: str
    tier: str = Field(..., description="free|premium")
    expires_at: Optional[str] = None
    duration_months: Optional[int] = Field(
        default=None,
        ge=1,
        le=36,
        description="Used when tier=premium and expires_at is omitted",
    )
    reason: str


class AdminChatEnergyGlobalConfigRequest(BaseModel):
    admin_user_id: str
    daily_limit_thb: float = Field(
        ..., ge=0, description="Global daily chat energy limit per student (THB/day)"
    )
    reason: Optional[str] = None


class AdminUserChatEnergyPolicyRequest(BaseModel):
    admin_user_id: str
    daily_limit_override_thb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Per-user override for daily limit in THB (null = use global default)",
    )
    daily_adjustment_thb: float = Field(
        default=0.0,
        description="Additional adjustment in THB (+/-) applied on top of base limit",
    )
    reason: Optional[str] = None


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


async def _stripe_request(
    method: str,
    path: str,
    secret_key: str,
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url = f"{STRIPE_API_BASE}{path}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.request(
            method=method.upper(),
            url=url,
            auth=(secret_key, ""),
            params=params,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    payload = response.json() if response.content else {}
    if response.status_code >= 400:
        error_obj = payload.get("error") if isinstance(payload, dict) else {}
        message = error_obj.get("message") if isinstance(error_obj, dict) else None
        raise HTTPException(status_code=400, detail=message or "Stripe request failed")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Invalid Stripe response")
    return payload


async def _find_user_enrollment(
    dynamodb_service,
    user_id: str,
    course_id: str,
) -> Optional[str]:
    enrolled_courses = await dynamodb_service.get_enrolled_courses_for_user(user_id)
    for row in enrolled_courses:
        enrolled_course_id = str(
            row.get("course_id") or row.get("id") or row.get("_id") or ""
        ).strip()
        if enrolled_course_id == str(course_id).strip():
            enrollment_id = str(row.get("enrollment_id") or "").strip()
            return enrollment_id or None
    return None


async def _get_existing_enrollment_with_schedule(
    dynamodb_service,
    user_id: str,
    course_id: str,
) -> Optional[Dict[str, Any]]:
    enrollment = await _get_user_course_enrollment(
        dynamodb_service=dynamodb_service,
        user_id=user_id,
        course_id=course_id,
    )
    if not enrollment:
        return None
    schedule = _build_enrollment_schedule(
        started_at_raw=enrollment.get("started_at") or enrollment.get("enrolled_at"),
        expires_at_raw=enrollment.get("expires_at"),
        duration_months_raw=enrollment.get("duration_months"),
    )
    return {
        "enrollment": enrollment,
        "schedule": schedule,
    }


async def _get_user_course_enrollment(
    dynamodb_service,
    user_id: str,
    course_id: str,
) -> Optional[Dict[str, Any]]:
    get_with_aliases = getattr(
        dynamodb_service, "get_user_enrollments_with_aliases", None
    )
    if callable(get_with_aliases):
        enrollments = await get_with_aliases(user_id)
    else:
        enrollments = await dynamodb_service.get_user_enrollments(user_id)
    target_course_id = str(course_id or "").strip()
    for row in enrollments:
        enrolled_course_id = str(
            row.get("course_id") or row.get("id") or row.get("_id") or ""
        ).strip()
        if enrolled_course_id != target_course_id:
            continue
        status = str(row.get("status") or "active").strip().lower()
        if status == "cancelled":
            continue
        return row
    return None


async def _get_all_user_enrollments(
    dynamodb_service,
    user_id: str,
) -> List[Dict[str, Any]]:
    get_with_aliases = getattr(
        dynamodb_service, "get_user_enrollments_with_aliases", None
    )
    if callable(get_with_aliases):
        return await get_with_aliases(user_id)
    return await dynamodb_service.get_user_enrollments(user_id)


def _is_premium_active(user: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(user, dict):
        return False
    subscription = user.get("premium_subscription")
    if not isinstance(subscription, dict) or not subscription:
        return False
    schedule = _build_enrollment_schedule(
        started_at_raw=subscription.get("started_at"),
        expires_at_raw=subscription.get("expires_at"),
        duration_months_raw=subscription.get("duration_months"),
    )
    if schedule["is_expired"] or not schedule.get("expires_at"):
        return False
    return str(subscription.get("status") or "active").strip().lower() != "expired"


PREMIUM_TIER_MODES = ("free", "premium")


def _build_admin_premium_summary(user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    subscription = (
        user.get("premium_subscription")
        if isinstance(user, dict) and isinstance(user.get("premium_subscription"), dict)
        else {}
    )
    is_active = _is_premium_active(user if isinstance(user, dict) else None)
    schedule = _build_enrollment_schedule(
        started_at_raw=subscription.get("started_at"),
        expires_at_raw=subscription.get("expires_at"),
        duration_months_raw=subscription.get("duration_months"),
    )
    admin_override = (
        subscription.get("admin_override")
        if isinstance(subscription.get("admin_override"), dict)
        else {}
    )
    payment_provider = str(subscription.get("payment_provider") or "").strip().lower()
    status_source = "none"
    if is_active or subscription:
        if admin_override:
            status_source = "admin_override"
        elif payment_provider == "stripe" or subscription.get("payment_intent_id"):
            status_source = "payment"
        elif payment_provider == "admin":
            status_source = "admin_override"
    return {
        "tier": "premium" if is_active else "free",
        "is_active": is_active,
        "plan_id": subscription.get("plan_id"),
        "plan_label": subscription.get("plan_label"),
        "expires_at": schedule.get("expires_at"),
        "started_at": schedule.get("started_at"),
        "days_remaining": schedule.get("days_remaining"),
        "is_expired": schedule.get("is_expired"),
        "status_source": status_source,
        "admin_override_updated_at": admin_override.get("updated_at"),
        "admin_override_updated_by": admin_override.get("updated_by"),
        "admin_override_reason": admin_override.get("reason"),
    }


async def _ensure_active_course_access(
    dynamodb_service,
    user_id: str,
    course_id: str,
) -> Dict[str, Any]:
    normalized_user_id = str(user_id or "").strip()
    normalized_course_id = str(course_id or "").strip()
    if not normalized_user_id or not normalized_course_id:
        raise HTTPException(
            status_code=400, detail="Missing user_id or course_id for access check"
        )

    get_user = getattr(dynamodb_service, "get_user", None)
    user = await get_user(normalized_user_id) if callable(get_user) else None

    if _is_premium_active(user):
        enrollment = await _get_user_course_enrollment(
            dynamodb_service=dynamodb_service,
            user_id=normalized_user_id,
            course_id=normalized_course_id,
        )
        if not enrollment:
            lazy_enrollment_data = {
                "progress": 0,
                "completed_quizzes": 0,
                "total_quizzes": 0,
                "completed_questions": 0,
                "total_questions": 0,
                "last_activity": "เพิ่งเข้าร่วมด้วยสิทธิ์ Premium",
                "enrollment_source": "premium",
                "enrollment_type": "premium",
            }
            await dynamodb_service.enroll_user_in_course(
                normalized_user_id, normalized_course_id, lazy_enrollment_data
            )
            enrollment = (
                await _get_user_course_enrollment(
                    dynamodb_service=dynamodb_service,
                    user_id=normalized_user_id,
                    course_id=normalized_course_id,
                )
                or lazy_enrollment_data
            )
        schedule = _build_enrollment_schedule(
            started_at_raw=enrollment.get("started_at") or enrollment.get("enrolled_at"),
            expires_at_raw=None,
            duration_months_raw=None,
        )
        return {"enrollment": enrollment, "schedule": schedule}

    enrollment = await _get_user_course_enrollment(
        dynamodb_service=dynamodb_service,
        user_id=normalized_user_id,
        course_id=normalized_course_id,
    )
    if enrollment and _is_free_course_enrollment(enrollment):
        schedule = _build_enrollment_schedule(
            started_at_raw=enrollment.get("started_at") or enrollment.get("enrolled_at"),
            expires_at_raw=None,
            duration_months_raw=None,
        )
        return {"enrollment": enrollment, "schedule": schedule}
    if enrollment:
        raise HTTPException(
            status_code=403,
            detail="PREMIUM_REQUIRED: your Premium subscription is required to access this course",
        )

    all_enrollments = await _get_all_user_enrollments(dynamodb_service, normalized_user_id)
    claimed_free_enrollment = next(
        (row for row in all_enrollments if _is_free_course_enrollment(row)), None
    )
    if claimed_free_enrollment:
        claimed_course_id = str(claimed_free_enrollment.get("course_id") or "").strip()
        raise HTTPException(
            status_code=403,
            detail=f"FREE_COURSE_LIMIT: your free course is {claimed_course_id}; upgrade to Premium for full access",
        )
    raise HTTPException(
        status_code=403, detail="COURSE_ACCESS_DENIED: not enrolled in this course"
    )


async def _ensure_user_matches_token(
    user_id: Optional[str],
    credentials: Optional[HTTPAuthorizationCredentials],
    auth_service: StudentAuthService,
) -> None:
    requested_user_id = str(user_id or "").strip()
    if not requested_user_id:
        return
    if not credentials or not str(credentials.credentials or "").strip():
        raise HTTPException(
            status_code=401, detail="UNAUTHORIZED: missing bearer token"
        )

    payload = await auth_service.verify_jwt_token(credentials.credentials)
    principals = {
        str(value).strip()
        for value in [
            payload.get("sub"),
            payload.get("username"),
            payload.get("cognito:username"),
            payload.get("user_id"),
            payload.get("custom:student_id"),
        ]
        if str(value or "").strip()
    }
    if not principals:
        raise HTTPException(
            status_code=401, detail="UNAUTHORIZED: token has no principal"
        )

    if requested_user_id in principals:
        return

    requested_lower = requested_user_id.lower()
    principals_lower = {value.lower() for value in principals}
    if requested_lower in principals_lower:
        return

    raise HTTPException(status_code=403, detail="USER_ID_TOKEN_MISMATCH")


def _parse_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(str(value).strip())
        return parsed if parsed > 0 else None
    except Exception:
        return None


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
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


def _build_enrollment_schedule(
    started_at_raw: Any,
    expires_at_raw: Any,
    duration_months_raw: Any,
) -> Dict[str, Any]:
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
    return {
        "started_at": _format_utc_iso(started_dt),
        "expires_at": _format_utc_iso(expires_dt),
        "duration_months": duration_months,
        "is_expired": is_expired,
        "days_remaining": days_remaining,
    }


def _is_free_course_enrollment(enrollment: Dict[str, Any]) -> bool:
    source = (
        str(
            enrollment.get("enrollment_source")
            or enrollment.get("enrollment_type")
            or ""
        )
        .strip()
        .lower()
    )
    if source == "free":
        return True
    return bool(enrollment.get("free_course_claimed_at"))


def _coerce_number(*values: Any) -> float:
    for value in values:
        try:
            parsed = float(value)
            if parsed == parsed and parsed not in (float("inf"), float("-inf")):
                return parsed
        except Exception:
            continue
    return 0.0


def _normalize_learning_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_lesson_id(value: Any) -> str:
    return str(value or "").strip()


def _get_lesson_name(lesson: Dict[str, Any], fallback_index: int = 0) -> str:
    raw = (
        lesson.get("title")
        or lesson.get("name")
        or lesson.get("lesson_name")
        or lesson.get("topic")
        or ""
    )
    text = str(raw or "").strip()
    return text or f"บทเรียน {fallback_index + 1}"


def _to_topic_label(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _resolve_question_topic_label(
    question: Dict[str, Any], fallback: str = "ไม่ระบุหัวข้อ"
) -> str:
    return (
        _to_topic_label(
            question.get("topic_tag"),
            question.get("topicTag"),
            question.get("topic"),
            question.get("subject_tag"),
            question.get("subject"),
            question.get("category"),
        )
        or fallback
    )


def _get_attempt_stats(item: Dict[str, Any]) -> Optional[Dict[str, float]]:
    total_questions = max(0.0, _coerce_number(item.get("total_questions")))
    correct_count = max(0.0, _coerce_number(item.get("correct_count")))
    if total_questions > 0:
        bounded_correct = min(correct_count, total_questions)
        return {
            "total": total_questions,
            "correct": bounded_correct,
            "accuracy": (bounded_correct / total_questions) * 100,
        }

    score_raw = item.get("score")
    try:
        score = float(score_raw)
    except Exception:
        return None
    if score != score:
        return None
    bounded_score = max(0.0, min(100.0, score))
    return {
        "total": 1.0,
        "correct": bounded_score / 100.0,
        "accuracy": bounded_score,
    }


def _to_difficulty_label(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        if value <= 1:
            return "ง่าย"
        if value == 2:
            return "กลาง"
        return "ยาก"

    text = _normalize_learning_text(value)
    if "easy" in text or "ง่าย" in text:
        return "ง่าย"
    if "hard" in text or "ยาก" in text or "advanced" in text:
        return "ยาก"
    if "medium" in text or "กลาง" in text or "intermediate" in text:
        return "กลาง"
    return None


def _to_difficulty_label_from_quiz(quiz: Dict[str, Any]) -> Optional[str]:
    direct = _to_difficulty_label(
        quiz.get("difficulty_avg")
        if quiz.get("difficulty_avg") is not None
        else quiz.get("difficulty")
        if quiz.get("difficulty") is not None
        else quiz.get("level_difficulty")
        if quiz.get("level_difficulty") is not None
        else quiz.get("difficulty_level")
        if quiz.get("difficulty_level") is not None
        else quiz.get("level")
    )
    if direct:
        return direct

    questions = quiz.get("questions")
    if not isinstance(questions, list):
        return None
    numeric_difficulties = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        try:
            parsed = float(question.get("difficulty"))
        except Exception:
            continue
        if parsed > 0:
            numeric_difficulties.append(parsed)
    if not numeric_difficulties:
        return None
    avg = sum(numeric_difficulties) / len(numeric_difficulties)
    if avg <= 2:
        return "ง่าย"
    if avg >= 4:
        return "ยาก"
    return "กลาง"


def _detect_quiz_kind(payload: Dict[str, Any]) -> str:
    tags = payload.get("tags")
    if isinstance(tags, list):
        tags_text = " ".join(str(item) for item in tags)
    else:
        tags_text = str(tags or "")
    text = " ".join(
        str(value or "")
        for value in (
            payload.get("title"),
            payload.get("name"),
            payload.get("quiz_type"),
            payload.get("type"),
            payload.get("purpose"),
            payload.get("description"),
            payload.get("document_type"),
            tags_text,
        )
        if value
    ).lower()
    if "mock_exam" in text or "mock exam" in text or "แบบทดสอบจำลอง" in text:
        return "mock_exam"
    return "lesson"


def _parse_timestamp_ms(value: Any) -> float:
    dt = _parse_iso_datetime(value)
    if not dt:
        return 0.0
    return dt.timestamp() * 1000


def _format_attempt_label(submitted_at_ms: float, fallback_index: int) -> str:
    if submitted_at_ms > 0:
        dt = datetime.fromtimestamp(submitted_at_ms / 1000)
        return dt.strftime("%d/%m")
    return f"ครั้งที่ {fallback_index + 1}"


def _merge_course_with_enrollment(
    course: Dict[str, Any], enrollment: Dict[str, Any]
) -> Dict[str, Any]:
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
    row["total_quizzes"] = enrollment.get("total_quizzes", row.get("total_quizzes", 0))
    row["completed_questions"] = enrollment.get(
        "completed_questions", row.get("completed_questions", 0)
    )
    row["total_questions"] = enrollment.get(
        "total_questions", row.get("total_questions", 0)
    )
    row["last_activity"] = enrollment.get("last_activity", row.get("last_activity"))
    return row


def _format_student_course(course: Dict[str, Any]) -> Dict[str, Any]:
    schedule = _build_enrollment_schedule(
        started_at_raw=course.get("started_at") or course.get("enrolled_at"),
        expires_at_raw=course.get("expires_at"),
        duration_months_raw=course.get("duration_months"),
    )
    return {
        "id": course.get("course_id"),
        "name": course.get("name"),
        "description": course.get("description"),
        "detail": course.get("detail"),
        "target_profile": course.get("target_profile"),
        "structure_summary": course.get("structure_summary"),
        "topics": course.get("topics", []),
        "tags": course.get("tags", []),
        "course_tags": course.get("tags", []),
        "benefits": course.get("benefits", []),
        "content_items": course.get("content_items", []),
        "instructor": course.get("instructor")
        or course.get("teacher_name")
        or "อาจารย์ระบบ",
        "teacher_name": course.get("teacher_name")
        or course.get("instructor")
        or "อาจารย์ระบบ",
        "category": course.get("category", "ทั่วไป"),
        "progress": course.get("progress", 0),
        "totalQuizzes": course.get("total_quizzes", 0),
        "completedQuizzes": course.get("completed_quizzes", 0),
        "totalQuestions": course.get("total_questions", 0),
        "completedQuestions": course.get("completed_questions", 0),
        "lastActivity": course.get("last_activity", "เพิ่งเข้าร่วม"),
        "color": "#4ecdc4",
        "image": "📚",
        "image_url": course.get("image_url"),
        "thumbnail_url": course.get("thumbnail_url"),
        "preview_image_url": course.get("preview_image_url"),
        "purchase_preview_image_url": course.get("purchase_preview_image_url"),
        "price": course.get("price"),
        "enrollment_id": course.get("enrollment_id"),
        "enrolled_at": course.get("enrolled_at"),
        "started_at": schedule["started_at"],
        "expires_at": schedule["expires_at"],
        "duration_months": schedule["duration_months"],
        "is_expired": schedule["is_expired"],
        "days_remaining": schedule["days_remaining"],
        "enrollment_source": course.get("enrollment_source"),
        "enrollment_type": course.get("enrollment_type"),
        "free_course_claimed_at": course.get("free_course_claimed_at"),
        "is_free_course": _is_free_course_enrollment(course),
    }


def _build_dashboard_course_stats(
    courses: List[Dict[str, Any]],
    lessons: List[Dict[str, Any]],
    quizzes: List[Dict[str, Any]],
    quiz_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    lessons_by_course: Dict[str, List[Dict[str, Any]]] = {}
    for lesson in lessons:
        course_id = str(lesson.get("course_id") or lesson.get("courseId") or "")
        lessons_by_course.setdefault(course_id, []).append(lesson)

    quizzes_by_course: Dict[str, List[Dict[str, Any]]] = {}
    for quiz in quizzes:
        course_id = str(quiz.get("course_id") or quiz.get("courseId") or "")
        quizzes_by_course.setdefault(course_id, []).append(quiz)

    stats: Dict[str, Dict[str, Any]] = {}
    week_start_ms = (datetime.utcnow() - timedelta(days=7)).timestamp() * 1000

    for course in courses:
        course_id = str(course.get("course_id") or course.get("id") or "")
        if not course_id:
            continue

        course_lessons = lessons_by_course.get(course_id, [])
        course_quizzes = quizzes_by_course.get(course_id, [])
        regular_quizzes = [
            quiz
            for quiz in course_quizzes
            if str(quiz.get("document_type") or "").lower() != "mock_exam"
        ]

        lesson_rows = []
        for lesson_index, lesson in enumerate(
            sorted(
                course_lessons,
                key=lambda row: _coerce_number(row.get("order"), 0)
                if row.get("order") is not None
                else 0,
            )
        ):
            lesson_id = _normalize_lesson_id(
                lesson.get("id") or lesson.get("lesson_id")
            )
            if not lesson_id:
                continue
            lesson_rows.append(
                {
                    "id": lesson_id,
                    "name": _get_lesson_name(lesson, lesson_index),
                    "order": int(_coerce_number(lesson.get("order"), lesson_index + 1)),
                }
            )

        lesson_name_by_id = {lesson["id"]: lesson["name"] for lesson in lesson_rows}
        quiz_to_lesson_id: Dict[str, str] = {}
        for lesson in course_lessons:
            lesson_id = _normalize_lesson_id(
                lesson.get("id") or lesson.get("lesson_id")
            )
            if not lesson_id:
                continue
            quiz_refs = []
            for key in ("quizzes", "selected_quizzes", "selectedQuizzes"):
                value = lesson.get(key)
                if isinstance(value, list):
                    quiz_refs.extend(value)
            for quiz_ref in quiz_refs:
                if not isinstance(quiz_ref, dict):
                    quiz_id = str(quiz_ref or "").strip()
                    if quiz_id:
                        quiz_to_lesson_id[quiz_id] = lesson_id
                    continue
                for raw_id in (
                    quiz_ref.get("quiz_id"),
                    quiz_ref.get("id"),
                    quiz_ref.get("document_id"),
                ):
                    quiz_id = str(raw_id or "").strip()
                    if quiz_id:
                        quiz_to_lesson_id[quiz_id] = lesson_id

        quiz_ids = {
            str(quiz.get("quiz_id") or quiz.get("id") or quiz.get("document_id") or "")
            for quiz in course_quizzes
            if str(
                quiz.get("quiz_id") or quiz.get("id") or quiz.get("document_id") or ""
            )
        }
        quiz_difficulty_by_id = {
            str(
                quiz.get("quiz_id") or quiz.get("id") or quiz.get("document_id")
            ): _to_difficulty_label_from_quiz(quiz)
            for quiz in regular_quizzes
            if str(
                quiz.get("quiz_id") or quiz.get("id") or quiz.get("document_id") or ""
            )
        }
        quiz_topic_by_id = {
            str(
                quiz.get("quiz_id") or quiz.get("id") or quiz.get("document_id")
            ): _to_topic_label(
                quiz.get("topic_tag"),
                quiz.get("topicTag"),
                quiz.get("topic"),
                quiz.get("category"),
                quiz.get("subject"),
            )
            for quiz in course_quizzes
            if str(
                quiz.get("quiz_id") or quiz.get("id") or quiz.get("document_id") or ""
            )
            and _to_topic_label(
                quiz.get("topic_tag"),
                quiz.get("topicTag"),
                quiz.get("topic"),
                quiz.get("category"),
                quiz.get("subject"),
            )
        }
        all_quiz_kind_by_id = {
            str(
                quiz.get("quiz_id") or quiz.get("id") or quiz.get("document_id")
            ): _detect_quiz_kind(quiz)
            for quiz in course_quizzes
            if str(
                quiz.get("quiz_id") or quiz.get("id") or quiz.get("document_id") or ""
            )
        }

        course_results = []
        for item in quiz_results:
            result_course_id = str(item.get("course_id") or "")
            result_quiz_id = str(item.get("quiz_id") or "")
            if result_course_id and result_course_id == course_id:
                course_results.append(item)
            elif result_quiz_id and result_quiz_id in quiz_ids:
                course_results.append(item)

        time_spent_seconds_this_week = 0.0
        for item in course_results:
            seconds = max(
                0.0,
                _coerce_number(
                    item.get("time_spent_seconds"),
                    item.get("total_time_spent_seconds"),
                ),
            )
            if seconds <= 0:
                continue
            timestamp_ms = _parse_timestamp_ms(
                item.get("submitted_at")
                or item.get("updated_at")
                or item.get("created_at")
            )
            if timestamp_ms <= 0 or timestamp_ms >= week_start_ms:
                time_spent_seconds_this_week += seconds

        attempted_quiz_ids = {
            str(item.get("quiz_id"))
            for item in course_results
            if str(item.get("quiz_id") or "")
        }
        question_attempts = sum(
            _coerce_number(item.get("total_questions")) for item in course_results
        )
        correct_answers = sum(
            _coerce_number(item.get("correct_count")) for item in course_results
        )
        attempted_lesson_ids = set()
        difficulty_buckets = {
            "easy": {"correct": 0.0, "total": 0.0},
            "medium": {"correct": 0.0, "total": 0.0},
            "hard": {"correct": 0.0, "total": 0.0},
        }
        score_buckets = {
            "lesson": {"correct": 0.0, "total": 0.0},
            "mockExam": {"correct": 0.0, "total": 0.0},
        }
        topic_buckets: Dict[str, Dict[str, Any]] = {}
        lesson_topic_buckets: Dict[str, Dict[str, Any]] = {}
        scored_attempt_count = 0
        lesson_buckets: Dict[str, Dict[str, Any]] = {}
        for lesson in lesson_rows:
            lesson_buckets[lesson["id"]] = {
                "id": lesson["id"],
                "name": lesson["name"],
                "order": lesson["order"],
                "minutes": 0.0,
                "lesson": {"correct": 0.0, "total": 0.0},
                "mockExam": {"correct": 0.0, "total": 0.0},
            }

        def _add_topic_stat(
            bucket: Dict[str, Dict[str, Any]],
            topic_label: str,
            total: float,
            correct: float,
        ) -> None:
            if topic_label not in bucket:
                bucket[topic_label] = {
                    "topic": topic_label,
                    "total": 0.0,
                    "correct": 0.0,
                }
            bucket[topic_label]["total"] += total
            bucket[topic_label]["correct"] += correct

        def _add_lesson_topic_stat(
            lesson_id: Optional[str],
            topic_label: str,
            total: float,
            correct: float,
        ) -> None:
            group_key = lesson_id or "__unassigned__"
            lesson_meta = next(
                (lesson for lesson in lesson_rows if lesson["id"] == lesson_id),
                None,
            )
            if group_key not in lesson_topic_buckets:
                lesson_topic_buckets[group_key] = {
                    "lessonId": lesson_id,
                    "lessonName": (
                        lesson_name_by_id.get(lesson_id) if lesson_id else None
                    )
                    or "ไม่ระบุบท",
                    "lessonOrder": lesson_meta.get("order")
                    if lesson_meta
                    else 9007199254740991,
                    "topics": {},
                }
            _add_topic_stat(
                lesson_topic_buckets[group_key]["topics"],
                topic_label,
                total,
                correct,
            )

        for item in course_results:
            attempt_stats = _get_attempt_stats(item)
            if not attempt_stats:
                continue
            scored_attempt_count += 1
            total_questions = attempt_stats["total"]
            correct_count = attempt_stats["correct"]

            mapped_difficulty = quiz_difficulty_by_id.get(
                str(item.get("quiz_id"))
            ) or _to_difficulty_label(
                item.get("difficulty")
                or item.get("level_difficulty")
                or item.get("difficulty_level")
            )
            bucket_key = (
                "easy"
                if mapped_difficulty == "ง่าย"
                else "hard"
                if mapped_difficulty == "ยาก"
                else "medium"
                if mapped_difficulty == "กลาง"
                else None
            )
            if bucket_key:
                difficulty_buckets[bucket_key]["total"] += total_questions
                difficulty_buckets[bucket_key]["correct"] += correct_count

            kind = all_quiz_kind_by_id.get(
                str(item.get("quiz_id"))
            ) or _detect_quiz_kind(item)
            kind_key = "mockExam" if kind == "mock_exam" else "lesson"
            is_lesson_practice = kind_key == "lesson"
            score_buckets[kind_key]["total"] += total_questions
            score_buckets[kind_key]["correct"] += correct_count

            explicit_lesson_id = _normalize_lesson_id(
                item.get("lesson_id") or item.get("lessonId")
            )
            quiz_mapped_lesson_id = _normalize_lesson_id(
                quiz_to_lesson_id.get(str(item.get("quiz_id") or ""))
            )
            mapped_lesson_id = (
                explicit_lesson_id
                if explicit_lesson_id and explicit_lesson_id in lesson_name_by_id
                else (quiz_mapped_lesson_id or explicit_lesson_id)
            )
            has_known_lesson = bool(
                mapped_lesson_id and mapped_lesson_id in lesson_buckets
            )

            fallback_topic_label = (
                _to_topic_label(
                    item.get("topic_tag"),
                    item.get("topicTag"),
                    item.get("topic"),
                    item.get("subject_tag"),
                    quiz_topic_by_id.get(str(item.get("quiz_id") or "")),
                )
                or "ไม่ระบุหัวข้อ"
            )
            question_insights = item.get("question_insights")
            answered_question_insights = []
            if isinstance(question_insights, list):
                answered_question_insights = [
                    question
                    for question in question_insights
                    if isinstance(question, dict)
                    and question.get("is_correct") in (True, False)
                ]
            if answered_question_insights:
                for question in answered_question_insights:
                    topic_label = _resolve_question_topic_label(
                        question, fallback_topic_label
                    )
                    correct_value = 1.0 if question.get("is_correct") is True else 0.0
                    _add_topic_stat(topic_buckets, topic_label, 1.0, correct_value)
                    if is_lesson_practice and has_known_lesson:
                        _add_lesson_topic_stat(
                            mapped_lesson_id,
                            topic_label,
                            1.0,
                            correct_value,
                        )
            elif total_questions > 0:
                _add_topic_stat(
                    topic_buckets,
                    fallback_topic_label,
                    total_questions,
                    correct_count,
                )
                if is_lesson_practice and has_known_lesson:
                    _add_lesson_topic_stat(
                        mapped_lesson_id,
                        fallback_topic_label,
                        total_questions,
                        correct_count,
                    )
            if not has_known_lesson:
                continue
            attempted_lesson_ids.add(mapped_lesson_id)
            lesson_buckets[mapped_lesson_id][kind_key]["total"] += total_questions
            lesson_buckets[mapped_lesson_id][kind_key]["correct"] += correct_count

            seconds = max(
                0.0,
                _coerce_number(
                    item.get("time_spent_seconds"),
                    item.get("total_time_spent_seconds"),
                ),
            )
            if seconds > 0:
                lesson_buckets[mapped_lesson_id]["minutes"] += seconds / 60

        computed_lesson_rows = []
        lesson_quiz_ids: Dict[str, set] = {}
        for quiz_id, lesson_id in quiz_to_lesson_id.items():
            if all_quiz_kind_by_id.get(quiz_id) == "mock_exam":
                continue
            lesson_quiz_ids.setdefault(lesson_id, set()).add(quiz_id)
        for lesson in sorted(
            lesson_buckets.values(),
            key=lambda row: (row.get("order") or 0, str(row.get("name") or "")),
        ):
            lesson_total = lesson["lesson"]["total"]
            mock_total = lesson["mockExam"]["total"]
            lesson_quiz_set = lesson_quiz_ids.get(lesson["id"], set())
            total_lesson_quizzes = len(lesson_quiz_set)
            completed_lesson_quizzes = len(
                [quiz_id for quiz_id in lesson_quiz_set if quiz_id in attempted_quiz_ids]
            )
            lesson_progress = (
                round((completed_lesson_quizzes / total_lesson_quizzes) * 100)
                if total_lesson_quizzes > 0
                else None
            )
            computed_lesson_rows.append(
                {
                    "id": lesson["id"],
                    "name": lesson["name"],
                    "scoreSplit": {
                        "lesson": round(
                            (lesson["lesson"]["correct"] / lesson_total) * 100
                        )
                        if lesson_total > 0
                        else None,
                        "mockExam": round(
                            (lesson["mockExam"]["correct"] / mock_total) * 100
                        )
                        if mock_total > 0
                        else None,
                    },
                    "minutes": round(lesson["minutes"]) if lesson["minutes"] > 0 else 0,
                    "totalQuizzes": total_lesson_quizzes,
                    "completedQuizzes": completed_lesson_quizzes,
                    "progress": lesson_progress,
                }
            )

        attempt_rows = []
        for index, item in enumerate(course_results):
            attempt_stats = _get_attempt_stats(item)
            if not attempt_stats or attempt_stats["total"] <= 0:
                continue
            submitted_at_raw = (
                item.get("submitted_at")
                or item.get("updated_at")
                or item.get("created_at")
            )
            submitted_at_ms = _parse_timestamp_ms(submitted_at_raw)
            safe_score = max(
                0,
                min(
                    100,
                    round((attempt_stats["correct"] / attempt_stats["total"]) * 100),
                ),
            )
            attempt_rows.append(
                {
                    "id": f"{course_id}-{item.get('result_id') or item.get('id') or item.get('quiz_id') or index}",
                    "score": safe_score,
                    "submittedAt": submitted_at_raw or None,
                    "submittedAtMs": submitted_at_ms if submitted_at_ms > 0 else 0,
                    "quizTitle": str(
                        item.get("quiz_title")
                        or item.get("quiz_name")
                        or item.get("title")
                        or ""
                    ).strip(),
                    "sequence": index + 1,
                }
            )
        attempt_rows.sort(
            key=lambda row: (
                row["submittedAtMs"] <= 0,
                row["submittedAtMs"] if row["submittedAtMs"] > 0 else row["sequence"],
            )
        )
        for index, row in enumerate(attempt_rows):
            row["label"] = _format_attempt_label(row["submittedAtMs"], index)
            row["attemptIndex"] = index + 1

        topic_rows = [
            {
                "id": f"{course_id}-{topic['topic']}",
                "topic": topic["topic"],
                "total": int(round(topic["total"])),
                "correct": int(round(topic["correct"])),
                "accuracy": round((topic["correct"] / topic["total"]) * 100),
            }
            for topic in topic_buckets.values()
            if topic["total"] > 0
        ]
        topic_rows.sort(
            key=lambda topic: (
                -topic["total"],
                topic["topic"] == "ไม่ระบุหัวข้อ",
                str(topic["topic"] or ""),
            )
        )
        topic_rows_by_lesson = []
        for group in lesson_topic_buckets.values():
            group_topics = [
                {
                    "id": f"{course_id}-{group['lessonId'] or 'unassigned'}-{topic['topic']}",
                    "topic": topic["topic"],
                    "total": int(round(topic["total"])),
                    "correct": int(round(topic["correct"])),
                    "accuracy": round((topic["correct"] / topic["total"]) * 100),
                }
                for topic in group["topics"].values()
                if topic["total"] > 0
            ]
            group_topics.sort(
                key=lambda topic: (
                    -topic["total"],
                    topic["topic"] == "ไม่ระบุหัวข้อ",
                    str(topic["topic"] or ""),
                )
            )
            if group_topics:
                topic_rows_by_lesson.append(
                    {
                        "lessonId": group["lessonId"],
                        "lessonName": group["lessonName"],
                        "lessonOrder": group["lessonOrder"],
                        "topics": group_topics,
                    }
                )
        topic_rows_by_lesson.sort(
            key=lambda group: (
                group.get("lessonOrder") or 9007199254740991,
                str(group.get("lessonName") or ""),
            )
        )

        difficulty_score = {
            key: round((bucket["correct"] / bucket["total"]) * 100)
            if bucket["total"] > 0
            else None
            for key, bucket in difficulty_buckets.items()
        }
        total_difficulty_questions = sum(
            bucket["total"] for bucket in difficulty_buckets.values()
        )
        total_difficulty_correct = sum(
            bucket["correct"] for bucket in difficulty_buckets.values()
        )
        score_split = {
            "lesson": round(
                (score_buckets["lesson"]["correct"] / score_buckets["lesson"]["total"])
                * 100
            )
            if score_buckets["lesson"]["total"] > 0
            else None,
            "mockExam": round(
                (
                    score_buckets["mockExam"]["correct"]
                    / score_buckets["mockExam"]["total"]
                )
                * 100
            )
            if score_buckets["mockExam"]["total"] > 0
            else None,
        }
        completed_quizzes = len(attempted_quiz_ids)
        total_quizzes = len(regular_quizzes)
        progress = (
            round((completed_quizzes / total_quizzes) * 100)
            if total_quizzes > 0
            else _coerce_number(course.get("progress"))
        )
        submitted_values = [
            str(item.get("submitted_at") or "")
            for item in course_results
            if str(item.get("submitted_at") or "")
        ]
        last_submitted_at = sorted(submitted_values)[-1] if submitted_values else None

        stats[course_id] = {
            "totalLessons": len(course_lessons),
            "completedLessons": len(attempted_lesson_ids),
            "totalQuizzes": total_quizzes,
            "completedQuizzes": completed_quizzes,
            "totalQuestions": question_attempts
            if question_attempts > 0
            else scored_attempt_count,
            "completedQuestions": correct_answers
            if question_attempts > 0
            else scored_attempt_count,
            "lessonRows": computed_lesson_rows,
            "attemptRows": attempt_rows,
            "topicRows": topic_rows,
            "topicRowsByLesson": topic_rows_by_lesson,
            "minutesThisWeek": int((time_spent_seconds_this_week + 59) // 60)
            if time_spent_seconds_this_week > 0
            else 0,
            "progress": max(0, min(100, round(progress))),
            "averageScore": round(
                (total_difficulty_correct / total_difficulty_questions) * 100
            )
            if total_difficulty_questions > 0
            else round((correct_answers / question_attempts) * 100)
            if question_attempts > 0
            else 0,
            "difficultyScore": difficulty_score,
            "scoreSplit": score_split,
            "lastActivity": last_submitted_at,
            "learningActivityDays": [
                str(day)
                for day in course.get("learning_activity_days", [])
                if str(day or "").strip()
            ]
            if isinstance(course.get("learning_activity_days"), list)
            else [],
        }

    return stats


TRIAL_OVERRIDE_MODES = {"auto", "available", "used"}


def _normalize_trial_override_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in TRIAL_OVERRIDE_MODES:
        return mode
    return "auto"


def _extract_trial_override(user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(user, dict):
        return {
            "mode": "auto",
            "updated_at": None,
            "updated_by": None,
            "reason": None,
        }
    override = user.get("admin_trial_override")
    if not isinstance(override, dict):
        return {
            "mode": _normalize_trial_override_mode(user.get("trial_override_mode")),
            "updated_at": None,
            "updated_by": None,
            "reason": None,
        }
    return {
        "mode": _normalize_trial_override_mode(override.get("mode")),
        "updated_at": str(override.get("updated_at") or "").strip() or None,
        "updated_by": str(override.get("updated_by") or "").strip() or None,
        "reason": str(override.get("reason") or "").strip() or None,
    }


def _resolve_effective_trial_used(
    trial_used_from_enrollments: bool,
    override_mode: Any,
) -> Dict[str, Any]:
    normalized_mode = _normalize_trial_override_mode(override_mode)
    if normalized_mode == "used":
        return {
            "trial_used": True,
            "trial_status_source": "admin_override",
        }
    if normalized_mode == "available":
        return {
            "trial_used": False,
            "trial_status_source": "admin_override",
        }
    return {
        "trial_used": bool(trial_used_from_enrollments),
        "trial_status_source": "enrollment",
    }


async def _set_user_trial_override(
    dynamodb_service,
    user_id: str,
    mode: str,
    updated_by: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_user_id = str(user_id or "").strip()
    normalized_mode = _normalize_trial_override_mode(mode)
    normalized_updated_by = str(updated_by or "").strip()
    normalized_reason = str(reason or "").strip() or None
    if not normalized_user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if not normalized_updated_by:
        raise HTTPException(status_code=400, detail="admin_user_id is required")

    now = datetime.utcnow().isoformat()
    user = await dynamodb_service.get_user(normalized_user_id)
    item = dict(user or {})
    if not str(item.get("user_id") or "").strip():
        item["user_id"] = normalized_user_id
    if not str(item.get("email") or "").strip():
        item["email"] = f"{normalized_user_id}@example.com"
    if not str(item.get("name") or "").strip():
        item["name"] = f"User {normalized_user_id}"
    if not str(item.get("role") or "").strip():
        item["role"] = "student"
    if not str(item.get("status") or "").strip():
        item["status"] = "active"
    if not str(item.get("created_at") or "").strip():
        item["created_at"] = now

    trial_override = {
        "mode": normalized_mode,
        "updated_at": now,
        "updated_by": normalized_updated_by,
        "reason": normalized_reason,
    }
    item["admin_trial_override"] = trial_override
    item["updated_at"] = now
    dynamodb_service.users_table.put_item(
        Item=dynamodb_service._convert_floats_to_decimal(item)
    )
    return trial_override


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_chat_energy_response(status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(status, dict):
        return {
            "daily_limit_thb": 0.0,
            "used_thb": 0.0,
            "remaining_thb": 0.0,
            "remaining_percent": 0.0,
            "is_exhausted": True,
            "daily_limit_override_thb": None,
            "daily_adjustment_thb": 0.0,
            "limit_source": "global_default",
            "usage_date": datetime.utcnow().date().isoformat(),
            "request_count": 0,
            "policy_updated_at": None,
            "policy_updated_by": None,
            "policy_reason": None,
            "platform_updated_at": None,
            "platform_updated_by": None,
            "platform_reason": None,
            "default_daily_limit_thb": 0.0,
        }
    return {
        "daily_limit_thb": _safe_float(status.get("daily_limit_thb"), 0.0),
        "used_thb": _safe_float(status.get("used_thb"), 0.0),
        "remaining_thb": _safe_float(status.get("remaining_thb"), 0.0),
        "remaining_percent": _safe_float(status.get("remaining_percent"), 0.0),
        "is_exhausted": bool(status.get("is_exhausted")),
        "daily_limit_override_thb": (
            _safe_float(status.get("daily_limit_override_thb"), 0.0)
            if status.get("daily_limit_override_thb") is not None
            else None
        ),
        "daily_adjustment_thb": _safe_float(status.get("daily_adjustment_thb"), 0.0),
        "limit_source": str(status.get("limit_source") or "global_default"),
        "usage_date": str(
            status.get("usage_date") or datetime.utcnow().date().isoformat()
        ),
        "request_count": int(_safe_float(status.get("request_count"), 0)),
        "policy_updated_at": str(status.get("policy_updated_at") or "").strip() or None,
        "policy_updated_by": str(status.get("policy_updated_by") or "").strip() or None,
        "policy_reason": str(status.get("policy_reason") or "").strip() or None,
        "platform_updated_at": str(status.get("platform_updated_at") or "").strip()
        or None,
        "platform_updated_by": str(status.get("platform_updated_by") or "").strip()
        or None,
        "platform_reason": str(status.get("platform_reason") or "").strip() or None,
        "default_daily_limit_thb": _safe_float(
            status.get("default_daily_limit_thb"), 0.0
        ),
    }


def _build_payment_order_id(paid_at: Any, payment_intent_id: Any) -> Optional[str]:
    intent_id = str(payment_intent_id or "").strip()
    if not intent_id:
        return None
    paid_at_dt = _parse_iso_datetime(paid_at) or datetime.utcnow()
    paid_at_dt = paid_at_dt.replace(tzinfo=timezone.utc).astimezone(PAYMENT_TIME_ZONE)
    suffix = hashlib.sha1(intent_id.encode("utf-8")).hexdigest()[:4].upper()
    return f"TM{paid_at_dt.strftime('%Y%m%d')}-{suffix}"


def _latest_charge_from_intent(intent: Dict[str, Any]) -> Dict[str, Any]:
    latest_charge = intent.get("latest_charge")
    if isinstance(latest_charge, dict):
        return latest_charge
    if isinstance(latest_charge, str) and latest_charge.strip():
        return {"id": latest_charge.strip()}
    return {}


async def _stripe_receipt_fields_from_payment_intent(
    payment_intent_id: Any, secret_key: str
) -> Dict[str, Optional[str]]:
    intent_id = str(payment_intent_id or "").strip()
    if not intent_id:
        return {
            "stripe_charge_id": None,
            "receipt_number": None,
            "receipt_url": None,
        }

    intent = await _stripe_request(
        method="GET",
        path=f"/payment_intents/{intent_id}",
        secret_key=secret_key,
        params={"expand[]": "latest_charge"},
    )
    latest_charge = _latest_charge_from_intent(intent)
    stripe_charge_id = str(latest_charge.get("id") or "").strip() or None
    receipt_number = str(latest_charge.get("receipt_number") or "").strip() or None
    receipt_url = str(latest_charge.get("receipt_url") or "").strip() or None
    if stripe_charge_id and (not receipt_number or not receipt_url):
        charge = await _stripe_request(
            method="GET",
            path=f"/charges/{stripe_charge_id}",
            secret_key=secret_key,
        )
        receipt_number = (
            receipt_number or str(charge.get("receipt_number") or "").strip() or None
        )
        receipt_url = (
            receipt_url or str(charge.get("receipt_url") or "").strip() or None
        )

    return {
        "stripe_charge_id": stripe_charge_id,
        "receipt_number": receipt_number,
        "receipt_url": receipt_url,
    }


async def _hydrate_payment_history_receipts(rows: List[Dict[str, Any]]) -> None:
    try:
        settings = get_settings()
    except Exception:
        return
    secret_key = str(getattr(settings, "stripe_private_key", "") or "").strip()
    if not secret_key:
        return

    receipt_cache: Dict[str, Dict[str, Optional[str]]] = {}
    for row in rows:
        if (
            str(row.get("receipt_url") or "").strip()
            and str(row.get("receipt_number") or "").strip()
        ):
            continue
        payment_intent_id = str(row.get("payment_intent_id") or "").strip()
        if not payment_intent_id:
            continue
        if str(row.get("payment_provider") or "").strip().lower() != "stripe":
            continue
        if str(row.get("payment_status") or "").strip().lower() != "succeeded":
            continue

        try:
            if payment_intent_id not in receipt_cache:
                receipt_cache[
                    payment_intent_id
                ] = await _stripe_receipt_fields_from_payment_intent(
                    payment_intent_id, secret_key
                )
            fields = receipt_cache[payment_intent_id]
        except Exception as exc:
            app_logger.warning(
                "Unable to hydrate payment receipt for "
                f"payment_intent_id={payment_intent_id}: {exc}"
            )
            continue

        if fields.get("stripe_charge_id") and not row.get("stripe_charge_id"):
            row["stripe_charge_id"] = fields["stripe_charge_id"]
        if fields.get("receipt_number") and not row.get("receipt_number"):
            row["receipt_number"] = fields["receipt_number"]
        if fields.get("receipt_url"):
            row["receipt_url"] = fields["receipt_url"]


def _to_payment_event(data: Dict[str, Any]) -> Dict[str, Any]:
    schedule = _build_enrollment_schedule(
        started_at_raw=data.get("started_at")
        or data.get("enrolled_at")
        or data.get("paid_at"),
        expires_at_raw=data.get("expires_at"),
        duration_months_raw=data.get("duration_months"),
    )
    paid_amount = data.get("paid_amount_thb")
    try:
        paid_amount = float(paid_amount) if paid_amount is not None else None
    except Exception:
        paid_amount = None

    payment_status = str(data.get("payment_status") or "").strip()
    event = {
        "order_id": data.get("order_id")
        or _build_payment_order_id(
            data.get("paid_at") or data.get("enrolled_at"),
            data.get("payment_intent_id"),
        ),
        "payment_provider": data.get("payment_provider") or "stripe",
        "payment_type": data.get("payment_type") or "promptpay",
        "payment_intent_id": data.get("payment_intent_id"),
        "stripe_charge_id": data.get("stripe_charge_id"),
        "receipt_number": data.get("receipt_number"),
        "receipt_url": data.get("receipt_url"),
        "payment_status": payment_status
        or ("succeeded" if paid_amount is not None else "active"),
        "paid_amount_thb": paid_amount,
        "paid_currency": data.get("paid_currency") or "THB",
        "billing_email": data.get("billing_email"),
        "plan_label": data.get("plan_label"),
        "duration_months": schedule["duration_months"],
        "paid_at": data.get("paid_at") or data.get("enrolled_at"),
        "started_at": schedule["started_at"],
        "expires_at": schedule["expires_at"],
    }
    for field_name in (
        "payment_success_email_status",
        "payment_success_email_job_id",
        "payment_success_email_sent_at",
    ):
        value = data.get(field_name)
        if value:
            event[field_name] = value
    return event


def _normalize_payment_history(enrollment: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = enrollment.get("payment_history")
    normalized_events: List[Dict[str, Any]] = []
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                normalized_events.append(_to_payment_event(event))

    # Backfill old records that do not have payment_history yet.
    if not normalized_events:
        payment_intent_id = str(enrollment.get("payment_intent_id") or "").strip()
        paid_amount = enrollment.get("paid_amount_thb")
        payment_status = str(enrollment.get("payment_status") or "").strip().lower()
        if (
            payment_intent_id
            or paid_amount is not None
            or payment_status == "succeeded"
        ):
            normalized_events.append(_to_payment_event(enrollment))
    return normalized_events


def _find_payment_event_by_intent(
    payment_events: List[Dict[str, Any]],
    payment_intent_id: Any,
) -> Optional[Dict[str, Any]]:
    target_intent_id = str(payment_intent_id or "").strip()
    if not target_intent_id:
        return None
    for event in payment_events:
        if not isinstance(event, dict):
            continue
        event_intent_id = str(event.get("payment_intent_id") or "").strip()
        if event_intent_id == target_intent_id:
            return event
    return None


def _extract_stripe_signature_parts(signature_header: str) -> Dict[str, List[str]]:
    parts: Dict[str, List[str]] = {}
    for item in str(signature_header or "").split(","):
        key, separator, value = item.partition("=")
        if not separator:
            continue
        parts.setdefault(key.strip(), []).append(value.strip())
    return parts


def _verify_stripe_webhook_signature(
    payload: bytes,
    signature_header: str,
    webhook_secret: str,
) -> None:
    signature_parts = _extract_stripe_signature_parts(signature_header)
    timestamps = signature_parts.get("t") or []
    signatures = signature_parts.get("v1") or []
    if not timestamps or not signatures:
        raise HTTPException(status_code=400, detail="Invalid Stripe signature")

    signed_payload = b".".join([timestamps[0].encode("utf-8"), payload])
    expected_signature = hmac.new(
        webhook_secret.encode("utf-8"),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()
    if not any(
        hmac.compare_digest(expected_signature, signature) for signature in signatures
    ):
        raise HTTPException(status_code=400, detail="Invalid Stripe signature")


def _payment_amount_thb_from_intent(intent: Dict[str, Any]) -> Optional[float]:
    amount_received_raw = intent.get("amount_received")
    amount_raw = intent.get("amount")
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
        created_ts = int(intent.get("created"))
        if created_ts > 0:
            paid_at = datetime.fromtimestamp(created_ts, timezone.utc).isoformat()
    except Exception:
        pass
    return paid_at


async def _enqueue_payment_success_email(
    *,
    enrollment_id: str,
    user_id: str,
    course_id: str,
    course_name: str,
    enrollment_data: Dict[str, Any],
) -> Optional[str]:
    to_email = str(enrollment_data.get("billing_email") or "").strip()
    if not to_email:
        app_logger.warning(
            "Skipping payment success email: missing billing email for "
            f"payment_intent_id={enrollment_data.get('payment_intent_id')}"
        )
        return None
    if QueueClient is None or get_queue_settings is None:
        app_logger.warning("Skipping payment success email: queue package unavailable")
        return None

    settings = get_queue_settings()
    client = QueueClient(settings.redis_url)
    try:
        job = await client.enqueue(
            "notification",
            "payment_success_email",
            {
                "payment_intent_id": enrollment_data.get("payment_intent_id"),
                "user_id": user_id,
                "course_id": course_id,
                "enrollment_id": enrollment_id,
                "to_email": to_email,
                "course_name": course_name,
                "amount_thb": enrollment_data.get("paid_amount_thb"),
                "currency": enrollment_data.get("paid_currency") or "THB",
                "paid_at": enrollment_data.get("paid_at"),
                "expires_at": enrollment_data.get("expires_at"),
                "plan_label": enrollment_data.get("plan_label"),
            },
        )
        return job.id
    except Exception as exc:
        app_logger.error(
            "Failed to enqueue payment success email for "
            f"payment_intent_id={enrollment_data.get('payment_intent_id')}: {exc}"
        )
        return None
    finally:
        await client.close()


async def _complete_promptpay_payment(
    *,
    payment_intent_id: str,
    dynamodb_service,
    expected_user_id: Optional[str] = None,
    expected_course_id: Optional[str] = None,
) -> Dict[str, Any]:
    settings = get_settings()
    if not settings.stripe_private_key:
        raise HTTPException(
            status_code=500, detail="Stripe private key is not configured"
        )

    payment_intent_id = str(payment_intent_id or "").strip()
    if not payment_intent_id:
        raise HTTPException(status_code=400, detail="payment_intent_id is required")

    intent = await _stripe_request(
        method="GET",
        path=f"/payment_intents/{payment_intent_id}",
        secret_key=settings.stripe_private_key,
        params={"expand[]": "latest_charge"},
    )
    metadata = (
        intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    )
    user_id = str(metadata.get("user_id") or expected_user_id or "").strip()
    course_id = str(metadata.get("course_id") or expected_course_id or "").strip()
    if not user_id or not course_id:
        raise HTTPException(
            status_code=400,
            detail="Payment is missing required user or course metadata",
        )
    if expected_user_id and user_id != str(expected_user_id).strip():
        raise HTTPException(
            status_code=400, detail="Payment does not belong to this user"
        )
    if expected_course_id and course_id != str(expected_course_id).strip():
        raise HTTPException(
            status_code=400, detail="Payment does not belong to this course"
        )

    payment_status = str(intent.get("status") or "").strip()
    if payment_status != "succeeded":
        return {
            "payment_intent_id": payment_intent_id,
            "payment_status": payment_status or "unknown",
            "enrolled": False,
            "message": "Payment is not completed yet",
        }

    existing_enrollment_with_schedule = await _get_existing_enrollment_with_schedule(
        dynamodb_service=dynamodb_service,
        user_id=user_id,
        course_id=course_id,
    )
    paid_at = _paid_at_from_intent(intent)
    latest_charge = _latest_charge_from_intent(intent)
    stripe_charge_id = str(latest_charge.get("id") or "").strip() or None
    receipt_number = str(latest_charge.get("receipt_number") or "").strip() or None
    receipt_url = str(latest_charge.get("receipt_url") or "").strip() or None
    duration_months = _parse_positive_int(metadata.get("duration_months"))
    schedule = _build_enrollment_schedule(
        started_at_raw=paid_at,
        expires_at_raw=None,
        duration_months_raw=duration_months,
    )
    enrollment_data = {
        "progress": 0,
        "completed_quizzes": 0,
        "total_quizzes": 0,
        "completed_questions": 0,
        "total_questions": 0,
        "last_activity": "เพิ่งชำระเงินและเข้าร่วม",
        "enrollment_source": "payment",
        "order_id": _build_payment_order_id(paid_at, payment_intent_id),
        "payment_provider": "stripe",
        "payment_type": "promptpay",
        "payment_intent_id": payment_intent_id,
        "stripe_charge_id": stripe_charge_id,
        "receipt_number": receipt_number,
        "receipt_url": receipt_url,
        "payment_status": payment_status,
        "paid_amount_thb": _payment_amount_thb_from_intent(intent),
        "paid_currency": str(intent.get("currency") or "THB").upper(),
        "billing_email": str(intent.get("receipt_email") or "").strip(),
        "plan_label": str(metadata.get("plan_label") or "").strip(),
        "duration_months": duration_months,
        "paid_at": paid_at,
        "started_at": schedule["started_at"],
        "expires_at": schedule["expires_at"],
    }

    if existing_enrollment_with_schedule:
        existing_enrollment = existing_enrollment_with_schedule["enrollment"]
        enrollment_id = str(existing_enrollment.get("enrollment_id") or "").strip()
        if not enrollment_id:
            raise HTTPException(
                status_code=500, detail="Existing enrollment is missing enrollment_id"
            )

        payment_history = _normalize_payment_history(existing_enrollment)
        existing_payment_event = _find_payment_event_by_intent(
            payment_events=payment_history,
            payment_intent_id=payment_intent_id,
        )
        if existing_payment_event:
            return {
                "payment_intent_id": payment_intent_id,
                "order_id": existing_payment_event.get("order_id")
                or _build_payment_order_id(
                    existing_payment_event.get("paid_at"), payment_intent_id
                ),
                "receipt_url": existing_payment_event.get("receipt_url"),
                "payment_status": payment_status,
                "enrolled": True,
                "enrollment_id": enrollment_id,
                "message": "Payment already confirmed for this enrollment",
            }

        current_schedule = existing_enrollment_with_schedule.get("schedule") or {}
        current_expires_at = _parse_iso_datetime(current_schedule.get("expires_at"))
        paid_at_dt = _parse_iso_datetime(paid_at) or datetime.utcnow()
        renewal_start_dt = paid_at_dt
        if current_expires_at and current_expires_at > paid_at_dt:
            renewal_start_dt = current_expires_at

        renewal_schedule = _build_enrollment_schedule(
            started_at_raw=renewal_start_dt.isoformat(),
            expires_at_raw=None,
            duration_months_raw=duration_months,
        )
        enrollment_data["started_at"] = renewal_schedule["started_at"]
        enrollment_data["expires_at"] = renewal_schedule["expires_at"]
        payment_event = _to_payment_event(enrollment_data)
        payment_history.append(payment_event)
        renewal_updates = {
            "status": "active",
            "order_id": enrollment_data["order_id"],
            "payment_provider": enrollment_data["payment_provider"],
            "payment_type": enrollment_data["payment_type"],
            "payment_intent_id": enrollment_data["payment_intent_id"],
            "stripe_charge_id": enrollment_data["stripe_charge_id"],
            "receipt_number": enrollment_data["receipt_number"],
            "receipt_url": enrollment_data["receipt_url"],
            "payment_status": enrollment_data["payment_status"],
            "paid_amount_thb": enrollment_data["paid_amount_thb"],
            "paid_currency": enrollment_data["paid_currency"],
            "billing_email": enrollment_data["billing_email"],
            "plan_label": enrollment_data["plan_label"],
            "duration_months": enrollment_data["duration_months"],
            "paid_at": enrollment_data["paid_at"],
            "started_at": enrollment_data["started_at"],
            "expires_at": enrollment_data["expires_at"],
            "payment_history": payment_history,
            "last_activity": "ต่ออายุคอร์สแล้ว",
        }
        success = await dynamodb_service.update_enrollment(
            enrollment_id, renewal_updates
        )
        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to renew existing enrollment"
            )
        return {
            "payment_intent_id": payment_intent_id,
            "order_id": payment_event.get("order_id"),
            "receipt_url": payment_event.get("receipt_url"),
            "payment_status": payment_status,
            "enrolled": True,
            "enrollment_id": enrollment_id,
            "message": "Payment verified and enrollment renewed",
        }

    payment_event = _to_payment_event(enrollment_data)
    enrollment_id = await dynamodb_service.enroll_user_in_course(
        user_id=user_id,
        course_id=course_id,
        enrollment_data={**enrollment_data, "payment_history": [payment_event]},
    )
    return {
        "payment_intent_id": payment_intent_id,
        "order_id": payment_event.get("order_id"),
        "receipt_url": payment_event.get("receipt_url"),
        "payment_status": payment_status,
        "enrolled": True,
        "enrollment_id": enrollment_id,
        "message": "Payment verified and enrollment completed",
    }


async def _set_quiz_gen_progress(
    job_id: Optional[str], payload: Dict[str, Any]
) -> None:
    if not job_id:
        return
    async with QUIZ_GEN_PROGRESS_LOCK:
        current = QUIZ_GEN_PROGRESS.get(job_id, {})
        merged = {**current, **payload, "updated_at": datetime.utcnow().isoformat()}
        QUIZ_GEN_PROGRESS[job_id] = merged


async def _advance_quiz_gen_progress(job_id: Optional[str], delta: int = 1) -> None:
    if not job_id:
        return
    async with QUIZ_GEN_PROGRESS_LOCK:
        current = QUIZ_GEN_PROGRESS.get(job_id, {})
        total = int(current.get("total") or 0)
        completed = int(current.get("completed") or 0) + max(0, int(delta))
        if total > 0:
            completed = min(completed, total)
            percent = int(round((completed / total) * 100))
        else:
            percent = 0
        current.update(
            {
                "completed": completed,
                "percent": max(0, min(100, percent)),
                "updated_at": datetime.utcnow().isoformat(),
            }
        )
        QUIZ_GEN_PROGRESS[job_id] = current


async def _append_quiz_gen_question(
    job_id: Optional[str],
    question: Dict[str, Any],
    set_index: int,
    question_index: int,
) -> None:
    if not job_id:
        return
    async with QUIZ_GEN_PROGRESS_LOCK:
        current = QUIZ_GEN_PROGRESS.get(job_id, {})
        generated = current.get("generated_questions")
        if not isinstance(generated, list):
            generated = []
        progress_key = f"{set_index}-{question_index}"
        if any(
            item.get("progress_key") == progress_key
            for item in generated
            if isinstance(item, dict)
        ):
            current["generated_questions"] = generated
            QUIZ_GEN_PROGRESS[job_id] = current
            return
        generated.append(
            {
                "progress_key": progress_key,
                "set_index": set_index,
                "question_index": question_index,
                "context": question.get("context"),
                "question": question.get("question"),
                "choices": question.get("choices"),
                "correct_answer": question.get("correct_answer"),
                "explanation": question.get("explanation"),
                "difficulty": question.get("difficulty"),
                "topic_tag": question.get("topic_tag"),
                "verification_status": question.get("verification_status"),
                "verification": question.get("verification"),
                "template_original": question.get("template_original"),
            }
        )
        current["generated_questions"] = generated
        current["updated_at"] = datetime.utcnow().isoformat()
        QUIZ_GEN_PROGRESS[job_id] = current


# Helper function to create filtered PDF with selected pages
async def create_filtered_pdf(
    original_pdf_path: Path, selected_pages: List[int]
) -> Path:
    """Create a new PDF containing only the selected pages from the original PDF."""
    try:
        app_logger.info(
            f"Creating filtered PDF from {original_pdf_path} with pages: {selected_pages}"
        )

        # Open the original PDF
        pdf_document = fitz.open(str(original_pdf_path))
        total_pages = pdf_document.page_count

        app_logger.info(f"Original PDF has {total_pages} pages")

        # Validate selected pages
        valid_pages = [p for p in selected_pages if 1 <= p <= total_pages]
        if not valid_pages:
            raise ValueError(f"No valid pages in selection: {selected_pages}")

        app_logger.info(f"Valid pages to extract: {valid_pages}")

        # Create new PDF with selected pages
        new_pdf = fitz.open()  # Create new empty PDF

        for page_num in sorted(valid_pages):  # Sort to maintain order
            # PyMuPDF uses 0-based indexing
            page_index = page_num - 1

            # Insert the page into the new PDF
            new_pdf.insert_pdf(pdf_document, from_page=page_index, to_page=page_index)
            app_logger.info(f"Added page {page_num} to filtered PDF")

        # Generate output file path
        output_path = (
            original_pdf_path.parent
            / f"filtered_{original_pdf_path.stem}_{'-'.join(map(str, valid_pages))}.pdf"
        )

        # Save the filtered PDF
        new_pdf.save(str(output_path))
        new_pdf.close()
        pdf_document.close()

        app_logger.info(f"Filtered PDF created successfully: {output_path}")
        return output_path

    except Exception as e:
        app_logger.error(f"Error creating filtered PDF: {e}")
        raise ValueError(f"Failed to create filtered PDF: {str(e)}")


# Dependency to get file service
async def get_file_service() -> FileService:
    return FileService()


# Dependency to get parsing service
async def get_parsing_service() -> GeminiOCRService:
    return GeminiOCRService()


# Dependency to get DynamoDB service
async def get_dynamodb_service():
    return get_db_service()


# Dependency to get chat/LLM service
async def get_chat_service() -> ChatService:
    return ChatService()


async def get_quiz_augment_service() -> QuizAugmentService:
    return QuizAugmentService()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Test Gemini OCR service initialization
        parsing_service = GeminiOCRService()
        gemini_status = "healthy"  # Gemini doesn't have a health endpoint

        return HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=0.0,  # Implement actual uptime tracking
            gemini_status=gemini_status,
        )
    except Exception as e:
        app_logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.get("/legal/terms-of-use", response_model=LegalDocumentResponse)
async def get_terms_of_use():
    """Get Terms of Use (Thai summary)."""
    return LegalDocumentResponse(
        document_name="เงื่อนไขการใช้งาน (Terms of Use)",
        version=LEGAL_DOC_VERSION,
        last_updated=LEGAL_DOC_LAST_UPDATED,
        summary="เอกสารนี้กำหนดเงื่อนไขการใช้งาน TEWMai แพลตฟอร์มฝึกโจทย์ ข้อสอบจำลอง วิเคราะห์ผล และผู้ช่วย AI สำหรับการเรียนรู้",
        sections=[
            LegalSection(
                heading="การยอมรับเงื่อนไข",
                details=[
                    "เมื่อผู้ใช้เข้าเว็บไซต์ ลงทะเบียน ชำระเงิน หรือใช้ฟีเจอร์ใด ๆ ของ TEWMai ถือว่าผู้ใช้ได้อ่าน เข้าใจ และยอมรับเงื่อนไขการใช้งานฉบับนี้แล้ว",
                    "หากผู้ใช้ไม่ยอมรับเงื่อนไขข้อใดข้อหนึ่ง ควรงดใช้บริการหรือหยุดใช้งานบัญชีจนกว่าจะเข้าใจรายละเอียดครบถ้วน",
                    "ในกรณีที่ผู้ใช้เป็นผู้เยาว์ ผู้ปกครองหรือผู้แทนโดยชอบธรรมควรรับทราบและยินยอมต่อการใช้งานบริการ",
                ],
            ),
            LegalSection(
                heading="รายละเอียดบริการ",
                details=[
                    "TEWMai ให้บริการแบบฝึกหัด ข้อสอบจำลอง บทเรียน รายงานวิเคราะห์คะแนน และผู้ช่วย AI เพื่ออธิบายแนวคิดในการทำโจทย์",
                    "ระบบอาจมีฟีเจอร์สำหรับอัปโหลดภาพโจทย์หรือวิธีทำ เพื่อให้ AI ช่วยอ่าน วิเคราะห์ และแนะนำแนวทางการเรียนรู้",
                    "บางฟีเจอร์ คอร์ส หรือรายงานเชิงลึกอาจเปิดให้ใช้เฉพาะผู้ใช้ที่สมัครคอร์ส ชำระเงิน หรือมีสิทธิ์เข้าถึงตามแพ็กเกจที่กำหนด",
                    "บริการมีเป้าหมายเพื่อสนับสนุนการเรียนรู้ ไม่ใช่การรับประกันผลสอบ คะแนนสอบ หรือการเข้าเรียนในสถาบันใดสถาบันหนึ่ง",
                ],
            ),
            LegalSection(
                heading="บัญชีผู้ใช้และสิทธิ์การเข้าใช้งาน",
                details=[
                    "ผู้ใช้ต้องให้ข้อมูลบัญชีที่ถูกต้อง เป็นปัจจุบัน และไม่แอบอ้างเป็นบุคคลอื่น",
                    "ผู้ใช้ต้องรักษารหัสผ่าน ลิงก์เข้าสู่ระบบ และอุปกรณ์ที่ใช้เข้าใช้งานให้ปลอดภัย หากพบการใช้งานผิดปกติควรแจ้งผู้ให้บริการโดยเร็ว",
                    "สิทธิ์การเข้าถึงคอร์สหรือฟีเจอร์เป็นสิทธิ์เฉพาะบัญชี ไม่ควรขาย ให้เช่า โอน หรือแบ่งปันบัญชีให้ผู้อื่นใช้งานโดยไม่ได้รับอนุญาต",
                ],
            ),
            LegalSection(
                heading="หน้าที่ของผู้ใช้งาน",
                details=[
                    "ผู้ใช้ต้องใช้บริการอย่างสุจริต ถูกกฎหมาย และไม่กระทำการที่รบกวนการทำงานของระบบหรือผู้ใช้อื่น",
                    "ผู้ใช้รับผิดชอบต่อความถูกต้อง ความเหมาะสม และสิทธิ์ในการใช้ข้อมูล รูปภาพ ไฟล์ หรือข้อความที่อัปโหลดเข้าสู่ระบบ",
                    "ผู้ใช้ไม่ควรอัปโหลดข้อมูลส่วนบุคคลที่ไม่จำเป็น ข้อมูลของผู้อื่นโดยไม่ได้รับอนุญาต หรือเนื้อหาที่ละเมิดกฎหมายและสิทธิของบุคคลภายนอก",
                ],
            ),
            LegalSection(
                heading="การใช้ผู้ช่วย AI และผลลัพธ์การเรียน",
                details=[
                    "คำตอบ คำอธิบาย และคำแนะนำจาก AI เป็นเครื่องมือช่วยเรียนรู้ ผู้ใช้ควรใช้วิจารณญาณ ตรวจสอบความถูกต้อง และปรึกษาครูหรือผู้ปกครองเมื่อจำเป็น",
                    "AI อาจตีความโจทย์ ภาพลายมือ หรือบริบทผิดพลาดได้ โดยเฉพาะภาพที่ไม่ชัดเจน ข้อมูลไม่ครบ หรือโจทย์ที่มีหลายวิธีคิด",
                    "ผู้ให้บริการอาจนำข้อมูลการใช้งานที่เหมาะสมไปใช้ปรับปรุงคุณภาพระบบ การตรวจจับข้อผิดพลาด และประสบการณ์การเรียน โดยเป็นไปตามนโยบายความเป็นส่วนตัว",
                ],
            ),
            LegalSection(
                heading="ข้อมูล การอัปโหลด และความเป็นส่วนตัว",
                details=[
                    "ผู้ใช้ยังคงเป็นเจ้าของเนื้อหา ไฟล์ ภาพโจทย์ และข้อมูลการเรียนที่ตนเองอัปโหลดหรือสร้างขึ้นผ่านระบบ",
                    "TEWMai จะเข้าถึงและประมวลผลข้อมูลเท่าที่จำเป็นเพื่อให้บริการ เช่น การตรวจคำตอบ การวิเคราะห์คะแนน การแสดงประวัติการเรียน และการช่วยเหลือผ่าน AI",
                    "รายละเอียดการเก็บ ใช้ เปิดเผย เก็บรักษา และลบข้อมูลส่วนบุคคลระบุไว้ในนโยบายความเป็นส่วนตัวของเรา",
                ],
            ),
            LegalSection(
                heading="เงื่อนไขการชำระเงิน คอร์ส และแพ็กเกจ",
                details=[
                    "ราคา ระยะเวลาเข้าถึง เนื้อหาคอร์ส ฟีเจอร์ และสิทธิ์การใช้งานเป็นไปตามข้อมูลที่แสดงในหน้าคอร์สหรือหน้าชำระเงิน ณ เวลาที่ผู้ใช้สมัคร",
                    "ผู้ให้บริการอาจปรับราคา แพ็กเกจ โปรโมชัน หรือโครงสร้างฟีเจอร์ในอนาคต โดยการเปลี่ยนแปลงจะไม่มีผลย้อนหลังต่อสิทธิ์ที่ผู้ใช้ชำระเงินและได้รับยืนยันแล้ว เว้นแต่ระบุไว้เป็นอย่างอื่น",
                    "หากมีการต่ออายุ อัปเกรด หรือเปลี่ยนแพ็กเกจ ระบบอาจคำนวณสิทธิ์หรือระยะเวลาใช้งานตามเงื่อนไขที่ประกาศในขณะทำรายการ",
                ],
            ),
            LegalSection(
                heading="การคืนเงินและการยกเลิกสิทธิ์",
                details=[
                    "นโยบายการคืนเงินหรือยกเลิกสิทธิ์เป็นไปตามเงื่อนไขที่แสดงในหน้าชำระเงิน ประกาศของระบบ และข้อกำหนดของผู้ให้บริการชำระเงินที่เกี่ยวข้อง",
                    "ผู้ใช้ควรตรวจสอบชื่อคอร์ส ราคา ระยะเวลา และรายละเอียดสิทธิ์ก่อนยืนยันการชำระเงิน",
                    "หากพบการเรียกเก็บเงินผิดพลาดหรือเข้าถึงคอร์สไม่ได้หลังชำระเงิน ผู้ใช้ควรติดต่อทีมสนับสนุนพร้อมหลักฐานการชำระเงินเพื่อให้ตรวจสอบ",
                ],
            ),
            LegalSection(
                heading="การใช้งานที่ห้าม",
                details=[
                    "ห้ามใช้ระบบเพื่อกระทำการผิดกฎหมาย ฉ้อโกง คุกคาม ละเมิดสิทธิผู้อื่น หรือเผยแพร่เนื้อหาที่ไม่เหมาะสม",
                    "ห้ามใช้บอท สคริปต์ การขูดข้อมูล หรือวิธีอัตโนมัติอื่นใดเพื่อดึงข้อมูลจำนวนมาก หลีกเลี่ยงข้อจำกัด หรือสร้างภาระเกินสมควรต่อระบบ",
                    "ห้ามพยายามเจาะระบบ ข้ามมาตรการรักษาความปลอดภัย แก้ไข ดัดแปลง ทำ reverse engineer หรือเข้าถึงข้อมูลที่ไม่ได้รับอนุญาต",
                    "การละเมิดเงื่อนไขอาจทำให้ถูกจำกัด ระงับ หรือยกเลิกบัญชี รวมถึงอาจดำเนินการตามกฎหมายหากมีความจำเป็น",
                ],
            ),
            LegalSection(
                heading="ความพร้อมใช้งานและการเปลี่ยนแปลงบริการ",
                details=[
                    "ผู้ให้บริการพยายามดูแลให้ระบบใช้งานได้ต่อเนื่อง แต่ไม่รับประกันว่าบริการจะไม่มีข้อผิดพลาด หยุดชะงัก หรือพร้อมใช้งานตลอดเวลา",
                    "ระบบอาจหยุดให้บริการชั่วคราวเนื่องจากการบำรุงรักษา การอัปเดต เหตุขัดข้องทางเทคนิค หรือปัจจัยภายนอก เช่น ผู้ให้บริการคลาวด์หรือระบบชำระเงิน",
                    "ผู้ให้บริการอาจปรับปรุง แก้ไข เพิ่ม ลด หรือยุติบางฟีเจอร์เมื่อจำเป็น โดยจะพยายามสื่อสารการเปลี่ยนแปลงที่มีผลสำคัญต่อผู้ใช้",
                ],
            ),
            LegalSection(
                heading="ข้อจำกัดความรับผิด",
                details=[
                    "TEWMai ให้บริการตามสภาพที่มีอยู่และตามขอบเขตที่ระบบรองรับ ผู้ใช้ยอมรับความเสี่ยงในการใช้ข้อมูล คำแนะนำ และผลวิเคราะห์เพื่อประกอบการเรียนรู้",
                    "ผู้ให้บริการไม่รับผิดชอบต่อความเสียหายทางอ้อม การสูญเสียโอกาส ผลสอบไม่เป็นไปตามคาด หรือความเสียหายที่เกิดจากการใช้งานผิดวัตถุประสงค์",
                    "ในขอบเขตสูงสุดที่กฎหมายอนุญาต ความรับผิดของผู้ให้บริการจะจำกัดตามมูลค่าบริการที่เกี่ยวข้องกับเหตุการณ์นั้น",
                ],
            ),
            LegalSection(
                heading="การเปลี่ยนแปลงเงื่อนไข",
                details=[
                    "ผู้ให้บริการอาจปรับปรุงเงื่อนไขการใช้งานเป็นครั้งคราว เพื่อให้สอดคล้องกับฟีเจอร์ใหม่ ข้อกำหนดทางกฎหมาย หรือวิธีดำเนินงานของระบบ",
                    "เมื่อมีการเปลี่ยนแปลงที่สำคัญ ระบบอาจแจ้งผ่านหน้าเว็บไซต์ แอป อีเมล หรือช่องทางที่เหมาะสม พร้อมปรับวันที่อัปเดตของเอกสาร",
                    "การใช้งานบริการต่อหลังจากเงื่อนไขใหม่มีผล ถือว่าผู้ใช้ยอมรับเงื่อนไขฉบับที่ปรับปรุงแล้ว",
                ],
            ),
            LegalSection(
                heading="ติดต่อเรา",
                details=[
                    "หากมีคำถามเกี่ยวกับเงื่อนไขการใช้งาน การชำระเงิน สิทธิ์การเข้าถึง หรือปัญหาเกี่ยวกับบัญชี สามารถติดต่อทีมสนับสนุนได้ที่ support@tewmai.com, LINE @tewmai หรือ Facebook: TEWMai - ติวอัจฉริยะด้วย AI",
                    "เพื่อให้ตรวจสอบได้รวดเร็ว โปรดระบุอีเมลบัญชี ชื่อคอร์ส หลักฐานการชำระเงิน หรือรายละเอียดปัญหาที่เกี่ยวข้องเมื่อส่งคำขอ",
                ],
            ),
        ],
        contact_email="support@tewmai.com",
    )


@router.get("/legal/privacy-policy", response_model=LegalDocumentResponse)
async def get_privacy_policy():
    """Get Privacy Policy (Thai summary)."""
    return LegalDocumentResponse(
        document_name="นโยบายความเป็นส่วนตัว (Privacy Policy)",
        version=LEGAL_DOC_VERSION,
        last_updated=LEGAL_DOC_LAST_UPDATED,
        summary="เอกสารนี้อธิบายวิธีที่ TEWMai เก็บ ใช้ เปิดเผย เก็บรักษา และคุ้มครองข้อมูลส่วนบุคคลของผู้เรียน ผู้ปกครอง และผู้ใช้งานระบบ",
        sections=[
            LegalSection(
                heading="ข้อมูลที่เราเก็บ",
                details=[
                    "ข้อมูลบัญชี เช่น ชื่อ อีเมล เบอร์โทรศัพท์ (ถ้ามี) รูปโปรไฟล์ ช่องทางเข้าสู่ระบบ และข้อมูลที่ใช้ระบุตัวตนของบัญชี",
                    "ข้อมูลการสมัครและการชำระเงิน เช่น คอร์สที่สมัคร สถานะการชำระเงิน ประวัติการต่ออายุ ใบเสร็จหรือหลักฐานที่เกี่ยวข้อง โดยอาจประมวลผลผ่านผู้ให้บริการชำระเงินภายนอก",
                    "ข้อมูลการเรียน เช่น คะแนน คำตอบ เวลาใช้งาน จำนวนครั้งที่ทำโจทย์ ความแม่นยำรายหัวข้อ ประวัติการเรียน และรายงานวิเคราะห์ผล",
                    "ข้อมูลทางเทคนิค เช่น ประเภทอุปกรณ์ เบราว์เซอร์ หมายเลข IP บันทึกการใช้งาน เหตุขัดข้อง และข้อมูลคุกกี้หรือเทคโนโลยีที่คล้ายกัน",
                ],
            ),
            LegalSection(
                heading="ข้อมูลจากภาพโจทย์ ไฟล์ และผู้ช่วย AI",
                details=[
                    "เมื่อผู้ใช้อัปโหลดภาพโจทย์ วิธีทำ ไฟล์ หรือข้อความ ระบบอาจประมวลผลข้อมูลดังกล่าวเพื่ออ่านโจทย์ ตรวจคำตอบ วิเคราะห์แนวคิด และสร้างคำแนะนำจาก AI",
                    "ข้อมูลที่ส่งให้ผู้ช่วย AI อาจประกอบด้วยข้อความที่ผู้ใช้พิมพ์ รูปภาพ คำตอบ คะแนน และบริบทการเรียนที่จำเป็นต่อการตอบคำถาม",
                    "ผู้ใช้ควรหลีกเลี่ยงการอัปโหลดข้อมูลส่วนบุคคลที่ไม่จำเป็น เช่น เลขบัตรประชาชน ที่อยู่ ข้อมูลสุขภาพ หรือข้อมูลของบุคคลอื่นที่ไม่ได้รับอนุญาต",
                    "เราใช้ข้อมูลดังกล่าวเพื่อให้บริการและปรับปรุงคุณภาพระบบ ไม่ได้ออกแบบมาเพื่อใช้เป็นเครื่องมือจัดเก็บข้อมูลอ่อนไหวหรือเอกสารสำคัญส่วนตัว",
                ],
            ),
            LegalSection(
                heading="วัตถุประสงค์การใช้ข้อมูล",
                details=[
                    "ให้บริการบัญชีผู้ใช้ คอร์ส แบบฝึกหัด ข้อสอบจำลอง ผู้ช่วย AI รายงานผล และฟีเจอร์ที่ผู้ใช้ร้องขอ",
                    "วิเคราะห์พัฒนาการ จุดแข็ง จุดที่ควรฝึกเพิ่ม และปรับคำแนะนำการเรียนให้เหมาะสมกับผู้ใช้แต่ละคน",
                    "ตรวจสอบการชำระเงิน จัดการสิทธิ์เข้าถึงคอร์ส ให้การสนับสนุนลูกค้า และแก้ไขปัญหาทางเทคนิค",
                    "รักษาความปลอดภัย ป้องกันการใช้งานผิดปกติ ตรวจจับการละเมิดเงื่อนไข และปรับปรุงเสถียรภาพของระบบ",
                    "พัฒนาคุณภาพโมเดล กระบวนการตรวจโจทย์ และประสบการณ์ใช้งาน โดยใช้ข้อมูลเท่าที่จำเป็นและลดการระบุตัวตนเมื่อเหมาะสม",
                ],
            ),
            LegalSection(
                heading="การเปิดเผยข้อมูล",
                details=[
                    "เราไม่ขายหรือให้เช่าข้อมูลส่วนบุคคลของผู้ใช้แก่บุคคลที่สาม",
                    "เราอาจเปิดเผยข้อมูลเท่าที่จำเป็นต่อผู้ให้บริการโครงสร้างพื้นฐาน ระบบยืนยันตัวตน ระบบชำระเงิน ระบบวิเคราะห์ข้อผิดพลาด หรือบริการ AI ที่ช่วยให้ระบบทำงานได้",
                    "ผู้ให้บริการภายนอกที่เกี่ยวข้องจะได้รับข้อมูลเฉพาะส่วนที่จำเป็นต่อการให้บริการ และต้องปฏิบัติตามมาตรการรักษาความลับและความปลอดภัยที่เหมาะสม",
                    "เราอาจเปิดเผยข้อมูลเมื่อกฎหมาย คำสั่งหน่วยงานรัฐ หรือกระบวนการทางกฎหมายกำหนด หรือเมื่อจำเป็นเพื่อปกป้องสิทธิ ความปลอดภัย และความมั่นคงของระบบ",
                ],
            ),
            LegalSection(
                heading="ความปลอดภัยของข้อมูล",
                details=[
                    "เราใช้มาตรการทางเทคนิคและองค์กรที่เหมาะสม เช่น การเชื่อมต่อที่ปลอดภัย การจำกัดสิทธิ์เข้าถึง และการตรวจสอบระบบ เพื่อลดความเสี่ยงจากการเข้าถึงโดยไม่ได้รับอนุญาต",
                    "ข้อมูลสำคัญบางประเภท เช่น token หรือ credential ที่ใช้เชื่อมต่อบริการ จะถูกจัดเก็บและควบคุมตามแนวทางความปลอดภัยของระบบ",
                    "แม้เราจะพยายามปกป้องข้อมูลอย่างเหมาะสม แต่ไม่มีระบบออนไลน์ใดปลอดภัยสมบูรณ์ ผู้ใช้ควรรักษารหัสผ่าน อุปกรณ์ และบัญชีอีเมลของตนเองให้ปลอดภัยด้วย",
                ],
            ),
            LegalSection(
                heading="การเก็บรักษาและการลบข้อมูล",
                details=[
                    "เราจะเก็บข้อมูลส่วนบุคคลเท่าที่จำเป็นต่อการให้บริการ การปฏิบัติตามกฎหมาย การบัญชี การตรวจสอบข้อพิพาท และการรักษาความปลอดภัยของระบบ",
                    "ข้อมูลการเรียนและประวัติการใช้งานอาจถูกเก็บไว้ตลอดระยะเวลาที่บัญชียังใช้งาน เพื่อให้ผู้ใช้ดูพัฒนาการย้อนหลังและใช้รายงานวิเคราะห์ได้ต่อเนื่อง",
                    "ข้อมูลชั่วคราวจากการประมวลผล เช่น ไฟล์หรือข้อมูลที่ใช้สำหรับอ่านภาพและวิเคราะห์โจทย์ อาจถูกลบหรือทำให้ลดการระบุตัวตนเมื่อหมดความจำเป็น",
                    "ผู้ใช้สามารถติดต่อเพื่อขอลบหรือปิดบัญชีได้ โดยบางข้อมูลอาจยังต้องเก็บไว้ตามที่กฎหมายกำหนดหรือเพื่อป้องกันการทุจริตและข้อพิพาท",
                ],
            ),
            LegalSection(
                heading="สิทธิของเจ้าของข้อมูล",
                details=[
                    "ผู้ใช้สามารถขอเข้าถึง สำเนา แก้ไข ลบ หรือจำกัดการประมวลผลข้อมูลส่วนบุคคลของตนเองได้ตามขอบเขตที่กฎหมายคุ้มครองข้อมูลส่วนบุคคลกำหนด",
                    "ผู้ใช้สามารถถอนความยินยอมหรือคัดค้านการประมวลผลบางประเภทได้ หากการดำเนินการนั้นอยู่บนฐานความยินยอมหรือเข้าข่ายที่กฎหมายอนุญาต",
                    "การลบหรือจำกัดข้อมูลบางอย่างอาจส่งผลให้ไม่สามารถใช้ฟีเจอร์บางส่วนได้ เช่น รายงานย้อนหลัง การวิเคราะห์ผล หรือการยืนยันสิทธิ์คอร์ส",
                    "เราจะพิจารณาคำขอตามขั้นตอนที่เหมาะสม และอาจขอข้อมูลเพิ่มเติมเพื่อยืนยันตัวตนก่อนดำเนินการ",
                ],
            ),
            LegalSection(
                heading="ข้อมูลของผู้เรียนและผู้เยาว์",
                details=[
                    "บริการของเราออกแบบเพื่อสนับสนุนการเรียนรู้ของผู้เรียน ซึ่งอาจรวมถึงผู้เยาว์ จึงควรมีผู้ปกครองหรือผู้ดูแลรับทราบการสมัครและการใช้งาน",
                    "หากผู้ปกครองพบว่าผู้เยาว์ให้ข้อมูลส่วนบุคคลโดยไม่ได้รับอนุญาต สามารถติดต่อเราเพื่อขอให้ตรวจสอบ แก้ไข หรือลบข้อมูลที่เกี่ยวข้อง",
                    "เราพยายามจำกัดการเก็บข้อมูลของผู้เรียนเท่าที่จำเป็นต่อการให้บริการด้านการเรียน การวิเคราะห์ผล และการดูแลความปลอดภัยของบัญชี",
                ],
            ),
            LegalSection(
                heading="คุกกี้ บันทึกการใช้งาน และการวิเคราะห์ระบบ",
                details=[
                    "เว็บไซต์หรือแอปอาจใช้คุกกี้ local storage หรือเทคโนโลยีที่คล้ายกันเพื่อจดจำสถานะการเข้าสู่ระบบ การตั้งค่า และปรับปรุงประสบการณ์ใช้งาน",
                    "เราอาจเก็บบันทึกการใช้งาน เหตุขัดข้อง และข้อมูลประสิทธิภาพของระบบ เพื่อแก้ปัญหา ป้องกันการใช้งานผิดปกติ และพัฒนาคุณภาพบริการ",
                    "ข้อมูลเชิงวิเคราะห์ที่ใช้เพื่อปรับปรุงระบบจะถูกใช้ในขอบเขตที่เหมาะสม และเมื่อเป็นไปได้จะลดการระบุตัวตนของผู้ใช้",
                ],
            ),
            LegalSection(
                heading="การโอนหรือประมวลผลข้อมูลโดยผู้ให้บริการภายนอก",
                details=[
                    "บางบริการ เช่น คลาวด์ โครงสร้างพื้นฐาน ระบบ AI ระบบอีเมล หรือระบบชำระเงิน อาจตั้งอยู่หรือประมวลผลข้อมูลในต่างประเทศ",
                    "เมื่อมีการใช้ผู้ให้บริการภายนอก เราจะพิจารณามาตรการคุ้มครองข้อมูลที่เหมาะสมกับลักษณะข้อมูลและความเสี่ยงของการประมวลผล",
                    "การใช้งานบริการต่อถือว่าผู้ใช้รับทราบว่าอาจมีการประมวลผลข้อมูลผ่านระบบหรือผู้ให้บริการที่จำเป็นต่อการให้บริการของ TEWMai",
                ],
            ),
            LegalSection(
                heading="การเปลี่ยนแปลงนโยบายนี้",
                details=[
                    "เราอาจปรับปรุงนโยบายความเป็นส่วนตัวเป็นครั้งคราว เพื่อให้สอดคล้องกับฟีเจอร์ใหม่ วิธีดำเนินงาน หรือข้อกำหนดทางกฎหมาย",
                    "เมื่อมีการเปลี่ยนแปลงสำคัญ เราอาจแจ้งผ่านเว็บไซต์ แอป อีเมล หรือช่องทางที่เหมาะสม พร้อมปรับวันที่อัปเดตของเอกสาร",
                    "การใช้งานบริการต่อหลังจากนโยบายฉบับใหม่มีผล ถือว่าผู้ใช้รับทราบการเปลี่ยนแปลงตามขอบเขตที่กฎหมายอนุญาต",
                ],
            ),
            LegalSection(
                heading="ติดต่อเรา",
                details=[
                    "หากมีคำถามเกี่ยวกับนโยบายความเป็นส่วนตัว การใช้ข้อมูล หรือการใช้สิทธิของเจ้าของข้อมูล สามารถติดต่อได้ที่ support@tewmai.com, LINE @tewmai หรือ Facebook: TEWMai - ติวอัจฉริยะด้วย AI",
                    "เพื่อความปลอดภัย เราอาจขอข้อมูลเพื่อยืนยันตัวตนก่อนเปิดเผย แก้ไข หรือลบข้อมูลตามคำขอ",
                ],
            ),
        ],
        contact_email="support@tewmai.com",
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    document_type: DocumentTypeEnum = Form(DocumentTypeEnum.DOCUMENT),
    file_service: FileService = Depends(get_file_service),
):
    """
    Upload a file for OCR processing.

    - **file**: Image or PDF file to upload
    - **document_type**: Type of document (document/book/exam)
    """
    try:
        app_logger.info(f"Uploading file: {file.filename}")

        # Save the uploaded file
        document_id, file_path, metadata = await file_service.save_upload_file(file)

        # Generate file URL
        file_url = file_service.get_file_url(metadata["saved_filename"])

        return UploadResponse(
            document_id=document_id,
            filename=metadata["original_filename"],
            file_size=metadata["file_size"],
            mime_type=metadata["mime_type"],
            upload_url=file_url,
            message="File uploaded successfully",
        )

    except BaseAPIException:
        raise
    except Exception as e:
        app_logger.error(f"Unexpected error during file upload: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/courses/upload-image")
async def upload_course_image(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    file_service: FileService = Depends(get_file_service),
):
    """Upload a course cover image and return a public URL."""
    try:
        settings = get_settings()
        if not settings.use_s3_storage:
            raise HTTPException(status_code=501, detail="Supabase storage is disabled")

        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        content_type = str(file.content_type or "").lower()
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files are allowed")

        upload_metadata = await file_service.upload_course_image_to_s3(file, user_id)
        s3_key = str(upload_metadata.get("s3_key") or "").strip()
        bucket_name = str(upload_metadata.get("bucket_name") or "").strip()
        image_url = str(upload_metadata.get("image_url") or "").strip()
        if not s3_key or not image_url:
            raise HTTPException(
                status_code=500, detail="Failed to resolve uploaded image URL"
            )

        app_logger.info(
            f"Course cover image uploaded by {user_id}: {bucket_name}/{s3_key}"
        )
        return {
            "filename": upload_metadata.get("original_filename") or file.filename,
            "saved_filename": Path(s3_key).name,
            "image_url": image_url,
            "s3_key": s3_key,
            "bucket_name": bucket_name,
            "message": "Course image uploaded successfully",
        }
    except HTTPException:
        raise
    except BaseAPIException as e:
        app_logger.error(f"Course image upload failed for {user_id}: {e}")
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        app_logger.error(f"Unexpected error during course image upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload course image")


@router.post("/process-document/{document_id}", response_model=OCRResponse)
async def process_document(
    document_id: str,
    parse_request: OCRRequest,
    file_service: FileService = Depends(get_file_service),
    parsing_service: GeminiOCRService = Depends(get_parsing_service),
):
    """
    Process document parsing for an uploaded document.

    - **document_id**: ID of the uploaded document
    - **document_type**: Type of document being processed
    - **language**: Primary language of the document
    - **enhance_markdown**: Whether to enhance output with markdown formatting
    """
    try:
        app_logger.info(f"Processing document for document: {document_id}")

        # Find the uploaded file
        upload_dir = Path(get_settings().upload_folder)

        # This is a simplified approach - in production you'd want to store
        # document metadata in a database and retrieve the actual file path
        matching_files = list(upload_dir.glob(f"{document_id}*"))

        if not matching_files:
            raise HTTPException(
                status_code=404, detail=f"Document {document_id} not found"
            )

        file_path = matching_files[0]

        # Validate file format
        if not parsing_service.validate_file_format(file_path):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {parsing_service.get_supported_formats()}",
            )

        # Process document using Gemini OCR
        result = await parsing_service.parse_document(
            file_path=file_path,
            document_id=document_id,
            document_type=parse_request.document_type,
            language=parse_request.language,
            enhance_markdown=parse_request.enhance_markdown,
            extract_questions=parse_request.document_type == DocumentTypeEnum.EXAM,
        )

        return result

    except HTTPException:
        raise
    except BaseAPIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        app_logger.error(f"Unexpected error during document processing: {e}")
        raise HTTPException(status_code=500, detail="Document processing failed")


@router.post("/upload-and-process", response_model=OCRResponse)
async def upload_and_process(
    file: UploadFile = File(...),
    user_id: str = Form("anonymous", description="User identifier for future storage"),
    document_type: DocumentTypeEnum = Form(DocumentTypeEnum.DOCUMENT),
    language: str = Form("auto"),
    enhance_markdown: bool = Form(True),
    extract_questions: bool = Form(True),
    selected_pages: Optional[str] = Form(
        None, description="JSON array of page numbers to process (for PDFs)"
    ),
    course_id: Optional[str] = Form(
        None, description="Course ID to associate this document with"
    ),
    file_service: FileService = Depends(get_file_service),
    parsing_service: GeminiOCRService = Depends(get_parsing_service),
):
    """
    Upload a file and process document parsing locally (no S3 upload until submitted).

    - **file**: Image file to upload and process (PNG, JPG, JPEG, TIFF, BMP, GIF, WebP, PDF, DOCX)
    - **document_type**: Type of document (document/book/exam)
    - **language**: Primary language of the document (auto, en, th, etc.)
    - **enhance_markdown**: Whether to enhance output formatting (deprecated - now returns JSON)
    - **extract_questions**: Whether to extract questions and choices as JSON (for exams)

    Returns structured JSON content. File is kept locally until user submits.
    """
    try:
        app_logger.info(f"Upload and process file: {file.filename} for user: {user_id}")

        # Parse selected pages if provided
        page_numbers = None
        if selected_pages:
            try:
                import json

                page_numbers = json.loads(selected_pages)
                app_logger.info(f"Processing selected pages: {page_numbers}")
            except json.JSONDecodeError as e:
                app_logger.warning(
                    f"Invalid selected_pages JSON: {e}, processing whole document"
                )
                page_numbers = None

        # Save the uploaded file ONLY locally (no S3 upload)
        document_id, file_path, metadata = await file_service.save_upload_file(file)

        # Validate file format for Gemini OCR
        if file_path and not parsing_service.validate_file_format(file_path):
            # Clean up uploaded files
            await file_service.cleanup_local_file(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {parsing_service.get_supported_formats()}",
            )

        # Process document using Gemini OCR (requires local file)
        if not file_path:
            raise HTTPException(
                status_code=500,
                detail="File processing failed - no local file available",
            )

        result = await parsing_service.parse_document(
            file_path=file_path,
            document_id=document_id,
            document_type=document_type,
            language=language,
            enhance_markdown=enhance_markdown,
            extract_questions=extract_questions
            and document_type == DocumentTypeEnum.EXAM,
            selected_pages=page_numbers,
        )

        # Add local file information to result metadata for future S3 upload
        result.metadata.update(
            {
                "local_file_path": str(file_path),
                "original_filename": file.filename,
                "user_id": user_id,
                "document_type": document_type.value,
                "course_id": course_id,
                "ready_for_submission": True,
                "selected_pages": page_numbers,  # Store selected pages for S3 upload filtering
            }
        )

        # Keep local file for potential S3 upload after user submission

        return result

    except BaseAPIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        app_logger.error(f"Unexpected error during upload and process: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")


@router.post("/submit-to-s3/{document_id}")
async def submit_document_to_s3(
    document_id: str,
    user_id: str = Form(...),
    questions_json: str = Form(
        ..., description="JSON string of extracted questions and choices"
    ),
    course_id: Optional[str] = Form(
        None, description="Course ID to associate this document with"
    ),
    file_service: FileService = Depends(get_file_service),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """
    Submit processed document to S3 and questions to DynamoDB.

    - **document_id**: ID of the processed document
    - **user_id**: User identifier for storage organization
    - **questions_json**: JSON string containing questions and choices data
    """
    try:
        app_logger.info(f"Submitting document {document_id} to S3 for user: {user_id}")

        # Find the local file - try both document_id pattern and any PDF in uploads
        upload_dir = Path(get_settings().upload_folder)
        matching_files = list(upload_dir.glob(f"{document_id}*"))

        # If not found by document_id, try to find any recent PDF file
        if not matching_files:
            # Get all PDF files, sorted by modification time (newest first)
            all_pdfs = list(upload_dir.glob("*.pdf"))
            if all_pdfs:
                all_pdfs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                local_file_path = all_pdfs[0]  # Use the most recently modified PDF
                app_logger.info(
                    f"Using most recent PDF file: {local_file_path.name} for document_id: {document_id}"
                )
            else:
                raise HTTPException(
                    status_code=404, detail=f"Document {document_id} not found locally"
                )
        else:
            local_file_path = matching_files[0]
        original_filename = local_file_path.name

        # Parse questions JSON
        import json

        try:
            questions_data = json.loads(questions_json)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON format for questions: {e}"
            )

        # Check if we need to create a filtered PDF (if selected pages were specified)
        selected_pages = questions_data.get("selected_pages")
        filtered_file_path = None

        if (
            selected_pages
            and len(selected_pages) > 0
            and local_file_path.suffix.lower() == ".pdf"
        ):
            app_logger.info(
                f"Creating filtered PDF with selected pages: {selected_pages}"
            )
            try:
                filtered_file_path = await create_filtered_pdf(
                    local_file_path, selected_pages
                )
                app_logger.info(f"Filtered PDF created: {filtered_file_path}")
            except Exception as e:
                app_logger.error(f"Failed to create filtered PDF: {e}")
                # Continue with original file if filtering fails
                filtered_file_path = None

        # Use filtered PDF if available, otherwise use original file
        file_to_upload = filtered_file_path if filtered_file_path else local_file_path
        upload_filename = (
            f"selected_pages_{original_filename}"
            if filtered_file_path
            else original_filename
        )

        # Upload file to S3
        with open(file_to_upload, "rb") as file_content:
            # Create a temporary UploadFile-like object
            class TempUploadFile:
                def __init__(self, file_path, upload_filename):
                    self.file = open(file_path, "rb")
                    self.filename = upload_filename
                    self.content_type = self._get_content_type(file_path)

                def _get_content_type(self, file_path):
                    suffix = file_path.suffix.lower()
                    if suffix == ".pdf":
                        return "application/pdf"
                    elif suffix in [".jpg", ".jpeg"]:
                        return "image/jpeg"
                    elif suffix == ".png":
                        return "image/png"
                    else:
                        return "application/octet-stream"

                async def read(self):
                    return self.file.read()

                def close(self):
                    self.file.close()

            temp_file = TempUploadFile(file_to_upload, upload_filename)

            # Upload to S3
            document_type = questions_data.get("document_type", "exam")
            s3_metadata = await file_service.upload_file_to_s3(
                temp_file, user_id, document_type
            )

            temp_file.close()

        # Store questions in DynamoDB instead of S3
        quiz_id = await dynamodb_service.store_quiz_questions(
            document_id=document_id,
            user_id=user_id,
            questions_data=questions_data,
            selected_pages=selected_pages,
            s3_file_key=s3_metadata["s3_key"],
            course_id=course_id,
        )

        # Clean up local files after successful S3 upload
        await file_service.cleanup_local_file(local_file_path)
        if filtered_file_path:
            # Clean up the temporary filtered PDF
            await file_service.cleanup_local_file(filtered_file_path)

        return {
            "message": "Document submitted successfully - PDF to S3, Questions to DynamoDB",
            "document_id": document_id,
            "quiz_id": quiz_id,
            "raw_file_s3_key": s3_metadata["s3_key"],
            "raw_file_s3_url": s3_metadata["s3_url"],
            "questions_stored_in": "DynamoDB",
            "questions_count": len(questions_data.get("questions", []))
            if isinstance(questions_data, dict)
            else 0,
            "selected_pages": selected_pages,
        }

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error submitting document {document_id} to S3: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to submit document to S3: {str(e)}"
        )


@router.get("/users/{user_id}/documents")
async def list_user_documents(
    user_id: str,
    limit: int = 100,
    file_service: FileService = Depends(get_file_service),
):
    """
    List documents for a specific user from S3 storage.

    - **user_id**: User identifier
    - **limit**: Maximum number of documents to return (default 100)
    """
    try:
        app_logger.info(f"Listing documents for user: {user_id}")

        documents = await file_service.list_user_documents_from_s3(user_id, limit)

        return {
            "user_id": user_id,
            "total_documents": len(documents),
            "documents": documents,
        }

    except Exception as e:
        app_logger.error(f"Error listing documents for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list user documents")


@router.get("/dynamodb/health")
async def dynamodb_health_check(dynamodb_service=Depends(get_dynamodb_service)):
    """Compatibility health endpoint for the Supabase-backed data service."""
    try:
        health_status = await dynamodb_service.check_tables_health()
        return {
            "dynamodb_enabled": False,
            "supabase_enabled": True,
            "compatibility_endpoint": "/dynamodb/health",
            "message": "DynamoDB has been replaced by Supabase Postgres",
            **health_status,
        }

    except Exception as e:
        app_logger.error(f"DynamoDB health check failed: {e}")
        return {
            "dynamodb_enabled": False,
            "supabase_enabled": True,
            "status": "error",
            "error": str(e),
            "healthy": False,
        }


@router.get("/s3/health")
async def s3_health_check(file_service: FileService = Depends(get_file_service)):
    """Compatibility health endpoint for Supabase Storage."""
    try:
        settings = get_settings()

        if not settings.use_s3_storage:
            return {
                "s3_enabled": False,
                "supabase_storage_enabled": False,
                "status": "disabled",
            }

        s3_service = file_service.s3_service
        if not s3_service:
            return {
                "s3_enabled": True,
                "status": "error",
                "message": "S3 service not initialized",
            }

        # Test bucket access
        bucket_accessible = await s3_service.check_bucket_access()

        return {
            "s3_enabled": False,
            "supabase_storage_enabled": True,
            "compatibility_endpoint": "/s3/health",
            "bucket_name": settings.supabase_storage_bucket,
            "bucket_accessible": bucket_accessible,
            "status": "healthy" if bucket_accessible else "bucket_inaccessible",
        }

    except Exception as e:
        app_logger.error(f"S3 health check failed: {e}")
        return {"s3_enabled": True, "status": "error", "error": str(e)}


@router.get("/files/{filename}")
async def get_file(
    filename: str, file_service: FileService = Depends(get_file_service)
):
    """
    Retrieve an uploaded file.

    - **filename**: Name of the file to retrieve
    """
    try:
        file_path = Path(get_settings().upload_folder) / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        file_info = await file_service.get_file_info(file_path)

        return FileResponse(
            path=str(file_path), media_type=file_info["mime_type"], filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error retrieving file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve file")


@router.delete("/files/{filename}")
async def delete_file(
    filename: str, file_service: FileService = Depends(get_file_service)
):
    """
    Delete an uploaded file.

    - **filename**: Name of the file to delete
    """
    try:
        file_path = Path(get_settings().upload_folder) / filename

        deleted = await file_service.delete_file(file_path)

        if not deleted:
            raise HTTPException(status_code=404, detail="File not found")

        return {"message": f"File {filename} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error deleting file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete file")


@router.get("/documents/{document_id}/status")
async def get_processing_status(document_id: str):
    """
    Get processing status for a document.

    - **document_id**: ID of the document to check
    """
    # This is a placeholder implementation
    # In production, you'd track processing status in a database or cache

    try:
        upload_dir = Path(get_settings().upload_folder)
        matching_files = list(upload_dir.glob(f"{document_id}*"))

        if not matching_files:
            raise HTTPException(
                status_code=404, detail=f"Document {document_id} not found"
            )

        # Return completed status if file exists
        # In production, implement proper status tracking
        return {
            "document_id": document_id,
            "status": ProcessingStatusEnum.COMPLETED,
            "message": "Document is ready for processing",
        }

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error checking status for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to check document status")


@router.get("/quiz/{quiz_id}")
async def get_quiz(
    quiz_id: str,
    user_id: Optional[str] = None,
    course_id: Optional[str] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        STUDENT_BEARER_OPTIONAL
    ),
    student_auth_service: StudentAuthService = Depends(_get_student_auth_service),
    dynamodb_service=Depends(get_db_service),
):
    """Get quiz questions by quiz ID using the configured DynamoDB service."""
    try:
        # The adapter exposes get_quiz for both separated/enhanced services
        quiz = await dynamodb_service.get_quiz(quiz_id)

        if not quiz:
            raise HTTPException(status_code=404, detail=f"Quiz {quiz_id} not found")

        if user_id:
            await _ensure_user_matches_token(
                user_id=user_id,
                credentials=credentials,
                auth_service=student_auth_service,
            )

        effective_course_id = str(course_id or quiz.get("course_id") or "").strip()
        if user_id and effective_course_id and effective_course_id != "default-course":
            await _ensure_active_course_access(
                dynamodb_service=dynamodb_service,
                user_id=user_id,
                course_id=effective_course_id,
            )

        return quiz

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error retrieving quiz {quiz_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve quiz")


@router.get("/users/{user_id}/quizzes")
async def list_user_quizzes(
    user_id: str,
    limit: int = 50,
    course_id: Optional[str] = None,
    dynamodb_service=Depends(get_dynamodb_service),
):
    """List all quizzes for a specific user, optionally filtered by course."""
    try:
        quizzes = await dynamodb_service.get_user_quizzes(user_id, course_id)

        return {"user_id": user_id, "total_quizzes": len(quizzes), "quizzes": quizzes}

    except Exception as e:
        app_logger.error(f"Error listing quizzes for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list user quizzes")


@router.get("/users/{user_id}/quiz-results")
async def list_user_quiz_results(
    user_id: str,
    course_id: Optional[str] = None,
    quiz_id: Optional[str] = None,
    dynamodb_service=Depends(get_dynamodb_service),
):
    """List quiz submission results for a user, optionally filtered by course_id/quiz_id."""
    try:
        results = await dynamodb_service.get_user_quiz_results(
            user_id=user_id,
            quiz_id=quiz_id,
            course_id=course_id,
        )

        return {
            "user_id": user_id,
            "total_results": len(results),
            "results": results,
        }
    except Exception as e:
        app_logger.error(f"Error listing quiz results for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list user quiz results")


@router.get("/courses/{course_id}/quizzes")
async def list_course_quizzes(
    course_id: str,
    user_id: Optional[str] = None,
    q: Optional[str] = None,
    difficulty: Optional[str] = None,
    sort: str = "latest",
    view: str = "full",
    page: int = 1,
    page_size: int = 20,
    quiz_ids: Optional[str] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        STUDENT_BEARER_OPTIONAL
    ),
    student_auth_service: StudentAuthService = Depends(_get_student_auth_service),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """List all quizzes associated with a course (any instructor)."""
    try:
        if user_id and str(course_id or "").strip():
            await _ensure_user_matches_token(
                user_id=user_id,
                credentials=credentials,
                auth_service=student_auth_service,
            )
            await _ensure_active_course_access(
                dynamodb_service=dynamodb_service,
                user_id=user_id,
                course_id=course_id,
            )

        def normalize_text(value: Any) -> str:
            return re.sub(r"\s+", " ", str(value or "").strip().lower())

        page = max(1, int(page or 1))
        page_size = max(1, min(100, int(page_size or 20)))
        sort_key = normalize_text(sort)
        summary_view = normalize_text(view) == "summary"
        difficulty_filter = normalize_text(difficulty)
        allowed_ids = []
        if quiz_ids and quiz_ids.strip():
            allowed_ids = [
                token.strip()
                for token in quiz_ids.split(",")
                if token and token.strip()
            ]

        get_quizzes_page = getattr(dynamodb_service, "get_course_quizzes_page", None)
        db_page_supported = (
            callable(get_quizzes_page)
            and sort_key in {"", "latest", "oldest"}
            and difficulty_filter
            not in {"easy", "ง่าย", "medium", "ปานกลาง", "hard", "ยาก"}
        )
        if db_page_supported:
            page_result = await get_quizzes_page(
                course_id,
                page=page,
                page_size=page_size,
                q=q,
                sort=sort_key or "latest",
                quiz_ids=allowed_ids or None,
                summary=summary_view,
            )
            total_filtered = int(page_result.get("total") or 0)
            total_pages = int(page_result.get("total_pages") or 1)
            current_page = int(page_result.get("page") or page)
            return {
                "course_id": course_id,
                "total_quizzes": total_filtered,
                "total_filtered": total_filtered,
                "page": current_page,
                "page_size": int(page_result.get("page_size") or page_size),
                "total_pages": total_pages,
                "has_next": current_page < total_pages,
                "has_prev": current_page > 1,
                "quizzes": page_result.get("rows") or [],
            }

        try:
            quizzes = await dynamodb_service.get_quizzes_by_course(
                course_id, summary=summary_view
            )
        except TypeError:
            quizzes = await dynamodb_service.get_quizzes_by_course(course_id)

        def natural_text_key(value: Any) -> tuple:
            text = normalize_text(value)
            if not text:
                return tuple()
            thai_digit_map = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
            parts = []
            cursor = 0
            for match in re.finditer(r"[0-9๐-๙]+", text):
                if match.start() > cursor:
                    parts.append((0, text[cursor : match.start()]))
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
            return normalize_text(
                item.get("quiz_id") or item.get("id") or item.get("document_id") or ""
            )

        def quiz_title_key(item: Dict[str, Any]) -> tuple:
            return natural_text_key(item.get("title") or item.get("name"))

        def to_question_count(item: Dict[str, Any]) -> int:
            if isinstance(item.get("total_questions"), (int, float)):
                return int(item.get("total_questions") or 0)
            qs = item.get("questions")
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
            if value in {"easy", "ง่าย"}:
                return 2.0
            if value in {"medium", "ปานกลาง"}:
                return 3.0
            if value in {"hard", "ยาก"}:
                return 4.0
            return None

        def to_difficulty_score(raw: Any) -> float:
            parsed = try_parse_difficulty_score(raw)
            if parsed is not None:
                return parsed
            return 3.0

        def pick_difficulty_value(item: Dict[str, Any]) -> Any:
            for key in (
                "difficulty_avg",
                "difficulty",
                "difficulty_level",
                "level_difficulty",
                "level",
            ):
                value = item.get(key)
                if isinstance(value, str) and not normalize_text(value):
                    continue
                if value is not None:
                    return value
            questions = item.get("questions")
            if isinstance(questions, list) and questions:
                scores: List[float] = []
                for question in questions:
                    if not isinstance(question, dict):
                        continue
                    parsed = try_parse_difficulty_score(
                        question.get("difficulty", question.get("level"))
                    )
                    if parsed is not None:
                        scores.append(parsed)
                if scores:
                    return sum(scores) / len(scores)
            return None

        def to_difficulty_bucket(score: float) -> str:
            # Keep bucketing consistent with student frontend (round-to-star first).
            stars = max(1, min(5, int(round(score))))
            if stars <= 2:
                return "easy"
            if stars == 3:
                return "medium"
            return "hard"

        total_before_filter = len(quizzes)

        # Optional scope: only include specific quiz IDs (useful for lesson-level listing)
        if allowed_ids:
            allowed_ids_set = set(allowed_ids)
            quizzes = [
                item
                for item in quizzes
                if str(
                    item.get("quiz_id")
                    or item.get("id")
                    or item.get("document_id")
                    or ""
                )
                in allowed_ids_set
            ]

        # Search by title/description
        q_text = normalize_text(q)
        if q_text:
            quizzes = [
                item
                for item in quizzes
                if q_text in normalize_text(item.get("title"))
                or q_text in normalize_text(item.get("description"))
            ]

        # Difficulty filter (easy | medium | hard)
        if difficulty_filter in {"easy", "ง่าย", "medium", "ปานกลาง", "hard", "ยาก"}:
            target_bucket = (
                "easy"
                if difficulty_filter in {"easy", "ง่าย"}
                else "medium"
                if difficulty_filter in {"medium", "ปานกลาง"}
                else "hard"
            )

            def in_bucket(score: float) -> bool:
                return to_difficulty_bucket(score) == target_bucket

            quizzes = [
                item
                for item in quizzes
                if in_bucket(to_difficulty_score(pick_difficulty_value(item)))
            ]

        if sort_key in {"oldest"}:
            quizzes.sort(key=lambda x: x.get("created_at", ""))
        elif sort_key in {"title_asc"}:
            quizzes.sort(key=lambda x: (quiz_title_key(x), quiz_identity_key(x)))
        elif sort_key in {"title_desc"}:
            quizzes.sort(
                key=lambda x: (quiz_title_key(x), quiz_identity_key(x)),
                reverse=True,
            )
        elif sort_key in {"difficulty_asc"}:
            quizzes.sort(
                key=lambda x: (
                    to_difficulty_score(pick_difficulty_value(x)),
                    quiz_title_key(x),
                    quiz_identity_key(x),
                )
            )
        elif sort_key in {"difficulty_desc"}:
            quizzes.sort(
                key=lambda x: (
                    -to_difficulty_score(pick_difficulty_value(x)),
                    quiz_title_key(x),
                    quiz_identity_key(x),
                )
            )
        elif sort_key in {"questions_asc"}:
            quizzes.sort(key=to_question_count)
        elif sort_key in {"questions_desc"}:
            quizzes.sort(key=to_question_count, reverse=True)
        else:  # latest
            quizzes.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        total_filtered = len(quizzes)
        total_pages = max(1, (total_filtered + page_size - 1) // page_size)
        if page > total_pages:
            page = total_pages
        start = (page - 1) * page_size
        end = start + page_size
        quizzes_page = quizzes[start:end]

        return {
            "course_id": course_id,
            "total_quizzes": total_before_filter,
            "total_filtered": total_filtered,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "quizzes": quizzes_page,
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error listing quizzes for course {course_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list course quizzes")


@router.put("/quiz/{quiz_id}")
async def update_quiz(
    quiz_id: str, quiz_update: dict, dynamodb_service=Depends(get_dynamodb_service)
):
    """Update quiz title and questions."""
    try:
        # Extract fields from the request body
        title = quiz_update.get("title")
        questions = quiz_update.get("questions")
        total_questions = quiz_update.get("total_questions")
        description = quiz_update.get("description")
        duration_minutes = quiz_update.get("duration_minutes")
        extra_prompt = quiz_update.get("extra_prompt")
        exam_details = quiz_update.get("exam_details")
        selection_reasons = (
            quiz_update.get("selection_reasons")
            or quiz_update.get("reasons")
            or quiz_update.get("pick_reasons")
        )

        app_logger.info(
            f"Updating quiz {quiz_id} - title: {title}, questions count: {total_questions}"
        )

        if not title:
            raise HTTPException(status_code=400, detail="Title is required")
        if not questions:
            raise HTTPException(status_code=400, detail="Questions are required")
        if total_questions is None:
            total_questions = len(questions) if isinstance(questions, list) else 0

        # Prepare updates
        updates = {
            "title": title,
            "questions": questions,
            "total_questions": total_questions,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        if description is not None:
            updates["description"] = str(description).strip()
        if duration_minutes is not None:
            try:
                updates["duration_minutes"] = max(0, int(duration_minutes))
            except Exception:
                raise HTTPException(
                    status_code=400, detail="duration_minutes must be an integer"
                )
        if extra_prompt is not None:
            updates["extra_prompt"] = str(extra_prompt).strip()
        if exam_details is not None:
            updates["exam_details"] = str(exam_details).strip()

        if selection_reasons is not None:
            if isinstance(selection_reasons, str):
                reasons_list = [
                    line.strip()
                    for line in selection_reasons.split("\n")
                    if line.strip()
                ]
            elif isinstance(selection_reasons, list):
                reasons_list = [
                    str(item).strip() for item in selection_reasons if str(item).strip()
                ]
            else:
                reasons_list = []
            updates["selection_reasons"] = reasons_list[:5]

        success = await dynamodb_service.update_quiz(quiz_id, updates)

        if success:
            # Return updated quiz
            quiz = await dynamodb_service.get_quiz_by_id(quiz_id)
            return {"message": "Quiz updated successfully", "quiz": quiz}
        else:
            raise HTTPException(status_code=500, detail="Failed to update quiz")

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error updating quiz {quiz_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update quiz")


@router.delete("/quiz/{quiz_id}")
async def delete_quiz(quiz_id: str, dynamodb_service=Depends(get_dynamodb_service)):
    """Delete a quiz (soft delete by setting status to 'deleted')."""
    try:
        success = await dynamodb_service.delete_quiz(quiz_id)

        if success:
            return {"message": f"Quiz {quiz_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete quiz")

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error deleting quiz {quiz_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete quiz")


# ---- Quiz Results (Submission + History) ----
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


@router.post("/users/{user_id}/quizzes/{quiz_id}/submit")
async def submit_quiz_answers(
    user_id: str,
    quiz_id: str,
    payload: QuizSubmitPayload,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        STUDENT_BEARER_OPTIONAL
    ),
    student_auth_service: StudentAuthService = Depends(_get_student_auth_service),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Submit quiz answers, compute score, and store result history in DynamoDB."""
    try:
        await _ensure_user_matches_token(
            user_id=user_id,
            credentials=credentials,
            auth_service=student_auth_service,
        )

        quiz = await dynamodb_service.get_quiz(quiz_id)
        if not quiz:
            raise HTTPException(status_code=404, detail=f"Quiz {quiz_id} not found")

        effective_course_id = str(
            payload.course_id or quiz.get("course_id") or ""
        ).strip()
        if effective_course_id and effective_course_id != "default-course":
            await _ensure_active_course_access(
                dynamodb_service=dynamodb_service,
                user_id=user_id,
                course_id=effective_course_id,
            )

        questions = quiz.get("questions") or []
        provided = payload.answers
        ordered_answers: list = []
        if isinstance(provided, list):
            ordered_answers = provided
        elif isinstance(provided, dict):
            for idx, q in enumerate(questions):
                qid = q.get("id") or f"q{idx+1}"
                ordered_answers.append(provided.get(qid))
        else:
            raise HTTPException(status_code=400, detail="Invalid answers format")

        # Normalize correct indices from various possible fields
        def _normalize_correct_index(q: dict) -> int:
            try:
                # Prefer explicit numeric fields
                for key in (
                    "correct_answer",
                    "correct_index",
                    "answer_index",
                    "correct",
                ):
                    if q.get(key) is not None:
                        val = q.get(key)
                        if isinstance(val, (int, float)):
                            return int(val)
                        if isinstance(val, str):
                            s = val.strip().lower()
                            # Map common letter/number labels
                            mapping = {
                                "a": 0,
                                "1": 0,
                                "ก": 0,
                                "b": 1,
                                "2": 1,
                                "ข": 1,
                                "c": 2,
                                "3": 2,
                                "ค": 2,
                                "d": 3,
                                "4": 3,
                                "ง": 3,
                            }
                            if s in mapping:
                                return mapping[s]
                            # Try to parse number within string (1-based)
                            import re

                            m = re.search(r"(\d+)", s)
                            if m:
                                n = int(m.group(1)) - 1
                                if n >= 0:
                                    return n
                            # Match exact option text
                            options = q.get("choices") or q.get("options") or []
                            for idx, opt in enumerate(options):
                                if str(opt).strip().lower() == s:
                                    return idx
                # Fallback: try boolean flags
                if "answer" in q and isinstance(q["answer"], (int, float)):
                    return int(q["answer"])
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

        score = (
            int(round((correct_count / total_questions) * 100))
            if total_questions > 0
            else 0
        )

        result = {
            "answers": ordered_answers,
            "correct_count": correct_count,
            "total_questions": total_questions,
            "score": score,
            "time_spent_seconds": payload.time_spent_seconds or 0,
            "course_id": payload.course_id or quiz.get("course_id", ""),
            "lesson_id": payload.lesson_id or "",
        }

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
            result["per_question_time_seconds"] = clean_question_times
        else:
            result["per_question_time_seconds"] = {}

        if isinstance(payload.confidence_by_question, dict):
            clean_confidence: Dict[str, str] = {}
            for qid, confidence in payload.confidence_by_question.items():
                if qid is None:
                    continue
                value = (
                    str(confidence).strip().lower() if confidence is not None else ""
                )
                if value in ("confident", "not_confident"):
                    clean_confidence[str(qid)] = value
            result["confidence_by_question"] = clean_confidence
        else:
            result["confidence_by_question"] = {}

        per_question_time_list: List[int] = []
        confidence_list: List[Optional[str]] = []
        for idx, q in enumerate(questions):
            qid = str(q.get("id") or f"q{idx+1}")
            per_question_time_list.append(
                int(result["per_question_time_seconds"].get(qid, 0))
            )
            confidence_list.append(result["confidence_by_question"].get(qid))
        result["per_question_time_list"] = per_question_time_list
        result["confidence_list"] = confidence_list

        result_id = await dynamodb_service.create_quiz_result(user_id, quiz_id, result)

        return {
            "message": "Submission recorded",
            "result_id": result_id,
            "user_id": user_id,
            "quiz_id": quiz_id,
            "score": score,
            "correct_count": correct_count,
            "total_questions": total_questions,
            "time_spent_seconds": result.get("time_spent_seconds", 0),
            "per_question_time_seconds": result.get("per_question_time_seconds", {}),
            "confidence_by_question": result.get("confidence_by_question", {}),
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error submitting quiz {quiz_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit quiz answers")


@router.post("/users/{user_id}/learning-activity")
async def record_user_learning_activity(
    user_id: str,
    payload: LearningActivityPayload,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        STUDENT_BEARER_OPTIONAL
    ),
    student_auth_service: StudentAuthService = Depends(_get_student_auth_service),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Record a durable lesson-view activity day for dashboard consistency."""
    try:
        await _ensure_user_matches_token(
            user_id=user_id,
            credentials=credentials,
            auth_service=student_auth_service,
        )

        course_id = str(payload.course_id or "").strip()
        if not course_id:
            raise HTTPException(status_code=400, detail="course_id is required")

        active_enrollment = None
        if course_id != "default-course":
            access = await _ensure_active_course_access(
                dynamodb_service=dynamodb_service,
                user_id=user_id,
                course_id=course_id,
            )
            active_enrollment = access["enrollment"]

        recorder = getattr(dynamodb_service, "record_learning_activity", None)
        if not callable(recorder):
            raise HTTPException(
                status_code=500, detail="Learning activity storage is unavailable"
            )

        result = await recorder(
            user_id=user_id,
            course_id=course_id,
            lesson_id=payload.lesson_id,
            activity_day=payload.activity_day,
            activity_days=payload.activity_days,
            enrollment=active_enrollment,
        )
        if not result:
            raise HTTPException(status_code=404, detail="Enrollment not found")

        return {"message": "Learning activity recorded", **result}

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error recording learning activity for {user_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to record learning activity"
        )


@router.get("/users/{user_id}/quizzes/{quiz_id}/results")
async def get_user_quiz_results(
    user_id: str, quiz_id: str, dynamodb_service=Depends(get_dynamodb_service)
):
    """Return submission history for a user and quiz."""
    try:
        results = await dynamodb_service.get_user_quiz_results(user_id, quiz_id)
        return {
            "user_id": user_id,
            "quiz_id": quiz_id,
            "total_results": len(results),
            "results": results,
        }
    except Exception as e:
        app_logger.error(
            f"Error getting quiz results for user {user_id}, quiz {quiz_id}: {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to get quiz results")


@router.get("/s3/presigned-url")
async def get_presigned_url(
    s3_key: str, file_service: FileService = Depends(get_file_service)
):
    """Get a pre-signed URL for an S3 object."""
    try:
        settings = get_settings()

        if not settings.use_s3_storage:
            raise HTTPException(status_code=501, detail="S3 storage is disabled")

        s3_service = file_service.s3_service
        if not s3_service:
            raise HTTPException(status_code=500, detail="S3 service not available")

        # Generate pre-signed URL (valid for 1 hour)
        presigned_url = await s3_service.generate_presigned_url(s3_key, expiration=3600)

        return {"s3_key": s3_key, "presigned_url": presigned_url, "expires_in": 3600}

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error generating presigned URL for {s3_key}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate presigned URL")


@router.post("/quiz/skeletons/upsample")
async def upsample_quiz_skeletons(
    payload: SkeletonUpsampleRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    """Create additional abstract skeleton variants for an existing template."""
    raise HTTPException(status_code=410, detail="Skeleton upsampling is disabled")
    normalized_seed_skeletons: List[Dict[str, Any]] = []
    normalized_seed_entries: List[Dict[str, Any]] = []
    for index, item in enumerate(payload.skeletons):
        normalized = _normalize_seed_skeleton(item)
        if normalized:
            source_material = _extract_seed_skeleton_source_material(item)
            seed_entry: Dict[str, Any] = {
                "skeleton": normalized,
                "source_skeleton_index": len(normalized_seed_skeletons),
            }
            if source_material:
                seed_entry["source_material"] = source_material
            normalized_seed_skeletons.append(normalized)
            normalized_seed_entries.append(seed_entry)

    if not normalized_seed_skeletons:
        raise HTTPException(status_code=400, detail="No valid skeletons to upsample")

    original_count = len(normalized_seed_skeletons)
    missing_count = max(0, payload.target_count - original_count)
    await _set_quiz_gen_progress(
        payload.progress_job_id,
        {
            "status": "running",
            "total": missing_count,
            "completed": 0,
            "percent": 0 if missing_count > 0 else 100,
            "error": None,
            "detail": "เตรียม skeleton ตั้งต้น",
        },
    )
    if missing_count <= 0:
        await _set_quiz_gen_progress(
            payload.progress_job_id,
            {
                "status": "completed",
                "total": 0,
                "completed": 0,
                "percent": 100,
                "detail": "skeleton pool ถึงเป้าหมายแล้ว",
            },
        )
        return {
            "original_count": original_count,
            "target_count": payload.target_count,
            "created_count": 0,
            "skeletons": [],
            "pool_count": original_count,
        }

    settings = get_settings()
    default_quiz_model = (
        str(settings.litellm_generate_quiz_model or settings.litellm_model).strip()
        or settings.litellm_model
    )
    resolved_model = str(payload.model or "").strip() or default_quiz_model
    litellm_user, litellm_metadata = await chat_service._get_litellm_user_context(
        payload.user_id
    )
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "skeleton_variants",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "skeletons": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "subject": {"type": "string"},
                                "topic_tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "learning_objective": {"type": "string"},
                                "core_logic_and_formulas": {"type": "string"},
                                "context_guidance": {"type": "string"},
                                "variables": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "given": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "target": {"type": "string"},
                                    },
                                    "required": ["given", "target"],
                                },
                                "constraints_and_tricks": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "distractor_logic": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "source_skeleton_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": max(0, original_count - 1),
                                },
                            },
                            "required": [
                                "subject",
                                "topic_tags",
                                "learning_objective",
                                "core_logic_and_formulas",
                                "context_guidance",
                                "variables",
                                "constraints_and_tricks",
                                "distractor_logic",
                                "source_skeleton_index",
                            ],
                        },
                    }
                },
                "required": ["skeletons"],
            },
        },
    }

    def extract_json(text: str) -> str:
        value = (text or "").strip()
        if value.startswith("```"):
            value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
            value = re.sub(r"\s*```$", "", value)
        start = value.find("{")
        end = value.rfind("}")
        return value[start : end + 1] if start >= 0 and end > start else value

    def skeleton_key(skeleton: Dict[str, Any]) -> str:
        return json.dumps(skeleton, ensure_ascii=False, sort_keys=True)

    def build_prompt(batch_count: int, source_entries: List[Dict[str, Any]]) -> str:
        return f"""
คุณเป็นผู้ออกแบบข้อสอบเชิงโครงสร้าง สร้าง skeleton variants แบบนามธรรมจาก skeleton ตั้งต้น
เพื่อนำไปเพิ่มคลัง template ให้มี pattern หลากหลายขึ้นสำหรับสร้างโจทย์ใหม่

ต้องสร้าง skeleton ใหม่จำนวน {batch_count} รายการเท่านั้น

กติกาสำคัญ:
- ใช้เฉพาะหลักวิชา skill, logic, formula roles, variable roles, constraints และ distractor patterns
- เปลี่ยน abstraction, variable roles, context guidance, constraints หรือ distractor patterns ให้หลากหลาย
- รักษาระดับความยากและจำนวนขั้นคิดให้ใกล้เคียง skeleton ตั้งต้น ห้ามยกระดับเป็นโจทย์ที่ซับซ้อนกว่าเดิม
- ห้ามเพิ่ม reasoning step ใหม่ เช่น จากจำ/แทนค่า/คำนวณตรง ๆ ไปเป็นอนุมานหลายชั้น วิเคราะห์หลายเงื่อนไข หรืออ่าน passage ยาว ถ้า skeleton ตั้งต้นไม่ได้ต้องการ
- constraints_and_tricks และ distractor_logic ต้องเป็นกับดักระดับเดียวกับต้นฉบับ ไม่ใช่เพิ่มความยากเพื่อให้ดูต่าง
- context_guidance ต้องสั้นและตรงตามรูปแบบเดิม ห้ามทำให้โจทย์ยากขึ้นด้วยบริบทยาว การอนุมานหลายชั้น idiom/register/nuance หรือข้อมูลรบกวน ถ้าต้นฉบับไม่มีสิ่งนั้น
- ห้ามคงถ้อยคำเฉพาะ narrative ตัวละคร สถานที่ ชื่อเฉพาะ หรือบริบทต้นฉบับ
- ห้ามคงเลขจริงจาก source; ใช้ placeholder เช่น a, b, n, k, rate, count, min, max
- ถ้า skeleton ตั้งต้นมี context_guidance ให้สร้าง context_guidance ใหม่ที่ใกล้เคียงเจตนา/รูปแบบบริบทเดิม และสอดคล้องกับ skeleton/โจทย์ใหม่ ห้ามส่งค่าว่าง แต่ห้ามคัดลอกถ้อยคำหรือรายละเอียดเฉพาะจากต้นฉบับตรง ๆ
- ถ้ามี source_material/source_context ให้ใช้เป็นบริบทต้นฉบับเต็มเพื่อเข้าใจรูปแบบ passage, ตาราง, บทสนทนา, คำสั่งร่วม หรือข้อมูลประกอบเท่านั้น แล้วสรุปเป็น context_guidance ใหม่แบบนามธรรม ห้ามคัดลอกประโยค รายละเอียดเฉพาะ ตัวละคร สถานที่ หรือค่าตัวเลขจาก source_material
- ห้ามสร้าง skeleton ที่ซ้ำกับ skeleton ตั้งต้นหรือซ้ำกันเองใน batch นี้
- แต่ละ skeleton ที่ตอบกลับต้องมี source_skeleton_index เป็นเลข index ของ skeleton ตั้งต้นที่ใช้เป็นแม่แบบ
- ถ้าดัดแปลงจาก variant ที่มี source_skeleton_index อยู่แล้ว ให้ส่ง source_skeleton_index เดิมต่อไป
- ถ้า skeleton ตั้งต้นไม่มี context_guidance และไม่ต้องใช้บริบท ให้ context_guidance เป็น string ว่าง
- ตอบกลับ JSON เท่านั้นตาม schema

Skeleton ตั้งต้นพร้อม index:
{json.dumps(source_entries, ensure_ascii=False)}
"""

    known_skeleton_keys = {
        skeleton_key(skeleton) for skeleton in normalized_seed_skeletons
    }
    variants: List[Dict[str, Any]] = []
    variant_entries: List[Dict[str, Any]] = []
    batch_size = min(12, missing_count)
    last_error_message = ""
    try:
        while len(variants) < missing_count:
            remaining_count = missing_count - len(variants)
            current_batch_size = min(batch_size, remaining_count)
            await _set_quiz_gen_progress(
                payload.progress_job_id,
                {
                    "status": "running",
                    "total": missing_count,
                    "completed": len(variants),
                    "percent": int(round((len(variants) / missing_count) * 100))
                    if missing_count > 0
                    else 100,
                    "detail": f"กำลังสร้าง skeleton batch {current_batch_size} รายการ",
                },
            )
            source_pool = [*normalized_seed_entries, *variant_entries]
            source_sample = (
                random.sample(source_pool, 40) if len(source_pool) > 40 else source_pool
            )
            prompt = build_prompt(current_batch_size, source_sample)

            try:
                raw_text = await asyncio.wait_for(
                    chat_service._call_gemini_chat(
                        prompt,
                        response_format=response_format,
                        model_name=resolved_model,
                        litellm_user=litellm_user,
                        litellm_metadata=litellm_metadata,
                    ),
                    timeout=max(45, int(settings.gemini_timeout or 60)),
                )
                parsed = json.loads(extract_json(raw_text))
                raw_variants = (
                    parsed.get("skeletons") if isinstance(parsed, dict) else []
                )
                if not isinstance(raw_variants, list):
                    raw_variants = []

                created_before = len(variants)
                for item in raw_variants:
                    normalized = _normalize_seed_skeleton(item)
                    if not normalized:
                        continue
                    try:
                        source_skeleton_index = int(item.get("source_skeleton_index"))
                    except (TypeError, ValueError):
                        source_skeleton_index = len(variants) % original_count
                    if (
                        source_skeleton_index < 0
                        or source_skeleton_index >= original_count
                    ):
                        source_skeleton_index = len(variants) % original_count
                    source_context_guidance = str(
                        normalized_seed_skeletons[source_skeleton_index].get(
                            "context_guidance"
                        )
                        or ""
                    ).strip()
                    if (
                        source_context_guidance
                        and not str(normalized.get("context_guidance") or "").strip()
                    ):
                        normalized["context_guidance"] = source_context_guidance
                    key = skeleton_key(normalized)
                    if key in known_skeleton_keys:
                        continue
                    known_skeleton_keys.add(key)
                    variants.append(
                        {
                            **normalized,
                            "source_skeleton_index": source_skeleton_index,
                        }
                    )
                    variant_entries.append(
                        {
                            "skeleton": normalized,
                            "source_skeleton_index": source_skeleton_index,
                            **(
                                {
                                    "source_material": normalized_seed_entries[
                                        source_skeleton_index
                                    ].get("source_material")
                                }
                                if normalized_seed_entries[source_skeleton_index].get(
                                    "source_material"
                                )
                                else {}
                            ),
                        }
                    )
                    if len(variants) >= missing_count:
                        break

                created_in_batch = len(variants) - created_before
                if created_in_batch <= 0:
                    last_error_message = "AI returned no valid skeleton variants"
                    if variants:
                        break
                    await _set_quiz_gen_progress(
                        payload.progress_job_id,
                        {
                            "status": "error",
                            "error": last_error_message,
                            "detail": last_error_message,
                        },
                    )
                    raise HTTPException(status_code=502, detail=last_error_message)
                await _set_quiz_gen_progress(
                    payload.progress_job_id,
                    {
                        "status": "running",
                        "total": missing_count,
                        "completed": len(variants),
                        "percent": int(round((len(variants) / missing_count) * 100))
                        if missing_count > 0
                        else 100,
                        "detail": f"สร้างแล้ว {len(variants)}/{missing_count} skeleton",
                    },
                )
                if batch_size < 12 and created_in_batch >= current_batch_size:
                    batch_size = min(12, batch_size + 2)
            except asyncio.TimeoutError:
                last_error_message = (
                    f"Timed out while creating {current_batch_size} skeleton variants"
                )
                if current_batch_size > 4:
                    batch_size = max(4, current_batch_size // 2)
                    app_logger.warning(
                        "Skeleton upsampling batch timed out; retrying smaller batch: "
                        f"{current_batch_size} -> {batch_size}"
                    )
                    continue
                if variants:
                    break
                await _set_quiz_gen_progress(
                    payload.progress_job_id,
                    {
                        "status": "error",
                        "error": last_error_message,
                        "detail": last_error_message,
                    },
                )
                raise HTTPException(status_code=504, detail=last_error_message)
            except json.JSONDecodeError as decode_error:
                last_error_message = (
                    f"AI returned invalid JSON for skeleton upsampling: {decode_error}"
                )
                if variants:
                    break
                await _set_quiz_gen_progress(
                    payload.progress_job_id,
                    {
                        "status": "error",
                        "error": last_error_message,
                        "detail": last_error_message,
                    },
                )
                raise HTTPException(status_code=502, detail=last_error_message)

        await _set_quiz_gen_progress(
            payload.progress_job_id,
            {
                "status": "partial" if len(variants) < missing_count else "completed",
                "total": missing_count,
                "completed": len(variants),
                "percent": int(round((len(variants) / missing_count) * 100))
                if missing_count > 0
                else 100,
                "detail": f"สร้าง skeleton เสร็จ {len(variants)}/{missing_count} รายการ",
                "error": last_error_message if len(variants) < missing_count else None,
            },
        )
        return {
            "original_count": original_count,
            "target_count": payload.target_count,
            "created_count": len(variants),
            "skeletons": variants,
            "pool_count": original_count + len(variants),
            "partial": len(variants) < missing_count,
            "warning": last_error_message if len(variants) < missing_count else None,
        }
    except HTTPException:
        raise
    except Exception as error:
        app_logger.exception("Skeleton upsampling failed")
        detail = str(error).strip() or error.__class__.__name__
        await _set_quiz_gen_progress(
            payload.progress_job_id,
            {
                "status": "error",
                "error": detail,
                "detail": detail,
            },
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upsample skeletons: {detail}",
        )


@router.get("/quiz/skeletons/upsample/progress/{job_id}")
async def get_skeleton_upsample_progress(job_id: str):
    """Get real-time progress for skeleton upsampling by job ID."""
    async with QUIZ_GEN_PROGRESS_LOCK:
        payload = QUIZ_GEN_PROGRESS.get(job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="progress job not found")
    return payload


@router.post("/quiz/generate")
async def generate_quiz_with_ai(
    topic: str = Form(..., description="Subject or topic for the quiz"),
    topics_json: Optional[str] = Form(
        None,
        description="Optional JSON array of sub-topics to distribute questions across",
    ),
    seed_questions_json: Optional[str] = Form(
        None,
        description="Deprecated. Reference-question generation is disabled.",
    ),
    seed_skeletons_json: Optional[str] = Form(
        None,
        description="Deprecated. Skeleton-based generation is disabled.",
    ),
    one_shot_examples_json: Optional[str] = Form(
        None,
        description="Deprecated alias for template_examples_json",
    ),
    template_examples_json: Optional[str] = Form(
        None,
        description="Optional JSON array of template questions to transform into new questions",
    ),
    grade_level: str = Form(
        ..., description="Grade level in Thai, e.g., ป.1-ป.6, ม.1-ม.6, มหาวิทยาลัย"
    ),
    output_language: str = Form(
        "th", description="Output language for generated question text: th | en"
    ),
    num_questions: int = Form(
        5, description="Number of questions to generate (1-2000)"
    ),
    difficulty: int = Form(3, description="Overall difficulty level (1-5)"),
    difficulty_strategy: str = Form(
        "single",
        description="Difficulty distribution strategy: single | balanced",
    ),
    exclude_context: bool = Form(
        False,
        description="Require self-contained questions and omit supporting context",
    ),
    num_sets: int = Form(1, description="Number of quiz sets to generate (1-5)"),
    user_id: str = Form(..., description="Instructor user ID"),
    model: Optional[str] = Form(
        None, description="Optional LiteLLM model override for quiz generation"
    ),
    judge_models_json: Optional[str] = Form(
        None,
        description="Optional JSON array of LiteLLM models for LLM-as-a-judge consensus (all selected models must pass)",
    ),
    course_id: Optional[str] = Form(
        None, description="Course ID to attach the quiz to"
    ),
    # Mock-exam specific options
    document_type: str = Form(
        "exam", description="Type of assessment: exam | mock_exam | manual"
    ),
    duration_minutes: Optional[int] = Form(
        None, description="Optional time limit in minutes"
    ),
    extra_prompt: Optional[str] = Form(
        None, description="Additional instructions to steer question generation"
    ),
    progress_job_id: Optional[str] = Form(
        None, description="Optional progress tracker ID"
    ),
    async_generation: bool = Form(
        False,
        description="Start generation in the background and return a progress job immediately",
    ),
    persist_quiz: bool = Form(True, description="Persist generated quiz to database"),
    chat_service: ChatService = Depends(get_chat_service),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Generate a multiple-choice quiz using the LLM and store it.

    Returns the created quiz record.
    """
    try:
        # Validate inputs
        exclude_context = (
            exclude_context if isinstance(exclude_context, bool) else False
        )
        async_generation = (
            async_generation if isinstance(async_generation, bool) else False
        )
        if not topic or len(topic.strip()) == 0:
            raise HTTPException(status_code=400, detail="Topic is required")
        if num_questions < 1 or num_questions > 2000:
            raise HTTPException(
                status_code=400, detail="num_questions must be between 1 and 2000"
            )
        if not isinstance(num_sets, int):
            num_sets = 1
        if num_sets < 1 or num_sets > 5:
            raise HTTPException(
                status_code=400, detail="num_sets must be between 1 and 5"
            )
        total_target = num_questions * num_sets
        if async_generation:
            progress_job_id = progress_job_id or str(uuid.uuid4())
        await _set_quiz_gen_progress(
            progress_job_id,
            {
                "status": "queued" if async_generation else "running",
                "total": total_target,
                "completed": 0,
                "percent": 0,
                "error": None,
                "generated_questions": [],
            },
        )
        if async_generation:

            async def _run_background_quiz_generation() -> None:
                try:
                    result = await generate_quiz_with_ai(
                        topic=topic,
                        topics_json=topics_json,
                        seed_questions_json=seed_questions_json,
                        seed_skeletons_json=seed_skeletons_json,
                        one_shot_examples_json=one_shot_examples_json,
                        template_examples_json=template_examples_json,
                        grade_level=grade_level,
                        output_language=output_language,
                        num_questions=num_questions,
                        difficulty=difficulty,
                        difficulty_strategy=difficulty_strategy,
                        exclude_context=exclude_context,
                        num_sets=num_sets,
                        user_id=user_id,
                        model=model,
                        judge_models_json=judge_models_json,
                        course_id=course_id,
                        document_type=document_type,
                        duration_minutes=duration_minutes,
                        extra_prompt=extra_prompt,
                        progress_job_id=progress_job_id,
                        async_generation=False,
                        persist_quiz=persist_quiz,
                        chat_service=chat_service,
                        dynamodb_service=dynamodb_service,
                    )
                    await _set_quiz_gen_progress(
                        progress_job_id,
                        {
                            "result": {
                                "message": result.get("message"),
                                "quiz_id": result.get("quiz_id"),
                                "quiz_ids": result.get("quiz_ids"),
                                "model": result.get("model"),
                                "partial_generation": result.get(
                                    "partial_generation"
                                ),
                                "generation_summary": result.get(
                                    "generation_summary"
                                ),
                                "quiz": result.get("quiz"),
                                "quizzes": result.get("quizzes"),
                            }
                        },
                    )
                except HTTPException as error:
                    await _set_quiz_gen_progress(
                        progress_job_id,
                        {
                            "status": "failed",
                            "error": str(error.detail),
                            "status_code": error.status_code,
                        },
                    )
                except Exception as error:
                    app_logger.exception("Background quiz generation failed")
                    await _set_quiz_gen_progress(
                        progress_job_id,
                        {
                            "status": "failed",
                            "error": str(error),
                        },
                    )

            asyncio.create_task(_run_background_quiz_generation())
            return {
                "message": "Quiz generation started",
                "job_id": progress_job_id,
                "progress_url": f"/quiz/generate/progress/{progress_job_id}",
                "status": "queued",
                "total": total_target,
            }

        # Clamp difficulty
        if not isinstance(difficulty, int):
            difficulty = 3
        difficulty = min(5, max(1, difficulty))
        normalized_output_language = str(output_language or "th").strip().lower()
        output_language_labels = {
            "th": "Thai",
            "thai": "Thai",
            "ไทย": "Thai",
            "en": "English",
            "eng": "English",
            "english": "English",
        }
        output_language_name = output_language_labels.get(
            normalized_output_language, "Thai"
        )
        difficulty_strategy = str(difficulty_strategy or "single").strip().lower()
        if difficulty_strategy not in {"single", "balanced"}:
            difficulty_strategy = "single"
        settings = get_settings()
        default_quiz_model = (
            str(settings.litellm_generate_quiz_model or settings.litellm_model).strip()
            or settings.litellm_model
        )
        default_quiz_verify_model = (
            str(
                settings.litellm_quiz_verify_model
                or settings.litellm_generate_quiz_model
                or settings.litellm_model
            ).strip()
            or settings.litellm_model
        )
        resolved_model = str(model or "").strip() or default_quiz_model
        judge_model_candidates: List[str] = []
        judge_verification_requested = False
        if isinstance(judge_models_json, str) and judge_models_json.strip():
            try:
                parsed_judge_models = json.loads(judge_models_json)
                if not isinstance(parsed_judge_models, list):
                    raise ValueError("judge_models_json must be a JSON array")
                for item in parsed_judge_models:
                    normalized_model = str(item or "").strip()
                    if (
                        normalized_model
                        and normalized_model not in judge_model_candidates
                    ):
                        judge_model_candidates.append(normalized_model)
                judge_verification_requested = bool(judge_model_candidates)
            except Exception:
                raise HTTPException(
                    status_code=400, detail="Invalid judge_models_json format"
                )
        if not judge_model_candidates:
            judge_model_candidates = [default_quiz_verify_model]
        litellm_user, litellm_metadata = await chat_service._get_litellm_user_context(
            user_id
        )

        topic_pool: List[str] = []
        if topics_json and topics_json.strip():
            try:
                parsed_topics = json.loads(topics_json)
                if isinstance(parsed_topics, list):
                    topic_pool = [
                        str(item).strip() for item in parsed_topics if str(item).strip()
                    ]
            except Exception:
                raise HTTPException(
                    status_code=400, detail="Invalid topics_json format"
                )
        if not topic_pool:
            topic_pool = [topic.strip()]

        if seed_questions_json and seed_questions_json.strip():
            raise HTTPException(
                status_code=400,
                detail="seed_questions_json is disabled",
            )

        template_examples: List[Dict[str, Any]] = []
        examples_payload = template_examples_json or one_shot_examples_json
        if isinstance(examples_payload, str) and examples_payload.strip():
            try:
                parsed_examples = json.loads(examples_payload)
                if not isinstance(parsed_examples, list):
                    raise ValueError("template_examples_json must be a JSON array")
                for item in parsed_examples:
                    normalized_example = _normalize_one_shot_example(item)
                    if normalized_example:
                        template_examples.append(normalized_example)
                    else:
                        normalized_pair = _normalize_one_shot_example_pair(item)
                        if normalized_pair:
                            template_examples.append(normalized_pair["example"])
            except Exception:
                raise HTTPException(
                    status_code=400, detail="Invalid template_examples_json format"
                )
        # Build strict JSON-only prompt in Thai (parallel per-question generation)
        system_instructions = (
            "คุณเป็นผู้ช่วยสร้างข้อสอบที่เชี่ยวชาญ สร้างโจทย์ปรนัย 4 ตัวเลือก "
            "และตอบกลับเป็น JSON เพียงอย่างเดียว ห้ามมีข้อความอื่น. "
            "โจทย์ทุกข้อต้องตอบได้จากข้อความใน question, context และ choices เท่านั้น "
            "ห้ามอ้างถึงหรือกำหนดให้ผู้เรียนดูรูป ภาพ แผนภาพ กราฟ หรือสื่อภายนอกที่ไม่ได้อยู่ในข้อความ. "
            "คำอธิบายเฉลยในฟิลด์ explanation ต้องเป็นภาษาไทยเสมอ แม้คำถามหรือตัวเลือกจะเป็นภาษาอื่น. "
            "หากมีนิพจน์คณิตศาสตร์ ให้ใช้ LaTeX โดยใส่ใน $...$ (inline) หรือ $$...$$ (display) "
            "เช่น $x^{2}$, $\\frac{a}{b}$, $(-3)\\times 4$, $18\\div (-3)$."
        )
        question_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "quiz_question",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 1200,
                        },
                        "context": {
                            "type": "string",
                            "maxLength": 2500,
                            "description": "Supporting passage, table, shared instruction, or empty string when not needed",
                        },
                        "choices": {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 4,
                            "items": {"type": "string", "minLength": 1},
                        },
                        "correct_answer": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 3,
                        },
                        "explanation": {
                            "type": "string",
                            "minLength": 30,
                            "maxLength": 800,
                        },
                        "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": [
                        "question",
                        "context",
                        "choices",
                        "correct_answer",
                        "explanation",
                        "difficulty",
                    ],
                },
            },
        }
        verification_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "quiz_verification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "is_correct": {"type": "boolean"},
                        "similarity_score_0_to_10": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 10,
                        },
                        "correctness_issues": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "similarity_issues": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "verdict_reason": {"type": "string"},
                    },
                    "required": [
                        "is_correct",
                        "similarity_score_0_to_10",
                        "correctness_issues",
                        "similarity_issues",
                        "verdict_reason",
                    ],
                },
            },
        }
        skeleton_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "skeleton_variants",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "skeletons": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "subject": {"type": "string"},
                                    "topic_tags": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "learning_objective": {"type": "string"},
                                    "core_logic_and_formulas": {"type": "string"},
                                    "context_guidance": {"type": "string"},
                                    "variables": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "given": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "target": {"type": "string"},
                                        },
                                        "required": ["given", "target"],
                                    },
                                    "constraints_and_tricks": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "distractor_logic": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [
                                    "subject",
                                    "topic_tags",
                                    "learning_objective",
                                    "core_logic_and_formulas",
                                    "context_guidance",
                                    "variables",
                                    "constraints_and_tricks",
                                    "distractor_logic",
                                ],
                            },
                        },
                    },
                    "required": ["skeletons"],
                },
            },
        }

        is_mock_exam = str(document_type or "").strip().lower() == "mock_exam"
        title_prefix = "ข้อสอบจำลอง" if is_mock_exam else "แบบทดสอบ"
        title_text = f"{title_prefix} {topic.strip()} ระดับชั้น {grade_level.strip()}"

        extra_block = ""
        if extra_prompt and extra_prompt.strip():
            extra_block = f"\nคำสั่งเพิ่มเติมจากผู้สอน:\n{extra_prompt.strip()}\n"
        context_rule_block = (
            '\n- ห้ามสร้าง context โดยเด็ดขาด ให้ตอบ context เป็น "" เท่านั้น '
            "และเขียน question ให้มีข้อมูลครบ อ่านและตอบได้ด้วยตัวเองโดยไม่อ้างถึง passage, ตาราง, "
            "บทสนทนา, คำสั่งร่วม หรือข้อมูลภายนอก\n"
            if exclude_context
            else ""
        )
        template_context_rule = (
            "- แม้ template ต้นฉบับมี context ก็ห้ามสร้าง context ในผลลัพธ์ "
            "ให้นำเฉพาะข้อมูลที่จำเป็นต่อการตอบมาเขียนรวมใน question แทน"
            if exclude_context
            else "- ถ้า template ต้นฉบับมี context ให้รักษารูปแบบและความยาวใกล้ต้นฉบับ "
            "เปลี่ยนเฉพาะรายละเอียดที่จำเป็น และใส่เฉพาะข้อมูลที่ใช้ตอบจริง"
        )

        difficulty_rubric = """
เกณฑ์ความยากที่ต้องใช้:
- ระดับ 1 ง่าย: ใช้ความรู้พื้นฐานโดยตรง คิด 1 ขั้น ไม่มีข้อมูลรบกวน
- ระดับ 2 ค่อนข้างง่าย: คิด 1-2 ขั้น มีการแปลงหรือตีความข้อมูลเล็กน้อย
- ระดับ 3 ปานกลาง: คิด 2-3 ขั้น ต้องเลือกหลักการที่เหมาะสม และมีตัวลวงจากข้อผิดพลาดทั่วไป
- ระดับ 4 ค่อนข้างยาก: คิด 3-4 ขั้น ต้องเชื่อมโยงข้อมูลหรือหลักการมากกว่าหนึ่งส่วน
- ระดับ 5 ยาก: คิดอย่างน้อย 4 ขั้น ต้องวิเคราะห์ วางแผน หรือประยุกต์หลายหลักการ

หลักการประเมินความยาก:
- ปรับความยากด้วยจำนวนขั้นคิด ระดับการประยุกต์ และคุณภาพตัวลวง ไม่ใช่เพิ่มความยาว ตัวเลขใหญ่ หรือข้อมูลรบกวน
- ห้ามใช้เนื้อหาเกินระดับชั้นเพื่อทำให้โจทย์ยากขึ้น
- ก่อนตอบ ให้ตรวจภายในว่าโจทย์ใช้จำนวนขั้นคิดและระดับการประยุกต์ตรงกับความยากเป้าหมาย
""".strip()

        def extract_json(text: str) -> str:
            """Best-effort extraction of a JSON blob from LLM output.
            Handles code fences, missing closing fences, and extra prose.
            """
            text = (text or "").strip()
            if not text:
                return text
            # 1) Proper fenced block ```json ... ```
            fence_match = re.search(r"```(?:json)?\n([\s\S]*?)```", text)
            if fence_match:
                return fence_match.group(1).strip()
            # 2) Incomplete opening fence like "```json" with no close
            if text.lower().startswith("```json"):
                # Remove the leading fence line
                text = text.split("\n", 1)[1] if "\n" in text else ""
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else ""

            # 3) Extract the first plausible top-level JSON object or array by brace matching
            def _extract_by_braces(s: str) -> Optional[str]:
                start_obj = s.find("{")
                start_arr = s.find("[")
                # choose earliest positive index
                starts = [i for i in [start_obj, start_arr] if i != -1]
                if not starts:
                    return None
                start = min(starts)
                open_char = s[start]
                close_char = "}" if open_char == "{" else "]"
                depth = 0
                for i in range(start, len(s)):
                    ch = s[i]
                    if ch == open_char:
                        depth += 1
                    elif ch == close_char:
                        depth -= 1
                        if depth == 0:
                            return s[start : i + 1]
                return None

            candidate = _extract_by_braces(text)
            return candidate.strip() if candidate else text

        async def upsample_seed_skeletons(
            skeletons: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            original_count = len(skeletons)
            pool_target = _compute_skeleton_pool_target(original_count, total_target)
            missing_count = max(0, pool_target - original_count)
            if missing_count <= 0:
                pool = skeletons[:]
                random.shuffle(pool)
                return pool

            prompt = f"""
คุณเป็นผู้ออกแบบข้อสอบเชิงโครงสร้าง สร้าง skeleton variants แบบนามธรรมจาก skeleton ตั้งต้น
เพื่อนำไปสร้างโจทย์ใหม่ที่หลากหลายและไม่ซ้ำ pattern เดิม

ต้องสร้าง skeleton ใหม่จำนวน {missing_count} รายการเท่านั้น

กติกาสำคัญ:
- ใช้เฉพาะหลักวิชา skill, logic, formula roles, variable roles, constraints และ distractor patterns
- เปลี่ยน abstraction, variable roles, context guidance, constraints หรือ distractor patterns ให้หลากหลาย
- รักษาระดับความยากและจำนวนขั้นคิดให้ใกล้เคียง skeleton ตั้งต้น ห้ามยกระดับเป็นโจทย์ที่ซับซ้อนกว่าเดิม
- ห้ามเพิ่ม reasoning step ใหม่ เช่น จากจำ/แทนค่า/คำนวณตรง ๆ ไปเป็นอนุมานหลายชั้น วิเคราะห์หลายเงื่อนไข หรืออ่าน passage ยาว ถ้า skeleton ตั้งต้นไม่ได้ต้องการ
- constraints_and_tricks และ distractor_logic ต้องเป็นกับดักระดับเดียวกับต้นฉบับ ไม่ใช่เพิ่มความยากเพื่อให้ดูต่าง
- context_guidance ต้องสั้นและตรงตามรูปแบบเดิม ห้ามทำให้โจทย์ยากขึ้นด้วยบริบทยาว การอนุมานหลายชั้น idiom/register/nuance หรือข้อมูลรบกวน ถ้าต้นฉบับไม่มีสิ่งนั้น
- ห้ามคงถ้อยคำเฉพาะ narrative ตัวละคร สถานที่ ชื่อเฉพาะ หรือบริบทต้นฉบับ
- ห้ามคงเลขจริงจาก source; ใช้ placeholder เช่น a, b, n, k, rate, count, min, max
- ถ้า skeleton ตั้งต้นมี context_guidance ให้สร้าง context_guidance ใหม่ที่ใกล้เคียงเจตนา/รูปแบบบริบทเดิม และสอดคล้องกับ skeleton/โจทย์ใหม่ ห้ามส่งค่าว่าง แต่ห้ามคัดลอกถ้อยคำหรือรายละเอียดเฉพาะจากต้นฉบับตรง ๆ
- ถ้า skeleton ตั้งต้นไม่มี context_guidance และไม่ต้องใช้บริบท ให้ context_guidance เป็น string ว่าง
- ตอบกลับ JSON เท่านั้นตาม schema

Skeleton ตั้งต้น:
{json.dumps(skeletons, ensure_ascii=False)}
"""
            try:
                raw_text = await asyncio.wait_for(
                    chat_service._call_gemini_chat(
                        prompt,
                        response_format=skeleton_response_format,
                        model_name=resolved_model,
                        litellm_user=litellm_user,
                        litellm_metadata=litellm_metadata,
                    ),
                    timeout=max(30, int(settings.gemini_timeout or 60)),
                )
                parsed = json.loads(extract_json(raw_text))
                raw_variants = (
                    parsed.get("skeletons") if isinstance(parsed, dict) else []
                )
                if not isinstance(raw_variants, list):
                    raw_variants = []
                variants: List[Dict[str, Any]] = []
                for index, item in enumerate(raw_variants):
                    normalized = _normalize_seed_skeleton(item)
                    if normalized:
                        source_skeleton = skeletons[index % original_count]
                        source_context_guidance = str(
                            source_skeleton.get("context_guidance") or ""
                        ).strip()
                        if (
                            source_context_guidance
                            and not str(
                                normalized.get("context_guidance") or ""
                            ).strip()
                        ):
                            normalized["context_guidance"] = source_context_guidance
                        variants.append(normalized)
                    if len(variants) >= missing_count:
                        break
                pool = [*skeletons, *variants]
                if len(variants) < missing_count:
                    app_logger.warning(
                        "Skeleton upsampling returned fewer variants than requested: "
                        f"{len(variants)}/{missing_count}"
                    )
                random.shuffle(pool)
                app_logger.info(
                    "Skeleton pool prepared: "
                    f"original={original_count} requested_pool={pool_target} actual={len(pool)}"
                )
                return pool
            except Exception as upsample_error:
                app_logger.warning(
                    f"Skeleton upsampling failed; falling back to original skeletons: {upsample_error}"
                )
                pool = skeletons[:]
                random.shuffle(pool)
                return pool

        def compact_explanation(text: str) -> str:
            text = re.sub(r"\s+", " ", (text or "").strip())
            if not text:
                return ""
            sentence_match = [
                item.strip()
                for item in re.split(r"(?<=[.!?])\s+", text)
                if item.strip()
            ]
            if not sentence_match:
                return text[:800].rstrip()
            # Keep enough detail for students: use up to first 5 sentences.
            concise = " ".join(sentence_match[:5]).strip()
            if len(concise) < 30:
                concise = text
            return concise[:800].rstrip()

        def normalize_math_artifacts(text: Any) -> str:
            value = str(text or "")
            if not value:
                return ""

            # Recover common escaped control chars from malformed JSON strings.
            for ch in (chr(8), chr(9), chr(10), chr(12), chr(13)):
                value = value.replace(ch, "\\")

            value = re.sub(r"\\imes\b", r"\\times", value)
            value = re.sub(r"\\hickspace\b", r"\\thickspace", value)
            value = re.sub(r"\\ext\{", r"\\text{", value)
            value = re.sub(r"\\rac\{", r"\\frac{", value)
            value = re.sub(r"\bextdiv\b", r"\\div", value)
            value = re.sub(r"\bimes\b", r"\\times", value)
            value = re.sub(r"\bhickspace\b", r"\\thickspace", value)
            value = re.sub(r"\bext\{", r"\\text{", value)
            value = re.sub(r"\brac\{", r"\\frac{", value)

            # Seen in payloads where division token degrades to "lat" between groups.
            value = re.sub(r"(?<=[\]\)\d])\s+lat\s+(?=[\[\(\-\d])", r" \\div ", value)
            return value.strip()

        def is_placeholder_context(text: str) -> bool:
            compact = re.sub(r"\s+", " ", str(text or "").strip())
            if not compact:
                return False
            lowered = compact.lower()

            # Generic section headers from extracted documents that should not be used
            # as question context.
            generic_patterns = [
                r"^(vocabulary|grammar|reading comprehension|conversation(?:\s*/\s*communication)?)(\s*\(.*\))?$",
                r"^(vocabulary|grammar|reading comprehension|conversation(?:\s*/\s*communication)?)\s*[\(\[]?\s*(items?|questions?)\s*\d+\s*[-–]\s*\d+\s*[\)\]]?$",
                r"^(items?|questions?)\s*\d+\s*[-–]\s*\d+$",
                r"^(คำศัพท์|ไวยากรณ์|อ่านจับใจความ|บทสนทนา)\s*[\(\[]?\s*(ข้อ)?\s*\d+\s*[-–]\s*\d+\s*[\)\]]?$",
                r"^(ข้อ)?\s*\d+\s*[-–]\s*\d+$",
            ]
            for pattern in generic_patterns:
                if re.fullmatch(pattern, lowered):
                    return True
            return False

        def normalize_question(
            raw_q: Dict[str, Any],
            fallback_difficulty: int,
        ) -> Dict[str, Any]:
            question_text = normalize_math_artifacts(raw_q.get("question"))
            context_text = normalize_math_artifacts(raw_q.get("context"))[:2500]
            choices = raw_q.get("choices") or []
            explanation = compact_explanation(
                normalize_math_artifacts(raw_q.get("explanation") or "")
            )
            correct = raw_q.get("correct_answer")
            qdiff = raw_q.get("difficulty")

            if not isinstance(choices, list):
                choices = []
            choices = [normalize_math_artifacts(c) for c in choices if str(c).strip()][
                :4
            ]
            while len(choices) < 4:
                choices.append("")

            if not isinstance(correct, int) or correct < 0 or correct > 3:
                correct = 0
            if not question_text:
                raise ValueError("Empty question text")

            if context_text and is_placeholder_context(context_text):
                context_text = ""

            normalized_question = {
                "question": question_text,
                "choices": choices,
                "correct_answer": correct,
                "explanation": explanation or None,
                "difficulty": int(qdiff)
                if isinstance(qdiff, int) and 1 <= qdiff <= 5
                else fallback_difficulty,
            }
            if context_text:
                normalized_question["context"] = context_text
            return normalized_question

        def build_difficulty_plan(
            total_questions_per_set: int,
            strategy: str,
            single_difficulty: int,
        ) -> List[int]:
            count = max(0, int(total_questions_per_set or 0))
            if count == 0:
                return []
            if strategy != "balanced":
                return [single_difficulty] * count

            buckets = [1, 3, 5]  # easy, medium, hard
            base_count = count // len(buckets)
            remainder = count % len(buckets)
            counts_by_bucket = {value: base_count for value in buckets}
            if remainder > 0:
                remainder_pick = buckets[:]
                random.shuffle(remainder_pick)
                for bucket in remainder_pick[:remainder]:
                    counts_by_bucket[bucket] += 1

            plan: List[int] = []
            for bucket in buckets:
                plan.extend([bucket] * counts_by_bucket[bucket])
            random.shuffle(plan)
            return plan

        max_concurrency = max(1, settings.quiz_gen_max_concurrency)
        semaphore = asyncio.Semaphore(
            max(1, min(max_concurrency, num_questions * num_sets))
        )
        per_question_timeout = max(15, int(settings.gemini_timeout or 60))
        max_retry_rounds = max(1, int(settings.gemini_max_retries or 3))
        accepted_questions: List[Dict[str, Any]] = []
        accepted_fingerprints: Set[str] = set()
        accepted_lock = asyncio.Lock()

        def clip_text(value: Any, max_len: int = 280) -> str:
            text = re.sub(r"\s+", " ", str(value or "").strip())
            if len(text) <= max_len:
                return text
            return f"{text[:max_len]}..."

        def _normalize_template_original(
            template_example: Optional[Dict[str, Any]],
        ) -> Optional[Dict[str, Any]]:
            if not isinstance(template_example, dict):
                return None
            question_text = re.sub(
                r"\s+", " ", str(template_example.get("question") or "").strip()
            )
            raw_choices = (
                template_example.get("choices")
                if isinstance(template_example.get("choices"), list)
                else []
            )
            choices = [
                re.sub(r"\s+", " ", str(choice or "").strip())
                for choice in raw_choices
                if str(choice or "").strip()
            ][:4]
            if not question_text or len(choices) < 4:
                return None

            raw_correct = template_example.get(
                "correct_answer", template_example.get("correctAnswer", 0)
            )
            try:
                correct_answer = int(raw_correct)
            except (TypeError, ValueError):
                correct_answer = 0
            correct_answer = min(3, max(0, correct_answer))
            context_text = re.sub(
                r"\s+", " ", str(template_example.get("context") or "").strip()
            )
            explanation_text = re.sub(
                r"\s+", " ", str(template_example.get("explanation") or "").strip()
            )
            normalized_original = {
                "question": question_text,
                "choices": choices,
                "correct_answer": correct_answer,
            }
            if context_text:
                normalized_original["context"] = context_text[:800]
            if explanation_text:
                normalized_original["explanation"] = explanation_text[:500]
            return normalized_original

        def _build_verification_result(
            verification_status: str,
            *,
            is_correct: Optional[bool],
            too_similar_to_template: Optional[bool],
            similarity_score_0_to_10: Optional[float],
            verdict_reason: str,
            similarity_threshold_0_to_10: Optional[float] = None,
            correctness_issues: Optional[List[str]] = None,
            similarity_issues: Optional[List[str]] = None,
            model_name: Optional[str] = None,
            judge_results: Optional[List[Dict[str, Any]]] = None,
        ) -> Dict[str, Any]:
            checked_at = datetime.utcnow().isoformat()
            return {
                "verification_status": verification_status,
                "verification": {
                    "is_correct": is_correct,
                    "too_similar_to_template": too_similar_to_template,
                    "similarity_score": similarity_score_0_to_10,
                    "similarity_score_0_to_10": similarity_score_0_to_10,
                    "similarity_score_scale": "0_to_10",
                    "similarity_threshold_0_to_10": similarity_threshold_0_to_10,
                    "correctness_issues": correctness_issues or [],
                    "similarity_issues": similarity_issues or [],
                    "verdict_reason": str(verdict_reason or "").strip(),
                    "model": model_name or default_quiz_verify_model,
                    "judge_results": judge_results or [],
                    "checked_at": checked_at,
                },
            }

        async def verify_generated_question(
            template_example: Optional[Dict[str, Any]],
            generated_question: Dict[str, Any],
            set_index: int,
            question_index: int,
        ) -> Dict[str, Any]:
            has_template_example = bool(template_example)
            if not has_template_example and not judge_verification_requested:
                return _build_verification_result(
                    "not_applicable",
                    is_correct=None,
                    too_similar_to_template=None,
                    similarity_score_0_to_10=None,
                    verdict_reason="template_missing",
                    model_name=default_quiz_verify_model,
                )

            similarity_rejection_threshold = 9.5
            if has_template_example:
                judge_prompt = f"""
คุณเป็นผู้ตรวจข้อสอบ (LLM as a judge) สำหรับโจทย์ที่ดัดแปลงจาก template

พิจารณา 2 มิติพร้อมกัน:
1) ความถูกต้องของโจทย์ใหม่: คำถาม/ตัวเลือก/เฉลย/คำอธิบายสัมพันธ์กัน ถูกหลักวิชา และมีคำตอบถูกเพียงข้อเดียว รวมถึงความสอดคล้องของ context
2) การคัดลอก template ต้นฉบับ: ให้ประเมินเป็นคะแนน similarity_score_0_to_10 โดยเน้นการคัดลอกถ้อยคำ ชื่อ ตัวเลข และตัวเลือกแบบแทบตรงกัน

เกณฑ์:
- is_correct=true เฉพาะเมื่อโจทย์ใหม่ถูกต้องโดยรวม
- ห้ามใช้ความยากจริงหรือความสอดคล้องกับ difficulty เป็นเหตุให้ is_correct=false
- โจทย์ใหม่ควรรักษาประเภทคำถาม โครงเรื่อง จำนวนขั้นคิด ระดับความยาก และความยาวใกล้ต้นฉบับ การคล้ายกันในส่วนเหล่านี้เป็นสิ่งที่ต้องการและไม่ใช่เหตุให้คะแนนสูง
- similarity_score_0_to_10 เป็นคะแนน 0 ถึง 10 โดย 0 = เปลี่ยนถ้อยคำ/ชื่อ/ตัวเลข/ตัวเลือกอย่างเหมาะสม, 10 = คัดลอกข้อความหรือรายละเอียดผิวหน้าแทบทั้งหมด
- ให้คะแนนสูงเฉพาะเมื่อยังคัดลอกประโยค ชื่อเฉพาะ ตัวเลข หรือ choices ใกล้ต้นฉบับจนเห็นได้ชัด
- คะแนน {similarity_rejection_threshold:g} ขึ้นไปถือว่าคัดลอกต้นฉบับมากเกินไปสำหรับระบบนี้
- correctness_issues/similarity_issues ใส่เหตุผลสั้นๆ เป็นรายการข้อความ
- verdict_reason สรุปสั้นๆ
- ถ้า question หรือ context ยาวกว่าต้นฉบับอย่างชัดเจนโดยไม่มีความจำเป็น ให้ถือว่าไม่ถูกต้อง
- ถ้า context ไม่ว่าง ต้องเกี่ยวข้องโดยตรงกับโจทย์และตัวเลือก และต้องช่วยให้ตอบได้จริง
- ถ้า context เป็นเพียงหัวข้อ/ป้าย เช่น "Vocabulary (Items 1-10)" หรือเป็นข้อความทั่วไปที่ไม่ช่วยตอบ ให้ถือว่าไม่ถูกต้อง
- ถ้าข้อคำถามต้องพึ่ง passage/table/dialogue แต่ context ว่างหรือไม่พอ ให้ถือว่าไม่ถูกต้อง
- ถ้าโจทย์อ้างถึงรูป ภาพ แผนภาพ กราฟ หรือสื่อภายนอกที่ไม่ได้ให้ข้อมูลครบเป็นข้อความ ให้ถือว่าไม่ถูกต้อง
- อ่าน question, context และ choices รวมกันแล้วต้องเป็นโจทย์ที่สมบูรณ์ อ่านรู้เรื่อง และมีคำสั่งชัดเจนว่าผู้เรียนต้องเลือก/หาอะไร
- ถ้า question เป็นเพียงรายการคำหรือวลีที่คั่นด้วย "/", comma หรือบรรทัดใหม่ และ choices ซ้ำรายการนั้น โดยไม่มีคำสั่ง เช่น definition, synonym, odd-one-out, translation หรือ fill-in-the-blank ให้ถือว่าไม่ถูกต้อง

Template original:
{json.dumps(template_example, ensure_ascii=False)}

Generated question:
{json.dumps(generated_question, ensure_ascii=False)}
"""
            else:
                judge_prompt = f"""
คุณเป็นผู้ตรวจข้อสอบ (LLM as a judge) สำหรับโจทย์ที่สร้างด้วย AI

ตรวจความถูกต้องของคำถาม ตัวเลือก เฉลย และคำอธิบายว่าถูกหลักวิชา
สัมพันธ์กัน และมีคำตอบถูกเพียงข้อเดียว รวมถึงตรวจว่า context เพียงพอ
และเกี่ยวข้องโดยตรงหากโจทย์จำเป็นต้องใช้ context

- is_correct=true เฉพาะเมื่อโจทย์ถูกต้องโดยรวม
- ห้ามใช้ความยากจริงหรือความสอดคล้องกับ difficulty เป็นเหตุให้ is_correct=false
- correctness_issues ใส่เหตุผลสั้นๆ เป็นรายการข้อความ
- similarity_score_0_to_10 ให้ตอบ 0 เพราะไม่มี template สำหรับเปรียบเทียบ
- similarity_issues ให้ตอบเป็นรายการว่าง
- verdict_reason สรุปสั้นๆ
- ถ้าโจทย์อ้างถึงรูป ภาพ แผนภาพ กราฟ หรือสื่อภายนอกที่ไม่ได้ให้ข้อมูลครบเป็นข้อความ ให้ถือว่าไม่ถูกต้อง
- อ่าน question, context และ choices รวมกันแล้วต้องเป็นโจทย์ที่สมบูรณ์ อ่านรู้เรื่อง และมีคำสั่งชัดเจนว่าผู้เรียนต้องเลือก/หาอะไร
- ถ้า question เป็นเพียงรายการคำหรือวลีที่คั่นด้วย "/", comma หรือบรรทัดใหม่ และ choices ซ้ำรายการนั้น โดยไม่มีคำสั่ง เช่น definition, synonym, odd-one-out, translation หรือ fill-in-the-blank ให้ถือว่าไม่ถูกต้อง

Generated question:
{json.dumps(generated_question, ensure_ascii=False)}
"""
            judge_results: List[Dict[str, Any]] = []
            judge_errors: List[str] = []
            for judge_model_name in judge_model_candidates:
                try:
                    raw_text = await asyncio.wait_for(
                        chat_service._call_gemini_chat(
                            judge_prompt,
                            response_format=verification_response_format,
                            model_name=judge_model_name,
                            litellm_user=litellm_user,
                            litellm_metadata=litellm_metadata,
                        ),
                        timeout=max(20, per_question_timeout),
                    )
                    parsed = json.loads(extract_json(raw_text))
                    if not isinstance(parsed, dict):
                        raise ValueError("judge response is not an object")

                    is_correct = bool(parsed.get("is_correct"))
                    score_raw = parsed.get("similarity_score_0_to_10")
                    if score_raw is None:
                        score_raw = parsed.get("similarity_score")
                    similarity_score_0_to_10: Optional[float] = None
                    too_similar_to_template: Optional[bool] = None
                    if has_template_example:
                        try:
                            similarity_score_0_to_10 = float(score_raw)
                        except (TypeError, ValueError):
                            similarity_score_0_to_10 = 10.0
                        if (
                            0.0 <= similarity_score_0_to_10 <= 1.0
                            and score_raw == parsed.get("similarity_score")
                        ):
                            similarity_score_0_to_10 *= 10.0
                        similarity_score_0_to_10 = max(
                            0.0, min(10.0, similarity_score_0_to_10)
                        )
                        too_similar_to_template = (
                            similarity_score_0_to_10 >= similarity_rejection_threshold
                        )

                    def _to_issue_list(value: Any) -> List[str]:
                        if not isinstance(value, list):
                            return []
                        return [
                            str(item).strip() for item in value if str(item).strip()
                        ][:8]

                    correctness_issues = _to_issue_list(
                        parsed.get("correctness_issues")
                    )
                    similarity_issues = _to_issue_list(parsed.get("similarity_issues"))
                    verdict_reason = str(parsed.get("verdict_reason") or "").strip()
                    judge_results.append(
                        {
                            "model": judge_model_name,
                            "is_correct": is_correct,
                            "too_similar_to_template": too_similar_to_template,
                            "similarity_score": similarity_score_0_to_10,
                            "similarity_score_0_to_10": similarity_score_0_to_10,
                            "similarity_score_scale": "0_to_10",
                            "correctness_issues": correctness_issues,
                            "similarity_issues": similarity_issues,
                            "verdict_reason": verdict_reason or "judge_completed",
                        }
                    )
                except Exception as verify_error:
                    judge_errors.append(f"{judge_model_name}: {verify_error}")
                    app_logger.warning(
                        "Quiz verification failed for generated question: "
                        f"set={set_index} q={question_index} model={judge_model_name} err={verify_error}"
                    )

            if judge_errors:
                app_logger.warning(
                    "Quiz verification failed due to judge model errors: "
                    f"set={set_index} q={question_index} models={judge_model_candidates} errors={judge_errors}"
                )
                return _build_verification_result(
                    "unverified",
                    is_correct=None,
                    too_similar_to_template=None,
                    similarity_score_0_to_10=None,
                    verdict_reason="judge_error",
                    correctness_issues=["judge_error", *judge_errors][:12],
                    similarity_issues=[],
                    model_name=", ".join(judge_model_candidates),
                    judge_results=judge_results,
                )

            if not judge_results:
                return _build_verification_result(
                    "unverified",
                    is_correct=None,
                    too_similar_to_template=None,
                    similarity_score_0_to_10=None,
                    verdict_reason="judge_unavailable",
                    correctness_issues=["judge_unavailable"],
                    similarity_issues=[],
                    model_name=", ".join(judge_model_candidates),
                )

            final_is_correct = all(
                bool(item.get("is_correct")) for item in judge_results
            )
            final_too_similar = any(
                bool(item.get("too_similar_to_template")) for item in judge_results
            )
            final_similarity_score = (
                max(
                    float(item.get("similarity_score_0_to_10") or 0.0)
                    for item in judge_results
                )
                if has_template_example
                else None
            )

            merged_correctness_issues: List[str] = []
            merged_similarity_issues: List[str] = []
            for item in judge_results:
                for issue in item.get("correctness_issues") or []:
                    normalized_issue = str(issue or "").strip()
                    if (
                        normalized_issue
                        and normalized_issue not in merged_correctness_issues
                    ):
                        merged_correctness_issues.append(normalized_issue)
                for issue in item.get("similarity_issues") or []:
                    normalized_issue = str(issue or "").strip()
                    if (
                        normalized_issue
                        and normalized_issue not in merged_similarity_issues
                    ):
                        merged_similarity_issues.append(normalized_issue)

            final_status = (
                "verified"
                if final_is_correct and not final_too_similar
                else "unverified"
            )
            final_reason = (
                "all_judges_passed"
                if final_status == "verified"
                else "one_or_more_judges_rejected"
            )
            return _build_verification_result(
                final_status,
                is_correct=final_is_correct,
                too_similar_to_template=(
                    final_too_similar if has_template_example else None
                ),
                similarity_score_0_to_10=final_similarity_score,
                similarity_threshold_0_to_10=(
                    similarity_rejection_threshold if has_template_example else None
                ),
                verdict_reason=final_reason,
                correctness_issues=merged_correctness_issues[:12],
                similarity_issues=merged_similarity_issues[:12],
                model_name=", ".join(judge_model_candidates),
                judge_results=judge_results,
            )

        async def generate_one_question(
            set_index: int,
            question_index: int,
            question_topic: str,
            question_difficulty: int,
            retry_round: int = 0,
            template_example: Optional[Dict[str, Any]] = None,
            avoid_questions: Optional[List[Dict[str, Any]]] = None,
        ) -> Dict[str, Any]:
            template_example_block = ""
            template_length_block = ""
            question_length_limit: Optional[int] = None
            context_length_limit: Optional[int] = None
            if template_example:
                original_question_length = len(
                    str(template_example.get("question") or "").strip()
                )
                original_context_length = len(
                    str(template_example.get("context") or "").strip()
                )
                question_length_limit = max(
                    80,
                    original_question_length + 60,
                    int(original_question_length * 1.5),
                )
                context_length_limit = (
                    max(
                        160,
                        original_context_length + 100,
                        int(original_context_length * 1.35),
                    )
                    if original_context_length and not exclude_context
                    else 0
                )
                template_example_block = (
                    "\nTemplate ต้นฉบับที่ต้องใช้เป็นฐานสำหรับแปลงโจทย์ "
                    "ให้ดู question, choices, correct_answer, explanation และ context (ถ้ามี) "
                    "เพื่อรักษาเป้าหมายการวัด ระดับชั้น ระดับความยาก โครงสร้างตัวเลือก และแนวคำอธิบายเฉลย "
                    "ผลลัพธ์ต้องคงประเภทคำถาม โครงสถานการณ์ จำนวนขั้นคิด และความยาวใกล้ต้นฉบับ "
                    "โดยเปลี่ยนเฉพาะถ้อยคำ ชื่อ ตัวเลข หรือ choices เท่าที่จำเป็น:\n"
                    f"{json.dumps(template_example, ensure_ascii=False)}\n"
                )
                template_length_block = (
                    "\nข้อจำกัดความยาวสำหรับ template นี้:\n"
                    f"- question ต้องยาวไม่เกินประมาณ {question_length_limit} ตัวอักษร\n"
                    + (
                        '- ห้ามสร้าง context สำหรับโจทย์นี้ ให้ตอบ context เป็น ""\n'
                        if exclude_context
                        else f"- context ต้องยาวไม่เกินประมาณ {context_length_limit} ตัวอักษร และคงรูปแบบเดิม\n"
                        if context_length_limit
                        else '- ต้นฉบับไม่มี context จึงต้องตอบ context เป็น ""\n'
                    )
                )
            avoid_block = ""
            if avoid_questions:
                examples = []
                for item in avoid_questions[-6:]:
                    examples.append(
                        {
                            "context": clip_text(item.get("context"), 180),
                            "question": clip_text(item.get("question"), 220),
                            "choices": [
                                clip_text(choice, 80)
                                for choice in (item.get("choices") or [])[:4]
                            ],
                        }
                    )
                avoid_block = (
                    "\nโจทย์ที่รับไปแล้วใน request นี้ ห้ามคัดลอกข้อความหรือคำตอบแทบตรงกัน "
                    "แต่อนุญาตให้ใช้ประเภทคำถามและรูปแบบเดียวกันได้:\n"
                    f"{json.dumps(examples, ensure_ascii=False)}\n"
                )
            prompt = f"""
{system_instructions}

หัวข้อหลัก: {topic.strip()}
หัวข้อย่อยที่ต้องเน้น: {question_topic}
ระดับชั้น: {grade_level.strip()}
ภาษา output สำหรับ question และ choices: {output_language_name}
ความยากเป้าหมาย: {question_difficulty} จาก 5 (1=ง่าย, 5=ยาก)
ชุดที่: {set_index}
ข้อที่: {question_index}
รอบ: {retry_round + 1}
{template_example_block}
{template_length_block}
{avoid_block}

เงื่อนไข:
- สร้างโจทย์ 1 ข้อเท่านั้น
- ฟิลด์ question และ choices ต้องเขียนเป็นภาษา {output_language_name}
- ฟิลด์ explanation ต้องเขียนเป็นภาษาไทยเท่านั้นเสมอ ห้ามใช้ภาษาอังกฤษเป็นหลัก
- โจทย์ต้องไม่ซ้ำข้อความและคำตอบแบบตรงๆ กับข้ออื่นในชุดเดียวกัน แต่ใช้รูปแบบโจทย์เดียวกันได้
- มี 4 ตัวเลือก และเฉลยถูกต้องเพียงข้อเดียว
- โจทย์ต้องตอบได้จากข้อความใน question, context และ choices เท่านั้น ห้ามต้องดูรูป ภาพ แผนภาพ กราฟ หรือสื่อภายนอก
- อ่าน question, context และ choices รวมกันแล้วต้องเป็นโจทย์ที่สมบูรณ์ อ่านรู้เรื่อง และมีคำสั่งชัดเจนว่าผู้เรียนต้องเลือก/หาอะไร
- ห้ามให้ question เป็นเพียงรายการคำหรือวลีที่คั่นด้วย "/", comma หรือบรรทัดใหม่แล้วซ้ำกับ choices ต้องมีคำสั่ง เช่น definition, synonym, odd-one-out, translation หรือ fill-in-the-blank ให้ชัดเจน
- ห้ามใช้ถ้อยคำอ้างภาพ เช่น "จากรูป", "ดังภาพ", "แผนภาพต่อไปนี้", "กราฟด้านล่าง", "figure below" หรือให้รูปภาพเป็นตัวเลือก
- หาก template ต้นฉบับต้องใช้รูปภาพ ให้แปลงข้อมูลที่จำเป็นเป็นข้อความให้ครบและกระชับ; ถ้าแปลงไม่ได้ ให้สร้างโจทย์ใหม่ในหัวข้อและระดับความยากเดียวกันที่ไม่ต้องใช้รูป
- หากมี template ต้นฉบับ ให้รักษาประเภทคำถาม โครงสถานการณ์ วิธีคิด จำนวนขั้นตอน และระดับความยากให้ใกล้ต้นฉบับ
- ดัดแปลงแบบพอดี โดยเลือกเปลี่ยนเพียง 1-2 ส่วน เช่น paraphrase ประโยค เปลี่ยนชื่อ เปลี่ยนตัวเลข หรือปรับ choices และตัวลวง
- ไม่จำเป็นต้องเปลี่ยนฉาก โครงเรื่อง หรือลำดับเหตุการณ์ทั้งหมด ถ้าการเปลี่ยนดังกล่าวทำให้โจทย์ยาวขึ้นหรือห่างจากต้นฉบับ
- question และ context ต้องกระชับและมีความยาวใกล้ต้นฉบับ ห้ามเติมเรื่องราว เงื่อนไข หรือข้อมูลรบกวนที่ไม่จำเป็น
- ห้ามคัดลอกข้อความต้นฉบับตรงๆ ทั้งประโยค แต่อนุญาตให้รักษาคำศัพท์เฉพาะวิชาและรูปแบบคำถามเดิม
- เขียน choices และ explanation ให้สอดคล้องกับส่วนที่ดัดแปลง โดยรักษารูปแบบคำตอบและตัวลวงเดิม
- ห้ามเพิ่มขั้นคิดใหม่หรือเงื่อนไขซ้อนที่ทำให้โจทย์ยากกว่า "ความยากเป้าหมาย"
- สร้างโจทย์ให้ตรงกับความยากเป้าหมายระดับ {question_difficulty} ตาม rubric ด้านล่าง
- ตัวเลือกผิดต้องสะท้อนข้อผิดพลาดที่เหมาะกับระดับความยาก ไม่ใช่ผิดแบบเห็นได้ทันทีทุกข้อ
- ฟิลด์ context ต้องเป็น string เสมอ
{context_rule_block}
- ใช้ context เฉพาะเมื่อจำเป็นจริงต่อการตอบ เช่น reading passage, ตารางข้อมูล, บทสนทนา หรือคำสั่งร่วม; ถ้าไม่จำเป็นให้ตั้ง context เป็น ""
- ถ้ามี context ต้องสอดคล้องกับ question และ choices โดยตรง และคำตอบที่ถูกต้องต้องอ้างอิง context ได้
- ห้ามใช้ context แบบหัวข้อหรือป้ายทั่วไป เช่น "Vocabulary (Items 1-10)", "Questions 1-5" หรือข้อความที่ไม่ช่วยตอบโจทย์
{template_context_rule}
- คำอธิบายเฉลยต้องสอนวิธีคิดให้ผู้เรียนเข้าใจง่าย ความยาว 3-5 ประโยค (ประมาณ 120-600 ตัวอักษร)
- ควรระบุหลักคิด, ขั้นตอนคำนวณสำคัญ, และจุดที่มักผิดพลาดอย่างน้อย 1 จุด
- ถ้ามีสัญลักษณ์คณิตศาสตร์ ให้เขียนในรูป LaTeX ที่ render ได้ทันที
- ยกกำลังต้องใช้รูปแบบเช่น $x^{{2}}$ ไม่ใช้ x2 หรือ x^2 แบบไม่มี $...$
- เศษส่วนใช้ $\\frac{{a}}{{b}}$ และการคูณ/หารใช้ $\\times$, $\\div$

{difficulty_rubric}
{extra_block}

ตอบกลับ JSON object เท่านั้น:
{{
  "question": "ข้อความคำถาม",
  "context": "บริบทประกอบโจทย์ ถ้าไม่มีให้เป็น string ว่าง",
  "choices": ["ตัวเลือกที่ 1", "ตัวเลือกที่ 2", "ตัวเลือกที่ 3", "ตัวเลือกที่ 4"],
  "correct_answer": 0,
  "explanation": "อธิบายเหตุผลแบบเข้าใจง่าย 3-5 ประโยค พร้อมหลักคิด ขั้นตอนสำคัญ และจุดที่มักผิดพลาด",
  "difficulty": {question_difficulty}
}}
"""
            async with semaphore:
                try:
                    raw_text = await asyncio.wait_for(
                        chat_service._call_gemini_chat(
                            prompt,
                            response_format=question_response_format,
                            model_name=resolved_model,
                            litellm_user=litellm_user,
                            litellm_metadata=litellm_metadata,
                        ),
                        timeout=per_question_timeout,
                    )
                except asyncio.TimeoutError as ex:
                    raise TimeoutError(
                        f"AI timeout set={set_index} q={question_index} after {per_question_timeout}s"
                    ) from ex

            json_str = extract_json(raw_text)
            try:
                payload: Dict[str, Any] = json.loads(json_str)
            except json.JSONDecodeError as ex:
                raise ValueError(f"Invalid JSON from AI: {ex}") from ex

            if (
                isinstance(payload, dict)
                and isinstance(payload.get("questions"), list)
                and payload["questions"]
            ):
                payload = payload["questions"][0]

            if not isinstance(payload, dict):
                raise ValueError("Unexpected AI response format")

            normalized = normalize_question(
                payload,
                fallback_difficulty=question_difficulty,
            )
            # The system difficulty plan is authoritative; do not trust the model's
            # self-reported difficulty value.
            normalized["difficulty"] = question_difficulty
            visual_reference = _find_external_visual_reference(normalized)
            if visual_reference:
                raise ValueError(
                    "Generated question requires an external image or visual"
                )
            incomplete_reason = _find_incomplete_quiz_question_reason(normalized)
            if incomplete_reason:
                raise ValueError(
                    "Generated question is incomplete or unclear: "
                    f"{incomplete_reason}"
                )
            if (
                question_length_limit
                and len(normalized["question"]) > question_length_limit
            ):
                raise ValueError(
                    "Generated question is too long compared with the template: "
                    f"{len(normalized['question'])}>{question_length_limit}"
                )
            normalized_context = str(normalized.get("context") or "")
            if exclude_context and normalized_context:
                raise ValueError("Generated context is not allowed")
            if template_example and context_length_limit == 0 and normalized_context:
                raise ValueError(
                    "Generated context is not allowed when the template has none"
                )
            if context_length_limit and len(normalized_context) > context_length_limit:
                raise ValueError(
                    "Generated context is too long compared with the template: "
                    f"{len(normalized_context)}>{context_length_limit}"
                )
            normalized["topic_tag"] = question_topic
            verification_payload = await verify_generated_question(
                template_example=template_example,
                generated_question=normalized,
                set_index=set_index,
                question_index=question_index,
            )
            normalized["verification_status"] = verification_payload.get(
                "verification_status", "unverified"
            )
            normalized["verification"] = verification_payload.get("verification") or {}
            normalized_template_original = _normalize_template_original(
                template_example
            )
            if normalized_template_original:
                normalized["template_original"] = normalized_template_original
            normalized["_order"] = question_index
            return normalized

        async def generate_one_set(set_index: int) -> Dict[str, Any]:
            ok_items: List[Dict[str, Any]] = []
            difficulty_plan = build_difficulty_plan(
                num_questions, difficulty_strategy, difficulty
            )
            difficulty_by_index = {
                idx + 1: difficulty_plan[idx]
                for idx in range(min(len(difficulty_plan), num_questions))
            }

            async def run_with_index(
                question_index: int, retry_round: int
            ) -> Dict[str, Any]:
                try:
                    question_topic = topic_pool[
                        (question_index + set_index - 2) % len(topic_pool)
                    ]
                    question_difficulty = int(
                        difficulty_by_index.get(question_index, difficulty)
                    )
                    template_example = (
                        template_examples[
                            ((set_index - 1) * num_questions + question_index - 1)
                            % len(template_examples)
                        ]
                        if template_examples
                        else None
                    )
                    async with accepted_lock:
                        avoid_questions = accepted_questions[-6:]
                    item = await generate_one_question(
                        set_index=set_index,
                        question_index=question_index,
                        question_topic=question_topic,
                        question_difficulty=question_difficulty,
                        retry_round=retry_round,
                        template_example=template_example,
                        avoid_questions=avoid_questions,
                    )
                    return {"index": question_index, "item": item, "error": None}
                except Exception as err:
                    return {"index": question_index, "item": None, "error": err}

            pending_indexes: List[int] = [idx + 1 for idx in range(num_questions)]
            for retry_round in range(max_retry_rounds):
                if not pending_indexes:
                    break

                next_pending: List[int] = []
                # Keep the number of scheduled tasks bounded. Large requests used
                # to create up to 2,000 waiting tasks at once, all with the same
                # stale duplicate-avoidance snapshot.
                for chunk_start in range(0, len(pending_indexes), max_concurrency):
                    chunk_indexes = pending_indexes[
                        chunk_start : chunk_start + max_concurrency
                    ]
                    round_tasks = [
                        asyncio.create_task(run_with_index(idx, retry_round))
                        for idx in chunk_indexes
                    ]
                    for completed_task in asyncio.as_completed(round_tasks):
                        payload = await completed_task
                        if payload["error"] is None and payload["item"] is not None:
                            async with accepted_lock:
                                is_duplicate = _is_duplicate_quiz_question(
                                    payload["item"],
                                    accepted_questions,
                                    accepted_fingerprints,
                                )
                                if not is_duplicate:
                                    accepted_questions.append(payload["item"])
                                    accepted_fingerprints.add(
                                        _quiz_question_fingerprint(payload["item"])
                                    )
                            if is_duplicate:
                                question_index = int(payload["index"])
                                next_pending.append(question_index)
                                if retry_round + 1 < max_retry_rounds:
                                    app_logger.warning(
                                        f"AI question duplicate rejected: set={set_index} "
                                        f"q={question_index} round={retry_round + 1}/{max_retry_rounds}"
                                    )
                                else:
                                    app_logger.error(
                                        f"AI question duplicate rejected permanently: set={set_index} "
                                        f"q={question_index} round={retry_round + 1}/{max_retry_rounds}"
                                    )
                                continue
                            ok_items.append(payload["item"])
                            await _append_quiz_gen_question(
                                progress_job_id,
                                payload["item"],
                                set_index,
                                int(payload["index"]),
                            )
                            await _advance_quiz_gen_progress(progress_job_id, 1)
                        else:
                            question_index = int(payload["index"])
                            error_text = str(payload["error"] or "")
                            if (
                                "resource_exhausted" in error_text.lower()
                                or "quota exceeded" in error_text.lower()
                            ):
                                for task in round_tasks:
                                    if not task.done():
                                        task.cancel()
                                await asyncio.gather(
                                    *round_tasks, return_exceptions=True
                                )
                                raise HTTPException(
                                    status_code=429,
                                    detail=(
                                        f"AI model quota exceeded for {resolved_model}. "
                                        "Please choose another model or retry after the provider quota resets."
                                    ),
                                )
                            next_pending.append(question_index)
                            if retry_round + 1 < max_retry_rounds:
                                app_logger.warning(
                                    f"AI question failed: set={set_index} q={question_index} "
                                    f"round={retry_round + 1}/{max_retry_rounds} err={payload['error']}"
                                )
                            else:
                                app_logger.error(
                                    f"AI question failed permanently: set={set_index} q={question_index} "
                                    f"round={retry_round + 1}/{max_retry_rounds} err={payload['error']}"
                                )
                pending_indexes = next_pending

            if len(ok_items) < num_questions:
                app_logger.warning(
                    f"AI generated partially for set {set_index}: "
                    f"{len(ok_items)}/{num_questions}; failed indexes: {pending_indexes}"
                )

            if len(ok_items) == 0:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"AI generated 0/{num_questions} questions for set {set_index}; "
                        f"failed indexes: {pending_indexes}"
                    ),
                )

            ok_items.sort(key=lambda q: q.get("_order", 0))
            normalized_questions = []
            for q in ok_items[:num_questions]:
                q.pop("_order", None)
                normalized_questions.append(q)
            return {
                "set_index": set_index,
                "questions": normalized_questions,
                "requested_count": num_questions,
                "generated_count": len(normalized_questions),
                "failed_indexes": pending_indexes,
            }

        # Generate all sets in parallel, each set generates each question in parallel
        set_tasks = [generate_one_set(set_index=i + 1) for i in range(num_sets)]
        all_sets_payload = await asyncio.gather(*set_tasks)

        if persist_quiz:
            for set_payload in all_sets_payload:
                set_index = int(set_payload.get("set_index") or 1)
                verified_questions = [
                    question
                    for question in (set_payload.get("questions") or [])
                    if str(question.get("verification_status") or "").strip().lower()
                    == "verified"
                ]
                if not verified_questions:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"No verified questions available for set {set_index}; "
                            "quiz was not created"
                        ),
                    )
                set_payload["questions"] = verified_questions
                set_payload["generated_count"] = len(verified_questions)

        quizzes: List[Dict[str, Any]] = []
        quiz_ids: List[str] = []
        effective_course = course_id if course_id else "default-course"
        generated_total_count = 0
        failed_total_count = 0
        set_generation_summary: List[Dict[str, Any]] = []

        for set_payload in all_sets_payload:
            set_index = int(set_payload.get("set_index") or 1)
            normalized_questions = set_payload.get("questions") or []
            requested_count = int(set_payload.get("requested_count") or num_questions)
            generated_count = int(
                set_payload.get("generated_count") or len(normalized_questions)
            )
            failed_indexes = set_payload.get("failed_indexes") or []
            generated_total_count += generated_count
            failed_total_count += max(0, requested_count - generated_count)
            set_generation_summary.append(
                {
                    "set_index": set_index,
                    "requested_count": requested_count,
                    "generated_count": generated_count,
                    "failed_indexes": failed_indexes,
                    "is_partial": generated_count < requested_count,
                }
            )
            quiz_data = {
                "title": f"{title_text} - ชุดที่ {set_index}",
                "document_type": document_type or "exam",
                "questions": normalized_questions,
                "total_questions": len(normalized_questions),
                "difficulty": difficulty,
                "set_index": set_index,
            }
            if document_type == "mock_exam" and duration_minutes is not None:
                quiz_data["duration_minutes"] = int(duration_minutes)
            if extra_prompt and extra_prompt.strip():
                quiz_data["extra_prompt"] = extra_prompt.strip()
            if persist_quiz:
                quiz_id = await dynamodb_service.create_quiz(
                    user_id, effective_course, quiz_data
                )
                quiz = await dynamodb_service.get_quiz(quiz_id)
                quizzes.append(quiz)
                quiz_ids.append(quiz_id)
            else:
                quizzes.append(
                    {
                        "quiz_id": None,
                        "title": quiz_data["title"],
                        "document_type": quiz_data["document_type"],
                        "questions": normalized_questions,
                        "total_questions": len(normalized_questions),
                        "difficulty": difficulty,
                        "set_index": set_index,
                        "course_id": course_id,
                        "created_by": user_id,
                        "created_at": datetime.utcnow().isoformat(),
                        "persisted": False,
                    }
                )
                quiz_ids.append(f"ephemeral-{set_index}")

        if num_sets == 1:
            first_summary = set_generation_summary[0] if set_generation_summary else {}
            partial_generation = bool(first_summary.get("is_partial"))
            result_payload = {
                "message": (
                    "Quiz generated successfully (partial)"
                    if partial_generation and persist_quiz
                    else (
                        "Questions generated successfully (partial, not persisted)"
                        if partial_generation
                        else (
                            "Quiz generated successfully"
                            if persist_quiz
                            else "Questions generated successfully (not persisted)"
                        )
                    )
                ),
                "quiz_id": quiz_ids[0],
                "quiz": quizzes[0],
                "model": resolved_model,
                "partial_generation": partial_generation,
                "generation_summary": first_summary,
            }
            await _set_quiz_gen_progress(
                progress_job_id,
                {
                    "status": "completed_with_partial"
                    if partial_generation
                    else "completed",
                    "completed": generated_total_count,
                    "percent": 100,
                    "requested_total": total_target,
                    "generated_total": generated_total_count,
                    "failed_total": failed_total_count,
                },
            )
            return result_payload
        else:
            partial_generation = any(
                item.get("is_partial") for item in set_generation_summary
            )
            result_payload = {
                "message": (
                    "Multiple quiz sets generated successfully (some sets partial)"
                    if partial_generation and persist_quiz
                    else (
                        "Multiple question sets generated successfully (some sets partial, not persisted)"
                        if partial_generation
                        else (
                            "Multiple quiz sets generated successfully"
                            if persist_quiz
                            else "Multiple question sets generated successfully (not persisted)"
                        )
                    )
                ),
                "total": num_sets,
                "quiz_ids": quiz_ids,
                "quizzes": quizzes,
                "quiz_id": quiz_ids[0],  # backward compatibility
                "quiz": quizzes[0],  # backward compatibility
                "model": resolved_model,
                "partial_generation": partial_generation,
                "generation_summary": set_generation_summary,
            }
            await _set_quiz_gen_progress(
                progress_job_id,
                {
                    "status": "completed_with_partial"
                    if partial_generation
                    else "completed",
                    "completed": generated_total_count,
                    "percent": 100,
                    "requested_total": total_target,
                    "generated_total": generated_total_count,
                    "failed_total": failed_total_count,
                },
            )
            return result_payload

    except HTTPException:
        await _set_quiz_gen_progress(
            progress_job_id,
            {
                "status": "failed",
                "error": "validation_failed",
            },
        )
        raise
    except Exception as e:
        app_logger.error(f"Error generating quiz with AI: {e}")
        await _set_quiz_gen_progress(
            progress_job_id,
            {
                "status": "failed",
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail="Failed to generate quiz with AI")


@router.get("/quiz/generate/progress/{job_id}")
async def get_quiz_generate_progress(job_id: str):
    """Get real-time progress for quiz generation by job ID."""
    async with QUIZ_GEN_PROGRESS_LOCK:
        payload = QUIZ_GEN_PROGRESS.get(job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="progress job not found")
    return payload


@router.post("/courses/ai-generate", response_model=CourseAIGenerateResponse)
async def generate_course_details_with_ai(
    request: CourseAIGenerateRequest = Body(...),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Generate course details with AI from a prompt."""
    try:
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        meta_lines = []
        if request.name:
            meta_lines.append(f"ชื่อคอร์ส: {request.name}")
        if request.category:
            meta_lines.append(f"หมวดหมู่: {request.category}")
        topics = [str(item or "").strip() for item in (request.topics or [])]
        topics = [item for item in topics if item]
        if topics:
            meta_lines.append(f"หัวข้อในคอร์ส: {', '.join(topics[:20])}")
        content_items = []
        for item in request.content_items or []:
            title = str(item.get("title", "")).strip()
            description = str(item.get("description", "")).strip()
            if title or description:
                content_items.append({"title": title, "description": description})
        if content_items:
            meta_lines.append(
                "หัวข้อเนื้อหาที่ต้องเติมคำอธิบาย: "
                + json.dumps(content_items[:30], ensure_ascii=False)
            )
        meta_block = "\n".join(meta_lines)

        system_prompt = (
            "คุณเป็นผู้ช่วยครู เขียนรายละเอียดคอร์สภาษาไทยที่กระชับและใช้งานได้จริง\n"
            "ตอบกลับเป็น JSON เท่านั้น ด้วยคีย์: description, target_profile, structure_summary, content_items\n"
            "แต่ละคีย์เป็นข้อความสั้นๆ 1-3 บรรทัด ไม่เกิน 300 ตัวอักษรต่อช่อง\n"
            "content_items เป็น array ของ object ที่มี title และ description โดยคง title เดิม และเขียน description ภาษาไทยสั้นๆ 1 ประโยค ไม่เกิน 160 ตัวอักษร\n"
            "ถ้าไม่มีหัวข้อเนื้อหา ให้ส่ง content_items เป็น array ว่าง\n"
        )
        user_prompt = f"{meta_block}\nคำสั่งผู้ใช้: {prompt}".strip()
        full_prompt = f"{system_prompt}\n{user_prompt}"

        raw_text = await chat_service._call_gemini_chat(full_prompt)
        match = re.search(r"\{.*\}", raw_text or "", re.DOTALL)
        if not match:
            app_logger.error(f"AI did not return JSON. Raw: {(raw_text or '')[:300]}")
            raise HTTPException(status_code=502, detail="AI returned invalid JSON")

        data = json.loads(match.group(0))
        return CourseAIGenerateResponse(
            description=str(data.get("description", "")).strip(),
            target_profile=str(data.get("target_profile", "")).strip(),
            structure_summary=str(data.get("structure_summary", "")).strip(),
            content_items=[
                {
                    "title": str(item.get("title", "")).strip(),
                    "description": str(item.get("description", "")).strip(),
                }
                for item in (data.get("content_items") or [])
                if isinstance(item, dict)
                and (
                    str(item.get("title", "")).strip()
                    or str(item.get("description", "")).strip()
                )
            ],
        )
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        app_logger.error(f"AI JSON decode failed: {e}")
        raise HTTPException(status_code=502, detail="AI returned invalid JSON")
    except Exception as e:
        app_logger.error(f"Course AI generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate course details")


@router.post("/quiz/augment", response_model=QuizAugmentResponse)
async def augment_quiz_questions(
    payload: QuizAugmentRequest,
    augment_service: QuizAugmentService = Depends(get_quiz_augment_service),
):
    """Paraphrase and augment quiz questions with answers and explanations."""
    try:
        if not payload.questions:
            raise HTTPException(
                status_code=400, detail="Questions list cannot be empty"
            )
        result = await augment_service.augment_questions(
            payload.questions,
            payload.language or "th",
            payload.num_questions,
            payload.num_sets,
            payload.mode,
            payload.classify_topic_tag,
            payload.course_topics or [],
        )
        sets = result.get("sets")
        questions = result.get("questions")
        if not sets and questions:
            sets = [{"questions": questions}]
        first_questions = []
        if sets and isinstance(sets, list) and sets:
            first_questions = (
                sets[0].get("questions", []) if isinstance(sets[0], dict) else []
            )
        return QuizAugmentResponse(
            questions=first_questions, sets=sets, model=get_settings().litellm_model
        )
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error augmenting quiz questions: {e}")
        raise HTTPException(status_code=500, detail="Failed to augment quiz questions")


@router.post("/users/{user_id}/quizzes")
async def create_quiz_manual(
    user_id: str,
    quiz_payload: Dict[str, Any] = Body(
        ...,
        description="Quiz data: title, questions, total_questions, course_id(optional)",
    ),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Create a quiz manually (used by tutor_frontend QuizCreationForm)."""
    try:

        def normalize_difficulty(value: Any) -> int:
            if isinstance(value, (int, float)):
                n = int(round(float(value)))
                return min(5, max(1, n))
            raw = str(value or "").strip().lower()
            try:
                n = int(round(float(raw)))
                return min(5, max(1, n))
            except (TypeError, ValueError):
                pass
            if raw in {"easy", "ง่าย", "1"}:
                return 1
            if raw in {"medium", "ปานกลาง", "3"}:
                return 3
            if raw in {"hard", "ยาก", "5"}:
                return 5
            return 3

        title = (quiz_payload.get("title") or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="Title is required")
        questions = quiz_payload.get("questions") or []
        if not isinstance(questions, list) or len(questions) == 0:
            raise HTTPException(status_code=400, detail="Questions are required")
        course_id = quiz_payload.get("course_id") or quiz_payload.get("courseId")

        normalized_questions: List[Dict[str, Any]] = []
        for q in questions:
            if not isinstance(q, dict):
                continue
            qcopy = dict(q)
            qcopy["difficulty"] = normalize_difficulty(qcopy.get("difficulty"))
            normalized_questions.append(qcopy)

        if not normalized_questions:
            raise HTTPException(status_code=400, detail="No valid questions provided")

        difficulty_values = [int(q.get("difficulty", 3)) for q in normalized_questions]
        avg_difficulty = (
            (sum(difficulty_values) / len(difficulty_values))
            if difficulty_values
            else 3.0
        )
        computed_difficulty = normalize_difficulty(
            quiz_payload.get("difficulty", round(avg_difficulty))
        )

        quiz_data = {
            "title": title,
            "description": str(quiz_payload.get("description") or "").strip(),
            "document_type": quiz_payload.get("document_type") or "manual",
            "questions": normalized_questions,
            "total_questions": quiz_payload.get("total_questions")
            or len(normalized_questions),
            "difficulty": computed_difficulty,
            "difficulty_avg": round(avg_difficulty, 2),
        }
        # Pass-through optional mock-exam fields
        if (quiz_data.get("document_type") or "").lower() == "mock_exam" and isinstance(
            quiz_payload.get("duration_minutes"), (int, float)
        ):
            quiz_data["duration_minutes"] = int(quiz_payload["duration_minutes"])
        if (
            isinstance(quiz_payload.get("extra_prompt"), str)
            and quiz_payload["extra_prompt"].strip()
        ):
            quiz_data["extra_prompt"] = quiz_payload["extra_prompt"].strip()
        if (
            isinstance(quiz_payload.get("exam_details"), str)
            and quiz_payload["exam_details"].strip()
        ):
            quiz_data["exam_details"] = quiz_payload["exam_details"].strip()
        if (quiz_data.get("document_type") or "").lower() == "mock_exam":
            topic_weights = quiz_payload.get("topic_weights")
            if isinstance(topic_weights, dict):
                normalized_topic_weights = {
                    str(topic or "").strip(): max(0, min(100, int(weight)))
                    for topic, weight in topic_weights.items()
                    if str(topic or "").strip() and isinstance(weight, (int, float))
                }
                if normalized_topic_weights:
                    quiz_data["topic_weights"] = normalized_topic_weights
            topic_question_counts = quiz_payload.get("topic_question_counts")
            if isinstance(topic_question_counts, dict):
                normalized_topic_question_counts = {
                    str(topic or "").strip(): max(0, int(count))
                    for topic, count in topic_question_counts.items()
                    if str(topic or "").strip() and isinstance(count, (int, float))
                }
                if normalized_topic_question_counts:
                    quiz_data[
                        "topic_question_counts"
                    ] = normalized_topic_question_counts
        selection_reasons = (
            quiz_payload.get("selection_reasons")
            or quiz_payload.get("reasons")
            or quiz_payload.get("pick_reasons")
        )
        if selection_reasons is not None:
            if isinstance(selection_reasons, str):
                reasons_list = [
                    line.strip()
                    for line in selection_reasons.split("\n")
                    if line.strip()
                ]
            elif isinstance(selection_reasons, list):
                reasons_list = [
                    str(item).strip() for item in selection_reasons if str(item).strip()
                ]
            else:
                reasons_list = []
            quiz_data["selection_reasons"] = reasons_list[:5]

        effective_course = course_id if course_id else "default-course"
        quiz_id = await dynamodb_service.create_quiz(
            user_id, effective_course, quiz_data
        )
        quiz = await dynamodb_service.get_quiz(quiz_id)

        return {
            "message": "Quiz created successfully",
            "quiz_id": quiz_id,
            "quiz": quiz,
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error creating manual quiz: {e}")
        raise HTTPException(status_code=500, detail="Failed to create quiz")


# Course Management Endpoints
@router.post("/courses")
async def create_course(
    user_id: str = Form(...),
    name: str = Form(...),
    instructor: str = Form(""),
    course_format: str = Form(""),
    grade_level: str = Form(""),
    description: str = Form(""),
    detail: str = Form(""),
    target_profile: str = Form(""),
    structure_summary: str = Form(""),
    category: str = Form("general"),
    image_url: str = Form(""),
    thumbnail_url: str = Form(""),
    preview_image_url: str = Form(""),
    purchase_preview_image_url: str = Form(""),
    price: Optional[str] = Form(None),
    topics_json: Optional[str] = Form(None),
    course_tags_json: Optional[str] = Form(None),
    benefits_json: Optional[str] = Form(None),
    content_items_json: Optional[str] = Form(None),
    pricing_plans_json: Optional[str] = Form(None),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Create a new course."""
    try:
        app_logger.info(f"Creating course '{name}' for user: {user_id}")
        normalized_instructor = (instructor or "").strip() or "อาจารย์ระบบ"

        topics: List[str] = []
        if topics_json and topics_json.strip():
            try:
                parsed_topics = json.loads(topics_json)
                if not isinstance(parsed_topics, list):
                    raise ValueError("topics_json must be a JSON array")
                topics = [
                    str(item).strip() for item in parsed_topics if str(item).strip()
                ]
            except Exception:
                raise HTTPException(
                    status_code=400, detail="Invalid topics_json format"
                )

        course_tags: List[str] = []
        if course_tags_json and course_tags_json.strip():
            try:
                parsed_tags = json.loads(course_tags_json)
                if not isinstance(parsed_tags, list):
                    raise ValueError("course_tags_json must be a JSON array")
                course_tags = [
                    str(item).strip() for item in parsed_tags if str(item).strip()
                ]
            except Exception:
                raise HTTPException(
                    status_code=400, detail="Invalid course_tags_json format"
                )
        if not course_tags:
            course_tags = [DEFAULT_COURSE_SUBJECT]

        benefits: List[str] = []
        if benefits_json and benefits_json.strip():
            try:
                parsed_benefits = json.loads(benefits_json)
                if not isinstance(parsed_benefits, list):
                    raise ValueError("benefits_json must be a JSON array")
                benefits = [
                    str(item).strip() for item in parsed_benefits if str(item).strip()
                ]
            except Exception:
                raise HTTPException(
                    status_code=400, detail="Invalid benefits_json format"
                )

        content_items: List[Dict[str, str]] = []
        if content_items_json and content_items_json.strip():
            try:
                parsed_items = json.loads(content_items_json)
                if not isinstance(parsed_items, list):
                    raise ValueError("content_items_json must be a JSON array")
                normalized_items: List[Dict[str, str]] = []
                for item in parsed_items:
                    if not isinstance(item, dict):
                        continue
                    title = str(item.get("title", "")).strip()
                    content_description = str(item.get("description", "")).strip()
                    if not title and not content_description:
                        continue
                    normalized_items.append(
                        {
                            "title": title or "หัวข้อ",
                            "description": content_description,
                        }
                    )
                content_items = normalized_items
            except Exception:
                raise HTTPException(
                    status_code=400, detail="Invalid content_items_json format"
                )

        parsed_price = None
        if price is not None and str(price).strip() != "":
            try:
                parsed_price = float(str(price).strip())
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid price format")
            if parsed_price < 0:
                raise HTTPException(status_code=400, detail="Price must be >= 0")

        pricing_plans: List[Dict[str, Any]] = []
        if pricing_plans_json and pricing_plans_json.strip():
            try:
                parsed_plans = json.loads(pricing_plans_json)
                if not isinstance(parsed_plans, list):
                    raise ValueError("pricing_plans_json must be a JSON array")
                normalized_plans: List[Dict[str, Any]] = []
                for plan in parsed_plans:
                    if not isinstance(plan, dict):
                        continue
                    duration_months = int(plan.get("duration_months", 0))
                    plan_price = float(plan.get("price", 0))
                    plan_label = str(plan.get("label", "")).strip()
                    if duration_months <= 0 or plan_price < 0:
                        continue
                    normalized_plans.append(
                        {
                            "duration_months": duration_months,
                            "price": plan_price,
                            "label": plan_label or f"{duration_months} เดือน",
                        }
                    )
                pricing_plans = normalized_plans
            except Exception:
                raise HTTPException(
                    status_code=400, detail="Invalid pricing_plans_json format"
                )

        course_data = {
            "name": name,
            "instructor": normalized_instructor,
            "teacher_name": normalized_instructor,
            "course_format": (course_format or "").strip(),
            "grade_level": (grade_level or "").strip(),
            "description": description,
            "detail": detail,
            "target_profile": target_profile,
            "structure_summary": structure_summary,
            "category": category,
            "topics": topics,
            "tags": course_tags,
            "benefits": benefits,
            "content_items": content_items,
            "image_url": (image_url or "").strip(),
            "thumbnail_url": (thumbnail_url or "").strip(),
            "preview_image_url": (preview_image_url or "").strip(),
            "purchase_preview_image_url": (purchase_preview_image_url or "").strip(),
            "price": parsed_price,
            "pricing_plans": pricing_plans,
        }

        course_id = await dynamodb_service.create_course(user_id, course_data)

        # Get the created course to return full data
        course = await dynamodb_service.get_course(course_id)

        return {
            "message": "Course created successfully",
            "course_id": course_id,
            "course": course,
        }

    except Exception as e:
        app_logger.error(f"Error creating course: {e}")
        raise HTTPException(status_code=500, detail="Failed to create course")


@router.get("/courses/{course_id}")
async def get_course(course_id: str, dynamodb_service=Depends(get_dynamodb_service)):
    """Get course by ID."""
    try:
        course = await dynamodb_service.get_course(course_id)

        if not course:
            raise HTTPException(status_code=404, detail=f"Course {course_id} not found")

        return course

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error retrieving course {course_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve course")


@router.get("/courses/{course_id}/learning-overview")
async def get_course_learning_overview(
    course_id: str,
    user_id: Optional[str] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        STUDENT_BEARER_OPTIONAL
    ),
    student_auth_service: StudentAuthService = Depends(_get_student_auth_service),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Return compact student course detail data in one request."""
    try:
        if user_id:
            await _ensure_user_matches_token(
                user_id=user_id,
                credentials=credentials,
                auth_service=student_auth_service,
            )
            await _ensure_active_course_access(
                dynamodb_service=dynamodb_service,
                user_id=user_id,
                course_id=course_id,
            )

        get_overview = getattr(dynamodb_service, "get_course_learning_overview", None)
        if callable(get_overview):
            overview = await get_overview(course_id, user_id=user_id)
        else:
            course, lessons, quizzes, quiz_results = await asyncio.gather(
                dynamodb_service.get_course(course_id),
                dynamodb_service.get_course_lessons(course_id),
                dynamodb_service.get_quizzes_by_course(course_id),
                dynamodb_service.get_user_quiz_results(user_id, course_id=course_id)
                if user_id
                else asyncio.sleep(0, result=[]),
            )
            overview = {
                "course_id": course_id,
                "user_id": user_id,
                "course": course,
                "enrollment": None,
                "lessons": lessons,
                "quizzes": quizzes,
                "quiz_results": quiz_results,
                "generated_at": datetime.utcnow().isoformat(),
            }
        if not overview.get("course"):
            raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
        return overview
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error building learning overview for {course_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get learning overview")


@router.get("/courses/{course_id}/question-bank")
async def get_course_question_bank(
    course_id: str,
    limit: int = 10000,
    verified_only: bool = False,
    compact: bool = False,
    page: int = 1,
    page_size: Optional[int] = None,
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Return the persisted tutor question bank for a course."""
    try:
        safe_limit = max(1, min(limit, 10000))
        if page_size is not None:
            safe_page = max(1, int(page or 1))
            safe_page_size = max(1, min(1000, int(page_size or 500)))
            items = await dynamodb_service.get_question_bank_items(
                course_id,
                limit=safe_limit,
                verified_only=verified_only,
                compact=compact,
                page=safe_page,
                page_size=safe_page_size,
            )
        else:
            safe_page = 1
            safe_page_size = None
            items = await dynamodb_service.get_question_bank_items(
                course_id,
                limit=safe_limit,
                verified_only=verified_only,
                compact=compact,
            )
        if verified_only and not compact:
            items = [
                item
                for item in items
                if str(
                    item.get("verification_status")
                    or item.get("verificationStatus")
                    or ""
                )
                .strip()
                .lower()
                == "verified"
            ]
        response = {"course_id": course_id, "total": len(items), "items": items}
        if safe_page_size is not None:
            response.update(
                {
                    "page": safe_page,
                    "page_size": safe_page_size,
                    "has_next": len(items) == safe_page_size
                    and safe_page * safe_page_size < safe_limit,
                    "has_prev": safe_page > 1,
                }
            )
        return response
    except Exception as e:
        app_logger.error(f"Error retrieving question bank for course {course_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve question bank")


@router.put("/courses/{course_id}/question-bank")
async def replace_course_question_bank(
    course_id: str,
    payload: QuestionBankReplaceRequest,
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Replace a course question bank with the supplied current snapshot."""
    try:
        course = await dynamodb_service.get_course(course_id)
        if not course:
            raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
        items = await dynamodb_service.replace_question_bank_items(
            course_id=course_id,
            user_id=payload.user_id,
            items=payload.items,
        )
        return {"course_id": course_id, "total": len(items)}
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error replacing question bank for course {course_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save question bank")


@router.get("/users/{user_id}/courses")
async def list_user_courses(
    user_id: str,
    limit: int = 50,
    aliases: Optional[str] = None,
    view: str = "full",
    dynamodb_service=Depends(get_dynamodb_service),
):
    """List all courses for a specific user."""
    try:
        alias_values = [
            token.strip()
            for token in str(aliases or "").split(",")
            if token and token.strip()
        ]
        summary_view = str(view or "").strip().lower() == "summary"
        try:
            courses = (
                await dynamodb_service.get_user_courses(
                    user_id, aliases=alias_values, summary=summary_view
                )
            )[:limit]
        except TypeError:
            courses = (await dynamodb_service.get_user_courses(user_id))[:limit]

        return {"user_id": user_id, "total_courses": len(courses), "courses": courses}

    except Exception as e:
        app_logger.error(f"Error listing courses for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list user courses")


@router.get("/courses")
async def list_all_courses(dynamodb_service=Depends(get_dynamodb_service)):
    """List all courses available on the platform (active only)."""
    try:
        courses = await dynamodb_service.get_all_courses()
        return {"total_courses": len(courses), "courses": courses}
    except Exception as e:
        app_logger.error(f"Error listing all courses: {e}")
        raise HTTPException(status_code=500, detail="Failed to list courses")


@router.put("/courses/{course_id}")
async def update_course(
    course_id: str,
    name: str = Form(None),
    instructor: str = Form(None),
    course_format: str = Form(None),
    grade_level: str = Form(None),
    description: str = Form(None),
    detail: str = Form(None),
    target_profile: str = Form(None),
    structure_summary: str = Form(None),
    category: str = Form(None),
    image_url: str = Form(None),
    thumbnail_url: str = Form(None),
    preview_image_url: str = Form(None),
    purchase_preview_image_url: str = Form(None),
    price: Optional[str] = Form(None),
    topics_json: Optional[str] = Form(None),
    course_tags_json: Optional[str] = Form(None),
    benefits_json: Optional[str] = Form(None),
    content_items_json: Optional[str] = Form(None),
    pricing_plans_json: Optional[str] = Form(None),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Update course information."""
    try:
        # Prepare updates (only include non-None values)
        updates = {}
        if name is not None:
            updates["name"] = name
        if instructor is not None:
            normalized_instructor = (instructor or "").strip() or "อาจารย์ระบบ"
            updates["instructor"] = normalized_instructor
            updates["teacher_name"] = normalized_instructor
        if course_format is not None:
            updates["course_format"] = (course_format or "").strip()
        if grade_level is not None:
            updates["grade_level"] = (grade_level or "").strip()
        if description is not None:
            updates["description"] = description
        if detail is not None:
            updates["detail"] = detail
        if target_profile is not None:
            updates["target_profile"] = target_profile
        if structure_summary is not None:
            updates["structure_summary"] = structure_summary
        if category is not None:
            updates["category"] = category
        if image_url is not None:
            updates["image_url"] = (image_url or "").strip()
        if thumbnail_url is not None:
            updates["thumbnail_url"] = (thumbnail_url or "").strip()
        if preview_image_url is not None:
            updates["preview_image_url"] = (preview_image_url or "").strip()
        if purchase_preview_image_url is not None:
            updates["purchase_preview_image_url"] = (
                purchase_preview_image_url or ""
            ).strip()
        if price is not None:
            raw_price = str(price).strip()
            if raw_price == "":
                updates["price"] = None
            else:
                try:
                    parsed_price = float(raw_price)
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid price format")
                if parsed_price < 0:
                    raise HTTPException(status_code=400, detail="Price must be >= 0")
                updates["price"] = parsed_price
        if topics_json is not None:
            topics: List[str] = []
            if topics_json.strip():
                try:
                    parsed_topics = json.loads(topics_json)
                    if not isinstance(parsed_topics, list):
                        raise ValueError("topics_json must be a JSON array")
                    topics = [
                        str(item).strip() for item in parsed_topics if str(item).strip()
                    ]
                except Exception:
                    raise HTTPException(
                        status_code=400, detail="Invalid topics_json format"
                    )
            updates["topics"] = topics
        if course_tags_json is not None:
            course_tags: List[str] = []
            if course_tags_json.strip():
                try:
                    parsed_tags = json.loads(course_tags_json)
                    if not isinstance(parsed_tags, list):
                        raise ValueError("course_tags_json must be a JSON array")
                    course_tags = [
                        str(item).strip() for item in parsed_tags if str(item).strip()
                    ]
                except Exception:
                    raise HTTPException(
                        status_code=400, detail="Invalid course_tags_json format"
                    )
            if not course_tags:
                course_tags = [DEFAULT_COURSE_SUBJECT]
            updates["tags"] = course_tags
        if benefits_json is not None:
            benefits: List[str] = []
            if benefits_json.strip():
                try:
                    parsed_benefits = json.loads(benefits_json)
                    if not isinstance(parsed_benefits, list):
                        raise ValueError("benefits_json must be a JSON array")
                    benefits = [
                        str(item).strip()
                        for item in parsed_benefits
                        if str(item).strip()
                    ]
                except Exception:
                    raise HTTPException(
                        status_code=400, detail="Invalid benefits_json format"
                    )
            updates["benefits"] = benefits
        if content_items_json is not None:
            content_items: List[Dict[str, str]] = []
            if content_items_json.strip():
                try:
                    parsed_items = json.loads(content_items_json)
                    if not isinstance(parsed_items, list):
                        raise ValueError("content_items_json must be a JSON array")
                    normalized_items: List[Dict[str, str]] = []
                    for item in parsed_items:
                        if not isinstance(item, dict):
                            continue
                        title = str(item.get("title", "")).strip()
                        content_description = str(item.get("description", "")).strip()
                        if not title and not content_description:
                            continue
                        normalized_items.append(
                            {
                                "title": title or "หัวข้อ",
                                "description": content_description,
                            }
                        )
                    content_items = normalized_items
                except Exception:
                    raise HTTPException(
                        status_code=400, detail="Invalid content_items_json format"
                    )
            updates["content_items"] = content_items
        if pricing_plans_json is not None:
            pricing_plans: List[Dict[str, Any]] = []
            if pricing_plans_json.strip():
                try:
                    parsed_plans = json.loads(pricing_plans_json)
                    if not isinstance(parsed_plans, list):
                        raise ValueError("pricing_plans_json must be a JSON array")
                    normalized_plans: List[Dict[str, Any]] = []
                    for plan in parsed_plans:
                        if not isinstance(plan, dict):
                            continue
                        duration_months = int(plan.get("duration_months", 0))
                        plan_price = float(plan.get("price", 0))
                        plan_label = str(plan.get("label", "")).strip()
                        if duration_months <= 0 or plan_price < 0:
                            continue
                        normalized_plans.append(
                            {
                                "duration_months": duration_months,
                                "price": plan_price,
                                "label": plan_label or f"{duration_months} เดือน",
                            }
                        )
                    pricing_plans = normalized_plans
                except Exception:
                    raise HTTPException(
                        status_code=400, detail="Invalid pricing_plans_json format"
                    )
            updates["pricing_plans"] = pricing_plans

        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")

        success = await dynamodb_service.update_course(course_id, updates)

        if success:
            # Return updated course
            course = await dynamodb_service.get_course(course_id)
            return {"message": "Course updated successfully", "course": course}
        else:
            raise HTTPException(status_code=500, detail="Failed to update course")

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error updating course {course_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update course")


@router.delete("/courses/{course_id}")
async def delete_course(course_id: str, dynamodb_service=Depends(get_dynamodb_service)):
    """Delete a course (soft delete)."""
    try:
        course = await dynamodb_service.get_course(course_id)
        if not course:
            raise HTTPException(status_code=404, detail=f"Course {course_id} not found")

        enrollments = await dynamodb_service.get_course_enrollments(course_id)
        protected_enrollments = [
            enrollment
            for enrollment in enrollments
            if str(enrollment.get("status") or "active").strip().lower()
            not in {"cancelled", "deleted"}
        ]
        if protected_enrollments:
            raise HTTPException(
                status_code=409,
                detail=(
                    "COURSE_HAS_ENROLLMENTS: cannot delete a course while "
                    f"{len(protected_enrollments)} active enrollment(s) still reference it"
                ),
            )

        success = await dynamodb_service.delete_course(course_id)

        if success:
            return {"message": f"Course {course_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete course")

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error deleting course {course_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete course")


# Payment Endpoints
@router.post(
    "/payments/promptpay/create-intent", response_model=PromptPayCreateIntentResponse
)
async def create_promptpay_payment_intent(
    body: PromptPayCreateIntentRequest,
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Create a Stripe PromptPay PaymentIntent for a course purchase."""
    settings = get_settings()
    if not settings.stripe_private_key or not settings.stripe_public_key:
        raise HTTPException(
            status_code=500,
            detail="Stripe keys are not configured (STRIPE_PRIVATE_KEY / STRIPE_PUBLIC_KEY)",
        )

    user_id = str(body.user_id or "").strip()
    course_id = str(body.course_id or "").strip()
    if not user_id or not course_id:
        raise HTTPException(
            status_code=400, detail="user_id and course_id are required"
        )

    course = await dynamodb_service.get_course(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    existing_enrollment_with_schedule = await _get_existing_enrollment_with_schedule(
        dynamodb_service=dynamodb_service,
        user_id=user_id,
        course_id=course_id,
    )
    has_existing_active_enrollment = bool(
        existing_enrollment_with_schedule
        and not existing_enrollment_with_schedule["schedule"]["is_expired"]
    )

    allowed_prices: List[float] = []
    base_price_raw = course.get("price")
    try:
        base_price = float(base_price_raw or 0)
    except Exception:
        base_price = 0.0
    if base_price > 0:
        allowed_prices.append(round(base_price, 2))

    pricing_plans = course.get("pricing_plans")
    if isinstance(pricing_plans, list):
        for plan in pricing_plans:
            if not isinstance(plan, dict):
                continue
            try:
                p = float(plan.get("price", 0))
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
            raise HTTPException(status_code=400, detail="Invalid amount_thb")
        if unique_allowed_prices and amount_thb not in unique_allowed_prices:
            raise HTTPException(
                status_code=400, detail="Selected amount does not match course pricing"
            )

    if amount_thb <= 0:
        raise HTTPException(
            status_code=400,
            detail="This course is free, please enroll directly without payment",
        )

    amount_satang = int(round(amount_thb * 100))
    if amount_satang < 100:
        raise HTTPException(status_code=400, detail="Amount is too low for payment")

    course_name = str(course.get("name") or course.get("title") or "คอร์สเรียน").strip()

    stripe_payload = {
        "amount": str(amount_satang),
        "currency": "thb",
        "payment_method_types[]": "promptpay",
        "description": f"Course payment: {course_name}",
        "metadata[user_id]": user_id,
        "metadata[course_id]": course_id,
        "metadata[payment_type]": "course_enrollment",
        "metadata[plan_label]": str(body.plan_label or "").strip(),
        "metadata[duration_months]": str(body.duration_months or ""),
    }
    intent = await _stripe_request(
        method="POST",
        path="/payment_intents",
        secret_key=settings.stripe_private_key,
        data=stripe_payload,
    )
    payment_intent_id = str(intent.get("id") or "").strip()
    client_secret = str(intent.get("client_secret") or "").strip()
    if not payment_intent_id or not client_secret:
        raise HTTPException(status_code=500, detail="Failed to create payment intent")

    return PromptPayCreateIntentResponse(
        payment_intent_id=payment_intent_id,
        client_secret=client_secret,
        publishable_key=settings.stripe_public_key,
        amount=amount_satang,
        currency=str(intent.get("currency") or "thb").upper(),
        payment_status=str(intent.get("status") or "requires_payment_method"),
        already_enrolled=has_existing_active_enrollment,
    )


@router.post("/payments/promptpay/confirm")
async def confirm_promptpay_payment_and_enroll(
    body: PromptPayConfirmRequest,
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Verify payment status from Stripe and enroll the user when payment succeeds."""
    user_id = str(body.user_id or "").strip()
    course_id = str(body.course_id or "").strip()
    payment_intent_id = str(body.payment_intent_id or "").strip()
    if not user_id or not course_id or not payment_intent_id:
        raise HTTPException(
            status_code=400,
            detail="user_id, course_id, and payment_intent_id are required",
        )

    return await _complete_promptpay_payment(
        payment_intent_id=payment_intent_id,
        dynamodb_service=dynamodb_service,
        expected_user_id=user_id,
        expected_course_id=course_id,
    )


@router.post("/payments/stripe/webhook")
async def handle_stripe_payment_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(default=None, alias="Stripe-Signature"),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Handle Stripe payment webhooks for payment fulfillment."""
    settings = get_settings()
    if not settings.stripe_webhook_secret:
        raise HTTPException(
            status_code=500, detail="Stripe webhook secret is not configured"
        )
    payload = await request.body()
    _verify_stripe_webhook_signature(
        payload=payload,
        signature_header=stripe_signature or "",
        webhook_secret=settings.stripe_webhook_secret,
    )
    try:
        event = json.loads(payload.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Stripe webhook payload")
    if not isinstance(event, dict):
        raise HTTPException(status_code=400, detail="Invalid Stripe webhook payload")

    event_type = str(event.get("type") or "").strip()
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    obj = data.get("object") if isinstance(data.get("object"), dict) else {}
    if event_type == "payment_intent.succeeded":
        payment_intent_id = str(obj.get("id") or "").strip()
        if not payment_intent_id:
            raise HTTPException(
                status_code=400, detail="Missing payment intent id in webhook"
            )
        await _complete_promptpay_payment(
            payment_intent_id=payment_intent_id,
            dynamodb_service=dynamodb_service,
        )
    return {"received": True}


@router.get("/users/{user_id}/payment-history")
async def get_user_payment_history(
    user_id: str,
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Return student payment history from enrollment records."""
    try:
        get_with_aliases = getattr(
            dynamodb_service, "get_user_enrollments_with_aliases", None
        )
        if callable(get_with_aliases):
            enrollments = await get_with_aliases(user_id)
        else:
            enrollments = await dynamodb_service.get_user_enrollments(user_id)
        rows: List[Dict[str, Any]] = []

        for enrollment in enrollments:
            course_id = str(enrollment.get("course_id") or "").strip()
            if not course_id:
                continue

            course = await dynamodb_service.get_course(course_id)
            course_name = (
                str(
                    (course or {}).get("name") or (course or {}).get("title") or ""
                ).strip()
                or "คอร์ส"
            )

            payment_events = _normalize_payment_history(enrollment)
            if not payment_events:
                schedule = _build_enrollment_schedule(
                    started_at_raw=enrollment.get("started_at")
                    or enrollment.get("enrolled_at"),
                    expires_at_raw=enrollment.get("expires_at"),
                    duration_months_raw=enrollment.get("duration_months"),
                )
                rows.append(
                    {
                        "enrollment_id": enrollment.get("enrollment_id"),
                        "course_id": course_id,
                        "course_name": course_name,
                        "order_id": enrollment.get("order_id")
                        or _build_payment_order_id(
                            enrollment.get("paid_at") or enrollment.get("enrolled_at"),
                            enrollment.get("payment_intent_id"),
                        ),
                        "payment_provider": enrollment.get("payment_provider")
                        or "stripe",
                        "payment_type": enrollment.get("payment_type") or "manual",
                        "payment_intent_id": enrollment.get("payment_intent_id"),
                        "stripe_charge_id": enrollment.get("stripe_charge_id"),
                        "receipt_number": enrollment.get("receipt_number"),
                        "receipt_url": enrollment.get("receipt_url"),
                        "payment_status": enrollment.get("payment_status") or "active",
                        "paid_amount_thb": None,
                        "paid_currency": enrollment.get("paid_currency") or "THB",
                        "billing_email": enrollment.get("billing_email"),
                        "plan_label": enrollment.get("plan_label"),
                        "duration_months": schedule["duration_months"],
                        "paid_at": enrollment.get("paid_at")
                        or enrollment.get("enrolled_at"),
                        "enrolled_at": enrollment.get("enrolled_at"),
                        "started_at": schedule["started_at"],
                        "expires_at": schedule["expires_at"],
                        "is_expired": schedule["is_expired"],
                        "days_remaining": schedule["days_remaining"],
                        "in_system": True,
                    }
                )
                continue

            for event in payment_events:
                schedule = _build_enrollment_schedule(
                    started_at_raw=event.get("started_at")
                    or enrollment.get("started_at")
                    or enrollment.get("enrolled_at"),
                    expires_at_raw=event.get("expires_at")
                    or enrollment.get("expires_at"),
                    duration_months_raw=event.get("duration_months"),
                )
                rows.append(
                    {
                        "enrollment_id": enrollment.get("enrollment_id"),
                        "course_id": course_id,
                        "course_name": course_name,
                        "order_id": event.get("order_id"),
                        "payment_provider": event.get("payment_provider") or "stripe",
                        "payment_type": event.get("payment_type") or "promptpay",
                        "payment_intent_id": event.get("payment_intent_id"),
                        "stripe_charge_id": event.get("stripe_charge_id"),
                        "receipt_number": event.get("receipt_number"),
                        "receipt_url": event.get("receipt_url"),
                        "payment_status": event.get("payment_status") or "active",
                        "paid_amount_thb": event.get("paid_amount_thb"),
                        "paid_currency": event.get("paid_currency") or "THB",
                        "billing_email": event.get("billing_email"),
                        "plan_label": event.get("plan_label"),
                        "duration_months": schedule["duration_months"],
                        "paid_at": event.get("paid_at")
                        or enrollment.get("enrolled_at"),
                        "enrolled_at": enrollment.get("enrolled_at"),
                        "started_at": schedule["started_at"],
                        "expires_at": schedule["expires_at"],
                        "is_expired": schedule["is_expired"],
                        "days_remaining": schedule["days_remaining"],
                        "in_system": True,
                    }
                )

        rows.sort(
            key=lambda row: str(row.get("paid_at") or row.get("enrolled_at") or ""),
            reverse=True,
        )
        await _hydrate_payment_history_receipts(rows)
        return {
            "user_id": user_id,
            "total": len(rows),
            "rows": rows,
        }
    except Exception as e:
        app_logger.error(f"Error fetching payment history for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get payment history")


@router.get("/admin/token-usage/daily")
async def get_admin_daily_token_usage(
    days: int = 30,
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Get platform token usage totals per day for admin dashboard."""
    try:
        settings = get_settings()
        usd_to_thb_rate = max(0.0, float(settings.litellm_cost_usd_to_thb or 0.0))
        days = max(1, min(int(days), 365))
        cache_key = f"days:{days}:rate:{usd_to_thb_rate}"
        cache_entry = ADMIN_TOKEN_USAGE_CACHE.get(cache_key)
        now_ts = datetime.utcnow().timestamp()
        if (
            cache_entry
            and now_ts - float(cache_entry.get("cached_at") or 0)
            < ADMIN_TOKEN_USAGE_CACHE_TTL_SECONDS
        ):
            return copy.deepcopy(cache_entry.get("payload") or {})
        today = datetime.utcnow().date()
        since_date = today - timedelta(days=days - 1)

        users: List[Dict[str, Any]] = []
        scan_kwargs: Dict[str, Any] = {}
        while True:
            response = dynamodb_service.users_table.scan(**scan_kwargs)
            users.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            scan_kwargs["ExclusiveStartKey"] = last_key

        daily_map: Dict[str, Dict[str, Any]] = {}
        per_student_map: Dict[str, Dict[str, Any]] = {}
        for item in users:
            user = dynamodb_service._convert_decimals_to_float(item)
            user_id = str(user.get("user_id") or "").strip()
            if not user_id:
                continue
            token_daily = user.get("token_usage_daily")
            if not isinstance(token_daily, dict):
                continue

            student_row = per_student_map.get(user_id)
            if not student_row:
                student_row = {
                    "user_id": user_id,
                    "name": str(user.get("name") or "").strip(),
                    "email": str(user.get("email") or "").strip(),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "request_count": 0,
                    "llm_cost_usd": 0.0,
                    "llm_cost_thb": 0.0,
                    "active_days": 0,
                }
                per_student_map[user_id] = student_row

            for date_key, usage_row in token_daily.items():
                row_date = _parse_iso_datetime(f"{date_key}T00:00:00")
                if not row_date:
                    continue
                if row_date.date() < since_date or row_date.date() > today:
                    continue
                if not isinstance(usage_row, dict):
                    continue
                input_tokens = int(_safe_float(usage_row.get("input_tokens"), 0))
                output_tokens = int(_safe_float(usage_row.get("output_tokens"), 0))
                total_tokens = int(_safe_float(usage_row.get("total_tokens"), 0))
                request_count = int(_safe_float(usage_row.get("request_count"), 0))
                llm_cost_usd = _safe_float(usage_row.get("llm_cost_usd"), 0.0)
                llm_cost_thb = llm_cost_usd * usd_to_thb_rate

                day_bucket = daily_map.get(date_key)
                if not day_bucket:
                    day_bucket = {
                        "date": date_key,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "request_count": 0,
                        "llm_cost_usd": 0.0,
                        "llm_cost_thb": 0.0,
                        "active_students": set(),
                    }
                    daily_map[date_key] = day_bucket
                day_bucket["input_tokens"] += input_tokens
                day_bucket["output_tokens"] += output_tokens
                day_bucket["total_tokens"] += total_tokens
                day_bucket["request_count"] += request_count
                day_bucket["llm_cost_usd"] += llm_cost_usd
                day_bucket["llm_cost_thb"] += llm_cost_thb
                day_bucket["active_students"].add(user_id)

                student_row["input_tokens"] += input_tokens
                student_row["output_tokens"] += output_tokens
                student_row["total_tokens"] += total_tokens
                student_row["request_count"] += request_count
                student_row["llm_cost_usd"] += llm_cost_usd
                student_row["llm_cost_thb"] += llm_cost_thb
                student_row["active_days"] += 1

        daily_rows = []
        for day_key in sorted(daily_map.keys()):
            row = daily_map[day_key]
            daily_rows.append(
                {
                    "date": day_key,
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "total_tokens": row["total_tokens"],
                    "request_count": row["request_count"],
                    "llm_cost_usd": round(float(row["llm_cost_usd"]), 8),
                    "llm_cost_thb": round(float(row["llm_cost_thb"]), 4),
                    "active_students": len(row["active_students"]),
                }
            )

        # Prefer true spend from LiteLLM logs (authoritative source of billed cost).
        # The chat completion payload may not always include response_cost fields.
        litellm_daily_spend: Dict[str, float] = {}
        try:
            litellm_key = str(settings.litellm_api_key or "").strip()
            litellm_base = str(settings.litellm_base_url or "").rstrip("/")
            if litellm_key and litellm_base:
                spend_url = f"{litellm_base}/spend/logs"
                # LiteLLM /spend/logs treats end_date as an exclusive boundary.
                # Add +1 day so "today" spend is included in the aggregation.
                spend_end_date = (today + timedelta(days=1)).isoformat()
                async with httpx.AsyncClient(timeout=15.0) as client:
                    spend_res = await client.get(
                        spend_url,
                        params={
                            "start_date": since_date.isoformat(),
                            "end_date": spend_end_date,
                        },
                        headers={"Authorization": f"Bearer {litellm_key}"},
                    )
                if spend_res.status_code < 400:
                    payload = spend_res.json()
                    if isinstance(payload, list):
                        for row in payload:
                            if not isinstance(row, dict):
                                continue
                            day = str(row.get("startTime") or "").strip()[:10]
                            if not day:
                                continue
                            litellm_daily_spend[day] = _safe_float(
                                row.get("spend"), 0.0
                            )
        except Exception as spend_exc:
            app_logger.warning(
                f"LiteLLM spend logs unavailable, fallback to stored per-request cost: {spend_exc}"
            )

        if litellm_daily_spend:
            for row in daily_rows:
                day = str(row.get("date") or "")
                true_usd = _safe_float(
                    litellm_daily_spend.get(day), row.get("llm_cost_usd")
                )
                row["llm_cost_usd"] = round(float(true_usd), 8)
                row["llm_cost_thb"] = round(float(true_usd * usd_to_thb_rate), 4)

        for row in per_student_map.values():
            row["llm_cost_usd"] = round(float(row["llm_cost_usd"]), 8)
            row["llm_cost_thb"] = round(float(row["llm_cost_thb"]), 4)

        student_rows = sorted(
            per_student_map.values(),
            key=lambda row: row["total_tokens"],
            reverse=True,
        )
        response_payload = {
            "days": days,
            "from_date": since_date.isoformat(),
            "to_date": today.isoformat(),
            "daily": daily_rows,
            "students": student_rows,
            "overall": {
                "input_tokens": sum(row["input_tokens"] for row in daily_rows),
                "output_tokens": sum(row["output_tokens"] for row in daily_rows),
                "total_tokens": sum(row["total_tokens"] for row in daily_rows),
                "request_count": sum(row["request_count"] for row in daily_rows),
                "llm_cost_usd": round(
                    sum(float(row["llm_cost_usd"]) for row in daily_rows), 8
                ),
                "llm_cost_thb": round(
                    sum(float(row["llm_cost_thb"]) for row in daily_rows), 4
                ),
                "llm_cost_usd_to_thb_rate": usd_to_thb_rate,
                "llm_cost_source": "litellm_spend_logs"
                if litellm_daily_spend
                else "app_usage_fallback",
                "active_students": len(student_rows),
            },
        }
        ADMIN_TOKEN_USAGE_CACHE[cache_key] = {
            "cached_at": now_ts,
            "payload": copy.deepcopy(response_payload),
        }
        return response_payload
    except Exception as e:
        app_logger.error(f"Error building admin token usage report: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get admin token usage report"
        )


@router.get("/chat/energy")
async def get_student_chat_energy_status(
    user_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        STUDENT_BEARER_OPTIONAL
    ),
    student_auth_service: StudentAuthService = Depends(_get_student_auth_service),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Get current student's chat energy status (daily THB budget)."""
    try:
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        if credentials and str(credentials.credentials or "").strip():
            await _ensure_user_matches_token(
                user_id=normalized_user_id,
                credentials=credentials,
                auth_service=student_auth_service,
            )

        status = await dynamodb_service.get_student_chat_energy_status(
            normalized_user_id
        )
        return {
            "user_id": normalized_user_id,
            **_to_chat_energy_response(status),
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to get chat energy status for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get chat energy status")


@router.get("/admin/students")
async def get_admin_students_overview(
    days: int = 30,
    page: int = 1,
    page_size: int = 50,
    q: Optional[str] = None,
    sort: str = "usage_desc",
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Get per-student token usage + purchase/expiry details for admin."""
    try:
        days = max(1, min(int(days), 365))
        page = max(1, int(page or 1))
        page_size = max(1, min(200, int(page_size or 50)))
        q_text = str(q or "").strip().lower()
        sort_key = str(sort or "usage_desc").strip().lower()
        today = datetime.utcnow().date()
        since_date = today - timedelta(days=days - 1)

        users: Dict[str, Dict[str, Any]] = {}
        scan_kwargs: Dict[str, Any] = {}
        while True:
            response = dynamodb_service.users_table.scan(**scan_kwargs)
            for raw_item in response.get("Items", []):
                user = dynamodb_service._convert_decimals_to_float(raw_item)
                user_id = str(user.get("user_id") or "").strip()
                if not user_id:
                    continue
                users[user_id] = user
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            scan_kwargs["ExclusiveStartKey"] = last_key

        identity_owners: Dict[str, Set[str]] = {}
        for user_id, user in users.items():
            for identity in (user_id, user.get("username"), user.get("student_id")):
                normalized_identity = str(identity or "").strip()
                if not normalized_identity:
                    continue
                identity_owners.setdefault(normalized_identity, set()).add(user_id)
        canonical_user_ids_by_identity: Dict[str, str] = {}
        for identity, owner_ids in identity_owners.items():
            if len(owner_ids) == 1:
                canonical_user_ids_by_identity[identity] = next(iter(owner_ids))
                continue
            uuid_owner_ids = []
            for owner_id in owner_ids:
                try:
                    uuid.UUID(owner_id)
                    uuid_owner_ids.append(owner_id)
                except (TypeError, ValueError):
                    continue
            if len(uuid_owner_ids) == 1:
                canonical_user_ids_by_identity[identity] = uuid_owner_ids[0]

        enrollments = await dynamodb_service.get_all_active_enrollments(limit=10000)
        course_ids = list(
            {
                str(enrollment.get("course_id") or "").strip()
                for enrollment in enrollments
                if str(enrollment.get("course_id") or "").strip()
            }
        )
        get_courses_by_ids = getattr(dynamodb_service, "get_courses_by_ids", None)
        courses_by_id: Dict[str, Dict[str, Any]] = {}
        if callable(get_courses_by_ids) and course_ids:
            courses_by_id = {
                str(course.get("course_id") or ""): course
                for course in await get_courses_by_ids(
                    course_ids, limit=len(course_ids)
                )
                if str(course.get("course_id") or "")
            }
        course_name_cache: Dict[str, str] = {}
        missing_course_ids = [
            course_id
            for course_id in course_ids
            if course_id and course_id not in courses_by_id
        ]
        if callable(get_courses_by_ids) and missing_course_ids:
            for course in await get_courses_by_ids(
                missing_course_ids, limit=len(missing_course_ids)
            ):
                course_id = str(course.get("course_id") or "").strip()
                if course_id:
                    courses_by_id[course_id] = course

        rows_by_user: Dict[str, Dict[str, Any]] = {}

        def _normalize_email(value: Any) -> str:
            email = str(value or "").strip()
            if not email:
                return ""
            lowered = email.lower()
            # Skip placeholders generated by system fallback users.
            if lowered.endswith("@example.com"):
                return ""
            return email

        def ensure_row(user_id: str) -> Dict[str, Any]:
            row = rows_by_user.get(user_id)
            if row:
                return row
            user = users.get(user_id) or {}
            token_daily = user.get("token_usage_daily")
            if not isinstance(token_daily, dict):
                token_daily = {}
            usage_input = 0
            usage_output = 0
            usage_total = 0
            usage_requests = 0
            usage_days = 0
            for date_key, usage in token_daily.items():
                dt = _parse_iso_datetime(f"{date_key}T00:00:00")
                if not dt or dt.date() < since_date or dt.date() > today:
                    continue
                if not isinstance(usage, dict):
                    continue
                usage_input += int(_safe_float(usage.get("input_tokens"), 0))
                usage_output += int(_safe_float(usage.get("output_tokens"), 0))
                usage_total += int(_safe_float(usage.get("total_tokens"), 0))
                usage_requests += int(_safe_float(usage.get("request_count"), 0))
                usage_days += 1

            row = {
                "user_id": user_id,
                "name": str(user.get("name") or "").strip(),
                "email": _normalize_email(user.get("email")),
                "token_usage": {
                    "input_tokens": usage_input,
                    "output_tokens": usage_output,
                    "total_tokens": usage_total,
                    "request_count": usage_requests,
                    "active_days": usage_days,
                    "window_days": days,
                },
                "courses": [],
                "active_courses": 0,
                "expired_courses": 0,
                "total_spend_thb": 0.0,
                "trial_used": False,
                "trial_courses": 0,
                "trial_last_used_at": None,
                "trial_available": True,
                "trial_status_source": "enrollment",
                "trial_override_mode": "auto",
                "trial_override_updated_at": None,
                "trial_override_updated_by": None,
                "trial_override_reason": None,
            }
            rows_by_user[user_id] = row
            return row

        for enrollment in enrollments:
            enrollment_user_id = str(enrollment.get("user_id") or "").strip()
            if not enrollment_user_id:
                continue
            user_id = canonical_user_ids_by_identity.get(
                enrollment_user_id, enrollment_user_id
            )
            row = ensure_row(user_id)
            if not row.get("email"):
                # Prefer billing email captured by payment flow when user profile email is missing.
                row["email"] = _normalize_email(enrollment.get("billing_email"))

            course_id = str(enrollment.get("course_id") or "").strip()
            if course_id:
                if course_id not in course_name_cache:
                    course = courses_by_id.get(course_id)
                    course_name_cache[course_id] = (
                        str(
                            (course or {}).get("name")
                            or (course or {}).get("title")
                            or ""
                        ).strip()
                        or "คอร์ส"
                    )
                course_name = course_name_cache[course_id]
            else:
                course_name = "คอร์ส"

            schedule = _build_enrollment_schedule(
                started_at_raw=enrollment.get("started_at")
                or enrollment.get("enrolled_at"),
                expires_at_raw=enrollment.get("expires_at"),
                duration_months_raw=enrollment.get("duration_months"),
            )
            amount = _safe_float(enrollment.get("paid_amount_thb"), 0.0)
            row["total_spend_thb"] += max(0.0, amount)
            is_free_course = _is_free_course_enrollment(enrollment)
            if is_free_course:
                row["trial_used"] = True
                row["trial_courses"] += 1
                consumed_at = str(
                    enrollment.get("free_course_claimed_at")
                    or enrollment.get("trial_consumed_at")
                    or enrollment.get("enrolled_at")
                    or ""
                ).strip()
                if consumed_at:
                    last_used = str(row.get("trial_last_used_at") or "").strip()
                    if not last_used or consumed_at > last_used:
                        row["trial_last_used_at"] = consumed_at
            if schedule["is_expired"]:
                row["expired_courses"] += 1
            else:
                row["active_courses"] += 1

            row["courses"].append(
                {
                    "enrollment_id": enrollment.get("enrollment_id"),
                    "course_id": course_id,
                    "course_name": course_name,
                    "payment_status": enrollment.get("payment_status") or "active",
                    "payment_type": enrollment.get("payment_type") or "manual",
                    "enrollment_source": enrollment.get("enrollment_source"),
                    "enrollment_type": enrollment.get("enrollment_type"),
                    "paid_amount_thb": amount,
                    "duration_months": schedule["duration_months"],
                    "started_at": schedule["started_at"],
                    "expires_at": schedule["expires_at"],
                    "is_expired": schedule["is_expired"],
                    "days_remaining": schedule["days_remaining"],
                    "paid_at": enrollment.get("paid_at")
                    or enrollment.get("enrolled_at"),
                    "enrolled_at": enrollment.get("enrolled_at"),
                    "is_trial": is_free_course,
                    "trial_consumed_at": enrollment.get("free_course_claimed_at"),
                    "trial_expires_at": None,
                }
            )

        # Ensure users with token usage but no enrollments still appear
        for user_id in users.keys():
            token_daily = users[user_id].get("token_usage_daily")
            if isinstance(token_daily, dict) and user_id not in rows_by_user:
                ensure_row(user_id)

        rows = list(rows_by_user.values())
        if q_text:
            rows = [
                row
                for row in rows
                if q_text
                in " ".join(
                    [
                        str(row.get("user_id") or ""),
                        str(row.get("name") or ""),
                        str(row.get("email") or ""),
                    ]
                ).lower()
            ]

        rows.sort(
            key=lambda item: (
                int(item["token_usage"].get("total_tokens") or 0),
                float(item.get("total_spend_thb") or 0),
            ),
            reverse=True,
        )
        if sort_key == "spend_desc":
            rows.sort(
                key=lambda item: float(item.get("total_spend_thb") or 0), reverse=True
            )
        elif sort_key == "name_asc":
            rows.sort(
                key=lambda item: str(
                    item.get("name") or item.get("email") or item.get("user_id") or ""
                ).lower()
            )
        elif sort_key == "recent_expiry":
            rows.sort(
                key=lambda item: min(
                    [
                        int(course.get("days_remaining"))
                        for course in item.get("courses", [])
                        if isinstance(course.get("days_remaining"), int)
                        and not course.get("is_expired")
                    ]
                    or [999999]
                )
            )
        total_students = len(rows)
        total_pages = max(1, (total_students + page_size - 1) // page_size)
        if page > total_pages:
            page = total_pages
        page_start = (page - 1) * page_size
        page_rows = rows[page_start : page_start + page_size]

        energy_settings_raw = await dynamodb_service.get_chat_energy_platform_config()
        energy_settings = {
            "daily_limit_thb": _safe_float(
                energy_settings_raw.get("default_daily_limit_thb"), 0.0
            ),
            "updated_at": str(energy_settings_raw.get("updated_at") or "").strip()
            or None,
            "updated_by": str(energy_settings_raw.get("updated_by") or "").strip()
            or None,
            "reason": str(energy_settings_raw.get("reason") or "").strip() or None,
        }

        async def _load_student_chat_energy(user_id: str) -> Dict[str, Any]:
            normalized_user_id = str(user_id or "").strip()
            if not normalized_user_id:
                return _to_chat_energy_response(None)
            try:
                status = await dynamodb_service.get_student_chat_energy_status(
                    normalized_user_id
                )
                return _to_chat_energy_response(status)
            except Exception as energy_error:
                app_logger.warning(
                    f"Unable to build chat energy status for user {normalized_user_id}: {energy_error}"
                )
                return _to_chat_energy_response(None)

        if page_rows:
            energy_statuses = await asyncio.gather(
                *[
                    _load_student_chat_energy(str(row.get("user_id") or ""))
                    for row in page_rows
                ]
            )
            for idx, row in enumerate(page_rows):
                row["chat_energy"] = energy_statuses[idx]

        for row in page_rows:
            user = users.get(str(row.get("user_id") or "").strip()) or {}
            trial_override = _extract_trial_override(user)
            effective_trial = _resolve_effective_trial_used(
                trial_used_from_enrollments=bool(row.get("trial_used")),
                override_mode=trial_override.get("mode"),
            )
            row["trial_used"] = bool(effective_trial.get("trial_used"))
            row["trial_available"] = not bool(row.get("trial_used"))
            row["trial_status_source"] = (
                str(effective_trial.get("trial_status_source") or "").strip()
                or "enrollment"
            )
            row["trial_override_mode"] = str(trial_override.get("mode") or "auto")
            row["trial_override_updated_at"] = trial_override.get("updated_at")
            row["trial_override_updated_by"] = trial_override.get("updated_by")
            row["trial_override_reason"] = trial_override.get("reason")
            row["premium"] = _build_admin_premium_summary(user)
            row["courses"].sort(
                key=lambda item: str(
                    item.get("paid_at") or item.get("enrolled_at") or ""
                ),
                reverse=True,
            )
            row["total_spend_thb"] = round(float(row["total_spend_thb"]), 2)

        return {
            "days": days,
            "from_date": since_date.isoformat(),
            "to_date": today.isoformat(),
            "total_students": total_students,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "chat_energy_settings": energy_settings,
            "students": page_rows,
        }
    except Exception as e:
        app_logger.error(f"Error building admin students overview: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get admin students overview"
        )


@router.get("/admin/transactions")
async def get_admin_transactions(
    days: int = 30,
    type: str = "all",
    payment_status: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Get admin transaction feed across all students (payment/trial/manual)."""
    try:
        days = max(1, min(int(days), 365))
        page = max(1, int(page or 1))
        page_size = max(1, min(200, int(page_size or 50)))
        today = datetime.utcnow().date()
        since_date = today - timedelta(days=days - 1)

        type_filter = str(type or "all").strip().lower() or "all"
        allowed_types = {"all", "payment", "trial", "manual"}
        if type_filter not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid type '{type_filter}'. "
                    f"Allowed: {', '.join(sorted(allowed_types))}"
                ),
            )

        payment_status_filter = (
            str(payment_status or "").strip().lower() if payment_status else ""
        )
        if payment_status_filter == "all":
            payment_status_filter = ""

        users: Dict[str, Dict[str, Any]] = {}
        scan_kwargs: Dict[str, Any] = {}
        while True:
            response = dynamodb_service.users_table.scan(**scan_kwargs)
            for raw_item in response.get("Items", []):
                user = dynamodb_service._convert_decimals_to_float(raw_item)
                user_id = str(user.get("user_id") or "").strip()
                if user_id:
                    users[user_id] = user
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            scan_kwargs["ExclusiveStartKey"] = last_key

        enrollments = await dynamodb_service.get_all_active_enrollments(limit=10000)
        course_ids = list(
            {
                str(enrollment.get("course_id") or "").strip()
                for enrollment in enrollments
                if str(enrollment.get("course_id") or "").strip()
            }
        )
        get_courses_by_ids = getattr(dynamodb_service, "get_courses_by_ids", None)
        courses_by_id: Dict[str, Dict[str, Any]] = {}
        if callable(get_courses_by_ids) and course_ids:
            courses_by_id = {
                str(course.get("course_id") or ""): course
                for course in await get_courses_by_ids(
                    course_ids, limit=len(course_ids)
                )
                if str(course.get("course_id") or "")
            }
        course_name_cache: Dict[str, str] = {}
        rows: List[Dict[str, Any]] = []

        def _normalize_email(value: Any) -> str:
            email = str(value or "").strip()
            if not email:
                return ""
            lowered = email.lower()
            if lowered.endswith("@example.com"):
                return ""
            return email

        def _is_in_window(value: Any) -> bool:
            dt = _parse_iso_datetime(value)
            if not dt:
                return True
            return since_date <= dt.date() <= today

        def _normalize_status(value: Any) -> str:
            return str(value or "").strip().lower() or "active"

        def _classify_transaction_type(
            trial_flag: bool, event: Optional[Dict[str, Any]]
        ) -> str:
            if trial_flag:
                return "trial"
            if not isinstance(event, dict):
                return "manual"

            normalized_payment_type = (
                str(event.get("payment_type") or "").strip().lower()
            )
            if normalized_payment_type in {"manual", "admin", "internal"}:
                return "manual"

            payment_intent_id = str(event.get("payment_intent_id") or "").strip()
            paid_amount = event.get("paid_amount_thb")
            status = _normalize_status(event.get("payment_status"))
            if (
                not payment_intent_id
                and paid_amount is None
                and status in {"active", "pending", "unknown", ""}
            ):
                return "manual"
            return "payment"

        for enrollment in enrollments:
            user_id = str(enrollment.get("user_id") or "").strip()
            if not user_id:
                continue

            user = users.get(user_id) or {}
            student_name = (
                str(user.get("name") or "").strip()
                or _normalize_email(user.get("email"))
                or user_id
            )
            student_email = _normalize_email(user.get("email")) or _normalize_email(
                enrollment.get("billing_email")
            )

            course_id = str(enrollment.get("course_id") or "").strip()
            if course_id:
                if course_id not in course_name_cache:
                    course = courses_by_id.get(course_id)
                    if not course:
                        course = await dynamodb_service.get_course(course_id)
                    course_name_cache[course_id] = (
                        str(
                            (course or {}).get("name")
                            or (course or {}).get("title")
                            or ""
                        ).strip()
                        or "คอร์ส"
                    )
                course_name = course_name_cache[course_id]
            else:
                course_name = "คอร์ส"

            schedule = _build_enrollment_schedule(
                started_at_raw=enrollment.get("started_at")
                or enrollment.get("enrolled_at"),
                expires_at_raw=enrollment.get("expires_at"),
                duration_months_raw=enrollment.get("duration_months"),
            )
            enrollment_id = str(enrollment.get("enrollment_id") or "").strip()
            trial_flag = _is_free_course_enrollment(enrollment)
            payment_events = _normalize_payment_history(enrollment)

            if trial_flag:
                trial_at = (
                    enrollment.get("free_course_claimed_at")
                    or enrollment.get("trial_consumed_at")
                    or enrollment.get("enrolled_at")
                    or enrollment.get("started_at")
                )
                if not _is_in_window(trial_at):
                    continue

                row = {
                    "transaction_id": f"{enrollment_id or user_id}:trial",
                    "transaction_type": "trial",
                    "user_id": user_id,
                    "student_name": student_name,
                    "student_email": student_email,
                    "course_id": course_id,
                    "course_name": course_name,
                    "enrollment_id": enrollment.get("enrollment_id"),
                    "enrollment_source": enrollment.get("enrollment_source"),
                    "enrollment_type": enrollment.get("enrollment_type"),
                    "payment_provider": None,
                    "payment_type": "trial",
                    "order_id": None,
                    "payment_intent_id": None,
                    "stripe_charge_id": None,
                    "receipt_number": None,
                    "receipt_url": None,
                    "payment_status": "used",
                    "paid_amount_thb": None,
                    "paid_currency": "THB",
                    "billing_email": student_email or enrollment.get("billing_email"),
                    "plan_label": enrollment.get("plan_label"),
                    "duration_months": schedule["duration_months"],
                    "paid_at": trial_at,
                    "enrolled_at": enrollment.get("enrolled_at"),
                    "started_at": schedule["started_at"],
                    "expires_at": schedule["expires_at"],
                    "is_expired": schedule["is_expired"],
                    "days_remaining": schedule["days_remaining"],
                    "trial_consumed_at": enrollment.get("trial_consumed_at"),
                    "trial_expires_at": enrollment.get("trial_expires_at"),
                }
                if (
                    payment_status_filter
                    and _normalize_status(row.get("payment_status"))
                    != payment_status_filter
                ):
                    continue
                if type_filter != "all" and row["transaction_type"] != type_filter:
                    continue
                rows.append(row)
                continue

            if not payment_events:
                payment_events = [
                    {
                        "payment_provider": enrollment.get("payment_provider")
                        or "manual",
                        "payment_type": enrollment.get("payment_type") or "manual",
                        "order_id": enrollment.get("order_id")
                        or _build_payment_order_id(
                            enrollment.get("paid_at") or enrollment.get("enrolled_at"),
                            enrollment.get("payment_intent_id"),
                        ),
                        "payment_intent_id": enrollment.get("payment_intent_id"),
                        "stripe_charge_id": enrollment.get("stripe_charge_id"),
                        "receipt_number": enrollment.get("receipt_number"),
                        "receipt_url": enrollment.get("receipt_url"),
                        "payment_status": enrollment.get("payment_status") or "active",
                        "paid_amount_thb": enrollment.get("paid_amount_thb"),
                        "paid_currency": enrollment.get("paid_currency") or "THB",
                        "billing_email": enrollment.get("billing_email"),
                        "plan_label": enrollment.get("plan_label"),
                        "duration_months": schedule["duration_months"],
                        "paid_at": enrollment.get("paid_at")
                        or enrollment.get("enrolled_at"),
                        "started_at": schedule["started_at"],
                        "expires_at": schedule["expires_at"],
                    }
                ]

            for idx, event in enumerate(payment_events):
                event_paid_at = event.get("paid_at") or enrollment.get("enrolled_at")
                if not _is_in_window(event_paid_at):
                    continue

                row_type = _classify_transaction_type(False, event)
                transaction_suffix = str(
                    event.get("payment_intent_id") or ""
                ).strip() or str(event_paid_at or "")
                row = {
                    "transaction_id": (
                        f"{enrollment_id or user_id}:{idx}:{row_type}:"
                        f"{transaction_suffix}"
                    ),
                    "transaction_type": row_type,
                    "user_id": user_id,
                    "student_name": student_name,
                    "student_email": student_email,
                    "course_id": course_id,
                    "course_name": course_name,
                    "enrollment_id": enrollment.get("enrollment_id"),
                    "enrollment_source": enrollment.get("enrollment_source"),
                    "enrollment_type": enrollment.get("enrollment_type"),
                    "payment_provider": event.get("payment_provider") or "manual",
                    "payment_type": event.get("payment_type") or "manual",
                    "order_id": event.get("order_id")
                    or _build_payment_order_id(
                        event_paid_at,
                        event.get("payment_intent_id"),
                    ),
                    "payment_intent_id": event.get("payment_intent_id"),
                    "stripe_charge_id": event.get("stripe_charge_id"),
                    "receipt_number": event.get("receipt_number"),
                    "receipt_url": event.get("receipt_url"),
                    "payment_status": event.get("payment_status") or "active",
                    "paid_amount_thb": event.get("paid_amount_thb"),
                    "paid_currency": event.get("paid_currency") or "THB",
                    "billing_email": event.get("billing_email")
                    or student_email
                    or enrollment.get("billing_email"),
                    "plan_label": event.get("plan_label"),
                    "duration_months": event.get("duration_months")
                    or schedule["duration_months"],
                    "paid_at": event_paid_at,
                    "enrolled_at": enrollment.get("enrolled_at"),
                    "started_at": event.get("started_at") or schedule["started_at"],
                    "expires_at": event.get("expires_at") or schedule["expires_at"],
                    "is_expired": schedule["is_expired"],
                    "days_remaining": schedule["days_remaining"],
                    "trial_consumed_at": enrollment.get("trial_consumed_at"),
                    "trial_expires_at": enrollment.get("trial_expires_at"),
                }

                if (
                    payment_status_filter
                    and _normalize_status(row.get("payment_status"))
                    != payment_status_filter
                ):
                    continue
                if type_filter != "all" and row["transaction_type"] != type_filter:
                    continue
                rows.append(row)

        rows.sort(
            key=lambda row: _parse_timestamp_ms(
                row.get("paid_at") or row.get("enrolled_at")
            ),
            reverse=True,
        )

        total_amount_thb = 0.0
        payment_count = 0
        trial_count = 0
        manual_count = 0
        for row in rows:
            row_type = str(row.get("transaction_type") or "").strip().lower()
            if row_type == "payment":
                payment_count += 1
            elif row_type == "trial":
                trial_count += 1
            elif row_type == "manual":
                manual_count += 1

            paid_amount = row.get("paid_amount_thb")
            if paid_amount is None:
                continue
            amount = _safe_float(paid_amount, 0.0)
            if amount > 0:
                total_amount_thb += amount

        total_transactions = len(rows)
        total_pages = max(1, (total_transactions + page_size - 1) // page_size)
        if page > total_pages:
            page = total_pages
        page_start = (page - 1) * page_size
        page_rows = rows[page_start : page_start + page_size]
        await _hydrate_payment_history_receipts(page_rows)

        return {
            "days": days,
            "from_date": since_date.isoformat(),
            "to_date": today.isoformat(),
            "type": type_filter,
            "payment_status": payment_status_filter or "all",
            "total_transactions": total_transactions,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "total_amount_thb": round(total_amount_thb, 2),
            "payment_count": payment_count,
            "trial_count": trial_count,
            "manual_count": manual_count,
            "rows": page_rows,
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error building admin transactions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get admin transactions")


@router.get("/admin/chat-energy/settings")
async def get_admin_chat_energy_settings(
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Get global chat energy settings for admin."""
    try:
        config = await dynamodb_service.get_chat_energy_platform_config()
        return {
            "daily_limit_thb": _safe_float(
                config.get("default_daily_limit_thb"), 0.0
            ),
            "updated_at": str(config.get("updated_at") or "").strip() or None,
            "updated_by": str(config.get("updated_by") or "").strip() or None,
            "reason": str(config.get("reason") or "").strip() or None,
        }
    except Exception as e:
        app_logger.error(f"Error getting admin chat energy settings: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get chat energy settings"
        )


@router.put("/admin/chat-energy/settings")
async def update_admin_chat_energy_settings(
    payload: AdminChatEnergyGlobalConfigRequest = Body(...),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        ADMIN_BEARER_OPTIONAL
    ),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Update global daily chat energy limit (THB/day) for all students."""
    try:
        admin_user_id = await validate_admin_actor(
            str(payload.admin_user_id or "").strip(), credentials
        )

        reason = str(payload.reason or "").strip() or None
        if reason and len(reason) > 500:
            raise HTTPException(
                status_code=400, detail="reason must be <= 500 characters"
            )

        config = await dynamodb_service.set_chat_energy_platform_config(
            default_daily_limit_thb=float(payload.daily_limit_thb or 0.0),
            updated_by=admin_user_id,
            reason=reason,
        )
        return {
            "message": "Global chat energy setting updated",
            "daily_limit_thb": _safe_float(
                config.get("default_daily_limit_thb"), 0.0
            ),
            "updated_at": str(config.get("updated_at") or "").strip() or None,
            "updated_by": str(config.get("updated_by") or "").strip() or None,
            "reason": str(config.get("reason") or "").strip() or None,
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error updating global chat energy settings: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to update chat energy settings"
        )


@router.put("/admin/users/{user_id}/chat-energy")
async def update_admin_user_chat_energy(
    user_id: str,
    payload: AdminUserChatEnergyPolicyRequest = Body(...),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        ADMIN_BEARER_OPTIONAL
    ),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Set per-user chat energy policy (override + adjustment)."""
    try:
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        admin_user_id = await validate_admin_actor(
            str(payload.admin_user_id or "").strip(), credentials
        )

        reason = str(payload.reason or "").strip() or None
        if reason and len(reason) > 500:
            raise HTTPException(
                status_code=400, detail="reason must be <= 500 characters"
            )

        policy = await dynamodb_service.set_user_chat_energy_policy(
            user_id=normalized_user_id,
            daily_limit_override_thb=payload.daily_limit_override_thb,
            daily_adjustment_thb=float(payload.daily_adjustment_thb or 0.0),
            updated_by=admin_user_id,
            reason=reason,
        )
        status = await dynamodb_service.get_student_chat_energy_status(
            normalized_user_id
        )
        return {
            "message": "User chat energy policy updated",
            "user_id": normalized_user_id,
            "policy": {
                "daily_limit_override_thb": (
                    _safe_float(policy.get("daily_limit_override_thb"), 0.0)
                    if policy.get("daily_limit_override_thb") is not None
                    else None
                ),
                "daily_adjustment_thb": _safe_float(
                    policy.get("daily_adjustment_thb"), 0.0
                ),
                "updated_at": str(policy.get("updated_at") or "").strip() or None,
                "updated_by": str(policy.get("updated_by") or "").strip() or None,
                "reason": str(policy.get("reason") or "").strip() or None,
            },
            "status": _to_chat_energy_response(status),
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error updating chat energy policy for user {user_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to update user chat energy policy"
        )


@router.put("/admin/users/{user_id}/trial-status")
async def admin_override_user_trial_status(
    user_id: str,
    payload: AdminUserTrialStatusOverrideRequest = Body(...),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        ADMIN_BEARER_OPTIONAL
    ),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Admin override for user trial/demo status."""
    try:
        normalized_user_id = str(user_id or "").strip()
        normalized_admin_user_id = await validate_admin_actor(
            str(payload.admin_user_id or "").strip(), credentials
        )
        normalized_mode = _normalize_trial_override_mode(payload.mode)
        normalized_reason = str(payload.reason or "").strip()
        if not normalized_user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        if normalized_mode not in TRIAL_OVERRIDE_MODES:
            raise HTTPException(
                status_code=400,
                detail="mode must be one of: auto, available, used",
            )
        if normalized_mode != "auto" and not normalized_reason:
            raise HTTPException(
                status_code=400,
                detail="reason is required when mode is available or used",
            )
        if len(normalized_reason) > 500:
            raise HTTPException(
                status_code=400, detail="reason must be <= 500 characters"
            )

        trial_override = await _set_user_trial_override(
            dynamodb_service=dynamodb_service,
            user_id=normalized_user_id,
            mode=normalized_mode,
            updated_by=normalized_admin_user_id,
            reason=normalized_reason or None,
        )
        user_enrollments = await dynamodb_service.get_user_enrollments(
            normalized_user_id
        )
        trial_used_from_enrollments = any(
            _is_free_course_enrollment(enrollment) for enrollment in user_enrollments
        )
        effective_trial = _resolve_effective_trial_used(
            trial_used_from_enrollments=trial_used_from_enrollments,
            override_mode=trial_override.get("mode"),
        )
        return {
            "message": "User free-course status updated",
            "user_id": normalized_user_id,
            "trial_override_mode": trial_override.get("mode"),
            "trial_override_updated_at": trial_override.get("updated_at"),
            "trial_override_updated_by": trial_override.get("updated_by"),
            "trial_override_reason": trial_override.get("reason"),
            "trial_used": bool(effective_trial.get("trial_used")),
            "trial_available": not bool(effective_trial.get("trial_used")),
            "trial_status_source": effective_trial.get("trial_status_source"),
            "trial_used_from_enrollments": trial_used_from_enrollments,
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error overriding user trial status for {user_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to override user trial status"
        )


@router.put("/admin/users/{user_id}/premium-status")
async def admin_override_user_premium_status(
    user_id: str,
    payload: AdminUserPremiumStatusRequest = Body(...),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        ADMIN_BEARER_OPTIONAL
    ),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Admin override for user Premium / Free tier."""
    try:
        normalized_user_id = str(user_id or "").strip()
        normalized_admin_user_id = await validate_admin_actor(
            str(payload.admin_user_id or "").strip(), credentials
        )
        normalized_tier = str(payload.tier or "").strip().lower()
        normalized_reason = str(payload.reason or "").strip()
        if not normalized_user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        if normalized_tier not in PREMIUM_TIER_MODES:
            raise HTTPException(
                status_code=400,
                detail="tier must be one of: free, premium",
            )
        if not normalized_reason:
            raise HTTPException(status_code=400, detail="reason is required")
        if len(normalized_reason) > 500:
            raise HTTPException(
                status_code=400, detail="reason must be <= 500 characters"
            )

        user = await dynamodb_service.get_user(normalized_user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        existing_subscription = (
            user.get("premium_subscription")
            if isinstance(user.get("premium_subscription"), dict)
            else {}
        )
        previous_tier = (
            "premium" if _is_premium_active(user) else "free"
        )
        now = datetime.utcnow()
        override_event = {
            "updated_at": now.isoformat(),
            "updated_by": normalized_admin_user_id,
            "reason": normalized_reason,
            "previous_tier": previous_tier,
            "requested_tier": normalized_tier,
        }

        if normalized_tier == "free":
            subscription_data = dict(existing_subscription)
            subscription_data["status"] = "expired"
            subscription_data["expires_at"] = _format_utc_iso(now)
            subscription_data["admin_override"] = override_event
        else:
            parsed_expires_at = None
            if payload.expires_at is not None:
                parsed_expires_at = _parse_iso_datetime(payload.expires_at)
                if not parsed_expires_at:
                    raise HTTPException(
                        status_code=400,
                        detail="expires_at must be a valid ISO datetime",
                    )
            duration_months = payload.duration_months
            if parsed_expires_at is None and not duration_months:
                duration_months = 1
            if parsed_expires_at and parsed_expires_at <= now:
                raise HTTPException(
                    status_code=400,
                    detail="expires_at must be in the future for premium tier",
                )
            schedule = _build_enrollment_schedule(
                started_at_raw=now.isoformat(),
                expires_at_raw=_format_utc_iso(parsed_expires_at)
                if parsed_expires_at
                else None,
                duration_months_raw=duration_months,
            )
            if schedule.get("is_expired"):
                raise HTTPException(
                    status_code=400,
                    detail="Premium expiry must be in the future",
                )
            subscription_data = dict(existing_subscription)
            subscription_data.update(
                {
                    "status": "active",
                    "plan_id": "admin",
                    "plan_label": "Admin grant",
                    "duration_months": schedule.get("duration_months"),
                    "started_at": schedule.get("started_at"),
                    "expires_at": schedule.get("expires_at"),
                    "payment_provider": "admin",
                    "payment_type": "admin",
                    "paid_amount_thb": 0.0,
                    "admin_override": override_event,
                }
            )

        save_premium = getattr(dynamodb_service, "save_premium_subscription", None)
        if not callable(save_premium):
            raise HTTPException(
                status_code=500,
                detail="Premium subscription storage is not configured",
            )
        try:
            await save_premium(normalized_user_id, subscription_data)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        updated_user = await dynamodb_service.get_user(normalized_user_id)
        premium_summary = _build_admin_premium_summary(updated_user)
        return {
            "message": "User premium status updated",
            "user_id": normalized_user_id,
            "tier": premium_summary.get("tier"),
            "is_active": premium_summary.get("is_active"),
            "expires_at": premium_summary.get("expires_at"),
            "started_at": premium_summary.get("started_at"),
            "days_remaining": premium_summary.get("days_remaining"),
            "status_source": premium_summary.get("status_source"),
            "admin_override": override_event,
            "premium": premium_summary,
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error overriding user premium status for {user_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to override user premium status"
        )


@router.put("/admin/enrollments/{enrollment_id}/expiry")
async def admin_override_enrollment_expiry(
    enrollment_id: str,
    payload: AdminEnrollmentExpiryOverrideRequest = Body(...),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        ADMIN_BEARER_OPTIONAL
    ),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Admin emergency override for enrollment expiry, used for debugging/incident handling."""
    try:
        admin_user_id = await validate_admin_actor(
            str(payload.admin_user_id or "").strip(), credentials
        )
        reason = str(payload.reason or "").strip()
        if not reason:
            raise HTTPException(status_code=400, detail="reason is required")
        if len(reason) > 500:
            raise HTTPException(
                status_code=400, detail="reason must be <= 500 characters"
            )

        enrollment = await dynamodb_service.get_enrollment_by_id(enrollment_id)
        if not enrollment:
            raise HTTPException(
                status_code=404, detail=f"Enrollment {enrollment_id} not found"
            )

        normalized_expires_at = None
        if payload.expires_at is not None:
            parsed_expires_at = _parse_iso_datetime(payload.expires_at)
            if not parsed_expires_at:
                raise HTTPException(
                    status_code=400, detail="expires_at must be a valid ISO datetime"
                )
            normalized_expires_at = _format_utc_iso(parsed_expires_at)

        previous_expires_at = enrollment.get("expires_at")
        override_event = {
            "overridden_at": datetime.utcnow().isoformat(),
            "overridden_by": admin_user_id,
            "reason": reason,
            "previous_expires_at": previous_expires_at,
            "new_expires_at": normalized_expires_at,
        }
        update_payload = {
            "expires_at": normalized_expires_at,
            "admin_expiry_override": override_event,
        }
        success = await dynamodb_service.update_enrollment(
            enrollment_id, update_payload
        )
        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to override enrollment expiry"
            )

        schedule = _build_enrollment_schedule(
            started_at_raw=enrollment.get("started_at")
            or enrollment.get("enrolled_at"),
            expires_at_raw=normalized_expires_at,
            duration_months_raw=enrollment.get("duration_months"),
        )
        return {
            "message": "Enrollment expiry overridden",
            "enrollment_id": enrollment_id,
            "user_id": enrollment.get("user_id"),
            "course_id": enrollment.get("course_id"),
            "previous_expires_at": previous_expires_at,
            "expires_at": schedule["expires_at"],
            "is_expired": schedule["is_expired"],
            "days_remaining": schedule["days_remaining"],
            "admin_override": override_event,
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error overriding expiry for enrollment {enrollment_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to override enrollment expiry"
        )


# User Enrollment Endpoints
@router.post("/enroll")
async def enroll_user_in_course(
    user_id: str = Form(...),
    course_id: str = Form(...),
    enrollment_mode: str = Form("standard"),
    progress: int = Form(0),
    completed_quizzes: int = Form(0),
    total_quizzes: int = Form(0),
    completed_questions: int = Form(0),
    total_questions: int = Form(0),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Enroll a user in a course."""
    try:
        app_logger.info(f"Enrolling user {user_id} in course {course_id}")

        enrollment_mode_normalized = str(enrollment_mode or "standard").strip().lower()
        if enrollment_mode_normalized not in {"standard", "free"}:
            raise HTTPException(
                status_code=400,
                detail="enrollment_mode must be either 'standard' or 'free'",
            )

        # Check if course exists
        course = await dynamodb_service.get_course(course_id)
        if not course:
            raise HTTPException(status_code=404, detail=f"Course {course_id} not found")

        get_user = getattr(dynamodb_service, "get_user", None)
        user_data = await get_user(user_id) if callable(get_user) else None
        free_course_override = _extract_trial_override(user_data or {})
        reset_override_after_enroll = False

        if enrollment_mode_normalized == "free":
            existing_enrollment = await _get_user_course_enrollment(
                dynamodb_service=dynamodb_service,
                user_id=user_id,
                course_id=course_id,
            )
            if existing_enrollment and _is_free_course_enrollment(existing_enrollment):
                schedule = _build_enrollment_schedule(
                    started_at_raw=existing_enrollment.get("started_at")
                    or existing_enrollment.get("enrolled_at"),
                    expires_at_raw=None,
                    duration_months_raw=None,
                )
                return {
                    "message": "Free course already claimed",
                    "enrollment_id": existing_enrollment.get("enrollment_id"),
                    "user_id": user_id,
                    "course_id": course_id,
                    "enrollment_mode": "free",
                    "is_free_course": True,
                    "expires_at": None,
                }

            user_enrollments = await dynamodb_service.get_user_enrollments(user_id)
            free_used_from_enrollments = any(
                _is_free_course_enrollment(enrollment) for enrollment in user_enrollments
            )
            effective_free = _resolve_effective_trial_used(
                trial_used_from_enrollments=free_used_from_enrollments,
                override_mode=free_course_override.get("mode"),
            )
            if bool(effective_free.get("trial_used")):
                if str(effective_free.get("trial_status_source")) == "admin_override":
                    raise HTTPException(
                        status_code=400,
                        detail="FREE_COURSE_ALREADY_CLAIMED: blocked by admin override mode=used",
                    )
                other_free_enrollment = next(
                    (
                        enrollment
                        for enrollment in user_enrollments
                        if _is_free_course_enrollment(enrollment)
                    ),
                    None,
                )
                other_course_id = str(
                    (other_free_enrollment or {}).get("course_id") or ""
                ).strip()
                raise HTTPException(
                    status_code=400,
                    detail=f"FREE_COURSE_ALREADY_CLAIMED: you already claimed {other_course_id} as your free course",
                )
            reset_override_after_enroll = (
                str(free_course_override.get("mode") or "").strip() == "available"
            )
            if existing_enrollment:
                if _is_premium_active(user_data):
                    raise HTTPException(
                        status_code=400,
                        detail="FREE_COURSE_NOT_ALLOWED: user already has enrollment for this course",
                    )
                started_at = datetime.utcnow().isoformat()
                enrollment_id = str(existing_enrollment.get("enrollment_id") or "").strip()
                if not enrollment_id:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to resolve enrollment for free course conversion",
                    )
                await dynamodb_service.update_enrollment(
                    enrollment_id,
                    {
                        "enrollment_source": "free",
                        "enrollment_type": "free",
                        "free_course_claimed_at": started_at,
                        "last_activity": "ลงทะเบียนเรียนฟรี",
                    },
                )
                if reset_override_after_enroll:
                    try:
                        await _set_user_trial_override(
                            dynamodb_service=dynamodb_service,
                            user_id=user_id,
                            mode="auto",
                            updated_by="system",
                            reason="Auto reset after successful free course enrollment",
                        )
                    except Exception as override_exc:
                        app_logger.warning(
                            "Failed to auto reset free course override after enrollment "
                            f"for user {user_id}: {override_exc}"
                        )
                return {
                    "message": "Free course enrollment created successfully",
                    "enrollment_id": enrollment_id,
                    "user_id": user_id,
                    "course_id": course_id,
                    "enrollment_mode": "free",
                    "is_free_course": True,
                    "expires_at": None,
                }

        # Prepare enrollment data
        started_at = datetime.utcnow().isoformat()
        schedule = _build_enrollment_schedule(
            started_at_raw=started_at,
            expires_at_raw=None,
            duration_months_raw=None,
        )
        default_last_activity = (
            "ลงทะเบียนเรียนฟรี"
            if enrollment_mode_normalized == "free"
            else "เพิ่งเข้าร่วม"
        )
        enrollment_data = {
            "progress": progress,
            "completed_quizzes": completed_quizzes,
            "total_quizzes": total_quizzes,
            "completed_questions": completed_questions,
            "total_questions": total_questions,
            "last_activity": (
                f"{completed_questions}/{total_questions} คำถาม"
                if total_questions > 0
                else default_last_activity
            ),
            "started_at": schedule["started_at"],
            "enrollment_source": "free"
            if enrollment_mode_normalized == "free"
            else "manual",
            "enrollment_type": enrollment_mode_normalized,
        }
        if enrollment_mode_normalized == "free":
            enrollment_data["free_course_claimed_at"] = started_at

        enrollment_id = await dynamodb_service.enroll_user_in_course(
            user_id, course_id, enrollment_data
        )

        if enrollment_mode_normalized == "free" and reset_override_after_enroll:
            try:
                await _set_user_trial_override(
                    dynamodb_service=dynamodb_service,
                    user_id=user_id,
                    mode="auto",
                    updated_by="system",
                    reason="Auto reset after successful free course enrollment",
                )
            except Exception as override_exc:
                app_logger.warning(
                    "Failed to auto reset free course override after enrollment "
                    f"for user {user_id}: {override_exc}"
                )

        return {
            "message": (
                "Free course enrollment created successfully"
                if enrollment_mode_normalized == "free"
                else "User enrolled successfully"
            ),
            "enrollment_id": enrollment_id,
            "user_id": user_id,
            "course_id": course_id,
            "enrollment_mode": enrollment_mode_normalized,
            "is_free_course": enrollment_mode_normalized == "free",
            "expires_at": None,
        }

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error enrolling user {user_id} in course {course_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to enroll user")


@router.get("/users/{user_id}/enrolled-courses")
async def get_user_enrolled_courses(
    user_id: str,
    limit: int = 50,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        STUDENT_BEARER_OPTIONAL
    ),
    student_auth_service: StudentAuthService = Depends(_get_student_auth_service),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Get courses the user is enrolled in."""
    try:
        if credentials:
            await _ensure_user_matches_token(
                user_id=user_id,
                credentials=credentials,
                auth_service=student_auth_service,
            )

        enrolled_courses = await dynamodb_service.get_enrolled_courses_for_user(
            user_id, limit=limit
        )

        # Transform the data to match the expected frontend format
        formatted_courses = []
        for course in enrolled_courses:
            formatted_courses.append(_format_student_course(course))

        return formatted_courses

    except HTTPException:
        raise
    except Exception as e:
        app_logger.exception(f"Error getting enrolled courses for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get enrolled courses")


@router.get("/users/{user_id}/dashboard-learning-summary")
async def get_dashboard_learning_summary(
    user_id: str,
    include_ai: bool = False,
    course_limit: int = 50,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        STUDENT_BEARER_OPTIONAL
    ),
    student_auth_service: StudentAuthService = Depends(_get_student_auth_service),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Return enrolled courses plus computed learning stats for dashboard views."""
    del include_ai
    try:
        if credentials:
            await _ensure_user_matches_token(
                user_id=user_id,
                credentials=credentials,
                auth_service=student_auth_service,
            )

        safe_limit = max(1, min(200, int(course_limit or 50)))
        get_inputs = getattr(dynamodb_service, "get_dashboard_learning_inputs", None)
        if not callable(get_inputs):
            enrolled_courses = await dynamodb_service.get_enrolled_courses_for_user(
                user_id, limit=safe_limit
            )
            quiz_results = await dynamodb_service.get_user_quiz_results(user_id)
            lessons: List[Dict[str, Any]] = []
            quizzes: List[Dict[str, Any]] = []
            for course in enrolled_courses:
                course_id = str(course.get("course_id") or course.get("id") or "")
                if not course_id:
                    continue
                lessons.extend(await dynamodb_service.get_course_lessons(course_id))
                quizzes.extend(await dynamodb_service.get_quizzes_by_course(course_id))
            formatted_courses = [
                _format_student_course(course) for course in enrolled_courses
            ]
            return {
                "user_id": user_id,
                "courses": formatted_courses,
                "course_stats": _build_dashboard_course_stats(
                    enrolled_courses, lessons, quizzes, quiz_results
                ),
                "generated_at": datetime.utcnow().isoformat(),
            }

        inputs = await get_inputs(user_id, limit=safe_limit)
        candidate_user_ids = inputs.get("candidate_user_ids") or []
        candidate_rank = {
            str(candidate_id): idx
            for idx, candidate_id in enumerate(candidate_user_ids)
        }
        courses_by_id = {
            str(course.get("course_id") or ""): course
            for course in inputs.get("courses", [])
            if str(course.get("course_id") or "")
        }

        merged_courses = []
        for enrollment in inputs.get("enrollments", []):
            course_id = str(enrollment.get("course_id") or "").strip()
            course = courses_by_id.get(course_id)
            if not course:
                continue
            merged_courses.append(_merge_course_with_enrollment(course, enrollment))

        deduped_results = []
        seen_result_ids = set()
        for row in sorted(
            inputs.get("quiz_results", []),
            key=lambda item: (
                candidate_rank.get(str(item.get("user_id") or ""), len(candidate_rank)),
                str(item.get("submitted_at") or item.get("created_at") or ""),
            ),
        ):
            result_id = str(row.get("result_id") or "").strip()
            if result_id and result_id in seen_result_ids:
                continue
            if result_id:
                seen_result_ids.add(result_id)
            deduped_results.append(row)
        deduped_results.sort(
            key=lambda row: str(row.get("submitted_at") or row.get("created_at") or ""),
            reverse=True,
        )

        return {
            "user_id": user_id,
            "courses": [_format_student_course(course) for course in merged_courses],
            "course_stats": _build_dashboard_course_stats(
                merged_courses,
                inputs.get("lessons", []),
                inputs.get("quizzes", []),
                deduped_results,
            ),
            "generated_at": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        app_logger.exception(
            f"Error building dashboard learning summary for {user_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail="Failed to get dashboard learning summary"
        )


@router.get("/courses/{course_id}/mock-exam-leaderboard")
async def get_course_mock_exam_leaderboard(
    course_id: str, limit: int = 50, dynamodb_service=Depends(get_dynamodb_service)
):
    """Get student ranking by average score from mock exams in a course."""
    try:
        course = await dynamodb_service.get_course(course_id)
        if not course:
            raise HTTPException(status_code=404, detail=f"Course {course_id} not found")

        quizzes = await dynamodb_service.get_quizzes_by_course(course_id)
        mock_quiz_ids = {
            str(item.get("quiz_id") or "")
            for item in quizzes
            if str(item.get("document_type") or "").strip().lower() == "mock_exam"
            and str(item.get("quiz_id") or "").strip()
        }

        enrollments = await dynamodb_service.get_course_enrollments(course_id)
        user_ids = [
            str(enrollment.get("user_id") or "").strip()
            for enrollment in enrollments
            if str(enrollment.get("user_id") or "").strip()
        ]
        get_users_by_ids = getattr(dynamodb_service, "get_users_by_ids", None)
        users_by_id: Dict[str, Dict[str, Any]] = {}
        if callable(get_users_by_ids) and user_ids:
            users_by_id = {
                str(user.get("user_id") or ""): user
                for user in await get_users_by_ids(user_ids, limit=len(user_ids))
                if str(user.get("user_id") or "")
            }

        get_results_for_course = getattr(
            dynamodb_service, "get_quiz_results_for_course", None
        )
        results_by_user: Dict[str, List[Dict[str, Any]]] = {
            user_id: [] for user_id in user_ids
        }
        if callable(get_results_for_course) and user_ids and mock_quiz_ids:
            course_results = await get_results_for_course(
                course_id,
                user_ids=user_ids,
                quiz_ids=list(mock_quiz_ids),
                limit=max(1000, len(user_ids) * max(1, len(mock_quiz_ids)) * 20),
            )
            for result in course_results:
                result_user_id = str(result.get("user_id") or "").strip()
                if result_user_id in results_by_user:
                    results_by_user[result_user_id].append(result)

        def _display_name(user_id: str, user_data: Optional[Dict[str, Any]]) -> str:
            user_data = user_data or {}
            onboarding_profile = (
                user_data.get("onboarding_profile")
                if isinstance(user_data, dict)
                else {}
            )
            nickname = str((onboarding_profile or {}).get("nickname") or "").strip()
            given_name = str((user_data or {}).get("given_name") or "").strip()
            family_name = str((user_data or {}).get("family_name") or "").strip()
            full_name = f"{given_name} {family_name}".strip()
            fallback_name = str((user_data or {}).get("name") or "").strip()
            fallback_email = str((user_data or {}).get("email") or "").strip()
            email_name = fallback_email.split("@")[0] if "@" in fallback_email else ""
            return (
                nickname
                or full_name
                or (fallback_name if fallback_name and fallback_name != user_id else "")
                or email_name
                or user_id
                or "ผู้เรียน"
            )

        rankings = []

        for enrollment in enrollments:
            user_id = str(enrollment.get("user_id") or "").strip()
            if not user_id:
                continue

            user_data = users_by_id.get(user_id)
            if not user_data and not callable(get_users_by_ids):
                user_data = await dynamodb_service.get_user(user_id)
            display_name = _display_name(user_id, user_data)

            if callable(get_results_for_course):
                candidate_results = results_by_user.get(user_id, [])
            else:
                candidate_results = await dynamodb_service.get_user_quiz_results(
                    user_id, course_id=course_id
                )
            mock_results = []
            for item in candidate_results:
                result_quiz_id = str(item.get("quiz_id") or "").strip()
                result_course_id = str(item.get("course_id") or "").strip()
                if result_course_id and result_course_id != str(course_id):
                    continue
                if result_quiz_id in mock_quiz_ids:
                    score = item.get("score")
                    if isinstance(score, (int, float)):
                        mock_results.append(item)

            if not mock_results:
                continue

            scores = [float(item.get("score")) for item in mock_results]
            time_values = [
                float(item.get("time_spent_seconds"))
                for item in mock_results
                if isinstance(item.get("time_spent_seconds"), (int, float))
                and float(item.get("time_spent_seconds")) >= 0
            ]
            average_score = round(sum(scores) / len(scores), 2)
            best_score = round(max(scores), 2)
            average_time_seconds = (
                round(sum(time_values) / len(time_values), 2) if time_values else None
            )
            latest_result = max(
                mock_results, key=lambda row: str(row.get("submitted_at") or "")
            )

            rankings.append(
                {
                    "user_id": user_id,
                    "display_name": display_name,
                    "average_score": average_score,
                    "best_score": best_score,
                    "average_time_seconds": average_time_seconds,
                    "attempt_count": len(scores),
                    "last_submitted_at": latest_result.get("submitted_at"),
                    "enrolled_at": enrollment.get("enrolled_at"),
                }
            )

        rankings.sort(
            key=lambda row: (
                float(row.get("average_score") or 0),
                int(row.get("attempt_count") or 0),
                str(row.get("last_submitted_at") or ""),
            ),
            reverse=True,
        )

        for index, row in enumerate(rankings, start=1):
            row["rank"] = index

        if isinstance(limit, int) and limit > 0:
            rankings = rankings[:limit]

        return {
            "course_id": course_id,
            "course_name": course.get("name") or course.get("title") or "คอร์สเรียน",
            "metric": "average_mock_exam_score",
            "total_students": len(rankings),
            "mock_exam_count": len(mock_quiz_ids),
            "rankings": rankings,
            "generated_at": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(
            f"Error getting mock exam leaderboard for course {course_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail="Failed to get mock exam leaderboard"
        )


@router.get("/courses/{course_id}/students")
async def get_course_students(
    course_id: str,
    page: int = 1,
    page_size: int = 50,
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Get enrolled students for a course."""
    try:
        page = max(1, int(page or 1))
        page_size = max(1, min(200, int(page_size or 50)))
        get_enrollments_page = getattr(
            dynamodb_service, "get_course_enrollments_page", None
        )
        if callable(get_enrollments_page):
            enrollments_page = await get_enrollments_page(
                course_id, page=page, page_size=page_size
            )
            enrollments = enrollments_page.get("rows") or []
            total_students = int(enrollments_page.get("total") or len(enrollments))
            total_pages = int(enrollments_page.get("total_pages") or 1)
            page = int(enrollments_page.get("page") or page)
            page_size = int(enrollments_page.get("page_size") or page_size)
        else:
            all_enrollments = await dynamodb_service.get_course_enrollments(course_id)
            total_students = len(all_enrollments)
            total_pages = max(1, (total_students + page_size - 1) // page_size)
            if page > total_pages:
                page = total_pages
            start = (page - 1) * page_size
            enrollments = all_enrollments[start : start + page_size]

        user_ids = [
            str(enrollment.get("user_id") or "").strip()
            for enrollment in enrollments
            if str(enrollment.get("user_id") or "").strip()
        ]
        users_by_id: Dict[str, Dict[str, Any]] = {}
        get_users_by_ids = getattr(dynamodb_service, "get_users_by_ids", None)
        if callable(get_users_by_ids) and user_ids:
            users_by_id = {
                str(user.get("user_id") or ""): user
                for user in await get_users_by_ids(user_ids, limit=len(user_ids))
                if str(user.get("user_id") or "")
            }

        results_by_user: Dict[str, List[Dict[str, Any]]] = {
            user_id: [] for user_id in user_ids
        }
        get_results_for_course = getattr(
            dynamodb_service, "get_quiz_results_for_course", None
        )
        if callable(get_results_for_course) and user_ids:
            course_results = await get_results_for_course(
                course_id,
                user_ids=user_ids,
                limit=max(1000, len(user_ids) * 200),
            )
            for result in course_results:
                result_user_id = str(result.get("user_id") or "").strip()
                if result_user_id in results_by_user:
                    results_by_user[result_user_id].append(result)

        def _display_name(user_id: str, user_data: Optional[Dict[str, Any]]) -> str:
            user_data = user_data or {}
            onboarding_profile = (
                user_data.get("onboarding_profile")
                if isinstance(user_data, dict)
                else {}
            )
            nickname = str((onboarding_profile or {}).get("nickname") or "").strip()
            given_name = str((user_data or {}).get("given_name") or "").strip()
            family_name = str((user_data or {}).get("family_name") or "").strip()
            full_name = f"{given_name} {family_name}".strip()
            fallback_name = str((user_data or {}).get("name") or "").strip()
            fallback_email = str((user_data or {}).get("email") or "").strip()
            email_name = fallback_email.split("@")[0] if "@" in fallback_email else ""
            return (
                nickname
                or full_name
                or (fallback_name if fallback_name and fallback_name != user_id else "")
                or email_name
                or user_id
                or "ผู้เรียน"
            )

        students = []

        for enrollment in enrollments:
            user_id = str(enrollment.get("user_id") or "").strip()
            user_data = users_by_id.get(user_id)
            if not user_data and user_id and not callable(get_users_by_ids):
                user_data = await dynamodb_service.get_user(user_id)
            display_name = _display_name(user_id, user_data)

            # Aggregate basic performance from quiz results in this course.
            if callable(get_results_for_course):
                course_results = results_by_user.get(user_id, [])
            else:
                all_results = (
                    await dynamodb_service.get_user_quiz_results(user_id)
                    if user_id
                    else []
                )
                course_results = [
                    item
                    for item in all_results
                    if str(item.get("course_id") or "") == str(course_id)
                ]
            attempts = len(course_results)
            latest_score = None
            best_score = None
            average_score = None
            last_submitted_at = None
            if attempts > 0:
                scores = [
                    float(item.get("score"))
                    for item in course_results
                    if isinstance(item.get("score"), (int, float))
                ]
                if scores:
                    latest_score = int(round(scores[0]))
                    best_score = int(round(max(scores)))
                    average_score = int(round(sum(scores) / len(scores)))
                last_submitted_at = course_results[0].get("submitted_at")

            students.append(
                {
                    "user_id": user_id,
                    "name": user_data.get("name") if user_data else user_id,
                    "display_name": display_name,
                    "email": user_data.get("email") if user_data else None,
                    "enrollment_id": enrollment.get("enrollment_id"),
                    "enrolled_at": enrollment.get("enrolled_at"),
                    "progress": enrollment.get("progress", 0),
                    "last_activity": enrollment.get("last_activity"),
                    "performance": {
                        "attempts": attempts,
                        "latest_score": latest_score,
                        "best_score": best_score,
                        "average_score": average_score,
                        "last_submitted_at": last_submitted_at,
                    },
                }
            )

        return {
            "course_id": course_id,
            "total_students": total_students,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "students": students,
        }
    except Exception as e:
        app_logger.error(f"Error getting students for course {course_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get course students")


@router.get("/courses/{course_id}/tutor-overview")
async def get_course_tutor_overview(
    course_id: str,
    page: int = 1,
    page_size: int = 50,
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Return compact tutor course management data in one request."""
    try:
        page = max(1, int(page or 1))
        page_size = max(1, min(200, int(page_size or 50)))
        course_task = dynamodb_service.get_course(course_id)
        try:
            lessons_task = dynamodb_service.get_course_lessons(
                course_id, summary=True
            )
        except TypeError:
            lessons_task = dynamodb_service.get_course_lessons(course_id)
        try:
            quizzes_task = dynamodb_service.get_quizzes_by_course(
                course_id, summary=True
            )
        except TypeError:
            quizzes_task = dynamodb_service.get_quizzes_by_course(course_id)
        students_task = get_course_students(
            course_id,
            page=page,
            page_size=page_size,
            dynamodb_service=dynamodb_service,
        )

        course, lessons, quizzes, students_payload = await asyncio.gather(
            course_task, lessons_task, quizzes_task, students_task
        )
        if not course:
            raise HTTPException(status_code=404, detail=f"Course {course_id} not found")

        total_questions = 0
        for quiz in quizzes or []:
            if isinstance(quiz.get("total_questions"), (int, float)):
                total_questions += int(quiz.get("total_questions") or 0)
            elif isinstance(quiz.get("questions"), list):
                total_questions += len(quiz.get("questions") or [])

        scores = []
        for student in students_payload.get("students", []):
            average_score = (student.get("performance") or {}).get("average_score")
            if isinstance(average_score, (int, float)):
                scores.append(float(average_score))

        return {
            "course_id": course_id,
            "course": course,
            "lessons": lessons,
            "quizzes": quizzes,
            "students": students_payload.get("students", []),
            "students_page": {
                key: students_payload.get(key)
                for key in (
                    "total_students",
                    "page",
                    "page_size",
                    "total_pages",
                    "has_next",
                    "has_prev",
                )
            },
            "stats": {
                "totalLessons": len(lessons or []),
                "totalQuizzes": len(quizzes or []),
                "totalQuestions": total_questions,
                "totalStudents": int(students_payload.get("total_students") or 0),
                "completedTests": sum(
                    int((student.get("performance") or {}).get("attempts") or 0)
                    for student in students_payload.get("students", [])
                ),
                "averageScore": int(round(sum(scores) / len(scores))) if scores else 0,
            },
            "generated_at": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error building tutor overview for {course_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get tutor overview")


@router.get("/enrollments/{user_id}")
async def get_user_enrollments(
    user_id: str, limit: int = 50, dynamodb_service=Depends(get_dynamodb_service)
):
    """Get all enrollments for a specific user."""
    try:
        # Compat: some DynamoDB service implementations accept only (user_id).
        try:
            enrollments = await dynamodb_service.get_user_enrollments(user_id, limit)
        except TypeError:
            enrollments = await dynamodb_service.get_user_enrollments(user_id)
        enrollments = list(enrollments or [])
        enrollments.sort(
            key=lambda row: str(row.get("enrolled_at") or ""), reverse=True
        )
        if isinstance(limit, int) and limit > 0:
            enrollments = enrollments[:limit]

        return {
            "user_id": user_id,
            "total_enrollments": len(enrollments),
            "enrollments": enrollments,
        }

    except Exception as e:
        app_logger.error(f"Error getting enrollments for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user enrollments")


@router.put("/enrollments/{enrollment_id}")
async def update_enrollment(
    enrollment_id: str,
    progress: Optional[int] = Form(None),
    completed_quizzes: Optional[int] = Form(None),
    total_quizzes: Optional[int] = Form(None),
    completed_questions: Optional[int] = Form(None),
    total_questions: Optional[int] = Form(None),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Update enrollment progress."""
    try:
        # Prepare updates (only include non-None values)
        updates = {}
        if progress is not None:
            updates["progress"] = progress
        if completed_quizzes is not None:
            updates["completed_quizzes"] = completed_quizzes
        if total_quizzes is not None:
            updates["total_quizzes"] = total_quizzes
        if completed_questions is not None:
            updates["completed_questions"] = completed_questions
        if total_questions is not None:
            updates["total_questions"] = total_questions

        # Update last activity
        if completed_questions is not None and total_questions is not None:
            updates["last_activity"] = f"{completed_questions}/{total_questions} คำถาม"

        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")

        success = await dynamodb_service.update_enrollment(enrollment_id, updates)

        if success:
            return {"message": f"Enrollment {enrollment_id} updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update enrollment")

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error updating enrollment {enrollment_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update enrollment")


@router.delete("/enrollments/{enrollment_id}")
async def cancel_enrollment(
    enrollment_id: str, dynamodb_service=Depends(get_dynamodb_service)
):
    """Cancel a user enrollment."""
    try:
        success = await dynamodb_service.cancel_enrollment(enrollment_id)

        if success:
            return {"message": f"Enrollment {enrollment_id} cancelled successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to cancel enrollment")

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error cancelling enrollment {enrollment_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel enrollment")


# Lesson endpoints
@router.post("/courses/{course_id}/lessons", response_model=LessonResponse)
async def create_lesson(
    course_id: str, request_body: dict, dynamodb_service=Depends(get_dynamodb_service)
):
    """Create a new lesson in a course."""
    try:
        # Extract lesson data from request body
        lesson_data = {
            "title": request_body.get("title"),
            "description": request_body.get("description", ""),
            "order": request_body.get("order", 1),
            "selectedDocuments": request_body.get("selectedDocuments", []),
            "selectedQuizzes": request_body.get("selectedQuizzes", []),
            "isPublished": request_body.get("isPublished", False),
        }

        # Extract user_id from request body (sent by frontend)
        user_id = request_body.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        # Validate required fields
        if not lesson_data.get("title"):
            raise HTTPException(status_code=400, detail="title is required")

        # Verify course exists and user has permission
        course = await dynamodb_service.get_course(course_id)
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")

        # Store lesson in DynamoDB
        lesson_id = await dynamodb_service.create_lesson(
            user_id=user_id, course_id=course_id, lesson_data=lesson_data
        )

        return LessonResponse(
            success=True, message="Lesson created successfully", lesson_id=lesson_id
        )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error creating lesson for course {course_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create lesson")


@router.get("/courses/{course_id}/lessons", response_model=LessonListResponse)
async def get_course_lessons(
    course_id: str,
    user_id: Optional[str] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        STUDENT_BEARER_OPTIONAL
    ),
    student_auth_service: StudentAuthService = Depends(_get_student_auth_service),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Get all lessons for a specific course."""
    try:
        if user_id and str(course_id or "").strip():
            await _ensure_user_matches_token(
                user_id=user_id,
                credentials=credentials,
                auth_service=student_auth_service,
            )
            await _ensure_active_course_access(
                dynamodb_service=dynamodb_service,
                user_id=user_id,
                course_id=course_id,
            )

        lessons = await dynamodb_service.get_course_lessons(
            course_id=course_id, user_id=None
        )

        normalized_lessons = []
        for lesson_item in lessons:
            # Normalize selected documents from both legacy and current keys
            documents = []
            raw_documents = (
                lesson_item.get("documents")
                or lesson_item.get("selectedDocuments")
                or lesson_item.get("selected_documents")
                or []
            )
            for doc in raw_documents:
                if isinstance(doc, str):
                    documents.append({"id": doc, "title": None, "type": None})
                    continue
                if isinstance(doc, dict):
                    doc_id = doc.get("id") or doc.get("document_id")
                    if not doc_id:
                        continue
                    documents.append(
                        {
                            "id": doc_id,
                            "title": doc.get("title"),
                            "type": doc.get("type"),
                        }
                    )

            # Normalize selected quizzes from both legacy and current keys
            quizzes = []
            raw_quizzes = (
                lesson_item.get("quizzes")
                or lesson_item.get("selectedQuizzes")
                or lesson_item.get("selected_quizzes")
                or []
            )
            for q in raw_quizzes:
                if isinstance(q, str):
                    quizzes.append({"id": q, "title": None, "questions": 0})
                    continue
                if isinstance(q, dict):
                    qid = q.get("id") or q.get("quiz_id") or q.get("document_id")
                    if not qid:
                        continue
                    questions_val = q.get("questions", 0)
                    if isinstance(questions_val, list):
                        questions_count = len(questions_val)
                    elif isinstance(questions_val, int):
                        questions_count = questions_val
                    else:
                        questions_count = 0
                    quizzes.append(
                        {
                            "id": qid,
                            "title": q.get("title"),
                            "questions": questions_count,
                        }
                    )

            normalized_lessons.append(
                {
                    "id": lesson_item.get("lesson_id") or lesson_item.get("id"),
                    "title": lesson_item.get("title"),
                    "description": lesson_item.get("description", ""),
                    "order": lesson_item.get("order", 1),
                    "courseId": lesson_item.get("course_id")
                    or lesson_item.get("courseId"),
                    "userId": lesson_item.get("user_id") or lesson_item.get("userId"),
                    "documents": documents,
                    "quizzes": quizzes,
                    "isPublished": lesson_item.get(
                        "isPublished", lesson_item.get("is_published", False)
                    ),
                    "createdAt": lesson_item.get("created_at")
                    or lesson_item.get("createdAt"),
                    "updatedAt": lesson_item.get("updated_at")
                    or lesson_item.get("updatedAt"),
                }
            )

        return LessonListResponse(
            lessons=normalized_lessons, total=len(normalized_lessons)
        )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error retrieving lessons for course {course_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve lessons")


@router.put("/lessons/{lesson_id}", response_model=LessonResponse)
async def update_lesson(
    lesson_id: str, request_body: dict, dynamodb_service=Depends(get_dynamodb_service)
):
    """Update a lesson."""
    try:
        # Extract user_id from request body (sent by frontend)
        user_id = request_body.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        # Extract lesson data from request body
        lesson_data = {}
        for field in [
            "title",
            "description",
            "order",
            "selectedDocuments",
            "selectedQuizzes",
            "isPublished",
        ]:
            if field in request_body:
                lesson_data[field] = request_body[field]

        # Update lesson in DynamoDB
        success = await dynamodb_service.update_lesson(
            lesson_id=lesson_id, lesson_data=lesson_data, user_id=user_id
        )

        if success:
            return LessonResponse(
                success=True, message="Lesson updated successfully", lesson_id=lesson_id
            )
        else:
            raise HTTPException(
                status_code=404, detail="Lesson not found or unauthorized"
            )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error updating lesson {lesson_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update lesson")


@router.delete("/lessons/{lesson_id}")
async def delete_lesson(
    lesson_id: str,
    user_id: str = Form(...),
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Delete a lesson (soft delete)."""
    try:
        success = await dynamodb_service.delete_lesson(
            lesson_id=lesson_id, user_id=user_id
        )

        if success:
            return {"message": f"Lesson {lesson_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=404, detail="Lesson not found or unauthorized"
            )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error deleting lesson {lesson_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete lesson")


@router.get("/lessons/{lesson_id}")
async def get_lesson(
    lesson_id: str,
    user_id: Optional[str] = None,
    dynamodb_service=Depends(get_dynamodb_service),
):
    """Get a specific lesson by ID using the configured DynamoDB service."""
    try:
        # Retrieve raw lesson item via adapter (handles separated/enhanced services)
        lesson_item = await dynamodb_service.get_lesson(lesson_id)

        if not lesson_item:
            raise HTTPException(status_code=404, detail="Lesson not found")

        # Ensure status is active
        if lesson_item.get("status") not in (None, "active"):
            raise HTTPException(status_code=404, detail="Lesson not found")

        # Normalize selected documents
        documents = []
        for doc in lesson_item.get("selected_documents", []) or []:
            if isinstance(doc, dict):
                if not doc.get("id"):
                    continue
                documents.append(
                    {
                        "id": doc.get("id"),
                        "title": doc.get("title"),
                        "type": doc.get("type"),
                    }
                )
        # Normalize selected quizzes: ensure id and convert questions list -> count
        quizzes = []
        for q in lesson_item.get("selected_quizzes", []) or []:
            if isinstance(q, str):
                quizzes.append({"id": q, "title": None, "questions": 0})
                continue
            if isinstance(q, dict):
                qid = q.get("id") or q.get("quiz_id") or q.get("document_id")
                if not qid:
                    # Skip entries without a resolvable id
                    continue
                questions_val = q.get("questions", 0)
                if isinstance(questions_val, list):
                    questions_count = len(questions_val)
                elif isinstance(questions_val, int):
                    questions_count = questions_val
                else:
                    questions_count = 0
                quizzes.append(
                    {
                        "id": qid,
                        "title": q.get("title"),
                        "questions": questions_count,
                    }
                )

        # Normalize to frontend shape
        lesson = {
            "id": lesson_item.get("lesson_id"),
            "title": lesson_item.get("title"),
            "description": lesson_item.get("description"),
            "order": lesson_item.get("order", 1),
            "courseId": lesson_item.get("course_id"),
            "userId": lesson_item.get("user_id"),
            "documents": documents,
            "quizzes": quizzes,
            "isPublished": lesson_item.get("is_published", False),
            "createdAt": lesson_item.get("created_at"),
            "updatedAt": lesson_item.get("updated_at"),
        }

        return LessonResponse(
            success=True,
            message="Lesson retrieved successfully",
            lesson=lesson,
        )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error retrieving lesson {lesson_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve lesson")
