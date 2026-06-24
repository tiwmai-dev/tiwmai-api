"""Chat service using OpenRouter (OpenAI-compatible) for intelligent course-aware responses."""

import asyncio
import base64
import json
import re
import time
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, TypedDict

try:
    from langchain.memory import ConversationBufferMemory
    from langchain_core.messages import AIMessage, HumanMessage
except Exception:  # pragma: no cover - optional dependency

    class _BaseMessage:
        def __init__(
            self, content: str, additional_kwargs: Optional[Dict[str, Any]] = None
        ):
            self.content = content
            self.additional_kwargs = additional_kwargs or {}

    class HumanMessage(_BaseMessage):
        type = "human"

    class AIMessage(_BaseMessage):
        type = "ai"

    class _SimpleChatMemory:
        def __init__(self):
            self.messages: List[Any] = []

        def add_message(self, message: Any) -> None:
            self.messages.append(message)

    class ConversationBufferMemory:  # type: ignore[override]
        def __init__(self, return_messages: bool = True):
            self.return_messages = return_messages
            self.chat_memory = _SimpleChatMemory()


from openai import AsyncOpenAI

try:
    from langgraph.graph import END, StateGraph

    _LANGGRAPH_AVAILABLE = True
except Exception:
    StateGraph = None
    END = None
    _LANGGRAPH_AVAILABLE = False

from app.core.config import get_settings
from app.core.exceptions import LLMProcessingError
from app.core.logging import app_logger
from app.models.schemas import ChatMessage, ChatResponse, ConversationHistory
from app.services.data_service import get_db_service
from app.services.llm_params import (
    build_llm_extra_params,
    supports_response_format,
)


class _ChatRouterState(TypedDict, total=False):
    user_message: str
    chat_mode: str
    has_image: bool
    parsed_question_context: Optional[Any]
    has_course_context: bool
    should_include_question_context: bool
    question_context_for_prompt: Optional[Any]
    should_include_course_context: bool
    should_include_system_context: bool
    context_route: str
    classifier_reason: str
    question_context_alignment_reason: str


class ChatService:
    """Service for intelligent chat responses using OpenRouter."""

    _session_memory: Dict[str, ConversationBufferMemory] = {}
    _session_summary: Dict[str, str] = {}
    _chat_router_graph = None
    _default_chat_mode = "study_solver"
    _allowed_chat_modes = {"study_solver", "learning_advisor"}
    _system_context = {
        "platform_name": "Tanaijarn",
        "capabilities": [
            "ตอบคำถามเรื่องการเรียน",
            "ช่วยวางแผนการอ่านและเป้าหมาย",
            "แนะนำการใช้งานระบบและเมนู",
            "ติดตามความคืบหน้าของคอร์ส",
        ],
    }

    def __init__(self):
        self.settings = get_settings()
        self._usage_events: List[Dict[str, Any]] = []

        # Configure OpenRouter OpenAI-compatible client
        self._aclient = AsyncOpenAI(
            base_url=self.settings.openrouter_base_url,
            api_key=self.settings.openrouter_api_key or "",
        )

        # In-memory conversation storage via LangChain (per session)

        # Course context information
        self.course_contexts = {
            "8b9476d2-634b-4204-a23d-09c143be1c8a": {  # Nursing course ID from frontend
                "name": "การพยาบาลพื้นฐาน",
                "description": "หลักสูตรการพยาบาลพื้นฐาน ครอบคลุมการดูแลผู้ป่วย การประเมินสัญญาณชีพ และการจัดการสุขภาพองค์รวม",
                "topics": [
                    "สัญญาณชีพ (Vital Signs)",
                    "การดูแลผู้ป่วยพื้นฐาน",
                    "หลักการพยาบาลองค์รวม",
                    "การประเมินสุขภาพ",
                    "การป้องกันการติดเชื้อ",
                    "การให้ยาและการคำนวณ",
                    "การดูแลบาดแผล",
                    "จริยธรรมการพยาบาล",
                    "การสื่อสารกับผู้ป่วย",
                    "การทำงานเป็นทีม",
                ],
                "language": "th",
            }
        }

    async def get_chat_response(
        self,
        user_message: str,
        user_id: str,
        course_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        question_context: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_mime: Optional[str] = None,
        chat_mode: Optional[str] = None,
    ) -> ChatResponse:
        """
        Generate intelligent AI response using Gemini API with course context.

        Args:
            user_message: User's message
            user_id: User identifier
            course_id: Course ID for context (optional)
            conversation_id: Conversation ID for context (optional)

        Returns:
            ChatResponse with AI response
        """
        start_time = datetime.utcnow()
        self._usage_events = []
        db_service = get_db_service()
        energy_status: Optional[Dict[str, Any]] = None

        def _energy_fields(status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            if not isinstance(status, dict):
                return {}
            return {
                "chat_energy_limit_thb": status.get("daily_limit_thb"),
                "chat_energy_used_thb": status.get("used_thb"),
                "chat_energy_remaining_thb": status.get("remaining_thb"),
                "chat_energy_percent": status.get("remaining_percent"),
                "chat_energy_exhausted": bool(status.get("is_exhausted")),
            }

        try:
            normalized_mode = self.normalize_chat_mode(chat_mode)
            # Generate or use existing conversation ID
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            context_started = time.monotonic()
            needs_learning_context = (
                normalized_mode == "learning_advisor"
                and self._learning_advisor_needs_learning_context(user_message)
            )
            needs_course_context = (
                normalized_mode == "study_solver"
                and bool(course_id)
                and course_id != "general-ai-tutor"
            )

            async def _load_energy_status() -> Optional[Dict[str, Any]]:
                try:
                    return await db_service.get_student_chat_energy_status(user_id)
                except Exception as energy_error:
                    app_logger.warning(
                        f"Unable to load chat energy for {user_id}: {energy_error}"
                    )
                    return None

            (
                energy_status,
                enrolled_course_context,
                advisor_learning_context,
                openrouter_context,
            ) = await asyncio.gather(
                _load_energy_status(),
                self._get_course_context_for_user(
                    user_id=user_id,
                    course_id=course_id,
                )
                if needs_course_context
                else asyncio.sleep(0, result=None),
                self._get_learning_overview_for_user(user_id)
                if needs_learning_context
                else asyncio.sleep(0, result=None),
                self._get_openrouter_user_context(user_id),
            )
            openrouter_user, openrouter_metadata = openrouter_context
            app_logger.info(
                "Chat context prepared for user {} in {}ms "
                "(mode={} learning_context={} course_context={})",
                user_id,
                int((time.monotonic() - context_started) * 1000),
                normalized_mode,
                needs_learning_context,
                needs_course_context,
            )

            if isinstance(energy_status, dict) and bool(
                energy_status.get("is_exhausted")
            ):
                end_time = datetime.utcnow()
                processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
                return ChatResponse(
                    message_id=str(uuid.uuid4()),
                    content=(
                        "พลังงานแชทหมดแล้วสำหรับวันนี้\n"
                        "กรุณารอรอบวันถัดไป หรือให้แอดมินเพิ่มพลังงานให้ก่อนใช้งานต่อ"
                    ),
                    timestamp=end_time,
                    confidence=1.0,
                    course_context=None,
                    conversation_id=conversation_id,
                    processing_time_ms=processing_time_ms,
                    **_energy_fields(energy_status),
                )

            # Get conversation history for context
            conversation_history = self._get_conversation_history(conversation_id)
            conversation_summary = self._session_summary.get(conversation_id, "")

            parsed_question_context = self._parse_question_context(question_context)
            routed_context = await self._route_question_context_with_langgraph(
                user_message=user_message,
                parsed_question_context=parsed_question_context,
                has_image=bool(image_bytes),
                has_course_context=bool(enrolled_course_context),
                chat_mode=normalized_mode,
            )
            effective_question_context = routed_context.get(
                "question_context_for_prompt"
            )
            effective_course_context = (
                enrolled_course_context
                if routed_context.get("should_include_course_context")
                else None
            )
            effective_system_context = (
                self._system_context
                if routed_context.get("should_include_system_context")
                else None
            )

            # Create intelligent prompt with context
            prompt = self._create_chat_prompt(
                user_message=user_message,
                course_context=effective_course_context,
                system_context=effective_system_context,
                conversation_history=conversation_history,
                conversation_summary=conversation_summary,
                user_id=user_id,
                question_context=effective_question_context,
                has_image=bool(image_bytes),
                chat_mode=normalized_mode,
                learning_context=advisor_learning_context,
            )

            app_logger.info(
                f"Generating chat response for user {user_id} in course {course_id}"
            )

            # Call OpenRouter Chat API
            base_max_tokens = 1000
            model_name = self._resolve_chat_model(
                chat_mode=normalized_mode,
                has_image=bool(image_bytes),
            )
            if image_bytes:
                app_logger.info(
                    f"Sending image to LLM: bytes={len(image_bytes)} mime={image_mime or 'image/png'}"
                )
                response = await self._call_gemini_chat_with_image(
                    prompt,
                    image_bytes,
                    image_mime or "image/png",
                    base_max_tokens,
                    model_name=model_name,
                    openrouter_user=openrouter_user,
                    openrouter_metadata=openrouter_metadata,
                )
            else:
                response = await self._call_gemini_chat(
                    prompt,
                    base_max_tokens,
                    model_name=model_name,
                    openrouter_user=openrouter_user,
                    openrouter_metadata=openrouter_metadata,
                )

            if not response or not response.strip():
                response = "ขออภัยค่ะ ฉันไม่สามารถสร้างคำตอบได้ในขณะนี้ กรุณาลองใหม่อีกครั้งค่ะ"
            response = self._sanitize_llm_text(response)
            response = self._normalize_thai_assistant_persona(response)
            if not response:
                response = "ขออภัยค่ะ น้องติวยังไม่สามารถสร้างคำตอบได้ในขณะนี้ กรุณาลองใหม่อีกครั้งนะคะ"

            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

            # Create response
            message_id = str(uuid.uuid4())
            chat_response_payload = {
                "message_id": message_id,
                "content": response,
                "timestamp": end_time,
                "confidence": self._calculate_response_confidence(
                    response, user_message
                ),
                "course_context": effective_course_context.get("name")
                if effective_course_context
                else None,
                "conversation_id": conversation_id,
                "processing_time_ms": processing_time_ms,
            }
            chat_response = ChatResponse(
                **chat_response_payload,
                **_energy_fields(energy_status),
            )

            # Store conversation history
            self._store_conversation_message(
                conversation_id=conversation_id,
                user_id=user_id,
                course_id=course_id,
                user_message=(
                    f"{user_message}\n[แนบรูปภาพ]" if image_bytes else user_message
                ),
                ai_response=response,
                metadata={
                    "used_question_context": bool(effective_question_context),
                    "used_course_context": bool(effective_course_context),
                    "used_system_context": bool(effective_system_context),
                    "question_context_reason": routed_context.get(
                        "classifier_reason", ""
                    ),
                    "context_route": routed_context.get("context_route", ""),
                    "chat_mode": normalized_mode,
                },
            )
            try:
                usage_events = [
                    event for event in self._usage_events if isinstance(event, dict)
                ]
                if usage_events:
                    input_tokens = sum(
                        max(0, int(event.get("input_tokens") or 0))
                        for event in usage_events
                    )
                    output_tokens = sum(
                        max(0, int(event.get("output_tokens") or 0))
                        for event in usage_events
                    )
                    total_tokens = sum(
                        max(0, int(event.get("total_tokens") or 0))
                        for event in usage_events
                    )
                    llm_cost_usd = sum(
                        max(0.0, float(event.get("llm_cost_usd") or 0.0))
                        for event in usage_events
                    )
                    model_name = (
                        str(usage_events[-1].get("model") or "").strip()
                        or str(model_name or "").strip()
                        or self.settings.openrouter_model
                    )
                    usage_date = datetime.utcnow().date().isoformat()
                    await db_service.record_student_token_usage(
                        user_id=user_id,
                        usage_date=usage_date,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        model=model_name,
                        llm_cost_usd=llm_cost_usd,
                    )
                    energy_status = self._apply_usage_to_energy_status(
                        energy_status,
                        llm_cost_usd=llm_cost_usd,
                    )
            except Exception as usage_error:
                app_logger.warning(
                    f"Failed to persist token usage for {user_id}: {usage_error}"
                )
            if isinstance(energy_status, dict):
                chat_response = ChatResponse(
                    **chat_response_payload,
                    **_energy_fields(energy_status),
                )
            app_logger.info(f"Chat response generated successfully for user {user_id}")
            return chat_response

        except Exception as e:
            app_logger.error(f"Chat response generation failed for user {user_id}: {e}")

            # Calculate processing time even for errors
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

            # Return error response
            return ChatResponse(
                message_id=str(uuid.uuid4()),
                content="ขออภัยค่ะ น้องติวเจอข้อผิดพลาดในระบบ กรุณาลองใหม่อีกครั้งนะคะ",
                timestamp=end_time,
                confidence=0.0,
                course_context=None,
                conversation_id=conversation_id or str(uuid.uuid4()),
                processing_time_ms=processing_time_ms,
            )

    def normalize_chat_mode(self, chat_mode: Optional[str]) -> str:
        mode = str(chat_mode or "").strip().lower()
        if mode in self._allowed_chat_modes:
            return mode
        return self._default_chat_mode

    @staticmethod
    def _learning_advisor_needs_learning_context(user_message: str) -> bool:
        """Load the expensive learning overview only for learning-data questions."""
        text = re.sub(r"\s+", " ", str(user_message or "").strip().lower())
        learning_keywords = (
            "ผลการเรียน",
            "คะแนน",
            "ความคืบหน้า",
            "คอร์ส",
            "บทเรียน",
            "แบบฝึก",
            "ข้อสอบ",
            "เรียนไปถึง",
            "ควรโฟกัส",
            "ควรฝึก",
            "วิชา",
            "หัวข้อ",
            "progress",
            "score",
            "course",
            "lesson",
            "quiz",
        )
        return any(keyword in text for keyword in learning_keywords)

    def _apply_usage_to_energy_status(
        self,
        energy_status: Optional[Dict[str, Any]],
        *,
        llm_cost_usd: float,
    ) -> Optional[Dict[str, Any]]:
        """Update response energy fields without a second database round trip."""
        if not isinstance(energy_status, dict):
            return energy_status
        updated = dict(energy_status)
        cost_thb = max(0.0, float(llm_cost_usd or 0.0)) * float(
            self.settings.openrouter_cost_usd_to_thb
        )
        used_thb = max(0.0, float(updated.get("used_thb") or 0.0)) + cost_thb
        daily_limit = max(0.0, float(updated.get("daily_limit_thb") or 0.0))
        remaining_thb = max(0.0, daily_limit - used_thb)
        updated["used_thb"] = used_thb
        updated["remaining_thb"] = remaining_thb
        updated["remaining_percent"] = (
            max(0.0, min(100.0, (remaining_thb / daily_limit) * 100.0))
            if daily_limit > 0
            else 0.0
        )
        updated["is_exhausted"] = remaining_thb <= 0.000001
        return updated

    def _resolve_chat_model(self, chat_mode: str, has_image: bool = False) -> str:
        """
        Resolve model by mode.
        - learning_advisor without image: use OPENROUTER_CHAT_MODEL when configured
        - otherwise: use OPENROUTER_MODEL
        """
        if chat_mode == "learning_advisor" and not has_image:
            preferred = str(self.settings.openrouter_chat_model or "").strip()
            if preferred:
                return preferred
        return self.settings.openrouter_model

    def _create_chat_prompt(
        self,
        user_message: str,
        course_context: Optional[Dict[str, Any]],
        system_context: Optional[Dict[str, Any]],
        conversation_history: List[ChatMessage],
        conversation_summary: str,
        user_id: str,
        question_context: Optional[Any] = None,
        has_image: bool = False,
        chat_mode: str = "study_solver",
        learning_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create chat prompt by mode and keep Thai responses."""
        history_turn_limit = 2
        history_msg_char_limit = 160

        def _compact_history_text(value: Any) -> str:
            text = re.sub(r"\s+", " ", str(value or "").strip())
            if len(text) <= history_msg_char_limit:
                return text
            return f"{text[:history_msg_char_limit].rstrip()}..."

        if chat_mode == "learning_advisor":
            system_prompt = (
                "คุณคือน้องติว เด็กผู้หญิงผู้ช่วยตอบคำถามผลการเรียนจากข้อมูลในระบบเท่านั้น\n"
                "บุคลิกของผู้ช่วย: สดใส น่ารัก เป็นกันเองแบบพอดี ใช้สรรพนามแทนตัวว่า 'น้องติว' และลงท้ายด้วย 'ค่ะ' หรือ 'นะคะ' อย่างสม่ำเสมอ ห้ามใช้ 'ผม', 'ครับ', 'ฉัน' หรือ 'ดิฉัน'\n"
                "ตอบภาษาไทยสุภาพ กระชับ ตรงคำถาม ไม่เกิน 4 บรรทัด\n"
                "ใช้อีโมจิได้เล็กน้อยเมื่อเหมาะสม แต่ห้ามเยอะจนรบกวนคำตอบ\n"
                "ใช้ข้อความธรรมดา ห้าม markdown/bullet/heading\n"
                "ให้ยึดข้อความผู้ใช้ปัจจุบันเป็นหลัก ใช้บทสนทนาก่อนหน้าเพื่อช่วยตีความเท่านั้น ห้ามตอบแทนข้อความเก่าหรือเปลี่ยนหัวข้อเอง\n"
                "ห้ามตีความข้อความกำกวมเป็นเรื่องเชิงปรัชญา เรื่องชีวิต หรือหัวข้อใหม่ที่ผู้ใช้ไม่ได้ถาม\n"
                "ถ้าข้อความปัจจุบันไม่ชัดว่าอยากรู้หรือให้ช่วยอะไร ให้ถามกลับเพียง 1 คำถามสั้นๆ แทนการเดาความหมาย\n"
                "ห้ามแต่งข้อมูล หากข้อมูลไม่พอให้บอกตรงๆ และถามเพิ่มได้ 1 คำถามสั้นๆ"
            )

            if learning_context:
                overview_lines = []
                for key, value in learning_context.items():
                    if key == "course_items":
                        continue
                    if value is None or value == "":
                        continue
                    overview_lines.append(f"- {key}: {value}")
                if overview_lines:
                    system_prompt += "\n\nภาพรวมผลการเรียนของผู้ใช้:\n" + "\n".join(
                        overview_lines
                    )
                course_items = (
                    learning_context.get("course_items")
                    if isinstance(learning_context, dict)
                    else []
                )
                if isinstance(course_items, list) and course_items:
                    compact_rows = [
                        f"- {item}" for item in course_items[:5] if str(item).strip()
                    ]
                    if compact_rows:
                        system_prompt += "\nคอร์สที่ลงทะเบียน (ย่อ):\n" + "\n".join(
                            compact_rows
                        )

            if conversation_summary:
                system_prompt += (
                    "\n\nความจำบทสนทนาก่อนหน้า (สรุป):\n" f"{conversation_summary}\n"
                )

            if conversation_history:
                system_prompt += "\n\nบริบทการสนทนาก่อนหน้า (ย่อ):\n"
                for msg in conversation_history[-history_turn_limit:]:
                    role = "ผู้ใช้" if msg.type == "user" else "AI"
                    system_prompt += f"{role}: {_compact_history_text(msg.content)}\n"

            return (
                f"{system_prompt}\n\n"
                f"ข้อความผู้ใช้ปัจจุบัน:\n{user_message}\n\n"
                "ตอบสั้น กระชับ ตรงคำถาม เป็นข้อความย่อหน้าเดียวหรือหลายบรรทัดสั้นๆ โดยไม่จัดรูปแบบพิเศษ"
            )

        # System instructions for study_solver: concise but readable responses.
        system_prompt = (
            "คุณคือน้องติว เด็กผู้หญิงผู้ช่วยติวสำหรับนักเรียน ให้ตอบภาษาไทยสุภาพ ชัดเจน อ่านง่าย และเป็นธรรมชาติ\n"
            "บุคลิกของผู้ช่วย: สดใส น่ารัก ใจดี ชวนคิดแบบเพื่อนรุ่นน้องที่ตั้งใจช่วยติว ใช้สรรพนามแทนตัวว่า 'น้องติว' และลงท้ายด้วย 'ค่ะ' หรือ 'นะคะ' อย่างสม่ำเสมอ ห้ามใช้ 'ผม', 'ครับ', 'ฉัน' หรือ 'ดิฉัน'\n"
            "ใช้อีโมจิได้เล็กน้อยเมื่อช่วยให้กำลังใจหรือทำให้ข้อความเป็นมิตร แต่ห้ามเยอะจนรบกวนการเรียน\n"
            "ถ้าคำตอบยาวเกิน 2 ประโยค ให้จัดเป็นบรรทัดสั้นๆ หรือ bullet สั้นๆ ได้\n"
            "เวลาตรวจโจทย์หรือวิธีทำ ให้เน้นชี้แนวคิดและจุดที่ควรแก้แบบเข้าใจง่าย\n"
            "หลีกเลี่ยงสำนวนแข็งหรือแปลก เช่น 'วิธีที่ของนักเรียนเป็นการ...'\n"
            "ถ้ามีบริบทโจทย์หรือเหตุผลเฉลย ให้ยึดข้อมูลนั้นเป็นหลัก ห้ามเดา\n"
            "ถ้าข้อมูลไม่พอให้บอกตรงๆ และตอบยาวประมาณ 3-7 บรรทัดตามความจำเป็น\n"
            "หากผู้ใช้ระบุจุดที่ไม่เข้าใจ ให้ตอบจุดนั้นก่อน"
        )

        allow_direct_answer = self._can_reveal_direct_answer(question_context)
        reveal_solution_after_method = isinstance(question_context, dict) and str(
            question_context.get("reveal_solution_after_method") or ""
        ).strip().lower() in {"true", "1", "yes", "y", "ok", "allow", "allowed"}
        if question_context and not allow_direct_answer:
            system_prompt += (
                "\n\nข้อกำหนดสำคัญสำหรับโจทย์ปัจจุบัน:\n"
                "ผู้ใช้ยังไม่ได้กดส่งคำตอบของข้อนี้ ห้ามเฉลยตรงๆ ห้ามบอกตัวเลือกที่ถูก "
                "ห้ามบอกคำตอบสุดท้ายเป็นตัวเลข/ข้อความตรงๆ\n"
                "ให้ตอบแบบโค้ชเพื่อกระตุ้นการคิด โดยอธิบายหลักคิดทีละขั้นและถามนำสั้นๆ 1 คำถามท้ายข้อความ\n"
                "รูปแบบที่ต้องการ:\n"
                "- เปิดด้วยบอกสั้นๆ ว่าตอนนี้ติดตรงไหนหรือควรระวังอะไร\n"
                "- ตามด้วย bullet 2-4 ข้อที่ช่วยไล่คิด\n"
                "- ปิดท้ายด้วยคำถามชวนคิด 1 ข้อ โดยยังไม่เฉลย"
            )
        elif question_context and allow_direct_answer:
            system_prompt += (
                "\n\nเมื่ออนุญาตให้เฉลยได้แล้ว ให้ตอบเป็น 3 ส่วนตามลำดับ:\n"
                "1. สิ่งที่ทำถูกหรือเข้าใจถูก\n"
                "2. จุดที่ควรแก้หรือเหตุผลสำคัญ\n"
                "3. คำตอบที่ถูกต้อง พร้อมเหตุผลสั้นๆ\n"
                "ถ้ามีหลายจุดให้ใช้ bullet สั้นๆ ได้ และห้ามตอบสั้นจนเหลือแค่เฉลยอย่างเดียว"
            )

        # Add current question context if provided
        if question_context:
            context_block = self._build_question_context_block(
                question_context,
                allow_direct_answer=allow_direct_answer,
            )
            if context_block:
                system_prompt += (
                    "\n\nบริบทคำถาม (ใช้ประกอบการอธิบาย):\n" f"{context_block}\n"
                )

        # Add compact long-term summary memory first, then recent turns.
        if conversation_summary:
            system_prompt += (
                "\n\nความจำบทสนทนาก่อนหน้า (สรุป):\n" f"{conversation_summary}\n"
            )

        # If image is provided, force answer-check workflow against question and solution context
        if has_image:
            if question_context and not allow_direct_answer:
                system_prompt += (
                    "\n\nเมื่อมีรูปวิธีทำจากผู้ใช้ และยังไม่ส่งคำตอบ:\n"
                    "ให้โฟกัสตรวจขั้นตอนคิดและชี้จุดที่ควรแก้ โดยห้ามเฉลยคำตอบสุดท้าย\n"
                    "ให้บอกด้วยว่าส่วนไหนในรูปที่ทำถูก และส่วนไหนควรลองใหม่"
                )
            else:
                system_prompt += (
                    "\n\nเมื่อมีรูปวิธีทำจากผู้ใช้ ให้ทำหน้าที่เป็นผู้ตรวจคำตอบอย่างเข้มงวด:\n"
                    "- ต้องยึดโจทย์/บริบทคำถามเป็นหลัก ไม่อธิบายรูปทั่วไป\n"
                    "- ถ้าอ่านรูปไม่ชัด ให้บอกจุดที่ไม่ชัดและบอกสิ่งที่ต้องส่งเพิ่ม\n"
                    "- ห้ามเดา และห้ามตัดสินโดยไม่เทียบกับโจทย์/เฉลย\n"
                    "- เวลาตอบให้สรุปเป็นส่วนๆ อ่านง่าย ไม่ใช่ย่อหน้ายาวก้อนเดียว"
                )
                if allow_direct_answer and reveal_solution_after_method:
                    system_prompt += (
                        "\n- ผู้ใช้ตอบผิดและส่งวิธีทำมาแล้ว: ต้องระบุเฉลยที่ถูกต้องให้ชัดเจน "
                        "พร้อมอธิบายเหตุผล/หลักคิดแบบเข้าใจง่าย"
                    )

        # Add recent conversation history (Thai labels)
        if conversation_history:
            system_prompt += "\n\nบริบทการสนทนาก่อนหน้า (ย่อ):\n"
            for msg in conversation_history[-history_turn_limit:]:
                role = "ผู้ใช้" if msg.type == "user" else "AI"
                system_prompt += f"{role}: {_compact_history_text(msg.content)}\n"

        # Compose final prompt with current user message
        full_prompt = (
            f"{system_prompt}\n\nคำถาม/คำขอปัจจุบันจากผู้ใช้:\n{user_message}\n\n"
            "โปรดตอบให้อ่านง่าย ใช้ข้อความสั้น กระชับ และใช้ bullet/ขึ้นบรรทัดใหม่ได้เมื่อช่วยให้ชัดเจน"
        )

        return full_prompt

    def _parse_question_context(self, question_context: Optional[Any]) -> Optional[Any]:
        """Parse question context that may arrive as plain text or JSON string."""
        if question_context is None:
            return None
        if isinstance(question_context, dict):
            return question_context
        if isinstance(question_context, str):
            raw = question_context.strip()
            if not raw:
                return None
            if raw.startswith("{") and raw.endswith("}"):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            return raw
        return str(question_context)

    def _build_question_context_block(
        self, question_context: Any, allow_direct_answer: bool = True
    ) -> str:
        """Build a readable question context block for the LLM prompt."""
        if isinstance(question_context, str):
            return f"คำถาม: {question_context.strip()}"
        if not isinstance(question_context, dict):
            return f"คำถาม: {str(question_context)}"

        lines: List[str] = []
        shared_context = (
            question_context.get("question_context_text")
            or question_context.get("context")
            or question_context.get("question_context")
            or question_context.get("passage")
            or question_context.get("shared_context")
            or question_context.get("sharedContext")
            or question_context.get("reading_passage")
            or question_context.get("readingPassage")
            or question_context.get("instructions")
            or question_context.get("instruction")
            or question_context.get("stimulus")
            or question_context.get("common_stem")
            or question_context.get("commonStem")
        )
        if shared_context:
            lines.append(f"บริบทประกอบโจทย์:\n{str(shared_context).strip()}")

        question_text = question_context.get("question_text") or question_context.get(
            "question"
        )
        if question_text:
            lines.append(f"โจทย์: {question_text}")

        options = question_context.get("options")
        if isinstance(options, list) and options:
            option_lines = []
            for idx, opt in enumerate(options):
                label = chr(65 + idx) if idx < 26 else str(idx + 1)
                option_lines.append(f"{label}) {opt}")
            lines.append("ตัวเลือก: " + " | ".join(option_lines))

        correct_answer_index = question_context.get("correct_answer_index")
        correct_answer_text = question_context.get("correct_answer_text")
        if allow_direct_answer and (
            correct_answer_index is not None or correct_answer_text
        ):
            lines.append(
                f"เฉลยที่คาดหวัง: index={correct_answer_index}, text={correct_answer_text}"
            )

        explanation = question_context.get("explanation")
        if allow_direct_answer and explanation:
            lines.append(f"ข้อมูลอธิบายเฉลย (ให้ยึดเป็นหลัก): {explanation}")

        user_answer_index = question_context.get("user_answer_index")
        user_answer_text = question_context.get("user_answer_text")
        if user_answer_index is not None or user_answer_text:
            lines.append(
                f"คำตอบที่ผู้ใช้เลือก: index={user_answer_index}, text={user_answer_text}"
            )

        if not lines:
            return ""
        return "\n".join(lines)

    def _can_reveal_direct_answer(self, question_context: Optional[Any]) -> bool:
        """Whether the assistant is allowed to reveal direct answers for the current question."""
        if not isinstance(question_context, dict):
            return False

        def as_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            text = str(value or "").strip().lower()
            return text in {"true", "1", "yes", "y", "ok", "allow", "allowed"}

        if as_bool(question_context.get("allow_direct_answer")):
            return True
        if as_bool(question_context.get("quiz_submitted")):
            return True
        if as_bool(question_context.get("answer_revealed_for_question")):
            return True
        if as_bool(question_context.get("is_answer_revealed")):
            return True
        return False

    def _sanitize_llm_text(self, raw: Optional[str]) -> str:
        """Normalize escaped artifacts from model output before returning to clients."""
        text = str(raw or "")
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        for _ in range(3):
            before = text
            text = (
                text.replace("\\\\r\\\\n", "\n")
                .replace("\\\\n", "\n")
                .replace("\\\\r", "\n")
                .replace("\\\\t", "  ")
                .replace('\\"', '"')
                .replace("\\'", "'")
                .replace("\\/", "/")
            )
            # Remove lines that contain only a stray backslash.
            text = re.sub(r"(?m)^\s*\\\s*$", "", text)
            if text == before:
                break

        text = (
            text.replace("\\n", "\n")
            .replace("\\r", "\n")
            .replace("\\t", "  ")
            .replace('\\"', '"')
            .replace("\\'", "'")
            .replace("\\/", "/")
        )
        # Drop stray backslashes that are not valid LaTeX-like commands.
        text = re.sub(r"\\(?=[\u0E00-\u0E7F])", "", text)
        text = re.sub(r"\\(?![a-zA-Z\\$()[\]{}])", "", text)
        text = re.sub(r"(?m)^\s*\\\s*$", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _normalize_thai_assistant_persona(self, raw: Optional[str]) -> str:
        """Keep Thai assistant persona consistent when the model drifts."""
        text = str(raw or "")
        if not text:
            return ""

        replacements = (
            ("ครับผม", "ค่ะ"),
            ("นะครับ", "นะคะ"),
            ("ครับ", "ค่ะ"),
            ("คับ", "ค่ะ"),
            ("ดิฉัน", "น้องติว"),
            ("ฉัน", "น้องติว"),
            ("ผม", "น้องติว"),
        )
        for source, target in replacements:
            text = text.replace(source, target)
        return text.strip()

    def _extract_openrouter_response_cost_usd(self, completion: Any) -> float:
        """Extract real USD cost reported by OpenRouter for one completion."""
        candidates: List[Any] = []

        if completion is None:
            return 0.0

        hidden = getattr(completion, "_hidden_params", None)
        if isinstance(hidden, dict):
            candidates.extend(
                [
                    hidden.get("response_cost"),
                    hidden.get("response_cost_usd"),
                    hidden.get("cost"),
                    hidden.get("spend"),
                ]
            )

        for attr in ("response_cost", "response_cost_usd", "cost", "spend"):
            candidates.append(getattr(completion, attr, None))

        if hasattr(completion, "get"):
            try:
                candidates.extend(
                    [
                        completion.get("response_cost"),
                        completion.get("response_cost_usd"),
                        completion.get("cost"),
                        completion.get("spend"),
                    ]
                )
            except Exception:
                pass

        if hasattr(completion, "model_dump"):
            try:
                dumped = completion.model_dump()
                if isinstance(dumped, dict):
                    candidates.extend(
                        [
                            dumped.get("response_cost"),
                            dumped.get("response_cost_usd"),
                            dumped.get("cost"),
                            dumped.get("spend"),
                        ]
                    )
                    usage = dumped.get("usage")
                    if isinstance(usage, dict):
                        candidates.extend(
                            [
                                usage.get("response_cost"),
                                usage.get("response_cost_usd"),
                                usage.get("cost"),
                                usage.get("spend"),
                            ]
                        )
            except Exception:
                pass

        for value in candidates:
            try:
                amount = float(value)
            except Exception:
                continue
            if amount >= 0:
                return amount

        return 0.0

    async def _get_openrouter_user_context(
        self, user_id: str
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Prepare per-user context fields without blocking chat on a profile lookup."""
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return None, {}

        return normalized_user_id, {
            "app_user_id": normalized_user_id,
            "source": "tanaijarn-backend",
        }

    async def _call_gemini_chat(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        openrouter_user: Optional[str] = None,
        openrouter_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Call OpenRouter Chat Completions API for chat response."""
        try:
            resolved_model = (
                str(model_name or self.settings.openrouter_model).strip()
                or self.settings.openrouter_model
            )

            extra_headers: Dict[str, str] = {}
            if self.settings.openrouter_site_url:
                extra_headers["HTTP-Referer"] = self.settings.openrouter_site_url
            if self.settings.openrouter_site_name:
                extra_headers["X-OpenRouter-Title"] = self.settings.openrouter_site_name
            extra_params = build_llm_extra_params(
                resolved_model,
                reasoning_effort=self.settings.openrouter_reasoning_effort,
            )

            request_payload: Dict[str, Any] = {
                "model": resolved_model,
                "messages": [{"role": "user", "content": prompt}],
                "extra_headers": extra_headers or None,
            }
            if openrouter_user:
                request_payload["user"] = openrouter_user
            if isinstance(openrouter_metadata, dict) and openrouter_metadata:
                request_payload["metadata"] = openrouter_metadata
            if response_format and supports_response_format(resolved_model):
                request_payload["response_format"] = response_format
            if max_tokens is not None:
                request_payload["max_tokens"] = max_tokens
                request_payload["max_completion_tokens"] = max_tokens

            completion = await self._aclient.chat.completions.create(
                **request_payload,
                **extra_params,
            )

            content = (
                completion.choices[0].message.content
                if completion and completion.choices
                else None
            )
            if not content or not content.strip():
                raise LLMProcessingError("Empty response from OpenRouter API")

            usage = None
            try:
                raw_usage = getattr(completion, "usage", None)
                if raw_usage:
                    prompt_tokens = getattr(raw_usage, "prompt_tokens", None)
                    completion_tokens = getattr(raw_usage, "completion_tokens", None)
                    total_tokens = getattr(raw_usage, "total_tokens", None)
                    if prompt_tokens is None and hasattr(raw_usage, "get"):
                        prompt_tokens = raw_usage.get("prompt_tokens")
                        completion_tokens = raw_usage.get("completion_tokens")
                        total_tokens = raw_usage.get("total_tokens")
                    usage = {
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": total_tokens,
                    }
            except Exception as exc:
                app_logger.warning(f"OpenRouter usage parse failed: {exc}")

            if usage:
                response_cost_usd = self._extract_openrouter_response_cost_usd(
                    completion
                )
                self._usage_events.append(
                    {
                        "model": resolved_model,
                        "input_tokens": int(usage.get("input") or 0),
                        "output_tokens": int(usage.get("output") or 0),
                        "total_tokens": int(usage.get("total") or 0),
                        "llm_cost_usd": response_cost_usd,
                    }
                )
            return content.strip()

        except Exception as e:
            msg = str(e)
            if "api key" in msg.lower() or "401" in msg:
                raise LLMProcessingError(f"OpenRouter API key error: {e}")
            elif "quota" in msg.lower() or "429" in msg:
                raise LLMProcessingError(f"OpenRouter API quota/rate limit: {e}")
            else:
                raise LLMProcessingError(f"OpenRouter API call failed: {e}")

    async def _call_gemini_chat_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        image_mime: str,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        openrouter_user: Optional[str] = None,
        openrouter_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Call OpenRouter Chat Completions API for chat response with an image."""
        try:
            resolved_model = (
                str(model_name or self.settings.openrouter_model).strip()
                or self.settings.openrouter_model
            )

            extra_headers: Dict[str, str] = {}
            if self.settings.openrouter_site_url:
                extra_headers["HTTP-Referer"] = self.settings.openrouter_site_url
            if self.settings.openrouter_site_name:
                extra_headers["X-OpenRouter-Title"] = self.settings.openrouter_site_name
            extra_params = build_llm_extra_params(
                resolved_model,
                reasoning_effort=self.settings.openrouter_reasoning_effort,
            )

            data_url = f"data:{image_mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
            image_url_payload: Dict[str, Any] = {"url": data_url}
            # OpenRouter Gemini docs: use OpenAI-style `detail` to map to Gemini `media_resolution`.
            # Apply only to Gemini 3+ models to align with provider support.
            if resolved_model.startswith("gemini/") and "gemini-3" in resolved_model:
                image_url_payload["detail"] = "low"
            request_payload: Dict[str, Any] = {
                "model": resolved_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": image_url_payload},
                        ],
                    }
                ],
                "extra_headers": extra_headers or None,
            }
            if openrouter_user:
                request_payload["user"] = openrouter_user
            if isinstance(openrouter_metadata, dict) and openrouter_metadata:
                request_payload["metadata"] = openrouter_metadata
            if max_tokens is not None:
                request_payload["max_tokens"] = max_tokens
                request_payload["max_completion_tokens"] = max_tokens

            completion = await self._aclient.chat.completions.create(
                **request_payload,
                **extra_params,
            )

            content = (
                completion.choices[0].message.content
                if completion and completion.choices
                else None
            )
            if not content or not content.strip():
                raise LLMProcessingError("Empty response from OpenRouter API")

            usage = None
            try:
                raw_usage = getattr(completion, "usage", None)
                if raw_usage:
                    prompt_tokens = getattr(raw_usage, "prompt_tokens", None)
                    completion_tokens = getattr(raw_usage, "completion_tokens", None)
                    total_tokens = getattr(raw_usage, "total_tokens", None)
                    if prompt_tokens is None and hasattr(raw_usage, "get"):
                        prompt_tokens = raw_usage.get("prompt_tokens")
                        completion_tokens = raw_usage.get("completion_tokens")
                        total_tokens = raw_usage.get("total_tokens")
                    usage = {
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": total_tokens,
                    }
            except Exception as exc:
                app_logger.warning(f"OpenRouter usage parse failed: {exc}")

            if usage:
                response_cost_usd = self._extract_openrouter_response_cost_usd(
                    completion
                )
                self._usage_events.append(
                    {
                        "model": resolved_model,
                        "input_tokens": int(usage.get("input") or 0),
                        "output_tokens": int(usage.get("output") or 0),
                        "total_tokens": int(usage.get("total") or 0),
                        "llm_cost_usd": response_cost_usd,
                    }
                )
            return content.strip()

        except Exception as e:
            msg = str(e)
            if "api key" in msg.lower() or "401" in msg:
                raise LLMProcessingError(f"OpenRouter API key error: {e}")
            elif "quota" in msg.lower() or "429" in msg:
                raise LLMProcessingError(f"OpenRouter API quota/rate limit: {e}")
            else:
                raise LLMProcessingError(f"OpenRouter API call failed: {e}")

    def _get_course_context(self, course_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get course context information."""
        if not course_id:
            return None
        return self.course_contexts.get(course_id)

    async def _get_course_context_for_user(
        self,
        user_id: str,
        course_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Return course context only when the course belongs to user's enrolled courses."""
        if not course_id:
            return None
        try:
            db_service = get_db_service()
            enrolled_courses = await db_service.get_enrolled_courses_for_user(user_id)
            matched = next(
                (
                    item
                    for item in enrolled_courses
                    if str(item.get("course_id") or item.get("id") or "")
                    == str(course_id)
                ),
                None,
            )
            if not matched:
                return None

            topics = (
                matched.get("topics") if isinstance(matched.get("topics"), list) else []
            )
            tags = matched.get("tags") if isinstance(matched.get("tags"), list) else []
            merged_topics = [str(v).strip() for v in (topics + tags) if str(v).strip()]
            learning_metrics = {
                "ความคืบหน้า": matched.get("progress"),
                "คะแนนเฉลี่ย": matched.get("averageScore") or matched.get("avg_score"),
                "บทเรียนที่ทำแล้ว": matched.get("completedLessons")
                or matched.get("completedLessonsCount"),
                "จำนวนบทเรียนทั้งหมด": matched.get("totalLessons")
                or matched.get("lessons_count")
                or matched.get("lesson_count"),
                "แบบฝึกที่ทำแล้ว": matched.get("completedQuizzes")
                or matched.get("completed_quizzes"),
                "จำนวนแบบฝึกทั้งหมด": matched.get("totalQuizzes")
                or matched.get("total_quizzes")
                or matched.get("quiz_count"),
                "จำนวนข้อที่ทำแล้ว": matched.get("completedQuestions")
                or matched.get("completed_questions"),
                "จำนวนข้อทั้งหมด": matched.get("totalQuestions")
                or matched.get("total_questions"),
            }

            return {
                "name": matched.get("name") or matched.get("title") or "คอร์สเรียน",
                "description": matched.get("description")
                or matched.get("detail")
                or "",
                "topics": merged_topics,
                "learning_metrics": learning_metrics,
                "language": "th",
            }
        except Exception as exc:
            app_logger.warning(
                f"Failed to resolve enrolled course context for user {user_id}: {exc}"
            )
            return self._get_course_context(course_id)

    async def _get_learning_overview_for_user(self, user_id: str) -> Dict[str, Any]:
        """
        Build compact learning summary from all enrolled courses.
        Used by learning_advisor mode when user asks without a specific course context.
        """
        try:
            db_service = get_db_service()
            enrolled_courses = await db_service.get_enrolled_courses_for_user(user_id)
        except Exception as exc:
            app_logger.warning(
                f"Failed to build learning overview for user {user_id}: {exc}"
            )
            return {}

        if not enrolled_courses:
            return {"จำนวนคอร์สที่ลงทะเบียน": 0}

        def _num(v: Any) -> Optional[float]:
            try:
                n = float(v)
                return n if n == n else None
            except Exception:
                return None

        progress_values = []
        score_values = []
        total_quizzes = 0.0
        completed_quizzes = 0.0
        course_items: List[str] = []

        for row in enrolled_courses:
            name = str(
                row.get("name") or row.get("title") or row.get("course_name") or "คอร์ส"
            ).strip()
            progress = _num(row.get("progress"))
            score = _num(row.get("averageScore") or row.get("avg_score"))
            done_q = (
                _num(row.get("completedQuizzes") or row.get("completed_quizzes")) or 0.0
            )
            all_q = (
                _num(
                    row.get("totalQuizzes")
                    or row.get("total_quizzes")
                    or row.get("quiz_count")
                )
                or 0.0
            )
            total_quizzes += all_q
            completed_quizzes += done_q
            if progress is not None:
                progress_values.append(max(0.0, min(100.0, progress)))
            if score is not None:
                score_values.append(max(0.0, min(100.0, score)))
            course_items.append(
                f"{name}: progress {int(round(progress)) if progress is not None else '-'}%, score {int(round(score)) if score is not None else '-'}%"
            )

        avg_progress = (
            int(round(sum(progress_values) / len(progress_values)))
            if progress_values
            else None
        )
        avg_score = (
            int(round(sum(score_values) / len(score_values))) if score_values else None
        )

        return {
            "จำนวนคอร์สที่ลงทะเบียน": len(enrolled_courses),
            "ความคืบหน้าเฉลี่ย": f"{avg_progress}%"
            if avg_progress is not None
            else None,
            "คะแนนเฉลี่ยรวม": f"{avg_score}%" if avg_score is not None else None,
            "แบบฝึกที่ทำแล้วรวม": int(round(completed_quizzes)),
            "จำนวนแบบฝึกรวม": int(round(total_quizzes)),
            "course_items": course_items,
        }

    def _get_conversation_history(self, conversation_id: str) -> List[ChatMessage]:
        """Get conversation history."""
        memory = self._get_or_create_memory(conversation_id)
        history: List[ChatMessage] = []

        for msg in memory.chat_memory.messages:
            msg_type = getattr(msg, "type", None)
            role = "user" if msg_type == "human" else "ai"
            ts = None
            if hasattr(msg, "additional_kwargs"):
                ts = msg.additional_kwargs.get("timestamp")
            if not isinstance(ts, datetime):
                ts = datetime.utcnow()
            history.append(
                ChatMessage(
                    id=str(uuid.uuid4()),
                    type=role,
                    content=msg.content,
                    timestamp=ts,
                    metadata=msg.additional_kwargs
                    if hasattr(msg, "additional_kwargs")
                    else None,
                )
            )

        return history

    def _store_conversation_message(
        self,
        conversation_id: str,
        user_id: str,
        course_id: Optional[str],
        user_message: str,
        ai_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store conversation message in memory."""
        memory = self._get_or_create_memory(conversation_id)
        timestamp = datetime.utcnow()
        base_meta = {"user_id": user_id, "course_id": course_id}
        if metadata:
            base_meta.update(metadata)

        memory.chat_memory.add_message(
            HumanMessage(
                content=user_message,
                additional_kwargs={"timestamp": timestamp, **base_meta},
            )
        )
        memory.chat_memory.add_message(
            AIMessage(
                content=ai_response,
                additional_kwargs={"timestamp": datetime.utcnow(), **base_meta},
            )
        )

        # Keep only recent messages; long-term continuity is maintained in summary memory.
        if len(memory.chat_memory.messages) > 40:
            memory.chat_memory.messages = memory.chat_memory.messages[-40:]

    def _get_or_create_memory(self, conversation_id: str) -> ConversationBufferMemory:
        memory = self._session_memory.get(conversation_id)
        if memory is None:
            memory = ConversationBufferMemory(return_messages=True)
            self._session_memory[conversation_id] = memory
        return memory

    @property
    def langgraph_available(self) -> bool:
        return _LANGGRAPH_AVAILABLE

    async def _route_question_context_with_langgraph(
        self,
        user_message: str,
        parsed_question_context: Optional[Any],
        has_image: bool = False,
        has_course_context: bool = False,
        chat_mode: str = "study_solver",
    ) -> Dict[str, Any]:
        """Use LangGraph routing to decide whether question context should be injected."""
        (
            aligned_question_context,
            alignment_reason,
        ) = await self._filter_incoherent_question_context(parsed_question_context)
        if alignment_reason.startswith("context_alignment:drop"):
            app_logger.info(
                f"Dropping mismatched question context before prompt injection ({alignment_reason})"
            )
        state: _ChatRouterState = {
            "user_message": user_message,
            "chat_mode": chat_mode,
            "has_image": bool(has_image),
            "parsed_question_context": aligned_question_context,
            "has_course_context": bool(has_course_context),
            "question_context_alignment_reason": alignment_reason,
        }
        if not _LANGGRAPH_AVAILABLE:
            should_include, reason, route = await self._decide_question_context_need(
                user_message=user_message,
                parsed_question_context=aligned_question_context,
                has_image=has_image,
                has_course_context=has_course_context,
                chat_mode=chat_mode,
            )
            classifier_reason = f"fallback:{reason}"
            if alignment_reason:
                classifier_reason = f"{alignment_reason}|{classifier_reason}"
            return {
                "should_include_question_context": should_include,
                "question_context_for_prompt": aligned_question_context
                if should_include
                else None,
                "should_include_course_context": route == "course_related"
                and has_course_context,
                "should_include_system_context": route == "system_related",
                "context_route": route,
                "classifier_reason": classifier_reason,
            }
        graph = self._get_or_create_chat_router_graph()
        result = await graph.ainvoke(state)
        classifier_reason = str(result.get("classifier_reason") or "")
        if alignment_reason:
            classifier_reason = (
                f"{alignment_reason}|{classifier_reason}"
                if classifier_reason
                else alignment_reason
            )
        return {
            "should_include_question_context": bool(
                result.get("should_include_question_context")
            ),
            "question_context_for_prompt": result.get("question_context_for_prompt"),
            "should_include_course_context": bool(
                result.get("should_include_course_context")
            ),
            "should_include_system_context": bool(
                result.get("should_include_system_context")
            ),
            "context_route": str(result.get("context_route") or ""),
            "classifier_reason": classifier_reason,
        }

    def _get_or_create_chat_router_graph(self):
        if self.__class__._chat_router_graph is not None:
            return self.__class__._chat_router_graph
        graph_builder = StateGraph(_ChatRouterState)
        graph_builder.add_node(
            "classify_context_need", self._langgraph_node_classify_context_need
        )
        graph_builder.add_node(
            "apply_context_selection", self._langgraph_node_apply_context_selection
        )
        graph_builder.set_entry_point("classify_context_need")
        graph_builder.add_edge("classify_context_need", "apply_context_selection")
        graph_builder.add_edge("apply_context_selection", END)
        self.__class__._chat_router_graph = graph_builder.compile()
        return self.__class__._chat_router_graph

    async def _langgraph_node_classify_context_need(
        self, state: _ChatRouterState
    ) -> Dict[str, Any]:
        user_message = str(state.get("user_message") or "")
        chat_mode = str(state.get("chat_mode") or self._default_chat_mode)
        parsed_question_context = state.get("parsed_question_context")
        has_image = bool(state.get("has_image"))
        has_course_context = bool(state.get("has_course_context"))

        should_include, reason, route = await self._decide_question_context_need(
            user_message=user_message,
            parsed_question_context=parsed_question_context,
            has_image=has_image,
            has_course_context=has_course_context,
            chat_mode=chat_mode,
        )

        return {
            "should_include_question_context": should_include,
            "should_include_course_context": route == "course_related"
            and has_course_context,
            "should_include_system_context": route == "system_related",
            "context_route": route,
            "classifier_reason": reason,
        }

    async def _langgraph_node_apply_context_selection(
        self, state: _ChatRouterState
    ) -> Dict[str, Any]:
        include_context = bool(state.get("should_include_question_context"))
        parsed_question_context = state.get("parsed_question_context")
        return {
            "question_context_for_prompt": parsed_question_context
            if include_context
            else None,
        }

    async def _filter_incoherent_question_context(
        self, parsed_question_context: Optional[Any]
    ) -> Tuple[Optional[Any], str]:
        """Use LLM to decide if shared context aligns with the question text."""
        if not isinstance(parsed_question_context, dict):
            return parsed_question_context, "context_alignment:skip_non_dict"

        question_text = str(
            parsed_question_context.get("question_text")
            or parsed_question_context.get("question")
            or ""
        ).strip()
        shared_context = str(
            parsed_question_context.get("question_context_text")
            or parsed_question_context.get("context")
            or parsed_question_context.get("question_context")
            or parsed_question_context.get("passage")
            or parsed_question_context.get("shared_context")
            or parsed_question_context.get("sharedContext")
            or parsed_question_context.get("reading_passage")
            or parsed_question_context.get("readingPassage")
            or parsed_question_context.get("instructions")
            or parsed_question_context.get("instruction")
            or parsed_question_context.get("stimulus")
            or parsed_question_context.get("common_stem")
            or parsed_question_context.get("commonStem")
            or ""
        ).strip()

        if not question_text or not shared_context:
            return parsed_question_context, "context_alignment:skip_missing_fields"

        question_tokens = set(re.findall(r"[a-zA-Z0-9ก-๙]+", question_text.lower()))
        context_tokens = set(re.findall(r"[a-zA-Z0-9ก-๙]+", shared_context.lower()))
        overlap = question_tokens & context_tokens
        overlap_ratio = len(overlap) / max(1, len(question_tokens))
        if len(overlap) >= 2 or overlap_ratio >= 0.2:
            return parsed_question_context, "context_alignment:token_overlap"
        compact_question = re.sub(r"\s+", "", question_text[:1200].lower())
        compact_context = re.sub(r"\s+", "", shared_context[:1800].lower())
        shared_substring_length = (
            SequenceMatcher(
                None,
                compact_question,
                compact_context,
                autojunk=False,
            )
            .find_longest_match()
            .size
        )
        if shared_substring_length >= 8:
            return parsed_question_context, "context_alignment:shared_substring"

        (
            is_aligned,
            reason,
            confidence,
        ) = await self._llm_question_context_alignment_classifier(
            question_text=question_text,
            shared_context=shared_context,
        )
        confidence_threshold = float(
            self.settings.chat_context_classifier_confidence_threshold
        )
        if confidence < confidence_threshold:
            return (
                parsed_question_context,
                (
                    "context_alignment:llm_low_confidence_keep:"
                    f"{reason}:confidence={confidence:.2f}:threshold={confidence_threshold:.2f}"
                ),
            )
        if is_aligned:
            return (
                parsed_question_context,
                f"context_alignment:llm_aligned:{reason}:confidence={confidence:.2f}",
            )

        sanitized_context = dict(parsed_question_context)
        for key in (
            "question_context_text",
            "context",
            "question_context",
            "passage",
            "shared_context",
            "sharedContext",
            "reading_passage",
            "readingPassage",
            "instructions",
            "instruction",
            "stimulus",
            "common_stem",
            "commonStem",
        ):
            if key in sanitized_context:
                sanitized_context.pop(key, None)
        return (
            sanitized_context,
            f"context_alignment:drop_llm_mismatch:{reason}:confidence={confidence:.2f}",
        )

    async def _llm_question_context_alignment_classifier(
        self,
        question_text: str,
        shared_context: str,
    ) -> Tuple[bool, str, float]:
        """Classify whether shared context belongs to the provided question."""
        prompt = (
            "คุณคือระบบตรวจสอบความสอดคล้องของโจทย์กับบริบทประกอบโจทย์\n"
            "ให้ตอบ JSON เพียงบรรทัดเดียวเท่านั้น (ห้าม markdown/code fence)\n"
            'รูปแบบ: {"aligned": true|false, "confidence": 0.0-1.0, "reason": "short"}\n'
            "เกณฑ์:\n"
            "- aligned=true เมื่อบริบทช่วยตอบโจทย์นี้ได้จริง หรือเป็นข้อความอ้างอิงเดียวกัน\n"
            "- aligned=false เมื่อบริบทคนละเรื่อง/คนละวิชา/ไม่เกี่ยวกับโจทย์\n"
            "- confidence เป็นตัวเลข 0 ถึง 1\n\n"
            f"โจทย์:\n{question_text[:1200]}\n\n"
            f"บริบทประกอบโจทย์:\n{shared_context[:1800]}"
        )
        try:
            raw = await self._call_gemini_chat(
                prompt,
                max_tokens=min(
                    self.settings.chat_context_classifier_max_tokens,
                    140,
                ),
                model_name=self._resolve_context_classifier_model(),
            )
            parsed = self._safe_extract_json_object(raw)
            if isinstance(parsed, dict):
                aligned_raw = parsed.get("aligned")
                if aligned_raw is None:
                    aligned_raw = parsed.get("related")
                aligned = self._coerce_bool(aligned_raw)
                confidence = self._coerce_confidence(parsed.get("confidence"))
                reason = str(parsed.get("reason") or "classified")
                return aligned, reason, confidence
            return True, "invalid_alignment_payload", 0.0
        except Exception as exc:
            app_logger.warning(f"LLM context alignment classifier failed: {exc}")
        return True, "alignment_classifier_failed", 0.0

    def _resolve_context_classifier_model(self) -> str:
        configured_model = str(
            self.settings.chat_context_classifier_model or ""
        ).strip()
        if configured_model:
            return configured_model
        preferred_chat_model = str(self.settings.openrouter_chat_model or "").strip()
        if preferred_chat_model:
            return preferred_chat_model
        return self.settings.openrouter_model

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        text = str(value or "").strip().lower()
        return text in {
            "true",
            "1",
            "yes",
            "y",
            "related",
            "need",
            "required",
            "include",
            "ใช่",
            "ต้อง",
        }

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        if value is None:
            return 0.0
        text = str(value).strip()
        if not text:
            return 0.0
        if text.endswith("%"):
            text = text[:-1].strip()
        try:
            confidence = float(text)
        except (TypeError, ValueError):
            return 0.0
        if confidence > 1.0 and confidence <= 100.0:
            confidence = confidence / 100.0
        return max(0.0, min(1.0, confidence))

    async def _decide_question_context_need(
        self,
        user_message: str,
        parsed_question_context: Optional[Any],
        has_image: bool = False,
        has_course_context: bool = False,
        chat_mode: str = "study_solver",
    ) -> Tuple[bool, str, str]:
        should_include, reason, route = self._heuristic_question_context_decision(
            user_message=user_message,
            parsed_question_context=parsed_question_context,
            has_image=has_image,
            has_course_context=has_course_context,
            chat_mode=chat_mode,
        )
        if reason != "uncertain" or not parsed_question_context:
            return should_include, reason, route

        if not self.settings.chat_context_classifier_enabled:
            return (
                True,
                "heuristic_uncertain_include_context",
                ("course_related" if has_course_context else "general"),
            )

        (
            llm_include,
            llm_reason,
            llm_confidence,
        ) = await self._llm_question_context_classifier(
            user_message=user_message,
            parsed_question_context=parsed_question_context,
        )
        confidence_threshold = float(
            self.settings.chat_context_classifier_confidence_threshold
        )
        if llm_confidence < confidence_threshold:
            return (
                True,
                (
                    "llm_low_confidence_safe_default:"
                    f"{llm_reason}:confidence={llm_confidence:.2f}:threshold={confidence_threshold:.2f}"
                ),
                ("course_related" if has_course_context else "general"),
            )

        route = "course_related" if (llm_include and has_course_context) else "general"
        return (
            llm_include,
            (
                "llm_classifier:"
                f"{llm_reason}:decision={'include' if llm_include else 'skip'}:confidence={llm_confidence:.2f}"
            ),
            route,
        )

    def _heuristic_question_context_decision(
        self,
        user_message: str,
        parsed_question_context: Optional[Any],
        has_image: bool = False,
        has_course_context: bool = False,
        chat_mode: str = "study_solver",
    ) -> Tuple[bool, str, str]:
        text = re.sub(r"\s+", " ", str(user_message or "").strip().lower())

        if chat_mode == "learning_advisor":
            system_keywords = (
                "ระบบ",
                "เมนู",
                "ใช้งาน",
                "ล็อกอิน",
                "สมัคร",
                "platform",
                "dashboard",
                "ชำระเงิน",
                "แพ็กเกจ",
                "account",
            )
            course_keywords = (
                "คอร์ส",
                "บทเรียน",
                "ข้อสอบ",
                "วิชา",
                "คะแนน",
                "ความคืบหน้า",
                "course",
                "lesson",
                "quiz",
                "progress",
            )
            if any(key in text for key in system_keywords):
                return False, "advisor_system_keywords", "system_related"
            if has_course_context and any(key in text for key in course_keywords):
                return False, "advisor_course_keywords", "course_related"
            return False, "advisor_general_chat", "general"

        if has_image:
            return (
                True,
                "image_attached",
                ("course_related" if has_course_context else "general"),
            )
        if not parsed_question_context:
            if any(
                key in text
                for key in (
                    "ระบบ",
                    "เมนู",
                    "ใช้งาน",
                    "dashboard",
                    "platform",
                    "account",
                )
            ):
                return False, "system_without_question_context", "system_related"
            if has_course_context and any(
                key in text
                for key in (
                    "คอร์ส",
                    "บทเรียน",
                    "วิชา",
                    "ข้อสอบ",
                    "course",
                    "lesson",
                    "quiz",
                )
            ):
                return False, "course_without_question_context", "course_related"
            return False, "no_question_context", "general"
        if not text:
            return False, "empty_user_message", "general"

        question_keywords = (
            "ข้อนี้",
            "ข้อ",
            "โจทย์",
            "คำตอบ",
            "เฉลย",
            "วิธีทำ",
            "ช่วยดู",
            "ถูกไหม",
            "ผิดไหม",
            "ทำยังไง",
            "เลือกข้อ",
            "อธิบายโจทย์",
            "answer",
            "solution",
            "choice",
            "explain",
            "solve",
            "question",
        )
        unsure_keywords = (
            "งง",
            "ไม่เข้าใจ",
            "ไม่มั่นใจ",
            "ติดตรง",
            "ตรงที่",
            "จุดที่งง",
            "ช่วยไล่",
            "ยังไม่ชัด",
            "ไม่เคลียร์",
            "why",
            "confuse",
            "stuck",
            "unclear",
        )
        generic_keywords = (
            "สวัสดี",
            "hello",
            "hi",
            "ขอบคุณ",
            "thanks",
            "ช่วยแนะนำ",
            "คอร์ส",
            "เรียนยังไง",
            "อ่านยังไง",
            "ทบทวนยังไง",
            "แผนการเรียน",
            "goal",
        )

        if any(key in text for key in question_keywords):
            return (
                True,
                "keyword_question_related",
                ("course_related" if has_course_context else "general"),
            )
        if any(key in text for key in unsure_keywords):
            return (
                True,
                "keyword_unsure_related",
                ("course_related" if has_course_context else "general"),
            )
        if any(key in text for key in generic_keywords):
            return False, "keyword_general_chat", "general"
        if any(
            key in text
            for key in (
                "ระบบ",
                "ล็อกอิน",
                "สมัคร",
                "dashboard",
                "platform",
                "account",
                "เมนู",
            )
        ):
            return False, "keyword_system_chat", "system_related"

        question_blob = self._build_question_context_block(parsed_question_context)
        msg_tokens = set(re.findall(r"[a-zA-Z0-9ก-๙]+", text))
        ctx_tokens = set(re.findall(r"[a-zA-Z0-9ก-๙]+", question_blob.lower()))
        overlap = msg_tokens & ctx_tokens
        ratio = (len(overlap) / max(1, len(msg_tokens))) if msg_tokens else 0.0
        if len(overlap) >= 2 or ratio >= 0.2:
            return (
                True,
                "token_overlap",
                ("course_related" if has_course_context else "general"),
            )

        return False, "uncertain", "general"

    async def _llm_question_context_classifier(
        self,
        user_message: str,
        parsed_question_context: Optional[Any],
    ) -> Tuple[bool, str, float]:
        """LLM-based fallback classification for ambiguous messages."""
        context_preview = self._build_question_context_block(parsed_question_context)
        context_preview = context_preview[:1200]
        classifier_prompt = (
            "คุณคือระบบจัดเส้นทางแชทติวเตอร์ ให้ตัดสินว่า 'ข้อความผู้ใช้' ต้องใช้บริบทโจทย์หรือไม่\n"
            "ตอบ JSON object เพียงบรรทัดเดียวเท่านั้น (ห้าม markdown/code fence)\n"
            "รูปแบบที่ต้องตอบ: "
            '{"requires_context": true|false, "confidence": 0.0-1.0, "reason": "short"}\n'
            "เกณฑ์:\n"
            "- requires_context=true เมื่อผู้ใช้ถาม/อ้างอิงโจทย์นี้โดยตรง, ขอเฉลย, ขออธิบายคำตอบ, ตรวจวิธีทำ, หรือมีคำที่ผูกกับข้อสอบ\n"
            "- requires_context=false เมื่อผู้ใช้คุยทั่วไป/ถามเรื่องระบบ/ถามเรื่องคอร์สแบบไม่ผูกโจทย์ปัจจุบัน\n"
            "- confidence ต้องอยู่ระหว่าง 0 และ 1\n\n"
            f"ข้อความผู้ใช้: {user_message}\n"
            f"บริบทโจทย์: {context_preview}"
        )
        try:
            raw = await self._call_gemini_chat(
                classifier_prompt,
                max_tokens=min(
                    self.settings.chat_context_classifier_max_tokens,
                    140,
                ),
                model_name=self._resolve_context_classifier_model(),
            )
            parsed = self._safe_extract_json_object(raw)
            if isinstance(parsed, dict):
                related_raw = parsed.get("requires_context")
                if related_raw is None:
                    related_raw = parsed.get("related")
                related = self._coerce_bool(related_raw)
                confidence_raw = parsed.get("confidence")
                if confidence_raw is None:
                    confidence_raw = parsed.get("score")
                confidence = self._coerce_confidence(confidence_raw)
                reason = str(parsed.get("reason") or "classified")
                return related, reason, confidence
            return False, "invalid_classifier_payload", 0.0
        except Exception as exc:
            app_logger.warning(f"LLM classifier fallback failed: {exc}")
        return False, "classifier_failed", 0.0

    def _safe_extract_json_object(self, raw: str) -> Optional[Dict[str, Any]]:
        text = (raw or "").strip()
        if not text:
            return None
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    async def _refresh_conversation_summary(self, conversation_id: str) -> None:
        """Refresh summary memory periodically to preserve long conversation context."""
        memory = self._session_memory.get(conversation_id)
        if memory is None:
            return
        messages = memory.chat_memory.messages
        if len(messages) < 8:
            return
        # Update every 3 user turns (6 messages) to keep cost predictable.
        if len(messages) % 6 != 0:
            return

        prev_summary = self._session_summary.get(conversation_id, "")
        recent = messages[-12:]
        rendered_turns: List[str] = []
        for msg in recent:
            role = "ผู้ใช้" if getattr(msg, "type", "") == "human" else "AI"
            content = re.sub(r"\s+", " ", str(msg.content or "").strip())
            if content:
                rendered_turns.append(f"{role}: {content}")
        if not rendered_turns:
            return

        summary_prompt = (
            "สรุปบทสนทนาเพื่อใช้เป็น memory สำหรับช่วยตอบต่อเนื่อง\n"
            "ข้อกำหนด:\n"
            "- ภาษาไทย กระชับ ไม่เกิน 8 bullet\n"
            "- เก็บเฉพาะข้อเท็จจริงสำคัญ, เจตนาผู้ใช้, สิ่งที่ทำไปแล้ว, ข้อจำกัดที่ต้องจำ\n"
            "- ไม่ต้องใส่คำเกริ่น\n\n"
            f"สรุปเดิม:\n{prev_summary or '-'}\n\n"
            f"บทสนทนาล่าสุด:\n" + "\n".join(rendered_turns)
        )
        try:
            new_summary = await self._call_gemini_chat(summary_prompt, max_tokens=220)
            clean_summary = re.sub(r"\s+\n", "\n", (new_summary or "").strip())
            if clean_summary:
                self._session_summary[conversation_id] = clean_summary
        except Exception as exc:
            app_logger.warning(f"Conversation summary update skipped: {exc}")

    def _calculate_response_confidence(self, response: str, user_message: str) -> float:
        """Calculate confidence score for the response."""
        if not response or not response.strip():
            return 0.0

        # Basic confidence calculation
        confidence = 0.8  # Base confidence for Gemini

        # Length factor
        if len(response) > 100:
            confidence += 0.1

        # Relevance factor (basic keyword matching)
        user_words = set(user_message.lower().split())
        response_words = set(response.lower().split())

        if user_words & response_words:  # If there are common words
            confidence += 0.05

        # Structure factor (if response has bullet points, numbered lists)
        if any(marker in response for marker in ["•", "-", "1.", "2.", "ก.", "ข."]):
            confidence += 0.05

        return min(0.95, confidence)

    def _is_response_incomplete(self, response: str) -> bool:
        """Heuristic check for clipped/incomplete model responses."""
        if not response:
            return True
        text = response.strip()

        terminal_chars = (".", "!", "?", "…", "ฯ", "”", '"', "'", ")", "】", "」")
        if text.endswith(terminal_chars):
            return False

        # Very short endings like "1. u" are usually clipped outputs.
        if len(text) < 40:
            if any(token in text for token in ("1.", "2.", "###", "**", "-", "•")):
                return True

        likely_incomplete_suffixes = (
            "ที่",
            "คือ",
            "ว่า",
            "และ",
            "หรือ",
            "โดย",
            "ซึ่ง",
            "เช่น",
            "ตัวเลข",
            "คำตอบ",
            ":",
        )
        if any(text.endswith(sfx) for sfx in likely_incomplete_suffixes):
            return True

        # Treat as complete by default unless there is a strong incomplete signal.
        return False

    def _merge_response_with_continuation(
        self, response: str, continuation: str
    ) -> str:
        """Merge continuation text while avoiding duplicate repeated content."""
        base = (response or "").strip()
        cont = (continuation or "").strip()
        if not cont:
            return base
        if not base:
            return cont
        if cont == base or cont in base:
            return base
        if base in cont:
            return cont

        # If continuation is mostly duplicated lines, skip it.
        base_lines = [ln.strip() for ln in base.splitlines() if ln.strip()]
        cont_lines = [ln.strip() for ln in cont.splitlines() if ln.strip()]
        if cont_lines:
            base_set = set(base_lines)
            overlap_ratio = sum(1 for ln in cont_lines if ln in base_set) / len(
                cont_lines
            )
            if overlap_ratio >= 0.6:
                return base

        # Merge by overlap between end of base and start of continuation.
        max_overlap = min(len(base), len(cont), 500)
        for size in range(max_overlap, 19, -1):
            if base[-size:] == cont[:size]:
                return f"{base}\n{cont[size:].lstrip()}".strip()

        # Fallback normalized duplicate check.
        norm_base = re.sub(r"\s+", " ", base).strip().lower()
        norm_cont = re.sub(r"\s+", " ", cont).strip().lower()
        if norm_cont and norm_cont in norm_base:
            return base

        return f"{base}\n{cont}".strip()

    async def get_conversation_history(
        self, conversation_id: str, user_id: str
    ) -> Optional[ConversationHistory]:
        """Get full conversation history."""
        messages = self._get_conversation_history(conversation_id)

        if not messages:
            return None

        # Find course_id from message metadata
        course_id = None
        for msg in messages:
            if msg.metadata and msg.metadata.get("course_id"):
                course_id = msg.metadata["course_id"]
                break

        return ConversationHistory(
            conversation_id=conversation_id,
            user_id=user_id,
            course_id=course_id,
            messages=messages,
            created_at=messages[0].timestamp if messages else datetime.utcnow(),
            updated_at=messages[-1].timestamp if messages else datetime.utcnow(),
        )

    def add_course_context(self, course_id: str, context: Dict[str, Any]):
        """Add or update course context."""
        self.course_contexts[course_id] = context
        app_logger.info(f"Updated course context for course {course_id}")

    async def clear_conversation(self, conversation_id: str):
        """Clear conversation history."""
        if conversation_id in self._session_memory:
            del self._session_memory[conversation_id]
        if conversation_id in self._session_summary:
            del self._session_summary[conversation_id]
            app_logger.info(f"Cleared conversation {conversation_id}")
