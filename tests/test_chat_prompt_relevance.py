import pytest
from unittest.mock import AsyncMock

from app.services.chat_service import ChatService


@pytest.mark.unit
def test_study_solver_prompt_requires_off_topic_scope_during_active_question():
    service = ChatService()

    prompt = service._create_chat_prompt(
        user_message="ช่วยอธิบายข้อนี้หน่อย",
        course_context=None,
        system_context=None,
        conversation_history=[],
        conversation_summary="",
        user_id="student-1",
        question_context={"question_text": "2 + 2 เท่ากับเท่าไร"},
        chat_mode="study_solver",
        active_question_session=True,
    )

    assert "ข้อกำหนดเรื่องขอบเขตการตอบ" in prompt
    assert "ให้ปฏิเสธอย่างสุภาพ" in prompt
    assert "ชวนกลับมาโฟกัสโจทย์ปัจจุบัน" in prompt


@pytest.mark.unit
def test_study_solver_prompt_skips_off_topic_scope_without_active_question():
    service = ChatService()

    prompt = service._create_chat_prompt(
        user_message="ช่วยแนะนำการเรียน",
        course_context=None,
        system_context=None,
        conversation_history=[],
        conversation_summary="",
        user_id="student-1",
        question_context=None,
        chat_mode="study_solver",
        active_question_session=False,
    )

    assert "ข้อกำหนดเรื่องขอบเขตการตอบ" not in prompt


@pytest.mark.unit
@pytest.mark.parametrize(
    ("message", "expected_fragment"),
    [
        ("สวัสดีค่ะ", "ลองบอกได้เลยว่าติดตรงไหน"),
        ("วันนี้อากาศดีไหม", "ช่วยเฉพาะเรื่องโจทย์ที่กำลังทำอยู่"),
        ("ช่วยแนะนำหนังหน่อย", "ช่วยเฉพาะเรื่องโจทย์ที่กำลังทำอยู่"),
    ],
)
def test_build_off_topic_refusal_message(message, expected_fragment):
    response = ChatService._build_off_topic_refusal_message(message)
    assert expected_fragment in response


@pytest.mark.unit
def test_heuristic_marks_general_chat_as_off_question_context():
    service = ChatService()
    should_include, reason, route = service._heuristic_question_context_decision(
        user_message="วันนี้อากาศดีไหม",
        parsed_question_context={"question_text": "2 + 2 เท่ากับเท่าไร"},
        chat_mode="study_solver",
    )

    assert should_include is False
    assert reason == "keyword_off_topic"
    assert route == "general"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_chat_response_refuses_off_topic_without_llm(monkeypatch):
    service = ChatService()

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called for off-topic question chat")

    monkeypatch.setattr(service, "_call_gemini_chat", fail_if_called)
    monkeypatch.setattr(service, "_call_gemini_chat_with_image", fail_if_called)

    async def fake_energy_status(user_id):
        return {
            "daily_limit_thb": 2.0,
            "used_thb": 0.0,
            "remaining_thb": 2.0,
            "remaining_percent": 100.0,
            "is_exhausted": False,
        }

    db_service = type(
        "FakeDb",
        (),
        {
            "get_student_chat_energy_status": fake_energy_status,
            "record_student_token_usage": AsyncMock(),
        },
    )()

    monkeypatch.setattr("app.services.chat_service.get_db_service", lambda: db_service)

    response = await service.get_chat_response(
        user_message="ช่วยแนะนำหนังสนุกๆ หน่อย",
        user_id="student-1",
        question_context='{"question_text": "2 + 2 เท่ากับเท่าไร"}',
        chat_mode="study_solver",
    )

    assert "ช่วยเฉพาะเรื่องโจทย์ที่กำลังทำอยู่" in response.content


@pytest.mark.unit
def test_learning_advisor_prompt_requires_clarification_for_ambiguous_messages():
    service = ChatService()

    prompt = service._create_chat_prompt(
        user_message="ก็เรา อยู่ในโลกนี้นา",
        course_context=None,
        system_context=None,
        conversation_history=[],
        conversation_summary="",
        user_id="student-1",
        chat_mode="learning_advisor",
        learning_context=None,
    )

    assert "ให้ยึดข้อความผู้ใช้ปัจจุบันเป็นหลัก" in prompt
    assert "ห้ามตีความข้อความกำกวมเป็นเรื่องเชิงปรัชญา" in prompt
    assert "ให้ถามกลับเพียง 1 คำถามสั้นๆ แทนการเดาความหมาย" in prompt
    assert "ข้อความผู้ใช้ปัจจุบัน:\nก็เรา อยู่ในโลกนี้นา" in prompt


@pytest.mark.unit
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("คะแนนเฉลี่ยล่าสุดเท่าไร", True),
        ("ควรโฟกัสวิชาไหน", True),
        ("สวัสดี วันนี้เป็นยังไงบ้าง", False),
        ("ก็เรา อยู่ในโลกนี้นา", False),
    ],
)
def test_learning_advisor_loads_learning_context_only_when_needed(message, expected):
    assert ChatService._learning_advisor_needs_learning_context(message) is expected


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openrouter_user_context_does_not_query_profile(monkeypatch):
    service = ChatService()

    def fail_if_called():
        raise AssertionError("profile database should not be queried")

    monkeypatch.setattr("app.services.chat_service.get_db_service", fail_if_called)

    user, metadata = await service._get_openrouter_user_context("student-1")

    assert user == "student-1"
    assert metadata == {
        "app_user_id": "student-1",
        "source": "tanaijarn-backend",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_question_context_alignment_skips_llm_when_tokens_overlap(monkeypatch):
    service = ChatService()

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("alignment LLM should not be called")

    monkeypatch.setattr(
        service,
        "_llm_question_context_alignment_classifier",
        fail_if_called,
    )

    context, reason = await service._filter_incoherent_question_context(
        {
            "question_text": "จากบทความ ข้อใดกล่าวถึงพลังงานแสงอาทิตย์",
            "question_context_text": (
                "พลังงานแสงอาทิตย์เป็นพลังงานหมุนเวียนที่นำมาใช้ผลิตไฟฟ้าได้"
            ),
        }
    )

    assert context is not None
    assert reason in {
        "context_alignment:token_overlap",
        "context_alignment:shared_substring",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_question_context_classifier_caps_output_tokens(monkeypatch):
    service = ChatService()
    service.settings.chat_context_classifier_max_tokens = 1000
    captured = {}

    async def fake_call(prompt, max_tokens=None, **kwargs):
        captured["max_tokens"] = max_tokens
        return '{"aligned": true, "confidence": 1, "reason": "ok"}'

    monkeypatch.setattr(service, "_call_gemini_chat", fake_call)

    await service._llm_question_context_alignment_classifier(
        question_text="โจทย์คนละถ้อยคำ",
        shared_context="บริบทประกอบอีกแบบหนึ่ง",
    )

    assert captured["max_tokens"] == 140


@pytest.mark.unit
def test_apply_usage_to_energy_status_avoids_refresh_query():
    service = ChatService()
    service.settings.openrouter_cost_usd_to_thb = 36.0

    status = service._apply_usage_to_energy_status(
        {
            "daily_limit_thb": 2.0,
            "used_thb": 0.5,
            "remaining_thb": 1.5,
            "remaining_percent": 75.0,
            "is_exhausted": False,
        },
        llm_cost_usd=0.01,
    )

    assert status["used_thb"] == pytest.approx(0.86)
    assert status["remaining_thb"] == pytest.approx(1.14)
    assert status["remaining_percent"] == pytest.approx(57.0)
    assert status["is_exhausted"] is False


@pytest.mark.unit
def test_usage_cost_is_zero_when_openrouter_omits_cost():
    service = ChatService()

    cost = service._extract_openrouter_response_cost_usd(None)

    assert cost == 0.0
