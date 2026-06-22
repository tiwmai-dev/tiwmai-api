from datetime import datetime, timezone

from app.models.schemas import ChatMessage, ChatResponse


def test_chat_response_serializes_naive_timestamp_as_utc_z():
    response = ChatResponse(
        message_id="message-1",
        content="ok",
        timestamp=datetime(2026, 6, 17, 12, 47),
        conversation_id="conversation-1",
    )

    assert response.model_dump(mode="json")["timestamp"] == "2026-06-17T12:47:00Z"


def test_chat_message_serializes_aware_timestamp_as_utc_z():
    message = ChatMessage(
        id="message-1",
        type="ai",
        content="ok",
        timestamp=datetime(2026, 6, 17, 19, 47, tzinfo=timezone.utc),
    )

    assert message.model_dump(mode="json")["timestamp"] == "2026-06-17T19:47:00Z"
