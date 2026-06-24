"""Chat API endpoints and student chat handler."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials

from app.core.exceptions import BaseAPIException
from app.core.logging import app_logger
from app.models.schemas import ChatMessageRequest, ChatResponse, ConversationHistory
from app.services.chat_service import ChatService
from app.services.student_auth_service import StudentAuthService
from app.api.student_handlers import (
    STUDENT_BEARER_OPTIONAL,
    _get_student_auth_service,
    _require_user_matches_token,
)

router = APIRouter()


async def get_chat_service() -> ChatService:
    return ChatService()


@router.post("/chat", response_model=ChatResponse)
async def send_chat_message(
    message: str = Form(..., description="User message"),
    user_id: str = Form(..., description="User ID"),
    course_id: Optional[str] = Form(None, description="Course ID for context"),
    conversation_id: Optional[str] = Form(
        None, description="Conversation ID for context"
    ),
    question_context: Optional[str] = Form(
        None, description="Current question text/context"
    ),
    chat_mode: Optional[str] = Form(
        "study_solver", description="Chat mode: study_solver|learning_advisor"
    ),
    image: Optional[UploadFile] = File(None, description="Optional image attachment"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(STUDENT_BEARER_OPTIONAL),
    student_auth_service: StudentAuthService = Depends(_get_student_auth_service),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Send a message to the AI chat assistant."""
    try:
        app_logger.info(f"Processing chat message from user {user_id}")

        await _require_user_matches_token(
            user_id=user_id,
            credentials=credentials,
            auth_service=student_auth_service,
        )

        if len(message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        if len(message) > 2000:
            raise HTTPException(
                status_code=400, detail="Message too long (max 2000 characters)"
            )

        normalized_mode = chat_service.normalize_chat_mode(chat_mode)

        image_bytes = None
        image_mime = None
        if image is not None:
            if normalized_mode == "learning_advisor":
                raise HTTPException(
                    status_code=400,
                    detail="Image attachment is not supported in learning advisor mode",
                )
            if not image.content_type or not image.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400, detail="Only image files are supported"
                )
            image_bytes = await image.read()
            if len(image_bytes) > 5 * 1024 * 1024:
                raise HTTPException(
                    status_code=400, detail="Image file too large (max 5MB)"
                )
            image_mime = image.content_type
            app_logger.info(
                f"Chat received image attachment: bytes={len(image_bytes)} mime={image_mime}"
            )

        response = await chat_service.get_chat_response(
            user_message=message.strip(),
            user_id=user_id,
            course_id=course_id,
            conversation_id=conversation_id,
            question_context=question_context,
            image_bytes=image_bytes,
            image_mime=image_mime,
            chat_mode=normalized_mode,
        )

        app_logger.info(f"Chat response generated successfully for user {user_id}")
        return response

    except HTTPException:
        raise
    except BaseAPIException as e:
        app_logger.error(f"Chat API error for user {user_id}: {e}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        app_logger.error(f"Unexpected error in chat for user {user_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Chat service temporarily unavailable"
        )


@router.post("/chat/json", response_model=ChatResponse)
async def send_chat_message_json(
    request: ChatMessageRequest, chat_service: ChatService = Depends(get_chat_service)
):
    """Send a message to the AI chat assistant in JSON format."""
    try:
        app_logger.info(f"Processing JSON chat message from user {request.user_id}")

        response = await chat_service.get_chat_response(
            user_message=request.message,
            user_id=request.user_id,
            course_id=request.course_id,
            conversation_id=request.conversation_id,
            question_context=request.question_context,
            chat_mode=request.chat_mode,
        )

        app_logger.info(
            f"Chat response generated successfully for user {request.user_id}"
        )
        return response

    except BaseAPIException as e:
        app_logger.error(f"Chat API error for user {request.user_id}: {e}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        app_logger.error(f"Unexpected error in chat for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Chat service temporarily unavailable"
        )


@router.get("/chat/conversation/{conversation_id}", response_model=ConversationHistory)
async def get_conversation_history(
    conversation_id: str,
    user_id: str,
    chat_service: ChatService = Depends(get_chat_service),
):
    """Get conversation history for a specific conversation."""
    try:
        app_logger.info(f"Retrieving conversation {conversation_id} for user {user_id}")

        history = await chat_service.get_conversation_history(
            conversation_id=conversation_id, user_id=user_id
        )

        if not history:
            raise HTTPException(
                status_code=404, detail=f"Conversation {conversation_id} not found"
            )

        return history

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error retrieving conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation")


@router.delete("/chat/conversation/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    user_id: str,
    chat_service: ChatService = Depends(get_chat_service),
):
    """Clear conversation history."""
    try:
        app_logger.info(f"Clearing conversation {conversation_id} for user {user_id}")

        await chat_service.clear_conversation(conversation_id)

        return {
            "message": f"Conversation {conversation_id} cleared successfully",
            "conversation_id": conversation_id,
            "cleared_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        app_logger.error(f"Error clearing conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear conversation")


@router.get("/chat/health")
async def chat_health_check():
    """Health check for chat service."""
    try:
        chat_service = ChatService()

        return {
            "service": "chat",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "llm_configured": bool(
                (chat_service.settings.openrouter_api_key or "").strip()
                or (chat_service.settings.litellm_api_key or "").strip()
                or (chat_service.settings.gemini_api_key or "").strip()
            ),
            "conversation_storage": "langchain_memory_with_summary",
            "context_router": "langgraph"
            if chat_service.langgraph_available
            else "heuristic",
            "supported_languages": ["th", "en"],
        }

    except Exception as e:
        app_logger.error(f"Chat health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "service": "chat",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
