"""Student chat endpoint handler."""

from typing import Optional

from fastapi import Depends, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials

from app.core.exceptions import BaseAPIException
from app.core.logging import app_logger
from app.models.schemas import ChatResponse
from app.services.chat_service import ChatService
from app.services.student_auth_service import StudentAuthService
from app.api.student_handlers import (
    STUDENT_BEARER_OPTIONAL,
    _get_student_auth_service,
    _require_user_matches_token,
)


async def get_chat_service() -> ChatService:
    return ChatService()


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
