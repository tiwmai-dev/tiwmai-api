"""Course invitation API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from app.api.auth_endpoints import get_current_user, security
from app.api.student_auth_endpoints import get_current_student
from app.core.logging import app_logger
from app.models.schemas import (
    AcceptInvitationRequest,
    CreateInvitationRequest,
    InvitationListResponse,
    InvitationResponse,
)
from app.services.auth_service import AuthService, UserInfo
from app.services.invitation_service import InvitationService
from app.services.student_auth_service import StudentAuthService, StudentInfo

# Create router
router = APIRouter(prefix="/invitations", tags=["invitations"])

# Single instance of invitation service (singleton pattern)
_invitation_service_instance = None


# Dependency to get invitation service
async def get_invitation_service() -> InvitationService:
    global _invitation_service_instance
    if _invitation_service_instance is None:
        _invitation_service_instance = InvitationService()
    return _invitation_service_instance


@router.post("/create", response_model=InvitationResponse)
async def create_course_invitation(
    invitation_request: CreateInvitationRequest,
    current_user: UserInfo = Depends(get_current_user),
    invitation_service: InvitationService = Depends(get_invitation_service),
):
    """
    Create a course invitation (instructor only).

    - **student_id**: ID of the student to invite
    - **course_id**: ID of the course to invite to
    - **message**: Optional invitation message
    """
    try:
        app_logger.info(f"=== INVITATION CREATE REQUEST ===")
        app_logger.info(f"Request data: {invitation_request}")
        app_logger.info(f"Current user: {current_user}")
        app_logger.info(
            f"Instructor {current_user.username} creating invitation for student {invitation_request.student_id}"
        )

        # Create invitation
        result = await invitation_service.create_invitation(
            instructor_id=current_user.user_id,
            instructor_name=current_user.given_name or current_user.username,
            request=invitation_request,
        )

        app_logger.info(f"Invitation created successfully: {result.invitation_id}")
        return result

    except HTTPException as e:
        app_logger.error(
            f"HTTPException in invitation creation: {e.status_code} - {e.detail}"
        )
        raise
    except Exception as e:
        app_logger.error(f"Unexpected error creating invitation: {e}")
        app_logger.error(f"Exception type: {type(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create invitation",
        )


@router.get("/student", response_model=InvitationListResponse)
async def get_student_invitations(
    current_student: StudentInfo = Depends(get_current_student),
    invitation_service: InvitationService = Depends(get_invitation_service),
):
    """
    Get all invitations for the current student.
    """
    try:
        app_logger.info(f"Getting invitations for student {current_student.username}")

        result = await invitation_service.get_student_invitations(
            student_id=current_student.username
        )

        app_logger.info(
            f"Found {result.total} invitations for student {current_student.username}"
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error getting student invitations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get invitations",
        )


@router.get("/course/{course_id}", response_model=InvitationListResponse)
async def get_course_invitations(
    course_id: str,
    current_user: UserInfo = Depends(get_current_user),
    invitation_service: InvitationService = Depends(get_invitation_service),
):
    """
    Get all invitations for a specific course (instructor only).
    """
    try:
        app_logger.info(
            f"Getting invitations for course {course_id} by instructor {current_user.username}"
        )

        result = await invitation_service.get_course_invitations(
            course_id=course_id, instructor_id=current_user.user_id
        )

        app_logger.info(f"Found {result.total} invitations for course {course_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error getting course invitations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get course invitations",
        )


@router.post("/accept", response_model=InvitationResponse)
async def accept_invitation(
    accept_request: AcceptInvitationRequest,
    current_student: StudentInfo = Depends(get_current_student),
    invitation_service: InvitationService = Depends(get_invitation_service),
):
    """
    Accept a course invitation (student only).

    - **invitation_id**: ID of the invitation to accept
    """
    try:
        app_logger.info(
            f"Student {current_student.username} accepting invitation {accept_request.invitation_id}"
        )

        result = await invitation_service.accept_invitation(
            student_id=current_student.username,
            invitation_id=accept_request.invitation_id,
        )

        app_logger.info(
            f"Invitation {accept_request.invitation_id} accepted successfully"
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error accepting invitation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to accept invitation",
        )


@router.post("/decline/{invitation_id}", response_model=InvitationResponse)
async def decline_invitation(
    invitation_id: str,
    current_student: StudentInfo = Depends(get_current_student),
    invitation_service: InvitationService = Depends(get_invitation_service),
):
    """
    Decline a course invitation (student only).
    """
    try:
        app_logger.info(
            f"Student {current_student.username} declining invitation {invitation_id}"
        )

        result = await invitation_service.decline_invitation(
            student_id=current_student.username, invitation_id=invitation_id
        )

        app_logger.info(f"Invitation {invitation_id} declined successfully")
        return result

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error declining invitation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to decline invitation",
        )
