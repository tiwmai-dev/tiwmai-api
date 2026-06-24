"""Student authentication service backed by Supabase Auth."""

from typing import Any, Dict, Optional

from pydantic import BaseModel

from app.services.auth_service import AuthService, UserInfo


class StudentInfo(BaseModel):
    """Student user information returned by auth dependencies."""

    user_id: str
    username: str
    email: str
    email_verified: bool
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    student_id: Optional[str] = None
    phone_number: Optional[str] = None
    groups: Optional[list] = None


class StudentAuthService(AuthService):
    """Supabase auth service with the student role default."""

    role = "student"

    def _student_info_from_user_info(
        self,
        user_info: UserInfo,
        profile: Optional[Dict[str, Any]] = None,
    ) -> StudentInfo:
        profile = profile or {}
        profile_user_id = str(profile.get("user_id") or "").strip()
        student_id = profile.get("student_id")
        if not student_id and profile_user_id and profile_user_id != user_info.user_id:
            student_id = profile_user_id
        return StudentInfo(
            user_id=user_info.user_id,
            username=user_info.username,
            email=user_info.email,
            email_verified=user_info.email_verified,
            given_name=user_info.given_name,
            family_name=user_info.family_name,
            student_id=student_id,
            phone_number=user_info.phone_number,
            groups=user_info.groups,
        )

    async def register_student(
        self,
        username: str,
        password: str,
        email: str,
        given_name: Optional[str] = None,
        family_name: Optional[str] = None,
        student_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = await self.register_user(
            username, password, email, given_name, family_name
        )
        if student_id:
            await self._upsert_profile(
                result["user_id"],
                email,
                username,
                "student",
                given_name,
                family_name,
                {"student_id": student_id},
            )
        result["student_id"] = student_id
        result["message"] = "Student registered successfully"
        return result

    async def resend_verification_email(self, email: str) -> Dict[str, str]:
        return await AuthService.resend_verification_email(self, email)

    async def authenticate_student(
        self, username: str, password: str
    ) -> Dict[str, Any]:
        return await self.authenticate_user(username, password)

    async def get_student_info(self, access_token: str) -> StudentInfo:
        user_info = await self.get_user_info(access_token)
        profile = await self._get_profile(user_info.user_id)
        return self._student_info_from_user_info(user_info, profile)

    async def logout_student(self, access_token: str) -> bool:
        return await self.logout_user(access_token)
