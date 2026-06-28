"""Authentication service backed by Supabase Auth."""

import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx
import jwt
from fastapi import HTTPException, status
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import app_logger
from app.services.supabase_service import get_supabase_service


class UserInfo(BaseModel):
    """User information returned by auth dependencies."""

    user_id: str
    username: str
    email: str
    email_verified: bool
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    phone_number: Optional[str] = None
    groups: Optional[list] = None


def _obj_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    raw = {}
    for key in ("user", "session", "access_token", "refresh_token", "expires_in"):
        if hasattr(value, key):
            raw[key] = getattr(value, key)
    return raw


def _user_to_dict(user: Any) -> Dict[str, Any]:
    raw = _obj_to_dict(user)
    metadata = raw.get("user_metadata") or raw.get("raw_user_meta_data") or {}
    app_metadata = raw.get("app_metadata") or raw.get("raw_app_meta_data") or {}
    return {
        "id": raw.get("id") or raw.get("sub") or raw.get("user_id"),
        "email": raw.get("email") or metadata.get("email"),
        "email_confirmed_at": raw.get("email_confirmed_at"),
        "phone": raw.get("phone"),
        "metadata": metadata,
        "app_metadata": app_metadata,
    }


def _google_legacy_alias(metadata: Dict[str, Any]) -> Optional[str]:
    """Return the legacy Cognito-style google_<id> alias when Supabase exposes it."""
    for key in ("provider_id", "sub"):
        value = str(metadata.get(key) or "").strip()
        if value and value.isdigit():
            return f"google_{value}"
    return None


def _google_user_id(google_sub: Optional[str]) -> Optional[str]:
    value = str(google_sub or "").strip()
    if not value:
        return None
    return f"google_{value}" if value.isdigit() else value


DUPLICATE_USERNAME_MESSAGE = "ชื่อผู้ใช้นี้ถูกใช้แล้ว"
DUPLICATE_EMAIL_MESSAGE = "อีเมลนี้ถูกใช้สมัครบัญชีแล้ว"
EMAIL_NOT_VERIFIED_MESSAGE = (
    "กรุณายืนยันอีเมลก่อนเข้าสู่ระบบ ตรวจสอบกล่องจดหมายหรือขอส่งอีเมลยืนยันอีกครั้ง"
)
REGISTRATION_VERIFY_EMAIL_MESSAGE = (
    "สมัครสมาชิกสำเร็จ กรุณาตรวจสอบอีเมลเพื่อยืนยันบัญชีก่อนเข้าสู่ระบบ"
)
RESEND_VERIFICATION_EMAIL_MESSAGE = (
    "หากมีบัญชีที่ยังไม่ได้ยืนยัน เราได้ส่งอีเมลยืนยันไปแล้ว กรุณาตรวจสอบกล่องจดหมาย"
)


def _registration_error_detail(error: Exception) -> str:
    """Return a stable client-facing message for common Supabase sign-up errors."""
    message = str(error or "").strip()
    lowered = message.lower()

    if any(
        token in lowered
        for token in (
            "username",
            "profiles_username",
            "user_name",
        )
    ) and any(
        token in lowered
        for token in ("duplicate", "already exists", "unique", "already been registered")
    ):
        return DUPLICATE_USERNAME_MESSAGE

    if any(
        token in lowered
        for token in (
            "already registered",
            "already been registered",
            "email_exists",
        )
    ):
        return DUPLICATE_EMAIL_MESSAGE

    if "duplicate key" in lowered or "already exists" in lowered:
        if "email" in lowered:
            return DUPLICATE_EMAIL_MESSAGE
        if "username" in lowered:
            return DUPLICATE_USERNAME_MESSAGE

    if "invalid email" in lowered or "email address is invalid" in lowered:
        return "Email address is invalid"

    if "password" in lowered and any(
        token in lowered for token in ("weak", "short", "least", "length")
    ):
        return "Password does not meet the minimum security requirements"

    return "Registration failed"


def _login_error_detail(error: Exception) -> str:
    """Return a stable client-facing message for common Supabase login errors."""
    message = str(error or "").strip()
    lowered = message.lower()

    if any(
        token in lowered
        for token in (
            "email not confirmed",
            "email_not_confirmed",
            "not confirmed",
            "confirm your email",
        )
    ):
        return EMAIL_NOT_VERIFIED_MESSAGE

    return "Invalid username or password"


class AuthService:
    """Service for handling instructor authentication through Supabase."""

    role = "instructor"

    def __init__(self):
        self.settings = get_settings()
        self.supabase = get_supabase_service()
        self._verified_auth_users: Dict[str, Dict[str, Any]] = {}
        self._profile_cache: Dict[str, Dict[str, Any]] = {}

    @property
    def client(self):
        return self.supabase.client

    @property
    def anon_client(self):
        return self.supabase.anon_client

    def _supabase_auth_api_key(self) -> Optional[str]:
        return (
            str(self.settings.supabase_anon_key or "").strip()
            or str(self.settings.supabase_service_role_key or "").strip()
            or None
        )

    async def _fetch_auth_user_via_rest(self, access_token: str) -> Dict[str, Any]:
        supabase_url = str(self.settings.supabase_url or "").strip().rstrip("/")
        api_key = self._supabase_auth_api_key()
        if not supabase_url or not api_key:
            raise ValueError("Supabase REST auth lookup is not configured")

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{supabase_url}/auth/v1/user",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "apikey": api_key,
                },
            )
        if response.status_code != 200:
            raise ValueError(f"REST /auth/v1/user returned {response.status_code}")

        payload = response.json()
        user = _user_to_dict(payload)
        if not user.get("id"):
            raise ValueError("REST /auth/v1/user returned no user id")
        return user

    async def _refresh_session_via_rest(
        self, refresh_token: str
    ) -> Optional[Dict[str, Any]]:
        supabase_url = str(self.settings.supabase_url or "").strip().rstrip("/")
        api_key = self._supabase_auth_api_key()
        token = str(refresh_token or "").strip()
        if not supabase_url or not api_key or not token:
            return None

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{supabase_url}/auth/v1/token",
                params={"grant_type": "refresh_token"},
                headers={
                    "apikey": api_key,
                    "Content-Type": "application/json",
                },
                json={"refresh_token": token},
            )
        if response.status_code != 200:
            app_logger.warning(
                "Supabase REST refresh token exchange failed: {}",
                response.status_code,
            )
            return None

        payload = response.json() if response.content else {}
        if not isinstance(payload, dict):
            return None
        return payload

    async def _fetch_auth_user(self, access_token: str) -> Dict[str, Any]:
        """Resolve the authenticated Supabase user from a bearer access token."""
        attempts = []

        try:
            response = await self.supabase.run(
                self.anon_client.auth.get_user, access_token
            )
            user = _user_to_dict(
                getattr(response, "user", None) or _obj_to_dict(response).get("user")
            )
            if user.get("id"):
                return user
            attempts.append("anon client returned no user")
        except Exception as e:
            attempts.append(f"anon client failed: {e}")

        try:
            response = await self.supabase.run(self.client.auth.get_user, access_token)
            user = _user_to_dict(
                getattr(response, "user", None) or _obj_to_dict(response).get("user")
            )
            if user.get("id"):
                return user
            attempts.append("service client returned no user")
        except Exception as e:
            attempts.append(f"service client failed: {e}")

        try:
            user = await self._fetch_auth_user_via_rest(access_token)
            if user.get("id"):
                return user
            attempts.append("rest auth endpoint returned no user")
        except Exception as e:
            attempts.append(f"rest auth endpoint failed: {e}")

        raise ValueError("; ".join(attempts) or "Supabase returned no user")

    async def _get_profile(
        self, user_id: str, email: Optional[str] = None
    ) -> Dict[str, Any]:
        profile_cache = getattr(self, "_profile_cache", {})
        if user_id and user_id in profile_cache:
            return profile_cache[user_id]

        try:
            if user_id:
                result = await self.supabase.run(
                    lambda: self.client.table("profiles")
                    .select("*")
                    .eq("user_id", user_id)
                    .limit(1)
                    .execute()
                )
                rows = getattr(result, "data", None) or []
                if rows:
                    profile_cache[user_id] = rows[0]
                    self._profile_cache = profile_cache
                    return rows[0]
            if email:
                result = await self.supabase.run(
                    lambda: self.client.table("profiles")
                    .select("*")
                    .eq("email", email)
                    .limit(1)
                    .execute()
                )
                rows = getattr(result, "data", None) or []
                if rows:
                    if user_id:
                        profile_cache[user_id] = rows[0]
                        self._profile_cache = profile_cache
                    return rows[0]
            return {}
        except Exception as e:
            app_logger.warning(f"Unable to load Supabase profile for {user_id}: {e}")
            return {}

    async def _get_profile_by_legacy_alias(
        self, alias: Optional[str]
    ) -> Dict[str, Any]:
        if not alias:
            return {}
        profile = await self._get_profile(alias)
        if profile:
            return profile
        try:
            result = await self.supabase.run(
                lambda: self.client.table("profiles")
                .select("*")
                .eq("username", alias)
                .limit(1)
                .execute()
            )
            rows = getattr(result, "data", None) or []
            return rows[0] if rows else {}
        except Exception as e:
            app_logger.warning(f"Unable to load legacy profile for {alias}: {e}")
            return {}

    async def _username_is_taken(self, username: str) -> bool:
        value = str(username or "").strip()
        if not value:
            return False
        try:
            result = await self.supabase.run(
                lambda: self.client.table("profiles")
                .select("user_id")
                .ilike("username", value)
                .limit(1)
                .execute()
            )
            rows = getattr(result, "data", None) or []
            return bool(rows)
        except Exception as e:
            app_logger.warning(f"Username availability check failed for {value}: {e}")
            return False

    def _frontend_callback_redirect_url(self) -> str:
        configured = str(self.settings.student_web_app_url or "").strip().rstrip("/")
        if configured:
            return f"{configured}/auth/callback"
        oauth_redirect = str(self.settings.supabase_oauth_redirect_uri or "").strip()
        if oauth_redirect:
            return oauth_redirect
        if self.settings.allowed_origins_list:
            return f"{self.settings.allowed_origins_list[0].rstrip('/')}/auth/callback"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Frontend callback redirect URL is not configured",
        )

    def _email_verification_redirect_url(self) -> str:
        return self._frontend_callback_redirect_url()

    async def _send_verification_email(self, email: str) -> None:
        normalized_email = str(email or "").strip()
        if not normalized_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email address is required",
            )
        redirect_to = self._email_verification_redirect_url()
        try:
            await self.supabase.run(
                self.anon_client.auth.resend,
                {
                    "type": "signup",
                    "email": normalized_email,
                    "options": {"email_redirect_to": redirect_to},
                },
            )
        except Exception as e:
            app_logger.error(
                "Failed to send verification email to %s: %s", normalized_email, e
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="ไม่สามารถส่งอีเมลยืนยันได้ กรุณาลองใหม่อีกครั้ง",
            )

    async def resend_verification_email(self, email: str) -> Dict[str, str]:
        normalized_email = str(email or "").strip()
        if not normalized_email or "@" not in normalized_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email address is invalid",
            )
        try:
            await self._send_verification_email(normalized_email)
        except HTTPException as exc:
            if exc.status_code == status.HTTP_502_BAD_GATEWAY:
                raise
            app_logger.warning(
                "Verification email resend skipped for %s: %s",
                normalized_email,
                exc.detail,
            )
        return {"message": RESEND_VERIFICATION_EMAIL_MESSAGE}

    async def _upsert_profile(
        self,
        user_id: str,
        email: str,
        username: Optional[str],
        role: Optional[str] = None,
        given_name: Optional[str] = None,
        family_name: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "user_id": user_id,
            "email": email,
            "username": username or email,
            "name": " ".join(part for part in [given_name, family_name] if part).strip()
            or username
            or email,
            "role": role or self.role,
            "status": "active",
            "given_name": given_name,
            "family_name": family_name,
        }
        if extra:
            payload.update(extra)
        result = await self.supabase.run(
            lambda: self.client.table("profiles")
            .upsert(payload, on_conflict="user_id")
            .execute()
        )
        rows = getattr(result, "data", None) or []
        return rows[0] if rows else payload

    async def _resolve_login_email(self, username_or_email: str) -> str:
        value = str(username_or_email or "").strip()
        if "@" in value:
            return value
        try:
            role_result = await self.supabase.run(
                lambda: self.client.table("profiles")
                .select("email")
                .eq("username", value)
                .eq("role", self.role)
                .limit(1)
                .execute()
            )
            role_rows = getattr(role_result, "data", None) or []
            if role_rows and role_rows[0].get("email"):
                return role_rows[0]["email"]

            result = await self.supabase.run(
                lambda: self.client.table("profiles")
                .select("email")
                .eq("username", value)
                .limit(1)
                .execute()
            )
            rows = getattr(result, "data", None) or []
            if rows and rows[0].get("email"):
                return rows[0]["email"]
        except Exception as e:
            app_logger.warning(f"Username lookup failed for {value}: {e}")
        return value

    def _to_user_info(
        self,
        user: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> UserInfo:
        profile = profile or {}
        payload = payload or {}
        metadata = user.get("metadata") or {}
        app_metadata = user.get("app_metadata") or {}
        legacy_alias = _google_legacy_alias(metadata)
        groups = []
        role = (
            profile.get("role")
            or app_metadata.get("role")
            or metadata.get("role")
            or self.role
        )
        if role:
            groups.append(role)
        user_id = str(
            user.get("id") or profile.get("user_id") or payload.get("sub") or ""
        )
        email = str(
            user.get("email") or profile.get("email") or payload.get("email") or ""
        )
        username = str(
            profile.get("username")
            or metadata.get("username")
            or legacy_alias
            or payload.get("preferred_username")
            or email
            or user_id
        )
        return UserInfo(
            user_id=user_id,
            username=username,
            email=email,
            email_verified=bool(
                user.get("email_confirmed_at")
                or payload.get("email_confirmed_at")
                or payload.get("email_verified")
            ),
            given_name=profile.get("given_name") or metadata.get("given_name"),
            family_name=profile.get("family_name") or metadata.get("family_name"),
            phone_number=user.get("phone") or profile.get("phone_number"),
            groups=groups,
        )

    def _to_user_info_from_local_payload(self, payload: Dict[str, Any]) -> UserInfo:
        groups = payload.get("groups")
        if not isinstance(groups, list):
            groups = [self.role]
        return UserInfo(
            user_id=str(payload.get("sub") or ""),
            username=str(
                payload.get("username")
                or payload.get("preferred_username")
                or payload.get("email")
                or payload.get("sub")
                or ""
            ),
            email=str(payload.get("email") or ""),
            email_verified=bool(payload.get("email_verified", True)),
            given_name=payload.get("given_name"),
            family_name=payload.get("family_name"),
            phone_number=payload.get("phone_number"),
            groups=groups,
        )

    def _create_local_access_token(self, user_info: UserInfo) -> str:
        now = int(time.time())
        expires_at = now + int(self.settings.access_token_expire_minutes or 30) * 60
        payload = {
            "iss": "tiwmai-api",
            "tiwmai_auth": True,
            "sub": user_info.user_id,
            "username": user_info.username,
            "email": user_info.email,
            "email_verified": user_info.email_verified,
            "given_name": user_info.given_name,
            "family_name": user_info.family_name,
            "phone_number": user_info.phone_number,
            "groups": user_info.groups or [self.role],
            "iat": now,
            "exp": expires_at,
        }
        return jwt.encode(payload, self.settings.secret_key, algorithm="HS256")

    def _build_local_session_payload(
        self,
        user_info: UserInfo,
        *,
        refresh_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        local_access_token = self._create_local_access_token(user_info)
        ttl_seconds = int(self.settings.access_token_expire_minutes or 30) * 60
        return {
            "access_token": local_access_token,
            "refresh_token": refresh_token,
            "id_token": local_access_token,
            "token_type": "Bearer",
            "expires_in": ttl_seconds,
            "user": user_info.model_dump(),
        }

    async def _get_google_user_info(self, access_token: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
        if response.status_code != 200:
            raise ValueError(f"Google userinfo failed with {response.status_code}")
        data = response.json()
        if not data.get("sub") or not data.get("email"):
            raise ValueError("Google userinfo response missing sub or email")
        return data

    async def _create_local_google_session(self, access_token: str) -> Dict[str, Any]:
        google_user = await self._get_google_user_info(access_token)
        google_id = _google_user_id(google_user.get("sub"))
        metadata = {
            "provider_id": google_user.get("sub"),
            "username": google_id,
            "given_name": google_user.get("given_name"),
            "family_name": google_user.get("family_name"),
        }
        profile = await self._get_profile_by_legacy_alias(google_id)
        if not profile:
            profile = await self._get_profile("", google_user.get("email"))
        if not profile and google_id:
            profile = await self._upsert_profile(
                google_id,
                google_user.get("email"),
                google_id,
                self.role,
                google_user.get("given_name") or google_user.get("name"),
                google_user.get("family_name"),
            )
        user = {
            "id": profile.get("user_id") or google_id,
            "email": google_user.get("email"),
            "email_confirmed_at": True,
            "phone": None,
            "metadata": metadata,
            "app_metadata": {"role": profile.get("role") or self.role},
        }
        user_info = self._to_user_info(user, profile)
        local_access_token = self._create_local_access_token(user_info)
        return {
            "access_token": local_access_token,
            "refresh_token": None,
            "id_token": local_access_token,
            "token_type": "Bearer",
            "expires_in": int(self.settings.access_token_expire_minutes or 30) * 60,
            "user": user_info.model_dump(),
        }

    async def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify a Supabase access token and return its JWT payload."""
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing access token"
            )
        verified_users = getattr(self, "_verified_auth_users", {})
        cached_user = verified_users.get(token)
        if cached_user:
            return {
                "sub": cached_user["id"],
                "email": cached_user.get("email"),
                "exp": int(time.time()) + 3600,
            }

        try:
            user = await self._fetch_auth_user(token)
            verified_users[token] = user
            self._verified_auth_users = verified_users
            return {
                "sub": user["id"],
                "email": user.get("email"),
                "exp": int(time.time()) + 3600,
            }
        except Exception as auth_api_error:
            app_logger.warning(
                "Supabase Auth API token verification failed (%s); trying local JWT verification.",
                auth_api_error,
            )

        jwt_algorithm = str(self.settings.jwt_algorithm or "HS256").upper()
        # Local secret verification works only for shared-secret algorithms.
        # New Supabase signing keys (e.g. ECC P-256 / ES256) must be validated
        # via Supabase Auth (or JWKS) instead of local secret decode.
        if self.settings.supabase_jwt_secret and jwt_algorithm.startswith("HS"):
            try:
                payload = jwt.decode(
                    token,
                    self.settings.supabase_jwt_secret,
                    algorithms=[self.settings.jwt_algorithm or "HS256"],
                    audience=self.settings.jwt_audience,
                    options={"verify_aud": bool(self.settings.jwt_audience)},
                    leeway=30,
                )
                return payload
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
                )
            except jwt.InvalidTokenError as e:
                app_logger.warning(
                    "Local Supabase JWT verification failed (%s).",
                    e,
                )

        try:
            return jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=["HS256"],
                options={"verify_aud": False},
                leeway=30,
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            app_logger.warning("Local app JWT verification failed (%s).", e)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token"
        )

    async def register_user(
        self,
        username: str,
        password: str,
        email: str,
        given_name: Optional[str] = None,
        family_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_username = str(username or "").strip()
        normalized_email = str(email or "").strip()
        if not normalized_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username is required",
            )
        if await self._username_is_taken(normalized_username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=DUPLICATE_USERNAME_MESSAGE,
            )
        try:
            metadata = {
                "username": normalized_username,
                "given_name": given_name,
                "family_name": family_name,
                "role": self.role,
            }
            response = await self.supabase.run(
                self.client.auth.admin.create_user,
                {
                    "email": normalized_email,
                    "password": password,
                    "email_confirm": False,
                    "user_metadata": metadata,
                    "app_metadata": {"role": self.role},
                },
            )
            user = _user_to_dict(
                getattr(response, "user", None) or _obj_to_dict(response).get("user")
            )
            user_id = user.get("id")
            if not user_id:
                raise ValueError("Supabase did not return a user id")
            await self._upsert_profile(
                user_id,
                normalized_email,
                normalized_username,
                self.role,
                given_name,
                family_name,
            )
            await self._send_verification_email(normalized_email)
            return {
                "user_id": user_id,
                "email": normalized_email,
                "message": REGISTRATION_VERIFY_EMAIL_MESSAGE,
                "email_verification_required": True,
            }
        except HTTPException:
            raise
        except Exception as e:
            app_logger.error(f"Supabase registration error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_registration_error_detail(e),
            )

    async def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        try:
            email = await self._resolve_login_email(username)
            response = await self.supabase.run(
                self.anon_client.auth.sign_in_with_password,
                {"email": email, "password": password},
            )
            raw = _obj_to_dict(response)
            session = _obj_to_dict(
                raw.get("session") or getattr(response, "session", None)
            )
            user = _user_to_dict(raw.get("user") or getattr(response, "user", None))
            profile = await self._get_profile(user.get("id", ""), user.get("email"))
            if not profile and user.get("id"):
                metadata = user.get("metadata") or {}
                profile = await self._upsert_profile(
                    user["id"],
                    user.get("email") or email,
                    metadata.get("username") or username,
                    self.role,
                    metadata.get("given_name"),
                    metadata.get("family_name"),
                )
            user_info = self._to_user_info(user, profile)
            if not user_info.email_verified:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=EMAIL_NOT_VERIFIED_MESSAGE,
                )
            return {
                "access_token": session.get("access_token"),
                "refresh_token": session.get("refresh_token"),
                "id_token": session.get("access_token"),
                "token_type": "Bearer",
                "expires_in": int(session.get("expires_in") or 3600),
                "user": user_info.model_dump(),
            }
        except HTTPException:
            raise
        except Exception as e:
            app_logger.error(f"Supabase login error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=_login_error_detail(e),
            )

    async def get_user_info(self, access_token: str) -> UserInfo:
        try:
            payload = await self.verify_jwt_token(access_token)
            if payload.get("tiwmai_auth"):
                return self._to_user_info_from_local_payload(payload)
            user = getattr(self, "_verified_auth_users", {}).get(access_token)
            if not user:
                user = await self._fetch_auth_user(access_token)
            if not user.get("id"):
                user["id"] = payload.get("sub")
                user["email"] = payload.get("email")
            profile = await self._get_profile(user.get("id", ""), user.get("email"))
            metadata = user.get("metadata") or {}
            legacy_alias = _google_legacy_alias(metadata)
            legacy_profile = {}
            if not profile and legacy_alias:
                legacy_profile = await self._get_profile_by_legacy_alias(legacy_alias)
                profile = legacy_profile
            if not profile and user.get("id"):
                profile = await self._upsert_profile(
                    user["id"],
                    user.get("email") or "",
                    metadata.get("username") or legacy_alias or user.get("email"),
                    self.role,
                    metadata.get("given_name") or metadata.get("name"),
                    metadata.get("family_name"),
                )
            elif legacy_profile and user.get("id"):
                await self._upsert_profile(
                    user["id"],
                    user.get("email") or legacy_profile.get("email") or "",
                    legacy_profile.get("username") or legacy_alias or user.get("email"),
                    legacy_profile.get("role") or self.role,
                    legacy_profile.get("given_name") or metadata.get("given_name"),
                    legacy_profile.get("family_name") or metadata.get("family_name"),
                    {"student_id": legacy_profile.get("user_id") or legacy_alias},
                )
            return self._to_user_info(user, profile, payload)
        except HTTPException:
            raise
        except Exception as e:
            app_logger.error(f"Unable to load Supabase user info: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token"
            )

    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        try:
            response = None
            try:
                response = await self.supabase.run(
                    self.anon_client.auth.refresh_session,
                    refresh_token,
                )
            except Exception as refresh_error:
                app_logger.warning(
                    "Supabase refresh_session failed during token refresh: {}",
                    refresh_error,
                )
                response = None

            if response is None:
                rest_session = await self._refresh_session_via_rest(refresh_token)
                if rest_session:
                    response = rest_session

            raw = _obj_to_dict(response)
            session = _obj_to_dict(
                raw.get("session") or getattr(response, "session", None)
            )
            normalized_access_token = session.get("access_token") or raw.get(
                "access_token"
            )
            normalized_refresh_token = (
                session.get("refresh_token")
                or raw.get("refresh_token")
                or refresh_token
            )
            if not normalized_access_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token refresh failed",
                )

            user_info = await self.get_user_info(normalized_access_token)
            return self._build_local_session_payload(
                user_info,
                refresh_token=normalized_refresh_token,
            )
        except HTTPException:
            raise
        except Exception as e:
            app_logger.error(f"Supabase token refresh failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token refresh failed"
            )

    async def normalize_oauth_session(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        provider_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            response = None
            if refresh_token:
                try:
                    response = await self.supabase.run(
                        self.anon_client.auth.set_session,
                        access_token,
                        refresh_token,
                    )
                except Exception as set_session_error:
                    app_logger.warning(
                        "Supabase set_session failed during OAuth callback: {}",
                        set_session_error,
                    )
                    try:
                        response = await self.supabase.run(
                            self.anon_client.auth.refresh_session,
                            refresh_token,
                        )
                    except Exception as refresh_error:
                        app_logger.warning(
                            "Supabase refresh_session failed during OAuth callback: {}",
                            refresh_error,
                        )
                        response = None

            # SDK fallback: refresh token via REST endpoint.
            if response is None and refresh_token:
                rest_session = await self._refresh_session_via_rest(refresh_token)
                if rest_session:
                    response = rest_session

            raw = _obj_to_dict(response) if response is not None else {}
            session = _obj_to_dict(
                raw.get("session") or getattr(response, "session", None)
            )
            normalized_access_token = (
                session.get("access_token") or raw.get("access_token") or access_token
            )
            normalized_refresh_token = (
                session.get("refresh_token")
                or raw.get("refresh_token")
                or refresh_token
            )

            try:
                user_info = await self.get_user_info(normalized_access_token)
            except HTTPException as verify_error:
                if provider_token:
                    app_logger.warning(
                        "Supabase OAuth token verification failed during callback; "
                        "falling back to provider token local session: {}",
                        verify_error,
                    )
                    local_session = await self._create_local_google_session(
                        provider_token
                    )
                    if normalized_refresh_token:
                        local_session["refresh_token"] = normalized_refresh_token
                    return local_session
                raise

            return self._build_local_session_payload(
                user_info,
                refresh_token=normalized_refresh_token,
            )
        except HTTPException:
            raise
        except Exception as e:
            app_logger.error(f"Supabase OAuth session normalization failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="OAuth token is invalid or expired",
            )

    async def logout_user(self, access_token: str) -> bool:
        try:
            await self.supabase.run(
                self.client.auth.admin.sign_out, access_token, "global"
            )
        except Exception as e:
            app_logger.warning(f"Supabase sign out failed; treating as logged out: {e}")
        return True

    def get_oauth_authorization_url(
        self,
        state: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> str:
        if not self.settings.supabase_url:
            raise HTTPException(
                status_code=500, detail="Supabase URL is not configured"
            )
        normalized_provider = str(provider or "google").strip().lower()
        allowed_providers = {"google", "github", "azure", "gitlab", "discord"}
        if normalized_provider not in allowed_providers:
            raise HTTPException(status_code=400, detail="Unsupported OAuth provider")
        redirect_to = self._frontend_callback_redirect_url()
        params = {
            "provider": normalized_provider,
            "redirect_to": redirect_to,
        }
        if state:
            params["state"] = state
        return f"{self.settings.supabase_url}/auth/v1/authorize?{urlencode(params)}"

    async def exchange_code_for_tokens(self, authorization_code: str) -> Dict[str, Any]:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Supabase OAuth PKCE callback should be completed by the frontend client",
        )
