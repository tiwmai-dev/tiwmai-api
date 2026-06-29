"""Shared Supabase client helpers for backend services."""

import asyncio
from functools import lru_cache
from typing import Any, Callable, Optional

from fastapi import HTTPException, status
from supabase import Client, create_client

from app.core.config import get_settings
from app.core.logging import app_logger
from app.services.supabase_metrics import increment_query_count


class SupabaseService:
    """Owns anon and service-role Supabase clients.

    The backend uses the service-role client for server-side data access. Frontend
    clients should only ever receive the anon key from their own environment.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._service_client: Optional[Client] = None
        self._anon_client: Optional[Client] = None
        # Keep sync supabase-py calls off the event loop while avoiding the old
        # single global queue. Bounded concurrency prevents request bursts from
        # creating unbounded worker threads against the same PostgREST backend.
        self._run_semaphore = asyncio.Semaphore(8)

    @staticmethod
    def _is_transient_connection_error(error: Exception) -> bool:
        message = str(error or "").strip().lower()
        transient_signatures = (
            "server disconnected",
            "connection reset",
            "connection aborted",
            "connection refused",
            "connectionterminated",
            "broken pipe",
            "read timeout",
            "timed out",
            "remoteprotocolerror",
            "protocol_error",
            "resource temporarily unavailable",
            "temporarily unavailable",
            "errno 11",
            "statement timeout",
            "canceling statement",
            "57014",
        )
        if any(signature in message for signature in transient_signatures):
            return True
        error_code = getattr(error, "code", None)
        if error_code is not None and str(error_code).strip() == "57014":
            return True
        return False

    def _reset_clients(self) -> None:
        self._service_client = None
        self._anon_client = None

    def _require_config(self) -> None:
        if not self.settings.supabase_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SUPABASE_URL is not configured",
            )
        if not self.settings.supabase_service_role_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SUPABASE_SERVICE_ROLE_KEY is not configured",
            )

    @property
    def client(self) -> Client:
        """Service-role client for trusted backend operations."""
        if self._service_client is None:
            self._require_config()
            self._service_client = create_client(
                self.settings.supabase_url,
                self.settings.supabase_service_role_key,
            )
            app_logger.info("Supabase service-role client initialized")
        return self._service_client

    @property
    def anon_client(self) -> Client:
        """Anon client for auth flows that should mirror browser behavior."""
        if self._anon_client is None:
            if not self.settings.supabase_url or not self.settings.supabase_anon_key:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Supabase anon client is not configured",
                )
            self._anon_client = create_client(
                self.settings.supabase_url,
                self.settings.supabase_anon_key,
            )
            app_logger.info("Supabase anon client initialized")
        return self._anon_client

    async def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run sync supabase-py calls without blocking the event loop."""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                async with self._run_semaphore:
                    increment_query_count()
                    return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as exc:
                is_retryable = self._is_transient_connection_error(exc)
                if attempt >= max_attempts or not is_retryable:
                    raise
                app_logger.warning(
                    "Transient Supabase error (%s). Retrying attempt %s/%s.",
                    exc,
                    attempt + 1,
                    max_attempts,
                )
                self._reset_clients()
                await asyncio.sleep(0.2 * attempt)


@lru_cache()
def get_supabase_service() -> SupabaseService:
    return SupabaseService()
