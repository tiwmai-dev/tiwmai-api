"""Helpers for registering canonical API routes alongside legacy aliases."""

from typing import Callable, Iterable, Sequence, Union

from fastapi import APIRouter

RouteHandler = Callable[..., object]
RouteSpec = tuple[str, str, RouteHandler]


def register_route_aliases(
    router: APIRouter,
    prefix: str,
    routes: Sequence[RouteSpec],
) -> None:
    """Mount existing handlers under a canonical prefix.

    Example: ``/courses`` becomes ``/tutor/courses`` when prefix is ``/tutor``.
    """
    normalized_prefix = prefix.rstrip("/")
    for method, path, handler in routes:
        alias_path = f"{normalized_prefix}{path}"
        router.add_api_route(
            alias_path,
            handler,
            methods=[method.upper()],
            include_in_schema=True,
        )


def register_auth_aliases(
    router: APIRouter,
    prefix: str,
    auth_router: APIRouter,
) -> None:
    """Duplicate auth router routes under ``{prefix}/auth/*``."""
    normalized_prefix = prefix.rstrip("/")
    for route in auth_router.routes:
        methods = sorted(getattr(route, "methods", None) or [])
        if not methods:
            continue
        legacy_path = getattr(route, "path", "")
        if not legacy_path.startswith("/auth"):
            continue
        suffix = legacy_path[len("/auth") :]
        alias_path = f"{normalized_prefix}/auth{suffix}"
        router.add_api_route(
            alias_path,
            route.endpoint,
            methods=methods,
            include_in_schema=True,
        )
