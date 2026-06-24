"""Lightweight Supabase query instrumentation for tests and diagnostics."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_query_count: ContextVar[int] = ContextVar("supabase_query_count", default=0)


def reset_query_count() -> None:
    _query_count.set(0)


def increment_query_count() -> None:
    _query_count.set(get_query_count() + 1)


def get_query_count() -> int:
    return int(_query_count.get())


@contextmanager
def track_supabase_queries() -> Iterator[None]:
    token = _query_count.set(0)
    try:
        yield
    finally:
        _query_count.reset(token)
