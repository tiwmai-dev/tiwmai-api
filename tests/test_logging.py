from pathlib import Path

from app.core.config import get_settings
from app.core.logging import setup_logging


def test_setup_logging_skips_file_sink_on_vercel(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("DEBUG", "false")
    get_settings.cache_clear()

    setup_logging()

    assert not Path("logs").exists()

    get_settings.cache_clear()
