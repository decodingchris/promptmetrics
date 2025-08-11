import logging
from pathlib import Path

from promptmetrics.logging_utils import setup_logger


def test_setup_logger_creates_file_and_handlers(tmp_path):
    log_dir = tmp_path / "logs"
    setup_logger(log_dir, "app.log")

    # Check file created
    log_file = log_dir / "app.log"
    assert log_file.exists()

    # Check logger configuration
    logger = logging.getLogger("promptmetrics")
    # Should have exactly 2 handlers: file and console
    assert len(logger.handlers) == 2

    # Ensure log content includes the init message
    content = log_file.read_text(encoding="utf-8")
    assert "Logger configured. Saving detailed logs to:" in content

    # Subsequent call should clear previous handlers and set again (no duplicates)
    setup_logger(log_dir, "app2.log")
    logger2 = logging.getLogger("promptmetrics")
    assert len(logger2.handlers) == 2