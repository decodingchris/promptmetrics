"""Logging helpers for PromptMetrics."""

import logging
import sys
from pathlib import Path
from typing import List


def setup_logger(log_dir: Path, filename: str):
    """Configure the 'promptmetrics' logger with file + console handlers.

    - File handler (INFO): detailed logs for post-mortem analysis.
    - Console handler (WARNING): concise feedback for CLI users.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = log_dir / filename

    logger = logging.getLogger("promptmetrics")
    logger.setLevel(logging.INFO)

    # Close and remove any existing handlers to avoid file descriptor leaks.
    if logger.hasHandlers():
        # Work on a copy since we'll mutate the list.
        existing_handlers: List[logging.Handler] = logger.handlers[:]
        for h in existing_handlers:
            try:
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)

    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)

    absolute_log_path = log_filepath.resolve()

    try:
        display_path = absolute_log_path.relative_to(Path.cwd())
    except ValueError:
        display_path = absolute_log_path

    logger.info(f"Logger configured. Saving detailed logs to: {display_path}")
