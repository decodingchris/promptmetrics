import logging
import sys
from pathlib import Path

def setup_logger(log_dir: Path, filename: str):
    """Configures a logger with detailed file output and clean console output."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = log_dir / filename

    logger = logging.getLogger("promptmetrics")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)

    absolute_log_path = log_filepath.resolve()
    
    try:
        display_path = absolute_log_path.relative_to(Path.cwd())
    except ValueError:
        display_path = absolute_log_path
        
    logger.info(f"Logger configured. Saving detailed logs to: {display_path}")