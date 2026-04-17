import sys
from pathlib import Path
from loguru import logger

from app.core.config import settings


def _exclude_content_log(record: dict) -> bool:
    """Sink filter: reject records bound with content_log=True so they only
    land in the dedicated content log file, not the main app log."""
    return record["extra"].get("content_log") is not True


def configure_logger():
    """Initialize the logger with console and optional file output."""
    logger.remove()

    logger.add(
        sys.stdout,
        level=settings.log_level.upper(),
        colorize=True,
        filter=_exclude_content_log,
    )

    if settings.log_to_file:
        log_file = Path(settings.log_file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            settings.log_file_path,
            level=settings.log_level.upper(),
            rotation=settings.log_file_rotation,
            retention=settings.log_file_retention,
            compression=settings.log_file_compression,
            enqueue=True,
            encoding="utf-8",
            filter=_exclude_content_log,
        )

    from app.utils.content_logger import configure_content_logger

    configure_content_logger()
