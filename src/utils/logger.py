import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR: Path = Path(__file__).resolve().parents[2] / "logs"
LOG_FILE: Path = LOG_DIR / "pipeline.log"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
MAX_BYTES: int = 5 * 1024 * 1024
BACKUP_COUNT: int = 3


def _build_console_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    return handler


def _build_file_handler() -> RotatingFileHandler:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    return handler


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Obtiene un logger configurado.

    Si la variable de entorno MCP_STDIO_MODE=1 está activa,
    solo escribe a archivo (nunca a stdout) para no interferir
    con el protocolo MCP stdio.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    
    if os.environ.get("MCP_STDIO_MODE") == "1":
        logger.addHandler(_build_file_handler())
    else:
        logger.addHandler(_build_console_handler())
        logger.addHandler(_build_file_handler())

    logger.propagate = False
    return logger