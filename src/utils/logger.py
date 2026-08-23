"""
Módulo de configuración centralizada de logging para el proyecto
construction-cost-forecasting.

Provee un logger configurado con handlers para consola y archivo,
con rotación automática para evitar archivos de log excesivamente grandes.

Uso:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Mensaje informativo")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ── Constantes ────────────────────────────────────────────
LOG_DIR: Path = Path(__file__).resolve().parents[2] / "logs"
LOG_FILE: Path = LOG_DIR / "pipeline.log"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT: int = 3


def _build_console_handler() -> logging.StreamHandler:
    """
    Construye un handler para salida por consola.

    Returns:
        StreamHandler configurado con formato estándar.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    return handler


def _build_file_handler() -> RotatingFileHandler:
    """
    Construye un handler para escritura en archivo con rotación automática.

    Returns:
        RotatingFileHandler configurado con rotación de 5MB y 3 backups.

    Raises:
        OSError: Si no se puede crear el directorio de logs.
    """
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
    Obtiene un logger configurado con handlers de consola y archivo.

    Evita agregar handlers duplicados si el logger ya fue inicializado,
    lo que puede ocurrir en entornos como Jupyter o al importar módulos
    múltiples veces.

    Args:
        name: Nombre del logger — usar __name__ para trazabilidad por módulo.
        level: Nivel mínimo de logging. Por defecto DEBUG.

    Returns:
        Logger configurado y listo para usar.

    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Pipeline iniciado")
        >>> logger.error("Error al cargar datos: %s", str(e))
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.addHandler(_build_console_handler())
    logger.addHandler(_build_file_handler())
    logger.propagate = False

    return logger