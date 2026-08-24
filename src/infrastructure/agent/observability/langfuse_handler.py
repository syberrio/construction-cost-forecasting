"""
Handler de observabilidad para LangFuse.

Configura el callback de LangFuse para trazar todas las llamadas
al LLM y ejecuciones de tools del agente LangGraph.

Usa langfuse.langchain.CallbackHandler (API actual del SDK 4.x)
— NO langfuse.callback que es la API vieja documentada en tutoriales
desactualizados.

Las credenciales se resuelven automáticamente desde variables
de entorno (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST)
sin pasarlas explícitas al construir el handler.
"""
import os
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from src.infrastructure.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_langfuse_handler(config: AppConfig) -> CallbackHandler:
    """
    Construye el handler de LangFuse para trazabilidad.

    Args:
        config: Configuración de la aplicación.

    Returns:
        CallbackHandler configurado.
    """
    logger.info(
        "Inicializando LangFuse handler — host: %s",
        config.langfuse_host,
    )

    
    os.environ["LANGFUSE_PUBLIC_KEY"] = config.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = config.langfuse_secret_key
    os.environ["LANGFUSE_HOST"] = config.langfuse_host
    os.environ["LANGFUSE_MODEL_NAME"] = config.litellm_model    
    os.environ["OTEL_SERVICE_NAME"] = "agent-cost-forecasting"
    os.environ["LANGFUSE_ENVIRONMENT"] = config.app_env
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = (
        f"application=construction-cost-forecasting,"
        f"version={config.app_version},"
        f"environment={config.app_env}"
    )

    # Inicializar cliente global primero
    client = Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host,
    )

    # Verificar conexión
    auth = client.auth_check()
    logger.info("LangFuse auth check: %s", auth)

    # Construir handler — usa el cliente global inicializado
    handler = CallbackHandler(
        public_key=config.langfuse_public_key,
    )

    logger.info("LangFuse handler inicializado exitosamente")
    return handler