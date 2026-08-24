"""
Configuración centralizada de la aplicación.

Lee todas las variables de entorno necesarias para el agente,
el gateway LiteLLM y LangFuse. Usa un dataclass frozen para
garantizar inmutabilidad en tiempo de ejecución.

No llama a load_dotenv() — se asume que el composition root
(streamlit_app.py o main.py) ya lo hizo antes.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """
    Configuración completa de la aplicación.

    Attributes:
        # Paths
        data_path: Ruta al directorio de datos procesados.
        models_path: Ruta al directorio de modelos entrenados.

        # LiteLLM Gateway
        litellm_base_url: URL base del gateway LiteLLM.
        litellm_master_key: Master key del gateway.
        litellm_model: Modelo a usar via gateway.
        litellm_model_name: Nombre del modelo para LangFuse (para trazabilidad).

        # LangFuse
        langfuse_public_key: Public key de LangFuse.
        langfuse_secret_key: Secret key de LangFuse.
        langfuse_host: URL del servidor LangFuse.

        # Agente
        agent_max_iterations: Máximo de iteraciones del agente ReAct.
        agent_temperature: Temperatura del LLM.
        tool_content_preview_length: Longitud del preview de tools.

        # App
        app_version: Versión de la aplicación.
        app_env: Ambiente (development, production).
    """

    # Paths
    data_path: Path
    models_path: Path

    # LiteLLM Gateway
    litellm_base_url: str
    litellm_master_key: str
    litellm_model: str
    litellm_model_name: str

    # LangFuse
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str

    # Agente
    agent_max_iterations: int
    agent_temperature: float
    tool_content_preview_length: int

    # App
    app_version: str
    app_env: str


def get_config() -> AppConfig:
    """
    Construye y retorna la configuración de la aplicación
    leyendo las variables de entorno.

    Cada llamada relee os.environ para que tests con
    monkeypatch.setenv() vean los cambios sin reimportar.

    Returns:
        AppConfig con todos los valores de configuración.

    Raises:
        ValueError: Si alguna variable de entorno crítica
                    no está definida.

    Example:
        >>> from src.infrastructure.config import get_config
        >>> config = get_config()
        >>> print(config.litellm_base_url)
        http://localhost:4000
    """
    
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY no está definida en las variables de entorno."
        )

    litellm_master_key = os.environ.get("LITELLM_MASTER_KEY")
    if not litellm_master_key:
        raise ValueError(
            "LITELLM_MASTER_KEY no está definida en las variables de entorno."
        )

    return AppConfig(        
        data_path=Path(
            os.environ.get("DATA_PROCESSED_PATH", "data/processed")
        ),
        models_path=Path(
            os.environ.get("MODELS_PATH", "models")
        ),

        # LiteLLM Gateway
        litellm_base_url=os.environ.get(
            "LITELLM_BASE_URL", "http://localhost:4000"
        ),
        litellm_master_key=litellm_master_key,
        litellm_model=os.environ.get(
            "OPENROUTER_MODEL",
            "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        ),
        litellm_model_name=os.environ.get(
            "LITELLM_MODEL_NAME", "nvidia-nemotron-ultra-free"
        ),

        # LangFuse
        langfuse_public_key=os.environ.get(
            "LANGFUSE_PUBLIC_KEY", ""
        ),
        langfuse_secret_key=os.environ.get(
            "LANGFUSE_SECRET_KEY", ""
        ),
        langfuse_host=os.environ.get(
            "LANGFUSE_HOST", "http://localhost:3000"
        ),

        # Agente
        agent_max_iterations=int(
            os.environ.get("AGENT_MAX_ITERATIONS", "25")
        ),
        agent_temperature=float(
            os.environ.get("AGENT_TEMPERATURE", "0.1")
        ),
        tool_content_preview_length=int(
            os.environ.get("TOOL_CONTENT_PREVIEW_LENGTH", "300")
        ),

        # App
        app_version=os.environ.get("APP_VERSION", "1.0.0"),
        app_env=os.environ.get("APP_ENV", "development"),
    )