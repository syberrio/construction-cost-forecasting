"""
Tool de búsqueda web para el agente de IA.

Usa DuckDuckGo como motor de búsqueda — no requiere API key.
El paquete PyPI es 'ddgs' (renombrado desde duckduckgo-search),
pero la clase de LangChain sigue siendo DuckDuckGoSearchRun.

Cuándo usar esta tool:
- Buscar tendencias actuales de precios de commodities
- Obtener contexto económico del mercado de construcción
- Noticias recientes sobre materias primas industriales
- Complementar el forecast con información externa
"""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import BaseTool

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_web_search_tool() -> BaseTool:
    """
    Construye y retorna la tool de búsqueda web con DuckDuckGo.

    Returns:
        Tool de búsqueda web lista para usar en el agente.

    Example:
        >>> tool = build_web_search_tool()
        >>> result = tool.invoke("precios acero construcción 2024")
        >>> print(result)
    """
    logger.info("Inicializando tool de búsqueda web — DuckDuckGo")

    tool = DuckDuckGoSearchRun(
        name="web_search",
        description=(
            "Busca información actualizada en la web usando DuckDuckGo. "
            "Úsala para: tendencias de precios de commodities, "
            "noticias del sector construcción, contexto económico "
            "que complemente el análisis de forecast. "
            "Input: query de búsqueda en español o inglés. "
            "Output: resumen de los resultados más relevantes."
        ),
    )

    logger.info("Tool de búsqueda web inicializada exitosamente")
    return tool