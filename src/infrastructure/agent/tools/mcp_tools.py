"""
Carga de tools MCP para el agente LangGraph.

Levanta el MCP Server como subproceso stdio y carga sus tools
usando langchain-mcp-adapters (MultiServerMCPClient).

El agente no llama directamente a los repositorios — solo
interactúa con las tools MCP, manteniendo el desacoplamiento
definido en la arquitectura DDD.

Diseño:
    Agente LangGraph
        └── load_mcp_tools()
                └── MultiServerMCPClient
                        └── mcp_server_fastmcp.py (subproceso stdio)
                                └── MCPDataServer
                                        ├── CSVHistoricalRepository
                                        └── CSVForecastRepository
"""

import sys
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Ruta al servidor MCP
MCP_SERVER_PATH = Path("src/infrastructure/data/mcp_server_fastmcp.py")


async def load_mcp_tools() -> list[BaseTool]:
    """
    Carga las tools del MCP Server como subproceso stdio.

    Levanta el servidor MCP en un subproceso y convierte sus
    tools en BaseTool de LangChain para uso en el agente ReAct.

    Returns:
        Lista de tools MCP listas para usar en el agente:
        - get_forecast_data
        - get_historical_data
        - get_model_metrics

    Raises:
        RuntimeError: Si el servidor MCP no puede inicializarse.

    Example:
        >>> import asyncio
        >>> tools = asyncio.run(load_mcp_tools())
        >>> print([t.name for t in tools])
        ['get_forecast_data', 'get_historical_data', 'get_model_metrics']
    """
    
    logger.info("Cargando tools MCP desde: %s", MCP_SERVER_PATH)    

    if not MCP_SERVER_PATH.exists():
        raise RuntimeError(
            f"MCP Server no encontrado: {MCP_SERVER_PATH}"
        )

    try:
        client = MultiServerMCPClient(
            {
                "construction-cost-forecasting": {
                    "command": sys.executable,
                    "args": [str(MCP_SERVER_PATH)],
                    "transport": "stdio",
                    "env": {
                        "PYTHONIOENCODING": "utf-8",
                        "PYTHONUTF8": "1",
                    },
                }
            }
        )

        tools = await client.get_tools()
        logger.info(
            "Tools MCP cargadas exitosamente: %s",
            [t.name for t in tools],
        )
        return tools

    except* Exception as eg:
        for exc in eg.exceptions:
            logger.error(
                "Sub-excepción MCP: %s — %s",
                type(exc).__name__,
                str(exc),
            )
        raise RuntimeError(
            f"Error al cargar tools MCP: {eg}"
        ) from eg