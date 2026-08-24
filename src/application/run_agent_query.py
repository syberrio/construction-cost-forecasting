"""
Caso de uso: ejecutar una consulta al agente de IA.

Orquesta la construcción y uso del agente LangGraph ReAct:
- build_agent_session(): construye todo una vez 
  → cacheado en Streamlit via st.cache_resource
- ask_agent(): invoca el grafo ya construido 
  → llamado por cada mensaje del usuario
- run_agent_query(): compatibilidad (build + ask en una llamada)

Siguiendo SRP, este módulo solo orquesta — no implementa
la lógica del agente ni el acceso a datos.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.pregel import Pregel as CompiledGraph

from src.infrastructure.agent.graph import SYSTEM_PROMPT, build_agent_graph
from src.infrastructure.agent.observability.langfuse_handler import (
    build_langfuse_handler,
)
from src.infrastructure.agent.tools.mcp_tools import load_mcp_tools
from src.infrastructure.agent.tools.web_search_tool import build_web_search_tool
from src.infrastructure.config import AppConfig, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ToolCallInfo:
    """
    Información sobre una tool call ejecutada por el agente.

    Attributes:
        tool_name: Nombre de la tool ejecutada.
        content_preview: Preview del contenido retornado.
    """

    tool_name: str
    content_preview: str


@dataclass
class AgentSession:
    """
    Sesión del agente con todos sus componentes inicializados.

    Attributes:
        graph: Grafo LangGraph compilado con memoria.
        memory: MemorySaver para persistencia de sesión.
        langfuse_handler: Handler de observabilidad.
        config: Configuración de la aplicación.
    """

    graph: CompiledGraph
    memory: MemorySaver
    langfuse_handler: Any
    config: AppConfig


def build_agent_session() -> AgentSession:
    """
    Construye la sesión completa del agente — operación costosa.

    Inicializa en orden:
    1. Configuración de la aplicación
    2. LLM via LiteLLM Gateway
    3. Tools MCP (subproceso stdio)
    4. Tool de búsqueda web
    5. Grafo LangGraph ReAct con memoria
    6. Handler de LangFuse

    Esta función debe llamarse UNA SOLA VEZ por proceso y
    cachearse (en Streamlit via st.cache_resource).

    Returns:
        AgentSession con todos los componentes listos.

    Raises:
        RuntimeError: Si algún componente no puede inicializarse.

    Example:
        >>> session = build_agent_session()
        >>> response = ask_agent(session, "¿Cuánto cuesta el Equipo1?")
    """
    logger.info("Construyendo sesión del agente — operación costosa")

    config = get_config()

    # LLM via LiteLLM Gateway
    logger.info(
        "Inicializando LLM — gateway: %s, modelo: %s",
        config.litellm_base_url,
        config.litellm_model,
    )
    llm = ChatOpenAI(
        base_url=f"{config.litellm_base_url}/v1",
        api_key=config.litellm_master_key,
        model=config.litellm_model_name,
        temperature=config.agent_temperature,
        max_tokens=2000,
        model_kwargs={
            "extra_headers": {
                "X-LiteLLM-Model": config.litellm_model,
            }
        },
    )

    # Tools MCP (async → sync)
    logger.info("Cargando tools MCP")
    try:
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        mcp_tools = loop.run_until_complete(load_mcp_tools())
    except Exception as e:
        logger.error("Error al cargar tools MCP: %s", str(e))
        raise RuntimeError(f"Error al cargar tools MCP: {e}") from e

    # Tool de búsqueda web
    web_tool = build_web_search_tool()

    # Todas las tools juntas
    all_tools = mcp_tools + [web_tool]
    logger.info(
        "Tools disponibles: %s",
        [t.name for t in all_tools],
    )

    # Grafo ReAct con memoria de sesión
    graph, memory = build_agent_graph(llm=llm, tools=all_tools)

    # Handler de LangFuse
    langfuse_handler = build_langfuse_handler(config)

    logger.info("Sesión del agente construida exitosamente")

    return AgentSession(
        graph=graph,
        memory=memory,
        langfuse_handler=langfuse_handler,
        config=config,
    )


def ask_agent(
    session: AgentSession,
    question: str,
    thread_id: str = "default-session",
) -> tuple[str, list[ToolCallInfo]]:
    """
    Invoca el agente con una pregunta y retorna la respuesta.

    Usa el grafo ya construido en build_agent_session().
    Mantiene memoria de sesión via thread_id.

    Args:
        session: Sesión del agente construida con build_agent_session().
        question: Pregunta del usuario en lenguaje natural.
        thread_id: Identificador de la sesión de conversación.
                   Usar el mismo thread_id mantiene el contexto.

    Returns:
        Tupla con:
        - Respuesta del agente como string
        - Lista de ToolCallInfo con las tools usadas

    Raises:
        Exception: Si el agente falla al procesar la pregunta.

    Example:
        >>> session = build_agent_session()
        >>> response, tools_used = ask_agent(
        ...     session,
        ...     "¿Cuánto cuesta el Equipo1?",
        ...     thread_id="user-123"
        ... )
        >>> print(response)
    """
    logger.info(
        "Invocando agente — thread_id: %s, pregunta: %.50s...",
        thread_id,
        question,
    )

    # Mensajes de entrada con system prompt
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    # Configuración de la invocación
    invoke_config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [session.langfuse_handler],
        "recursion_limit": session.config.agent_max_iterations,
    }

    try:
        import nest_asyncio
        nest_asyncio.apply()

        async def _invoke():
            return await session.graph.ainvoke(
                {"messages": messages},
                config=invoke_config,
            )

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(_invoke())

    except Exception as e:
        logger.error("Error al invocar el agente: %s", str(e))
        raise

    # Extraer respuesta final
    final_message = result["messages"][-1]
    response = (
        final_message.content
        if hasattr(final_message, "content")
        else str(final_message)
    )

    # Extraer tool calls usadas
    tool_calls_info = _extract_tool_calls(
        messages=result["messages"],
        preview_length=session.config.tool_content_preview_length,
    )

    logger.info(
        "Agente respondió — tools usadas: %d, respuesta: %.50s...",
        len(tool_calls_info),
        response,
    )

    return response, tool_calls_info


def _extract_tool_calls(
    messages: list,
    preview_length: int = 300,
) -> list[ToolCallInfo]:
    """
    Extrae información de las tool calls del historial de mensajes.

    Args:
        messages: Lista de mensajes del resultado del agente.
        preview_length: Longitud máxima del preview del contenido.

    Returns:
        Lista de ToolCallInfo con nombre y preview de cada tool usada.
    """
    tool_calls = []

    for message in messages:
        if hasattr(message, "name") and message.name:
            content = (
                message.content
                if isinstance(message.content, str)
                else str(message.content)
            )
            tool_calls.append(
                ToolCallInfo(
                    tool_name=message.name,
                    content_preview=content[:preview_length],
                )
            )

    return tool_calls


def run_agent_query(
    question: str,
    thread_id: str = "default-session",
) -> tuple[str, list[ToolCallInfo]]:
    """
    Función de compatibilidad — construye sesión y consulta en una llamada.

    Útil para uso desde scripts o tests. Para uso en Streamlit,
    preferir build_agent_session() + ask_agent() por separado
    para evitar reconstruir el grafo en cada mensaje.

    Args:
        question: Pregunta del usuario.
        thread_id: Identificador de sesión.

    Returns:
        Tupla con respuesta y lista de tools usadas.

    Example:
        >>> response, tools = run_agent_query(
        ...     "¿Cuál es el forecast del Equipo2?"
        ... )
    """
    session = build_agent_session()
    return ask_agent(session, question, thread_id)