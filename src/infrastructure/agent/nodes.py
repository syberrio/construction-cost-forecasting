"""
Nodos del grafo ReAct del agente de IA.

Define los tres nodos del ciclo ReAct:
- reason: invoca el LLM con tools bindeadas para razonar
- act: ejecuta las tools seleccionadas por el LLM
- observe: loggea el resultado (punto de extensión futuro)

El flujo es:
    reason → act → observe → reason (loop)
    reason → END (cuando el LLM no llama ninguna tool)
"""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_reason_node(llm_with_tools: Runnable) -> callable:
    """
    Construye el nodo 'reason' del grafo ReAct.

    El nodo reason invoca el LLM con las tools bindeadas.
    El LLM decide si usar una tool o responder directamente.

    Args:
        llm_with_tools: LLM con tools bindeadas via bind_tools().

    Returns:
        Función nodo compatible con StateGraph de LangGraph.

    Example:
        >>> reason = build_reason_node(llm_with_tools)
        >>> # Se registra en el grafo como nodo
        >>> graph.add_node("reason", reason)
    """

    def reason(state: MessagesState) -> dict[str, Any]:
        """
        Nodo de razonamiento — invoca el LLM con el estado actual.

        Args:
            state: Estado actual del grafo con historial de mensajes.

        Returns:
            Diccionario con los nuevos mensajes generados por el LLM.
        """
        logger.debug(
            "Nodo reason — %d mensajes en estado",
            len(state["messages"]),
        )

        response: AIMessage = llm_with_tools.invoke(state["messages"])

        logger.debug(
            "Nodo reason — respuesta generada, tool_calls: %d",
            len(response.tool_calls) if hasattr(response, "tool_calls")
            and response.tool_calls else 0,
        )

        return {"messages": [response]}

    return reason


def build_observe_node() -> callable:
    """
    Construye el nodo 'observe' del grafo ReAct.

    Nodo liviano que loggea el resultado de las tools
    y sirve como punto de extensión para lógica futura
    (ej: validación de resultados, enriquecimiento de contexto).

    Returns:
        Función nodo compatible con StateGraph de LangGraph.

    Example:
        >>> observe = build_observe_node()
        >>> graph.add_node("observe", observe)
    """

    def observe(state: MessagesState) -> dict[str, Any]:
        """
        Nodo de observación — loggea el resultado de las tools.

        Args:
            state: Estado actual del grafo con resultados de tools.

        Returns:
            Diccionario vacío — no modifica el estado.
        """
        messages: list[BaseMessage] = state["messages"]
        last_message = messages[-1] if messages else None

        if last_message:
            logger.debug(
                "Nodo observe — último mensaje tipo: %s",
                type(last_message).__name__,
            )

        return {}

    return observe


def build_act_node(tools: list) -> ToolNode:
    """
    Construye el nodo 'act' del grafo ReAct.

    Usa ToolNode de langgraph.prebuilt para ejecutar
    las tools seleccionadas por el LLM en el nodo reason.

    Args:
        tools: Lista de tools disponibles para el agente.

    Returns:
        ToolNode configurado con las tools del agente.

    Example:
        >>> act = build_act_node(tools)
        >>> graph.add_node("act", act)
    """
    logger.info(
        "Construyendo nodo act con %d tools: %s",
        len(tools),
        [t.name for t in tools],
    )
    return ToolNode(tools)