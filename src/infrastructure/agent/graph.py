"""
Grafo ReAct del agente de IA para forecast de costos de construcción.

Construye el grafo LangGraph con el ciclo ReAct:
    reason → act → observe → reason (loop)
    reason → END (cuando el LLM no necesita tools)

Incluye memoria de sesión via MemorySaver para mantener
el contexto de la conversación dentro de una misma sesión.

El agente puede:
- Consultar forecasts de costos de equipos
- Consultar datos históricos de precios
- Consultar métricas de los modelos
- Buscar contexto externo de mercado en la web
- Combinar toda esa información para responder preguntas
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import tools_condition

from src.infrastructure.agent.nodes import (
    build_act_node,
    build_observe_node,
    build_reason_node,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# System prompt del agente
SYSTEM_PROMPT = """Eres un asistente experto en análisis de costos de equipos
de construcción. Tienes acceso a un sistema de forecast que proyecta los precios
de dos equipos críticos (Equipo1 y Equipo2) basado en materias primas.

## Contexto del análisis

**Variables seleccionadas (basado en evidencia estadística):**
- Price_Y → Equipo1 (correlación Pearson 0.997, cointegración p=0.013)
- Price_Z → Equipo2 (correlación Pearson 0.983, cointegración p=0.008)
- Price_X → DESCARTADA (ruido, sin cointegración con ningún equipo)

**Modelos utilizados:**
- Equipo1: ARIMAX(1,1,2) con Price_Y — MAPE 1.55%, R²=0.928
- Equipo2: ARIMAX(0,1,1) con Price_Z — MAPE 1.75%, R²=0.829

**Forecast más reciente (Monte Carlo):**
- Equipo1 sep 2023: precio central ~451, IC 95% [414, 488]
- Equipo2 sep 2023: precio central ~933, IC 95% [866, 1000]
- Horizonte confiable: 1 mes para ambos equipos
- La incertidumbre se propaga correctamente via simulación Monte Carlo

**Período atípico:** abril 2021 → mayo 2022 (disrupciones post-pandemia)

## Tus capacidades

1. **get_forecast_data(equipment_id):** obtén proyecciones futuras de precios
2. **get_historical_data(start_date, end_date):** consulta precios históricos
3. **get_model_metrics():** obtén métricas de evaluación de los modelos
4. **web_search(query):** busca contexto externo de mercado

## Instrucciones

- Siempre basa tus respuestas en datos reales de las tools
- Para preguntas de forecast: usa get_forecast_data UNA SOLA VEZ
- Para preguntas de mercado: usa web_search MÁXIMO 2 veces con queries distintas
- Para preguntas históricas: usa get_historical_data UNA SOLA VEZ
- Combina el forecast con contexto externo cuando sea relevante
- Menciona siempre el horizonte confiable (1 mes) al hablar de proyecciones
- Si el usuario pregunta por más de 1 mes, explica la limitación de incertidumbre
- NO repitas búsquedas con la misma query — si ya buscaste algo, usa ese resultado
- Al usar web_search, menciona explícitamente las fechas
  de las fuentes encontradas para dar contexto temporal
- Aclara siempre que el histórico del modelo llega hasta
  agosto 2023 pero el contexto web es información actual
- Si encuentras URLs en los resultados, cítalas al final
  de la respuesta como referencias
- Después de obtener resultados de tools, genera SIEMPRE una respuesta final
- Responde en español
- Sé conciso pero completo — incluye números concretos cuando estén disponibles
"""


def build_agent_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
) -> tuple[StateGraph, MemorySaver]:
    """
    Construye el grafo ReAct del agente con memoria de sesión.

    Args:
        llm: Modelo de lenguaje a usar como cerebro del agente.
        tools: Lista de tools disponibles (MCP + web search).

    Returns:
        Tupla con:
        - Grafo compilado listo para invocar
        - MemorySaver para acceso externo si se necesita

    Example:
        >>> graph, memory = build_agent_graph(llm, tools)
        >>> config = {"configurable": {"thread_id": "session-1"}}
        >>> result = graph.invoke(
        ...     {"messages": [("human", "¿Cuánto cuesta el Equipo1?")]},
        ...     config=config
        ... )
    """
    logger.info(
        "Construyendo grafo ReAct — %d tools disponibles: %s",
        len(tools),
        [t.name for t in tools],
    )

    # Bindear tools al LLM
    llm_with_tools = llm.bind_tools(tools)

    # Construir nodos
    reason_node = build_reason_node(llm_with_tools)
    act_node = build_act_node(tools)
    observe_node = build_observe_node()

    # Construir grafo
    builder = StateGraph(MessagesState)

    # Agregar nodos
    builder.add_node("reason", reason_node)
    builder.add_node("act", act_node)
    builder.add_node("observe", observe_node)

    # Punto de entrada
    builder.set_entry_point("reason")

    # Bordes condicionales
    # reason → act (si hay tool calls) | END (si no hay tool calls)
    builder.add_conditional_edges(
        "reason",
        tools_condition,
        {
            "tools": "act",
            END: END,
        },
    )

    # Bordes fijos
    builder.add_edge("act", "observe")
    builder.add_edge("observe", "reason")

    # Memoria de sesión — mantiene contexto dentro de una sesión
    memory = MemorySaver()

    # Compilar grafo con checkpointer
    graph = builder.compile(checkpointer=memory)

    logger.info("Grafo ReAct compilado exitosamente con memoria de sesión")

    return graph, memory