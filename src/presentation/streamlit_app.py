"""
Aplicación Streamlit — UI híbrida para el sistema de forecast
de costos de equipos de construcción.

Composition root de la aplicación — único lugar donde se
instancian los adaptadores concretos y se construye la sesión
del agente. Expone dos tabs:

- Dashboard: visualización de histórico y forecast con IC
- Chat: agente conversacional con fuentes citadas

Principio DDD respetado: los repositorios concretos se instancian
aquí y se pasan por parámetro — los paneles solo importan
interfaces de dominio.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.application.run_agent_query import (
    AgentSession,
    ToolCallInfo,
    ask_agent,
    build_agent_session,
)
from src.infrastructure.data.csv_repository import (
    CSVForecastRepository,
    CSVHistoricalRepository,
)
from src.utils.logger import get_logger

# ── Configuración ─────────────────────────────────────────
load_dotenv()
logger = get_logger(__name__)

DATA_PATH = Path("data/processed")

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Construction Cost Forecasting",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cache de recursos  ────────────────────────────

@st.cache_resource
def get_historical_repo() -> CSVHistoricalRepository:
    """Repositorio histórico — inicializado una sola vez."""
    return CSVHistoricalRepository(DATA_PATH)


@st.cache_resource
def get_forecast_repo() -> CSVForecastRepository:
    """Repositorio de forecast — inicializado una sola vez."""
    return CSVForecastRepository(DATA_PATH)


@st.cache_resource
def get_agent_session() -> AgentSession:
    """
    Sesión del agente — construida una sola vez por proceso.
    Incluye LLM, tools MCP, web search y memoria de sesión.
    """
    logger.info("Construyendo sesión del agente (cache_resource)")
    return build_agent_session()


# ── Helpers ───────────────────────────────────────────────

def load_historical_data(
    equipment_id: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Carga datos históricos del repositorio."""
    repo = get_historical_repo()
    try:
        df = repo.get_historical_prices(
            start_date=start_date,
            end_date=end_date,
        )
        col = f"Price_{equipment_id}"
        return df[["Date", col]].rename(columns={col: "Price"})
    except Exception as e:
        logger.error("Error cargando histórico: %s", str(e))
        return pd.DataFrame()


def load_forecast_data(equipment_id: str) -> pd.DataFrame:
    """Carga datos de forecast del repositorio."""
    repo = get_forecast_repo()
    try:
        return repo.get_forecast_summary(equipment_id=equipment_id)
    except Exception as e:
        logger.error("Error cargando forecast: %s", str(e))
        return pd.DataFrame()


def dashboard_context_summary(
    equipment_id: str,
    date_range: tuple[date, date],
) -> str:
    """
    Genera un resumen del contexto del dashboard para el agente.
    Sincroniza los filtros del dashboard con el chat.
    """
    return (
        f"El usuario está viendo el dashboard de {equipment_id} "
        f"con datos desde {date_range[0]} hasta {date_range[1]}. "
        f"Tiene acceso al forecast de este equipo y puede preguntar "
        f"sobre proyecciones, históricos o contexto de mercado."
    )


# ── Panels ────────────────────────────────────────────────

def render_dashboard_panel(
    equipment_id: str,
    date_range: tuple[date, date],
) -> None:
    """Renderiza el panel de dashboard con histórico y forecast."""

    st.subheader(f"📊 Histórico y Forecast — {equipment_id}")

    # Cargar datos
    hist_df = load_historical_data(
        equipment_id=equipment_id,
        start_date=date_range[0],
        end_date=date_range[1],
    )
    forecast_df = load_forecast_data(equipment_id=equipment_id)

    if hist_df.empty:
        st.error("No se pudieron cargar los datos históricos.")
        return

    # ── Métricas rápidas ──────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        last_price = hist_df["Price"].iloc[-1]
        st.metric("Último precio histórico", f"{last_price:.2f}")

    with col2:
        avg_price = hist_df["Price"].mean()
        st.metric("Precio promedio", f"{avg_price:.2f}")

    if not forecast_df.empty:
        with col3:
            first_forecast = forecast_df["precio_medio"].iloc[0]
            delta = first_forecast - last_price
            st.metric(
                "Forecast próximo mes",
                f"{first_forecast:.2f}",
                delta=f"{delta:+.2f}",
            )
        with col4:
            st.metric(
                "Horizonte confiable",
                "~1 mes",
                help="IC < 25% del precio proyectado",
            )

    # ── Gráfica principal ─────────────────────────────────
    fig = go.Figure()

    # Serie histórica
    fig.add_trace(go.Scatter(
        x=hist_df["Date"],
        y=hist_df["Price"],
        name="Histórico",
        line=dict(color="#7B2D8B", width=1),
        mode="lines",
    ))

    # Forecast con IC
    if not forecast_df.empty:
        # Convertir mes a fecha
        forecast_df["fecha"] = pd.to_datetime(
            forecast_df["mes"].astype(str)
        )

        # Banda IC 95%
        fig.add_trace(go.Scatter(
            x=pd.concat([
                forecast_df["fecha"],
                forecast_df["fecha"].iloc[::-1]
            ]),
            y=pd.concat([
                forecast_df["ic_upper"],
                forecast_df["ic_lower"].iloc[::-1]
            ]),
            fill="toself",
            fillcolor="rgba(33, 150, 243, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="IC 95%",
            showlegend=True,
        ))

        # Línea central del forecast
        fig.add_trace(go.Scatter(
            x=forecast_df["fecha"],
            y=forecast_df["precio_medio"],
            name="Forecast central",
            line=dict(color="#2196F3", width=2, dash="dash"),
            mode="lines+markers",
        ))

    fig.update_layout(
        title=f"Evolución de precios — {equipment_id}",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Tabla de forecast ─────────────────────────────────
    if not forecast_df.empty:
        st.subheader("📅 Resumen mensual del forecast")

        display_df = forecast_df[[
            "mes", "precio_medio", "ic_lower",
            "ic_upper", "amplitud_95"
        ]].copy()

        display_df.columns = [
            "Mes", "Precio central",
            "IC lower (95%)", "IC upper (95%)", "Amplitud IC"
        ]

        display_df["% del precio"] = (
            display_df["Amplitud IC"] /
            display_df["Precio central"] * 100
        ).round(1).astype(str) + "%"

        st.dataframe(
            display_df.style.format({
                "Precio central": "{:.2f}",
                "IC lower (95%)": "{:.2f}",
                "IC upper (95%)": "{:.2f}",
                "Amplitud IC": "{:.2f}",
            }),
            use_container_width=True,
        )

        st.info(
            "⚠️ **Horizonte confiable: 1 mes** — "
            "A partir del mes 2 la amplitud del IC supera el 25% "
            "del precio proyectado. Se recomienda actualizar el "
            "forecast mensualmente con nuevos datos de las "
            "materias primas."
        )


def render_chat_panel(
    equipment_id: str,
    date_range: tuple[date, date],
) -> None:
    """Renderiza el panel de chat con el agente."""

    st.subheader("🤖 Agente de IA — Consultas sobre costos")

    # Inicializar historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "thread_id" not in st.session_state:
        import uuid
        st.session_state.thread_id = str(uuid.uuid4())

    # Contexto del dashboard para el agente
    context = dashboard_context_summary(equipment_id, date_range)

    # Mostrar historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("tools_used"):
                with st.expander("🔍 Fuentes consultadas"):
                    for tool in message["tools_used"]:
                        st.markdown(f"**{tool.tool_name}**")
                        st.code(tool.content_preview, language="json")

    # Input del usuario
    if prompt := st.chat_input(
        "Pregunta sobre costos, forecast o mercado..."
    ):
        # Agregar mensaje del usuario
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Respuesta del agente
        with st.chat_message("assistant"):
            with st.spinner("Consultando..."):
                try:
                    session = get_agent_session()
                    full_question = (
                        f"Contexto del dashboard: {context}\n\n"
                        f"Pregunta: {prompt}"
                    )
                    response, tools_used = ask_agent(
                        session=session,
                        question=full_question,
                        thread_id=st.session_state.thread_id,
                    )

                    st.markdown(response)

                    if tools_used:
                        with st.expander("🔍 Fuentes consultadas"):
                            for tool in tools_used:
                                st.markdown(f"**{tool.tool_name}**")
                                st.code(
                                    tool.content_preview,
                                    language="json",
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "tools_used": tools_used,
                    })

                except Exception as e:
                    error_msg = f"Error al consultar el agente: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)

    # Botón para limpiar historial
    if st.session_state.messages:
        if st.button("🗑️ Limpiar conversación"):
            st.session_state.messages = []
            import uuid
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()


# ── Main ──────────────────────────────────────────────────

def main() -> None:
    """Función principal de la aplicación Streamlit."""

    # Header
    st.title("🏗️ Construction Cost Forecasting")
    st.markdown(
        "Sistema de estimación y proyección de costos de equipos "
        "de construcción basado en precios de materias primas."
    )

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuración")

        equipment_id = st.selectbox(
            "Equipo",
            options=["Equipo1", "Equipo2"],
            help="Equipo1: predictor Price_Y | Equipo2: predictor Price_Z",
        )

        st.divider()

        st.subheader("📅 Rango histórico")
        start_date = st.date_input(
            "Desde",
            value=date(2022, 1, 1),
            min_value=date(2010, 1, 4),
            max_value=date(2023, 8, 31),
        )
        end_date = st.date_input(
            "Hasta",
            value=date(2023, 8, 31),
            min_value=date(2010, 1, 4),
            max_value=date(2023, 8, 31),
        )

        st.divider()

        st.subheader("ℹ️ Información del modelo")
        if equipment_id == "Equipo1":
            st.markdown("""
            - **Modelo:** ARIMAX(1,1,2)
            - **Predictor:** Price_Y
            - **MAPE:** 1.55%
            - **R²:** 0.928
            """)
        else:
            st.markdown("""
            - **Modelo:** ARIMAX(0,1,1)
            - **Predictor:** Price_Z
            - **MAPE:** 1.75%
            - **R²:** 0.829
            """)

        st.divider()
        st.caption("🔗 [LangFuse](http://localhost:3000) | "
                   "🔗 [LiteLLM](http://localhost:4000)")

    # ── Tabs ─────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Chat con Agente"])

    date_range = (start_date, end_date)

    with tab1:
        render_dashboard_panel(
            equipment_id=equipment_id,
            date_range=date_range,
        )

    with tab2:
        render_chat_panel(
            equipment_id=equipment_id,
            date_range=date_range,
        )


if __name__ == "__main__":
    main()