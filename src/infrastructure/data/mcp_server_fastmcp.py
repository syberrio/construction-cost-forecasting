import os
import sys


os.environ["MCP_STDIO_MODE"] = "1"

from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from src.infrastructure.data.mcp_server import MCPDataServer
from src.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

DATA_PATH = Path("data/processed")

mcp = FastMCP(
    name="construction-cost-forecasting",
    instructions=(
        "Servidor de datos para el sistema de forecast de costos "
        "de equipos de construcción. "
        "Usa get_forecast_data para obtener proyecciones futuras de precios. "
        "Usa get_historical_data para consultar precios históricos. "
        "Usa get_model_metrics para obtener métricas de evaluación de modelos. "
        "IMPORTANTE: Si get_forecast_data falla por falta de modelo, "
        "no inventes predicciones — informa al usuario y ofrece "
        "datos históricos como alternativa."
    ),
)

server = MCPDataServer(data_path=DATA_PATH)


@mcp.tool()
def get_forecast_data(equipment_id: str) -> dict:
    """
    Obtiene el forecast más reciente de costos para un equipo específico.

    Retorna el resumen mensual con precio central e intervalos de
    confianza calculados con simulación Monte Carlo (1000 trayectorias).
    La incertidumbre está correctamente propagada desde las predictoras.

    Cuándo usar esta tool:
    - El usuario pregunta por el precio futuro de un equipo
    - El usuario pregunta por proyecciones o estimaciones de costos
    - El usuario quiere saber el horizonte confiable del forecast

    Args:
        equipment_id: Identificador del equipo.
                      Valores válidos: 'Equipo1' o 'Equipo2'

    Returns:
        Diccionario con:
        - status: 'success' o 'error'
        - forecast_summary: resumen mensual con precio central e IC 95%
        - reliable_months: meses con IC < 25% del precio (~1 mes)
        - message: descripción del resultado
    """
    logger.info("MCP tool get_forecast_data — equipment_id: %s", equipment_id)
    return server.get_forecast_data(equipment_id=equipment_id)


@mcp.tool()
def get_historical_data(start_date: str, end_date: str) -> dict:
    """
    Obtiene precios históricos de equipos y materias primas en un rango.

    Cuándo usar esta tool:
    - El usuario pregunta por precios pasados de equipos o materias primas
    - El usuario quiere comparar precios históricos
    - El usuario pregunta por tendencias históricas
    - Como alternativa cuando no hay forecast disponible

    NO usar esta tool para predicciones futuras —
    usar get_forecast_data en su lugar.

    Args:
        start_date: Fecha de inicio en formato 'YYYY-MM-DD'.
                    Rango disponible: 2010-01-04 en adelante.
        end_date: Fecha de fin en formato 'YYYY-MM-DD'.
                  Rango disponible: hasta 2023-08-31.

    Returns:
        Diccionario con:
        - status: 'success' o 'error'
        - records: lista de registros con Date, Price_X, Price_Y,
                   Price_Z, Price_Equipo1, Price_Equipo2
        - n_records: número de registros encontrados
        - message: descripción del resultado
    """
    logger.info(
        "MCP tool get_historical_data — rango: %s → %s",
        start_date,
        end_date,
    )
    return server.get_historical_data(
        start_date=start_date,
        end_date=end_date,
    )


@mcp.tool()
def get_model_metrics() -> dict:
    """
    Obtiene las métricas de evaluación de los modelos de predicción.

    Retorna la comparación entre OLS (baseline), ARIMAX y XGBoost
    para ambos equipos, con métricas MAE, RMSE, MAPE y R².

    Cuándo usar esta tool:
    - El usuario pregunta por la precisión o rendimiento del modelo
    - El usuario quiere saber qué modelo se usó y por qué
    - El usuario pregunta por el error esperado en las predicciones
    - El usuario quiere comparar modelos

    Returns:
        Diccionario con:
        - status: 'success' o 'error'
        - metrics: lista con métricas por modelo y equipo
        - best_model: modelo ganador (ARIMAX)
        - message: descripción del resultado con contexto
    """
    logger.info("MCP tool get_model_metrics")
    return server.get_model_metrics()


if __name__ == "__main__":
    logger.info("Iniciando MCP Server — construction-cost-forecasting")
    mcp.run(transport="stdio")