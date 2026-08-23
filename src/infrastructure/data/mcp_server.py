"""
MCP Server como Data Access Layer para el agente de IA.

Implementa el patrón Adapter exponiendo los repositorios de datos
como herramientas que el agente LangGraph puede consumir.

El agente no conoce si los datos vienen de CSV o base de datos
relacional — solo interactúa con este servidor MCP.

Diseño:
    Agente LangGraph
        └── MCP Server (este módulo)
                ├── get_forecast_data     → CSVForecastRepository
                └── get_historical_data   → CSVHistoricalRepository

Extensibilidad:
    Para migrar a base de datos relacional, solo se necesita
    reemplazar CSVHistoricalRepository y CSVForecastRepository
    por implementaciones concretas alternativas sin modificar
    este servidor ni el agente.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.domain.exceptions import DataNotFoundError, RepositoryError
from src.infrastructure.data.csv_repository import (
    CSVForecastRepository,
    CSVHistoricalRepository,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Constantes ────────────────────────────────────────────
VALID_EQUIPMENT_IDS: list[str] = ["Equipo1", "Equipo2"]
VALID_COMMODITY_IDS: list[str] = ["X", "Y", "Z"]
DEFAULT_DATE_FORMAT: str = "%Y-%m-%d"


class MCPDataServer:
    """
    Servidor MCP que expone los datos del proyecto como herramientas
    para el agente de IA.

    Actúa como fachada (Facade) sobre los repositorios concretos,
    simplificando la interfaz para el agente y centralizando
    el manejo de errores y logging.

    Attributes:
        historical_repo: Repositorio de datos históricos.
        forecast_repo: Repositorio de forecasts.

    Example:
        >>> server = MCPDataServer(data_path=Path("data/processed"))
        >>> result = server.get_forecast_data("Equipo1")
        >>> result = server.get_historical_data(
        ...     start_date="2023-01-01",
        ...     end_date="2023-08-31"
        ... )
    """

    def __init__(self, data_path: Path) -> None:
        """
        Inicializa el servidor MCP con los repositorios concretos.

        Args:
            data_path: Ruta al directorio con los archivos CSV procesados.

        Raises:
            RepositoryError: Si el directorio no existe.
        """
        self._historical_repo = CSVHistoricalRepository(data_path)
        self._forecast_repo = CSVForecastRepository(data_path)
        logger.info(
            "MCPDataServer inicializado — data_path: %s", data_path
        )

    def get_forecast_data(
        self,
        equipment_id: str,
    ) -> dict[str, Any]:
        """
        Herramienta MCP: obtiene el forecast más reciente para un equipo.

        Retorna el resumen mensual del forecast con precio central
        e intervalos de confianza, listo para ser consumido por el agente.

        Args:
            equipment_id: Identificador del equipo ('Equipo1' o 'Equipo2').

        Returns:
            Diccionario con:
            - equipment_id: Identificador del equipo
            - forecast_summary: Lista de dicts con resumen mensual
            - reliable_months: Meses con IC < 25% del precio
            - status: 'success' o 'error'
            - message: Descripción del resultado

        Example:
            >>> result = server.get_forecast_data("Equipo1")
            >>> print(result["forecast_summary"][0])
        """
        logger.info(
            "MCP get_forecast_data — equipment_id: %s", equipment_id
        )

        if equipment_id not in VALID_EQUIPMENT_IDS:
            return {
                "status": "error",
                "message": (
                    f"Equipo no válido: {equipment_id}. "
                    f"Valores válidos: {VALID_EQUIPMENT_IDS}"
                ),
            }

        try:
            summary_df = self._forecast_repo.get_forecast_summary(
                equipment_id=equipment_id
            )
            points = self._forecast_repo.get_latest_forecast(
                equipment_id=equipment_id
            )

            reliable_days = sum(1 for p in points if p.is_reliable)

            # Aproximar a meses hábiles (~21 días hábiles por mes)
            reliable_months = max(1, reliable_days // 21) if reliable_days > 0 else 0

            logger.info(
                "Forecast obtenido para %s — %d días confiables (~%d mes/es)",
                equipment_id,
                reliable_days,
                reliable_months
            )

            return {
                "status": "success",
                "equipment_id": equipment_id,
                "forecast_summary": summary_df.to_dict(orient="records"),
                "reliable_months": reliable_months,
                "message": (
                    f"Forecast disponible para {equipment_id}. "
                    f"Meses con alta confianza: {reliable_months}"
                ),
            }

        except DataNotFoundError as e:
            logger.warning(
                "Forecast no encontrado para %s: %s", equipment_id, e.message
            )
            return {
                "status": "error",
                "message": f"No hay forecast disponible para {equipment_id}",
            }
        except Exception as e:
            logger.error(
                "Error inesperado en get_forecast_data para %s: %s",
                equipment_id,
                str(e),
            )
            return {
                "status": "success",
                "equipment_id": equipment_id,
                "forecast_summary": summary_df.to_dict(orient="records"),
                "reliable_days": reliable_days,
                "reliable_months": reliable_months,
                "message": (
                    f"Forecast disponible para {equipment_id}. "
                    f"Días con IC < 25% del precio: {reliable_days} "
                    f"(~{reliable_months} mes/es de alta confianza)"
                ),
            }

    def get_historical_data(
        self,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """
        Herramienta MCP: obtiene datos históricos de precios en un rango.

        Args:
            start_date: Fecha de inicio en formato 'YYYY-MM-DD'.
            end_date: Fecha de fin en formato 'YYYY-MM-DD'.

        Returns:
            Diccionario con:
            - status: 'success' o 'error'
            - records: Lista de dicts con los precios históricos
            - n_records: Número de registros encontrados
            - date_range: Rango de fechas efectivo
            - message: Descripción del resultado

        Example:
            >>> result = server.get_historical_data(
            ...     start_date="2023-01-01",
            ...     end_date="2023-08-31"
            ... )
        """
        logger.info(
            "MCP get_historical_data — rango: %s → %s",
            start_date,
            end_date,
        )

        try:
            start = datetime.strptime(
                start_date, DEFAULT_DATE_FORMAT
            ).date()
            end = datetime.strptime(
                end_date, DEFAULT_DATE_FORMAT
            ).date()
        except ValueError as e:
            return {
                "status": "error",
                "message": (
                    f"Formato de fecha inválido. "
                    f"Use '{DEFAULT_DATE_FORMAT}': {e}"
                ),
            }

        if start > end:
            return {
                "status": "error",
                "message": (
                    f"La fecha de inicio ({start_date}) debe ser "
                    f"anterior a la fecha de fin ({end_date})"
                ),
            }

        try:
            df = self._historical_repo.get_historical_prices(
                start_date=start,
                end_date=end,
            )

            # Convertir fechas a string para serialización
            df["Date"] = df["Date"].dt.strftime(DEFAULT_DATE_FORMAT)

            logger.info(
                "Histórico obtenido: %d registros (%s → %s)",
                len(df),
                start_date,
                end_date,
            )

            return {
                "status": "success",
                "n_records": len(df),
                "date_range": {
                    "start": start_date,
                    "end": end_date,
                },
                "records": df.to_dict(orient="records"),
                "message": (
                    f"Se encontraron {len(df)} registros "
                    f"entre {start_date} y {end_date}"
                ),
            }

        except DataNotFoundError as e:
            logger.warning(
                "Datos no encontrados: %s → %s: %s",
                start_date,
                end_date,
                e.message,
            )
            return {
                "status": "error",
                "message": (
                    f"No hay datos históricos para el rango "
                    f"{start_date} → {end_date}"
                ),
            }
        except Exception as e:
            logger.error(
                "Error inesperado en get_historical_data: %s", str(e)
            )
            return {
                "status": "error",
                "message": f"Error al obtener datos históricos: {str(e)}",
            }

    def get_model_metrics(self) -> dict[str, Any]:
        """
        Herramienta MCP: obtiene las métricas de evaluación de los modelos.

        Retorna la tabla comparativa de modelos generada en el notebook 03,
        útil para que el agente pueda responder preguntas sobre el
        rendimiento de los modelos.

        Returns:
            Diccionario con:
            - status: 'success' o 'error'
            - metrics: Lista de dicts con métricas por modelo y equipo
            - best_model: Mejor modelo seleccionado
            - message: Descripción del resultado

        Example:
            >>> result = server.get_model_metrics()
            >>> print(result["best_model"])
        """
        logger.info("MCP get_model_metrics")

        metrics_path = (
            self._historical_repo._data_path / "model_comparison.csv"
        )

        try:
            df = pd.read_csv(metrics_path)
            logger.info(
                "Métricas cargadas: %d modelos evaluados", len(df)
            )

            return {
                "status": "success",
                "metrics": df.to_dict(orient="records"),
                "best_model": "ARIMAX",
                "message": (
                    "Modelos evaluados: OLS (baseline), ARIMAX y XGBoost. "
                    "ARIMAX fue seleccionado como modelo ganador para "
                    "ambos equipos basado en MAPE y R²."
                ),
            }

        except FileNotFoundError:
            logger.warning("Archivo de métricas no encontrado: %s", metrics_path)
            return {
                "status": "error",
                "message": "No se encontró el archivo de métricas de modelos",
            }
        except Exception as e:
            logger.error(
                "Error inesperado en get_model_metrics: %s", str(e)
            )
            return {
                "status": "error",
                "message": f"Error al obtener métricas: {str(e)}",
            }