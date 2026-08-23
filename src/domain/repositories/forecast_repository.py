"""
Interfaz abstracta para el repositorio de resultados de forecast.

Define el contrato que cualquier implementación concreta debe cumplir
para persistir y recuperar los resultados de las proyecciones de precios.

Siguiendo el principio de inversión de dependencias (DIP) de SOLID,
la capa de dominio no conoce el mecanismo de almacenamiento concreto.
"""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from src.domain.value_objects.forecast import ForecastPoint


class ForecastRepository(ABC):
    """
    Interfaz abstracta para persistencia y recuperación de forecasts.

    Cualquier implementación concreta (CSV, PostgreSQL, API, etc.)
    debe heredar de esta clase e implementar todos sus métodos.

    Example:
        >>> class CSVForecastRepository(ForecastRepository):
        ...     def save_forecast(self, equipment_id, forecast_points):
        ...         # implementación concreta
        ...         pass
    """

    @abstractmethod
    def save_forecast(
        self,
        equipment_id: str,
        forecast_points: list[ForecastPoint],
    ) -> None:
        """
        Persiste los puntos de forecast para un equipo específico.

        Args:
            equipment_id: Identificador del equipo (ej: 'Equipo1').
            forecast_points: Lista de puntos de forecast a persistir.

        Raises:
            RepositoryError: Si ocurre un error al persistir los datos.
        """
        ...

    @abstractmethod
    def get_forecast(
        self,
        equipment_id: str,
        start_date: date,
        end_date: date,
    ) -> list[ForecastPoint]:
        """
        Recupera los puntos de forecast para un equipo en un rango
        de fechas.

        Args:
            equipment_id: Identificador del equipo (ej: 'Equipo1').
            start_date: Fecha de inicio del rango (inclusive).
            end_date: Fecha de fin del rango (inclusive).

        Returns:
            Lista de ForecastPoint ordenada por fecha ascendente.

        Raises:
            DataNotFoundError: Si no hay forecast para el rango solicitado.
            RepositoryError: Si ocurre un error al recuperar los datos.
        """
        ...

    @abstractmethod
    def get_latest_forecast(
        self,
        equipment_id: str,
    ) -> list[ForecastPoint]:
        """
        Recupera el forecast más reciente disponible para un equipo.

        Args:
            equipment_id: Identificador del equipo (ej: 'Equipo1').

        Returns:
            Lista de ForecastPoint del forecast más reciente.

        Raises:
            DataNotFoundError: Si no hay forecast disponible.
            RepositoryError: Si ocurre un error al recuperar los datos.
        """
        ...

    @abstractmethod
    def get_forecast_summary(
        self,
        equipment_id: str,
    ) -> pd.DataFrame:
        """
        Recupera un resumen mensual del forecast más reciente
        para un equipo.

        Args:
            equipment_id: Identificador del equipo (ej: 'Equipo1').

        Returns:
            DataFrame con columnas:
            [mes, precio_medio, ic_lower, ic_upper, amplitud_95]

        Raises:
            DataNotFoundError: Si no hay forecast disponible.
            RepositoryError: Si ocurre un error al recuperar los datos.
        """
        ...