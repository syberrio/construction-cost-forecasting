"""
Interfaz abstracta para el repositorio de datos históricos de precios.

Define el contrato que cualquier implementación concreta debe cumplir,
siguiendo el principio de inversión de dependencias (DIP) de SOLID.
La capa de dominio no conoce si los datos vienen de CSV, base de datos
o cualquier otra fuente — solo conoce esta interfaz.
"""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class HistoricalRepository(ABC):
    """
    Interfaz abstracta para acceso a datos históricos de precios.

    Cualquier implementación concreta (CSV, PostgreSQL, API, etc.)
    debe heredar de esta clase e implementar todos sus métodos.

    Example:
        >>> class CSVRepository(HistoricalRepository):
        ...     def get_historical_prices(self, start_date, end_date):
        ...         # implementación concreta
        ...         pass
    """

    @abstractmethod
    def get_historical_prices(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Obtiene los precios históricos de equipos y materias primas
        en un rango de fechas.

        Args:
            start_date: Fecha de inicio del rango (inclusive).
            end_date: Fecha de fin del rango (inclusive).

        Returns:
            DataFrame con columnas:
            [Date, Price_X, Price_Y, Price_Z,
             Price_Equipo1, Price_Equipo2]

        Raises:
            DataNotFoundError: Si no hay datos en el rango solicitado.
            RepositoryError: Si ocurre un error al acceder a los datos.
        """
        ...

    @abstractmethod
    def get_commodity_prices(
        self,
        commodity_id: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Obtiene los precios históricos de una materia prima específica.

        Args:
            commodity_id: Identificador de la materia prima ('X', 'Y', 'Z').
            start_date: Fecha de inicio del rango (inclusive).
            end_date: Fecha de fin del rango (inclusive).

        Returns:
            DataFrame con columnas: [Date, Price]

        Raises:
            DataNotFoundError: Si no hay datos para el commodity solicitado.
            RepositoryError: Si ocurre un error al acceder a los datos.
        """
        ...

    @abstractmethod
    def get_last_known_date(self) -> date:
        """
        Obtiene la fecha del último registro disponible en el histórico.

        Returns:
            Fecha del último registro disponible.

        Raises:
            RepositoryError: Si ocurre un error al acceder a los datos.
        """
        ...