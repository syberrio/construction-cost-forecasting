"""
Implementación concreta del repositorio de datos históricos
usando archivos CSV como fuente de datos.

Esta clase implementa las interfaces HistoricalRepository y
ForecastRepository definidas en el dominio, siguiendo el patrón
Adapter — permite que la lógica de negocio sea independiente
del mecanismo de almacenamiento concreto.

Si en el futuro se desea usar una base de datos relacional,
solo se necesita crear un nuevo adaptador que implemente
las mismas interfaces sin modificar la lógica de negocio.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from src.domain.entities.commodity import Commodity
from src.domain.exceptions import DataNotFoundError, RepositoryError
from src.domain.repositories.forecast_repository import ForecastRepository
from src.domain.repositories.historical_repository import HistoricalRepository
from src.domain.value_objects.forecast import ConfidenceInterval, ForecastPoint
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Constantes ────────────────────────────────────────────
COMMODITY_FILE_MAP: dict[str, str] = {
    "X": "X.csv",
    "Y": "Y.csv",
    "Z": "Z.csv",
}

HISTORICAL_FILE: str = "historico_equipos.csv"
FORECAST_FILE_TEMPLATE: str = "forecast_equipo{equipment_number}.csv"
FORECAST_SUMMARY_TEMPLATE: str = "forecast_resumen_equipo{equipment_number}.csv"


class CSVHistoricalRepository(HistoricalRepository):
    """
    Repositorio de datos históricos basado en archivos CSV.

    Implementa HistoricalRepository usando pandas para leer
    los archivos procesados en data/processed/.

    Attributes:
        data_path: Ruta al directorio con los archivos CSV procesados.

    Example:
        >>> from pathlib import Path
        >>> repo = CSVHistoricalRepository(Path("data/processed"))
        >>> df = repo.get_historical_prices(date(2020, 1, 1), date(2023, 8, 31))
    """

    def __init__(self, data_path: Path) -> None:
        """
        Inicializa el repositorio con la ruta a los datos procesados.

        Args:
            data_path: Ruta al directorio con los archivos CSV procesados.

        Raises:
            RepositoryError: Si el directorio no existe.
        """
        if not data_path.exists():
            raise RepositoryError(
                f"El directorio de datos no existe: {data_path}"
            )
        self._data_path = data_path
        logger.info("CSVHistoricalRepository inicializado en: %s", data_path)

    def get_historical_prices(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Obtiene los precios históricos del archivo maestro.

        Args:
            start_date: Fecha de inicio del rango (inclusive).
            end_date: Fecha de fin del rango (inclusive).

        Returns:
            DataFrame con columnas:
            [Date, Price_X, Price_Y, Price_Z,
             Price_Equipo1, Price_Equipo2]

        Raises:
            DataNotFoundError: Si no hay datos en el rango solicitado.
            RepositoryError: Si ocurre un error al leer el archivo.
        """
        file_path = self._data_path / HISTORICAL_FILE
        logger.debug("Cargando histórico desde: %s", file_path)

        try:
            df = pd.read_csv(file_path, parse_dates=["Date"])
        except FileNotFoundError as e:
            raise RepositoryError(
                f"Archivo no encontrado: {file_path}"
            ) from e
        except Exception as e:
            raise RepositoryError(
                f"Error al leer {file_path}: {e}"
            ) from e

        df = df.sort_values("Date").reset_index(drop=True)
        mask = (df["Date"].dt.date >= start_date) & (
            df["Date"].dt.date <= end_date
        )
        result = df[mask].copy()

        if result.empty:
            raise DataNotFoundError(
                f"No hay datos históricos para el rango "
                f"{start_date} → {end_date}"
            )

        logger.info(
            "Histórico cargado: %d registros (%s → %s)",
            len(result),
            start_date,
            end_date,
        )
        return result

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
            RepositoryError: Si ocurre un error al leer el archivo.
        """
        if commodity_id not in COMMODITY_FILE_MAP:
            raise DataNotFoundError(
                f"Commodity no reconocido: {commodity_id}. "
                f"Valores válidos: {list(COMMODITY_FILE_MAP.keys())}"
            )

        file_path = self._data_path / COMMODITY_FILE_MAP[commodity_id]
        logger.debug("Cargando commodity %s desde: %s", commodity_id, file_path)

        try:
            df = pd.read_csv(file_path, parse_dates=["Date"])
        except FileNotFoundError as e:
            raise RepositoryError(
                f"Archivo no encontrado: {file_path}"
            ) from e
        except Exception as e:
            raise RepositoryError(
                f"Error al leer {file_path}: {e}"
            ) from e

        df = df.sort_values("Date").reset_index(drop=True)
        mask = (df["Date"].dt.date >= start_date) & (
            df["Date"].dt.date <= end_date
        )
        result = df[mask].copy()

        if result.empty:
            raise DataNotFoundError(
                f"No hay datos para commodity {commodity_id} "
                f"en el rango {start_date} → {end_date}"
            )

        logger.info(
            "Commodity %s cargado: %d registros",
            commodity_id,
            len(result),
        )
        return result

    def get_last_known_date(self) -> date:
        """
        Obtiene la fecha del último registro disponible en el histórico.

        Returns:
            Fecha del último registro disponible.

        Raises:
            RepositoryError: Si ocurre un error al leer el archivo.
        """
        file_path = self._data_path / HISTORICAL_FILE
        logger.debug("Obteniendo última fecha conocida desde: %s", file_path)

        try:
            df = pd.read_csv(file_path, parse_dates=["Date"])
            last_date = df["Date"].max().date()
            logger.info("Última fecha conocida: %s", last_date)
            return last_date
        except FileNotFoundError as e:
            raise RepositoryError(
                f"Archivo no encontrado: {file_path}"
            ) from e
        except Exception as e:
            raise RepositoryError(
                f"Error al obtener última fecha: {e}"
            ) from e


class CSVForecastRepository(ForecastRepository):
    """
    Repositorio de forecasts basado en archivos CSV.

    Implementa ForecastRepository leyendo los archivos de
    proyecciones generados en el notebook 04.

    Attributes:
        data_path: Ruta al directorio con los archivos CSV procesados.

    Example:
        >>> from pathlib import Path
        >>> repo = CSVForecastRepository(Path("data/processed"))
        >>> points = repo.get_latest_forecast("Equipo1")
    """

    def __init__(self, data_path: Path) -> None:
        """
        Inicializa el repositorio con la ruta a los datos procesados.

        Args:
            data_path: Ruta al directorio con los archivos CSV procesados.

        Raises:
            RepositoryError: Si el directorio no existe.
        """
        if not data_path.exists():
            raise RepositoryError(
                f"El directorio de datos no existe: {data_path}"
            )
        self._data_path = data_path
        logger.info("CSVForecastRepository inicializado en: %s", data_path)

    def _get_equipment_number(self, equipment_id: str) -> str:
        """
        Extrae el número del equipo del identificador.

        Args:
            equipment_id: Identificador del equipo (ej: 'Equipo1').

        Returns:
            Número del equipo como string (ej: '1').

        Raises:
            DataNotFoundError: Si el equipment_id no tiene formato válido.
        """
        number = "".join(filter(str.isdigit, equipment_id))
        if not number:
            raise DataNotFoundError(
                f"No se pudo extraer el número del equipo: {equipment_id}"
            )
        return number

    def save_forecast(
        self,
        equipment_id: str,
        forecast_points: list[ForecastPoint],
    ) -> None:
        """
        Persiste los puntos de forecast en un archivo CSV.

        Args:
            equipment_id: Identificador del equipo.
            forecast_points: Lista de puntos de forecast a persistir.

        Raises:
            RepositoryError: Si ocurre un error al escribir el archivo.
        """
        number = self._get_equipment_number(equipment_id)
        file_path = self._data_path / FORECAST_FILE_TEMPLATE.format(
            equipment_number=number
        )

        try:
            records = [
                {
                    "Date": fp.forecast_date,
                    f"Price_Equipo{number}": fp.predicted_price,
                    f"Equipo{number}_lower": fp.confidence_interval.lower,
                    f"Equipo{number}_upper": fp.confidence_interval.upper,
                }
                for fp in forecast_points
            ]
            pd.DataFrame(records).to_csv(file_path, index=False)
            logger.info(
                "Forecast guardado para %s: %d puntos en %s",
                equipment_id,
                len(forecast_points),
                file_path,
            )
        except Exception as e:
            raise RepositoryError(
                f"Error al guardar forecast para {equipment_id}: {e}"
            ) from e

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
            equipment_id: Identificador del equipo.
            start_date: Fecha de inicio del rango (inclusive).
            end_date: Fecha de fin del rango (inclusive).

        Returns:
            Lista de ForecastPoint ordenada por fecha ascendente.

        Raises:
            DataNotFoundError: Si no hay forecast para el rango solicitado.
            RepositoryError: Si ocurre un error al leer el archivo.
        """
        points = self.get_latest_forecast(equipment_id)
        filtered = [
            p for p in points
            if start_date <= p.forecast_date <= end_date
        ]

        if not filtered:
            raise DataNotFoundError(
                f"No hay forecast para {equipment_id} "
                f"en el rango {start_date} → {end_date}"
            )

        return filtered

    def get_latest_forecast(
        self,
        equipment_id: str,
    ) -> list[ForecastPoint]:
        """
        Recupera el forecast más reciente para un equipo.

        Args:
            equipment_id: Identificador del equipo.

        Returns:
            Lista de ForecastPoint del forecast más reciente.

        Raises:
            DataNotFoundError: Si no hay forecast disponible.
            RepositoryError: Si ocurre un error al leer el archivo.
        """
        number = self._get_equipment_number(equipment_id)
        file_path = self._data_path / FORECAST_FILE_TEMPLATE.format(
            equipment_number=number
        )
        logger.debug(
            "Cargando forecast de %s desde: %s", equipment_id, file_path
        )

        try:
            df = pd.read_csv(file_path, parse_dates=["Date"])
        except FileNotFoundError as e:
            raise DataNotFoundError(
                f"No se encontró forecast para {equipment_id}: {file_path}"
            ) from e
        except Exception as e:
            raise RepositoryError(
                f"Error al leer forecast de {equipment_id}: {e}"
            ) from e

        # Detectar nombres de columnas automáticamente
        # Soporta 'Price_Equipo1'/'Equipo1_lower' y
        # 'Eq1_central'/'Eq1_lower' generados por notebook 04
        price_col = next(
            (c for c in df.columns if "central" in c
             or f"Price_Equipo{number}" in c),
            None,
        )
        lower_col = next(
            (c for c in df.columns if "lower" in c),
            None,
        )
        upper_col = next(
            (c for c in df.columns if "upper" in c),
            None,
        )

        if not all([price_col, lower_col, upper_col]):
            raise RepositoryError(
                f"No se encontraron columnas esperadas en {file_path}. "
                f"Columnas disponibles: {df.columns.tolist()}"
            )

        logger.debug(
            "Columnas detectadas — precio: %s, lower: %s, upper: %s",
            price_col,
            lower_col,
            upper_col,
        )

        points = [
            ForecastPoint(
                forecast_date=row["Date"].date(),
                predicted_price=float(row[price_col]),
                confidence_interval=ConfidenceInterval(
                    lower=float(row[lower_col]),
                    upper=float(row[upper_col]),
                ),
                equipment_id=equipment_id,
            )
            for _, row in df.iterrows()
        ]

        logger.info(
            "Forecast cargado para %s: %d puntos",
            equipment_id,
            len(points),
        )
        return points

    def get_forecast_summary(
        self,
        equipment_id: str,
    ) -> pd.DataFrame:
        """
        Recupera el resumen mensual del forecast para un equipo.

        Args:
            equipment_id: Identificador del equipo.

        Returns:
            DataFrame con resumen mensual del forecast.

        Raises:
            DataNotFoundError: Si no hay resumen disponible.
            RepositoryError: Si ocurre un error al leer el archivo.
        """
        number = self._get_equipment_number(equipment_id)
        file_path = self._data_path / FORECAST_SUMMARY_TEMPLATE.format(
            equipment_number=number
        )
        logger.debug(
            "Cargando resumen forecast de %s desde: %s",
            equipment_id,
            file_path,
        )

        try:
            df = pd.read_csv(file_path)
            logger.info(
                "Resumen forecast cargado para %s: %d meses",
                equipment_id,
                len(df),
            )
            return df
        except FileNotFoundError as e:
            raise DataNotFoundError(
                f"No se encontró resumen de forecast para {equipment_id}"
            ) from e
        except Exception as e:
            raise RepositoryError(
                f"Error al leer resumen de forecast: {e}"
            ) from e