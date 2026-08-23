"""
Caso de uso: ejecutar el pipeline completo de forecast.

Orquesta la secuencia completa de pasos para generar el forecast
de costos de equipos de construcción:
1. Carga los datos históricos
2. Proyecta las predictoras (Price_Y, Price_Z)
3. Ejecuta Monte Carlo para cada equipo
4. Persiste los resultados

Siguiendo el principio de responsabilidad única (SRP), este módulo
solo orquesta — no implementa la lógica de negocio ni el acceso
a datos directamente.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.domain.entities.equipment import Equipment
from src.domain.exceptions import ForecastError
from src.domain.repositories.forecast_repository import ForecastRepository
from src.domain.repositories.historical_repository import HistoricalRepository
from src.domain.value_objects.forecast import ConfidenceInterval, ForecastPoint
from src.infrastructure.ml.arimax_model import ARIMAXModel
from src.infrastructure.ml.forecast_pipeline import (
    PredictorForecast,
    project_predictor,
    run_monte_carlo_forecast,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastPipelineConfig:
    """
    Configuración del pipeline de forecast.

    Attributes:
        models_path: Ruta al directorio con los modelos entrenados.
        forecast_start: Fecha de inicio del forecast (ej: '2023-09-01').
        forecast_steps: Número de pasos a proyectar.
        n_simulations: Número de simulaciones Monte Carlo.
        random_seed: Semilla para reproducibilidad.
    """

    models_path: Path
    forecast_start: str
    forecast_steps: int = 147
    n_simulations: int = 1000
    random_seed: int = 42


class RunForecastPipeline:
    """
    Caso de uso que orquesta el pipeline completo de forecast.

    Coordina la interacción entre los repositorios de datos,
    los modelos ARIMAX y el pipeline de Monte Carlo para
    generar proyecciones de costos de equipos.

    Attributes:
        historical_repo: Repositorio de datos históricos.
        forecast_repo: Repositorio de resultados de forecast.
        config: Configuración del pipeline.

    Example:
        >>> pipeline = RunForecastPipeline(
        ...     historical_repo=csv_historical_repo,
        ...     forecast_repo=csv_forecast_repo,
        ...     config=ForecastPipelineConfig(
        ...         models_path=Path("models"),
        ...         forecast_start="2023-09-01",
        ...     )
        ... )
        >>> results = pipeline.execute(equipments)
    """

    def __init__(
        self,
        historical_repo: HistoricalRepository,
        forecast_repo: ForecastRepository,
        config: ForecastPipelineConfig,
    ) -> None:
        """
        Inicializa el pipeline con sus dependencias.

        Args:
            historical_repo: Repositorio de datos históricos.
            forecast_repo: Repositorio de resultados de forecast.
            config: Configuración del pipeline.
        """
        self._historical_repo = historical_repo
        self._forecast_repo = forecast_repo
        self._config = config
        logger.info(
            "RunForecastPipeline inicializado — forecast desde: %s, "
            "%d pasos, %d simulaciones",
            config.forecast_start,
            config.forecast_steps,
            config.n_simulations,
        )

    def execute(
        self,
        equipments: list[Equipment],
    ) -> dict[str, pd.DataFrame]:
        """
        Ejecuta el pipeline completo de forecast para una lista de equipos.

        Args:
            equipments: Lista de equipos a proyectar.

        Returns:
            Diccionario con equipment_id como clave y DataFrame
            de forecast como valor.

        Raises:
            ForecastError: Si ocurre un error durante el pipeline.
        """
        logger.info(
            "Iniciando pipeline de forecast para %d equipos",
            len(equipments),
        )

        # Cargar datos históricos completos
        hist = self._load_historical_data()

        results: dict[str, pd.DataFrame] = {}

        for equipment in equipments:
            logger.info(
                "Procesando equipo: %s", equipment.equipment_id
            )

            try:
                # Proyectar predictora
                predictor_forecast = self._project_predictor(
                    hist=hist,
                    predictor_column=equipment.predictor_column,
                )

                # Cargar modelo ARIMAX
                arimax_model = self._load_arimax_model(equipment)

                # Ejecutar Monte Carlo
                forecast_df = run_monte_carlo_forecast(
                    arimax_model=arimax_model,
                    predictor_forecast=predictor_forecast,
                    predictor_column=equipment.predictor_column,
                    n_simulations=self._config.n_simulations,
                    random_seed=self._config.random_seed,
                )

                # Persistir resultados
                forecast_points = self._df_to_forecast_points(
                    forecast_df=forecast_df,
                    equipment=equipment,
                )
                self._forecast_repo.save_forecast(
                    equipment_id=equipment.equipment_id,
                    forecast_points=forecast_points,
                )

                results[equipment.equipment_id] = forecast_df
                logger.info(
                    "Equipo %s procesado exitosamente",
                    equipment.equipment_id,
                )

            except Exception as e:
                logger.error(
                    "Error procesando equipo %s: %s",
                    equipment.equipment_id,
                    str(e),
                )
                raise ForecastError(
                    f"Error en pipeline para {equipment.equipment_id}: {e}"
                ) from e

        logger.info(
            "Pipeline completado — %d equipos procesados",
            len(results),
        )
        return results

    def _load_historical_data(self) -> pd.DataFrame:
        """
        Carga todos los datos históricos disponibles.

        Returns:
            DataFrame con el histórico completo.

        Raises:
            ForecastError: Si ocurre un error al cargar los datos.
        """
        logger.info("Cargando datos históricos")
        try:
            last_date = self._historical_repo.get_last_known_date()
            from datetime import date
            hist = self._historical_repo.get_historical_prices(
                start_date=date(2010, 1, 1),
                end_date=last_date,
            )
            logger.info(
                "Datos históricos cargados: %d registros", len(hist)
            )
            return hist
        except Exception as e:
            raise ForecastError(
                f"Error al cargar datos históricos: {e}"
            ) from e

    def _project_predictor(
        self,
        hist: pd.DataFrame,
        predictor_column: str,
    ) -> PredictorForecast:
        """
        Proyecta una variable predictora usando ARIMA.

        Args:
            hist: DataFrame con datos históricos.
            predictor_column: Nombre de la columna predictora.

        Returns:
            PredictorForecast con la proyección y sus intervalos.

        Raises:
            ForecastError: Si ocurre un error al proyectar.
        """
        logger.info("Proyectando predictora: %s", predictor_column)
        try:
            return project_predictor(
                historical_series=hist[predictor_column],
                steps=self._config.forecast_steps,
                forecast_start=self._config.forecast_start,
            )
        except Exception as e:
            raise ForecastError(
                f"Error al proyectar {predictor_column}: {e}"
            ) from e

    def _load_arimax_model(self, equipment: Equipment) -> ARIMAXModel:
        """
        Carga el modelo ARIMAX para un equipo específico.

        Args:
            equipment: Entidad del equipo.

        Returns:
            ARIMAXModel cargado y listo para usar.

        Raises:
            ForecastError: Si ocurre un error al cargar el modelo.
        """
        number = "".join(filter(str.isdigit, equipment.equipment_id))
        model_path = (
            self._config.models_path / f"arimax_equipo{number}.pkl"
        )
        logger.info(
            "Cargando modelo ARIMAX para %s desde %s",
            equipment.equipment_id,
            model_path,
        )
        model = ARIMAXModel(
            equipment_id=equipment.equipment_id,
            model_path=model_path,
        )
        model.load()
        return model

    def _df_to_forecast_points(
        self,
        forecast_df: pd.DataFrame,
        equipment: Equipment,
    ) -> list[ForecastPoint]:
        """
        Convierte el DataFrame de forecast en lista de ForecastPoint.

        Args:
            forecast_df: DataFrame con resultados del forecast.
            equipment: Entidad del equipo.

        Returns:
            Lista de ForecastPoint para persistir.
        """
        number = "".join(filter(str.isdigit, equipment.equipment_id))
        points = []

        for _, row in forecast_df.iterrows():
            point = ForecastPoint(
                forecast_date=row["Date"].date(),
                predicted_price=float(row[f"Price_Equipo{number}"]),
                confidence_interval=ConfidenceInterval(
                    lower=float(row[f"Equipo{number}_lower"]),
                    upper=float(row[f"Equipo{number}_upper"]),
                ),
                equipment_id=equipment.equipment_id,
            )
            points.append(point)

        logger.info(
            "Convertidos %d puntos de forecast para %s",
            len(points),
            equipment.equipment_id,
        )
        return points