"""
Wrapper del modelo ARIMAX para predicción de costos de equipos.

Encapsula la lógica de carga, predicción y forecast del modelo
ARIMAX entrenado en el notebook 03, siguiendo el principio de
responsabilidad única (SRP) de SOLID.

El wrapper abstrae la implementación de statsmodels y expone
una interfaz limpia y tipada para su uso en la capa de aplicación.
"""

from pathlib import Path

import joblib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from src.domain.exceptions import ForecastError, ModelNotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ARIMAXModel:
    """
    Wrapper del modelo ARIMAX para predicción de precios de equipos.

    Encapsula la carga del modelo entrenado y expone métodos
    para predicción en período de test y forecast hacia el futuro.

    Attributes:
        equipment_id: Identificador del equipo que predice.
        model_path: Ruta al archivo del modelo serializado.

    Example:
        >>> from pathlib import Path
        >>> model = ARIMAXModel("Equipo1", Path("models/arimax_equipo1.pkl"))
        >>> model.load()
        >>> forecast_df = model.forecast(exog_future=price_y_future, steps=21)
    """

    def __init__(
        self,
        equipment_id: str,
        model_path: Path,
    ) -> None:
        """
        Inicializa el wrapper con el identificador del equipo
        y la ruta al modelo serializado.

        Args:
            equipment_id: Identificador del equipo (ej: 'Equipo1').
            model_path: Ruta al archivo .pkl del modelo entrenado.
        """
        self._equipment_id = equipment_id
        self._model_path = model_path
        self._model: SARIMAXResultsWrapper | None = None
        logger.info(
            "ARIMAXModel inicializado para %s — path: %s",
            equipment_id,
            model_path,
        )

    @property
    def equipment_id(self) -> str:
        """Retorna el identificador del equipo."""
        return self._equipment_id

    @property
    def is_loaded(self) -> bool:
        """Retorna True si el modelo está cargado en memoria."""
        return self._model is not None

    def load(self) -> None:
        """
        Carga el modelo ARIMAX desde el archivo serializado.

        Raises:
            ModelNotFoundError: Si el archivo del modelo no existe.
            ForecastError: Si ocurre un error al deserializar el modelo.
        """
        if not self._model_path.exists():
            raise ModelNotFoundError(
                f"No se encontró el modelo para {self._equipment_id}: "
                f"{self._model_path}"
            )

        logger.info("Cargando modelo ARIMAX para %s", self._equipment_id)
        try:
            self._model = joblib.load(self._model_path)
            logger.info(
                "Modelo ARIMAX cargado exitosamente para %s",
                self._equipment_id,
            )
        except Exception as e:
            raise ForecastError(
                f"Error al cargar el modelo para {self._equipment_id}: {e}"
            ) from e

    def predict(
        self,
        exog_test: pd.DataFrame,
    ) -> pd.Series:
        """
        Genera predicciones sobre el conjunto de test.

        Args:
            exog_test: DataFrame con la variable exógena para el período
                       de test (ej: Price_Y o Price_Z).

        Returns:
            Series con las predicciones para el período de test.

        Raises:
            ForecastError: Si el modelo no está cargado o falla la predicción.
        """
        self._validate_loaded()
        logger.info(
            "Generando predicciones para %s — %d pasos",
            self._equipment_id,
            len(exog_test),
        )

        try:
            predictions = self._model.forecast(
                steps=len(exog_test),
                exog=exog_test.values,
            )
            logger.info(
                "Predicciones generadas exitosamente para %s",
                self._equipment_id,
            )
            return predictions
        except Exception as e:
            raise ForecastError(
                f"Error al generar predicciones para "
                f"{self._equipment_id}: {e}"
            ) from e

    def forecast(
        self,
        exog_future: pd.DataFrame,
        steps: int,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Genera forecast hacia el futuro con intervalos de confianza.

        Args:
            exog_future: DataFrame con la variable exógena proyectada
                         para el período de forecast.
            steps: Número de pasos a proyectar.

        Returns:
            Tupla con:
            - Series con la predicción puntual (valor central)
            - DataFrame con los límites del IC (lower, upper)

        Raises:
            ForecastError: Si el modelo no está cargado o falla el forecast.
        """
        self._validate_loaded()
        logger.info(
            "Generando forecast para %s — %d pasos",
            self._equipment_id,
            steps,
        )

        try:
            forecast_result = self._model.get_forecast(
                steps=steps,
                exog=exog_future.values,
            )
            mean = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=0.05)

            logger.info(
                "Forecast generado exitosamente para %s",
                self._equipment_id,
            )
            return mean, conf_int
        except Exception as e:
            raise ForecastError(
                f"Error al generar forecast para "
                f"{self._equipment_id}: {e}"
            ) from e

    def _validate_loaded(self) -> None:
        """
        Valida que el modelo esté cargado antes de usarlo.

        Raises:
            ForecastError: Si el modelo no ha sido cargado.
        """
        if not self.is_loaded:
            raise ForecastError(
                f"El modelo para {self._equipment_id} no está cargado. "
                f"Llama a load() primero."
            )