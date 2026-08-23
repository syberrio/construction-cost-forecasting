"""
Pipeline de forecast con simulación Monte Carlo para propagación
correcta de la incertidumbre de las predictoras.

Implementa el enfoque desarrollado en el notebook 04:
1. Proyecta las predictoras (Price_Y, Price_Z) con ARIMA propio
2. Genera N trayectorias de la predictora muestreando su IC
3. Para cada trayectoria calcula el forecast del equipo
4. Calcula percentiles para obtener el IC real del equipo

Este enfoque corrige el error de 'inyección estática' donde
statsmodels trata los valores futuros de la covariable como
datos conocidos con certeza, subestimando la incertidumbre real.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.domain.exceptions import ForecastError
from src.domain.value_objects.forecast import ConfidenceInterval, ForecastPoint
from src.infrastructure.ml.arimax_model import ARIMAXModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Constantes ────────────────────────────────────────────
N_SIMULATIONS: int = 1000
FORECAST_STEPS: int = 147  # 7 meses × ~21 días hábiles
CONFIDENCE_LEVEL: float = 0.95
PERCENTILE_LOWER: float = 2.5
PERCENTILE_UPPER: float = 97.5
ARIMA_MAX_P: int = 5
ARIMA_MAX_Q: int = 5


@dataclass
class PredictorForecast:
    """
    Resultado de la proyección de una variable predictora.

    Attributes:
        dates: Fechas futuras proyectadas.
        mean: Valores medios proyectados.
        lower: Límite inferior del IC 95%.
        upper: Límite superior del IC 95%.
        std: Desviación estándar derivada del IC.
    """

    dates: pd.DatetimeIndex
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    std: np.ndarray


def project_predictor(
    historical_series: pd.Series,
    steps: int,
    forecast_start: str,
) -> PredictorForecast:
    """
    Proyecta una serie predictora usando auto_arima.

    Selecciona automáticamente los mejores parámetros p, q
    usando AIC y genera el forecast con IC 95%.

    Args:
        historical_series: Serie histórica de la predictora.
        steps: Número de pasos a proyectar.
        forecast_start: Fecha de inicio del forecast (ej: '2023-09-01').

    Returns:
        PredictorForecast con la proyección y sus intervalos.

    Raises:
        ForecastError: Si ocurre un error al proyectar la predictora.

    Example:
        >>> forecast = project_predictor(hist["Price_Y"], 147, "2023-09-01")
        >>> print(forecast.mean[:5])
    """
    logger.info(
        "Proyectando predictora — %d pasos desde %s",
        steps,
        forecast_start,
    )

    try:
        # Selección automática de parámetros
        auto = auto_arima(
            historical_series,
            d=1,
            start_p=0, max_p=ARIMA_MAX_P,
            start_q=0, max_q=ARIMA_MAX_Q,
            seasonal=False,
            information_criterion="aic",
            stepwise=True,
            verbose=False,
        )
        order = auto.order
        logger.info("Mejor modelo ARIMA seleccionado: %s", order)

        # Entrenar con parámetros óptimos
        model = SARIMAX(
            historical_series,
            order=order,
            trend="n",
        ).fit(disp=False)

        # Generar forecast con IC
        forecast_result = model.get_forecast(steps=steps)
        mean = forecast_result.predicted_mean.values
        conf_int = forecast_result.conf_int(alpha=1 - CONFIDENCE_LEVEL)
        lower = conf_int.iloc[:, 0].values
        upper = conf_int.iloc[:, 1].values
        std = (upper - lower) / (2 * 1.96)

        # Fechas futuras en días hábiles
        future_dates = pd.bdate_range(
            start=forecast_start,
            periods=steps,
        )

        logger.info(
            "Predictora proyectada exitosamente: %d pasos",
            steps,
        )

        return PredictorForecast(
            dates=future_dates,
            mean=mean,
            lower=lower,
            upper=upper,
            std=std,
        )

    except Exception as e:
        raise ForecastError(
            f"Error al proyectar predictora: {e}"
        ) from e


def run_monte_carlo_forecast(
    arimax_model: ARIMAXModel,
    predictor_forecast: PredictorForecast,
    predictor_column: str,
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Ejecuta simulación Monte Carlo para propagar la incertidumbre
    de la predictora al forecast del equipo.

    Para cada simulación:
    1. Muestrea una trayectoria de la predictora de su distribución normal
    2. Genera el forecast del equipo con esa trayectoria
    3. Calcula percentiles sobre todas las simulaciones

    Args:
        arimax_model: Modelo ARIMAX cargado para el equipo.
        predictor_forecast: Proyección de la predictora con IC.
        predictor_column: Nombre de la columna predictora
                          (ej: 'Price_Y').
        n_simulations: Número de simulaciones Monte Carlo.
        random_seed: Semilla para reproducibilidad.

    Returns:
        DataFrame con columnas:
        [Date, Price_Equipo{N}, Equipo{N}_lower, Equipo{N}_upper,
         Equipo{N}_p10, Equipo{N}_p90, amplitud_ic, mes]

    Raises:
        ForecastError: Si ocurre un error durante la simulación.

    Example:
        >>> df_forecast = run_monte_carlo_forecast(
        ...     arimax_model=model,
        ...     predictor_forecast=pred_forecast,
        ...     predictor_column="Price_Y",
        ... )
    """
    equipment_id = arimax_model.equipment_id
    number = "".join(filter(str.isdigit, equipment_id))
    steps = len(predictor_forecast.dates)

    logger.info(
        "Iniciando Monte Carlo para %s — %d simulaciones, %d pasos",
        equipment_id,
        n_simulations,
        steps,
    )

    np.random.seed(random_seed)
    simulations = np.zeros((n_simulations, steps))

    try:
        for i in range(n_simulations):
            # Muestrear trayectoria aleatoria de la predictora
            y_sim = np.random.normal(
                loc=predictor_forecast.mean,
                scale=predictor_forecast.std,
            ).reshape(-1, 1)

            # Forecast del equipo con esa trayectoria
            exog_df = pd.DataFrame(
                y_sim,
                columns=[predictor_column],
            )
            mean, _ = arimax_model.forecast(
                exog_future=exog_df,
                steps=steps,
            )
            simulations[i, :] = mean.values

            if (i + 1) % 100 == 0:
                logger.debug(
                    "Monte Carlo %s: %d/%d simulaciones completadas",
                    equipment_id,
                    i + 1,
                    n_simulations,
                )

    except Exception as e:
        raise ForecastError(
            f"Error en simulación Monte Carlo para {equipment_id}: {e}"
        ) from e

    # Calcular percentiles
    central = np.median(simulations, axis=0)
    lower = np.percentile(simulations, PERCENTILE_LOWER, axis=0)
    upper = np.percentile(simulations, PERCENTILE_UPPER, axis=0)
    p10 = np.percentile(simulations, 10, axis=0)
    p90 = np.percentile(simulations, 90, axis=0)

    result = pd.DataFrame({
        "Date":                     predictor_forecast.dates,
        f"Price_Equipo{number}":    central,
        f"Equipo{number}_lower":    lower,
        f"Equipo{number}_upper":    upper,
        f"Equipo{number}_p10":      p10,
        f"Equipo{number}_p90":      p90,
    })

    result["amplitud_ic"] = result[f"Equipo{number}_upper"] - result[f"Equipo{number}_lower"]
    result["mes"] = result["Date"].dt.to_period("M")

    logger.info(
        "Monte Carlo completado para %s — mediana central: %.2f",
        equipment_id,
        float(central.mean()),
    )

    return result