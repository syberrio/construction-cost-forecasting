"""
Módulo de feature engineering para los modelos de predicción
de costos de equipos de construcción.

Implementa la creación de features temporales identificadas
en el notebook 03 como relevantes para el modelo XGBoost,
incluyendo lags, medias móviles, volatilidad y calendario.

También incluye la variable dummy para el período atípico
2021-2022 identificado en el EDA (notebook 01).
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Constantes ────────────────────────────────────────────
DUMMY_START: str = "2021-04-08"
DUMMY_END: str = "2022-05-23"

LAG_PERIODS: list[int] = [1, 2, 3, 5, 10, 20]
ROLLING_WINDOWS_MEAN: list[int] = [5, 10, 20, 60]
ROLLING_WINDOWS_STD: list[int] = [5, 20]
DIFF_PERIODS: list[int] = [1, 5]


def create_temporal_features(
    df: pd.DataFrame,
    predictor: str,
) -> pd.DataFrame:
    """
    Crea features temporales para XGBoost a partir de una predictora.

    Genera lags, medias móviles, volatilidad móvil, diferencias
    y features de calendario para capturar la estructura temporal
    de la serie sin depender de los mecanismos internos de ARIMA.

    Args:
        df: DataFrame con columnas Date y la predictora.
        predictor: Nombre de la columna predictora
                   (ej: 'Price_Y' o 'Price_Z').

    Returns:
        DataFrame con features adicionales agregadas.

    Raises:
        ValueError: Si el predictor no existe en el DataFrame.

    Example:
        >>> df_features = create_temporal_features(hist, "Price_Y")
        >>> print(df_features.shape)
    """
    if predictor not in df.columns:
        raise ValueError(
            f"La columna '{predictor}' no existe en el DataFrame. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    logger.debug("Creando features temporales para: %s", predictor)
    df = df.copy()

    # Features de lag
    for lag in LAG_PERIODS:
        df[f"{predictor}_lag{lag}"] = df[predictor].shift(lag)
        logger.debug("Feature creada: %s_lag%d", predictor, lag)

    # Medias móviles
    for window in ROLLING_WINDOWS_MEAN:
        df[f"{predictor}_ma{window}"] = df[predictor].rolling(window).mean()
        logger.debug("Feature creada: %s_ma%d", predictor, window)

    # Volatilidad móvil
    for window in ROLLING_WINDOWS_STD:
        df[f"{predictor}_std{window}"] = df[predictor].rolling(window).std()
        logger.debug("Feature creada: %s_std%d", predictor, window)

    # Diferencias
    for period in DIFF_PERIODS:
        df[f"{predictor}_diff{period}"] = df[predictor].diff(period)
        logger.debug("Feature creada: %s_diff%d", predictor, period)

    # Cambio porcentual diario
    df[f"{predictor}_pct1"] = df[predictor].pct_change(1)
    logger.debug("Feature creada: %s_pct1", predictor)

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features de calendario al DataFrame.

    Extrae mes, trimestre y día de la semana de la columna Date
    para capturar patrones estacionales identificados en la
    descomposición STL del notebook 01.

    Args:
        df: DataFrame con columna Date de tipo datetime.

    Returns:
        DataFrame con features de calendario agregadas.

    Raises:
        ValueError: Si la columna Date no existe en el DataFrame.

    Example:
        >>> df_cal = add_calendar_features(df)
        >>> print(df_cal[["month", "quarter", "day_of_week"]].head())
    """
    if "Date" not in df.columns:
        raise ValueError(
            "La columna 'Date' no existe en el DataFrame."
        )

    logger.debug("Agregando features de calendario")
    df = df.copy()

    df["month"]       = df["Date"].dt.month
    df["quarter"]     = df["Date"].dt.quarter
    df["day_of_week"] = df["Date"].dt.dayofweek

    return df


def add_pandemic_dummy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variable dummy para el período atípico 2021-2022.

    El período abril 2021 → mayo 2022 fue identificado en el EDA
    como un régimen atípico por disrupciones post-pandemia.
    Esta dummy permite a XGBoost distinguir el comportamiento
    normal del anómalo durante el entrenamiento.

    Args:
        df: DataFrame con columna Date de tipo datetime.

    Returns:
        DataFrame con columna dummy_pandemic agregada
        (1 = período atípico, 0 = período normal).

    Raises:
        ValueError: Si la columna Date no existe en el DataFrame.

    Example:
        >>> df_dummy = add_pandemic_dummy(df)
        >>> print(df_dummy["dummy_pandemic"].value_counts())
    """
    if "Date" not in df.columns:
        raise ValueError(
            "La columna 'Date' no existe en el DataFrame."
        )

    logger.debug(
        "Agregando dummy pandemia: %s → %s", DUMMY_START, DUMMY_END
    )
    df = df.copy()

    df["dummy_pandemic"] = (
        (df["Date"] >= DUMMY_START) & (df["Date"] <= DUMMY_END)
    ).astype(int)

    n_dummy = df["dummy_pandemic"].sum()
    logger.info(
        "Dummy pandemia agregada: %d registros marcados (%.1f%%)",
        n_dummy,
        n_dummy / len(df) * 100,
    )

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    predictors: list[str],
) -> pd.DataFrame:
    """
    Construye la matriz completa de features para el modelado.

    Ejecuta en secuencia: features temporales por predictor,
    features de calendario y dummy de pandemia. Elimina filas
    con NaN generados por lags y rolling windows.

    Args:
        df: DataFrame con columnas Date y predictoras.
        predictors: Lista de columnas predictoras
                    (ej: ['Price_Y', 'Price_Z']).

    Returns:
        DataFrame limpio con todas las features listo para modelar.

    Raises:
        ValueError: Si algún predictor no existe en el DataFrame.

    Example:
        >>> df_features = build_feature_matrix(hist, ["Price_Y", "Price_Z"])
        >>> print(f"Shape: {df_features.shape}")
    """
    logger.info(
        "Construyendo matriz de features para predictores: %s", predictors
    )

    result = df.copy()

    for predictor in predictors:
        result = create_temporal_features(result, predictor)

    result = add_calendar_features(result)
    result = add_pandemic_dummy(result)

    rows_before = len(result)
    result = result.dropna().reset_index(drop=True)
    rows_after = len(result)

    logger.info(
        "Matriz de features construida: %d → %d filas (%d eliminadas por NaN), %d columnas",
        rows_before,
        rows_after,
        rows_before - rows_after,
        result.shape[1],
    )

    return result