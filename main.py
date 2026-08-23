"""
Punto de entrada principal del pipeline de forecast de costos
de equipos de construcción.

Orquesta la ejecución completa del pipeline:
1. Carga la configuración desde variables de entorno
2. Inicializa los repositorios y el pipeline
3. Ejecuta el forecast para todos los equipos
4. Reporta los resultados

Uso:
    uv run python main.py
"""

from pathlib import Path

from dotenv import load_dotenv

from src.application.run_forecast_pipeline import (
    ForecastPipelineConfig,
    RunForecastPipeline,
)
from src.domain.entities.equipment import Equipment
from src.domain.exceptions import DomainError
from src.infrastructure.data.csv_repository import (
    CSVForecastRepository,
    CSVHistoricalRepository,
)
from src.utils.logger import get_logger

# ── Configuración ─────────────────────────────────────────
load_dotenv()
logger = get_logger(__name__)

# ── Constantes ────────────────────────────────────────────
DATA_PATH: Path = Path("data/processed")
MODELS_PATH: Path = Path("models")
FORECAST_START: str = "2023-09-01"
FORECAST_STEPS: int = 147
N_SIMULATIONS: int = 1000
RANDOM_SEED: int = 42

# ── Equipos definidos según hallazgos del notebook 02 ─────
EQUIPMENTS: list[Equipment] = [
    Equipment(
        equipment_id="Equipo1",
        price_column="Price_Equipo1",
        predictor_column="Price_Y",
        description=(
            "Equipo crítico tipo 1 — predictor principal: Price_Y "
            "(correlación Pearson 0.997, cointegración p=0.013)"
        ),
    ),
    Equipment(
        equipment_id="Equipo2",
        price_column="Price_Equipo2",
        predictor_column="Price_Z",
        description=(
            "Equipo crítico tipo 2 — predictor principal: Price_Z "
            "(correlación Pearson 0.983, cointegración p=0.008)"
        ),
    ),
]


def main() -> None:
    """
    Función principal que ejecuta el pipeline completo de forecast.

    Raises:
        DomainError: Si ocurre un error en el pipeline de forecast.
        SystemExit: Si ocurre un error no recuperable.
    """
    logger.info("=" * 60)
    logger.info("Iniciando pipeline de forecast — construction-cost-forecasting")
    logger.info("=" * 60)
    logger.info("Configuración:")
    logger.info("  Data path:      %s", DATA_PATH)
    logger.info("  Models path:    %s", MODELS_PATH)
    logger.info("  Forecast start: %s", FORECAST_START)
    logger.info("  Forecast steps: %d (~7 meses)", FORECAST_STEPS)
    logger.info("  Simulaciones:   %d (Monte Carlo)", N_SIMULATIONS)
    logger.info("  Equipos:        %d", len(EQUIPMENTS))

    try:
        # Inicializar repositorios
        historical_repo = CSVHistoricalRepository(DATA_PATH)
        forecast_repo = CSVForecastRepository(DATA_PATH)

        # Inicializar pipeline
        config = ForecastPipelineConfig(
            models_path=MODELS_PATH,
            forecast_start=FORECAST_START,
            forecast_steps=FORECAST_STEPS,
            n_simulations=N_SIMULATIONS,
            random_seed=RANDOM_SEED,
        )

        pipeline = RunForecastPipeline(
            historical_repo=historical_repo,
            forecast_repo=forecast_repo,
            config=config,
        )

        # Ejecutar pipeline
        results = pipeline.execute(EQUIPMENTS)

        # Reportar resultados
        logger.info("=" * 60)
        logger.info("Resultados del forecast:")
        logger.info("=" * 60)

        for equipment_id, df in results.items():
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            central_col = next(
                (c for c in numeric_cols if "central" in c),
                numeric_cols[0],
            )
            lower_col = next(
                (c for c in numeric_cols if "lower" in c),
                None,
            )
            upper_col = next(
                (c for c in numeric_cols if "upper" in c),
                None,
            )

            # Calcular amplitud IC como % del precio para
            # determinar horizonte confiable
            if lower_col and upper_col:
                df["amplitud_pct"] = (
                    (df[upper_col] - df[lower_col]) / df[central_col] * 100
                )
                reliable_days = int((df["amplitud_pct"] < 25).sum())
                reliable_months = max(1, reliable_days // 21) if reliable_days > 0 else 0
            else:
                reliable_days = 0
                reliable_months = 0

            logger.info("%s:", equipment_id)
            logger.info(
                "  Puntos calculados:    %d (horizonte máximo ~7 meses)",
                len(df),
            )
            logger.info(
                "  Precio central medio: %.2f",
                df[central_col].mean(),
            )
            if lower_col and upper_col:
                logger.info(
                    "  IC 95%% primer mes:   [%.2f, %.2f]",
                    df[lower_col].iloc[0],
                    df[upper_col].iloc[0],
                )
            logger.info(
                "  Horizonte confiable:  ~%d mes/es (%d días con IC < 25%%)",
                reliable_months,
                reliable_days,
            )
            logger.info(
                "Más de %d mes: usar solo como referencia de escenarios",
                reliable_months,
            )

        logger.info("=" * 60)
        logger.info("Pipeline completado exitosamente")
        logger.info("=" * 60)

    except DomainError as e:
        logger.error("Error de dominio en el pipeline: %s", e.message)
        raise SystemExit(1) from e
    except Exception as e:
        logger.error("Error inesperado en el pipeline: %s", str(e))
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()