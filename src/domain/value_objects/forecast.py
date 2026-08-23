"""
Value object que representa el resultado de un forecast
de precio de equipo con intervalos de confianza.

Un value object es inmutable y su identidad está definida
por sus valores, no por un identificador único.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ConfidenceInterval:
    """
    Representa un intervalo de confianza para una predicción.

    Attributes:
        lower: Límite inferior del intervalo.
        upper: Límite superior del intervalo.
        confidence_level: Nivel de confianza (ej: 0.95 para IC 95%).

    Example:
        >>> ic = ConfidenceInterval(lower=414.0, upper=488.0, confidence_level=0.95)
        >>> print(ic.width)
        74.0
    """

    lower: float
    upper: float
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        """
        Valida que el intervalo sea válido.

        Raises:
            ValueError: Si lower >= upper o confidence_level no está en (0, 1).
        """
        if self.lower >= self.upper:
            raise ValueError(
                f"El límite inferior ({self.lower}) debe ser menor "
                f"que el límite superior ({self.upper})."
            )
        if not 0 < self.confidence_level < 1:
            raise ValueError(
                f"El nivel de confianza ({self.confidence_level}) "
                f"debe estar entre 0 y 1."
            )

    @property
    def width(self) -> float:
        """
        Calcula la amplitud del intervalo de confianza.

        Returns:
            Diferencia entre límite superior e inferior.
        """
        return self.upper - self.lower

    @property
    def width_pct(self) -> float:
        """
        Calcula la amplitud como porcentaje del punto medio.

        Returns:
            Amplitud relativa al punto medio del intervalo en %.
        """
        midpoint = (self.upper + self.lower) / 2
        return (self.width / midpoint) * 100 if midpoint != 0 else 0.0


@dataclass(frozen=True)
class ForecastPoint:
    """
    Representa un punto de forecast para una fecha específica.

    Attributes:
        forecast_date: Fecha de la predicción.
        predicted_price: Precio predicho (valor central).
        confidence_interval: Intervalo de confianza asociado.
        equipment_id: Identificador del equipo predicho.

    Example:
        >>> from datetime import date
        >>> fp = ForecastPoint(
        ...     forecast_date=date(2023, 9, 1),
        ...     predicted_price=451.38,
        ...     confidence_interval=ConfidenceInterval(414.0, 488.0),
        ...     equipment_id="Equipo1"
        ... )
        >>> print(fp.is_reliable)
        True
    """

    forecast_date: date
    predicted_price: float
    confidence_interval: ConfidenceInterval
    equipment_id: str

    def __post_init__(self) -> None:
        """
        Valida que el precio predicho sea positivo.

        Raises:
            ValueError: Si predicted_price es negativo o cero.
        """
        if self.predicted_price <= 0:
            raise ValueError(
                f"El precio predicho ({self.predicted_price}) "
                f"debe ser positivo."
            )

    @property
    def is_reliable(self) -> bool:
        """
        Determina si el forecast es confiable basado en la
        amplitud del IC respecto al precio predicho.

        Un forecast se considera confiable si la amplitud del IC
        es menor al 25% del precio predicho — umbral derivado del
        CV histórico de los equipos (24.71% y 19.11%).

        Returns:
            True si el IC es menor al 25% del precio predicho.
        """
        return (self.confidence_interval.width / self.predicted_price) < 0.25