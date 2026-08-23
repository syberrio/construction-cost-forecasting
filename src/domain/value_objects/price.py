"""
Value object que representa un precio de mercado en una fecha específica.

Un precio es inmutable y su identidad está definida por su valor
y fecha, no por un identificador único.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Price:
    """
    Representa un precio de mercado en una fecha específica.

    Attributes:
        value: Valor numérico del precio.
        price_date: Fecha a la que corresponde el precio.
        source: Fuente o identificador de la variable
                (ej: 'Price_Y', 'Price_Equipo1').

    Example:
        >>> from datetime import date
        >>> price = Price(
        ...     value=451.38,
        ...     price_date=date(2023, 9, 1),
        ...     source="Price_Equipo1"
        ... )
        >>> print(price.is_positive)
        True
    """

    value: float
    price_date: date
    source: str

    def __post_init__(self) -> None:
        """
        Valida que el precio sea positivo y la fuente no esté vacía.

        Raises:
            ValueError: Si value es negativo o cero, o source está vacío.
        """
        if self.value <= 0:
            raise ValueError(
                f"El valor del precio ({self.value}) debe ser positivo."
            )
        if not self.source.strip():
            raise ValueError("La fuente del precio no puede estar vacía.")

    @property
    def is_positive(self) -> bool:
        """
        Verifica si el precio es positivo.

        Returns:
            True si el valor es mayor a cero.
        """
        return self.value > 0

    def scale(self, factor: float) -> "Price":
        """
        Retorna un nuevo Price escalado por un factor.

        Args:
            factor: Factor de escala positivo.

        Returns:
            Nuevo Price con el valor escalado.

        Raises:
            ValueError: Si el factor es negativo o cero.

        Example:
            >>> price = Price(value=100.0, price_date=date.today(), source="X")
            >>> scaled = price.scale(1.1)
            >>> print(scaled.value)
            110.0
        """
        if factor <= 0:
            raise ValueError(
                f"El factor de escala ({factor}) debe ser positivo."
            )
        return Price(
            value=self.value * factor,
            price_date=self.price_date,
            source=self.source
        )