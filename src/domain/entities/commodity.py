"""
Entidad de dominio que representa una materia prima (commodity).

En el contexto del caso de negocio, una materia prima es el insumo
cuyo precio de mercado explica el comportamiento del costo de los
equipos de construcción.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Commodity:
    """
    Representa una materia prima con su identificador y
    columna de precio asociada en el dataset.

    Attributes:
        commodity_id: Identificador único de la materia prima
                      (ej: 'X', 'Y', 'Z').
        price_column: Nombre de la columna de precio en el dataset
                      (ej: 'Price_Y').
        is_predictor: Indica si esta materia prima es predictora
                      principal de algún equipo.
        description: Descripción opcional de la materia prima.

    Example:
        >>> commodity_y = Commodity(
        ...     commodity_id="Y",
        ...     price_column="Price_Y",
        ...     is_predictor=True,
        ...     description="Materia prima principal Equipo 1"
        ... )
        >>> print(commodity_y.is_predictor)
        True
    """

    commodity_id: str
    price_column: str
    is_predictor: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        """
        Valida que los campos obligatorios no estén vacíos.

        Raises:
            ValueError: Si commodity_id o price_column están vacíos.
        """
        if not self.commodity_id.strip():
            raise ValueError("commodity_id no puede estar vacío.")
        if not self.price_column.strip():
            raise ValueError("price_column no puede estar vacío.")