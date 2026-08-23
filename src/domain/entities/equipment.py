"""
Entidad de dominio que representa un equipo de construcción.

En el contexto del caso de negocio, un equipo es el activo cuyo
costo de adquisición se desea predecir y proyectar.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Equipment:
    """
    Representa un equipo de construcción con su identificador
    y materia prima asociada.

    Attributes:
        equipment_id: Identificador único del equipo (ej: 'Equipo1').
        price_column: Nombre de la columna de precio en el dataset
                      (ej: 'Price_Equipo1').
        predictor_column: Nombre de la columna de la materia prima
                          predictora principal (ej: 'Price_Y').
        description: Descripción opcional del equipo.

    Example:
        >>> equipo1 = Equipment(
        ...     equipment_id="Equipo1",
        ...     price_column="Price_Equipo1",
        ...     predictor_column="Price_Y",
        ...     description="Equipo crítico tipo 1"
        ... )
        >>> print(equipo1.equipment_id)
        Equipo1
    """

    equipment_id: str
    price_column: str
    predictor_column: str
    description: str = ""

    def __post_init__(self) -> None:
        """
        Valida que los campos obligatorios no estén vacíos.

        Raises:
            ValueError: Si equipment_id, price_column o predictor_column
                        están vacíos.
        """
        if not self.equipment_id.strip():
            raise ValueError("equipment_id no puede estar vacío.")
        if not self.price_column.strip():
            raise ValueError("price_column no puede estar vacío.")
        if not self.predictor_column.strip():
            raise ValueError("predictor_column no puede estar vacío.")