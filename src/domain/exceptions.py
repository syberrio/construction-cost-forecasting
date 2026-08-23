"""
Excepciones personalizadas del dominio de construction-cost-forecasting.

Centralizar las excepciones en el dominio permite que las capas superiores
(application, infrastructure) las capturen y manejen de forma consistente,
sin depender de excepciones genéricas de Python o de librerías externas.

Jerarquía:
    DomainError
    ├── DataNotFoundError
    ├── RepositoryError
    ├── ModelNotFoundError
    ├── ForecastError
    └── ValidationError
"""


class DomainError(Exception):
    """
    Excepción base para todos los errores del dominio.

    Todas las excepciones personalizadas del proyecto heredan
    de esta clase para facilitar el manejo genérico de errores
    de dominio en las capas superiores.

    Example:
        >>> try:
        ...     raise DomainError("Error genérico de dominio")
        ... except DomainError as e:
        ...     print(f"Error de dominio: {e}")
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class DataNotFoundError(DomainError):
    """
    Se lanza cuando no se encuentran datos para el rango
    o entidad solicitada.

    Example:
        >>> raise DataNotFoundError(
        ...     "No hay datos históricos para el rango 2024-01-01 → 2024-03-31"
        ... )
    """
    pass


class RepositoryError(DomainError):
    """
    Se lanza cuando ocurre un error al acceder o persistir
    datos en el repositorio.

    Example:
        >>> raise RepositoryError(
        ...     "Error al leer el archivo historico_equipos.csv"
        ... )
    """
    pass


class ModelNotFoundError(DomainError):
    """
    Se lanza cuando no se encuentra el modelo entrenado
    para un equipo específico.

    Example:
        >>> raise ModelNotFoundError(
        ...     "No se encontró el modelo ARIMAX para Equipo1"
        ... )
    """
    pass


class ForecastError(DomainError):
    """
    Se lanza cuando ocurre un error durante la generación
    del forecast.

    Example:
        >>> raise ForecastError(
        ...     "Error al ejecutar simulación Monte Carlo para Equipo2"
        ... )
    """
    pass


class ValidationError(DomainError):
    """
    Se lanza cuando los datos de entrada no cumplen
    con las validaciones requeridas.

    Example:
        >>> raise ValidationError(
        ...     "La fecha de inicio debe ser anterior a la fecha de fin"
        ... )
    """
    pass