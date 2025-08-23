"""
Reusable validation mixins and utilities for VRP data models.

This module provides centralized validation logic that can be mixed into
data models to ensure consistency and reduce code duplication.
"""

from decimal import Decimal
from typing import List, Union

from pydantic import field_validator, model_validator

from .constants import BusinessConstants, ErrorMessages, FieldNames, ValidationConstants


class PriceValidationMixin:
    """Mixin for price field validation logic."""

    @field_validator('open', 'high', 'low', 'close', 'iv', mode='before')
    @classmethod
    def validate_positive_prices(cls, v: Union[float, Decimal], info) -> Decimal:
        """Ensure price fields are positive with descriptive errors."""
        value = Decimal(str(v))
        field_name = info.field_name

        if value <= ValidationConstants.MIN_POSITIVE_VALUE:
            human_name = getattr(FieldNames, field_name.upper(), field_name)
            raise ValueError(
                ErrorMessages.PRICE_MUST_BE_POSITIVE.format(
                    field_name=human_name,
                    value=value
                )
            )

        if value > ValidationConstants.MAX_PRICE:
            human_name = getattr(FieldNames, field_name.upper(), field_name)
            raise ValueError(
                ErrorMessages.PRICE_EXCEEDS_MAXIMUM.format(
                    field_name=human_name,
                    max_value=ValidationConstants.MAX_PRICE,
                    value=value
                )
            )

        return value


class VolumeValidationMixin:
    """Mixin for volume field validation logic."""

    @field_validator('volume', mode='before')
    @classmethod
    def validate_volume(cls, v: int, info) -> int:
        """Ensure volume is non-negative and within reasonable bounds."""
        if v < 0:
            raise ValueError(
                ErrorMessages.VOLUME_MUST_BE_NON_NEGATIVE.format(value=v)
            )

        if v > ValidationConstants.MAX_VOLUME:
            raise ValueError(
                ErrorMessages.VOLUME_EXCEEDS_MAXIMUM.format(
                    max_value=ValidationConstants.MAX_VOLUME,
                    value=v
                )
            )

        return v


class VolatilityValidationMixin:
    """Mixin for volatility field validation logic."""

    @field_validator('realized_vol_30d', 'implied_vol', 'vrp', mode='before')
    @classmethod
    def validate_positive_volatilities(cls, v: Union[float, Decimal], info) -> Decimal:
        """Ensure volatility fields are positive."""
        value = Decimal(str(v))
        field_name = info.field_name

        if value <= ValidationConstants.MIN_POSITIVE_VALUE:
            if field_name == 'realized_vol_30d':
                human_name = FieldNames.REALIZED_VOLATILITY
            elif field_name == 'implied_vol':
                human_name = FieldNames.IMPLIED_VOLATILITY
            elif field_name == 'vrp':
                human_name = FieldNames.VRP_RATIO
            else:
                human_name = field_name

            raise ValueError(
                ErrorMessages.PRICE_MUST_BE_POSITIVE.format(
                    field_name=human_name,
                    value=value
                )
            )

        return value


class ProbabilityValidationMixin:
    """Mixin for probability and confidence score validation."""

    @classmethod
    def validate_probability_field(cls, v: Union[float, Decimal], field_name: str) -> Decimal:
        """Validate that a probability field is between 0 and 1."""
        value = Decimal(str(v))

        if not (ValidationConstants.MIN_POSITIVE_VALUE <= value <= Decimal('1.0')):
            raise ValueError(
                ErrorMessages.PROBABILITY_OUT_OF_BOUNDS.format(
                    field_name=field_name,
                    value=value
                )
            )

        return value


class PositionSizeValidationMixin:
    """Mixin for position size validation."""

    @classmethod
    def validate_position_size_field(cls, v: Union[float, Decimal], field_name: str) -> Decimal:
        """Ensure position sizes are between 0 and 1."""
        value = Decimal(str(v))

        if not (BusinessConstants.MIN_POSITION_SIZE <=
                value <= BusinessConstants.MAX_POSITION_SIZE):
            raise ValueError(
                ErrorMessages.PROBABILITY_OUT_OF_BOUNDS.format(
                    field_name=field_name,
                    value=value
                )
            )

        return value


class EnumValidationMixin:
    """Mixin for enum field validation."""

    @field_validator('signal_type', mode='before')
    @classmethod
    def validate_signal_type(cls, v: str, info) -> str:
        """Validate signal type is one of allowed values."""
        if v not in BusinessConstants.SIGNAL_TYPES:
            raise ValueError(
                ErrorMessages.INVALID_ENUM_VALUE.format(
                    field_name="signal type",
                    allowed_values=BusinessConstants.SIGNAL_TYPES,
                    value=v
                )
            )
        return v

    @field_validator('entropy_trend', mode='before')
    @classmethod
    def validate_entropy_trend(cls, v: str, info) -> str:
        """Validate entropy trend is one of allowed values."""
        if v not in BusinessConstants.ENTROPY_TRENDS:
            raise ValueError(
                ErrorMessages.INVALID_ENUM_VALUE.format(
                    field_name="entropy trend",
                    allowed_values=BusinessConstants.ENTROPY_TRENDS,
                    value=v
                )
            )
        return v

    @field_validator('alert_level', mode='before')
    @classmethod
    def validate_alert_level(cls, v: str, info) -> str:
        """Validate alert level is one of allowed values."""
        if v not in BusinessConstants.ALERT_LEVELS:
            raise ValueError(
                ErrorMessages.INVALID_ENUM_VALUE.format(
                    field_name="alert level",
                    allowed_values=BusinessConstants.ALERT_LEVELS,
                    value=v
                )
            )
        return v


class BusinessLogicValidationMixin:
    """Mixin for complex business logic validation."""

    @field_validator('profit_factor', mode='before')
    @classmethod
    def validate_profit_factor(cls, v: Union[float, Decimal], info) -> Decimal:
        """Ensure profit factor is positive."""
        value = Decimal(str(v))

        if value <= ValidationConstants.MIN_POSITIVE_VALUE:
            raise ValueError(
                ErrorMessages.PRICE_MUST_BE_POSITIVE.format(
                    field_name=FieldNames.PROFIT_FACTOR,
                    value=value
                )
            )

        return value


class MatrixValidationMixin:
    """Mixin for transition matrix validation."""

    @field_validator('matrix', mode='before')
    @classmethod
    def validate_transition_matrix(
            cls, v: List[List[Union[float, Decimal]]], info) -> List[List[Decimal]]:
        """Validate transition matrix properties with detailed error messages."""
        # Convert to Decimal matrix
        matrix = [[Decimal(str(cell)) for cell in row] for row in v]

        # Validate dimensions
        if len(matrix) != ValidationConstants.MATRIX_SIZE:
            raise ValueError(
                ErrorMessages.MATRIX_WRONG_SIZE.format(
                    size=ValidationConstants.MATRIX_SIZE,
                    actual_rows=len(matrix),
                    actual_cols=len(matrix[0]) if matrix else 0
                )
            )

        # Validate each row
        for i, row in enumerate(matrix):
            if len(row) != ValidationConstants.MATRIX_SIZE:
                raise ValueError(
                    ErrorMessages.MATRIX_WRONG_SIZE.format(
                        size=ValidationConstants.MATRIX_SIZE,
                        actual_rows=len(matrix),
                        actual_cols=len(row)
                    )
                )

            # Validate probabilities
            for j, prob in enumerate(row):
                if not (Decimal('0') <= prob <= Decimal('1')):
                    raise ValueError(
                        ErrorMessages.PROBABILITY_INVALID.format(
                            row=i, col=j, value=prob
                        )
                    )

            # Validate row sums to 1 (within tolerance)
            row_sum = sum(row)
            tolerance = ValidationConstants.DECIMAL_TOLERANCE
            target = ValidationConstants.PROBABILITY_SUM_TARGET

            if not (target - tolerance <= row_sum <= target + tolerance):
                raise ValueError(
                    ErrorMessages.ROW_SUM_INVALID.format(
                        row_index=i,
                        actual_sum=row_sum
                    )
                )

        return matrix


class ConfigurationValidationMixin:
    """Mixin for configuration parameter validation."""

    @field_validator('min_data_years', 'preferred_data_years', mode='before')
    @classmethod
    def validate_data_years(cls, v: int, info) -> int:
        """Validate data year requirements."""
        if not (BusinessConstants.MIN_DATA_YEARS <= v <= BusinessConstants.MAX_DATA_YEARS):
            raise ValueError(
                ErrorMessages.YEAR_RANGE_INVALID.format(
                    min_years=BusinessConstants.MIN_DATA_YEARS,
                    max_years=BusinessConstants.MAX_DATA_YEARS,
                    value=v
                )
            )
        return v

    @field_validator('rolling_window_days', mode='before')
    @classmethod
    def validate_rolling_window(cls, v: int, info) -> int:
        """Validate rolling window size."""
        if not (BusinessConstants.MIN_ROLLING_WINDOW_DAYS <=
                v <= BusinessConstants.MAX_ROLLING_WINDOW_DAYS):
            raise ValueError(
                ErrorMessages.WINDOW_SIZE_INVALID.format(
                    min_days=BusinessConstants.MIN_ROLLING_WINDOW_DAYS,
                    max_days=BusinessConstants.MAX_ROLLING_WINDOW_DAYS,
                    value=v
                )
            )
        return v


def create_cross_field_validator(validation_func, fields: List[str]):
    """
    Factory function to create cross-field validators.

    Args:
        validation_func: Function that takes model instance and validates
        fields: List of field names this validator depends on

    Returns:
        Decorated model validator
    """
    def validator(cls):
        return model_validator(mode='after')(validation_func)

    return validator
