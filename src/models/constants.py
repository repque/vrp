"""
Constants and configuration values for VRP data models.

This module centralizes all hardcoded values, magic numbers, and configuration
constants to ensure consistency and maintainability across the system.
"""

from decimal import Decimal
from typing import List


class ValidationConstants:
    """Validation constants for data models."""

    # Floating point precision tolerance for financial calculations
    DECIMAL_TOLERANCE = Decimal('0.000001')

    # Matrix validation constants
    MATRIX_SIZE = 5
    PROBABILITY_SUM_TARGET = Decimal('1.0')

    # Minimum positive value for prices and volatilities
    MIN_POSITIVE_VALUE = Decimal('0.0')

    # Maximum reasonable values for validation
    MAX_VIX_VALUE = Decimal('200.0')
    MAX_VOLUME = 10_000_000_000  # 10 billion shares
    MAX_PRICE = Decimal('100000.0')  # $100k per share


class BusinessConstants:
    """Business logic constants for trading operations."""

    # Valid trading signal types
    SIGNAL_TYPES: List[str] = ["BUY_VOL", "SELL_VOL", "HOLD"]

    # Valid entropy trend indicators
    ENTROPY_TRENDS: List[str] = ["INCREASING", "STABLE", "DECREASING"]

    # Valid alert levels for system monitoring
    ALERT_LEVELS: List[str] = ["GREEN", "YELLOW", "RED"]

    # VRP quantile boundaries for adaptive state classification
    VRP_QUANTILE_BOUNDS = [0.1, 0.3, 0.7, 0.9]  # Percentile-based state boundaries

    # Position sizing constraints
    MIN_POSITION_SIZE = Decimal('0.0')
    MAX_POSITION_SIZE = Decimal('1.0')

    # Performance metric bounds
    MIN_WIN_RATE = Decimal('0.0')
    MAX_WIN_RATE = Decimal('1.0')

    # Time window constraints
    MIN_DATA_YEARS = 1
    MAX_DATA_YEARS = 50
    MIN_ROLLING_WINDOW_DAYS = 1
    MAX_ROLLING_WINDOW_DAYS = 365


class DefaultConfiguration:
    """Default configuration values for the VRP system."""

    # Data requirements - synchronized with settings.py
    MIN_DATA_YEARS = 3
    PREFERRED_DATA_YEARS = 5
    ROLLING_WINDOW_DAYS = 60  # Same as transition_window_days in settings.py

    # Volatility calculation constants
    VOLATILITY_ANNUALIZATION_FACTOR = 252  # Trading days per year for volatility annualization

    # VRP adaptive classification - quantile-based boundaries
    VRP_QUANTILE_BOUNDS = BusinessConstants.VRP_QUANTILE_BOUNDS

    # Model parameters - synchronized with settings.py
    LAPLACE_SMOOTHING_ALPHA = Decimal('1.0')  # Matches settings.py default
    MIN_CONFIDENCE_THRESHOLD = Decimal('0.6')  # Matches min_confidence_for_signal

    # Risk management - matches TradingConfig defaults
    BASE_POSITION_SIZE = Decimal('0.02')  # From settings.py base_position_size_pct
    MAX_POSITION_SIZE = Decimal('0.05')   # From settings.py max_position_size_pct
    TARGET_SHARPE_RATIO = Decimal('0.8')

    # System monitoring
    MAX_DATA_FRESHNESS_HOURS = 24
    MAX_MATRIX_AGE_DAYS = 30


class ErrorMessages:
    """Standardized error messages for validation failures."""

    # Price validation messages
    PRICE_MUST_BE_POSITIVE = "{field_name} must be positive, got {value}"
    PRICE_EXCEEDS_MAXIMUM = "{field_name} cannot exceed {max_value}, got {value}"

    # Volume validation messages
    VOLUME_MUST_BE_NON_NEGATIVE = "Volume must be non-negative, got {value}"
    VOLUME_EXCEEDS_MAXIMUM = "Volume cannot exceed {max_value}, got {value}"

    # Probability validation messages
    PROBABILITY_OUT_OF_BOUNDS = "{field_name} must be between 0 and 1, got {value}"

    # Matrix validation messages
    MATRIX_WRONG_SIZE = "Matrix must be {size}x{size}, got {actual_rows}x{actual_cols}"
    ROW_SUM_INVALID = "Row {row_index} must sum to 1.0, got {actual_sum}"
    PROBABILITY_INVALID = "Probability at [{row},{col}] must be between 0 and 1, got {value}"

    # Business logic validation messages
    INVALID_OHLC_RELATIONSHIP = (
        "Invalid OHLC price relationship: low={low}, open={open}, "
        "high={high}, close={close}"
    )
    TRADE_COUNT_MISMATCH = (
        "Winning trades ({winning}) + losing trades ({losing}) "
        "must equal total trades ({total})"
    )
    THRESHOLD_ORDER_INVALID = "VRP thresholds must be in ascending order"
    POSITION_SIZE_INVALID = (
        "Risk adjusted size ({risk_size}) cannot exceed "
        "recommended size ({rec_size})"
    )

    # Configuration validation messages
    INVALID_ENUM_VALUE = "{field_name} must be one of {allowed_values}, got {value}"
    YEAR_RANGE_INVALID = "Data years must be between {min_years} and {max_years}, got {value}"
    WINDOW_SIZE_INVALID = (
        "Rolling window must be between {min_days} and {max_days} days, "
        "got {value}"
    )


class FieldNames:
    """Standardized field names for consistent error reporting."""

    # Market data fields
    SPY_OPEN = "SPY opening price"
    SPY_HIGH = "SPY high price"
    SPY_LOW = "SPY low price"
    SPY_CLOSE = "SPY closing price"
    SPY_VOLUME = "SPY trading volume"
    VIX_CLOSE = "VIX closing value"

    # Volatility fields
    REALIZED_VOLATILITY = "realized volatility"
    IMPLIED_VOLATILITY = "implied volatility"
    VRP_RATIO = "VRP ratio"

    # Performance fields
    PROFIT_FACTOR = "profit factor"
    WIN_RATE = "win rate"
    SHARPE_RATIO = "Sharpe ratio"

    # Position sizing fields
    POSITION_SIZE = "position size"
    RECOMMENDED_SIZE = "recommended position size"
    RISK_ADJUSTED_SIZE = "risk adjusted position size"
