"""
Constants and configuration values for VRP data models.

This module centralizes all hardcoded values, magic numbers, and configuration
constants to ensure consistency and maintainability across the system.
"""

from decimal import Decimal
from typing import List, Tuple


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
    
    # VRP state boundaries (default thresholds)
    VRP_THRESHOLDS = {
        'underpriced': Decimal('0.90'),
        'fair_upper': Decimal('1.10'), 
        'normal_upper': Decimal('1.30'),
        'elevated_upper': Decimal('1.50')
    }
    
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
    
    # VRP classification thresholds - converted from settings.py list format
    VRP_UNDERPRICED_THRESHOLD = BusinessConstants.VRP_THRESHOLDS['underpriced']
    VRP_FAIR_UPPER_THRESHOLD = BusinessConstants.VRP_THRESHOLDS['fair_upper']
    VRP_NORMAL_UPPER_THRESHOLD = BusinessConstants.VRP_THRESHOLDS['normal_upper']
    VRP_ELEVATED_UPPER_THRESHOLD = BusinessConstants.VRP_THRESHOLDS['elevated_upper']
    
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
    INVALID_OHLC_RELATIONSHIP = "Invalid OHLC price relationship: low={low}, open={open}, high={high}, close={close}"
    TRADE_COUNT_MISMATCH = "Winning trades ({winning}) + losing trades ({losing}) must equal total trades ({total})"
    THRESHOLD_ORDER_INVALID = "VRP thresholds must be in ascending order"
    POSITION_SIZE_INVALID = "Risk adjusted size ({risk_size}) cannot exceed recommended size ({rec_size})"
    
    # Configuration validation messages
    INVALID_ENUM_VALUE = "{field_name} must be one of {allowed_values}, got {value}"
    YEAR_RANGE_INVALID = "Data years must be between {min_years} and {max_years}, got {value}"
    WINDOW_SIZE_INVALID = "Rolling window must be between {min_days} and {max_days} days, got {value}"


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


def validate_vrp_threshold_order(thresholds: dict) -> bool:
    """
    Validate that VRP thresholds are in ascending order.
    
    Args:
        thresholds: Dictionary of threshold values
        
    Returns:
        bool: True if thresholds are properly ordered
        
    Raises:
        ValueError: If thresholds are not in ascending order
    """
    threshold_list = [
        thresholds['underpriced'],
        thresholds['fair_upper'], 
        thresholds['normal_upper'],
        thresholds['elevated_upper']
    ]
    
    for i in range(len(threshold_list) - 1):
        if threshold_list[i] >= threshold_list[i + 1]:
            raise ValueError(ErrorMessages.THRESHOLD_ORDER_INVALID)
    
    return True


def get_vrp_state_ranges(thresholds: dict = None) -> List[Tuple[Decimal, Decimal]]:
    """
    Get VRP state ranges based on threshold configuration.
    
    Args:
        thresholds: Optional custom thresholds, defaults to business constants
        
    Returns:
        List of (min, max) tuples for each VRP state
    """
    if thresholds is None:
        thresholds = BusinessConstants.VRP_THRESHOLDS
    
    validate_vrp_threshold_order(thresholds)
    
    return [
        (Decimal('0'), thresholds['underpriced']),  # EXTREME_LOW
        (thresholds['underpriced'], thresholds['fair_upper']),  # FAIR_VALUE
        (thresholds['fair_upper'], thresholds['normal_upper']),  # NORMAL_PREMIUM
        (thresholds['normal_upper'], thresholds['elevated_upper']),  # ELEVATED_PREMIUM
        (thresholds['elevated_upper'], Decimal('999')),  # EXTREME_HIGH
    ]