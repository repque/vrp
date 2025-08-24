"""
Constants and configuration values for VRP data models.

This module centralizes all hardcoded values, magic numbers, and configuration
constants to ensure consistency and maintainability across the system.
"""

from decimal import Decimal
from typing import Any, Dict, List


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


class ConfigurationSection:
    """Base class for configuration sections with validation."""
    
    def __init__(self, **kwargs):
        """Initialize configuration section with provided values."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __setattr__(self, name: str, value) -> None:
        """Set attribute with type validation."""
        # Type validation for numeric fields
        if name.endswith('_days') or name.endswith('_years'):
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric (int or float), got {type(value).__name__}")
            if isinstance(value, (int, float)) and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        
        # Convert strings to Decimal for financial fields
        if name.endswith('_threshold') or name.endswith('_pct') or name.endswith('_alpha'):
            if not isinstance(value, (str, int, float, Decimal)):
                raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
            if isinstance(value, (str, int, float)):
                try:
                    value = Decimal(str(value))
                except (ValueError, TypeError, Exception) as e:
                    raise ValueError(f"Invalid value for {name}: {value}. Must be numeric.") from e
        
        # Range validation for percentage fields
        if name.endswith('_pct') and isinstance(value, (Decimal, float)):
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        
        # Additional validation for threshold fields
        if name.endswith('_threshold') and isinstance(value, (Decimal, float)):
            if value < 0:
                raise ValueError(f"{name} cannot be negative, got {value}")
            if value > 100:  # Reasonable upper bound for VRP thresholds
                raise ValueError(f"{name} exceeds reasonable maximum (100), got {value}")
        
        super().__setattr__(name, value)


class ModelConfiguration(ConfigurationSection):
    """Model parameters and thresholds configuration."""
    
    def __init__(self):
        super().__init__(
            # VRP thresholds
            vrp_underpriced_threshold=Decimal('0.90'),
            vrp_fair_upper_threshold=Decimal('1.10'),
            vrp_normal_upper_threshold=Decimal('1.30'),
            vrp_elevated_upper_threshold=Decimal('1.50'),
            
            # Time windows
            transition_window_days=60,
            realized_vol_window_days=30,
            
            # Model parameters
            laplace_smoothing_alpha=Decimal('1.0'),
            min_confidence_for_signal=Decimal('0.6'),
            
            # Volatility calculation
            volatility_annualization_factor=252.0,
            
            # Validation thresholds
            vrp_min_reasonable=Decimal('0.1'),
            vrp_max_reasonable=Decimal('10.0')
        )
    
    def __setattr__(self, name: str, value) -> None:
        """Set attribute with VRP threshold validation."""
        super().__setattr__(name, value)
        
        # Validate threshold ordering after setting
        if hasattr(self, 'vrp_underpriced_threshold') and hasattr(self, 'vrp_fair_upper_threshold'):
            if self.vrp_underpriced_threshold >= self.vrp_fair_upper_threshold:
                raise ValueError("VRP underpriced threshold must be less than fair upper threshold")


class TradingConfiguration(ConfigurationSection):
    """Trading and risk management configuration."""
    
    def __init__(self):
        super().__init__(
            # Position sizing
            base_position_size_pct=Decimal('0.02'),
            max_position_size_pct=Decimal('0.05'),
            
            # Signal generation
            extreme_low_confidence_threshold=Decimal('0.3'),
            extreme_high_confidence_threshold=Decimal('0.6'),
            
            # Transaction costs
            transaction_cost_bps=Decimal('10.0'),
            
            # Risk management
            target_sharpe_ratio=Decimal('0.8')
        )
    
    def __setattr__(self, name: str, value) -> None:
        """Set attribute with trading parameter validation."""
        super().__setattr__(name, value)
        
        # Validate position sizing
        if hasattr(self, 'base_position_size_pct') and hasattr(self, 'max_position_size_pct'):
            if self.base_position_size_pct >= self.max_position_size_pct:
                raise ValueError("Base position size must be less than max position size")


class DataConfiguration(ConfigurationSection):
    """Data fetching and processing configuration."""
    
    def __init__(self):
        super().__init__(
            # Data requirements
            min_data_years=3,
            preferred_data_years=5,
            
            # Data validation
            max_missing_days_pct=Decimal('0.02'),
            
            # Performance requirements
            max_daily_processing_seconds=30,
            max_memory_usage_mb=500
        )


class DatabaseConfiguration(ConfigurationSection):
    """Database connection and storage configuration."""
    
    def __init__(self):
        super().__init__(
            database_url="sqlite:///vrp_model.db",
            database_path="vrp_model.db",
            connection_timeout=30,
            connection_pool_size=5,
            max_overflow=10,
            enable_wal_mode=True,
            enable_foreign_keys=True,
            backup_retention_days=7,
            auto_vacuum="INCREMENTAL"
        )


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
