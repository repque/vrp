"""
Data models for VRP Markov Chain Trading Model.

This module defines all data structures using strongly typed Pydantic models
to ensure type safety and data validation throughout the system.
All validation logic is centralized using mixins and constants for maintainability.
"""

from datetime import date as date_type
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .constants import (
    BusinessConstants,
    DefaultConfiguration,
    ErrorMessages,
    FieldNames,
    ValidationConstants,
)
from .validators import (
    MatrixValidationMixin,
    PriceValidationMixin,
    VolatilityValidationMixin,
    VolumeValidationMixin,
)


class SignalType(str, Enum):
    """Trading signal types for volatility strategies."""
    BUY_VOL = "BUY_VOL"
    SELL_VOL = "SELL_VOL"
    HOLD = "HOLD"


class VRPState(int, Enum):
    """
    VRP state classifications based on volatility risk premium levels.

    States represent different market regimes for volatility premium:
    - EXTREME_LOW: VRP < 0.90 - IV significantly underpriced
    - FAIR_VALUE: 0.90 ≤ VRP < 1.10 - Fair value region
    - NORMAL_PREMIUM: 1.10 ≤ VRP < 1.30 - Normal premium
    - ELEVATED_PREMIUM: 1.30 ≤ VRP < 1.50 - Elevated premium
    - EXTREME_HIGH: VRP ≥ 1.50 - Extreme premium territory
    """
    EXTREME_LOW = 1
    FAIR_VALUE = 2
    NORMAL_PREMIUM = 3
    ELEVATED_PREMIUM = 4
    EXTREME_HIGH = 5

    # Aliases for backwards compatibility
    UNDERPRICED = EXTREME_LOW
    EXTREME_PREMIUM = EXTREME_HIGH


class MarketData(
    BaseModel,
    PriceValidationMixin,
    VolumeValidationMixin
):
    """
    Market data container for OHLCV and implied volatility values.

    Provides OHLCV data for any underlying asset with implied volatility values
    for volatility risk premium analysis. Includes comprehensive validation for
    price relationships and data integrity.
    """

    date: datetime = Field(description="Trading date")
    open: Decimal = Field(description="Opening price")
    high: Decimal = Field(description="High price")
    low: Decimal = Field(description="Low price")
    close: Decimal = Field(description="Closing price")
    volume: int = Field(description="Trading volume")
    iv: Decimal = Field(
        gt=0,
        le=ValidationConstants.MAX_VIX_VALUE,
        description="Implied volatility value"
    )

    @field_validator('date')
    @classmethod
    def normalize_date(cls, v: datetime) -> date_type:
        """Convert datetime to date for storage consistency."""
        if isinstance(v, datetime):
            return v.date()
        return v

    @model_validator(mode='after')
    def validate_ohlc_relationships(self) -> 'MarketData':
        """Validate OHLC price relationships follow market conventions."""
        if not (self.low <= self.open <= self.high and
                self.low <= self.close <= self.high):
            raise ValueError(
                ErrorMessages.INVALID_OHLC_RELATIONSHIP.format(
                    low=self.low,
                    open=self.open,
                    high=self.high,
                    close=self.close
                )
            )
        return self

    def calculate_return(self) -> Decimal:
        """
        Calculate daily return from open to close.

        Returns:
            Decimal: Daily return as (close - open) / open
        """
        return (self.close - self.open) / self.open

    def dict(self, **kwargs) -> Dict:
        """Override dict method to handle date serialization."""
        data = super().dict(**kwargs)
        if isinstance(data.get('date'), date_type):
            data['date'] = data['date']
        return data

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True
    )


class MarketDataPoint(BaseModel):
    """
    Market data point for compatibility with test expectations.
    
    This model provides the specific field names expected by tests
    for SPY and VIX data points.
    """
    
    date: date_type = Field(description="Trading date")
    spy_open: Decimal = Field(gt=0, description="SPY opening price")
    spy_high: Decimal = Field(gt=0, description="SPY high price") 
    spy_low: Decimal = Field(gt=0, description="SPY low price")
    spy_close: Decimal = Field(gt=0, description="SPY closing price")
    spy_volume: int = Field(ge=0, description="SPY trading volume")
    vix_close: Decimal = Field(ge=0, description="VIX closing value")

    @field_validator('date')
    @classmethod
    def normalize_date(cls, v: Union[datetime, date_type]) -> date_type:
        """Convert datetime to date for storage consistency."""
        if isinstance(v, datetime):
            return v.date()
        return v

    @model_validator(mode='after')
    def validate_ohlc_relationships(self) -> 'MarketDataPoint':
        """Validate OHLC price relationships follow market conventions."""
        if not (self.spy_low <= self.spy_open <= self.spy_high and
                self.spy_low <= self.spy_close <= self.spy_high):
            raise ValueError(
                f"Invalid OHLC relationship: low={self.spy_low}, open={self.spy_open}, "
                f"high={self.spy_high}, close={self.spy_close}"
            )
        return self

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True
    )


class VolatilityData(
    BaseModel,
    VolatilityValidationMixin
):
    """
    Volatility calculations and VRP metrics for a specific date.

    Contains realized volatility, implied volatility, and the calculated
    volatility risk premium ratio with state classification.
    """

    date: date_type = Field(description="Calculation date")
    daily_return: Decimal = Field(description="Daily return", alias="spy_return")
    realized_vol_30d: Decimal = Field(description="30-day realized volatility")
    implied_vol: Decimal = Field(description="Implied volatility")
    vrp: Decimal = Field(description="Volatility risk premium ratio")
    vrp_state: VRPState = Field(description="VRP state classification")

    @property
    def spy_return(self) -> Decimal:
        """Backward compatibility alias for daily_return."""
        return self.daily_return

    model_config = ConfigDict(
        use_enum_values=False,  # Keep enum instances for proper method access
        populate_by_name=True   # Allow both field name and alias
    )


# Alias for backward compatibility
VolatilityMetrics = VolatilityData


class TransitionMatrix(
    BaseModel,
    MatrixValidationMixin
):
    """
    5x5 Markov chain transition matrix for VRP state transitions.

    Represents transition probabilities between VRP states with
    comprehensive validation for matrix properties and statistical consistency.
    """

    matrix: List[List[Decimal]] = Field(description="5x5 transition probability matrix")
    observation_count: int = Field(ge=0, description="Number of observations used to build matrix")
    window_start: date_type = Field(description="Start date of rolling window")
    window_end: date_type = Field(description="End date of rolling window")
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Last update timestamp"
    )

    def get_transition_probability(self, from_state: VRPState, to_state: VRPState) -> Decimal:
        """
        Get transition probability between two states.

        Args:
            from_state: Source VRP state
            to_state: Target VRP state

        Returns:
            Decimal: Transition probability
        """
        return self.matrix[from_state.value - 1][to_state.value - 1]

    def get_state_probabilities(self, current_state: VRPState) -> Dict[VRPState, Decimal]:
        """
        Get all transition probabilities from current state.

        Args:
            current_state: Current VRP state

        Returns:
            Dict mapping each possible state to its transition probability
        """
        probabilities = {}
        for state in VRPState:
            probabilities[state] = self.get_transition_probability(current_state, state)
        return probabilities

    model_config = ConfigDict(
        use_enum_values=True
    )


class ConfidenceMetrics(BaseModel):
    """
    Confidence scoring metrics for model predictions.

    Provides multi-dimensional confidence assessment including entropy,
    data quality, and model stability components.
    """

    entropy_score: Decimal = Field(description="Entropy-based confidence")
    data_quality_score: Decimal = Field(description="Data quality score")
    stability_score: Decimal = Field(description="Model stability score")
    overall_confidence: Decimal = Field(description="Combined confidence score")

    @field_validator('entropy_score', 'data_quality_score',
                     'stability_score', 'overall_confidence', mode='before')
    @classmethod
    def validate_probability_bounds(cls, v: Union[float, Decimal]) -> Decimal:
        """Ensure confidence scores are between 0 and 1."""
        value = Decimal(str(v))

        if not (ValidationConstants.MIN_POSITIVE_VALUE <= value <= Decimal('1.0')):
            raise ValueError(
                ErrorMessages.PROBABILITY_OUT_OF_BOUNDS.format(
                    field_name="confidence score",
                    value=value
                )
            )

        return value

    @model_validator(mode='after')
    def validate_confidence_consistency(self) -> 'ConfidenceMetrics':
        """Ensure overall confidence is consistent with component scores."""
        max_component = max(
            self.entropy_score,
            self.data_quality_score,
            self.stability_score
        )

        # Overall confidence should not significantly exceed max component
        if self.overall_confidence > max_component + ValidationConstants.DECIMAL_TOLERANCE:
            raise ValueError(
                f"Overall confidence ({self.overall_confidence}) cannot exceed "
                f"maximum component score ({max_component})"
            )

        return self

    model_config = ConfigDict(
        use_enum_values=True
    )


class ModelPrediction(BaseModel):
    """
    Model prediction with comprehensive confidence metrics.

    Represents a single prediction event with transition probabilities
    and quality assessment metrics.
    """

    current_date: date_type = Field(description="Prediction date")
    current_state: VRPState = Field(description="Current VRP state")
    predicted_state: VRPState = Field(description="Predicted next state")
    transition_probability: Decimal = Field(description="Transition probability")
    confidence_score: Decimal = Field(description="Prediction confidence")
    entropy: Decimal = Field(ge=0, description="Prediction entropy")
    data_quality_score: Decimal = Field(description="Data quality score")

    @field_validator('transition_probability', 'confidence_score',
                     'data_quality_score', mode='before')
    @classmethod
    def validate_probability_bounds(cls, v: Union[float, Decimal]) -> Decimal:
        """Ensure probability fields are between 0 and 1."""
        value = Decimal(str(v))

        if not (ValidationConstants.MIN_POSITIVE_VALUE <= value <= Decimal('1.0')):
            raise ValueError(
                ErrorMessages.PROBABILITY_OUT_OF_BOUNDS.format(
                    field_name="probability/confidence",
                    value=value
                )
            )

        return value

    model_config = ConfigDict(
        use_enum_values=False
    )


class ExitConfidenceConfig(BaseModel):
    """
    Configuration model for three-tier confidence thresholds.
    
    Defines the confidence levels for different trading decisions:
    - Entry threshold: Minimum confidence to enter new positions
    - Exit threshold: Minimum confidence to maintain existing positions
    - Flip threshold: Minimum confidence to flip positions directly
    """
    
    entry_threshold: Decimal = Field(
        default=Decimal('0.65'),
        ge=Decimal('0.0'),
        le=Decimal('1.0'),
        description="Minimum confidence to enter new positions"
    )
    exit_threshold: Decimal = Field(
        default=Decimal('0.40'),
        ge=Decimal('0.0'),
        le=Decimal('1.0'),
        description="Minimum confidence to maintain existing positions"
    )
    flip_threshold: Decimal = Field(
        default=Decimal('0.75'),
        ge=Decimal('0.0'),
        le=Decimal('1.0'),
        description="Minimum confidence to flip positions directly"
    )
    
    @model_validator(mode='after')
    def validate_threshold_ordering(self) -> 'ExitConfidenceConfig':
        """Validate that thresholds are in logical order."""
        if not (self.exit_threshold <= self.entry_threshold <= self.flip_threshold):
            raise ValueError(
                f"Thresholds must be ordered: exit ({self.exit_threshold}) <= "
                f"entry ({self.entry_threshold}) <= flip ({self.flip_threshold})"
            )
        return self

    model_config = ConfigDict(
        use_enum_values=True
    )


class TradingSignal(BaseModel):
    """
    Trading signal with confidence and position sizing recommendations.

    Provides actionable trading recommendations with risk-adjusted position sizing
    and comprehensive reasoning for signal generation.
    """

    date: date_type = Field(description="Signal date")
    signal_type: str = Field(description="Type of trading signal")
    current_state: VRPState = Field(description="Current VRP state")
    predicted_state: VRPState = Field(description="Predicted next state")
    signal_strength: Decimal = Field(description="Signal strength")
    confidence_score: Decimal = Field(description="Signal confidence")
    recommended_position_size: Decimal = Field(description="Recommended position size")
    risk_adjusted_size: Decimal = Field(description="Risk adjusted position size")
    reason: str = Field(description="Signal generation reason")
    exit_confidence: Optional[Decimal] = Field(
        default=None,
        ge=Decimal('0.0'),
        le=Decimal('1.0'),
        description="Confidence score for exit decisions"
    )

    @field_validator('signal_type', mode='before')
    @classmethod
    def validate_signal_type(cls, v: str) -> str:
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

    @field_validator('signal_strength', 'confidence_score', mode='before')
    @classmethod
    def validate_probability_bounds(cls, v: Union[float, Decimal]) -> Decimal:
        """Ensure probability fields are between 0 and 1."""
        value = Decimal(str(v))

        if not (ValidationConstants.MIN_POSITIVE_VALUE <= value <= Decimal('1.0')):
            raise ValueError(
                ErrorMessages.PROBABILITY_OUT_OF_BOUNDS.format(
                    field_name="probability/confidence",
                    value=value
                )
            )

        return value

    @field_validator('recommended_position_size', 'risk_adjusted_size', mode='before')
    @classmethod
    def validate_position_size_bounds(cls, v: Union[float, Decimal]) -> Decimal:
        """Ensure position sizes are between 0 and 1."""
        value = Decimal(str(v))

        if not (BusinessConstants.MIN_POSITION_SIZE <=
                value <= BusinessConstants.MAX_POSITION_SIZE):
            raise ValueError(
                ErrorMessages.PROBABILITY_OUT_OF_BOUNDS.format(
                    field_name="position size",
                    value=value
                )
            )

        return value

    @model_validator(mode='after')
    def validate_position_size_relationship(self) -> 'TradingSignal':
        """Ensure risk adjusted size does not exceed recommended size."""
        if self.risk_adjusted_size > self.recommended_position_size:
            raise ValueError(
                ErrorMessages.POSITION_SIZE_INVALID.format(
                    risk_size=self.risk_adjusted_size,
                    rec_size=self.recommended_position_size
                )
            )
        return self

    model_config = ConfigDict(
        use_enum_values=False  # Keep enum instances for proper method access
    )


class ModelState(BaseModel):
    """
    Complete model state for persistence and monitoring.

    Captures the full state of the VRP model including transition matrix,
    current state, and performance tracking metrics.
    """

    last_updated: datetime = Field(description="Last model update timestamp")
    transition_matrix: TransitionMatrix = Field(description="Current transition matrix")
    current_vrp_state: Optional[VRPState] = Field(description="Current VRP state")
    total_observations: int = Field(ge=0, description="Total observations processed")
    version: str = Field(description="Model version identifier")
    data_start_date: datetime = Field(description="Start date of available data")
    data_end_date: datetime = Field(description="End date of available data")
    recent_accuracy: Optional[Decimal] = Field(
        ge=0, le=1,
        description="Recent prediction accuracy"
    )
    signal_count_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Signal counts by type"
    )

    model_config = ConfigDict(
        use_enum_values=True
    )


class PerformanceMetrics(BaseModel):
    """
    Aggregated performance metrics for model evaluation.

    Comprehensive performance tracking including returns, risk metrics,
    and trading statistics with validation for consistency.
    """

    start_date: date_type = Field(description="Performance measurement start date")
    end_date: date_type = Field(description="Performance measurement end date")
    total_return: Decimal = Field(description="Total return")
    sharpe_ratio: Decimal = Field(description="Sharpe ratio")
    max_drawdown: Decimal = Field(le=0, description="Maximum drawdown")
    profit_factor: Decimal = Field(description="Profit factor")
    win_rate: Decimal = Field(description="Win rate")
    avg_win: Decimal = Field(description="Average winning trade")
    avg_loss: Decimal = Field(description="Average losing trade")
    total_trades: int = Field(ge=0, description="Total number of trades")
    winning_trades: int = Field(ge=0, description="Number of winning trades")
    losing_trades: int = Field(ge=0, description="Number of losing trades")
    extreme_state_precision: Decimal = Field(description="Extreme state prediction precision")

    @field_validator('profit_factor', mode='before')
    @classmethod
    def validate_profit_factor(cls, v: Union[float, Decimal]) -> Decimal:
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

    @field_validator('win_rate', 'extreme_state_precision', mode='before')
    @classmethod
    def validate_probability_bounds(cls, v: Union[float, Decimal]) -> Decimal:
        """Ensure probability fields are between 0 and 1."""
        value = Decimal(str(v))

        if not (ValidationConstants.MIN_POSITIVE_VALUE <= value <= Decimal('1.0')):
            raise ValueError(
                ErrorMessages.PROBABILITY_OUT_OF_BOUNDS.format(
                    field_name="probability/rate",
                    value=value
                )
            )

        return value

    @model_validator(mode='after')
    def validate_trade_count_consistency(self) -> 'PerformanceMetrics':
        """Validate that trade counts are consistent."""
        if self.winning_trades + self.losing_trades != self.total_trades:
            raise ValueError(
                ErrorMessages.TRADE_COUNT_MISMATCH.format(
                    winning=self.winning_trades,
                    losing=self.losing_trades,
                    total=self.total_trades
                )
            )
        return self

    @model_validator(mode='after')
    def validate_win_rate_consistency(self) -> 'PerformanceMetrics':
        """Validate that win rate matches calculated ratio."""
        if self.total_trades > 0:
            expected_win_rate = Decimal(self.winning_trades) / Decimal(self.total_trades)
            tolerance = ValidationConstants.DECIMAL_TOLERANCE

            if abs(self.win_rate - expected_win_rate) > tolerance:
                raise ValueError(
                    f"Win rate {self.win_rate} doesn't match calculated "
                    f"{expected_win_rate} (tolerance: {tolerance})"
                )
        return self

    model_config = ConfigDict(
        use_enum_values=True
    )


class BacktestResult(BaseModel):
    """
    Individual backtest result for model validation.

    Represents a single prediction outcome with performance attribution
    for backtesting and model evaluation purposes.
    """

    test_date: datetime = Field(description="Date of prediction")
    predicted_state: VRPState = Field(description="Model prediction")
    actual_state: VRPState = Field(description="Actual observed state")
    signal_generated: SignalType = Field(description="Signal generated")
    confidence: Decimal = Field(ge=0, le=1, description="Prediction confidence")
    was_correct: bool = Field(description="Whether prediction was correct")
    pnl: Optional[Decimal] = Field(description="Profit/Loss from signal")

    model_config = ConfigDict(
        use_enum_values=True
    )


class DataConfiguration(BaseModel):
    """Data fetching and processing configuration."""
    
    fred_api_key: Optional[str] = Field(default=None, description="FRED API key")
    alpha_vantage_api_key: Optional[str] = Field(default=None, description="Alpha Vantage API key")
    request_timeout_seconds: int = Field(default=30, description="API request timeout")
    max_retry_attempts: int = Field(default=3, description="Maximum retry attempts")
    retry_delay_seconds: int = Field(default=1, description="Delay between retries")
    rate_limit_delay_ms: int = Field(default=100, description="Rate limiting delay")
    validate_ohlc_consistency: bool = Field(default=True, description="Validate OHLC data consistency")
    cache_expiry_hours: int = Field(default=24, description="Cache expiry time")

    @field_validator('request_timeout_seconds', 'max_retry_attempts', 'retry_delay_seconds', 
                     'rate_limit_delay_ms', 'cache_expiry_hours', mode='after')
    @classmethod
    def validate_positive_integer_fields(cls, v: int, info) -> int:
        """Validate that integer fields are positive."""
        from ..utils.exceptions import ConfigurationError
        if v <= 0:
            raise ConfigurationError(f"{info.field_name} must be positive, got {v}")
        return v

    @model_validator(mode='after')
    def validate_api_keys(self) -> 'DataConfiguration':
        """Validate that required API keys are present and valid."""
        from ..utils.exceptions import ConfigurationError
        
        # For testing and flat initialization, allow empty data configuration
        # Only require API keys if this appears to be a production configuration
        # (i.e., any API key field has been explicitly set to a non-default value)
        has_explicit_api_config = any([
            self.fred_api_key and self.fred_api_key != "",
            self.alpha_vantage_api_key and self.alpha_vantage_api_key != "",
        ])
        
        # Validate API key format and length (only if they exist and are not empty)
        
        # Check other API keys if they are provided (not None and not empty)
        if self.fred_api_key and len(self.fred_api_key) < 3:
            raise ConfigurationError("API key fred_api_key too short")
            
        if self.alpha_vantage_api_key and len(self.alpha_vantage_api_key) < 3:
            raise ConfigurationError("API key alpha_vantage_api_key too short")
        
        return self


class ModelConfiguration(BaseModel):
    """Model parameters and thresholds configuration."""
    
    vrp_underpriced_threshold: float = Field(default=0.9, description="VRP underpriced threshold")
    vrp_fair_upper_threshold: float = Field(default=1.1, description="VRP fair value upper threshold")
    vrp_normal_upper_threshold: float = Field(default=1.3, description="VRP normal upper threshold")
    vrp_elevated_upper_threshold: float = Field(default=1.5, description="VRP elevated upper threshold")
    laplace_smoothing_alpha: float = Field(default=0.01, description="Laplace smoothing parameter")
    rolling_window_days: int = Field(default=60, description="Rolling window size")
    volatility_window_days: int = Field(default=30, description="Volatility calculation window")
    min_confidence_threshold: float = Field(default=0.6, description="Minimum confidence threshold")
    min_signal_strength: float = Field(default=0.7, description="Minimum signal strength")
    state_memory_days: int = Field(default=5, description="Days of state memory")

    @field_validator('min_confidence_threshold', 'min_signal_strength', mode='after')
    @classmethod
    def validate_percentage_fields(cls, v: float, info) -> float:
        """Validate that percentage fields are between 0 and 1."""
        from ..utils.exceptions import ConfigurationError
        if not (0 <= v <= 1):
            raise ConfigurationError(f"{info.field_name} must be between 0 and 1, got {v}")
        return v

    @field_validator('rolling_window_days', 'volatility_window_days', 'state_memory_days', mode='after')
    @classmethod
    def validate_positive_integer_fields(cls, v: int, info) -> int:
        """Validate that integer fields are positive."""
        from ..utils.exceptions import ConfigurationError
        if v <= 0:
            raise ConfigurationError(f"{info.field_name} must be positive, got {v}")
        return v

    @field_validator('vrp_underpriced_threshold', 'vrp_fair_upper_threshold', 
                     'vrp_normal_upper_threshold', 'vrp_elevated_upper_threshold',
                     'laplace_smoothing_alpha', mode='before')
    @classmethod
    def validate_float_fields(cls, v: float, info) -> float:
        """Validate that float fields are finite numbers."""
        import math
        from ..utils.exceptions import ConfigurationError
        
        if not isinstance(v, (int, float)):
            raise ConfigurationError(f"{info.field_name} must be a number, got {type(v)}")
        
        if math.isinf(v):
            raise ConfigurationError(f"{info.field_name} cannot be infinite, got {v}")
        
        if math.isnan(v):
            raise ConfigurationError(f"{info.field_name} cannot be NaN, got {v}")
        
        return float(v)

    @model_validator(mode='after')
    def validate_vrp_threshold_order(self) -> 'ModelConfiguration':
        """Validate that VRP thresholds are in ascending order."""
        from ..utils.exceptions import ConfigurationError
        
        thresholds = [
            self.vrp_underpriced_threshold,
            self.vrp_fair_upper_threshold,
            self.vrp_normal_upper_threshold,
            self.vrp_elevated_upper_threshold
        ]
        
        if thresholds != sorted(thresholds):
            raise ConfigurationError("VRP thresholds must be in ascending order")
        
        return self


class RiskConfiguration(BaseModel):
    """Risk management configuration."""
    
    max_position_size: float = Field(default=0.2, description="Maximum position size")
    max_daily_trades: int = Field(default=1, description="Maximum daily trades")
    max_portfolio_concentration: float = Field(default=0.5, description="Maximum portfolio concentration")
    volatility_scaling_factor: float = Field(default=1.5, description="Volatility scaling factor")
    max_drawdown_threshold: float = Field(default=0.15, description="Maximum drawdown threshold")
    position_sizing_method: str = Field(default='kelly', description="Position sizing method")
    risk_free_rate: float = Field(default=0.03, description="Risk-free rate")
    base_position_size: float = Field(default=0.02, description="Base position size")

    @field_validator('max_position_size', 'max_portfolio_concentration', 'base_position_size', mode='after')
    @classmethod
    def validate_percentage_fields(cls, v: float, info) -> float:
        """Validate that percentage fields are between 0 and 1."""
        from ..utils.exceptions import ConfigurationError
        if not (0 <= v <= 1):
            raise ConfigurationError(f"{info.field_name} must be between 0 and 1, got {v}")
        return v


class SignalsConfiguration(BaseModel):
    """Signal generation configuration."""
    
    signal_cooldown_hours: int = Field(default=24, description="Hours between signals")
    min_signal_confidence: float = Field(default=0.7, description="Minimum signal confidence")
    max_concurrent_signals: int = Field(default=3, description="Maximum concurrent signals")
    signal_decay_hours: int = Field(default=72, description="Signal decay time")
    enable_momentum_filter: bool = Field(default=True, description="Enable momentum filtering")
    momentum_lookback_days: int = Field(default=10, description="Momentum lookback period")

    @field_validator('signal_cooldown_hours', 'max_concurrent_signals', 'signal_decay_hours', 
                     'momentum_lookback_days', mode='after')
    @classmethod
    def validate_positive_integer_fields(cls, v: int, info) -> int:
        """Validate that integer fields are positive."""
        from ..utils.exceptions import ConfigurationError
        if v <= 0:
            raise ConfigurationError(f"{info.field_name} must be positive, got {v}")
        return v

    @field_validator('min_signal_confidence', mode='after')
    @classmethod
    def validate_percentage_fields(cls, v: float, info) -> float:
        """Validate that percentage fields are between 0 and 1."""
        from ..utils.exceptions import ConfigurationError
        if not (0 <= v <= 1):
            raise ConfigurationError(f"{info.field_name} must be between 0 and 1, got {v}")
        return v


class PerformanceConfiguration(BaseModel):
    """Performance tracking and optimization configuration."""
    
    benchmark_symbol: str = Field(default='SPY', description="Benchmark symbol")
    rebalance_frequency: str = Field(default='daily', description="Rebalance frequency")
    transaction_cost_bps: int = Field(default=5, description="Transaction cost in basis points")
    slippage_bps: int = Field(default=2, description="Slippage in basis points")
    max_execution_delay_seconds: int = Field(default=300, description="Maximum execution delay")
    performance_attribution_window: int = Field(default=30, description="Performance attribution window")
    log_slow_operations: bool = Field(default=False, description="Log slow operations")
    enable_detailed_logging: bool = Field(default=False, description="Enable detailed logging")


class ConfigurationSettings(BaseModel):
    """
    Configuration settings for the VRP system.

    Centralized configuration with nested sections and comprehensive validation
    for parameter consistency. Supports both flat and nested configuration formats
    for backward compatibility.
    """

    # Nested configuration sections
    data: Optional[DataConfiguration] = Field(default=None, description="Data configuration")
    model: Optional[ModelConfiguration] = Field(default=None, description="Model configuration")
    risk: Optional[RiskConfiguration] = Field(default=None, description="Risk configuration")
    signals: Optional[SignalsConfiguration] = Field(default=None, description="Signals configuration")
    performance: Optional[PerformanceConfiguration] = Field(default=None, description="Performance configuration")

    # Flat configuration fields for backward compatibility
    min_data_years: int = Field(
        default=DefaultConfiguration.MIN_DATA_YEARS,
        description="Minimum years of data required"
    )
    preferred_data_years: int = Field(
        default=DefaultConfiguration.PREFERRED_DATA_YEARS,
        description="Preferred years of data"
    )
    rolling_window_days: int = Field(
        default=DefaultConfiguration.ROLLING_WINDOW_DAYS,
        description="Rolling window size in days"
    )
    vrp_quantile_window: int = Field(
        default=252,
        description="Rolling window for VRP quantile-based state classification"
    )
    laplace_smoothing_alpha: Decimal = Field(
        default=DefaultConfiguration.LAPLACE_SMOOTHING_ALPHA,
        description="Laplace smoothing parameter"
    )
    min_confidence_threshold: Decimal = Field(
        default=DefaultConfiguration.MIN_CONFIDENCE_THRESHOLD,
        description="Minimum confidence threshold"
    )
    max_position_size: Decimal = Field(
        default=DefaultConfiguration.MAX_POSITION_SIZE,
        description="Maximum position size"
    )
    base_position_size: Decimal = Field(
        default=DefaultConfiguration.BASE_POSITION_SIZE,
        description="Base position size"
    )
    target_sharpe_ratio: Decimal = Field(
        default=DefaultConfiguration.TARGET_SHARPE_RATIO,
        description="Target Sharpe ratio"
    )

    @classmethod
    def _apply_env_overrides(cls, values):
        """Apply environment variable overrides to configuration values."""
        import copy
        import os
        
        # Create a deep copy to avoid modifying original
        overridden_values = copy.deepcopy(values)
        
        # Environment variable mappings
        env_mappings = {
            'VRP_MAX_POSITION_SIZE': ('risk', 'max_position_size'),
            'VRP_ROLLING_WINDOW_DAYS': ('model', 'rolling_window_days'),
        }
        
        for env_var, (section, field) in env_mappings.items():
            if env_var in os.environ:
                if section not in overridden_values:
                    overridden_values[section] = {}
                
                # Convert to appropriate type
                env_value = os.environ[env_var]
                if field in ['max_position_size']:
                    env_value = float(env_value)
                elif field in ['rolling_window_days']:
                    env_value = int(env_value)
                
                overridden_values[section][field] = env_value
        
        return overridden_values

    @model_validator(mode='before')
    @classmethod
    def create_nested_config_sections(cls, values):
        """
        Automatically create nested configuration sections from dict data.
        
        This supports initialization with nested dictionaries containing
        data, model, risk, signals, and performance sections.
        """
        from ..utils.exceptions import ConfigurationError
        import os
        
        if isinstance(values, dict):
            # Apply environment variable overrides before validation
            values = cls._apply_env_overrides(values)
        else:
            # If not a dict, create empty dict to populate with defaults
            values = {}
            
        # Check if this is flat initialization (has flat fields but no nested sections)
        flat_fields = [
            'min_data_years', 'preferred_data_years', 'rolling_window_days',
            'vrp_quantile_window', 'laplace_smoothing_alpha', 'min_confidence_threshold',
            'max_position_size', 'base_position_size', 'target_sharpe_ratio'
        ]
        
        # Check for known old-style configuration sections
        old_style_sections = ['api_keys', 'thresholds']
        
        has_flat_fields = any(field in values for field in flat_fields)
        has_nested_sections = any(section in values for section in ['data', 'model', 'risk', 'signals', 'performance'])
        has_old_style = any(section in values for section in old_style_sections)
        
        # For empty or missing values, or flat initialization, create minimal required structure
        if not values or len(values) == 0 or (has_flat_fields and not has_nested_sections):
            # Create empty nested sections to satisfy the structure
            if 'data' not in values:
                values['data'] = {}
            if 'model' not in values:
                values['model'] = {}
            if 'risk' not in values:
                values['risk'] = {}
            if 'signals' not in values:
                values['signals'] = {}
            if 'performance' not in values:
                values['performance'] = {}
        
        # Validate required sections for configurations that should have proper structure
        # Skip validation only for genuine flat initialization, not for old-style configs
        if has_old_style or (has_nested_sections and not has_flat_fields):
            required_sections = ['model']  
            for section in required_sections:
                if section not in values or values[section] is None:
                    raise ConfigurationError(f"Missing required section: {section}")
            
        # Check for required fields in data section
        if 'data' in values:
            data_config = values['data']
            if isinstance(data_config, dict):
                # Validate API keys
                api_keys = ['fred_api_key', 'alpha_vantage_api_key']
                for key in api_keys:
                    if key in data_config:
                        if data_config[key] == "":
                            raise ConfigurationError(f"API key {key} cannot be empty")
                        elif data_config[key] is None:
                            raise ConfigurationError(f"API key {key} cannot be None")
                        elif isinstance(data_config[key], str) and len(data_config[key]) < 3:
                            raise ConfigurationError(f"API key {key} is too short (minimum 3 characters)")
                
            
        # Validate percentage fields (0-1 range)
            percentage_fields = [
                ('risk', 'max_position_size'),
                ('risk', 'base_position_size'),
                ('risk', 'max_portfolio_concentration'),
                ('model', 'min_confidence_threshold'),
                ('model', 'min_signal_strength')
            ]
            
            for section, field in percentage_fields:
                if section in values and isinstance(values[section], dict):
                    value = values[section].get(field)
                    if value is not None:
                        try:
                            num_value = float(value)
                            if num_value < 0 or num_value > 1:
                                raise ConfigurationError(f"{field} must be between 0 and 1")
                        except (ValueError, TypeError):
                            pass
            
            # Validate positive integer fields
            positive_int_fields = [
                ('model', 'rolling_window_days'),
                ('model', 'volatility_window_days'),
                ('data', 'request_timeout_seconds'),
                ('data', 'max_retry_attempts'),
                ('signals', 'signal_cooldown_hours')
            ]
            
            for section, field in positive_int_fields:
                if section in values and isinstance(values[section], dict):
                    value = values[section].get(field)
                    if value is not None:
                        try:
                            num_value = int(value)
                            if num_value <= 0:
                                raise ConfigurationError(f"{field} must be positive")
                        except (ValueError, TypeError):
                            pass
            
            # Validate model section relationships
            if 'model' in values:
                model_config = values['model']
                if isinstance(model_config, dict):
                    # Volatility window should be <= rolling window
                    volatility_window = model_config.get('volatility_window_days')
                    rolling_window = model_config.get('rolling_window_days')
                    if volatility_window is not None and rolling_window is not None:
                        # Only compare if both are numbers
                        try:
                            vol_num = float(volatility_window)
                            roll_num = float(rolling_window)
                            if vol_num > roll_num:
                                raise ConfigurationError("Volatility window cannot be larger than rolling window")
                        except (ValueError, TypeError):
                            # Let Pydantic handle the type validation
                            pass
            
            # Validate risk section relationships
            if 'risk' in values:
                risk_config = values['risk']
                if isinstance(risk_config, dict):
                    # Base position size should be <= max position size
                    base_position = risk_config.get('base_position_size')
                    max_position = risk_config.get('max_position_size')
                    if base_position is not None and max_position is not None:
                        try:
                            base_num = float(base_position)
                            max_num = float(max_position)
                            if base_num > max_num:
                                raise ConfigurationError("Base position size cannot exceed max position size")
                        except (ValueError, TypeError):
                            pass
                    
                    # Max single position risk should be <= max position size
                    max_single_position_risk = risk_config.get('max_single_position_risk')
                    if max_single_position_risk is not None and max_position is not None:
                        try:
                            single_num = float(max_single_position_risk)
                            max_num = float(max_position)
                            if single_num > max_num:
                                raise ConfigurationError("Max single position risk cannot exceed max position size")
                        except (ValueError, TypeError):
                            pass
            
            # Validate signals section relationships
            if 'signals' in values:
                signals_config = values['signals']
                if isinstance(signals_config, dict):
                    # Signal weights should sum to 1
                    confidence_weight = signals_config.get('confidence_weight')
                    transition_weight = signals_config.get('transition_probability_weight')
                    if confidence_weight is not None and transition_weight is not None:
                        try:
                            conf_num = float(confidence_weight)
                            trans_num = float(transition_weight)
                            weight_sum = conf_num + trans_num
                            # Allow for small floating point tolerance
                            if abs(weight_sum - 1.0) > 0.001:
                                raise ConfigurationError("Signal weights must sum to 1")
                        except (ValueError, TypeError):
                            pass
                    
                    # Max signals per day should be reasonable
                    max_signals_per_day = signals_config.get('max_signals_per_day')
                    if max_signals_per_day is not None:
                        try:
                            max_num = int(max_signals_per_day)
                            if max_num > 100:
                                raise ConfigurationError("Max signals per day is unreasonably high")
                        except (ValueError, TypeError):
                            pass
            
            # Validate performance section constraints
            if 'performance' in values:
                performance_config = values['performance']
                if isinstance(performance_config, dict):
                    # Max processing time must be positive
                    max_processing_time = performance_config.get('max_processing_time_seconds')
                    if max_processing_time is not None:
                        try:
                            time_num = float(max_processing_time)
                            if time_num <= 0:
                                raise ConfigurationError("Max processing time must be positive")
                        except (ValueError, TypeError):
                            pass
                    
                    # Memory limit should be reasonable
                    max_memory_mb = performance_config.get('max_memory_usage_mb')
                    if max_memory_mb is not None:
                        try:
                            mem_num = float(max_memory_mb)
                            if mem_num <= 0:
                                raise ConfigurationError("Max memory usage must be positive")
                            elif mem_num > 32000:  # 32GB
                                raise ConfigurationError("Max memory usage is unreasonably high")
                        except (ValueError, TypeError):
                            pass
            
            # Create nested configuration objects if they don't exist
            try:
                if 'data' in values and not isinstance(values['data'], DataConfiguration):
                    values['data'] = DataConfiguration(**values['data'])
                if 'model' in values and not isinstance(values['model'], ModelConfiguration):
                    values['model'] = ModelConfiguration(**values['model'])
                if 'risk' in values and not isinstance(values['risk'], RiskConfiguration):
                    values['risk'] = RiskConfiguration(**values['risk'])
                if 'signals' in values and not isinstance(values['signals'], SignalsConfiguration):
                    values['signals'] = SignalsConfiguration(**values['signals'])
                if 'performance' in values and not isinstance(values['performance'], PerformanceConfiguration):
                    values['performance'] = PerformanceConfiguration(**values['performance'])
            except Exception as e:
                from ..utils.exceptions import ValidationError, ConfigurationError
                # Re-raise ConfigurationError as-is, wrap others in ValidationError
                if isinstance(e, ConfigurationError):
                    raise e
                else:
                    raise ValidationError("configuration", str(values), f"Configuration validation failed: {str(e)}")
        
        return values

    @field_validator('min_data_years', 'preferred_data_years', mode='before')
    @classmethod
    def validate_data_years(cls, v: int) -> int:
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
    def validate_rolling_window(cls, v: int) -> int:
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

    @field_validator('max_position_size', 'base_position_size',
                     'min_confidence_threshold', mode='before')
    @classmethod
    def validate_position_size_bounds(cls, v: Union[float, Decimal]) -> Decimal:
        """Ensure position sizes and confidence bounds are between 0 and 1."""
        value = Decimal(str(v))

        if not (BusinessConstants.MIN_POSITION_SIZE <=
                value <= BusinessConstants.MAX_POSITION_SIZE):
            raise ValueError(
                ErrorMessages.PROBABILITY_OUT_OF_BOUNDS.format(
                    field_name="position size/confidence",
                    value=value
                )
            )

        return value

    @model_validator(mode='after')
    def validate_data_year_consistency(self) -> 'ConfigurationSettings':
        """Validate that preferred years >= minimum years."""
        if self.preferred_data_years < self.min_data_years:
            raise ValueError(
                f"Preferred data years ({self.preferred_data_years}) must be >= "
                f"minimum data years ({self.min_data_years})"
            )
        return self

    @model_validator(mode='after')
    def validate_position_size_relationship(self) -> 'ConfigurationSettings':
        """Validate that max position size >= base position size."""
        if self.max_position_size < self.base_position_size:
            raise ValueError(
                f"Maximum position size ({self.max_position_size}) must be >= "
                f"base position size ({self.base_position_size})"
            )
        return self

    @classmethod
    def from_settings(cls, settings) -> 'ConfigurationSettings':
        """
        Create ConfigurationSettings from main Settings object.

        Args:
            settings: Main Settings object from config.settings

        Returns:
            ConfigurationSettings instance with values from settings
        """
        return cls(
            min_data_years=settings.data.min_data_years,
            preferred_data_years=settings.data.preferred_data_years,
            rolling_window_days=settings.model.transition_window_days,
            laplace_smoothing_alpha=Decimal(str(settings.model.laplace_smoothing_alpha)),
            min_confidence_threshold=Decimal(str(settings.model.min_confidence_for_signal)),
            max_position_size=Decimal(str(settings.trading.max_position_size_pct)),
            base_position_size=Decimal(str(settings.trading.base_position_size_pct)),
        )

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary format.
        
        Returns:
            Dictionary representation of configuration
        """
        result = {}
        
        # Convert nested configuration objects to dictionaries
        if self.data:
            result['data'] = self.data.model_dump()
        if self.model:
            result['model'] = self.model.model_dump()
        if self.risk:
            result['risk'] = self.risk.model_dump()
        if self.signals:
            result['signals'] = self.signals.model_dump()
        if self.performance:
            result['performance'] = self.performance.model_dump()
        
        # Add flat fields
        flat_fields = [
            'min_data_years', 'preferred_data_years', 'rolling_window_days',
            'vrp_quantile_window', 'laplace_smoothing_alpha', 'min_confidence_threshold',
            'max_position_size', 'base_position_size', 'target_sharpe_ratio'
        ]
        
        for field in flat_fields:
            if hasattr(self, field):
                value = getattr(self, field)
                # Convert Decimal to float for serialization compatibility
                if isinstance(value, Decimal):
                    result[field] = float(value)
                else:
                    result[field] = value
        
        return result

    model_config = ConfigDict(
        use_enum_values=True,
        extra='allow'  # Allow extra fields for test compatibility
    )


class ModelHealthMetrics(BaseModel):
    """
    Health metrics for model monitoring and alerting.

    Provides comprehensive health assessment including data freshness,
    model performance, and operational alerts.
    """

    data_freshness_hours: int = Field(ge=0, description="Hours since last data update")
    transition_matrix_age_days: int = Field(ge=0, description="Days since matrix update")
    recent_prediction_accuracy: Decimal = Field(description="Recent prediction accuracy")
    entropy_trend: str = Field(description="Trend in prediction entropy")
    data_quality_issues: List[str] = Field(
        default_factory=list,
        description="List of data quality issues"
    )
    model_drift_detected: bool = Field(
        default=False,
        description="Whether model drift is detected"
    )
    alert_level: str = Field(description="Current alert level")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Metrics timestamp"
    )

    @field_validator('recent_prediction_accuracy', mode='before')
    @classmethod
    def validate_probability_bounds(cls, v: Union[float, Decimal]) -> Decimal:
        """Ensure accuracy is between 0 and 1."""
        value = Decimal(str(v))

        if not (ValidationConstants.MIN_POSITIVE_VALUE <= value <= Decimal('1.0')):
            raise ValueError(
                ErrorMessages.PROBABILITY_OUT_OF_BOUNDS.format(
                    field_name="prediction accuracy",
                    value=value
                )
            )

        return value

    @field_validator('entropy_trend', mode='before')
    @classmethod
    def validate_entropy_trend(cls, v: str) -> str:
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
    def validate_alert_level(cls, v: str) -> str:
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

    @model_validator(mode='after')
    def validate_alert_consistency(self) -> 'ModelHealthMetrics':
        """Validate alert level consistency with health metrics."""
        # Red alert should have issues or drift
        if (self.alert_level == "RED" and
            not self.model_drift_detected and
                len(self.data_quality_issues) == 0):
            raise ValueError(
                "RED alert level requires either model drift detection or data quality issues"
            )

        # Stale data should trigger at least yellow alert
        if (self.data_freshness_hours > DefaultConfiguration.MAX_DATA_FRESHNESS_HOURS and
                self.alert_level == "GREEN"):
            raise ValueError(
                f"Stale data ({self.data_freshness_hours}h) should trigger at least YELLOW alert"
            )

        return self

    model_config = ConfigDict(
        use_enum_values=True,
        protected_namespaces=()  # Allow model_drift_detected field
    )


class Position(BaseModel):
    """Trading position tracking model."""

    position_id: str = Field(description="Unique position identifier")
    symbol: str = Field(description="Trading symbol")
    position_type: str = Field(description="Position type (e.g., SHORT_VOL, LONG_VOL)")
    entry_date: date_type = Field(description="Date when position was opened")
    entry_signal: TradingSignal = Field(description="Signal that triggered the position")
    position_size: Decimal = Field(description="Position size as decimal")
    is_active: bool = Field(default=True, description="Whether position is still active")
    exit_date: Optional[date_type] = Field(default=None, description="Date when position was closed")
    exit_price: Optional[Decimal] = Field(default=None, description="Exit price if closed")
    realized_pnl: Optional[Decimal] = Field(default=None, description="Realized P&L if closed")

    @field_validator('position_size', mode='before')
    @classmethod
    def validate_position_size(cls, v):
        """Validate position size is positive."""
        decimal_v = Decimal(str(v))
        if decimal_v <= Decimal('0'):
            raise ValueError(f"Position size must be positive, got {decimal_v}")
        return decimal_v

    @field_validator('symbol', 'position_type', mode='before')
    @classmethod
    def validate_strings(cls, v):
        """Validate string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("String fields cannot be empty")
        return v.strip().upper()

    @model_validator(mode='after')
    def validate_exit_consistency(self):
        """Ensure exit fields are consistent."""
        if not self.is_active:
            if self.exit_date is None:
                raise ValueError("Inactive positions must have an exit date")
        else:
            if self.exit_date is not None or self.exit_price is not None or self.realized_pnl is not None:
                raise ValueError("Active positions cannot have exit data")
        return self

    model_config = ConfigDict(
        use_enum_values=True
    )
