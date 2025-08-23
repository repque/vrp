"""
Data models for VRP Markov Chain Trading Model.

This module defines all data structures using strongly typed Pydantic models
to ensure type safety and data validation throughout the system.
All validation logic is centralized using mixins and constants for maintainability.
"""

from datetime import date as date_type, datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from .constants import (
    ValidationConstants,
    BusinessConstants,
    DefaultConfiguration,
    ErrorMessages,
    FieldNames,
    validate_vrp_threshold_order,
)
from .validators import (
    PriceValidationMixin,
    VolumeValidationMixin,
    VolatilityValidationMixin,
    ProbabilityValidationMixin,
    PositionSizeValidationMixin,
    EnumValidationMixin,
    BusinessLogicValidationMixin,
    MatrixValidationMixin,
    ConfigurationValidationMixin,
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


class MarketData(
    BaseModel,
    PriceValidationMixin,
    VolumeValidationMixin
):
    """
    Market data container for SPY and VIX daily values.
    
    Provides OHLCV data for SPY with VIX closing values for volatility analysis.
    Includes comprehensive validation for price relationships and data integrity.
    """
    
    date: datetime = Field(description="Trading date")
    spy_open: Decimal = Field(description="SPY opening price")
    spy_high: Decimal = Field(description="SPY high price")
    spy_low: Decimal = Field(description="SPY low price") 
    spy_close: Decimal = Field(description="SPY closing price")
    spy_volume: int = Field(description="SPY trading volume")
    vix_close: Decimal = Field(
        gt=0, 
        le=ValidationConstants.MAX_VIX_VALUE,
        description="VIX closing value"
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
        if not (self.spy_low <= self.spy_open <= self.spy_high and
                self.spy_low <= self.spy_close <= self.spy_high):
            raise ValueError(
                ErrorMessages.INVALID_OHLC_RELATIONSHIP.format(
                    low=self.spy_low,
                    open=self.spy_open,
                    high=self.spy_high,
                    close=self.spy_close
                )
            )
        return self

    def calculate_return(self) -> Decimal:
        """
        Calculate daily return from open to close.
        
        Returns:
            Decimal: Daily return as (close - open) / open
        """
        return (self.spy_close - self.spy_open) / self.spy_open

    def dict(self, **kwargs) -> Dict:
        """Override dict method to handle date serialization."""
        data = super().dict(**kwargs)
        if isinstance(data.get('date'), date_type):
            data['date'] = data['date']
        return data

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


# Alias for backward compatibility with tests
MarketDataPoint = MarketData


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
    spy_return: Decimal = Field(description="SPY daily return")
    realized_vol_30d: Decimal = Field(description="30-day realized volatility")
    implied_vol: Decimal = Field(description="Implied volatility from VIX")
    vrp: Decimal = Field(description="Volatility risk premium ratio")
    vrp_state: VRPState = Field(description="VRP state classification")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = False  # Keep enum instances for proper method access


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
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


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
    
    @field_validator('entropy_score', 'data_quality_score', 'stability_score', 'overall_confidence', mode='before')
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
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


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
    
    @field_validator('transition_probability', 'confidence_score', 'data_quality_score', mode='before')
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
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


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
        
        if not (BusinessConstants.MIN_POSITION_SIZE <= value <= BusinessConstants.MAX_POSITION_SIZE):
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
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = False  # Keep enum instances for proper method access


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
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


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
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


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
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class ConfigurationSettings(BaseModel):
    """
    Configuration settings for the VRP system.
    
    Centralized configuration with defaults from constants and
    comprehensive validation for parameter consistency.
    """
    
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
    vrp_underpriced_threshold: Decimal = Field(
        default=DefaultConfiguration.VRP_UNDERPRICED_THRESHOLD,
        description="VRP underpriced threshold"
    )
    vrp_fair_upper_threshold: Decimal = Field(
        default=DefaultConfiguration.VRP_FAIR_UPPER_THRESHOLD,
        description="VRP fair value upper threshold"
    )
    vrp_normal_upper_threshold: Decimal = Field(
        default=DefaultConfiguration.VRP_NORMAL_UPPER_THRESHOLD,
        description="VRP normal premium upper threshold"
    )
    vrp_elevated_upper_threshold: Decimal = Field(
        default=DefaultConfiguration.VRP_ELEVATED_UPPER_THRESHOLD,
        description="VRP elevated premium upper threshold"
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
        if not (BusinessConstants.MIN_ROLLING_WINDOW_DAYS <= v <= BusinessConstants.MAX_ROLLING_WINDOW_DAYS):
            raise ValueError(
                ErrorMessages.WINDOW_SIZE_INVALID.format(
                    min_days=BusinessConstants.MIN_ROLLING_WINDOW_DAYS,
                    max_days=BusinessConstants.MAX_ROLLING_WINDOW_DAYS,
                    value=v
                )
            )
        return v
    
    @field_validator('max_position_size', 'base_position_size', 'min_confidence_threshold', mode='before')
    @classmethod
    def validate_position_size_bounds(cls, v: Union[float, Decimal]) -> Decimal:
        """Ensure position sizes and confidence bounds are between 0 and 1."""
        value = Decimal(str(v))
        
        if not (BusinessConstants.MIN_POSITION_SIZE <= value <= BusinessConstants.MAX_POSITION_SIZE):
            raise ValueError(
                ErrorMessages.PROBABILITY_OUT_OF_BOUNDS.format(
                    field_name="position size/confidence",
                    value=value
                )
            )
        
        return value
    
    @model_validator(mode='after')
    def validate_threshold_ordering(self) -> 'ConfigurationSettings':
        """Validate that VRP thresholds are in ascending order."""
        thresholds = {
            'underpriced': self.vrp_underpriced_threshold,
            'fair_upper': self.vrp_fair_upper_threshold,
            'normal_upper': self.vrp_normal_upper_threshold,
            'elevated_upper': self.vrp_elevated_upper_threshold,
        }
        
        validate_vrp_threshold_order(thresholds)
        return self
    
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
        # Convert list format thresholds to individual fields
        thresholds = settings.model.vrp_thresholds
        
        return cls(
            min_data_years=settings.data.min_data_years,
            preferred_data_years=settings.data.preferred_data_years,
            rolling_window_days=settings.model.transition_window_days,
            vrp_underpriced_threshold=Decimal(str(thresholds[0])),
            vrp_fair_upper_threshold=Decimal(str(thresholds[1])),
            vrp_normal_upper_threshold=Decimal(str(thresholds[2])),
            vrp_elevated_upper_threshold=Decimal(str(thresholds[3])),
            laplace_smoothing_alpha=Decimal(str(settings.model.laplace_smoothing_alpha)),
            min_confidence_threshold=Decimal(str(settings.model.min_confidence_for_signal)),
            max_position_size=Decimal(str(settings.trading.max_position_size_pct)),
            base_position_size=Decimal(str(settings.trading.base_position_size_pct)),
        )
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


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
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        protected_namespaces = ()  # Allow model_drift_detected field