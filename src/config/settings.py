"""
Configuration management for VRP Markov Chain Trading Model.

This module centralizes all configuration parameters, environment variables,
and system settings to ensure consistent behavior across environments.
"""

import os
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
    """Core model configuration parameters."""
    
    # State thresholds for VRP classification
    vrp_thresholds: List[float] = Field(
        default=[0.9, 1.1, 1.3, 1.5],
        description="VRP state boundary thresholds"
    )
    
    # Rolling window parameters
    transition_window_days: int = Field(
        default=60,
        description="Days for transition matrix counting"
    )
    
    realized_vol_window_days: int = Field(
        default=30,
        description="Days for realized volatility calculation"
    )
    
    # Smoothing parameters
    laplace_smoothing_alpha: float = Field(
        default=1.0,
        description="Laplace smoothing parameter for sparse data"
    )
    
    # Confidence thresholds
    min_confidence_for_signal: float = Field(
        default=0.6,
        description="Minimum confidence required for trading signal"
    )
    
    # Annualization factor for volatility
    volatility_annualization_factor: float = Field(
        default=252.0,
        description="Trading days per year for volatility scaling"
    )
    
    # Validation thresholds
    vrp_min_reasonable: float = Field(
        default=0.1,
        description="Minimum reasonable VRP ratio"
    )
    
    vrp_max_reasonable: float = Field(
        default=10.0,
        description="Maximum reasonable VRP ratio"
    )
    
    @field_validator('vrp_thresholds')
    @classmethod
    def validate_thresholds(cls, v):
        """Ensure thresholds are sorted and valid."""
        if len(v) != 4:
            raise ValueError("Must have exactly 4 VRP thresholds")
        if not all(v[i] < v[i+1] for i in range(len(v)-1)):
            raise ValueError("VRP thresholds must be in ascending order")
        return v


class TradingConfig(BaseSettings):
    """Trading and risk management configuration."""
    
    # Position sizing
    base_position_size_pct: float = Field(
        default=0.02,
        description="Base position size as percentage of portfolio"
    )
    
    max_position_size_pct: float = Field(
        default=0.05,
        description="Maximum position size as percentage of portfolio"
    )
    
    # Signal generation
    extreme_low_confidence_threshold: float = Field(
        default=0.3,
        description="Confidence threshold for BUY_VOL signals (State 1)"
    )
    
    extreme_high_confidence_threshold: float = Field(
        default=0.6,
        description="Confidence threshold for SELL_VOL signals (State 5)"
    )
    
    # Transaction costs
    transaction_cost_bps: float = Field(
        default=10.0,
        description="Transaction costs in basis points (0.10%)"
    )


class DataConfig(BaseSettings):
    """Data fetching and processing configuration."""
    
    # API configuration
    yahoo_finance_base_url: str = Field(
        default="https://query1.finance.yahoo.com/v8/finance/chart/",
        description="Yahoo Finance API base URL"
    )
    
    # Data requirements
    min_data_years: int = Field(
        default=3,
        description="Minimum years of data required"
    )
    
    preferred_data_years: int = Field(
        default=5,
        description="Preferred years of data for optimal model performance"
    )
    
    # Data validation
    max_missing_days_pct: float = Field(
        default=0.02,
        description="Maximum percentage of missing days allowed"
    )
    
    # Performance requirements
    max_daily_processing_seconds: int = Field(
        default=30,
        description="Maximum seconds allowed for daily processing"
    )
    
    max_memory_usage_mb: int = Field(
        default=500,
        description="Maximum memory usage in MB"
    )


class DatabaseConfig(BaseSettings):
    """Database connection and storage configuration."""
    
    database_url: str = Field(
        default="sqlite:///vrp_model.db",
        description="Database connection URL"
    )
    
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis cache connection URL"
    )
    
    # Table names
    market_data_table: str = Field(
        default="market_data",
        description="Table name for market data storage"
    )
    
    model_state_table: str = Field(
        default="model_state",
        description="Table name for model state persistence"
    )
    
    signals_table: str = Field(
        default="trading_signals",
        description="Table name for trading signals"
    )


class LoggingConfig(BaseSettings):
    """Logging and monitoring configuration."""
    
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    enable_file_logging: bool = Field(
        default=True,
        description="Enable logging to file"
    )
    
    log_file_path: str = Field(
        default="logs/vrp_model.log",
        description="Path to log file"
    )
    
    max_log_file_mb: int = Field(
        default=100,
        description="Maximum log file size in MB"
    )
    
    log_retention_days: int = Field(
        default=30,
        description="Days to retain log files"
    )


class Settings(BaseSettings):
    """Main settings class combining all configuration sections."""
    
    model: ModelConfig = ModelConfig()
    trading: TradingConfig = TradingConfig()
    data: DataConfig = DataConfig()
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # Environment settings
    environment: str = Field(
        default="development",
        description="Current environment (development, production, testing)"
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def validate_environment_setup() -> None:
    """
    Validate that the environment is properly configured.
    
    Raises:
        ValueError: If critical configuration is missing or invalid
    """
    # Check required directories exist
    os.makedirs("logs", exist_ok=True)
    
    # Validate model parameters
    if settings.model.transition_window_days < 30:
        raise ValueError("Transition window must be at least 30 days")
    
    if settings.model.realized_vol_window_days < 20:
        raise ValueError("Realized volatility window must be at least 20 days")
    
    # Validate trading parameters
    if settings.trading.max_position_size_pct <= settings.trading.base_position_size_pct:
        raise ValueError("Max position size must be greater than base position size")
    
    # Validate data requirements
    if settings.data.min_data_years < 1:
        raise ValueError("Minimum data years must be at least 1")


if __name__ == "__main__":
    # Validate configuration when run directly
    validate_environment_setup()
    print("Configuration validated successfully")
    print(f"Environment: {settings.environment}")
    print(f"VRP Thresholds: {settings.model.vrp_thresholds}")