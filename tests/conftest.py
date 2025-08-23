"""
Pytest configuration and shared fixtures for VRP Trading System tests.

This module provides comprehensive test fixtures that can be reused across
all test modules, ensuring consistent test data and reducing duplication.
"""

import numpy as np
import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List
from unittest.mock import Mock

from src.config.settings import Settings, TradingConfig
from src.models.data_models import (
    MarketData,
    VolatilityData,
    VRPState,
    TransitionMatrix,
    ModelPrediction,
    TradingSignal,
    PerformanceMetrics,
    ConfigurationSettings,
)


@pytest.fixture
def vrp_config():
    """Create comprehensive VRP trading configuration for testing."""
    config = Mock(spec=TradingConfig)
    
    # Model configuration
    config.model = Mock()
    config.model.vrp_underpriced_threshold = Decimal('0.90')
    config.model.vrp_fair_upper_threshold = Decimal('1.10')
    config.model.vrp_normal_upper_threshold = Decimal('1.30')
    config.model.vrp_elevated_upper_threshold = Decimal('1.50')
    config.model.laplace_smoothing_alpha = Decimal('0.01')
    config.model.rolling_window_days = 60
    config.model.min_confidence_threshold = Decimal('0.6')
    config.model.min_signal_strength = Decimal('0.7')
    
    # Risk management configuration
    config.risk = Mock()
    config.risk.max_position_size = Decimal('0.25')
    config.risk.max_drawdown_limit = Decimal('0.15')
    config.risk.position_sizing_method = "FIXED_FRACTION"
    
    # Data configuration
    config.data = Mock()
    config.data.min_data_years = 3
    config.data.preferred_data_years = 5
    
    # Performance targets
    config.performance = Mock()
    config.performance.target_sharpe_ratio = Decimal('0.8')
    config.performance.target_profit_factor = Decimal('1.3')
    config.performance.target_extreme_state_precision = Decimal('0.6')
    
    return config


@pytest.fixture
def realistic_market_data():
    """Create realistic market data with various volatility regimes."""
    base_date = date(2023, 1, 1)
    data = []
    
    # Start with base values
    spy_price = Decimal('400.0')
    vix_value = Decimal('20.0')
    
    # Create 252 trading days (1 year) with different volatility regimes
    for i in range(252):
        current_date = base_date + timedelta(days=i)
        
        # Create different volatility regimes throughout the year
        if i < 60:  # Low volatility period
            daily_return = np.random.normal(0.0005, 0.01)  # ~1% daily vol
            vix_multiplier = np.random.uniform(0.9, 1.1)
        elif i < 120:  # Normal volatility period
            daily_return = np.random.normal(0.0003, 0.015)  # ~1.5% daily vol
            vix_multiplier = np.random.uniform(0.95, 1.2)
        elif i < 180:  # High volatility period
            daily_return = np.random.normal(-0.0002, 0.025)  # ~2.5% daily vol
            vix_multiplier = np.random.uniform(1.1, 1.8)
        else:  # Return to normal
            daily_return = np.random.normal(0.0004, 0.012)  # ~1.2% daily vol
            vix_multiplier = np.random.uniform(0.95, 1.15)
        
        # Update prices
        spy_price = spy_price * (1 + Decimal(str(daily_return)))
        vix_value = max(Decimal('10.0'), vix_value * Decimal(str(vix_multiplier)))
        
        # Create OHLC data
        open_price = spy_price * Decimal('0.999')
        high_price = spy_price * Decimal(str(1 + abs(daily_return) * 0.5))
        low_price = spy_price * Decimal(str(1 - abs(daily_return) * 0.5))
        
        point = MarketData(
            date=current_date,
            open=open_price,
            high=max(high_price, low_price, spy_price),
            low=min(high_price, low_price, spy_price),
            close=spy_price,
            volume=int(np.random.uniform(50_000_000, 150_000_000)),
            iv=vix_value
        )
        
        data.append(point)
    
    return data


@pytest.fixture
def volatility_metrics_sequence():
    """Create sequence of volatility metrics covering all VRP states."""
    base_date = date(2023, 1, 1)
    metrics = []
    
    # Pattern that cycles through all states
    vrp_values = [0.85, 0.95, 1.15, 1.35, 1.55, 1.45, 1.25, 1.05, 0.88, 1.05] * 12
    states = [
        VRPState.UNDERPRICED, VRPState.FAIR_VALUE, VRPState.NORMAL_PREMIUM,
        VRPState.ELEVATED_PREMIUM, VRPState.EXTREME_PREMIUM, VRPState.ELEVATED_PREMIUM,
        VRPState.NORMAL_PREMIUM, VRPState.FAIR_VALUE, VRPState.UNDERPRICED, VRPState.FAIR_VALUE
    ] * 12
    
    for i, (vrp, state) in enumerate(zip(vrp_values, states)):
        # Calculate realistic volatilities based on VRP
        implied_vol = Decimal('0.20')  # Base 20% implied vol
        realized_vol = implied_vol / Decimal(str(vrp))
        
        metric = VolatilityData(
            date=base_date + timedelta(days=i),
            daily_return=Decimal(str(np.random.normal(0.0005, 0.015))),
            realized_vol_30d=realized_vol,
            implied_vol=implied_vol,
            vrp=Decimal(str(vrp)),
            vrp_state=state
        )
        metrics.append(metric)
    
    return metrics


@pytest.fixture
def sample_transition_matrix():
    """Create sample transition matrix with realistic probabilities."""
    # Realistic transition probabilities based on market behavior
    matrix = [
        # From UNDERPRICED: tends to revert to higher states
        [Decimal('0.2'), Decimal('0.4'), Decimal('0.25'), Decimal('0.1'), Decimal('0.05')],
        # From FAIR_VALUE: most stable, tends to stay or move to normal
        [Decimal('0.1'), Decimal('0.4'), Decimal('0.3'), Decimal('0.15'), Decimal('0.05')],
        # From NORMAL_PREMIUM: can move in either direction
        [Decimal('0.05'), Decimal('0.25'), Decimal('0.35'), Decimal('0.25'), Decimal('0.1')],
        # From ELEVATED_PREMIUM: higher chance of extremes or reversion
        [Decimal('0.05'), Decimal('0.15'), Decimal('0.2'), Decimal('0.35'), Decimal('0.25')],
        # From EXTREME_PREMIUM: strong mean reversion tendency
        [Decimal('0.1'), Decimal('0.2'), Decimal('0.3'), Decimal('0.3'), Decimal('0.1')]
    ]
    
    return TransitionMatrix(
        matrix=matrix,
        observation_count=120,
        window_start=date(2023, 1, 1),
        window_end=date(2023, 3, 1),
        last_updated=datetime.now()
    )


@pytest.fixture
def sample_model_prediction():
    """Create sample model prediction for testing."""
    return ModelPrediction(
        current_date=date(2023, 3, 15),
        current_state=VRPState.FAIR_VALUE,
        predicted_state=VRPState.ELEVATED_PREMIUM,
        transition_probability=Decimal('0.75'),
        confidence_score=Decimal('0.82'),
        entropy=Decimal('0.65'),
        data_quality_score=Decimal('0.88')
    )


@pytest.fixture
def extreme_state_predictions():
    """Create predictions for extreme states (for signal generation testing)."""
    predictions = []
    
    # Extreme premium prediction (should generate SELL_VOL signal)
    predictions.append(ModelPrediction(
        current_date=date(2023, 3, 15),
        current_state=VRPState.ELEVATED_PREMIUM,
        predicted_state=VRPState.EXTREME_PREMIUM,
        transition_probability=Decimal('0.85'),
        confidence_score=Decimal('0.9'),
        entropy=Decimal('0.3'),
        data_quality_score=Decimal('0.92')
    ))
    
    # Underpriced prediction (should generate BUY_VOL signal)
    predictions.append(ModelPrediction(
        current_date=date(2023, 3, 16),
        current_state=VRPState.FAIR_VALUE,
        predicted_state=VRPState.UNDERPRICED,
        transition_probability=Decimal('0.8'),
        confidence_score=Decimal('0.85'),
        entropy=Decimal('0.4'),
        data_quality_score=Decimal('0.9')
    ))
    
    # Low confidence prediction (should not generate signal)
    predictions.append(ModelPrediction(
        current_date=date(2023, 3, 17),
        current_state=VRPState.NORMAL_PREMIUM,
        predicted_state=VRPState.EXTREME_PREMIUM,
        transition_probability=Decimal('0.45'),
        confidence_score=Decimal('0.3'),
        entropy=Decimal('1.2'),
        data_quality_score=Decimal('0.6')
    ))
    
    return predictions


@pytest.fixture
def sample_trading_signals():
    """Create sample trading signals for testing."""
    signals = []
    
    # Strong sell volatility signal
    signals.append(TradingSignal(
        date=date(2023, 3, 15),
        signal_type="SELL_VOL",
        current_state=VRPState.ELEVATED_PREMIUM,
        predicted_state=VRPState.EXTREME_PREMIUM,
        signal_strength=Decimal('0.9'),
        confidence_score=Decimal('0.85'),
        recommended_position_size=Decimal('0.2'),
        risk_adjusted_size=Decimal('0.15'),
        reason="High confidence transition to extreme premium state"
    ))
    
    # Buy volatility signal
    signals.append(TradingSignal(
        date=date(2023, 3, 16),
        signal_type="BUY_VOL",
        current_state=VRPState.FAIR_VALUE,
        predicted_state=VRPState.UNDERPRICED,
        signal_strength=Decimal('0.8'),
        confidence_score=Decimal('0.75'),
        recommended_position_size=Decimal('0.15'),
        risk_adjusted_size=Decimal('0.12'),
        reason="Volatility appears underpriced, potential mean reversion"
    ))
    
    # Hold signal
    signals.append(TradingSignal(
        date=date(2023, 3, 17),
        signal_type="HOLD",
        current_state=VRPState.NORMAL_PREMIUM,
        predicted_state=VRPState.NORMAL_PREMIUM,
        signal_strength=Decimal('0.3'),
        confidence_score=Decimal('0.5'),
        recommended_position_size=Decimal('0.0'),
        risk_adjusted_size=Decimal('0.0'),
        reason="No clear directional signal, maintain current positions"
    ))
    
    return signals


@pytest.fixture
def edge_case_market_data():
    """Create edge case market data for robustness testing."""
    edge_cases = []
    
    # Extreme price movements
    extreme_crash = MarketData(
        date=date(2023, 3, 1),
        open=Decimal('400.0'),
        high=Decimal('405.0'),
        low=Decimal('320.0'),  # -20% crash day
        close=Decimal('325.0'),
        volume=500_000_000,
        iv=Decimal('80.0')  # IV spike
    )
    edge_cases.append(extreme_crash)
    
    # Minimal price movement
    flat_day = MarketData(
        date=date(2023, 3, 2),
        open=Decimal('400.0'),
        high=Decimal('400.01'),
        low=Decimal('399.99'),
        close=Decimal('400.0'),
        volume=10_000_000,
        iv=Decimal('8.0')  # Very low IV
    )
    edge_cases.append(flat_day)
    
    # Gap up/down scenarios
    gap_up = MarketData(
        date=date(2023, 3, 3),
        open=Decimal('440.0'),  # 10% gap up
        high=Decimal('445.0'),
        low=Decimal('435.0'),
        close=Decimal('442.0'),
        volume=200_000_000,
        iv=Decimal('15.0')
    )
    edge_cases.append(gap_up)
    
    return edge_cases


@pytest.fixture
def invalid_market_data():
    """Create invalid market data for validation testing."""
    invalid_cases = []
    
    # Negative prices
    try:
        negative_price = MarketData(
            date=date(2023, 3, 1),
            open=Decimal('-400.0'),
            high=Decimal('405.0'),
            low=Decimal('395.0'),
            close=Decimal('400.0'),
            volume=1_000_000,
            iv=Decimal('20.0')
        )
        invalid_cases.append(negative_price)
    except Exception:
        pass  # Expected to fail validation
    
    # High < Low (impossible)
    try:
        impossible_ohlc = MarketData(
            date=date(2023, 3, 2),
            open=Decimal('400.0'),
            high=Decimal('395.0'),  # High < Low
            low=Decimal('405.0'),
            close=Decimal('400.0'),
            volume=1_000_000,
            iv=Decimal('20.0')
        )
        invalid_cases.append(impossible_ohlc)
    except Exception:
        pass  # Expected to fail validation
    
    return invalid_cases


@pytest.fixture
def performance_test_data():
    """Create large dataset for performance testing."""
    base_date = date(2020, 1, 1)
    data = []
    
    # Create 5 years of daily data (1,260 trading days)
    spy_price = Decimal('300.0')
    vix_value = Decimal('18.0')
    
    np.random.seed(42)  # Reproducible random data
    
    for i in range(1260):
        daily_return = np.random.normal(0.0004, 0.016)
        vix_change = np.random.normal(0.0, 0.15)
        
        spy_price = spy_price * (1 + Decimal(str(daily_return)))
        vix_value = max(Decimal('8.0'), vix_value * (1 + Decimal(str(vix_change))))
        
        point = MarketData(
            date=base_date + timedelta(days=i),
            open=spy_price * Decimal('0.999'),
            high=spy_price * Decimal('1.01'),
            low=spy_price * Decimal('0.99'),
            close=spy_price,
            volume=int(np.random.uniform(80_000_000, 120_000_000)),
            iv=vix_value
        )
        
        data.append(point)
    
    return data


@pytest.fixture
def configuration_test_cases():
    """Create various configuration scenarios for testing."""
    configs = {}
    
    # Valid configuration
    configs['valid'] = ConfigurationSettings(
        min_data_years=3,
        preferred_data_years=5,
        rolling_window_days=60,
        vrp_underpriced_threshold=Decimal('0.90'),
        vrp_fair_upper_threshold=Decimal('1.10'),
        vrp_normal_upper_threshold=Decimal('1.30'),
        vrp_elevated_upper_threshold=Decimal('1.50'),
        laplace_smoothing_alpha=Decimal('0.01'),
        min_confidence_threshold=Decimal('0.6'),
        max_position_size=Decimal('0.25'),
        target_sharpe_ratio=Decimal('0.8')
    )
    
    # Edge case - minimal settings
    configs['minimal'] = ConfigurationSettings(
        min_data_years=1,
        rolling_window_days=30,
        vrp_underpriced_threshold=Decimal('0.85'),
        vrp_elevated_upper_threshold=Decimal('1.60'),
        max_position_size=Decimal('0.1'),
        target_sharpe_ratio=Decimal('0.5')
    )
    
    # Aggressive settings
    configs['aggressive'] = ConfigurationSettings(
        min_data_years=2,
        rolling_window_days=30,
        vrp_underpriced_threshold=Decimal('0.95'),
        vrp_fair_upper_threshold=Decimal('1.05'),
        vrp_normal_upper_threshold=Decimal('1.15'),
        vrp_elevated_upper_threshold=Decimal('1.25'),
        min_confidence_threshold=Decimal('0.4'),
        max_position_size=Decimal('0.5'),
        target_sharpe_ratio=Decimal('1.2')
    )
    
    return configs


@pytest.fixture
def mock_external_apis():
    """Create mock external API responses for testing."""
    mocks = {}
    
    # Mock Yahoo Finance API response
    mocks['yahoo_finance'] = {
        'SPY': {
            '2023-01-01': {'Open': 400.0, 'High': 405.0, 'Low': 395.0, 'Close': 402.0, 'Volume': 100000000},
            '2023-01-02': {'Open': 402.0, 'High': 407.0, 'Low': 397.0, 'Close': 405.0, 'Volume': 110000000}
        },
        'VIX': {
            '2023-01-01': {'Close': 20.0},
            '2023-01-02': {'Close': 19.5}
        }
    }
    
    # Mock FRED API response (for risk-free rates)
    mocks['fred'] = {
        'DGS3MO': {
            '2023-01-01': 4.5,
            '2023-01-02': 4.6
        }
    }
    
    return mocks