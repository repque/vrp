"""
Unit tests for VRP Trading System data models.

This module contains comprehensive tests for all Pydantic data models,
focusing on validation rules, edge cases, and serialization behavior.
Tests ensure data integrity and type safety throughout the system.
"""

import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List

from pydantic import ValidationError

from src.models.data_models import (
    VRPState,
    SignalType,
    MarketData,
    VolatilityData,
    TransitionMatrix,
    ConfidenceMetrics,
    TradingSignal,
    ModelState,
    PerformanceMetrics,
    BacktestResult,
    ModelHealthMetrics,
    ModelPrediction,
    ConfigurationSettings,
)


class TestVRPState:
    """Test suite for VRPState enum."""
    
    def test_vrp_state_values(self):
        """Test that VRP states have correct values."""
        assert VRPState.EXTREME_LOW.value == 1
        assert VRPState.FAIR_VALUE.value == 2
        assert VRPState.NORMAL_PREMIUM.value == 3
        assert VRPState.ELEVATED_PREMIUM.value == 4
        assert VRPState.EXTREME_HIGH.value == 5
    
    def test_vrp_state_ordering(self):
        """Test that VRP states can be compared."""
        assert VRPState.EXTREME_LOW < VRPState.FAIR_VALUE
        assert VRPState.EXTREME_HIGH > VRPState.ELEVATED_PREMIUM
        assert VRPState.NORMAL_PREMIUM == VRPState.NORMAL_PREMIUM
    
    def test_vrp_state_iteration(self):
        """Test that all VRP states are accessible."""
        states = list(VRPState)
        assert len(states) == 5
        assert all(isinstance(state, VRPState) for state in states)


class TestMarketData:
    """Test suite for MarketData model."""
    
    def test_valid_market_data_creation(self):
        """Test creation of valid market data point."""
        data = MarketData(
            date=datetime(2023, 1, 1),
            open=400.0,
            high=405.0,
            low=395.0,
            close=402.0,
            volume=100_000_000,
            iv=20.0
        )
        
        assert data.date == date(2023, 1, 1)
        assert data.open == Decimal('400.0')
        assert data.high == Decimal('405.0')
        assert data.low == Decimal('395.0')
        assert data.close == Decimal('402.0')
        assert data.volume == 100_000_000
        assert data.iv == Decimal('20.0')
    
    def test_negative_prices_validation(self):
        """Test that negative prices are rejected."""
        with pytest.raises(ValidationError, match="must be positive"):
            MarketData(
                date=date(2023, 1, 1),
                open=Decimal('-400.0'),
                high=Decimal('405.0'),
                low=Decimal('395.0'),
                close=Decimal('402.0'),
                volume=100_000_000,
                iv=Decimal('20.0')
            )
    
    def test_zero_prices_validation(self):
        """Test that zero prices are rejected."""
        with pytest.raises(ValidationError, match="must be positive"):
            MarketData(
                date=date(2023, 1, 1),
                open=Decimal('400.0'),
                high=Decimal('0.0'),
                low=Decimal('395.0'),
                close=Decimal('402.0'),
                volume=100_000_000,
                iv=Decimal('20.0')
            )
    
    def test_negative_volume_validation(self):
        """Test that negative volume is rejected."""
        with pytest.raises(ValidationError, match="must be positive|non-negative"):
            MarketData(
                date=date(2023, 1, 1),
                open=Decimal('400.0'),
                high=Decimal('405.0'),
                low=Decimal('395.0'),
                close=Decimal('402.0'),
                volume=-100_000_000,
                iv=Decimal('20.0')
            )
    
    def test_zero_volume_allowed(self):
        """Test that zero volume is allowed."""
        data = MarketData(
            date=date(2023, 1, 1),
            open=Decimal('400.0'),
            high=Decimal('405.0'),
            low=Decimal('395.0'),
            close=Decimal('402.0'),
            volume=0,
            iv=Decimal('20.0')
        )
        assert data.volume == 0
    
    def test_extreme_values_handling(self):
        """Test handling of extreme but valid values."""
        data = MarketData(
            date=date(2023, 1, 1),
            open=Decimal('0.01'),  # Very small price
            high=Decimal('10000.0'),  # Very large price
            low=Decimal('0.01'),
            close=Decimal('5000.0'),
            volume=1_000_000_000,  # Very large volume
            iv=Decimal('100.0')  # Very high VIX
        )
        
        assert data.open == Decimal('0.01')
        assert data.high == Decimal('10000.0')
        assert data.iv == Decimal('100.0')
    
    def test_serialization_deserialization(self):
        """Test model serialization and deserialization."""
        original = MarketData(
            date=date(2023, 1, 1),
            open=Decimal('400.0'),
            high=Decimal('405.0'),
            low=Decimal('395.0'),
            close=Decimal('402.0'),
            volume=100_000_000,
            iv=Decimal('20.0')
        )
        
        # Serialize to dict
        data_dict = original.dict()
        assert 'date' in data_dict
        assert 'open' in data_dict
        
        # Deserialize from dict
        restored = MarketData(**data_dict)
        assert restored.date == original.date
        assert restored.close == original.close


class TestVolatilityData:
    """Test suite for VolatilityData model."""
    
    def test_valid_volatility_metrics(self):
        """Test creation of valid volatility metrics."""
        metrics = VolatilityData(
            date=date(2023, 1, 1),
            daily_return=Decimal('0.015'),
            realized_vol_30d=Decimal('0.20'),
            implied_vol=Decimal('0.25'),
            vrp=Decimal('1.25'),
            vrp_state=VRPState.NORMAL_PREMIUM
        )
        
        assert metrics.date == date(2023, 1, 1)
        assert metrics.daily_return == Decimal('0.015')
        assert metrics.realized_vol_30d == Decimal('0.20')
        assert metrics.implied_vol == Decimal('0.25')
        assert metrics.vrp == Decimal('1.25')
        assert metrics.vrp_state == VRPState.NORMAL_PREMIUM
    
    def test_negative_volatility_validation(self):
        """Test that negative volatilities are rejected."""
        with pytest.raises(ValidationError, match="must be positive"):
            VolatilityData(
                date=date(2023, 1, 1),
                daily_return=Decimal('0.015'),
                realized_vol_30d=Decimal('-0.20'),
                implied_vol=Decimal('0.25'),
                vrp=Decimal('1.25'),
                vrp_state=VRPState.NORMAL_PREMIUM
            )
    
    def test_zero_volatility_validation(self):
        """Test that zero volatilities are rejected."""
        with pytest.raises(ValidationError, match="must be positive"):
            VolatilityData(
                date=date(2023, 1, 1),
                daily_return=Decimal('0.015'),
                realized_vol_30d=Decimal('0.20'),
                implied_vol=Decimal('0.0'),
                vrp=Decimal('1.25'),
                vrp_state=VRPState.NORMAL_PREMIUM
            )
    
    def test_negative_return_allowed(self):
        """Test that negative returns are allowed."""
        metrics = VolatilityData(
            date=date(2023, 1, 1),
            daily_return=Decimal('-0.025'),  # Negative return allowed
            realized_vol_30d=Decimal('0.20'),
            implied_vol=Decimal('0.25'),
            vrp=Decimal('1.25'),
            vrp_state=VRPState.NORMAL_PREMIUM
        )
        
        assert metrics.daily_return == Decimal('-0.025')
    
    def test_vrp_state_enum_validation(self):
        """Test that VRP state must be valid enum value."""
        with pytest.raises(ValidationError):
            VolatilityData(
                date=date(2023, 1, 1),
                daily_return=Decimal('0.015'),
                realized_vol_30d=Decimal('0.20'),
                implied_vol=Decimal('0.25'),
                vrp=Decimal('1.25'),
                vrp_state="INVALID_STATE"  # Invalid state
            )


class TestTransitionMatrix:
    """Test suite for TransitionMatrix model."""
    
    def test_valid_transition_matrix(self):
        """Test creation of valid transition matrix."""
        matrix = [
            [Decimal('0.2'), Decimal('0.3'), Decimal('0.3'), Decimal('0.15'), Decimal('0.05')],
            [Decimal('0.1'), Decimal('0.4'), Decimal('0.3'), Decimal('0.15'), Decimal('0.05')],
            [Decimal('0.05'), Decimal('0.25'), Decimal('0.4'), Decimal('0.25'), Decimal('0.05')],
            [Decimal('0.05'), Decimal('0.15'), Decimal('0.2'), Decimal('0.35'), Decimal('0.25')],
            [Decimal('0.1'), Decimal('0.2'), Decimal('0.3'), Decimal('0.3'), Decimal('0.1')]
        ]
        
        tm = TransitionMatrix(
            matrix=matrix,
            observation_count=120,
            window_start=date(2023, 1, 1),
            window_end=date(2023, 3, 1)
        )
        
        assert len(tm.matrix) == 5
        assert len(tm.matrix[0]) == 5
        assert tm.observation_count == 120
        assert tm.window_start == date(2023, 1, 1)
        assert tm.window_end == date(2023, 3, 1)
        assert isinstance(tm.last_updated, datetime)
    
    def test_wrong_matrix_dimensions(self):
        """Test rejection of matrices with wrong dimensions."""
        # Wrong number of rows
        with pytest.raises(ValidationError, match="Matrix must be|wrong size"):
            TransitionMatrix(
                matrix=[
                    [Decimal('0.5'), Decimal('0.5')]  # Only 2 columns, 1 row
                ],
                observation_count=10,
                window_start=date(2023, 1, 1),
                window_end=date(2023, 2, 1)
            )
        
        # Wrong number of columns
        with pytest.raises(ValidationError, match="Matrix must be|wrong size"):
            TransitionMatrix(
                matrix=[
                    [Decimal('0.2'), Decimal('0.3'), Decimal('0.5')],  # Only 3 columns
                    [Decimal('0.2'), Decimal('0.3'), Decimal('0.5')],
                    [Decimal('0.2'), Decimal('0.3'), Decimal('0.5')],
                    [Decimal('0.2'), Decimal('0.3'), Decimal('0.5')],
                    [Decimal('0.2'), Decimal('0.3'), Decimal('0.5')]
                ],
                observation_count=10,
                window_start=date(2023, 1, 1),
                window_end=date(2023, 2, 1)
            )
    
    def test_rows_must_sum_to_one(self):
        """Test that matrix rows must sum to 1.0."""
        matrix = [
            [Decimal('0.2'), Decimal('0.3'), Decimal('0.3'), Decimal('0.15'), Decimal('0.04')],  # Sums to 0.99
            [Decimal('0.1'), Decimal('0.4'), Decimal('0.3'), Decimal('0.15'), Decimal('0.05')],
            [Decimal('0.05'), Decimal('0.25'), Decimal('0.4'), Decimal('0.25'), Decimal('0.05')],
            [Decimal('0.05'), Decimal('0.15'), Decimal('0.2'), Decimal('0.35'), Decimal('0.25')],
            [Decimal('0.1'), Decimal('0.2'), Decimal('0.3'), Decimal('0.3'), Decimal('0.1')]
        ]
        
        with pytest.raises(ValidationError, match="sum to 1|sum to.*1"):
            TransitionMatrix(
                matrix=matrix,
                observation_count=120,
                window_start=date(2023, 1, 1),
                window_end=date(2023, 3, 1)
            )
    
    def test_floating_point_precision_tolerance(self):
        """Test that small floating point errors are tolerated."""
        matrix = [
            [Decimal('0.2'), Decimal('0.3'), Decimal('0.3'), Decimal('0.15'), Decimal('0.0500001')],  # Very small error
            [Decimal('0.1'), Decimal('0.4'), Decimal('0.3'), Decimal('0.15'), Decimal('0.05')],
            [Decimal('0.05'), Decimal('0.25'), Decimal('0.4'), Decimal('0.25'), Decimal('0.05')],
            [Decimal('0.05'), Decimal('0.15'), Decimal('0.2'), Decimal('0.35'), Decimal('0.25')],
            [Decimal('0.1'), Decimal('0.2'), Decimal('0.3'), Decimal('0.3'), Decimal('0.0999999')]  # Very small error
        ]
        
        # Should succeed due to 1e-6 tolerance
        tm = TransitionMatrix(
            matrix=matrix,
            observation_count=120,
            window_start=date(2023, 1, 1),
            window_end=date(2023, 3, 1)
        )
        
        assert tm is not None


class TestModelPrediction:
    """Test suite for ModelPrediction model."""
    
    def test_valid_model_prediction(self):
        """Test creation of valid model prediction."""
        prediction = ModelPrediction(
            current_date=date(2023, 3, 15),
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.ELEVATED_PREMIUM,
            transition_probability=Decimal('0.75'),
            confidence_score=Decimal('0.82'),
            entropy=Decimal('0.65'),
            data_quality_score=Decimal('0.88')
        )
        
        assert prediction.current_date == date(2023, 3, 15)
        assert prediction.current_state == VRPState.FAIR_VALUE
        assert prediction.predicted_state == VRPState.ELEVATED_PREMIUM
        assert prediction.transition_probability == Decimal('0.75')
        assert prediction.confidence_score == Decimal('0.82')
        assert prediction.entropy == Decimal('0.65')
        assert prediction.data_quality_score == Decimal('0.88')
    
    def test_probability_bounds_validation(self):
        """Test that probabilities must be between 0 and 1."""
        # Test transition probability > 1
        with pytest.raises(ValidationError):
            ModelPrediction(
                current_date=date(2023, 3, 15),
                current_state=VRPState.FAIR_VALUE,
                predicted_state=VRPState.ELEVATED_PREMIUM,
                transition_probability=Decimal('1.5'),  # > 1
                confidence_score=Decimal('0.82'),
                entropy=Decimal('0.65'),
                data_quality_score=Decimal('0.88')
            )
        
        # Test confidence score < 0
        with pytest.raises(ValidationError):
            ModelPrediction(
                current_date=date(2023, 3, 15),
                current_state=VRPState.FAIR_VALUE,
                predicted_state=VRPState.ELEVATED_PREMIUM,
                transition_probability=Decimal('0.75'),
                confidence_score=Decimal('-0.1'),  # < 0
                entropy=Decimal('0.65'),
                data_quality_score=Decimal('0.88')
            )
    
    def test_edge_case_probability_values(self):
        """Test edge case probability values (0 and 1)."""
        prediction = ModelPrediction(
            current_date=date(2023, 3, 15),
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.ELEVATED_PREMIUM,
            transition_probability=Decimal('1.0'),  # Exactly 1
            confidence_score=Decimal('0.0'),  # Exactly 0
            entropy=Decimal('2.0'),  # High entropy
            data_quality_score=Decimal('1.0')  # Perfect quality
        )
        
        assert prediction.transition_probability == Decimal('1.0')
        assert prediction.confidence_score == Decimal('0.0')


class TestTradingSignal:
    """Test suite for TradingSignal model."""
    
    def test_valid_trading_signal(self):
        """Test creation of valid trading signal."""
        signal = TradingSignal(
            date=date(2023, 3, 15),
            signal_type="SELL_VOL",
            current_state=VRPState.ELEVATED_PREMIUM,
            predicted_state=VRPState.ELEVATED_PREMIUM,
            signal_strength=Decimal('0.9'),
            confidence_score=Decimal('0.85'),
            recommended_position_size=Decimal('0.2'),
            risk_adjusted_size=Decimal('0.15'),
            reason="High confidence transition to extreme premium state"
        )
        
        assert signal.date == date(2023, 3, 15)
        assert signal.signal_type == "SELL_VOL"
        assert signal.current_state == VRPState.ELEVATED_PREMIUM
        assert signal.predicted_state == VRPState.ELEVATED_PREMIUM
        assert signal.signal_strength == Decimal('0.9')
        assert signal.confidence_score == Decimal('0.85')
        assert signal.recommended_position_size == Decimal('0.2')
        assert signal.risk_adjusted_size == Decimal('0.15')
        assert signal.reason == "High confidence transition to extreme premium state"
    
    def test_signal_type_validation(self):
        """Test that signal type must be valid."""
        # Valid signal types
        for signal_type in ["BUY_VOL", "SELL_VOL", "HOLD"]:
            signal = TradingSignal(
                date=date(2023, 3, 15),
                signal_type=signal_type,
                current_state=VRPState.FAIR_VALUE,
                predicted_state=VRPState.ELEVATED_PREMIUM,
                signal_strength=Decimal('0.8'),
                confidence_score=Decimal('0.7'),
                recommended_position_size=Decimal('0.15'),
                risk_adjusted_size=Decimal('0.12'),
                reason="Test signal"
            )
            assert signal.signal_type == signal_type
        
        # Invalid signal type
        with pytest.raises(ValidationError):
            TradingSignal(
                date=date(2023, 3, 15),
                signal_type="INVALID_SIGNAL",
                current_state=VRPState.FAIR_VALUE,
                predicted_state=VRPState.ELEVATED_PREMIUM,
                signal_strength=Decimal('0.8'),
                confidence_score=Decimal('0.7'),
                recommended_position_size=Decimal('0.15'),
                risk_adjusted_size=Decimal('0.12'),
                reason="Test signal"
            )
    
    def test_position_size_bounds(self):
        """Test that position sizes are bounded between 0 and 1."""
        # Valid position sizes
        signal = TradingSignal(
            date=date(2023, 3, 15),
            signal_type="SELL_VOL",
            current_state=VRPState.ELEVATED_PREMIUM,
            predicted_state=VRPState.ELEVATED_PREMIUM,
            signal_strength=Decimal('0.9'),
            confidence_score=Decimal('0.85'),
            recommended_position_size=Decimal('1.0'),  # Maximum allowed
            risk_adjusted_size=Decimal('0.0'),  # Minimum allowed
            reason="Edge case test"
        )
        
        assert signal.recommended_position_size == Decimal('1.0')
        assert signal.risk_adjusted_size == Decimal('0.0')
        
        # Invalid position size > 1
        with pytest.raises(ValidationError):
            TradingSignal(
                date=date(2023, 3, 15),
                signal_type="SELL_VOL",
                current_state=VRPState.ELEVATED_PREMIUM,
                predicted_state=VRPState.ELEVATED_PREMIUM,
                signal_strength=Decimal('0.9'),
                confidence_score=Decimal('0.85'),
                recommended_position_size=Decimal('1.5'),  # > 1
                risk_adjusted_size=Decimal('0.15'),
                reason="Invalid test"
            )


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics model."""
    
    def test_valid_performance_metrics(self):
        """Test creation of valid performance metrics."""
        metrics = PerformanceMetrics(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            total_return=Decimal('0.15'),
            sharpe_ratio=Decimal('1.2'),
            max_drawdown=Decimal('-0.08'),
            profit_factor=Decimal('1.8'),
            win_rate=Decimal('0.6466666666666667'),
            avg_win=Decimal('0.025'),
            avg_loss=Decimal('-0.015'),
            total_trades=150,
            winning_trades=97,
            losing_trades=53,
            extreme_state_precision=Decimal('0.72')
        )
        
        assert metrics.start_date == date(2023, 1, 1)
        assert metrics.end_date == date(2023, 12, 31)
        assert metrics.total_return == Decimal('0.15')
        assert metrics.sharpe_ratio == Decimal('1.2')
        assert metrics.max_drawdown == Decimal('-0.08')
        assert metrics.profit_factor == Decimal('1.8')
        assert metrics.win_rate == Decimal('0.6466666666666667')
        assert metrics.total_trades == 150
        assert metrics.winning_trades == 97
        assert metrics.losing_trades == 53
    
    def test_profit_factor_validation(self):
        """Test that profit factor must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            PerformanceMetrics(
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                total_return=Decimal('0.15'),
                sharpe_ratio=Decimal('1.2'),
                max_drawdown=Decimal('-0.08'),
                profit_factor=Decimal('-1.8'),  # Negative profit factor
                win_rate=Decimal('0.65'),
                avg_win=Decimal('0.025'),
                avg_loss=Decimal('-0.015'),
                total_trades=150,
                winning_trades=98,
                losing_trades=52,
                extreme_state_precision=Decimal('0.72')
            )
    
    def test_win_rate_bounds(self):
        """Test that win rate is bounded between 0 and 1."""
        # Valid win rate at boundaries
        metrics = PerformanceMetrics(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            total_return=Decimal('0.15'),
            sharpe_ratio=Decimal('1.2'),
            max_drawdown=Decimal('-0.08'),
            profit_factor=Decimal('1.8'),
            win_rate=Decimal('1.0'),  # Perfect win rate
            avg_win=Decimal('0.025'),
            avg_loss=Decimal('0.0'),  # No losses
            total_trades=150,
            winning_trades=150,
            losing_trades=0,
            extreme_state_precision=Decimal('0.72')
        )
        
        assert metrics.win_rate == Decimal('1.0')
        assert metrics.losing_trades == 0
        
        # Invalid win rate > 1
        with pytest.raises(ValidationError):
            PerformanceMetrics(
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                total_return=Decimal('0.15'),
                sharpe_ratio=Decimal('1.2'),
                max_drawdown=Decimal('-0.08'),
                profit_factor=Decimal('1.8'),
                win_rate=Decimal('1.2'),  # > 1
                avg_win=Decimal('0.025'),
                avg_loss=Decimal('-0.015'),
                total_trades=150,
                winning_trades=98,
                losing_trades=52,
                extreme_state_precision=Decimal('0.72')
            )


class TestConfigurationSettings:
    """Test suite for ConfigurationSettings model."""
    
    def test_default_configuration(self):
        """Test that default configuration is valid."""
        config = ConfigurationSettings()
        
        assert config.min_data_years == 3
        assert config.preferred_data_years == 5
        assert config.rolling_window_days == 60
        # Old threshold-based assertions removed - using adaptive quantiles now
        # Old threshold assertions removed - system now uses adaptive quantiles
        # Removed deprecated threshold assertions - system uses adaptive quantiles
        assert config.laplace_smoothing_alpha == Decimal('1.0')
        assert config.min_confidence_threshold == Decimal('0.6')
        assert config.max_position_size == Decimal('0.05')
        assert config.target_sharpe_ratio == Decimal('0.8')
    
    def test_custom_configuration(self):
        """Test creation of custom configuration."""
        config = ConfigurationSettings(
            min_data_years=2,
            preferred_data_years=4,
            rolling_window_days=45,
            vrp_quantile_window=200,
            max_position_size=Decimal('0.3'),
            target_sharpe_ratio=Decimal('1.0')
        )
        
        assert config.min_data_years == 2
        assert config.preferred_data_years == 4
        assert config.rolling_window_days == 45
        assert config.vrp_quantile_window == 200
        assert config.max_position_size == Decimal('0.3')
        assert config.target_sharpe_ratio == Decimal('1.0')
    
    def test_position_size_bounds(self):
        """Test that position size is bounded."""
        # Valid position size at boundary
        config = ConfigurationSettings(max_position_size=Decimal('1.0'))
        assert config.max_position_size == Decimal('1.0')
        
        # Invalid position size > 1
        with pytest.raises(ValidationError):
            ConfigurationSettings(max_position_size=Decimal('1.5'))
        
        # Invalid position size < 0
        with pytest.raises(ValidationError):
            ConfigurationSettings(max_position_size=Decimal('-0.1'))
    
    def test_minimum_data_years_validation(self):
        """Test that minimum data years is validated."""
        # Valid minimum
        config = ConfigurationSettings(min_data_years=1)
        assert config.min_data_years == 1
        
        # Invalid minimum (< 1)
        with pytest.raises(ValidationError):
            ConfigurationSettings(min_data_years=0)
    
    def test_adaptive_quantile_configuration(self):
        """Test adaptive quantile configuration."""
        config = ConfigurationSettings(
            vrp_quantile_window=300,
            rolling_window_days=90
        )
        
        assert config.vrp_quantile_window == 300
        assert config.rolling_window_days == 90
        # Validate that quantile window is reasonable
        assert config.vrp_quantile_window >= 30


class TestModelHealthMetrics:
    """Test suite for ModelHealthMetrics model."""
    
    def test_valid_health_metrics(self):
        """Test creation of valid health metrics."""
        metrics = ModelHealthMetrics(
            data_freshness_hours=2,
            transition_matrix_age_days=5,
            recent_prediction_accuracy=Decimal('0.75'),
            entropy_trend="STABLE",
            data_quality_issues=["Minor gap in VIX data"],
            model_drift_detected=False,
            alert_level="GREEN"
        )
        
        assert metrics.data_freshness_hours == 2
        assert metrics.transition_matrix_age_days == 5
        assert metrics.recent_prediction_accuracy == Decimal('0.75')
        assert metrics.entropy_trend == "STABLE"
        assert len(metrics.data_quality_issues) == 1
        assert not metrics.model_drift_detected
        assert metrics.alert_level == "GREEN"
        assert isinstance(metrics.timestamp, datetime)
    
    def test_trend_validation(self):
        """Test that entropy trend must be valid."""
        # Valid trends
        for trend in ["INCREASING", "STABLE", "DECREASING"]:
            metrics = ModelHealthMetrics(
                data_freshness_hours=2,
                transition_matrix_age_days=5,
                recent_prediction_accuracy=Decimal('0.75'),
                entropy_trend=trend,
                alert_level="GREEN"
            )
            assert metrics.entropy_trend == trend
        
        # Invalid trend
        with pytest.raises(ValidationError):
            ModelHealthMetrics(
                data_freshness_hours=2,
                transition_matrix_age_days=5,
                recent_prediction_accuracy=Decimal('0.75'),
                entropy_trend="INVALID_TREND",
                alert_level="GREEN"
            )
    
    def test_alert_level_validation(self):
        """Test that alert level must be valid."""
        # Valid alert levels
        for level in ["GREEN", "YELLOW"]:
            metrics = ModelHealthMetrics(
                data_freshness_hours=2,
                transition_matrix_age_days=5,
                recent_prediction_accuracy=Decimal('0.75'),
                entropy_trend="STABLE",
                alert_level=level
            )
            assert metrics.alert_level == level
        
        # RED alert requires model drift or data quality issues
        metrics = ModelHealthMetrics(
            data_freshness_hours=2,
            transition_matrix_age_days=5,
            recent_prediction_accuracy=Decimal('0.75'),
            entropy_trend="STABLE",
            alert_level="RED",
            model_drift_detected=True  # Required for RED alert
        )
        assert metrics.alert_level == "RED"
        
        # Invalid alert level
        with pytest.raises(ValidationError):
            ModelHealthMetrics(
                data_freshness_hours=2,
                transition_matrix_age_days=5,
                recent_prediction_accuracy=Decimal('0.75'),
                entropy_trend="STABLE",
                alert_level="INVALID_LEVEL"
            )
    
    def test_empty_data_quality_issues(self):
        """Test that data quality issues can be empty."""
        metrics = ModelHealthMetrics(
            data_freshness_hours=1,
            transition_matrix_age_days=3,
            recent_prediction_accuracy=Decimal('0.85'),
            entropy_trend="STABLE",
            alert_level="GREEN"
        )
        
        assert len(metrics.data_quality_issues) == 0