"""
Unit tests for SignalGenerator service.

This module contains comprehensive tests for trading signal generation,
focusing on extreme state detection, confidence thresholds, position sizing,
and signal validation logic. Tests ensure accurate signal generation for
the VRP Markov Chain Trading Model.
"""

import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
from typing import Optional, List

from src.config.settings import VRPTradingConfig
from src.trading.signal_generator import SignalGenerator
from src.models.data_models import (
    ModelPrediction,
    VRPState,
    VolatilityMetrics,
    TradingSignal,
    TransitionMatrix,
    Position
)
from src.utils.exceptions import SignalGenerationError, ValidationError


class TestSignalGenerator:
    """Test suite for SignalGenerator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration for SignalGenerator."""
        config = Mock(spec=VRPTradingConfig)
        config.model = Mock()
        config.model.min_confidence_threshold = Decimal('0.6')
        config.model.min_signal_strength = Decimal('0.7')
        config.model.entropy_threshold = Decimal('1.0')
        config.model.data_quality_threshold = Decimal('0.7')
        
        config.risk = Mock()
        config.risk.max_position_size = Decimal('0.25')
        config.risk.base_position_size = Decimal('0.1')
        config.risk.position_scaling_factor = Decimal('1.5')
        config.risk.max_portfolio_concentration = Decimal('0.4')
        
        config.signals = Mock()
        config.signals.extreme_state_only = True
        config.signals.require_state_transition = True
        config.signals.min_prediction_horizon_days = 1
        config.signals.signal_cooldown_days = 3
        
        return config
    
    @pytest.fixture
    def signal_generator(self, config):
        """Create SignalGenerator instance."""
        return SignalGenerator(config)
    
    @pytest.fixture
    def extreme_premium_prediction(self):
        """Create prediction for extreme premium state."""
        return ModelPrediction(
            current_date=date(2023, 3, 15),
            current_state=VRPState.ELEVATED_PREMIUM,
            predicted_state=VRPState.EXTREME_PREMIUM,
            transition_probability=Decimal('0.85'),
            confidence_score=Decimal('0.9'),
            entropy=Decimal('0.3'),
            data_quality_score=Decimal('0.92')
        )
    
    @pytest.fixture
    def underpriced_prediction(self):
        """Create prediction for underpriced state."""
        return ModelPrediction(
            current_date=date(2023, 3, 16),
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.UNDERPRICED,
            transition_probability=Decimal('0.8'),
            confidence_score=Decimal('0.85'),
            entropy=Decimal('0.4'),
            data_quality_score=Decimal('0.88')
        )
    
    @pytest.fixture
    def low_confidence_prediction(self):
        """Create low confidence prediction."""
        return ModelPrediction(
            current_date=date(2023, 3, 17),
            current_state=VRPState.NORMAL_PREMIUM,
            predicted_state=VRPState.EXTREME_PREMIUM,
            transition_probability=Decimal('0.45'),
            confidence_score=Decimal('0.3'),
            entropy=Decimal('1.5'),
            data_quality_score=Decimal('0.6')
        )
    
    @pytest.fixture
    def sample_volatility_metrics(self):
        """Create sample volatility metrics."""
        return VolatilityMetrics(
            date=date(2023, 3, 15),
            spy_return=Decimal('0.015'),
            realized_vol_30d=Decimal('0.18'),
            implied_vol=Decimal('0.27'),
            vrp=Decimal('1.5'),
            vrp_state=VRPState.EXTREME_PREMIUM
        )
    
    def test_generate_sell_vol_signal_success(self, signal_generator, extreme_premium_prediction, sample_volatility_metrics):
        """Test successful SELL_VOL signal generation."""
        signal = signal_generator.generate_signal(extreme_premium_prediction, sample_volatility_metrics)
        
        assert signal is not None
        assert signal.signal_type == "SELL_VOL"
        assert signal.current_state == VRPState.ELEVATED_PREMIUM
        assert signal.predicted_state == VRPState.EXTREME_PREMIUM
        assert signal.signal_strength >= Decimal('0.7')  # Above minimum threshold
        assert signal.confidence_score >= Decimal('0.6')  # Above minimum threshold
        assert Decimal('0.0') < signal.recommended_position_size <= Decimal('0.25')
        assert signal.risk_adjusted_size <= signal.recommended_position_size
        assert signal.date == extreme_premium_prediction.current_date
        assert "extreme premium" in signal.reason.lower()
    
    def test_generate_buy_vol_signal_success(self, signal_generator, underpriced_prediction, sample_volatility_metrics):
        """Test successful BUY_VOL signal generation."""
        # Adjust volatility metrics for underpriced scenario
        underpriced_metrics = VolatilityMetrics(
            date=date(2023, 3, 16),
            spy_return=Decimal('-0.008'),
            realized_vol_30d=Decimal('0.25'),
            implied_vol=Decimal('0.18'),
            vrp=Decimal('0.72'),
            vrp_state=VRPState.UNDERPRICED
        )
        
        signal = signal_generator.generate_signal(underpriced_prediction, underpriced_metrics)
        
        assert signal is not None
        assert signal.signal_type == "BUY_VOL"
        assert signal.current_state == VRPState.FAIR_VALUE
        assert signal.predicted_state == VRPState.UNDERPRICED
        assert signal.signal_strength >= Decimal('0.7')
        assert signal.confidence_score >= Decimal('0.6')
        assert Decimal('0.0') < signal.recommended_position_size <= Decimal('0.25')
        assert signal.risk_adjusted_size <= signal.recommended_position_size
        assert signal.date == underpriced_prediction.current_date
        assert "underpriced" in signal.reason.lower()
    
    def test_no_signal_low_confidence(self, signal_generator, low_confidence_prediction, sample_volatility_metrics):
        """Test no signal generation for low confidence predictions."""
        signal = signal_generator.generate_signal(low_confidence_prediction, sample_volatility_metrics)
        
        # Should return None due to low confidence
        assert signal is None
    
    def test_no_signal_non_extreme_states(self, signal_generator, sample_volatility_metrics):
        """Test no signal generation for non-extreme state transitions."""
        # Normal to elevated premium (not extreme enough)
        normal_prediction = ModelPrediction(
            current_date=date(2023, 3, 15),
            current_state=VRPState.NORMAL_PREMIUM,
            predicted_state=VRPState.ELEVATED_PREMIUM,
            transition_probability=Decimal('0.85'),
            confidence_score=Decimal('0.9'),
            entropy=Decimal('0.3'),
            data_quality_score=Decimal('0.92')
        )
        
        signal = signal_generator.generate_signal(normal_prediction, sample_volatility_metrics)
        
        # Should return None as it's not an extreme state transition
        assert signal is None
        
        # Fair value to normal premium
        fair_to_normal = ModelPrediction(
            current_date=date(2023, 3, 15),
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.NORMAL_PREMIUM,
            transition_probability=Decimal('0.85'),
            confidence_score=Decimal('0.9'),
            entropy=Decimal('0.3'),
            data_quality_score=Decimal('0.92')
        )
        
        signal = signal_generator.generate_signal(fair_to_normal, sample_volatility_metrics)
        assert signal is None
    
    def test_signal_strength_calculation(self, signal_generator, extreme_premium_prediction):
        """Test signal strength calculation logic."""
        strength = signal_generator.calculate_signal_strength(extreme_premium_prediction)
        
        # Signal strength should be based on confidence score and transition probability
        expected_strength = (
            float(extreme_premium_prediction.confidence_score) * 0.6 +
            float(extreme_premium_prediction.transition_probability) * 0.4
        )
        
        assert abs(float(strength) - expected_strength) < 1e-6
        assert 0 <= strength <= 1
    
    def test_signal_strength_edge_cases(self, signal_generator):
        """Test signal strength calculation with edge case values."""
        # Maximum strength case
        max_prediction = ModelPrediction(
            current_date=date(2023, 3, 15),
            current_state=VRPState.ELEVATED_PREMIUM,
            predicted_state=VRPState.EXTREME_PREMIUM,
            transition_probability=Decimal('1.0'),
            confidence_score=Decimal('1.0'),
            entropy=Decimal('0.0'),
            data_quality_score=Decimal('1.0')
        )
        
        max_strength = signal_generator.calculate_signal_strength(max_prediction)
        assert max_strength == Decimal('1.0')
        
        # Minimum strength case
        min_prediction = ModelPrediction(
            current_date=date(2023, 3, 15),
            current_state=VRPState.NORMAL_PREMIUM,
            predicted_state=VRPState.ELEVATED_PREMIUM,
            transition_probability=Decimal('0.0'),
            confidence_score=Decimal('0.0'),
            entropy=Decimal('2.0'),
            data_quality_score=Decimal('0.0')
        )
        
        min_strength = signal_generator.calculate_signal_strength(min_prediction)
        assert min_strength == Decimal('0.0')
    
    def test_position_size_calculation(self, signal_generator, extreme_premium_prediction):
        """Test position size calculation based on signal strength."""
        # High confidence/strength should result in larger position
        high_strength = Decimal('0.9')
        large_position = signal_generator.calculate_position_size(
            high_strength, 
            extreme_premium_prediction.confidence_score
        )
        
        # Low confidence/strength should result in smaller position
        low_strength = Decimal('0.3')
        small_position = signal_generator.calculate_position_size(
            low_strength, 
            Decimal('0.4')
        )
        
        assert large_position > small_position
        assert large_position <= Decimal('0.25')  # Max position size
        assert small_position >= Decimal('0.0')
    
    def test_risk_adjusted_position_sizing(self, signal_generator, extreme_premium_prediction, sample_volatility_metrics):
        """Test risk adjustment of position sizes."""
        signal = signal_generator.generate_signal(extreme_premium_prediction, sample_volatility_metrics)
        
        assert signal is not None
        assert signal.risk_adjusted_size <= signal.recommended_position_size
        
        # Risk adjustment should consider current market volatility
        # Higher volatility should result in smaller risk-adjusted size
        high_vol_metrics = VolatilityMetrics(
            date=date(2023, 3, 15),
            spy_return=Decimal('0.035'),  # Higher return
            realized_vol_30d=Decimal('0.35'),  # Higher volatility
            implied_vol=Decimal('0.45'),
            vrp=Decimal('1.8'),
            vrp_state=VRPState.EXTREME_PREMIUM
        )
        
        high_vol_signal = signal_generator.generate_signal(extreme_premium_prediction, high_vol_metrics)
        
        assert high_vol_signal is not None
        # Risk-adjusted size should be smaller for higher volatility
        assert high_vol_signal.risk_adjusted_size <= signal.risk_adjusted_size
    
    def test_signal_validation_conditions(self, signal_generator):
        """Test signal validation conditions."""
        # Valid extreme state prediction
        valid_prediction = ModelPrediction(
            current_date=date(2023, 3, 15),
            current_state=VRPState.ELEVATED_PREMIUM,
            predicted_state=VRPState.EXTREME_PREMIUM,
            transition_probability=Decimal('0.85'),
            confidence_score=Decimal('0.8'),
            entropy=Decimal('0.5'),
            data_quality_score=Decimal('0.85')
        )
        
        assert signal_generator.validate_signal_conditions(valid_prediction) == True
        
        # Invalid - low confidence
        low_conf_prediction = ModelPrediction(
            current_date=date(2023, 3, 15),
            current_state=VRPState.ELEVATED_PREMIUM,
            predicted_state=VRPState.EXTREME_PREMIUM,
            transition_probability=Decimal('0.85'),
            confidence_score=Decimal('0.4'),  # Below threshold
            entropy=Decimal('0.5'),
            data_quality_score=Decimal('0.85')
        )
        
        assert signal_generator.validate_signal_conditions(low_conf_prediction) == False
        
        # Invalid - high entropy
        high_entropy_prediction = ModelPrediction(
            current_date=date(2023, 3, 15),
            current_state=VRPState.ELEVATED_PREMIUM,
            predicted_state=VRPState.EXTREME_PREMIUM,
            transition_probability=Decimal('0.85'),
            confidence_score=Decimal('0.8'),
            entropy=Decimal('1.5'),  # Above threshold
            data_quality_score=Decimal('0.85')
        )
        
        assert signal_generator.validate_signal_conditions(high_entropy_prediction) == False
        
        # Invalid - low data quality
        low_quality_prediction = ModelPrediction(
            current_date=date(2023, 3, 15),
            current_state=VRPState.ELEVATED_PREMIUM,
            predicted_state=VRPState.EXTREME_PREMIUM,
            transition_probability=Decimal('0.85'),
            confidence_score=Decimal('0.8'),
            entropy=Decimal('0.5'),
            data_quality_score=Decimal('0.5')  # Below threshold
        )
        
        assert signal_generator.validate_signal_conditions(low_quality_prediction) == False
    
    def test_extreme_state_detection(self, signal_generator):
        """Test detection of extreme states for signal generation."""
        # Extreme states should be detected
        assert signal_generator._is_extreme_state_transition(
            VRPState.ELEVATED_PREMIUM, VRPState.EXTREME_PREMIUM
        ) == True
        
        assert signal_generator._is_extreme_state_transition(
            VRPState.FAIR_VALUE, VRPState.UNDERPRICED
        ) == True
        
        assert signal_generator._is_extreme_state_transition(
            VRPState.NORMAL_PREMIUM, VRPState.UNDERPRICED
        ) == True
        
        # Non-extreme transitions should not be detected
        assert signal_generator._is_extreme_state_transition(
            VRPState.FAIR_VALUE, VRPState.NORMAL_PREMIUM
        ) == False
        
        assert signal_generator._is_extreme_state_transition(
            VRPState.NORMAL_PREMIUM, VRPState.ELEVATED_PREMIUM
        ) == False
        
        assert signal_generator._is_extreme_state_transition(
            VRPState.ELEVATED_PREMIUM, VRPState.NORMAL_PREMIUM
        ) == False
        
        # Same state transitions
        assert signal_generator._is_extreme_state_transition(
            VRPState.FAIR_VALUE, VRPState.FAIR_VALUE
        ) == False
    
    def test_signal_cooldown_period(self, signal_generator, extreme_premium_prediction, sample_volatility_metrics):
        """Test signal cooldown period functionality."""
        # Generate first signal
        signal1 = signal_generator.generate_signal(extreme_premium_prediction, sample_volatility_metrics)
        assert signal1 is not None
        
        # Try to generate signal within cooldown period (should be blocked)
        recent_prediction = ModelPrediction(
            current_date=date(2023, 3, 16),  # Next day
            current_state=VRPState.ELEVATED_PREMIUM,
            predicted_state=VRPState.EXTREME_PREMIUM,
            transition_probability=Decimal('0.85'),
            confidence_score=Decimal('0.9'),
            entropy=Decimal('0.3'),
            data_quality_score=Decimal('0.92')
        )
        
        with patch.object(signal_generator, '_get_last_signal_date', return_value=date(2023, 3, 15)):
            signal2 = signal_generator.generate_signal(recent_prediction, sample_volatility_metrics)
            assert signal2 is None  # Should be blocked by cooldown
        
        # Signal after cooldown period should be allowed
        later_prediction = ModelPrediction(
            current_date=date(2023, 3, 20),  # After cooldown
            current_state=VRPState.ELEVATED_PREMIUM,
            predicted_state=VRPState.EXTREME_PREMIUM,
            transition_probability=Decimal('0.85'),
            confidence_score=Decimal('0.9'),
            entropy=Decimal('0.3'),
            data_quality_score=Decimal('0.92')
        )
        
        with patch.object(signal_generator, '_get_last_signal_date', return_value=date(2023, 3, 15)):
            signal3 = signal_generator.generate_signal(later_prediction, sample_volatility_metrics)
            assert signal3 is not None  # Should be allowed after cooldown
    
    def test_portfolio_concentration_limits(self, signal_generator, extreme_premium_prediction, sample_volatility_metrics):
        """Test portfolio concentration limits in position sizing."""
        # Mock existing positions that would create concentration
        existing_positions = [
            Position(
                position_id="pos1",
                symbol="VIX_CALL",
                position_type="SHORT_VOL",
                entry_date=date(2023, 3, 10),
                entry_signal=Mock(),
                position_size=Decimal('0.3'),  # High concentration
                is_active=True
            )
        ]
        
        with patch.object(signal_generator, '_get_active_positions', return_value=existing_positions):
            signal = signal_generator.generate_signal(extreme_premium_prediction, sample_volatility_metrics)
            
            if signal is not None:
                # Position size should be reduced due to concentration limits
                assert signal.risk_adjusted_size < signal.recommended_position_size
                
                # Total concentration should not exceed limit
                total_concentration = signal.risk_adjusted_size + Decimal('0.3')
                assert total_concentration <= Decimal('0.4')  # Max concentration
    
    def test_signal_reason_generation(self, signal_generator, extreme_premium_prediction, sample_volatility_metrics):
        """Test generation of clear signal reasoning."""
        signal = signal_generator.generate_signal(extreme_premium_prediction, sample_volatility_metrics)
        
        assert signal is not None
        assert len(signal.reason) > 20  # Should be descriptive
        assert "confidence" in signal.reason.lower()
        assert str(extreme_premium_prediction.transition_probability) in signal.reason
        
        # Check that reason includes key metrics
        reason_lower = signal.reason.lower()
        assert any(word in reason_lower for word in ["extreme", "premium", "elevated"])
    
    def test_signal_type_determination(self, signal_generator):
        """Test correct signal type determination based on state transitions."""
        # Sell volatility signals (moving to extreme premium)
        sell_transitions = [
            (VRPState.ELEVATED_PREMIUM, VRPState.EXTREME_PREMIUM),
            (VRPState.NORMAL_PREMIUM, VRPState.EXTREME_PREMIUM),
            (VRPState.FAIR_VALUE, VRPState.EXTREME_PREMIUM),
        ]
        
        for current, predicted in sell_transitions:
            signal_type = signal_generator._determine_signal_type(current, predicted)
            assert signal_type == "SELL_VOL"
        
        # Buy volatility signals (moving to underpriced)
        buy_transitions = [
            (VRPState.FAIR_VALUE, VRPState.UNDERPRICED),
            (VRPState.NORMAL_PREMIUM, VRPState.UNDERPRICED),
            (VRPState.ELEVATED_PREMIUM, VRPState.UNDERPRICED),
        ]
        
        for current, predicted in buy_transitions:
            signal_type = signal_generator._determine_signal_type(current, predicted)
            assert signal_type == "BUY_VOL"
    
    def test_invalid_state_transitions(self, signal_generator, sample_volatility_metrics):
        """Test handling of invalid or impossible state transitions."""
        # Same state transition
        same_state_prediction = ModelPrediction(
            current_date=date(2023, 3, 15),
            current_state=VRPState.EXTREME_PREMIUM,
            predicted_state=VRPState.EXTREME_PREMIUM,  # Same state
            transition_probability=Decimal('0.85'),
            confidence_score=Decimal('0.9'),
            entropy=Decimal('0.3'),
            data_quality_score=Decimal('0.92')
        )
        
        signal = signal_generator.generate_signal(same_state_prediction, sample_volatility_metrics)
        assert signal is None  # No signal for same state
        
        # Reverse extreme transition (extreme to underpriced - highly unlikely)
        extreme_reverse_prediction = ModelPrediction(
            current_date=date(2023, 3, 15),
            current_state=VRPState.EXTREME_PREMIUM,
            predicted_state=VRPState.UNDERPRICED,
            transition_probability=Decimal('0.05'),  # Very low probability
            confidence_score=Decimal('0.2'),  # Low confidence
            entropy=Decimal('1.8'),
            data_quality_score=Decimal('0.5')
        )
        
        signal = signal_generator.generate_signal(extreme_reverse_prediction, sample_volatility_metrics)
        assert signal is None  # Should be filtered out by validation
    
    @patch('src.trading.signal_generator.logger')
    def test_logging_during_signal_generation(self, mock_logger, signal_generator, extreme_premium_prediction, sample_volatility_metrics):
        """Test that appropriate logging occurs during signal generation."""
        signal = signal_generator.generate_signal(extreme_premium_prediction, sample_volatility_metrics)
        
        # Should have logged the signal generation
        mock_logger.info.assert_called()
        
        # Test logging for rejected signals
        mock_logger.reset_mock()
        
        low_conf_prediction = ModelPrediction(
            current_date=date(2023, 3, 15),
            current_state=VRPState.ELEVATED_PREMIUM,
            predicted_state=VRPState.EXTREME_PREMIUM,
            transition_probability=Decimal('0.85'),
            confidence_score=Decimal('0.3'),  # Below threshold
            entropy=Decimal('0.5'),
            data_quality_score=Decimal('0.85')
        )
        
        rejected_signal = signal_generator.generate_signal(low_conf_prediction, sample_volatility_metrics)
        
        # Should have logged why signal was rejected
        mock_logger.debug.assert_called()
    
    def test_configuration_impact_on_signals(self, signal_generator):
        """Test how configuration changes impact signal generation."""
        # Lower confidence threshold should allow more signals
        original_threshold = signal_generator.config.model.min_confidence_threshold
        signal_generator.config.model.min_confidence_threshold = Decimal('0.4')
        
        low_conf_prediction = ModelPrediction(
            current_date=date(2023, 3, 15),
            current_state=VRPState.ELEVATED_PREMIUM,
            predicted_state=VRPState.EXTREME_PREMIUM,
            transition_probability=Decimal('0.85'),
            confidence_score=Decimal('0.5'),  # Between 0.4 and 0.6
            entropy=Decimal('0.5'),
            data_quality_score=Decimal('0.85')
        )
        
        assert signal_generator.validate_signal_conditions(low_conf_prediction) == True
        
        # Restore original threshold
        signal_generator.config.model.min_confidence_threshold = original_threshold
        assert signal_generator.validate_signal_conditions(low_conf_prediction) == False
    
    def test_batch_signal_generation(self, signal_generator, sample_volatility_metrics):
        """Test batch processing of multiple predictions."""
        predictions = [
            ModelPrediction(
                current_date=date(2023, 3, 15),
                current_state=VRPState.ELEVATED_PREMIUM,
                predicted_state=VRPState.EXTREME_PREMIUM,
                transition_probability=Decimal('0.85'),
                confidence_score=Decimal('0.9'),
                entropy=Decimal('0.3'),
                data_quality_score=Decimal('0.92')
            ),
            ModelPrediction(
                current_date=date(2023, 3, 16),
                current_state=VRPState.FAIR_VALUE,
                predicted_state=VRPState.UNDERPRICED,
                transition_probability=Decimal('0.8'),
                confidence_score=Decimal('0.85'),
                entropy=Decimal('0.4'),
                data_quality_score=Decimal('0.88')
            ),
            ModelPrediction(
                current_date=date(2023, 3, 17),
                current_state=VRPState.NORMAL_PREMIUM,
                predicted_state=VRPState.ELEVATED_PREMIUM,
                transition_probability=Decimal('0.7'),
                confidence_score=Decimal('0.6'),
                entropy=Decimal('0.8'),
                data_quality_score=Decimal('0.8')
            )
        ]
        
        metrics_list = [sample_volatility_metrics] * len(predictions)
        signals = signal_generator.generate_batch_signals(predictions, metrics_list)
        
        # Should generate signals for extreme states only
        valid_signals = [s for s in signals if s is not None]
        assert len(valid_signals) == 2  # First two should generate signals
        
        signal_types = [s.signal_type for s in valid_signals]
        assert "SELL_VOL" in signal_types
        assert "BUY_VOL" in signal_types