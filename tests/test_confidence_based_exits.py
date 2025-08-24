"""
Comprehensive Test Suite for Confidence-Based Exits Enhancement

This test suite validates the three-tier confidence system for VRP trading:
- Entry threshold: 0.65
- Exit threshold: 0.40 
- Flip threshold: 0.75

Tests cover position state transitions, signal enhancement, flat period behavior,
and performance improvements vs legacy system.
"""

import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch

from src.models.data_models import (
    TradingSignal,
    VRPState,
    VolatilityData,
    Position,
    ModelPrediction,
    ConfidenceMetrics,
    MarketData,
    ExitConfidenceConfig
)
from src.utils.exceptions import ValidationError, ConfigurationError
from src.trading.position_manager import PositionManager, PositionState, ConfidenceThresholds

# Use the actual TradingSignal with exit_confidence field
EnhancedTradingSignal = TradingSignal


# Using imported PositionManager, PositionState, and ConfidenceThresholds

# Global fixtures available to all test classes
@pytest.fixture
def position_manager():
    """Create position manager instance."""
    return PositionManager()

@pytest.fixture
def high_confidence_buy_signal():
    """Create high confidence BUY signal."""
    return EnhancedTradingSignal(
        date=date(2023, 3, 15),
        signal_type="BUY_VOL",
        current_state=VRPState.FAIR_VALUE,
        predicted_state=VRPState.UNDERPRICED,
        signal_strength=Decimal('0.85'),
        confidence_score=Decimal('0.80'),  # Above entry threshold
        recommended_position_size=Decimal('0.15'),
        risk_adjusted_size=Decimal('0.12'),
        reason="High confidence underpriced prediction",
        exit_confidence=Decimal('0.70')  # High exit confidence
    )

@pytest.fixture
def high_confidence_sell_signal():
    """Create high confidence SELL signal."""
    return EnhancedTradingSignal(
        date=date(2023, 3, 15),
        signal_type="SELL_VOL",
        current_state=VRPState.ELEVATED_PREMIUM,
        predicted_state=VRPState.EXTREME_HIGH,
        signal_strength=Decimal('0.90'),
        confidence_score=Decimal('0.85'),  # Above entry threshold
        recommended_position_size=Decimal('0.18'),
        risk_adjusted_size=Decimal('0.15'),
        reason="High confidence extreme premium prediction",
        exit_confidence=Decimal('0.75')  # High exit confidence
    )

@pytest.fixture
def low_confidence_signal():
    """Create low confidence signal."""
    return EnhancedTradingSignal(
        date=date(2023, 3, 16),
        signal_type="BUY_VOL",
        current_state=VRPState.NORMAL_PREMIUM,
        predicted_state=VRPState.FAIR_VALUE,
        signal_strength=Decimal('0.30'),
        confidence_score=Decimal('0.25'),  # Below exit threshold
        recommended_position_size=Decimal('0.05'),
        risk_adjusted_size=Decimal('0.03'),
        reason="Low confidence signal",
        exit_confidence=Decimal('0.20')  # Very low exit confidence
    )

@pytest.fixture
def flip_threshold_signal():
    """Create signal at flip threshold."""
    return EnhancedTradingSignal(
        date=date(2023, 3, 17),
        signal_type="SELL_VOL",
        current_state=VRPState.UNDERPRICED,
        predicted_state=VRPState.EXTREME_HIGH,
        signal_strength=Decimal('0.95'),
        confidence_score=Decimal('0.85'),  # Above flip threshold
        recommended_position_size=Decimal('0.20'),
        risk_adjusted_size=Decimal('0.18'),
        reason="Very high confidence state flip",
        exit_confidence=Decimal('0.80')  # High exit confidence
    )


class TestConfidenceBasedExits:
    """Test suite for confidence-based exits system."""


class TestConfidenceThresholds:
    """Test confidence threshold logic."""
    
    def test_threshold_values(self):
        """Test that threshold values are correctly defined."""
        assert ConfidenceThresholds.ENTRY == Decimal('0.65')
        assert ConfidenceThresholds.EXIT == Decimal('0.40')
        assert ConfidenceThresholds.FLIP == Decimal('0.75')
        
        # Verify threshold ordering
        assert ConfidenceThresholds.EXIT < ConfidenceThresholds.ENTRY
        assert ConfidenceThresholds.ENTRY < ConfidenceThresholds.FLIP
    
    def test_threshold_comparisons(self):
        """Test threshold comparison logic."""
        # Entry threshold tests
        assert Decimal('0.70') >= ConfidenceThresholds.ENTRY
        assert Decimal('0.60') < ConfidenceThresholds.ENTRY
        
        # Exit threshold tests
        assert Decimal('0.35') <= ConfidenceThresholds.EXIT
        assert Decimal('0.45') > ConfidenceThresholds.EXIT
        
        # Flip threshold tests
        assert Decimal('0.80') >= ConfidenceThresholds.FLIP
        assert Decimal('0.70') < ConfidenceThresholds.FLIP


class TestPositionEntryLogic:
    """Test position entry logic from flat state."""
    
    def test_high_confidence_entry_buy(self, position_manager, high_confidence_buy_signal):
        """Test high confidence BUY entry from flat."""
        result = position_manager.process_signal(high_confidence_buy_signal)
        
        assert result['old_position'] == PositionState.FLAT
        assert result['new_position'] == PositionState.LONG_VOL
        assert result['action'] == "ENTER_LONG"
        assert result['new_size'] == high_confidence_buy_signal.risk_adjusted_size
        assert "confidence" in result['reasoning'].lower()
        assert position_manager.entry_date == high_confidence_buy_signal.date
    
    def test_high_confidence_entry_sell(self, position_manager, high_confidence_sell_signal):
        """Test high confidence SELL entry from flat."""
        result = position_manager.process_signal(high_confidence_sell_signal)
        
        assert result['old_position'] == PositionState.FLAT
        assert result['new_position'] == PositionState.SHORT_VOL
        assert result['action'] == "ENTER_SHORT"
        assert result['new_size'] == -high_confidence_sell_signal.risk_adjusted_size
        assert "confidence" in result['reasoning'].lower()
        assert position_manager.entry_date == high_confidence_sell_signal.date
    
    def test_low_confidence_no_entry(self, position_manager, low_confidence_signal):
        """Test low confidence signal does not trigger entry."""
        result = position_manager.process_signal(low_confidence_signal)
        
        assert result['old_position'] == PositionState.FLAT
        assert result['new_position'] == PositionState.FLAT
        assert result['action'] == "HOLD"
        assert result['new_size'] == Decimal('0.0')
        assert position_manager.entry_date is None
    
    def test_medium_confidence_no_entry(self, position_manager):
        """Test medium confidence (below entry threshold) does not trigger entry."""
        medium_confidence_signal = EnhancedTradingSignal(
            date=date(2023, 3, 15),
            signal_type="BUY_VOL",
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.UNDERPRICED,
            signal_strength=Decimal('0.60'),
            confidence_score=Decimal('0.55'),  # Below entry threshold
            recommended_position_size=Decimal('0.10'),
            risk_adjusted_size=Decimal('0.08'),
            reason="Medium confidence signal",
            exit_confidence=Decimal('0.50')
        )
        
        result = position_manager.process_signal(medium_confidence_signal)
        
        assert result['old_position'] == PositionState.FLAT
        assert result['new_position'] == PositionState.FLAT
        assert result['action'] == "HOLD"
        assert position_manager.current_position == PositionState.FLAT


class TestPositionExitLogic:
    """Test position exit logic."""
    
    def test_low_confidence_exit_from_long(self, position_manager, high_confidence_buy_signal, low_confidence_signal):
        """Test low confidence triggers exit from long position."""
        # First enter long position
        position_manager.process_signal(high_confidence_buy_signal)
        assert position_manager.current_position == PositionState.LONG_VOL
        
        # Then process low confidence signal
        result = position_manager.process_signal(low_confidence_signal)
        
        assert result['old_position'] == PositionState.LONG_VOL
        assert result['new_position'] == PositionState.FLAT
        assert result['action'] == "EXIT_TO_FLAT"
        assert result['new_size'] == Decimal('0.0')
        assert "confidence" in result['reasoning'].lower()
        assert position_manager.entry_date is None
    
    def test_low_confidence_exit_from_short(self, position_manager, high_confidence_sell_signal, low_confidence_signal):
        """Test low confidence triggers exit from short position."""
        # First enter short position
        position_manager.process_signal(high_confidence_sell_signal)
        assert position_manager.current_position == PositionState.SHORT_VOL
        
        # Then process low confidence signal
        result = position_manager.process_signal(low_confidence_signal)
        
        assert result['old_position'] == PositionState.SHORT_VOL
        assert result['new_position'] == PositionState.FLAT
        assert result['action'] == "EXIT_TO_FLAT"
        assert result['new_size'] == Decimal('0.0')
        assert "confidence" in result['reasoning'].lower()
        assert position_manager.entry_date is None
    
    def test_low_exit_confidence_triggers_exit(self, position_manager, high_confidence_buy_signal):
        """Test low exit confidence specifically triggers exit."""
        # Enter long position
        position_manager.process_signal(high_confidence_buy_signal)
        assert position_manager.current_position == PositionState.LONG_VOL
        
        # Signal with decent confidence but low exit confidence
        low_exit_conf_signal = EnhancedTradingSignal(
            date=date(2023, 3, 16),
            signal_type="HOLD",
            current_state=VRPState.NORMAL_PREMIUM,
            predicted_state=VRPState.FAIR_VALUE,
            signal_strength=Decimal('0.50'),
            confidence_score=Decimal('0.55'),  # Above exit threshold
            recommended_position_size=Decimal('0.10'),
            risk_adjusted_size=Decimal('0.08'),
            reason="Medium confidence hold",
            exit_confidence=Decimal('0.30')  # Below exit threshold
        )
        
        result = position_manager.process_signal(low_exit_conf_signal)
        
        assert result['old_position'] == PositionState.LONG_VOL
        assert result['new_position'] == PositionState.FLAT
        assert result['action'] == "EXIT_TO_FLAT"
        assert "exit confidence" in result['reasoning']
    
    def test_medium_confidence_maintains_position(self, position_manager, high_confidence_buy_signal):
        """Test medium confidence maintains existing position."""
        # Enter long position
        position_manager.process_signal(high_confidence_buy_signal)
        original_size = position_manager.position_size
        
        # Medium confidence signal
        medium_conf_signal = EnhancedTradingSignal(
            date=date(2023, 3, 16),
            signal_type="BUY_VOL",
            current_state=VRPState.NORMAL_PREMIUM,
            predicted_state=VRPState.FAIR_VALUE,
            signal_strength=Decimal('0.50'),
            confidence_score=Decimal('0.55'),  # Between exit and entry
            recommended_position_size=Decimal('0.10'),
            risk_adjusted_size=Decimal('0.08'),
            reason="Medium confidence signal",
            exit_confidence=Decimal('0.50')  # Above exit threshold
        )
        
        result = position_manager.process_signal(medium_conf_signal)
        
        assert result['old_position'] == PositionState.LONG_VOL
        assert result['new_position'] == PositionState.LONG_VOL
        assert result['action'] == "HOLD"
        assert result['new_size'] == original_size


class TestPositionFlipLogic:
    """Test position flip logic for very high confidence."""
    
    def test_flip_long_to_short_high_confidence(self, position_manager, high_confidence_buy_signal, flip_threshold_signal):
        """Test flip from long to short with very high confidence."""
        # Enter long position
        position_manager.process_signal(high_confidence_buy_signal)
        assert position_manager.current_position == PositionState.LONG_VOL
        
        # Process flip signal
        result = position_manager.process_signal(flip_threshold_signal)
        
        assert result['old_position'] == PositionState.LONG_VOL
        assert result['new_position'] == PositionState.SHORT_VOL
        assert result['action'] == "FLIP_TO_SHORT"
        assert result['new_size'] == -flip_threshold_signal.risk_adjusted_size
        assert "confidence" in result['reasoning'].lower()
        assert position_manager.entry_date == flip_threshold_signal.date
    
    def test_flip_short_to_long_high_confidence(self, position_manager, high_confidence_sell_signal):
        """Test flip from short to long with very high confidence."""
        # Enter short position
        position_manager.process_signal(high_confidence_sell_signal)
        assert position_manager.current_position == PositionState.SHORT_VOL
        
        # Create flip signal to long
        flip_to_long_signal = EnhancedTradingSignal(
            date=date(2023, 3, 17),
            signal_type="BUY_VOL",
            current_state=VRPState.EXTREME_HIGH,
            predicted_state=VRPState.UNDERPRICED,
            signal_strength=Decimal('0.95'),
            confidence_score=Decimal('0.90'),  # Above flip threshold
            recommended_position_size=Decimal('0.22'),
            risk_adjusted_size=Decimal('0.20'),
            reason="Very high confidence flip to underpriced",
            exit_confidence=Decimal('0.85')
        )
        
        result = position_manager.process_signal(flip_to_long_signal)
        
        assert result['old_position'] == PositionState.SHORT_VOL
        assert result['new_position'] == PositionState.LONG_VOL
        assert result['action'] == "FLIP_TO_LONG"
        assert result['new_size'] == flip_to_long_signal.risk_adjusted_size
        assert "confidence" in result['reasoning'].lower()
    
    def test_no_flip_below_threshold(self, position_manager, high_confidence_buy_signal):
        """Test no flip occurs below flip threshold."""
        # Enter long position
        position_manager.process_signal(high_confidence_buy_signal)
        original_size = position_manager.position_size
        
        # High confidence but below flip threshold
        high_but_not_flip_signal = EnhancedTradingSignal(
            date=date(2023, 3, 17),
            signal_type="SELL_VOL",
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.ELEVATED_PREMIUM,
            signal_strength=Decimal('0.80'),
            confidence_score=Decimal('0.70'),  # Below flip threshold
            recommended_position_size=Decimal('0.15'),
            risk_adjusted_size=Decimal('0.12'),
            reason="High confidence but not flip level",
            exit_confidence=Decimal('0.60')
        )
        
        result = position_manager.process_signal(high_but_not_flip_signal)
        
        assert result['old_position'] == PositionState.LONG_VOL
        assert result['new_position'] == PositionState.LONG_VOL
        assert result['action'] == "HOLD"
        assert result['new_size'] == original_size  # No change


class TestFlatPeriodBehavior:
    """Test flat period behavior during uncertainty."""
    
    def test_extended_flat_period(self, position_manager):
        """Test system stays flat during extended low confidence period."""
        flat_periods = []
        
        # Simulate 10 days of low confidence signals
        for i in range(10):
            low_conf_signal = EnhancedTradingSignal(
                date=date(2023, 3, 15) + timedelta(days=i),
                signal_type="HOLD",
                current_state=VRPState.NORMAL_PREMIUM,
                predicted_state=VRPState.NORMAL_PREMIUM,
                signal_strength=Decimal('0.30'),
                confidence_score=Decimal('0.35'),  # Below entry threshold
                recommended_position_size=Decimal('0.05'),
                risk_adjusted_size=Decimal('0.03'),
                reason=f"Low confidence day {i+1}",
                exit_confidence=Decimal('0.25')
            )
            
            result = position_manager.process_signal(low_conf_signal)
            flat_periods.append(result['new_position'] == PositionState.FLAT)
        
        # Should remain flat throughout
        assert all(flat_periods)
        assert position_manager.current_position == PositionState.FLAT
        assert position_manager.position_size == Decimal('0.0')
    
    def test_flat_to_position_transition(self, position_manager, high_confidence_buy_signal):
        """Test transition from flat period to position when confidence rises."""
        # Start with low confidence
        low_conf_signal = EnhancedTradingSignal(
            date=date(2023, 3, 14),
            signal_type="BUY_VOL",
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.UNDERPRICED,
            signal_strength=Decimal('0.40'),
            confidence_score=Decimal('0.50'),  # Below entry
            recommended_position_size=Decimal('0.08'),
            risk_adjusted_size=Decimal('0.06'),
            reason="Low confidence signal",
            exit_confidence=Decimal('0.40')
        )
        
        result1 = position_manager.process_signal(low_conf_signal)
        assert result1['new_position'] == PositionState.FLAT
        
        # Then high confidence signal should trigger entry
        result2 = position_manager.process_signal(high_confidence_buy_signal)
        assert result2['old_position'] == PositionState.FLAT
        assert result2['new_position'] == PositionState.LONG_VOL
        assert result2['action'] == "ENTER_LONG"
    
    def test_position_to_flat_transition(self, position_manager, high_confidence_buy_signal, low_confidence_signal):
        """Test transition from position to flat when confidence drops."""
        # Enter position
        result1 = position_manager.process_signal(high_confidence_buy_signal)
        assert result1['new_position'] == PositionState.LONG_VOL
        
        # Exit to flat
        result2 = position_manager.process_signal(low_confidence_signal)
        assert result2['old_position'] == PositionState.LONG_VOL
        assert result2['new_position'] == PositionState.FLAT
        assert result2['action'] == "EXIT_TO_FLAT"


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""
    
    def test_exact_threshold_values(self, position_manager):
        """Test behavior at exact threshold values."""
        # Exactly at entry threshold
        exact_entry_signal = EnhancedTradingSignal(
            date=date(2023, 3, 15),
            signal_type="BUY_VOL",
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.UNDERPRICED,
            signal_strength=Decimal('0.70'),
            confidence_score=ConfidenceThresholds.ENTRY,  # Exactly 0.65
            recommended_position_size=Decimal('0.10'),
            risk_adjusted_size=Decimal('0.08'),
            reason="Exact entry threshold",
            exit_confidence=Decimal('0.60')
        )
        
        result = position_manager.process_signal(exact_entry_signal)
        assert result['action'] == "ENTER_LONG"  # Should trigger entry
        
        # Reset position manager
        position_manager._exit_position()
        
        # Just below entry threshold
        below_entry_signal = EnhancedTradingSignal(
            date=date(2023, 3, 16),
            signal_type="BUY_VOL",
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.UNDERPRICED,
            signal_strength=Decimal('0.70'),
            confidence_score=ConfidenceThresholds.ENTRY - Decimal('0.01'),  # 0.64
            recommended_position_size=Decimal('0.10'),
            risk_adjusted_size=Decimal('0.08'),
            reason="Just below entry threshold",
            exit_confidence=Decimal('0.60')
        )
        
        result = position_manager.process_signal(below_entry_signal)
        assert result['action'] == "HOLD"  # Should not trigger entry
    
    def test_zero_confidence_handling(self, position_manager, high_confidence_buy_signal):
        """Test handling of zero confidence signals."""
        # Enter position first
        position_manager.process_signal(high_confidence_buy_signal)
        
        zero_conf_signal = EnhancedTradingSignal(
            date=date(2023, 3, 16),
            signal_type="HOLD",
            current_state=VRPState.NORMAL_PREMIUM,
            predicted_state=VRPState.NORMAL_PREMIUM,
            signal_strength=Decimal('0.0'),
            confidence_score=Decimal('0.0'),
            recommended_position_size=Decimal('0.0'),
            risk_adjusted_size=Decimal('0.0'),
            reason="Zero confidence",
            exit_confidence=Decimal('0.0')
        )
        
        result = position_manager.process_signal(zero_conf_signal)
        assert result['action'] == "EXIT_TO_FLAT"  # Should exit on zero confidence
    
    def test_maximum_confidence_handling(self, position_manager):
        """Test handling of maximum confidence signals."""
        max_conf_signal = EnhancedTradingSignal(
            date=date(2023, 3, 15),
            signal_type="BUY_VOL",
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.UNDERPRICED,
            signal_strength=Decimal('1.0'),
            confidence_score=Decimal('1.0'),  # Maximum confidence
            recommended_position_size=Decimal('0.25'),
            risk_adjusted_size=Decimal('0.20'),
            reason="Maximum confidence",
            exit_confidence=Decimal('1.0')
        )
        
        result = position_manager.process_signal(max_conf_signal)
        assert result['action'] == "ENTER_LONG"
        assert result['new_size'] == max_conf_signal.risk_adjusted_size


class TestSignalEnhancement:
    """Test enhanced signals with exit confidence scores."""
    
    def test_enhanced_signal_creation(self):
        """Test creation of enhanced signals with exit confidence."""
        signal = EnhancedTradingSignal(
            date=date(2023, 3, 15),
            signal_type="BUY_VOL",
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.UNDERPRICED,
            signal_strength=Decimal('0.80'),
            confidence_score=Decimal('0.75'),
            recommended_position_size=Decimal('0.15'),
            risk_adjusted_size=Decimal('0.12'),
            reason="Enhanced signal test",
            exit_confidence=Decimal('0.65')
        )
        
        assert hasattr(signal, 'exit_confidence')
        assert signal.exit_confidence == Decimal('0.65')
        assert signal.confidence_score == Decimal('0.75')
    
    def test_exit_confidence_validation(self):
        """Test exit confidence validation."""
        # Valid exit confidence
        valid_signal = EnhancedTradingSignal(
            date=date(2023, 3, 15),
            signal_type="BUY_VOL",
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.UNDERPRICED,
            signal_strength=Decimal('0.80'),
            confidence_score=Decimal('0.75'),
            recommended_position_size=Decimal('0.15'),
            risk_adjusted_size=Decimal('0.12'),
            reason="Valid signal",
            exit_confidence=Decimal('0.65')
        )
        
        assert Decimal('0.0') <= valid_signal.exit_confidence <= Decimal('1.0')
    
    def test_signal_with_different_confidences(self, position_manager):
        """Test signal where entry and exit confidences differ significantly."""
        # High entry confidence, low exit confidence
        mixed_signal = EnhancedTradingSignal(
            date=date(2023, 3, 15),
            signal_type="BUY_VOL",
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.UNDERPRICED,
            signal_strength=Decimal('0.80'),
            confidence_score=Decimal('0.70'),  # High entry confidence
            recommended_position_size=Decimal('0.15'),
            risk_adjusted_size=Decimal('0.12'),
            reason="Mixed confidence signal",
            exit_confidence=Decimal('0.30')  # Low exit confidence
        )
        
        # From flat - should enter based on entry confidence
        result1 = position_manager.process_signal(mixed_signal)
        assert result1['action'] == "ENTER_LONG"
        
        # Create similar signal for existing position
        hold_signal = EnhancedTradingSignal(
            date=date(2023, 3, 16),
            signal_type="HOLD",
            current_state=VRPState.NORMAL_PREMIUM,
            predicted_state=VRPState.NORMAL_PREMIUM,
            signal_strength=Decimal('0.50'),
            confidence_score=Decimal('0.60'),  # Above exit threshold
            recommended_position_size=Decimal('0.10'),
            risk_adjusted_size=Decimal('0.08'),
            reason="Hold with low exit confidence",
            exit_confidence=Decimal('0.30')  # Low exit confidence
        )
        
        # Should exit based on low exit confidence
        result2 = position_manager.process_signal(hold_signal)
        assert result2['action'] == "EXIT_TO_FLAT"


class TestPerformanceMetrics:
    """Test performance improvement metrics."""
    
    def test_transaction_cost_reduction(self):
        """Test that confidence-based exits reduce transaction costs."""
        # Simulate traditional system (always positioned)
        traditional_transactions = 0
        position = "LONG"
        
        signals = ["BUY", "SELL", "BUY", "SELL", "BUY"]
        for signal in signals:
            if (position == "LONG" and signal == "SELL") or (position == "SHORT" and signal == "BUY"):
                traditional_transactions += 1  # Position change
                position = "SHORT" if signal == "SELL" else "LONG"
        
        # Simulate confidence-based system
        position_manager = PositionManager()
        confidence_transactions = 0
        
        test_signals = [
            ("BUY_VOL", Decimal('0.70')),   # Enter long
            ("SELL_VOL", Decimal('0.50')),  # Stay long (low confidence)
            ("HOLD", Decimal('0.30')),      # Exit to flat
            ("SELL_VOL", Decimal('0.80')),  # Enter short
            ("BUY_VOL", Decimal('0.40'))    # Stay short (low confidence)
        ]
        
        for signal_type, confidence in test_signals:
            signal = EnhancedTradingSignal(
                date=date(2023, 3, 15),
                signal_type=signal_type,
                current_state=VRPState.NORMAL_PREMIUM,
                predicted_state=VRPState.NORMAL_PREMIUM,
                signal_strength=confidence,
                confidence_score=confidence,
                recommended_position_size=Decimal('0.10'),
                risk_adjusted_size=Decimal('0.08'),
                reason="Performance test",
                exit_confidence=confidence
            )
            
            result = position_manager.process_signal(signal)
            if result['action'] in ["ENTER_LONG", "ENTER_SHORT", "EXIT_TO_FLAT", "FLIP_TO_SHORT", "FLIP_TO_LONG"]:
                confidence_transactions += 1
        
        # Confidence-based system should have fewer transactions
        assert confidence_transactions <= traditional_transactions
    
    def test_flat_period_percentage(self):
        """Test calculation of flat period percentage."""
        position_manager = PositionManager()
        total_signals = 20
        flat_periods = 0
        
        # Mix of high and low confidence signals
        for i in range(total_signals):
            confidence = Decimal('0.80') if i % 4 == 0 else Decimal('0.30')  # 25% high confidence
            
            signal = EnhancedTradingSignal(
                date=date(2023, 3, 15) + timedelta(days=i),
                signal_type="BUY_VOL",
                current_state=VRPState.FAIR_VALUE,
                predicted_state=VRPState.UNDERPRICED,
                signal_strength=confidence,
                confidence_score=confidence,
                recommended_position_size=Decimal('0.10'),
                risk_adjusted_size=Decimal('0.08'),
                reason="Test signal",
                exit_confidence=confidence
            )
            
            result = position_manager.process_signal(signal)
            if result['new_position'] == PositionState.FLAT:
                flat_periods += 1
        
        flat_percentage = (flat_periods / total_signals) * 100
        
        # Should have significant flat periods (>50% with mostly low confidence)
        assert flat_percentage > 50.0
    
    def test_whipsaw_reduction(self):
        """Test reduction in whipsaw trades."""
        position_manager = PositionManager()
        
        # Simulate alternating signals with varying confidence
        whipsaws = 0
        previous_position = PositionState.FLAT
        
        alternating_signals = [
            ("BUY_VOL", Decimal('0.80')),   # High confidence
            ("SELL_VOL", Decimal('0.45')),  # Low confidence - should not flip
            ("BUY_VOL", Decimal('0.40')),   # Low confidence
            ("SELL_VOL", Decimal('0.85')),  # High confidence
            ("BUY_VOL", Decimal('0.50')),   # Medium confidence
        ]
        
        for signal_type, confidence in alternating_signals:
            signal = EnhancedTradingSignal(
                date=date(2023, 3, 15),
                signal_type=signal_type,
                current_state=VRPState.NORMAL_PREMIUM,
                predicted_state=VRPState.NORMAL_PREMIUM,
                signal_strength=confidence,
                confidence_score=confidence,
                recommended_position_size=Decimal('0.10'),
                risk_adjusted_size=Decimal('0.08'),
                reason="Whipsaw test",
                exit_confidence=confidence
            )
            
            result = position_manager.process_signal(signal)
            
            # Count whipsaws (rapid position changes)
            if (previous_position != PositionState.FLAT and 
                result['new_position'] != PositionState.FLAT and
                previous_position != result['new_position']):
                whipsaws += 1
            
            previous_position = result['new_position']
        
        # Should have minimal whipsaws due to confidence thresholds
        assert whipsaws <= 1  # Allow at most 1 whipsaw


class TestSystemIntegration:
    """Integration tests for the complete confidence-based system."""
    
    def test_end_to_end_signal_processing(self):
        """Test end-to-end signal processing through confidence system."""
        position_manager = PositionManager()
        
        # Day 1: High confidence buy signal
        day1_signal = EnhancedTradingSignal(
            date=date(2023, 3, 15),
            signal_type="BUY_VOL",
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.UNDERPRICED,
            signal_strength=Decimal('0.85'),
            confidence_score=Decimal('0.80'),
            recommended_position_size=Decimal('0.15'),
            risk_adjusted_size=Decimal('0.12'),
            reason="Day 1: High confidence buy",
            exit_confidence=Decimal('0.75')
        )
        
        result1 = position_manager.process_signal(day1_signal)
        assert result1['action'] == "ENTER_LONG"
        
        # Day 2: Medium confidence - hold position
        day2_signal = EnhancedTradingSignal(
            date=date(2023, 3, 16),
            signal_type="HOLD",
            current_state=VRPState.NORMAL_PREMIUM,
            predicted_state=VRPState.NORMAL_PREMIUM,
            signal_strength=Decimal('0.55'),
            confidence_score=Decimal('0.55'),
            recommended_position_size=Decimal('0.10'),
            risk_adjusted_size=Decimal('0.08'),
            reason="Day 2: Medium confidence hold",
            exit_confidence=Decimal('0.50')
        )
        
        result2 = position_manager.process_signal(day2_signal)
        assert result2['action'] == "HOLD"
        assert result2['new_position'] == PositionState.LONG_VOL
        
        # Day 3: Low confidence - exit position
        day3_signal = EnhancedTradingSignal(
            date=date(2023, 3, 17),
            signal_type="HOLD",
            current_state=VRPState.NORMAL_PREMIUM,
            predicted_state=VRPState.NORMAL_PREMIUM,
            signal_strength=Decimal('0.25'),
            confidence_score=Decimal('0.30'),
            recommended_position_size=Decimal('0.05'),
            risk_adjusted_size=Decimal('0.03'),
            reason="Day 3: Low confidence",
            exit_confidence=Decimal('0.25')
        )
        
        result3 = position_manager.process_signal(day3_signal)
        assert result3['action'] == "EXIT_TO_FLAT"
        assert result3['new_position'] == PositionState.FLAT
        
        # Day 4: Stay flat during uncertainty
        day4_signal = EnhancedTradingSignal(
            date=date(2023, 3, 18),
            signal_type="BUY_VOL",
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.UNDERPRICED,
            signal_strength=Decimal('0.45'),
            confidence_score=Decimal('0.50'),  # Below entry threshold
            recommended_position_size=Decimal('0.08'),
            risk_adjusted_size=Decimal('0.06'),
            reason="Day 4: Uncertainty",
            exit_confidence=Decimal('0.40')
        )
        
        result4 = position_manager.process_signal(day4_signal)
        assert result4['action'] == "HOLD"
        assert result4['new_position'] == PositionState.FLAT
        
        # Day 5: Very high confidence - enter opposite position
        day5_signal = EnhancedTradingSignal(
            date=date(2023, 3, 19),
            signal_type="SELL_VOL",
            current_state=VRPState.ELEVATED_PREMIUM,
            predicted_state=VRPState.EXTREME_HIGH,
            signal_strength=Decimal('0.95'),
            confidence_score=Decimal('0.90'),
            recommended_position_size=Decimal('0.20'),
            risk_adjusted_size=Decimal('0.18'),
            reason="Day 5: Very high confidence sell",
            exit_confidence=Decimal('0.85')
        )
        
        result5 = position_manager.process_signal(day5_signal)
        assert result5['action'] == "ENTER_SHORT"
        assert result5['new_position'] == PositionState.SHORT_VOL
    
    def test_backward_compatibility(self):
        """Test backward compatibility with existing signal format."""
        position_manager = PositionManager()
        
        # Regular TradingSignal without exit_confidence
        regular_signal = TradingSignal(
            date=date(2023, 3, 15),
            signal_type="BUY_VOL",
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.UNDERPRICED,
            signal_strength=Decimal('0.80'),
            confidence_score=Decimal('0.75'),
            recommended_position_size=Decimal('0.15'),
            risk_adjusted_size=Decimal('0.12'),
            reason="Regular signal without exit confidence"
        )
        
        # Should handle regular signals (exit_confidence defaults to 0.0)
        result = position_manager.process_signal(regular_signal)
        
        # Should work but exit_confidence will be 0.0, potentially causing immediate exit
        # This tests that the system doesn't crash on regular signals
        assert result is not None
        assert 'action' in result


class TestRiskManagementScenarios:
    """Test risk management improvements with confidence-based exits."""
    
    def test_early_exit_on_confidence_loss(self, position_manager, high_confidence_buy_signal):
        """Test early exit when confidence deteriorates."""
        # Enter position
        position_manager.process_signal(high_confidence_buy_signal)
        original_entry_date = position_manager.entry_date
        
        # Confidence deteriorates rapidly
        deteriorating_signal = EnhancedTradingSignal(
            date=date(2023, 3, 16),
            signal_type="HOLD",
            current_state=VRPState.NORMAL_PREMIUM,
            predicted_state=VRPState.NORMAL_PREMIUM,
            signal_strength=Decimal('0.20'),
            confidence_score=Decimal('0.25'),  # Rapid confidence loss
            recommended_position_size=Decimal('0.03'),
            risk_adjusted_size=Decimal('0.02'),
            reason="Rapid confidence deterioration",
            exit_confidence=Decimal('0.15')  # Very low
        )
        
        result = position_manager.process_signal(deteriorating_signal)
        
        assert result['action'] == "EXIT_TO_FLAT"
        assert position_manager.entry_date is None
        
        # Position was held for only 1 day - early exit prevented larger loss
        assert (deteriorating_signal.date - original_entry_date).days == 1
    
    def test_volatility_spike_response(self, position_manager):
        """Test response to volatility spikes with confidence adjustment."""
        # Normal entry
        normal_signal = EnhancedTradingSignal(
            date=date(2023, 3, 15),
            signal_type="BUY_VOL",
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.UNDERPRICED,
            signal_strength=Decimal('0.70'),
            confidence_score=Decimal('0.70'),
            recommended_position_size=Decimal('0.15'),
            risk_adjusted_size=Decimal('0.12'),
            reason="Normal market entry",
            exit_confidence=Decimal('0.65')
        )
        
        position_manager.process_signal(normal_signal)
        assert position_manager.current_position == PositionState.LONG_VOL
        
        # Volatility spike with adjusted confidence
        spike_signal = EnhancedTradingSignal(
            date=date(2023, 3, 16),
            signal_type="HOLD",
            current_state=VRPState.ELEVATED_PREMIUM,
            predicted_state=VRPState.ELEVATED_PREMIUM,
            signal_strength=Decimal('0.40'),
            confidence_score=Decimal('0.35'),  # Confidence drops due to volatility
            recommended_position_size=Decimal('0.06'),
            risk_adjusted_size=Decimal('0.04'),
            reason="Volatility spike - confidence adjusted",
            exit_confidence=Decimal('0.30')  # Low due to spike
        )
        
        result = position_manager.process_signal(spike_signal)
        
        # Should exit due to low confidence during volatility spike
        assert result['action'] == "EXIT_TO_FLAT"
        assert "confidence" in result['reasoning']
    
    def test_drawdown_protection(self):
        """Test drawdown protection through confidence-based exits."""
        position_manager = PositionManager()
        
        # Track position changes and simulate drawdown scenario
        positions = []
        actions = []
        
        # Simulated market scenario with declining confidence
        market_scenario = [
            ("BUY_VOL", Decimal('0.80'), Decimal('0.75')),  # Strong entry
            ("HOLD", Decimal('0.60'), Decimal('0.55')),     # Weakening
            ("HOLD", Decimal('0.45'), Decimal('0.40')),     # Further weakness  
            ("HOLD", Decimal('0.30'), Decimal('0.25')),     # Should trigger exit
            ("BUY_VOL", Decimal('0.25'), Decimal('0.20')),  # Should stay flat
        ]
        
        for i, (signal_type, confidence, exit_conf) in enumerate(market_scenario):
            signal = EnhancedTradingSignal(
                date=date(2023, 3, 15) + timedelta(days=i),
                signal_type=signal_type,
                current_state=VRPState.NORMAL_PREMIUM,
                predicted_state=VRPState.NORMAL_PREMIUM,
                signal_strength=confidence,
                confidence_score=confidence,
                recommended_position_size=Decimal('0.10'),
                risk_adjusted_size=Decimal('0.08'),
                reason=f"Drawdown test day {i+1}",
                exit_confidence=exit_conf
            )
            
            result = position_manager.process_signal(signal)
            positions.append(result['new_position'])
            actions.append(result['action'])
        
        # Should have exited before final low confidence signal
        assert PositionState.FLAT in positions[2:]  # Exit by day 3-4
        assert "EXIT_TO_FLAT" in actions
        
        # Should not re-enter on final low confidence signal
        assert positions[-1] == PositionState.FLAT
        assert actions[-1] == "HOLD"