"""
Position Manager for Confidence-Based Exits

Manages trading positions using a three-tier confidence system for improved
risk management and reduced transaction costs.

This module implements:
- Three position states: FLAT, LONG_VOL, SHORT_VOL  
- Confidence thresholds: Entry (0.65), Exit (0.40), Flip (0.75)
- Enhanced position state transitions with confidence validation
- Comprehensive logging for position state changes
"""

import logging
from decimal import Decimal
from datetime import date
from typing import Dict, Any, Optional
from enum import Enum

from ..models.data_models import TradingSignal, ExitConfidenceConfig

logger = logging.getLogger(__name__)


class PositionState(str, Enum):
    """Enhanced position states for confidence-based exits."""
    FLAT = "FLAT"
    LONG_VOL = "LONG_VOL" 
    SHORT_VOL = "SHORT_VOL"
    PENDING_EXIT = "PENDING_EXIT"


class ConfidenceThresholds:
    """Three-tier confidence thresholds."""
    ENTRY = Decimal('0.65')
    EXIT = Decimal('0.40') 
    FLIP = Decimal('0.75')


class PositionManager:
    """
    Enhanced position manager with confidence-based exits.
    
    Implements three-tier confidence system for better risk management
    and reduced transaction costs. Uses confidence scores to determine
    when to enter, hold, or exit positions.
    
    Architecture principles:
    - Encapsulated state mutations in dedicated methods
    - Clear breadcrumb logging for all state changes
    - Centralized position management logic
    - Type-safe with Pydantic integration
    """
    
    def __init__(self, config: Optional[ExitConfidenceConfig] = None):
        """
        Initialize position manager with confidence configuration.
        
        Args:
            config: Configuration for confidence thresholds
        """
        self.config = config or ExitConfidenceConfig()
        self.current_position = PositionState.FLAT
        self.position_size = Decimal('0.0')
        self.entry_date: Optional[date] = None
        self.last_signal_confidence = Decimal('0.0')
        
        logger.info("[SYSTEM] PositionManager initialized with confidence thresholds: "
                   f"entry={self.config.entry_threshold}, exit={self.config.exit_threshold}, "
                   f"flip={self.config.flip_threshold}")
    
    def process_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """
        Process trading signal with confidence-based logic.
        
        Implements the core confidence-based position management algorithm:
        1. From FLAT: Enter only if confidence >= entry_threshold
        2. From position: Exit if confidence <= exit_threshold OR exit_confidence <= exit_threshold
        3. From position: Flip if opposite signal confidence >= flip_threshold
        
        Args:
            signal: Trading signal with confidence metrics
            
        Returns:
            Dict with position state changes, actions, and reasoning
        """
        old_position = self.current_position
        old_size = self.position_size
        confidence = signal.confidence_score
        exit_confidence = getattr(signal, 'exit_confidence', Decimal('0.0'))
        
        # Default to signal confidence if exit_confidence is not provided or None
        if exit_confidence is None:
            exit_confidence = confidence
            
        action = "HOLD"
        reasoning = "No position change"
        
        logger.debug(f"[SYSTEM] Processing signal: type={signal.signal_type}, "
                    f"confidence={confidence:.3f}, exit_confidence={exit_confidence:.3f}, "
                    f"current_position={self.current_position}")
        
        # Position logic based on current state
        if self.current_position == PositionState.FLAT:
            action, reasoning = self._handle_flat_position(signal, confidence)
            
        elif self.current_position == PositionState.LONG_VOL:
            action, reasoning = self._handle_long_position(signal, confidence, exit_confidence)
            
        elif self.current_position == PositionState.SHORT_VOL:
            action, reasoning = self._handle_short_position(signal, confidence, exit_confidence)
        
        # Log the decision
        if action != "HOLD":
            logger.info(f"[SYSTEM] Position action: {action} - Reason: {reasoning}")
        
        return {
            'old_position': old_position,
            'new_position': self.current_position,
            'old_size': old_size,
            'new_size': self.position_size,
            'action': action,
            'reasoning': reasoning,
            'confidence': confidence,
            'exit_confidence': exit_confidence
        }
    
    def _handle_flat_position(self, signal: TradingSignal, confidence: Decimal) -> tuple[str, str]:
        """Handle signal processing when position is flat."""
        if confidence >= self.config.entry_threshold:
            if signal.signal_type == "BUY_VOL":
                self._enter_long_position(signal)
                return "ENTER_LONG", f"High confidence ({confidence:.3f}) entry signal"
            elif signal.signal_type == "SELL_VOL":
                self._enter_short_position(signal)
                return "ENTER_SHORT", f"High confidence ({confidence:.3f}) entry signal"
        
        return "HOLD", f"Confidence ({confidence:.3f}) below entry threshold ({self.config.entry_threshold})"
    
    def _handle_long_position(self, signal: TradingSignal, confidence: Decimal, exit_confidence: Decimal) -> tuple[str, str]:
        """Handle signal processing when in long volatility position."""
        # Check exit conditions first
        if exit_confidence <= self.config.exit_threshold:
            self._exit_position("Low exit confidence")
            return "EXIT_TO_FLAT", f"Low exit confidence ({exit_confidence:.3f}) - go flat"
        
        if confidence <= self.config.exit_threshold:
            self._exit_position("Low confidence")
            return "EXIT_TO_FLAT", f"Low confidence ({confidence:.3f}) - exit position"
        
        # Check flip condition
        if (signal.signal_type == "SELL_VOL" and confidence >= self.config.flip_threshold):
            self._flip_to_short_position(signal)
            return "FLIP_TO_SHORT", f"Very high confidence ({confidence:.3f}) flip signal"
        
        return "HOLD", f"Maintaining long position - confidence ({confidence:.3f}) sufficient"
    
    def _handle_short_position(self, signal: TradingSignal, confidence: Decimal, exit_confidence: Decimal) -> tuple[str, str]:
        """Handle signal processing when in short volatility position."""
        # Check exit conditions first
        if exit_confidence <= self.config.exit_threshold:
            self._exit_position("Low exit confidence")
            return "EXIT_TO_FLAT", f"Low exit confidence ({exit_confidence:.3f}) - go flat"
        
        if confidence <= self.config.exit_threshold:
            self._exit_position("Low confidence")
            return "EXIT_TO_FLAT", f"Low confidence ({confidence:.3f}) - exit position"
        
        # Check flip condition
        if (signal.signal_type == "BUY_VOL" and confidence >= self.config.flip_threshold):
            self._flip_to_long_position(signal)
            return "FLIP_TO_LONG", f"Very high confidence ({confidence:.3f}) flip signal"
        
        return "HOLD", f"Maintaining short position - confidence ({confidence:.3f}) sufficient"
    
    def _enter_long_position(self, signal: TradingSignal):
        """
        Enter long volatility position.
        
        Args:
            signal: Trading signal triggering the position
        """
        old_position = self.current_position
        self.current_position = PositionState.LONG_VOL
        self.position_size = signal.risk_adjusted_size
        self.entry_date = signal.date
        self.last_signal_confidence = signal.confidence_score
        
        logger.info(f"[SYSTEM] Entered long position - Size: {self.position_size}, "
                   f"Date: {self.entry_date}, Previous: {old_position}")
    
    def _enter_short_position(self, signal: TradingSignal):
        """
        Enter short volatility position.
        
        Args:
            signal: Trading signal triggering the position
        """
        old_position = self.current_position
        self.current_position = PositionState.SHORT_VOL
        self.position_size = -signal.risk_adjusted_size
        self.entry_date = signal.date
        self.last_signal_confidence = signal.confidence_score
        
        logger.info(f"[SYSTEM] Entered short position - Size: {self.position_size}, "
                   f"Date: {self.entry_date}, Previous: {old_position}")
    
    def _exit_position(self, reason: str = "Exit signal"):
        """
        Exit current position to flat state.
        
        Args:
            reason: Reason for exiting position
        """
        old_position = self.current_position
        old_size = self.position_size
        old_entry_date = self.entry_date
        
        self.current_position = PositionState.FLAT
        self.position_size = Decimal('0.0')
        self.entry_date = None
        self.last_signal_confidence = Decimal('0.0')
        
        logger.info(f"[SYSTEM] Exited position to flat - Reason: {reason}, "
                   f"Previous: {old_position}, Size: {old_size}, "
                   f"Entry date: {old_entry_date}")
    
    def _flip_to_short_position(self, signal: TradingSignal):
        """
        Flip from long to short position.
        
        Args:
            signal: Trading signal triggering the flip
        """
        old_position = self.current_position
        old_size = self.position_size
        
        self.current_position = PositionState.SHORT_VOL
        self.position_size = -signal.risk_adjusted_size
        self.entry_date = signal.date
        self.last_signal_confidence = signal.confidence_score
        
        logger.info(f"[SYSTEM] Flipped to short position - Previous: {old_position} ({old_size}), "
                   f"New size: {self.position_size}, Date: {self.entry_date}")
    
    def _flip_to_long_position(self, signal: TradingSignal):
        """
        Flip from short to long position.
        
        Args:
            signal: Trading signal triggering the flip
        """
        old_position = self.current_position
        old_size = self.position_size
        
        self.current_position = PositionState.LONG_VOL
        self.position_size = signal.risk_adjusted_size
        self.entry_date = signal.date
        self.last_signal_confidence = signal.confidence_score
        
        logger.info(f"[SYSTEM] Flipped to long position - Previous: {old_position} ({old_size}), "
                   f"New size: {self.position_size}, Date: {self.entry_date}")
    
    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get current position summary.
        
        Returns:
            Dict with position state, size, and metadata
        """
        return {
            'position': self.current_position,
            'size': self.position_size,
            'entry_date': self.entry_date,
            'last_confidence': self.last_signal_confidence,
            'is_flat': self.current_position == PositionState.FLAT,
            'is_long': self.current_position == PositionState.LONG_VOL,
            'is_short': self.current_position == PositionState.SHORT_VOL
        }
    
    def reset_position(self):
        """
        Reset position manager to flat state.
        
        Used for backtesting and system resets.
        """
        old_state = self.get_position_summary()
        self._exit_position("System reset")
        
        logger.info(f"[SYSTEM] Position manager reset - Previous state: {old_state}")