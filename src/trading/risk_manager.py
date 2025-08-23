"""
Risk management for VRP Trading System.

This module implements comprehensive risk management including position sizing,
risk limit validation, portfolio risk monitoring, and position exit criteria.
It ensures the system operates within defined risk parameters while
maximizing risk-adjusted returns.
"""

import logging
from decimal import Decimal
from typing import Dict, List

import numpy as np

from src.config.settings import VRPTradingConfig
from src.interfaces.contracts import RiskManagerInterface
from src.models.data_models import Position, TradingSignal, VRPState, VolatilityMetrics
from src.utils.exceptions import RiskViolationError


logger = logging.getLogger(__name__)


class VRPRiskManager(RiskManagerInterface):
    """
    Comprehensive risk management for VRP trading strategy.
    
    This class implements multiple layers of risk control including:
    - Position sizing based on Kelly criterion and volatility targeting
    - Portfolio-level risk limits and diversification rules
    - Dynamic position monitoring and exit criteria
    - Circuit breaker functionality for extreme scenarios
    """
    
    def __init__(self, config: VRPTradingConfig):
        """
        Initialize risk manager with configuration.
        
        Args:
            config: System configuration containing risk parameters
        """
        self.config = config
        
        # Risk tracking
        self.circuit_breaker_triggered = False
        self.consecutive_losses = 0
        self.daily_pnl_tracking = []
        
        logger.info("Initialized VRPRiskManager")
    
    def calculate_position_size(
        self, 
        signal: TradingSignal,
        current_portfolio_value: float,
        current_positions: List[Position]
    ) -> float:
        """
        Calculate appropriate position size for signal.
        
        Uses multiple approaches and takes the most conservative:
        - Fixed fraction based on configuration
        - Volatility targeting
        - Kelly criterion estimation
        - Risk parity considerations
        
        Args:
            signal: Trading signal to size
            current_portfolio_value: Current portfolio value
            current_positions: Existing positions
            
        Returns:
            Position size as dollar amount
            
        Raises:
            RiskViolationError: If position would violate risk limits
        """
        logger.info(f"Calculating position size for {signal.signal_type} signal")
        
        try:
            # Method 1: Fixed fraction
            fixed_fraction_size = self._calculate_fixed_fraction_size(
                signal, current_portfolio_value
            )
            
            # Method 2: Volatility targeting
            vol_target_size = self._calculate_volatility_target_size(
                signal, current_portfolio_value
            )
            
            # Method 3: Kelly criterion (if enough historical data)
            kelly_size = self._calculate_kelly_size(signal, current_portfolio_value)
            
            # Take the most conservative (smallest) size
            candidate_sizes = [fixed_fraction_size, vol_target_size]
            if kelly_size > 0:
                candidate_sizes.append(kelly_size)
            
            base_position_size = min(candidate_sizes)
            
            # Apply portfolio-level adjustments
            adjusted_size = self._apply_portfolio_adjustments(
                base_position_size, signal, current_positions, current_portfolio_value
            )
            
            # Final validation
            self._validate_position_size(
                adjusted_size, signal, current_positions, current_portfolio_value
            )
            
            logger.info(f"Calculated position size: ${adjusted_size:,.2f}")
            return adjusted_size
        
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            raise RiskViolationError(f"Position sizing failed: {str(e)}")
    
    def validate_risk_limits(
        self, 
        proposed_position: Position,
        current_positions: List[Position]
    ) -> bool:
        """
        Validate position against risk limits.
        
        Checks multiple risk constraints:
        - Maximum position size
        - Portfolio concentration limits
        - Correlation exposure limits
        - Drawdown limits
        
        Args:
            proposed_position: Position to validate
            current_positions: Current portfolio positions
            
        Returns:
            True if position passes all risk checks
        """
        logger.debug(f"Validating risk limits for position {proposed_position.position_id}")
        
        try:
            # Check 1: Maximum position size
            max_position_value = float(self.config.risk.max_position_size) * 100000  # Assume 100k base
            if abs(proposed_position.position_size) > max_position_value:
                logger.warning(f"Position size exceeds maximum: {proposed_position.position_size}")
                return False
            
            # Check 2: Total volatility exposure
            total_vol_exposure = self._calculate_total_volatility_exposure(
                current_positions + [proposed_position]
            )
            
            max_vol_exposure = float(self.config.risk.max_correlation_exposure)
            if abs(total_vol_exposure) > max_vol_exposure:
                logger.warning(f"Total volatility exposure exceeds limit: {total_vol_exposure}")
                return False
            
            # Check 3: Circuit breaker
            if self.circuit_breaker_triggered:
                logger.warning("Circuit breaker is active - no new positions allowed")
                return False
            
            # Check 4: Maximum number of positions
            if len(current_positions) >= 10:  # Arbitrary limit
                logger.warning("Maximum number of positions reached")
                return False
            
            # Check 5: Opposite position validation
            if not self._validate_opposite_positions(proposed_position, current_positions):
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating risk limits: {str(e)}")
            return False
    
    def calculate_portfolio_risk_metrics(self, positions: List[Position]) -> Dict[str, float]:
        """
        Calculate current portfolio risk metrics.
        
        Args:
            positions: Current portfolio positions
            
        Returns:
            Dictionary with risk metrics
        """
        if not positions:
            return {
                'total_exposure': 0.0,
                'volatility_exposure': 0.0,
                'num_positions': 0,
                'concentration_risk': 0.0,
                'estimated_portfolio_vol': 0.0
            }
        
        try:
            # Calculate exposures
            total_exposure = sum(abs(pos.position_size) for pos in positions if pos.is_active)
            vol_exposure = self._calculate_total_volatility_exposure(positions)
            
            # Calculate concentration risk (max single position / total)
            if total_exposure > 0:
                max_position = max(abs(pos.position_size) for pos in positions if pos.is_active)
                concentration_risk = max_position / total_exposure
            else:
                concentration_risk = 0.0
            
            # Estimate portfolio volatility
            portfolio_vol = self._estimate_portfolio_volatility(positions)
            
            metrics = {
                'total_exposure': total_exposure,
                'volatility_exposure': vol_exposure,
                'num_positions': len([p for p in positions if p.is_active]),
                'concentration_risk': concentration_risk,
                'estimated_portfolio_vol': portfolio_vol,
                'long_vol_exposure': sum(
                    pos.position_size for pos in positions 
                    if pos.is_active and pos.position_type == "LONG_VOL"
                ),
                'short_vol_exposure': sum(
                    abs(pos.position_size) for pos in positions 
                    if pos.is_active and pos.position_type == "SHORT_VOL"
                )
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {str(e)}")
            return {}
    
    def should_exit_position(
        self, 
        position: Position,
        current_metrics: VolatilityMetrics
    ) -> bool:
        """
        Determine if position should be exited.
        
        Considers multiple exit criteria:
        - Stop loss levels
        - Profit taking levels
        - State change criteria
        - Time-based exits
        - Risk limit breaches
        
        Args:
            position: Position to evaluate
            current_metrics: Current market metrics
            
        Returns:
            True if position should be exited
        """
        logger.debug(f"Evaluating exit criteria for position {position.position_id}")
        
        try:
            # Exit criterion 1: Stop loss
            if self._check_stop_loss(position):
                logger.info(f"Stop loss triggered for position {position.position_id}")
                return True
            
            # Exit criterion 2: Profit target
            if self._check_profit_target(position):
                logger.info(f"Profit target reached for position {position.position_id}")
                return True
            
            # Exit criterion 3: State-based exit
            if self._check_state_based_exit(position, current_metrics):
                logger.info(f"State-based exit triggered for position {position.position_id}")
                return True
            
            # Exit criterion 4: Time-based exit
            if self._check_time_based_exit(position):
                logger.info(f"Time-based exit triggered for position {position.position_id}")
                return True
            
            # Exit criterion 5: Risk limit breach
            if self._check_risk_limit_exit(position, current_metrics):
                logger.info(f"Risk limit exit triggered for position {position.position_id}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error evaluating exit criteria: {str(e)}")
            return True  # Conservative: exit on error
    
    def update_circuit_breaker(self, daily_pnl: float, portfolio_value: float) -> None:
        """
        Update circuit breaker status based on recent performance.
        
        Args:
            daily_pnl: Today's P&L
            portfolio_value: Current portfolio value
        """
        self.daily_pnl_tracking.append(daily_pnl)
        
        # Keep only last 5 days
        if len(self.daily_pnl_tracking) > 5:
            self.daily_pnl_tracking.pop(0)
        
        # Check for consecutive losses
        if daily_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Trigger circuit breaker conditions
        threshold = int(self.config.risk.circuit_breaker_threshold)
        
        if self.consecutive_losses >= threshold:
            self.circuit_breaker_triggered = True
            logger.warning(f"Circuit breaker triggered: {self.consecutive_losses} consecutive losses")
        
        # Check for large daily loss
        daily_loss_threshold = 0.05  # 5% daily loss
        if daily_pnl / portfolio_value < -daily_loss_threshold:
            self.circuit_breaker_triggered = True
            logger.warning(f"Circuit breaker triggered: Large daily loss {daily_pnl/portfolio_value:.2%}")
        
        # Auto-reset circuit breaker after period of good performance
        if (len(self.daily_pnl_tracking) >= 3 and 
            all(pnl >= 0 for pnl in self.daily_pnl_tracking[-3:]) and
            self.circuit_breaker_triggered):
            self.circuit_breaker_triggered = False
            self.consecutive_losses = 0
            logger.info("Circuit breaker reset: Recent positive performance")
    
    def _calculate_fixed_fraction_size(
        self, 
        signal: TradingSignal, 
        portfolio_value: float
    ) -> float:
        """Calculate position size using fixed fraction method."""
        base_fraction = float(signal.risk_adjusted_size)
        return portfolio_value * base_fraction
    
    def _calculate_volatility_target_size(
        self, 
        signal: TradingSignal, 
        portfolio_value: float
    ) -> float:
        """Calculate position size using volatility targeting."""
        # Target portfolio volatility of 15%
        target_vol = 0.15
        
        # Estimate position volatility (simplified)
        # In practice, this would use option Greeks or volatility sensitivity
        position_vol = 0.25  # Assume 25% volatility for volatility positions
        
        # Calculate size to achieve target portfolio volatility contribution
        vol_contribution_target = target_vol * 0.3  # 30% of target vol from this position
        position_size = portfolio_value * vol_contribution_target / position_vol
        
        return position_size
    
    def _calculate_kelly_size(self, signal: TradingSignal, portfolio_value: float) -> float:
        """Calculate position size using Kelly criterion."""
        # Simplified Kelly calculation
        # In practice, would need historical win rate and average win/loss
        
        # Placeholder values (would be calculated from historical performance)
        win_rate = 0.55  # 55% win rate
        avg_win = 0.08   # 8% average win
        avg_loss = 0.05  # 5% average loss
        
        if win_rate <= 0.5 or avg_loss <= 0:
            return 0.0  # No Kelly sizing if no edge or invalid parameters
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply conservative scaling (typically use 25-50% of Kelly)
        conservative_kelly = kelly_fraction * 0.25
        
        # Apply signal strength scaling
        signal_adjusted_kelly = conservative_kelly * float(signal.signal_strength)
        
        return portfolio_value * max(0.0, min(0.2, signal_adjusted_kelly))  # Cap at 20%
    
    def _apply_portfolio_adjustments(
        self, 
        base_size: float, 
        signal: TradingSignal,
        current_positions: List[Position],
        portfolio_value: float
    ) -> float:
        """Apply portfolio-level adjustments to position size."""
        adjusted_size = base_size
        
        # Adjustment 1: Reduce size if we already have similar positions
        similar_exposure = self._calculate_similar_exposure(signal, current_positions)
        if similar_exposure > 0:
            reduction_factor = max(0.5, 1.0 - similar_exposure / portfolio_value)
            adjusted_size *= reduction_factor
            logger.debug(f"Similar exposure adjustment: {reduction_factor:.2f}")
        
        # Adjustment 2: Scale down if portfolio is highly concentrated
        risk_metrics = self.calculate_portfolio_risk_metrics(current_positions)
        if risk_metrics.get('concentration_risk', 0) > 0.4:  # 40% concentration
            adjusted_size *= 0.8
            logger.debug("High concentration adjustment applied")
        
        # Adjustment 3: Circuit breaker proximity
        if self.consecutive_losses >= 3:
            adjusted_size *= 0.7
            logger.debug("Circuit breaker proximity adjustment applied")
        
        return adjusted_size
    
    def _validate_position_size(
        self, 
        position_size: float, 
        signal: TradingSignal,
        current_positions: List[Position],
        portfolio_value: float
    ) -> None:
        """Validate final position size against limits."""
        max_position_value = portfolio_value * float(self.config.risk.max_position_size)
        
        if abs(position_size) > max_position_value:
            raise RiskViolationError(
                f"Position size ${position_size:,.2f} exceeds maximum ${max_position_value:,.2f}"
            )
        
        # Check total exposure after this position
        current_exposure = sum(abs(p.position_size) for p in current_positions if p.is_active)
        total_exposure = current_exposure + abs(position_size)
        
        if total_exposure > portfolio_value * 2.0:  # 200% gross exposure limit
            raise RiskViolationError(
                f"Total exposure ${total_exposure:,.2f} would exceed 200% of portfolio"
            )
    
    def _calculate_total_volatility_exposure(self, positions: List[Position]) -> float:
        """Calculate net volatility exposure across positions."""
        long_vol_exposure = sum(
            pos.position_size for pos in positions 
            if pos.is_active and pos.position_type == "LONG_VOL"
        )
        
        short_vol_exposure = sum(
            pos.position_size for pos in positions 
            if pos.is_active and pos.position_type == "SHORT_VOL"
        )
        
        return long_vol_exposure - short_vol_exposure
    
    def _validate_opposite_positions(
        self, 
        proposed_position: Position, 
        current_positions: List[Position]
    ) -> bool:
        """Validate that opposite positions don't create excessive hedge."""
        proposed_type = proposed_position.position_type
        
        for pos in current_positions:
            if not pos.is_active:
                continue
                
            # Check for opposite position types
            if ((proposed_type == "LONG_VOL" and pos.position_type == "SHORT_VOL") or
                (proposed_type == "SHORT_VOL" and pos.position_type == "LONG_VOL")):
                
                # Allow small opposite positions for hedging
                if abs(proposed_position.position_size) < abs(pos.position_size) * 0.5:
                    continue
                else:
                    logger.warning("Large opposite position would create excessive hedge")
                    return False
        
        return True
    
    def _estimate_portfolio_volatility(self, positions: List[Position]) -> float:
        """Estimate portfolio volatility from positions."""
        if not positions:
            return 0.0
        
        # Simplified portfolio volatility estimation
        # In practice, would use proper correlation matrices and position Greeks
        
        position_vols = []
        position_weights = []
        total_value = sum(abs(p.position_size) for p in positions if p.is_active)
        
        if total_value == 0:
            return 0.0
        
        for pos in positions:
            if not pos.is_active:
                continue
                
            # Estimate individual position volatility
            if pos.position_type == "LONG_VOL":
                pos_vol = 0.30  # 30% volatility for long vol positions
            else:
                pos_vol = 0.35  # 35% volatility for short vol positions
            
            weight = abs(pos.position_size) / total_value
            position_vols.append(pos_vol)
            position_weights.append(weight)
        
        # Simple weighted average (ignoring correlations)
        # Real implementation would use covariance matrix
        weighted_vol = sum(w * v for w, v in zip(position_weights, position_vols))
        
        return weighted_vol
    
    def _calculate_similar_exposure(
        self, 
        signal: TradingSignal, 
        current_positions: List[Position]
    ) -> float:
        """Calculate exposure to similar positions."""
        signal_type = signal.signal_type
        similar_exposure = 0.0
        
        for pos in current_positions:
            if not pos.is_active:
                continue
                
            # Check for similar position types
            if ((signal_type == "BUY_VOL" and pos.position_type == "LONG_VOL") or
                (signal_type == "SELL_VOL" and pos.position_type == "SHORT_VOL")):
                similar_exposure += abs(pos.position_size)
        
        return similar_exposure
    
    def _check_stop_loss(self, position: Position) -> bool:
        """Check if stop loss should trigger."""
        if (self.config.risk.stop_loss_threshold is None or 
            position.unrealized_pnl is None):
            return False
        
        stop_loss_threshold = float(self.config.risk.stop_loss_threshold)
        unrealized_pnl_pct = float(position.unrealized_pnl) / abs(position.position_size)
        
        return unrealized_pnl_pct < -stop_loss_threshold
    
    def _check_profit_target(self, position: Position) -> bool:
        """Check if profit target should trigger."""
        if position.unrealized_pnl is None:
            return False
        
        # Take profits at 25% gain
        profit_target = 0.25
        unrealized_pnl_pct = float(position.unrealized_pnl) / abs(position.position_size)
        
        return unrealized_pnl_pct > profit_target
    
    def _check_state_based_exit(self, position: Position, current_metrics: VolatilityMetrics) -> bool:
        """Check if state change should trigger exit."""
        entry_signal_state = position.entry_signal.predicted_state
        current_state = current_metrics.vrp_state
        
        # Exit if we've moved significantly away from the predicted state
        # This is a simplified logic - could be more sophisticated
        
        if position.position_type == "LONG_VOL":
            # Exit long vol if we're no longer in underpriced territory
            return current_state not in {VRPState.UNDERPRICED, VRPState.FAIR_VALUE}
        
        elif position.position_type == "SHORT_VOL":
            # Exit short vol if we're no longer in overpriced territory
            return current_state not in {VRPState.EXTREME_PREMIUM, VRPState.ELEVATED_PREMIUM}
        
        return False
    
    def _check_time_based_exit(self, position: Position) -> bool:
        """Check if time-based exit should trigger."""
        from datetime import date, timedelta
        
        # Exit positions after 30 days
        max_holding_period = timedelta(days=30)
        holding_period = date.today() - position.entry_date
        
        return holding_period > max_holding_period
    
    def _check_risk_limit_exit(self, position: Position, current_metrics: VolatilityMetrics) -> bool:
        """Check if risk limits require position exit."""
        # Exit if realized volatility becomes extremely high
        realized_vol = float(current_metrics.realized_vol_30d)
        
        if realized_vol > 0.6:  # 60% realized volatility
            logger.warning(f"Extremely high realized volatility: {realized_vol:.1%}")
            return True
        
        return False