"""
Adaptive VRP State Classification Service

Provides quantile-based VRP state classification that adapts to market conditions
without using fixed thresholds. States are determined by historical distribution.
"""

import logging
from collections import deque
from decimal import Decimal
from typing import List, Optional

import numpy as np

from src.config.settings import get_settings
from src.data.volatility_calculator import VolatilityCalculator
from src.interfaces.contracts import IVRPClassifier
from src.models.data_models import VRPState
from src.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


class VRPClassifier(IVRPClassifier):
    """
    Adaptive VRP state classification using quantile-based boundaries.

    Eliminates fixed thresholds and adapts to market regime changes by using
    rolling quantiles from historical VRP distribution.
    """

    def __init__(self, config=None):
        """Initialize VRP classifier with adaptive quantile-based boundaries."""
        self.settings = config if config is not None else get_settings()

        # Initialize volatility calculator for VRP calculations
        self.volatility_calculator = VolatilityCalculator(self.settings)

        # Adaptive quantile-based configuration
        self.quantile_percentiles = [10, 30, 70, 90]  # For 5-state classification
        self.quantile_window = 252  # Rolling window for quantile calculation (1 year)
        self.vrp_history = deque(maxlen=self.quantile_window)  # Efficient rolling window

        logger.info("VRP Classifier initialized with adaptive quantile-based boundaries")

    def calculate_vrp(self, implied_vol, realized_vol):
        """
        Calculate VRP ratio from implied and realized volatility.
        
        VRP = (Implied Volatility / 100) / Realized Volatility
        
        Args:
            implied_vol: Implied volatility (VIX value)
            realized_vol: Realized volatility (annualized)
            
        Returns:
            VRP ratio
        """
        from ..utils.exceptions import CalculationError
        
        # Validate VIX input
        if implied_vol <= 0:
            raise CalculationError(
                calculation="calculate_vrp",
                message="VIX must be positive"
            )
        
        # Validate realized volatility input
        if realized_vol <= 0:
            raise CalculationError(
                calculation="calculate_vrp", 
                message="Realized volatility must be positive"
            )
        
        # Convert VIX to decimal form (VIX is in percentage)
        implied_decimal = implied_vol / Decimal('100.0')
        
        vrp_ratio = implied_decimal / realized_vol
        
        # Log unusual VRP values (consistent with volatility calculator)
        vrp_float = float(vrp_ratio)
        if vrp_float < 0.1 or vrp_float > 10.0:
            logger.warning(
                f"Unusual VRP value: {vrp_float:.3f} (IV: {float(implied_decimal):.3f}, RV: {float(realized_vol):.3f})"
            )
            
        return vrp_ratio

    def classify_vrp_state(self, vrp_ratio: float) -> VRPState:
        """
        Classify VRP ratio using adaptive quantile-based boundaries.

        State boundaries are dynamically calculated from historical VRP distribution:
        - EXTREME_LOW: VRP ≤ 10th percentile - Extremely undervalued
        - FAIR_VALUE: 10th < VRP ≤ 30th percentile - Undervalued
        - NORMAL_PREMIUM: 30th < VRP ≤ 70th percentile - Normal range
        - ELEVATED_PREMIUM: 70th < VRP ≤ 90th percentile - Overvalued
        - EXTREME_HIGH: VRP > 90th percentile - Extremely overvalued

        Args:
            vrp_ratio: Volatility risk premium ratio to classify

        Returns:
            VRP state classification based on historical distribution

        Raises:
            ValidationError: If VRP ratio is invalid
        """
        try:
            # Validate input
            if vrp_ratio <= 0:
                raise ValidationError(
                    field="vrp_ratio",
                    value=str(vrp_ratio),
                    message="VRP ratio must be positive"
                )

            # Add to rolling history
            self.vrp_history.append(vrp_ratio)

            # Use adaptive quantile-based classification if we have sufficient history
            if len(self.vrp_history) >= 30:
                return self._classify_by_quantiles(vrp_ratio)
            else:
                # Fallback classification for insufficient history
                return self._classify_by_fallback(vrp_ratio)

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(
                field="vrp_ratio",
                value=str(vrp_ratio),
                message=f"Failed to classify VRP ratio: {str(e)}"
            )

    def _classify_by_quantiles(self, vrp_ratio: float) -> VRPState:
        """
        Classify VRP using quantiles from historical distribution.

        Args:
            vrp_ratio: VRP ratio to classify

        Returns:
            VRP state based on quantile boundaries
        """
        # Calculate quantile boundaries from historical data
        # Convert Decimals to floats for numpy compatibility
        history_floats = [float(x) for x in self.vrp_history]
        quantiles = np.percentile(history_floats, self.quantile_percentiles)
        
        vrp_ratio_float = float(vrp_ratio)

        if vrp_ratio_float <= quantiles[0]:  # ≤ 10th percentile
            return VRPState.EXTREME_LOW
        elif vrp_ratio_float <= quantiles[1]:  # ≤ 30th percentile
            return VRPState.FAIR_VALUE
        elif vrp_ratio_float <= quantiles[2]:  # ≤ 70th percentile
            return VRPState.NORMAL_PREMIUM
        elif vrp_ratio_float <= quantiles[3]:  # ≤ 90th percentile
            return VRPState.ELEVATED_PREMIUM
        else:  # > 90th percentile
            return VRPState.EXTREME_HIGH

    def _classify_by_fallback(self, vrp_ratio: float) -> VRPState:
        """
        Fallback classification for insufficient historical data.

        Uses the documented thresholds from VRPState enum comments as fallback
        until sufficient historical data is available for quantile-based classification.

        Args:
            vrp_ratio: VRP ratio to classify

        Returns:
            VRP state using documented threshold boundaries
        """
        # Use documented thresholds from VRPState enum
        # Convert to Decimal for precise comparison
        vrp_decimal = Decimal(str(vrp_ratio))
        
        if vrp_decimal < Decimal('0.90'):  # VRP < 0.90 - IV significantly underpriced
            return VRPState.EXTREME_LOW
        elif vrp_decimal < Decimal('1.10'):  # 0.90 ≤ VRP < 1.10 - Fair value region  
            return VRPState.FAIR_VALUE
        elif vrp_decimal < Decimal('1.30'):  # 1.10 ≤ VRP < 1.30 - Normal premium
            return VRPState.NORMAL_PREMIUM
        elif vrp_decimal < Decimal('1.50'):  # 1.30 ≤ VRP < 1.50 - Elevated premium
            return VRPState.ELEVATED_PREMIUM
        else:  # VRP ≥ 1.50 - Extreme premium territory
            return VRPState.EXTREME_HIGH

    def get_current_boundaries(self) -> Optional[List[float]]:
        """
        Get current quantile boundaries used for classification.

        Returns:
            List of current quantile boundaries or None if insufficient data
        """
        if len(self.vrp_history) >= 30:
            return list(np.percentile(list(self.vrp_history), self.quantile_percentiles))
        return None

    def get_state_distribution(self) -> dict:
        """
        Get distribution of states in current history window.

        Returns:
            Dictionary mapping states to their frequency in current window
        """
        if len(self.vrp_history) < 30:
            return {}

        state_counts = {state: 0 for state in VRPState}

        for vrp_value in self.vrp_history:
            state = self._classify_by_quantiles(vrp_value)
            state_counts[state] += 1

        return state_counts

    def reset_history(self):
        """Reset VRP history (useful for backtesting)."""
        self.vrp_history.clear()
        logger.debug("VRP history reset")

    def get_state_boundaries(self) -> List[float]:
        """
        Get the current quantile boundaries for state classification.

        Returns:
            List of quantile boundary values, or empty list if insufficient data
        """
        boundaries = self.get_current_boundaries()
        return boundaries if boundaries is not None else []

    def validate_state_transition(self, from_state: VRPState, to_state: VRPState) -> bool:
        """
        Validate if a state transition is reasonable.

        With adaptive quantile-based classification, all transitions are valid
        since boundaries adapt to market conditions.

        Args:
            from_state: Starting state
            to_state: Ending state

        Returns:
            True (all transitions are valid in adaptive system)
        """
        return True

    def calculate_realized_volatility(self, market_data, window_days=30):
        """
        Delegate to volatility calculator for realized volatility calculation.
        
        Args:
            market_data: List of market data points
            window_days: Rolling window size in days
            
        Returns:
            Volatility calculation results
        """
        return self.volatility_calculator.calculate_realized_volatility(market_data, window_days)

    def calculate_daily_returns(self, market_data):
        """
        Calculate daily returns for market data.
        
        Args:
            market_data: List of market data points
            
        Returns:
            List of daily returns
        """
        return self.volatility_calculator._calculate_daily_returns(market_data)

    def annualized_volatility_calculation(self, returns):
        """
        Calculate annualized volatility from daily returns.
        
        Args:
            returns: List of daily returns
            
        Returns:
            Annualized volatility
        """
        return self.volatility_calculator._calculate_window_volatility(returns)
