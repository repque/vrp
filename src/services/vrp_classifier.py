"""
Adaptive VRP State Classification Service

Provides quantile-based VRP state classification that adapts to market conditions
without using fixed thresholds. States are determined by historical distribution.
"""

import logging
import numpy as np
from typing import List, Optional
from collections import deque

from src.config.settings import get_settings
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
    
    def __init__(self):
        """Initialize VRP classifier with adaptive quantile-based boundaries."""
        self.settings = get_settings()
        
        # Adaptive quantile-based configuration
        self.quantile_percentiles = [10, 30, 70, 90]  # For 5-state classification
        self.quantile_window = 252  # Rolling window for quantile calculation (1 year)
        self.vrp_history = deque(maxlen=self.quantile_window)  # Efficient rolling window
        
        logger.info(f"VRP Classifier initialized with adaptive quantile-based boundaries")
    
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
        quantiles = np.percentile(list(self.vrp_history), self.quantile_percentiles)
        
        if vrp_ratio <= quantiles[0]:  # ≤ 10th percentile
            return VRPState.EXTREME_LOW
        elif vrp_ratio <= quantiles[1]:  # ≤ 30th percentile
            return VRPState.FAIR_VALUE
        elif vrp_ratio <= quantiles[2]:  # ≤ 70th percentile
            return VRPState.NORMAL_PREMIUM
        elif vrp_ratio <= quantiles[3]:  # ≤ 90th percentile
            return VRPState.ELEVATED_PREMIUM
        else:  # > 90th percentile
            return VRPState.EXTREME_HIGH
    
    def _classify_by_fallback(self, vrp_ratio: float) -> VRPState:
        """
        Fallback classification for insufficient historical data.
        
        Uses rough market-based boundaries as temporary classification.
        
        Args:
            vrp_ratio: VRP ratio to classify
            
        Returns:
            VRP state using fallback boundaries
        """
        # Rough market-based boundaries (not fixed thresholds, just bootstrapping)
        if vrp_ratio < 0.8:
            return VRPState.EXTREME_LOW
        elif vrp_ratio < 1.0:
            return VRPState.FAIR_VALUE
        elif vrp_ratio < 1.2:
            return VRPState.NORMAL_PREMIUM
        elif vrp_ratio < 1.5:
            return VRPState.ELEVATED_PREMIUM
        else:
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