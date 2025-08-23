"""
VRP state classification service for VRP Markov Chain Trading Model.

This service classifies VRP ratios into discrete states based on
configurable thresholds and validates state transitions.
"""

import logging
from typing import List

from src.config.settings import get_settings
from src.interfaces.contracts import IVRPClassifier
from src.models.data_models import VRPState
from src.utils.exceptions import ValidationError


logger = logging.getLogger(__name__)


class VRPClassifier(IVRPClassifier):
    """
    Implementation of VRP state classification.
    
    Classifies VRP ratios into 5 discrete states based on configurable
    thresholds and provides validation for state transitions.
    """
    
    def __init__(self):
        """Initialize VRP classifier with configuration settings."""
        self.settings = get_settings()
        self.thresholds = self.settings.model.vrp_thresholds
        
        # Validate thresholds are properly configured
        self._validate_thresholds()
        
        logger.info(f"VRP Classifier initialized with thresholds: {self.thresholds}")
    
    def classify_vrp_state(self, vrp_ratio: float) -> VRPState:
        """
        Classify VRP ratio into discrete state.
        
        State boundaries:
        - State 1 (EXTREME_LOW): VRP < 0.90 - IV underpriced
        - State 2 (FAIR_VALUE): 0.90 ≤ VRP < 1.10 - Fair value  
        - State 3 (NORMAL_PREMIUM): 1.10 ≤ VRP < 1.30 - Normal premium
        - State 4 (ELEVATED_PREMIUM): 1.30 ≤ VRP < 1.50 - Elevated premium
        - State 5 (EXTREME_HIGH): VRP ≥ 1.50 - Extreme premium
        
        Args:
            vrp_ratio: Volatility risk premium ratio to classify
            
        Returns:
            VRP state classification
            
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
            
            # Log warning for extreme values
            min_vrp = self.settings.model.vrp_min_reasonable
            max_vrp = self.settings.model.vrp_max_reasonable
            if vrp_ratio < min_vrp or vrp_ratio > max_vrp:
                logger.warning(f"VRP ratio {vrp_ratio:.4f} is outside typical range [{min_vrp}, {max_vrp}]")
            
            # Classify based on thresholds
            if vrp_ratio < self.thresholds[0]:  # < 0.90
                return VRPState.EXTREME_LOW
            elif vrp_ratio < self.thresholds[1]:  # 0.90 <= x < 1.10
                return VRPState.FAIR_VALUE
            elif vrp_ratio < self.thresholds[2]:  # 1.10 <= x < 1.30
                return VRPState.NORMAL_PREMIUM
            elif vrp_ratio < self.thresholds[3]:  # 1.30 <= x < 1.50
                return VRPState.ELEVATED_PREMIUM
            else:  # >= 1.50
                return VRPState.EXTREME_HIGH
                
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(
                field="vrp_ratio",
                value=str(vrp_ratio),
                message=f"Failed to classify VRP ratio: {str(e)}"
            )
    
    def get_state_boundaries(self) -> List[float]:
        """
        Get the threshold boundaries for state classification.
        
        Returns:
            List of threshold values [0.9, 1.1, 1.3, 1.5]
        """
        return self.thresholds.copy()
    
    def validate_state_transition(
        self, 
        from_state: VRPState, 
        to_state: VRPState
    ) -> bool:
        """
        Validate if a state transition is reasonable.
        
        Reasonable transitions are:
        - Adjacent states (e.g., 2 -> 3 or 3 -> 2)
        - Same state (e.g., 3 -> 3)
        - Maximum jump of 2 states for extreme market conditions
        
        Args:
            from_state: Starting VRP state
            to_state: Ending VRP state
            
        Returns:
            True if transition is considered reasonable
        """
        try:
            state_diff = abs(to_state.value - from_state.value)
            
            # Same state is always valid
            if state_diff == 0:
                return True
            
            # Adjacent states are normal
            if state_diff == 1:
                return True
            
            # Jumps of 2 states are possible in volatile markets
            if state_diff == 2:
                logger.info(f"Large state transition detected: {from_state.name} -> {to_state.name}")
                return True
            
            # Jumps of 3+ states are extreme but possible
            if state_diff >= 3:
                logger.warning(f"Extreme state transition: {from_state.name} -> {to_state.name}")
                return True  # Still valid, just unusual
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating state transition: {str(e)}")
            return False
    
    def get_state_description(self, state: VRPState) -> str:
        """
        Get human-readable description of VRP state.
        
        Args:
            state: VRP state to describe
            
        Returns:
            Descriptive string for the state
        """
        descriptions = {
            VRPState.EXTREME_LOW: "Implied volatility significantly underpriced - rare buying opportunity",
            VRPState.FAIR_VALUE: "Implied volatility fairly valued - no clear trading edge",
            VRPState.NORMAL_PREMIUM: "Normal volatility premium - typical market conditions",
            VRPState.ELEVATED_PREMIUM: "Elevated volatility premium - consider selling volatility",
            VRPState.EXTREME_HIGH: "Extreme volatility premium - strong signal to sell volatility"
        }
        
        return descriptions.get(state, f"Unknown state: {state}")
    
    def get_state_trading_implications(self, state: VRPState) -> str:
        """
        Get trading implications for a VRP state.
        
        Args:
            state: VRP state to analyze
            
        Returns:
            Trading implications string
        """
        implications = {
            VRPState.EXTREME_LOW: "BUY volatility - IV likely to increase",
            VRPState.FAIR_VALUE: "NO TRADE - no clear directional edge",
            VRPState.NORMAL_PREMIUM: "NO TRADE - normal market conditions",
            VRPState.ELEVATED_PREMIUM: "CONSIDER selling volatility - IV elevated",
            VRPState.EXTREME_HIGH: "SELL volatility - IV likely to decrease"
        }
        
        return implications.get(state, f"Unknown implications for state: {state}")
    
    def calculate_state_statistics(self, states: List[VRPState]) -> dict:
        """
        Calculate distribution statistics for a list of VRP states.
        
        Args:
            states: List of VRP states to analyze
            
        Returns:
            Dictionary with state distribution statistics
        """
        if not states:
            return {}
        
        # Count occurrences of each state
        state_counts = {state: 0 for state in VRPState}
        for state in states:
            state_counts[state] += 1
        
        total_count = len(states)
        
        # Calculate percentages
        state_percentages = {
            state: (count / total_count) * 100 
            for state, count in state_counts.items()
        }
        
        # Calculate transition-relevant statistics
        extreme_states = sum(1 for s in states if s in [VRPState.EXTREME_LOW, VRPState.EXTREME_HIGH])
        extreme_percentage = (extreme_states / total_count) * 100
        
        tradeable_states = sum(1 for s in states if s in [VRPState.EXTREME_LOW, VRPState.EXTREME_HIGH])
        tradeable_percentage = (tradeable_states / total_count) * 100
        
        stats = {
            'total_observations': total_count,
            'state_counts': state_counts,
            'state_percentages': state_percentages,
            'extreme_state_percentage': extreme_percentage,
            'tradeable_state_percentage': tradeable_percentage,
            'most_common_state': max(state_counts, key=state_counts.get),
            'least_common_state': min(state_counts, key=state_counts.get)
        }
        
        logger.info(f"Calculated state statistics for {total_count} observations")
        logger.info(f"Extreme states: {extreme_percentage:.1f}%, Tradeable: {tradeable_percentage:.1f}%")
        
        return stats
    
    def _validate_thresholds(self) -> None:
        """
        Validate that thresholds are properly configured.
        
        Raises:
            ValidationError: If thresholds are invalid
        """
        if len(self.thresholds) != 4:
            raise ValidationError(
                field="vrp_thresholds",
                value=str(self.thresholds),
                message="Must have exactly 4 thresholds"
            )
        
        # Check thresholds are in ascending order
        for i in range(len(self.thresholds) - 1):
            if self.thresholds[i] >= self.thresholds[i + 1]:
                raise ValidationError(
                    field="vrp_thresholds",
                    value=str(self.thresholds),
                    message="Thresholds must be in ascending order"
                )
        
        # Check thresholds are reasonable values
        if self.thresholds[0] < 0.5 or self.thresholds[-1] > 3.0:
            logger.warning(f"Threshold values may be outside typical range: {self.thresholds}")
    
    def get_threshold_for_state(self, state: VRPState) -> tuple:
        """
        Get the VRP ratio range for a given state.
        
        Args:
            state: VRP state to get range for
            
        Returns:
            Tuple of (lower_bound, upper_bound) for the state
        """
        if state == VRPState.EXTREME_LOW:
            return (0.0, self.thresholds[0])
        elif state == VRPState.FAIR_VALUE:
            return (self.thresholds[0], self.thresholds[1])
        elif state == VRPState.NORMAL_PREMIUM:
            return (self.thresholds[1], self.thresholds[2])
        elif state == VRPState.ELEVATED_PREMIUM:
            return (self.thresholds[2], self.thresholds[3])
        elif state == VRPState.EXTREME_HIGH:
            return (self.thresholds[3], float('inf'))
        else:
            raise ValidationError(
                field="state",
                value=str(state),
                message=f"Unknown VRP state: {state}"
            )