"""
Predictive Signal Generation Service

Uses Markov chain state transition predictions to generate trading signals
based on forecasted VRP state probabilities rather than simple thresholds.
"""

import logging
from typing import Dict, List, Optional, Tuple
from decimal import Decimal

from src.models.data_models import VolatilityData, VRPState, TradingSignal, ConfidenceMetrics, TransitionMatrix
from src.models.constants import DefaultConfiguration
from src.services.markov_chain_model import MarkovChainModel
from src.utils.exceptions import CalculationError, InsufficientDataError, ModelStateError

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Predictive signal generation using Markov chain state transitions.
    
    Generates trading signals based on predicted future VRP states rather
    than reactive threshold-based rules. Uses confidence scoring to size positions.
    """
    
    def __init__(self, config: Optional[DefaultConfiguration] = None):
        """
        Initialize predictive signal generator.
        
        Args:
            config: System configuration
        """
        self.config = config or DefaultConfiguration()
        self.markov_model = MarkovChainModel()
        
        logger.info("SignalGenerator initialized with Markov chain prediction")
    
    def generate_signal(
        self, 
        volatility_data: List[VolatilityData],
        window_days: int = 60
    ) -> TradingSignal:
        """
        Generate predictive trading signal using Markov chain state transitions.
        
        Args:
            volatility_data: Historical volatility data with states
            window_days: Rolling window for transition matrix
            
        Returns:
            TradingSignal with predictions and confidence metrics
            
        Raises:
            InsufficientDataError: If insufficient data for prediction
            ModelStateError: If model prediction fails
        """
        if not volatility_data or len(volatility_data) < window_days:
            raise InsufficientDataError(
                required=window_days,
                available=len(volatility_data) if volatility_data else 0
            )
        
        try:
            # Get current state from latest data point
            current_state = volatility_data[-1].vrp_state
            current_vrp = float(volatility_data[-1].vrp)
            
            # Update transition matrix with recent data
            transition_matrix = self.markov_model.update_transition_matrix(
                volatility_data, window_days
            )
            
            # Predict next state probabilities
            state_probabilities = self.markov_model.predict_next_state(
                current_state, transition_matrix
            )
            
            # Generate signal based on state predictions
            signal_type, signal_strength, reason = self._generate_predictive_signal(
                current_state, state_probabilities, current_vrp
            )
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(
                state_probabilities, transition_matrix, len(volatility_data)
            )
            
            # Determine predicted state (highest probability)
            predicted_state = max(state_probabilities.items(), key=lambda x: x[1])[0]
            
            # Calculate position sizes
            recommended_size = Decimal(str(float(self.config.BASE_POSITION_SIZE)))
            risk_adjusted_size = self._calculate_position_size(
                signal_strength, float(confidence_metrics.overall_confidence)
            )
            
            # Create trading signal
            signal = TradingSignal(
                date=volatility_data[-1].date,
                signal_type=signal_type,
                current_state=current_state,
                predicted_state=predicted_state,
                signal_strength=Decimal(str(signal_strength)),
                confidence_score=confidence_metrics.overall_confidence,
                recommended_position_size=recommended_size,
                risk_adjusted_size=risk_adjusted_size,
                reason=reason
            )
            
            logger.info(f"Generated predictive signal: {signal_type} | "
                       f"{current_state.name} -> {predicted_state.name} | "
                       f"Confidence: {confidence_metrics.overall_confidence:.3f}")
            
            return signal
            
        except Exception as e:
            if isinstance(e, (InsufficientDataError, ModelStateError)):
                raise
            raise ModelStateError(f"Failed to generate predictive signal: {str(e)}")
    
    def _generate_predictive_signal(
        self, 
        current_state: VRPState,
        state_probabilities: Dict[VRPState, float],
        current_vrp: float
    ) -> Tuple[str, float, str]:
        """
        Generate signal based on state transition predictions.
        
        Args:
            current_state: Current VRP state
            state_probabilities: Predicted next state probabilities
            current_vrp: Current VRP value
            
        Returns:
            Tuple of (signal_type, signal_strength, reason)
        """
        # Calculate expected VRP movement based on state probabilities
        low_states_prob = (
            state_probabilities.get(VRPState.EXTREME_LOW, 0) + 
            state_probabilities.get(VRPState.FAIR_VALUE, 0)
        )
        
        high_states_prob = (
            state_probabilities.get(VRPState.ELEVATED_PREMIUM, 0) + 
            state_probabilities.get(VRPState.EXTREME_HIGH, 0)
        )
        
        normal_prob = state_probabilities.get(VRPState.NORMAL_PREMIUM, 0)
        
        # Signal logic based on predicted state transitions
        # CORRECTED LOGIC: When VRP will be LOW (undervalued), SELL volatility for mean reversion
        # When VRP will be HIGH (overvalued), BUY volatility for mean reversion
        if low_states_prob > 0.6:  # High probability of moving to undervalued states
            signal_type = "SELL_VOL"  # FIXED: Sell when VRP will be low (undervalued)
            signal_strength = low_states_prob
            reason = f"Model predicts {low_states_prob:.1%} probability of undervalued VRP states - sell for mean reversion"
            
        elif high_states_prob > 0.6:  # High probability of moving to overvalued states
            signal_type = "BUY_VOL"  # FIXED: Buy when VRP will be high (overvalued)
            signal_strength = high_states_prob
            reason = f"Model predicts {high_states_prob:.1%} probability of overvalued VRP states - buy for mean reversion"
            
        elif current_state == VRPState.EXTREME_LOW and low_states_prob < 0.3:
            # Currently extremely low but model predicts mean reversion UP
            signal_type = "BUY_VOL"  # FIXED: Buy when VRP is extremely low (will mean revert higher)
            signal_strength = 1.0 - low_states_prob
            reason = f"Mean reversion UP from extreme low VRP expected ({low_states_prob:.1%} persistence)"
            
        elif current_state == VRPState.EXTREME_HIGH and high_states_prob < 0.3:
            # Currently extremely high but model predicts mean reversion DOWN
            signal_type = "SELL_VOL"  # FIXED: Sell when VRP is extremely high (will mean revert lower)
            signal_strength = 1.0 - high_states_prob
            reason = f"Mean reversion DOWN from extreme high VRP expected ({high_states_prob:.1%} persistence)"
            
        else:
            # No clear directional signal
            signal_type = "HOLD"
            signal_strength = normal_prob + max(low_states_prob, high_states_prob)
            reason = f"No clear directional signal (balanced probabilities: L={low_states_prob:.1%}, H={high_states_prob:.1%})"
        
        return signal_type, signal_strength, reason
    
    def _calculate_confidence_metrics(
        self, 
        state_probabilities: Dict[VRPState, float],
        transition_matrix: TransitionMatrix,
        data_length: int
    ) -> ConfidenceMetrics:
        """
        Calculate confidence metrics for the prediction.
        
        Args:
            state_probabilities: Predicted state probabilities
            transition_matrix: Current transition matrix
            data_length: Length of data used
            
        Returns:
            ConfidenceMetrics with scoring breakdown
        """
        # Entropy-based confidence (lower entropy = higher confidence)
        import math
        probabilities = list(state_probabilities.values())
        entropy = -sum(p * math.log(p + 1e-10) for p in probabilities if p > 0)
        max_entropy = math.log(len(VRPState))  # Maximum possible entropy
        entropy_score = 1.0 - (entropy / max_entropy)  # Invert so higher = more confident
        
        # Data quality score based on observation count
        min_observations = 60
        data_quality_score = min(1.0, data_length / min_observations)
        
        # Model stability (simplified - could compare with previous matrix)
        stability_score = 0.8  # Default assumption of reasonable stability
        
        # Overall confidence (weighted combination)
        overall_confidence = (
            0.4 * entropy_score + 
            0.4 * data_quality_score + 
            0.2 * stability_score
        )
        
        return ConfidenceMetrics(
            entropy_score=Decimal(str(entropy_score)),
            data_quality_score=Decimal(str(data_quality_score)),
            stability_score=Decimal(str(stability_score)),
            overall_confidence=Decimal(str(overall_confidence))
        )
    
    def _calculate_position_size(self, signal_strength: float, confidence: float) -> Decimal:
        """
        Calculate position size based on signal strength and confidence.
        
        Args:
            signal_strength: Signal strength (0-1)
            confidence: Model confidence (0-1)
            
        Returns:
            Position size as decimal percentage
        """
        # Base position size from configuration
        base_size = float(self.config.BASE_POSITION_SIZE)
        
        # Adjust based on signal strength and confidence
        adjusted_size = base_size * signal_strength * confidence
        
        # Cap at maximum position size
        max_size = float(self.config.MAX_POSITION_SIZE)
        final_size = min(adjusted_size, max_size)
        
        return Decimal(str(final_size))
    
    def validate_signal_requirements(self, volatility_data: List[VolatilityData]) -> bool:
        """
        Validate that we have sufficient data for signal generation.
        
        Args:
            volatility_data: Historical volatility data
            
        Returns:
            True if requirements are met
        """
        if not volatility_data:
            logger.warning("No volatility data provided")
            return False
        
        if len(volatility_data) < 60:
            logger.warning(f"Insufficient data for reliable predictions: {len(volatility_data)} < 60")
            return False
        
        # Check for recent data
        from datetime import datetime, timedelta
        if volatility_data[-1].date < (datetime.now().date() - timedelta(days=7)):
            logger.warning("Data appears stale (older than 1 week)")
            return False
        
        return True