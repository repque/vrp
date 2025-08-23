"""
Trading signal generation for VRP Trading System.

This module generates trading signals based on Markov Chain predictions,
validates signal conditions, and calculates signal strength with proper
risk considerations. It focuses on extreme VRP states where the model
has the highest edge.
"""

import logging
from datetime import date
from decimal import Decimal
from typing import Optional
from uuid import uuid4

from src.config.settings import VRPTradingConfig
from src.interfaces.contracts import SignalGeneratorInterface
from src.models.data_models import (
    ModelPrediction,
    TradingSignal,
    VRPState,
    VolatilityMetrics,
)
from src.utils.exceptions import SignalGenerationError


logger = logging.getLogger(__name__)


class VRPSignalGenerator(SignalGeneratorInterface):
    """
    Generates trading signals from VRP Markov Chain predictions.
    
    This class implements a conservative signal generation strategy that
    focuses on extreme VRP states where the model has demonstrated edge.
    It validates multiple conditions before generating signals and provides
    detailed reasoning for all decisions.
    """
    
    def __init__(self, config: VRPTradingConfig):
        """
        Initialize signal generator with configuration.
        
        Args:
            config: System configuration containing signal parameters
        """
        self.config = config
        
        # Define extreme states for signal generation
        self.vol_buying_states = {VRPState.UNDERPRICED}
        self.vol_selling_states = {VRPState.EXTREME_PREMIUM}
        
        # Signal strength thresholds
        self.min_confidence = float(config.model.min_confidence_threshold)
        self.min_signal_strength = float(config.model.min_signal_strength)
        
        logger.info("Initialized VRPSignalGenerator")
    
    def generate_signal(
        self, 
        prediction: ModelPrediction,
        volatility_metrics: VolatilityMetrics
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal from model prediction.
        
        Only generates signals for extreme VRP states with sufficient
        confidence. Implements multiple validation layers to ensure
        signal quality.
        
        Args:
            prediction: Model prediction with confidence metrics
            volatility_metrics: Current volatility metrics
            
        Returns:
            TradingSignal if conditions are met, None otherwise
        """
        logger.info(
            f"Evaluating signal for state {prediction.current_state.name} -> "
            f"{prediction.predicted_state.name} (confidence: {prediction.confidence_score})"
        )
        
        try:
            # Validate signal conditions
            if not self.validate_signal_conditions(prediction):
                logger.info("Signal conditions not met")
                return None
            
            # Determine signal type
            signal_type = self._determine_signal_type(
                prediction.current_state, 
                prediction.predicted_state
            )
            
            if signal_type == "HOLD":
                logger.info("No tradeable signal identified")
                return None
            
            # Calculate signal strength
            signal_strength = self.calculate_signal_strength(prediction)
            
            if signal_strength < self.min_signal_strength:
                logger.info(f"Signal strength too low: {signal_strength:.3f}")
                return None
            
            # Calculate position sizing recommendations
            base_position_size = self._calculate_base_position_size(signal_strength)
            risk_adjusted_size = self._apply_risk_adjustments(
                base_position_size, 
                volatility_metrics,
                prediction
            )
            
            # Generate detailed reasoning
            reason = self._generate_signal_reasoning(
                prediction, 
                volatility_metrics, 
                signal_type,
                signal_strength
            )
            
            # Create trading signal
            signal = TradingSignal(
                date=prediction.current_date,
                signal_type=signal_type,
                current_state=prediction.current_state,
                predicted_state=prediction.predicted_state,
                signal_strength=Decimal(str(signal_strength)),
                confidence_score=prediction.confidence_score,
                recommended_position_size=Decimal(str(base_position_size)),
                risk_adjusted_size=Decimal(str(risk_adjusted_size)),
                reason=reason
            )
            
            logger.info(
                f"Generated {signal_type} signal with strength {signal_strength:.3f} "
                f"and size {risk_adjusted_size:.3f}"
            )
            
            return signal
        
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            raise SignalGenerationError(f"Signal generation failed: {str(e)}")
    
    def validate_signal_conditions(self, prediction: ModelPrediction) -> bool:
        """
        Validate if conditions are met for signal generation.
        
        Checks multiple criteria:
        - Confidence above threshold
        - Data quality acceptable
        - Extreme state involvement
        - Prediction consistency
        
        Args:
            prediction: Model prediction to validate
            
        Returns:
            True if signal conditions are met
        """
        # Check confidence threshold
        if float(prediction.confidence_score) < self.min_confidence:
            logger.debug(f"Confidence too low: {prediction.confidence_score} < {self.min_confidence}")
            return False
        
        # Check data quality
        if float(prediction.data_quality_score) < 0.5:
            logger.debug(f"Data quality too low: {prediction.data_quality_score}")
            return False
        
        # Check for extreme state involvement
        extreme_states = self.vol_buying_states | self.vol_selling_states
        if (prediction.current_state not in extreme_states and 
            prediction.predicted_state not in extreme_states):
            logger.debug("No extreme states involved")
            return False
        
        # Check transition probability
        if float(prediction.transition_probability) < 0.3:
            logger.debug(f"Transition probability too low: {prediction.transition_probability}")
            return False
        
        return True
    
    def calculate_signal_strength(self, prediction: ModelPrediction) -> float:
        """
        Calculate signal strength based on prediction confidence.
        
        Combines multiple factors:
        - Model confidence
        - Transition probability
        - Data quality
        - State extremeness
        
        Args:
            prediction: Model prediction
            
        Returns:
            Signal strength between 0 and 1
        """
        try:
            # Factor 1: Base confidence score
            confidence_score = float(prediction.confidence_score)
            
            # Factor 2: Transition probability (higher for more certain transitions)
            prob_score = float(prediction.transition_probability)
            
            # Factor 3: Data quality score
            data_score = float(prediction.data_quality_score)
            
            # Factor 4: State extremeness score
            extremeness_score = self._calculate_state_extremeness(
                prediction.current_state, 
                prediction.predicted_state
            )
            
            # Factor 5: Entropy adjustment (lower entropy = higher strength)
            entropy_adjustment = max(0.0, 1.0 - float(prediction.entropy) / 2.3)  # log(5) ≈ 1.6
            
            # Weighted combination
            weights = {
                'confidence': 0.35,
                'probability': 0.25,
                'data_quality': 0.15,
                'extremeness': 0.15,
                'entropy': 0.10
            }
            
            signal_strength = (
                weights['confidence'] * confidence_score +
                weights['probability'] * prob_score +
                weights['data_quality'] * data_score +
                weights['extremeness'] * extremeness_score +
                weights['entropy'] * entropy_adjustment
            )
            
            return max(0.0, min(1.0, signal_strength))
        
        except Exception as e:
            logger.warning(f"Error calculating signal strength: {str(e)}")
            return 0.0  # Conservative fallback
    
    def _determine_signal_type(
        self, 
        current_state: VRPState, 
        predicted_state: VRPState
    ) -> str:
        """
        Determine signal type based on current and predicted states.
        
        Args:
            current_state: Current VRP state
            predicted_state: Predicted next VRP state
            
        Returns:
            Signal type: "BUY_VOL", "SELL_VOL", or "HOLD"
        """
        # Buy volatility signals (IV underpriced)
        if (current_state in self.vol_buying_states or 
            predicted_state in self.vol_buying_states):
            return "BUY_VOL"
        
        # Sell volatility signals (IV overpriced)
        if (current_state in self.vol_selling_states or 
            predicted_state in self.vol_selling_states):
            return "SELL_VOL"
        
        # No signal for neutral states
        return "HOLD"
    
    def _calculate_base_position_size(self, signal_strength: float) -> float:
        """
        Calculate base position size based on signal strength.
        
        Args:
            signal_strength: Signal strength (0-1)
            
        Returns:
            Base position size as fraction of capital
        """
        max_size = float(self.config.risk.max_position_size)
        
        # Scale position size with signal strength
        # Use square root scaling for more conservative sizing
        base_size = max_size * (signal_strength ** 0.75)
        
        # Minimum threshold for position size
        min_size = max_size * 0.1
        
        return max(min_size, base_size)
    
    def _apply_risk_adjustments(
        self, 
        base_size: float, 
        volatility_metrics: VolatilityMetrics,
        prediction: ModelPrediction
    ) -> float:
        """
        Apply risk adjustments to position size.
        
        Args:
            base_size: Base position size
            volatility_metrics: Current volatility metrics
            prediction: Model prediction
            
        Returns:
            Risk-adjusted position size
        """
        adjusted_size = base_size
        
        # Adjustment 1: High volatility environments
        realized_vol = float(volatility_metrics.realized_vol_30d)
        if realized_vol > 0.3:  # High volatility (>30% annualized)
            adjusted_size *= 0.75
            logger.debug(f"High volatility adjustment: {realized_vol:.3f}")
        
        # Adjustment 2: Low confidence adjustment
        confidence = float(prediction.confidence_score)
        if confidence < 0.7:
            confidence_multiplier = confidence / 0.7
            adjusted_size *= confidence_multiplier
            logger.debug(f"Low confidence adjustment: {confidence:.3f}")
        
        # Adjustment 3: Extreme VRP values
        vrp = float(volatility_metrics.vrp)
        if vrp > 3.0 or vrp < 0.3:  # Very extreme values
            adjusted_size *= 1.2  # Increase size for extreme opportunities
            logger.debug(f"Extreme VRP adjustment: {vrp:.3f}")
        
        # Ensure we don't exceed maximum position size
        max_size = float(self.config.risk.max_position_size)
        adjusted_size = min(adjusted_size, max_size)
        
        return max(0.01, adjusted_size)  # Minimum 1% position
    
    def _calculate_state_extremeness(
        self, 
        current_state: VRPState, 
        predicted_state: VRPState
    ) -> float:
        """
        Calculate extremeness score based on states involved.
        
        Args:
            current_state: Current VRP state
            predicted_state: Predicted VRP state
            
        Returns:
            Extremeness score (0-1)
        """
        # Score each state by extremeness
        state_scores = {
            VRPState.UNDERPRICED: 1.0,      # Most extreme
            VRPState.EXTREME_PREMIUM: 1.0,  # Most extreme
            VRPState.ELEVATED_PREMIUM: 0.6,
            VRPState.NORMAL_PREMIUM: 0.3,
            VRPState.FAIR_VALUE: 0.1
        }
        
        current_score = state_scores.get(current_state, 0.0)
        predicted_score = state_scores.get(predicted_state, 0.0)
        
        # Return maximum extremeness between current and predicted
        return max(current_score, predicted_score)
    
    def _generate_signal_reasoning(
        self, 
        prediction: ModelPrediction,
        volatility_metrics: VolatilityMetrics,
        signal_type: str,
        signal_strength: float
    ) -> str:
        """
        Generate detailed reasoning for signal generation.
        
        Args:
            prediction: Model prediction
            volatility_metrics: Current volatility metrics
            signal_type: Type of signal generated
            signal_strength: Calculated signal strength
            
        Returns:
            Detailed reasoning string
        """
        reasons = []
        
        # State transition reasoning
        reasons.append(
            f"VRP state transition: {prediction.current_state.name} -> "
            f"{prediction.predicted_state.name} (prob: {prediction.transition_probability:.3f})"
        )
        
        # VRP value context
        vrp = float(volatility_metrics.vrp)
        rv = float(volatility_metrics.realized_vol_30d)
        iv = float(volatility_metrics.implied_vol)
        
        reasons.append(
            f"VRP context: {vrp:.3f} (IV: {iv:.1%}, RV: {rv:.1%})"
        )
        
        # Signal strength factors
        reasons.append(
            f"Signal strength: {signal_strength:.3f} "
            f"(confidence: {prediction.confidence_score}, "
            f"data quality: {prediction.data_quality_score})"
        )
        
        # Signal type specific reasoning
        if signal_type == "BUY_VOL":
            if prediction.current_state == VRPState.UNDERPRICED:
                reasons.append("Volatility appears underpriced - rare buying opportunity")
            elif prediction.predicted_state == VRPState.UNDERPRICED:
                reasons.append("Model predicts transition to underpriced state")
        
        elif signal_type == "SELL_VOL":
            if prediction.current_state == VRPState.EXTREME_PREMIUM:
                reasons.append("Volatility extremely overpriced - strong selling opportunity")
            elif prediction.predicted_state == VRPState.EXTREME_PREMIUM:
                reasons.append("Model predicts transition to extreme premium state")
        
        # Risk considerations
        if float(volatility_metrics.realized_vol_30d) > 0.3:
            reasons.append("High volatility environment - position size reduced")
        
        if float(prediction.confidence_score) < 0.8:
            reasons.append("Moderate confidence - position size adjusted")
        
        return "; ".join(reasons)
    
    def get_signal_performance_attribution(
        self, 
        signal: TradingSignal,
        actual_outcome: Optional[VolatilityMetrics] = None
    ) -> dict:
        """
        Analyze signal performance and attribute to factors.
        
        Args:
            signal: Original trading signal
            actual_outcome: Actual volatility metrics after signal
            
        Returns:
            Dictionary with performance attribution
        """
        attribution = {
            'signal_id': str(uuid4()),
            'signal_date': signal.date,
            'signal_type': signal.signal_type,
            'predicted_state': signal.predicted_state.name,
            'signal_strength': float(signal.signal_strength),
            'confidence_score': float(signal.confidence_score),
            'position_size': float(signal.risk_adjusted_size)
        }
        
        if actual_outcome:
            attribution.update({
                'actual_state': actual_outcome.vrp_state.name,
                'prediction_correct': signal.predicted_state == actual_outcome.vrp_state,
                'actual_vrp': float(actual_outcome.vrp),
                'vrp_change': float(actual_outcome.vrp) - float(
                    # Would need historical VRP for comparison
                    1.0  # Placeholder
                )
            })
        
        return attribution