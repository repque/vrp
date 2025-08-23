"""
Markov Chain model for VRP state transitions.

This module implements the core Markov Chain model that learns transition
probabilities between VRP states and generates predictions with confidence
scoring. It handles rolling window updates, Laplace smoothing, and entropy
calculations for model health monitoring.
"""

import logging
import math
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple

import numpy as np

from src.config.settings import VRPTradingConfig
from src.interfaces.contracts import MarkovChainInterface
from src.models.data_models import (
    ModelPrediction,
    TransitionMatrix,
    VRPState,
    VolatilityMetrics,
)
from src.utils.exceptions import InsufficientDataError, ModelError


logger = logging.getLogger(__name__)


class VRPMarkovChain(MarkovChainInterface):
    """
    Markov Chain model for VRP state transitions.
    
    This class implements a sophisticated Markov Chain that learns transition
    probabilities between VRP states using rolling windows, applies Laplace
    smoothing for robustness, and provides confidence scoring based on
    entropy and data quality metrics.
    """
    
    def __init__(self, config: VRPTradingConfig):
        """
        Initialize Markov Chain model with configuration.
        
        Args:
            config: System configuration containing model parameters
        """
        self.config = config
        self.num_states = 5  # Number of VRP states
        
        # State mapping for array indexing
        self.state_to_index = {
            VRPState.UNDERPRICED: 0,
            VRPState.FAIR_VALUE: 1,
            VRPState.NORMAL_PREMIUM: 2,
            VRPState.ELEVATED_PREMIUM: 3,
            VRPState.EXTREME_PREMIUM: 4
        }
        
        self.index_to_state = {v: k for k, v in self.state_to_index.items()}
        
        # Cache for efficiency
        self._transition_cache = {}
        
        logger.info("Initialized VRPMarkovChain model")
    
    def update_transition_matrix(
        self, 
        state_sequence: List[VolatilityMetrics],
        window_days: int = 60
    ) -> TransitionMatrix:
        """
        Update transition matrix with new state observations.
        
        Uses a rolling window approach to maintain model adaptivity while
        ensuring sufficient observations. Applies Laplace smoothing to
        handle sparse transitions.
        
        Args:
            state_sequence: List of volatility metrics with VRP states
            window_days: Rolling window size in days
            
        Returns:
            Updated TransitionMatrix with probabilities and metadata
            
        Raises:
            InsufficientDataError: If not enough data for reliable estimation
            ModelError: If matrix calculation fails
        """
        if len(state_sequence) < window_days:
            raise InsufficientDataError(
                f"Need at least {window_days} observations, got {len(state_sequence)}"
            )
        
        logger.info(f"Updating transition matrix with {len(state_sequence)} observations")
        
        try:
            # Get the most recent window_days observations
            recent_observations = state_sequence[-window_days:]
            
            # Count state transitions
            transition_counts = self._count_transitions(recent_observations)
            
            # Apply Laplace smoothing
            smoothed_matrix = self.apply_laplace_smoothing(
                transition_counts, 
                alpha=float(self.config.model.laplace_smoothing_alpha)
            )
            
            # Convert to Decimal for precision
            decimal_matrix = [
                [Decimal(str(prob)) for prob in row] 
                for row in smoothed_matrix
            ]
            
            # Create TransitionMatrix object
            transition_matrix = TransitionMatrix(
                matrix=decimal_matrix,
                observation_count=len(recent_observations),
                window_start=recent_observations[0].date,
                window_end=recent_observations[-1].date,
                last_updated=datetime.now()
            )
            
            logger.info(f"Updated transition matrix with {len(recent_observations)} observations")
            return transition_matrix
        
        except Exception as e:
            logger.error(f"Error updating transition matrix: {str(e)}")
            raise ModelError(f"Transition matrix update failed: {str(e)}")
    
    def predict_next_state(
        self, 
        current_state: VRPState,
        transition_matrix: TransitionMatrix
    ) -> ModelPrediction:
        """
        Predict next state given current state and transition matrix.
        
        Uses the transition probabilities to determine the most likely next
        state and calculates confidence metrics based on probability
        distribution and matrix quality.
        
        Args:
            current_state: Current VRP state
            transition_matrix: Transition probability matrix
            
        Returns:
            ModelPrediction with predicted state and confidence metrics
        """
        try:
            current_index = self.state_to_index[current_state]
            
            # Get transition probabilities for current state
            transition_probs = [
                float(prob) for prob in transition_matrix.matrix[current_index]
            ]
            
            # Find most likely next state
            max_prob_index = np.argmax(transition_probs)
            predicted_state = self.index_to_state[max_prob_index]
            transition_probability = transition_probs[max_prob_index]
            
            # Calculate confidence score
            confidence_score = self.calculate_confidence_score(
                transition_matrix, 
                ModelPrediction(
                    current_date=date.today(),
                    current_state=current_state,
                    predicted_state=predicted_state,
                    transition_probability=Decimal(str(transition_probability)),
                    confidence_score=Decimal('0.0'),  # Will be calculated
                    entropy=Decimal('0.0'),  # Will be calculated
                    data_quality_score=Decimal('0.0')  # Will be calculated
                )
            )
            
            # Calculate entropy of prediction distribution
            entropy = self._calculate_entropy(transition_probs)
            
            # Calculate data quality score
            data_quality_score = self._calculate_data_quality_score(transition_matrix)
            
            prediction = ModelPrediction(
                current_date=date.today(),
                current_state=current_state,
                predicted_state=predicted_state,
                transition_probability=Decimal(str(transition_probability)),
                confidence_score=Decimal(str(confidence_score)),
                entropy=Decimal(str(entropy)),
                data_quality_score=Decimal(str(data_quality_score))
            )
            
            logger.info(
                f"Predicted {predicted_state.name} from {current_state.name} "
                f"(prob: {transition_probability:.3f}, confidence: {confidence_score:.3f})"
            )
            
            return prediction
        
        except Exception as e:
            logger.error(f"Error predicting next state: {str(e)}")
            raise ModelError(f"State prediction failed: {str(e)}")
    
    def calculate_confidence_score(
        self, 
        transition_matrix: TransitionMatrix,
        prediction: ModelPrediction
    ) -> float:
        """
        Calculate confidence score for prediction.
        
        Combines multiple factors:
        - Transition probability (higher is better)
        - Entropy of distribution (lower is better)
        - Data quality (more observations is better)
        - Matrix stability (consistent probabilities over time)
        
        Args:
            transition_matrix: Current transition matrix
            prediction: Model prediction to score
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Factor 1: Transition probability (0-1)
            prob_score = float(prediction.transition_probability)
            
            # Factor 2: Inverse entropy score (lower entropy = higher confidence)
            # Normalize entropy by log(num_states) for 0-1 scale
            max_entropy = math.log(self.num_states)
            entropy_score = 1.0 - (float(prediction.entropy) / max_entropy)
            entropy_score = max(0.0, min(1.0, entropy_score))  # Clamp to [0,1]
            
            # Factor 3: Data quality score
            data_score = self._calculate_data_quality_score(transition_matrix)
            
            # Factor 4: Matrix stability score
            stability_score = self._calculate_matrix_stability(transition_matrix)
            
            # Weighted combination
            weights = {
                'probability': 0.4,
                'entropy': 0.3,
                'data_quality': 0.2,
                'stability': 0.1
            }
            
            confidence = (
                weights['probability'] * prob_score +
                weights['entropy'] * entropy_score +
                weights['data_quality'] * data_score +
                weights['stability'] * stability_score
            )
            
            return max(0.0, min(1.0, confidence))  # Ensure [0,1] range
        
        except Exception as e:
            logger.warning(f"Error calculating confidence score: {str(e)}")
            return 0.0  # Conservative fallback
    
    def apply_laplace_smoothing(
        self, 
        raw_matrix: List[List[int]], 
        alpha: float = 0.01
    ) -> List[List[float]]:
        """
        Apply Laplace smoothing to transition counts.
        
        Adds a small constant (alpha) to all transition counts to handle
        zero probabilities and improve model robustness.
        
        Args:
            raw_matrix: Raw transition count matrix
            alpha: Smoothing parameter (small positive value)
            
        Returns:
            Smoothed probability matrix
        """
        if len(raw_matrix) != self.num_states or any(len(row) != self.num_states for row in raw_matrix):
            raise ModelError(f"Invalid matrix dimensions: expected {self.num_states}x{self.num_states}")
        
        smoothed_matrix = []
        
        for i in range(self.num_states):
            row_counts = raw_matrix[i]
            
            # Add smoothing constant
            smoothed_counts = [count + alpha for count in row_counts]
            
            # Normalize to probabilities
            row_sum = sum(smoothed_counts)
            if row_sum == 0:
                # Uniform distribution if no observations
                probabilities = [1.0 / self.num_states] * self.num_states
            else:
                probabilities = [count / row_sum for count in smoothed_counts]
            
            smoothed_matrix.append(probabilities)
        
        return smoothed_matrix
    
    def get_steady_state_distribution(self, transition_matrix: TransitionMatrix) -> List[float]:
        """
        Calculate steady-state distribution of the Markov chain.
        
        Finds the stationary distribution by solving the eigenvector equation
        or using iterative methods.
        
        Args:
            transition_matrix: Transition probability matrix
            
        Returns:
            Steady-state probability distribution
        """
        try:
            # Convert to numpy array
            matrix = np.array([
                [float(prob) for prob in row] 
                for row in transition_matrix.matrix
            ])
            
            # Find steady state using eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
            
            # Find eigenvector corresponding to eigenvalue 1
            steady_state_index = np.argmin(np.abs(eigenvalues - 1.0))
            steady_state = np.real(eigenvectors[:, steady_state_index])
            
            # Normalize to sum to 1 and ensure non-negative
            steady_state = np.abs(steady_state)
            steady_state = steady_state / np.sum(steady_state)
            
            return steady_state.tolist()
        
        except Exception as e:
            logger.warning(f"Error calculating steady state: {str(e)}")
            # Fallback to uniform distribution
            return [1.0 / self.num_states] * self.num_states
    
    def simulate_future_states(
        self, 
        current_state: VRPState,
        transition_matrix: TransitionMatrix,
        num_steps: int,
        num_simulations: int = 1000
    ) -> Dict[VRPState, float]:
        """
        Simulate future state probabilities using Monte Carlo.
        
        Args:
            current_state: Starting state
            transition_matrix: Transition probabilities
            num_steps: Number of steps to simulate
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary of state probabilities after num_steps
        """
        # Convert matrix to numpy for efficiency
        matrix = np.array([
            [float(prob) for prob in row] 
            for row in transition_matrix.matrix
        ])
        
        current_index = self.state_to_index[current_state]
        final_states = []
        
        for _ in range(num_simulations):
            state = current_index
            
            for _ in range(num_steps):
                # Sample next state based on transition probabilities
                state = np.random.choice(
                    self.num_states, 
                    p=matrix[state]
                )
            
            final_states.append(state)
        
        # Calculate probabilities
        state_probs = {}
        for state_idx in range(self.num_states):
            count = sum(1 for s in final_states if s == state_idx)
            probability = count / num_simulations
            state_probs[self.index_to_state[state_idx]] = probability
        
        return state_probs
    
    def _count_transitions(self, observations: List[VolatilityMetrics]) -> List[List[int]]:
        """
        Count state transitions from observation sequence.
        
        Args:
            observations: Sequence of volatility metrics with states
            
        Returns:
            Matrix of transition counts
        """
        # Initialize count matrix
        counts = [[0 for _ in range(self.num_states)] for _ in range(self.num_states)]
        
        # Count transitions
        for i in range(len(observations) - 1):
            from_state = observations[i].vrp_state
            to_state = observations[i + 1].vrp_state
            
            from_index = self.state_to_index[from_state]
            to_index = self.state_to_index[to_state]
            
            counts[from_index][to_index] += 1
        
        return counts
    
    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """
        Calculate Shannon entropy of probability distribution.
        
        Args:
            probabilities: Probability distribution
            
        Returns:
            Entropy value
        """
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log(p)
        
        return entropy
    
    def _calculate_data_quality_score(self, transition_matrix: TransitionMatrix) -> float:
        """
        Calculate data quality score based on observation count and recency.
        
        Args:
            transition_matrix: Transition matrix with metadata
            
        Returns:
            Quality score between 0 and 1
        """
        # Factor 1: Number of observations
        min_observations = self.config.model.rolling_window_days
        obs_score = min(1.0, transition_matrix.observation_count / (min_observations * 2))
        
        # Factor 2: Data recency
        days_old = (date.today() - transition_matrix.window_end).days
        max_age_days = 7  # Data older than a week gets penalized
        recency_score = max(0.0, 1.0 - (days_old / max_age_days))
        
        # Combine scores
        quality_score = 0.7 * obs_score + 0.3 * recency_score
        
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_matrix_stability(self, transition_matrix: TransitionMatrix) -> float:
        """
        Calculate matrix stability score based on diagonal dominance.
        
        A more stable matrix has higher probabilities on the diagonal
        (states tend to persist).
        
        Args:
            transition_matrix: Transition matrix
            
        Returns:
            Stability score between 0 and 1
        """
        try:
            diagonal_sum = 0.0
            total_sum = 0.0
            
            for i in range(self.num_states):
                for j in range(self.num_states):
                    prob = float(transition_matrix.matrix[i][j])
                    total_sum += prob
                    if i == j:
                        diagonal_sum += prob
            
            if total_sum > 0:
                return diagonal_sum / total_sum
            else:
                return 0.0
        
        except Exception:
            return 0.0