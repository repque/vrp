"""
Markov chain model service for VRP state transitions.

This service manages the transition matrix, applies Laplace smoothing,
and provides state predictions based on historical VRP state sequences.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

from src.config.settings import get_settings
from src.interfaces.contracts import IMarkovChainModel
from src.models.data_models import TransitionMatrix, VRPState, VolatilityData
from src.utils.exceptions import CalculationError, InsufficientDataError, ModelStateError


logger = logging.getLogger(__name__)


class MarkovChainModel(IMarkovChainModel):
    """
    Implementation of Markov chain model for VRP state transitions.
    
    Manages transition matrix construction, Laplace smoothing application,
    and next-state probability predictions using rolling windows.
    """
    
    def __init__(self):
        """Initialize Markov chain model with configuration settings."""
        self.settings = get_settings()
        self.num_states = 5  # VRP has 5 states
        
        logger.info("Markov Chain Model initialized")
    
    def update_transition_matrix(
        self, 
        volatility_data: List[VolatilityData],
        window_days: int = 60
    ) -> TransitionMatrix:
        """
        Update transition matrix with new data using rolling window.
        
        Counts state transitions within the rolling window and applies
        Laplace smoothing to handle sparse data conditions.
        
        Args:
            volatility_data: Historical volatility data with states
            window_days: Rolling window size for transition counting
            
        Returns:
            Updated transition matrix with probabilities
            
        Raises:
            InsufficientDataError: If insufficient data for matrix construction
            CalculationError: If matrix construction fails
        """
        if len(volatility_data) < window_days:
            raise InsufficientDataError(
                required=window_days,
                available=len(volatility_data)
            )
        
        try:
            # Sort data by date to ensure chronological order
            sorted_data = sorted(volatility_data, key=lambda x: x.date)
            
            # Use the most recent window_days of data
            recent_data = sorted_data[-window_days:]
            
            # Count transitions
            transition_counts = self._count_transitions(recent_data)
            
            # Apply Laplace smoothing
            smoothed_matrix = self.apply_laplace_smoothing(
                transition_counts, 
                self.settings.model.laplace_smoothing_alpha
            )
            
            # Create TransitionMatrix object
            transition_matrix = TransitionMatrix(
                matrix=smoothed_matrix,
                observation_count=len(recent_data),
                last_updated=datetime.now(),
                window_start=recent_data[0].date,
                window_end=recent_data[-1].date
            )
            
            logger.info(f"Updated transition matrix with {len(recent_data)} observations")
            logger.debug(f"Window: {recent_data[0].date} to {recent_data[-1].date}")
            
            return transition_matrix
            
        except Exception as e:
            if isinstance(e, InsufficientDataError):
                raise
            raise CalculationError(
                calculation="transition_matrix_update",
                message=f"Failed to update transition matrix: {str(e)}"
            )
    
    def predict_next_state(
        self, 
        current_state: VRPState,
        transition_matrix: TransitionMatrix
    ) -> Dict[VRPState, float]:
        """
        Predict next state probabilities based on current state.
        
        Uses the transition matrix to get probability distribution
        over next possible states given the current state.
        
        Args:
            current_state: Current VRP state (enum or int)
            transition_matrix: Current transition matrix
            
        Returns:
            Dictionary mapping states to transition probabilities
            
        Raises:
            ModelStateError: If matrix is invalid
        """
        try:
            # Validate transition matrix
            self._validate_transition_matrix(transition_matrix)
            
            # Get probabilities for current state
            state_probabilities = transition_matrix.get_state_probabilities(current_state)
            
            # Convert Decimal probabilities to float for consistency
            float_probabilities = {
                state: float(prob) for state, prob in state_probabilities.items()
            }
            
            # Validate probabilities sum to 1
            prob_sum = sum(float_probabilities.values())
            if not (0.99 <= prob_sum <= 1.01):
                logger.warning(f"Transition probabilities sum to {prob_sum:.4f}, expected ~1.0")
            
            logger.debug(f"Predicted from state {current_state.name}: {float_probabilities}")
            
            return float_probabilities
            
        except Exception as e:
            raise ModelStateError(f"Failed to predict next state: {str(e)}")
    
    def apply_laplace_smoothing(
        self, 
        transition_counts: List[List[int]], 
        alpha: float = 1.0
    ) -> List[List[float]]:
        """
        Apply Laplace smoothing to transition counts.
        
        Converts raw transition counts to probabilities using:
        P[i,j] = (count[i,j] + α) / (count[i] + k*α)
        where k is the number of states.
        
        Args:
            transition_counts: Raw transition count matrix (5x5)
            alpha: Smoothing parameter (default 1.0)
            
        Returns:
            Smoothed probability matrix (5x5)
            
        Raises:
            CalculationError: If smoothing fails
        """
        try:
            # Validate input matrix
            if len(transition_counts) != self.num_states:
                raise ValueError(f"Expected {self.num_states}x{self.num_states} matrix, got {len(transition_counts)} rows")
            
            for i, row in enumerate(transition_counts):
                if len(row) != self.num_states:
                    raise ValueError(f"Row {i} has {len(row)} columns, expected {self.num_states}")
            
            # Apply Laplace smoothing
            smoothed_matrix = []
            
            for i in range(self.num_states):
                row_counts = transition_counts[i]
                row_total = sum(row_counts)
                
                # Apply smoothing formula: (count + alpha) / (total + k*alpha)
                smoothed_row = []
                denominator = row_total + self.num_states * alpha
                
                for j in range(self.num_states):
                    numerator = row_counts[j] + alpha
                    probability = numerator / denominator
                    smoothed_row.append(probability)
                
                # Ensure row sums to 1.0 (handle floating point precision)
                row_sum = sum(smoothed_row)
                if row_sum > 0:
                    smoothed_row = [p / row_sum for p in smoothed_row]
                
                smoothed_matrix.append(smoothed_row)
            
            logger.debug(f"Applied Laplace smoothing with alpha={alpha}")
            return smoothed_matrix
            
        except Exception as e:
            raise CalculationError(
                calculation="laplace_smoothing",
                message=f"Failed to apply Laplace smoothing: {str(e)}"
            )
    
    def _count_transitions(self, volatility_data: List[VolatilityData]) -> List[List[int]]:
        """
        Count state transitions in chronologically ordered data.
        
        Args:
            volatility_data: Sorted volatility data with VRP states
            
        Returns:
            5x5 matrix of transition counts
        """
        # Initialize count matrix
        counts = [[0 for _ in range(self.num_states)] for _ in range(self.num_states)]
        
        # Count transitions between consecutive observations
        for i in range(len(volatility_data) - 1):
            from_state = volatility_data[i].vrp_state
            to_state = volatility_data[i + 1].vrp_state
            
            # Convert states to 0-based indices
            from_idx = from_state.value - 1
            to_idx = to_state.value - 1
            
            counts[from_idx][to_idx] += 1
        
        # Log transition counts for debugging
        total_transitions = sum(sum(row) for row in counts)
        logger.debug(f"Counted {total_transitions} total transitions")
        
        return counts
    
    def _validate_transition_matrix(self, transition_matrix: TransitionMatrix) -> None:
        """
        Validate transition matrix structure and properties.
        
        Args:
            transition_matrix: Matrix to validate
            
        Raises:
            ModelStateError: If matrix is invalid
        """
        try:
            matrix = transition_matrix.matrix
            
            # Check dimensions
            if len(matrix) != self.num_states:
                raise ValueError(f"Matrix must have {self.num_states} rows, got {len(matrix)}")
            
            for i, row in enumerate(matrix):
                if len(row) != self.num_states:
                    raise ValueError(f"Row {i} must have {self.num_states} columns, got {len(row)}")
                
                # Check probabilities are valid
                for j, prob in enumerate(row):
                    if not (0 <= prob <= 1):
                        raise ValueError(f"Probability at [{i},{j}] = {prob} is not in [0,1]")
                
                # Check row sums to 1 (within tolerance)
                row_sum = sum(row)
                if not (0.98 <= row_sum <= 1.02):
                    raise ValueError(f"Row {i} sums to {row_sum}, expected ~1.0")
                    
        except Exception as e:
            raise ModelStateError(f"Invalid transition matrix: {str(e)}")
    
    def calculate_steady_state_distribution(
        self, 
        transition_matrix: TransitionMatrix,
        max_iterations: int = 1000,
        tolerance: float = 1e-8
    ) -> Dict[VRPState, float]:
        """
        Calculate steady-state distribution of the Markov chain.
        
        Uses power iteration to find the stationary distribution.
        
        Args:
            transition_matrix: Transition matrix
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary mapping states to steady-state probabilities
            
        Raises:
            CalculationError: If calculation fails to converge
        """
        try:
            matrix = np.array(transition_matrix.matrix)
            
            # Start with uniform distribution
            state_dist = np.ones(self.num_states) / self.num_states
            
            # Power iteration
            for iteration in range(max_iterations):
                new_dist = state_dist @ matrix
                
                # Check convergence
                if np.allclose(state_dist, new_dist, atol=tolerance):
                    logger.info(f"Steady state converged after {iteration} iterations")
                    break
                
                state_dist = new_dist
            else:
                logger.warning(f"Steady state did not converge after {max_iterations} iterations")
            
            # Convert to dictionary
            steady_state = {}
            for i, state in enumerate(VRPState):
                steady_state[state] = float(state_dist[i])
            
            return steady_state
            
        except Exception as e:
            raise CalculationError(
                calculation="steady_state_distribution",
                message=f"Failed to calculate steady state: {str(e)}"
            )
    
    def analyze_matrix_stability(
        self, 
        current_matrix: TransitionMatrix,
        previous_matrix: TransitionMatrix
    ) -> float:
        """
        Analyze stability between two transition matrices.
        
        Uses Frobenius norm to measure matrix difference.
        
        Args:
            current_matrix: Current transition matrix
            previous_matrix: Previous transition matrix
            
        Returns:
            Stability score (1.0 = identical, 0.0 = completely different)
        """
        try:
            current = np.array(current_matrix.matrix)
            previous = np.array(previous_matrix.matrix)
            
            # Calculate Frobenius norm of difference
            diff_norm = np.linalg.norm(current - previous, 'fro')
            
            # Normalize to [0, 1] scale (max possible difference is ~sqrt(20))
            max_possible_diff = np.sqrt(self.num_states * self.num_states * 2)
            stability = max(0.0, 1.0 - (diff_norm / max_possible_diff))
            
            logger.debug(f"Matrix stability score: {stability:.4f}")
            return float(stability)
            
        except Exception as e:
            logger.warning(f"Failed to calculate matrix stability: {str(e)}")
            return 0.5  # Return neutral score on error
    
    def get_transition_statistics(self, transition_matrix: TransitionMatrix) -> Dict[str, float]:
        """
        Calculate statistics for transition matrix analysis.
        
        Args:
            transition_matrix: Matrix to analyze
            
        Returns:
            Dictionary of matrix statistics
        """
        try:
            matrix = np.array(transition_matrix.matrix)
            
            # Calculate basic statistics
            stats = {
                'determinant': float(np.linalg.det(matrix)),
                'trace': float(np.trace(matrix)),
                'frobenius_norm': float(np.linalg.norm(matrix, 'fro')),
                'max_eigenvalue': float(max(np.real(np.linalg.eigvals(matrix)))),
                'observation_count': transition_matrix.observation_count,
                'diagonal_sum': float(np.sum(np.diag(matrix))),  # Self-transition probability
            }
            
            # Calculate entropy of each row (state)
            row_entropies = []
            for i in range(self.num_states):
                row = matrix[i]
                # Avoid log(0) by adding small epsilon
                row_probs = row + 1e-10
                entropy = -np.sum(row_probs * np.log(row_probs))
                row_entropies.append(entropy)
            
            stats['mean_row_entropy'] = float(np.mean(row_entropies))
            stats['max_row_entropy'] = float(np.max(row_entropies))
            stats['min_row_entropy'] = float(np.min(row_entropies))
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate transition statistics: {str(e)}")
            return {}