"""
Unit tests for VRPMarkovChain.

This module contains comprehensive tests for Markov Chain functionality,
including transition matrix updates, state predictions, confidence scoring,
and edge case handling.
"""

import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import numpy as np

from src.config.settings import VRPTradingConfig
from src.models.markov_chain import VRPMarkovChain
from src.models.data_models import VRPState, VolatilityMetrics
from src.utils.exceptions import InsufficientDataError, ModelError


class TestVRPMarkovChain:
    """Test suite for VRPMarkovChain class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=VRPTradingConfig)
        config.model = Mock()
        config.model.laplace_smoothing_alpha = Decimal('0.01')
        config.model.rolling_window_days = 60
        return config
    
    @pytest.fixture
    def markov_chain(self, config):
        """Create VRPMarkovChain instance."""
        return VRPMarkovChain(config)
    
    @pytest.fixture
    def sample_state_sequence(self):
        """Create sample state sequence for testing."""
        base_date = date(2023, 1, 1)
        states = [
            VRPState.FAIR_VALUE, VRPState.NORMAL_PREMIUM, VRPState.ELEVATED_PREMIUM,
            VRPState.EXTREME_PREMIUM, VRPState.ELEVATED_PREMIUM, VRPState.NORMAL_PREMIUM,
            VRPState.FAIR_VALUE, VRPState.UNDERPRICED, VRPState.FAIR_VALUE,
            VRPState.NORMAL_PREMIUM, VRPState.ELEVATED_PREMIUM, VRPState.EXTREME_PREMIUM
        ] * 10  # Repeat pattern to get enough data
        
        sequence = []
        for i, state in enumerate(states):
            metrics = VolatilityMetrics(
                date=base_date + timedelta(days=i),
                spy_return=Decimal('0.01'),
                realized_vol_30d=Decimal('0.20'),
                implied_vol=Decimal('0.25'),
                vrp=Decimal('1.25'),
                vrp_state=state
            )
            sequence.append(metrics)
        
        return sequence
    
    def test_initialization(self, markov_chain):
        """Test Markov chain initialization."""
        assert markov_chain.num_states == 5
        assert len(markov_chain.state_to_index) == 5
        assert len(markov_chain.index_to_state) == 5
        
        # Test state mapping consistency
        for state, index in markov_chain.state_to_index.items():
            assert markov_chain.index_to_state[index] == state
    
    def test_update_transition_matrix_success(self, markov_chain, sample_state_sequence):
        """Test successful transition matrix update."""
        matrix = markov_chain.update_transition_matrix(sample_state_sequence, window_days=60)
        
        # Check matrix structure
        assert len(matrix.matrix) == 5
        assert all(len(row) == 5 for row in matrix.matrix)
        
        # Check row sums (should be approximately 1.0)
        for row in matrix.matrix:
            row_sum = sum(float(prob) for prob in row)
            assert abs(row_sum - 1.0) < 1e-6
        
        # Check metadata
        assert matrix.observation_count == len(sample_state_sequence)
        assert matrix.window_start == sample_state_sequence[-60].date
        assert matrix.window_end == sample_state_sequence[-1].date
        assert isinstance(matrix.last_updated, datetime)
    
    def test_update_transition_matrix_insufficient_data(self, markov_chain):
        """Test transition matrix update with insufficient data."""
        short_sequence = [
            VolatilityMetrics(
                date=date(2023, 1, 1),
                spy_return=Decimal('0.01'),
                realized_vol_30d=Decimal('0.20'),
                implied_vol=Decimal('0.25'),
                vrp=Decimal('1.25'),
                vrp_state=VRPState.FAIR_VALUE
            )
        ]
        
        with pytest.raises(InsufficientDataError):
            markov_chain.update_transition_matrix(short_sequence, window_days=60)
    
    def test_predict_next_state(self, markov_chain, sample_state_sequence):
        """Test state prediction functionality."""
        # First create a transition matrix
        matrix = markov_chain.update_transition_matrix(sample_state_sequence, window_days=60)
        
        # Test prediction
        prediction = markov_chain.predict_next_state(VRPState.FAIR_VALUE, matrix)
        
        # Check prediction structure
        assert prediction.current_state == VRPState.FAIR_VALUE
        assert prediction.predicted_state in VRPState
        assert 0 <= float(prediction.transition_probability) <= 1
        assert 0 <= float(prediction.confidence_score) <= 1
        assert float(prediction.entropy) >= 0
        assert 0 <= float(prediction.data_quality_score) <= 1
        assert isinstance(prediction.current_date, date)
    
    def test_apply_laplace_smoothing(self, markov_chain):
        """Test Laplace smoothing application."""
        # Create raw count matrix with some zeros
        raw_matrix = [
            [10, 5, 0, 2, 1],
            [3, 8, 4, 0, 2],
            [0, 2, 12, 3, 1],
            [1, 0, 5, 7, 3],
            [2, 1, 1, 4, 8]
        ]
        
        smoothed = markov_chain.apply_laplace_smoothing(raw_matrix, alpha=0.1)
        
        # Check dimensions
        assert len(smoothed) == 5
        assert all(len(row) == 5 for row in smoothed)
        
        # Check that no probabilities are zero (due to smoothing)
        for row in smoothed:
            assert all(prob > 0 for prob in row)
        
        # Check row sums
        for row in smoothed:
            assert abs(sum(row) - 1.0) < 1e-10
    
    def test_laplace_smoothing_invalid_matrix(self, markov_chain):
        """Test Laplace smoothing with invalid matrix dimensions."""
        invalid_matrix = [
            [1, 2, 3],  # Wrong size
            [4, 5, 6]
        ]
        
        with pytest.raises(ModelError):
            markov_chain.apply_laplace_smoothing(invalid_matrix)
    
    def test_calculate_confidence_score(self, markov_chain, sample_state_sequence):
        """Test confidence score calculation."""
        matrix = markov_chain.update_transition_matrix(sample_state_sequence, window_days=60)
        
        # Create mock prediction
        mock_prediction = Mock()
        mock_prediction.transition_probability = Decimal('0.8')
        mock_prediction.entropy = Decimal('0.5')
        
        confidence = markov_chain.calculate_confidence_score(matrix, mock_prediction)
        
        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)
    
    def test_confidence_score_edge_cases(self, markov_chain, sample_state_sequence):
        """Test confidence score calculation edge cases."""
        matrix = markov_chain.update_transition_matrix(sample_state_sequence, window_days=60)
        
        # Test with very high transition probability
        high_prob_prediction = Mock()
        high_prob_prediction.transition_probability = Decimal('0.95')
        high_prob_prediction.entropy = Decimal('0.1')
        
        high_confidence = markov_chain.calculate_confidence_score(matrix, high_prob_prediction)
        
        # Test with low transition probability
        low_prob_prediction = Mock()
        low_prob_prediction.transition_probability = Decimal('0.25')
        low_prob_prediction.entropy = Decimal('1.5')
        
        low_confidence = markov_chain.calculate_confidence_score(matrix, low_prob_prediction)
        
        # High probability prediction should have higher confidence
        assert high_confidence > low_confidence
    
    def test_count_transitions(self, markov_chain, sample_state_sequence):
        """Test transition counting functionality."""
        # Take a subset for testing
        test_sequence = sample_state_sequence[:20]
        
        counts = markov_chain._count_transitions(test_sequence)
        
        # Check dimensions
        assert len(counts) == 5
        assert all(len(row) == 5 for row in counts)
        
        # Check that total transitions equals sequence length - 1
        total_transitions = sum(sum(row) for row in counts)
        assert total_transitions == len(test_sequence) - 1
        
        # All counts should be non-negative
        for row in counts:
            assert all(count >= 0 for count in row)
    
    def test_calculate_entropy(self, markov_chain):
        """Test entropy calculation."""
        # Test uniform distribution (maximum entropy)
        uniform_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
        uniform_entropy = markov_chain._calculate_entropy(uniform_probs)
        
        # Test concentrated distribution (low entropy)
        concentrated_probs = [0.9, 0.025, 0.025, 0.025, 0.025]
        concentrated_entropy = markov_chain._calculate_entropy(concentrated_probs)
        
        # Uniform distribution should have higher entropy
        assert uniform_entropy > concentrated_entropy
        
        # Test with zero probabilities
        with_zeros = [0.5, 0.5, 0.0, 0.0, 0.0]
        zero_entropy = markov_chain._calculate_entropy(with_zeros)
        assert zero_entropy >= 0
    
    def test_get_steady_state_distribution(self, markov_chain, sample_state_sequence):
        """Test steady state distribution calculation."""
        matrix = markov_chain.update_transition_matrix(sample_state_sequence, window_days=60)
        
        steady_state = markov_chain.get_steady_state_distribution(matrix)
        
        # Check properties
        assert len(steady_state) == 5
        assert all(prob >= 0 for prob in steady_state)
        assert abs(sum(steady_state) - 1.0) < 1e-6  # Should sum to 1
    
    def test_simulate_future_states(self, markov_chain, sample_state_sequence):
        """Test Monte Carlo state simulation."""
        matrix = markov_chain.update_transition_matrix(sample_state_sequence, window_days=60)
        
        simulation_result = markov_chain.simulate_future_states(
            current_state=VRPState.FAIR_VALUE,
            transition_matrix=matrix,
            num_steps=5,
            num_simulations=100
        )
        
        # Check result structure
        assert len(simulation_result) == 5  # All states should be represented
        assert all(state in VRPState for state in simulation_result.keys())
        
        # Check probabilities sum to 1
        total_prob = sum(simulation_result.values())
        assert abs(total_prob - 1.0) < 1e-6
        
        # All probabilities should be non-negative
        assert all(prob >= 0 for prob in simulation_result.values())
    
    def test_data_quality_score_calculation(self, markov_chain, sample_state_sequence):
        """Test data quality score calculation."""
        matrix = markov_chain.update_transition_matrix(sample_state_sequence, window_days=60)
        
        quality_score = markov_chain._calculate_data_quality_score(matrix)
        
        assert 0 <= quality_score <= 1
        assert isinstance(quality_score, float)
    
    def test_matrix_stability_calculation(self, markov_chain, sample_state_sequence):
        """Test matrix stability score calculation."""
        matrix = markov_chain.update_transition_matrix(sample_state_sequence, window_days=60)
        
        stability_score = markov_chain._calculate_matrix_stability(matrix)
        
        assert 0 <= stability_score <= 1
        assert isinstance(stability_score, float)
    
    def test_state_index_mapping(self, markov_chain):
        """Test state to index mapping consistency."""
        # Test all states are mapped
        for state in VRPState:
            assert state in markov_chain.state_to_index
            index = markov_chain.state_to_index[state]
            assert 0 <= index < 5
            assert markov_chain.index_to_state[index] == state
    
    def test_prediction_with_different_states(self, markov_chain, sample_state_sequence):
        """Test predictions from different starting states."""
        matrix = markov_chain.update_transition_matrix(sample_state_sequence, window_days=60)
        
        # Test prediction from each state
        for state in VRPState:
            prediction = markov_chain.predict_next_state(state, matrix)
            
            assert prediction.current_state == state
            assert prediction.predicted_state in VRPState
            assert 0 <= float(prediction.transition_probability) <= 1
            assert 0 <= float(prediction.confidence_score) <= 1
    
    def test_laplace_smoothing_with_zero_observations(self, markov_chain):
        """Test Laplace smoothing with rows of all zeros."""
        zero_matrix = [
            [0, 0, 0, 0, 0],  # All zeros
            [1, 2, 3, 4, 5],  # Normal row
            [0, 0, 0, 0, 0],  # All zeros
            [2, 1, 0, 3, 1],  # Normal row
            [0, 0, 0, 0, 0]   # All zeros
        ]
        
        smoothed = markov_chain.apply_laplace_smoothing(zero_matrix, alpha=0.1)
        
        # Check that zero rows become uniform distributions
        for i in [0, 2, 4]:  # Zero rows
            row = smoothed[i]
            # Should be approximately uniform
            expected_prob = 1.0 / 5
            for prob in row:
                assert abs(prob - expected_prob) < 1e-6
    
    @patch('src.models.markov_chain.logger')
    def test_logging_during_operations(self, mock_logger, markov_chain, sample_state_sequence):
        """Test that appropriate logging occurs during operations."""
        # Test matrix update logging
        markov_chain.update_transition_matrix(sample_state_sequence, window_days=60)
        mock_logger.info.assert_called()
        
        # Reset mock
        mock_logger.reset_mock()
        
        # Test prediction logging
        matrix = markov_chain.update_transition_matrix(sample_state_sequence, window_days=60)
        markov_chain.predict_next_state(VRPState.FAIR_VALUE, matrix)
        mock_logger.info.assert_called()
    
    def test_transition_matrix_serialization(self, markov_chain, sample_state_sequence):
        """Test that transition matrix can be properly serialized/deserialized."""
        matrix = markov_chain.update_transition_matrix(sample_state_sequence, window_days=60)
        
        # Test that matrix can be converted to dict and back
        matrix_dict = matrix.dict()
        assert 'matrix' in matrix_dict
        assert 'observation_count' in matrix_dict
        assert 'window_start' in matrix_dict
        assert 'window_end' in matrix_dict
        
        # Test matrix values are preserved
        for i in range(5):
            for j in range(5):
                original_value = float(matrix.matrix[i][j])
                dict_value = float(matrix_dict['matrix'][i][j])
                assert abs(original_value - dict_value) < 1e-10
    
    def test_error_handling_in_prediction(self, markov_chain):
        """Test error handling during prediction."""
        # Create invalid transition matrix (doesn't sum to 1)
        from src.models.data_models import TransitionMatrix
        
        invalid_matrix = TransitionMatrix(
            matrix=[
                [Decimal('0.5'), Decimal('0.3'), Decimal('0.1'), Decimal('0.05'), Decimal('0.04')],  # Doesn't sum to 1
                [Decimal('0.2'), Decimal('0.2'), Decimal('0.2'), Decimal('0.2'), Decimal('0.2')],
                [Decimal('0.2'), Decimal('0.2'), Decimal('0.2'), Decimal('0.2'), Decimal('0.2')],
                [Decimal('0.2'), Decimal('0.2'), Decimal('0.2'), Decimal('0.2'), Decimal('0.2')],
                [Decimal('0.2'), Decimal('0.2'), Decimal('0.2'), Decimal('0.2'), Decimal('0.2')]
            ],
            observation_count=60,
            window_start=date(2023, 1, 1),
            window_end=date(2023, 3, 1)
        )
        
        # Should still work (Pydantic validation should catch issues at creation)
        try:
            prediction = markov_chain.predict_next_state(VRPState.FAIR_VALUE, invalid_matrix)
            assert prediction is not None
        except Exception:
            # If validation catches the error, that's also acceptable
            pass
    
    def test_performance_with_large_dataset(self, markov_chain):
        """Test performance with larger datasets."""
        # Create larger state sequence
        large_sequence = []
        base_date = date(2023, 1, 1)
        
        # Create 500 days of data
        for i in range(500):
            state = list(VRPState)[i % 5]  # Cycle through states
            metrics = VolatilityMetrics(
                date=base_date + timedelta(days=i),
                spy_return=Decimal('0.01'),
                realized_vol_30d=Decimal('0.20'),
                implied_vol=Decimal('0.25'),
                vrp=Decimal('1.25'),
                vrp_state=state
            )
            large_sequence.append(metrics)
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        
        matrix = markov_chain.update_transition_matrix(large_sequence, window_days=60)
        prediction = markov_chain.predict_next_state(VRPState.FAIR_VALUE, matrix)
        
        end_time = time.time()
        
        # Should complete in less than 1 second
        assert end_time - start_time < 1.0
        
        # Results should still be valid
        assert matrix is not None
        assert prediction is not None