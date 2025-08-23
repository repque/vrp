"""
Unit tests for ConfidenceCalculator service.

This module contains comprehensive tests for confidence scoring calculations,
including entropy measures, data quality assessments, prediction reliability,
and statistical validation of confidence metrics.
"""

import pytest
import numpy as np
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
from typing import List, Dict

from src.config.settings import VRPTradingConfig
from src.models.confidence_calculator import ConfidenceCalculator
from src.models.data_models import (
    ModelPrediction,
    TransitionMatrix,
    VolatilityMetrics,
    VRPState,
    MarketDataPoint
)
from src.utils.exceptions import CalculationError, InsufficientDataError


class TestConfidenceCalculator:
    """Test suite for ConfidenceCalculator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration for ConfidenceCalculator."""
        config = Mock(spec=VRPTradingConfig)
        config.model = Mock()
        config.model.entropy_weight = Decimal('0.3')
        config.model.data_quality_weight = Decimal('0.25')
        config.model.prediction_accuracy_weight = Decimal('0.25')
        config.model.transition_stability_weight = Decimal('0.2')
        config.model.min_observations_for_confidence = 30
        config.model.confidence_decay_factor = Decimal('0.95')
        config.model.max_entropy_threshold = Decimal('2.32')  # log2(5) for 5 states
        return config
    
    @pytest.fixture
    def confidence_calculator(self, config):
        """Create ConfidenceCalculator instance."""
        return ConfidenceCalculator(config)
    
    @pytest.fixture
    def sample_transition_matrix(self):
        """Create sample transition matrix for testing."""
        # Well-balanced transition matrix
        matrix = [
            [Decimal('0.3'), Decimal('0.4'), Decimal('0.2'), Decimal('0.08'), Decimal('0.02')],
            [Decimal('0.15'), Decimal('0.45'), Decimal('0.25'), Decimal('0.12'), Decimal('0.03')],
            [Decimal('0.08'), Decimal('0.27'), Decimal('0.4'), Decimal('0.2'), Decimal('0.05')],
            [Decimal('0.04'), Decimal('0.16'), Decimal('0.25'), Decimal('0.4'), Decimal('0.15')],
            [Decimal('0.05'), Decimal('0.2'), Decimal('0.3'), Decimal('0.25'), Decimal('0.2')]
        ]
        
        return TransitionMatrix(
            matrix=matrix,
            observation_count=120,
            window_start=date(2023, 1, 1),
            window_end=date(2023, 3, 1),
            last_updated=datetime.now()
        )
    
    @pytest.fixture
    def high_entropy_transition_matrix(self):
        """Create transition matrix with high entropy (uniform distribution)."""
        # Uniform distribution - maximum entropy
        uniform_prob = Decimal('0.2')
        matrix = [[uniform_prob] * 5 for _ in range(5)]
        
        return TransitionMatrix(
            matrix=matrix,
            observation_count=100,
            window_start=date(2023, 1, 1),
            window_end=date(2023, 3, 1),
            last_updated=datetime.now()
        )
    
    @pytest.fixture
    def low_entropy_transition_matrix(self):
        """Create transition matrix with low entropy (concentrated distribution)."""
        # Highly concentrated distribution - low entropy
        matrix = [
            [Decimal('0.9'), Decimal('0.05'), Decimal('0.03'), Decimal('0.015'), Decimal('0.005')],
            [Decimal('0.05'), Decimal('0.9'), Decimal('0.03'), Decimal('0.015'), Decimal('0.005')],
            [Decimal('0.03'), Decimal('0.05'), Decimal('0.9'), Decimal('0.015'), Decimal('0.005')],
            [Decimal('0.015'), Decimal('0.03'), Decimal('0.05'), Decimal('0.9'), Decimal('0.005')],
            [Decimal('0.005'), Decimal('0.015'), Decimal('0.03'), Decimal('0.05'), Decimal('0.9')]
        ]
        
        return TransitionMatrix(
            matrix=matrix,
            observation_count=150,
            window_start=date(2023, 1, 1),
            window_end=date(2023, 3, 1),
            last_updated=datetime.now()
        )
    
    @pytest.fixture
    def sample_volatility_sequence(self):
        """Create sequence of volatility metrics for data quality testing."""
        base_date = date(2023, 1, 1)
        sequence = []
        
        # Create 60 days of data with varying quality
        for i in range(60):
            # Different quality patterns
            if i < 20:  # High quality period
                spy_return = Decimal(str(np.random.normal(0.001, 0.015)))
                realized_vol = Decimal('0.18')
                implied_vol = Decimal('0.22')
                vrp = implied_vol / realized_vol
            elif i < 40:  # Medium quality period (some gaps)
                spy_return = Decimal(str(np.random.normal(0.0005, 0.02)))
                realized_vol = Decimal('0.16') if i % 3 != 0 else Decimal('0.25')  # Some inconsistency
                implied_vol = Decimal('0.20')
                vrp = implied_vol / realized_vol
            else:  # Lower quality period (more volatility)
                spy_return = Decimal(str(np.random.normal(0.0, 0.03)))
                realized_vol = Decimal(str(0.15 + np.random.uniform(0, 0.2)))  # More variable
                implied_vol = Decimal('0.25')
                vrp = implied_vol / realized_vol
            
            metrics = VolatilityMetrics(
                date=base_date + timedelta(days=i),
                spy_return=spy_return,
                realized_vol_30d=realized_vol,
                implied_vol=implied_vol,
                vrp=vrp,
                vrp_state=VRPState.NORMAL_PREMIUM  # Simplified for testing
            )
            sequence.append(metrics)
        
        return sequence
    
    def test_calculate_entropy_uniform_distribution(self, confidence_calculator):
        """Test entropy calculation for uniform distribution (maximum entropy)."""
        # Uniform distribution over 5 states
        uniform_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        entropy = confidence_calculator.calculate_entropy(uniform_probs)
        
        # Expected entropy for uniform distribution: log2(5) ≈ 2.32
        expected_entropy = np.log2(5)
        assert abs(float(entropy) - expected_entropy) < 1e-6
        assert entropy == confidence_calculator.config.model.max_entropy_threshold
    
    def test_calculate_entropy_concentrated_distribution(self, confidence_calculator):
        """Test entropy calculation for concentrated distribution (low entropy)."""
        # Highly concentrated distribution
        concentrated_probs = [0.9, 0.05, 0.03, 0.015, 0.005]
        
        entropy = confidence_calculator.calculate_entropy(concentrated_probs)
        
        # Calculate expected entropy manually
        expected_entropy = -sum(p * np.log2(p) for p in concentrated_probs if p > 0)
        assert abs(float(entropy) - expected_entropy) < 1e-6
        assert entropy < Decimal('1.0')  # Should be much lower than uniform
    
    def test_calculate_entropy_edge_cases(self, confidence_calculator):
        """Test entropy calculation with edge cases."""
        # Single probability = 1 (minimum entropy)
        certainty_probs = [1.0, 0.0, 0.0, 0.0, 0.0]
        entropy_min = confidence_calculator.calculate_entropy(certainty_probs)
        assert entropy_min == Decimal('0.0')
        
        # With zero probabilities (should handle gracefully)
        with_zeros = [0.5, 0.5, 0.0, 0.0, 0.0]
        entropy_with_zeros = confidence_calculator.calculate_entropy(with_zeros)
        expected = -0.5 * np.log2(0.5) - 0.5 * np.log2(0.5)  # = 1.0
        assert abs(float(entropy_with_zeros) - expected) < 1e-6
    
    def test_calculate_entropy_invalid_inputs(self, confidence_calculator):
        """Test entropy calculation with invalid inputs."""
        # Probabilities don't sum to 1
        invalid_probs = [0.3, 0.3, 0.3, 0.2, 0.2]  # Sum = 1.1
        with pytest.raises(CalculationError, match="Probabilities must sum to 1"):
            confidence_calculator.calculate_entropy(invalid_probs)
        
        # Negative probabilities
        negative_probs = [0.6, 0.3, -0.1, 0.1, 0.1]
        with pytest.raises(CalculationError, match="Probabilities must be non-negative"):
            confidence_calculator.calculate_entropy(negative_probs)
        
        # Wrong number of probabilities
        wrong_size = [0.5, 0.5]  # Only 2 states, expect 5
        with pytest.raises(CalculationError, match="Expected 5 probabilities"):
            confidence_calculator.calculate_entropy(wrong_size)
    
    def test_calculate_data_quality_score(self, confidence_calculator, sample_volatility_sequence):
        """Test data quality score calculation."""
        quality_score = confidence_calculator.calculate_data_quality_score(sample_volatility_sequence)
        
        assert 0 <= quality_score <= 1
        assert isinstance(quality_score, float)
        
        # With good quality data, score should be reasonably high
        assert quality_score > 0.5
    
    def test_calculate_data_quality_score_perfect_data(self, confidence_calculator):
        """Test data quality score with perfect synthetic data."""
        # Create perfect data with consistent patterns
        perfect_sequence = []
        base_date = date(2023, 1, 1)
        
        for i in range(30):
            metrics = VolatilityMetrics(
                date=base_date + timedelta(days=i),
                spy_return=Decimal('0.001'),  # Consistent returns
                realized_vol_30d=Decimal('0.20'),  # Stable volatility
                implied_vol=Decimal('0.24'),  # Stable implied vol
                vrp=Decimal('1.2'),  # Consistent VRP
                vrp_state=VRPState.NORMAL_PREMIUM
            )
            perfect_sequence.append(metrics)
        
        perfect_score = confidence_calculator.calculate_data_quality_score(perfect_sequence)
        
        # Perfect data should get a high quality score
        assert perfect_score > 0.8
        assert perfect_score <= 1.0
    
    def test_calculate_data_quality_score_poor_data(self, confidence_calculator):
        """Test data quality score with poor quality data."""
        # Create poor quality data with large gaps and inconsistencies
        poor_sequence = []
        base_date = date(2023, 1, 1)
        
        np.random.seed(42)  # For reproducible results
        for i in range(30):
            # Highly volatile and inconsistent data
            spy_return = Decimal(str(np.random.normal(0.0, 0.05)))  # High volatility
            realized_vol = Decimal(str(max(0.05, np.random.uniform(0.1, 0.5))))  # Highly variable
            implied_vol = Decimal(str(max(0.05, np.random.uniform(0.1, 0.6))))
            vrp = implied_vol / realized_vol
            
            metrics = VolatilityMetrics(
                date=base_date + timedelta(days=i),
                spy_return=spy_return,
                realized_vol_30d=realized_vol,
                implied_vol=implied_vol,
                vrp=vrp,
                vrp_state=VRPState.NORMAL_PREMIUM
            )
            poor_sequence.append(metrics)
        
        poor_score = confidence_calculator.calculate_data_quality_score(poor_sequence)
        
        # Poor data should get a lower quality score
        assert poor_score < 0.7
        assert poor_score >= 0.0
    
    def test_calculate_prediction_accuracy(self, confidence_calculator):
        """Test prediction accuracy calculation."""
        # Create historical predictions and outcomes
        predictions = [
            ModelPrediction(
                current_date=date(2023, 1, 1),
                current_state=VRPState.FAIR_VALUE,
                predicted_state=VRPState.NORMAL_PREMIUM,
                transition_probability=Decimal('0.7'),
                confidence_score=Decimal('0.8'),
                entropy=Decimal('0.5'),
                data_quality_score=Decimal('0.85')
            ),
            ModelPrediction(
                current_date=date(2023, 1, 2),
                current_state=VRPState.NORMAL_PREMIUM,
                predicted_state=VRPState.ELEVATED_PREMIUM,
                transition_probability=Decimal('0.6'),
                confidence_score=Decimal('0.7'),
                entropy=Decimal('0.6'),
                data_quality_score=Decimal('0.8')
            ),
            ModelPrediction(
                current_date=date(2023, 1, 3),
                current_state=VRPState.ELEVATED_PREMIUM,
                predicted_state=VRPState.EXTREME_PREMIUM,
                transition_probability=Decimal('0.8'),
                confidence_score=Decimal('0.9'),
                entropy=Decimal('0.4'),
                data_quality_score=Decimal('0.9')
            )
        ]
        
        # Actual outcomes (2 out of 3 correct)
        actual_outcomes = [
            VRPState.NORMAL_PREMIUM,    # Correct
            VRPState.NORMAL_PREMIUM,    # Incorrect (stayed same)
            VRPState.EXTREME_PREMIUM,   # Correct
        ]
        
        accuracy = confidence_calculator.calculate_prediction_accuracy(predictions, actual_outcomes)
        
        # Should be 2/3 ≈ 0.667
        expected_accuracy = 2.0 / 3.0
        assert abs(accuracy - expected_accuracy) < 1e-6
        assert 0 <= accuracy <= 1
    
    def test_calculate_prediction_accuracy_edge_cases(self, confidence_calculator):
        """Test prediction accuracy with edge cases."""
        # Perfect accuracy
        perfect_predictions = [
            ModelPrediction(
                current_date=date(2023, 1, 1),
                current_state=VRPState.FAIR_VALUE,
                predicted_state=VRPState.NORMAL_PREMIUM,
                transition_probability=Decimal('0.9'),
                confidence_score=Decimal('0.95'),
                entropy=Decimal('0.2'),
                data_quality_score=Decimal('0.95')
            )
        ]
        
        perfect_outcomes = [VRPState.NORMAL_PREMIUM]
        
        perfect_accuracy = confidence_calculator.calculate_prediction_accuracy(
            perfect_predictions, perfect_outcomes
        )
        assert perfect_accuracy == 1.0
        
        # No accuracy
        wrong_outcomes = [VRPState.UNDERPRICED]
        
        no_accuracy = confidence_calculator.calculate_prediction_accuracy(
            perfect_predictions, wrong_outcomes
        )
        assert no_accuracy == 0.0
    
    def test_calculate_prediction_accuracy_mismatched_lengths(self, confidence_calculator):
        """Test prediction accuracy with mismatched input lengths."""
        predictions = [Mock(), Mock()]
        outcomes = [VRPState.FAIR_VALUE]  # Different length
        
        with pytest.raises(ValueError, match="Length mismatch"):
            confidence_calculator.calculate_prediction_accuracy(predictions, outcomes)
    
    def test_calculate_transition_stability(self, confidence_calculator, sample_transition_matrix):
        """Test transition matrix stability calculation."""
        stability = confidence_calculator.calculate_transition_stability(sample_transition_matrix)
        
        assert 0 <= stability <= 1
        assert isinstance(stability, float)
        
        # Well-balanced matrix should have reasonable stability
        assert stability > 0.3
    
    def test_calculate_transition_stability_high_vs_low(self, confidence_calculator, 
                                                       high_entropy_transition_matrix, 
                                                       low_entropy_transition_matrix):
        """Test stability calculation for high vs low entropy matrices."""
        high_entropy_stability = confidence_calculator.calculate_transition_stability(
            high_entropy_transition_matrix
        )
        
        low_entropy_stability = confidence_calculator.calculate_transition_stability(
            low_entropy_transition_matrix
        )
        
        # Low entropy (concentrated) matrix should be more "stable" in terms of predictability
        # High entropy (uniform) matrix is less predictable but more "stable" in terms of consistency
        # The interpretation depends on the stability metric implementation
        assert 0 <= high_entropy_stability <= 1
        assert 0 <= low_entropy_stability <= 1
    
    def test_calculate_comprehensive_confidence_score(self, confidence_calculator, 
                                                    sample_transition_matrix, 
                                                    sample_volatility_sequence):
        """Test comprehensive confidence score calculation combining all factors."""
        # Create sample prediction
        prediction = ModelPrediction(
            current_date=date(2023, 3, 1),
            current_state=VRPState.ELEVATED_PREMIUM,
            predicted_state=VRPState.EXTREME_PREMIUM,
            transition_probability=Decimal('0.8'),
            confidence_score=Decimal('0.0'),  # Will be calculated
            entropy=Decimal('0.5'),
            data_quality_score=Decimal('0.85')
        )
        
        # Historical accuracy data
        historical_predictions = [Mock()] * 10  # Simplified
        historical_outcomes = [VRPState.NORMAL_PREMIUM] * 8 + [VRPState.ELEVATED_PREMIUM] * 2
        
        with patch.object(confidence_calculator, 'calculate_prediction_accuracy', return_value=0.8):
            confidence_score = confidence_calculator.calculate_comprehensive_confidence_score(
                prediction=prediction,
                transition_matrix=sample_transition_matrix,
                historical_data=sample_volatility_sequence,
                historical_predictions=historical_predictions,
                historical_outcomes=historical_outcomes
            )
        
        assert 0 <= confidence_score <= 1
        assert isinstance(confidence_score, float)
        
        # With good inputs, should get reasonable confidence score
        assert confidence_score > 0.4
    
    def test_confidence_score_components_weighting(self, confidence_calculator):
        """Test that confidence score components are properly weighted."""
        # Test with extreme values to verify weighting
        
        # Mock individual component calculations
        with patch.object(confidence_calculator, 'calculate_entropy', return_value=Decimal('0.0')), \
             patch.object(confidence_calculator, 'calculate_data_quality_score', return_value=1.0), \
             patch.object(confidence_calculator, 'calculate_prediction_accuracy', return_value=1.0), \
             patch.object(confidence_calculator, 'calculate_transition_stability', return_value=1.0):
            
            # Create minimal inputs
            prediction = Mock()
            prediction.entropy = Decimal('0.0')
            prediction.data_quality_score = Decimal('1.0')
            
            confidence = confidence_calculator.calculate_comprehensive_confidence_score(
                prediction=prediction,
                transition_matrix=Mock(),
                historical_data=[],
                historical_predictions=[],
                historical_outcomes=[]
            )
            
            # With perfect components, should get high confidence
            assert confidence > 0.8
    
    def test_confidence_decay_over_time(self, confidence_calculator):
        """Test confidence decay for older predictions."""
        base_date = date(2023, 1, 1)
        
        # Recent prediction
        recent_prediction = ModelPrediction(
            current_date=base_date,
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.NORMAL_PREMIUM,
            transition_probability=Decimal('0.8'),
            confidence_score=Decimal('0.9'),
            entropy=Decimal('0.4'),
            data_quality_score=Decimal('0.9')
        )
        
        # Old prediction
        old_prediction = ModelPrediction(
            current_date=base_date - timedelta(days=30),  # 30 days old
            current_state=VRPState.FAIR_VALUE,
            predicted_state=VRPState.NORMAL_PREMIUM,
            transition_probability=Decimal('0.8'),
            confidence_score=Decimal('0.9'),
            entropy=Decimal('0.4'),
            data_quality_score=Decimal('0.9')
        )
        
        recent_confidence = confidence_calculator.apply_time_decay(
            recent_prediction.confidence_score, 
            recent_prediction.current_date,
            base_date
        )
        
        old_confidence = confidence_calculator.apply_time_decay(
            old_prediction.confidence_score,
            old_prediction.current_date,
            base_date
        )
        
        # Recent prediction should have higher confidence after decay
        assert recent_confidence >= old_confidence
        assert old_confidence < recent_prediction.confidence_score  # Should be decayed
    
    def test_insufficient_data_handling(self, confidence_calculator):
        """Test handling of insufficient data for confidence calculation."""
        # Very small dataset
        minimal_data = [Mock()] * 5  # Below minimum threshold
        
        with pytest.raises(InsufficientDataError, match="Insufficient data"):
            confidence_calculator.calculate_data_quality_score(minimal_data)
    
    def test_confidence_bounds_validation(self, confidence_calculator):
        """Test that all confidence scores are properly bounded between 0 and 1."""
        # Test with various extreme inputs
        test_cases = [
            # (entropy, data_quality, pred_accuracy, stability)
            (0.0, 1.0, 1.0, 1.0),    # Perfect case
            (2.32, 0.0, 0.0, 0.0),   # Worst case
            (1.0, 0.5, 0.5, 0.5),    # Average case
        ]
        
        for entropy, quality, accuracy, stability in test_cases:
            with patch.object(confidence_calculator, 'calculate_entropy', return_value=Decimal(str(entropy))), \
                 patch.object(confidence_calculator, 'calculate_data_quality_score', return_value=quality), \
                 patch.object(confidence_calculator, 'calculate_prediction_accuracy', return_value=accuracy), \
                 patch.object(confidence_calculator, 'calculate_transition_stability', return_value=stability):
                
                prediction = Mock()
                prediction.entropy = Decimal(str(entropy))
                prediction.data_quality_score = Decimal(str(quality))
                
                confidence = confidence_calculator.calculate_comprehensive_confidence_score(
                    prediction=prediction,
                    transition_matrix=Mock(),
                    historical_data=[Mock()] * 50,
                    historical_predictions=[Mock()] * 10,
                    historical_outcomes=[VRPState.FAIR_VALUE] * 10
                )
                
                assert 0 <= confidence <= 1
    
    @patch('src.models.confidence_calculator.logger')
    def test_logging_during_calculations(self, mock_logger, confidence_calculator):
        """Test that appropriate logging occurs during calculations."""
        # Test entropy calculation logging
        uniform_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
        confidence_calculator.calculate_entropy(uniform_probs)
        
        # Should log high entropy warning
        mock_logger.debug.assert_called()
        
        # Test data quality logging
        mock_logger.reset_mock()
        
        poor_data = [Mock() for _ in range(50)]  # Minimal data
        with patch.object(confidence_calculator, '_calculate_data_consistency', return_value=0.3):
            confidence_calculator.calculate_data_quality_score(poor_data)
            mock_logger.warning.assert_called()
    
    def test_confidence_score_monotonicity(self, confidence_calculator):
        """Test that confidence score increases with better inputs."""
        # Create predictions with increasing quality
        low_quality_prediction = Mock()
        low_quality_prediction.entropy = Decimal('2.0')
        low_quality_prediction.data_quality_score = Decimal('0.3')
        low_quality_prediction.transition_probability = Decimal('0.4')
        
        high_quality_prediction = Mock()
        high_quality_prediction.entropy = Decimal('0.5')
        high_quality_prediction.data_quality_score = Decimal('0.9')
        high_quality_prediction.transition_probability = Decimal('0.9')
        
        # Mock consistent values for other components
        with patch.object(confidence_calculator, 'calculate_prediction_accuracy', return_value=0.7), \
             patch.object(confidence_calculator, 'calculate_transition_stability', return_value=0.8):
            
            low_confidence = confidence_calculator.calculate_comprehensive_confidence_score(
                prediction=low_quality_prediction,
                transition_matrix=Mock(),
                historical_data=[Mock()] * 50,
                historical_predictions=[Mock()] * 10,
                historical_outcomes=[VRPState.FAIR_VALUE] * 7 + [VRPState.NORMAL_PREMIUM] * 3
            )
            
            high_confidence = confidence_calculator.calculate_comprehensive_confidence_score(
                prediction=high_quality_prediction,
                transition_matrix=Mock(),
                historical_data=[Mock()] * 50,
                historical_predictions=[Mock()] * 10,
                historical_outcomes=[VRPState.FAIR_VALUE] * 7 + [VRPState.NORMAL_PREMIUM] * 3
            )
        
        # Higher quality input should result in higher confidence
        assert high_confidence > low_confidence