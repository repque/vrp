"""
Unit tests for VRP Classifier service.

This module contains comprehensive tests for VRP state classification,
focusing on boundary conditions, edge cases, and mathematical accuracy
of the VRP calculation and state assignment logic.
"""

import pytest
import numpy as np
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
from typing import List

from src.config.settings import Settings
from src.services.vrp_classifier import VRPClassifier
from src.models.data_models import MarketDataPoint, VolatilityMetrics, VRPState
from src.utils.exceptions import CalculationError, InsufficientDataError, ValidationError


class TestVRPClassifier:
    """Test suite for VRPClassifier class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration for VRPClassifier."""
        config = Mock(spec=Settings)
        config.model = Mock()
        config.model.vrp_underpriced_threshold = Decimal('0.90')
        config.model.vrp_fair_lower_threshold = Decimal('0.90')
        config.model.vrp_fair_upper_threshold = Decimal('1.10')
        config.model.vrp_normal_upper_threshold = Decimal('1.30')
        config.model.vrp_elevated_upper_threshold = Decimal('1.50')
        config.model.volatility_window_days = 30
        config.model.min_data_points = 20
        config.model.annualization_factor = Decimal('252')  # Trading days per year
        return config
    
    @pytest.fixture
    def vrp_classifier(self, config):
        """Create VRPClassifier instance."""
        return VRPClassifier(config)
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        base_date = date(2023, 1, 1)
        data = []
        base_price = Decimal('400.0')
        base_vix = Decimal('20.0')
        
        # Create 60 days of data with controlled volatility
        np.random.seed(42)  # For reproducible tests
        for i in range(60):
            # Simulate different volatility regimes
            if i < 20:  # Low volatility period
                daily_return = np.random.normal(0.0005, 0.008)  # ~0.8% daily vol
                vix_multiplier = np.random.uniform(0.95, 1.05)
            elif i < 40:  # Normal volatility period
                daily_return = np.random.normal(0.0003, 0.015)  # ~1.5% daily vol
                vix_multiplier = np.random.uniform(0.9, 1.2)
            else:  # High volatility period
                daily_return = np.random.normal(-0.0002, 0.025)  # ~2.5% daily vol
                vix_multiplier = np.random.uniform(1.1, 1.6)
            
            base_price = base_price * (1 + Decimal(str(daily_return)))
            base_vix = max(Decimal('8.0'), base_vix * Decimal(str(vix_multiplier)))
            
            # Create OHLC with realistic intraday range
            daily_range = abs(daily_return) * 2
            open_price = base_price * Decimal(str(1 - daily_return / 2))
            high_price = base_price * Decimal(str(1 + daily_range / 2))
            low_price = base_price * Decimal(str(1 - daily_range / 2))
            
            point = MarketDataPoint(
                date=base_date + timedelta(days=i),
                spy_open=open_price,
                spy_high=max(high_price, low_price, base_price),
                spy_low=min(high_price, low_price, base_price),
                spy_close=base_price,
                spy_volume=int(np.random.uniform(80_000_000, 120_000_000)),
                vix_close=base_vix
            )
            
            data.append(point)
        
        return data
    
    def test_calculate_vrp_normal_cases(self, vrp_classifier):
        """Test VRP calculation with normal input values."""
        # Test case 1: VRP = 1.25 (normal premium)
        implied_vol = Decimal('25.0')  # VIX value
        realized_vol = Decimal('0.20')  # 20% annualized
        
        vrp = vrp_classifier.calculate_vrp(implied_vol, realized_vol)
        expected_vrp = (implied_vol / 100) / realized_vol  # 0.25 / 0.20 = 1.25
        
        assert abs(vrp - expected_vrp) < Decimal('0.0001')
        assert vrp == Decimal('1.25')
        
        # Test case 2: VRP = 0.8 (underpriced)
        implied_vol = Decimal('16.0')
        realized_vol = Decimal('0.20')
        
        vrp = vrp_classifier.calculate_vrp(implied_vol, realized_vol)
        expected_vrp = Decimal('0.8')  # 0.16 / 0.20
        
        assert vrp == expected_vrp
        
        # Test case 3: VRP = 2.0 (extreme premium)
        implied_vol = Decimal('40.0')
        realized_vol = Decimal('0.20')
        
        vrp = vrp_classifier.calculate_vrp(implied_vol, realized_vol)
        expected_vrp = Decimal('2.0')  # 0.40 / 0.20
        
        assert vrp == expected_vrp
    
    def test_calculate_vrp_edge_cases(self, vrp_classifier):
        """Test VRP calculation with edge case values."""
        # Very low VIX
        vrp = vrp_classifier.calculate_vrp(Decimal('8.0'), Decimal('0.15'))
        assert vrp == Decimal('0.08') / Decimal('0.15')  # ≈ 0.533
        
        # Very high VIX
        vrp = vrp_classifier.calculate_vrp(Decimal('80.0'), Decimal('0.15'))
        assert vrp == Decimal('0.80') / Decimal('0.15')  # ≈ 5.333
        
        # Very low realized volatility
        vrp = vrp_classifier.calculate_vrp(Decimal('20.0'), Decimal('0.05'))
        assert vrp == Decimal('0.20') / Decimal('0.05')  # = 4.0
        
        # Very high realized volatility
        vrp = vrp_classifier.calculate_vrp(Decimal('20.0'), Decimal('0.80'))
        assert vrp == Decimal('0.20') / Decimal('0.80')  # = 0.25
    
    def test_calculate_vrp_invalid_inputs(self, vrp_classifier):
        """Test VRP calculation with invalid inputs."""
        # Zero VIX
        with pytest.raises(CalculationError, match="VIX must be positive"):
            vrp_classifier.calculate_vrp(Decimal('0.0'), Decimal('0.20'))
        
        # Negative VIX
        with pytest.raises(CalculationError, match="VIX must be positive"):
            vrp_classifier.calculate_vrp(Decimal('-5.0'), Decimal('0.20'))
        
        # Zero realized volatility
        with pytest.raises(CalculationError, match="Realized volatility must be positive"):
            vrp_classifier.calculate_vrp(Decimal('20.0'), Decimal('0.0'))
        
        # Negative realized volatility
        with pytest.raises(CalculationError, match="Realized volatility must be positive"):
            vrp_classifier.calculate_vrp(Decimal('20.0'), Decimal('-0.10'))
    
    def test_classify_vrp_state_all_states(self, vrp_classifier):
        """Test VRP state classification for all possible states."""
        # UNDERPRICED state
        assert vrp_classifier.classify_vrp_state(Decimal('0.85')) == VRPState.UNDERPRICED
        assert vrp_classifier.classify_vrp_state(Decimal('0.50')) == VRPState.UNDERPRICED
        assert vrp_classifier.classify_vrp_state(Decimal('0.89')) == VRPState.UNDERPRICED
        
        # FAIR_VALUE state
        assert vrp_classifier.classify_vrp_state(Decimal('0.90')) == VRPState.FAIR_VALUE
        assert vrp_classifier.classify_vrp_state(Decimal('1.00')) == VRPState.FAIR_VALUE
        assert vrp_classifier.classify_vrp_state(Decimal('1.09')) == VRPState.FAIR_VALUE
        
        # NORMAL_PREMIUM state
        assert vrp_classifier.classify_vrp_state(Decimal('1.10')) == VRPState.NORMAL_PREMIUM
        assert vrp_classifier.classify_vrp_state(Decimal('1.20')) == VRPState.NORMAL_PREMIUM
        assert vrp_classifier.classify_vrp_state(Decimal('1.29')) == VRPState.NORMAL_PREMIUM
        
        # ELEVATED_PREMIUM state
        assert vrp_classifier.classify_vrp_state(Decimal('1.30')) == VRPState.ELEVATED_PREMIUM
        assert vrp_classifier.classify_vrp_state(Decimal('1.40')) == VRPState.ELEVATED_PREMIUM
        assert vrp_classifier.classify_vrp_state(Decimal('1.49')) == VRPState.ELEVATED_PREMIUM
        
        # EXTREME_PREMIUM state
        assert vrp_classifier.classify_vrp_state(Decimal('1.50')) == VRPState.EXTREME_PREMIUM
        assert vrp_classifier.classify_vrp_state(Decimal('2.00')) == VRPState.EXTREME_PREMIUM
        assert vrp_classifier.classify_vrp_state(Decimal('5.00')) == VRPState.EXTREME_PREMIUM
    
    def test_classify_vrp_state_boundary_conditions(self, vrp_classifier):
        """Test VRP state classification at exact boundary values."""
        # Test exact threshold boundaries
        assert vrp_classifier.classify_vrp_state(Decimal('0.900000')) == VRPState.FAIR_VALUE
        assert vrp_classifier.classify_vrp_state(Decimal('1.100000')) == VRPState.NORMAL_PREMIUM
        assert vrp_classifier.classify_vrp_state(Decimal('1.300000')) == VRPState.ELEVATED_PREMIUM
        assert vrp_classifier.classify_vrp_state(Decimal('1.500000')) == VRPState.EXTREME_PREMIUM
        
        # Test values just below thresholds
        assert vrp_classifier.classify_vrp_state(Decimal('0.899999')) == VRPState.UNDERPRICED
        assert vrp_classifier.classify_vrp_state(Decimal('1.099999')) == VRPState.FAIR_VALUE
        assert vrp_classifier.classify_vrp_state(Decimal('1.299999')) == VRPState.NORMAL_PREMIUM
        assert vrp_classifier.classify_vrp_state(Decimal('1.499999')) == VRPState.ELEVATED_PREMIUM
    
    def test_classify_vrp_state_precision_handling(self, vrp_classifier):
        """Test VRP state classification with high precision values."""
        # Test with many decimal places
        assert vrp_classifier.classify_vrp_state(Decimal('1.100000001')) == VRPState.NORMAL_PREMIUM
        assert vrp_classifier.classify_vrp_state(Decimal('1.099999999')) == VRPState.FAIR_VALUE
        
        # Test with floating point precision issues
        slightly_below_threshold = Decimal('1.1') - Decimal('0.0000000001')
        assert vrp_classifier.classify_vrp_state(slightly_below_threshold) == VRPState.FAIR_VALUE
        
        slightly_above_threshold = Decimal('1.1') + Decimal('0.0000000001')
        assert vrp_classifier.classify_vrp_state(slightly_above_threshold) == VRPState.NORMAL_PREMIUM
    
    def test_calculate_realized_volatility_success(self, vrp_classifier, sample_market_data):
        """Test successful realized volatility calculation."""
        # Use sufficient data points
        result = vrp_classifier.calculate_realized_volatility(sample_market_data, window_days=30)
        
        assert len(result) > 0
        assert len(result) == len(sample_market_data) - 30  # Rolling windows
        
        # Check that all volatilities are positive and reasonable
        for vol_metrics in result:
            assert vol_metrics.realized_vol_30d > 0
            assert vol_metrics.realized_vol_30d < Decimal('2.0')  # Less than 200% annualized
            assert vol_metrics.implied_vol > 0
            assert vol_metrics.vrp > 0
            assert vol_metrics.vrp_state in VRPState
    
    def test_calculate_realized_volatility_insufficient_data(self, vrp_classifier):
        """Test realized volatility calculation with insufficient data."""
        # Create minimal data
        short_data = [
            MarketDataPoint(
                date=date(2023, 1, 1),
                spy_open=Decimal('400.0'),
                spy_high=Decimal('405.0'),
                spy_low=Decimal('395.0'),
                spy_close=Decimal('402.0'),
                spy_volume=100000000,
                vix_close=Decimal('20.0')
            )
        ]
        
        with pytest.raises(InsufficientDataError, match="Insufficient data:"):
            vrp_classifier.calculate_realized_volatility(short_data, window_days=30)
    
    def test_calculate_daily_returns(self, vrp_classifier):
        """Test daily returns calculation accuracy."""
        # Create controlled price data
        prices = [
            Decimal('100.0'),
            Decimal('102.0'),  # +2% return
            Decimal('100.98'),  # -1% return  
            Decimal('103.0'),   # +2.02% return
            Decimal('103.0')    # 0% return
        ]
        
        market_data = []
        for i, price in enumerate(prices):
            point = MarketDataPoint(
                date=date(2023, 1, 1) + timedelta(days=i),
                spy_open=price,
                spy_high=price * Decimal('1.01'),
                spy_low=price * Decimal('0.99'),
                spy_close=price,
                spy_volume=100000000,
                vix_close=Decimal('20.0')
            )
            market_data.append(point)
        
        returns = vrp_classifier.calculate_daily_returns(market_data)
        
        # Check calculated returns
        assert len(returns) == 4  # n-1 returns
        assert abs(returns[0] - 0.02) < 1e-10    # 2% return
        assert abs(returns[1] - (-0.01)) < 1e-10  # -1% return (approximately)
        assert abs(returns[2] - 0.02) < 1e-5      # ~2% return
        assert abs(returns[3] - 0.0) < 1e-10      # 0% return
    
    def test_annualized_volatility_calculation(self, vrp_classifier):
        """Test annualized volatility calculation."""
        # Known daily returns for testing
        daily_returns = [0.01, -0.015, 0.02, -0.005, 0.008, -0.012, 0.018]
        
        annualized_vol = vrp_classifier.annualized_volatility_calculation(daily_returns)
        
        # Calculate expected volatility
        import numpy as np
        daily_vol = np.std(daily_returns, ddof=1)
        expected_vol = daily_vol * np.sqrt(252)
        
        assert abs(float(annualized_vol) - expected_vol) < 1e-10
        assert annualized_vol > 0
        assert annualized_vol < Decimal('2.0')  # Reasonable upper bound
    
    def test_extreme_volatility_handling(self, vrp_classifier):
        """Test handling of extreme volatility scenarios."""
        # Extreme crash scenario
        extreme_returns = [-0.20, -0.15, -0.10, 0.25, 0.15, -0.08, 0.12]
        
        annualized_vol = vrp_classifier.annualized_volatility_calculation(extreme_returns)
        
        # Should handle extreme values without error
        assert annualized_vol > 0
        assert annualized_vol > Decimal('1.0')  # Should be very high volatility
        
        # Very low volatility scenario
        low_vol_returns = [0.0001, -0.0002, 0.0001, 0.0000, -0.0001, 0.0002, -0.0001]
        
        low_annualized_vol = vrp_classifier.annualized_volatility_calculation(low_vol_returns)
        
        assert low_annualized_vol > 0
        assert low_annualized_vol < Decimal('0.1')  # Should be very low volatility
    
    def test_nan_inf_handling_in_returns(self, vrp_classifier):
        """Test handling of NaN and Inf values in return calculations."""
        # Create data with potential for NaN/Inf
        problematic_data = [
            MarketDataPoint(
                date=date(2023, 1, 1),
                spy_open=Decimal('0.01'),  # Very small price
                spy_high=Decimal('0.02'),
                spy_low=Decimal('0.005'),
                spy_close=Decimal('0.01'),
                spy_volume=100000000,
                vix_close=Decimal('20.0')
            ),
            MarketDataPoint(
                date=date(2023, 1, 2),
                spy_open=Decimal('100.0'),  # Large price jump
                spy_high=Decimal('105.0'),
                spy_low=Decimal('95.0'),
                spy_close=Decimal('100.0'),
                spy_volume=100000000,
                vix_close=Decimal('20.0')
            )
        ]
        
        returns = vrp_classifier.calculate_daily_returns(problematic_data)
        
        # Should handle the extreme jump gracefully
        assert len(returns) == 1
        assert not np.isnan(returns[0])
        assert not np.isinf(returns[0])
        assert returns[0] > 0  # Massive positive return
    
    def test_volatility_metrics_validation(self, vrp_classifier, sample_market_data):
        """Test validation of calculated volatility metrics."""
        result = vrp_classifier.calculate_realized_volatility(sample_market_data, window_days=30)
        
        for metrics in result:
            # All volatilities should be positive
            assert metrics.realized_vol_30d > 0
            assert metrics.implied_vol > 0
            assert metrics.vrp > 0
            
            # VRP should be reasonable (allow for extreme values in test data)
            assert Decimal('0.1') <= metrics.vrp <= Decimal('500.0')
            
            # Implied volatility should be VIX/100
            corresponding_data = next(d for d in sample_market_data if d.date == metrics.date)
            expected_iv = corresponding_data.vix_close / 100
            assert abs(metrics.implied_vol - expected_iv) < Decimal('0.0001')
            
            # VRP should equal implied_vol / realized_vol
            expected_vrp = metrics.implied_vol / metrics.realized_vol_30d
            assert abs(metrics.vrp - expected_vrp) < Decimal('0.0001')
    
    def test_custom_threshold_configuration(self, vrp_classifier):
        """Test VRP classification with quantile-based adaptive thresholds."""
        # Reset history for clean test
        vrp_classifier.reset_history()
        
        # Feed historical data to build quantile boundaries
        historical_vrp_values = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7] * 5  # 50 values
        
        # Feed values to build history (need at least 30 for quantile classification)
        for vrp_value in historical_vrp_values:
            vrp_classifier.classify_vrp_state(vrp_value)
        
        # Now test classification - should use quantile-based boundaries
        # Test edge case values
        low_value = vrp_classifier.classify_vrp_state(Decimal('0.7'))  # Should be EXTREME_LOW
        high_value = vrp_classifier.classify_vrp_state(Decimal('2.0'))  # Should be EXTREME_HIGH
        
        assert low_value in [VRPState.EXTREME_LOW, VRPState.FAIR_VALUE]  # Depends on quantiles
        assert high_value in [VRPState.EXTREME_HIGH, VRPState.ELEVATED_PREMIUM]  # Depends on quantiles
        
        # Verify we can get boundaries
        boundaries = vrp_classifier.get_current_boundaries()
        assert boundaries is not None
        assert len(boundaries) == 4  # 4 quantile boundaries
    
    def test_state_transition_edge_cases(self, vrp_classifier):
        """Test state transitions at boundaries."""
        # Values very close to thresholds
        boundary_values = [
            (Decimal('0.8999999'), VRPState.UNDERPRICED),
            (Decimal('0.9000000'), VRPState.FAIR_VALUE),
            (Decimal('0.9000001'), VRPState.FAIR_VALUE),
            (Decimal('1.0999999'), VRPState.FAIR_VALUE),
            (Decimal('1.1000000'), VRPState.NORMAL_PREMIUM),
            (Decimal('1.1000001'), VRPState.NORMAL_PREMIUM),
        ]
        
        for vrp_value, expected_state in boundary_values:
            actual_state = vrp_classifier.classify_vrp_state(vrp_value)
            assert actual_state == expected_state, f"VRP {vrp_value} should be {expected_state}, got {actual_state}"
    
    @patch('src.services.vrp_classifier.logger')
    def test_logging_unusual_values(self, mock_logger, vrp_classifier):
        """Test that unusual VRP values trigger appropriate logging."""
        # Extremely high VRP (should log warning)
        vrp_classifier.calculate_vrp(Decimal('80.0'), Decimal('0.05'))  # VRP = 16.0
        mock_logger.warning.assert_called()
        
        # Reset mock
        mock_logger.reset_mock()
        
        # Extremely low VRP (should log warning)  
        vrp_classifier.calculate_vrp(Decimal('5.0'), Decimal('0.80'))  # VRP = 0.0625
        mock_logger.warning.assert_called()
    
    def test_rolling_window_calculation(self, vrp_classifier, sample_market_data):
        """Test rolling window volatility calculation."""
        window_sizes = [20, 30, 45]
        
        for window_size in window_sizes:
            if len(sample_market_data) >= window_size:
                result = vrp_classifier.calculate_realized_volatility(sample_market_data, window_days=window_size)
                
                expected_length = len(sample_market_data) - window_size
                assert len(result) == expected_length
                
                # Check that dates are correct
                first_result_date = sample_market_data[window_size].date
                assert result[0].date == first_result_date
    
    def test_performance_large_dataset(self, vrp_classifier):
        """Test performance with large dataset."""
        # Create large dataset (1000 days)
        large_dataset = []
        base_date = date(2020, 1, 1)
        base_price = Decimal('300.0')
        base_vix = Decimal('18.0')
        
        np.random.seed(42)
        for i in range(1000):
            daily_return = np.random.normal(0.0005, 0.016)
            vix_change = np.random.normal(0.0, 0.12)
            
            base_price = base_price * (1 + Decimal(str(daily_return)))
            base_vix = max(Decimal('8.0'), base_vix * (1 + Decimal(str(vix_change))))
            
            point = MarketDataPoint(
                date=base_date + timedelta(days=i),
                spy_open=base_price * Decimal('0.999'),
                spy_high=base_price * Decimal('1.01'),
                spy_low=base_price * Decimal('0.99'),
                spy_close=base_price,
                spy_volume=100000000,
                vix_close=base_vix
            )
            large_dataset.append(point)
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        
        result = vrp_classifier.calculate_realized_volatility(large_dataset, window_days=30)
        
        end_time = time.time()
        
        # Should complete in less than 2 seconds
        assert end_time - start_time < 2.0
        assert len(result) == len(large_dataset) - 30
        assert all(r.vrp > 0 for r in result)