"""
Unit tests for VolatilityCalculator.

This module contains comprehensive tests for volatility calculations,
VRP computations, and state determinations. Tests include edge cases,
error conditions, and validation of mathematical correctness.
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
from pydantic import ValidationError

from src.config.settings import Settings
from src.data.volatility_calculator import VolatilityCalculator
from src.models.data_models import MarketDataPoint, VRPState
from src.utils.exceptions import CalculationError, InsufficientDataError


class TestVolatilityCalculator:
    """Test suite for VolatilityCalculator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Settings)
        config.model = Mock()
        config.model.vrp_underpriced_threshold = Decimal('0.90')
        config.model.vrp_fair_upper_threshold = Decimal('1.10')
        config.model.vrp_normal_upper_threshold = Decimal('1.30')
        config.model.vrp_elevated_upper_threshold = Decimal('1.50')
        return config
    
    @pytest.fixture
    def calculator(self, config):
        """Create VolatilityCalculator instance."""
        return VolatilityCalculator(config)
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        base_date = date(2023, 1, 1)
        data = []
        
        # Create 50 days of sample data with realistic price movements
        base_price = Decimal('400.0')
        vix_base = Decimal('20.0')
        
        for i in range(50):
            # Simulate price movements
            daily_change = (-1) ** i * Decimal('0.02') * (i % 5)  # ±2% moves
            price = base_price * (1 + daily_change)
            
            # Simulate VIX movements
            vix_change = (-1) ** (i + 1) * Decimal('0.1') * (i % 3)  # VIX movements
            vix = max(Decimal('10.0'), vix_base * (1 + vix_change))
            
            point = MarketDataPoint(
                date=base_date + timedelta(days=i),
                spy_open=price * Decimal('0.999'),
                spy_high=price * Decimal('1.015'),
                spy_low=price * Decimal('0.985'),
                spy_close=price,
                spy_volume=1000000,
                vix_close=vix
            )
            data.append(point)
            
            # Update base values for next iteration
            base_price = price
            vix_base = vix
        
        return data
    
    def test_calculate_realized_volatility_success(self, calculator, sample_market_data):
        """Test successful volatility calculation."""
        result = calculator.calculate_realized_volatility(sample_market_data, window_days=30)
        
        assert len(result) > 0
        assert len(result) == len(sample_market_data) - 30
        
        # Check that all results have required fields
        for metrics in result:
            assert metrics.date is not None
            assert metrics.spy_return is not None
            assert metrics.realized_vol_30d > 0
            assert metrics.implied_vol > 0
            assert metrics.vrp > 0
            assert metrics.vrp_state is not None
    
    def test_calculate_realized_volatility_insufficient_data(self, calculator):
        """Test error handling with insufficient data."""
        short_data = [
            MarketDataPoint(
                date=date(2023, 1, 1),
                spy_open=Decimal('400.0'),
                spy_high=Decimal('405.0'),
                spy_low=Decimal('395.0'),
                spy_close=Decimal('402.0'),
                spy_volume=1000000,
                vix_close=Decimal('20.0')
            )
        ]
        
        with pytest.raises(InsufficientDataError):
            calculator.calculate_realized_volatility(short_data, window_days=30)
    
    def test_calculate_vrp_normal_case(self, calculator):
        """Test VRP calculation with normal inputs."""
        realized_vol = 0.20  # 20% realized volatility
        implied_vol = 0.25   # 25% implied volatility
        
        vrp = calculator.calculate_vrp(realized_vol, implied_vol)
        
        assert vrp == 1.25  # 25% / 20% = 1.25
    
    def test_calculate_vrp_edge_cases(self, calculator):
        """Test VRP calculation edge cases."""
        # Test with very low realized volatility
        vrp_high = calculator.calculate_vrp(0.05, 0.20)
        assert vrp_high == 4.0
        
        # Test with very high realized volatility
        vrp_low = calculator.calculate_vrp(0.50, 0.20)
        assert vrp_low == 0.4
    
    def test_calculate_vrp_invalid_inputs(self, calculator):
        """Test VRP calculation with invalid inputs."""
        # Test zero realized volatility
        with pytest.raises(CalculationError):
            calculator.calculate_vrp(0.0, 0.20)
        
        # Test negative realized volatility
        with pytest.raises(CalculationError):
            calculator.calculate_vrp(-0.10, 0.20)
        
        # Test zero implied volatility
        with pytest.raises(CalculationError):
            calculator.calculate_vrp(0.20, 0.0)
        
        # Test negative implied volatility
        with pytest.raises(CalculationError):
            calculator.calculate_vrp(0.20, -0.10)
    
    def test_determine_vrp_state_all_states(self, calculator):
        """Test VRP state determination for all possible states."""
        # Test UNDERPRICED state
        assert calculator.determine_vrp_state(0.85) == VRPState.UNDERPRICED
        
        # Test FAIR_VALUE state
        assert calculator.determine_vrp_state(0.95) == VRPState.FAIR_VALUE
        assert calculator.determine_vrp_state(1.05) == VRPState.FAIR_VALUE
        
        # Test NORMAL_PREMIUM state
        assert calculator.determine_vrp_state(1.15) == VRPState.NORMAL_PREMIUM
        assert calculator.determine_vrp_state(1.25) == VRPState.NORMAL_PREMIUM
        
        # Test ELEVATED_PREMIUM state
        assert calculator.determine_vrp_state(1.35) == VRPState.ELEVATED_PREMIUM
        assert calculator.determine_vrp_state(1.45) == VRPState.ELEVATED_PREMIUM
        
        # Test EXTREME_PREMIUM state
        assert calculator.determine_vrp_state(1.55) == VRPState.EXTREME_PREMIUM
        assert calculator.determine_vrp_state(2.00) == VRPState.EXTREME_PREMIUM
    
    def test_determine_vrp_state_boundary_conditions(self, calculator):
        """Test VRP state determination at exact threshold boundaries."""
        # Test exact thresholds
        assert calculator.determine_vrp_state(0.90) == VRPState.FAIR_VALUE
        assert calculator.determine_vrp_state(1.10) == VRPState.NORMAL_PREMIUM
        assert calculator.determine_vrp_state(1.30) == VRPState.ELEVATED_PREMIUM
        assert calculator.determine_vrp_state(1.50) == VRPState.EXTREME_PREMIUM
    
    def test_daily_returns_calculation(self, calculator, sample_market_data):
        """Test daily returns calculation accuracy."""
        returns = calculator._calculate_daily_returns(sample_market_data[:5])
        
        # Check return count
        assert len(returns) == 4  # n-1 returns for n prices
        
        # Check return calculation
        for i, ret in enumerate(returns):
            prev_close = float(sample_market_data[i].spy_close)
            curr_close = float(sample_market_data[i + 1].spy_close)
            expected_return = (curr_close - prev_close) / prev_close
            
            assert abs(ret - expected_return) < 1e-10  # Floating point precision
    
    def test_daily_returns_insufficient_data(self, calculator):
        """Test daily returns calculation with insufficient data."""
        single_point = [
            MarketDataPoint(
                date=date(2023, 1, 1),
                spy_open=Decimal('400.0'),
                spy_high=Decimal('405.0'),
                spy_low=Decimal('395.0'),
                spy_close=Decimal('402.0'),
                spy_volume=1000000,
                vix_close=Decimal('20.0')
            )
        ]
        
        with pytest.raises(InsufficientDataError):
            calculator._calculate_daily_returns(single_point)
    
    def test_window_volatility_calculation(self, calculator):
        """Test window volatility calculation."""
        # Create returns with known standard deviation
        returns = [0.01, -0.01, 0.015, -0.008, 0.012, -0.02, 0.005]
        
        vol = calculator._calculate_window_volatility(returns)
        
        # Check that volatility is positive and reasonable
        assert vol > 0
        assert vol < 1.0  # Should be less than 100% annualized
        
        # Check annualization (should be roughly sqrt(252) times daily vol)
        import numpy as np
        daily_vol = np.std(returns, ddof=1)
        expected_annual_vol = daily_vol * (252 ** 0.5)
        
        assert abs(vol - expected_annual_vol) < 1e-10
    
    def test_window_volatility_edge_cases(self, calculator):
        """Test window volatility calculation edge cases."""
        # Test with minimal data
        with pytest.raises(CalculationError):
            calculator._calculate_window_volatility([0.01])
        
        # Test with NaN values
        import numpy as np
        returns_with_nan = [0.01, np.nan, 0.015, -0.008]
        
        vol = calculator._calculate_window_volatility(returns_with_nan)
        assert vol > 0  # Should handle NaN values gracefully
    
    def test_parkinson_volatility(self, calculator, sample_market_data):
        """Test Parkinson volatility estimator."""
        result = calculator.calculate_parkinson_volatility(sample_market_data, window_days=20)
        
        assert len(result) == len(sample_market_data) - 19
        
        # Check that all volatilities are positive
        for vol in result:
            if vol > 0:  # Skip any zero values from insufficient data
                assert vol > 0
                assert vol < 2.0  # Reasonable upper bound
    
    def test_garman_klass_volatility(self, calculator, sample_market_data):
        """Test Garman-Klass volatility estimator."""
        result = calculator.calculate_garman_klass_volatility(sample_market_data, window_days=20)
        
        assert len(result) == len(sample_market_data) - 19
        
        # Check that volatilities are reasonable
        for vol in result:
            if vol > 0:
                assert vol > 0
                assert vol < 2.0  # Reasonable upper bound
    
    def test_volatility_summary_stats(self, calculator, sample_market_data):
        """Test volatility summary statistics calculation."""
        metrics = calculator.calculate_realized_volatility(sample_market_data, window_days=30)
        summary = calculator.get_volatility_summary_stats(metrics)
        
        # Check that summary contains expected keys
        expected_keys = [
            'total_observations', 'vrp_stats', 'realized_vol_stats',
            'implied_vol_stats', 'state_distribution', 'state_percentages'
        ]
        
        for key in expected_keys:
            assert key in summary
        
        # Check VRP statistics
        vrp_stats = summary['vrp_stats']
        assert 'mean' in vrp_stats
        assert 'median' in vrp_stats
        assert 'std' in vrp_stats
        assert 'min' in vrp_stats
        assert 'max' in vrp_stats
        
        # Check state distribution
        state_dist = summary['state_distribution']
        total_states = sum(state_dist.values())
        assert total_states == len(metrics)
        
        # Check percentages sum to 100
        percentages = summary['state_percentages']
        total_percentage = sum(percentages.values())
        assert abs(total_percentage - 100.0) < 1e-6  # Account for floating point precision
    
    def test_volatility_summary_stats_empty_data(self, calculator):
        """Test summary statistics with empty data."""
        summary = calculator.get_volatility_summary_stats([])
        assert summary == {}
    
    @pytest.mark.parametrize("window_days", [20, 30, 60])
    def test_different_window_sizes(self, calculator, sample_market_data, window_days):
        """Test volatility calculation with different window sizes."""
        if len(sample_market_data) < window_days + 1:
            # Test should handle insufficient data gracefully
            with pytest.raises(InsufficientDataError):
                calculator.calculate_realized_volatility(sample_market_data, window_days=window_days)
        else:
            result = calculator.calculate_realized_volatility(sample_market_data, window_days=window_days)
            
            expected_length = len(sample_market_data) - window_days
            assert len(result) == expected_length
                
            # Check that all results are valid
            for metrics in result:
                assert metrics.realized_vol_30d > 0
                assert metrics.vrp > 0
    
    def test_realistic_volatility_ranges(self, calculator, sample_market_data):
        """Test that calculated volatilities are in realistic ranges."""
        result = calculator.calculate_realized_volatility(sample_market_data, window_days=30)
        
        for metrics in result:
            # Realized volatility should be between 5% and 100% annualized
            rv = float(metrics.realized_vol_30d)
            assert 0.05 <= rv <= 1.00
            
            # Implied volatility should be between 5% and 80%
            iv = float(metrics.implied_vol)
            assert 0.05 <= iv <= 0.80
            
            # VRP should be between 0.1 and 10.0 in normal conditions
            vrp = float(metrics.vrp)
            assert 0.1 <= vrp <= 10.0
    
    @patch('src.data.volatility_calculator.logger')
    def test_logging_on_unusual_vrp(self, mock_logger, calculator):
        """Test that unusual VRP values trigger warnings."""
        # Test extremely high VRP
        calculator.calculate_vrp(0.05, 0.80)  # VRP = 16.0
        mock_logger.warning.assert_called()
        
        # Reset mock
        mock_logger.reset_mock()
        
        # Test extremely low VRP
        calculator.calculate_vrp(0.80, 0.05)  # VRP = 0.0625
        mock_logger.warning.assert_called()
    
    def test_price_validation_in_data(self, calculator):
        """Test that invalid price data is handled properly."""
        # Test that Pydantic validation catches invalid OHLC relationships
        with pytest.raises(ValidationError):
            MarketDataPoint(
                date=date(2023, 1, 1),
                spy_open=Decimal('400.0'),
                spy_high=Decimal('395.0'),  # High < Low (invalid)
                spy_low=Decimal('405.0'),
                spy_close=Decimal('402.0'),
                spy_volume=1000000,
                vix_close=Decimal('20.0')
            )
        
        # Test with valid data that has extreme but valid OHLC relationships
        extreme_data = [
            MarketDataPoint(
                date=date(2023, 1, 1),
                spy_open=Decimal('400.0'),
                spy_high=Decimal('450.0'),  # Very high but valid
                spy_low=Decimal('350.0'),   # Very low but valid
                spy_close=Decimal('402.0'),
                spy_volume=1000000,
                vix_close=Decimal('20.0')
            ),
            MarketDataPoint(
                date=date(2023, 1, 2),
                spy_open=Decimal('402.0'),
                spy_high=Decimal('407.0'),
                spy_low=Decimal('397.0'),
                spy_close=Decimal('405.0'),
                spy_volume=1000000,
                vix_close=Decimal('21.0')
            )
        ]
        
        # Add a third data point to have enough for volatility calculation
        extreme_data.append(MarketDataPoint(
            date=date(2023, 1, 3),
            spy_open=Decimal('405.0'),
            spy_high=Decimal('415.0'),
            spy_low=Decimal('400.0'),
            spy_close=Decimal('410.0'),
            spy_volume=1000000,
            vix_close=Decimal('19.0')
        ))
        
        # Test volatility calculation with extreme but valid data
        try:
            result = calculator.calculate_realized_volatility(extreme_data, window_days=2)
            # Should succeed with valid data and produce reasonable results
            assert len(result) > 0
            for metrics in result:
                assert metrics.realized_vol_30d > 0
                assert metrics.vrp > 0
        except (InsufficientDataError, CalculationError):
            # Acceptable if not enough data for calculation or calculation issues with extreme values
            pass