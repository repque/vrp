"""
Unit tests for temporal validation utilities.

This module tests the forward-looking bias prevention mechanisms
to ensure trading systems maintain temporal integrity.
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal

from src.utils.temporal_validation import (
    validate_no_forward_looking,
    validate_trading_decision_temporal_integrity,
    create_temporal_safe_window,
    ForwardLookingError
)
from src.models.data_models import MarketData, VolatilityData, VRPState


class TestTemporalValidation:
    """Test suite for temporal validation utilities."""
    
    def test_validate_no_forward_looking_passes_with_historical_data(self):
        """Test validation passes with only historical data."""
        decision_date = date(2023, 6, 15)
        
        # Create historical data
        historical_data = [
            MarketData(
                date=date(2023, 6, 14),
                open=Decimal('400.0'),
                high=Decimal('405.0'),
                low=Decimal('395.0'),
                close=Decimal('402.0'),
                volume=100000000,
                iv=Decimal('20.0')
            ),
            MarketData(
                date=date(2023, 6, 15),  # Same day is OK
                open=Decimal('402.0'),
                high=Decimal('407.0'),
                low=Decimal('397.0'),
                close=Decimal('405.0'),
                volume=100000000,
                iv=Decimal('21.0')
            )
        ]
        
        # Should pass validation
        result = validate_no_forward_looking(decision_date, historical_data)
        assert result is True
    
    def test_validate_no_forward_looking_fails_with_future_data(self):
        """Test validation fails with future data."""
        decision_date = date(2023, 6, 15)
        
        # Create data with future dates
        mixed_data = [
            MarketData(
                date=date(2023, 6, 14),  # Historical - OK
                open=Decimal('400.0'),
                high=Decimal('405.0'),
                low=Decimal('395.0'),
                close=Decimal('402.0'),
                volume=100000000,
                iv=Decimal('20.0')
            ),
            MarketData(
                date=date(2023, 6, 16),  # Future - NOT OK
                open=Decimal('402.0'),
                high=Decimal('407.0'),
                low=Decimal('397.0'),
                close=Decimal('405.0'),
                volume=100000000,
                iv=Decimal('21.0')
            )
        ]
        
        # Should raise ForwardLookingError
        with pytest.raises(ForwardLookingError) as exc_info:
            validate_no_forward_looking(decision_date, mixed_data)
        
        assert "Forward-looking bias detected" in str(exc_info.value)
        assert "Item date 2023-06-16 > Decision date 2023-06-15" in str(exc_info.value)
    
    def test_validate_trading_decision_temporal_integrity(self):
        """Test comprehensive trading decision validation."""
        decision_date = date(2023, 6, 15)
        
        # Create valid historical market data
        market_data = [
            MarketData(
                date=date(2023, 6, 14),
                open=Decimal('400.0'),
                high=Decimal('405.0'),
                low=Decimal('395.0'),
                close=Decimal('402.0'),
                volume=100000000,
                iv=Decimal('20.0')
            )
        ]
        
        # Create valid historical volatility data
        volatility_data = [
            VolatilityData(
                date=date(2023, 6, 14),
                daily_return=Decimal('0.015'),
                realized_vol_30d=Decimal('0.20'),
                implied_vol=Decimal('0.25'),
                vrp=Decimal('1.25'),
                vrp_state=VRPState.NORMAL_PREMIUM
            )
        ]
        
        # Should pass validation
        result = validate_trading_decision_temporal_integrity(
            decision_date, market_data, volatility_data
        )
        assert result is True
    
    def test_create_temporal_safe_window(self):
        """Test creation of temporal-safe data windows."""
        current_date = date(2023, 6, 15)
        
        # Create mixed data (historical and future)
        all_data = [
            MarketData(
                date=date(2023, 6, 12),
                open=Decimal('395.0'),
                high=Decimal('400.0'),
                low=Decimal('390.0'),
                close=Decimal('398.0'),
                volume=100000000,
                iv=Decimal('19.0')
            ),
            MarketData(
                date=date(2023, 6, 14),
                open=Decimal('400.0'),
                high=Decimal('405.0'),
                low=Decimal('395.0'),
                close=Decimal('402.0'),
                volume=100000000,
                iv=Decimal('20.0')
            ),
            MarketData(
                date=date(2023, 6, 15),  # Current day
                open=Decimal('402.0'),
                high=Decimal('407.0'),
                low=Decimal('397.0'),
                close=Decimal('405.0'),
                volume=100000000,
                iv=Decimal('21.0')
            ),
            MarketData(
                date=date(2023, 6, 16),  # Future - should be excluded
                open=Decimal('405.0'),
                high=Decimal('410.0'),
                low=Decimal('400.0'),
                close=Decimal('408.0'),
                volume=100000000,
                iv=Decimal('22.0')
            )
        ]
        
        # Create safe window
        safe_window = create_temporal_safe_window(
            all_data, current_date, window_size=2
        )
        
        # Should only include current and previous day
        assert len(safe_window) == 2
        assert safe_window[0].date == date(2023, 6, 14)
        assert safe_window[1].date == date(2023, 6, 15)
        
        # Future data should be excluded
        for item in safe_window:
            assert item.date <= current_date
    
    def test_temporal_safe_window_with_insufficient_data(self):
        """Test temporal safe window when less data than requested."""
        current_date = date(2023, 6, 15)
        
        # Only one historical data point
        limited_data = [
            MarketData(
                date=date(2023, 6, 15),
                open=Decimal('400.0'),
                high=Decimal('405.0'),
                low=Decimal('395.0'),
                close=Decimal('402.0'),
                volume=100000000,
                iv=Decimal('20.0')
            )
        ]
        
        # Request window of 5, but only 1 available
        safe_window = create_temporal_safe_window(
            limited_data, current_date, window_size=5
        )
        
        # Should return all available data (1 item)
        assert len(safe_window) == 1
        assert safe_window[0].date == date(2023, 6, 15)
    
    def test_backtest_temporal_integrity_simulation(self):
        """Test that backtest-style operations maintain temporal integrity."""
        # Simulate what happens in backtest loop
        all_data = []
        base_date = date(2023, 1, 1)
        
        # Create 10 days of data
        for i in range(10):
            data_point = MarketData(
                date=base_date + timedelta(days=i),
                open=Decimal('400.0') + Decimal(str(i)),
                high=Decimal('405.0') + Decimal(str(i)),
                low=Decimal('395.0') + Decimal(str(i)),
                close=Decimal('402.0') + Decimal(str(i)),
                volume=100000000,
                iv=Decimal('20.0') + Decimal(str(i * 0.1))
            )
            all_data.append(data_point)
        
        # Simulate backtest loop - decision on day 5
        decision_day_index = 5
        decision_date = all_data[decision_day_index].date
        
        # Create historical window (only up to current day)
        historical_data = all_data[:decision_day_index + 1]
        
        # Validate temporal integrity
        result = validate_no_forward_looking(decision_date, historical_data)
        assert result is True
        
        # Verify no future data
        for data_point in historical_data:
            assert data_point.date <= decision_date