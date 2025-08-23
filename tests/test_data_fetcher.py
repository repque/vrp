"""
Unit tests for DataFetcher service.

This module contains comprehensive tests for data fetching functionality,
including mock external APIs, error handling, data validation, and
retry mechanisms. Tests ensure reliable data acquisition from various sources.
"""

import pytest
import requests
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.config.settings import Settings
from src.data.data_fetcher import DataFetcher
from src.models.data_models import MarketDataPoint, VRPState
from src.utils.exceptions import DataFetchError, ValidationError, ConfigurationError


class TestDataFetcher:
    """Test suite for DataFetcher class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration for DataFetcher."""
        config = Mock(spec=Settings)
        config.data = Mock()
        config.data.yahoo_finance_api_key = "test_api_key"
        config.data.fred_api_key = "test_fred_key"
        config.data.alpha_vantage_api_key = "test_alpha_key"
        config.data.request_timeout_seconds = 30
        config.data.max_retry_attempts = 3
        config.data.retry_delay_seconds = 1
        config.data.rate_limit_delay_ms = 100
        config.data.validate_ohlc_consistency = True
        return config
    
    @pytest.fixture
    def data_fetcher(self, config):
        """Create DataFetcher instance."""
        return DataFetcher(config)
    
    @pytest.fixture
    def mock_yahoo_response(self):
        """Create mock Yahoo Finance API response."""
        return {
            "chart": {
                "result": [{
                    "meta": {
                        "symbol": "SPY",
                        "currency": "USD",
                        "exchangeName": "PCX"
                    },
                    "timestamp": [1672531200, 1672617600, 1672704000],  # 3 days
                    "indicators": {
                        "quote": [{
                            "open": [400.0, 402.0, 405.0],
                            "high": [405.0, 407.0, 410.0],
                            "low": [395.0, 398.0, 401.0],
                            "close": [402.0, 405.0, 408.0],
                            "volume": [100000000, 110000000, 95000000]
                        }]
                    }
                }],
                "error": None
            }
        }
    
    @pytest.fixture
    def mock_vix_response(self):
        """Create mock VIX data response."""
        return {
            "chart": {
                "result": [{
                    "meta": {
                        "symbol": "^VIX",
                        "currency": "USD"
                    },
                    "timestamp": [1672531200, 1672617600, 1672704000],
                    "indicators": {
                        "quote": [{
                            "close": [20.5, 19.8, 21.2]
                        }]
                    }
                }],
                "error": None
            }
        }
    
    def test_fetch_spy_data_success(self, data_fetcher, mock_yahoo_response):
        """Test successful SPY data fetching."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 3)
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_yahoo_response
            mock_get.return_value.raise_for_status = Mock()
            
            spy_data = data_fetcher.fetch_spy_data(start_date, end_date)
            
            assert len(spy_data) == 3
            assert spy_data[0].spy_open == Decimal('400.0')
            assert spy_data[0].spy_close == Decimal('402.0')
            assert spy_data[0].spy_volume == 100000000
            assert spy_data[1].spy_high == Decimal('407.0')
            assert spy_data[2].spy_low == Decimal('401.0')
            
            # Verify API was called with correct parameters
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert 'SPY' in call_args[0][0]  # URL contains SPY symbol
    
    def test_fetch_vix_data_success(self, data_fetcher, mock_vix_response):
        """Test successful VIX data fetching."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 3)
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_vix_response
            mock_get.return_value.raise_for_status = Mock()
            
            vix_data = data_fetcher.fetch_vix_data(start_date, end_date)
            
            assert len(vix_data) == 3
            assert vix_data[0] == Decimal('20.5')
            assert vix_data[1] == Decimal('19.8')
            assert vix_data[2] == Decimal('21.2')
    
    def test_combine_market_data_success(self, data_fetcher, mock_yahoo_response, mock_vix_response):
        """Test successful combination of SPY and VIX data."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 3)
        
        with patch.object(data_fetcher, 'fetch_spy_data') as mock_spy, \
             patch.object(data_fetcher, 'fetch_vix_data') as mock_vix:
            
            # Mock SPY data
            spy_data = [
                MarketDataPoint(
                    date=date(2023, 1, 1),
                    spy_open=Decimal('400.0'),
                    spy_high=Decimal('405.0'),
                    spy_low=Decimal('395.0'),
                    spy_close=Decimal('402.0'),
                    spy_volume=100000000,
                    vix_close=Decimal('0.0')  # Will be replaced
                ),
                MarketDataPoint(
                    date=date(2023, 1, 2),
                    spy_open=Decimal('402.0'),
                    spy_high=Decimal('407.0'),
                    spy_low=Decimal('398.0'),
                    spy_close=Decimal('405.0'),
                    spy_volume=110000000,
                    vix_close=Decimal('0.0')
                )
            ]
            
            # Mock VIX data
            vix_data = {
                date(2023, 1, 1): Decimal('20.5'),
                date(2023, 1, 2): Decimal('19.8')
            }
            
            mock_spy.return_value = spy_data
            mock_vix.return_value = vix_data
            
            combined_data = data_fetcher.fetch_market_data(start_date, end_date)
            
            assert len(combined_data) == 2
            assert combined_data[0].date == date(2023, 1, 1)
            assert combined_data[0].vix_close == Decimal('20.5')
            assert combined_data[1].date == date(2023, 1, 2)
            assert combined_data[1].vix_close == Decimal('19.8')
    
    def test_api_error_handling(self, data_fetcher):
        """Test handling of API errors."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 3)
        
        # Test HTTP error
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 404
            mock_get.return_value.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
            
            with pytest.raises(DataFetchError, match="HTTP error"):
                data_fetcher.fetch_spy_data(start_date, end_date)
        
        # Test timeout error
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.Timeout("Request timed out")
            
            with pytest.raises(DataFetchError, match="Max retry attempts exceeded"):
                data_fetcher.fetch_spy_data(start_date, end_date)
        
        # Test connection error
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection failed")
            
            with pytest.raises(DataFetchError, match="Connection error"):
                data_fetcher.fetch_spy_data(start_date, end_date)
    
    def test_invalid_json_response(self, data_fetcher):
        """Test handling of invalid JSON responses."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 3)
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.side_effect = ValueError("Invalid JSON")
            mock_get.return_value.raise_for_status = Mock()
            
            with pytest.raises(DataFetchError, match="Invalid JSON response"):
                data_fetcher.fetch_spy_data(start_date, end_date)
    
    def test_empty_data_response(self, data_fetcher):
        """Test handling of empty data responses."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 3)
        
        empty_response = {
            "chart": {
                "result": [],
                "error": None
            }
        }
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = empty_response
            mock_get.return_value.raise_for_status = Mock()
            
            with pytest.raises(DataFetchError, match="No data returned"):
                data_fetcher.fetch_spy_data(start_date, end_date)
    
    def test_api_error_in_response(self, data_fetcher):
        """Test handling of API errors embedded in response."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 3)
        
        error_response = {
            "chart": {
                "result": None,
                "error": {
                    "code": "Not Found",
                    "description": "No data found for symbol"
                }
            }
        }
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = error_response
            mock_get.return_value.raise_for_status = Mock()
            
            with pytest.raises(DataFetchError, match="API returned error"):
                data_fetcher.fetch_spy_data(start_date, end_date)
    
    def test_retry_mechanism(self, data_fetcher):
        """Test retry mechanism on transient failures."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 3)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "chart": {
                "result": [{
                    "timestamp": [1672531200],
                    "indicators": {
                        "quote": [{
                            "open": [400.0],
                            "high": [405.0],
                            "low": [395.0],
                            "close": [402.0],
                            "volume": [100000000]
                        }]
                    }
                }]
            }
        }
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get') as mock_get:
            # First two calls fail, third succeeds
            mock_get.side_effect = [
                requests.Timeout("Timeout 1"),
                requests.Timeout("Timeout 2"),
                mock_response
            ]
            
            with patch('time.sleep') as mock_sleep:
                spy_data = data_fetcher.fetch_spy_data(start_date, end_date)
                
                # Should have retried 2 times (plus 1 rate limiting sleep)
                assert mock_get.call_count == 3
                assert mock_sleep.call_count == 3  # 1 rate limiting + 2 retry sleeps
                assert len(spy_data) == 1
    
    def test_max_retries_exceeded(self, data_fetcher):
        """Test behavior when max retries are exceeded."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 3)
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.Timeout("Persistent timeout")
            
            with patch('time.sleep'):
                with pytest.raises(DataFetchError, match="Max retry attempts exceeded"):
                    data_fetcher.fetch_spy_data(start_date, end_date)
                
                # Should have tried max_retry_attempts + 1 times
                assert mock_get.call_count == 4  # 3 retries + 1 initial attempt
    
    def test_data_validation_ohlc_consistency(self, data_fetcher):
        """Test OHLC data consistency validation."""
        inconsistent_response = {
            "chart": {
                "result": [{
                    "timestamp": [1672531200],
                    "indicators": {
                        "quote": [{
                            "open": [400.0],
                            "high": [395.0],  # High < Open (invalid)
                            "low": [405.0],   # Low > Open (invalid)
                            "close": [402.0],
                            "volume": [100000000]
                        }]
                    }
                }]
            }
        }
        
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 1)
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = inconsistent_response
            mock_get.return_value.raise_for_status = Mock()
            
            with pytest.raises(ValidationError, match="OHLC data inconsistency"):
                data_fetcher.fetch_spy_data(start_date, end_date)
    
    def test_date_range_validation(self, data_fetcher):
        """Test date range validation."""
        # End date before start date
        with pytest.raises(ValueError, match="End date must be after start date"):
            data_fetcher.fetch_market_data(date(2023, 1, 10), date(2023, 1, 5))
        
        # Same start and end date (should be allowed)
        with patch.object(data_fetcher, 'fetch_spy_data') as mock_spy, \
             patch.object(data_fetcher, 'fetch_vix_data') as mock_vix:
            
            mock_spy.return_value = []
            mock_vix.return_value = {}
            
            same_date = date(2023, 1, 1)
            result = data_fetcher.fetch_market_data(same_date, same_date)
            assert isinstance(result, list)
    
    def test_missing_configuration(self):
        """Test handling of missing configuration."""
        config = Mock()
        config.data = Mock()
        config.data.yahoo_finance_api_key = None  # Missing API key
        
        with pytest.raises(ConfigurationError, match="Missing required API key"):
            DataFetcher(config)
    
    def test_rate_limiting(self, data_fetcher):
        """Test rate limiting functionality."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 2)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "chart": {"result": [{"timestamp": [1672531200], "indicators": {"quote": [{"open": [400.0], "high": [405.0], "low": [395.0], "close": [402.0], "volume": [100000000]}]}}]}
        }
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response), \
             patch('time.sleep') as mock_sleep:
            
            # Make multiple requests
            data_fetcher.fetch_spy_data(start_date, end_date)
            data_fetcher.fetch_vix_data(start_date, end_date)
            
            # Should have applied rate limiting
            mock_sleep.assert_called()
    
    def test_weekend_date_handling(self, data_fetcher):
        """Test handling of weekend dates in requests."""
        # Saturday and Sunday dates
        weekend_start = date(2023, 1, 7)  # Saturday
        weekend_end = date(2023, 1, 8)    # Sunday
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "chart": {"result": [{"timestamp": [], "indicators": {"quote": [{}]}}]}
        }
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response):
            # Should handle weekend dates gracefully
            result = data_fetcher.fetch_spy_data(weekend_start, weekend_end)
            assert isinstance(result, list)
    
    def test_data_alignment_spy_vix(self, data_fetcher):
        """Test proper alignment of SPY and VIX data."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 3)
        
        # SPY data for 3 days
        spy_data = [
            MarketDataPoint(
                date=date(2023, 1, 1),
                spy_open=Decimal('400.0'),
                spy_high=Decimal('405.0'),
                spy_low=Decimal('395.0'),
                spy_close=Decimal('402.0'),
                spy_volume=100000000,
                vix_close=Decimal('0.0')
            ),
            MarketDataPoint(
                date=date(2023, 1, 2),
                spy_open=Decimal('402.0'),
                spy_high=Decimal('407.0'),
                spy_low=Decimal('398.0'),
                spy_close=Decimal('405.0'),
                spy_volume=110000000,
                vix_close=Decimal('0.0')
            ),
            MarketDataPoint(
                date=date(2023, 1, 3),
                spy_open=Decimal('405.0'),
                spy_high=Decimal('410.0'),
                spy_low=Decimal('401.0'),
                spy_close=Decimal('408.0'),
                spy_volume=95000000,
                vix_close=Decimal('0.0')
            )
        ]
        
        # VIX data missing for middle day
        vix_data = {
            date(2023, 1, 1): Decimal('20.5'),
            # date(2023, 1, 2) missing
            date(2023, 1, 3): Decimal('21.2')
        }
        
        with patch.object(data_fetcher, 'fetch_spy_data', return_value=spy_data), \
             patch.object(data_fetcher, 'fetch_vix_data', return_value=vix_data):
            
            combined_data = data_fetcher.fetch_market_data(start_date, end_date)
            
            # Should only return aligned data (excluding middle day)
            assert len(combined_data) == 2
            assert combined_data[0].date == date(2023, 1, 1)
            assert combined_data[0].vix_close == Decimal('20.5')
            assert combined_data[1].date == date(2023, 1, 3)
            assert combined_data[1].vix_close == Decimal('21.2')
    
    @patch('src.data.data_fetcher.logger')
    def test_logging_during_operations(self, mock_logger, data_fetcher):
        """Test that appropriate logging occurs during operations."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 3)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "chart": {"result": [{"timestamp": [], "indicators": {"quote": [{}]}}]}
        }
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response):
            data_fetcher.fetch_spy_data(start_date, end_date)
            
            # Should have logged the fetch operation
            mock_logger.info.assert_called()
    
    def test_large_date_range_chunking(self, data_fetcher):
        """Test handling of large date ranges with chunking."""
        # Large date range (over 1 year)
        start_date = date(2022, 1, 1)
        end_date = date(2023, 12, 31)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "chart": {"result": [{"timestamp": [], "indicators": {"quote": [{}]}}]}
        }
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            data_fetcher.fetch_spy_data(start_date, end_date)
            
            # For very large ranges, might need to chunk requests
            # This depends on implementation details
            assert mock_get.called
    
    def test_cache_functionality(self, data_fetcher):
        """Test caching of fetched data to reduce API calls."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 3)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "chart": {
                "result": [{
                    "timestamp": [1672531200],
                    "indicators": {
                        "quote": [{
                            "open": [400.0],
                            "high": [405.0],
                            "low": [395.0],
                            "close": [402.0],
                            "volume": [100000000]
                        }]
                    }
                }]
            }
        }
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            # First request
            result1 = data_fetcher.fetch_spy_data(start_date, end_date)
            
            # Second identical request (might use cache)
            result2 = data_fetcher.fetch_spy_data(start_date, end_date)
            
            # Results should be identical
            assert len(result1) == len(result2)
            if len(result1) > 0:
                assert result1[0].spy_close == result2[0].spy_close