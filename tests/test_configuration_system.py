"""
Comprehensive Test Suite for Settings Configuration System

This test suite validates the Settings configuration system that provides:
1. Nested access patterns (settings.model.field)
2. Type safety and validation with Pydantic
3. Environment variable integration
4. Serialization/deserialization

Tests verify the Settings system works correctly for all VRP trading components.
"""

import os
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, ValidationError

from src.config.settings import Settings, get_settings


class TestSettingsStructure:
    """Test that Settings provides proper nested structure access."""

    def test_settings_has_nested_model_section(self):
        """Test that Settings has a model section with nested access."""
        settings = get_settings()
        
        # Should have nested model section
        assert hasattr(settings, 'model'), "Settings must have 'model' attribute"
        assert settings.model is not None, "settings.model must not be None"
        
        # Should have required model fields
        assert hasattr(settings.model, 'vrp_thresholds')
        assert hasattr(settings.model, 'transition_window_days')
        assert hasattr(settings.model, 'realized_vol_window_days')
        assert hasattr(settings.model, 'laplace_smoothing_alpha')
        assert hasattr(settings.model, 'min_confidence_for_signal')
        assert hasattr(settings.model, 'volatility_annualization_factor')
        assert hasattr(settings.model, 'vrp_min_reasonable')
        assert hasattr(settings.model, 'vrp_max_reasonable')

    def test_settings_has_nested_trading_section(self):
        """Test that Settings has a trading section."""
        settings = get_settings()
        
        assert hasattr(settings, 'trading'), "Settings must have 'trading' attribute"
        assert settings.trading is not None, "settings.trading must not be None"
        
        # Should have required trading fields
        assert hasattr(settings.trading, 'base_position_size_pct')
        assert hasattr(settings.trading, 'max_position_size_pct')
        assert hasattr(settings.trading, 'extreme_low_confidence_threshold')
        assert hasattr(settings.trading, 'extreme_high_confidence_threshold')
        assert hasattr(settings.trading, 'transaction_cost_bps')

    def test_settings_has_nested_data_section(self):
        """Test that Settings has a data section."""
        settings = get_settings()
        
        assert hasattr(settings, 'data'), "Settings must have 'data' attribute"
        assert settings.data is not None, "settings.data must not be None"
        
        # Should have required data fields
        assert hasattr(settings.data, 'min_data_years')
        assert hasattr(settings.data, 'preferred_data_years')
        assert hasattr(settings.data, 'max_missing_days_pct')
        assert hasattr(settings.data, 'max_daily_processing_seconds')
        assert hasattr(settings.data, 'max_memory_usage_mb')

    def test_settings_has_nested_database_section(self):
        """Test that Settings has a database section."""
        settings = get_settings()
        
        assert hasattr(settings, 'database'), "Settings must have 'database' attribute"
        assert settings.database is not None, "settings.database must not be None"
        
        # Should have required database fields
        assert hasattr(settings.database, 'database_url')
        assert hasattr(settings.database, 'database_path')
        assert hasattr(settings.database, 'connection_timeout')
        assert hasattr(settings.database, 'enable_wal_mode')

    def test_settings_nested_access_types(self):
        """Test that nested access returns correct types."""
        settings = get_settings()
        
        # Model section types
        assert isinstance(settings.model.vrp_thresholds, list)
        assert isinstance(settings.model.transition_window_days, int)
        assert isinstance(settings.model.laplace_smoothing_alpha, float)
        assert isinstance(settings.model.volatility_annualization_factor, float)
        
        # Trading section types
        assert isinstance(settings.trading.base_position_size_pct, float)
        assert isinstance(settings.trading.max_position_size_pct, float)
        
        # Data section types
        assert isinstance(settings.data.min_data_years, int)
        assert isinstance(settings.data.preferred_data_years, int)


class TestSettingsValidation:
    """Test configuration validation and type safety."""

    def test_settings_validation_vrp_thresholds(self):
        """Test that VRP thresholds are properly validated."""
        settings = get_settings()
        
        # Should have exactly 4 thresholds
        assert len(settings.model.vrp_thresholds) == 4
        
        # Should be in ascending order
        thresholds = settings.model.vrp_thresholds
        assert thresholds[0] < thresholds[1] < thresholds[2] < thresholds[3]
        
        # Should be reasonable values
        for threshold in thresholds:
            assert 0.5 <= threshold <= 3.0

    def test_settings_position_sizing_validation(self):
        """Test that position sizing parameters are validated."""
        settings = get_settings()
        
        # Base position size should be less than max
        assert settings.trading.base_position_size_pct < settings.trading.max_position_size_pct
        
        # Both should be reasonable percentages
        assert 0 < settings.trading.base_position_size_pct < 0.2  # < 20%
        assert 0 < settings.trading.max_position_size_pct < 0.5   # < 50%

    def test_settings_data_requirements_validation(self):
        """Test that data requirements are validated."""
        settings = get_settings()
        
        # Min data years should be positive
        assert settings.data.min_data_years > 0
        
        # Preferred should be >= min
        assert settings.data.preferred_data_years >= settings.data.min_data_years
        
        # Missing days percentage should be reasonable
        assert 0 <= settings.data.max_missing_days_pct <= 0.1  # <= 10%


class TestSettingsDefaults:
    """Test that settings have reasonable default values."""

    def test_model_section_defaults(self):
        """Test model section default values."""
        settings = get_settings()
        
        # VRP thresholds should be reasonable
        thresholds = settings.model.vrp_thresholds
        assert len(thresholds) == 4
        assert thresholds == [0.9, 1.1, 1.3, 1.5]
        
        # Time windows should be reasonable
        assert 30 <= settings.model.transition_window_days <= 120
        assert 20 <= settings.model.realized_vol_window_days <= 60
        
        # Parameters should be reasonable
        assert settings.model.laplace_smoothing_alpha > 0
        assert 0.5 <= settings.model.min_confidence_for_signal <= 1.0

    def test_trading_section_defaults(self):
        """Test trading section default values.""" 
        settings = get_settings()
        
        # Position sizes should be reasonable
        assert 0 < settings.trading.base_position_size_pct < 0.1  # < 10%
        assert settings.trading.max_position_size_pct > settings.trading.base_position_size_pct
        assert settings.trading.max_position_size_pct < 0.2  # < 20%
        
        # Confidence thresholds should be reasonable
        assert 0 < settings.trading.extreme_low_confidence_threshold < 1.0
        assert 0 < settings.trading.extreme_high_confidence_threshold < 1.0

    def test_data_section_defaults(self):
        """Test data section default values."""
        settings = get_settings()
        
        # Data requirements should be reasonable
        assert 1 <= settings.data.min_data_years <= 5
        assert settings.data.preferred_data_years >= settings.data.min_data_years
        assert settings.data.preferred_data_years <= 10

    def test_database_section_defaults(self):
        """Test database section default values."""
        settings = get_settings()
        
        # Database configuration should be reasonable
        assert settings.database.database_path is not None
        assert len(settings.database.database_path) > 0
        assert settings.database.database_url is not None
        assert len(settings.database.database_url) > 0


class TestSettingsIntegration:
    """Test integration with existing system components."""
    
    def test_settings_with_vrp_trader(self):
        """Test that VRPTrader works with Settings."""
        from vrp import VRPTrader
        
        settings = get_settings()
        
        # Should initialize without errors
        trader = VRPTrader(settings=settings)
        assert trader.settings == settings

    def test_settings_with_daily_trader(self):
        """Test that DailyVRPTrader works with Settings."""
        from src.production.daily_trader import DailyVRPTrader
        
        settings = get_settings()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "test.db")
            trader = DailyVRPTrader(settings=settings, database_path=db_path)
            
            assert trader.settings == settings

    def test_settings_with_volatility_calculator(self):
        """Test that VolatilityCalculator works with Settings."""
        from src.services.volatility_calculator import VolatilityCalculator
        
        settings = get_settings()
        
        # Should initialize without errors
        calculator = VolatilityCalculator(settings=settings)
        assert calculator.settings == settings

    def test_settings_with_signal_generator(self):
        """Test that SignalGenerator works with Settings."""
        from services.signal_generator import SignalGenerator
        
        settings = get_settings()
        
        # Should initialize without errors
        signal_generator = SignalGenerator(settings=settings)
        assert signal_generator.settings == settings