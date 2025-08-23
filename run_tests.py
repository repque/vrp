#!/usr/bin/env python3
"""
Direct test runner for VRP data models.
Runs tests without pytest to avoid environment issues.
"""

import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime, date
from pydantic import ValidationError

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.data_models import (
    VRPState,
    SignalType,
    MarketData,
    VolatilityData,
    TransitionMatrix,
    ConfidenceMetrics,
    TradingSignal,
    ModelState,
    PerformanceMetrics,
    BacktestResult,
    ModelPrediction,
    ConfigurationSettings,
    ModelHealthMetrics,
    # Aliases
    MarketDataPoint,
    VolatilityMetrics,
)


class TestRunner:
    def __init__(self):
        self.tests_passed = 0
        self.tests_total = 0
        self.failed_tests = []

    def assert_equal(self, actual, expected, msg=""):
        self.tests_total += 1
        if actual == expected:
            self.tests_passed += 1
        else:
            self.failed_tests.append(f"Expected {expected}, got {actual}. {msg}")

    def assert_raises(self, exception_type, func, expected_msg=""):
        self.tests_total += 1
        try:
            func()
            self.failed_tests.append(f"Expected {exception_type.__name__} but no exception was raised")
        except exception_type as e:
            if expected_msg and expected_msg not in str(e):
                self.failed_tests.append(f"Expected message '{expected_msg}' in exception, got: {e}")
            else:
                self.tests_passed += 1
        except Exception as e:
            self.failed_tests.append(f"Expected {exception_type.__name__}, got {type(e).__name__}: {e}")

    def test_vrp_state_values(self):
        """Test that VRP states have correct values."""
        print("Testing VRP state values...")
        self.assert_equal(VRPState.EXTREME_LOW.value, 1)
        self.assert_equal(VRPState.FAIR_VALUE.value, 2)
        self.assert_equal(VRPState.NORMAL_PREMIUM.value, 3)
        self.assert_equal(VRPState.ELEVATED_PREMIUM.value, 4)
        self.assert_equal(VRPState.EXTREME_HIGH.value, 5)

    def test_vrp_state_ordering(self):
        """Test that VRP states can be compared."""
        print("Testing VRP state ordering...")
        self.tests_total += 3
        if VRPState.EXTREME_LOW < VRPState.FAIR_VALUE:
            self.tests_passed += 1
        else:
            self.failed_tests.append("VRPState ordering failed: EXTREME_LOW < FAIR_VALUE")
        
        if VRPState.EXTREME_HIGH > VRPState.ELEVATED_PREMIUM:
            self.tests_passed += 1
        else:
            self.failed_tests.append("VRPState ordering failed: EXTREME_HIGH > ELEVATED_PREMIUM")
        
        if VRPState.NORMAL_PREMIUM == VRPState.NORMAL_PREMIUM:
            self.tests_passed += 1
        else:
            self.failed_tests.append("VRPState equality failed")

    def test_valid_market_data_creation(self):
        """Test creation of valid market data point."""
        print("Testing valid market data creation...")
        
        data = MarketData(
            date=datetime(2023, 1, 1),
            spy_open=400.0,
            spy_high=405.0,
            spy_low=395.0,
            spy_close=402.0,
            spy_volume=100_000_000,
            vix_close=20.0
        )
        
        self.assert_equal(data.date, date(2023, 1, 1))
        self.assert_equal(data.spy_open, Decimal('400.0'))
        self.assert_equal(data.spy_high, Decimal('405.0'))
        self.assert_equal(data.spy_low, Decimal('395.0'))
        self.assert_equal(data.spy_close, Decimal('402.0'))
        self.assert_equal(data.spy_volume, 100_000_000)
        self.assert_equal(data.vix_close, Decimal('20.0'))

    def test_negative_prices_validation(self):
        """Test that negative prices are rejected."""
        print("Testing negative prices validation...")
        
        def create_negative_price_data():
            return MarketDataPoint(
                date=date(2023, 1, 1),
                spy_open=Decimal('-400.0'),
                spy_high=Decimal('405.0'),
                spy_low=Decimal('395.0'),
                spy_close=Decimal('402.0'),
                spy_volume=100_000_000,
                vix_close=Decimal('20.0')
            )
        
        self.assert_raises(ValidationError, create_negative_price_data, "must be positive")

    def test_valid_volatility_metrics(self):
        """Test creation of valid volatility metrics."""
        print("Testing valid volatility metrics...")
        
        metrics = VolatilityMetrics(
            date=date(2023, 1, 1),
            spy_return=Decimal('0.015'),
            realized_vol_30d=Decimal('0.20'),
            implied_vol=Decimal('0.25'),
            vrp=Decimal('1.25'),
            vrp_state=VRPState.NORMAL_PREMIUM
        )
        
        self.assert_equal(metrics.date, date(2023, 1, 1))
        self.assert_equal(metrics.spy_return, Decimal('0.015'))
        self.assert_equal(metrics.realized_vol_30d, Decimal('0.20'))
        self.assert_equal(metrics.implied_vol, Decimal('0.25'))
        self.assert_equal(metrics.vrp, Decimal('1.25'))
        self.assert_equal(metrics.vrp_state, VRPState.NORMAL_PREMIUM)

    def test_valid_transition_matrix(self):
        """Test creation of valid transition matrix."""
        print("Testing valid transition matrix...")
        
        matrix = [
            [Decimal('0.2'), Decimal('0.3'), Decimal('0.3'), Decimal('0.15'), Decimal('0.05')],
            [Decimal('0.1'), Decimal('0.4'), Decimal('0.3'), Decimal('0.15'), Decimal('0.05')],
            [Decimal('0.05'), Decimal('0.25'), Decimal('0.4'), Decimal('0.25'), Decimal('0.05')],
            [Decimal('0.05'), Decimal('0.15'), Decimal('0.2'), Decimal('0.35'), Decimal('0.25')],
            [Decimal('0.1'), Decimal('0.2'), Decimal('0.3'), Decimal('0.3'), Decimal('0.1')]
        ]
        
        tm = TransitionMatrix(
            matrix=matrix,
            observation_count=120,
            window_start=date(2023, 1, 1),
            window_end=date(2023, 3, 1)
        )
        
        self.assert_equal(len(tm.matrix), 5)
        self.assert_equal(len(tm.matrix[0]), 5)
        self.assert_equal(tm.observation_count, 120)
        self.assert_equal(tm.window_start, date(2023, 1, 1))
        self.assert_equal(tm.window_end, date(2023, 3, 1))
        
        # Check that last_updated is a datetime
        self.tests_total += 1
        if isinstance(tm.last_updated, datetime):
            self.tests_passed += 1
        else:
            self.failed_tests.append(f"Expected datetime for last_updated, got {type(tm.last_updated)}")

    def test_configuration_defaults(self):
        """Test that default configuration is valid."""
        print("Testing configuration defaults...")
        
        config = ConfigurationSettings()
        
        self.assert_equal(config.min_data_years, 3)
        self.assert_equal(config.preferred_data_years, 5)
        self.assert_equal(config.rolling_window_days, 60)
        self.assert_equal(config.vrp_underpriced_threshold, Decimal('0.90'))
        self.assert_equal(config.vrp_fair_upper_threshold, Decimal('1.10'))
        self.assert_equal(config.vrp_normal_upper_threshold, Decimal('1.30'))
        self.assert_equal(config.vrp_elevated_upper_threshold, Decimal('1.50'))
        self.assert_equal(config.laplace_smoothing_alpha, Decimal('1.0'))
        self.assert_equal(config.min_confidence_threshold, Decimal('0.6'))
        self.assert_equal(config.max_position_size, Decimal('0.05'))
        self.assert_equal(config.target_sharpe_ratio, Decimal('0.8'))

    def run_all_tests(self):
        """Run all tests and report results."""
        print("=" * 60)
        print("VRP Data Models Test Suite")
        print("=" * 60)
        
        # Run all test methods
        test_methods = [method for method in dir(self) if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                method = getattr(self, method_name)
                method()
            except Exception as e:
                self.failed_tests.append(f"Error in {method_name}: {e}")
        
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Tests run: {self.tests_total}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Tests failed: {self.tests_total - self.tests_passed}")
        
        if self.failed_tests:
            print("\nFAILED TESTS:")
            for i, failure in enumerate(self.failed_tests, 1):
                print(f"{i}. {failure}")
        
        success_rate = (self.tests_passed / self.tests_total * 100) if self.tests_total > 0 else 0
        print(f"\nSuccess rate: {success_rate:.1f}%")
        
        return self.tests_passed == self.tests_total


if __name__ == "__main__":
    runner = TestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)