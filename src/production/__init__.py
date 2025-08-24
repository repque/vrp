"""
Production components for daily VRP trading operations.

This module provides production-ready components for running VRP trading
system in daily batch mode with state persistence and error recovery.
"""

from .daily_trader import DailyVRPTrader, run_daily_processing

__all__ = ['DailyVRPTrader', 'run_daily_processing']