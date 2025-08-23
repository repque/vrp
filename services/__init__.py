"""
VRP Trading Services Package

This package contains decomposed service classes that follow the Single
Responsibility Principle for better maintainability and testability.
"""

from .vrp_calculator import VRPCalculator
from .signal_generator import SignalGenerator
from .backtest_engine import BacktestEngine

__all__ = [
    'VRPCalculator',
    'SignalGenerator', 
    'BacktestEngine'
]