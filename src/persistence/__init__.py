"""
Persistence layer for VRP state management.

This module provides SQLite-based persistence for maintaining model state,
trading positions, and signal history across daily production runs.
"""

from .database import SQLitePersistence
from .state_manager import StateManager

__all__ = ['SQLitePersistence', 'StateManager']