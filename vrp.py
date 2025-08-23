#!/usr/bin/env python3
"""
VRP Trading System

Simple interface for VRP (Volatility Risk Premium) analysis.
Reads CSV files with market data and generates trading signals.
"""

import pandas as pd
import numpy as np
import logging
from decimal import Decimal
from datetime import datetime, date
from typing import Optional, Dict, Any, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
# Add services to path
sys.path.insert(0, os.path.dirname(__file__))

from src.models.data_models import MarketData, VRPState, TradingSignal, VolatilityData
from src.models.constants import DefaultConfiguration, BusinessConstants
from src.utils.exceptions import DataQualityError, CalculationError, InsufficientDataError, ModelStateError
from services import VRPCalculator, SignalGenerator, BacktestEngine

# Configure logging for internal operations
logger = logging.getLogger(__name__)


class VRPTrader:
    """
    Predictive VRP Trading System
    
    Orchestrates Markov chain-based prediction services for VRP state transitions.
    Uses predictive modeling instead of simple thresholds for signal generation.
    
    Usage:
        trader = VRPTrader()
        trader.load_data('market_data.csv')
        signal = trader.get_signal()
    """
    
    def __init__(self, config: Optional[DefaultConfiguration] = None):
        """
        Initialize VRP trader with service dependencies.
        
        Args:
            config: System configuration, uses default if not provided
        """
        self.config = config or DefaultConfiguration()
        self.data = None
        self.current_vrp = None
        self.current_state = None
        self._volatility_data_cache = None
        
        # Initialize predictive service dependencies
        self.calculator = VRPCalculator(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.backtest_engine = BacktestEngine(self.calculator, self.signal_generator)
        
        logger.info(f"VRPTrader initialized with predictive Markov chain services")
    
    def load_data(self, data_source) -> bool:
        """
        Load market data from CSV file or DataFrame.
        
        Expected columns: date, spy_open, spy_high, spy_low, spy_close, spy_volume, vix_close
        
        Args:
            data_source: CSV file path or pandas DataFrame
            
        Returns:
            True if successful
        """
        try:
            # Handle different input types
            if isinstance(data_source, str):
                # CSV file path
                df = pd.read_csv(data_source)
                logger.info(f"Loaded data from {data_source}")
            elif isinstance(data_source, pd.DataFrame):
                # DataFrame
                df = data_source.copy()
                logger.info("Loaded DataFrame")
            else:
                logger.error("Data source must be CSV file path or DataFrame")
                return False
            
            # Validate required columns
            required_cols = ['date', 'spy_open', 'spy_high', 'spy_low', 'spy_close', 'spy_volume', 'vix_close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}. Required: {required_cols}")
                return False
            
            # Convert data types
            df['date'] = pd.to_datetime(df['date'])
            
            # Convert to our data models for validation
            validated_data = []
            for _, row in df.iterrows():
                try:
                    market_data = MarketData(
                        date=row['date'],
                        spy_open=Decimal(str(row['spy_open'])),
                        spy_high=Decimal(str(row['spy_high'])),
                        spy_low=Decimal(str(row['spy_low'])),
                        spy_close=Decimal(str(row['spy_close'])),
                        spy_volume=int(row['spy_volume']),
                        vix_close=Decimal(str(row['vix_close']))
                    )
                    validated_data.append(market_data)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid row {row['date']}: {e}")
                    continue
            
            if not validated_data:
                raise DataQualityError("No valid data rows found after validation")
            
            self.data = validated_data
            # Clear cache when new data is loaded
            self._volatility_data_cache = None
            logger.info(f"Successfully loaded and validated {len(validated_data)} data points")
            return True
            
        except DataQualityError:
            logger.error("No valid data rows found")
            return False
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            logger.error(f"Error loading data file: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading data: {e}")
            return False
    
    def _generate_volatility_data(self) -> List[VolatilityData]:
        """
        Generate VolatilityData objects from market data for Markov chain processing.
        
        Returns:
            List of VolatilityData objects
            
        Raises:
            InsufficientDataError: If insufficient data
            CalculationError: If generation fails
        """
        if self._volatility_data_cache is None:
            try:
                self._volatility_data_cache = self.calculator.generate_volatility_data(self.data)
                logger.info(f"Generated {len(self._volatility_data_cache)} volatility data points")
            except (InsufficientDataError, CalculationError):
                raise
        
        return self._volatility_data_cache
    
    def get_current_vrp_state(self) -> VRPState:
        """Get current VRP state from latest data using calculator service."""
        if not self.data:
            raise InsufficientDataError("No data loaded for VRP state classification")
        
        return self.calculator.get_current_vrp_state(self.data)
    
    def get_signal(self) -> Optional[str]:
        """
        Get predictive trading signal based on Markov chain state transitions.
        
        Returns:
            "BUY_VOL", "SELL_VOL", or "HOLD"
        """
        if not self.data:
            logger.error("No data loaded. Use load_data() first.")
            return None
        
        try:
            # Generate volatility data for Markov chain processing
            volatility_data = self._generate_volatility_data()
            
            # Get current state
            self.current_state = volatility_data[-1].vrp_state
            self.current_vrp = float(volatility_data[-1].vrp)
            
            # Generate predictive signal using Markov chain model
            trading_signal = self.signal_generator.generate_signal(volatility_data)
            
            logger.info(f"Generated predictive signal: {trading_signal.signal_type} | "
                       f"Current: {trading_signal.current_state.name} | "
                       f"Predicted: {trading_signal.predicted_state.name} | "
                       f"Confidence: {trading_signal.confidence_score:.3f} | "
                       f"Reason: {trading_signal.reason}")
            
            return trading_signal.signal_type
            
        except (InsufficientDataError, CalculationError, ModelStateError) as e:
            logger.error(f"Failed to generate predictive signal: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in signal generation: {str(e)}")
            return None
    
    def backtest(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Backtest functionality is temporarily disabled pending API updates.
        
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            
        Returns:
            Dictionary with backtest results
        """
        logger.warning("Backtest functionality temporarily disabled - predictive system updates in progress")
        return {
            "status": "disabled",
            "message": "Backtest functionality temporarily disabled pending API updates for predictive system",
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0
        }
    
    def get_trading_signal_details(self) -> Optional[TradingSignal]:
        """
        Get detailed predictive trading signal with full metadata.
        
        Returns:
            Complete TradingSignal object with predictions and confidence metrics
        """
        if not self.data:
            logger.error("No data loaded. Use load_data() first.")
            return None
        
        try:
            # Generate volatility data for Markov chain processing
            volatility_data = self._generate_volatility_data()
            
            # Generate full trading signal with all details
            trading_signal = self.signal_generator.generate_signal(volatility_data)
            
            return trading_signal
            
        except (InsufficientDataError, CalculationError, ModelStateError) as e:
            logger.error(f"Failed to generate detailed signal: {str(e)}")
            return None


def create_sample_data(filename: str = "sample_data.csv"):
    """Create sample CSV file for testing."""
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    
    # Generate realistic sample data
    np.random.seed(42)
    spy_prices = 400 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    vix_values = 15 + np.abs(np.random.randn(len(dates)) * 3)
    
    data = []
    for i, date in enumerate(dates):
        if i == 0:
            continue
            
        spy_open = spy_prices[i] + np.random.randn() * 0.5
        spy_close = spy_prices[i]
        spy_high = max(spy_open, spy_close) + abs(np.random.randn() * 0.3)
        spy_low = min(spy_open, spy_close) - abs(np.random.randn() * 0.3)
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'spy_open': round(spy_open, 2),
            'spy_high': round(spy_high, 2),
            'spy_low': round(spy_low, 2),
            'spy_close': round(spy_close, 2),
            'spy_volume': int(80_000_000 + np.random.randn() * 20_000_000),
            'vix_close': round(vix_values[i], 2)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    logger.info(f"Created sample data file: {filename}")
    return filename


if __name__ == "__main__":
    # Configure logging for demo
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("VRP Trading System starting")
    
    # Create sample data for demo
    sample_file = create_sample_data()
    
    # Example usage
    trader = VRPTrader()
    
    if trader.load_data(sample_file):
        signal = trader.get_signal()
        trader.backtest()
        
        logger.info(f"Demo completed. Current signal: {signal}")
    else:
        logger.error("Failed to load sample data")