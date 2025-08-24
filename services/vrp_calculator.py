"""
VRP Calculation Service

Generates VolatilityData objects for Markov chain model processing.
Handles volatility calculations and state classifications.
"""

import logging
import numpy as np
from typing import List, Optional
from decimal import Decimal

from src.models.data_models import MarketData, VRPState, VolatilityData
from src.config.settings import Settings, get_settings
from src.services.vrp_classifier import VRPClassifier
from src.utils.exceptions import CalculationError, InsufficientDataError

logger = logging.getLogger(__name__)


class VRPCalculator:
    """
    Service for generating VolatilityData objects for Markov chain model.
    
    Converts raw market data into structured volatility data with VRP
    calculations and state classifications for predictive modeling.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize VRP calculator with configuration.
        
        Args:
            settings: System configuration, uses default if not provided
        """
        self.settings = settings or get_settings()
        self.trading_days_per_year = self.settings.model.volatility_annualization_factor
        self.vrp_classifier = VRPClassifier()
        
        logger.info("VRPCalculator initialized with adaptive VRP classification")
    
    def generate_volatility_data(self, data: List[MarketData]) -> List[VolatilityData]:
        """
        Generate VolatilityData objects from market data for Markov chain processing.
        
        Args:
            data: Historical market data
            
        Returns:
            List of VolatilityData objects with VRP calculations and state classifications
            
        Raises:
            InsufficientDataError: If insufficient data for calculations
            CalculationError: If calculation fails
        """
        if not data or len(data) < 31:  # Need at least 31 days for 30-day rolling
            raise InsufficientDataError(
                required=31,
                available=len(data) if data else 0
            )
        
        volatility_data = []
        
        # Pre-calculate all returns for vectorized operations
        closes = [float(d.close) for d in data]
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        
        # Start from day 30 to have enough data for 30-day realized volatility
        for i in range(30, len(data)):
            try:
                current_date = data[i].date
                
                # Get pre-calculated daily return
                daily_return = Decimal(str(returns[i-1]))  # returns is 0-indexed from day 1
                
                # Calculate 30-day realized volatility using cached returns
                window_returns = returns[i-30:i]  # Last 30 returns
                realized_vol = self._calculate_volatility_from_returns(window_returns)
                
                # Get implied volatility (cache division by 100)
                implied_vol = Decimal(str(float(data[i].iv) * 0.01))
                
                # Calculate VRP ratio
                if realized_vol <= 0:
                    logger.warning(f"Skipping date {current_date}: realized volatility <= 0")
                    continue
                
                vrp_ratio = implied_vol / realized_vol
                
                # Classify VRP state using adaptive quantile-based classifier
                vrp_state = self.vrp_classifier.classify_vrp_state(float(vrp_ratio))
                
                # Create VolatilityData object
                vol_data = VolatilityData(
                    date=current_date,
                    daily_return=daily_return,
                    realized_vol_30d=realized_vol,
                    implied_vol=implied_vol,
                    vrp=vrp_ratio,
                    vrp_state=vrp_state
                )
                
                volatility_data.append(vol_data)
                
            except Exception as e:
                logger.warning(f"Skipping date {data[i].date}: {str(e)}")
                continue
        
        if not volatility_data:
            raise CalculationError("No valid VolatilityData could be generated")
        
        logger.info(f"Generated {len(volatility_data)} VolatilityData points")
        return volatility_data
    
    def _calculate_rolling_volatility(self, data_window: List[MarketData]) -> Decimal:
        """
        Calculate annualized volatility for a rolling window.
        
        Args:
            data_window: 30-day window of market data
            
        Returns:
            Annualized realized volatility
        """
        returns = []
        for i in range(1, len(data_window)):
            prev_close = float(data_window[i-1].close)
            curr_close = float(data_window[i].close)
            daily_return = (curr_close - prev_close) / prev_close
            returns.append(daily_return)
        
        if not returns:
            return Decimal('0')
        
        volatility = np.std(returns) * np.sqrt(self.trading_days_per_year)
        return Decimal(str(volatility))
    
    def _calculate_volatility_from_returns(self, returns: List[float]) -> Decimal:
        """
        Optimized volatility calculation from pre-computed returns.
        
        Args:
            returns: List of daily returns
            
        Returns:
            Annualized realized volatility
        """
        if not returns:
            return Decimal('0')
        
        # Use numpy for efficient calculation
        volatility = np.std(returns, ddof=1) * np.sqrt(self.trading_days_per_year)
        return Decimal(str(volatility))
    
    
    def get_current_vrp_state(self, data: List[MarketData]) -> VRPState:
        """
        Get current VRP state from latest market data.
        
        Args:
            data: Market data
            
        Returns:
            Current VRP state classification
        """
        if not data:
            raise InsufficientDataError("No data available for VRP state classification")
        
        # Generate volatility data to get the latest VRP state
        volatility_data = self.generate_volatility_data(data)
        
        if not volatility_data:
            raise CalculationError("Could not generate volatility data for current state")
        
        return volatility_data[-1].vrp_state