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
from src.models.constants import DefaultConfiguration
from src.utils.exceptions import CalculationError, InsufficientDataError

logger = logging.getLogger(__name__)


class VRPCalculator:
    """
    Service for generating VolatilityData objects for Markov chain model.
    
    Converts raw market data into structured volatility data with VRP
    calculations and state classifications for predictive modeling.
    """
    
    def __init__(self, config: Optional[DefaultConfiguration] = None):
        """
        Initialize VRP calculator with configuration.
        
        Args:
            config: System configuration, uses default if not provided
        """
        self.config = config or DefaultConfiguration()
        self.trading_days_per_year = self.config.VOLATILITY_ANNUALIZATION_FACTOR
        
        logger.info("VRPCalculator initialized for VolatilityData generation")
    
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
        
        # Start from day 30 to have enough data for 30-day realized volatility
        for i in range(30, len(data)):
            try:
                current_date = data[i].date
                
                # Calculate daily return
                prev_close = float(data[i-1].spy_close)
                curr_close = float(data[i].spy_close)
                daily_return = Decimal(str((curr_close - prev_close) / prev_close))
                
                # Calculate 30-day realized volatility
                realized_vol = self._calculate_rolling_volatility(data[i-29:i+1])
                
                # Get implied volatility
                implied_vol = Decimal(str(float(data[i].vix_close) / 100))
                
                # Calculate VRP ratio
                if realized_vol <= 0:
                    logger.warning(f"Skipping date {current_date}: realized volatility <= 0")
                    continue
                
                vrp_ratio = implied_vol / realized_vol
                
                # Classify VRP state
                vrp_state = self._classify_vrp_state(float(vrp_ratio))
                
                # Create VolatilityData object
                vol_data = VolatilityData(
                    date=current_date,
                    spy_return=daily_return,
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
            prev_close = float(data_window[i-1].spy_close)
            curr_close = float(data_window[i].spy_close)
            daily_return = (curr_close - prev_close) / prev_close
            returns.append(daily_return)
        
        if not returns:
            return Decimal('0')
        
        volatility = np.std(returns) * np.sqrt(self.trading_days_per_year)
        return Decimal(str(volatility))
    
    def _classify_vrp_state(self, vrp_ratio: float) -> VRPState:
        """
        Classify VRP ratio into discrete states for Markov chain.
        
        Args:
            vrp_ratio: The VRP ratio to classify
            
        Returns:
            VRP state classification
        """
        config = self.config
        
        if vrp_ratio < float(config.VRP_UNDERPRICED_THRESHOLD):
            return VRPState.EXTREME_LOW
        elif vrp_ratio < float(config.VRP_FAIR_UPPER_THRESHOLD):
            return VRPState.FAIR_VALUE
        elif vrp_ratio < float(config.VRP_NORMAL_UPPER_THRESHOLD):
            return VRPState.NORMAL_PREMIUM
        elif vrp_ratio < float(config.VRP_ELEVATED_UPPER_THRESHOLD):
            return VRPState.ELEVATED_PREMIUM
        else:
            return VRPState.EXTREME_HIGH
    
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