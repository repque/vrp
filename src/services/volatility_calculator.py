"""
Volatility calculation service for VRP Markov Chain Trading Model.

This service handles all volatility-related calculations including realized volatility,
VRP ratio computation, and volatility data generation.
"""

import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.interfaces.contracts import IVRPClassifier, IVolatilityCalculator
from src.models.data_models import MarketDataPoint, VRPState, VolatilityData
from src.utils.exceptions import CalculationError, InsufficientDataError


logger = logging.getLogger(__name__)


class VolatilityCalculator(IVolatilityCalculator):
    """
    Implementation of volatility calculations for VRP analysis.
    
    This service calculates realized volatility using rolling windows and
    computes volatility risk premium ratios with proper error handling.
    """
    
    def __init__(self, vrp_classifier: IVRPClassifier):
        """
        Initialize volatility calculator.
        
        Args:
            vrp_classifier: VRP state classifier for state assignment
        """
        self.settings = get_settings()
        self.vrp_classifier = vrp_classifier
        
    def calculate_realized_volatility(
        self, 
        market_data: List[MarketDataPoint], 
        window_days: int = 30
    ) -> Dict[datetime, float]:
        """
        Calculate rolling realized volatility for market data.
        
        Uses log returns and rolling standard deviation with annualization
        factor to compute realized volatility.
        
        Args:
            market_data: Historical market data for SPY
            window_days: Rolling window size in days
            
        Returns:
            Dictionary mapping dates to annualized volatility values
            
        Raises:
            InsufficientDataError: If insufficient data for calculation
            CalculationError: If calculation fails
        """
        if len(market_data) < window_days:
            raise InsufficientDataError(
                required=window_days,
                available=len(market_data)
            )
        
        try:
            # Convert to DataFrame for easier calculation
            df = pd.DataFrame([
                {
                    'date': data.date,
                    'close': data.spy_close,
                    'open': data.spy_open
                }
                for data in market_data
            ])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Calculate log returns using close prices
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # Calculate rolling standard deviation
            df['rolling_std'] = df['log_return'].rolling(
                window=window_days,
                min_periods=window_days
            ).std()
            
            # Annualize volatility
            annualization_factor = np.sqrt(self.settings.model.volatility_annualization_factor)
            df['realized_vol'] = df['rolling_std'] * annualization_factor
            
            # Remove NaN values and convert to dictionary
            valid_data = df.dropna()
            volatility_dict = dict(zip(valid_data['date'], valid_data['realized_vol']))
            
            logger.info(f"Calculated realized volatility for {len(volatility_dict)} dates")
            return volatility_dict
            
        except Exception as e:
            raise CalculationError(
                calculation="realized_volatility",
                message=f"Failed to calculate realized volatility: {str(e)}"
            )
    
    def calculate_vrp_ratio(
        self, 
        implied_vol: float, 
        realized_vol: float
    ) -> float:
        """
        Calculate volatility risk premium ratio.
        
        VRP = Implied Volatility / Realized Volatility
        where Implied Volatility = VIX / 100
        
        Args:
            implied_vol: Implied volatility (VIX/100)
            realized_vol: Realized volatility (annualized)
            
        Returns:
            VRP ratio (implied/realized)
            
        Raises:
            CalculationError: If calculation fails or inputs are invalid
        """
        try:
            # Validate inputs
            if implied_vol <= 0:
                raise ValueError(f"Implied volatility must be positive, got {implied_vol}")
            
            if realized_vol <= 0:
                raise ValueError(f"Realized volatility must be positive, got {realized_vol}")
            
            # Calculate VRP ratio
            vrp_ratio = implied_vol / realized_vol
            
            # Validate result is reasonable
            min_vrp = self.settings.model.vrp_min_reasonable
            max_vrp = self.settings.model.vrp_max_reasonable
            if vrp_ratio < min_vrp or vrp_ratio > max_vrp:
                logger.warning(f"VRP ratio {vrp_ratio:.4f} is outside typical bounds [{min_vrp}, {max_vrp}]")
            
            return vrp_ratio
            
        except Exception as e:
            raise CalculationError(
                calculation="vrp_ratio",
                message=f"Failed to calculate VRP ratio: {str(e)}"
            )
    
    def generate_volatility_data(
        self, 
        market_data: List[MarketDataPoint]
    ) -> List[VolatilityData]:
        """
        Generate complete volatility analysis for market data.
        
        Processes SPY and VIX data to produce VolatilityData objects
        with VRP calculations and state classifications.
        
        Args:
            market_data: Combined SPY and VIX market data
            
        Returns:
            List of VolatilityData objects with VRP calculations
            
        Raises:
            InsufficientDataError: If insufficient data
            CalculationError: If processing fails
        """
        try:
            # Separate SPY and VIX data
            spy_data = []
            vix_data = {}
            
            for data in market_data:
                # All data has both volume and iv in generic schema
                if hasattr(data, 'volume') and data.volume > 0:
                    spy_data.append(data)
                # All data has implied volatility values
                if hasattr(data, 'iv') and data.iv > 0:
                    vix_data[data.date] = data.iv
            
            # Calculate realized volatility for SPY
            realized_vol_dict = self.calculate_realized_volatility(
                spy_data, 
                self.settings.model.realized_vol_window_days
            )
            
            # Generate volatility data for dates with both SPY and VIX data
            volatility_data_list = []
            
            for date, realized_vol in realized_vol_dict.items():
                if date in vix_data:
                    try:
                        # Convert VIX to implied volatility
                        implied_vol = vix_data[date] / 100.0
                        
                        # Calculate VRP ratio
                        vrp_ratio = self.calculate_vrp_ratio(implied_vol, realized_vol)
                        
                        # Classify VRP state
                        vrp_state = self.vrp_classifier.classify_vrp_state(vrp_ratio)
                        
                        # Create VolatilityData object
                        vol_data = VolatilityData(
                            date=date,
                            realized_volatility_30d=realized_vol,
                            implied_volatility=implied_vol,
                            vrp_ratio=vrp_ratio,
                            vrp_state=vrp_state
                        )
                        
                        volatility_data_list.append(vol_data)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process data for {date}: {str(e)}")
                        continue
            
            if not volatility_data_list:
                raise InsufficientDataError(
                    required=1,
                    available=0
                )
            
            # Sort by date
            volatility_data_list.sort(key=lambda x: x.date)
            
            logger.info(f"Generated volatility data for {len(volatility_data_list)} dates")
            return volatility_data_list
            
        except Exception as e:
            if isinstance(e, (InsufficientDataError, CalculationError)):
                raise
            raise CalculationError(
                calculation="volatility_data_generation",
                message=f"Failed to generate volatility data: {str(e)}"
            )
    
    def validate_volatility_calculation(
        self, 
        returns: List[float], 
        window_size: int
    ) -> bool:
        """
        Validate volatility calculation inputs and intermediate results.
        
        Args:
            returns: List of returns for validation
            window_size: Rolling window size
            
        Returns:
            True if validation passes
            
        Raises:
            CalculationError: If validation fails
        """
        try:
            # Check for sufficient data
            if len(returns) < window_size:
                raise ValueError(f"Insufficient returns: need {window_size}, got {len(returns)}")
            
            # Check for excessive missing values
            valid_returns = [r for r in returns if not np.isnan(r)]
            missing_pct = (len(returns) - len(valid_returns)) / len(returns)
            
            if missing_pct > self.settings.data.max_missing_days_pct:
                raise ValueError(f"Too many missing values: {missing_pct:.2%}")
            
            # Check for extreme outliers (>10 standard deviations)
            returns_array = np.array(valid_returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            outliers = np.abs(returns_array - mean_return) > 10 * std_return
            outlier_pct = np.sum(outliers) / len(returns_array)
            
            if outlier_pct > 0.01:  # More than 1% outliers
                logger.warning(f"High percentage of outliers detected: {outlier_pct:.2%}")
            
            return True
            
        except Exception as e:
            raise CalculationError(
                calculation="volatility_validation",
                message=f"Validation failed: {str(e)}"
            )
    
    def calculate_volatility_statistics(
        self, 
        volatility_data: List[VolatilityData]
    ) -> Dict[str, float]:
        """
        Calculate summary statistics for volatility data.
        
        Args:
            volatility_data: List of volatility data objects
            
        Returns:
            Dictionary of volatility statistics
        """
        if not volatility_data:
            return {}
        
        realized_vols = [data.realized_volatility_30d for data in volatility_data]
        implied_vols = [data.implied_volatility for data in volatility_data]
        vrp_ratios = [data.vrp_ratio for data in volatility_data]
        
        stats = {
            'count': len(volatility_data),
            'realized_vol_mean': np.mean(realized_vols),
            'realized_vol_std': np.std(realized_vols),
            'realized_vol_min': np.min(realized_vols),
            'realized_vol_max': np.max(realized_vols),
            'implied_vol_mean': np.mean(implied_vols),
            'implied_vol_std': np.std(implied_vols),
            'vrp_ratio_mean': np.mean(vrp_ratios),
            'vrp_ratio_std': np.std(vrp_ratios),
            'vrp_ratio_min': np.min(vrp_ratios),
            'vrp_ratio_max': np.max(vrp_ratios),
        }
        
        logger.info(f"Calculated volatility statistics for {stats['count']} observations")
        return stats