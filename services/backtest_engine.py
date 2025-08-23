"""
Backtesting Engine Service

Handles backtesting logic with proper separation of concerns,
performance metrics calculation, and comprehensive result analysis.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, date

from src.models.data_models import MarketData
from src.utils.exceptions import InsufficientDataError, CalculationError
from .vrp_calculator import VRPCalculator
from .signal_generator import SignalGenerator

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Service for backtesting VRP trading strategies.
    
    Handles historical simulation, performance metrics calculation,
    and trade analysis with proper validation and error handling.
    """
    
    def __init__(
        self, 
        calculator: VRPCalculator,
        signal_generator: SignalGenerator
    ):
        """
        Initialize backtest engine with required services.
        
        Args:
            calculator: VRP calculation service
            signal_generator: Signal generation service
        """
        self.calculator = calculator
        self.signal_generator = signal_generator
        
        logger.info("BacktestEngine initialized")
    
    def run_backtest(
        self,
        data: List[MarketData],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtest on historical data.
        
        Args:
            data: Historical market data
            start_date: Start date for backtest (YYYY-MM-DD), optional
            end_date: End date for backtest (YYYY-MM-DD), optional
            lookback_days: Days to use for volatility calculations
            
        Returns:
            Dictionary with backtest results
            
        Raises:
            InsufficientDataError: If not enough data for backtest
            CalculationError: If backtest calculation fails
        """
        if not data:
            raise InsufficientDataError("No data provided for backtest")
        
        logger.info("Starting backtest")
        
        try:
            # Filter data by date range if specified
            test_data = self._filter_data_by_date(data, start_date, end_date)
            
            # Validate minimum data requirements
            min_required = lookback_days + 20  # Buffer for calculations
            if len(test_data) < min_required:
                raise InsufficientDataError(f"Need at least {min_required} days of data, got {len(test_data)}")
            
            # Run historical simulation
            trades = self._simulate_trading(test_data, lookback_days)
            
            # Calculate performance metrics
            results = self._calculate_performance_metrics(trades)
            
            logger.info(f"Backtest completed: return={results['total_return']:.4f}, win_rate={results['win_rate']:.2f}, trades={results['total_trades']}")
            
            return results
            
        except (InsufficientDataError, CalculationError):
            # Re-raise specific errors
            raise
        except Exception as e:
            raise CalculationError(
                calculation="backtest_execution",
                message=f"Backtest error: {e}"
            )
    
    def _filter_data_by_date(
        self,
        data: List[MarketData],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> List[MarketData]:
        """
        Filter data by specified date range.
        
        Args:
            data: Original market data
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            Filtered market data
        """
        filtered_data = data
        
        if start_date:
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
            filtered_data = [d for d in filtered_data if d.date >= start]
            logger.debug(f"Filtered data from {start_date}: {len(filtered_data)} points")
        
        if end_date:
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
            filtered_data = [d for d in filtered_data if d.date <= end]
            logger.debug(f"Filtered data to {end_date}: {len(filtered_data)} points")
        
        return filtered_data
    
    def _simulate_trading(
        self,
        data: List[MarketData],
        lookback_days: int
    ) -> List[Dict]:
        """
        Simulate trading based on historical signals.
        
        Args:
            data: Historical market data
            lookback_days: Days for volatility calculation
            
        Returns:
            List of trade records
        """
        trades = []
        current_position = 0
        
        # Start after lookback period to ensure we have enough data
        for i in range(lookback_days, len(data)):
            # Calculate VRP for this historical point
            historical_data = data[max(0, i-lookback_days):i+1]
            
            try:
                vrp_ratio = self.calculator.calculate_vrp_ratio(historical_data, lookback_days)
                vrp_state = self.calculator.classify_vrp_state(vrp_ratio)
                
                # Generate signal for this point in time
                signal, reason = self.signal_generator.generate_signal(vrp_ratio, vrp_state)
                
                # Execute trades based on signal
                new_position = self._execute_trade(signal, current_position)
                
                # Calculate P&L if position changed
                pnl = 0
                if new_position != current_position and i > 0:
                    pnl = self._calculate_trade_pnl(
                        data[i-1], data[i], current_position, new_position
                    )
                
                # Record trade if action taken
                if signal != "HOLD" or new_position != current_position:
                    trades.append({
                        'date': data[i].date,
                        'signal': signal,
                        'vrp_ratio': vrp_ratio,
                        'vrp_state': vrp_state.name,
                        'position_before': current_position,
                        'position_after': new_position,
                        'pnl': pnl,
                        'reason': reason
                    })
                
                current_position = new_position
                
            except (InsufficientDataError, CalculationError) as e:
                logger.warning(f"Skipping date {data[i].date} due to calculation error: {e}")
                continue
        
        logger.debug(f"Simulation completed with {len(trades)} trades")
        return trades
    
    def _execute_trade(self, signal: str, current_position: float) -> float:
        """
        Execute trade based on signal and current position.
        
        Args:
            signal: Trading signal ("BUY_VOL", "SELL_VOL", "HOLD")
            current_position: Current position size
            
        Returns:
            New position size
        """
        position_size = self.signal_generator.get_position_size()
        
        if signal == "BUY_VOL" and current_position <= 0:
            return position_size
        elif signal == "SELL_VOL" and current_position >= 0:
            return -position_size
        elif signal == "HOLD":
            return 0
        else:
            return current_position  # No change
    
    def _calculate_trade_pnl(
        self,
        prev_data: MarketData,
        curr_data: MarketData,
        old_position: float,
        new_position: float
    ) -> float:
        """
        Calculate P&L for a trade.
        
        Args:
            prev_data: Previous day's market data
            curr_data: Current day's market data
            old_position: Position before trade
            new_position: Position after trade
            
        Returns:
            P&L for the trade
        """
        try:
            # Simplified P&L calculation using VIX change
            # In practice, this would use actual option or ETF prices
            vix_change = (
                float(curr_data.vix_close) - float(prev_data.vix_close)
            ) / float(prev_data.vix_close)
            
            # Volatility positions have inverse relationship with VIX
            # (buying vol benefits from VIX increase)
            effective_position = (old_position + new_position) / 2
            pnl = effective_position * vix_change
            
            return pnl
            
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def _calculate_performance_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics from trades.
        
        Args:
            trades: List of trade records
            
        Returns:
            Dictionary with performance metrics
        """
        if not trades:
            return {
                'total_return': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Extract P&L data
        pnls = [trade['pnl'] for trade in trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        # Calculate basic metrics
        total_return = sum(pnls)
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        # Calculate advanced metrics
        max_drawdown = self._calculate_max_drawdown(pnls)
        sharpe_ratio = self._calculate_sharpe_ratio(pnls)
        
        return {
            'total_return': total_return,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
        }
    
    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown from P&L series."""
        if not pnls:
            return 0.0
        
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        
        return float(np.max(drawdown))
    
    def _calculate_sharpe_ratio(self, pnls: List[float]) -> float:
        """Calculate Sharpe ratio from P&L series."""
        if not pnls or len(pnls) < 2:
            return 0.0
        
        mean_return = np.mean(pnls)
        std_return = np.std(pnls)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe (assuming daily returns)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return float(sharpe)