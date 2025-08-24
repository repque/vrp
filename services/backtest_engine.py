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
from src.utils.exceptions import InsufficientDataError, CalculationError, ModelStateError
from src.utils.temporal_validation import validate_trading_decision_temporal_integrity
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
        
        # Need sufficient data for volatility calculations and Markov chain
        min_data_required = max(lookback_days, 90)  # Need more data for predictions
        
        # Memory-efficient approach: maintain rolling windows instead of regenerating
        window_size = min(200, len(data))  # Limit window size to prevent excessive memory usage
        volatility_cache = {}  # Cache volatility calculations to avoid recomputation
        
        # Start after minimum required period
        for i in range(min_data_required, len(data)):
            try:
                # CRITICAL: Only use data available up to current point (no forward-looking)
                # Use sliding window approach for memory efficiency
                start_idx = max(0, i - window_size + 1)
                historical_data = data[start_idx:i+1]  # Up to and including current day
                
                # Check cache first to avoid redundant calculations
                cache_key = (start_idx, i)
                if cache_key not in volatility_cache:
                    # Generate volatility data for Markov chain processing
                    # This ensures no future data leakage in calculations
                    volatility_data = self.calculator.generate_volatility_data(historical_data)
                    volatility_cache[cache_key] = volatility_data
                    
                    # Limit cache size to prevent memory bloat
                    if len(volatility_cache) > 100:  # Keep last 100 calculations
                        oldest_key = min(volatility_cache.keys())
                        del volatility_cache[oldest_key]
                else:
                    volatility_data = volatility_cache[cache_key]
                
                if len(volatility_data) < 60:  # Need minimum data for transition matrix
                    continue
                
                # CRITICAL: Validate no forward-looking bias before trading decision
                current_decision_date = data[i].date
                validate_trading_decision_temporal_integrity(
                    current_decision_date, historical_data, volatility_data
                )
                    
                # Generate predictive signal using Markov chain model
                trading_signal = self.signal_generator.generate_signal(volatility_data)
                
                signal = trading_signal.signal_type
                reason = trading_signal.reason
                current_state = trading_signal.current_state
                predicted_state = trading_signal.predicted_state
                confidence = float(trading_signal.confidence_score)
                
                # Execute trades based on signal
                new_position = self._execute_trade(signal, current_position)
                
                # Calculate P&L for position held during this period
                pnl = 0
                if i > 0:  # Need previous day for P&L calculation
                    pnl = self._calculate_trade_pnl(
                        data[i-1], data[i], current_position, new_position
                    )
                
                # Record trade for every signal (not just position changes)
                if signal != "HOLD":
                    # Get VRP data for this date
                    current_vrp_data = volatility_data[-1]  # Latest data point
                    
                    trades.append({
                        'date': data[i].date,
                        'signal': signal,
                        'vrp_ratio': float(current_vrp_data.vrp),
                        'current_state': current_state.name,
                        'predicted_state': predicted_state.name,
                        'confidence': confidence,
                        'position_before': current_position,
                        'position_after': new_position,
                        'pnl': pnl,
                        'reason': reason
                    })
                
                current_position = new_position
                
            except (InsufficientDataError, CalculationError, ModelStateError) as e:
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
        # Use base position size from configuration
        position_size = float(self.calculator.config.BASE_POSITION_SIZE)
        
        if signal == "BUY_VOL":
            # Always go long volatility (positive position)
            return position_size
        elif signal == "SELL_VOL":
            # Always go short volatility (negative position)
            return -position_size
        elif signal == "HOLD":
            # Keep current position (don't close positions on HOLD)
            return current_position
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
            # Calculate P&L based on position held during the period
            # Use the position we held BEFORE the trade (old_position)
            if abs(old_position) < 1e-6:  # No position held
                return 0.0
            
            # Robust IV change calculation with edge case handling
            prev_iv = float(prev_data.iv)
            curr_iv = float(curr_data.iv)
            
            # Edge case: prevent division by zero or near-zero IV
            if abs(prev_iv) < 1e-10:
                logger.warning(f"Previous IV too small ({prev_iv}), using zero P&L")
                return 0.0
            
            # Edge case: check for invalid IV values
            if prev_iv <= 0 or curr_iv <= 0:
                logger.warning(f"Invalid IV values (prev: {prev_iv}, curr: {curr_iv}), using zero P&L")
                return 0.0
            
            # Edge case: prevent extreme IV changes (likely data errors)
            iv_change_ratio = curr_iv / prev_iv
            if iv_change_ratio > 10.0 or iv_change_ratio < 0.1:  # 1000% or 90% change
                logger.warning(f"Extreme IV change detected ({iv_change_ratio:.2f}x), capping at reasonable bounds")
                iv_change_ratio = max(0.1, min(10.0, iv_change_ratio))
            
            iv_change_pct = iv_change_ratio - 1.0
            
            # Volatility positions profit from IV changes in the same direction:
            # - Long vol (positive position) profits when IV increases (positive change)
            # - Short vol (negative position) profits when IV decreases (negative change)
            pnl = old_position * iv_change_pct
            
            # Edge case: prevent infinite or NaN P&L
            if not (abs(pnl) < 1e10):  # Catches NaN, inf, and extreme values
                logger.warning(f"Extreme P&L calculated ({pnl}), using zero")
                return 0.0
            
            return pnl
            
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            logger.warning(f"P&L calculation error: {str(e)}, using zero P&L")
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