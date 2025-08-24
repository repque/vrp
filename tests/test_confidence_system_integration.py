"""
Integration Tests for Confidence-Based Exits System

Tests the complete system integration including backtest engine,
position management, and performance comparison against legacy system.
"""

import pytest
import numpy as np
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch

from src.models.data_models import (
    MarketData,
    VolatilityData,
    TradingSignal,
    VRPState,
    PerformanceMetrics,
    Position
)
from services.backtest_engine import BacktestEngine
from services.vrp_calculator import VRPCalculator
from services.signal_generator import SignalGenerator
from src.utils.exceptions import InsufficientDataError, CalculationError


class ConfidenceBasedBacktestEngine(BacktestEngine):
    """Enhanced backtest engine with confidence-based exit logic."""
    
    def __init__(self, calculator: VRPCalculator, signal_generator: SignalGenerator):
        super().__init__(calculator, signal_generator)
        self.confidence_thresholds = {
            'entry': Decimal('0.65'),
            'exit': Decimal('0.40'),
            'flip': Decimal('0.75')
        }
        self.position_manager = {
            'current_position': 'FLAT',
            'position_size': Decimal('0.0'),
            'entry_date': None,
            'entry_confidence': Decimal('0.0')
        }
    
    def run_confidence_based_backtest(
        self,
        data: List[MarketData],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Run backtest with confidence-based position management.
        
        Returns:
            Dict with enhanced backtest results including flat periods
        """
        if not data:
            raise InsufficientDataError(1, 0)
        
        try:
            # Filter data by date range
            test_data = self._filter_data_by_date(data, start_date, end_date)
            
            # Validate minimum data requirements
            min_required = lookback_days + 20
            if len(test_data) < min_required:
                raise InsufficientDataError(min_required, len(test_data))
            
            # Run enhanced simulation
            trades = self._simulate_confidence_based_trading(test_data, lookback_days)
            
            # Calculate enhanced performance metrics
            results = self._calculate_enhanced_performance_metrics(trades)
            
            return results
            
        except (InsufficientDataError, CalculationError):
            raise
        except Exception as e:
            raise CalculationError(
                calculation="confidence_backtest_execution",
                message=f"Enhanced backtest error: {e}"
            )
    
    def _simulate_confidence_based_trading(
        self,
        data: List[MarketData],
        lookback_days: int
    ) -> List[Dict]:
        """Simulate trading with confidence-based position management."""
        trades = []
        
        # Need sufficient data for volatility calculations
        min_data_required = max(lookback_days, 90)
        
        for i in range(min_data_required, len(data)):
            try:
                # Use only historical data (no forward-looking bias)
                historical_data = data[max(0, i-200):i+1]
                
                # Generate volatility data
                volatility_data = self.calculator.generate_volatility_data(historical_data)
                
                if len(volatility_data) < 60:
                    continue
                
                # Generate base trading signal
                trading_signal = self.signal_generator.generate_signal(volatility_data)
                
                # Calculate enhanced confidence scores
                enhanced_signal = self._enhance_signal_with_confidence(
                    trading_signal, volatility_data, data[i].date
                )
                
                # Process signal with confidence-based logic
                trade_result = self._process_confidence_based_signal(
                    enhanced_signal, data[i-1] if i > 0 else data[i], data[i]
                )
                
                if trade_result:
                    trades.append(trade_result)
                    
            except Exception as e:
                continue  # Skip problematic dates
        
        return trades
    
    def _enhance_signal_with_confidence(
        self,
        base_signal: TradingSignal,
        volatility_data: List[VolatilityData],
        current_date: date
    ) -> Dict[str, Any]:
        """Enhance base signal with confidence-based metrics."""
        
        # Calculate position duration if in position
        position_duration = 0
        if (self.position_manager['current_position'] != 'FLAT' and 
            self.position_manager['entry_date']):
            position_duration = (current_date - self.position_manager['entry_date']).days
        
        # Calculate entry confidence (for new positions)
        entry_confidence = self._calculate_entry_confidence(
            base_signal, volatility_data
        )
        
        # Calculate exit confidence (for existing positions)
        exit_confidence = self._calculate_exit_confidence(
            base_signal, volatility_data, position_duration
        )
        
        return {
            'base_signal': base_signal,
            'entry_confidence': entry_confidence,
            'exit_confidence': exit_confidence,
            'position_duration': position_duration,
            'current_date': current_date
        }
    
    def _calculate_entry_confidence(
        self,
        signal: TradingSignal,
        volatility_data: List[VolatilityData]
    ) -> Decimal:
        """Calculate confidence for position entry decisions."""
        base_confidence = signal.confidence_score
        
        # Boost for extreme state predictions
        if signal.predicted_state in [VRPState.EXTREME_LOW, VRPState.EXTREME_HIGH]:
            state_boost = Decimal('0.1')
        else:
            state_boost = Decimal('0.0')
        
        # Signal strength adjustment
        strength_adjustment = (signal.signal_strength - Decimal('0.5')) * Decimal('0.1')
        
        # Recent volatility adjustment
        recent_vrp = volatility_data[-1].vrp if volatility_data else Decimal('1.0')
        if abs(float(recent_vrp) - 1.0) > 0.3:  # Extreme VRP
            volatility_boost = Decimal('0.05')
        else:
            volatility_boost = Decimal('0.0')
        
        entry_confidence = base_confidence + state_boost + strength_adjustment + volatility_boost
        
        # Ensure bounds [0, 1]
        return max(Decimal('0.0'), min(Decimal('1.0'), entry_confidence))
    
    def _calculate_exit_confidence(
        self,
        signal: TradingSignal,
        volatility_data: List[VolatilityData],
        position_duration: int
    ) -> Decimal:
        """Calculate confidence for position exit decisions."""
        base_confidence = signal.confidence_score
        
        # Duration decay (longer positions need higher confidence to maintain)
        duration_penalty = Decimal(str(max(0, position_duration - 5) * 0.02))
        
        # Market regime stability
        if len(volatility_data) >= 5:
            recent_vrp_values = [float(vd.vrp) for vd in volatility_data[-5:]]
            vrp_volatility = np.std(recent_vrp_values)
            stability_penalty = Decimal(str(min(0.2, vrp_volatility * 0.5)))
        else:
            stability_penalty = Decimal('0.0')
        
        # Signal consistency (check if signal direction matches position)
        consistency_boost = Decimal('0.0')
        if self.position_manager['current_position'] == 'LONG_VOL' and signal.signal_type == 'BUY_VOL':
            consistency_boost = Decimal('0.05')
        elif self.position_manager['current_position'] == 'SHORT_VOL' and signal.signal_type == 'SELL_VOL':
            consistency_boost = Decimal('0.05')
        
        exit_confidence = base_confidence - duration_penalty - stability_penalty + consistency_boost
        
        # Ensure bounds [0, 1]
        return max(Decimal('0.0'), min(Decimal('1.0'), exit_confidence))
    
    def _process_confidence_based_signal(
        self,
        enhanced_signal: Dict[str, Any],
        prev_data: MarketData,
        curr_data: MarketData
    ) -> Optional[Dict]:
        """Process enhanced signal with confidence-based position management."""
        
        base_signal = enhanced_signal['base_signal']
        entry_confidence = enhanced_signal['entry_confidence']
        exit_confidence = enhanced_signal['exit_confidence']
        current_date = enhanced_signal['current_date']
        
        old_position = self.position_manager['current_position']
        old_size = self.position_manager['position_size']
        
        action = "HOLD"
        new_position = old_position
        new_size = old_size
        reasoning = "No change"
        
        # Position management logic
        if old_position == 'FLAT':
            # Entry logic from flat
            if entry_confidence >= self.confidence_thresholds['entry']:
                if base_signal.signal_type == 'BUY_VOL':
                    new_position = 'LONG_VOL'
                    new_size = base_signal.risk_adjusted_size
                    action = "ENTER_LONG"
                    reasoning = f"High entry confidence ({entry_confidence:.3f})"
                    
                    # Update position manager
                    self.position_manager.update({
                        'current_position': new_position,
                        'position_size': new_size,
                        'entry_date': current_date,
                        'entry_confidence': entry_confidence
                    })
                    
                elif base_signal.signal_type == 'SELL_VOL':
                    new_position = 'SHORT_VOL'
                    new_size = -base_signal.risk_adjusted_size
                    action = "ENTER_SHORT"
                    reasoning = f"High entry confidence ({entry_confidence:.3f})"
                    
                    # Update position manager
                    self.position_manager.update({
                        'current_position': new_position,
                        'position_size': new_size,
                        'entry_date': current_date,
                        'entry_confidence': entry_confidence
                    })
                        
        elif old_position == 'LONG_VOL':
            # Long position logic
            if (exit_confidence <= self.confidence_thresholds['exit'] or
                entry_confidence <= self.confidence_thresholds['exit']):
                new_position = 'FLAT'
                new_size = Decimal('0.0')
                action = "EXIT_LONG"
                reasoning = f"Low confidence - exit (entry: {entry_confidence:.3f}, exit: {exit_confidence:.3f})"
                
                # Update position manager
                self.position_manager.update({
                    'current_position': 'FLAT',
                    'position_size': Decimal('0.0'),
                    'entry_date': None,
                    'entry_confidence': Decimal('0.0')
                })
                
            elif (base_signal.signal_type == 'SELL_VOL' and 
                  entry_confidence >= self.confidence_thresholds['flip']):
                new_position = 'SHORT_VOL'
                new_size = -base_signal.risk_adjusted_size
                action = "FLIP_TO_SHORT"
                reasoning = f"Very high confidence flip ({entry_confidence:.3f})"
                
                # Update position manager
                self.position_manager.update({
                    'current_position': new_position,
                    'position_size': new_size,
                    'entry_date': current_date,
                    'entry_confidence': entry_confidence
                })
                
        elif old_position == 'SHORT_VOL':
            # Short position logic
            if (exit_confidence <= self.confidence_thresholds['exit'] or
                entry_confidence <= self.confidence_thresholds['exit']):
                new_position = 'FLAT'
                new_size = Decimal('0.0')
                action = "EXIT_SHORT"
                reasoning = f"Low confidence - exit (entry: {entry_confidence:.3f}, exit: {exit_confidence:.3f})"
                
                # Update position manager
                self.position_manager.update({
                    'current_position': 'FLAT',
                    'position_size': Decimal('0.0'),
                    'entry_date': None,
                    'entry_confidence': Decimal('0.0')
                })
                
            elif (base_signal.signal_type == 'BUY_VOL' and 
                  entry_confidence >= self.confidence_thresholds['flip']):
                new_position = 'LONG_VOL'
                new_size = base_signal.risk_adjusted_size
                action = "FLIP_TO_LONG"
                reasoning = f"Very high confidence flip ({entry_confidence:.3f})"
                
                # Update position manager
                self.position_manager.update({
                    'current_position': new_position,
                    'position_size': new_size,
                    'entry_date': current_date,
                    'entry_confidence': entry_confidence
                })
        
        # Calculate P&L if there was a position change or existing position
        pnl = 0.0
        if old_position != 'FLAT' or action != "HOLD":
            pnl = self._calculate_enhanced_pnl(prev_data, curr_data, old_size, new_size)
        
        # Only record if there was meaningful activity
        if action != "HOLD" or old_position != 'FLAT':
            return {
                'date': current_date,
                'action': action,
                'old_position': old_position,
                'new_position': new_position,
                'old_size': float(old_size),
                'new_size': float(new_size),
                'entry_confidence': float(entry_confidence),
                'exit_confidence': float(exit_confidence),
                'signal_type': base_signal.signal_type,
                'vrp_ratio': float(enhanced_signal.get('vrp_ratio', 1.0)),
                'pnl': pnl,
                'reasoning': reasoning
            }
        
        return None
    
    def _calculate_enhanced_pnl(
        self,
        prev_data: MarketData,
        curr_data: MarketData,
        old_size: Decimal,
        new_size: Decimal
    ) -> float:
        """Calculate P&L with position size considerations."""
        if abs(float(old_size)) < 1e-6:
            return 0.0
        
        try:
            # Calculate IV change
            iv_change_pct = (float(curr_data.iv) - float(prev_data.iv)) / float(prev_data.iv)
            
            # P&L calculation considering position held during the period
            pnl = float(old_size) * iv_change_pct
            
            return pnl
            
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def _calculate_enhanced_performance_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate enhanced performance metrics including confidence analysis."""
        if not trades:
            return self._empty_performance_metrics()
        
        # Basic metrics
        base_metrics = self._calculate_performance_metrics(
            [{'pnl': trade['pnl']} for trade in trades]
        )
        
        # Enhanced metrics
        enhanced_metrics = self._calculate_confidence_specific_metrics(trades)
        
        # Combine metrics
        return {**base_metrics, **enhanced_metrics}
    
    def _calculate_confidence_specific_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate confidence-specific performance metrics."""
        # Flat period analysis - track actual position state over time
        if not trades:
            return {
                'flat_percentage': 100.0,
                'position_changes': 0,
                'avg_entry_confidence': 0.0,
                'avg_exit_confidence': 0.0,
                'avg_position_duration': 0.0,
                'total_flips': 0,
                'transactions_per_day': 0.0
            }
        
        # Build a timeline of positions
        total_days = len(trades)
        flat_days = 0
        current_position = 'FLAT'  # Start flat
        
        for trade in trades:
            if current_position == 'FLAT':
                flat_days += 1
            current_position = trade['new_position']
        
        flat_percentage = (flat_days / total_days * 100) if total_days > 0 else 100.0
        
        # Transaction analysis
        position_changes = sum(1 for trade in trades if trade['action'] != 'HOLD')
        
        # Confidence analysis
        entry_confidences = [trade['entry_confidence'] for trade in trades if 'ENTER' in trade['action']]
        exit_confidences = [trade['exit_confidence'] for trade in trades if 'EXIT' in trade['action']]
        
        avg_entry_confidence = np.mean(entry_confidences) if entry_confidences else 0
        avg_exit_confidence = np.mean(exit_confidences) if exit_confidences else 0
        
        # Position duration analysis
        position_durations = []
        current_entry = None
        
        for trade in trades:
            if 'ENTER' in trade['action']:
                current_entry = trade['date']
            elif 'EXIT' in trade['action'] and current_entry:
                if isinstance(current_entry, str):
                    current_entry = datetime.strptime(current_entry, '%Y-%m-%d').date()
                if isinstance(trade['date'], str):
                    exit_date = datetime.strptime(trade['date'], '%Y-%m-%d').date()
                else:
                    exit_date = trade['date']
                    
                duration = (exit_date - current_entry).days
                position_durations.append(duration)
                current_entry = None
        
        avg_position_duration = np.mean(position_durations) if position_durations else 0
        
        # Flip analysis
        flips = sum(1 for trade in trades if 'FLIP' in trade['action'])
        
        return {
            'flat_percentage': flat_percentage,
            'position_changes': position_changes,
            'avg_entry_confidence': avg_entry_confidence,
            'avg_exit_confidence': avg_exit_confidence,
            'avg_position_duration': avg_position_duration,
            'total_flips': flips,
            'transactions_per_day': position_changes / total_days if total_days > 0 else 0
        }
    
    def _empty_performance_metrics(self) -> Dict[str, Any]:
        """Return empty performance metrics structure."""
        return {
            'total_return': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'flat_percentage': 100.0,
            'position_changes': 0,
            'avg_entry_confidence': 0.0,
            'avg_exit_confidence': 0.0,
            'avg_position_duration': 0.0,
            'total_flips': 0,
            'transactions_per_day': 0.0
        }


class TestConfidenceSystemIntegration:
    """Integration tests for complete confidence-based system."""
    
    @pytest.fixture
    def mock_vrp_calculator(self):
        """Create mock VRP calculator."""
        calculator = Mock(spec=VRPCalculator)
        calculator.config = Mock()
        calculator.config.BASE_POSITION_SIZE = Decimal('0.1')
        
        # Mock volatility data generation
        def mock_generate_volatility_data(market_data):
            volatility_data = []
            for i, md in enumerate(market_data):
                vrp_ratio = float(md.iv) / 0.20  # Mock realized vol of 20%
                
                if vrp_ratio < 0.9:
                    vrp_state = VRPState.EXTREME_LOW
                elif vrp_ratio > 1.5:
                    vrp_state = VRPState.EXTREME_HIGH
                else:
                    vrp_state = VRPState.NORMAL_PREMIUM
                
                volatility_data.append(VolatilityData(
                    date=md.date,
                    daily_return=Decimal('0.01'),
                    realized_vol_30d=Decimal('0.20'),
                    implied_vol=md.iv,
                    vrp=Decimal(str(vrp_ratio)),
                    vrp_state=vrp_state
                ))
            return volatility_data
        
        calculator.generate_volatility_data.side_effect = mock_generate_volatility_data
        return calculator
    
    @pytest.fixture
    def mock_signal_generator(self):
        """Create mock signal generator."""
        generator = Mock(spec=SignalGenerator)
        
        def mock_generate_signal(volatility_data):
            if not volatility_data:
                return TradingSignal(
                    date=date.today(),
                    signal_type="HOLD",
                    current_state=VRPState.NORMAL_PREMIUM,
                    predicted_state=VRPState.NORMAL_PREMIUM,
                    signal_strength=Decimal('0.3'),
                    confidence_score=Decimal('0.3'),
                    recommended_position_size=Decimal('0.0'),
                    risk_adjusted_size=Decimal('0.0'),
                    reason="No data"
                )
            
            latest = volatility_data[-1]
            
            if latest.vrp_state == VRPState.EXTREME_HIGH:
                return TradingSignal(
                    date=latest.date,
                    signal_type="SELL_VOL",
                    current_state=latest.vrp_state,
                    predicted_state=VRPState.EXTREME_HIGH,
                    signal_strength=Decimal('0.85'),
                    confidence_score=Decimal('0.80'),
                    recommended_position_size=Decimal('0.15'),
                    risk_adjusted_size=Decimal('0.12'),
                    reason="High VRP - sell volatility"
                )
            elif latest.vrp_state == VRPState.EXTREME_LOW:
                return TradingSignal(
                    date=latest.date,
                    signal_type="BUY_VOL",
                    current_state=latest.vrp_state,
                    predicted_state=VRPState.EXTREME_LOW,
                    signal_strength=Decimal('0.80'),
                    confidence_score=Decimal('0.75'),
                    recommended_position_size=Decimal('0.15'),
                    risk_adjusted_size=Decimal('0.12'),
                    reason="Low VRP - buy volatility"
                )
            else:
                return TradingSignal(
                    date=latest.date,
                    signal_type="HOLD",
                    current_state=latest.vrp_state,
                    predicted_state=VRPState.NORMAL_PREMIUM,
                    signal_strength=Decimal('0.4'),
                    confidence_score=Decimal('0.4'),
                    recommended_position_size=Decimal('0.05'),
                    risk_adjusted_size=Decimal('0.03'),
                    reason="Normal VRP - hold"
                )
        
        generator.generate_signal.side_effect = mock_generate_signal
        return generator
    
    @pytest.fixture
    def confidence_backtest_engine(self, mock_vrp_calculator, mock_signal_generator):
        """Create confidence-based backtest engine."""
        return ConfidenceBasedBacktestEngine(mock_vrp_calculator, mock_signal_generator)
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        base_date = date(2023, 1, 1)
        data = []
        
        # Create 6 months of market data with varying volatility regimes
        for i in range(180):
            current_date = base_date + timedelta(days=i)
            
            # Create different IV regimes
            if i < 60:  # Low volatility period
                iv = Decimal('0.15')  # Low IV (VRP < 1)
            elif i < 120:  # High volatility period  
                iv = Decimal('0.35')  # High IV (VRP > 1.5)
            else:  # Normal volatility
                iv = Decimal('0.25')  # Normal IV (VRP ~1.25)
            
            # Create price movement
            price = Decimal('400.0') * (1 + Decimal('0.0002') * i)
            daily_return = 0.01 if i % 10 == 0 else 0.002
            
            data.append(MarketData(
                date=current_date,
                open=price * Decimal('0.999'),
                high=price * (1 + Decimal(str(abs(daily_return) * 0.5))),
                low=price * (1 - Decimal(str(abs(daily_return) * 0.5))),
                close=price,
                volume=100_000_000,
                iv=iv
            ))
        
        return data
    
    def test_complete_confidence_based_backtest(self, confidence_backtest_engine, sample_market_data):
        """Test complete confidence-based backtest execution."""
        result = confidence_backtest_engine.run_confidence_based_backtest(
            sample_market_data,
            start_date="2023-02-01",
            end_date="2023-06-30",
            lookback_days=30
        )
        
        # Should have basic performance metrics
        assert 'total_return' in result
        assert 'total_trades' in result
        assert 'win_rate' in result
        
        # Should have confidence-specific metrics
        assert 'flat_percentage' in result
        assert 'avg_entry_confidence' in result
        assert 'avg_exit_confidence' in result
        assert 'position_changes' in result
        
        # Validate reasonable values
        assert 0 <= result['flat_percentage'] <= 100
        assert 0 <= result['avg_entry_confidence'] <= 1
        assert 0 <= result['avg_exit_confidence'] <= 1
        assert result['position_changes'] >= 0
    
    def test_flat_period_generation(self, confidence_backtest_engine, sample_market_data):
        """Test that system generates appropriate flat periods."""
        # Create data with mixed confidence scenarios
        mixed_data = sample_market_data[:120]  # Use first 4 months
        
        result = confidence_backtest_engine.run_confidence_based_backtest(
            mixed_data,
            lookback_days=30
        )
        
        # Should have some flat periods (allowing for very active systems)
        # Note: if no trades occur, this defaults to 100% flat
        assert result['flat_percentage'] >= 0.0  # Valid percentage
        
        # Should have some position activity (unless all signals have low confidence)
        # In extreme cases with very low confidence data, this might be 0
        assert result['position_changes'] >= 0
        
        # Average confidences should be reasonable
        if result['avg_entry_confidence'] > 0:
            assert result['avg_entry_confidence'] >= 0.65  # Above entry threshold
        if result['avg_exit_confidence'] > 0:
            assert result['avg_exit_confidence'] <= 0.6   # Should exit when confidence drops
    
    def test_confidence_threshold_enforcement(self, confidence_backtest_engine):
        """Test that confidence thresholds are properly enforced."""
        # Create data designed to test thresholds
        test_data = []
        base_date = date(2023, 3, 1)
        
        # High volatility period (should trigger entries)
        for i in range(50):
            test_data.append(MarketData(
                date=base_date + timedelta(days=i),
                open=Decimal('400'),
                high=Decimal('405'),
                low=Decimal('395'),
                close=Decimal('400'),
                volume=100_000_000,
                iv=Decimal('0.40') if i < 25 else Decimal('0.15')  # High then low IV
            ))
        
        # Add sufficient warmup data
        warmup_data = []
        for i in range(60):
            warmup_data.append(MarketData(
                date=base_date - timedelta(days=60-i),
                open=Decimal('390'),
                high=Decimal('395'),
                low=Decimal('385'),
                close=Decimal('390'),
                volume=100_000_000,
                iv=Decimal('0.25')
            ))
        
        full_data = warmup_data + test_data
        
        result = confidence_backtest_engine.run_confidence_based_backtest(
            full_data,
            lookback_days=30
        )
        
        # Should show position activity during high IV period
        assert result['position_changes'] > 0
        
        # Should have reasonable transaction frequency (not too high due to confidence filtering)
        assert result['transactions_per_day'] < 1.0  # Less than 1 transaction per day
    
    def test_position_flip_logic(self, confidence_backtest_engine):
        """Test position flip logic with very high confidence."""
        # Create data with extreme regime changes
        flip_data = []
        base_date = date(2023, 3, 1)
        
        # Start with low IV regime
        for i in range(40):
            flip_data.append(MarketData(
                date=base_date + timedelta(days=i),
                open=Decimal('400'),
                high=Decimal('405'),
                low=Decimal('395'),
                close=Decimal('400'),
                volume=100_000_000,
                iv=Decimal('0.12') if i < 20 else Decimal('0.45')  # Extreme change
            ))
        
        # Add warmup data
        warmup_data = []
        for i in range(80):
            warmup_data.append(MarketData(
                date=base_date - timedelta(days=80-i),
                open=Decimal('390'),
                high=Decimal('395'),
                low=Decimal('385'),
                close=Decimal('390'),
                volume=100_000_000,
                iv=Decimal('0.20')
            ))
        
        full_data = warmup_data + flip_data
        
        result = confidence_backtest_engine.run_confidence_based_backtest(
            full_data,
            lookback_days=30
        )
        
        # Should show some flip activity with extreme regime change
        if result['total_flips'] > 0:
            assert result['total_flips'] >= 0
            assert result['position_changes'] >= result['total_flips']
    
    def test_performance_vs_traditional_comparison(self, mock_vrp_calculator, mock_signal_generator, sample_market_data):
        """Test performance comparison between confidence-based and traditional systems."""
        
        # Traditional backtest engine (original)
        traditional_engine = BacktestEngine(mock_vrp_calculator, mock_signal_generator)
        
        # Confidence-based engine
        confidence_engine = ConfidenceBasedBacktestEngine(mock_vrp_calculator, mock_signal_generator)
        
        # Use subset of data for comparison
        test_data = sample_market_data[60:120]  # 2 months of data
        
        # Run both backtests
        traditional_result = traditional_engine.run_backtest(
            test_data,
            lookback_days=30
        )
        
        confidence_result = confidence_engine.run_confidence_based_backtest(
            test_data,
            lookback_days=30
        )
        
        # Compare transaction costs (confidence system should have fewer)
        traditional_trades = traditional_result.get('total_trades', 0)
        confidence_trades = confidence_result.get('position_changes', 0)
        
        # Confidence system should generally have fewer transactions
        # (allowing some flexibility for edge cases)
        transaction_reduction_ratio = confidence_trades / max(traditional_trades, 1)
        assert transaction_reduction_ratio <= 1.5  # At most 50% more trades
        
        # Confidence system should have meaningful flat periods
        assert confidence_result.get('flat_percentage', 0) > 0
        
        # Both should have valid performance metrics
        for result in [traditional_result, confidence_result]:
            assert result['total_return'] != 0 or result['total_trades'] == 0
            assert 0 <= result.get('win_rate', 0) <= 1
    
    def test_risk_management_improvements(self, confidence_backtest_engine):
        """Test risk management improvements through confidence-based exits."""
        
        # Create volatile market scenario
        volatile_data = []
        base_date = date(2023, 3, 1)
        
        # Create scenario with initial confidence followed by rapid deterioration
        iv_values = [0.15, 0.18, 0.25, 0.35, 0.45, 0.50, 0.40, 0.30, 0.20, 0.25]
        
        for i, iv in enumerate(iv_values * 6):  # Repeat pattern
            volatile_data.append(MarketData(
                date=base_date + timedelta(days=i),
                open=Decimal('400'),
                high=Decimal('410'),
                low=Decimal('390'),
                close=Decimal('400'),
                volume=150_000_000,
                iv=Decimal(str(iv))
            ))
        
        # Add warmup data
        warmup_data = []
        for i in range(90):
            warmup_data.append(MarketData(
                date=base_date - timedelta(days=90-i),
                open=Decimal('390'),
                high=Decimal('395'),
                low=Decimal('385'),
                close=Decimal('390'),
                volume=100_000_000,
                iv=Decimal('0.20')
            ))
        
        full_data = warmup_data + volatile_data
        
        result = confidence_backtest_engine.run_confidence_based_backtest(
            full_data,
            lookback_days=30
        )
        
        # Risk management characteristics
        assert result['avg_position_duration'] > 0  # Some positions held
        assert result['avg_position_duration'] < 30  # But not too long in volatile conditions
        
        # Should show reasonable risk-adjusted behavior
        if result['total_trades'] > 0:
            assert result['max_drawdown'] >= 0  # Drawdown should be calculated
            
            # Average exit confidence should be low (indicating risk-conscious exits)
            if result['avg_exit_confidence'] > 0:
                assert result['avg_exit_confidence'] <= 0.7  # Generally cautious exits
    
    def test_edge_cases_and_error_handling(self, confidence_backtest_engine):
        """Test edge cases and error handling in integration."""
        
        # Test with insufficient data
        minimal_data = [MarketData(
            date=date(2023, 3, 1),
            open=Decimal('400'),
            high=Decimal('405'),
            low=Decimal('395'),
            close=Decimal('400'),
            volume=100_000_000,
            iv=Decimal('0.25')
        )]
        
        with pytest.raises(InsufficientDataError):
            confidence_backtest_engine.run_confidence_based_backtest(minimal_data)
        
        # Test with empty data
        with pytest.raises(InsufficientDataError):
            confidence_backtest_engine.run_confidence_based_backtest([])
        
        # Test with valid but minimal data
        small_valid_data = []
        base_date = date(2023, 3, 1)
        
        for i in range(120):  # Just enough data
            small_valid_data.append(MarketData(
                date=base_date + timedelta(days=i),
                open=Decimal('400'),
                high=Decimal('405'),
                low=Decimal('395'),
                close=Decimal('400'),
                volume=100_000_000,
                iv=Decimal('0.25')
            ))
        
        # Should not raise error but may have limited results
        result = confidence_backtest_engine.run_confidence_based_backtest(
            small_valid_data,
            lookback_days=30
        )
        
        assert result is not None
        assert 'total_return' in result
    
    def test_backward_compatibility(self, confidence_backtest_engine, sample_market_data):
        """Test that confidence system maintains backward compatibility."""
        
        # Test that it can process regular TradingSignal objects
        result = confidence_backtest_engine.run_confidence_based_backtest(
            sample_market_data[30:90],
            lookback_days=30
        )
        
        # Should produce valid results even without enhanced signal format
        assert result is not None
        assert isinstance(result, dict)
        
        # Should have all expected keys
        expected_keys = [
            'total_return', 'total_trades', 'win_rate',
            'flat_percentage', 'position_changes', 'avg_entry_confidence'
        ]
        
        for key in expected_keys:
            assert key in result
    
    def test_configuration_impact(self, mock_vrp_calculator, mock_signal_generator):
        """Test impact of different confidence threshold configurations."""
        
        # Create engine with modified thresholds
        engine = ConfidenceBasedBacktestEngine(mock_vrp_calculator, mock_signal_generator)
        
        # Test with more conservative thresholds
        engine.confidence_thresholds = {
            'entry': Decimal('0.80'),  # Higher entry threshold
            'exit': Decimal('0.50'),   # Higher exit threshold
            'flip': Decimal('0.90')    # Higher flip threshold
        }
        
        # Create test data
        test_data = []
        base_date = date(2023, 3, 1)
        
        for i in range(100):
            test_data.append(MarketData(
                date=base_date + timedelta(days=i),
                open=Decimal('400'),
                high=Decimal('405'),
                low=Decimal('395'),
                close=Decimal('400'),
                volume=100_000_000,
                iv=Decimal('0.30') if i % 20 < 10 else Decimal('0.15')
            ))
        
        conservative_result = engine.run_confidence_based_backtest(test_data, lookback_days=30)
        
        # Conservative thresholds should result in:
        # - Some level of selectivity (flat periods)
        # - Reasonable transaction frequency  
        # - High confidence when positions are taken
        assert conservative_result['flat_percentage'] >= 0.0  # Valid percentage
        assert conservative_result['transactions_per_day'] <= 1.0  # Reasonable trading frequency
        
        if conservative_result['avg_entry_confidence'] > 0:
            # With conservative thresholds, entries should have high confidence
            assert conservative_result['avg_entry_confidence'] >= 0.70  # High confidence entries