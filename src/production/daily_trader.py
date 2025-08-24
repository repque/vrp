"""
Production Daily VRP Trader

Enhanced VRPTrader with state persistence for daily production operations.
Handles loading previous state, processing new data, generating signals,
and saving updated state for next day's operations.
"""

import asyncio
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from ..models.data_models import (
    MarketData, VRPState, TradingSignal, VolatilityData, 
    ModelState, Position, TransitionMatrix
)
from ..config.settings import Settings, get_settings
from ..utils.exceptions import (
    DataQualityError, CalculationError, InsufficientDataError, 
    ModelStateError, PersistenceError
)
from ..persistence.state_manager import StateManager, SystemState
from ..trading.position_manager import PositionManager, PositionState
from services import VRPCalculator, SignalGenerator, BacktestEngine

logger = logging.getLogger(__name__)


class DailyVRPTrader:
    """
    Production VRP Trading System with State Persistence
    
    Extends VRPTrader with daily production capabilities:
    - Loads previous day's state on startup
    - Maintains position tracking across sessions  
    - Persists model state and transitions
    - Handles daily signal generation workflow
    - Saves complete state for next day
    
    Usage:
        trader = DailyVRPTrader()
        await trader.initialize_for_date(date.today())
        signal = trader.process_daily_data('new_data.csv')
        await trader.finalize_daily_processing()
    """
    
    def __init__(self, 
                 settings: Optional[Settings] = None,
                 database_path: str = "vrp_model.db"):
        """
        Initialize daily VRP trader with state persistence.
        
        Args:
            settings: System configuration
            database_path: Path to SQLite database for state persistence
        """
        self.settings = settings or get_settings()
        self.database_path = database_path
        
        # Core components
        self.calculator = VRPCalculator(self.settings)
        self.signal_generator = SignalGenerator(self.settings)
        self.backtest_engine = BacktestEngine(self.calculator, self.signal_generator)
        
        # State management
        self.state_manager = StateManager(database_path)
        self.position_manager = PositionManager()
        self.system_state: Optional[SystemState] = None
        
        # Current session data
        self.current_data: Optional[List[MarketData]] = None
        self.current_date: Optional[date] = None
        self.latest_signal: Optional[TradingSignal] = None
        
        logger.info(f"DailyVRPTrader initialized with database: {database_path}")
    
    async def initialize_for_date(self, target_date: date) -> bool:
        """
        Initialize trader for processing target date.
        
        Loads previous state, validates consistency, and prepares
        for daily processing workflow.
        
        Args:
            target_date: Date to process
            
        Returns:
            True if initialization successful
        """
        try:
            logger.info(f"Initializing DailyVRPTrader for {target_date}")
            
            # Check if already processed
            if not await self.state_manager.should_process_date(target_date):
                logger.warning(f"Date {target_date} already processed - skipping")
                return False
            
            # Load complete previous state
            self.system_state = await self.state_manager.load_complete_state()
            self.current_date = target_date
            
            # Initialize position manager with previous state
            if self.system_state.current_position_state:
                self.position_manager.current_position = PositionState(
                    self.system_state.current_position_state
                )
                self.position_manager.position_size = self.system_state.current_position_size
            
            # Validate state consistency
            is_valid, issues = await self.state_manager.validate_state_consistency()
            if not is_valid:
                logger.warning(f"State consistency issues found: {issues}")
            
            # Create backup before processing
            backup_id = self.state_manager.create_model_state_backup()
            if backup_id:
                logger.info(f"State backup created: {backup_id}")
            
            self._log_initialization_summary()
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize for {target_date}: {str(e)}")
            return False
    
    def process_daily_data(self, data_source) -> Optional[TradingSignal]:
        """
        Process new daily market data and generate trading signal.
        
        Args:
            data_source: CSV file path or DataFrame with daily data
            
        Returns:
            Generated trading signal or None if failed
        """
        try:
            logger.info(f"Processing daily data for {self.current_date}")
            
            # Load and validate new data
            if not self._load_daily_data(data_source):
                return None
            
            # Generate volatility data for analysis
            volatility_data = self.calculator.generate_volatility_data(self.current_data)
            
            # Update current state from latest data
            current_vrp_state = volatility_data[-1].vrp_state
            current_vrp = float(volatility_data[-1].vrp)
            
            try:
                vrp_state_name = getattr(current_vrp_state, 'name', str(current_vrp_state))
                logger.info(f"Current VRP state: {vrp_state_name} (ratio: {current_vrp:.3f})")
            except Exception as log_e:
                logger.info(f"Current VRP state: {type(current_vrp_state).__name__} (ratio: {current_vrp:.3f})")
            
            # Generate trading signal using enhanced Markov model
            if self.system_state.model_state:
                # Use existing model state for better predictions
                self.signal_generator.initialize_from_state(self.system_state.model_state)
            
            trading_signal = self.signal_generator.generate_signal(volatility_data)
            self.latest_signal = trading_signal
            
            # Process signal through position manager
            position_action = self.position_manager.process_signal(trading_signal)
            
            try:
                signal_type_name = getattr(trading_signal.signal_type, 'name', str(trading_signal.signal_type))
                confidence_score = getattr(trading_signal, 'confidence_score', 0.0)
                position_action_name = 'HOLD'
                if isinstance(position_action, dict) and 'action' in position_action:
                    position_action_name = str(position_action['action'])
                logger.info(f"Generated signal: {signal_type_name} | "
                           f"Position action: {position_action_name} | "
                           f"Confidence: {confidence_score:.3f}")
            except Exception as log_e:
                logger.info(f"Generated signal: {type(trading_signal).__name__} completed")
            
            return trading_signal
            
        except (InsufficientDataError, CalculationError, ModelStateError) as e:
            logger.error(f"Failed to process daily data: {str(e)}")
            return None
        except Exception as e:
            try:
                error_message = str(e)
            except Exception:
                error_message = "Unknown error"
            logger.error(f"Unexpected error processing daily data: {error_message}")
            return None
    
    async def finalize_daily_processing(self) -> bool:
        """
        Finalize daily processing and save complete state.
        
        Saves model state, positions, signals, and system state
        for next day's operations.
        
        Returns:
            True if finalization successful
        """
        try:
            logger.info(f"Finalizing daily processing for {self.current_date}")
            
            if not self.latest_signal or not self.current_date:
                logger.error("Cannot finalize - no signal generated or date not set")
                return False
            
            # Build updated model state
            updated_model_state = self._build_updated_model_state()
            
            # Get current active positions
            active_positions = self._get_updated_positions()
            
            # Save complete state
            await self.state_manager.save_complete_state(
                model_state=updated_model_state,
                active_positions=active_positions,
                latest_signal=self.latest_signal,
                processed_date=self.current_date,
                position_state=self.position_manager.current_position.value,
                position_size=self.position_manager.position_size
            )
            
            # Log completion summary
            self._log_completion_summary()
            
            logger.info(f"Daily processing completed successfully for {self.current_date}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to finalize daily processing: {str(e)}")
            return False
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of current system state.
        
        Returns:
            Dictionary with current status information
        """
        try:
            positions_summary = self.state_manager.get_current_positions_summary()
            
            status = {
                'current_date': self.current_date,
                'position_state': self.position_manager.current_position.value,
                'position_size': float(self.position_manager.position_size),
                'latest_signal': self.latest_signal.signal_type if self.latest_signal else None,
                'signal_confidence': float(self.latest_signal.confidence_score) if self.latest_signal else None,
                'positions_summary': positions_summary,
                'data_loaded': self.current_data is not None,
                'total_data_points': len(self.current_data) if self.current_data else 0
            }
            
            if self.system_state:
                status.update({
                    'has_model_state': self.system_state.model_state is not None,
                    'total_signals_generated': self.system_state.total_signals_generated,
                    'last_processed_date': self.system_state.last_processed_date
                })
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get current status: {str(e)}")
            return {'error': str(e)}
    
    def _load_daily_data(self, data_source) -> bool:
        """Load and validate daily market data."""
        try:
            import pandas as pd
            
            # Handle different input types
            if isinstance(data_source, str):
                df = pd.read_csv(data_source)
                logger.info(f"Loaded data from {data_source}")
            elif isinstance(data_source, pd.DataFrame):
                df = data_source.copy()
                logger.info("Loaded DataFrame")
            else:
                logger.error("Data source must be CSV file path or DataFrame")
                return False
            
            # Validate required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'iv']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Convert data types and validate
            df['date'] = pd.to_datetime(df['date'])
            
            validated_data = []
            for _, row in df.iterrows():
                try:
                    market_data = MarketData(
                        date=row['date'],
                        open=Decimal(str(row['open'])),
                        high=Decimal(str(row['high'])),
                        low=Decimal(str(row['low'])),
                        close=Decimal(str(row['close'])),
                        volume=int(row['volume']),
                        iv=Decimal(str(row['iv']))
                    )
                    validated_data.append(market_data)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid row {row['date']}: {e}")
                    continue
            
            if not validated_data:
                raise DataQualityError("No valid data rows found after validation")
            
            self.current_data = validated_data
            logger.info(f"Successfully loaded and validated {len(validated_data)} data points")
            return True
            
        except Exception as e:
            logger.error(f"Error loading daily data: {str(e)}")
            return False
    
    def _build_updated_model_state(self) -> ModelState:
        """Build updated model state from current processing."""
        try:
            # Get transition matrix from signal generator
            transition_matrix = self.signal_generator.get_current_transition_matrix()
            
            # Calculate metrics
            current_vrp_state = self.latest_signal.current_state if self.latest_signal else None
            total_observations = len(self.current_data) if self.current_data else 0
            
            if self.system_state.model_state:
                total_observations += self.system_state.model_state.total_observations
            
            model_state = ModelState(
                last_updated=datetime.now(),
                transition_matrix=transition_matrix,
                current_vrp_state=current_vrp_state,
                total_observations=total_observations,
                version="1.0.0",
                data_start_date=self.current_data[0].date if self.current_data else datetime.now(),
                data_end_date=self.current_data[-1].date if self.current_data else datetime.now(),
                recent_accuracy=None,  # TODO: Implement accuracy tracking
                signal_count_by_type=self._calculate_signal_counts()
            )
            
            return model_state
            
        except Exception as e:
            logger.error(f"Failed to build updated model state: {str(e)}")
            # Return minimal valid state
            from decimal import Decimal
            from datetime import date
            
            default_transition_matrix = TransitionMatrix(
                matrix=[[Decimal('0.2')] * 5 for _ in range(5)],  # Default uniform probabilities
                observation_count=0,
                window_start=date.today(),
                window_end=date.today(),
                last_updated=datetime.now()
            )
            
            return ModelState(
                last_updated=datetime.now(),
                transition_matrix=default_transition_matrix,
                total_observations=0,
                version="1.0.0",
                data_start_date=datetime.now(),
                data_end_date=datetime.now(),
                signal_count_by_type={}
            )
    
    def _get_updated_positions(self) -> List[Position]:
        """
        Get current active positions from position manager.
        
        Retrieves active positions from the position manager and creates Position
        objects for persistence. This ensures positions are tracked across daily runs.
        
        Returns:
            List of current active Position objects
        """
        try:
            active_positions = []
            
            # Load existing active positions from database
            existing_positions = self.state_manager.persistence.load_active_positions()
            
            # Process position manager state changes
            current_position = self.position_manager.current_position
            current_size = self.position_manager.position_size
            
            logger.info(f"[POSITION UPDATE] Current state: {current_position.value}, size: {current_size}")
            
            # Handle position state changes
            if current_position.value == "FLAT":
                # Close all existing positions
                for position in existing_positions:
                    if position.is_active:
                        position.is_active = False
                        position.exit_date = self.current_date
                        position.exit_price = self._get_current_market_price()
                        position.realized_pnl = self._calculate_position_pnl(position)
                        logger.info(f"[POSITION UPDATE] Closed position {position.position_id}")
                        
                        # Save the closed position
                        self.state_manager.persistence.save_position(position)
            
            elif current_position.value in ["LONG_VOL", "SHORT_VOL"] and current_size > 0:
                # Check if we need to create a new position or update existing
                existing_active = [p for p in existing_positions if p.is_active]
                
                if not existing_active:
                    # Create new position
                    new_position = self._create_new_position(
                        position_type=current_position.value,
                        size=current_size,
                        entry_date=self.current_date,
                        signal=self.latest_signal
                    )
                    active_positions.append(new_position)
                    logger.info(f"[POSITION UPDATE] Created new position: {new_position.position_id}")
                
                else:
                    # Update existing position size if changed
                    for position in existing_active:
                        if abs(float(position.position_size) - float(current_size)) > 0.01:
                            position.position_size = Decimal(str(current_size))
                            logger.info(f"[POSITION UPDATE] Updated position size: {position.position_id}")
                        active_positions.append(position)
            
            # Include any other active positions that weren't modified
            for position in existing_positions:
                if position.is_active and position not in active_positions:
                    active_positions.append(position)
            
            logger.info(f"[POSITION UPDATE] Total active positions: {len(active_positions)}")
            return active_positions
            
        except Exception as e:
            logger.error(f"Failed to get updated positions: {str(e)}")
            # Return existing positions if update fails
            try:
                return self.state_manager.persistence.load_active_positions()
            except Exception:
                return []
    
    def _calculate_signal_counts(self) -> Dict[str, int]:
        """Calculate signal counts by type."""
        counts = {"BUY_VOL": 0, "SELL_VOL": 0, "HOLD": 0}
        if self.latest_signal:
            counts[self.latest_signal.signal_type] = 1
        return counts
    
    def _log_initialization_summary(self) -> None:
        """Log initialization summary for monitoring."""
        logger.info(f"[DAILY INIT] Date: {self.current_date}")
        logger.info(f"[DAILY INIT] Position: {self.position_manager.current_position.value} "
                   f"(size: {self.position_manager.position_size})")
        
        if self.system_state.model_state:
            logger.info(f"[DAILY INIT] Model: {self.system_state.model_state.total_observations} observations")
        else:
            logger.info("[DAILY INIT] Model: No previous state")
        
        logger.info(f"[DAILY INIT] Active positions: {len(self.system_state.active_positions)}")
    
    def _log_completion_summary(self) -> None:
        """Log completion summary for monitoring."""
        if self.latest_signal:
            try:
                signal_type = getattr(self.latest_signal.signal_type, 'name', str(self.latest_signal.signal_type))
                confidence_score = getattr(self.latest_signal, 'confidence_score', 0.0)
                logger.info(f"[DAILY COMPLETE] Signal: {signal_type} "
                           f"(confidence: {confidence_score:.3f})")
            except Exception:
                logger.info(f"[DAILY COMPLETE] Signal processing completed")
        
        logger.info(f"[DAILY COMPLETE] Final position: {self.position_manager.current_position.value} "
                   f"(size: {self.position_manager.position_size})")
        logger.info(f"[DAILY COMPLETE] Data processed: {len(self.current_data) if self.current_data else 0} points")
    
    def _create_new_position(self, position_type: str, size: Decimal, 
                           entry_date: date, signal: TradingSignal) -> Position:
        """
        Create a new trading position.
        
        Args:
            position_type: Type of position (LONG_VOL, SHORT_VOL)
            size: Position size
            entry_date: Entry date
            signal: Trading signal that triggered the position
            
        Returns:
            New Position object
        """
        import uuid
        
        position_id = f"POS_{position_type}_{entry_date.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        entry_price = self._get_current_market_price()
        
        position = Position(
            position_id=position_id,
            symbol="VRP",  # VRP strategy symbol
            position_type=position_type,
            entry_date=entry_date,
            entry_signal=signal,
            position_size=size,
            is_active=True,
            exit_date=None,
            exit_price=None,
            realized_pnl=None
        )
        
        logger.info(f"[POSITION CREATE] Created position {position_id} - {position_type} size {size}")
        return position
    
    def _get_current_market_price(self) -> Optional[Decimal]:
        """
        Get current market price for position valuation.
        
        Uses the latest close price from current data.
        
        Returns:
            Current market price or None if unavailable
        """
        try:
            if self.current_data and len(self.current_data) > 0:
                return self.current_data[-1].close
            return None
        except Exception as e:
            logger.warning(f"Failed to get current market price: {e}")
            return None
    
    def _calculate_position_pnl(self, position: Position) -> Optional[Decimal]:
        """
        Calculate realized P&L for a closed position.
        
        Args:
            position: Position to calculate P&L for
            
        Returns:
            Realized P&L or None if cannot calculate
        """
        try:
            if not position.exit_price:
                return None
            
            entry_price = self._get_position_entry_price(position)
            if not entry_price:
                return None
            
            # Calculate P&L based on position type
            if position.position_type == "LONG_VOL":
                pnl = (position.exit_price - entry_price) * position.position_size
            elif position.position_type == "SHORT_VOL":
                pnl = (entry_price - position.exit_price) * position.position_size
            else:
                logger.warning(f"Unknown position type: {position.position_type}")
                return None
            
            logger.info(f"[POSITION PNL] Position {position.position_id}: {pnl}")
            return pnl
            
        except Exception as e:
            logger.error(f"Failed to calculate P&L for position {position.position_id}: {e}")
            return None
    
    def _get_position_entry_price(self, position: Position) -> Optional[Decimal]:
        """
        Get entry price for a position from its entry signal.
        
        Args:
            position: Position to get entry price for
            
        Returns:
            Entry price or current market price as fallback
        """
        try:
            # Try to get price from entry signal data
            if position.entry_signal and hasattr(position.entry_signal, 'market_price'):
                return Decimal(str(position.entry_signal.market_price))
            
            # Fallback to current market price
            return self._get_current_market_price()
            
        except Exception as e:
            logger.warning(f"Failed to get entry price for position {position.position_id}: {e}")
            return self._get_current_market_price()


# Convenience function for daily operations
async def run_daily_processing(target_date: date, 
                              data_source,
                              database_path: str = "vrp_model.db") -> Optional[TradingSignal]:
    """
    Run complete daily VRP processing workflow.
    
    Args:
        target_date: Date to process
        data_source: Market data CSV or DataFrame
        database_path: Database path for state persistence
        
    Returns:
        Generated trading signal or None if failed
    """
    try:
        trader = DailyVRPTrader(database_path=database_path)
        # Initialize for target date
        if not await trader.initialize_for_date(target_date):
            return None
        
        # Process daily data
        signal = trader.process_daily_data(data_source)
        if not signal:
            return None
        
        # Finalize and save state
        if not await trader.finalize_daily_processing():
            logger.error("Failed to finalize daily processing")
        
        return signal
        
    except Exception as e:
        logger.error(f"Daily processing workflow failed: {str(e)}")
        return None