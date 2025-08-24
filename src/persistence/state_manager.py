"""
State Manager for VRP Daily Operations

Orchestrates loading and saving of all system state for daily production runs.
Handles model state, positions, and system operational state with proper
error handling and recovery mechanisms.
"""

import asyncio
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from .database import SQLitePersistence
from ..models.data_models import ModelState, Position, TradingSignal, TransitionMatrix, VRPState
from ..utils.exceptions import PersistenceError, ModelStateError

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Complete system state for daily operations."""
    last_processed_date: Optional[date] = None
    current_position_state: str = "FLAT"
    current_position_size: Decimal = Decimal('0.0')
    active_positions: List[Position] = None
    model_state: Optional[ModelState] = None
    total_signals_generated: int = 0
    
    def __post_init__(self):
        if self.active_positions is None:
            self.active_positions = []


class StateManager:
    """
    Orchestrates all state persistence for daily VRP operations.
    
    Provides high-level interface for saving/loading complete system state
    including model parameters, trading positions, and operational metrics.
    Handles error recovery and state validation.
    """
    
    def __init__(self, database_path: str = "vrp_model.db"):
        """
        Initialize state manager.
        
        Args:
            database_path: Path to SQLite database file
        """
        self.persistence = SQLitePersistence(database_path)
        logger.info(f"StateManager initialized with database: {database_path}")
    
    async def load_complete_state(self) -> SystemState:
        """
        Load complete system state for daily startup.
        
        Returns:
            SystemState with all persisted data or defaults if first run
        """
        try:
            logger.info("Loading complete system state for daily startup")
            
            # Load all state components in parallel
            model_state_task = self.persistence.load_model_state()
            active_positions_task = asyncio.create_task(
                asyncio.to_thread(self.persistence.load_active_positions)
            )
            system_state_task = asyncio.create_task(
                asyncio.to_thread(self.persistence.load_system_state)
            )
            
            # Wait for all loads to complete
            model_state = await model_state_task
            active_positions = await active_positions_task
            system_data = await system_state_task
            
            # Build complete state
            state = SystemState()
            state.model_state = model_state
            state.active_positions = active_positions
            
            if system_data:
                state.last_processed_date = system_data['last_processed_date']
                state.current_position_state = system_data['current_position_state']
                state.current_position_size = Decimal(str(system_data['current_position_size']))
                state.total_signals_generated = system_data['total_signals_generated']
            
            self._log_loaded_state(state)
            return state
            
        except Exception as e:
            logger.error(f"Failed to load complete state: {str(e)}")
            # Return default state on error
            return SystemState()
    
    async def save_complete_state(self, 
                                  model_state: Optional[ModelState] = None,
                                  active_positions: Optional[List[Position]] = None,
                                  latest_signal: Optional[TradingSignal] = None,
                                  processed_date: Optional[date] = None,
                                  position_state: str = "FLAT",
                                  position_size: Decimal = Decimal('0.0')) -> None:
        """
        Save complete system state after daily processing.
        
        Args:
            model_state: Updated model state with transition matrix
            active_positions: Current active positions
            latest_signal: Latest trading signal generated
            processed_date: Date that was processed
            position_state: Current position state (FLAT, LONG_VOL, SHORT_VOL)  
            position_size: Current position size
        """
        try:
            logger.info("Saving complete system state after daily processing")
            
            save_tasks = []
            
            # Save model state
            if model_state:
                save_tasks.append(self.persistence.save_model_state(model_state))
            
            # Save latest signal
            if latest_signal:
                save_tasks.append(self.persistence.save_trading_signal(latest_signal))
            
            # Save positions
            if active_positions:
                for position in active_positions:
                    save_tasks.append(
                        asyncio.to_thread(self.persistence.save_position, position)
                    )
            
            # Save system state
            if processed_date:
                save_tasks.append(
                    asyncio.to_thread(
                        self.persistence.update_system_state,
                        processed_date,
                        position_state,
                        float(position_size)
                    )
                )
            
            # Execute all saves in parallel
            if save_tasks:
                await asyncio.gather(*save_tasks)
            
            logger.info("Complete system state saved successfully")
            
        except Exception as e:
            raise PersistenceError("save_complete_state", f"Failed to save complete state: {str(e)}")
    
    async def should_process_date(self, target_date: date) -> bool:
        """
        Check if target date should be processed based on last processed date.
        
        Args:
            target_date: Date to potentially process
            
        Returns:
            True if should process, False if already processed
        """
        try:
            system_data = await asyncio.to_thread(self.persistence.load_system_state)
            
            if not system_data:
                logger.info("No previous processing history - will process")
                return True
            
            last_processed = system_data['last_processed_date']
            should_process = target_date > last_processed
            
            if should_process:
                logger.info(f"Will process {target_date} (last processed: {last_processed})")
            else:
                logger.info(f"Already processed {target_date} - skipping")
            
            return should_process
            
        except Exception as e:
            logger.warning(f"Error checking processing history: {e} - will process")
            return True
    
    def get_current_positions_summary(self) -> Dict[str, Any]:
        """
        Get summary of current position state for monitoring.
        
        Returns:
            Dictionary with position summary information
        """
        try:
            active_positions = self.persistence.load_active_positions()
            system_data = self.persistence.load_system_state()
            
            total_exposure = sum(float(pos.position_size) for pos in active_positions)
            
            summary = {
                'total_active_positions': len(active_positions),
                'total_exposure': total_exposure,
                'current_position_state': system_data['current_position_state'] if system_data else 'FLAT',
                'positions_by_type': {},
                'last_processed_date': system_data['last_processed_date'] if system_data else None
            }
            
            # Group positions by type
            for position in active_positions:
                pos_type = position.position_type
                if pos_type not in summary['positions_by_type']:
                    summary['positions_by_type'][pos_type] = {'count': 0, 'total_size': 0.0}
                
                summary['positions_by_type'][pos_type]['count'] += 1
                summary['positions_by_type'][pos_type]['total_size'] += float(position.position_size)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get positions summary: {str(e)}")
            return {'error': str(e)}
    
    def create_model_state_backup(self) -> Optional[str]:
        """
        Create backup of current model state.
        
        Returns:
            Backup identifier or None if failed
        """
        try:
            import time
            import uuid
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Add microseconds and random suffix for uniqueness
            unique_suffix = f"{datetime.now().microsecond:06d}_{uuid.uuid4().hex[:6]}"
            backup_path = f"vrp_model_backup_{timestamp}_{unique_suffix}.db"
            
            # Simple file copy for SQLite backup
            import shutil
            shutil.copy2(self.persistence.database_path, backup_path)
            
            logger.info(f"Model state backed up to: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create model state backup: {str(e)}")
            return None
    
    async def validate_state_consistency(self) -> Tuple[bool, List[str]]:
        """
        Validate consistency of stored state data.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Load all state components
            system_state = await self.load_complete_state()
            
            # Check model state consistency
            if system_state.model_state:
                model = system_state.model_state
                if model.data_end_date < model.data_start_date:
                    issues.append("Model data end date is before start date")
                
                if model.total_observations <= 0:
                    issues.append("Model has zero or negative observations")
            
            # Check position consistency
            total_exposure = Decimal('0.0')
            for position in system_state.active_positions:
                if position.position_size <= 0:
                    issues.append(f"Position {position.position_id} has non-positive size")
                
                if position.is_active and position.exit_date:
                    issues.append(f"Position {position.position_id} marked active but has exit date")
                
                total_exposure += position.position_size
            
            # Check system state consistency
            if abs(total_exposure - system_state.current_position_size) > Decimal('0.01'):
                issues.append("System position size doesn't match sum of active positions")
            
            if system_state.last_processed_date and system_state.last_processed_date > date.today():
                issues.append("Last processed date is in the future")
            
            is_valid = len(issues) == 0
            if is_valid:
                logger.info("State consistency validation passed")
            else:
                logger.warning(f"State consistency validation failed: {len(issues)} issues found")
            
            return is_valid, issues
            
        except Exception as e:
            error_msg = f"State validation error: {str(e)}"
            logger.error(error_msg)
            return False, [error_msg]
    
    def _log_loaded_state(self, state: SystemState) -> None:
        """Log summary of loaded state for monitoring."""
        if state.model_state:
            logger.info(f"[STATE] Loaded model: {state.model_state.total_observations} observations, "
                       f"last updated {state.model_state.last_updated}")
        else:
            logger.info("[STATE] No previous model state found")
        
        logger.info(f"[STATE] Active positions: {len(state.active_positions)}")
        logger.info(f"[STATE] Position state: {state.current_position_state} "
                   f"(size: {state.current_position_size})")
        
        if state.last_processed_date:
            logger.info(f"[STATE] Last processed: {state.last_processed_date}")
        else:
            logger.info("[STATE] No previous processing history")
        
        logger.info(f"[STATE] Total signals generated: {state.total_signals_generated}")


# Convenience functions for common operations
async def load_daily_state(database_path: str = "vrp_model.db") -> SystemState:
    """Load complete system state for daily operations."""
    state_manager = StateManager(database_path)
    return await state_manager.load_complete_state()


async def save_daily_state(model_state: ModelState, 
                          latest_signal: TradingSignal,
                          processed_date: date,
                          database_path: str = "vrp_model.db") -> None:
    """Save daily processing results."""
    state_manager = StateManager(database_path)
    await state_manager.save_complete_state(
        model_state=model_state,
        latest_signal=latest_signal,
        processed_date=processed_date
    )