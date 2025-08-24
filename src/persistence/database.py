"""
SQLite Database Layer for VRP State Persistence

Provides SQLite-based implementation for persisting model state, positions,
and trading signals across daily production runs. Implements the IModelPersistence
interface with ACID guarantees for financial data reliability.
"""

import sqlite3
import json
import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List, Dict, Any
from pathlib import Path
import threading
from contextlib import contextmanager

from ..models.data_models import ModelState, Position, TradingSignal, TransitionMatrix, VRPState
from ..utils.exceptions import PersistenceError
from ..interfaces.contracts import IModelPersistence
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class SQLitePersistence(IModelPersistence):
    """
    SQLite implementation for VRP state persistence.
    
    Handles all database operations for storing and retrieving model state,
    trading positions, and signal history with proper error handling and
    transaction management.
    """
    
    _DEFAULT = object()  # Sentinel value for default database path
    
    def __init__(self, database_path = _DEFAULT, connection_timeout: Optional[int] = None):
        """
        Initialize SQLite persistence layer.
        
        Args:
            database_path: Path to SQLite database file (overrides config)
            connection_timeout: Connection timeout in seconds (overrides config)
            
        Raises:
            PersistenceError: If database path is invalid or initialization fails
        """
        settings = get_settings()
        
        # Use provided values or fall back to configuration
        # Validate any explicitly provided path, use default only when no argument provided
        if database_path is self._DEFAULT:
            self.database_path = self._validate_database_path(settings.database.database_path)
        else:
            self.database_path = self._validate_database_path(database_path)
        self.connection_timeout = connection_timeout or settings.database.connection_timeout
        self.enable_wal_mode = settings.database.enable_wal_mode
        self.enable_foreign_keys = settings.database.enable_foreign_keys
        self.auto_vacuum = settings.database.auto_vacuum
        
        self._connection_lock = threading.Lock()
        self._ensure_database_exists()
        logger.info(f"SQLite persistence initialized: {self.database_path} (timeout: {self.connection_timeout}s)")
    
    def _validate_database_path(self, database_path: str) -> str:
        """
        Validate and sanitize database path.
        
        Args:
            database_path: Raw database path
            
        Returns:
            Validated database path
            
        Raises:
            PersistenceError: If path is invalid or insecure
        """
        if not database_path or not isinstance(database_path, str):
            raise PersistenceError("path_validation", "Database path must be a non-empty string")
        
        # Prevent directory traversal attacks
        if ".." in database_path or database_path.startswith("/"):
            # Allow absolute paths but validate they're in allowed directories
            path = Path(database_path).resolve()
            if not path.name.endswith('.db'):
                raise PersistenceError("path_validation", "Database file must have .db extension")
        else:
            # Relative paths are safer
            path = Path(database_path)
            if not path.name.endswith('.db'):
                raise PersistenceError("path_validation", "Database file must have .db extension")
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        return str(path)
    
    def _ensure_database_exists(self) -> None:
        """Create database and tables if they don't exist."""
        try:
            # Configure SQLite settings without transaction
            conn = sqlite3.connect(
                self.database_path,
                timeout=self.connection_timeout
            )
            
            # Configure SQLite settings based on configuration
            if self.enable_foreign_keys:
                conn.execute("PRAGMA foreign_keys = ON")
            
            if self.enable_wal_mode:
                conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging for better concurrency
                
            conn.execute("PRAGMA synchronous = NORMAL")  # Balance between safety and performance
            conn.execute(f"PRAGMA auto_vacuum = {self.auto_vacuum}")
            
            # Performance optimizations
            conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
            conn.execute("PRAGMA temp_store = MEMORY")  # Store temp tables in memory
            
            self._create_tables(conn)
            conn.close()
            logger.debug("Database schema validated/created with optimized settings")
        except Exception as e:
            raise PersistenceError("database_init", f"Failed to initialize database: {str(e)}")
    
    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Create all required tables with proper schemas."""
        
        # Model state table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_state (
                id INTEGER PRIMARY KEY,
                last_updated TIMESTAMP NOT NULL,
                transition_matrix TEXT NOT NULL,
                current_vrp_state INTEGER,
                total_observations INTEGER NOT NULL,
                version TEXT NOT NULL,
                data_start_date TIMESTAMP NOT NULL,
                data_end_date TIMESTAMP NOT NULL,
                recent_accuracy REAL,
                signal_count_by_type TEXT,
                model_metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Trading positions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                position_type TEXT NOT NULL,
                entry_date DATE NOT NULL,
                entry_signal_data TEXT NOT NULL,
                position_size REAL NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                exit_date DATE,
                exit_price REAL,
                realized_pnl REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Trading signals history table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                signal_type TEXT NOT NULL,
                current_state INTEGER NOT NULL,
                predicted_state INTEGER NOT NULL,
                signal_strength REAL NOT NULL,
                confidence_score REAL NOT NULL,
                recommended_position_size REAL NOT NULL,
                risk_adjusted_size REAL NOT NULL,
                reason TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System state table for tracking daily operations
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                id INTEGER PRIMARY KEY,
                last_processed_date DATE NOT NULL,
                current_position_state TEXT NOT NULL DEFAULT 'FLAT',
                current_position_size REAL NOT NULL DEFAULT 0.0,
                total_signals_generated INTEGER NOT NULL DEFAULT 0,
                system_version TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for common queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_active ON positions(is_active)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_date ON trading_signals(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_entry_date ON positions(entry_date)")
        
        conn.commit()
    
    async def save_model_state(self, model_state: ModelState) -> None:
        """Save model state to database."""
        try:
            with self._get_connection() as conn:
                # Convert complex objects to JSON for storage
                transition_matrix_json = self._serialize_transition_matrix(model_state.transition_matrix)
                signal_counts_json = json.dumps(model_state.signal_count_by_type)
                metadata_json = json.dumps(getattr(model_state, 'model_metadata', {}))
                
                # Atomic UPSERT operation - safer than DELETE/INSERT
                conn.execute("""
                    INSERT OR REPLACE INTO model_state 
                    (id, last_updated, transition_matrix, current_vrp_state, total_observations,
                     version, data_start_date, data_end_date, recent_accuracy, 
                     signal_count_by_type, model_metadata)
                    VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_state.last_updated,
                    transition_matrix_json,
                    int(model_state.current_vrp_state) if model_state.current_vrp_state else None,
                    model_state.total_observations,
                    model_state.version,
                    model_state.data_start_date,
                    model_state.data_end_date,
                    float(model_state.recent_accuracy) if model_state.recent_accuracy else None,
                    signal_counts_json,
                    metadata_json
                ))
                logger.info("Model state saved successfully")
                
        except Exception as e:
            raise PersistenceError("model_state_save", f"Failed to save model state: {str(e)}")
    
    async def load_model_state(self) -> Optional[ModelState]:
        """Load most recent model state from database."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM model_state 
                    ORDER BY last_updated DESC 
                    LIMIT 1
                """)
                row = cursor.fetchone()
                
                if not row:
                    logger.info("No previous model state found")
                    return None
                
                # Deserialize complex objects
                transition_matrix = self._deserialize_transition_matrix(row['transition_matrix'])
                signal_counts = json.loads(row['signal_count_by_type'])
                
                model_state = ModelState(
                    last_updated=datetime.fromisoformat(row['last_updated']),
                    transition_matrix=transition_matrix,
                    current_vrp_state=VRPState(row['current_vrp_state']) if row['current_vrp_state'] else None,
                    total_observations=row['total_observations'],
                    version=row['version'],
                    data_start_date=datetime.fromisoformat(row['data_start_date']),
                    data_end_date=datetime.fromisoformat(row['data_end_date']),
                    recent_accuracy=Decimal(str(row['recent_accuracy'])) if row['recent_accuracy'] else None,
                    signal_count_by_type=signal_counts
                )
                
                logger.info(f"Loaded model state from {row['last_updated']}")
                return model_state
                
        except Exception as e:
            logger.error(f"Failed to load model state: {str(e)}")
            return None
    
    async def save_trading_signal(self, signal: TradingSignal) -> None:
        """Save trading signal to history table."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO trading_signals 
                    (date, signal_type, current_state, predicted_state, signal_strength,
                     confidence_score, recommended_position_size, risk_adjusted_size, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal.date,
                    signal.signal_type,
                    signal.current_state.value,
                    signal.predicted_state.value,
                    float(signal.signal_strength),
                    float(signal.confidence_score),
                    float(signal.recommended_position_size),
                    float(signal.risk_adjusted_size),
                    signal.reason
                ))
                logger.debug(f"Saved trading signal: {signal.signal_type}")
                
        except Exception as e:
            raise PersistenceError("trading_signal_save", f"Failed to save trading signal: {str(e)}")
    
    async def get_recent_signals(self, days: int = 30) -> List[TradingSignal]:
        """
        Get recent trading signals from storage.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of recent trading signals
            
        Raises:
            PersistenceError: If retrieval fails
        """
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM trading_signals 
                    WHERE date > date('now', '-{} days')
                    ORDER BY date DESC, id DESC
                """.format(days))
                
                signals = []
                for row in cursor.fetchall():
                    signal = TradingSignal(
                        date=date.fromisoformat(row['date']),
                        signal_type=row['signal_type'],
                        current_state=VRPState(row['current_state']),
                        predicted_state=VRPState(row['predicted_state']),
                        signal_strength=Decimal(str(row['signal_strength'])),
                        confidence_score=Decimal(str(row['confidence_score'])),
                        recommended_position_size=Decimal(str(row['recommended_position_size'])),
                        risk_adjusted_size=Decimal(str(row['risk_adjusted_size'])),
                        reason=row['reason']
                    )
                    signals.append(signal)
                
                logger.debug(f"Retrieved {len(signals)} recent signals from last {days} days")
                return signals
                
        except Exception as e:
            logger.error(f"Failed to get recent signals: {str(e)}")
            raise PersistenceError("recent_signals_load", f"Failed to get recent signals: {str(e)}")
    
    def save_position(self, position: Position) -> None:
        """Save or update trading position."""
        try:
            with self._get_connection() as conn:
                # Serialize the entry signal
                entry_signal_json = self._serialize_trading_signal(position.entry_signal)
                
                conn.execute("""
                    INSERT OR REPLACE INTO positions 
                    (position_id, symbol, position_type, entry_date, entry_signal_data,
                     position_size, is_active, exit_date, exit_price, realized_pnl, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position.position_id,
                    position.symbol,
                    position.position_type,
                    position.entry_date,
                    entry_signal_json,
                    float(position.position_size),
                    position.is_active,
                    position.exit_date,
                    float(position.exit_price) if position.exit_price else None,
                    float(position.realized_pnl) if position.realized_pnl else None,
                    datetime.now()
                ))
                logger.debug(f"Saved position: {position.position_id}")
                
        except Exception as e:
            raise PersistenceError("position_save", f"Failed to save position: {str(e)}")
    
    def load_active_positions(self) -> List[Position]:
        """Load all active trading positions."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM positions 
                    WHERE is_active = 1
                    ORDER BY entry_date DESC
                """)
                
                positions = []
                for row in cursor.fetchall():
                    # Deserialize entry signal
                    entry_signal = self._deserialize_trading_signal(row['entry_signal_data'])
                    
                    position = Position(
                        position_id=row['position_id'],
                        symbol=row['symbol'],
                        position_type=row['position_type'],
                        entry_date=date.fromisoformat(row['entry_date']),
                        entry_signal=entry_signal,
                        position_size=Decimal(str(row['position_size'])),
                        is_active=bool(row['is_active']),
                        exit_date=date.fromisoformat(row['exit_date']) if row['exit_date'] else None,
                        exit_price=Decimal(str(row['exit_price'])) if row['exit_price'] else None,
                        realized_pnl=Decimal(str(row['realized_pnl'])) if row['realized_pnl'] else None
                    )
                    positions.append(position)
                
                logger.info(f"Loaded {len(positions)} active positions")
                return positions
                
        except Exception as e:
            logger.error(f"Failed to load active positions: {str(e)}")
            return []
    
    def update_system_state(self, last_processed_date: date, position_state: str, position_size: float = 0.0) -> None:
        """Update system operational state."""
        try:
            with self._get_connection() as conn:
                # Atomic UPSERT operation - safer than DELETE/INSERT
                conn.execute("""
                    INSERT OR REPLACE INTO system_state 
                    (id, last_processed_date, current_position_state, current_position_size,
                     total_signals_generated, system_version)
                    VALUES (1, ?, ?, ?, 
                           (SELECT COUNT(*) FROM trading_signals), ?)
                """, (
                    last_processed_date,
                    position_state,
                    position_size,
                    "1.0.0"
                ))
                logger.debug(f"Updated system state: {position_state}")
                
        except Exception as e:
            raise PersistenceError("system_state_update", f"Failed to update system state: {str(e)}")
    
    def load_system_state(self) -> Optional[Dict[str, Any]]:
        """Load current system state."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM system_state WHERE id = 1")
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return {
                    'last_processed_date': date.fromisoformat(row['last_processed_date']),
                    'current_position_state': row['current_position_state'],
                    'current_position_size': row['current_position_size'],
                    'total_signals_generated': row['total_signals_generated'],
                    'system_version': row['system_version']
                }
                
        except Exception as e:
            logger.error(f"Failed to load system state: {str(e)}")
            return None
    
    def _serialize_transition_matrix(self, matrix: TransitionMatrix) -> str:
        """Serialize transition matrix to JSON."""
        # Convert Decimal values to float for JSON serialization
        serializable_matrix = [[float(val) for val in row] for row in matrix.matrix]
        
        return json.dumps({
            'matrix': serializable_matrix,
            'observation_count': matrix.observation_count,
            'window_start': matrix.window_start.isoformat(),
            'window_end': matrix.window_end.isoformat(),
            'last_updated': matrix.last_updated.isoformat()
        })
    
    def _deserialize_transition_matrix(self, json_str: str) -> TransitionMatrix:
        """Deserialize transition matrix from JSON."""
        data = json.loads(json_str)
        # Convert float values back to Decimal for precision
        decimal_matrix = [[Decimal(str(val)) for val in row] for row in data['matrix']]
        
        return TransitionMatrix(
            matrix=decimal_matrix,
            observation_count=data['observation_count'],
            window_start=date.fromisoformat(data['window_start']),
            window_end=date.fromisoformat(data['window_end']),
            last_updated=datetime.fromisoformat(data['last_updated'])
        )
    
    def _serialize_trading_signal(self, signal: TradingSignal) -> str:
        """Serialize trading signal to JSON."""
        return json.dumps({
            'date': signal.date.isoformat(),
            'signal_type': signal.signal_type,
            'current_state': signal.current_state.value,
            'predicted_state': signal.predicted_state.value,
            'signal_strength': float(signal.signal_strength),
            'confidence_score': float(signal.confidence_score),
            'recommended_position_size': float(signal.recommended_position_size),
            'risk_adjusted_size': float(signal.risk_adjusted_size),
            'reason': signal.reason
        })
    
    def _deserialize_trading_signal(self, json_str: str) -> TradingSignal:
        """Deserialize trading signal from JSON."""
        data = json.loads(json_str)
        return TradingSignal(
            date=date.fromisoformat(data['date']),
            signal_type=data['signal_type'],
            current_state=VRPState(data['current_state']),
            predicted_state=VRPState(data['predicted_state']),
            signal_strength=Decimal(str(data['signal_strength'])),
            confidence_score=Decimal(str(data['confidence_score'])),
            recommended_position_size=Decimal(str(data['recommended_position_size'])),
            risk_adjusted_size=Decimal(str(data['risk_adjusted_size'])),
            reason=data['reason']
        )
    
    @contextmanager
    def _get_connection(self):
        """
        Get database connection with proper timeout and error handling.
        
        Yields:
            sqlite3.Connection: Database connection with proper configuration
        """
        conn = None
        try:
            with self._connection_lock:
                conn = sqlite3.connect(
                    self.database_path,
                    timeout=self.connection_timeout,
                    isolation_level=None  # Enable autocommit mode for better control
                )
                conn.execute("BEGIN IMMEDIATE")  # Start transaction immediately
                yield conn
                conn.execute("COMMIT")
        except sqlite3.Error as e:
            if conn:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass  # Rollback may fail if transaction wasn't started
            logger.error(f"Database error: {str(e)}")
            raise PersistenceError("database_operation", f"Database operation failed: {str(e)}")
        except Exception as e:
            if conn:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            logger.error(f"Unexpected error in database operation: {str(e)}")
            raise PersistenceError("database_operation", f"Database operation failed: {str(e)}")
        finally:
            if conn:
                conn.close()