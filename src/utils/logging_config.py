"""
Logging configuration for VRP Trading System.

This module sets up comprehensive logging with proper formatting, rotation,
and different handlers for various log levels. It implements structured
logging with context information for debugging and monitoring.
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

from src.config.settings import Settings
from src.models.data_models import (
    ModelPrediction,
    PerformanceMetrics,
    Position,
    TradingSignal,
    VRPState,
)


class VRPFormatter(logging.Formatter):
    """
    Custom formatter for VRP Trading System logs.

    Provides structured logging with context information and proper
    formatting for different log levels and message types.
    """

    def __init__(self):
        """Initialize formatter with structured format."""
        super().__init__()

        # Different formats for different log levels
        self.formats = {
            logging.DEBUG: "[%(asctime)s] [%(name)s] [DEBUG] [%(funcName)s:%(lineno)d] %(message)s",
            logging.INFO: "[%(asctime)s] [%(name)s] [INFO] %(message)s",
            logging.WARNING: (
                "[%(asctime)s] [%(name)s] [WARNING] "
                "[%(funcName)s:%(lineno)d] %(message)s"
            ),
            logging.ERROR: "[%(asctime)s] [%(name)s] [ERROR] [%(funcName)s:%(lineno)d] %(message)s",
            logging.CRITICAL: (
                "[%(asctime)s] [%(name)s] [CRITICAL] "
                "[%(funcName)s:%(lineno)d] %(message)s"
            )}

        # Date format
        self.datefmt = "%Y-%m-%d %H:%M:%S"

    def format(self, record):
        """
        Format log record with appropriate format for log level.

        Args:
            record: LogRecord to format

        Returns:
            Formatted log string
        """
        # Get format for this log level
        log_fmt = self.formats.get(record.levelno, self.formats[logging.INFO])

        # Create formatter with appropriate format
        formatter = logging.Formatter(log_fmt, self.datefmt)

        # Add custom attributes if present
        if hasattr(record, 'component'):
            record.name = f"{record.name}.{record.component}"

        if hasattr(record, 'trade_id'):
            record.msg = f"[TRADE:{record.trade_id}] {record.msg}"

        if hasattr(record, 'position_id'):
            record.msg = f"[POS:{record.position_id}] {record.msg}"

        if hasattr(record, 'signal_id'):
            record.msg = f"[SIG:{record.signal_id}] {record.msg}"

        return formatter.format(record)


class VRPLogger:
    """
    Centralized logging system for VRP Trading System.

    Provides structured logging with different handlers for different
    purposes (file, console, alerts) and convenience methods for
    logging trading-specific events.
    """

    def __init__(self, config: Settings):
        """
        Initialize logging system with configuration.

        Args:
            config: System configuration containing logging settings
        """
        self.config = config
        self.loggers: Dict[str, logging.Logger] = {}

        # Setup root logger
        self._setup_root_logger()

        # Setup specialized loggers
        self._setup_specialized_loggers()

    def _setup_root_logger(self) -> None:
        """Setup root logger with handlers and formatters."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.logging.log_level))

        # Clear any existing handlers
        root_logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(VRPFormatter())
        root_logger.addHandler(console_handler)

        # File handler with rotation
        if self.config.logging.log_file_path:
            log_file = Path(self.config.logging.log_file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=self.config.logging.log_max_bytes,
                backupCount=self.config.logging.log_backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(VRPFormatter())
            root_logger.addHandler(file_handler)

        # Error file handler (errors and above only)
        if self.config.logging.log_file_path:
            error_log_file = log_file.parent / f"{log_file.stem}_errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                filename=error_log_file,
                maxBytes=self.config.logging.log_max_bytes,
                backupCount=self.config.logging.log_backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(VRPFormatter())
            root_logger.addHandler(error_handler)

    def _setup_specialized_loggers(self) -> None:
        """Setup specialized loggers for different components."""
        # Trading logger
        self.loggers['trading'] = self._create_logger(
            'vrp.trading',
            'trading_activities.log'
        )

        # Model logger
        self.loggers['model'] = self._create_logger(
            'vrp.model',
            'model_activities.log'
        )

        # Data logger
        self.loggers['data'] = self._create_logger(
            'vrp.data',
            'data_activities.log'
        )

        # Risk logger
        self.loggers['risk'] = self._create_logger(
            'vrp.risk',
            'risk_activities.log'
        )

        # Performance logger
        self.loggers['performance'] = self._create_logger(
            'vrp.performance',
            'performance_activities.log'
        )

        # Alert logger (for important events that need attention)
        self.loggers['alerts'] = self._create_logger(
            'vrp.alerts',
            'alerts.log'
        )

    def _create_logger(self, name: str, filename: str) -> logging.Logger:
        """
        Create specialized logger with dedicated file handler.

        Args:
            name: Logger name
            filename: Log filename

        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if self.config.logging.log_file_path:
            log_dir = Path(self.config.logging.log_file_path).parent
            log_file = log_dir / filename

            handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=self.config.logging.log_max_bytes,
                backupCount=self.config.logging.log_backup_count
            )
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(VRPFormatter())
            logger.addHandler(handler)

        return logger

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get logger by name.

        Args:
            name: Logger name

        Returns:
            Logger instance
        """
        if name in self.loggers:
            return self.loggers[name]
        else:
            return logging.getLogger(name)

    def log_state_transition(
        self,
        old_state: VRPState,
        new_state: VRPState,
        reason: str,
        vrp_value: float = None,
        confidence: float = None
    ) -> None:
        """
        Log VRP state transitions with detailed context.

        Args:
            old_state: Previous VRP state
            new_state: New VRP state
            reason: Reason for state change
            vrp_value: Current VRP value
            confidence: Model confidence score
        """
        logger = self.loggers['model']

        extra_info = []
        if vrp_value is not None:
            extra_info.append(f"VRP={vrp_value:.3f}")
        if confidence is not None:
            extra_info.append(f"Confidence={confidence:.3f}")

        extra_str = f" ({', '.join(extra_info)})" if extra_info else ""

        logger.info(
            f"[STATE_TRANSITION] {old_state.name} -> {new_state.name}: {reason}{extra_str}",
            extra={'component': 'markov_chain'}
        )

    def log_signal_generation(
        self,
        signal: TradingSignal,
        signal_id: str = None
    ) -> None:
        """
        Log trading signal generation with full context.

        Args:
            signal: Generated trading signal
            signal_id: Optional signal identifier
        """
        logger = self.loggers['trading']

        logger.info(
            f"[SIGNAL_GENERATED] {signal.signal_type} | "
            f"State: {signal.current_state.name} -> {signal.predicted_state.name} | "
            f"Strength: {signal.signal_strength:.3f} | "
            f"Confidence: {signal.confidence_score:.3f} | "
            f"Size: {signal.risk_adjusted_size:.3f} | "
            f"Reason: {signal.reason}",
            extra={'component': 'signal_generator', 'signal_id': signal_id}
        )

    def log_position_action(
        self,
        action: str,
        position: Position,
        reason: str,
        pnl: float = None
    ) -> None:
        """
        Log position management actions.

        Args:
            action: Action taken (OPEN, CLOSE, UPDATE)
            position: Position object
            reason: Reason for action
            pnl: P&L if applicable
        """
        logger = self.loggers['trading']

        pnl_str = f" | P&L: ${pnl:,.2f}" if pnl is not None else ""

        logger.info(
            f"[POSITION_{action.upper()}] {position.symbol} | "
            f"Type: {position.position_type} | "
            f"Size: {position.position_size} | "
            f"Entry: {position.entry_date} | "
            f"Reason: {reason}{pnl_str}",
            extra={'component': 'portfolio_manager', 'position_id': position.position_id}
        )

    def log_risk_event(
        self,
        event_type: str,
        severity: str,
        details: str,
        position_id: str = None,
        risk_metrics: dict = None
    ) -> None:
        """
        Log risk management events.

        Args:
            event_type: Type of risk event
            severity: Event severity (LOW, MEDIUM, HIGH, CRITICAL)
            details: Event details
            position_id: Related position ID
            risk_metrics: Risk metrics context
        """
        logger = self.loggers['risk']

        log_level = {
            'LOW': logging.INFO,
            'MEDIUM': logging.WARNING,
            'HIGH': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }.get(severity, logging.WARNING)

        extra_context = {'component': 'risk_manager'}
        if position_id:
            extra_context['position_id'] = position_id

        metrics_str = ""
        if risk_metrics:
            metrics_str = " | " + ", ".join(f"{k}={v}" for k, v in risk_metrics.items())

        logger.log(
            log_level,
            f"[RISK_{event_type.upper()}] {details}{metrics_str}",
            extra=extra_context
        )

    def log_performance_update(
        self,
        metrics: PerformanceMetrics,
        period: str = "DAILY"
    ) -> None:
        """
        Log performance metrics updates.

        Args:
            metrics: Performance metrics
            period: Performance period (DAILY, WEEKLY, MONTHLY)
        """
        logger = self.loggers['performance']

        logger.info(
            f"[PERFORMANCE_{period}] "
            f"Return: {metrics.total_return:.2%} | "
            f"Sharpe: {metrics.sharpe_ratio:.2f} | "
            f"Max DD: {metrics.max_drawdown:.2%} | "
            f"Win Rate: {metrics.win_rate:.2%} | "
            f"Profit Factor: {metrics.profit_factor:.2f} | "
            f"Trades: {metrics.total_trades}",
            extra={'component': 'performance_analyzer'}
        )

    def log_model_prediction(
        self,
        prediction: ModelPrediction,
        prediction_id: str = None
    ) -> None:
        """
        Log model predictions with confidence metrics.

        Args:
            prediction: Model prediction
            prediction_id: Optional prediction identifier
        """
        logger = self.loggers['model']

        logger.info(
            f"[MODEL_PREDICTION] "
            f"{prediction.current_state.name} -> {prediction.predicted_state.name} | "
            f"Prob: {prediction.transition_probability:.3f} | "
            f"Confidence: {prediction.confidence_score:.3f} | "
            f"Entropy: {prediction.entropy:.3f} | "
            f"Data Quality: {prediction.data_quality_score:.3f}",
            extra={'component': 'markov_chain', 'prediction_id': prediction_id}
        )

    def log_data_quality_issue(
        self,
        issue_type: str,
        details: str,
        severity: str = "MEDIUM",
        data_source: str = None,
        affected_dates: list = None
    ) -> None:
        """
        Log data quality issues.

        Args:
            issue_type: Type of data quality issue
            details: Issue details
            severity: Issue severity
            data_source: Affected data source
            affected_dates: List of affected dates
        """
        logger = self.loggers['data']

        log_level = {
            'LOW': logging.INFO,
            'MEDIUM': logging.WARNING,
            'HIGH': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }.get(severity, logging.WARNING)

        extra_info = []
        if data_source:
            extra_info.append(f"Source: {data_source}")
        if affected_dates:
            extra_info.append(f"Dates: {len(affected_dates)} affected")

        extra_str = f" | {' | '.join(extra_info)}" if extra_info else ""

        logger.log(
            log_level,
            f"[DATA_QUALITY_{issue_type.upper()}] {details}{extra_str}",
            extra={'component': 'data_validator'}
        )

    def log_system_startup(self, components: list, config_summary: dict) -> None:
        """
        Log system startup information.

        Args:
            components: List of initialized components
            config_summary: Configuration summary
        """
        logger = logging.getLogger('vrp.system')

        logger.info("=" * 60)
        logger.info("VRP TRADING SYSTEM STARTUP")
        logger.info("=" * 60)

        logger.info(f"System Environment: {config_summary.get('environment', 'unknown')}")
        logger.info(f"Trading Mode: {config_summary.get('trading_mode', 'unknown')}")
        logger.info(f"Log Level: {config_summary.get('log_level', 'unknown')}")

        logger.info("Initialized Components:")
        for component in components:
            logger.info(f"  ✓ {component}")

        logger.info("Configuration Summary:")
        for key, value in config_summary.items():
            logger.info(f"  {key}: {value}")

        logger.info("=" * 60)

    def log_system_shutdown(self, reason: str = "Normal shutdown") -> None:
        """
        Log system shutdown information.

        Args:
            reason: Reason for shutdown
        """
        logger = logging.getLogger('vrp.system')

        logger.info("=" * 60)
        logger.info(f"VRP TRADING SYSTEM SHUTDOWN: {reason}")
        logger.info(f"Shutdown Time: {datetime.now()}")
        logger.info("=" * 60)

    def log_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "HIGH",
        additional_context: dict = None
    ) -> None:
        """
        Log alerts that require immediate attention.

        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity
            additional_context: Additional context information
        """
        logger = self.loggers['alerts']

        log_level = {
            'LOW': logging.INFO,
            'MEDIUM': logging.WARNING,
            'HIGH': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }.get(severity, logging.ERROR)

        context_str = ""
        if additional_context:
            context_str = " | " + ", ".join(f"{k}={v}" for k, v in additional_context.items())

        logger.log(
            log_level,
            f"[ALERT_{alert_type.upper()}] {message}{context_str}",
            extra={'component': 'alerting_system'}
        )


def setup_logging(config: Settings) -> VRPLogger:
    """
    Setup logging system for VRP Trading System.

    Args:
        config: System configuration

    Returns:
        Configured VRPLogger instance
    """
    return VRPLogger(config)
