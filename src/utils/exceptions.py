"""
Custom exceptions for VRP Markov Chain Trading Model.

This module defines specific exception types to enable precise error handling
and clear communication of failure modes throughout the system.
"""


class VRPModelError(Exception):
    """Base exception for all VRP model errors."""

    def __init__(self, message: str, error_code: str = None):
        """
        Initialize VRP model error.

        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class DataFetchError(VRPModelError):
    """Raised when data fetching from external sources fails."""

    def __init__(self, symbol: str, message: str, error_code: str = "DATA_FETCH_FAILED"):
        """
        Initialize data fetch error.

        Args:
            symbol: Symbol that failed to fetch
            message: Detailed error message
            error_code: Error code for this type of failure
        """
        super().__init__(f"Failed to fetch data for {symbol}: {message}", error_code)
        self.symbol = symbol


class DataQualityError(VRPModelError):
    """Raised when data quality validation fails."""

    def __init__(self, issue: str, error_code: str = "DATA_QUALITY_FAILED"):
        """
        Initialize data quality error.

        Args:
            issue: Description of the data quality issue
            error_code: Error code for this type of failure
        """
        super().__init__(f"Data quality validation failed: {issue}", error_code)


class DataValidationError(VRPModelError):
    """Raised when data validation fails."""

    def __init__(self, message: str, data_field: str = None, error_code: str = "DATA_VALIDATION_FAILED"):
        """
        Initialize data validation error.

        Args:
            message: Detailed error message
            data_field: Optional field that failed validation
            error_code: Error code for this type of failure
        """
        super().__init__(f"Data validation failed: {message}", error_code)
        self.data_field = data_field


class CalculationError(VRPModelError):
    """Raised when mathematical calculations fail."""

    def __init__(self, message: str, calculation: str = None, error_code: str = "CALCULATION_FAILED"):
        """
        Initialize calculation error.

        Args:
            message: Detailed error message or calculation name (for backward compatibility)
            calculation: Optional name of the specific calculation that failed
            error_code: Error code for this type of failure
        """
        if calculation is not None:
            # Traditional usage: CalculationError(message, calculation)
            formatted_message = f"Calculation '{calculation}' failed: {message}"
            self.calculation = calculation
        else:
            # Simple usage: CalculationError("Calculation failed")
            formatted_message = f"Calculation failed: {message}"
            self.calculation = None
        
        super().__init__(formatted_message, error_code)


class ModelStateError(VRPModelError):
    """Raised when model state is invalid or corrupted."""

    def __init__(self, state_issue: str, error_code: str = "MODEL_STATE_INVALID"):
        """
        Initialize model state error.

        Args:
            state_issue: Description of the state issue
            error_code: Error code for this type of failure
        """
        super().__init__(f"Model state error: {state_issue}", error_code)


class ModelError(VRPModelError):
    """Raised when model operations fail."""

    def __init__(self, message: str, error_code: str = "MODEL_ERROR"):
        """
        Initialize model error.

        Args:
            message: Detailed error message
            error_code: Error code for this type of failure
        """
        super().__init__(f"Model error: {message}", error_code)


class PersistenceError(VRPModelError):
    """Raised when persistence operations fail."""

    def __init__(self, operation: str, message: str, error_code: str = "PERSISTENCE_FAILED"):
        """
        Initialize persistence error.

        Args:
            operation: Operation that failed (save, load, etc.)
            message: Detailed error message
            error_code: Error code for this type of failure
        """
        super().__init__(f"Persistence operation '{operation}' failed: {message}", error_code)
        self.operation = operation


class ValidationError(VRPModelError):
    """Raised when data validation fails."""

    def __init__(self, field: str, value: str, message: str, error_code: str = "VALIDATION_FAILED"):
        """
        Initialize validation error.

        Args:
            field: Field that failed validation
            value: Value that failed validation
            message: Detailed error message
            error_code: Error code for this type of failure
        """
        super().__init__(
            f"Validation failed for '{field}' with value '{value}': {message}",
            error_code)
        self.field = field
        self.value = value


class ConfigurationError(VRPModelError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, config_issue: str, error_code: str = "CONFIGURATION_INVALID"):
        """
        Initialize configuration error.

        Args:
            config_issue: Description of the configuration issue
            error_code: Error code for this type of failure
        """
        super().__init__(f"Configuration error: {config_issue}", error_code)


class OperationError(VRPModelError):
    """Raised when high-level operations fail."""

    def __init__(self, operation: str, message: str, error_code: str = "OPERATION_FAILED"):
        """
        Initialize operation error.

        Args:
            operation: Name of the operation that failed
            message: Detailed error message
            error_code: Error code for this type of failure
        """
        super().__init__(f"Operation '{operation}' failed: {message}", error_code)
        self.operation = operation


class InsufficientDataError(VRPModelError):
    """Raised when there is insufficient data for model operations."""

    def __init__(self, message_or_required=None, available: int | None = None, *, 
                 required: int | None = None, error_code: str = "INSUFFICIENT_DATA"):
        """
        Initialize insufficient data error.

        Supports multiple usage patterns:
        1. Positional: InsufficientDataError(required, available)
        2. Keyword: InsufficientDataError(required=10, available=5)
        3. Simple: InsufficientDataError("Custom message")

        Args:
            message_or_required: Either error message (simple usage) or required count (positional usage)
            available: Available number of data points (positional or keyword usage)
            required: Required number of data points (keyword-only usage)
            error_code: Error code for this type of failure
        """
        # Handle keyword-only usage: InsufficientDataError(required=10, available=5)
        if required is not None:
            if message_or_required is not None:
                raise ValueError("Cannot specify both positional 'message_or_required' and keyword 'required'")
            formatted_message = f"Insufficient data: required {required}, available {available or 0}"
            self.required = required
            self.available = available or 0
        elif available is not None:
            # Positional detailed usage: InsufficientDataError(required, available)
            required_count = message_or_required
            formatted_message = f"Insufficient data: required {required_count}, available {available}"
            self.required = required_count
            self.available = available
        else:
            # Simple usage: InsufficientDataError("Not enough data")
            formatted_message = f"Insufficient data: {message_or_required}"
            self.required = None
            self.available = None
        
        super().__init__(formatted_message, error_code)


class ModelConvergenceError(VRPModelError):
    """Raised when model fails to converge or produces unstable results."""

    def __init__(self, metric: str, value: float, threshold: float,
                 error_code: str = "CONVERGENCE_FAILED"):
        """
        Initialize model convergence error.

        Args:
            metric: Name of the convergence metric
            value: Actual value of the metric
            threshold: Required threshold value
            error_code: Error code for this type of failure
        """
        super().__init__(
            f"Model convergence failed: {metric} = {value}, required < {threshold}",
            error_code
        )
        self.metric = metric
        self.value = value
        self.threshold = threshold


class SignalGenerationError(VRPModelError):
    """Raised when trading signal generation fails."""

    def __init__(self, reason: str, error_code: str = "SIGNAL_GENERATION_FAILED"):
        """
        Initialize signal generation error.

        Args:
            reason: Reason why signal generation failed
            error_code: Error code for this type of failure
        """
        super().__init__(f"Signal generation failed: {reason}", error_code)


class RiskViolationError(VRPModelError):
    """Raised when risk management constraints are violated."""

    def __init__(self, violation: str, current_value: float = None, limit: float = None, 
                 error_code: str = "RISK_VIOLATION"):
        """
        Initialize risk violation error.

        Args:
            violation: Description of the risk violation
            current_value: Current value that triggered the violation
            limit: Risk limit that was violated
            error_code: Error code for this type of failure
        """
        if current_value is not None and limit is not None:
            message = f"Risk violation: {violation} (current: {current_value}, limit: {limit})"
        else:
            message = f"Risk violation: {violation}"
        super().__init__(message, error_code)
        self.violation = violation
        self.current_value = current_value
        self.limit = limit


class PerformanceThresholdError(VRPModelError):
    """Raised when model performance falls below required thresholds."""

    def __init__(self, metric: str, actual: float, required: float,
                 error_code: str = "PERFORMANCE_THRESHOLD_FAILED"):
        """
        Initialize performance threshold error.

        Args:
            metric: Performance metric that failed
            actual: Actual performance value
            required: Required performance threshold
            error_code: Error code for this type of failure
        """
        super().__init__(
            f"Performance threshold failed: {metric} = {actual:.4f}, required >= {required:.4f}",
            error_code
        )
        self.metric = metric
        self.actual = actual
        self.required = required


class ExternalServiceError(VRPModelError):
    """Raised when external service calls fail."""

    def __init__(self, service: str, message: str, error_code: str = "EXTERNAL_SERVICE_FAILED"):
        """
        Initialize external service error.

        Args:
            service: Name of the external service
            message: Detailed error message
            error_code: Error code for this type of failure
        """
        super().__init__(f"External service '{service}' failed: {message}", error_code)
        self.service = service


class TimeoutError(VRPModelError):
    """Raised when operations exceed their timeout limits."""

    def __init__(self, operation: str, timeout_seconds: int, error_code: str = "TIMEOUT_EXCEEDED"):
        """
        Initialize timeout error.

        Args:
            operation: Operation that timed out
            timeout_seconds: Timeout limit in seconds
            error_code: Error code for this type of failure
        """
        super().__init__(
            f"Operation '{operation}' timed out after {timeout_seconds} seconds",
            error_code
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class ResourceExhaustedError(VRPModelError):
    """Raised when system resources are exhausted."""

    def __init__(self, resource: str, limit: str, error_code: str = "RESOURCE_EXHAUSTED"):
        """
        Initialize resource exhausted error.

        Args:
            resource: Type of resource that was exhausted
            limit: Description of the limit that was exceeded
            error_code: Error code for this type of failure
        """
        super().__init__(f"Resource '{resource}' exhausted: {limit}", error_code)
        self.resource = resource
        self.limit = limit


# Error code constants for programmatic handling
class ErrorCodes:
    """Constants for error codes used throughout the system."""

    # Data-related errors
    DATA_FETCH_FAILED = "DATA_FETCH_FAILED"
    DATA_QUALITY_FAILED = "DATA_QUALITY_FAILED"
    DATA_VALIDATION_FAILED = "DATA_VALIDATION_FAILED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

    # Calculation errors
    CALCULATION_FAILED = "CALCULATION_FAILED"
    CONVERGENCE_FAILED = "CONVERGENCE_FAILED"

    # Model errors
    MODEL_STATE_INVALID = "MODEL_STATE_INVALID"
    MODEL_ERROR = "MODEL_ERROR"
    SIGNAL_GENERATION_FAILED = "SIGNAL_GENERATION_FAILED"
    PERFORMANCE_THRESHOLD_FAILED = "PERFORMANCE_THRESHOLD_FAILED"
    RISK_VIOLATION = "RISK_VIOLATION"

    # System errors
    PERSISTENCE_FAILED = "PERSISTENCE_FAILED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    CONFIGURATION_INVALID = "CONFIGURATION_INVALID"
    OPERATION_FAILED = "OPERATION_FAILED"

    # External errors
    EXTERNAL_SERVICE_FAILED = "EXTERNAL_SERVICE_FAILED"
    TIMEOUT_EXCEEDED = "TIMEOUT_EXCEEDED"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"


def handle_exception_with_logging(logger, operation: str):
    """
    Decorator to handle exceptions with consistent logging.

    Args:
        logger: Logger instance to use
        operation: Name of the operation being performed

    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except VRPModelError as e:
                logger.error(f"VRP Model error in {operation}: {e.message} (Code: {e.error_code})")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in {operation}: {str(e)}")
                raise OperationError(operation, str(e))
        return wrapper
    return decorator
