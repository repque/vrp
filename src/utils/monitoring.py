"""
Production monitoring and observability utilities.

Provides hooks for system health monitoring, performance metrics collection,
and alerting for the VRP Trading System.
"""

import time
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for system monitoring."""
    execution_time: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_pct: Optional[float] = None
    error_count: int = 0
    success_count: int = 0


@dataclass
class SystemHealthStatus:
    """Overall system health status."""
    timestamp: datetime
    component: str
    status: str  # GREEN, YELLOW, RED
    metrics: PerformanceMetrics
    alerts: list = field(default_factory=list)


class MonitoringHooks:
    """Production monitoring hooks for VRP Trading System."""
    
    def __init__(self):
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._alerts: list = []
        self._health_checks: Dict[str, Callable] = {}
    
    @contextmanager
    def performance_monitor(self, component_name: str):
        """Context manager for monitoring component performance."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
            # Success case
            execution_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            memory_delta = end_memory - start_memory if start_memory and end_memory else None
            
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=memory_delta,
                success_count=1
            )
            
            self._record_metrics(component_name, metrics)
            
            # Log performance warnings
            if execution_time > 10.0:  # Slow operation
                logger.warning(f"Slow operation in {component_name}: {execution_time:.2f}s")
            
            if memory_delta and memory_delta > 100:  # High memory usage
                logger.warning(f"High memory usage in {component_name}: {memory_delta:.2f}MB")
                
        except Exception as e:
            execution_time = time.time() - start_time
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                error_count=1
            )
            self._record_metrics(component_name, metrics)
            
            # Log error with context
            logger.error(f"Error in {component_name} after {execution_time:.2f}s: {str(e)}")
            raise
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None
    
    def _record_metrics(self, component: str, metrics: PerformanceMetrics):
        """Record performance metrics for a component."""
        if component in self._metrics:
            # Aggregate metrics
            existing = self._metrics[component]
            existing.success_count += metrics.success_count
            existing.error_count += metrics.error_count
            # Use exponential moving average for timing
            existing.execution_time = 0.8 * existing.execution_time + 0.2 * metrics.execution_time
        else:
            self._metrics[component] = metrics
    
    def register_health_check(self, component: str, check_func: Callable[[], bool]):
        """Register a health check function for a component."""
        self._health_checks[component] = check_func
    
    def run_health_checks(self) -> Dict[str, SystemHealthStatus]:
        """Run all registered health checks."""
        results = {}
        
        for component, check_func in self._health_checks.items():
            try:
                is_healthy = check_func()
                status = "GREEN" if is_healthy else "RED"
                
                metrics = self._metrics.get(component, PerformanceMetrics(execution_time=0))
                
                results[component] = SystemHealthStatus(
                    timestamp=datetime.now(),
                    component=component,
                    status=status,
                    metrics=metrics
                )
                
            except Exception as e:
                logger.error(f"Health check failed for {component}: {str(e)}")
                results[component] = SystemHealthStatus(
                    timestamp=datetime.now(),
                    component=component,
                    status="RED",
                    metrics=PerformanceMetrics(execution_time=0, error_count=1),
                    alerts=[f"Health check failed: {str(e)}"]
                )
        
        return results
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'component_metrics': {
                component: {
                    'avg_execution_time': metrics.execution_time,
                    'success_count': metrics.success_count,
                    'error_count': metrics.error_count,
                    'error_rate': metrics.error_count / (metrics.success_count + metrics.error_count) 
                                 if (metrics.success_count + metrics.error_count) > 0 else 0,
                    'memory_usage_mb': metrics.memory_usage_mb
                }
                for component, metrics in self._metrics.items()
            }
        }
    
    def check_trading_performance(self, win_rate: float, sharpe_ratio: float) -> str:
        """Check if trading performance is within acceptable bounds."""
        alerts = []
        
        if win_rate < 0.45:  # Below production threshold
            alerts.append(f"Win rate below threshold: {win_rate:.2%} < 45%")
        
        if sharpe_ratio < 0.5:  # Below production threshold
            alerts.append(f"Sharpe ratio below threshold: {sharpe_ratio:.2f} < 0.5")
        
        if alerts:
            self._alerts.extend(alerts)
            logger.warning(f"Performance alerts: {'; '.join(alerts)}")
            return "RED"
        elif win_rate < 0.50 or sharpe_ratio < 0.8:  # Warning levels
            return "YELLOW"
        else:
            return "GREEN"


# Global monitoring instance
monitoring = MonitoringHooks()


def monitor_performance(component_name: str):
    """Decorator for monitoring function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with monitoring.performance_monitor(component_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator