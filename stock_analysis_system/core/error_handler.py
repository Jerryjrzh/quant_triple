"""
Comprehensive Error Handler System

This module implements a sophisticated error handling system for the stock analysis system.
It provides centralized error management, classification, retry mechanisms, and recovery strategies.
"""

import logging
import time
import traceback
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import json
import functools
import inspect
from pathlib import Path


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification"""
    NETWORK = "network"
    DATABASE = "database"
    DATA_FORMAT = "data_format"
    BUSINESS_LOGIC = "business_logic"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    EXTERNAL_API = "external_api"
    SYSTEM_RESOURCE = "system_resource"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class RetryStrategy(str, Enum):
    """Retry strategies"""
    NONE = "none"
    IMMEDIATE = "immediate"
    LINEAR_BACKOFF = "linear_backoff"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    CUSTOM = "custom"


@dataclass
class ErrorContext:
    """Context information for an error"""
    function_name: str
    module_name: str
    line_number: int
    arguments: Dict[str, Any] = field(default_factory=dict)
    local_variables: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    error_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    retry_count: int = 0
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_method: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_on_exceptions: List[Type[Exception]] = field(default_factory=list)
    stop_on_exceptions: List[Type[Exception]] = field(default_factory=list)


@dataclass
class ErrorPattern:
    """Pattern for error classification and handling"""
    pattern_id: str
    name: str
    description: str
    error_types: List[str]
    keywords: List[str]
    category: ErrorCategory
    severity: ErrorSeverity
    retry_config: RetryConfig
    custom_handler: Optional[Callable] = None
    auto_resolve: bool = False


class ErrorClassifier:
    """
    Intelligent error classifier that categorizes errors based on patterns.
    """
    
    def __init__(self):
        self.patterns: List[ErrorPattern] = []
        self.logger = logging.getLogger(__name__)
        self._initialize_default_patterns()
    
    def _initialize_default_patterns(self):
        """Initialize default error patterns"""
        # Network errors
        network_pattern = ErrorPattern(
            pattern_id="network_errors",
            name="Network Connectivity Issues",
            description="Network-related errors including timeouts and connection failures",
            error_types=["ConnectionError", "TimeoutError", "HTTPError", "URLError"],
            keywords=["connection", "timeout", "network", "unreachable", "dns", "socket"],
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            retry_config=RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=5,
                base_delay=2.0,
                max_delay=30.0
            )
        )
        self.patterns.append(network_pattern)
        
        # Database errors
        database_pattern = ErrorPattern(
            pattern_id="database_errors",
            name="Database Operation Failures",
            description="Database connection and query errors",
            error_types=["DatabaseError", "OperationalError", "IntegrityError", "ProgrammingError"],
            keywords=["database", "sql", "connection", "query", "transaction", "deadlock"],
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            retry_config=RetryConfig(
                strategy=RetryStrategy.LINEAR_BACKOFF,
                max_attempts=3,
                base_delay=1.0
            )
        )
        self.patterns.append(database_pattern)
        
        # Data format errors
        data_format_pattern = ErrorPattern(
            pattern_id="data_format_errors",
            name="Data Format and Parsing Issues",
            description="JSON, XML, CSV parsing and data validation errors",
            error_types=["JSONDecodeError", "ValueError", "KeyError", "TypeError"],
            keywords=["json", "parse", "format", "decode", "invalid", "missing"],
            category=ErrorCategory.DATA_FORMAT,
            severity=ErrorSeverity.MEDIUM,
            retry_config=RetryConfig(
                strategy=RetryStrategy.NONE,
                max_attempts=1
            )
        )
        self.patterns.append(data_format_pattern)
        
        # External API errors
        api_pattern = ErrorPattern(
            pattern_id="external_api_errors",
            name="External API Failures",
            description="Third-party API errors and rate limiting",
            error_types=["HTTPError", "RequestException", "APIError"],
            keywords=["api", "rate limit", "quota", "forbidden", "unauthorized", "service unavailable"],
            category=ErrorCategory.EXTERNAL_API,
            severity=ErrorSeverity.MEDIUM,
            retry_config=RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=4,
                base_delay=5.0,
                max_delay=120.0
            )
        )
        self.patterns.append(api_pattern)
        
        # Authentication errors
        auth_pattern = ErrorPattern(
            pattern_id="authentication_errors",
            name="Authentication Failures",
            description="Login, token, and credential errors",
            error_types=["AuthenticationError", "PermissionError", "Unauthorized"],
            keywords=["authentication", "login", "token", "credential", "unauthorized", "forbidden"],
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            retry_config=RetryConfig(
                strategy=RetryStrategy.NONE,
                max_attempts=1
            )
        )
        self.patterns.append(auth_pattern)
        
        # System resource errors
        resource_pattern = ErrorPattern(
            pattern_id="system_resource_errors",
            name="System Resource Issues",
            description="Memory, disk, and CPU resource errors",
            error_types=["MemoryError", "OSError", "IOError", "DiskSpaceError"],
            keywords=["memory", "disk", "space", "resource", "limit", "quota"],
            category=ErrorCategory.SYSTEM_RESOURCE,
            severity=ErrorSeverity.CRITICAL,
            retry_config=RetryConfig(
                strategy=RetryStrategy.LINEAR_BACKOFF,
                max_attempts=2,
                base_delay=10.0
            )
        )
        self.patterns.append(resource_pattern)
    
    def classify_error(self, error: Exception, error_message: str = "") -> ErrorPattern:
        """Classify an error based on patterns"""
        error_type = type(error).__name__
        full_message = f"{error_message} {str(error)}".lower()
        
        # Find matching pattern
        for pattern in self.patterns:
            # Check error type match
            if error_type in pattern.error_types:
                return pattern
            
            # Check keyword match
            if any(keyword in full_message for keyword in pattern.keywords):
                return pattern
        
        # Default pattern for unclassified errors
        return ErrorPattern(
            pattern_id="unknown_error",
            name="Unknown Error",
            description="Unclassified error",
            error_types=[error_type],
            keywords=[],
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            retry_config=RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=2
            )
        )
    
    def add_pattern(self, pattern: ErrorPattern):
        """Add a custom error pattern"""
        self.patterns.append(pattern)
        self.logger.info(f"Added error pattern: {pattern.name}")
    
    def remove_pattern(self, pattern_id: str) -> bool:
        """Remove an error pattern"""
        for i, pattern in enumerate(self.patterns):
            if pattern.pattern_id == pattern_id:
                del self.patterns[i]
                self.logger.info(f"Removed error pattern: {pattern_id}")
                return True
        return False


class RetryManager:
    """
    Manages retry logic with different strategies and backoff algorithms.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_delay(self, retry_config: RetryConfig, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if retry_config.strategy == RetryStrategy.NONE:
            return 0
        
        elif retry_config.strategy == RetryStrategy.IMMEDIATE:
            return 0
        
        elif retry_config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = retry_config.base_delay * attempt
        
        elif retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = retry_config.base_delay * (retry_config.backoff_multiplier ** (attempt - 1))
        
        else:
            delay = retry_config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, retry_config.max_delay)
        
        # Add jitter to prevent thundering herd
        if retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def should_retry(self, error: Exception, retry_config: RetryConfig, attempt: int) -> bool:
        """Determine if an error should be retried"""
        if attempt >= retry_config.max_attempts:
            return False
        
        error_type = type(error)
        
        # Check stop conditions
        if retry_config.stop_on_exceptions and error_type in retry_config.stop_on_exceptions:
            return False
        
        # Check retry conditions
        if retry_config.retry_on_exceptions:
            return error_type in retry_config.retry_on_exceptions
        
        # Default retry logic based on error type
        non_retryable_errors = [
            ValueError, TypeError, AttributeError, KeyError, IndexError,
            SyntaxError, NameError, ImportError
        ]
        
        return error_type not in non_retryable_errors
    
    async def execute_with_retry(self, func: Callable, retry_config: RetryConfig, 
                                *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_error = None
        
        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as error:
                last_error = error
                
                if not self.should_retry(error, retry_config, attempt):
                    break
                
                if attempt < retry_config.max_attempts:
                    delay = self.calculate_delay(retry_config, attempt)
                    self.logger.warning(
                        f"Attempt {attempt} failed: {error}. Retrying in {delay:.2f}s"
                    )
                    
                    if delay > 0:
                        await asyncio.sleep(delay)
        
        # All retries exhausted
        raise last_error


class ErrorHandler:
    """
    Comprehensive error handler with classification, retry logic, and recovery strategies.
    """
    
    def __init__(self, 
                 max_error_history: int = 10000,
                 error_aggregation_window: int = 300,  # 5 minutes
                 enable_auto_recovery: bool = True):
        """
        Initialize error handler.
        
        Args:
            max_error_history: Maximum number of errors to keep in history
            error_aggregation_window: Time window for error aggregation (seconds)
            enable_auto_recovery: Enable automatic error recovery
        """
        self.max_error_history = max_error_history
        self.error_aggregation_window = error_aggregation_window
        self.enable_auto_recovery = enable_auto_recovery
        
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.classifier = ErrorClassifier()
        self.retry_manager = RetryManager()
        
        # Error tracking
        self.error_history: deque = deque(maxlen=max_error_history)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_patterns_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Recovery handlers
        self.recovery_handlers: Dict[ErrorCategory, List[Callable]] = defaultdict(list)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_errors': 0,
            'resolved_errors': 0,
            'retry_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0
        }
        
        self.logger.info("ErrorHandler initialized")
    
    def _extract_error_context(self, error: Exception) -> ErrorContext:
        """Extract context information from an error"""
        # Get current frame information
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the actual error location
            while frame and frame.f_code.co_filename == __file__:
                frame = frame.f_back
            
            if frame:
                function_name = frame.f_code.co_name
                module_name = frame.f_globals.get('__name__', 'unknown')
                line_number = frame.f_lineno
                
                # Extract local variables (be careful with sensitive data)
                local_vars = {}
                for key, value in frame.f_locals.items():
                    if not key.startswith('_') and not callable(value):
                        try:
                            # Only include serializable values
                            json.dumps(value, default=str)
                            local_vars[key] = value
                        except (TypeError, ValueError):
                            local_vars[key] = str(type(value))
            else:
                function_name = "unknown"
                module_name = "unknown"
                line_number = 0
                local_vars = {}
            
        finally:
            del frame
        
        return ErrorContext(
            function_name=function_name,
            module_name=module_name,
            line_number=line_number,
            local_variables=local_vars,
            stack_trace=traceback.format_exc(),
            timestamp=datetime.now()
        )
    
    def handle_error(self, error: Exception, 
                    context: Optional[ErrorContext] = None,
                    custom_message: str = "",
                    user_id: Optional[str] = None,
                    request_id: Optional[str] = None) -> ErrorRecord:
        """
        Handle an error with classification and recovery.
        
        Args:
            error: The exception that occurred
            context: Optional error context
            custom_message: Custom error message
            user_id: User ID associated with the error
            request_id: Request ID for tracing
            
        Returns:
            ErrorRecord: Record of the handled error
        """
        with self.lock:
            # Extract context if not provided
            if context is None:
                context = self._extract_error_context(error)
            
            # Add user and request information
            context.user_id = user_id
            context.request_id = request_id
            
            # Classify the error
            pattern = self.classifier.classify_error(error, custom_message)
            
            # Create error record
            error_id = f"{pattern.category.value}_{int(time.time())}_{id(error)}"
            error_record = ErrorRecord(
                error_id=error_id,
                error_type=type(error).__name__,
                error_message=custom_message or str(error),
                severity=pattern.severity,
                category=pattern.category,
                context=context
            )
            
            # Update statistics
            self.stats['total_errors'] += 1
            self.error_counts[pattern.category.value] += 1
            
            # Add to history
            self.error_history.append(error_record)
            
            # Update pattern statistics
            pattern_stats = self.error_patterns_stats[pattern.pattern_id]
            pattern_stats['count'] = pattern_stats.get('count', 0) + 1
            pattern_stats['last_occurrence'] = datetime.now()
            
            # Log the error
            log_level = {
                ErrorSeverity.LOW: logging.INFO,
                ErrorSeverity.MEDIUM: logging.WARNING,
                ErrorSeverity.HIGH: logging.ERROR,
                ErrorSeverity.CRITICAL: logging.CRITICAL
            }.get(pattern.severity, logging.ERROR)
            
            self.logger.log(
                log_level,
                f"Error handled: {error_record.error_type} - {error_record.error_message} "
                f"[Category: {pattern.category.value}, Severity: {pattern.severity.value}]"
            )
            
            # Attempt automatic recovery if enabled
            if self.enable_auto_recovery and pattern.auto_resolve:
                self._attempt_recovery(error_record, pattern)
            
            return error_record
    
    def _attempt_recovery(self, error_record: ErrorRecord, pattern: ErrorPattern):
        """Attempt automatic error recovery"""
        try:
            # Try custom handler first
            if pattern.custom_handler:
                success = pattern.custom_handler(error_record)
                if success:
                    error_record.resolved = True
                    error_record.resolution_time = datetime.now()
                    error_record.resolution_method = "custom_handler"
                    self.stats['successful_recoveries'] += 1
                    return
            
            # Try registered recovery handlers
            handlers = self.recovery_handlers.get(pattern.category, [])
            for handler in handlers:
                try:
                    success = handler(error_record)
                    if success:
                        error_record.resolved = True
                        error_record.resolution_time = datetime.now()
                        error_record.resolution_method = f"recovery_handler_{handler.__name__}"
                        self.stats['successful_recoveries'] += 1
                        return
                except Exception as recovery_error:
                    self.logger.error(f"Recovery handler failed: {recovery_error}")
            
            self.stats['failed_recoveries'] += 1
            
        except Exception as e:
            self.logger.error(f"Error in recovery attempt: {e}")
            self.stats['failed_recoveries'] += 1
    
    def register_recovery_handler(self, category: ErrorCategory, handler: Callable):
        """Register a recovery handler for a specific error category"""
        self.recovery_handlers[category].append(handler)
        self.logger.info(f"Registered recovery handler for {category.value}")
    
    def with_error_handling(self, 
                           retry_config: Optional[RetryConfig] = None,
                           custom_message: str = "",
                           user_id: Optional[str] = None,
                           request_id: Optional[str] = None,
                           raise_on_failure: bool = True):
        """
        Decorator for automatic error handling and retry.
        
        Args:
            retry_config: Retry configuration
            custom_message: Custom error message
            user_id: User ID for context
            request_id: Request ID for tracing
            raise_on_failure: Whether to raise exception after all retries fail
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_error_record = None
                
                # Use default retry config if not provided
                if retry_config is None:
                    default_config = RetryConfig()
                else:
                    default_config = retry_config
                
                for attempt in range(1, default_config.max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                        
                    except Exception as error:
                        # Handle the error
                        error_record = self.handle_error(
                            error, 
                            custom_message=custom_message,
                            user_id=user_id,
                            request_id=request_id
                        )
                        last_error_record = error_record
                        error_record.retry_count = attempt - 1
                        
                        # Update retry statistics
                        self.stats['retry_attempts'] += 1
                        
                        # Check if we should retry
                        if not self.retry_manager.should_retry(error, default_config, attempt):
                            break
                        
                        if attempt < default_config.max_attempts:
                            delay = self.retry_manager.calculate_delay(default_config, attempt)
                            if delay > 0:
                                await asyncio.sleep(delay)
                
                # All retries exhausted
                if raise_on_failure and last_error_record:
                    raise Exception(f"All retry attempts failed: {last_error_record.error_message}")
                
                return None
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_error_record = None
                
                # Use default retry config if not provided
                if retry_config is None:
                    default_config = RetryConfig()
                else:
                    default_config = retry_config
                
                for attempt in range(1, default_config.max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                        
                    except Exception as error:
                        # Handle the error
                        error_record = self.handle_error(
                            error,
                            custom_message=custom_message,
                            user_id=user_id,
                            request_id=request_id
                        )
                        last_error_record = error_record
                        error_record.retry_count = attempt - 1
                        
                        # Update retry statistics
                        self.stats['retry_attempts'] += 1
                        
                        # Check if we should retry
                        if not self.retry_manager.should_retry(error, default_config, attempt):
                            break
                        
                        if attempt < default_config.max_attempts:
                            delay = self.retry_manager.calculate_delay(default_config, attempt)
                            if delay > 0:
                                time.sleep(delay)
                
                # All retries exhausted
                if raise_on_failure and last_error_record:
                    raise Exception(f"All retry attempts failed: {last_error_record.error_message}")
                
                return None
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        with self.lock:
            # Calculate error rates by category
            category_stats = {}
            for category, count in self.error_counts.items():
                category_stats[category] = {
                    'count': count,
                    'percentage': (count / self.stats['total_errors'] * 100) if self.stats['total_errors'] > 0 else 0
                }
            
            # Calculate recent error trends
            recent_errors = [
                error for error in self.error_history
                if (datetime.now() - error.context.timestamp).total_seconds() <= self.error_aggregation_window
            ]
            
            return {
                'total_errors': self.stats['total_errors'],
                'resolved_errors': self.stats['resolved_errors'],
                'retry_attempts': self.stats['retry_attempts'],
                'successful_recoveries': self.stats['successful_recoveries'],
                'failed_recoveries': self.stats['failed_recoveries'],
                'resolution_rate': (self.stats['resolved_errors'] / self.stats['total_errors'] * 100) if self.stats['total_errors'] > 0 else 0,
                'category_breakdown': category_stats,
                'recent_errors_count': len(recent_errors),
                'pattern_statistics': dict(self.error_patterns_stats),
                'error_history_size': len(self.error_history)
            }
    
    def get_error_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get error trends over specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_errors = [
                error for error in self.error_history
                if error.context.timestamp > cutoff_time
            ]
            
            # Group by hour
            hourly_counts = defaultdict(int)
            category_trends = defaultdict(lambda: defaultdict(int))
            
            for error in recent_errors:
                hour_key = error.context.timestamp.strftime('%Y-%m-%d %H:00')
                hourly_counts[hour_key] += 1
                category_trends[error.category.value][hour_key] += 1
            
            return {
                'time_period_hours': hours,
                'total_errors_in_period': len(recent_errors),
                'hourly_distribution': dict(hourly_counts),
                'category_trends': {k: dict(v) for k, v in category_trends.items()},
                'average_errors_per_hour': len(recent_errors) / hours if hours > 0 else 0
            }
    
    def export_error_report(self, file_path: str, include_context: bool = False):
        """Export comprehensive error report"""
        try:
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'statistics': self.get_error_statistics(),
                'trends': self.get_error_trends(),
                'error_patterns': [
                    {
                        'pattern_id': pattern.pattern_id,
                        'name': pattern.name,
                        'category': pattern.category.value,
                        'severity': pattern.severity.value,
                        'statistics': self.error_patterns_stats.get(pattern.pattern_id, {})
                    }
                    for pattern in self.classifier.patterns
                ]
            }
            
            if include_context:
                report['recent_errors'] = [
                    {
                        'error_id': error.error_id,
                        'error_type': error.error_type,
                        'error_message': error.error_message,
                        'severity': error.severity.value,
                        'category': error.category.value,
                        'timestamp': error.context.timestamp.isoformat(),
                        'function_name': error.context.function_name,
                        'module_name': error.context.module_name,
                        'retry_count': error.retry_count,
                        'resolved': error.resolved
                    }
                    for error in list(self.error_history)[-100:]  # Last 100 errors
                ]
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Error report exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export error report: {e}")
    
    def clear_error_history(self):
        """Clear error history (use with caution)"""
        with self.lock:
            self.error_history.clear()
            self.error_counts.clear()
            self.error_patterns_stats.clear()
            self.stats = {
                'total_errors': 0,
                'resolved_errors': 0,
                'retry_attempts': 0,
                'successful_recoveries': 0,
                'failed_recoveries': 0
            }
        
        self.logger.info("Error history cleared")


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def handle_error(error: Exception, **kwargs) -> ErrorRecord:
    """Convenience function to handle errors using global handler"""
    return get_error_handler().handle_error(error, **kwargs)


def with_error_handling(**kwargs):
    """Convenience decorator for error handling"""
    return get_error_handler().with_error_handling(**kwargs)


# Example usage and testing
if __name__ == "__main__":
    import random
    import asyncio
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create error handler
    error_handler = ErrorHandler()
    
    # Example recovery handler
    def network_recovery_handler(error_record: ErrorRecord) -> bool:
        """Example recovery handler for network errors"""
        print(f"Attempting network recovery for: {error_record.error_message}")
        # Simulate recovery logic
        return random.random() > 0.5  # 50% success rate
    
    # Register recovery handler
    error_handler.register_recovery_handler(ErrorCategory.NETWORK, network_recovery_handler)
    
    # Example function with error handling
    @error_handler.with_error_handling(
        retry_config=RetryConfig(max_attempts=3, strategy=RetryStrategy.EXPONENTIAL_BACKOFF),
        custom_message="Test function failed"
    )
    def test_function(should_fail: bool = False):
        if should_fail:
            if random.random() > 0.7:
                raise ConnectionError("Network connection failed")
            elif random.random() > 0.5:
                raise ValueError("Invalid data format")
            else:
                raise Exception("Unknown error occurred")
        return "Success!"
    
    # Test error handling
    print("Testing Error Handler...")
    
    # Generate some test errors
    for i in range(10):
        try:
            result = test_function(should_fail=True)
            print(f"Test {i}: {result}")
        except Exception as e:
            print(f"Test {i}: Final failure - {e}")
    
    # Print statistics
    stats = error_handler.get_error_statistics()
    print(f"\nError Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Export error report
    error_handler.export_error_report("error_report.json", include_context=True)
    print("\nError report exported to error_report.json")