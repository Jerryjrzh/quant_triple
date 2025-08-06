"""
Jaeger Distributed Tracing System

This module implements comprehensive distributed tracing for the stock analysis system
using Jaeger. It provides automatic instrumentation, custom span creation, and
trace correlation across services.
"""

import logging
import time
import functools
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor

# Jaeger and OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.celery import CeleryInstrumentor
    from opentelemetry.propagate import inject, extract
    from opentelemetry.trace.status import Status, StatusCode
    from opentelemetry.sdk.resources import Resource
    JAEGER_AVAILABLE = True
except ImportError:
    JAEGER_AVAILABLE = False
    # Mock classes for when Jaeger is not available
    class MockTracer:
        def start_span(self, *args, **kwargs):
            return MockSpan()
    
    class MockSpan:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def set_attribute(self, *args):
            pass
        def set_status(self, *args):
            pass
        def add_event(self, *args):
            pass


@dataclass
class TracingConfig:
    """Jaeger tracing configuration"""
    service_name: str
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    agent_host: str = "localhost"
    agent_port: int = 6831
    sampling_rate: float = 1.0
    max_tag_value_length: int = 1024
    batch_span_processor: bool = True
    enable_auto_instrumentation: bool = True


@dataclass
class SpanInfo:
    """Span information for tracking"""
    span_id: str
    trace_id: str
    operation_name: str
    start_time: datetime
    duration: Optional[float] = None
    status: str = "ok"
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


class JaegerTracingManager:
    """
    Comprehensive Jaeger distributed tracing manager.
    
    Features:
    - Automatic service instrumentation
    - Custom span creation and management
    - Trace correlation across services
    - Performance monitoring
    - Error tracking and debugging
    - Sampling configuration
    - Batch span processing
    """
    
    def __init__(self, config: TracingConfig):
        """
        Initialize Jaeger tracing manager.
        
        Args:
            config: Tracing configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tracer = None
        self.span_processor = None
        self.active_spans: Dict[str, SpanInfo] = {}
        self._lock = threading.Lock()
        
        if JAEGER_AVAILABLE:
            self._setup_tracing()
        else:
            self.logger.warning("Jaeger/OpenTelemetry not available, using mock tracer")
            self.tracer = MockTracer()
    
    def _setup_tracing(self):
        """Setup Jaeger tracing with OpenTelemetry"""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.config.service_name,
                "service.version": "1.0.0"
            })
            
            # Create tracer provider
            trace.set_tracer_provider(TracerProvider(resource=resource))
            
            # Create Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.config.agent_host,
                agent_port=self.config.agent_port,
                collector_endpoint=self.config.jaeger_endpoint,
            )
            
            # Create span processor
            if self.config.batch_span_processor:
                self.span_processor = BatchSpanProcessor(jaeger_exporter)
            else:
                from opentelemetry.sdk.trace.export import SimpleSpanProcessor
                self.span_processor = SimpleSpanProcessor(jaeger_exporter)
            
            # Add span processor to tracer provider
            trace.get_tracer_provider().add_span_processor(self.span_processor)
            
            # Get tracer
            self.tracer = trace.get_tracer(__name__)
            
            # Setup auto-instrumentation
            if self.config.enable_auto_instrumentation:
                self._setup_auto_instrumentation()
            
            self.logger.info(f"Initialized Jaeger tracing for service: {self.config.service_name}")
            
        except Exception as e:
            self.logger.error(f"Error setting up Jaeger tracing: {e}")
            self.tracer = MockTracer()
    
    def _setup_auto_instrumentation(self):
        """Setup automatic instrumentation for common libraries"""
        try:
            # Instrument HTTP requests
            RequestsInstrumentor().instrument()
            
            # Instrument SQLAlchemy (if available)
            try:
                SQLAlchemyInstrumentor().instrument()
            except Exception:
                pass
            
            # Instrument Redis (if available)
            try:
                RedisInstrumentor().instrument()
            except Exception:
                pass
            
            # Instrument Celery (if available)
            try:
                CeleryInstrumentor().instrument()
            except Exception:
                pass
            
            self.logger.info("Auto-instrumentation setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up auto-instrumentation: {e}")
    
    @contextmanager
    def start_span(self, operation_name: str, 
                   parent_span=None,
                   tags: Optional[Dict[str, Any]] = None,
                   kind: Optional[str] = None):
        """
        Start a new span with context management.
        
        Args:
            operation_name: Name of the operation
            parent_span: Parent span (optional)
            tags: Span tags
            kind: Span kind (server, client, producer, consumer, internal)
            
        Yields:
            Span object
        """
        span = self.tracer.start_span(
            operation_name,
            context=parent_span
        )
        
        try:
            # Set span kind
            if kind:
                span.set_attribute("span.kind", kind)
            
            # Set tags
            if tags:
                for key, value in tags.items():
                    span.set_attribute(key, str(value)[:self.config.max_tag_value_length])
            
            # Track span
            span_info = SpanInfo(
                span_id=format(span.get_span_context().span_id, '016x'),
                trace_id=format(span.get_span_context().trace_id, '032x'),
                operation_name=operation_name,
                start_time=datetime.now(),
                tags=tags or {}
            )
            
            with self._lock:
                self.active_spans[span_info.span_id] = span_info
            
            yield span
            
        except Exception as e:
            # Record exception
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            
            # Update span info
            with self._lock:
                if span_info.span_id in self.active_spans:
                    self.active_spans[span_info.span_id].status = "error"
            
            raise
        finally:
            # Finish span
            span.end()
            
            # Update span info
            with self._lock:
                if span_info.span_id in self.active_spans:
                    span_info = self.active_spans[span_info.span_id]
                    span_info.duration = (datetime.now() - span_info.start_time).total_seconds()
                    # Keep span info for a while for monitoring
                    # In production, you might want to clean this up periodically
    
    def trace_function(self, operation_name: Optional[str] = None,
                      tags: Optional[Dict[str, Any]] = None,
                      kind: str = "internal"):
        """
        Decorator to trace function calls.
        
        Args:
            operation_name: Custom operation name (defaults to function name)
            tags: Additional tags for the span
            kind: Span kind
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                # Prepare tags
                span_tags = {
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                }
                if tags:
                    span_tags.update(tags)
                
                with self.start_span(op_name, tags=span_tags, kind=kind) as span:
                    try:
                        # Add function arguments as tags (be careful with sensitive data)
                        if args:
                            span.set_attribute("function.args.count", len(args))
                        if kwargs:
                            span.set_attribute("function.kwargs.count", len(kwargs))
                            # Only add non-sensitive kwargs
                            safe_kwargs = {k: v for k, v in kwargs.items() 
                                         if not any(sensitive in k.lower() 
                                                  for sensitive in ['password', 'token', 'key', 'secret'])}
                            for k, v in safe_kwargs.items():
                                span.set_attribute(f"function.kwargs.{k}", str(v)[:100])
                        
                        # Execute function
                        result = func(*args, **kwargs)
                        
                        # Add result information
                        if result is not None:
                            span.set_attribute("function.result.type", type(result).__name__)
                            if hasattr(result, '__len__'):
                                try:
                                    span.set_attribute("function.result.length", len(result))
                                except:
                                    pass
                        
                        return result
                        
                    except Exception as e:
                        span.add_event("exception", {
                            "exception.type": type(e).__name__,
                            "exception.message": str(e)
                        })
                        raise
            
            return wrapper
        return decorator
    
    def trace_async_function(self, operation_name: Optional[str] = None,
                           tags: Optional[Dict[str, Any]] = None,
                           kind: str = "internal"):
        """
        Decorator to trace async function calls.
        
        Args:
            operation_name: Custom operation name
            tags: Additional tags for the span
            kind: Span kind
            
        Returns:
            Decorated async function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                span_tags = {
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                    "function.async": True
                }
                if tags:
                    span_tags.update(tags)
                
                with self.start_span(op_name, tags=span_tags, kind=kind) as span:
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception as e:
                        span.add_event("exception", {
                            "exception.type": type(e).__name__,
                            "exception.message": str(e)
                        })
                        raise
            
            return wrapper
        return decorator
    
    def trace_database_operation(self, operation: str, table: str, 
                               query: Optional[str] = None):
        """
        Create a span for database operations.
        
        Args:
            operation: Database operation (SELECT, INSERT, UPDATE, DELETE)
            table: Table name
            query: SQL query (optional, will be truncated)
            
        Returns:
            Context manager for the span
        """
        tags = {
            "db.operation": operation,
            "db.table": table,
            "component": "database"
        }
        
        if query:
            # Truncate query for security and readability
            tags["db.statement"] = query[:500] + "..." if len(query) > 500 else query
        
        return self.start_span(
            f"db.{operation.lower()}",
            tags=tags,
            kind="client"
        )
    
    def trace_api_request(self, method: str, endpoint: str, 
                         status_code: Optional[int] = None):
        """
        Create a span for API requests.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            
        Returns:
            Context manager for the span
        """
        tags = {
            "http.method": method,
            "http.url": endpoint,
            "component": "http"
        }
        
        if status_code:
            tags["http.status_code"] = status_code
        
        return self.start_span(
            f"http.{method.lower()}",
            tags=tags,
            kind="server"
        )
    
    def trace_ml_operation(self, model_name: str, operation: str,
                          model_version: Optional[str] = None,
                          input_size: Optional[int] = None):
        """
        Create a span for ML operations.
        
        Args:
            model_name: Name of the ML model
            operation: ML operation (train, predict, evaluate)
            model_version: Model version
            input_size: Size of input data
            
        Returns:
            Context manager for the span
        """
        tags = {
            "ml.model.name": model_name,
            "ml.operation": operation,
            "component": "ml"
        }
        
        if model_version:
            tags["ml.model.version"] = model_version
        
        if input_size:
            tags["ml.input.size"] = input_size
        
        return self.start_span(
            f"ml.{operation}",
            tags=tags,
            kind="internal"
        )
    
    def trace_stock_analysis(self, symbol: str, analysis_type: str,
                           time_range: Optional[str] = None):
        """
        Create a span for stock analysis operations.
        
        Args:
            symbol: Stock symbol
            analysis_type: Type of analysis
            time_range: Time range for analysis
            
        Returns:
            Context manager for the span
        """
        tags = {
            "stock.symbol": symbol,
            "stock.analysis.type": analysis_type,
            "component": "stock_analysis"
        }
        
        if time_range:
            tags["stock.time_range"] = time_range
        
        return self.start_span(
            f"stock.analysis.{analysis_type}",
            tags=tags,
            kind="internal"
        )
    
    def inject_trace_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Inject trace context into headers for distributed tracing.
        
        Args:
            headers: HTTP headers dictionary
            
        Returns:
            Headers with trace context injected
        """
        if JAEGER_AVAILABLE:
            inject(headers)
        return headers
    
    def extract_trace_context(self, headers: Dict[str, str]):
        """
        Extract trace context from headers.
        
        Args:
            headers: HTTP headers dictionary
            
        Returns:
            Trace context
        """
        if JAEGER_AVAILABLE:
            return extract(headers)
        return None
    
    def get_current_span(self):
        """Get the current active span"""
        if JAEGER_AVAILABLE:
            return trace.get_current_span()
        return None
    
    def get_trace_id(self) -> Optional[str]:
        """Get current trace ID"""
        span = self.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().trace_id, '032x')
        return None
    
    def get_span_id(self) -> Optional[str]:
        """Get current span ID"""
        span = self.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().span_id, '016x')
        return None
    
    def get_active_spans(self) -> Dict[str, SpanInfo]:
        """Get information about active spans"""
        with self._lock:
            return self.active_spans.copy()
    
    def get_span_statistics(self) -> Dict[str, Any]:
        """Get span statistics"""
        with self._lock:
            spans = list(self.active_spans.values())
        
        if not spans:
            return {
                "total_spans": 0,
                "active_spans": 0,
                "average_duration": 0,
                "error_rate": 0
            }
        
        completed_spans = [s for s in spans if s.duration is not None]
        error_spans = [s for s in spans if s.status == "error"]
        
        return {
            "total_spans": len(spans),
            "active_spans": len(spans) - len(completed_spans),
            "completed_spans": len(completed_spans),
            "average_duration": sum(s.duration for s in completed_spans) / len(completed_spans) if completed_spans else 0,
            "error_rate": len(error_spans) / len(spans) if spans else 0,
            "operations": list(set(s.operation_name for s in spans))
        }
    
    def flush_spans(self):
        """Flush pending spans"""
        if self.span_processor:
            self.span_processor.force_flush()
    
    def shutdown(self):
        """Shutdown tracing"""
        try:
            if self.span_processor:
                self.span_processor.shutdown()
            self.logger.info("Jaeger tracing shutdown completed")
        except Exception as e:
            self.logger.error(f"Error shutting down tracing: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of tracing system"""
        stats = self.get_span_statistics()
        
        return {
            "status": "healthy" if JAEGER_AVAILABLE else "degraded",
            "service_name": self.config.service_name,
            "jaeger_available": JAEGER_AVAILABLE,
            "jaeger_endpoint": self.config.jaeger_endpoint,
            "sampling_rate": self.config.sampling_rate,
            "span_statistics": stats
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import time
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create tracing configuration
    config = TracingConfig(
        service_name="stock_analysis_test",
        jaeger_endpoint="http://localhost:14268/api/traces",
        sampling_rate=1.0
    )
    
    # Create tracing manager
    tracer = JaegerTracingManager(config)
    
    # Example traced function
    @tracer.trace_function("example_function", tags={"component": "test"})
    def example_function(x: int, y: int) -> int:
        time.sleep(0.1)  # Simulate work
        return x + y
    
    # Example async traced function
    @tracer.trace_async_function("async_example", tags={"component": "test"})
    async def async_example(delay: float) -> str:
        await asyncio.sleep(delay)
        return f"Completed after {delay}s"
    
    async def main():
        # Test basic tracing
        with tracer.start_span("main_operation", tags={"test": "true"}) as span:
            span.add_event("Starting test operations")
            
            # Test function tracing
            result = example_function(5, 3)
            span.set_attribute("function_result", result)
            
            # Test async function tracing
            async_result = await async_example(0.2)
            span.set_attribute("async_result", async_result)
            
            # Test database operation tracing
            with tracer.trace_database_operation("SELECT", "stocks", "SELECT * FROM stocks WHERE symbol = 'AAPL'"):
                time.sleep(0.05)  # Simulate DB query
            
            # Test API request tracing
            with tracer.trace_api_request("GET", "/api/stocks/AAPL", 200):
                time.sleep(0.03)  # Simulate API processing
            
            # Test ML operation tracing
            with tracer.trace_ml_operation("pattern_detector", "predict", "v1.0", 1000):
                time.sleep(0.1)  # Simulate ML inference
            
            # Test stock analysis tracing
            with tracer.trace_stock_analysis("AAPL", "spring_festival", "2023-2024"):
                time.sleep(0.15)  # Simulate analysis
            
            span.add_event("Test operations completed")
        
        # Print statistics
        stats = tracer.get_span_statistics()
        print(f"Span statistics: {stats}")
        
        # Print health status
        health = tracer.get_health_status()
        print(f"Health status: {health}")
        
        # Flush spans
        tracer.flush_spans()
        
        print("Tracing test completed. Check Jaeger UI at http://localhost:16686")
    
    # Run the test
    try:
        asyncio.run(main())
    finally:
        tracer.shutdown()