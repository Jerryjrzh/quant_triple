"""
ELK Stack Centralized Logging System

This module implements comprehensive centralized logging for the stock analysis system
using the ELK Stack (Elasticsearch, Logstash, Kibana). It provides structured logging,
log aggregation, and advanced log analysis capabilities.
"""

import json
import logging
import logging.handlers
import socket
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import threading
from queue import Queue, Empty
import gzip
import os

# Elasticsearch client (optional)
try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False


@dataclass
class ELKConfig:
    """ELK Stack configuration"""
    service_name: str
    environment: str = "development"
    elasticsearch_hosts: List[str] = field(default_factory=lambda: ["localhost:9200"])
    elasticsearch_index: str = "stock-analysis-logs"
    logstash_host: str = "localhost"
    logstash_port: int = 5000
    kibana_url: str = "http://localhost:5601"
    log_level: str = "INFO"
    enable_elasticsearch: bool = True
    enable_logstash: bool = True
    enable_file_logging: bool = True
    log_file_path: str = "logs/stock_analysis.log"
    max_log_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 5
    batch_size: int = 100
    flush_interval: int = 5


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: str
    message: str
    service: str
    environment: str
    logger_name: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def __init__(self, service_name: str, environment: str):
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.hostname = socket.gethostname()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        # Extract trace information if available
        trace_id = getattr(record, 'trace_id', None)
        span_id = getattr(record, 'span_id', None)
        user_id = getattr(record, 'user_id', None)
        request_id = getattr(record, 'request_id', None)
        
        # Create structured log entry
        log_entry = {
            "@timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "service": self.service_name,
            "environment": self.environment,
            "hostname": self.hostname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread_id": record.thread,
            "process_id": record.process,
        }
        
        # Add trace information
        if trace_id:
            log_entry["trace_id"] = trace_id
        if span_id:
            log_entry["span_id"] = span_id
        if user_id:
            log_entry["user_id"] = user_id
        if request_id:
            log_entry["request_id"] = request_id
        
        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info', 'trace_id', 'span_id', 
                          'user_id', 'request_id']:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, default=str)


class ElasticsearchHandler(logging.Handler):
    """Custom logging handler for Elasticsearch"""
    
    def __init__(self, hosts: List[str], index: str, batch_size: int = 100, 
                 flush_interval: int = 5):
        super().__init__()
        self.hosts = hosts
        self.index = index
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Initialize Elasticsearch client
        if ELASTICSEARCH_AVAILABLE:
            self.es_client = Elasticsearch(hosts)
            self.log_buffer = Queue()
            self.last_flush = time.time()
            self._setup_index_template()
        else:
            self.es_client = None
            logging.warning("Elasticsearch not available, logs will not be sent to ES")
    
    def _setup_index_template(self):
        """Setup Elasticsearch index template"""
        if not self.es_client:
            return
        
        template = {
            "index_patterns": [f"{self.index}-*"],
            "template": {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "index.refresh_interval": "5s"
                },
                "mappings": {
                    "properties": {
                        "@timestamp": {"type": "date"},
                        "level": {"type": "keyword"},
                        "message": {"type": "text"},
                        "service": {"type": "keyword"},
                        "environment": {"type": "keyword"},
                        "hostname": {"type": "keyword"},
                        "logger": {"type": "keyword"},
                        "module": {"type": "keyword"},
                        "function": {"type": "keyword"},
                        "line": {"type": "integer"},
                        "thread_id": {"type": "long"},
                        "process_id": {"type": "long"},
                        "trace_id": {"type": "keyword"},
                        "span_id": {"type": "keyword"},
                        "user_id": {"type": "keyword"},
                        "request_id": {"type": "keyword"}
                    }
                }
            }
        }
        
        try:
            self.es_client.indices.put_index_template(
                name=f"{self.index}-template",
                body=template
            )
        except Exception as e:
            logging.error(f"Failed to create index template: {e}")
    
    def emit(self, record: logging.LogRecord):
        """Emit log record to Elasticsearch"""
        if not self.es_client:
            return
        
        try:
            # Format the record
            log_data = json.loads(self.format(record))
            
            # Add to buffer
            self.log_buffer.put({
                "_index": f"{self.index}-{datetime.now().strftime('%Y.%m.%d')}",
                "_source": log_data
            })
            
            # Check if we should flush
            current_time = time.time()
            if (self.log_buffer.qsize() >= self.batch_size or 
                current_time - self.last_flush >= self.flush_interval):
                self._flush_buffer()
                
        except Exception as e:
            self.handleError(record)
    
    def _flush_buffer(self):
        """Flush log buffer to Elasticsearch"""
        if not self.es_client or self.log_buffer.empty():
            return
        
        docs = []
        try:
            while not self.log_buffer.empty() and len(docs) < self.batch_size:
                docs.append(self.log_buffer.get_nowait())
        except Empty:
            pass
        
        if docs:
            try:
                bulk(self.es_client, docs)
                self.last_flush = time.time()
            except Exception as e:
                logging.error(f"Failed to bulk insert logs to Elasticsearch: {e}")
                # Put docs back in queue for retry
                for doc in docs:
                    self.log_buffer.put(doc)
    
    def flush(self):
        """Flush all pending logs"""
        self._flush_buffer()
    
    def close(self):
        """Close handler and flush remaining logs"""
        self.flush()
        super().close()


class LogstashHandler(logging.handlers.SocketHandler):
    """Custom logging handler for Logstash"""
    
    def __init__(self, host: str, port: int):
        super().__init__(host, port)
        self.retries = 3
        self.retry_delay = 1
    
    def emit(self, record: logging.LogRecord):
        """Emit log record to Logstash"""
        for attempt in range(self.retries):
            try:
                super().emit(record)
                break
            except Exception as e:
                if attempt == self.retries - 1:
                    self.handleError(record)
                else:
                    time.sleep(self.retry_delay * (attempt + 1))
    
    def makePickle(self, record: logging.LogRecord) -> bytes:
        """Convert log record to bytes for transmission"""
        # Format as JSON instead of pickle for Logstash compatibility
        log_data = self.format(record)
        return (log_data + '\n').encode('utf-8')


class ELKLoggingManager:
    """
    Comprehensive ELK Stack logging manager.
    
    Features:
    - Structured JSON logging
    - Multiple output destinations (Elasticsearch, Logstash, files)
    - Automatic log rotation
    - Batch processing for performance
    - Trace correlation
    - Custom log fields
    - Health monitoring
    """
    
    def __init__(self, config: ELKConfig):
        """
        Initialize ELK logging manager.
        
        Args:
            config: ELK configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.root_logger = logging.getLogger()
        self.handlers = []
        self.log_stats = {
            "total_logs": 0,
            "logs_by_level": {},
            "errors": 0,
            "start_time": datetime.now()
        }
        self._stats_lock = threading.Lock()
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Set root logger level
        self.root_logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Create structured formatter
        formatter = StructuredFormatter(
            self.config.service_name,
            self.config.environment
        )
        
        # Setup file logging
        if self.config.enable_file_logging:
            self._setup_file_logging(formatter)
        
        # Setup Elasticsearch logging
        if self.config.enable_elasticsearch and ELASTICSEARCH_AVAILABLE:
            self._setup_elasticsearch_logging(formatter)
        
        # Setup Logstash logging
        if self.config.enable_logstash:
            self._setup_logstash_logging(formatter)
        
        # Setup console logging for development
        if self.config.environment == "development":
            self._setup_console_logging()
        
        # Add custom filter for statistics
        self.root_logger.addFilter(self._log_filter)
        
        self.logger.info(f"ELK logging initialized for service: {self.config.service_name}")
    
    def _setup_file_logging(self, formatter: StructuredFormatter):
        """Setup file logging with rotation"""
        try:
            # Create log directory
            log_path = Path(self.config.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.log_file_path,
                maxBytes=self.config.max_log_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setFormatter(formatter)
            
            self.root_logger.addHandler(file_handler)
            self.handlers.append(file_handler)
            
            self.logger.info(f"File logging enabled: {self.config.log_file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup file logging: {e}")
    
    def _setup_elasticsearch_logging(self, formatter: StructuredFormatter):
        """Setup Elasticsearch logging"""
        try:
            es_handler = ElasticsearchHandler(
                self.config.elasticsearch_hosts,
                self.config.elasticsearch_index,
                self.config.batch_size,
                self.config.flush_interval
            )
            es_handler.setFormatter(formatter)
            
            self.root_logger.addHandler(es_handler)
            self.handlers.append(es_handler)
            
            self.logger.info(f"Elasticsearch logging enabled: {self.config.elasticsearch_hosts}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Elasticsearch logging: {e}")
    
    def _setup_logstash_logging(self, formatter: StructuredFormatter):
        """Setup Logstash logging"""
        try:
            logstash_handler = LogstashHandler(
                self.config.logstash_host,
                self.config.logstash_port
            )
            logstash_handler.setFormatter(formatter)
            
            self.root_logger.addHandler(logstash_handler)
            self.handlers.append(logstash_handler)
            
            self.logger.info(f"Logstash logging enabled: {self.config.logstash_host}:{self.config.logstash_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Logstash logging: {e}")
    
    def _setup_console_logging(self):
        """Setup console logging for development"""
        try:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            
            self.root_logger.addHandler(console_handler)
            self.handlers.append(console_handler)
            
            self.logger.info("Console logging enabled for development")
            
        except Exception as e:
            self.logger.error(f"Failed to setup console logging: {e}")
    
    def _log_filter(self, record: logging.LogRecord) -> bool:
        """Filter to collect log statistics"""
        with self._stats_lock:
            self.log_stats["total_logs"] += 1
            level = record.levelname
            self.log_stats["logs_by_level"][level] = self.log_stats["logs_by_level"].get(level, 0) + 1
            
            if record.levelno >= logging.ERROR:
                self.log_stats["errors"] += 1
        
        return True
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name"""
        return logging.getLogger(name)
    
    def log_with_context(self, logger: logging.Logger, level: str, message: str,
                        trace_id: Optional[str] = None,
                        span_id: Optional[str] = None,
                        user_id: Optional[str] = None,
                        request_id: Optional[str] = None,
                        **extra_fields):
        """
        Log message with additional context.
        
        Args:
            logger: Logger instance
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            trace_id: Trace ID for distributed tracing
            span_id: Span ID for distributed tracing
            user_id: User ID
            request_id: Request ID
            **extra_fields: Additional fields to include in log
        """
        # Create log record with extra context
        extra = extra_fields.copy()
        if trace_id:
            extra['trace_id'] = trace_id
        if span_id:
            extra['span_id'] = span_id
        if user_id:
            extra['user_id'] = user_id
        if request_id:
            extra['request_id'] = request_id
        
        # Log with appropriate level
        log_method = getattr(logger, level.lower())
        log_method(message, extra=extra)
    
    def log_api_request(self, method: str, endpoint: str, status_code: int,
                       duration: float, user_id: Optional[str] = None,
                       request_id: Optional[str] = None):
        """Log API request information"""
        logger = self.get_logger("api")
        self.log_with_context(
            logger, "INFO",
            f"API request: {method} {endpoint}",
            user_id=user_id,
            request_id=request_id,
            http_method=method,
            http_endpoint=endpoint,
            http_status_code=status_code,
            response_time_ms=duration * 1000,
            component="api"
        )
    
    def log_database_operation(self, operation: str, table: str, duration: float,
                             rows_affected: Optional[int] = None,
                             query: Optional[str] = None):
        """Log database operation information"""
        logger = self.get_logger("database")
        extra_fields = {
            "db_operation": operation,
            "db_table": table,
            "duration_ms": duration * 1000,
            "component": "database"
        }
        
        if rows_affected is not None:
            extra_fields["rows_affected"] = rows_affected
        
        if query:
            # Truncate query for security
            extra_fields["db_query"] = query[:500] + "..." if len(query) > 500 else query
        
        self.log_with_context(
            logger, "INFO",
            f"Database operation: {operation} on {table}",
            **extra_fields
        )
    
    def log_ml_operation(self, model_name: str, operation: str, duration: float,
                        model_version: Optional[str] = None,
                        input_size: Optional[int] = None,
                        accuracy: Optional[float] = None):
        """Log ML operation information"""
        logger = self.get_logger("ml")
        extra_fields = {
            "ml_model_name": model_name,
            "ml_operation": operation,
            "duration_ms": duration * 1000,
            "component": "ml"
        }
        
        if model_version:
            extra_fields["ml_model_version"] = model_version
        if input_size:
            extra_fields["ml_input_size"] = input_size
        if accuracy:
            extra_fields["ml_accuracy"] = accuracy
        
        self.log_with_context(
            logger, "INFO",
            f"ML operation: {operation} with model {model_name}",
            **extra_fields
        )
    
    def log_stock_analysis(self, symbol: str, analysis_type: str, duration: float,
                          result: Optional[Dict[str, Any]] = None):
        """Log stock analysis operation"""
        logger = self.get_logger("stock_analysis")
        extra_fields = {
            "stock_symbol": symbol,
            "analysis_type": analysis_type,
            "duration_ms": duration * 1000,
            "component": "stock_analysis"
        }
        
        if result:
            # Add selected result fields (be careful not to log sensitive data)
            if "pattern_count" in result:
                extra_fields["pattern_count"] = result["pattern_count"]
            if "confidence_score" in result:
                extra_fields["confidence_score"] = result["confidence_score"]
        
        self.log_with_context(
            logger, "INFO",
            f"Stock analysis: {analysis_type} for {symbol}",
            **extra_fields
        )
    
    def log_error(self, logger: logging.Logger, message: str, exception: Exception,
                 trace_id: Optional[str] = None, **extra_fields):
        """Log error with exception details"""
        extra_fields.update({
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "component": "error_handler"
        })
        
        self.log_with_context(
            logger, "ERROR", message,
            trace_id=trace_id,
            **extra_fields
        )
    
    def flush_all_handlers(self):
        """Flush all logging handlers"""
        for handler in self.handlers:
            try:
                handler.flush()
            except Exception as e:
                self.logger.error(f"Error flushing handler {handler}: {e}")
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        with self._stats_lock:
            stats = self.log_stats.copy()
        
        # Calculate uptime
        uptime = datetime.now() - stats["start_time"]
        stats["uptime_seconds"] = uptime.total_seconds()
        
        # Calculate rates
        if stats["uptime_seconds"] > 0:
            stats["logs_per_second"] = stats["total_logs"] / stats["uptime_seconds"]
            stats["error_rate"] = stats["errors"] / stats["total_logs"] if stats["total_logs"] > 0 else 0
        else:
            stats["logs_per_second"] = 0
            stats["error_rate"] = 0
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of logging system"""
        stats = self.get_log_statistics()
        
        # Check handler health
        handler_status = {}
        for handler in self.handlers:
            handler_name = type(handler).__name__
            try:
                # Try to emit a test log (at debug level to avoid spam)
                test_record = logging.LogRecord(
                    name="health_check",
                    level=logging.DEBUG,
                    pathname="",
                    lineno=0,
                    msg="Health check",
                    args=(),
                    exc_info=None
                )
                handler.handle(test_record)
                handler_status[handler_name] = "healthy"
            except Exception as e:
                handler_status[handler_name] = f"unhealthy: {str(e)}"
        
        return {
            "status": "healthy" if all(status == "healthy" for status in handler_status.values()) else "degraded",
            "service_name": self.config.service_name,
            "environment": self.config.environment,
            "handlers": handler_status,
            "elasticsearch_available": ELASTICSEARCH_AVAILABLE,
            "statistics": stats
        }
    
    def shutdown(self):
        """Shutdown logging system"""
        try:
            self.flush_all_handlers()
            
            for handler in self.handlers:
                try:
                    handler.close()
                    self.root_logger.removeHandler(handler)
                except Exception as e:
                    self.logger.error(f"Error closing handler {handler}: {e}")
            
            self.logger.info("ELK logging system shutdown completed")
            
        except Exception as e:
            print(f"Error shutting down logging system: {e}")


# Example usage and testing
if __name__ == "__main__":
    import time
    import random
    
    # Create ELK configuration
    config = ELKConfig(
        service_name="stock_analysis_test",
        environment="development",
        elasticsearch_hosts=["localhost:9200"],
        logstash_host="localhost",
        logstash_port=5000,
        enable_elasticsearch=False,  # Disable for testing without ES
        enable_logstash=False,       # Disable for testing without Logstash
        enable_file_logging=True,
        log_file_path="test_logs/stock_analysis.log"
    )
    
    # Create logging manager
    elk_manager = ELKLoggingManager(config)
    
    try:
        # Get loggers
        api_logger = elk_manager.get_logger("api")
        db_logger = elk_manager.get_logger("database")
        ml_logger = elk_manager.get_logger("ml")
        
        # Test different types of logging
        for i in range(10):
            # API request logging
            elk_manager.log_api_request(
                "GET", f"/api/stocks/AAPL{i}", 200, 0.1 + random.random() * 0.5,
                user_id=f"user_{i % 3}", request_id=f"req_{i}"
            )
            
            # Database operation logging
            elk_manager.log_database_operation(
                "SELECT", "stocks", 0.05 + random.random() * 0.1,
                rows_affected=random.randint(1, 100)
            )
            
            # ML operation logging
            elk_manager.log_ml_operation(
                "pattern_detector", "predict", 0.2 + random.random() * 0.3,
                model_version="v1.0", input_size=1000, accuracy=0.85 + random.random() * 0.1
            )
            
            # Stock analysis logging
            elk_manager.log_stock_analysis(
                f"STOCK{i}", "spring_festival", 0.5 + random.random() * 1.0,
                result={"pattern_count": random.randint(1, 10), "confidence_score": random.random()}
            )
            
            # Error logging
            if i % 5 == 0:
                try:
                    raise ValueError(f"Test error {i}")
                except Exception as e:
                    elk_manager.log_error(api_logger, f"Test error occurred", e, trace_id=f"trace_{i}")
            
            time.sleep(0.1)
        
        # Print statistics
        stats = elk_manager.get_log_statistics()
        print(f"Log statistics: {json.dumps(stats, indent=2, default=str)}")
        
        # Print health status
        health = elk_manager.get_health_status()
        print(f"Health status: {json.dumps(health, indent=2, default=str)}")
        
        print("Logging test completed. Check log files and ELK stack.")
        
    finally:
        elk_manager.shutdown()