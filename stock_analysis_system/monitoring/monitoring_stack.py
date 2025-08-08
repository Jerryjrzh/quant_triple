"""
Comprehensive Monitoring Stack

This module orchestrates the complete monitoring stack including Prometheus metrics,
Grafana dashboards, Jaeger tracing, and ELK logging for the stock analysis system.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from .prometheus_metrics import PrometheusMetricsCollector
from .grafana_dashboards import GrafanaDashboardManager, GrafanaConfig
from .jaeger_tracing import JaegerTracingManager, TracingConfig
from .elk_logging import ELKLogger


@dataclass
class MonitoringStackConfig:
    """Complete monitoring stack configuration"""
    service_name: str
    environment: str = "development"
    
    # Prometheus configuration
    prometheus_port: int = 8000
    prometheus_push_gateway: Optional[str] = None
    metrics_collection_interval: int = 15
    
    # Grafana configuration
    grafana_url: str = "http://localhost:3000"
    grafana_api_key: Optional[str] = None
    
    # Jaeger configuration
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831
    tracing_sampling_rate: float = 1.0
    
    # ELK configuration
    elasticsearch_hosts: List[str] = field(default_factory=lambda: ["localhost:9200"])
    elasticsearch_index: str = "stock-analysis-logs"
    logstash_host: str = "localhost"
    logstash_port: int = 5000
    kibana_url: str = "http://localhost:5601"
    
    # General configuration
    log_level: str = "INFO"
    enable_prometheus: bool = True
    enable_grafana: bool = True
    enable_jaeger: bool = True
    enable_elk: bool = True
    health_check_interval: int = 60


class MonitoringStack:
    """
    Comprehensive monitoring stack orchestrator.
    
    This class manages the complete observability stack including:
    - Prometheus metrics collection and exposure
    - Grafana dashboard management and deployment
    - Jaeger distributed tracing
    - ELK centralized logging
    - Health monitoring and alerting
    - Unified configuration and lifecycle management
    """
    
    def __init__(self, config: MonitoringStackConfig):
        """
        Initialize monitoring stack.
        
        Args:
            config: Monitoring stack configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Component instances
        self.prometheus_collector: Optional[PrometheusMetricsCollector] = None
        self.grafana_manager: Optional[GrafanaDashboardManager] = None
        self.jaeger_tracer: Optional[JaegerTracingManager] = None
        self.elk_manager: Optional[ELKLoggingSystem] = None
        
        # Health monitoring
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_health_check = threading.Event()
        self._component_health: Dict[str, Dict[str, Any]] = {}
        self._health_lock = threading.Lock()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all monitoring components"""
        try:
            # Initialize Prometheus metrics collector
            if self.config.enable_prometheus:
                self._initialize_prometheus()
            
            # Initialize Grafana dashboard manager
            if self.config.enable_grafana and self.config.grafana_api_key:
                self._initialize_grafana()
            
            # Initialize Jaeger tracing
            if self.config.enable_jaeger:
                self._initialize_jaeger()
            
            # Initialize ELK logging
            if self.config.enable_elk:
                self._initialize_elk()
            
            self.logger.info("Monitoring stack initialization completed")
            
        except Exception as e:
            self.logger.error(f"Error initializing monitoring stack: {e}")
            raise
    
    def _initialize_prometheus(self):
        """Initialize Prometheus metrics collector"""
        try:
            self.prometheus_collector = PrometheusMetricsCollector(
                push_gateway_url=self.config.prometheus_push_gateway,
                job_name=self.config.service_name,
                collection_interval=self.config.metrics_collection_interval
            )
            
            # Start metrics collection
            self.prometheus_collector.start_collection()
            
            # Start HTTP server for metrics endpoint
            self.prometheus_collector.start_http_server(self.config.prometheus_port)
            
            self.logger.info(f"Prometheus metrics initialized on port {self.config.prometheus_port}")
            
        except Exception as e:
            self.logger.error(f"Error initializing Prometheus: {e}")
            self.prometheus_collector = None
    
    def _initialize_grafana(self):
        """Initialize Grafana dashboard manager"""
        try:
            grafana_config = GrafanaConfig(
                url=self.config.grafana_url,
                api_key=self.config.grafana_api_key,
                org_id=1
            )
            
            self.grafana_manager = GrafanaDashboardManager(grafana_config)
            
            self.logger.info(f"Grafana dashboard manager initialized: {self.config.grafana_url}")
            
        except Exception as e:
            self.logger.error(f"Error initializing Grafana: {e}")
            self.grafana_manager = None
    
    def _initialize_jaeger(self):
        """Initialize Jaeger tracing"""
        try:
            tracing_config = TracingConfig(
                service_name=self.config.service_name,
                jaeger_endpoint=self.config.jaeger_endpoint,
                agent_host=self.config.jaeger_agent_host,
                agent_port=self.config.jaeger_agent_port,
                sampling_rate=self.config.tracing_sampling_rate
            )
            
            self.jaeger_tracer = JaegerTracingManager(tracing_config)
            
            self.logger.info(f"Jaeger tracing initialized: {self.config.jaeger_endpoint}")
            
        except Exception as e:
            self.logger.error(f"Error initializing Jaeger: {e}")
            self.jaeger_tracer = None
    
    def _initialize_elk(self):
        """Initialize ELK logging"""
        try:
            self.elk_manager = ELKLoggingSystem(
                elasticsearch_hosts=self.config.elasticsearch_hosts,
                log_level=self.config.log_level
            )
            
            self.logger.info("ELK logging initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing ELK: {e}")
            self.elk_manager = None
    
    def start(self):
        """Start the monitoring stack"""
        try:
            self.logger.info("Starting monitoring stack...")
            
            # Deploy Grafana dashboards if available
            if self.grafana_manager:
                self._deploy_dashboards()
            
            # Start health monitoring
            self._start_health_monitoring()
            
            self.logger.info("Monitoring stack started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring stack: {e}")
            raise
    
    def _deploy_dashboards(self):
        """Deploy Grafana dashboards"""
        try:
            results = self.grafana_manager.deploy_all_dashboards()
            
            successful = sum(1 for r in results.values() if r.get("status") == "success")
            total = len(results)
            
            self.logger.info(f"Deployed {successful}/{total} Grafana dashboards")
            
            # Log any failures
            for name, result in results.items():
                if result.get("status") != "success":
                    self.logger.error(f"Failed to deploy dashboard {name}: {result.get('error')}")
            
        except Exception as e:
            self.logger.error(f"Error deploying Grafana dashboards: {e}")
    
    def _start_health_monitoring(self):
        """Start health monitoring thread"""
        if self._health_check_thread and self._health_check_thread.is_alive():
            return
        
        self._stop_health_check.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def _health_check_loop(self):
        """Health check monitoring loop"""
        while not self._stop_health_check.is_set():
            try:
                self._check_component_health()
                
                # Wait for next check
                self._stop_health_check.wait(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                time.sleep(10)  # Brief pause before retrying
    
    def _check_component_health(self):
        """Check health of all components"""
        health_status = {}
        
        # Check Prometheus
        if self.prometheus_collector:
            try:
                health_status["prometheus"] = self.prometheus_collector.get_health_status()
            except Exception as e:
                health_status["prometheus"] = {"status": "unhealthy", "error": str(e)}
        
        # Check Grafana
        if self.grafana_manager:
            try:
                health_status["grafana"] = self.grafana_manager.get_health_status()
            except Exception as e:
                health_status["grafana"] = {"status": "unhealthy", "error": str(e)}
        
        # Check Jaeger
        if self.jaeger_tracer:
            try:
                health_status["jaeger"] = self.jaeger_tracer.get_health_status()
            except Exception as e:
                health_status["jaeger"] = {"status": "unhealthy", "error": str(e)}
        
        # Check ELK
        if self.elk_manager:
            try:
                health_status["elk"] = self.elk_manager.get_health_status()
            except Exception as e:
                health_status["elk"] = {"status": "unhealthy", "error": str(e)}
        
        # Update health status
        with self._health_lock:
            self._component_health = health_status
        
        # Log unhealthy components
        for component, status in health_status.items():
            if status.get("status") != "healthy":
                self.logger.warning(f"Component {component} is unhealthy: {status}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of monitoring stack"""
        with self._health_lock:
            component_health = self._component_health.copy()
        
        # Determine overall status
        overall_status = "healthy"
        unhealthy_components = []
        
        for component, status in component_health.items():
            if status.get("status") != "healthy":
                overall_status = "degraded"
                unhealthy_components.append(component)
        
        if len(unhealthy_components) == len(component_health):
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "service_name": self.config.service_name,
            "environment": self.config.environment,
            "components": component_health,
            "unhealthy_components": unhealthy_components,
            "last_check": datetime.now().isoformat()
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        summary = {}
        
        if self.prometheus_collector:
            try:
                # Get system metrics
                system_metrics = self.prometheus_collector.last_system_metrics
                if system_metrics:
                    summary["system"] = {
                        "cpu_usage": system_metrics.cpu_usage,
                        "memory_usage": system_metrics.memory_usage,
                        "process_count": system_metrics.process_count,
                        "load_average": system_metrics.load_average
                    }
                
                # Get business metrics
                business_metrics = self.prometheus_collector.last_business_metrics
                if business_metrics:
                    summary["business"] = {
                        "api_requests": business_metrics.api_requests_total,
                        "active_users": business_metrics.active_users,
                        "error_rate": business_metrics.error_rate,
                        "cache_hit_rate": business_metrics.cache_hit_rate
                    }
            except Exception as e:
                self.logger.error(f"Error getting metrics summary: {e}")
        
        return summary
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics"""
        if self.jaeger_tracer:
            try:
                return self.jaeger_tracer.get_span_statistics()
            except Exception as e:
                self.logger.error(f"Error getting trace statistics: {e}")
        
        return {}
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        if self.elk_manager:
            try:
                return self.elk_manager.get_log_statistics()
            except Exception as e:
                self.logger.error(f"Error getting log statistics: {e}")
        
        return {}
    
    def record_api_request(self, method: str, endpoint: str, status: int, 
                          duration: float, user_id: Optional[str] = None,
                          request_id: Optional[str] = None):
        """Record API request across all monitoring systems"""
        # Record in Prometheus
        if self.prometheus_collector:
            self.prometheus_collector.record_api_request(method, endpoint, status, duration)
        
        # Log in ELK
        if self.elk_manager:
            self.elk_manager.log_api_request(method, endpoint, status, duration, user_id, request_id)
    
    def record_database_operation(self, operation: str, table: str, duration: float,
                                rows_affected: Optional[int] = None,
                                query: Optional[str] = None):
        """Record database operation across monitoring systems"""
        # Log in ELK
        if self.elk_manager:
            self.elk_manager.log_database_operation(operation, table, duration, rows_affected, query)
    
    def record_ml_operation(self, model_name: str, operation: str, duration: float,
                           model_version: Optional[str] = None,
                           input_size: Optional[int] = None,
                           accuracy: Optional[float] = None):
        """Record ML operation across monitoring systems"""
        # Record in Prometheus
        if self.prometheus_collector:
            self.prometheus_collector.record_ml_prediction(model_name, model_version or "unknown", accuracy)
        
        # Log in ELK
        if self.elk_manager:
            self.elk_manager.log_ml_operation(model_name, operation, duration, model_version, input_size, accuracy)
    
    def record_stock_analysis(self, symbol: str, analysis_type: str, duration: float,
                            result: Optional[Dict[str, Any]] = None):
        """Record stock analysis across monitoring systems"""
        # Record in Prometheus
        if self.prometheus_collector:
            self.prometheus_collector.record_stock_analysis(analysis_type)
        
        # Log in ELK
        if self.elk_manager:
            self.elk_manager.log_stock_analysis(symbol, analysis_type, duration, result)
    
    def create_trace_span(self, operation_name: str, **kwargs):
        """Create a trace span"""
        if self.jaeger_tracer:
            return self.jaeger_tracer.start_span(operation_name, **kwargs)
        else:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()
    
    def trace_function(self, operation_name: Optional[str] = None, **kwargs):
        """Decorator to trace function calls"""
        if self.jaeger_tracer:
            return self.jaeger_tracer.trace_function(operation_name, **kwargs)
        else:
            # Return a no-op decorator
            def decorator(func):
                return func
            return decorator
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance"""
        if self.elk_manager:
            return self.elk_manager.get_logger(name)
        else:
            return logging.getLogger(name)
    
    def export_configuration(self, file_path: str):
        """Export monitoring configuration to file"""
        try:
            config_dict = {
                "service_name": self.config.service_name,
                "environment": self.config.environment,
                "prometheus": {
                    "enabled": self.config.enable_prometheus,
                    "port": self.config.prometheus_port,
                    "push_gateway": self.config.prometheus_push_gateway,
                    "collection_interval": self.config.metrics_collection_interval
                },
                "grafana": {
                    "enabled": self.config.enable_grafana,
                    "url": self.config.grafana_url,
                    "api_key_configured": bool(self.config.grafana_api_key)
                },
                "jaeger": {
                    "enabled": self.config.enable_jaeger,
                    "endpoint": self.config.jaeger_endpoint,
                    "agent_host": self.config.jaeger_agent_host,
                    "agent_port": self.config.jaeger_agent_port,
                    "sampling_rate": self.config.tracing_sampling_rate
                },
                "elk": {
                    "enabled": self.config.enable_elk,
                    "elasticsearch_hosts": self.config.elasticsearch_hosts,
                    "elasticsearch_index": self.config.elasticsearch_index,
                    "logstash_host": self.config.logstash_host,
                    "logstash_port": self.config.logstash_port,
                    "kibana_url": self.config.kibana_url
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"Configuration exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
    
    def stop(self):
        """Stop the monitoring stack"""
        try:
            self.logger.info("Stopping monitoring stack...")
            
            # Stop health monitoring
            if self._health_check_thread:
                self._stop_health_check.set()
                self._health_check_thread.join(timeout=5)
            
            # Stop Prometheus collection
            if self.prometheus_collector:
                self.prometheus_collector.stop_collection()
            
            # Flush Jaeger spans
            if self.jaeger_tracer:
                self.jaeger_tracer.flush_spans()
                self.jaeger_tracer.shutdown()
            
            # Flush ELK logs
            if self.elk_manager:
                self.elk_manager.flush_all_handlers()
                self.elk_manager.shutdown()
            
            self.logger.info("Monitoring stack stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring stack: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


# Example usage and testing
if __name__ == "__main__":
    import time
    import random
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create monitoring stack configuration
    config = MonitoringStackConfig(
        service_name="stock_analysis_test",
        environment="development",
        prometheus_port=8000,
        grafana_url="http://localhost:3000",
        grafana_api_key=None,  # Set this if you have Grafana running
        enable_grafana=False,  # Disable for testing without Grafana
        enable_elk=True,
        log_level="INFO"
    )
    
    # Create and start monitoring stack
    with MonitoringStack(config) as monitoring:
        try:
            # Simulate application activity
            for i in range(20):
                # Record API requests
                monitoring.record_api_request(
                    "GET", f"/api/stocks/AAPL{i % 5}", 
                    200 if random.random() > 0.1 else 500,
                    random.uniform(0.1, 2.0),
                    user_id=f"user_{i % 3}",
                    request_id=f"req_{i}"
                )
                
                # Record database operations
                monitoring.record_database_operation(
                    "SELECT", "stocks", random.uniform(0.01, 0.5),
                    rows_affected=random.randint(1, 100)
                )
                
                # Record ML operations
                monitoring.record_ml_operation(
                    "pattern_detector", "predict", random.uniform(0.1, 1.0),
                    model_version="v1.0", input_size=1000,
                    accuracy=random.uniform(0.8, 0.95)
                )
                
                # Record stock analysis
                monitoring.record_stock_analysis(
                    f"STOCK{i % 10}", "spring_festival", random.uniform(0.5, 3.0),
                    result={"pattern_count": random.randint(1, 5)}
                )
                
                # Use tracing
                with monitoring.create_trace_span("test_operation", tags={"iteration": i}):
                    time.sleep(0.1)
                
                time.sleep(0.5)
            
            # Print health status
            health = monitoring.get_health_status()
            print(f"\nHealth Status:")
            print(json.dumps(health, indent=2, default=str))
            
            # Print metrics summary
            metrics = monitoring.get_metrics_summary()
            print(f"\nMetrics Summary:")
            print(json.dumps(metrics, indent=2, default=str))
            
            # Print trace statistics
            traces = monitoring.get_trace_statistics()
            print(f"\nTrace Statistics:")
            print(json.dumps(traces, indent=2, default=str))
            
            # Print log statistics
            logs = monitoring.get_log_statistics()
            print(f"\nLog Statistics:")
            print(json.dumps(logs, indent=2, default=str))
            
            # Export configuration
            monitoring.export_configuration("monitoring_config.json")
            
            print("\nMonitoring stack test completed!")
            print("Check the following:")
            print("- Prometheus metrics: http://localhost:8000/metrics")
            print("- Log files: logs/stock_analysis.log")
            print("- Configuration: monitoring_config.json")
            
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        except Exception as e:
            print(f"Test error: {e}")
            raise