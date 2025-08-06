"""
Prometheus Metrics Collection System

This module implements comprehensive metrics collection for the stock analysis system
using Prometheus. It provides system metrics, business metrics, and custom metrics
with proper labeling and aggregation.
"""

import time
import psutil
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    start_http_server, push_to_gateway
)
import threading
from concurrent.futures import ThreadPoolExecutor


@dataclass
class MetricConfig:
    """Configuration for Prometheus metrics"""
    name: str
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None
    metric_type: str = "counter"  # counter, gauge, histogram, summary


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    load_average: List[float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BusinessMetrics:
    """Business logic metrics"""
    api_requests_total: int
    active_users: int
    data_processing_time: float
    error_rate: float
    cache_hit_rate: float
    database_connections: int
    timestamp: datetime = field(default_factory=datetime.now)


class PrometheusMetricsCollector:
    """
    Comprehensive Prometheus metrics collector for the stock analysis system.
    
    Features:
    - System resource monitoring
    - Business logic metrics
    - Custom application metrics
    - Automatic metric registration
    - Push gateway support
    - Multi-threaded collection
    """
    
    def __init__(self, 
                 registry: Optional[CollectorRegistry] = None,
                 push_gateway_url: Optional[str] = None,
                 job_name: str = "stock_analysis_system",
                 collection_interval: int = 15):
        """
        Initialize Prometheus metrics collector.
        
        Args:
            registry: Custom Prometheus registry
            push_gateway_url: Push gateway URL for batch metrics
            job_name: Job name for push gateway
            collection_interval: Metrics collection interval in seconds
        """
        self.registry = registry or CollectorRegistry()
        self.push_gateway_url = push_gateway_url
        self.job_name = job_name
        self.collection_interval = collection_interval
        self.logger = logging.getLogger(__name__)
        
        # Thread management
        self._collection_thread = None
        self._stop_collection = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize metrics
        self._init_system_metrics()
        self._init_business_metrics()
        self._init_custom_metrics()
        
        # Metric storage
        self.last_system_metrics: Optional[SystemMetrics] = None
        self.last_business_metrics: Optional[BusinessMetrics] = None
        
    def _init_system_metrics(self):
        """Initialize system resource metrics"""
        # CPU metrics
        self.cpu_usage_gauge = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            ['core'],
            registry=self.registry
        )
        
        # Memory metrics
        self.memory_usage_gauge = Gauge(
            'system_memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        # Disk metrics
        self.disk_usage_gauge = Gauge(
            'system_disk_usage_bytes',
            'Disk usage in bytes',
            ['device', 'type'],
            registry=self.registry
        )
        
        # Network metrics
        self.network_io_counter = Counter(
            'system_network_io_bytes_total',
            'Network I/O bytes total',
            ['direction'],
            registry=self.registry
        )
        
        # Process metrics
        self.process_count_gauge = Gauge(
            'system_process_count',
            'Number of running processes',
            registry=self.registry
        )
        
        # Load average
        self.load_average_gauge = Gauge(
            'system_load_average',
            'System load average',
            ['period'],
            registry=self.registry
        )
        
    def _init_business_metrics(self):
        """Initialize business logic metrics"""
        # API metrics
        self.api_requests_counter = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # User metrics
        self.active_users_gauge = Gauge(
            'active_users_count',
            'Number of active users',
            registry=self.registry
        )
        
        # Data processing metrics
        self.data_processing_duration = Histogram(
            'data_processing_duration_seconds',
            'Data processing duration',
            ['operation', 'source'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
            registry=self.registry
        )
        
        # Error metrics
        self.error_rate_gauge = Gauge(
            'error_rate_percent',
            'Error rate percentage',
            ['component'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hit_rate_gauge = Gauge(
            'cache_hit_rate_percent',
            'Cache hit rate percentage',
            ['cache_type'],
            registry=self.registry
        )
        
        # Database metrics
        self.database_connections_gauge = Gauge(
            'database_connections_active',
            'Active database connections',
            ['database'],
            registry=self.registry
        )
        
    def _init_custom_metrics(self):
        """Initialize custom application metrics"""
        # Stock analysis specific metrics
        self.stocks_analyzed_counter = Counter(
            'stocks_analyzed_total',
            'Total stocks analyzed',
            ['analysis_type'],
            registry=self.registry
        )
        
        self.spring_festival_patterns_gauge = Gauge(
            'spring_festival_patterns_detected',
            'Number of Spring Festival patterns detected',
            ['pattern_type'],
            registry=self.registry
        )
        
        # Risk management metrics
        self.risk_calculations_counter = Counter(
            'risk_calculations_total',
            'Total risk calculations performed',
            ['calculation_type'],
            registry=self.registry
        )
        
        # ML model metrics
        self.model_predictions_counter = Counter(
            'ml_model_predictions_total',
            'Total ML model predictions',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.model_accuracy_gauge = Gauge(
            'ml_model_accuracy_score',
            'ML model accuracy score',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            for i, usage in enumerate(cpu_percent):
                self.cpu_usage_gauge.labels(core=f"cpu{i}").set(usage)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage_gauge.labels(type="used").set(memory.used)
            self.memory_usage_gauge.labels(type="available").set(memory.available)
            self.memory_usage_gauge.labels(type="total").set(memory.total)
            
            # Disk usage
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    device = partition.device.replace('/', '_')
                    self.disk_usage_gauge.labels(device=device, type="used").set(usage.used)
                    self.disk_usage_gauge.labels(device=device, type="free").set(usage.free)
                    self.disk_usage_gauge.labels(device=device, type="total").set(usage.total)
                except PermissionError:
                    continue
            
            # Network I/O
            network = psutil.net_io_counters()
            self.network_io_counter.labels(direction="sent").inc(network.bytes_sent)
            self.network_io_counter.labels(direction="recv").inc(network.bytes_recv)
            
            # Process count
            self.process_count_gauge.set(len(psutil.pids()))
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                self.load_average_gauge.labels(period="1min").set(load_avg[0])
                self.load_average_gauge.labels(period="5min").set(load_avg[1])
                self.load_average_gauge.labels(period="15min").set(load_avg[2])
            except AttributeError:
                # Windows doesn't have load average
                pass
            
            # Create metrics object
            metrics = SystemMetrics(
                cpu_usage=sum(cpu_percent) / len(cpu_percent),
                memory_usage=memory.percent,
                disk_usage=0,  # Will be calculated separately
                network_io={
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv
                },
                process_count=len(psutil.pids()),
                load_average=list(load_avg) if 'load_avg' in locals() else [0, 0, 0]
            )
            
            self.last_system_metrics = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(0, 0, 0, {}, 0, [0, 0, 0])
    
    def collect_business_metrics(self) -> BusinessMetrics:
        """Collect business logic metrics"""
        try:
            # This would typically be populated by the application
            # For now, we'll create a placeholder structure
            metrics = BusinessMetrics(
                api_requests_total=0,
                active_users=0,
                data_processing_time=0.0,
                error_rate=0.0,
                cache_hit_rate=0.0,
                database_connections=0
            )
            
            self.last_business_metrics = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting business metrics: {e}")
            return BusinessMetrics(0, 0, 0.0, 0.0, 0.0, 0)
    
    def record_api_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record API request metrics"""
        self.api_requests_counter.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        self.api_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_data_processing(self, operation: str, source: str, duration: float):
        """Record data processing metrics"""
        self.data_processing_duration.labels(
            operation=operation,
            source=source
        ).observe(duration)
    
    def record_stock_analysis(self, analysis_type: str, count: int = 1):
        """Record stock analysis metrics"""
        self.stocks_analyzed_counter.labels(
            analysis_type=analysis_type
        ).inc(count)
    
    def record_spring_festival_pattern(self, pattern_type: str, count: int):
        """Record Spring Festival pattern detection"""
        self.spring_festival_patterns_gauge.labels(
            pattern_type=pattern_type
        ).set(count)
    
    def record_risk_calculation(self, calculation_type: str, count: int = 1):
        """Record risk calculation metrics"""
        self.risk_calculations_counter.labels(
            calculation_type=calculation_type
        ).inc(count)
    
    def record_ml_prediction(self, model_name: str, model_version: str, 
                           accuracy: Optional[float] = None, count: int = 1):
        """Record ML model prediction metrics"""
        self.model_predictions_counter.labels(
            model_name=model_name,
            model_version=model_version
        ).inc(count)
        
        if accuracy is not None:
            self.model_accuracy_gauge.labels(
                model_name=model_name,
                model_version=model_version
            ).set(accuracy)
    
    def update_active_users(self, count: int):
        """Update active users count"""
        self.active_users_gauge.set(count)
    
    def update_error_rate(self, component: str, rate: float):
        """Update error rate for a component"""
        self.error_rate_gauge.labels(component=component).set(rate)
    
    def update_cache_hit_rate(self, cache_type: str, rate: float):
        """Update cache hit rate"""
        self.cache_hit_rate_gauge.labels(cache_type=cache_type).set(rate)
    
    def update_database_connections(self, database: str, count: int):
        """Update database connection count"""
        self.database_connections_gauge.labels(database=database).set(count)
    
    def start_collection(self):
        """Start automatic metrics collection"""
        if self._collection_thread and self._collection_thread.is_alive():
            self.logger.warning("Metrics collection already running")
            return
        
        self._stop_collection.clear()
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self._collection_thread.start()
        self.logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop automatic metrics collection"""
        if self._collection_thread:
            self._stop_collection.set()
            self._collection_thread.join(timeout=5)
            self.logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """Main collection loop"""
        while not self._stop_collection.is_set():
            try:
                # Collect metrics in parallel
                futures = [
                    self._executor.submit(self.collect_system_metrics),
                    self._executor.submit(self.collect_business_metrics)
                ]
                
                # Wait for completion
                for future in futures:
                    future.result(timeout=10)
                
                # Push to gateway if configured
                if self.push_gateway_url:
                    self.push_metrics()
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
            
            # Wait for next collection
            self._stop_collection.wait(self.collection_interval)
    
    def push_metrics(self):
        """Push metrics to Prometheus push gateway"""
        if not self.push_gateway_url:
            return
        
        try:
            from prometheus_client.gateway import push_to_gateway
            push_to_gateway(
                self.push_gateway_url,
                job=self.job_name,
                registry=self.registry
            )
            self.logger.debug("Pushed metrics to gateway")
        except Exception as e:
            self.logger.error(f"Error pushing metrics to gateway: {e}")
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format"""
        return generate_latest(self.registry).decode('utf-8')
    
    def start_http_server(self, port: int = 8000):
        """Start HTTP server for metrics endpoint"""
        try:
            start_http_server(port, registry=self.registry)
            self.logger.info(f"Started Prometheus metrics server on port {port}")
        except Exception as e:
            self.logger.error(f"Error starting metrics server: {e}")
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of metrics collection"""
        return {
            "status": "healthy" if self._collection_thread and self._collection_thread.is_alive() else "unhealthy",
            "collection_interval": self.collection_interval,
            "push_gateway_configured": self.push_gateway_url is not None,
            "last_system_metrics": self.last_system_metrics.timestamp if self.last_system_metrics else None,
            "last_business_metrics": self.last_business_metrics.timestamp if self.last_business_metrics else None,
            "registry_metrics_count": len(list(self.registry._collector_to_names.keys()))
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start_collection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_collection()
        self._executor.shutdown(wait=True)


# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create metrics collector
    collector = PrometheusMetricsCollector(
        collection_interval=5,
        job_name="stock_analysis_test"
    )
    
    try:
        # Start collection
        collector.start_collection()
        
        # Start HTTP server
        collector.start_http_server(8000)
        
        # Simulate some metrics
        for i in range(10):
            collector.record_api_request("GET", "/api/stocks", 200, 0.5)
            collector.record_stock_analysis("spring_festival", 1)
            collector.record_ml_prediction("pattern_detector", "v1.0", 0.85)
            collector.update_active_users(i * 10)
            
            time.sleep(2)
        
        print("Metrics collection running. Check http://localhost:8000/metrics")
        print("Press Ctrl+C to stop...")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping metrics collection...")
    finally:
        collector.stop_collection()