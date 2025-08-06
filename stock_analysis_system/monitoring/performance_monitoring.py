"""
Application Performance Monitoring System

This module implements comprehensive application performance monitoring for the stock analysis system.
It provides custom metrics for business logic monitoring, performance profiling, alerting for
performance degradation, and capacity planning recommendations.
"""

import time
import psutil
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import functools
from concurrent.futures import ThreadPoolExecutor
import asyncio
import gc
import tracemalloc
from pathlib import Path
import json


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    metric_name: str
    alert_type: str  # warning, critical
    current_value: float
    threshold: float
    timestamp: datetime
    message: str
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class ProfileResult:
    """Performance profiling result"""
    function_name: str
    module_name: str
    total_time: float
    call_count: int
    average_time: float
    max_time: float
    min_time: float
    memory_usage: Optional[int] = None
    cpu_usage: Optional[float] = None


@dataclass
class CapacityRecommendation:
    """Capacity planning recommendation"""
    component: str
    current_usage: float
    projected_usage: float
    recommendation_type: str  # scale_up, scale_down, optimize
    confidence: float
    details: str
    estimated_impact: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """
    Performance profiler for function-level monitoring.
    
    Features:
    - Function execution time tracking
    - Memory usage monitoring
    - CPU usage tracking
    - Call frequency analysis
    - Performance regression detection
    """
    
    def __init__(self, enable_memory_profiling: bool = True):
        self.enable_memory_profiling = enable_memory_profiling
        self.profiles: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.memory_snapshots: Dict[str, List[int]] = defaultdict(list)
        self.lock = threading.Lock()
        
        if enable_memory_profiling:
            tracemalloc.start()
    
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator to profile function performance"""
        def decorator(func: Callable) -> Callable:
            profile_name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                start_memory = None
                
                if self.enable_memory_profiling:
                    try:
                        current, peak = tracemalloc.get_traced_memory()
                        start_memory = current
                    except:
                        pass
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    
                    end_memory = None
                    if self.enable_memory_profiling and start_memory:
                        try:
                            current, peak = tracemalloc.get_traced_memory()
                            end_memory = current - start_memory
                        except:
                            pass
                    
                    with self.lock:
                        self.profiles[profile_name].append(execution_time)
                        self.call_counts[profile_name] += 1
                        
                        if end_memory:
                            self.memory_snapshots[profile_name].append(end_memory)
                        
                        # Keep only recent measurements (last 1000)
                        if len(self.profiles[profile_name]) > 1000:
                            self.profiles[profile_name] = self.profiles[profile_name][-1000:]
                        if len(self.memory_snapshots[profile_name]) > 1000:
                            self.memory_snapshots[profile_name] = self.memory_snapshots[profile_name][-1000:]
            
            return wrapper
        return decorator
    
    def get_profile_results(self) -> List[ProfileResult]:
        """Get profiling results for all functions"""
        results = []
        
        with self.lock:
            for func_name, times in self.profiles.items():
                if not times:
                    continue
                
                memory_usage = None
                if func_name in self.memory_snapshots and self.memory_snapshots[func_name]:
                    memory_usage = statistics.mean(self.memory_snapshots[func_name])
                
                result = ProfileResult(
                    function_name=func_name.split('.')[-1],
                    module_name='.'.join(func_name.split('.')[:-1]),
                    total_time=sum(times),
                    call_count=self.call_counts[func_name],
                    average_time=statistics.mean(times),
                    max_time=max(times),
                    min_time=min(times),
                    memory_usage=memory_usage
                )
                results.append(result)
        
        return sorted(results, key=lambda x: x.total_time, reverse=True)
    
    def reset_profiles(self):
        """Reset all profiling data"""
        with self.lock:
            self.profiles.clear()
            self.call_counts.clear()
            self.memory_snapshots.clear()


class PerformanceMonitor:
    """
    Comprehensive application performance monitor.
    
    Features:
    - Custom business metrics tracking
    - Performance threshold monitoring
    - Alerting for performance degradation
    - Capacity planning and scaling recommendations
    - Performance trend analysis
    - Resource utilization monitoring
    """
    
    def __init__(self, 
                 alert_callback: Optional[Callable[[PerformanceAlert], None]] = None,
                 monitoring_interval: int = 30,
                 history_retention_hours: int = 24):
        """
        Initialize performance monitor.
        
        Args:
            alert_callback: Callback function for performance alerts
            monitoring_interval: Monitoring interval in seconds
            history_retention_hours: How long to retain performance history
        """
        self.alert_callback = alert_callback
        self.monitoring_interval = monitoring_interval
        self.history_retention = timedelta(hours=history_retention_hours)
        
        self.logger = logging.getLogger(__name__)
        
        # Performance data storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        
        # Profiler
        self.profiler = PerformanceProfiler()
        
        # Monitoring thread
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Performance thresholds
        self.thresholds = {
            "api_response_time": {"warning": 1.0, "critical": 5.0},
            "database_query_time": {"warning": 0.5, "critical": 2.0},
            "memory_usage_percent": {"warning": 80.0, "critical": 95.0},
            "cpu_usage_percent": {"warning": 80.0, "critical": 95.0},
            "error_rate_percent": {"warning": 5.0, "critical": 10.0},
            "queue_size": {"warning": 1000, "critical": 5000},
            "cache_hit_rate_percent": {"warning": 70.0, "critical": 50.0}
        }
        
        # Business metrics
        self.business_metrics = {
            "stocks_processed_per_minute": 0,
            "analysis_completion_rate": 0,
            "user_session_duration": 0,
            "data_freshness_minutes": 0
        }
        
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.warning("Performance monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5)
        
        self._executor.shutdown(wait=True)
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect business metrics
                self._collect_business_metrics()
                
                # Check thresholds and generate alerts
                self._check_thresholds()
                
                # Clean old data
                self._cleanup_old_data()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next monitoring cycle
            self._stop_monitoring.wait(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("cpu_usage_percent", cpu_percent, "percent")
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric("memory_usage_percent", memory.percent, "percent")
            self.record_metric("memory_available_mb", memory.available / 1024 / 1024, "MB")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric("disk_usage_percent", disk_percent, "percent")
            
            # Network I/O
            network = psutil.net_io_counters()
            self.record_metric("network_bytes_sent", network.bytes_sent, "bytes")
            self.record_metric("network_bytes_recv", network.bytes_recv, "bytes")
            
            # Process information
            process = psutil.Process()
            self.record_metric("process_memory_mb", process.memory_info().rss / 1024 / 1024, "MB")
            self.record_metric("process_cpu_percent", process.cpu_percent(), "percent")
            self.record_metric("process_threads", process.num_threads(), "count")
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_business_metrics(self):
        """Collect business-specific metrics"""
        try:
            # Record current business metrics
            for metric_name, value in self.business_metrics.items():
                self.record_metric(metric_name, value, "count")
            
        except Exception as e:
            self.logger.error(f"Error collecting business metrics: {e}")
    
    def record_metric(self, name: str, value: float, unit: str, 
                     tags: Optional[Dict[str, str]] = None,
                     threshold_warning: Optional[float] = None,
                     threshold_critical: Optional[float] = None):
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            tags: Additional tags
            threshold_warning: Warning threshold
            threshold_critical: Critical threshold
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            tags=tags or {},
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical
        )
        
        with self.lock:
            self.metrics_history[name].append(metric)
    
    def record_api_performance(self, endpoint: str, method: str, 
                             response_time: float, status_code: int):
        """Record API performance metrics"""
        tags = {"endpoint": endpoint, "method": method, "status": str(status_code)}
        
        self.record_metric(
            "api_response_time",
            response_time,
            "seconds",
            tags=tags,
            threshold_warning=self.thresholds["api_response_time"]["warning"],
            threshold_critical=self.thresholds["api_response_time"]["critical"]
        )
        
        # Record error rate
        is_error = 1 if status_code >= 400 else 0
        self.record_metric("api_error_count", is_error, "count", tags=tags)
    
    def record_database_performance(self, operation: str, table: str, 
                                  query_time: float, rows_affected: int = 0):
        """Record database performance metrics"""
        tags = {"operation": operation, "table": table}
        
        self.record_metric(
            "database_query_time",
            query_time,
            "seconds",
            tags=tags,
            threshold_warning=self.thresholds["database_query_time"]["warning"],
            threshold_critical=self.thresholds["database_query_time"]["critical"]
        )
        
        self.record_metric("database_rows_affected", rows_affected, "count", tags=tags)
    
    def record_ml_performance(self, model_name: str, operation: str,
                            processing_time: float, input_size: int,
                            accuracy: Optional[float] = None):
        """Record ML model performance metrics"""
        tags = {"model": model_name, "operation": operation}
        
        self.record_metric("ml_processing_time", processing_time, "seconds", tags=tags)
        self.record_metric("ml_input_size", input_size, "count", tags=tags)
        
        if accuracy is not None:
            self.record_metric("ml_accuracy", accuracy, "percent", tags=tags)
    
    def record_business_metric(self, name: str, value: float):
        """Record business metric and update internal tracking"""
        with self.lock:
            self.business_metrics[name] = value
        
        self.record_metric(name, value, "count")
    
    def _check_thresholds(self):
        """Check performance thresholds and generate alerts"""
        current_time = datetime.now()
        
        with self.lock:
            for metric_name, metrics in self.metrics_history.items():
                if not metrics:
                    continue
                
                # Get latest metric
                latest_metric = metrics[-1]
                
                # Check if we have thresholds for this metric
                thresholds = self.thresholds.get(metric_name)
                if not thresholds and not latest_metric.threshold_warning:
                    continue
                
                # Determine thresholds
                warning_threshold = latest_metric.threshold_warning or thresholds.get("warning")
                critical_threshold = latest_metric.threshold_critical or thresholds.get("critical")
                
                # Check for threshold violations
                alert_type = None
                threshold_value = None
                
                if critical_threshold and latest_metric.value >= critical_threshold:
                    alert_type = "critical"
                    threshold_value = critical_threshold
                elif warning_threshold and latest_metric.value >= warning_threshold:
                    alert_type = "warning"
                    threshold_value = warning_threshold
                
                # Generate alert if threshold violated
                if alert_type:
                    alert_key = f"{metric_name}_{alert_type}"
                    
                    # Check if alert already exists
                    if alert_key not in self.active_alerts:
                        alert = PerformanceAlert(
                            metric_name=metric_name,
                            alert_type=alert_type,
                            current_value=latest_metric.value,
                            threshold=threshold_value,
                            timestamp=current_time,
                            message=f"{metric_name} ({latest_metric.value:.2f} {latest_metric.unit}) "
                                   f"exceeded {alert_type} threshold ({threshold_value:.2f})"
                        )
                        
                        self.active_alerts[alert_key] = alert
                        self.alert_history.append(alert)
                        
                        # Trigger callback
                        if self.alert_callback:
                            try:
                                self.alert_callback(alert)
                            except Exception as e:
                                self.logger.error(f"Error in alert callback: {e}")
                        
                        self.logger.warning(f"Performance alert: {alert.message}")
                
                # Check for alert resolution
                else:
                    for alert_type in ["warning", "critical"]:
                        alert_key = f"{metric_name}_{alert_type}"
                        if alert_key in self.active_alerts:
                            alert = self.active_alerts[alert_key]
                            alert.resolved = True
                            alert.resolution_time = current_time
                            del self.active_alerts[alert_key]
                            
                            self.logger.info(f"Performance alert resolved: {alert.message}")
    
    def _cleanup_old_data(self):
        """Clean up old performance data"""
        cutoff_time = datetime.now() - self.history_retention
        
        with self.lock:
            for metric_name, metrics in self.metrics_history.items():
                # Remove old metrics
                while metrics and metrics[0].timestamp < cutoff_time:
                    metrics.popleft()
            
            # Clean up old alerts
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.timestamp > cutoff_time
            ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self.lock:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "active_alerts": len(self.active_alerts),
                "total_metrics": sum(len(metrics) for metrics in self.metrics_history.values()),
                "monitoring_interval": self.monitoring_interval,
                "recent_metrics": {}
            }
            
            # Get recent values for key metrics
            for metric_name, metrics in self.metrics_history.items():
                if metrics:
                    latest = metrics[-1]
                    summary["recent_metrics"][metric_name] = {
                        "value": latest.value,
                        "unit": latest.unit,
                        "timestamp": latest.timestamp.isoformat()
                    }
            
            return summary
    
    def get_metric_statistics(self, metric_name: str, 
                            hours: int = 1) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific metric"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            if metric_name not in self.metrics_history:
                return None
            
            metrics = [
                m for m in self.metrics_history[metric_name]
                if m.timestamp > cutoff_time
            ]
            
            if not metrics:
                return None
            
            values = [m.value for m in metrics]
            
            return {
                "metric_name": metric_name,
                "time_range_hours": hours,
                "sample_count": len(values),
                "current_value": values[-1],
                "average": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "unit": metrics[-1].unit
            }
    
    def get_capacity_recommendations(self) -> List[CapacityRecommendation]:
        """Generate capacity planning recommendations"""
        recommendations = []
        
        # CPU capacity recommendation
        cpu_stats = self.get_metric_statistics("cpu_usage_percent", hours=24)
        if cpu_stats:
            if cpu_stats["average"] > 70:
                recommendations.append(CapacityRecommendation(
                    component="CPU",
                    current_usage=cpu_stats["current_value"],
                    projected_usage=cpu_stats["average"] * 1.2,  # 20% growth projection
                    recommendation_type="scale_up",
                    confidence=0.8,
                    details=f"CPU usage averaging {cpu_stats['average']:.1f}% over 24h, consider scaling up",
                    estimated_impact={"performance_improvement": "20-30%", "cost_increase": "moderate"}
                ))
            elif cpu_stats["average"] < 30:
                recommendations.append(CapacityRecommendation(
                    component="CPU",
                    current_usage=cpu_stats["current_value"],
                    projected_usage=cpu_stats["average"],
                    recommendation_type="scale_down",
                    confidence=0.7,
                    details=f"CPU usage averaging {cpu_stats['average']:.1f}% over 24h, consider scaling down",
                    estimated_impact={"cost_savings": "20-40%", "risk": "low"}
                ))
        
        # Memory capacity recommendation
        memory_stats = self.get_metric_statistics("memory_usage_percent", hours=24)
        if memory_stats:
            if memory_stats["average"] > 80:
                recommendations.append(CapacityRecommendation(
                    component="Memory",
                    current_usage=memory_stats["current_value"],
                    projected_usage=memory_stats["average"] * 1.15,
                    recommendation_type="scale_up",
                    confidence=0.9,
                    details=f"Memory usage averaging {memory_stats['average']:.1f}% over 24h, urgent scaling needed",
                    estimated_impact={"performance_improvement": "significant", "stability": "improved"}
                ))
        
        # API performance recommendation
        api_stats = self.get_metric_statistics("api_response_time", hours=6)
        if api_stats:
            if api_stats["average"] > 1.0:
                recommendations.append(CapacityRecommendation(
                    component="API",
                    current_usage=api_stats["current_value"],
                    projected_usage=api_stats["average"],
                    recommendation_type="optimize",
                    confidence=0.8,
                    details=f"API response time averaging {api_stats['average']:.2f}s, optimization needed",
                    estimated_impact={"user_experience": "improved", "throughput": "increased"}
                ))
        
        return recommendations
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get currently active performance alerts"""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[PerformanceAlert]:
        """Get alert history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            return [
                alert for alert in self.alert_history
                if alert.timestamp > cutoff_time
            ]
    
    def export_performance_report(self, file_path: str, hours: int = 24):
        """Export performance report to file"""
        try:
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "time_range_hours": hours,
                "summary": self.get_performance_summary(),
                "capacity_recommendations": [
                    {
                        "component": r.component,
                        "current_usage": r.current_usage,
                        "projected_usage": r.projected_usage,
                        "recommendation_type": r.recommendation_type,
                        "confidence": r.confidence,
                        "details": r.details,
                        "estimated_impact": r.estimated_impact
                    }
                    for r in self.get_capacity_recommendations()
                ],
                "active_alerts": [
                    {
                        "metric_name": a.metric_name,
                        "alert_type": a.alert_type,
                        "current_value": a.current_value,
                        "threshold": a.threshold,
                        "timestamp": a.timestamp.isoformat(),
                        "message": a.message
                    }
                    for a in self.get_active_alerts()
                ],
                "profile_results": [
                    {
                        "function_name": p.function_name,
                        "module_name": p.module_name,
                        "total_time": p.total_time,
                        "call_count": p.call_count,
                        "average_time": p.average_time,
                        "max_time": p.max_time,
                        "min_time": p.min_time,
                        "memory_usage": p.memory_usage
                    }
                    for p in self.profiler.get_profile_results()[:20]  # Top 20
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Performance report exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting performance report: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of performance monitoring"""
        return {
            "status": "healthy" if self._monitoring_thread and self._monitoring_thread.is_alive() else "unhealthy",
            "monitoring_active": self._monitoring_thread and self._monitoring_thread.is_alive(),
            "monitoring_interval": self.monitoring_interval,
            "active_alerts": len(self.active_alerts),
            "total_metrics_tracked": len(self.metrics_history),
            "profiler_enabled": self.profiler.enable_memory_profiling,
            "last_collection": datetime.now().isoformat()
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()


# Example usage and testing
if __name__ == "__main__":
    import random
    import asyncio
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    def alert_handler(alert: PerformanceAlert):
        """Handle performance alerts"""
        print(f"🚨 ALERT: {alert.message}")
    
    # Create performance monitor
    monitor = PerformanceMonitor(
        alert_callback=alert_handler,
        monitoring_interval=5,  # 5 seconds for testing
        history_retention_hours=1
    )
    
    # Example profiled function
    @monitor.profiler.profile_function("test_function")
    def test_function(duration: float):
        time.sleep(duration)
        return f"Completed in {duration}s"
    
    try:
        with monitor:
            print("Performance monitoring started...")
            
            # Simulate application activity
            for i in range(30):
                # Record API performance
                response_time = random.uniform(0.1, 2.0)
                status_code = 200 if random.random() > 0.1 else 500
                monitor.record_api_performance(
                    f"/api/endpoint{i % 3}", "GET", response_time, status_code
                )
                
                # Record database performance
                query_time = random.uniform(0.01, 1.0)
                monitor.record_database_performance(
                    "SELECT", "stocks", query_time, random.randint(1, 100)
                )
                
                # Record ML performance
                processing_time = random.uniform(0.1, 3.0)
                monitor.record_ml_performance(
                    "pattern_detector", "predict", processing_time, 1000, 
                    accuracy=random.uniform(0.8, 0.95)
                )
                
                # Record business metrics
                monitor.record_business_metric("stocks_processed_per_minute", random.randint(50, 200))
                monitor.record_business_metric("analysis_completion_rate", random.uniform(85, 99))
                
                # Test profiled function
                test_function(random.uniform(0.01, 0.1))
                
                time.sleep(0.5)
            
            # Wait for some monitoring cycles
            time.sleep(10)
            
            # Print performance summary
            summary = monitor.get_performance_summary()
            print(f"\nPerformance Summary:")
            print(json.dumps(summary, indent=2, default=str))
            
            # Print capacity recommendations
            recommendations = monitor.get_capacity_recommendations()
            print(f"\nCapacity Recommendations:")
            for rec in recommendations:
                print(f"- {rec.component}: {rec.recommendation_type} ({rec.confidence:.0%} confidence)")
                print(f"  {rec.details}")
            
            # Print active alerts
            alerts = monitor.get_active_alerts()
            print(f"\nActive Alerts: {len(alerts)}")
            for alert in alerts:
                print(f"- {alert.message}")
            
            # Print profiling results
            profiles = monitor.profiler.get_profile_results()
            print(f"\nTop Profiled Functions:")
            for profile in profiles[:5]:
                print(f"- {profile.function_name}: {profile.average_time:.4f}s avg, {profile.call_count} calls")
            
            # Export performance report
            monitor.export_performance_report("performance_report.json", hours=1)
            print("\nPerformance report exported to performance_report.json")
            
            print("\nPerformance monitoring test completed!")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
        raise