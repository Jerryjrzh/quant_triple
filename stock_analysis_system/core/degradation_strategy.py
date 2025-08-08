"""
System Degradation Strategy Implementation

This module implements a comprehensive system degradation strategy that automatically
reduces system functionality and performance to maintain core operations during
high error rates, resource constraints, or external service failures.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import threading
import json
from pathlib import Path

from .error_handler import ErrorHandler, ErrorCategory, ErrorSeverity


class DegradationLevel(str, Enum):
    """System degradation levels"""
    NORMAL = "normal"
    LIGHT = "light"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class DegradationTrigger(str, Enum):
    """Triggers that can cause system degradation"""
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    RESOURCE_USAGE = "resource_usage"
    EXTERNAL_SERVICE_FAILURE = "external_service_failure"
    DATABASE_ISSUES = "database_issues"
    CACHE_FAILURE = "cache_failure"
    MANUAL = "manual"


class ServicePriority(str, Enum):
    """Service priority levels for degradation decisions"""
    CRITICAL = "critical"      # Core functionality, never degrade
    HIGH = "high"             # Important features, degrade only in severe cases
    MEDIUM = "medium"         # Standard features, degrade in moderate cases
    LOW = "low"              # Nice-to-have features, degrade first


@dataclass
class DegradationRule:
    """Rule defining when and how to degrade a service"""
    rule_id: str
    name: str
    description: str
    trigger: DegradationTrigger
    threshold: float
    degradation_level: DegradationLevel
    affected_services: List[str]
    actions: List[str]
    recovery_threshold: Optional[float] = None
    cooldown_period: int = 300  # seconds
    enabled: bool = True


@dataclass
class ServiceConfig:
    """Configuration for a service in the degradation system"""
    service_name: str
    priority: ServicePriority
    degradation_actions: Dict[DegradationLevel, List[str]]
    health_check_endpoint: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    resource_limits: Dict[str, float] = field(default_factory=dict)


@dataclass
class DegradationEvent:
    """Record of a degradation event"""
    event_id: str
    timestamp: datetime
    trigger: DegradationTrigger
    level: DegradationLevel
    affected_services: List[str]
    actions_taken: List[str]
    trigger_value: float
    threshold: float
    auto_triggered: bool = True
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class SystemMetrics:
    """System metrics collector for degradation decisions"""
    
    def __init__(self):
        self.metrics = {
            'error_rate': deque(maxlen=100),
            'response_times': deque(maxlen=100),
            'cpu_usage': deque(maxlen=50),
            'memory_usage': deque(maxlen=50),
            'database_connections': deque(maxlen=50),
            'cache_hit_rate': deque(maxlen=50),
            'active_requests': deque(maxlen=100)
        }
        self.lock = threading.Lock()
        self.last_update = datetime.now()
    
    def update_metric(self, metric_name: str, value: float):
        """Update a system metric"""
        with self.lock:
            if metric_name in self.metrics:
                self.metrics[metric_name].append({
                    'value': value,
                    'timestamp': datetime.now()
                })
                self.last_update = datetime.now()
    
    def get_metric_average(self, metric_name: str, window_minutes: int = 5) -> float:
        """Get average value of a metric over time window"""
        with self.lock:
            if metric_name not in self.metrics:
                return 0.0
            
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            recent_values = [
                entry['value'] for entry in self.metrics[metric_name]
                if entry['timestamp'] > cutoff_time
            ]
            
            return sum(recent_values) / len(recent_values) if recent_values else 0.0
    
    def get_metric_trend(self, metric_name: str, window_minutes: int = 10) -> str:
        """Get trend direction of a metric (increasing, decreasing, stable)"""
        with self.lock:
            if metric_name not in self.metrics or len(self.metrics[metric_name]) < 2:
                return "stable"
            
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            recent_values = [
                entry['value'] for entry in self.metrics[metric_name]
                if entry['timestamp'] > cutoff_time
            ]
            
            if len(recent_values) < 2:
                return "stable"
            
            # Simple trend calculation
            first_half = recent_values[:len(recent_values)//2]
            second_half = recent_values[len(recent_values)//2:]
            
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg > first_avg * 1.1:
                return "increasing"
            elif second_avg < first_avg * 0.9:
                return "decreasing"
            else:
                return "stable"


class DegradationStrategy:
    """
    Main system degradation strategy implementation.
    
    This class monitors system health and automatically degrades functionality
    to maintain core operations during adverse conditions.
    """
    
    def __init__(self, 
                 error_handler: ErrorHandler,
                 config_file: Optional[str] = None,
                 enable_auto_degradation: bool = True):
        """
        Initialize degradation strategy.
        
        Args:
            error_handler: Error handler instance for monitoring errors
            config_file: Path to configuration file
            enable_auto_degradation: Enable automatic degradation
        """
        self.error_handler = error_handler
        self.enable_auto_degradation = enable_auto_degradation
        
        self.logger = logging.getLogger(__name__)
        
        # System state
        self.current_level = DegradationLevel.NORMAL
        self.active_degradations: Set[str] = set()
        self.degradation_history: deque = deque(maxlen=1000)
        
        # Configuration
        self.rules: List[DegradationRule] = []
        self.services: Dict[str, ServiceConfig] = {}
        self.degradation_actions: Dict[str, Callable] = {}
        
        # Monitoring
        self.metrics = SystemMetrics()
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_interval = 30  # seconds
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_degradations': 0,
            'auto_degradations': 0,
            'manual_degradations': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'current_degraded_services': 0
        }
        
        # Initialize default configuration
        self._initialize_default_config()
        
        # Load custom configuration if provided
        if config_file:
            self._load_config(config_file)
        
        self.logger.info("DegradationStrategy initialized")
    
    def _initialize_default_config(self):
        """Initialize default degradation rules and service configurations"""
        
        # Default degradation rules
        default_rules = [
            DegradationRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                description="Degrade when error rate exceeds threshold",
                trigger=DegradationTrigger.ERROR_RATE,
                threshold=0.1,  # 10% error rate
                degradation_level=DegradationLevel.LIGHT,
                affected_services=["data_collection", "analysis"],
                actions=["reduce_collection_frequency", "disable_complex_analysis"],
                recovery_threshold=0.05,
                cooldown_period=300
            ),
            DegradationRule(
                rule_id="critical_error_rate",
                name="Critical Error Rate",
                description="Severe degradation for very high error rates",
                trigger=DegradationTrigger.ERROR_RATE,
                threshold=0.25,  # 25% error rate
                degradation_level=DegradationLevel.SEVERE,
                affected_services=["data_collection", "analysis", "visualization"],
                actions=["minimal_data_collection", "disable_analysis", "basic_visualization"],
                recovery_threshold=0.15,
                cooldown_period=600
            ),
            DegradationRule(
                rule_id="slow_response_time",
                name="Slow Response Time",
                description="Degrade when response time is too high",
                trigger=DegradationTrigger.RESPONSE_TIME,
                threshold=5.0,  # 5 seconds
                degradation_level=DegradationLevel.MODERATE,
                affected_services=["api", "visualization"],
                actions=["reduce_api_complexity", "simplify_charts"],
                recovery_threshold=2.0,
                cooldown_period=180
            ),
            DegradationRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                description="Degrade when memory usage is critical",
                trigger=DegradationTrigger.RESOURCE_USAGE,
                threshold=0.85,  # 85% memory usage
                degradation_level=DegradationLevel.MODERATE,
                affected_services=["caching", "data_processing"],
                actions=["clear_cache", "reduce_batch_size"],
                recovery_threshold=0.70,
                cooldown_period=120
            ),
            DegradationRule(
                rule_id="database_issues",
                name="Database Connection Issues",
                description="Degrade when database has problems",
                trigger=DegradationTrigger.DATABASE_ISSUES,
                threshold=0.5,  # 50% database error rate
                degradation_level=DegradationLevel.SEVERE,
                affected_services=["data_storage", "historical_analysis"],
                actions=["use_cache_only", "disable_historical_queries"],
                recovery_threshold=0.1,
                cooldown_period=300
            )
        ]
        
        for rule in default_rules:
            self.rules.append(rule)
        
        # Default service configurations
        default_services = {
            "data_collection": ServiceConfig(
                service_name="data_collection",
                priority=ServicePriority.HIGH,
                degradation_actions={
                    DegradationLevel.LIGHT: ["reduce_frequency_25"],
                    DegradationLevel.MODERATE: ["reduce_frequency_50"],
                    DegradationLevel.SEVERE: ["minimal_collection"],
                    DegradationLevel.CRITICAL: ["disable_collection"]
                },
                dependencies=["database", "external_apis"]
            ),
            "analysis": ServiceConfig(
                service_name="analysis",
                priority=ServicePriority.MEDIUM,
                degradation_actions={
                    DegradationLevel.LIGHT: ["disable_complex_analysis"],
                    DegradationLevel.MODERATE: ["basic_analysis_only"],
                    DegradationLevel.SEVERE: ["disable_analysis"],
                    DegradationLevel.CRITICAL: ["disable_analysis"]
                },
                dependencies=["data_collection", "database"]
            ),
            "visualization": ServiceConfig(
                service_name="visualization",
                priority=ServicePriority.MEDIUM,
                degradation_actions={
                    DegradationLevel.LIGHT: ["reduce_chart_complexity"],
                    DegradationLevel.MODERATE: ["basic_charts_only"],
                    DegradationLevel.SEVERE: ["minimal_visualization"],
                    DegradationLevel.CRITICAL: ["disable_visualization"]
                },
                dependencies=["analysis"]
            ),
            "api": ServiceConfig(
                service_name="api",
                priority=ServicePriority.CRITICAL,
                degradation_actions={
                    DegradationLevel.LIGHT: ["reduce_api_features"],
                    DegradationLevel.MODERATE: ["basic_api_only"],
                    DegradationLevel.SEVERE: ["minimal_api"],
                    DegradationLevel.CRITICAL: ["emergency_api_only"]
                }
            ),
            "caching": ServiceConfig(
                service_name="caching",
                priority=ServicePriority.HIGH,
                degradation_actions={
                    DegradationLevel.LIGHT: ["reduce_cache_size"],
                    DegradationLevel.MODERATE: ["clear_old_cache"],
                    DegradationLevel.SEVERE: ["minimal_cache"],
                    DegradationLevel.CRITICAL: ["disable_cache"]
                }
            )
        }
        
        self.services.update(default_services)
    
    def _load_config(self, config_file: str):
        """Load configuration from file"""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Load rules
                if 'rules' in config:
                    for rule_data in config['rules']:
                        rule = DegradationRule(**rule_data)
                        self.rules.append(rule)
                
                # Load services
                if 'services' in config:
                    for service_name, service_data in config['services'].items():
                        service = ServiceConfig(**service_data)
                        self.services[service_name] = service
                
                self.logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
    
    def register_degradation_action(self, action_name: str, action_func: Callable):
        """Register a degradation action function"""
        self.degradation_actions[action_name] = action_func
        self.logger.info(f"Registered degradation action: {action_name}")
    
    def add_rule(self, rule: DegradationRule):
        """Add a custom degradation rule"""
        with self.lock:
            self.rules.append(rule)
            self.logger.info(f"Added degradation rule: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a degradation rule"""
        with self.lock:
            for i, rule in enumerate(self.rules):
                if rule.rule_id == rule_id:
                    del self.rules[i]
                    self.logger.info(f"Removed degradation rule: {rule_id}")
                    return True
            return False
    
    def add_service(self, service: ServiceConfig):
        """Add a service configuration"""
        with self.lock:
            self.services[service.service_name] = service
            self.logger.info(f"Added service configuration: {service.service_name}")
    
    async def start_monitoring(self):
        """Start the monitoring task"""
        if self.monitoring_task is None or self.monitoring_task.done():
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Started degradation monitoring")
    
    async def stop_monitoring(self):
        """Stop the monitoring task"""
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Stopped degradation monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                await self._check_degradation_conditions()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _check_degradation_conditions(self):
        """Check all degradation conditions and trigger if necessary"""
        if not self.enable_auto_degradation:
            return
        
        # Update system metrics
        await self._update_system_metrics()
        
        # Check each rule
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                should_trigger = await self._evaluate_rule(rule)
                
                if should_trigger:
                    await self._trigger_degradation(rule)
                else:
                    # Check for recovery
                    await self._check_recovery(rule)
                    
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
    
    async def _update_system_metrics(self):
        """Update system metrics from various sources"""
        try:
            # Get error statistics from error handler
            error_stats = self.error_handler.get_error_statistics()
            total_errors = error_stats.get('total_errors', 0)
            
            if total_errors > 0:
                # Calculate recent error rate
                recent_errors = len([
                    error for error in self.error_handler.error_history
                    if (datetime.now() - error.context.timestamp).total_seconds() <= 300
                ])
                error_rate = recent_errors / 100  # Approximate rate
                self.metrics.update_metric('error_rate', error_rate)
            
            # Update other metrics (these would come from actual monitoring systems)
            # For now, we'll use placeholder values
            import psutil
            
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent / 100
            
            self.metrics.update_metric('cpu_usage', cpu_percent / 100)
            self.metrics.update_metric('memory_usage', memory_percent)
            
        except Exception as e:
            self.logger.error(f"Error updating system metrics: {e}")
    
    async def _evaluate_rule(self, rule: DegradationRule) -> bool:
        """Evaluate if a degradation rule should trigger"""
        try:
            if rule.trigger == DegradationTrigger.ERROR_RATE:
                current_value = self.metrics.get_metric_average('error_rate', 5)
                return current_value > rule.threshold
            
            elif rule.trigger == DegradationTrigger.RESPONSE_TIME:
                current_value = self.metrics.get_metric_average('response_times', 5)
                return current_value > rule.threshold
            
            elif rule.trigger == DegradationTrigger.RESOURCE_USAGE:
                memory_usage = self.metrics.get_metric_average('memory_usage', 3)
                cpu_usage = self.metrics.get_metric_average('cpu_usage', 3)
                return max(memory_usage, cpu_usage) > rule.threshold
            
            elif rule.trigger == DegradationTrigger.DATABASE_ISSUES:
                # Check database-related errors
                db_error_rate = self._get_category_error_rate(ErrorCategory.DATABASE)
                return db_error_rate > rule.threshold
            
            elif rule.trigger == DegradationTrigger.EXTERNAL_SERVICE_FAILURE:
                # Check external API errors
                api_error_rate = self._get_category_error_rate(ErrorCategory.EXTERNAL_API)
                return api_error_rate > rule.threshold
            
            elif rule.trigger == DegradationTrigger.CACHE_FAILURE:
                cache_hit_rate = self.metrics.get_metric_average('cache_hit_rate', 5)
                return cache_hit_rate < rule.threshold
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
            return False
    
    def _get_category_error_rate(self, category: ErrorCategory) -> float:
        """Get error rate for a specific category"""
        try:
            recent_errors = [
                error for error in self.error_handler.error_history
                if (datetime.now() - error.context.timestamp).total_seconds() <= 300
                and error.category == category
            ]
            
            total_recent = len([
                error for error in self.error_handler.error_history
                if (datetime.now() - error.context.timestamp).total_seconds() <= 300
            ])
            
            return len(recent_errors) / max(total_recent, 1)
            
        except Exception:
            return 0.0
    
    async def _trigger_degradation(self, rule: DegradationRule):
        """Trigger degradation based on a rule"""
        with self.lock:
            # Check cooldown period
            recent_events = [
                event for event in self.degradation_history
                if event.trigger == rule.trigger 
                and (datetime.now() - event.timestamp).total_seconds() < rule.cooldown_period
            ]
            
            if recent_events:
                return  # Still in cooldown
            
            # Create degradation event
            event = DegradationEvent(
                event_id=f"{rule.rule_id}_{int(time.time())}",
                timestamp=datetime.now(),
                trigger=rule.trigger,
                level=rule.degradation_level,
                affected_services=rule.affected_services.copy(),
                actions_taken=[],
                trigger_value=0.0,  # Would be set with actual metric value
                threshold=rule.threshold,
                auto_triggered=True
            )
            
            # Execute degradation actions
            for action in rule.actions:
                try:
                    if action in self.degradation_actions:
                        await self.degradation_actions[action]()
                        event.actions_taken.append(action)
                    else:
                        # Default action handling
                        await self._execute_default_action(action, rule.affected_services)
                        event.actions_taken.append(action)
                        
                except Exception as e:
                    self.logger.error(f"Failed to execute degradation action {action}: {e}")
            
            # Update system state
            self.current_level = max(self.current_level, rule.degradation_level, key=lambda x: list(DegradationLevel).index(x))
            self.active_degradations.update(rule.affected_services)
            self.degradation_history.append(event)
            
            # Update statistics
            self.stats['total_degradations'] += 1
            self.stats['auto_degradations'] += 1
            self.stats['current_degraded_services'] = len(self.active_degradations)
            
            self.logger.warning(
                f"Degradation triggered: {rule.name} - Level: {rule.degradation_level.value} - "
                f"Services: {', '.join(rule.affected_services)}"
            )
    
    async def _execute_default_action(self, action: str, services: List[str]):
        """Execute default degradation actions"""
        if action == "reduce_collection_frequency":
            # Reduce data collection frequency
            self.logger.info("Reducing data collection frequency")
        
        elif action == "disable_complex_analysis":
            # Disable complex analysis features
            self.logger.info("Disabling complex analysis features")
        
        elif action == "clear_cache":
            # Clear system cache
            self.logger.info("Clearing system cache")
        
        elif action == "reduce_api_complexity":
            # Reduce API response complexity
            self.logger.info("Reducing API response complexity")
        
        elif action == "minimal_data_collection":
            # Switch to minimal data collection
            self.logger.info("Switching to minimal data collection")
        
        # Add more default actions as needed
    
    async def _check_recovery(self, rule: DegradationRule):
        """Check if system can recover from degradation"""
        if rule.recovery_threshold is None:
            return
        
        # Find active degradation events for this rule
        active_events = [
            event for event in self.degradation_history
            if event.trigger == rule.trigger 
            and not event.resolved
            and any(service in self.active_degradations for service in event.affected_services)
        ]
        
        if not active_events:
            return
        
        # Check if recovery conditions are met
        can_recover = False
        
        if rule.trigger == DegradationTrigger.ERROR_RATE:
            current_value = self.metrics.get_metric_average('error_rate', 5)
            can_recover = current_value < rule.recovery_threshold
        
        elif rule.trigger == DegradationTrigger.RESPONSE_TIME:
            current_value = self.metrics.get_metric_average('response_times', 5)
            can_recover = current_value < rule.recovery_threshold
        
        elif rule.trigger == DegradationTrigger.RESOURCE_USAGE:
            memory_usage = self.metrics.get_metric_average('memory_usage', 3)
            cpu_usage = self.metrics.get_metric_average('cpu_usage', 3)
            can_recover = max(memory_usage, cpu_usage) < rule.recovery_threshold
        
        if can_recover:
            await self._recover_from_degradation(active_events)
    
    async def _recover_from_degradation(self, events: List[DegradationEvent]):
        """Recover from degradation"""
        with self.lock:
            for event in events:
                # Mark event as resolved
                event.resolved = True
                event.resolution_time = datetime.now()
                
                # Remove services from active degradations
                for service in event.affected_services:
                    self.active_degradations.discard(service)
                
                # Execute recovery actions (reverse of degradation actions)
                for action in event.actions_taken:
                    try:
                        recovery_action = f"recover_{action}"
                        if recovery_action in self.degradation_actions:
                            await self.degradation_actions[recovery_action]()
                    except Exception as e:
                        self.logger.error(f"Failed to execute recovery action {recovery_action}: {e}")
                
                self.logger.info(f"Recovered from degradation: {event.event_id}")
            
            # Update system state
            if not self.active_degradations:
                self.current_level = DegradationLevel.NORMAL
            
            # Update statistics
            self.stats['successful_recoveries'] += len(events)
            self.stats['current_degraded_services'] = len(self.active_degradations)
    
    async def manual_degradation(self, 
                                level: DegradationLevel, 
                                services: List[str], 
                                reason: str = "Manual trigger") -> str:
        """Manually trigger system degradation"""
        with self.lock:
            event = DegradationEvent(
                event_id=f"manual_{int(time.time())}",
                timestamp=datetime.now(),
                trigger=DegradationTrigger.MANUAL,
                level=level,
                affected_services=services,
                actions_taken=[],
                trigger_value=0.0,
                threshold=0.0,
                auto_triggered=False
            )
            
            # Execute degradation for each service
            for service_name in services:
                if service_name in self.services:
                    service = self.services[service_name]
                    actions = service.degradation_actions.get(level, [])
                    
                    for action in actions:
                        try:
                            if action in self.degradation_actions:
                                await self.degradation_actions[action]()
                            else:
                                await self._execute_default_action(action, [service_name])
                            event.actions_taken.append(f"{service_name}:{action}")
                        except Exception as e:
                            self.logger.error(f"Failed to execute manual degradation action {action}: {e}")
            
            # Update system state
            self.current_level = max(self.current_level, level, key=lambda x: list(DegradationLevel).index(x))
            self.active_degradations.update(services)
            self.degradation_history.append(event)
            
            # Update statistics
            self.stats['total_degradations'] += 1
            self.stats['manual_degradations'] += 1
            self.stats['current_degraded_services'] = len(self.active_degradations)
            
            self.logger.warning(f"Manual degradation triggered: {reason} - Level: {level.value}")
            
            return event.event_id
    
    async def manual_recovery(self, event_id: Optional[str] = None) -> bool:
        """Manually recover from degradation"""
        with self.lock:
            if event_id:
                # Recover specific event
                events_to_recover = [
                    event for event in self.degradation_history
                    if event.event_id == event_id and not event.resolved
                ]
            else:
                # Recover all active degradations
                events_to_recover = [
                    event for event in self.degradation_history
                    if not event.resolved
                ]
            
            if events_to_recover:
                await self._recover_from_degradation(events_to_recover)
                return True
            
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system degradation status"""
        with self.lock:
            active_events = [
                {
                    'event_id': event.event_id,
                    'trigger': event.trigger.value,
                    'level': event.level.value,
                    'affected_services': event.affected_services,
                    'timestamp': event.timestamp.isoformat(),
                    'auto_triggered': event.auto_triggered
                }
                for event in self.degradation_history
                if not event.resolved
            ]
            
            return {
                'current_level': self.current_level.value,
                'active_degradations': list(self.active_degradations),
                'active_events': active_events,
                'total_services': len(self.services),
                'degraded_services': len(self.active_degradations),
                'monitoring_enabled': self.enable_auto_degradation,
                'last_metric_update': self.metrics.last_update.isoformat(),
                'statistics': self.stats.copy()
            }
    
    def get_degradation_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get degradation history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            return [
                {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp.isoformat(),
                    'trigger': event.trigger.value,
                    'level': event.level.value,
                    'affected_services': event.affected_services,
                    'actions_taken': event.actions_taken,
                    'resolved': event.resolved,
                    'resolution_time': event.resolution_time.isoformat() if event.resolution_time else None,
                    'auto_triggered': event.auto_triggered
                }
                for event in self.degradation_history
                if event.timestamp > cutoff_time
            ]
    
    def export_degradation_report(self, file_path: str):
        """Export comprehensive degradation report"""
        try:
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'system_status': self.get_system_status(),
                'degradation_history': self.get_degradation_history(168),  # 1 week
                'rules': [
                    {
                        'rule_id': rule.rule_id,
                        'name': rule.name,
                        'trigger': rule.trigger.value,
                        'threshold': rule.threshold,
                        'level': rule.degradation_level.value,
                        'enabled': rule.enabled
                    }
                    for rule in self.rules
                ],
                'services': {
                    name: {
                        'priority': service.priority.value,
                        'dependencies': service.dependencies
                    }
                    for name, service in self.services.items()
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Degradation report exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export degradation report: {e}")


# Global degradation strategy instance
_degradation_strategy: Optional[DegradationStrategy] = None


def get_degradation_strategy() -> Optional[DegradationStrategy]:
    """Get the global degradation strategy instance"""
    return _degradation_strategy


def initialize_degradation_strategy(error_handler: ErrorHandler, 
                                  config_file: Optional[str] = None,
                                  enable_auto_degradation: bool = True) -> DegradationStrategy:
    """Initialize the global degradation strategy"""
    global _degradation_strategy
    _degradation_strategy = DegradationStrategy(
        error_handler=error_handler,
        config_file=config_file,
        enable_auto_degradation=enable_auto_degradation
    )
    return _degradation_strategy