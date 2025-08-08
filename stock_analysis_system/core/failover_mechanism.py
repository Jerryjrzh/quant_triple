"""
Failover Mechanism Implementation

This module implements a comprehensive failover mechanism that automatically
switches between primary and backup resources (data sources, databases, cache)
when failures are detected, ensuring system continuity and high availability.
"""

import logging
import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import json
from pathlib import Path
import random

from .error_handler import ErrorHandler, ErrorCategory, ErrorSeverity


class ResourceType(str, Enum):
    """Types of resources that can failover"""
    DATA_SOURCE = "data_source"
    DATABASE = "database"
    CACHE = "cache"
    API_ENDPOINT = "api_endpoint"
    SERVICE = "service"
    STORAGE = "storage"


class ResourceStatus(str, Enum):
    """Status of a resource"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"


class FailoverStrategy(str, Enum):
    """Failover strategies"""
    IMMEDIATE = "immediate"           # Switch immediately on failure
    GRADUAL = "gradual"              # Gradually shift traffic
    LOAD_BALANCED = "load_balanced"   # Distribute across available resources
    PRIORITY_BASED = "priority_based" # Use highest priority available resource


@dataclass
class ResourceConfig:
    """Configuration for a failover resource"""
    resource_id: str
    resource_type: ResourceType
    name: str
    connection_string: str
    priority: int = 1  # Lower number = higher priority
    weight: float = 1.0  # For load balancing
    health_check_url: Optional[str] = None
    health_check_interval: int = 30  # seconds
    max_failures: int = 3
    failure_timeout: int = 300  # seconds before retry
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    resource_id: str
    timestamp: datetime
    status: ResourceStatus
    response_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailoverEvent:
    """Record of a failover event"""
    event_id: str
    timestamp: datetime
    resource_type: ResourceType
    from_resource: str
    to_resource: str
    reason: str
    strategy: FailoverStrategy
    success: bool
    response_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResourceHealthMonitor:
    """
    Monitors health of resources and triggers failover when needed.
    """
    
    def __init__(self, failover_manager: 'FailoverManager'):
        self.failover_manager = failover_manager
        self.logger = logging.getLogger(__name__)
        
        # Health check tasks
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Failure tracking
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.last_failure_time: Dict[str, datetime] = {}
        
        # Custom health check functions
        self.custom_health_checks: Dict[str, Callable] = {}
        
        self.logger.info("ResourceHealthMonitor initialized")
    
    def register_health_check(self, resource_id: str, health_check_func: Callable):
        """Register a custom health check function for a resource"""
        self.custom_health_checks[resource_id] = health_check_func
        self.logger.info(f"Registered custom health check for resource: {resource_id}")
    
    async def start_monitoring(self, resource_configs: List[ResourceConfig]):
        """Start health monitoring for all resources"""
        for config in resource_configs:
            if config.resource_id not in self.health_check_tasks:
                task = asyncio.create_task(
                    self._monitor_resource_health(config)
                )
                self.health_check_tasks[config.resource_id] = task
                self.logger.info(f"Started health monitoring for: {config.resource_id}")
    
    async def stop_monitoring(self):
        """Stop all health monitoring tasks"""
        for resource_id, task in self.health_check_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                self.logger.info(f"Stopped health monitoring for: {resource_id}")
        
        self.health_check_tasks.clear()
    
    async def _monitor_resource_health(self, config: ResourceConfig):
        """Monitor health of a single resource"""
        while True:
            try:
                # Perform health check
                health_result = await self._perform_health_check(config)
                
                # Store health result
                self.health_history[config.resource_id].append(health_result)
                
                # Update resource status in failover manager
                await self.failover_manager._update_resource_status(
                    config.resource_id, 
                    health_result.status
                )
                
                # Check if failover is needed
                if health_result.status in [ResourceStatus.FAILED, ResourceStatus.DEGRADED]:
                    await self._handle_resource_failure(config, health_result)
                elif health_result.status == ResourceStatus.HEALTHY:
                    await self._handle_resource_recovery(config, health_result)
                
                # Wait for next check
                await asyncio.sleep(config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error monitoring resource {config.resource_id}: {e}")
                await asyncio.sleep(config.health_check_interval)
    
    async def _perform_health_check(self, config: ResourceConfig) -> HealthCheckResult:
        """Perform health check on a resource"""
        start_time = time.time()
        
        try:
            # Use custom health check if available
            if config.resource_id in self.custom_health_checks:
                result = await self.custom_health_checks[config.resource_id](config)
                if isinstance(result, HealthCheckResult):
                    return result
                else:
                    # Convert boolean result to HealthCheckResult
                    status = ResourceStatus.HEALTHY if result else ResourceStatus.FAILED
                    return HealthCheckResult(
                        resource_id=config.resource_id,
                        timestamp=datetime.now(),
                        status=status,
                        response_time=time.time() - start_time
                    )
            
            # Default health checks based on resource type
            if config.resource_type == ResourceType.DATABASE:
                return await self._check_database_health(config, start_time)
            elif config.resource_type == ResourceType.DATA_SOURCE:
                return await self._check_data_source_health(config, start_time)
            elif config.resource_type == ResourceType.CACHE:
                return await self._check_cache_health(config, start_time)
            elif config.resource_type == ResourceType.API_ENDPOINT:
                return await self._check_api_health(config, start_time)
            else:
                return await self._check_generic_health(config, start_time)
                
        except Exception as e:
            return HealthCheckResult(
                resource_id=config.resource_id,
                timestamp=datetime.now(),
                status=ResourceStatus.FAILED,
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _check_database_health(self, config: ResourceConfig, start_time: float) -> HealthCheckResult:
        """Check database health"""
        try:
            # Simulate database health check
            # In real implementation, this would connect to the database and run a simple query
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Random failure simulation for demo
            if random.random() < 0.05:  # 5% failure rate
                raise Exception("Database connection timeout")
            
            response_time = time.time() - start_time
            
            # Determine status based on response time
            if response_time > 2.0:
                status = ResourceStatus.DEGRADED
            else:
                status = ResourceStatus.HEALTHY
            
            return HealthCheckResult(
                resource_id=config.resource_id,
                timestamp=datetime.now(),
                status=status,
                response_time=response_time,
                metadata={'connection_pool_size': 10, 'active_connections': 3}
            )
            
        except Exception as e:
            return HealthCheckResult(
                resource_id=config.resource_id,
                timestamp=datetime.now(),
                status=ResourceStatus.FAILED,
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _check_data_source_health(self, config: ResourceConfig, start_time: float) -> HealthCheckResult:
        """Check data source health"""
        try:
            # Simulate data source health check
            await asyncio.sleep(0.05)  # Simulate API call
            
            # Random failure simulation
            if random.random() < 0.03:  # 3% failure rate
                raise Exception("Data source API rate limit exceeded")
            
            response_time = time.time() - start_time
            
            status = ResourceStatus.HEALTHY if response_time < 1.0 else ResourceStatus.DEGRADED
            
            return HealthCheckResult(
                resource_id=config.resource_id,
                timestamp=datetime.now(),
                status=status,
                response_time=response_time,
                metadata={'api_quota_remaining': 1000, 'last_update': datetime.now().isoformat()}
            )
            
        except Exception as e:
            return HealthCheckResult(
                resource_id=config.resource_id,
                timestamp=datetime.now(),
                status=ResourceStatus.FAILED,
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _check_cache_health(self, config: ResourceConfig, start_time: float) -> HealthCheckResult:
        """Check cache health"""
        try:
            # Simulate cache health check
            await asyncio.sleep(0.02)  # Simulate cache ping
            
            # Random failure simulation
            if random.random() < 0.02:  # 2% failure rate
                raise Exception("Cache server unreachable")
            
            response_time = time.time() - start_time
            
            return HealthCheckResult(
                resource_id=config.resource_id,
                timestamp=datetime.now(),
                status=ResourceStatus.HEALTHY,
                response_time=response_time,
                metadata={'memory_usage': 0.65, 'hit_rate': 0.85}
            )
            
        except Exception as e:
            return HealthCheckResult(
                resource_id=config.resource_id,
                timestamp=datetime.now(),
                status=ResourceStatus.FAILED,
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _check_api_health(self, config: ResourceConfig, start_time: float) -> HealthCheckResult:
        """Check API endpoint health"""
        try:
            # Simulate API health check
            if config.health_check_url:
                # In real implementation, make HTTP request to health_check_url
                await asyncio.sleep(0.1)
            
            # Random failure simulation
            if random.random() < 0.04:  # 4% failure rate
                raise Exception("API endpoint returned 503 Service Unavailable")
            
            response_time = time.time() - start_time
            
            status = ResourceStatus.HEALTHY if response_time < 0.5 else ResourceStatus.DEGRADED
            
            return HealthCheckResult(
                resource_id=config.resource_id,
                timestamp=datetime.now(),
                status=status,
                response_time=response_time,
                metadata={'status_code': 200, 'version': '1.0.0'}
            )
            
        except Exception as e:
            return HealthCheckResult(
                resource_id=config.resource_id,
                timestamp=datetime.now(),
                status=ResourceStatus.FAILED,
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _check_generic_health(self, config: ResourceConfig, start_time: float) -> HealthCheckResult:
        """Generic health check for unknown resource types"""
        try:
            await asyncio.sleep(0.05)
            
            return HealthCheckResult(
                resource_id=config.resource_id,
                timestamp=datetime.now(),
                status=ResourceStatus.HEALTHY,
                response_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheckResult(
                resource_id=config.resource_id,
                timestamp=datetime.now(),
                status=ResourceStatus.FAILED,
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _handle_resource_failure(self, config: ResourceConfig, health_result: HealthCheckResult):
        """Handle resource failure"""
        self.failure_counts[config.resource_id] += 1
        self.last_failure_time[config.resource_id] = datetime.now()
        
        self.logger.warning(
            f"Resource failure detected: {config.resource_id} - "
            f"Status: {health_result.status.value} - "
            f"Failures: {self.failure_counts[config.resource_id]}/{config.max_failures}"
        )
        
        # Trigger failover if max failures reached
        if self.failure_counts[config.resource_id] >= config.max_failures:
            await self.failover_manager.trigger_failover(
                resource_type=config.resource_type,
                failed_resource=config.resource_id,
                reason=f"Max failures reached: {health_result.error_message or 'Health check failed'}"
            )
    
    async def _handle_resource_recovery(self, config: ResourceConfig, health_result: HealthCheckResult):
        """Handle resource recovery"""
        if config.resource_id in self.failure_counts and self.failure_counts[config.resource_id] > 0:
            self.logger.info(f"Resource recovered: {config.resource_id}")
            self.failure_counts[config.resource_id] = 0
            
            # Notify failover manager of recovery
            await self.failover_manager.handle_resource_recovery(config.resource_id)
    
    def get_health_status(self, resource_id: str) -> Optional[HealthCheckResult]:
        """Get latest health status for a resource"""
        if resource_id in self.health_history and self.health_history[resource_id]:
            return self.health_history[resource_id][-1]
        return None
    
    def get_health_history(self, resource_id: str, hours: int = 24) -> List[HealthCheckResult]:
        """Get health history for a resource"""
        if resource_id not in self.health_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            result for result in self.health_history[resource_id]
            if result.timestamp > cutoff_time
        ]


class FailoverManager:
    """
    Main failover manager that coordinates resource failover and recovery.
    """
    
    def __init__(self, 
                 error_handler: ErrorHandler,
                 default_strategy: FailoverStrategy = FailoverStrategy.PRIORITY_BASED):
        """
        Initialize failover manager.
        
        Args:
            error_handler: Error handler for logging failures
            default_strategy: Default failover strategy
        """
        self.error_handler = error_handler
        self.default_strategy = default_strategy
        
        self.logger = logging.getLogger(__name__)
        
        # Resource management
        self.resources: Dict[str, ResourceConfig] = {}
        self.resource_groups: Dict[ResourceType, List[str]] = defaultdict(list)
        self.active_resources: Dict[ResourceType, str] = {}
        self.resource_status: Dict[str, ResourceStatus] = {}
        
        # Failover tracking
        self.failover_history: deque = deque(maxlen=1000)
        self.failover_strategies: Dict[ResourceType, FailoverStrategy] = {}
        
        # Health monitoring
        self.health_monitor = ResourceHealthMonitor(self)
        
        # Custom failover handlers
        self.failover_handlers: Dict[ResourceType, List[Callable]] = defaultdict(list)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_failovers': 0,
            'successful_failovers': 0,
            'failed_failovers': 0,
            'total_recoveries': 0,
            'average_failover_time': 0.0
        }
        
        self.logger.info("FailoverManager initialized")
    
    def add_resource(self, config: ResourceConfig):
        """Add a resource to the failover manager"""
        with self.lock:
            self.resources[config.resource_id] = config
            self.resource_groups[config.resource_type].append(config.resource_id)
            self.resource_status[config.resource_id] = ResourceStatus.HEALTHY
            
            # Set as active resource if it's the first or highest priority
            current_active = self.active_resources.get(config.resource_type)
            if (not current_active or 
                config.priority < self.resources[current_active].priority):
                self.active_resources[config.resource_type] = config.resource_id
            
            self.logger.info(f"Added resource: {config.resource_id} ({config.resource_type.value})")
    
    def remove_resource(self, resource_id: str) -> bool:
        """Remove a resource from the failover manager"""
        with self.lock:
            if resource_id not in self.resources:
                return False
            
            config = self.resources[resource_id]
            
            # Remove from resource groups
            if resource_id in self.resource_groups[config.resource_type]:
                self.resource_groups[config.resource_type].remove(resource_id)
            
            # Update active resource if this was the active one
            if self.active_resources.get(config.resource_type) == resource_id:
                self._select_new_active_resource(config.resource_type)
            
            # Clean up
            del self.resources[resource_id]
            del self.resource_status[resource_id]
            
            self.logger.info(f"Removed resource: {resource_id}")
            return True
    
    def set_failover_strategy(self, resource_type: ResourceType, strategy: FailoverStrategy):
        """Set failover strategy for a resource type"""
        self.failover_strategies[resource_type] = strategy
        self.logger.info(f"Set failover strategy for {resource_type.value}: {strategy.value}")
    
    def register_failover_handler(self, resource_type: ResourceType, handler: Callable):
        """Register a custom failover handler"""
        self.failover_handlers[resource_type].append(handler)
        self.logger.info(f"Registered failover handler for {resource_type.value}")
    
    async def start_monitoring(self):
        """Start health monitoring for all resources"""
        resource_configs = list(self.resources.values())
        await self.health_monitor.start_monitoring(resource_configs)
        self.logger.info("Started failover monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        await self.health_monitor.stop_monitoring()
        self.logger.info("Stopped failover monitoring")
    
    async def trigger_failover(self, 
                              resource_type: ResourceType, 
                              failed_resource: str, 
                              reason: str,
                              strategy: Optional[FailoverStrategy] = None) -> bool:
        """
        Trigger failover for a resource type.
        
        Args:
            resource_type: Type of resource to failover
            failed_resource: ID of the failed resource
            reason: Reason for failover
            strategy: Failover strategy to use (optional)
            
        Returns:
            bool: True if failover was successful
        """
        start_time = time.time()
        
        with self.lock:
            # Determine strategy
            if strategy is None:
                strategy = self.failover_strategies.get(resource_type, self.default_strategy)
            
            # Find target resource
            target_resource = self._select_failover_target(resource_type, failed_resource, strategy)
            
            if not target_resource:
                self.logger.error(f"No available failover target for {resource_type.value}")
                self._record_failover_event(
                    resource_type, failed_resource, None, reason, strategy, False, 
                    time.time() - start_time
                )
                return False
            
            # Execute failover
            try:
                success = await self._execute_failover(
                    resource_type, failed_resource, target_resource, strategy
                )
                
                if success:
                    # Update active resource
                    self.active_resources[resource_type] = target_resource
                    self.resource_status[failed_resource] = ResourceStatus.FAILED
                    
                    # Execute custom handlers
                    await self._execute_failover_handlers(
                        resource_type, failed_resource, target_resource
                    )
                    
                    self.stats['successful_failovers'] += 1
                    self.logger.info(
                        f"Failover successful: {failed_resource} -> {target_resource} "
                        f"({resource_type.value})"
                    )
                else:
                    self.stats['failed_failovers'] += 1
                    self.logger.error(f"Failover failed: {failed_resource} -> {target_resource}")
                
                # Record event
                failover_time = time.time() - start_time
                self._record_failover_event(
                    resource_type, failed_resource, target_resource, reason, 
                    strategy, success, failover_time
                )
                
                # Update statistics
                self.stats['total_failovers'] += 1
                self._update_average_failover_time(failover_time)
                
                return success
                
            except Exception as e:
                self.logger.error(f"Error during failover: {e}")
                self.error_handler.handle_error(e, custom_message=f"Failover error: {reason}")
                
                self._record_failover_event(
                    resource_type, failed_resource, target_resource, reason, 
                    strategy, False, time.time() - start_time
                )
                
                self.stats['failed_failovers'] += 1
                return False
    
    def _select_failover_target(self, 
                               resource_type: ResourceType, 
                               failed_resource: str, 
                               strategy: FailoverStrategy) -> Optional[str]:
        """Select the best failover target based on strategy"""
        available_resources = [
            resource_id for resource_id in self.resource_groups[resource_type]
            if (resource_id != failed_resource and 
                self.resource_status.get(resource_id) == ResourceStatus.HEALTHY)
        ]
        
        if not available_resources:
            return None
        
        if strategy == FailoverStrategy.PRIORITY_BASED:
            # Select highest priority (lowest number) available resource
            return min(available_resources, 
                      key=lambda r: self.resources[r].priority)
        
        elif strategy == FailoverStrategy.LOAD_BALANCED:
            # Select resource with highest weight
            return max(available_resources, 
                      key=lambda r: self.resources[r].weight)
        
        elif strategy == FailoverStrategy.IMMEDIATE:
            # Select first available resource
            return available_resources[0]
        
        elif strategy == FailoverStrategy.GRADUAL:
            # For gradual strategy, select based on current load (simplified)
            return min(available_resources, 
                      key=lambda r: self.resources[r].priority)
        
        else:
            return available_resources[0]
    
    async def _execute_failover(self, 
                               resource_type: ResourceType, 
                               from_resource: str, 
                               to_resource: str, 
                               strategy: FailoverStrategy) -> bool:
        """Execute the actual failover"""
        try:
            # Simulate failover execution
            # In real implementation, this would:
            # - Update connection pools
            # - Redirect traffic
            # - Update configuration
            # - Notify dependent services
            
            await asyncio.sleep(0.1)  # Simulate failover time
            
            self.logger.info(f"Executing failover: {from_resource} -> {to_resource}")
            
            # Simulate potential failover failure
            if random.random() < 0.05:  # 5% chance of failover failure
                raise Exception("Failover target also unavailable")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failover execution failed: {e}")
            return False
    
    async def _execute_failover_handlers(self, 
                                        resource_type: ResourceType, 
                                        from_resource: str, 
                                        to_resource: str):
        """Execute custom failover handlers"""
        handlers = self.failover_handlers.get(resource_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(from_resource, to_resource)
                else:
                    handler(from_resource, to_resource)
            except Exception as e:
                self.logger.error(f"Failover handler error: {e}")
    
    def _select_new_active_resource(self, resource_type: ResourceType):
        """Select new active resource when current one is removed"""
        available_resources = [
            resource_id for resource_id in self.resource_groups[resource_type]
            if self.resource_status.get(resource_id) == ResourceStatus.HEALTHY
        ]
        
        if available_resources:
            # Select highest priority available resource
            new_active = min(available_resources, 
                           key=lambda r: self.resources[r].priority)
            self.active_resources[resource_type] = new_active
            self.logger.info(f"Selected new active resource: {new_active}")
        else:
            # No healthy resources available
            if resource_type in self.active_resources:
                del self.active_resources[resource_type]
            self.logger.warning(f"No healthy resources available for {resource_type.value}")
    
    def _record_failover_event(self, 
                              resource_type: ResourceType, 
                              from_resource: str, 
                              to_resource: Optional[str], 
                              reason: str, 
                              strategy: FailoverStrategy, 
                              success: bool, 
                              response_time: float):
        """Record a failover event"""
        event = FailoverEvent(
            event_id=f"failover_{int(time.time())}_{id(self)}",
            timestamp=datetime.now(),
            resource_type=resource_type,
            from_resource=from_resource,
            to_resource=to_resource or "none",
            reason=reason,
            strategy=strategy,
            success=success,
            response_time=response_time
        )
        
        self.failover_history.append(event)
    
    def _update_average_failover_time(self, failover_time: float):
        """Update average failover time statistic"""
        current_avg = self.stats['average_failover_time']
        total_failovers = self.stats['total_failovers']
        
        if total_failovers == 1:
            self.stats['average_failover_time'] = failover_time
        else:
            # Calculate running average
            self.stats['average_failover_time'] = (
                (current_avg * (total_failovers - 1) + failover_time) / total_failovers
            )
    
    async def _update_resource_status(self, resource_id: str, status: ResourceStatus):
        """Update resource status (called by health monitor)"""
        if resource_id in self.resource_status:
            old_status = self.resource_status[resource_id]
            self.resource_status[resource_id] = status
            
            if old_status != status:
                self.logger.info(f"Resource status changed: {resource_id} {old_status.value} -> {status.value}")
    
    async def handle_resource_recovery(self, resource_id: str):
        """Handle resource recovery"""
        if resource_id not in self.resources:
            return
        
        config = self.resources[resource_id]
        self.resource_status[resource_id] = ResourceStatus.HEALTHY
        
        # Check if this resource should become active again
        current_active = self.active_resources.get(config.resource_type)
        
        if (not current_active or 
            config.priority < self.resources[current_active].priority):
            
            # Switch back to recovered resource if it has higher priority
            old_active = current_active
            self.active_resources[config.resource_type] = resource_id
            
            self.logger.info(f"Switched back to recovered resource: {resource_id}")
            
            # Record recovery event
            self.stats['total_recoveries'] += 1
    
    def get_active_resource(self, resource_type: ResourceType) -> Optional[str]:
        """Get currently active resource for a type"""
        return self.active_resources.get(resource_type)
    
    def get_resource_status(self, resource_id: str) -> Optional[ResourceStatus]:
        """Get status of a specific resource"""
        return self.resource_status.get(resource_id)
    
    def get_all_resources(self, resource_type: Optional[ResourceType] = None) -> Dict[str, ResourceConfig]:
        """Get all resources, optionally filtered by type"""
        if resource_type is None:
            return self.resources.copy()
        else:
            return {
                resource_id: config for resource_id, config in self.resources.items()
                if config.resource_type == resource_type
            }
    
    def get_failover_statistics(self) -> Dict[str, Any]:
        """Get comprehensive failover statistics"""
        with self.lock:
            # Calculate success rate
            total_failovers = self.stats['total_failovers']
            success_rate = (
                (self.stats['successful_failovers'] / total_failovers * 100) 
                if total_failovers > 0 else 0
            )
            
            # Resource status summary
            status_summary = defaultdict(int)
            for status in self.resource_status.values():
                status_summary[status.value] += 1
            
            # Active resources by type
            active_resources = {
                resource_type.value: resource_id 
                for resource_type, resource_id in self.active_resources.items()
            }
            
            return {
                'total_resources': len(self.resources),
                'total_failovers': total_failovers,
                'successful_failovers': self.stats['successful_failovers'],
                'failed_failovers': self.stats['failed_failovers'],
                'success_rate': success_rate,
                'total_recoveries': self.stats['total_recoveries'],
                'average_failover_time': self.stats['average_failover_time'],
                'resource_status_summary': dict(status_summary),
                'active_resources': active_resources,
                'resource_groups': {
                    resource_type.value: resources 
                    for resource_type, resources in self.resource_groups.items()
                }
            }
    
    def get_failover_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get failover history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            {
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'resource_type': event.resource_type.value,
                'from_resource': event.from_resource,
                'to_resource': event.to_resource,
                'reason': event.reason,
                'strategy': event.strategy.value,
                'success': event.success,
                'response_time': event.response_time,
                'metadata': event.metadata
            }
            for event in self.failover_history
            if event.timestamp > cutoff_time
        ]
    
    def export_failover_report(self, file_path: str):
        """Export comprehensive failover report"""
        try:
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'statistics': self.get_failover_statistics(),
                'failover_history': self.get_failover_history(168),  # 1 week
                'resource_configurations': {
                    resource_id: {
                        'name': config.name,
                        'type': config.resource_type.value,
                        'priority': config.priority,
                        'status': self.resource_status.get(resource_id, 'unknown').value
                    }
                    for resource_id, config in self.resources.items()
                },
                'health_status': {
                    resource_id: {
                        'latest_check': self.health_monitor.get_health_status(resource_id).__dict__ 
                        if self.health_monitor.get_health_status(resource_id) else None
                    }
                    for resource_id in self.resources.keys()
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Failover report exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export failover report: {e}")


# Global failover manager instance
_failover_manager: Optional[FailoverManager] = None


def get_failover_manager() -> Optional[FailoverManager]:
    """Get the global failover manager instance"""
    return _failover_manager


def initialize_failover_manager(error_handler: ErrorHandler, 
                               default_strategy: FailoverStrategy = FailoverStrategy.PRIORITY_BASED) -> FailoverManager:
    """Initialize the global failover manager"""
    global _failover_manager
    _failover_manager = FailoverManager(
        error_handler=error_handler,
        default_strategy=default_strategy
    )
    return _failover_manager