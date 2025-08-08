"""
Unit tests for Failover Mechanism

This module contains comprehensive unit tests for the failover mechanism
implementation, covering resource management, health monitoring, and failover logic.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from stock_analysis_system.core.error_handler import ErrorHandler
from stock_analysis_system.core.failover_mechanism import (
    FailoverManager, ResourceHealthMonitor, ResourceType, ResourceStatus,
    FailoverStrategy, ResourceConfig, HealthCheckResult, FailoverEvent,
    initialize_failover_manager
)


class TestResourceConfig:
    """Test ResourceConfig dataclass"""
    
    def test_init(self):
        """Test ResourceConfig initialization"""
        config = ResourceConfig(
            resource_id="test_db",
            resource_type=ResourceType.DATABASE,
            name="Test Database",
            connection_string="postgresql://test",
            priority=1
        )
        
        assert config.resource_id == "test_db"
        assert config.resource_type == ResourceType.DATABASE
        assert config.name == "Test Database"
        assert config.connection_string == "postgresql://test"
        assert config.priority == 1
        assert config.weight == 1.0  # default value
        assert config.health_check_interval == 30  # default value


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass"""
    
    def test_init(self):
        """Test HealthCheckResult initialization"""
        timestamp = datetime.now()
        result = HealthCheckResult(
            resource_id="test_resource",
            timestamp=timestamp,
            status=ResourceStatus.HEALTHY,
            response_time=0.5,
            error_message="Test error"
        )
        
        assert result.resource_id == "test_resource"
        assert result.timestamp == timestamp
        assert result.status == ResourceStatus.HEALTHY
        assert result.response_time == 0.5
        assert result.error_message == "Test error"


class TestResourceHealthMonitor:
    """Test ResourceHealthMonitor class"""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler fixture"""
        return ErrorHandler()
    
    @pytest.fixture
    def failover_manager(self, error_handler):
        """Create failover manager fixture"""
        return FailoverManager(error_handler)
    
    @pytest.fixture
    def health_monitor(self, failover_manager):
        """Create health monitor fixture"""
        return ResourceHealthMonitor(failover_manager)
    
    def test_init(self, health_monitor):
        """Test ResourceHealthMonitor initialization"""
        assert health_monitor.failover_manager is not None
        assert len(health_monitor.health_check_tasks) == 0
        assert len(health_monitor.health_history) == 0
        assert len(health_monitor.failure_counts) == 0
        assert len(health_monitor.custom_health_checks) == 0
    
    def test_register_health_check(self, health_monitor):
        """Test registering custom health check"""
        async def custom_check(config):
            return True
        
        health_monitor.register_health_check("test_resource", custom_check)
        
        assert "test_resource" in health_monitor.custom_health_checks
        assert health_monitor.custom_health_checks["test_resource"] == custom_check
    
    @pytest.mark.asyncio
    async def test_perform_health_check_database(self, health_monitor):
        """Test database health check"""
        config = ResourceConfig(
            resource_id="test_db",
            resource_type=ResourceType.DATABASE,
            name="Test Database",
            connection_string="postgresql://test"
        )
        
        result = await health_monitor._perform_health_check(config)
        
        assert isinstance(result, HealthCheckResult)
        assert result.resource_id == "test_db"
        assert result.status in [ResourceStatus.HEALTHY, ResourceStatus.DEGRADED, ResourceStatus.FAILED]
        assert result.response_time >= 0
    
    @pytest.mark.asyncio
    async def test_perform_health_check_data_source(self, health_monitor):
        """Test data source health check"""
        config = ResourceConfig(
            resource_id="test_api",
            resource_type=ResourceType.DATA_SOURCE,
            name="Test API",
            connection_string="https://api.test.com"
        )
        
        result = await health_monitor._perform_health_check(config)
        
        assert isinstance(result, HealthCheckResult)
        assert result.resource_id == "test_api"
        assert result.status in [ResourceStatus.HEALTHY, ResourceStatus.DEGRADED, ResourceStatus.FAILED]
    
    @pytest.mark.asyncio
    async def test_perform_health_check_custom(self, health_monitor):
        """Test custom health check"""
        async def custom_check(config):
            return HealthCheckResult(
                resource_id=config.resource_id,
                timestamp=datetime.now(),
                status=ResourceStatus.HEALTHY,
                response_time=0.1
            )
        
        config = ResourceConfig(
            resource_id="test_custom",
            resource_type=ResourceType.SERVICE,
            name="Test Custom Service",
            connection_string="custom://test"
        )
        
        health_monitor.register_health_check("test_custom", custom_check)
        result = await health_monitor._perform_health_check(config)
        
        assert result.resource_id == "test_custom"
        assert result.status == ResourceStatus.HEALTHY
        assert result.response_time == 0.1
    
    def test_get_health_status(self, health_monitor):
        """Test getting health status"""
        # Initially no health status
        status = health_monitor.get_health_status("test_resource")
        assert status is None
        
        # Add health result
        result = HealthCheckResult(
            resource_id="test_resource",
            timestamp=datetime.now(),
            status=ResourceStatus.HEALTHY,
            response_time=0.5
        )
        health_monitor.health_history["test_resource"].append(result)
        
        # Should return the result
        status = health_monitor.get_health_status("test_resource")
        assert status == result
    
    def test_get_health_history(self, health_monitor):
        """Test getting health history"""
        resource_id = "test_resource"
        
        # Add some health results
        for i in range(5):
            result = HealthCheckResult(
                resource_id=resource_id,
                timestamp=datetime.now() - timedelta(hours=i),
                status=ResourceStatus.HEALTHY,
                response_time=0.1 * i
            )
            health_monitor.health_history[resource_id].append(result)
        
        # Get history for last 3 hours
        history = health_monitor.get_health_history(resource_id, hours=3)
        
        # Should return results from last 3 hours
        assert len(history) == 4  # 0, 1, 2 hours ago + current


class TestFailoverManager:
    """Test FailoverManager class"""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler fixture"""
        return ErrorHandler()
    
    @pytest.fixture
    def failover_manager(self, error_handler):
        """Create failover manager fixture"""
        return FailoverManager(error_handler)
    
    def test_init(self, failover_manager):
        """Test FailoverManager initialization"""
        assert failover_manager.default_strategy == FailoverStrategy.PRIORITY_BASED
        assert len(failover_manager.resources) == 0
        assert len(failover_manager.resource_groups) == 0
        assert len(failover_manager.active_resources) == 0
        assert len(failover_manager.resource_status) == 0
        assert isinstance(failover_manager.health_monitor, ResourceHealthMonitor)
    
    def test_add_resource(self, failover_manager):
        """Test adding resources"""
        config = ResourceConfig(
            resource_id="test_db",
            resource_type=ResourceType.DATABASE,
            name="Test Database",
            connection_string="postgresql://test",
            priority=1
        )
        
        failover_manager.add_resource(config)
        
        assert "test_db" in failover_manager.resources
        assert failover_manager.resources["test_db"] == config
        assert "test_db" in failover_manager.resource_groups[ResourceType.DATABASE]
        assert failover_manager.resource_status["test_db"] == ResourceStatus.HEALTHY
        assert failover_manager.active_resources[ResourceType.DATABASE] == "test_db"
    
    def test_add_multiple_resources_priority(self, failover_manager):
        """Test adding multiple resources with different priorities"""
        # Add lower priority resource first
        config1 = ResourceConfig(
            resource_id="backup_db",
            resource_type=ResourceType.DATABASE,
            name="Backup Database",
            connection_string="postgresql://backup",
            priority=2
        )
        failover_manager.add_resource(config1)
        
        # Add higher priority resource
        config2 = ResourceConfig(
            resource_id="primary_db",
            resource_type=ResourceType.DATABASE,
            name="Primary Database",
            connection_string="postgresql://primary",
            priority=1
        )
        failover_manager.add_resource(config2)
        
        # Primary should be active due to higher priority (lower number)
        assert failover_manager.active_resources[ResourceType.DATABASE] == "primary_db"
    
    def test_remove_resource(self, failover_manager):
        """Test removing resources"""
        config = ResourceConfig(
            resource_id="test_db",
            resource_type=ResourceType.DATABASE,
            name="Test Database",
            connection_string="postgresql://test",
            priority=1
        )
        
        failover_manager.add_resource(config)
        assert "test_db" in failover_manager.resources
        
        success = failover_manager.remove_resource("test_db")
        
        assert success is True
        assert "test_db" not in failover_manager.resources
        assert "test_db" not in failover_manager.resource_groups[ResourceType.DATABASE]
        assert "test_db" not in failover_manager.resource_status
        assert ResourceType.DATABASE not in failover_manager.active_resources
    
    def test_remove_nonexistent_resource(self, failover_manager):
        """Test removing non-existent resource"""
        success = failover_manager.remove_resource("nonexistent")
        assert success is False
    
    def test_set_failover_strategy(self, failover_manager):
        """Test setting failover strategy"""
        failover_manager.set_failover_strategy(
            ResourceType.DATABASE, 
            FailoverStrategy.LOAD_BALANCED
        )
        
        assert failover_manager.failover_strategies[ResourceType.DATABASE] == FailoverStrategy.LOAD_BALANCED
    
    def test_register_failover_handler(self, failover_manager):
        """Test registering failover handler"""
        def test_handler(from_resource, to_resource):
            pass
        
        failover_manager.register_failover_handler(ResourceType.DATABASE, test_handler)
        
        assert test_handler in failover_manager.failover_handlers[ResourceType.DATABASE]
    
    def test_select_failover_target_priority_based(self, failover_manager):
        """Test selecting failover target with priority-based strategy"""
        # Add resources with different priorities
        configs = [
            ResourceConfig("db1", ResourceType.DATABASE, "DB1", "conn1", priority=3),
            ResourceConfig("db2", ResourceType.DATABASE, "DB2", "conn2", priority=1),
            ResourceConfig("db3", ResourceType.DATABASE, "DB3", "conn3", priority=2)
        ]
        
        for config in configs:
            failover_manager.add_resource(config)
        
        # Select target when db2 (priority 1) fails
        target = failover_manager._select_failover_target(
            ResourceType.DATABASE, "db2", FailoverStrategy.PRIORITY_BASED
        )
        
        # Should select db3 (priority 2) as it's the next highest priority
        assert target == "db3"
    
    def test_select_failover_target_load_balanced(self, failover_manager):
        """Test selecting failover target with load-balanced strategy"""
        # Add resources with different weights
        configs = [
            ResourceConfig("api1", ResourceType.API_ENDPOINT, "API1", "conn1", weight=0.5),
            ResourceConfig("api2", ResourceType.API_ENDPOINT, "API2", "conn2", weight=1.0),
            ResourceConfig("api3", ResourceType.API_ENDPOINT, "API3", "conn3", weight=0.8)
        ]
        
        for config in configs:
            failover_manager.add_resource(config)
        
        # Select target when api1 fails
        target = failover_manager._select_failover_target(
            ResourceType.API_ENDPOINT, "api1", FailoverStrategy.LOAD_BALANCED
        )
        
        # Should select api2 (highest weight)
        assert target == "api2"
    
    def test_select_failover_target_no_available(self, failover_manager):
        """Test selecting failover target when no resources available"""
        config = ResourceConfig(
            resource_id="only_db",
            resource_type=ResourceType.DATABASE,
            name="Only Database",
            connection_string="postgresql://only",
            priority=1
        )
        
        failover_manager.add_resource(config)
        
        # Mark as failed
        failover_manager.resource_status["only_db"] = ResourceStatus.FAILED
        
        target = failover_manager._select_failover_target(
            ResourceType.DATABASE, "only_db", FailoverStrategy.PRIORITY_BASED
        )
        
        assert target is None
    
    @pytest.mark.asyncio
    async def test_trigger_failover_success(self, failover_manager):
        """Test successful failover trigger"""
        # Add resources
        configs = [
            ResourceConfig("primary_db", ResourceType.DATABASE, "Primary", "conn1", priority=1),
            ResourceConfig("backup_db", ResourceType.DATABASE, "Backup", "conn2", priority=2)
        ]
        
        for config in configs:
            failover_manager.add_resource(config)
        
        # Mock execute_failover to always succeed
        async def mock_execute_failover(*args):
            return True
        
        failover_manager._execute_failover = mock_execute_failover
        
        # Trigger failover
        success = await failover_manager.trigger_failover(
            ResourceType.DATABASE, "primary_db", "Test failover"
        )
        
        assert success is True
        assert failover_manager.active_resources[ResourceType.DATABASE] == "backup_db"
        assert failover_manager.resource_status["primary_db"] == ResourceStatus.FAILED
        assert failover_manager.stats['successful_failovers'] == 1
        assert len(failover_manager.failover_history) == 1
    
    @pytest.mark.asyncio
    async def test_trigger_failover_no_target(self, failover_manager):
        """Test failover trigger when no target available"""
        config = ResourceConfig(
            resource_id="only_db",
            resource_type=ResourceType.DATABASE,
            name="Only Database",
            connection_string="postgresql://only",
            priority=1
        )
        
        failover_manager.add_resource(config)
        
        # Trigger failover (no backup available)
        success = await failover_manager.trigger_failover(
            ResourceType.DATABASE, "only_db", "Test failover"
        )
        
        assert success is False
        assert failover_manager.stats['failed_failovers'] == 1
    
    @pytest.mark.asyncio
    async def test_update_resource_status(self, failover_manager):
        """Test updating resource status"""
        config = ResourceConfig(
            resource_id="test_db",
            resource_type=ResourceType.DATABASE,
            name="Test Database",
            connection_string="postgresql://test",
            priority=1
        )
        
        failover_manager.add_resource(config)
        
        # Update status
        await failover_manager._update_resource_status("test_db", ResourceStatus.DEGRADED)
        
        assert failover_manager.resource_status["test_db"] == ResourceStatus.DEGRADED
    
    @pytest.mark.asyncio
    async def test_handle_resource_recovery(self, failover_manager):
        """Test handling resource recovery"""
        # Add resources
        configs = [
            ResourceConfig("primary_db", ResourceType.DATABASE, "Primary", "conn1", priority=1),
            ResourceConfig("backup_db", ResourceType.DATABASE, "Backup", "conn2", priority=2)
        ]
        
        for config in configs:
            failover_manager.add_resource(config)
        
        # Simulate failover to backup
        failover_manager.active_resources[ResourceType.DATABASE] = "backup_db"
        failover_manager.resource_status["primary_db"] = ResourceStatus.FAILED
        
        # Handle recovery of primary
        await failover_manager.handle_resource_recovery("primary_db")
        
        # Should switch back to primary (higher priority)
        assert failover_manager.active_resources[ResourceType.DATABASE] == "primary_db"
        assert failover_manager.resource_status["primary_db"] == ResourceStatus.HEALTHY
        assert failover_manager.stats['total_recoveries'] == 1
    
    def test_get_active_resource(self, failover_manager):
        """Test getting active resource"""
        config = ResourceConfig(
            resource_id="test_db",
            resource_type=ResourceType.DATABASE,
            name="Test Database",
            connection_string="postgresql://test",
            priority=1
        )
        
        failover_manager.add_resource(config)
        
        active = failover_manager.get_active_resource(ResourceType.DATABASE)
        assert active == "test_db"
        
        # Test non-existent type
        active = failover_manager.get_active_resource(ResourceType.CACHE)
        assert active is None
    
    def test_get_resource_status(self, failover_manager):
        """Test getting resource status"""
        config = ResourceConfig(
            resource_id="test_db",
            resource_type=ResourceType.DATABASE,
            name="Test Database",
            connection_string="postgresql://test",
            priority=1
        )
        
        failover_manager.add_resource(config)
        
        status = failover_manager.get_resource_status("test_db")
        assert status == ResourceStatus.HEALTHY
        
        # Test non-existent resource
        status = failover_manager.get_resource_status("nonexistent")
        assert status is None
    
    def test_get_all_resources(self, failover_manager):
        """Test getting all resources"""
        configs = [
            ResourceConfig("db1", ResourceType.DATABASE, "DB1", "conn1"),
            ResourceConfig("api1", ResourceType.API_ENDPOINT, "API1", "conn2")
        ]
        
        for config in configs:
            failover_manager.add_resource(config)
        
        # Get all resources
        all_resources = failover_manager.get_all_resources()
        assert len(all_resources) == 2
        assert "db1" in all_resources
        assert "api1" in all_resources
        
        # Get filtered resources
        db_resources = failover_manager.get_all_resources(ResourceType.DATABASE)
        assert len(db_resources) == 1
        assert "db1" in db_resources
        assert "api1" not in db_resources
    
    def test_get_failover_statistics(self, failover_manager):
        """Test getting failover statistics"""
        # Add some resources
        configs = [
            ResourceConfig("db1", ResourceType.DATABASE, "DB1", "conn1"),
            ResourceConfig("api1", ResourceType.API_ENDPOINT, "API1", "conn2")
        ]
        
        for config in configs:
            failover_manager.add_resource(config)
        
        # Update some statistics
        failover_manager.stats['total_failovers'] = 5
        failover_manager.stats['successful_failovers'] = 4
        failover_manager.stats['failed_failovers'] = 1
        
        stats = failover_manager.get_failover_statistics()
        
        assert stats['total_resources'] == 2
        assert stats['total_failovers'] == 5
        assert stats['successful_failovers'] == 4
        assert stats['failed_failovers'] == 1
        assert stats['success_rate'] == 80.0
        assert 'resource_status_summary' in stats
        assert 'active_resources' in stats
        assert 'resource_groups' in stats
    
    def test_get_failover_history(self, failover_manager):
        """Test getting failover history"""
        # Add some failover events
        events = [
            FailoverEvent(
                event_id="event1",
                timestamp=datetime.now() - timedelta(hours=1),
                resource_type=ResourceType.DATABASE,
                from_resource="db1",
                to_resource="db2",
                reason="Test",
                strategy=FailoverStrategy.PRIORITY_BASED,
                success=True,
                response_time=0.5
            ),
            FailoverEvent(
                event_id="event2",
                timestamp=datetime.now() - timedelta(hours=25),  # Outside 24h window
                resource_type=ResourceType.API_ENDPOINT,
                from_resource="api1",
                to_resource="api2",
                reason="Test",
                strategy=FailoverStrategy.IMMEDIATE,
                success=False,
                response_time=1.0
            )
        ]
        
        for event in events:
            failover_manager.failover_history.append(event)
        
        # Get history for last 24 hours
        history = failover_manager.get_failover_history(hours=24)
        
        # Should only return event1
        assert len(history) == 1
        assert history[0]['event_id'] == "event1"
    
    def test_export_failover_report(self, failover_manager, tmp_path):
        """Test exporting failover report"""
        # Add a resource
        config = ResourceConfig(
            resource_id="test_db",
            resource_type=ResourceType.DATABASE,
            name="Test Database",
            connection_string="postgresql://test",
            priority=1
        )
        failover_manager.add_resource(config)
        
        report_file = tmp_path / "test_report.json"
        
        failover_manager.export_failover_report(str(report_file))
        
        assert report_file.exists()
        
        # Read and verify report content
        import json
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        assert 'report_timestamp' in report
        assert 'statistics' in report
        assert 'failover_history' in report
        assert 'resource_configurations' in report
        assert 'health_status' in report


class TestGlobalFunctions:
    """Test global functions"""
    
    def test_initialize_failover_manager(self):
        """Test global failover manager initialization"""
        error_handler = ErrorHandler()
        
        manager = initialize_failover_manager(
            error_handler=error_handler,
            default_strategy=FailoverStrategy.LOAD_BALANCED
        )
        
        assert manager is not None
        assert isinstance(manager, FailoverManager)
        assert manager.default_strategy == FailoverStrategy.LOAD_BALANCED
        
        # Test getting the global instance
        from stock_analysis_system.core.failover_mechanism import get_failover_manager
        global_manager = get_failover_manager()
        
        assert global_manager is manager


if __name__ == "__main__":
    pytest.main([__file__, "-v"])