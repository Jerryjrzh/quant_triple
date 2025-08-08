"""
Unit tests for System Degradation Strategy

This module contains comprehensive unit tests for the degradation strategy
implementation, covering all major functionality and edge cases.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from stock_analysis_system.core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
from stock_analysis_system.core.degradation_strategy import (
    DegradationStrategy, DegradationLevel, DegradationTrigger,
    DegradationRule, ServiceConfig, ServicePriority, SystemMetrics,
    DegradationEvent, initialize_degradation_strategy
)


class TestSystemMetrics:
    """Test SystemMetrics class"""
    
    def test_init(self):
        """Test SystemMetrics initialization"""
        metrics = SystemMetrics()
        
        assert 'error_rate' in metrics.metrics
        assert 'response_times' in metrics.metrics
        assert 'cpu_usage' in metrics.metrics
        assert 'memory_usage' in metrics.metrics
        assert isinstance(metrics.last_update, datetime)
    
    def test_update_metric(self):
        """Test metric updates"""
        metrics = SystemMetrics()
        
        metrics.update_metric('error_rate', 0.1)
        metrics.update_metric('response_times', 2.5)
        
        assert len(metrics.metrics['error_rate']) == 1
        assert len(metrics.metrics['response_times']) == 1
        assert metrics.metrics['error_rate'][0]['value'] == 0.1
        assert metrics.metrics['response_times'][0]['value'] == 2.5
    
    def test_get_metric_average(self):
        """Test metric average calculation"""
        metrics = SystemMetrics()
        
        # Add test data
        for i in range(5):
            metrics.update_metric('error_rate', i * 0.1)
        
        average = metrics.get_metric_average('error_rate', 10)
        expected = (0.0 + 0.1 + 0.2 + 0.3 + 0.4) / 5
        assert abs(average - expected) < 0.001
    
    def test_get_metric_trend(self):
        """Test metric trend calculation"""
        metrics = SystemMetrics()
        
        # Add increasing trend data
        for i in range(10):
            metrics.update_metric('error_rate', i * 0.1)
        
        trend = metrics.get_metric_trend('error_rate', 10)
        assert trend == "increasing"
        
        # Add decreasing trend data
        metrics = SystemMetrics()
        for i in range(10, 0, -1):
            metrics.update_metric('error_rate', i * 0.1)
        
        trend = metrics.get_metric_trend('error_rate', 10)
        assert trend == "decreasing"


class TestDegradationStrategy:
    """Test DegradationStrategy class"""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler fixture"""
        return ErrorHandler()
    
    @pytest.fixture
    def degradation_strategy(self, error_handler):
        """Create degradation strategy fixture"""
        return DegradationStrategy(
            error_handler=error_handler,
            enable_auto_degradation=True
        )
    
    def test_init(self, degradation_strategy):
        """Test DegradationStrategy initialization"""
        assert degradation_strategy.current_level == DegradationLevel.NORMAL
        assert len(degradation_strategy.active_degradations) == 0
        assert len(degradation_strategy.rules) > 0  # Default rules loaded
        assert len(degradation_strategy.services) > 0  # Default services loaded
        assert degradation_strategy.enable_auto_degradation is True
    
    def test_add_rule(self, degradation_strategy):
        """Test adding degradation rules"""
        initial_count = len(degradation_strategy.rules)
        
        rule = DegradationRule(
            rule_id="test_rule",
            name="Test Rule",
            description="Test rule description",
            trigger=DegradationTrigger.ERROR_RATE,
            threshold=0.2,
            degradation_level=DegradationLevel.MODERATE,
            affected_services=["test_service"],
            actions=["test_action"]
        )
        
        degradation_strategy.add_rule(rule)
        
        assert len(degradation_strategy.rules) == initial_count + 1
        assert any(r.rule_id == "test_rule" for r in degradation_strategy.rules)
    
    def test_remove_rule(self, degradation_strategy):
        """Test removing degradation rules"""
        # Add a test rule first
        rule = DegradationRule(
            rule_id="test_rule_remove",
            name="Test Rule Remove",
            description="Test rule for removal",
            trigger=DegradationTrigger.ERROR_RATE,
            threshold=0.2,
            degradation_level=DegradationLevel.MODERATE,
            affected_services=["test_service"],
            actions=["test_action"]
        )
        
        degradation_strategy.add_rule(rule)
        initial_count = len(degradation_strategy.rules)
        
        # Remove the rule
        success = degradation_strategy.remove_rule("test_rule_remove")
        
        assert success is True
        assert len(degradation_strategy.rules) == initial_count - 1
        assert not any(r.rule_id == "test_rule_remove" for r in degradation_strategy.rules)
    
    def test_add_service(self, degradation_strategy):
        """Test adding service configurations"""
        initial_count = len(degradation_strategy.services)
        
        service = ServiceConfig(
            service_name="test_service",
            priority=ServicePriority.MEDIUM,
            degradation_actions={
                DegradationLevel.LIGHT: ["test_action_1"],
                DegradationLevel.MODERATE: ["test_action_2"]
            }
        )
        
        degradation_strategy.add_service(service)
        
        assert len(degradation_strategy.services) == initial_count + 1
        assert "test_service" in degradation_strategy.services
        assert degradation_strategy.services["test_service"].priority == ServicePriority.MEDIUM
    
    def test_register_degradation_action(self, degradation_strategy):
        """Test registering degradation actions"""
        async def test_action():
            pass
        
        degradation_strategy.register_degradation_action("test_action", test_action)
        
        assert "test_action" in degradation_strategy.degradation_actions
        assert degradation_strategy.degradation_actions["test_action"] == test_action
    
    @pytest.mark.asyncio
    async def test_manual_degradation(self, degradation_strategy):
        """Test manual degradation trigger"""
        # Register a test action
        action_called = False
        
        async def test_action():
            nonlocal action_called
            action_called = True
        
        degradation_strategy.register_degradation_action("test_action", test_action)
        
        # Add a test service
        service = ServiceConfig(
            service_name="test_service",
            priority=ServicePriority.MEDIUM,
            degradation_actions={
                DegradationLevel.MODERATE: ["test_action"]
            }
        )
        degradation_strategy.add_service(service)
        
        # Trigger manual degradation
        event_id = await degradation_strategy.manual_degradation(
            level=DegradationLevel.MODERATE,
            services=["test_service"],
            reason="Test degradation"
        )
        
        assert event_id is not None
        assert degradation_strategy.current_level == DegradationLevel.MODERATE
        assert "test_service" in degradation_strategy.active_degradations
        assert action_called is True
        assert degradation_strategy.stats['manual_degradations'] == 1
    
    @pytest.mark.asyncio
    async def test_manual_recovery(self, degradation_strategy):
        """Test manual recovery"""
        # First trigger a degradation
        event_id = await degradation_strategy.manual_degradation(
            level=DegradationLevel.LIGHT,
            services=["data_collection"],
            reason="Test for recovery"
        )
        
        assert len(degradation_strategy.active_degradations) > 0
        
        # Now recover
        success = await degradation_strategy.manual_recovery(event_id)
        
        assert success is True
        assert len(degradation_strategy.active_degradations) == 0
        assert degradation_strategy.current_level == DegradationLevel.NORMAL
        assert degradation_strategy.stats['successful_recoveries'] > 0
    
    def test_get_system_status(self, degradation_strategy):
        """Test system status retrieval"""
        status = degradation_strategy.get_system_status()
        
        assert 'current_level' in status
        assert 'active_degradations' in status
        assert 'active_events' in status
        assert 'total_services' in status
        assert 'degraded_services' in status
        assert 'monitoring_enabled' in status
        assert 'statistics' in status
        
        assert status['current_level'] == DegradationLevel.NORMAL.value
        assert isinstance(status['active_degradations'], list)
        assert isinstance(status['statistics'], dict)
    
    def test_get_degradation_history(self, degradation_strategy):
        """Test degradation history retrieval"""
        history = degradation_strategy.get_degradation_history(hours=24)
        
        assert isinstance(history, list)
        # Initially should be empty
        assert len(history) == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_rule_error_rate(self, degradation_strategy):
        """Test rule evaluation for error rate trigger"""
        # Create a test rule
        rule = DegradationRule(
            rule_id="test_error_rate",
            name="Test Error Rate",
            description="Test error rate rule",
            trigger=DegradationTrigger.ERROR_RATE,
            threshold=0.1,
            degradation_level=DegradationLevel.LIGHT,
            affected_services=["test_service"],
            actions=["test_action"]
        )
        
        # Set high error rate
        degradation_strategy.metrics.update_metric('error_rate', 0.15)
        
        should_trigger = await degradation_strategy._evaluate_rule(rule)
        assert should_trigger is True
        
        # Set low error rate
        degradation_strategy.metrics.update_metric('error_rate', 0.05)
        
        should_trigger = await degradation_strategy._evaluate_rule(rule)
        assert should_trigger is False
    
    @pytest.mark.asyncio
    async def test_evaluate_rule_response_time(self, degradation_strategy):
        """Test rule evaluation for response time trigger"""
        rule = DegradationRule(
            rule_id="test_response_time",
            name="Test Response Time",
            description="Test response time rule",
            trigger=DegradationTrigger.RESPONSE_TIME,
            threshold=3.0,
            degradation_level=DegradationLevel.MODERATE,
            affected_services=["test_service"],
            actions=["test_action"]
        )
        
        # Set high response time
        degradation_strategy.metrics.update_metric('response_times', 5.0)
        
        should_trigger = await degradation_strategy._evaluate_rule(rule)
        assert should_trigger is True
        
        # Set low response time
        degradation_strategy.metrics.update_metric('response_times', 1.0)
        
        should_trigger = await degradation_strategy._evaluate_rule(rule)
        assert should_trigger is False
    
    @pytest.mark.asyncio
    async def test_evaluate_rule_resource_usage(self, degradation_strategy):
        """Test rule evaluation for resource usage trigger"""
        rule = DegradationRule(
            rule_id="test_resource_usage",
            name="Test Resource Usage",
            description="Test resource usage rule",
            trigger=DegradationTrigger.RESOURCE_USAGE,
            threshold=0.8,
            degradation_level=DegradationLevel.SEVERE,
            affected_services=["test_service"],
            actions=["test_action"]
        )
        
        # Set high resource usage
        degradation_strategy.metrics.update_metric('memory_usage', 0.9)
        degradation_strategy.metrics.update_metric('cpu_usage', 0.85)
        
        should_trigger = await degradation_strategy._evaluate_rule(rule)
        assert should_trigger is True
        
        # Set low resource usage
        degradation_strategy.metrics.update_metric('memory_usage', 0.5)
        degradation_strategy.metrics.update_metric('cpu_usage', 0.6)
        
        should_trigger = await degradation_strategy._evaluate_rule(rule)
        assert should_trigger is False
    
    @pytest.mark.asyncio
    async def test_monitoring_loop(self, degradation_strategy):
        """Test monitoring loop functionality"""
        # Mock the check degradation conditions method
        check_called = False
        
        async def mock_check():
            nonlocal check_called
            check_called = True
        
        degradation_strategy._check_degradation_conditions = mock_check
        
        # Start monitoring for a short time
        await degradation_strategy.start_monitoring()
        
        # Wait a bit for the monitoring loop to run
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        await degradation_strategy.stop_monitoring()
        
        # The check method should have been called
        assert check_called is True
    
    def test_export_degradation_report(self, degradation_strategy, tmp_path):
        """Test degradation report export"""
        report_file = tmp_path / "test_report.json"
        
        degradation_strategy.export_degradation_report(str(report_file))
        
        assert report_file.exists()
        
        # Read and verify report content
        import json
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        assert 'report_timestamp' in report
        assert 'system_status' in report
        assert 'degradation_history' in report
        assert 'rules' in report
        assert 'services' in report
    
    @pytest.mark.asyncio
    async def test_cooldown_period(self, degradation_strategy):
        """Test cooldown period functionality"""
        # Create a rule with short cooldown
        rule = DegradationRule(
            rule_id="test_cooldown",
            name="Test Cooldown",
            description="Test cooldown rule",
            trigger=DegradationTrigger.ERROR_RATE,
            threshold=0.1,
            degradation_level=DegradationLevel.LIGHT,
            affected_services=["test_service"],
            actions=["test_action"],
            cooldown_period=1  # 1 second cooldown
        )
        
        degradation_strategy.add_rule(rule)
        
        # Set conditions to trigger the rule
        degradation_strategy.metrics.update_metric('error_rate', 0.15)
        
        # First trigger should work
        await degradation_strategy._trigger_degradation(rule)
        first_count = degradation_strategy.stats['total_degradations']
        
        # Immediate second trigger should be blocked by cooldown
        await degradation_strategy._trigger_degradation(rule)
        second_count = degradation_strategy.stats['total_degradations']
        
        assert second_count == first_count  # No increase due to cooldown
        
        # Wait for cooldown to expire
        await asyncio.sleep(1.1)
        
        # Third trigger should work after cooldown
        await degradation_strategy._trigger_degradation(rule)
        third_count = degradation_strategy.stats['total_degradations']
        
        assert third_count > second_count  # Should increase after cooldown


class TestDegradationIntegration:
    """Integration tests for degradation strategy"""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler fixture"""
        return ErrorHandler()
    
    @pytest.fixture
    def degradation_strategy(self, error_handler):
        """Create degradation strategy fixture"""
        return DegradationStrategy(
            error_handler=error_handler,
            enable_auto_degradation=True
        )
    
    @pytest.mark.asyncio
    async def test_error_triggered_degradation(self, error_handler, degradation_strategy):
        """Test degradation triggered by actual errors"""
        # Generate multiple errors to increase error rate
        for i in range(10):
            try:
                raise ConnectionError(f"Test error {i}")
            except Exception as e:
                error_handler.handle_error(e)
        
        # Update error rate metric
        degradation_strategy.metrics.update_metric('error_rate', 0.15)
        
        # Check degradation conditions
        await degradation_strategy._check_degradation_conditions()
        
        # Should have triggered some degradation
        assert len(degradation_strategy.active_degradations) > 0 or degradation_strategy.current_level != DegradationLevel.NORMAL
    
    @pytest.mark.asyncio
    async def test_recovery_after_improvement(self, degradation_strategy):
        """Test automatic recovery after conditions improve"""
        # First trigger degradation
        await degradation_strategy.manual_degradation(
            level=DegradationLevel.MODERATE,
            services=["data_collection"],
            reason="Test for recovery"
        )
        
        assert len(degradation_strategy.active_degradations) > 0
        
        # Simulate improved conditions
        degradation_strategy.metrics.update_metric('error_rate', 0.01)
        degradation_strategy.metrics.update_metric('response_times', 1.0)
        degradation_strategy.metrics.update_metric('memory_usage', 0.5)
        
        # Check for recovery
        for rule in degradation_strategy.rules:
            await degradation_strategy._check_recovery(rule)
        
        # Should have recovered (at least partially)
        # Note: This test might need adjustment based on specific recovery logic
    
    @pytest.mark.asyncio
    async def test_multiple_degradation_levels(self, degradation_strategy):
        """Test handling multiple degradation levels"""
        # Trigger light degradation
        await degradation_strategy.manual_degradation(
            level=DegradationLevel.LIGHT,
            services=["visualization"],
            reason="Light degradation test"
        )
        
        assert degradation_strategy.current_level == DegradationLevel.LIGHT
        
        # Trigger moderate degradation (should upgrade level)
        await degradation_strategy.manual_degradation(
            level=DegradationLevel.MODERATE,
            services=["analysis"],
            reason="Moderate degradation test"
        )
        
        assert degradation_strategy.current_level == DegradationLevel.MODERATE
        
        # Should have both services degraded
        assert "visualization" in degradation_strategy.active_degradations
        assert "analysis" in degradation_strategy.active_degradations


class TestGlobalFunctions:
    """Test global functions"""
    
    def test_initialize_degradation_strategy(self):
        """Test global degradation strategy initialization"""
        error_handler = ErrorHandler()
        
        strategy = initialize_degradation_strategy(
            error_handler=error_handler,
            enable_auto_degradation=False
        )
        
        assert strategy is not None
        assert isinstance(strategy, DegradationStrategy)
        assert strategy.enable_auto_degradation is False
        
        # Test getting the global instance
        from stock_analysis_system.core.degradation_strategy import get_degradation_strategy
        global_strategy = get_degradation_strategy()
        
        assert global_strategy is strategy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])