"""
Comprehensive tests for the Cost Management and Optimization System

This module tests all components of task 14: cost monitoring, auto-scaling,
and resource optimization dashboard functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json
import numpy as np

from stock_analysis_system.infrastructure.cost_optimization_manager import (
    CostOptimizationManager, CostAlert, ResourceUsage, ResourceType, CostCategory
)
from stock_analysis_system.infrastructure.intelligent_autoscaling import (
    IntelligentAutoScaling, AutoScalingConfig, SpotInstanceConfig, 
    ScalingAction, InstanceType, ScalingMetrics
)
from stock_analysis_system.infrastructure.resource_optimization_dashboard import (
    ResourceOptimizationDashboard, DashboardView, TimeRange, BudgetPlan, DashboardConfig
)

class TestCostOptimizationManager:
    """Test cases for Cost Optimization Manager"""
    
    @pytest.fixture
    def cost_manager(self):
        """Create cost optimization manager for testing"""
        config = {
            'daily_budget': 1000.0,
            'monthly_budget': 30000.0,
            'spike_threshold': 0.5,
            'utilization_threshold': 0.3,
            'base_cost_per_hour': 0.1
        }
        return CostOptimizationManager(config)
    
    @pytest.mark.asyncio
    async def test_collect_resource_usage(self, cost_manager):
        """Test resource usage collection"""
        with patch('psutil.cpu_percent', return_value=45.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_network:
            
            # Mock system metrics
            mock_memory.return_value = Mock(percent=60.0)
            mock_disk.return_value = Mock(used=50*1024**3, total=100*1024**3)  # 50% usage
            mock_network.return_value = Mock(bytes_recv=1024**3, bytes_sent=512*1024**2)
            
            usage = await cost_manager.collect_resource_usage()
            
            assert isinstance(usage, ResourceUsage)
            assert usage.cpu_usage == 45.0
            assert usage.memory_usage == 60.0
            assert usage.disk_usage == 50.0
            assert usage.resource_type == ResourceType.COMPUTE
            assert usage.cost_per_hour > 0
            assert 0 <= usage.utilization_score <= 1
    
    @pytest.mark.asyncio
    async def test_calculate_cost_metrics(self, cost_manager):
        """Test cost metrics calculation"""
        # Add some mock usage history
        for i in range(10):
            usage = ResourceUsage(
                resource_id=f"test_{i}",
                resource_type=ResourceType.COMPUTE,
                timestamp=datetime.now() - timedelta(hours=i),
                cpu_usage=50.0 + i * 2,
                memory_usage=40.0 + i * 3,
                cost_per_hour=0.1 + i * 0.01,
                utilization_score=0.5 + i * 0.02
            )
            cost_manager.resource_usage_history.append(usage)
        
        metrics = await cost_manager.calculate_cost_metrics()
        
        assert metrics.total_cost > 0
        assert metrics.daily_cost > 0
        assert metrics.monthly_cost > 0
        assert len(metrics.cost_by_category) > 0
        assert len(metrics.cost_trend) > 0
        assert isinstance(metrics.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_cost_alert_creation(self, cost_manager):
        """Test cost alert creation and checking"""
        alert = CostAlert(
            alert_id="test_alert_1",
            name="Daily Budget Alert",
            threshold_type="budget",
            threshold_value=500.0,
            period="daily",
            enabled=True
        )
        
        success = await cost_manager.create_cost_alert(alert)
        assert success
        assert "test_alert_1" in cost_manager.cost_alerts
        
        # Test alert checking
        triggered_alerts = await cost_manager.check_cost_alerts()
        assert isinstance(triggered_alerts, list)
    
    @pytest.mark.asyncio
    async def test_optimization_recommendations(self, cost_manager):
        """Test optimization recommendations generation"""
        # Add mock usage data with low utilization
        for i in range(20):
            usage = ResourceUsage(
                resource_id=f"low_util_{i}",
                resource_type=ResourceType.COMPUTE,
                timestamp=datetime.now() - timedelta(minutes=i*5),
                cpu_usage=15.0,  # Low CPU usage
                memory_usage=25.0,  # Low memory usage
                cost_per_hour=0.1,
                utilization_score=0.2  # Low utilization
            )
            cost_manager.resource_usage_history.append(usage)
        
        recommendations = await cost_manager.get_optimization_recommendations()
        
        assert 'cost_metrics' in recommendations
        assert 'rightsizing_recommendations' in recommendations
        assert 'potential_savings' in recommendations
        assert isinstance(recommendations['priority_actions'], list)
    
    @pytest.mark.asyncio
    async def test_export_cost_report(self, cost_manager):
        """Test cost report export"""
        # Add some mock data
        usage = ResourceUsage(
            resource_id="export_test",
            resource_type=ResourceType.COMPUTE,
            timestamp=datetime.now(),
            cpu_usage=50.0,
            memory_usage=60.0,
            cost_per_hour=0.15,
            utilization_score=0.55
        )
        cost_manager.resource_usage_history.append(usage)
        
        report = await cost_manager.export_cost_report("json")
        
        assert isinstance(report, str)
        report_data = json.loads(report)
        assert 'report_timestamp' in report_data
        assert 'cost_metrics' in report_data
        assert 'optimization_recommendations' in report_data

class TestIntelligentAutoScaling:
    """Test cases for Intelligent Auto-scaling System"""
    
    @pytest.fixture
    def autoscaling_config(self):
        """Create auto-scaling configuration for testing"""
        return AutoScalingConfig(
            min_instances=1,
            max_instances=5,
            target_cpu_utilization=70.0,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            cooldown_period=60  # Short cooldown for testing
        )
    
    @pytest.fixture
    def spot_config(self):
        """Create spot instance configuration for testing"""
        return SpotInstanceConfig(
            enabled=True,
            max_spot_percentage=0.7,
            spot_price_threshold=0.5
        )
    
    @pytest.fixture
    def autoscaling(self, autoscaling_config, spot_config):
        """Create intelligent auto-scaling instance for testing"""
        return IntelligentAutoScaling(autoscaling_config, spot_config)
    
    @pytest.mark.asyncio
    async def test_collect_scaling_metrics(self, autoscaling):
        """Test scaling metrics collection"""
        metrics = await autoscaling.collect_scaling_metrics()
        
        assert isinstance(metrics, ScalingMetrics)
        assert 0 <= metrics.cpu_utilization <= 100
        assert 0 <= metrics.memory_utilization <= 100
        assert metrics.request_rate >= 0
        assert metrics.response_time > 0
        assert metrics.queue_length >= 0
        assert metrics.cost_per_hour > 0
        assert 0 <= metrics.performance_score <= 100
    
    @pytest.mark.asyncio
    async def test_prediction_model_training(self, autoscaling):
        """Test prediction model training"""
        # Add sufficient mock data for training
        for i in range(60):  # Need at least 50 data points
            metrics = ScalingMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                cpu_utilization=50.0 + np.sin(i * 0.1) * 20,
                memory_utilization=60.0 + np.cos(i * 0.1) * 15,
                network_utilization=40.0 + np.sin(i * 0.05) * 10,
                request_rate=100.0 + np.sin(i * 0.2) * 50,
                response_time=200.0 + np.cos(i * 0.15) * 50,
                queue_length=max(0, int(np.sin(i * 0.3) * 5)),
                cost_per_hour=0.1 + np.sin(i * 0.1) * 0.05,
                performance_score=70.0 + np.cos(i * 0.1) * 20
            )
            autoscaling.metrics_history.append(metrics)
        
        success = await autoscaling.train_prediction_model()
        assert success
        assert autoscaling.model_trained
    
    @pytest.mark.asyncio
    async def test_scaling_decision_scale_up(self, autoscaling):
        """Test scaling decision for scale up scenario"""
        # Add high utilization metrics
        high_util_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=85.0,  # Above scale_up_threshold
            memory_utilization=85.0,
            network_utilization=70.0,
            request_rate=200.0,
            response_time=300.0,
            queue_length=8,  # High queue length
            cost_per_hour=0.15,
            performance_score=40.0  # Low performance due to high load
        )
        autoscaling.metrics_history.append(high_util_metrics)
        
        decision = await autoscaling.make_scaling_decision()
        
        assert decision is not None
        assert decision.action == ScalingAction.SCALE_UP
        assert decision.confidence > 0.5
        assert decision.expected_cost_change > 0  # Cost should increase
        assert "High utilization" in decision.rationale
    
    @pytest.mark.asyncio
    async def test_scaling_decision_scale_down(self, autoscaling):
        """Test scaling decision for scale down scenario"""
        # Add low utilization metrics
        low_util_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=20.0,  # Below scale_down_threshold
            memory_utilization=25.0,
            network_utilization=15.0,
            request_rate=30.0,
            response_time=150.0,
            queue_length=0,  # No queue
            cost_per_hour=0.1,
            performance_score=85.0  # Good performance with low load
        )
        autoscaling.metrics_history.append(low_util_metrics)
        
        decision = await autoscaling.make_scaling_decision()
        
        assert decision is not None
        assert decision.action == ScalingAction.SCALE_DOWN
        assert decision.confidence > 0.5
        assert decision.expected_cost_change < 0  # Cost should decrease
        assert "Low utilization" in decision.rationale
    
    @pytest.mark.asyncio
    async def test_rightsizing_recommendations(self, autoscaling):
        """Test resource right-sizing recommendations"""
        # Add metrics with consistent low CPU usage
        for i in range(25):
            metrics = ScalingMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                cpu_utilization=15.0,  # Consistently low
                memory_utilization=80.0,  # High memory usage
                network_utilization=30.0,
                request_rate=50.0,
                response_time=180.0,
                queue_length=0,
                cost_per_hour=0.1,
                performance_score=70.0
            )
            autoscaling.metrics_history.append(metrics)
        
        recommendations = await autoscaling.get_rightsizing_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend CPU downsizing and memory optimization
        cpu_rec = next((r for r in recommendations if r['type'] == 'downsize_cpu'), None)
        memory_rec = next((r for r in recommendations if r['type'] == 'increase_memory'), None)
        
        assert cpu_rec is not None
        assert memory_rec is not None
    
    @pytest.mark.asyncio
    async def test_scaling_analytics(self, autoscaling):
        """Test scaling analytics generation"""
        # Add some mock scaling history
        from stock_analysis_system.infrastructure.intelligent_autoscaling import ScalingDecision
        
        decision1 = ScalingDecision(
            action=ScalingAction.SCALE_UP,
            target_instances=2,
            target_instance_type=InstanceType.MEDIUM,
            confidence=0.8,
            rationale="High CPU utilization",
            expected_cost_change=0.1,
            expected_performance_change=20.0,
            timestamp=datetime.now() - timedelta(hours=1)
        )
        
        decision2 = ScalingDecision(
            action=ScalingAction.SCALE_DOWN,
            target_instances=1,
            target_instance_type=InstanceType.SMALL,
            confidence=0.7,
            rationale="Low utilization",
            expected_cost_change=-0.05,
            expected_performance_change=-5.0,
            timestamp=datetime.now()
        )
        
        autoscaling.scaling_history.extend([decision1, decision2])
        
        # Add some metrics
        for i in range(10):
            metrics = ScalingMetrics(
                timestamp=datetime.now() - timedelta(minutes=i*5),
                cpu_utilization=50.0,
                memory_utilization=60.0,
                network_utilization=40.0,
                request_rate=100.0,
                response_time=200.0,
                queue_length=2,
                cost_per_hour=0.12,
                performance_score=75.0
            )
            autoscaling.metrics_history.append(metrics)
        
        analytics = await autoscaling.get_scaling_analytics()
        
        assert 'performance_metrics' in analytics
        assert 'scaling_effectiveness' in analytics
        assert 'cost_optimization' in analytics
        assert analytics['scaling_effectiveness']['total_scaling_actions'] == 2

class TestResourceOptimizationDashboard:
    """Test cases for Resource Optimization Dashboard"""
    
    @pytest.fixture
    def mock_cost_manager(self):
        """Create mock cost optimization manager"""
        manager = Mock(spec=CostOptimizationManager)
        manager.calculate_cost_metrics = AsyncMock()
        manager.check_cost_alerts = AsyncMock(return_value=[])
        manager.get_optimization_recommendations = AsyncMock(return_value={
            'cost_metrics': {},
            'rightsizing_recommendations': [],
            'potential_savings': 100.0,
            'priority_actions': ['Test recommendation']
        })
        return manager
    
    @pytest.fixture
    def mock_autoscaling(self):
        """Create mock intelligent auto-scaling"""
        autoscaling = Mock(spec=IntelligentAutoScaling)
        autoscaling.get_scaling_analytics = AsyncMock(return_value={
            'performance_metrics': {
                'average_performance_score': 75.0,
                'average_cost_per_hour': 0.12,
                'cost_performance_ratio': 0.0016
            },
            'scaling_effectiveness': {
                'total_scaling_actions': 5,
                'successful_scalings': 4,
                'success_rate': 0.8
            }
        })
        autoscaling.get_rightsizing_recommendations = AsyncMock(return_value=[])
        return autoscaling
    
    @pytest.fixture
    def dashboard(self, mock_cost_manager, mock_autoscaling):
        """Create resource optimization dashboard for testing"""
        config = DashboardConfig(refresh_interval=60)
        return ResourceOptimizationDashboard(mock_cost_manager, mock_autoscaling, config)
    
    @pytest.mark.asyncio
    async def test_generate_overview_dashboard(self, dashboard, mock_cost_manager, mock_autoscaling):
        """Test overview dashboard generation"""
        from stock_analysis_system.infrastructure.cost_optimization_manager import CostMetrics
        
        # Mock cost metrics
        mock_cost_metrics = CostMetrics(
            total_cost=100.0,
            daily_cost=50.0,
            monthly_cost=1500.0,
            cost_by_category={'infrastructure': 60.0, 'analytics': 40.0},
            cost_by_resource={'compute': 80.0, 'storage': 20.0},
            cost_trend=[(datetime.now() - timedelta(hours=i), 10.0 + i) for i in range(24)],
            optimization_potential=20.0,
            recommendations=['Reduce instance size', 'Use spot instances']
        )
        
        mock_cost_manager.calculate_cost_metrics.return_value = mock_cost_metrics
        
        overview = await dashboard.generate_overview_dashboard(TimeRange.LAST_DAY)
        
        assert 'timestamp' in overview
        assert 'time_range' in overview
        assert 'kpis' in overview
        assert 'charts' in overview
        assert 'optimization_summary' in overview
        
        # Check KPIs
        kpis = overview['kpis']
        assert 'total_cost' in kpis
        assert 'daily_cost' in kpis
        assert 'potential_savings' in kpis
        
        # Check optimization summary
        opt_summary = overview['optimization_summary']
        assert opt_summary['potential_savings'] == 20.0
        assert len(opt_summary['top_recommendations']) <= 3
    
    @pytest.mark.asyncio
    async def test_cost_forecast_generation(self, dashboard):
        """Test cost forecast generation"""
        # Mock some cost trend data
        dashboard.cost_manager.calculate_cost_metrics = AsyncMock()
        mock_metrics = Mock()
        mock_metrics.cost_trend = [
            (datetime.now() - timedelta(days=i), 100.0 + i * 2) 
            for i in range(30)
        ]
        dashboard.cost_manager.calculate_cost_metrics.return_value = mock_metrics
        
        forecasts = await dashboard.generate_cost_forecast(30)
        
        assert len(forecasts) == 30
        for forecast in forecasts:
            assert hasattr(forecast, 'forecast_date')
            assert hasattr(forecast, 'predicted_cost')
            assert hasattr(forecast, 'confidence_interval')
            assert hasattr(forecast, 'trend')
            assert forecast.predicted_cost >= 0
    
    @pytest.mark.asyncio
    async def test_budget_plan_creation(self, dashboard):
        """Test budget plan creation and status checking"""
        budget_plan = BudgetPlan(
            name="Test Budget",
            period="monthly",
            budget_amount=5000.0,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            categories={"infrastructure": 3000.0, "analytics": 2000.0},
            alerts_enabled=True,
            alert_thresholds=[0.8, 0.9, 1.0]
        )
        
        success = await dashboard.create_budget_plan(budget_plan)
        assert success
        assert "Test Budget" in dashboard.budget_plans
        
        # Mock cost metrics for budget checking
        from stock_analysis_system.infrastructure.cost_optimization_manager import CostMetrics
        mock_metrics = CostMetrics(
            total_cost=100.0,
            daily_cost=150.0,  # $150/day = $4500/month
            monthly_cost=4500.0,
            cost_by_category={"infrastructure": 2700.0, "analytics": 1800.0},
            cost_by_resource={},
            cost_trend=[],
            optimization_potential=0.0,
            recommendations=[]
        )
        dashboard.cost_manager.calculate_cost_metrics.return_value = mock_metrics
        
        status = await dashboard.check_budget_status("Test Budget")
        
        assert status['plan_name'] == "Test Budget"
        assert status['budget_utilization'] == 90.0  # 4500/5000 = 90%
        assert status['status'] == 'critical'  # >= 90%
        assert status['remaining_budget'] == 500.0
    
    @pytest.mark.asyncio
    async def test_optimization_recommendations(self, dashboard):
        """Test comprehensive optimization recommendations"""
        recommendations = await dashboard.generate_optimization_recommendations()
        
        assert 'timestamp' in recommendations
        assert 'cost_recommendations' in recommendations
        assert 'rightsizing_recommendations' in recommendations
        assert 'prioritized_actions' in recommendations
        assert 'implementation_roadmap' in recommendations
    
    @pytest.mark.asyncio
    async def test_dashboard_data_export(self, dashboard, mock_cost_manager):
        """Test dashboard data export functionality"""
        # Mock cost metrics for export
        from stock_analysis_system.infrastructure.cost_optimization_manager import CostMetrics
        mock_metrics = CostMetrics(
            total_cost=100.0,
            daily_cost=50.0,
            monthly_cost=1500.0,
            cost_by_category={},
            cost_by_resource={},
            cost_trend=[],
            optimization_potential=20.0,
            recommendations=[]
        )
        mock_cost_manager.calculate_cost_metrics.return_value = mock_metrics
        
        exported_data = await dashboard.export_dashboard_data(
            DashboardView.OVERVIEW, 
            TimeRange.LAST_DAY, 
            "json"
        )
        
        assert isinstance(exported_data, str)
        data = json.loads(exported_data)
        assert 'export_timestamp' in data
        assert 'view' in data
        assert 'time_range' in data
        assert 'data' in data

class TestCostManagementIntegration:
    """Integration tests for the complete cost management system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_cost_optimization_workflow(self):
        """Test complete cost optimization workflow"""
        # Initialize components
        cost_manager = CostOptimizationManager({
            'daily_budget': 1000.0,
            'base_cost_per_hour': 0.1
        })
        
        autoscaling_config = AutoScalingConfig(min_instances=1, max_instances=3)
        autoscaling = IntelligentAutoScaling(autoscaling_config)
        
        dashboard = ResourceOptimizationDashboard(cost_manager, autoscaling)
        
        # Step 1: Collect resource usage
        with patch('psutil.cpu_percent', return_value=75.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_network:
            
            mock_memory.return_value = Mock(percent=80.0)
            mock_disk.return_value = Mock(used=60*1024**3, total=100*1024**3)
            mock_network.return_value = Mock(bytes_recv=2*1024**3, bytes_sent=1024**3)
            
            usage = await cost_manager.collect_resource_usage()
            assert usage.cpu_usage == 75.0
        
        # Step 2: Get scaling metrics
        scaling_metrics = await autoscaling.collect_scaling_metrics()
        assert scaling_metrics.cpu_utilization > 0
        
        # Step 3: Make scaling decision
        decision = await autoscaling.make_scaling_decision()
        assert decision is not None
        
        # Step 4: Calculate cost metrics
        cost_metrics = await cost_manager.calculate_cost_metrics()
        assert cost_metrics.total_cost > 0
        
        # Step 5: Generate dashboard overview
        overview = await dashboard.generate_overview_dashboard()
        assert 'kpis' in overview
        assert 'optimization_summary' in overview
        
        # Step 6: Get optimization recommendations
        recommendations = await dashboard.generate_optimization_recommendations()
        assert 'prioritized_actions' in recommendations
    
    @pytest.mark.asyncio
    async def test_alert_and_budget_integration(self):
        """Test integration between alerts and budget management"""
        cost_manager = CostOptimizationManager({'daily_budget': 100.0})
        autoscaling = IntelligentAutoScaling(AutoScalingConfig())
        dashboard = ResourceOptimizationDashboard(cost_manager, autoscaling)
        
        # Create budget plan
        budget_plan = BudgetPlan(
            name="Integration Test Budget",
            period="daily",
            budget_amount=100.0,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=1),
            alert_thresholds=[0.8, 0.9, 1.0]
        )
        
        await dashboard.create_budget_plan(budget_plan)
        
        # Create cost alert
        alert = CostAlert(
            alert_id="integration_alert",
            name="Integration Test Alert",
            threshold_type="budget",
            threshold_value=80.0,
            period="daily"
        )
        
        await cost_manager.create_cost_alert(alert)
        
        # Simulate high cost usage
        for i in range(10):
            usage = ResourceUsage(
                resource_id=f"integration_{i}",
                resource_type=ResourceType.COMPUTE,
                timestamp=datetime.now() - timedelta(minutes=i),
                cost_per_hour=15.0,  # High cost to trigger alerts
                utilization_score=0.8
            )
            cost_manager.resource_usage_history.append(usage)
        
        # Check alerts
        triggered_alerts = await cost_manager.check_cost_alerts()
        assert len(triggered_alerts) > 0
        
        # Check budget status
        budget_status = await dashboard.check_budget_status("Integration Test Budget")
        assert budget_status['budget_utilization'] > 80.0

if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_cost_optimization or test_autoscaling or test_dashboard"
    ])