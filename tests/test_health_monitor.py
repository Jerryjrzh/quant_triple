#!/usr/bin/env python3
"""
健康监控器测试

测试健康检查监控器的各项功能，包括组件状态检查、故障检测、自动恢复等。
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from stock_analysis_system.monitoring.health_monitor import (
    HealthMonitor, HealthStatus, ComponentType, HealthCheckResult, SystemMetrics
)


class TestHealthMonitor:
    """健康监控器测试类"""
    
    @pytest.fixture
    def health_monitor(self):
        """创建健康监控器实例"""
        return HealthMonitor()
    
    @pytest.mark.asyncio
    async def test_health_monitor_initialization(self, health_monitor):
        """测试健康监控器初始化"""
        assert health_monitor.check_interval == 30
        assert health_monitor.history_retention_days == 7
        assert not health_monitor.is_monitoring
        assert health_monitor.monitor_task is None
        assert len(health_monitor.health_history) == 0
        assert len(health_monitor.component_status) == 0
    
    @pytest.mark.asyncio
    async def test_database_health_check(self, health_monitor):
        """测试数据库健康检查"""
        # Mock数据库管理器
        health_monitor.db_manager = Mock()
        health_monitor.db_manager.initialize = AsyncMock()
        health_monitor.db_manager.fetch_one = AsyncMock(return_value={'test': 1})
        
        result = await health_monitor._check_database_health()
        
        assert isinstance(result, HealthCheckResult)
        assert result.component_name == "database"
        assert result.component_type == ComponentType.DATABASE
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time > 0
        assert "数据库连接正常" in result.message
    
    @pytest.mark.asyncio
    async def test_database_health_check_failure(self, health_monitor):
        """测试数据库健康检查失败"""
        # Mock数据库管理器抛出异常
        health_monitor.db_manager = Mock()
        health_monitor.db_manager.initialize = AsyncMock(side_effect=Exception("Connection failed"))
        
        result = await health_monitor._check_database_health()
        
        assert result.status == HealthStatus.CRITICAL
        assert "数据库连接失败" in result.message
        assert result.error == "Connection failed"
    
    @pytest.mark.asyncio
    async def test_cache_health_check(self, health_monitor):
        """测试缓存健康检查"""
        # Mock缓存管理器
        health_monitor.cache_manager = Mock()
        health_monitor.cache_manager.set = AsyncMock()
        health_monitor.cache_manager.get = AsyncMock(return_value="test_value_123")
        health_monitor.cache_manager.delete = AsyncMock()
        
        with patch('time.time', side_effect=[0, 0.1]):  # Mock时间
            result = await health_monitor._check_cache_health()
        
        assert result.component_name == "cache"
        assert result.component_type == ComponentType.CACHE
        assert result.status == HealthStatus.HEALTHY
        assert "缓存读写正常" in result.message
    
    @pytest.mark.asyncio
    async def test_cache_health_check_failure(self, health_monitor):
        """测试缓存健康检查失败"""
        # Mock缓存管理器抛出异常
        health_monitor.cache_manager = Mock()
        health_monitor.cache_manager.set = AsyncMock(side_effect=Exception("Redis connection failed"))
        
        result = await health_monitor._check_cache_health()
        
        assert result.status == HealthStatus.CRITICAL
        assert "缓存连接失败" in result.message
        assert result.error == "Redis connection failed"
    
    @pytest.mark.asyncio
    async def test_data_source_health_check(self, health_monitor):
        """测试数据源健康检查"""
        # Mock数据源管理器
        import pandas as pd
        mock_data = pd.DataFrame({'price': [100.0], 'volume': [1000]})
        
        health_monitor.data_source_manager = Mock()
        health_monitor.data_source_manager.get_realtime_data = AsyncMock(return_value=mock_data)
        
        result = await health_monitor._check_data_source_health()
        
        assert result.component_name == "data_source"
        assert result.component_type == ComponentType.DATA_SOURCE
        assert result.status == HealthStatus.HEALTHY
        assert "数据源连接正常" in result.message
        assert result.details['data_rows'] == 1
    
    @pytest.mark.asyncio
    async def test_data_source_health_check_empty_data(self, health_monitor):
        """测试数据源健康检查返回空数据"""
        import pandas as pd
        
        health_monitor.data_source_manager = Mock()
        health_monitor.data_source_manager.get_realtime_data = AsyncMock(return_value=pd.DataFrame())
        
        result = await health_monitor._check_data_source_health()
        
        assert result.status == HealthStatus.WARNING
        assert "返回空数据" in result.message
    
    @pytest.mark.asyncio
    async def test_system_health_check(self, health_monitor):
        """测试系统健康检查"""
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Mock内存和磁盘信息
            mock_memory.return_value = Mock(percent=60.0, available=4*1024*1024*1024)
            mock_disk.return_value = Mock(percent=70.0, free=100*1024*1024*1024)
            
            result = await health_monitor._check_system_health()
            
            assert result.component_name == "system"
            assert result.component_type == ComponentType.SYSTEM
            assert result.status == HealthStatus.HEALTHY
            assert "系统资源正常" in result.message
            assert result.details['cpu_percent'] == 50.0
            assert result.details['memory_percent'] == 60.0
    
    @pytest.mark.asyncio
    async def test_system_health_check_high_usage(self, health_monitor):
        """测试系统健康检查高使用率"""
        with patch('psutil.cpu_percent', return_value=90.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Mock高使用率
            mock_memory.return_value = Mock(percent=90.0, available=1*1024*1024*1024)
            mock_disk.return_value = Mock(percent=95.0, free=10*1024*1024*1024)
            
            result = await health_monitor._check_system_health()
            
            assert result.status == HealthStatus.CRITICAL
            assert "系统资源告警" in result.message
    
    @pytest.mark.asyncio
    async def test_perform_health_check(self, health_monitor):
        """测试执行健康检查"""
        # Mock所有检查方法
        health_monitor._check_database_health = AsyncMock(return_value=HealthCheckResult(
            "database", ComponentType.DATABASE, HealthStatus.HEALTHY, 0.1, "OK"
        ))
        health_monitor._check_cache_health = AsyncMock(return_value=HealthCheckResult(
            "cache", ComponentType.CACHE, HealthStatus.HEALTHY, 0.05, "OK"
        ))
        health_monitor._check_data_source_health = AsyncMock(return_value=HealthCheckResult(
            "data_source", ComponentType.DATA_SOURCE, HealthStatus.HEALTHY, 0.2, "OK"
        ))
        health_monitor._check_api_health = AsyncMock(return_value=HealthCheckResult(
            "api", ComponentType.API, HealthStatus.HEALTHY, 0.15, "OK"
        ))
        health_monitor._check_system_health = AsyncMock(return_value=HealthCheckResult(
            "system", ComponentType.SYSTEM, HealthStatus.HEALTHY, 0.01, "OK"
        ))
        health_monitor._check_network_health = AsyncMock(return_value=HealthCheckResult(
            "network", ComponentType.NETWORK, HealthStatus.HEALTHY, 0.3, "OK"
        ))
        
        results = await health_monitor.perform_health_check()
        
        assert len(results) == 6
        assert "database" in results
        assert "cache" in results
        assert "data_source" in results
        assert "api" in results
        assert "system" in results
        assert "network" in results
        
        # 检查组件状态是否更新
        assert len(health_monitor.component_status) == 6
        assert len(health_monitor.health_history) == 1
    
    @pytest.mark.asyncio
    async def test_collect_system_metrics(self, health_monitor):
        """测试收集系统指标"""
        with patch('psutil.cpu_percent', return_value=45.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_connections', return_value=[1, 2, 3]), \
             patch('psutil.getloadavg', return_value=(1.0, 1.5, 2.0)):
            
            mock_memory.return_value = Mock(percent=65.0, available=3*1024*1024*1024)
            mock_disk.return_value = Mock(percent=75.0)
            
            metrics = await health_monitor.collect_system_metrics()
            
            assert isinstance(metrics, SystemMetrics)
            assert metrics.cpu_percent == 45.0
            assert metrics.memory_percent == 65.0
            assert metrics.disk_usage_percent == 75.0
            assert metrics.network_connections == 3
            assert metrics.load_average == (1.0, 1.5, 2.0)
            
            # 检查是否保存到历史记录
            assert len(health_monitor.system_metrics_history) == 1
    
    @pytest.mark.asyncio
    async def test_failure_count_tracking(self, health_monitor):
        """测试故障计数跟踪"""
        # 创建一个失败的健康检查结果
        failed_result = HealthCheckResult(
            "test_component", ComponentType.DATABASE, HealthStatus.CRITICAL, 1.0, "Failed"
        )
        
        # Mock故障处理方法
        health_monitor._handle_component_failure = AsyncMock()
        
        # 模拟连续失败
        for i in range(health_monitor.max_failure_count):
            await health_monitor._update_failure_count(failed_result)
        
        # 检查故障计数
        assert health_monitor.failure_counts["test_component"] == health_monitor.max_failure_count
        
        # 检查是否触发故障处理
        health_monitor._handle_component_failure.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_failure_count_reset(self, health_monitor):
        """测试故障计数重置"""
        # 先设置一个故障计数
        health_monitor.failure_counts["test_component"] = 2
        
        # 创建一个成功的健康检查结果
        success_result = HealthCheckResult(
            "test_component", ComponentType.DATABASE, HealthStatus.HEALTHY, 0.1, "OK"
        )
        
        await health_monitor._update_failure_count(success_result)
        
        # 检查故障计数是否重置
        assert health_monitor.failure_counts["test_component"] == 0
    
    def test_get_current_status(self, health_monitor):
        """测试获取当前状态"""
        # 添加一些组件状态
        health_monitor.component_status = {
            "database": HealthCheckResult("database", ComponentType.DATABASE, HealthStatus.HEALTHY, 0.1, "OK"),
            "cache": HealthCheckResult("cache", ComponentType.CACHE, HealthStatus.WARNING, 0.2, "Slow"),
            "api": HealthCheckResult("api", ComponentType.API, HealthStatus.CRITICAL, 1.0, "Failed")
        }
        
        status = health_monitor.get_current_status()
        
        assert status['overall_status'] == HealthStatus.CRITICAL.value
        assert len(status['critical_components']) == 1
        assert len(status['warning_components']) == 1
        assert 'api' in status['critical_components']
        assert 'cache' in status['warning_components']
        assert status['components']['database'] == HealthStatus.HEALTHY.value
    
    def test_get_health_report(self, health_monitor):
        """测试生成健康状态报告"""
        # 添加一些测试数据
        health_monitor.component_status = {
            "database": HealthCheckResult("database", ComponentType.DATABASE, HealthStatus.HEALTHY, 0.1, "OK")
        }
        health_monitor.system_metrics_history = [
            SystemMetrics(50.0, 60.0, 2048.0, 70.0, 100, (1.0, 1.5, 2.0))
        ]
        health_monitor.failure_counts = {"cache": 1}
        
        report = health_monitor.get_health_report()
        
        assert 'report_time' in report
        assert 'current_status' in report
        assert 'system_metrics' in report
        assert 'component_details' in report
        assert 'failure_counts' in report
        assert report['failure_counts']['cache'] == 1
        assert report['system_metrics']['cpu_percent'] == 50.0
    
    def test_calculate_availability_stats(self, health_monitor):
        """测试计算可用性统计"""
        # 添加历史数据
        health_monitor.component_status = {"database": Mock(), "cache": Mock()}
        health_monitor.health_history = [
            {
                "database": HealthCheckResult("database", ComponentType.DATABASE, HealthStatus.HEALTHY, 0.1, "OK"),
                "cache": HealthCheckResult("cache", ComponentType.CACHE, HealthStatus.HEALTHY, 0.1, "OK")
            },
            {
                "database": HealthCheckResult("database", ComponentType.DATABASE, HealthStatus.HEALTHY, 0.1, "OK"),
                "cache": HealthCheckResult("cache", ComponentType.CACHE, HealthStatus.CRITICAL, 1.0, "Failed")
            }
        ]
        
        stats = health_monitor._calculate_availability_stats()
        
        assert stats["database"] == 1.0  # 100% 可用
        assert stats["cache"] == 0.5     # 50% 可用
    
    def test_get_trend_analysis(self, health_monitor):
        """测试获取趋势分析"""
        # 添加历史数据
        now = datetime.now()
        health_monitor.component_status = {"database": Mock()}
        health_monitor.health_history = [
            {
                "database": HealthCheckResult("database", ComponentType.DATABASE, HealthStatus.HEALTHY, 0.1, "OK", timestamp=now)
            }
        ]
        health_monitor.system_metrics_history = [
            SystemMetrics(50.0, 60.0, 2048.0, 70.0, 100, (1.0, 1.5, 2.0), timestamp=now)
        ]
        
        analysis = health_monitor.get_trend_analysis(hours=24)
        
        assert analysis['time_range_hours'] == 24
        assert analysis['total_checks'] == 1
        assert 'component_trends' in analysis
        assert 'system_trends' in analysis
        assert 'database' in analysis['component_trends']
        assert analysis['component_trends']['database']['healthy_ratio'] == 1.0
        assert analysis['system_trends']['avg_cpu_percent'] == 50.0
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, health_monitor):
        """测试监控生命周期"""
        # Mock健康检查方法以避免实际执行
        health_monitor.perform_health_check = AsyncMock()
        health_monitor.collect_system_metrics = AsyncMock()
        health_monitor._cleanup_history = AsyncMock()
        
        # 启动监控
        await health_monitor.start_monitoring()
        assert health_monitor.is_monitoring
        assert health_monitor.monitor_task is not None
        
        # 等待一小段时间让监控循环运行
        await asyncio.sleep(0.1)
        
        # 停止监控
        await health_monitor.stop_monitoring()
        assert not health_monitor.is_monitoring
        assert health_monitor.monitor_task.cancelled()


@pytest.mark.asyncio
async def test_health_monitor_integration():
    """健康监控器集成测试"""
    monitor = HealthMonitor()
    
    # 执行一次完整的健康检查
    results = await monitor.perform_health_check()
    
    # 验证结果
    assert isinstance(results, dict)
    assert len(results) > 0
    
    # 获取当前状态
    status = monitor.get_current_status()
    assert 'overall_status' in status
    assert 'components' in status
    
    # 生成健康报告
    report = monitor.get_health_report()
    assert 'report_time' in report
    assert 'current_status' in report


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])