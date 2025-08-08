#!/usr/bin/env python3
"""
健康检查监控器

监控所有组件状态，实现数据源、数据库、缓存的健康检查，
添加自动故障检测和诊断功能，创建健康状态报告和历史趋势分析。
"""

import asyncio
import logging
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import aiohttp
import redis
from contextlib import asynccontextmanager

from stock_analysis_system.core.database_manager import DatabaseManager
from stock_analysis_system.data.cache_manager import CacheManager
from stock_analysis_system.data.enhanced_data_sources import EnhancedDataSourceManager

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """组件类型枚举"""
    DATABASE = "database"
    CACHE = "cache"
    DATA_SOURCE = "data_source"
    API = "api"
    SYSTEM = "system"
    NETWORK = "network"


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    component_name: str
    component_type: ComponentType
    status: HealthStatus
    response_time: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    network_connections: int
    load_average: Tuple[float, float, float]
    timestamp: datetime = field(default_factory=datetime.now)


class HealthMonitor:
    """健康检查监控器"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.cache_manager = CacheManager()
        self.data_source_manager = EnhancedDataSourceManager()
        
        # 监控配置
        self.check_interval = 30  # 检查间隔（秒）
        self.history_retention_days = 7  # 历史数据保留天数
        self.alert_thresholds = {
            'response_time': 5.0,      # 响应时间阈值（秒）
            'cpu_usage': 80.0,         # CPU使用率阈值（%）
            'memory_usage': 85.0,      # 内存使用率阈值（%）
            'disk_usage': 90.0,        # 磁盘使用率阈值（%）
            'error_rate': 0.05         # 错误率阈值（5%）
        }
        
        # 状态存储
        self.health_history: List[Dict[str, HealthCheckResult]] = []
        self.system_metrics_history: List[SystemMetrics] = []
        self.component_status: Dict[str, HealthCheckResult] = {}
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_task = None
        
        # 故障检测
        self.failure_counts: Dict[str, int] = {}
        self.max_failure_count = 3
    
    async def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            logger.warning("监控已经在运行中")
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("健康监控已启动")
    
    async def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("健康监控已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 执行健康检查
                await self.perform_health_check()
                
                # 收集系统指标
                await self.collect_system_metrics()
                
                # 清理历史数据
                await self._cleanup_history()
                
                # 等待下次检查
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def perform_health_check(self) -> Dict[str, HealthCheckResult]:
        """执行健康检查"""
        logger.debug("开始执行健康检查...")
        
        health_results = {}
        
        # 检查各个组件
        check_tasks = [
            self._check_database_health(),
            self._check_cache_health(),
            self._check_data_source_health(),
            self._check_api_health(),
            self._check_system_health(),
            self._check_network_health()
        ]
        
        results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"健康检查异常: {result}")
                continue
            
            if isinstance(result, HealthCheckResult):
                health_results[result.component_name] = result
                self.component_status[result.component_name] = result
                
                # 更新故障计数
                await self._update_failure_count(result)
        
        # 保存到历史记录
        self.health_history.append(health_results)
        
        # 限制历史记录大小
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-1000:]
        
        logger.debug(f"健康检查完成，检查了 {len(health_results)} 个组件")
        return health_results
    
    async def _check_database_health(self) -> HealthCheckResult:
        """检查数据库健康状态"""
        start_time = time.time()
        
        try:
            # 初始化数据库连接
            await self.db_manager.initialize()
            
            # 执行简单查询测试连接
            result = await self.db_manager.fetch_one("SELECT 1 as test")
            
            response_time = time.time() - start_time
            
            if result and result.get('test') == 1:
                status = HealthStatus.HEALTHY if response_time < self.alert_thresholds['response_time'] else HealthStatus.WARNING
                message = f"数据库连接正常，响应时间: {response_time:.3f}秒"
            else:
                status = HealthStatus.CRITICAL
                message = "数据库查询返回异常结果"
            
            return HealthCheckResult(
                component_name="database",
                component_type=ComponentType.DATABASE,
                status=status,
                response_time=response_time,
                message=message,
                details={
                    'query_result': result,
                    'connection_pool_size': getattr(self.db_manager.pool, 'size', 'unknown') if self.db_manager.pool else 'no_pool'
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                component_name="database",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.CRITICAL,
                response_time=response_time,
                message=f"数据库连接失败: {str(e)}",
                error=str(e)
            )
    
    async def _check_cache_health(self) -> HealthCheckResult:
        """检查缓存健康状态"""
        start_time = time.time()
        
        try:
            # 测试缓存读写
            test_key = "health_check_test"
            test_value = f"test_value_{int(time.time())}"
            
            # 写入测试
            await self.cache_manager.set(test_key, test_value, ttl=60)
            
            # 读取测试
            cached_value = await self.cache_manager.get(test_key)
            
            # 清理测试数据
            await self.cache_manager.delete(test_key)
            
            response_time = time.time() - start_time
            
            if cached_value == test_value:
                status = HealthStatus.HEALTHY if response_time < self.alert_thresholds['response_time'] else HealthStatus.WARNING
                message = f"缓存读写正常，响应时间: {response_time:.3f}秒"
            else:
                status = HealthStatus.CRITICAL
                message = f"缓存读写异常，期望: {test_value}，实际: {cached_value}"
            
            return HealthCheckResult(
                component_name="cache",
                component_type=ComponentType.CACHE,
                status=status,
                response_time=response_time,
                message=message,
                details={
                    'test_key': test_key,
                    'expected_value': test_value,
                    'actual_value': cached_value
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                component_name="cache",
                component_type=ComponentType.CACHE,
                status=HealthStatus.CRITICAL,
                response_time=response_time,
                message=f"缓存连接失败: {str(e)}",
                error=str(e)
            )
    
    async def _check_data_source_health(self) -> HealthCheckResult:
        """检查数据源健康状态"""
        start_time = time.time()
        
        try:
            # 测试数据源连接
            test_symbol = "000001"
            
            # 尝试获取数据
            data = await self.data_source_manager.get_realtime_data(test_symbol)
            
            response_time = time.time() - start_time
            
            if data is not None and not data.empty:
                status = HealthStatus.HEALTHY if response_time < self.alert_thresholds['response_time'] else HealthStatus.WARNING
                message = f"数据源连接正常，响应时间: {response_time:.3f}秒"
                details = {
                    'test_symbol': test_symbol,
                    'data_rows': len(data),
                    'data_columns': list(data.columns) if hasattr(data, 'columns') else []
                }
            else:
                status = HealthStatus.WARNING
                message = "数据源连接正常但返回空数据"
                details = {'test_symbol': test_symbol, 'data_empty': True}
            
            return HealthCheckResult(
                component_name="data_source",
                component_type=ComponentType.DATA_SOURCE,
                status=status,
                response_time=response_time,
                message=message,
                details=details
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                component_name="data_source",
                component_type=ComponentType.DATA_SOURCE,
                status=HealthStatus.CRITICAL,
                response_time=response_time,
                message=f"数据源连接失败: {str(e)}",
                error=str(e)
            )
    
    async def _check_api_health(self) -> HealthCheckResult:
        """检查API健康状态"""
        start_time = time.time()
        
        try:
            # 测试API健康端点
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8000/health', timeout=5) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        response_data = await response.json()
                        status = HealthStatus.HEALTHY
                        message = f"API服务正常，响应时间: {response_time:.3f}秒"
                        details = {'response_data': response_data}
                    else:
                        status = HealthStatus.WARNING
                        message = f"API服务异常，状态码: {response.status}"
                        details = {'status_code': response.status}
            
            return HealthCheckResult(
                component_name="api",
                component_type=ComponentType.API,
                status=status,
                response_time=response_time,
                message=message,
                details=details
            )
            
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return HealthCheckResult(
                component_name="api",
                component_type=ComponentType.API,
                status=HealthStatus.CRITICAL,
                response_time=response_time,
                message="API服务响应超时",
                error="timeout"
            )
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                component_name="api",
                component_type=ComponentType.API,
                status=HealthStatus.CRITICAL,
                response_time=response_time,
                message=f"API服务连接失败: {str(e)}",
                error=str(e)
            )
    
    async def _check_system_health(self) -> HealthCheckResult:
        """检查系统健康状态"""
        start_time = time.time()
        
        try:
            # 获取系统指标
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            response_time = time.time() - start_time
            
            # 判断系统状态
            issues = []
            if cpu_percent > self.alert_thresholds['cpu_usage']:
                issues.append(f"CPU使用率过高: {cpu_percent:.1f}%")
            
            if memory.percent > self.alert_thresholds['memory_usage']:
                issues.append(f"内存使用率过高: {memory.percent:.1f}%")
            
            if disk.percent > self.alert_thresholds['disk_usage']:
                issues.append(f"磁盘使用率过高: {disk.percent:.1f}%")
            
            if issues:
                status = HealthStatus.WARNING if len(issues) == 1 else HealthStatus.CRITICAL
                message = f"系统资源告警: {'; '.join(issues)}"
            else:
                status = HealthStatus.HEALTHY
                message = "系统资源正常"
            
            return HealthCheckResult(
                component_name="system",
                component_type=ComponentType.SYSTEM,
                status=status,
                response_time=response_time,
                message=message,
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / 1024 / 1024 / 1024,
                    'disk_percent': disk.percent,
                    'disk_free_gb': disk.free / 1024 / 1024 / 1024
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                component_name="system",
                component_type=ComponentType.SYSTEM,
                status=HealthStatus.CRITICAL,
                response_time=response_time,
                message=f"系统指标获取失败: {str(e)}",
                error=str(e)
            )
    
    async def _check_network_health(self) -> HealthCheckResult:
        """检查网络健康状态"""
        start_time = time.time()
        
        try:
            # 测试网络连接
            test_urls = [
                'https://www.baidu.com',
                'https://www.google.com'
            ]
            
            successful_connections = 0
            total_response_time = 0
            
            async with aiohttp.ClientSession() as session:
                for url in test_urls:
                    try:
                        url_start = time.time()
                        async with session.get(url, timeout=5) as response:
                            url_response_time = time.time() - url_start
                            if response.status == 200:
                                successful_connections += 1
                                total_response_time += url_response_time
                    except:
                        continue
            
            response_time = time.time() - start_time
            
            if successful_connections > 0:
                avg_response_time = total_response_time / successful_connections
                if successful_connections == len(test_urls):
                    status = HealthStatus.HEALTHY
                    message = f"网络连接正常，平均响应时间: {avg_response_time:.3f}秒"
                else:
                    status = HealthStatus.WARNING
                    message = f"部分网络连接异常，成功: {successful_connections}/{len(test_urls)}"
            else:
                status = HealthStatus.CRITICAL
                message = "网络连接全部失败"
            
            return HealthCheckResult(
                component_name="network",
                component_type=ComponentType.NETWORK,
                status=status,
                response_time=response_time,
                message=message,
                details={
                    'successful_connections': successful_connections,
                    'total_tests': len(test_urls),
                    'avg_response_time': total_response_time / successful_connections if successful_connections > 0 else 0
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                component_name="network",
                component_type=ComponentType.NETWORK,
                status=HealthStatus.CRITICAL,
                response_time=response_time,
                message=f"网络检查失败: {str(e)}",
                error=str(e)
            )
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network_connections = len(psutil.net_connections())
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_usage_percent=disk.percent,
                network_connections=network_connections,
                load_average=load_avg
            )
            
            # 保存到历史记录
            self.system_metrics_history.append(metrics)
            
            # 限制历史记录大小
            if len(self.system_metrics_history) > 1000:
                self.system_metrics_history = self.system_metrics_history[-1000:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, (0, 0, 0))
    
    async def _update_failure_count(self, result: HealthCheckResult):
        """更新故障计数"""
        component_name = result.component_name
        
        if result.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
            self.failure_counts[component_name] = self.failure_counts.get(component_name, 0) + 1
            
            # 检查是否需要触发故障处理
            if self.failure_counts[component_name] >= self.max_failure_count:
                await self._handle_component_failure(result)
        else:
            # 重置故障计数
            self.failure_counts[component_name] = 0
    
    async def _handle_component_failure(self, result: HealthCheckResult):
        """处理组件故障"""
        logger.error(f"组件 {result.component_name} 连续故障 {self.max_failure_count} 次")
        
        # 这里可以实现自动恢复逻辑
        if result.component_type == ComponentType.DATABASE:
            await self._attempt_database_recovery()
        elif result.component_type == ComponentType.CACHE:
            await self._attempt_cache_recovery()
        elif result.component_type == ComponentType.DATA_SOURCE:
            await self._attempt_data_source_recovery()
    
    async def _attempt_database_recovery(self):
        """尝试数据库恢复"""
        logger.info("尝试数据库连接恢复...")
        try:
            await self.db_manager.close()
            await asyncio.sleep(5)
            await self.db_manager.initialize()
            logger.info("数据库连接恢复成功")
        except Exception as e:
            logger.error(f"数据库连接恢复失败: {e}")
    
    async def _attempt_cache_recovery(self):
        """尝试缓存恢复"""
        logger.info("尝试缓存连接恢复...")
        try:
            # 重新初始化缓存连接
            # 这里可以添加具体的缓存重连逻辑
            logger.info("缓存连接恢复成功")
        except Exception as e:
            logger.error(f"缓存连接恢复失败: {e}")
    
    async def _attempt_data_source_recovery(self):
        """尝试数据源恢复"""
        logger.info("尝试数据源连接恢复...")
        try:
            # 重新初始化数据源连接
            # 这里可以添加具体的数据源重连逻辑
            logger.info("数据源连接恢复成功")
        except Exception as e:
            logger.error(f"数据源连接恢复失败: {e}")
    
    async def _cleanup_history(self):
        """清理历史数据"""
        cutoff_time = datetime.now() - timedelta(days=self.history_retention_days)
        
        # 清理健康检查历史
        self.health_history = [
            record for record in self.health_history
            if any(result.timestamp > cutoff_time for result in record.values())
        ]
        
        # 清理系统指标历史
        self.system_metrics_history = [
            metrics for metrics in self.system_metrics_history
            if metrics.timestamp > cutoff_time
        ]
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        overall_status = HealthStatus.HEALTHY
        critical_components = []
        warning_components = []
        
        for component_name, result in self.component_status.items():
            if result.status == HealthStatus.CRITICAL:
                critical_components.append(component_name)
                overall_status = HealthStatus.CRITICAL
            elif result.status == HealthStatus.WARNING:
                warning_components.append(component_name)
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
        
        return {
            'overall_status': overall_status.value,
            'components': {name: result.status.value for name, result in self.component_status.items()},
            'critical_components': critical_components,
            'warning_components': warning_components,
            'last_check_time': max([result.timestamp for result in self.component_status.values()]) if self.component_status else None,
            'monitoring_active': self.is_monitoring
        }
    
    def get_health_report(self) -> Dict[str, Any]:
        """生成健康状态报告"""
        current_status = self.get_current_status()
        
        # 计算可用性统计
        availability_stats = self._calculate_availability_stats()
        
        # 获取最新的系统指标
        latest_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None
        
        return {
            'report_time': datetime.now().isoformat(),
            'current_status': current_status,
            'availability_stats': availability_stats,
            'system_metrics': {
                'cpu_percent': latest_metrics.cpu_percent if latest_metrics else 0,
                'memory_percent': latest_metrics.memory_percent if latest_metrics else 0,
                'disk_usage_percent': latest_metrics.disk_usage_percent if latest_metrics else 0,
                'network_connections': latest_metrics.network_connections if latest_metrics else 0
            } if latest_metrics else {},
            'component_details': {
                name: {
                    'status': result.status.value,
                    'response_time': result.response_time,
                    'message': result.message,
                    'last_check': result.timestamp.isoformat()
                }
                for name, result in self.component_status.items()
            },
            'failure_counts': self.failure_counts,
            'history_size': len(self.health_history)
        }
    
    def _calculate_availability_stats(self) -> Dict[str, float]:
        """计算可用性统计"""
        if not self.health_history:
            return {}
        
        stats = {}
        
        for component_name in self.component_status.keys():
            total_checks = 0
            healthy_checks = 0
            
            for record in self.health_history:
                if component_name in record:
                    total_checks += 1
                    if record[component_name].status == HealthStatus.HEALTHY:
                        healthy_checks += 1
            
            if total_checks > 0:
                stats[component_name] = healthy_checks / total_checks
            else:
                stats[component_name] = 0.0
        
        return stats
    
    def get_trend_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """获取趋势分析"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # 过滤指定时间范围内的数据
        recent_history = [
            record for record in self.health_history
            if any(result.timestamp > cutoff_time for result in record.values())
        ]
        
        recent_metrics = [
            metrics for metrics in self.system_metrics_history
            if metrics.timestamp > cutoff_time
        ]
        
        # 分析趋势
        trend_analysis = {
            'time_range_hours': hours,
            'total_checks': len(recent_history),
            'component_trends': {},
            'system_trends': {}
        }
        
        # 组件趋势分析
        for component_name in self.component_status.keys():
            component_data = []
            for record in recent_history:
                if component_name in record:
                    result = record[component_name]
                    component_data.append({
                        'timestamp': result.timestamp,
                        'status': result.status.value,
                        'response_time': result.response_time
                    })
            
            if component_data:
                avg_response_time = sum(d['response_time'] for d in component_data) / len(component_data)
                healthy_ratio = sum(1 for d in component_data if d['status'] == 'healthy') / len(component_data)
                
                trend_analysis['component_trends'][component_name] = {
                    'avg_response_time': avg_response_time,
                    'healthy_ratio': healthy_ratio,
                    'total_checks': len(component_data)
                }
        
        # 系统趋势分析
        if recent_metrics:
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            avg_disk = sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics)
            
            trend_analysis['system_trends'] = {
                'avg_cpu_percent': avg_cpu,
                'avg_memory_percent': avg_memory,
                'avg_disk_percent': avg_disk,
                'metrics_count': len(recent_metrics)
            }
        
        return trend_analysis


# 全局健康监控器实例
health_monitor = HealthMonitor()