#!/usr/bin/env python3
"""
压力测试和负载测试

创建高并发用户访问的压力测试，测试系统在极限负载下的稳定性。
包含故障注入和恢复能力的测试，以及系统容量规划和扩展性验证。
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
import psutil
import time
import threading
import gc
import logging
import random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any, Callable
import json
import statistics
from dataclasses import dataclass, field
from unittest.mock import Mock, patch
import aiohttp
import multiprocessing
from contextlib import asynccontextmanager

from stock_analysis_system.data.enhanced_data_sources import EnhancedDataSourceManager
from stock_analysis_system.data.data_source_manager import DataSourceManager
from stock_analysis_system.data.cache_manager import CacheManager
from stock_analysis_system.data.data_quality_engine import EnhancedDataQualityEngine
from stock_analysis_system.core.database_manager import DatabaseManager
from stock_analysis_system.api.main import app
from fastapi.testclient import TestClient

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """负载测试配置"""
    name: str
    duration: int  # 测试持续时间（秒）
    concurrent_users: int  # 并发用户数
    ramp_up_time: int  # 用户启动时间（秒）
    think_time: float  # 用户思考时间（秒）
    operations: List[str]  # 操作类型列表
    failure_threshold: float = 0.05  # 失败率阈值
    response_time_threshold: float = 5.0  # 响应时间阈值


@dataclass
class StressTestResult:
    """压力测试结果"""
    test_name: str
    config: LoadTestConfig
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    percentile_95: float
    percentile_99: float
    throughput: float  # 请求/秒
    error_rate: float
    system_resources: Dict
    errors: List[Dict] = field(default_factory=list)
    success: bool = True


@dataclass
class FailureInjection:
    """故障注入配置"""
    name: str
    failure_type: str  # 'network', 'database', 'cache', 'memory', 'cpu'
    probability: float  # 故障概率 0-1
    duration: float  # 故障持续时间（秒）
    recovery_time: float  # 恢复时间（秒）


class SystemResourceTracker:
    """系统资源跟踪器"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.tracking = False
        self.metrics = []
        self.track_thread = None
    
    def start_tracking(self):
        """开始跟踪"""
        self.tracking = True
        self.metrics = []
        self.track_thread = threading.Thread(target=self._track_loop)
        self.track_thread.start()
    
    def stop_tracking(self) -> Dict:
        """停止跟踪并返回统计结果"""
        self.tracking = False
        if self.track_thread:
            self.track_thread.join()
        
        if not self.metrics:
            return {}
        
        # 计算统计数据
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_percent'] for m in self.metrics]
        memory_mb_values = [m['memory_mb'] for m in self.metrics]
        disk_io_read = [m['disk_io_read'] for m in self.metrics]
        disk_io_write = [m['disk_io_write'] for m in self.metrics]
        network_sent = [m['network_sent'] for m in self.metrics]
        network_recv = [m['network_recv'] for m in self.metrics]
        
        return {
            'duration': len(self.metrics) * self.interval,
            'samples': len(self.metrics),
            'cpu': {
                'avg': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'p95': np.percentile(cpu_values, 95) if cpu_values else 0
            },
            'memory': {
                'avg_percent': statistics.mean(memory_values),
                'max_percent': max(memory_values),
                'avg_mb': statistics.mean(memory_mb_values),
                'max_mb': max(memory_mb_values)
            },
            'disk_io': {
                'avg_read_mb_s': statistics.mean(disk_io_read),
                'avg_write_mb_s': statistics.mean(disk_io_write),
                'max_read_mb_s': max(disk_io_read),
                'max_write_mb_s': max(disk_io_write)
            },
            'network': {
                'avg_sent_mb_s': statistics.mean(network_sent),
                'avg_recv_mb_s': statistics.mean(network_recv),
                'max_sent_mb_s': max(network_sent),
                'max_recv_mb_s': max(network_recv)
            }
        }
    
    def _track_loop(self):
        """跟踪循环"""
        process = psutil.Process()
        last_disk_io = process.io_counters()
        last_network = psutil.net_io_counters()
        last_time = time.time()
        
        while self.tracking:
            try:
                current_time = time.time()
                time_delta = current_time - last_time
                
                # CPU和内存
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                # 磁盘IO
                current_disk_io = process.io_counters()
                disk_read_mb_s = (current_disk_io.read_bytes - last_disk_io.read_bytes) / time_delta / 1024 / 1024
                disk_write_mb_s = (current_disk_io.write_bytes - last_disk_io.write_bytes) / time_delta / 1024 / 1024
                
                # 网络IO
                current_network = psutil.net_io_counters()
                network_sent_mb_s = (current_network.bytes_sent - last_network.bytes_sent) / time_delta / 1024 / 1024
                network_recv_mb_s = (current_network.bytes_recv - last_network.bytes_recv) / time_delta / 1024 / 1024
                
                self.metrics.append({
                    'timestamp': current_time,
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_info.rss / 1024 / 1024,
                    'memory_percent': memory_percent,
                    'disk_io_read': disk_read_mb_s,
                    'disk_io_write': disk_write_mb_s,
                    'network_sent': network_sent_mb_s,
                    'network_recv': network_recv_mb_s
                })
                
                last_disk_io = current_disk_io
                last_network = current_network
                last_time = current_time
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.warning(f"资源跟踪异常: {e}")
                break


class FailureInjector:
    """故障注入器"""
    
    def __init__(self):
        self.active_failures = {}
        self.failure_history = []
    
    @asynccontextmanager
    async def inject_failure(self, failure: FailureInjection):
        """注入故障"""
        if random.random() > failure.probability:
            # 不触发故障
            yield
            return
        
        failure_id = f"{failure.name}_{int(time.time())}"
        logger.info(f"注入故障: {failure.name} ({failure.failure_type})")
        
        self.active_failures[failure_id] = failure
        self.failure_history.append({
            'failure_id': failure_id,
            'name': failure.name,
            'type': failure.failure_type,
            'start_time': datetime.now(),
            'duration': failure.duration
        })
        
        try:
            # 根据故障类型执行不同的故障注入
            if failure.failure_type == 'network':
                await self._inject_network_failure(failure)
            elif failure.failure_type == 'database':
                await self._inject_database_failure(failure)
            elif failure.failure_type == 'cache':
                await self._inject_cache_failure(failure)
            elif failure.failure_type == 'memory':
                await self._inject_memory_failure(failure)
            elif failure.failure_type == 'cpu':
                await self._inject_cpu_failure(failure)
            
            yield
            
        finally:
            # 故障恢复
            logger.info(f"故障恢复: {failure.name}")
            if failure_id in self.active_failures:
                del self.active_failures[failure_id]
            
            # 等待恢复时间
            if failure.recovery_time > 0:
                await asyncio.sleep(failure.recovery_time)
    
    async def _inject_network_failure(self, failure: FailureInjection):
        """注入网络故障"""
        # 模拟网络延迟或超时
        await asyncio.sleep(failure.duration)
    
    async def _inject_database_failure(self, failure: FailureInjection):
        """注入数据库故障"""
        # 模拟数据库连接问题
        await asyncio.sleep(failure.duration)
    
    async def _inject_cache_failure(self, failure: FailureInjection):
        """注入缓存故障"""
        # 模拟缓存不可用
        await asyncio.sleep(failure.duration)
    
    async def _inject_memory_failure(self, failure: FailureInjection):
        """注入内存故障"""
        # 模拟内存不足
        memory_hog = []
        try:
            # 分配大量内存
            for _ in range(int(failure.duration * 10)):
                memory_hog.append(np.random.random(100000))
                await asyncio.sleep(0.1)
        finally:
            del memory_hog
            gc.collect()
    
    async def _inject_cpu_failure(self, failure: FailureInjection):
        """注入CPU故障"""
        # 模拟CPU高负载
        end_time = time.time() + failure.duration
        
        def cpu_intensive_task():
            while time.time() < end_time:
                # CPU密集型计算
                sum(i * i for i in range(1000))
        
        # 启动多个CPU密集型任务
        with ThreadPoolExecutor(max_workers=psutil.cpu_count()) as executor:
            futures = [executor.submit(cpu_intensive_task) for _ in range(psutil.cpu_count())]
            await asyncio.sleep(failure.duration)


class VirtualUser:
    """虚拟用户"""
    
    def __init__(self, user_id: int, config: LoadTestConfig, failure_injector: FailureInjector):
        self.user_id = user_id
        self.config = config
        self.failure_injector = failure_injector
        self.client = TestClient(app)
        self.data_sources = EnhancedDataSourceManager()
        self.cache_manager = CacheManager()
        self.db_manager = DatabaseManager()
        
        # 用户状态
        self.requests_made = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = []
        self.errors = []
        self.active = False
    
    async def start_session(self):
        """开始用户会话"""
        self.active = True
        logger.debug(f"用户 {self.user_id} 开始会话")
        
        try:
            session_end_time = time.time() + self.config.duration
            
            while time.time() < session_end_time and self.active:
                # 选择随机操作
                operation = random.choice(self.config.operations)
                
                # 执行操作
                await self._execute_operation(operation)
                
                # 思考时间
                if self.config.think_time > 0:
                    think_time = random.uniform(0, self.config.think_time * 2)
                    await asyncio.sleep(think_time)
        
        except Exception as e:
            logger.error(f"用户 {self.user_id} 会话异常: {e}")
            self.errors.append({
                'timestamp': datetime.now(),
                'operation': 'session',
                'error': str(e)
            })
        
        finally:
            self.active = False
            logger.debug(f"用户 {self.user_id} 结束会话")
    
    async def _execute_operation(self, operation: str):
        """执行操作"""
        start_time = time.time()
        self.requests_made += 1
        
        try:
            # 随机故障注入
            failure = self._get_random_failure()
            
            if failure:
                async with self.failure_injector.inject_failure(failure):
                    await self._perform_operation(operation)
            else:
                await self._perform_operation(operation)
            
            self.successful_requests += 1
            
        except Exception as e:
            self.failed_requests += 1
            self.errors.append({
                'timestamp': datetime.now(),
                'operation': operation,
                'error': str(e)
            })
            logger.debug(f"用户 {self.user_id} 操作 {operation} 失败: {e}")
        
        finally:
            response_time = time.time() - start_time
            self.response_times.append(response_time)
    
    async def _perform_operation(self, operation: str):
        """执行具体操作"""
        if operation == 'get_realtime_data':
            symbol = random.choice(['000001', '000002', '600000', '600036', '300001'])
            await self.data_sources.get_realtime_data(symbol)
        
        elif operation == 'get_historical_data':
            symbol = random.choice(['000001', '000002', '600000', '600036', '300001'])
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            await self.data_sources.get_historical_data(symbol, start_date, end_date)
        
        elif operation == 'cache_operation':
            key = f"user_{self.user_id}_key_{random.randint(1, 100)}"
            value = f"value_{random.randint(1, 1000)}"
            
            # 随机选择缓存操作
            cache_op = random.choice(['set', 'get', 'delete'])
            if cache_op == 'set':
                await self.cache_manager.set(key, value, ttl=300)
            elif cache_op == 'get':
                await self.cache_manager.get(key)
            elif cache_op == 'delete':
                await self.cache_manager.delete(key)
        
        elif operation == 'database_query':
            # 随机数据库查询
            queries = [
                "SELECT COUNT(*) FROM stock_data",
                "SELECT symbol, AVG(close_price) FROM stock_data GROUP BY symbol LIMIT 10",
                "SELECT * FROM stock_data ORDER BY trade_date DESC LIMIT 5"
            ]
            query = random.choice(queries)
            await self.db_manager.fetch_all(query)
        
        elif operation == 'api_call':
            # 随机API调用
            endpoints = [
                '/health',
                '/api/v1/stocks/000001',
                '/api/v1/realtime/000001',
                '/api/v1/market/summary'
            ]
            endpoint = random.choice(endpoints)
            response = self.client.get(endpoint)
            
            if response.status_code != 200:
                raise Exception(f"API调用失败: {response.status_code}")
        
        elif operation == 'data_processing':
            # 模拟数据处理
            data_size = random.randint(100, 1000)
            test_data = pd.DataFrame({
                'price': np.random.uniform(10, 100, data_size),
                'volume': np.random.randint(1000, 100000, data_size)
            })
            
            # 执行一些计算
            test_data['ma5'] = test_data['price'].rolling(window=5).mean()
            test_data['volatility'] = test_data['price'].rolling(window=10).std()
            result = test_data.describe()
    
    def _get_random_failure(self) -> Optional[FailureInjection]:
        """获取随机故障（用于故障注入测试）"""
        if random.random() > 0.1:  # 90%的时候不注入故障
            return None
        
        failures = [
            FailureInjection("network_delay", "network", 0.05, 1.0, 0.5),
            FailureInjection("database_timeout", "database", 0.03, 2.0, 1.0),
            FailureInjection("cache_miss", "cache", 0.08, 0.5, 0.2),
            FailureInjection("memory_pressure", "memory", 0.02, 3.0, 2.0),
            FailureInjection("cpu_spike", "cpu", 0.02, 2.0, 1.0)
        ]
        
        return random.choice(failures)
    
    def get_statistics(self) -> Dict:
        """获取用户统计信息"""
        return {
            'user_id': self.user_id,
            'requests_made': self.requests_made,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'error_rate': self.failed_requests / self.requests_made if self.requests_made > 0 else 0,
            'avg_response_time': statistics.mean(self.response_times) if self.response_times else 0,
            'max_response_time': max(self.response_times) if self.response_times else 0,
            'min_response_time': min(self.response_times) if self.response_times else 0,
            'errors': self.errors
        }


class StressLoadTester:
    """压力和负载测试器"""
    
    def __init__(self):
        self.resource_tracker = SystemResourceTracker()
        self.failure_injector = FailureInjector()
        
        # 预定义的测试配置
        self.test_configs = {
            'light_load': LoadTestConfig(
                name="轻负载测试",
                duration=60,
                concurrent_users=10,
                ramp_up_time=10,
                think_time=2.0,
                operations=['get_realtime_data', 'api_call', 'cache_operation']
            ),
            'medium_load': LoadTestConfig(
                name="中等负载测试",
                duration=120,
                concurrent_users=50,
                ramp_up_time=30,
                think_time=1.0,
                operations=['get_realtime_data', 'get_historical_data', 'api_call', 'cache_operation', 'database_query']
            ),
            'heavy_load': LoadTestConfig(
                name="重负载测试",
                duration=180,
                concurrent_users=100,
                ramp_up_time=60,
                think_time=0.5,
                operations=['get_realtime_data', 'get_historical_data', 'api_call', 'cache_operation', 'database_query', 'data_processing']
            ),
            'stress_test': LoadTestConfig(
                name="压力测试",
                duration=300,
                concurrent_users=200,
                ramp_up_time=120,
                think_time=0.1,
                operations=['get_realtime_data', 'get_historical_data', 'api_call', 'cache_operation', 'database_query', 'data_processing'],
                failure_threshold=0.1,
                response_time_threshold=10.0
            ),
            'spike_test': LoadTestConfig(
                name="峰值测试",
                duration=60,
                concurrent_users=500,
                ramp_up_time=10,
                think_time=0.05,
                operations=['api_call', 'get_realtime_data'],
                failure_threshold=0.15,
                response_time_threshold=15.0
            ),
            'endurance_test': LoadTestConfig(
                name="耐久性测试",
                duration=1800,  # 30分钟
                concurrent_users=30,
                ramp_up_time=60,
                think_time=3.0,
                operations=['get_realtime_data', 'get_historical_data', 'api_call', 'cache_operation', 'database_query']
            )
        }
    
    async def setup_test_environment(self):
        """设置测试环境"""
        logger.info("设置压力负载测试环境...")
        
        # 初始化数据库连接
        db_manager = DatabaseManager()
        await db_manager.initialize()
        await db_manager.close()
        
        # 预热系统
        await self._warmup_system()
        
        logger.info("压力负载测试环境设置完成")
    
    async def teardown_test_environment(self):
        """清理测试环境"""
        logger.info("清理压力负载测试环境...")
        
        # 清理内存
        gc.collect()
        
        logger.info("压力负载测试环境清理完成")
    
    async def _warmup_system(self):
        """系统预热"""
        logger.info("系统预热中...")
        
        # 创建少量虚拟用户进行预热
        warmup_config = LoadTestConfig(
            name="预热测试",
            duration=10,
            concurrent_users=3,
            ramp_up_time=2,
            think_time=1.0,
            operations=['api_call', 'get_realtime_data']
        )
        
        await self.run_load_test(warmup_config)
        logger.info("系统预热完成")
    
    async def run_load_test(self, config: LoadTestConfig) -> StressTestResult:
        """运行负载测试"""
        logger.info(f"开始负载测试: {config.name}")
        logger.info(f"配置: {config.concurrent_users}用户, {config.duration}秒, 启动时间{config.ramp_up_time}秒")
        
        start_time = datetime.now()
        self.resource_tracker.start_tracking()
        
        # 创建虚拟用户
        users = [
            VirtualUser(i, config, self.failure_injector)
            for i in range(config.concurrent_users)
        ]
        
        try:
            # 逐步启动用户（ramp-up）
            user_tasks = []
            ramp_up_interval = config.ramp_up_time / config.concurrent_users if config.concurrent_users > 0 else 0
            
            for i, user in enumerate(users):
                # 延迟启动用户
                if i > 0:
                    await asyncio.sleep(ramp_up_interval)
                
                task = asyncio.create_task(user.start_session())
                user_tasks.append(task)
                logger.debug(f"启动用户 {user.user_id}")
            
            # 等待所有用户完成
            await asyncio.gather(*user_tasks, return_exceptions=True)
            
            end_time = datetime.now()
            system_resources = self.resource_tracker.stop_tracking()
            
            # 收集统计信息
            result = self._collect_test_results(config, users, start_time, end_time, system_resources)
            
            logger.info(f"负载测试完成: {config.name}")
            logger.info(f"总请求: {result.total_requests}, 成功: {result.successful_requests}, 失败: {result.failed_requests}")
            logger.info(f"错误率: {result.error_rate:.2%}, 平均响应时间: {result.avg_response_time:.3f}秒")
            logger.info(f"吞吐量: {result.throughput:.2f} 请求/秒")
            
            return result
            
        except Exception as e:
            logger.error(f"负载测试异常: {e}")
            end_time = datetime.now()
            system_resources = self.resource_tracker.stop_tracking()
            
            return StressTestResult(
                test_name=config.name,
                config=config,
                start_time=start_time,
                end_time=end_time,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time=0,
                max_response_time=0,
                min_response_time=0,
                percentile_95=0,
                percentile_99=0,
                throughput=0,
                error_rate=1.0,
                system_resources=system_resources,
                errors=[{'error': str(e), 'timestamp': datetime.now()}],
                success=False
            )
    
    def _collect_test_results(self, config: LoadTestConfig, users: List[VirtualUser], 
                            start_time: datetime, end_time: datetime, 
                            system_resources: Dict) -> StressTestResult:
        """收集测试结果"""
        
        # 汇总所有用户的统计信息
        total_requests = sum(user.requests_made for user in users)
        successful_requests = sum(user.successful_requests for user in users)
        failed_requests = sum(user.failed_requests for user in users)
        
        # 收集所有响应时间
        all_response_times = []
        all_errors = []
        
        for user in users:
            all_response_times.extend(user.response_times)
            all_errors.extend(user.errors)
        
        # 计算统计指标
        duration = (end_time - start_time).total_seconds()
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        throughput = successful_requests / duration if duration > 0 else 0
        
        if all_response_times:
            avg_response_time = statistics.mean(all_response_times)
            max_response_time = max(all_response_times)
            min_response_time = min(all_response_times)
            percentile_95 = np.percentile(all_response_times, 95)
            percentile_99 = np.percentile(all_response_times, 99)
        else:
            avg_response_time = max_response_time = min_response_time = 0
            percentile_95 = percentile_99 = 0
        
        # 判断测试是否成功
        success = (
            error_rate <= config.failure_threshold and
            avg_response_time <= config.response_time_threshold
        )
        
        return StressTestResult(
            test_name=config.name,
            config=config,
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time,
            percentile_95=percentile_95,
            percentile_99=percentile_99,
            throughput=throughput,
            error_rate=error_rate,
            system_resources=system_resources,
            errors=all_errors,
            success=success
        )
    
    async def run_all_tests(self) -> Dict:
        """运行所有测试"""
        logger.info("开始运行所有压力负载测试...")
        
        results = {
            'test_start_time': datetime.now().isoformat(),
            'test_results': {},
            'overall_summary': {},
            'system_capacity': {},
            'recommendations': []
        }
        
        try:
            # 按顺序运行测试（从轻到重）
            test_order = ['light_load', 'medium_load', 'heavy_load', 'stress_test', 'spike_test', 'endurance_test']
            
            for test_name in test_order:
                if test_name in self.test_configs:
                    logger.info(f"运行测试: {test_name}")
                    
                    try:
                        config = self.test_configs[test_name]
                        result = await self.run_load_test(config)
                        results['test_results'][test_name] = result
                        
                        # 如果关键测试失败，可能需要停止后续测试
                        if not result.success and test_name in ['stress_test', 'spike_test']:
                            logger.warning(f"关键测试 {test_name} 失败，考虑停止后续测试")
                            # 这里可以选择是否继续
                        
                        # 测试间隔，让系统恢复
                        await asyncio.sleep(30)
                        
                    except Exception as e:
                        logger.error(f"测试 {test_name} 执行失败: {e}")
                        results['test_results'][test_name] = {
                            'error': str(e),
                            'success': False
                        }
            
            # 生成总体摘要
            results['overall_summary'] = self._generate_overall_summary(results['test_results'])
            
            # 分析系统容量
            results['system_capacity'] = self._analyze_system_capacity(results['test_results'])
            
            # 生成建议
            results['recommendations'] = self._generate_recommendations(results['test_results'])
            
            results['test_end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"测试套件执行失败: {e}")
            results['error'] = str(e)
        
        return results
    
    def _generate_overall_summary(self, test_results: Dict) -> Dict:
        """生成总体摘要"""
        summary = {
            'total_tests': len(test_results),
            'passed_tests': 0,
            'failed_tests': 0,
            'max_concurrent_users': 0,
            'max_throughput': 0,
            'min_error_rate': 1.0,
            'avg_response_time_range': [float('inf'), 0]
        }
        
        for test_name, result in test_results.items():
            if isinstance(result, StressTestResult):
                if result.success:
                    summary['passed_tests'] += 1
                else:
                    summary['failed_tests'] += 1
                
                summary['max_concurrent_users'] = max(summary['max_concurrent_users'], result.config.concurrent_users)
                summary['max_throughput'] = max(summary['max_throughput'], result.throughput)
                summary['min_error_rate'] = min(summary['min_error_rate'], result.error_rate)
                
                summary['avg_response_time_range'][0] = min(summary['avg_response_time_range'][0], result.avg_response_time)
                summary['avg_response_time_range'][1] = max(summary['avg_response_time_range'][1], result.avg_response_time)
        
        return summary
    
    def _analyze_system_capacity(self, test_results: Dict) -> Dict:
        """分析系统容量"""
        capacity = {
            'max_stable_users': 0,
            'breaking_point': None,
            'resource_bottlenecks': [],
            'scalability_assessment': 'unknown'
        }
        
        # 找到最大稳定用户数
        for test_name, result in test_results.items():
            if isinstance(result, StressTestResult) and result.success:
                capacity['max_stable_users'] = max(capacity['max_stable_users'], result.config.concurrent_users)
        
        # 找到系统崩溃点
        for test_name, result in test_results.items():
            if isinstance(result, StressTestResult) and not result.success:
                if capacity['breaking_point'] is None or result.config.concurrent_users < capacity['breaking_point']:
                    capacity['breaking_point'] = result.config.concurrent_users
        
        # 分析资源瓶颈
        for test_name, result in test_results.items():
            if isinstance(result, StressTestResult) and result.system_resources:
                resources = result.system_resources
                
                if resources.get('cpu', {}).get('max', 0) > 80:
                    capacity['resource_bottlenecks'].append('CPU')
                
                if resources.get('memory', {}).get('max_percent', 0) > 80:
                    capacity['resource_bottlenecks'].append('Memory')
                
                if resources.get('disk_io', {}).get('max_read_mb_s', 0) > 100:
                    capacity['resource_bottlenecks'].append('Disk I/O')
        
        # 可扩展性评估
        if capacity['max_stable_users'] >= 100:
            capacity['scalability_assessment'] = 'excellent'
        elif capacity['max_stable_users'] >= 50:
            capacity['scalability_assessment'] = 'good'
        elif capacity['max_stable_users'] >= 20:
            capacity['scalability_assessment'] = 'fair'
        else:
            capacity['scalability_assessment'] = 'poor'
        
        return capacity
    
    def _generate_recommendations(self, test_results: Dict) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 分析测试结果并生成建议
        failed_tests = [name for name, result in test_results.items() 
                       if isinstance(result, StressTestResult) and not result.success]
        
        if failed_tests:
            recommendations.append(f"以下测试失败: {', '.join(failed_tests)}，需要优化系统性能")
        
        # 检查响应时间
        high_response_time_tests = [
            name for name, result in test_results.items()
            if isinstance(result, StressTestResult) and result.avg_response_time > 2.0
        ]
        
        if high_response_time_tests:
            recommendations.append("部分测试响应时间过长，建议优化数据库查询和缓存策略")
        
        # 检查错误率
        high_error_rate_tests = [
            name for name, result in test_results.items()
            if isinstance(result, StressTestResult) and result.error_rate > 0.05
        ]
        
        if high_error_rate_tests:
            recommendations.append("部分测试错误率过高，建议增强错误处理和重试机制")
        
        # 检查资源使用
        resource_intensive_tests = []
        for name, result in test_results.items():
            if isinstance(result, StressTestResult) and result.system_resources:
                resources = result.system_resources
                if (resources.get('cpu', {}).get('max', 0) > 80 or 
                    resources.get('memory', {}).get('max_percent', 0) > 80):
                    resource_intensive_tests.append(name)
        
        if resource_intensive_tests:
            recommendations.append("系统资源使用率过高，建议优化算法效率或增加硬件资源")
        
        # 通用建议
        if not recommendations:
            recommendations.append("系统性能表现良好，建议继续监控并定期进行压力测试")
        
        return recommendations


# 测试用例
@pytest.mark.asyncio
async def test_stress_load_testing_suite():
    """压力负载测试套件"""
    tester = StressLoadTester()
    
    try:
        # 设置测试环境
        await tester.setup_test_environment()
        
        # 运行轻负载测试（快速验证）
        light_config = tester.test_configs['light_load']
        light_result = await tester.run_load_test(light_config)
        
        # 验证测试结果
        assert light_result.success, f"轻负载测试失败: 错误率 {light_result.error_rate:.2%}"
        assert light_result.error_rate <= 0.05, "错误率过高"
        assert light_result.avg_response_time <= 5.0, "平均响应时间过长"
        
        logger.info("压力负载测试套件通过")
        
    finally:
        # 清理测试环境
        await tester.teardown_test_environment()


@pytest.mark.asyncio
async def test_full_stress_load_suite():
    """完整压力负载测试套件（耗时较长）"""
    tester = StressLoadTester()
    
    try:
        # 设置测试环境
        await tester.setup_test_environment()
        
        # 运行所有测试
        results = await tester.run_all_tests()
        
        # 验证结果
        assert results['overall_summary']['passed_tests'] > 0, "没有测试通过"
        
        # 保存详细结果
        result_file = f"stress_load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"完整压力负载测试完成，结果保存到: {result_file}")
        
    finally:
        # 清理测试环境
        await tester.teardown_test_environment()


if __name__ == "__main__":
    # 运行压力负载测试
    async def run_stress_tests():
        tester = StressLoadTester()
        
        try:
            await tester.setup_test_environment()
            
            print("=" * 80)
            print("开始压力负载测试套件")
            print("=" * 80)
            
            # 运行所有测试
            results = await tester.run_all_tests()
            
            # 显示结果摘要
            summary = results['overall_summary']
            capacity = results['system_capacity']
            
            print(f"\n测试摘要:")
            print(f"  总测试数: {summary['total_tests']}")
            print(f"  通过测试: {summary['passed_tests']}")
            print(f"  失败测试: {summary['failed_tests']}")
            print(f"  最大并发用户: {summary['max_concurrent_users']}")
            print(f"  最大吞吐量: {summary['max_throughput']:.2f} 请求/秒")
            print(f"  最低错误率: {summary['min_error_rate']:.2%}")
            
            print(f"\n系统容量分析:")
            print(f"  最大稳定用户数: {capacity['max_stable_users']}")
            print(f"  系统崩溃点: {capacity['breaking_point'] or '未达到'}")
            print(f"  资源瓶颈: {', '.join(capacity['resource_bottlenecks']) or '无'}")
            print(f"  可扩展性评估: {capacity['scalability_assessment']}")
            
            print(f"\n优化建议:")
            for i, recommendation in enumerate(results['recommendations'], 1):
                print(f"  {i}. {recommendation}")
            
            # 保存详细结果
            result_file = f"stress_load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            
            print(f"\n详细结果已保存到: {result_file}")
            
            print("\n" + "=" * 80)
            print("压力负载测试完成")
            print("=" * 80)
            
        finally:
            await tester.teardown_test_environment()
    
    # 运行测试
    asyncio.run(run_stress_tests())