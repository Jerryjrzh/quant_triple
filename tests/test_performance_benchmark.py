#!/usr/bin/env python3
"""
性能基准测试

创建系统性能基准测试套件，测试数据获取延迟、处理吞吐量等关键指标。
包含内存使用和CPU负载的性能监控，以及性能回归测试和持续监控。
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
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
import json
import statistics
from dataclasses import dataclass
from unittest.mock import Mock, patch

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
class PerformanceMetric:
    """性能指标数据类"""
    name: str
    value: float
    unit: str
    threshold: float
    passed: bool
    timestamp: datetime
    additional_data: Dict = None


@dataclass
class BenchmarkResult:
    """基准测试结果数据类"""
    test_name: str
    duration: float
    metrics: List[PerformanceMetric]
    system_resources: Dict
    success: bool
    error_message: str = None


class SystemResourceMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        self.interval = 0.1  # 100ms采样间隔
    
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict:
        """停止监控并返回统计结果"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if not self.metrics:
            return {}
        
        # 计算统计数据
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_percent'] for m in self.metrics]
        memory_mb_values = [m['memory_mb'] for m in self.metrics]
        
        return {
            'duration': len(self.metrics) * self.interval,
            'samples': len(self.metrics),
            'cpu': {
                'avg': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'std': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            },
            'memory': {
                'avg_percent': statistics.mean(memory_values),
                'max_percent': max(memory_values),
                'min_percent': min(memory_values),
                'avg_mb': statistics.mean(memory_mb_values),
                'max_mb': max(memory_mb_values),
                'min_mb': min(memory_mb_values)
            }
        }
    
    def _monitor_loop(self):
        """监控循环"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                self.metrics.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_info.rss / 1024 / 1024,  # MB
                    'memory_percent': memory_percent
                })
                
                time.sleep(self.interval)
            except Exception as e:
                logger.warning(f"资源监控异常: {e}")
                break


class PerformanceBenchmarkTest:
    """性能基准测试类"""
    
    def __init__(self):
        self.client = TestClient(app)
        self.data_sources = EnhancedDataSourceManager()
        self.data_source_manager = DataSourceManager()
        self.cache_manager = CacheManager()
        self.quality_engine = EnhancedDataQualityEngine()
        self.db_manager = DatabaseManager()
        self.resource_monitor = SystemResourceMonitor()
        
        # 测试配置
        self.test_symbols = ['000001', '000002', '600000', '600036', '300001', '000858', '002415', '600519']
        self.large_symbol_set = self.test_symbols * 10  # 扩大测试集
        self.test_dates = [
            (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
            for i in range(1, 8)
        ]
        
        # 性能基准阈值
        self.performance_thresholds = {
            'data_acquisition_latency': 2.0,      # 数据获取延迟 < 2秒
            'data_processing_throughput': 1000,   # 处理吞吐量 > 1000条/秒
            'database_query_latency': 0.5,        # 数据库查询延迟 < 0.5秒
            'cache_hit_ratio': 0.8,               # 缓存命中率 > 80%
            'api_response_time': 1.0,             # API响应时间 < 1秒
            'memory_usage_mb': 500,               # 内存使用 < 500MB
            'cpu_usage_percent': 80,              # CPU使用率 < 80%
            'concurrent_users': 50,               # 并发用户数 > 50
            'error_rate': 0.01                    # 错误率 < 1%
        }
        
        # 基准测试套件
        self.benchmark_suites = [
            'data_acquisition_benchmark',
            'data_processing_benchmark',
            'database_performance_benchmark',
            'cache_performance_benchmark',
            'api_performance_benchmark',
            'concurrent_performance_benchmark',
            'memory_performance_benchmark',
            'stress_test_benchmark'
        ]
    
    async def setup_test_environment(self):
        """设置测试环境"""
        logger.info("设置性能基准测试环境...")
        
        # 初始化数据库连接
        await self.db_manager.initialize()
        
        # 预热系统
        await self._warmup_system()
        
        # 清理垃圾回收
        gc.collect()
        
        logger.info("性能基准测试环境设置完成")
    
    async def teardown_test_environment(self):
        """清理测试环境"""
        logger.info("清理性能基准测试环境...")
        
        # 关闭数据库连接
        await self.db_manager.close()
        
        # 清理内存
        gc.collect()
        
        logger.info("性能基准测试环境清理完成")
    
    async def _warmup_system(self):
        """系统预热"""
        logger.info("系统预热中...")
        
        # 预热数据源
        for symbol in self.test_symbols[:3]:
            try:
                await self.data_sources.get_stock_info(symbol)
            except Exception:
                pass
        
        # 预热缓存
        await self.cache_manager.set("warmup_key", "warmup_value", ttl=10)
        await self.cache_manager.get("warmup_key")
        
        # 预热数据库
        try:
            await self.db_manager.fetch_one("SELECT 1")
        except Exception:
            pass
        
        logger.info("系统预热完成")
    
    async def run_all_benchmarks(self) -> Dict:
        """运行所有基准测试"""
        logger.info("开始运行性能基准测试套件...")
        
        results = {
            'test_start_time': datetime.now().isoformat(),
            'benchmark_results': {},
            'overall_metrics': {},
            'system_info': self._get_system_info(),
            'success': True,
            'summary': {}
        }
        
        try:
            # 运行各个基准测试
            for suite_name in self.benchmark_suites:
                logger.info(f"运行基准测试: {suite_name}")
                
                try:
                    suite_method = getattr(self, suite_name)
                    benchmark_result = await suite_method()
                    results['benchmark_results'][suite_name] = benchmark_result
                    
                    if not benchmark_result.success:
                        results['success'] = False
                
                except Exception as e:
                    logger.error(f"基准测试 {suite_name} 失败: {e}")
                    results['benchmark_results'][suite_name] = BenchmarkResult(
                        test_name=suite_name,
                        duration=0,
                        metrics=[],
                        system_resources={},
                        success=False,
                        error_message=str(e)
                    )
                    results['success'] = False
            
            # 计算总体指标
            results['overall_metrics'] = self._calculate_overall_metrics(results['benchmark_results'])
            
            # 生成测试摘要
            results['summary'] = self._generate_test_summary(results)
            
            results['test_end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"基准测试套件执行失败: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    async def data_acquisition_benchmark(self) -> BenchmarkResult:
        """数据获取性能基准测试"""
        logger.info("执行数据获取性能基准测试...")
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        metrics = []
        
        try:
            # 测试单个数据源获取延迟
            latencies = []
            for symbol in self.test_symbols:
                symbol_start = time.time()
                try:
                    data = await self.data_sources.get_realtime_data(symbol)
                    latency = time.time() - symbol_start
                    latencies.append(latency)
                    
                    if data is not None and not data.empty:
                        logger.debug(f"股票 {symbol} 数据获取延迟: {latency:.3f}秒")
                except Exception as e:
                    logger.warning(f"获取股票 {symbol} 数据失败: {e}")
                    latencies.append(float('inf'))
            
            # 计算延迟指标
            valid_latencies = [l for l in latencies if l != float('inf')]
            if valid_latencies:
                avg_latency = statistics.mean(valid_latencies)
                max_latency = max(valid_latencies)
                min_latency = min(valid_latencies)
                
                metrics.append(PerformanceMetric(
                    name="average_acquisition_latency",
                    value=avg_latency,
                    unit="seconds",
                    threshold=self.performance_thresholds['data_acquisition_latency'],
                    passed=avg_latency <= self.performance_thresholds['data_acquisition_latency'],
                    timestamp=datetime.now()
                ))
                
                metrics.append(PerformanceMetric(
                    name="max_acquisition_latency",
                    value=max_latency,
                    unit="seconds",
                    threshold=self.performance_thresholds['data_acquisition_latency'] * 2,
                    passed=max_latency <= self.performance_thresholds['data_acquisition_latency'] * 2,
                    timestamp=datetime.now()
                ))
            
            # 测试批量获取性能
            batch_start = time.time()
            batch_tasks = [
                self.data_sources.get_realtime_data(symbol) 
                for symbol in self.test_symbols
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            batch_duration = time.time() - batch_start
            
            successful_batch = sum(1 for r in batch_results if not isinstance(r, Exception))
            batch_throughput = successful_batch / batch_duration if batch_duration > 0 else 0
            
            metrics.append(PerformanceMetric(
                name="batch_acquisition_throughput",
                value=batch_throughput,
                unit="requests/second",
                threshold=len(self.test_symbols) / 2,  # 期望至少一半的吞吐量
                passed=batch_throughput >= len(self.test_symbols) / 2,
                timestamp=datetime.now()
            ))
            
            duration = time.time() - start_time
            system_resources = self.resource_monitor.stop_monitoring()
            
            return BenchmarkResult(
                test_name="data_acquisition_benchmark",
                duration=duration,
                metrics=metrics,
                system_resources=system_resources,
                success=all(m.passed for m in metrics)
            )
            
        except Exception as e:
            self.resource_monitor.stop_monitoring()
            return BenchmarkResult(
                test_name="data_acquisition_benchmark",
                duration=time.time() - start_time,
                metrics=metrics,
                system_resources={},
                success=False,
                error_message=str(e)
            )
    
    async def data_processing_benchmark(self) -> BenchmarkResult:
        """数据处理性能基准测试"""
        logger.info("执行数据处理性能基准测试...")
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        metrics = []
        
        try:
            # 生成测试数据
            test_data_size = 10000
            test_data = pd.DataFrame({
                'symbol': np.random.choice(self.test_symbols, test_data_size),
                'price': np.random.uniform(10, 100, test_data_size),
                'volume': np.random.randint(1000, 100000, test_data_size),
                'timestamp': pd.date_range(start='2024-01-01', periods=test_data_size, freq='1min')
            })
            
            # 测试数据验证性能
            validation_start = time.time()
            validation_results = []
            
            # 分批处理以测试吞吐量
            batch_size = 1000
            for i in range(0, len(test_data), batch_size):
                batch = test_data.iloc[i:i+batch_size]
                try:
                    result = await self.quality_engine.validate_realtime_data(batch)
                    validation_results.append(result)
                except Exception as e:
                    logger.warning(f"数据验证批次 {i//batch_size} 失败: {e}")
            
            validation_duration = time.time() - validation_start
            validation_throughput = test_data_size / validation_duration if validation_duration > 0 else 0
            
            metrics.append(PerformanceMetric(
                name="data_validation_throughput",
                value=validation_throughput,
                unit="records/second",
                threshold=self.performance_thresholds['data_processing_throughput'],
                passed=validation_throughput >= self.performance_thresholds['data_processing_throughput'],
                timestamp=datetime.now()
            ))
            
            # 测试数据转换性能
            transformation_start = time.time()
            
            # 模拟数据转换操作
            transformed_data = test_data.copy()
            transformed_data['price_change'] = transformed_data['price'].pct_change()
            transformed_data['volume_ma'] = transformed_data['volume'].rolling(window=5).mean()
            transformed_data['price_volatility'] = transformed_data['price'].rolling(window=10).std()
            
            transformation_duration = time.time() - transformation_start
            transformation_throughput = test_data_size / transformation_duration if transformation_duration > 0 else 0
            
            metrics.append(PerformanceMetric(
                name="data_transformation_throughput",
                value=transformation_throughput,
                unit="records/second",
                threshold=self.performance_thresholds['data_processing_throughput'] * 2,  # 转换应该更快
                passed=transformation_throughput >= self.performance_thresholds['data_processing_throughput'] * 2,
                timestamp=datetime.now()
            ))
            
            # 测试聚合计算性能
            aggregation_start = time.time()
            
            aggregated_data = test_data.groupby('symbol').agg({
                'price': ['mean', 'std', 'min', 'max'],
                'volume': ['sum', 'mean'],
                'timestamp': ['min', 'max']
            })
            
            aggregation_duration = time.time() - aggregation_start
            
            metrics.append(PerformanceMetric(
                name="data_aggregation_latency",
                value=aggregation_duration,
                unit="seconds",
                threshold=1.0,  # 聚合应该在1秒内完成
                passed=aggregation_duration <= 1.0,
                timestamp=datetime.now()
            ))
            
            duration = time.time() - start_time
            system_resources = self.resource_monitor.stop_monitoring()
            
            return BenchmarkResult(
                test_name="data_processing_benchmark",
                duration=duration,
                metrics=metrics,
                system_resources=system_resources,
                success=all(m.passed for m in metrics)
            )
            
        except Exception as e:
            self.resource_monitor.stop_monitoring()
            return BenchmarkResult(
                test_name="data_processing_benchmark",
                duration=time.time() - start_time,
                metrics=metrics,
                system_resources={},
                success=False,
                error_message=str(e)
            )
    
    async def database_performance_benchmark(self) -> BenchmarkResult:
        """数据库性能基准测试"""
        logger.info("执行数据库性能基准测试...")
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        metrics = []
        
        try:
            # 测试简单查询性能
            simple_query_times = []
            for _ in range(10):
                query_start = time.time()
                try:
                    result = await self.db_manager.fetch_one("SELECT COUNT(*) FROM stock_data LIMIT 1")
                    query_time = time.time() - query_start
                    simple_query_times.append(query_time)
                except Exception as e:
                    logger.warning(f"简单查询失败: {e}")
                    simple_query_times.append(float('inf'))
            
            valid_simple_times = [t for t in simple_query_times if t != float('inf')]
            if valid_simple_times:
                avg_simple_query_time = statistics.mean(valid_simple_times)
                
                metrics.append(PerformanceMetric(
                    name="simple_query_latency",
                    value=avg_simple_query_time,
                    unit="seconds",
                    threshold=self.performance_thresholds['database_query_latency'],
                    passed=avg_simple_query_time <= self.performance_thresholds['database_query_latency'],
                    timestamp=datetime.now()
                ))
            
            # 测试复杂查询性能
            complex_query_start = time.time()
            try:
                complex_query = """
                    SELECT symbol, AVG(close_price) as avg_price, 
                           COUNT(*) as record_count,
                           MAX(high_price) as max_high,
                           MIN(low_price) as min_low
                    FROM stock_data 
                    WHERE trade_date >= $1
                    GROUP BY symbol
                    ORDER BY avg_price DESC
                    LIMIT 10
                """
                result = await self.db_manager.fetch_all(
                    complex_query, 
                    (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                )
                complex_query_time = time.time() - complex_query_start
                
                metrics.append(PerformanceMetric(
                    name="complex_query_latency",
                    value=complex_query_time,
                    unit="seconds",
                    threshold=self.performance_thresholds['database_query_latency'] * 5,  # 复杂查询允许更长时间
                    passed=complex_query_time <= self.performance_thresholds['database_query_latency'] * 5,
                    timestamp=datetime.now()
                ))
                
            except Exception as e:
                logger.warning(f"复杂查询失败: {e}")
            
            # 测试批量插入性能
            batch_insert_start = time.time()
            try:
                # 生成测试数据
                test_records = []
                for i in range(100):
                    test_records.append((
                        f"TEST{i:03d}",
                        datetime.now().date(),
                        10.0 + i * 0.1,
                        10.5 + i * 0.1,
                        11.0 + i * 0.1,
                        9.5 + i * 0.1,
                        1000 + i * 10
                    ))
                
                # 批量插入
                insert_query = """
                    INSERT INTO stock_data (symbol, trade_date, open_price, close_price, high_price, low_price, volume)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (symbol, trade_date) DO NOTHING
                """
                
                for record in test_records:
                    await self.db_manager.execute(insert_query, *record)
                
                batch_insert_time = time.time() - batch_insert_start
                insert_throughput = len(test_records) / batch_insert_time if batch_insert_time > 0 else 0
                
                metrics.append(PerformanceMetric(
                    name="batch_insert_throughput",
                    value=insert_throughput,
                    unit="records/second",
                    threshold=100,  # 期望至少100条/秒
                    passed=insert_throughput >= 100,
                    timestamp=datetime.now()
                ))
                
                # 清理测试数据
                await self.db_manager.execute("DELETE FROM stock_data WHERE symbol LIKE 'TEST%'")
                
            except Exception as e:
                logger.warning(f"批量插入测试失败: {e}")
            
            duration = time.time() - start_time
            system_resources = self.resource_monitor.stop_monitoring()
            
            return BenchmarkResult(
                test_name="database_performance_benchmark",
                duration=duration,
                metrics=metrics,
                system_resources=system_resources,
                success=all(m.passed for m in metrics)
            )
            
        except Exception as e:
            self.resource_monitor.stop_monitoring()
            return BenchmarkResult(
                test_name="database_performance_benchmark",
                duration=time.time() - start_time,
                metrics=metrics,
                system_resources={},
                success=False,
                error_message=str(e)
            )
    
    async def cache_performance_benchmark(self) -> BenchmarkResult:
        """缓存性能基准测试"""
        logger.info("执行缓存性能基准测试...")
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        metrics = []
        
        try:
            # 测试缓存写入性能
            cache_keys = [f"benchmark_key_{i}" for i in range(1000)]
            cache_values = [f"benchmark_value_{i}" for i in range(1000)]
            
            write_start = time.time()
            write_tasks = [
                self.cache_manager.set(key, value, ttl=300)
                for key, value in zip(cache_keys, cache_values)
            ]
            await asyncio.gather(*write_tasks, return_exceptions=True)
            write_duration = time.time() - write_start
            write_throughput = len(cache_keys) / write_duration if write_duration > 0 else 0
            
            metrics.append(PerformanceMetric(
                name="cache_write_throughput",
                value=write_throughput,
                unit="operations/second",
                threshold=1000,  # 期望至少1000次/秒
                passed=write_throughput >= 1000,
                timestamp=datetime.now()
            ))
            
            # 测试缓存读取性能
            read_start = time.time()
            read_tasks = [self.cache_manager.get(key) for key in cache_keys]
            read_results = await asyncio.gather(*read_tasks, return_exceptions=True)
            read_duration = time.time() - read_start
            read_throughput = len(cache_keys) / read_duration if read_duration > 0 else 0
            
            metrics.append(PerformanceMetric(
                name="cache_read_throughput",
                value=read_throughput,
                unit="operations/second",
                threshold=2000,  # 读取应该比写入更快
                passed=read_throughput >= 2000,
                timestamp=datetime.now()
            ))
            
            # 计算缓存命中率
            successful_reads = sum(1 for r in read_results if not isinstance(r, Exception) and r is not None)
            hit_ratio = successful_reads / len(cache_keys) if cache_keys else 0
            
            metrics.append(PerformanceMetric(
                name="cache_hit_ratio",
                value=hit_ratio,
                unit="ratio",
                threshold=self.performance_thresholds['cache_hit_ratio'],
                passed=hit_ratio >= self.performance_thresholds['cache_hit_ratio'],
                timestamp=datetime.now()
            ))
            
            # 测试缓存删除性能
            delete_start = time.time()
            delete_tasks = [self.cache_manager.delete(key) for key in cache_keys[:100]]
            await asyncio.gather(*delete_tasks, return_exceptions=True)
            delete_duration = time.time() - delete_start
            delete_throughput = 100 / delete_duration if delete_duration > 0 else 0
            
            metrics.append(PerformanceMetric(
                name="cache_delete_throughput",
                value=delete_throughput,
                unit="operations/second",
                threshold=500,  # 期望至少500次/秒
                passed=delete_throughput >= 500,
                timestamp=datetime.now()
            ))
            
            # 清理剩余测试数据
            remaining_keys = cache_keys[100:]
            cleanup_tasks = [self.cache_manager.delete(key) for key in remaining_keys]
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            duration = time.time() - start_time
            system_resources = self.resource_monitor.stop_monitoring()
            
            return BenchmarkResult(
                test_name="cache_performance_benchmark",
                duration=duration,
                metrics=metrics,
                system_resources=system_resources,
                success=all(m.passed for m in metrics)
            )
            
        except Exception as e:
            self.resource_monitor.stop_monitoring()
            return BenchmarkResult(
                test_name="cache_performance_benchmark",
                duration=time.time() - start_time,
                metrics=metrics,
                system_resources={},
                success=False,
                error_message=str(e)
            )
    
    async def api_performance_benchmark(self) -> BenchmarkResult:
        """API性能基准测试"""
        logger.info("执行API性能基准测试...")
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        metrics = []
        
        try:
            # 测试不同API端点的响应时间
            api_endpoints = [
                ('/api/v1/stocks/000001', {}),
                ('/api/v1/realtime/000001', {}),
                ('/api/v1/history/000001', {'start_date': '2024-01-01', 'end_date': '2024-01-31'}),
                ('/health', {}),
                ('/api/v1/market/summary', {})
            ]
            
            endpoint_response_times = {}
            
            for endpoint, params in api_endpoints:
                response_times = []
                
                # 每个端点测试10次
                for _ in range(10):
                    request_start = time.time()
                    try:
                        if params:
                            response = self.client.get(endpoint, params=params)
                        else:
                            response = self.client.get(endpoint)
                        
                        response_time = time.time() - request_start
                        
                        if response.status_code == 200:
                            response_times.append(response_time)
                        else:
                            logger.warning(f"API {endpoint} 返回状态码: {response.status_code}")
                    
                    except Exception as e:
                        logger.warning(f"API {endpoint} 请求失败: {e}")
                
                if response_times:
                    avg_response_time = statistics.mean(response_times)
                    endpoint_response_times[endpoint] = avg_response_time
                    
                    metrics.append(PerformanceMetric(
                        name=f"api_response_time_{endpoint.replace('/', '_').replace('-', '_')}",
                        value=avg_response_time,
                        unit="seconds",
                        threshold=self.performance_thresholds['api_response_time'],
                        passed=avg_response_time <= self.performance_thresholds['api_response_time'],
                        timestamp=datetime.now()
                    ))
            
            # 计算总体API性能
            if endpoint_response_times:
                overall_avg_response_time = statistics.mean(endpoint_response_times.values())
                
                metrics.append(PerformanceMetric(
                    name="overall_api_response_time",
                    value=overall_avg_response_time,
                    unit="seconds",
                    threshold=self.performance_thresholds['api_response_time'],
                    passed=overall_avg_response_time <= self.performance_thresholds['api_response_time'],
                    timestamp=datetime.now()
                ))
            
            # 测试API吞吐量
            throughput_start = time.time()
            concurrent_requests = 50
            
            async def make_request():
                try:
                    response = self.client.get('/health')
                    return response.status_code == 200
                except:
                    return False
            
            throughput_tasks = [make_request() for _ in range(concurrent_requests)]
            throughput_results = await asyncio.gather(*throughput_tasks)
            throughput_duration = time.time() - throughput_start
            
            successful_requests = sum(throughput_results)
            api_throughput = successful_requests / throughput_duration if throughput_duration > 0 else 0
            
            metrics.append(PerformanceMetric(
                name="api_throughput",
                value=api_throughput,
                unit="requests/second",
                threshold=100,  # 期望至少100请求/秒
                passed=api_throughput >= 100,
                timestamp=datetime.now()
            ))
            
            duration = time.time() - start_time
            system_resources = self.resource_monitor.stop_monitoring()
            
            return BenchmarkResult(
                test_name="api_performance_benchmark",
                duration=duration,
                metrics=metrics,
                system_resources=system_resources,
                success=all(m.passed for m in metrics)
            )
            
        except Exception as e:
            self.resource_monitor.stop_monitoring()
            return BenchmarkResult(
                test_name="api_performance_benchmark",
                duration=time.time() - start_time,
                metrics=metrics,
                system_resources={},
                success=False,
                error_message=str(e)
            )
    
    async def concurrent_performance_benchmark(self) -> BenchmarkResult:
        """并发性能基准测试"""
        logger.info("执行并发性能基准测试...")
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        metrics = []
        
        try:
            # 测试不同并发级别下的性能
            concurrency_levels = [10, 25, 50, 100]
            
            for concurrency in concurrency_levels:
                logger.info(f"测试并发级别: {concurrency}")
                
                concurrent_start = time.time()
                
                async def concurrent_task(task_id):
                    try:
                        # 模拟混合工作负载
                        symbol = self.test_symbols[task_id % len(self.test_symbols)]
                        
                        # 数据获取
                        data = await self.data_sources.get_realtime_data(symbol)
                        
                        # 缓存操作
                        cache_key = f"concurrent_test_{task_id}"
                        await self.cache_manager.set(cache_key, f"value_{task_id}", ttl=60)
                        cached_value = await self.cache_manager.get(cache_key)
                        
                        # API调用
                        response = self.client.get('/health')
                        
                        return {
                            'task_id': task_id,
                            'success': response.status_code == 200,
                            'data_acquired': data is not None,
                            'cache_worked': cached_value is not None
                        }
                    
                    except Exception as e:
                        return {
                            'task_id': task_id,
                            'success': False,
                            'error': str(e)
                        }
                
                # 执行并发任务
                concurrent_tasks = [concurrent_task(i) for i in range(concurrency)]
                concurrent_results = await asyncio.gather(*concurrent_tasks)
                concurrent_duration = time.time() - concurrent_start
                
                # 分析结果
                successful_tasks = sum(1 for r in concurrent_results if r.get('success', False))
                success_rate = successful_tasks / concurrency if concurrency > 0 else 0
                throughput = successful_tasks / concurrent_duration if concurrent_duration > 0 else 0
                
                metrics.append(PerformanceMetric(
                    name=f"concurrent_success_rate_{concurrency}",
                    value=success_rate,
                    unit="ratio",
                    threshold=0.95,  # 期望95%成功率
                    passed=success_rate >= 0.95,
                    timestamp=datetime.now(),
                    additional_data={'concurrency_level': concurrency}
                ))
                
                metrics.append(PerformanceMetric(
                    name=f"concurrent_throughput_{concurrency}",
                    value=throughput,
                    unit="tasks/second",
                    threshold=concurrency * 0.5,  # 期望至少50%的理论吞吐量
                    passed=throughput >= concurrency * 0.5,
                    timestamp=datetime.now(),
                    additional_data={'concurrency_level': concurrency}
                ))
                
                # 清理并发测试的缓存数据
                cleanup_tasks = [
                    self.cache_manager.delete(f"concurrent_test_{i}")
                    for i in range(concurrency)
                ]
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            duration = time.time() - start_time
            system_resources = self.resource_monitor.stop_monitoring()
            
            return BenchmarkResult(
                test_name="concurrent_performance_benchmark",
                duration=duration,
                metrics=metrics,
                system_resources=system_resources,
                success=all(m.passed for m in metrics)
            )
            
        except Exception as e:
            self.resource_monitor.stop_monitoring()
            return BenchmarkResult(
                test_name="concurrent_performance_benchmark",
                duration=time.time() - start_time,
                metrics=metrics,
                system_resources={},
                success=False,
                error_message=str(e)
            )
    
    async def memory_performance_benchmark(self) -> BenchmarkResult:
        """内存性能基准测试"""
        logger.info("执行内存性能基准测试...")
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        metrics = []
        
        try:
            # 记录初始内存使用
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # 测试大数据集处理的内存使用
            large_data_size = 50000
            large_dataset = pd.DataFrame({
                'symbol': np.random.choice(self.test_symbols, large_data_size),
                'price': np.random.uniform(10, 100, large_data_size),
                'volume': np.random.randint(1000, 100000, large_data_size),
                'timestamp': pd.date_range(start='2024-01-01', periods=large_data_size, freq='1min')
            })
            
            # 执行内存密集型操作
            memory_operations = [
                lambda: large_dataset.groupby('symbol').agg({'price': ['mean', 'std'], 'volume': 'sum'}),
                lambda: large_dataset.sort_values(['symbol', 'timestamp']),
                lambda: large_dataset.merge(large_dataset.sample(1000), on='symbol', how='inner'),
                lambda: large_dataset.pivot_table(values='price', index='symbol', columns=large_dataset['timestamp'].dt.hour, aggfunc='mean')
            ]
            
            peak_memory = initial_memory
            
            for i, operation in enumerate(memory_operations):
                operation_start = time.time()
                try:
                    result = operation()
                    operation_duration = time.time() - operation_start
                    
                    # 检查内存使用
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    peak_memory = max(peak_memory, current_memory)
                    
                    logger.debug(f"内存操作 {i+1} 完成，耗时: {operation_duration:.2f}秒，内存: {current_memory:.1f}MB")
                    
                except Exception as e:
                    logger.warning(f"内存操作 {i+1} 失败: {e}")
            
            # 强制垃圾回收
            del large_dataset
            gc.collect()
            
            # 检查内存回收
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_growth = peak_memory - initial_memory
            memory_recovered = peak_memory - final_memory
            
            metrics.append(PerformanceMetric(
                name="peak_memory_usage",
                value=peak_memory,
                unit="MB",
                threshold=self.performance_thresholds['memory_usage_mb'],
                passed=peak_memory <= self.performance_thresholds['memory_usage_mb'],
                timestamp=datetime.now()
            ))
            
            metrics.append(PerformanceMetric(
                name="memory_growth",
                value=memory_growth,
                unit="MB",
                threshold=200,  # 期望内存增长不超过200MB
                passed=memory_growth <= 200,
                timestamp=datetime.now()
            ))
            
            metrics.append(PerformanceMetric(
                name="memory_recovery_ratio",
                value=memory_recovered / memory_growth if memory_growth > 0 else 1.0,
                unit="ratio",
                threshold=0.8,  # 期望至少回收80%的内存
                passed=(memory_recovered / memory_growth if memory_growth > 0 else 1.0) >= 0.8,
                timestamp=datetime.now()
            ))
            
            duration = time.time() - start_time
            system_resources = self.resource_monitor.stop_monitoring()
            
            return BenchmarkResult(
                test_name="memory_performance_benchmark",
                duration=duration,
                metrics=metrics,
                system_resources=system_resources,
                success=all(m.passed for m in metrics)
            )
            
        except Exception as e:
            self.resource_monitor.stop_monitoring()
            return BenchmarkResult(
                test_name="memory_performance_benchmark",
                duration=time.time() - start_time,
                metrics=metrics,
                system_resources={},
                success=False,
                error_message=str(e)
            )
    
    async def stress_test_benchmark(self) -> BenchmarkResult:
        """压力测试基准"""
        logger.info("执行压力测试基准...")
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        metrics = []
        
        try:
            # 高强度混合负载测试
            stress_duration = 30  # 30秒压力测试
            stress_end_time = time.time() + stress_duration
            
            error_count = 0
            success_count = 0
            response_times = []
            
            async def stress_worker(worker_id):
                worker_errors = 0
                worker_successes = 0
                worker_response_times = []
                
                while time.time() < stress_end_time:
                    try:
                        operation_start = time.time()
                        
                        # 随机选择操作类型
                        operation_type = np.random.choice(['data_fetch', 'cache_op', 'api_call', 'db_query'])
                        
                        if operation_type == 'data_fetch':
                            symbol = np.random.choice(self.test_symbols)
                            await self.data_sources.get_realtime_data(symbol)
                        
                        elif operation_type == 'cache_op':
                            key = f"stress_key_{worker_id}_{int(time.time())}"
                            await self.cache_manager.set(key, f"value_{worker_id}", ttl=10)
                            await self.cache_manager.get(key)
                        
                        elif operation_type == 'api_call':
                            self.client.get('/health')
                        
                        elif operation_type == 'db_query':
                            await self.db_manager.fetch_one("SELECT 1")
                        
                        operation_time = time.time() - operation_start
                        worker_response_times.append(operation_time)
                        worker_successes += 1
                        
                        # 短暂休息避免过度压力
                        await asyncio.sleep(0.01)
                    
                    except Exception as e:
                        worker_errors += 1
                        logger.debug(f"压力测试工作者 {worker_id} 操作失败: {e}")
                
                return {
                    'worker_id': worker_id,
                    'errors': worker_errors,
                    'successes': worker_successes,
                    'response_times': worker_response_times
                }
            
            # 启动多个压力测试工作者
            num_workers = 20
            stress_tasks = [stress_worker(i) for i in range(num_workers)]
            stress_results = await asyncio.gather(*stress_tasks)
            
            # 汇总结果
            for result in stress_results:
                error_count += result['errors']
                success_count += result['successes']
                response_times.extend(result['response_times'])
            
            total_operations = error_count + success_count
            error_rate = error_count / total_operations if total_operations > 0 else 0
            avg_response_time = statistics.mean(response_times) if response_times else 0
            operations_per_second = total_operations / stress_duration if stress_duration > 0 else 0
            
            metrics.append(PerformanceMetric(
                name="stress_test_error_rate",
                value=error_rate,
                unit="ratio",
                threshold=self.performance_thresholds['error_rate'],
                passed=error_rate <= self.performance_thresholds['error_rate'],
                timestamp=datetime.now()
            ))
            
            metrics.append(PerformanceMetric(
                name="stress_test_throughput",
                value=operations_per_second,
                unit="operations/second",
                threshold=100,  # 期望至少100操作/秒
                passed=operations_per_second >= 100,
                timestamp=datetime.now()
            ))
            
            metrics.append(PerformanceMetric(
                name="stress_test_avg_response_time",
                value=avg_response_time,
                unit="seconds",
                threshold=2.0,  # 压力测试下允许更长响应时间
                passed=avg_response_time <= 2.0,
                timestamp=datetime.now()
            ))
            
            duration = time.time() - start_time
            system_resources = self.resource_monitor.stop_monitoring()
            
            return BenchmarkResult(
                test_name="stress_test_benchmark",
                duration=duration,
                metrics=metrics,
                system_resources=system_resources,
                success=all(m.passed for m in metrics)
            )
            
        except Exception as e:
            self.resource_monitor.stop_monitoring()
            return BenchmarkResult(
                test_name="stress_test_benchmark",
                duration=time.time() - start_time,
                metrics=metrics,
                system_resources={},
                success=False,
                error_message=str(e)
            )
    
    def _get_system_info(self) -> Dict:
        """获取系统信息"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'disk_usage': psutil.disk_usage('/')._asdict(),
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
            'platform': psutil.platform.platform()
        }
    
    def _calculate_overall_metrics(self, benchmark_results: Dict) -> Dict:
        """计算总体指标"""
        overall_metrics = {
            'total_tests': len(benchmark_results),
            'passed_tests': 0,
            'failed_tests': 0,
            'total_duration': 0,
            'overall_success_rate': 0,
            'performance_score': 0
        }
        
        total_metrics = 0
        passed_metrics = 0
        
        for test_name, result in benchmark_results.items():
            overall_metrics['total_duration'] += result.duration
            
            if result.success:
                overall_metrics['passed_tests'] += 1
            else:
                overall_metrics['failed_tests'] += 1
            
            # 统计指标通过情况
            for metric in result.metrics:
                total_metrics += 1
                if metric.passed:
                    passed_metrics += 1
        
        overall_metrics['overall_success_rate'] = overall_metrics['passed_tests'] / overall_metrics['total_tests'] if overall_metrics['total_tests'] > 0 else 0
        overall_metrics['performance_score'] = passed_metrics / total_metrics if total_metrics > 0 else 0
        
        return overall_metrics
    
    def _generate_test_summary(self, results: Dict) -> Dict:
        """生成测试摘要"""
        summary = {
            'test_status': 'PASSED' if results['success'] else 'FAILED',
            'total_duration': results['overall_metrics']['total_duration'],
            'performance_score': results['overall_metrics']['performance_score'],
            'key_findings': [],
            'recommendations': []
        }
        
        # 分析关键发现
        performance_score = results['overall_metrics']['performance_score']
        if performance_score >= 0.9:
            summary['key_findings'].append("系统性能表现优秀，所有关键指标均达标")
        elif performance_score >= 0.7:
            summary['key_findings'].append("系统性能表现良好，部分指标需要优化")
        else:
            summary['key_findings'].append("系统性能存在问题，需要重点关注和优化")
        
        # 生成建议
        if performance_score < 0.8:
            summary['recommendations'].append("建议对性能较差的组件进行优化")
        
        if results['overall_metrics']['total_duration'] > 300:  # 5分钟
            summary['recommendations'].append("测试执行时间较长，建议优化测试效率")
        
        return summary


# 测试用例
@pytest.mark.asyncio
async def test_performance_benchmark_suite():
    """性能基准测试套件"""
    benchmark_test = PerformanceBenchmarkTest()
    
    try:
        # 设置测试环境
        await benchmark_test.setup_test_environment()
        
        # 运行所有基准测试
        results = await benchmark_test.run_all_benchmarks()
        
        # 验证测试结果
        assert results['success'], f"性能基准测试失败"
        assert results['overall_metrics']['performance_score'] >= 0.7, "性能评分过低"
        
        logger.info("性能基准测试套件通过")
        
        # 保存详细结果
        with open(f"performance_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
    finally:
        # 清理测试环境
        await benchmark_test.teardown_test_environment()


if __name__ == "__main__":
    # 运行性能基准测试
    async def run_benchmark():
        benchmark_test = PerformanceBenchmarkTest()
        
        try:
            await benchmark_test.setup_test_environment()
            
            print("=" * 80)
            print("开始性能基准测试套件")
            print("=" * 80)
            
            # 运行基准测试
            results = await benchmark_test.run_all_benchmarks()
            
            # 显示结果摘要
            print(f"\n测试状态: {results['summary']['test_status']}")
            print(f"总耗时: {results['overall_metrics']['total_duration']:.2f}秒")
            print(f"性能评分: {results['overall_metrics']['performance_score']:.2f}")
            print(f"通过测试: {results['overall_metrics']['passed_tests']}/{results['overall_metrics']['total_tests']}")
            
            print("\n关键发现:")
            for finding in results['summary']['key_findings']:
                print(f"  • {finding}")
            
            if results['summary']['recommendations']:
                print("\n建议:")
                for recommendation in results['summary']['recommendations']:
                    print(f"  • {recommendation}")
            
            # 保存详细结果
            result_file = f"performance_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            
            print(f"\n详细结果已保存到: {result_file}")
            
            print("\n" + "=" * 80)
            print("性能基准测试完成")
            print("=" * 80)
            
        finally:
            await benchmark_test.teardown_test_environment()
    
    # 运行基准测试
    asyncio.run(run_benchmark())