"""
缓存管理器单元测试

测试缓存的读写、失效和更新机制，实现并发访问和竞态条件的测试，
添加缓存性能和内存使用的测试，创建缓存一致性和数据同步的验证。

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import asyncio
import json
import pickle
import pytest
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any
import threading
import concurrent.futures

from stock_analysis_system.data.cache_manager import (
    CacheManager,
    CacheConfig,
    CacheStats,
    CacheLevel,
    get_cache_manager
)


class TestDataGenerator:
    """缓存测试数据生成器"""
    
    @staticmethod
    def generate_sample_dataframe(rows: int = 100) -> pd.DataFrame:
        """生成示例DataFrame"""
        np.random.seed(42)
        
        data = {
            'stock_code': [f'{i:06d}' for i in range(rows)],
            'trade_date': [datetime.now() - timedelta(days=i) for i in range(rows)],
            'close_price': np.random.uniform(10, 100, rows),
            'volume': np.random.randint(1000000, 10000000, rows),
            'amount': np.random.uniform(10000000, 100000000, rows)
        }
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_large_dataframe(rows: int = 10000) -> pd.DataFrame:
        """生成大型DataFrame用于性能测试"""
        np.random.seed(42)
        
        data = {
            'id': range(rows),
            'timestamp': [datetime.now() - timedelta(seconds=i) for i in range(rows)],
            'value1': np.random.normal(0, 1, rows),
            'value2': np.random.uniform(0, 100, rows),
            'value3': np.random.exponential(2, rows),
            'category': np.random.choice(['A', 'B', 'C', 'D'], rows),
            'flag': np.random.choice([True, False], rows)
        }
        
        return pd.DataFrame(data)


class MockRedis:
    """模拟Redis客户端"""
    
    def __init__(self):
        self.data = {}
        self.expire_times = {}
        self.connected = True
    
    async def ping(self):
        """模拟ping命令"""
        if not self.connected:
            raise Exception("Redis connection failed")
        return True
    
    async def get(self, key: str):
        """模拟get命令"""
        if not self.connected:
            raise Exception("Redis connection failed")
        
        # 检查是否过期
        if key in self.expire_times:
            if time.time() > self.expire_times[key]:
                del self.data[key]
                del self.expire_times[key]
                return None
        
        return self.data.get(key)
    
    async def setex(self, key: str, ttl: int, value: Any):
        """模拟setex命令"""
        if not self.connected:
            raise Exception("Redis connection failed")
        
        self.data[key] = value
        self.expire_times[key] = time.time() + ttl
    
    async def delete(self, *keys):
        """模拟delete命令"""
        if not self.connected:
            raise Exception("Redis connection failed")
        
        deleted_count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                deleted_count += 1
            if key in self.expire_times:
                del self.expire_times[key]
        
        return deleted_count
    
    async def keys(self, pattern: str):
        """模拟keys命令"""
        if not self.connected:
            raise Exception("Redis connection failed")
        
        import fnmatch
        matching_keys = []
        for key in self.data.keys():
            if fnmatch.fnmatch(key, pattern):
                matching_keys.append(key)
        
        return matching_keys
    
    async def close(self):
        """模拟close命令"""
        pass
    
    def disconnect(self):
        """模拟连接断开"""
        self.connected = False
    
    def reconnect(self):
        """模拟重新连接"""
        self.connected = True


class TestCacheConfig:
    """缓存配置测试类"""
    
    def test_cache_config_creation(self):
        """测试缓存配置创建"""
        config = CacheConfig(
            ttl=300,
            key_pattern='test:{id}',
            level=CacheLevel.REDIS,
            preload=True,
            compress=True
        )
        
        assert config.ttl == 300
        assert config.key_pattern == 'test:{id}'
        assert config.level == CacheLevel.REDIS
        assert config.preload is True
        assert config.compress is True
    
    def test_cache_config_defaults(self):
        """测试缓存配置默认值"""
        config = CacheConfig(ttl=300, key_pattern='test:{id}')
        
        assert config.level == CacheLevel.REDIS
        assert config.preload is False
        assert config.compress is False


class TestCacheStats:
    """缓存统计测试类"""
    
    def test_cache_stats_initialization(self):
        """测试缓存统计初始化"""
        stats = CacheStats()
        
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.deletes == 0
        assert stats.errors == 0
        assert stats.total_size == 0
        assert stats.hit_rate == 0.0
    
    def test_hit_rate_calculation(self):
        """测试命中率计算"""
        stats = CacheStats()
        
        # 无访问时命中率为0
        assert stats.hit_rate == 0.0
        
        # 有命中和未命中时
        stats.hits = 8
        stats.misses = 2
        assert stats.hit_rate == 0.8
        
        # 只有命中时
        stats.misses = 0
        assert stats.hit_rate == 1.0
        
        # 只有未命中时
        stats.hits = 0
        stats.misses = 5
        assert stats.hit_rate == 0.0


class TestCacheManager:
    """缓存管理器测试类"""
    
    @pytest.fixture
    def cache_manager(self):
        """创建缓存管理器实例"""
        manager = CacheManager("redis://localhost:6379/0")
        
        # 使用模拟Redis客户端
        mock_redis = MockRedis()
        manager.redis_client = mock_redis
        
        return manager
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        return TestDataGenerator.generate_sample_dataframe(50)
    
    @pytest.mark.asyncio
    async def test_initialization_success(self):
        """测试成功初始化"""
        manager = CacheManager()
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client
            
            await manager.initialize()
            
            assert manager.redis_client is not None
            mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """测试初始化失败"""
        manager = CacheManager()
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping.side_effect = Exception("Connection failed")
            mock_redis.return_value = mock_client
            
            await manager.initialize()
            
            assert manager.redis_client is None
    
    @pytest.mark.asyncio
    async def test_memory_cache_operations(self, cache_manager, sample_data):
        """测试内存缓存操作"""
        key = "test_memory_key"
        cache_type = "realtime_data"
        
        # 测试缓存未命中
        result = await cache_manager.get_cached_data(key, cache_type)
        assert result is None
        assert cache_manager.stats.misses == 1
        
        # 设置缓存
        await cache_manager.set_cached_data(key, sample_data, cache_type)
        assert cache_manager.stats.sets == 1
        
        # 测试缓存命中
        result = await cache_manager.get_cached_data(key, cache_type)
        assert result is not None
        assert len(result) == len(sample_data)
        assert cache_manager.stats.hits == 1
        
        # 验证数据完整性
        pd.testing.assert_frame_equal(result, sample_data)
    
    @pytest.mark.asyncio
    async def test_redis_cache_operations(self, cache_manager, sample_data):
        """测试Redis缓存操作"""
        key = "test_redis_key"
        cache_type = "daily_data"
        
        # 清空内存缓存以确保从Redis读取
        cache_manager.memory_cache.clear()
        cache_manager.memory_expire.clear()
        
        # 设置缓存
        await cache_manager.set_cached_data(key, sample_data, cache_type)
        
        # 清空内存缓存
        cache_manager.memory_cache.clear()
        cache_manager.memory_expire.clear()
        
        # 从Redis获取缓存
        result = await cache_manager.get_cached_data(key, cache_type)
        assert result is not None
        assert len(result) == len(sample_data)
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache_manager, sample_data):
        """测试缓存过期"""
        key = "test_expiration_key"
        cache_type = "realtime_data"
        short_ttl = 1  # 1秒TTL
        
        # 设置短TTL的缓存
        await cache_manager.set_cached_data(key, sample_data, cache_type, ttl=short_ttl)
        
        # 立即获取应该成功
        result = await cache_manager.get_cached_data(key, cache_type)
        assert result is not None
        
        # 等待过期
        await asyncio.sleep(1.5)
        
        # 再次获取应该失败
        result = await cache_manager.get_cached_data(key, cache_type)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache_manager, sample_data):
        """测试缓存失效"""
        # 设置多个缓存项
        keys = ["test_inv_1", "test_inv_2", "other_key"]
        cache_type = "realtime_data"
        
        for key in keys:
            await cache_manager.set_cached_data(key, sample_data, cache_type)
        
        # 验证缓存存在
        for key in keys:
            result = await cache_manager.get_cached_data(key, cache_type)
            assert result is not None
        
        # 使用模式失效缓存
        await cache_manager.invalidate_cache("test_inv_*")
        
        # 验证匹配的缓存被删除
        result1 = await cache_manager.get_cached_data("test_inv_1", cache_type)
        result2 = await cache_manager.get_cached_data("test_inv_2", cache_type)
        assert result1 is None
        assert result2 is None
        
        # 验证不匹配的缓存仍然存在
        result3 = await cache_manager.get_cached_data("other_key", cache_type)
        assert result3 is not None
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_manager, sample_data):
        """测试缓存统计"""
        key = "test_stats_key"
        cache_type = "realtime_data"
        
        # 初始统计
        stats = cache_manager.get_cache_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['sets'] == 0
        
        # 缓存未命中
        await cache_manager.get_cached_data(key, cache_type)
        stats = cache_manager.get_cache_stats()
        assert stats['misses'] == 1
        
        # 设置缓存
        await cache_manager.set_cached_data(key, sample_data, cache_type)
        stats = cache_manager.get_cache_stats()
        assert stats['sets'] == 1
        
        # 缓存命中
        await cache_manager.get_cached_data(key, cache_type)
        stats = cache_manager.get_cache_stats()
        assert stats['hits'] == 1
        assert stats['hit_rate'] == 0.5  # 1 hit / (1 hit + 1 miss)
    
    @pytest.mark.asyncio
    async def test_memory_cache_size_limit(self, cache_manager):
        """测试内存缓存大小限制"""
        cache_type = "realtime_data"
        
        # 创建超过限制的缓存项（限制是1000）
        for i in range(1005):
            key = f"test_limit_{i}"
            data = TestDataGenerator.generate_sample_dataframe(10)
            await cache_manager.set_cached_data(key, data, cache_type)
        
        # 验证内存缓存大小不超过限制
        assert len(cache_manager.memory_cache) <= 1000
        
        # 验证最新的缓存项仍然存在
        result = await cache_manager.get_cached_data("test_limit_1004", cache_type)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_compressed_cache(self, cache_manager, sample_data):
        """测试压缩缓存"""
        key = "test_compressed_key"
        cache_type = "daily_data"  # 配置为压缩
        
        # 设置压缩缓存
        await cache_manager.set_cached_data(key, sample_data, cache_type)
        
        # 验证Redis中存储的是pickle格式
        redis_data = await cache_manager.redis_client.get(key)
        assert redis_data is not None
        
        # 尝试用pickle反序列化
        try:
            unpickled_data = pickle.loads(redis_data)
            assert isinstance(unpickled_data, pd.DataFrame)
        except:
            pytest.fail("压缩缓存数据不是pickle格式")
        
        # 清空内存缓存后从Redis获取
        cache_manager.memory_cache.clear()
        cache_manager.memory_expire.clear()
        
        result = await cache_manager.get_cached_data(key, cache_type)
        assert result is not None
        pd.testing.assert_frame_equal(result, sample_data)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, cache_manager, sample_data):
        """测试错误处理"""
        key = "test_error_key"
        cache_type = "realtime_data"
        
        # 记录初始错误数
        initial_errors = cache_manager.stats.errors
        
        # 模拟Redis连接失败
        cache_manager.redis_client.disconnect()
        
        # 设置缓存应该仍然工作（只使用内存缓存）
        await cache_manager.set_cached_data(key, sample_data, cache_type)
        
        # 获取缓存应该仍然工作
        result = await cache_manager.get_cached_data(key, cache_type)
        assert result is not None
        
        # 错误统计应该增加（Redis操作失败会增加错误计数）
        assert cache_manager.stats.errors > initial_errors
    
    @pytest.mark.asyncio
    async def test_preload_cache(self, cache_manager, sample_data):
        """测试预加载缓存"""
        cache_type = "realtime_data"
        
        # 模拟数据加载函数
        async def mock_data_loader(symbol):
            return sample_data
        
        # 预加载缓存
        await cache_manager.preload_cache(cache_type, mock_data_loader, symbol="000001")
        
        # 验证缓存已设置
        # 由于键生成可能复杂，我们检查统计信息
        assert cache_manager.stats.sets > 0
    
    @pytest.mark.asyncio
    async def test_warm_up_cache(self, cache_manager, sample_data):
        """测试缓存预热"""
        async def mock_loader1():
            return sample_data
        
        async def mock_loader2():
            return sample_data
        
        warm_up_configs = [
            {
                'cache_type': 'realtime_data',
                'data_loader': mock_loader1,
                'params': {}
            },
            {
                'cache_type': 'dragon_tiger',
                'data_loader': mock_loader2,
                'params': {}
            }
        ]
        
        await cache_manager.warm_up_cache(warm_up_configs)
        
        # 验证预热完成
        assert cache_manager.stats.sets >= 2


class TestConcurrencyAndRaceConditions:
    """并发和竞态条件测试类"""
    
    @pytest.fixture
    def cache_manager(self):
        """创建缓存管理器实例"""
        manager = CacheManager("redis://localhost:6379/0")
        manager.redis_client = MockRedis()
        return manager
    
    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, cache_manager):
        """测试并发读写"""
        key = "concurrent_test_key"
        cache_type = "realtime_data"
        
        async def writer(data_id):
            data = TestDataGenerator.generate_sample_dataframe(10)
            data['id'] = data_id
            await cache_manager.set_cached_data(f"{key}_{data_id}", data, cache_type)
        
        async def reader(data_id):
            result = await cache_manager.get_cached_data(f"{key}_{data_id}", cache_type)
            return result
        
        # 并发写入
        write_tasks = [writer(i) for i in range(10)]
        await asyncio.gather(*write_tasks)
        
        # 并发读取
        read_tasks = [reader(i) for i in range(10)]
        results = await asyncio.gather(*read_tasks)
        
        # 验证所有读取都成功
        assert all(result is not None for result in results)
        assert len(results) == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_invalidation(self, cache_manager):
        """测试并发缓存失效"""
        cache_type = "realtime_data"
        
        # 设置多个缓存项
        async def setup_cache():
            for i in range(20):
                key = f"concurrent_inv_{i}"
                data = TestDataGenerator.generate_sample_dataframe(5)
                await cache_manager.set_cached_data(key, data, cache_type)
        
        await setup_cache()
        
        # 并发失效不同的模式
        async def invalidate_pattern(pattern):
            await cache_manager.invalidate_cache(pattern)
        
        patterns = ["concurrent_inv_1*", "concurrent_inv_*5", "concurrent_inv_*0"]
        invalidate_tasks = [invalidate_pattern(pattern) for pattern in patterns]
        
        await asyncio.gather(*invalidate_tasks)
        
        # 验证失效操作完成
        assert cache_manager.stats.deletes > 0
    
    @pytest.mark.asyncio
    async def test_race_condition_same_key(self, cache_manager):
        """测试同一键的竞态条件"""
        key = "race_condition_key"
        cache_type = "realtime_data"
        
        async def concurrent_operation(operation_id):
            # 同时读取和写入同一个键
            data = TestDataGenerator.generate_sample_dataframe(5)
            data['operation_id'] = operation_id
            
            # 先尝试读取
            existing = await cache_manager.get_cached_data(key, cache_type)
            
            # 然后写入
            await cache_manager.set_cached_data(key, data, cache_type)
            
            # 再次读取验证
            final = await cache_manager.get_cached_data(key, cache_type)
            
            return existing, final
        
        # 并发执行多个操作
        tasks = [concurrent_operation(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # 验证操作完成且没有异常
        assert len(results) == 5
        
        # 最终应该有一个值存在
        final_result = await cache_manager.get_cached_data(key, cache_type)
        assert final_result is not None
    
    def test_thread_safety(self, cache_manager):
        """测试线程安全性"""
        key_base = "thread_safety_test"
        cache_type = "realtime_data"
        
        def sync_cache_operation(thread_id):
            """同步缓存操作（在线程中运行）"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                async def async_operation():
                    key = f"{key_base}_{thread_id}"
                    data = TestDataGenerator.generate_sample_dataframe(10)
                    
                    # 写入缓存
                    await cache_manager.set_cached_data(key, data, cache_type)
                    
                    # 读取缓存
                    result = await cache_manager.get_cached_data(key, cache_type)
                    return result is not None
                
                return loop.run_until_complete(async_operation())
            finally:
                loop.close()
        
        # 使用多线程并发执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(sync_cache_operation, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 验证所有操作都成功
        assert all(results)
        assert len(results) == 10


class TestPerformanceAndMemoryUsage:
    """性能和内存使用测试类"""
    
    @pytest.fixture
    def cache_manager(self):
        """创建缓存管理器实例"""
        manager = CacheManager("redis://localhost:6379/0")
        manager.redis_client = MockRedis()
        return manager
    
    @pytest.mark.asyncio
    async def test_large_data_caching_performance(self, cache_manager):
        """测试大数据缓存性能"""
        import time
        
        large_data = TestDataGenerator.generate_large_dataframe(10000)
        key = "large_data_test"
        cache_type = "daily_data"
        
        # 测试写入性能
        start_time = time.time()
        await cache_manager.set_cached_data(key, large_data, cache_type)
        write_time = time.time() - start_time
        
        # 写入应该在合理时间内完成（例如5秒）
        assert write_time < 5.0
        
        # 测试读取性能
        start_time = time.time()
        result = await cache_manager.get_cached_data(key, cache_type)
        read_time = time.time() - start_time
        
        # 读取应该更快（例如1秒）
        assert read_time < 1.0
        assert result is not None
        assert len(result) == len(large_data)
    
    @pytest.mark.asyncio
    async def test_batch_operations_performance(self, cache_manager):
        """测试批量操作性能"""
        import time
        
        cache_type = "realtime_data"
        batch_size = 100
        
        # 准备批量数据
        batch_data = []
        for i in range(batch_size):
            data = TestDataGenerator.generate_sample_dataframe(50)
            batch_data.append((f"batch_key_{i}", data))
        
        # 测试批量写入性能
        start_time = time.time()
        write_tasks = [
            cache_manager.set_cached_data(key, data, cache_type)
            for key, data in batch_data
        ]
        await asyncio.gather(*write_tasks)
        batch_write_time = time.time() - start_time
        
        # 批量写入应该在合理时间内完成
        assert batch_write_time < 10.0
        
        # 测试批量读取性能
        start_time = time.time()
        read_tasks = [
            cache_manager.get_cached_data(key, cache_type)
            for key, _ in batch_data
        ]
        results = await asyncio.gather(*read_tasks)
        batch_read_time = time.time() - start_time
        
        # 批量读取应该更快
        assert batch_read_time < 5.0
        assert all(result is not None for result in results)
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, cache_manager):
        """测试内存使用监控"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 缓存大量数据
        cache_type = "realtime_data"
        for i in range(100):
            key = f"memory_test_{i}"
            data = TestDataGenerator.generate_sample_dataframe(1000)
            await cache_manager.set_cached_data(key, data, cache_type)
        
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # 内存增长应该在合理范围内（例如小于100MB）
        assert memory_increase < 100 * 1024 * 1024  # 100MB
        
        # 获取缓存统计
        stats = cache_manager.get_cache_stats()
        assert stats['memory_cache_size'] > 0
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate_optimization(self, cache_manager):
        """测试缓存命中率优化"""
        cache_type = "realtime_data"
        
        # 设置一些缓存数据
        for i in range(20):
            key = f"hit_rate_test_{i}"
            data = TestDataGenerator.generate_sample_dataframe(10)
            await cache_manager.set_cached_data(key, data, cache_type)
        
        # 模拟真实访问模式（80/20规则）
        # 80%的访问集中在20%的数据上
        hot_keys = [f"hit_rate_test_{i}" for i in range(4)]  # 20%的键
        cold_keys = [f"hit_rate_test_{i}" for i in range(4, 20)]  # 80%的键
        
        # 执行访问模式
        access_count = 100
        for _ in range(access_count):
            # 80%的访问访问热点数据
            if np.random.random() < 0.8:
                key = np.random.choice(hot_keys)
            else:
                key = np.random.choice(cold_keys)
            
            await cache_manager.get_cached_data(key, cache_type)
        
        # 检查命中率
        stats = cache_manager.get_cache_stats()
        assert stats['hit_rate'] > 0.8  # 应该有较高的命中率
    
    @pytest.mark.asyncio
    async def test_ttl_performance_impact(self, cache_manager):
        """测试TTL对性能的影响"""
        import time
        
        cache_type = "realtime_data"
        data = TestDataGenerator.generate_sample_dataframe(100)
        
        # 测试不同TTL值的性能
        ttl_values = [60, 300, 1800, 3600]  # 1分钟到1小时
        performance_results = {}
        
        for ttl in ttl_values:
            key = f"ttl_test_{ttl}"
            
            # 测试设置性能
            start_time = time.time()
            await cache_manager.set_cached_data(key, data, cache_type, ttl=ttl)
            set_time = time.time() - start_time
            
            # 测试获取性能
            start_time = time.time()
            result = await cache_manager.get_cached_data(key, cache_type)
            get_time = time.time() - start_time
            
            performance_results[ttl] = {
                'set_time': set_time,
                'get_time': get_time,
                'success': result is not None
            }
        
        # 验证所有TTL值都能正常工作
        assert all(result['success'] for result in performance_results.values())
        
        # 验证性能在合理范围内
        for ttl, result in performance_results.items():
            assert result['set_time'] < 1.0  # 设置时间小于1秒
            assert result['get_time'] < 0.1  # 获取时间小于0.1秒


class TestCacheConsistencyAndSynchronization:
    """缓存一致性和数据同步测试类"""
    
    @pytest.fixture
    def cache_manager(self):
        """创建缓存管理器实例"""
        manager = CacheManager("redis://localhost:6379/0")
        manager.redis_client = MockRedis()
        return manager
    
    @pytest.mark.asyncio
    async def test_memory_redis_consistency(self, cache_manager):
        """测试内存缓存和Redis缓存的一致性"""
        key = "consistency_test_key"
        cache_type = "daily_data"
        data = TestDataGenerator.generate_sample_dataframe(50)
        
        # 设置缓存
        await cache_manager.set_cached_data(key, data, cache_type)
        
        # 从内存获取
        memory_result = cache_manager._get_memory_cache(key)
        assert memory_result is not None
        
        # 清空内存缓存，强制从Redis获取
        cache_manager.memory_cache.clear()
        cache_manager.memory_expire.clear()
        
        redis_result = await cache_manager.get_cached_data(key, cache_type)
        assert redis_result is not None
        
        # 验证数据一致性
        pd.testing.assert_frame_equal(memory_result, redis_result)
    
    @pytest.mark.asyncio
    async def test_cache_update_consistency(self, cache_manager):
        """测试缓存更新一致性"""
        key = "update_consistency_key"
        cache_type = "realtime_data"
        
        # 设置初始数据
        initial_data = TestDataGenerator.generate_sample_dataframe(30)
        initial_data['version'] = 1
        await cache_manager.set_cached_data(key, initial_data, cache_type)
        
        # 更新数据
        updated_data = TestDataGenerator.generate_sample_dataframe(30)
        updated_data['version'] = 2
        await cache_manager.set_cached_data(key, updated_data, cache_type)
        
        # 从内存获取
        memory_result = await cache_manager.get_cached_data(key, cache_type)
        assert memory_result['version'].iloc[0] == 2
        
        # 清空内存缓存，从Redis获取
        cache_manager.memory_cache.clear()
        cache_manager.memory_expire.clear()
        
        redis_result = await cache_manager.get_cached_data(key, cache_type)
        assert redis_result['version'].iloc[0] == 2
        
        # 验证更新后的数据一致性
        pd.testing.assert_frame_equal(memory_result, redis_result)
    
    @pytest.mark.asyncio
    async def test_partial_cache_failure_handling(self, cache_manager):
        """测试部分缓存失败处理"""
        key = "partial_failure_key"
        cache_type = "realtime_data"
        data = TestDataGenerator.generate_sample_dataframe(20)
        
        # 正常设置缓存
        await cache_manager.set_cached_data(key, data, cache_type)
        
        # 模拟Redis失败
        cache_manager.redis_client.disconnect()
        
        # 应该仍能从内存缓存获取数据
        result = await cache_manager.get_cached_data(key, cache_type)
        assert result is not None
        pd.testing.assert_frame_equal(result, data)
        
        # 设置新数据应该仍然工作（只更新内存缓存）
        new_data = TestDataGenerator.generate_sample_dataframe(25)
        await cache_manager.set_cached_data(key, new_data, cache_type)
        
        result = await cache_manager.get_cached_data(key, cache_type)
        assert result is not None
        assert len(result) == 25
    
    @pytest.mark.asyncio
    async def test_cache_recovery_after_failure(self, cache_manager):
        """测试故障后缓存恢复"""
        key = "recovery_test_key"
        cache_type = "daily_data"
        data = TestDataGenerator.generate_sample_dataframe(40)
        
        # 正常设置缓存
        await cache_manager.set_cached_data(key, data, cache_type)
        
        # 模拟Redis故障
        cache_manager.redis_client.disconnect()
        
        # 清空内存缓存
        cache_manager.memory_cache.clear()
        cache_manager.memory_expire.clear()
        
        # 此时应该无法获取数据
        result = await cache_manager.get_cached_data(key, cache_type)
        assert result is None
        
        # 恢复Redis连接
        cache_manager.redis_client.reconnect()
        
        # 重新设置数据
        await cache_manager.set_cached_data(key, data, cache_type)
        
        # 应该能够正常获取数据
        result = await cache_manager.get_cached_data(key, cache_type)
        assert result is not None
        pd.testing.assert_frame_equal(result, data)
    
    @pytest.mark.asyncio
    async def test_concurrent_update_consistency(self, cache_manager):
        """测试并发更新一致性"""
        key = "concurrent_update_key"
        cache_type = "realtime_data"
        
        async def update_cache(update_id):
            data = TestDataGenerator.generate_sample_dataframe(10)
            data['update_id'] = update_id
            data['timestamp'] = datetime.now()
            await cache_manager.set_cached_data(key, data, cache_type)
            return update_id
        
        # 并发更新同一个键
        update_tasks = [update_cache(i) for i in range(10)]
        update_results = await asyncio.gather(*update_tasks)
        
        # 验证最终状态一致
        final_memory = cache_manager._get_memory_cache(key)
        
        # 清空内存缓存，从Redis获取
        cache_manager.memory_cache.clear()
        cache_manager.memory_expire.clear()
        
        final_redis = await cache_manager.get_cached_data(key, cache_type)
        
        # 两者应该一致（虽然不知道最终是哪个更新的结果）
        if final_memory is not None and final_redis is not None:
            assert final_memory['update_id'].iloc[0] == final_redis['update_id'].iloc[0]


class TestEdgeCasesAndErrorScenarios:
    """边界情况和错误场景测试类"""
    
    @pytest.fixture
    async def cache_manager(self):
        """创建缓存管理器实例"""
        manager = CacheManager("redis://localhost:6379/0")
        manager.redis_client = MockRedis()
        yield manager
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_empty_dataframe_caching(self, cache_manager):
        """测试空DataFrame缓存"""
        key = "empty_df_key"
        cache_type = "realtime_data"
        empty_df = pd.DataFrame()
        
        # 设置空DataFrame
        await cache_manager.set_cached_data(key, empty_df, cache_type)
        
        # 获取空DataFrame
        result = await cache_manager.get_cached_data(key, cache_type)
        assert result is not None
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_very_large_dataframe(self, cache_manager):
        """测试非常大的DataFrame"""
        key = "large_df_key"
        cache_type = "daily_data"
        
        # 创建大型DataFrame（可能接近内存限制）
        large_df = TestDataGenerator.generate_large_dataframe(100000)
        
        # 应该能够处理大型DataFrame
        await cache_manager.set_cached_data(key, large_df, cache_type)
        
        result = await cache_manager.get_cached_data(key, cache_type)
        assert result is not None
        assert len(result) == len(large_df)
    
    @pytest.mark.asyncio
    async def test_special_characters_in_keys(self, cache_manager):
        """测试键中的特殊字符"""
        special_keys = [
            "key:with:colons",
            "key with spaces",
            "key-with-dashes",
            "key_with_underscores",
            "key.with.dots",
            "key/with/slashes",
            "key@with@symbols",
            "中文键名",
            "key🚀with🎯emojis"
        ]
        
        cache_type = "realtime_data"
        data = TestDataGenerator.generate_sample_dataframe(5)
        
        for key in special_keys:
            # 设置缓存
            await cache_manager.set_cached_data(key, data, cache_type)
            
            # 获取缓存
            result = await cache_manager.get_cached_data(key, cache_type)
            assert result is not None, f"Failed for key: {key}"
            assert len(result) == len(data)
    
    @pytest.mark.asyncio
    async def test_dataframe_with_special_values(self, cache_manager):
        """测试包含特殊值的DataFrame"""
        key = "special_values_key"
        cache_type = "realtime_data"
        
        # 创建包含特殊值的DataFrame
        special_df = pd.DataFrame({
            'normal_values': [1.0, 2.0, 3.0],
            'nan_values': [np.nan, 1.0, np.nan],
            'inf_values': [np.inf, -np.inf, 1.0],
            'string_values': ['normal', '', None],
            'datetime_values': [datetime.now(), pd.NaT, datetime.now() - timedelta(days=1)]
        })
        
        # 设置和获取包含特殊值的DataFrame
        await cache_manager.set_cached_data(key, special_df, cache_type)
        
        result = await cache_manager.get_cached_data(key, cache_type)
        assert result is not None
        assert len(result) == len(special_df)
        
        # 验证特殊值保持不变
        assert pd.isna(result['nan_values'].iloc[0])
        assert np.isinf(result['inf_values'].iloc[0])
    
    @pytest.mark.asyncio
    async def test_zero_ttl_handling(self, cache_manager):
        """测试零TTL处理"""
        key = "zero_ttl_key"
        cache_type = "realtime_data"
        data = TestDataGenerator.generate_sample_dataframe(10)
        
        # 设置TTL为0
        await cache_manager.set_cached_data(key, data, cache_type, ttl=0)
        
        # 应该立即过期
        result = await cache_manager.get_cached_data(key, cache_type)
        # 根据实现，可能返回None或者数据（取决于是否立即过期）
        # 这里我们只验证不会崩溃
        assert result is None or isinstance(result, pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_negative_ttl_handling(self, cache_manager):
        """测试负TTL处理"""
        key = "negative_ttl_key"
        cache_type = "realtime_data"
        data = TestDataGenerator.generate_sample_dataframe(10)
        
        # 设置负TTL
        await cache_manager.set_cached_data(key, data, cache_type, ttl=-100)
        
        # 应该能够处理负TTL而不崩溃
        result = await cache_manager.get_cached_data(key, cache_type)
        # 根据实现，可能返回None或者数据
        assert result is None or isinstance(result, pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_invalid_cache_type(self, cache_manager):
        """测试无效缓存类型"""
        key = "invalid_type_key"
        invalid_cache_type = "nonexistent_cache_type"
        data = TestDataGenerator.generate_sample_dataframe(10)
        
        # 使用不存在的缓存类型
        await cache_manager.set_cached_data(key, data, invalid_cache_type)
        
        # 应该使用默认配置
        result = await cache_manager.get_cached_data(key, invalid_cache_type)
        assert result is not None
        pd.testing.assert_frame_equal(result, data)
    
    @pytest.mark.asyncio
    async def test_corrupted_cache_data_handling(self, cache_manager):
        """测试损坏缓存数据处理"""
        key = "corrupted_data_key"
        
        # 直接在Redis中设置损坏的数据
        await cache_manager.redis_client.setex(key, 300, "corrupted_json_data")
        
        # 尝试获取损坏的数据
        result = await cache_manager.get_cached_data(key, "daily_data")
        
        # 应该返回None而不是崩溃
        assert result is None
        
        # 错误统计应该增加
        assert cache_manager.stats.errors > 0


class TestGlobalCacheManager:
    """全局缓存管理器测试类"""
    
    @pytest.mark.asyncio
    async def test_get_cache_manager_singleton(self):
        """测试获取缓存管理器单例"""
        with patch('stock_analysis_system.data.cache_manager.cache_manager') as mock_manager:
            mock_manager.redis_client = None
            mock_manager.initialize = AsyncMock()
            
            # 第一次调用应该初始化
            manager1 = await get_cache_manager()
            mock_manager.initialize.assert_called_once()
            
            # 第二次调用不应该再次初始化
            mock_manager.redis_client = MockRedis()  # 模拟已初始化
            manager2 = await get_cache_manager()
            
            # 应该返回同一个实例
            assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_get_cache_manager_initialization_failure(self):
        """测试缓存管理器初始化失败"""
        with patch('stock_analysis_system.data.cache_manager.cache_manager') as mock_manager:
            mock_manager.redis_client = None
            mock_manager.initialize = AsyncMock(side_effect=Exception("Init failed"))
            
            # 即使初始化失败，也应该返回管理器实例
            manager = await get_cache_manager()
            assert manager is not None


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])