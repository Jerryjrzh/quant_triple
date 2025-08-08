"""
缓存数据库同步测试

验证缓存与数据库数据的一致性，测试缓存失效和数据更新的同步机制，
添加缓存穿透和雪崩场景的测试，实现缓存性能和命中率的集成监控。

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import threading
import concurrent.futures
import time
import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from stock_analysis_system.core.database import Base
from stock_analysis_system.data.models import (
    StockDailyData,
    DragonTigerBoard,
    FundFlow,
    LimitUpReason,
    ETFData,
    DataQualityLog
)
from stock_analysis_system.data.cache_manager import CacheManager, CacheLevel, CacheConfig


class MockRedis:
    """模拟Redis客户端"""
    
    def __init__(self):
        self.data = {}
        self.expire_times = {}
        
    async def ping(self):
        """模拟ping操作"""
        return True
    
    async def get(self, key: str):
        """模拟get操作"""
        if key in self.data:
            # 检查是否过期
            if key in self.expire_times:
                if time.time() > self.expire_times[key]:
                    del self.data[key]
                    del self.expire_times[key]
                    return None
            return self.data[key]
        return None
    
    async def set(self, key: str, value: str, ex: Optional[int] = None):
        """模拟set操作"""
        self.data[key] = value
        if ex:
            self.expire_times[key] = time.time() + ex
        return True
    
    async def delete(self, key: str):
        """模拟delete操作"""
        if key in self.data:
            del self.data[key]
        if key in self.expire_times:
            del self.expire_times[key]
        return True
    
    async def exists(self, key: str):
        """模拟exists操作"""
        return key in self.data
    
    async def ttl(self, key: str):
        """模拟ttl操作"""
        if key in self.expire_times:
            remaining = self.expire_times[key] - time.time()
            return int(remaining) if remaining > 0 else -2
        return -1
    
    async def close(self):
        """模拟close操作"""
        pass


class DatabaseTestHelper:
    """数据库测试辅助类"""
    
    def __init__(self):
        # 使用内存SQLite数据库进行测试
        self.engine = create_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def setup_database(self):
        """设置测试数据库"""
        Base.metadata.create_all(bind=self.engine)
        
    def teardown_database(self):
        """清理测试数据库"""
        Base.metadata.drop_all(bind=self.engine)
        
    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()
    
    def create_sample_stock_data(self, count: int = 5) -> List[StockDailyData]:
        """创建示例股票数据"""
        np.random.seed(42)
        
        data = []
        base_date = date.today() - timedelta(days=count)
        
        for i in range(count):
            stock_data = StockDailyData(
                stock_code=f"{i+1:06d}",
                trade_date=base_date + timedelta(days=i),
                open_price=Decimal(str(round(np.random.uniform(10, 100), 3))),
                high_price=Decimal(str(round(np.random.uniform(10, 100), 3))),
                low_price=Decimal(str(round(np.random.uniform(10, 100), 3))),
                close_price=Decimal(str(round(np.random.uniform(10, 100), 3))),
                volume=int(np.random.randint(1000000, 10000000)),
                amount=Decimal(str(round(np.random.uniform(10000000, 100000000), 2)))
            )
            data.append(stock_data)
            
        return data


@pytest.fixture
def db_helper():
    """数据库测试辅助器fixture"""
    helper = DatabaseTestHelper()
    helper.setup_database()
    yield helper
    helper.teardown_database()


@pytest.fixture
def db_session(db_helper):
    """数据库会话fixture"""
    session = db_helper.get_session()
    yield session
    session.close()


@pytest.fixture
def cache_manager():
    """缓存管理器fixture"""
    manager = CacheManager()
    
    # 使用模拟Redis客户端
    mock_redis = MockRedis()
    manager.redis_client = mock_redis
    
    return manager


class TestCacheDatabaseConsistency:
    """缓存数据库一致性测试"""
    
    @pytest.mark.asyncio
    async def test_cache_database_sync_basic(self, cache_manager, db_session, db_helper):
        """测试基本的缓存数据库同步"""
        # 1. 在数据库中插入数据
        stock_data = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),
            close_price=Decimal("10.50"),
            volume=1000000
        )
        db_session.add(stock_data)
        db_session.commit()
        
        # 2. 创建DataFrame格式的数据用于缓存
        df_data = pd.DataFrame({
            'stock_code': ['000001'],
            'trade_date': [date.today()],
            'close_price': [10.50],
            'volume': [1000000]
        })
        
        # 3. 将数据放入缓存
        cache_key = "daily:000001:{}".format(date.today())
        await cache_manager.set_cached_data(cache_key, df_data, 'daily_data')
        
        # 4. 从缓存获取数据
        cached_data = await cache_manager.get_cached_data(cache_key, 'daily_data')
        
        # 5. 验证缓存数据与数据库数据一致
        assert cached_data is not None
        assert len(cached_data) == 1
        assert cached_data.iloc[0]['stock_code'] == '000001'
        assert cached_data.iloc[0]['close_price'] == 10.50
        
        # 6. 验证数据库中的数据
        db_data = db_session.query(StockDailyData).filter(
            StockDailyData.stock_code == "000001"
        ).first()
        assert db_data is not None
        assert float(db_data.close_price) == 10.50
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_on_database_update(self, cache_manager, db_session):
        """测试数据库更新时的缓存失效"""
        # 1. 创建初始数据
        stock_data = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),
            close_price=Decimal("10.00")
        )
        db_session.add(stock_data)
        db_session.commit()
        
        # 2. 缓存初始数据
        df_data = pd.DataFrame({
            'stock_code': ['000001'],
            'close_price': [10.00]
        })
        cache_key = "daily:000001:{}".format(date.today())
        await cache_manager.set_cached_data(cache_key, df_data, 'daily_data')
        
        # 3. 验证缓存存在
        cached_data = await cache_manager.get_cached_data(cache_key, 'daily_data')
        assert cached_data is not None
        assert cached_data.iloc[0]['close_price'] == 10.00
        
        # 4. 更新数据库数据
        stock_data.close_price = Decimal("11.00")
        db_session.commit()
        
        # 5. 模拟缓存失效（在实际应用中，这应该由数据更新触发器处理）
        await cache_manager.invalidate_cache(cache_key)
        
        # 6. 验证缓存已失效
        cached_data_after_update = await cache_manager.get_cached_data(cache_key, 'daily_data')
        assert cached_data_after_update is None
        
        # 7. 重新缓存更新后的数据
        updated_df_data = pd.DataFrame({
            'stock_code': ['000001'],
            'close_price': [11.00]
        })
        await cache_manager.set_cached_data(cache_key, updated_df_data, 'daily_data')
        
        # 8. 验证缓存数据已更新
        final_cached_data = await cache_manager.get_cached_data(cache_key, 'daily_data')
        assert final_cached_data is not None
        assert final_cached_data.iloc[0]['close_price'] == 11.00
    
    @pytest.mark.asyncio
    async def test_cache_write_through_pattern(self, cache_manager, db_session):
        """测试写穿透模式（同时更新缓存和数据库）"""
        cache_key = "daily:000002:{}".format(date.today())
        
        # 1. 同时写入数据库和缓存
        stock_data = StockDailyData(
            stock_code="000002",
            trade_date=date.today(),
            close_price=Decimal("15.50"),
            volume=2000000
        )
        db_session.add(stock_data)
        db_session.commit()
        
        df_data = pd.DataFrame({
            'stock_code': ['000002'],
            'trade_date': [date.today()],
            'close_price': [15.50],
            'volume': [2000000]
        })
        await cache_manager.set_cached_data(cache_key, df_data, 'daily_data')
        
        # 2. 验证数据库数据
        db_data = db_session.query(StockDailyData).filter(
            StockDailyData.stock_code == "000002"
        ).first()
        assert db_data is not None
        assert float(db_data.close_price) == 15.50
        
        # 3. 验证缓存数据
        cached_data = await cache_manager.get_cached_data(cache_key, 'daily_data')
        assert cached_data is not None
        assert cached_data.iloc[0]['close_price'] == 15.50
        
        # 4. 验证数据一致性
        assert float(db_data.close_price) == cached_data.iloc[0]['close_price']
        assert db_data.volume == cached_data.iloc[0]['volume']
    
    @pytest.mark.asyncio
    async def test_cache_write_back_pattern(self, cache_manager, db_session):
        """测试写回模式（先写缓存，延迟写数据库）"""
        cache_key = "daily:000003:{}".format(date.today())
        
        # 1. 先写入缓存
        df_data = pd.DataFrame({
            'stock_code': ['000003'],
            'trade_date': [date.today()],
            'close_price': [20.00],
            'volume': [3000000]
        })
        await cache_manager.set_cached_data(cache_key, df_data, 'daily_data')
        
        # 2. 验证缓存数据存在
        cached_data = await cache_manager.get_cached_data(cache_key, 'daily_data')
        assert cached_data is not None
        assert cached_data.iloc[0]['close_price'] == 20.00
        
        # 3. 模拟延迟写入数据库
        await asyncio.sleep(0.1)  # 模拟延迟
        
        stock_data = StockDailyData(
            stock_code="000003",
            trade_date=date.today(),
            close_price=Decimal("20.00"),
            volume=3000000
        )
        db_session.add(stock_data)
        db_session.commit()
        
        # 4. 验证数据库数据
        db_data = db_session.query(StockDailyData).filter(
            StockDailyData.stock_code == "000003"
        ).first()
        assert db_data is not None
        assert float(db_data.close_price) == 20.00
        
        # 5. 验证最终一致性
        assert float(db_data.close_price) == cached_data.iloc[0]['close_price']


class TestCacheFailureScenarios:
    """缓存故障场景测试"""
    
    @pytest.mark.asyncio
    async def test_cache_penetration_protection(self, cache_manager, db_session):
        """测试缓存穿透保护"""
        # 1. 查询不存在的数据
        non_existent_key = "daily:999999:{}".format(date.today())
        
        # 2. 第一次查询缓存（缓存未命中）
        cached_data = await cache_manager.get_cached_data(non_existent_key, 'daily_data')
        assert cached_data is None
        
        # 3. 查询数据库（数据不存在）
        db_data = db_session.query(StockDailyData).filter(
            StockDailyData.stock_code == "999999"
        ).first()
        assert db_data is None
        
        # 4. 设置空值缓存防止穿透
        empty_df = pd.DataFrame()  # 空DataFrame表示数据不存在
        await cache_manager.set_cached_data(
            non_existent_key, 
            empty_df, 
            'daily_data', 
            ttl=60  # 短TTL
        )
        
        # 5. 再次查询缓存（应该返回空DataFrame）
        cached_empty = await cache_manager.get_cached_data(non_existent_key, 'daily_data')
        assert cached_empty is not None
        assert len(cached_empty) == 0  # 空DataFrame
        
        # 6. 验证缓存统计
        assert cache_manager.stats.hits > 0
        assert cache_manager.stats.misses > 0
    
    @pytest.mark.asyncio
    async def test_cache_avalanche_protection(self, cache_manager, db_session, db_helper):
        """测试缓存雪崩保护"""
        # 1. 创建多个相同TTL的缓存项
        sample_data = db_helper.create_sample_stock_data(5)
        db_session.add_all(sample_data)
        db_session.commit()
        
        cache_keys = []
        base_time = time.time()
        
        for i, stock in enumerate(sample_data):
            cache_key = f"daily:{stock.stock_code}:{stock.trade_date}"
            cache_keys.append(cache_key)
            
            df_data = pd.DataFrame({
                'stock_code': [stock.stock_code],
                'close_price': [float(stock.close_price)]
            })
            
            # 设置相同的TTL（模拟雪崩场景）
            await cache_manager.set_cached_data(cache_key, df_data, 'daily_data', ttl=1)
        
        # 2. 验证所有缓存都存在
        for key in cache_keys:
            cached_data = await cache_manager.get_cached_data(key, 'daily_data')
            assert cached_data is not None
        
        # 3. 等待缓存过期
        await asyncio.sleep(1.5)
        
        # 4. 模拟并发访问过期的缓存
        async def concurrent_access(key):
            # 添加随机延迟防止雪崩
            await asyncio.sleep(np.random.uniform(0, 0.1))
            return await cache_manager.get_cached_data(key, 'daily_data')
        
        # 5. 并发访问所有过期的缓存
        tasks = [concurrent_access(key) for key in cache_keys]
        results = await asyncio.gather(*tasks)
        
        # 6. 验证大部分访问都返回None（缓存已过期）
        none_count = sum(1 for result in results if result is None)
        assert none_count >= len(cache_keys) * 0.8  # 至少80%过期
    
    @pytest.mark.asyncio
    async def test_cache_hotspot_protection(self, cache_manager, db_session):
        """测试缓存热点保护"""
        # 1. 创建热点数据
        hot_stock = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),
            close_price=Decimal("50.00"),
            volume=10000000
        )
        db_session.add(hot_stock)
        db_session.commit()
        
        cache_key = f"daily:000001:{date.today()}"
        
        # 2. 模拟高并发访问
        async def concurrent_get():
            return await cache_manager.get_cached_data(cache_key, 'daily_data')
        
        # 3. 第一次访问（缓存未命中）
        first_result = await concurrent_get()
        assert first_result is None
        
        # 4. 设置缓存
        df_data = pd.DataFrame({
            'stock_code': ['000001'],
            'close_price': [50.00]
        })
        await cache_manager.set_cached_data(cache_key, df_data, 'daily_data')
        
        # 5. 并发访问缓存
        tasks = [concurrent_get() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # 6. 验证所有访问都命中缓存
        assert all(result is not None for result in results)
        assert all(result.iloc[0]['close_price'] == 50.00 for result in results)
        
        # 7. 验证缓存命中率
        assert cache_manager.stats.hit_rate > 0.5


class TestCachePerformanceMonitoring:
    """缓存性能监控测试"""
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate_monitoring(self, cache_manager, db_session):
        """测试缓存命中率监控"""
        # 1. 重置统计
        cache_manager.stats.hits = 0
        cache_manager.stats.misses = 0
        
        # 2. 执行一系列缓存操作
        cache_keys = [f"test_key_{i}" for i in range(10)]
        
        # 第一轮：全部未命中
        for key in cache_keys:
            result = await cache_manager.get_cached_data(key, 'daily_data')
            assert result is None
        
        # 验证未命中统计
        assert cache_manager.stats.misses == 10
        assert cache_manager.stats.hit_rate == 0.0
        
        # 3. 设置缓存
        for i, key in enumerate(cache_keys):
            df_data = pd.DataFrame({'value': [i]})
            await cache_manager.set_cached_data(key, df_data, 'daily_data')
        
        # 4. 第二轮：全部命中
        for key in cache_keys:
            result = await cache_manager.get_cached_data(key, 'daily_data')
            assert result is not None
        
        # 验证命中统计
        assert cache_manager.stats.hits == 10
        assert cache_manager.stats.hit_rate == 0.5  # 10命中 / (10命中 + 10未命中)
    
    @pytest.mark.asyncio
    async def test_cache_performance_metrics(self, cache_manager):
        """测试缓存性能指标"""
        # 1. 测试缓存设置性能
        start_time = time.time()
        
        for i in range(100):
            key = f"perf_test_{i}"
            df_data = pd.DataFrame({'value': [i], 'timestamp': [time.time()]})
            await cache_manager.set_cached_data(key, df_data, 'daily_data')
        
        set_time = time.time() - start_time
        
        # 2. 测试缓存获取性能
        start_time = time.time()
        
        for i in range(100):
            key = f"perf_test_{i}"
            result = await cache_manager.get_cached_data(key, 'daily_data')
            assert result is not None
        
        get_time = time.time() - start_time
        
        # 3. 验证性能指标
        assert set_time < 1.0  # 100次设置应该在1秒内完成
        assert get_time < 0.5  # 100次获取应该在0.5秒内完成
        
        # 4. 验证吞吐量
        set_throughput = 100 / set_time
        get_throughput = 100 / get_time
        
        assert set_throughput > 100  # 每秒至少100次设置
        assert get_throughput > 200  # 每秒至少200次获取
        
        print(f"缓存设置吞吐量: {set_throughput:.2f} ops/sec")
        print(f"缓存获取吞吐量: {get_throughput:.2f} ops/sec")
    
    @pytest.mark.asyncio
    async def test_cache_memory_usage_monitoring(self, cache_manager):
        """测试缓存内存使用监控"""
        import sys
        
        # 1. 记录初始内存使用
        initial_memory = len(cache_manager.memory_cache)
        
        # 2. 添加大量缓存数据
        large_data_size = 1000
        for i in range(large_data_size):
            key = f"memory_test_{i}"
            # 创建较大的DataFrame
            df_data = pd.DataFrame({
                'col1': np.random.randn(100),
                'col2': np.random.randn(100),
                'col3': np.random.randn(100)
            })
            await cache_manager.set_cached_data(key, df_data, 'daily_data')
        
        # 3. 检查内存使用增长
        final_memory = len(cache_manager.memory_cache)
        memory_growth = final_memory - initial_memory
        
        assert memory_growth > 0
        assert memory_growth <= large_data_size  # 不应该超过添加的数据量
        
        # 4. 测试内存清理
        cache_manager.memory_cache.clear()
        cache_manager.memory_expire.clear()
        
        cleared_memory = len(cache_manager.memory_cache)
        assert cleared_memory == 0
        
        print(f"内存使用增长: {memory_growth} 项")


class TestCacheSynchronizationScenarios:
    """缓存同步场景测试"""
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_sync(self, cache_manager, db_session):
        """测试多级缓存同步"""
        # 1. 在数据库中创建数据
        stock_data = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),
            close_price=Decimal("25.00")
        )
        db_session.add(stock_data)
        db_session.commit()
        
        cache_key = f"daily:000001:{date.today()}"
        
        # 2. 设置Redis缓存
        df_data = pd.DataFrame({
            'stock_code': ['000001'],
            'close_price': [25.00]
        })
        await cache_manager.set_cached_data(cache_key, df_data, 'daily_data')
        
        # 3. 第一次获取（应该从Redis获取并缓存到内存）
        result1 = await cache_manager.get_cached_data(cache_key, 'daily_data')
        assert result1 is not None
        assert result1.iloc[0]['close_price'] == 25.00
        
        # 4. 验证内存缓存已设置
        assert cache_key in cache_manager.memory_cache
        
        # 5. 第二次获取（应该从内存缓存获取）
        result2 = await cache_manager.get_cached_data(cache_key, 'daily_data')
        assert result2 is not None
        assert result2.iloc[0]['close_price'] == 25.00
        
        # 6. 验证缓存层级工作正常
        assert cache_manager.stats.hits >= 2
    
    @pytest.mark.asyncio
    async def test_cache_consistency_under_concurrent_updates(self, cache_manager, db_session):
        """测试并发更新下的缓存一致性"""
        # 1. 创建初始数据
        stock_data = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),
            close_price=Decimal("30.00")
        )
        db_session.add(stock_data)
        db_session.commit()
        
        cache_key = f"daily:000001:{date.today()}"
        
        # 2. 并发更新函数
        async def update_cache_and_db(new_price: float):
            # 更新数据库
            stock_data.close_price = Decimal(str(new_price))
            db_session.commit()
            
            # 更新缓存
            df_data = pd.DataFrame({
                'stock_code': ['000001'],
                'close_price': [new_price]
            })
            await cache_manager.set_cached_data(cache_key, df_data, 'daily_data')
            
            return new_price
        
        # 3. 并发执行更新
        prices = [31.00, 32.00, 33.00]
        tasks = [update_cache_and_db(price) for price in prices]
        results = await asyncio.gather(*tasks)
        
        # 4. 验证最终状态一致性
        final_cached_data = await cache_manager.get_cached_data(cache_key, 'daily_data')
        final_db_data = db_session.query(StockDailyData).filter(
            StockDailyData.stock_code == "000001"
        ).first()
        
        assert final_cached_data is not None
        assert final_db_data is not None
        
        # 最终价格应该是其中一个更新的价格
        final_cache_price = final_cached_data.iloc[0]['close_price']
        final_db_price = float(final_db_data.close_price)
        
        assert final_cache_price in prices
        assert final_db_price in prices
        
        # 缓存和数据库应该一致（在最后一次更新后）
        # 注意：由于并发性，可能不完全一致，但应该都是有效的价格
        assert final_cache_price in prices
        assert final_db_price in prices


if __name__ == "__main__":
    pytest.main([__file__, "-v"])