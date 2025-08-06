"""
缓存管理系统

实现多级缓存策略，支持Redis缓存的读写和失效机制，
包括缓存预热和智能预加载功能。
"""

import asyncio
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """缓存级别枚举"""
    MEMORY = "memory"
    REDIS = "redis"
    DATABASE = "database"


@dataclass
class CacheConfig:
    """缓存配置"""
    ttl: int  # 生存时间（秒）
    key_pattern: str  # 键模式
    level: CacheLevel = CacheLevel.REDIS
    preload: bool = False  # 是否预加载
    compress: bool = False  # 是否压缩


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheManager:
    """
    缓存管理器
    
    支持多级缓存策略：
    1. 内存缓存（最快，容量小）
    2. Redis缓存（快速，容量中等）
    3. 数据库缓存（慢，容量大）
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        初始化缓存管理器
        
        Args:
            redis_url: Redis连接URL
        """
        self.redis_url = redis_url
        self.redis_client: Optional[Redis] = None
        self.memory_cache: Dict[str, Any] = {}
        self.memory_expire: Dict[str, float] = {}
        self.stats = CacheStats()
        
        # 缓存配置
        self.cache_configs = {
            'realtime_data': CacheConfig(
                ttl=60, 
                key_pattern='rt:{symbol}',
                preload=True
            ),
            'daily_data': CacheConfig(
                ttl=3600, 
                key_pattern='daily:{symbol}:{date}',
                compress=True
            ),
            'dragon_tiger': CacheConfig(
                ttl=1800, 
                key_pattern='dt:{date}',
                preload=True
            ),
            'fund_flow': CacheConfig(
                ttl=900, 
                key_pattern='ff:{symbol}:{period}'
            ),
            'limitup_reason': CacheConfig(
                ttl=1800, 
                key_pattern='lr:{date}'
            ),
            'etf_data': CacheConfig(
                ttl=300, 
                key_pattern='etf:{symbol}:{data_type}'
            ),
            'market_overview': CacheConfig(
                ttl=120, 
                key_pattern='mo:{timestamp}',
                preload=True
            )
        }
        
        # 预加载任务
        self.preload_tasks: List[asyncio.Task] = []
        
    async def initialize(self):
        """初始化缓存管理器"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Redis连接成功")
            
            # 启动预加载任务
            await self._start_preload_tasks()
            
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            self.redis_client = None
    
    async def close(self):
        """关闭缓存管理器"""
        # 停止预加载任务
        for task in self.preload_tasks:
            task.cancel()
        
        if self.redis_client:
            await self.redis_client.close()
    
    async def get_cached_data(self, key: str, cache_type: str = 'default') -> Optional[pd.DataFrame]:
        """
        获取缓存数据
        
        Args:
            key: 缓存键
            cache_type: 缓存类型
            
        Returns:
            缓存的数据，如果不存在返回None
        """
        try:
            # 1. 先检查内存缓存
            memory_data = self._get_memory_cache(key)
            if memory_data is not None:
                self.stats.hits += 1
                logger.debug(f"内存缓存命中: {key}")
                return memory_data
            
            # 2. 检查Redis缓存
            if self.redis_client:
                redis_data = await self._get_redis_cache(key, cache_type)
                if redis_data is not None:
                    # 将数据放入内存缓存
                    self._set_memory_cache(key, redis_data, cache_type)
                    self.stats.hits += 1
                    logger.debug(f"Redis缓存命中: {key}")
                    return redis_data
            
            self.stats.misses += 1
            logger.debug(f"缓存未命中: {key}")
            return None
            
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"获取缓存数据失败: {key}, 错误: {e}")
            return None
    
    async def set_cached_data(
        self, 
        key: str, 
        data: pd.DataFrame, 
        cache_type: str = 'default',
        ttl: Optional[int] = None
    ):
        """
        设置缓存数据
        
        Args:
            key: 缓存键
            data: 要缓存的数据
            cache_type: 缓存类型
            ttl: 生存时间（秒），如果为None则使用配置的TTL
        """
        try:
            config = self.cache_configs.get(cache_type, CacheConfig(ttl=300, key_pattern='{key}'))
            actual_ttl = ttl or config.ttl
            
            # 1. 设置内存缓存
            self._set_memory_cache(key, data, cache_type, actual_ttl)
            
            # 2. 设置Redis缓存
            if self.redis_client:
                await self._set_redis_cache(key, data, config, actual_ttl)
            
            self.stats.sets += 1
            logger.debug(f"缓存数据已设置: {key}, TTL: {actual_ttl}秒")
            
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"设置缓存数据失败: {key}, 错误: {e}")
    
    async def invalidate_cache(self, pattern: str):
        """
        使缓存失效
        
        Args:
            pattern: 缓存键模式，支持通配符
        """
        try:
            # 1. 清理内存缓存
            keys_to_delete = []
            for key in self.memory_cache.keys():
                if self._match_pattern(key, pattern):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self.memory_cache[key]
                if key in self.memory_expire:
                    del self.memory_expire[key]
            
            # 2. 清理Redis缓存
            if self.redis_client:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                    self.stats.deletes += len(keys)
            
            logger.info(f"缓存已失效: {pattern}")
            
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"使缓存失效失败: {pattern}, 错误: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息字典
        """
        return {
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'hit_rate': self.stats.hit_rate,
            'sets': self.stats.sets,
            'deletes': self.stats.deletes,
            'errors': self.stats.errors,
            'memory_cache_size': len(self.memory_cache),
            'total_size': self.stats.total_size
        }
    
    async def preload_cache(self, cache_type: str, data_loader_func, *args, **kwargs):
        """
        预加载缓存
        
        Args:
            cache_type: 缓存类型
            data_loader_func: 数据加载函数
            *args, **kwargs: 传递给数据加载函数的参数
        """
        try:
            config = self.cache_configs.get(cache_type)
            if not config or not config.preload:
                return
            
            logger.info(f"开始预加载缓存: {cache_type}")
            
            # 调用数据加载函数获取数据
            data = await data_loader_func(*args, **kwargs)
            
            if data is not None and not data.empty:
                # 生成缓存键
                key = self._generate_cache_key(config.key_pattern, *args, **kwargs)
                await self.set_cached_data(key, data, cache_type)
                logger.info(f"预加载缓存完成: {key}")
            
        except Exception as e:
            logger.error(f"预加载缓存失败: {cache_type}, 错误: {e}")
    
    async def warm_up_cache(self, warm_up_configs: List[Dict[str, Any]]):
        """
        缓存预热
        
        Args:
            warm_up_configs: 预热配置列表
        """
        logger.info("开始缓存预热")
        
        tasks = []
        for config in warm_up_configs:
            cache_type = config.get('cache_type')
            data_loader = config.get('data_loader')
            params = config.get('params', {})
            
            if cache_type and data_loader:
                task = asyncio.create_task(
                    self.preload_cache(cache_type, data_loader, **params)
                )
                tasks.append(task)
        
        # 并发执行预热任务
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("缓存预热完成")
    
    def _get_memory_cache(self, key: str) -> Optional[pd.DataFrame]:
        """获取内存缓存"""
        if key not in self.memory_cache:
            return None
        
        # 检查是否过期
        if key in self.memory_expire:
            if time.time() > self.memory_expire[key]:
                del self.memory_cache[key]
                del self.memory_expire[key]
                return None
        
        return self.memory_cache[key]
    
    def _set_memory_cache(
        self, 
        key: str, 
        data: pd.DataFrame, 
        cache_type: str, 
        ttl: Optional[int] = None
    ):
        """设置内存缓存"""
        config = self.cache_configs.get(cache_type, CacheConfig(ttl=300, key_pattern='{key}'))
        actual_ttl = ttl or config.ttl
        
        self.memory_cache[key] = data
        self.memory_expire[key] = time.time() + actual_ttl
        
        # 限制内存缓存大小（最多1000个条目）
        if len(self.memory_cache) > 1000:
            # 删除最旧的条目
            oldest_key = min(self.memory_expire.keys(), key=self.memory_expire.get)
            del self.memory_cache[oldest_key]
            del self.memory_expire[oldest_key]
    
    async def _get_redis_cache(self, key: str, cache_type: str) -> Optional[pd.DataFrame]:
        """获取Redis缓存"""
        try:
            config = self.cache_configs.get(cache_type, CacheConfig(ttl=300, key_pattern='{key}'))
            
            cached_data = await self.redis_client.get(key)
            if cached_data is None:
                return None
            
            # 反序列化数据
            if config.compress:
                data = pickle.loads(cached_data)
            else:
                data_dict = json.loads(cached_data)
                data = pd.DataFrame(data_dict)
            
            return data
            
        except Exception as e:
            logger.error(f"获取Redis缓存失败: {key}, 错误: {e}")
            return None
    
    async def _set_redis_cache(
        self, 
        key: str, 
        data: pd.DataFrame, 
        config: CacheConfig, 
        ttl: int
    ):
        """设置Redis缓存"""
        try:
            # 序列化数据
            if config.compress:
                serialized_data = pickle.dumps(data)
            else:
                serialized_data = json.dumps(data.to_dict())
            
            await self.redis_client.setex(key, ttl, serialized_data)
            
        except Exception as e:
            logger.error(f"设置Redis缓存失败: {key}, 错误: {e}")
    
    def _generate_cache_key(self, pattern: str, *args, **kwargs) -> str:
        """生成缓存键"""
        try:
            return pattern.format(*args, **kwargs)
        except (KeyError, IndexError):
            # 如果格式化失败，使用简单的键
            return f"{pattern}_{hash(str(args) + str(kwargs))}"
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """匹配模式"""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    async def _start_preload_tasks(self):
        """启动预加载任务"""
        for cache_type, config in self.cache_configs.items():
            if config.preload:
                task = asyncio.create_task(self._preload_task(cache_type))
                self.preload_tasks.append(task)
    
    async def _preload_task(self, cache_type: str):
        """预加载任务"""
        config = self.cache_configs[cache_type]
        
        while True:
            try:
                # 根据缓存类型执行不同的预加载逻辑
                if cache_type == 'realtime_data':
                    await self._preload_realtime_data()
                elif cache_type == 'dragon_tiger':
                    await self._preload_dragon_tiger_data()
                elif cache_type == 'market_overview':
                    await self._preload_market_overview()
                
                # 等待下次预加载
                await asyncio.sleep(config.ttl // 2)  # 在TTL的一半时间后重新加载
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"预加载任务失败: {cache_type}, 错误: {e}")
                await asyncio.sleep(60)  # 出错后等待1分钟再重试
    
    async def _preload_realtime_data(self):
        """预加载实时数据"""
        # 这里应该调用实际的数据获取函数
        # 为了演示，我们创建一个空的DataFrame
        logger.debug("预加载实时数据")
        pass
    
    async def _preload_dragon_tiger_data(self):
        """预加载龙虎榜数据"""
        logger.debug("预加载龙虎榜数据")
        pass
    
    async def _preload_market_overview(self):
        """预加载市场概览数据"""
        logger.debug("预加载市场概览数据")
        pass


# 全局缓存管理器实例
cache_manager = CacheManager()


async def get_cache_manager() -> CacheManager:
    """获取缓存管理器实例"""
    if cache_manager.redis_client is None:
        await cache_manager.initialize()
    return cache_manager