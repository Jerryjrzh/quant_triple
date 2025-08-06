"""
缓存管理器演示脚本

演示缓存管理器的各种功能，包括：
1. 基本缓存操作
2. 多级缓存策略
3. 缓存预热和预加载
4. 性能监控和统计
5. 并发访问测试
"""

import asyncio
import time
import pandas as pd
from datetime import datetime, timedelta
import logging

from stock_analysis_system.data.cache_manager import CacheManager, get_cache_manager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_sample_data(symbol: str, size: int = 100) -> pd.DataFrame:
    """创建示例数据"""
    return pd.DataFrame({
        'symbol': [symbol] * size,
        'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(size)],
        'price': [100 + i * 0.1 for i in range(size)],
        'volume': [1000 + i * 10 for i in range(size)],
        'amount': [100000 + i * 1000 for i in range(size)]
    })


async def demo_basic_cache_operations():
    """演示基本缓存操作"""
    print("\n=== 基本缓存操作演示 ===")
    
    cache_manager = await get_cache_manager()
    
    # 创建测试数据
    test_data = await create_sample_data("000001", 50)
    print(f"创建测试数据: {len(test_data)} 行")
    
    # 设置缓存
    key = "demo_basic_000001"
    cache_type = "realtime_data"
    
    start_time = time.time()
    await cache_manager.set_cached_data(key, test_data, cache_type)
    set_time = time.time() - start_time
    print(f"设置缓存耗时: {set_time:.4f} 秒")
    
    # 获取缓存（内存缓存命中）
    start_time = time.time()
    cached_data = await cache_manager.get_cached_data(key, cache_type)
    get_time = time.time() - start_time
    print(f"获取缓存耗时（内存）: {get_time:.4f} 秒")
    print(f"缓存数据行数: {len(cached_data) if cached_data is not None else 0}")
    
    # 清空内存缓存，测试Redis缓存
    cache_manager.memory_cache.clear()
    
    start_time = time.time()
    cached_data = await cache_manager.get_cached_data(key, cache_type)
    get_time = time.time() - start_time
    print(f"获取缓存耗时（Redis）: {get_time:.4f} 秒")
    print(f"缓存数据行数: {len(cached_data) if cached_data is not None else 0}")
    
    # 显示统计信息
    stats = cache_manager.get_cache_stats()
    print(f"缓存统计: 命中={stats['hits']}, 未命中={stats['misses']}, 命中率={stats['hit_rate']:.2%}")


async def demo_cache_expiration():
    """演示缓存过期机制"""
    print("\n=== 缓存过期机制演示 ===")
    
    cache_manager = await get_cache_manager()
    test_data = await create_sample_data("000002", 30)
    
    key = "demo_expiration_000002"
    cache_type = "realtime_data"
    
    # 设置短TTL缓存
    await cache_manager.set_cached_data(key, test_data, cache_type, ttl=3)
    print("设置3秒TTL的缓存")
    
    # 立即获取
    cached_data = await cache_manager.get_cached_data(key, cache_type)
    print(f"立即获取: {'成功' if cached_data is not None else '失败'}")
    
    # 等待2秒
    print("等待2秒...")
    await asyncio.sleep(2)
    cached_data = await cache_manager.get_cached_data(key, cache_type)
    print(f"2秒后获取: {'成功' if cached_data is not None else '失败'}")
    
    # 等待2秒（总共4秒，超过TTL）
    print("再等待2秒...")
    await asyncio.sleep(2)
    cached_data = await cache_manager.get_cached_data(key, cache_type)
    print(f"4秒后获取: {'成功' if cached_data is not None else '失败'}")


async def demo_cache_invalidation():
    """演示缓存失效"""
    print("\n=== 缓存失效演示 ===")
    
    cache_manager = await get_cache_manager()
    
    # 设置多个缓存
    symbols = ["000001", "000002", "000003", "600001"]
    cache_type = "daily_data"
    
    for symbol in symbols:
        test_data = await create_sample_data(symbol, 20)
        key = f"demo_invalid_{symbol}"
        await cache_manager.set_cached_data(key, test_data, cache_type)
    
    print(f"设置了 {len(symbols)} 个缓存")
    
    # 验证缓存存在
    for symbol in symbols:
        key = f"demo_invalid_{symbol}"
        cached_data = await cache_manager.get_cached_data(key, cache_type)
        print(f"缓存 {key}: {'存在' if cached_data is not None else '不存在'}")
    
    # 使用模式失效部分缓存
    print("\n使用模式 'demo_invalid_0000*' 失效缓存")
    await cache_manager.invalidate_cache("demo_invalid_0000*")
    
    # 再次验证缓存
    for symbol in symbols:
        key = f"demo_invalid_{symbol}"
        cached_data = await cache_manager.get_cached_data(key, cache_type)
        print(f"缓存 {key}: {'存在' if cached_data is not None else '不存在'}")


async def demo_concurrent_access():
    """演示并发访问"""
    print("\n=== 并发访问演示 ===")
    
    cache_manager = await get_cache_manager()
    
    async def concurrent_operation(task_id: int):
        """并发操作任务"""
        symbol = f"TASK{task_id:03d}"
        test_data = await create_sample_data(symbol, 10)
        key = f"demo_concurrent_{symbol}"
        cache_type = "realtime_data"
        
        # 设置缓存
        await cache_manager.set_cached_data(key, test_data, cache_type)
        
        # 多次获取缓存
        for _ in range(5):
            cached_data = await cache_manager.get_cached_data(key, cache_type)
            if cached_data is None:
                print(f"任务 {task_id}: 缓存获取失败")
                return False
        
        return True
    
    # 创建并发任务
    tasks = []
    task_count = 20
    
    start_time = time.time()
    for i in range(task_count):
        task = asyncio.create_task(concurrent_operation(i))
        tasks.append(task)
    
    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    success_count = sum(results)
    print(f"并发任务: {task_count} 个")
    print(f"成功任务: {success_count} 个")
    print(f"总耗时: {end_time - start_time:.4f} 秒")
    print(f"平均耗时: {(end_time - start_time) / task_count:.4f} 秒/任务")
    
    # 显示最终统计
    stats = cache_manager.get_cache_stats()
    print(f"最终统计: 命中={stats['hits']}, 未命中={stats['misses']}, 命中率={stats['hit_rate']:.2%}")


async def demo_cache_warm_up():
    """演示缓存预热"""
    print("\n=== 缓存预热演示 ===")
    
    cache_manager = await get_cache_manager()
    
    # 定义数据加载函数
    async def load_realtime_data():
        """加载实时数据"""
        print("加载实时数据...")
        await asyncio.sleep(0.1)  # 模拟网络延迟
        return await create_sample_data("RT_DATA", 100)
    
    async def load_market_overview():
        """加载市场概览"""
        print("加载市场概览...")
        await asyncio.sleep(0.2)  # 模拟网络延迟
        return await create_sample_data("MARKET", 50)
    
    async def load_dragon_tiger_data():
        """加载龙虎榜数据"""
        print("加载龙虎榜数据...")
        await asyncio.sleep(0.15)  # 模拟网络延迟
        return await create_sample_data("DRAGON_TIGER", 30)
    
    # 配置预热任务
    warm_up_configs = [
        {
            'cache_type': 'realtime_data',
            'data_loader': load_realtime_data,
            'params': {}
        },
        {
            'cache_type': 'market_overview',
            'data_loader': load_market_overview,
            'params': {}
        },
        {
            'cache_type': 'dragon_tiger',
            'data_loader': load_dragon_tiger_data,
            'params': {}
        }
    ]
    
    # 执行预热
    start_time = time.time()
    await cache_manager.warm_up_cache(warm_up_configs)
    warm_up_time = time.time() - start_time
    
    print(f"缓存预热完成，耗时: {warm_up_time:.4f} 秒")
    
    # 显示预热后的统计
    stats = cache_manager.get_cache_stats()
    print(f"预热后统计: 设置={stats['sets']}, 内存缓存大小={stats['memory_cache_size']}")


async def demo_performance_comparison():
    """演示性能对比"""
    print("\n=== 性能对比演示 ===")
    
    cache_manager = await get_cache_manager()
    
    # 创建大量测试数据
    large_data = await create_sample_data("PERF_TEST", 1000)
    key = "demo_performance_large"
    cache_type = "daily_data"
    
    # 测试设置性能
    start_time = time.time()
    await cache_manager.set_cached_data(key, large_data, cache_type)
    set_time = time.time() - start_time
    print(f"设置大数据缓存耗时: {set_time:.4f} 秒")
    
    # 测试内存缓存获取性能
    times = []
    for i in range(10):
        start_time = time.time()
        cached_data = await cache_manager.get_cached_data(key, cache_type)
        get_time = time.time() - start_time
        times.append(get_time)
    
    avg_memory_time = sum(times) / len(times)
    print(f"内存缓存平均获取时间: {avg_memory_time:.6f} 秒")
    
    # 清空内存缓存，测试Redis性能
    cache_manager.memory_cache.clear()
    
    times = []
    for i in range(10):
        start_time = time.time()
        cached_data = await cache_manager.get_cached_data(key, cache_type)
        get_time = time.time() - start_time
        times.append(get_time)
        
        # 清空内存缓存以确保每次都从Redis获取
        cache_manager.memory_cache.clear()
    
    avg_redis_time = sum(times) / len(times)
    print(f"Redis缓存平均获取时间: {avg_redis_time:.6f} 秒")
    print(f"性能比较: Redis比内存慢 {avg_redis_time / avg_memory_time:.1f} 倍")


async def demo_error_handling():
    """演示错误处理"""
    print("\n=== 错误处理演示 ===")
    
    # 创建一个使用无效Redis URL的缓存管理器
    error_manager = CacheManager("redis://invalid_host:6379/0")
    
    try:
        await error_manager.initialize()
        print("连接无效Redis: 应该失败但被处理")
    except Exception as e:
        print(f"连接无效Redis失败: {e}")
    
    # 测试在Redis不可用时的降级行为
    test_data = await create_sample_data("ERROR_TEST", 10)
    key = "demo_error_test"
    cache_type = "realtime_data"
    
    # 设置缓存（应该只使用内存缓存）
    await error_manager.set_cached_data(key, test_data, cache_type)
    
    # 获取缓存（应该从内存缓存获取）
    cached_data = await error_manager.get_cached_data(key, cache_type)
    print(f"Redis不可用时缓存操作: {'成功' if cached_data is not None else '失败'}")
    
    # 显示错误统计
    stats = error_manager.get_cache_stats()
    print(f"错误统计: 错误数={stats['errors']}")
    
    await error_manager.close()


async def main():
    """主演示函数"""
    print("缓存管理器功能演示")
    print("=" * 50)
    
    try:
        # 基本操作演示
        await demo_basic_cache_operations()
        
        # 缓存过期演示
        await demo_cache_expiration()
        
        # 缓存失效演示
        await demo_cache_invalidation()
        
        # 并发访问演示
        await demo_concurrent_access()
        
        # 缓存预热演示
        await demo_cache_warm_up()
        
        # 性能对比演示
        await demo_performance_comparison()
        
        # 错误处理演示
        await demo_error_handling()
        
        print("\n=== 演示完成 ===")
        
        # 显示最终统计
        cache_manager = await get_cache_manager()
        final_stats = cache_manager.get_cache_stats()
        print("\n最终缓存统计:")
        print(f"  命中次数: {final_stats['hits']}")
        print(f"  未命中次数: {final_stats['misses']}")
        print(f"  命中率: {final_stats['hit_rate']:.2%}")
        print(f"  设置次数: {final_stats['sets']}")
        print(f"  删除次数: {final_stats['deletes']}")
        print(f"  错误次数: {final_stats['errors']}")
        print(f"  内存缓存大小: {final_stats['memory_cache_size']}")
        
        # 清理
        await cache_manager.close()
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())