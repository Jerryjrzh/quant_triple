#!/usr/bin/env python3
"""
测试增强数据源管理器

验证 EnhancedDataSourceManager 的功能：
1. 数据源优先级管理
2. 负载均衡机制
3. 健康状态监控
4. 故障转移功能

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_analysis_system.data.enhanced_data_sources import (
    EnhancedDataSourceManager, 
    MarketDataRequest
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_data_source_priority():
    """测试数据源优先级功能"""
    print("\n🔄 Testing Data Source Priority Management")
    print("-" * 50)
    
    manager = EnhancedDataSourceManager()
    
    # 显示当前优先级配置
    print("Current priority configuration:")
    for data_type, sources in manager.source_priority.items():
        print(f"  {data_type}: {sources}")
    
    # 测试优先级更新
    print("\nTesting priority update...")
    original_priority = manager.source_priority.get("stock_realtime", []).copy()
    new_priority = ["eastmoney_adapter", "akshare", "eastmoney", "tushare"]
    manager.update_source_priority("stock_realtime", new_priority)
    
    print(f"Original: {original_priority}")
    print(f"Updated:  {manager.source_priority['stock_realtime']}")
    
    return manager


async def test_load_balancing():
    """测试负载均衡功能"""
    print("\n⚖️ Testing Load Balancing Mechanism")
    print("-" * 50)
    
    manager = EnhancedDataSourceManager()
    
    # 模拟多次请求以测试负载均衡
    requests = [
        MarketDataRequest(symbol="000001", start_date="", end_date="", data_type="stock_realtime"),
        MarketDataRequest(symbol="000002", start_date="", end_date="", data_type="stock_realtime"),
        MarketDataRequest(symbol="600000", start_date="", end_date="", data_type="stock_realtime"),
    ]
    
    print("Simulating multiple requests to test load balancing...")
    for i, request in enumerate(requests):
        print(f"\nRequest {i+1}: {request.symbol}")
        try:
            # 这里只测试负载均衡逻辑，不实际获取数据
            data_sources = manager.source_priority.get(request.data_type, [])
            selected_sources = manager._apply_load_balancing(data_sources)
            print(f"  Original order: {data_sources}")
            print(f"  Load balanced:  {selected_sources}")
            
            # 模拟更新统计
            if selected_sources:
                manager._update_load_balancer_stats(selected_sources[0], True, 1.5)
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # 显示负载均衡统计
    print("\nLoad balancer statistics:")
    lb_stats = manager.get_load_balancer_stats()
    for source, weight in lb_stats["health_weights"].items():
        print(f"  {source}: weight={weight:.3f}")
    
    return manager


async def test_health_monitoring():
    """测试健康状态监控"""
    print("\n💓 Testing Health Status Monitoring")
    print("-" * 50)
    
    manager = EnhancedDataSourceManager()
    
    # 执行健康检查
    print("Performing health check...")
    try:
        health_results = await manager.perform_health_check(force_check=True)
        
        print("\nHealth check results:")
        for source_name, health_info in health_results.items():
            status = health_info.get("status", "unknown")
            response_time = health_info.get("response_time", 0.0)
            data_available = health_info.get("data_available", False)
            
            print(f"  {source_name}:")
            print(f"    Status: {status}")
            print(f"    Response Time: {response_time:.3f}s")
            print(f"    Data Available: {data_available}")
            
            if "error" in health_info:
                print(f"    Error: {health_info['error']}")
    
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # 获取详细健康状态
    print("\nDetailed health status:")
    try:
        health_status = await manager.get_data_source_health_status()
        
        for source_name, details in health_status.items():
            print(f"  {source_name}:")
            print(f"    Success Rate: {details['success_rate']:.2%}")
            print(f"    Health Weight: {details['health_weight']:.3f}")
            print(f"    Circuit Breaker: {details['circuit_breaker_state']}")
            print(f"    Status: {details['status']}")
    
    except Exception as e:
        print(f"Failed to get health status: {e}")
    
    return manager


async def test_circuit_breaker():
    """测试熔断器功能"""
    print("\n🔌 Testing Circuit Breaker Functionality")
    print("-" * 50)
    
    manager = EnhancedDataSourceManager()
    
    # 模拟连续失败以触发熔断器
    test_source = "eastmoney_adapter"
    print(f"Simulating failures for {test_source}...")
    
    for i in range(6):  # 超过失败阈值
        manager._record_failure(test_source)
        circuit_state = manager.health_monitor["circuit_breakers"][test_source]["state"]
        failure_count = manager.health_monitor["failure_counts"][test_source]
        
        print(f"  Failure {i+1}: Circuit state = {circuit_state}, Failures = {failure_count}")
    
    # 检查是否可以使用该数据源
    can_use = manager._can_use_source(test_source)
    print(f"\nCan use {test_source}: {can_use}")
    
    # 模拟成功以恢复熔断器
    print(f"\nSimulating success to recover circuit breaker...")
    manager._record_success(test_source)
    circuit_state = manager.health_monitor["circuit_breakers"][test_source]["state"]
    print(f"After success: Circuit state = {circuit_state}")
    
    return manager


async def test_failover_mechanism():
    """测试故障转移机制"""
    print("\n🔄 Testing Failover Mechanism")
    print("-" * 50)
    
    manager = EnhancedDataSourceManager()
    
    # 创建测试请求
    request = MarketDataRequest(
        symbol="000001",
        start_date="20240101",
        end_date="20240131",
        data_type="stock_history"
    )
    
    print(f"Testing failover for request: {request.symbol} ({request.data_type})")
    
    # 显示数据源优先级
    data_sources = manager.source_priority.get(request.data_type, [])
    print(f"Available data sources: {data_sources}")
    
    # 模拟第一个数据源失败
    if data_sources:
        first_source = data_sources[0]
        print(f"\nSimulating failure of primary source: {first_source}")
        
        # 触发熔断器
        for _ in range(6):
            manager._record_failure(first_source)
        
        # 检查负载均衡后的顺序
        balanced_sources = manager._apply_load_balancing(data_sources)
        print(f"Load balanced order: {balanced_sources}")
        
        # 检查哪些源可用
        available_sources = [s for s in balanced_sources if manager._can_use_source(s)]
        print(f"Available sources after circuit breaker: {available_sources}")
    
    return manager


async def main():
    """主测试函数"""
    print("🧪 Enhanced Data Source Manager Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 运行所有测试
        await test_data_source_priority()
        await test_load_balancing()
        await test_health_monitoring()
        await test_circuit_breaker()
        await test_failover_mechanism()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.exception("Test execution failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)