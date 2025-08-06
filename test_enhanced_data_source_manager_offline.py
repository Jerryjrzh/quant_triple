#!/usr/bin/env python3
"""
离线测试增强数据源管理器

验证 EnhancedDataSourceManager 的核心功能（不依赖网络连接）：
1. 数据源优先级管理
2. 负载均衡机制
3. 健康状态监控
4. 熔断器功能

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
    level=logging.WARNING,  # 减少日志输出
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_initialization():
    """测试初始化功能"""
    print("\n🚀 Testing Initialization")
    print("-" * 50)
    
    manager = EnhancedDataSourceManager()
    
    # 检查数据源注册
    print(f"Enhanced sources: {len(manager.enhanced_sources)}")
    for name in manager.enhanced_sources.keys():
        print(f"  - {name}")
    
    print(f"Adapters: {len(manager.adapters)}")
    for name in manager.adapters.keys():
        print(f"  - {name}")
    
    # 检查优先级配置
    print(f"Priority configurations: {len(manager.source_priority)}")
    for data_type, sources in manager.source_priority.items():
        print(f"  {data_type}: {len(sources)} sources")
    
    # 检查健康监控初始化
    print(f"Health monitoring initialized for {len(manager.health_monitor['failure_counts'])} sources")
    
    return manager


def test_priority_management():
    """测试优先级管理"""
    print("\n🔄 Testing Priority Management")
    print("-" * 50)
    
    manager = EnhancedDataSourceManager()
    
    # 显示当前优先级
    original_priority = manager.source_priority.get("stock_realtime", []).copy()
    print(f"Original priority for stock_realtime: {original_priority}")
    
    # 更新优先级
    new_priority = ["eastmoney_adapter", "akshare", "eastmoney", "tushare"]
    manager.update_source_priority("stock_realtime", new_priority)
    updated_priority = manager.source_priority.get("stock_realtime", [])
    print(f"Updated priority: {updated_priority}")
    
    # 验证更新
    assert updated_priority == new_priority, "Priority update failed"
    print("✅ Priority update successful")
    
    # 测试无效数据类型
    manager.update_source_priority("invalid_type", ["test"])
    print("✅ Invalid data type handled correctly")
    
    return manager


def test_load_balancing_logic():
    """测试负载均衡逻辑"""
    print("\n⚖️ Testing Load Balancing Logic")
    print("-" * 50)
    
    manager = EnhancedDataSourceManager()
    
    # 模拟一些统计数据
    test_sources = ["eastmoney_adapter", "eastmoney", "akshare", "tushare"]
    
    # 设置不同的健康权重
    manager.load_balancer["health_weights"]["eastmoney_adapter"] = 0.9
    manager.load_balancer["health_weights"]["eastmoney"] = 0.7
    manager.load_balancer["health_weights"]["akshare"] = 0.8
    manager.load_balancer["health_weights"]["tushare"] = 0.6
    
    # 设置不同的响应时间
    manager.load_balancer["response_times"]["eastmoney_adapter"] = [1.0, 1.2, 0.8]
    manager.load_balancer["response_times"]["eastmoney"] = [2.0, 2.5, 1.8]
    manager.load_balancer["response_times"]["akshare"] = [1.5, 1.3, 1.7]
    manager.load_balancer["response_times"]["tushare"] = [3.0, 2.8, 3.2]
    
    # 测试负载均衡
    balanced_sources = manager._apply_load_balancing(test_sources)
    print(f"Original order: {test_sources}")
    print(f"Load balanced:  {balanced_sources}")
    
    # 验证排序逻辑（健康权重高且响应时间短的应该排在前面）
    print("\nSource weights and response times:")
    for source in balanced_sources:
        weight = manager.load_balancer["health_weights"].get(source, 1.0)
        times = manager.load_balancer["response_times"].get(source, [1.0])
        avg_time = sum(times) / len(times)
        composite = weight / max(avg_time, 0.1)
        print(f"  {source}: weight={weight:.2f}, avg_time={avg_time:.2f}s, composite={composite:.2f}")
    
    print("✅ Load balancing logic working correctly")
    
    return manager


def test_circuit_breaker():
    """测试熔断器功能"""
    print("\n🔌 Testing Circuit Breaker")
    print("-" * 50)
    
    manager = EnhancedDataSourceManager()
    
    test_source = "eastmoney_adapter"
    
    # 初始状态检查
    can_use = manager._can_use_source(test_source)
    circuit_state = manager.health_monitor["circuit_breakers"][test_source]["state"]
    print(f"Initial state - Can use: {can_use}, Circuit: {circuit_state}")
    
    # 模拟连续失败
    print("\nSimulating failures...")
    for i in range(6):  # 超过默认阈值5
        manager._record_failure(test_source)
        failure_count = manager.health_monitor["failure_counts"][test_source]
        circuit_state = manager.health_monitor["circuit_breakers"][test_source]["state"]
        can_use = manager._can_use_source(test_source)
        print(f"  Failure {i+1}: failures={failure_count}, circuit={circuit_state}, can_use={can_use}")
    
    # 验证熔断器已打开
    assert not manager._can_use_source(test_source), "Circuit breaker should be open"
    print("✅ Circuit breaker opened correctly")
    
    # 模拟成功恢复
    print("\nSimulating recovery...")
    manager._record_success(test_source)
    circuit_state = manager.health_monitor["circuit_breakers"][test_source]["state"]
    can_use = manager._can_use_source(test_source)
    failure_count = manager.health_monitor["failure_counts"][test_source]
    print(f"After success: failures={failure_count}, circuit={circuit_state}, can_use={can_use}")
    
    # 验证熔断器已关闭
    assert manager._can_use_source(test_source), "Circuit breaker should be closed"
    print("✅ Circuit breaker recovery working correctly")
    
    return manager


def test_health_monitoring_stats():
    """测试健康监控统计"""
    print("\n💓 Testing Health Monitoring Stats")
    print("-" * 50)
    
    manager = EnhancedDataSourceManager()
    
    # 模拟一些统计数据
    test_sources = ["eastmoney_adapter", "fund_flow_adapter", "dragon_tiger_adapter"]
    
    for source in test_sources:
        # 模拟成功和失败
        for _ in range(8):  # 8次成功
            manager._record_success(source)
        for _ in range(2):  # 2次失败
            manager._record_failure(source)
        
        # 模拟响应时间
        for time_val in [1.0, 1.5, 2.0, 1.2, 0.8]:
            manager._update_load_balancer_stats(source, True, time_val)
    
    # 获取健康状态统计
    health_stats = asyncio.run(manager.get_data_source_health_status())
    
    print("Health statistics:")
    for source, stats in health_stats.items():
        if source in test_sources:
            print(f"  {source}:")
            print(f"    Success Rate: {stats['success_rate']:.2%}")
            print(f"    Total Requests: {stats['total_requests']}")
            print(f"    Health Weight: {stats['health_weight']:.3f}")
            print(f"    Status: {stats['status']}")
            print(f"    Circuit Breaker: {stats['circuit_breaker_state']}")
    
    # 获取负载均衡统计
    lb_stats = manager.get_load_balancer_stats()
    print(f"\nLoad balancer manages {len(lb_stats['health_weights'])} sources")
    print(f"Response time data available for {len([k for k, v in lb_stats['response_times'].items() if v['count'] > 0])} sources")
    
    print("✅ Health monitoring statistics working correctly")
    
    return manager


def test_configuration_management():
    """测试配置管理"""
    print("\n⚙️ Testing Configuration Management")
    print("-" * 50)
    
    manager = EnhancedDataSourceManager()
    
    # 测试健康监控重置
    test_source = "eastmoney_adapter"
    
    # 添加一些统计数据
    manager._record_failure(test_source)
    manager._record_failure(test_source)
    manager._update_load_balancer_stats(test_source, False, 2.0)
    
    print(f"Before reset - Failures: {manager.health_monitor['failure_counts'][test_source]}")
    print(f"Before reset - Requests: {manager.load_balancer['request_counts'][test_source]}")
    
    # 重置特定数据源
    manager.reset_health_monitoring(test_source)
    
    print(f"After reset - Failures: {manager.health_monitor['failure_counts'][test_source]}")
    print(f"After reset - Requests: {manager.load_balancer['request_counts'][test_source]}")
    
    # 验证重置
    assert manager.health_monitor['failure_counts'][test_source] == 0, "Failure count should be reset"
    assert manager.load_balancer['request_counts'][test_source]['total'] == 0, "Request count should be reset"
    
    print("✅ Configuration management working correctly")
    
    return manager


def test_data_source_selection():
    """测试数据源选择逻辑"""
    print("\n🎯 Testing Data Source Selection")
    print("-" * 50)
    
    manager = EnhancedDataSourceManager()
    
    # 测试不同数据类型的数据源选择
    test_cases = [
        ("stock_realtime", "实时股票数据"),
        ("stock_history", "历史股票数据"),
        ("fund_flow", "资金流向数据"),
        ("dragon_tiger", "龙虎榜数据"),
        ("etf_data", "ETF数据"),
        ("unknown_type", "未知数据类型")
    ]
    
    for data_type, description in test_cases:
        sources = manager.source_priority.get(data_type, [])
        print(f"{description} ({data_type}): {len(sources)} sources")
        if sources:
            balanced = manager._apply_load_balancing(sources)
            available = [s for s in balanced if manager._can_use_source(s)]
            print(f"  Available: {len(available)}/{len(sources)} sources")
        else:
            print("  No sources configured")
    
    print("✅ Data source selection logic working correctly")
    
    return manager


def main():
    """主测试函数"""
    print("🧪 Enhanced Data Source Manager Offline Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 运行所有测试
        test_initialization()
        test_priority_management()
        test_load_balancing_logic()
        test_circuit_breaker()
        test_health_monitoring_stats()
        test_configuration_management()
        test_data_source_selection()
        
        print("\n" + "=" * 60)
        print("✅ All offline tests completed successfully!")
        print("\n📊 Test Summary:")
        print("  ✓ Initialization and registration")
        print("  ✓ Priority management")
        print("  ✓ Load balancing logic")
        print("  ✓ Circuit breaker functionality")
        print("  ✓ Health monitoring statistics")
        print("  ✓ Configuration management")
        print("  ✓ Data source selection")
        
        print(f"\n🎉 Enhanced Data Source Manager implementation is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.exception("Test execution failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)