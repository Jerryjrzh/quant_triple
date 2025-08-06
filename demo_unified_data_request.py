"""
统一数据请求接口演示脚本

该脚本演示如何使用统一数据请求接口获取各种市场数据：
- 股票实时行情
- 股票历史数据
- 龙虎榜数据
- 资金流向数据
- 涨停原因数据
- ETF数据

作者: Kiro
创建时间: 2024-01-01
"""

import asyncio
import json
from datetime import date, datetime
from stock_analysis_system.data.market_data_request import (
    MarketDataRequest,
    DataType,
    DataSource,
    Period,
    process_market_data_request,
    unified_request_interface
)


async def demo_stock_realtime_data():
    """演示获取股票实时行情数据"""
    print("\n=== 股票实时行情数据演示 ===")
    
    # 单个股票实时数据
    request_data = {
        'data_type': DataType.STOCK_REALTIME,
        'symbol': '000001.SZ',
        'data_source': DataSource.AUTO,
        'use_cache': True,
        'timeout': 10
    }
    
    result = await process_market_data_request(request_data)
    
    if result['success']:
        print(f"✅ 成功获取 {request_data['symbol']} 实时数据")
        print(f"数据源: {result['data_source']}")
        print(f"记录数: {result['record_count']}")
        print(f"请求ID: {result['request_id']}")
        print("数据预览:")
        print(result['data'].head())
    else:
        print(f"❌ 获取数据失败: {result['error']}")
    
    # 多个股票实时数据
    print("\n--- 多股票实时数据 ---")
    request_data = {
        'data_type': DataType.STOCK_REALTIME,
        'symbols': ['000001.SZ', '000002.SZ', '600000.SH'],
        'data_source': DataSource.AUTO,
        'fields': ['symbol', 'price', 'change', 'change_pct', 'volume']
    }
    
    result = await process_market_data_request(request_data)
    
    if result['success']:
        print(f"✅ 成功获取多股票实时数据")
        print(f"记录数: {result['record_count']}")
        print("数据预览:")
        print(result['data'])
    else:
        print(f"❌ 获取数据失败: {result['error']}")


async def demo_stock_history_data():
    """演示获取股票历史数据"""
    print("\n=== 股票历史数据演示 ===")
    
    request_data = {
        'data_type': DataType.STOCK_HISTORY,
        'symbol': '000001.SZ',
        'start_date': '2024-01-01',
        'end_date': '2024-01-31',
        'period': Period.DAILY,
        'data_source': DataSource.EASTMONEY,
        'limit': 100
    }
    
    result = await process_market_data_request(request_data)
    
    if result['success']:
        print(f"✅ 成功获取 {request_data['symbol']} 历史数据")
        print(f"时间范围: {request_data['start_date']} 到 {request_data['end_date']}")
        print(f"数据源: {result['data_source']}")
        print(f"记录数: {result['record_count']}")
        print("数据预览:")
        print(result['data'].head())
    else:
        print(f"❌ 获取数据失败: {result['error']}")


async def demo_dragon_tiger_data():
    """演示获取龙虎榜数据"""
    print("\n=== 龙虎榜数据演示 ===")
    
    request_data = {
        'data_type': DataType.DRAGON_TIGER,
        'trade_date': '2024-01-15',
        'data_source': DataSource.EASTMONEY,
        'limit': 50,
        'sort_by': 'net_buy_amount',
        'sort_order': 'desc'
    }
    
    result = await process_market_data_request(request_data)
    
    if result['success']:
        print(f"✅ 成功获取龙虎榜数据")
        print(f"交易日期: {request_data['trade_date']}")
        print(f"数据源: {result['data_source']}")
        print(f"记录数: {result['record_count']}")
        print("数据预览:")
        print(result['data'].head())
    else:
        print(f"❌ 获取数据失败: {result['error']}")


async def demo_fund_flow_data():
    """演示获取资金流向数据"""
    print("\n=== 资金流向数据演示 ===")
    
    request_data = {
        'data_type': DataType.FUND_FLOW,
        'symbol': '000001.SZ',
        'start_date': '2024-01-01',
        'end_date': '2024-01-31',
        'data_source': DataSource.EASTMONEY,
        'filters': {
            'period_type': '今日'
        }
    }
    
    result = await process_market_data_request(request_data)
    
    if result['success']:
        print(f"✅ 成功获取 {request_data['symbol']} 资金流向数据")
        print(f"数据源: {result['data_source']}")
        print(f"记录数: {result['record_count']}")
        print("数据预览:")
        print(result['data'].head())
    else:
        print(f"❌ 获取数据失败: {result['error']}")


async def demo_limitup_reason_data():
    """演示获取涨停原因数据"""
    print("\n=== 涨停原因数据演示 ===")
    
    request_data = {
        'data_type': DataType.LIMITUP_REASON,
        'trade_date': '2024-01-15',
        'data_source': DataSource.AUTO,
        'limit': 20,
        'fields': ['stock_code', 'stock_name', 'reason', 'change_rate']
    }
    
    result = await process_market_data_request(request_data)
    
    if result['success']:
        print(f"✅ 成功获取涨停原因数据")
        print(f"交易日期: {request_data['trade_date']}")
        print(f"数据源: {result['data_source']}")
        print(f"记录数: {result['record_count']}")
        print("数据预览:")
        print(result['data'].head())
    else:
        print(f"❌ 获取数据失败: {result['error']}")


async def demo_etf_data():
    """演示获取ETF数据"""
    print("\n=== ETF数据演示 ===")
    
    request_data = {
        'data_type': DataType.ETF_DATA,
        'symbol': '510300.SH',  # 沪深300ETF
        'start_date': '2024-01-01',
        'end_date': '2024-01-31',
        'period': Period.DAILY,
        'data_source': DataSource.EASTMONEY
    }
    
    result = await process_market_data_request(request_data)
    
    if result['success']:
        print(f"✅ 成功获取 {request_data['symbol']} ETF数据")
        print(f"数据源: {result['data_source']}")
        print(f"记录数: {result['record_count']}")
        print("数据预览:")
        print(result['data'].head())
    else:
        print(f"❌ 获取数据失败: {result['error']}")


async def demo_concurrent_requests():
    """演示并发请求处理"""
    print("\n=== 并发请求演示 ===")
    
    # 创建多个并发请求
    requests = [
        {
            'data_type': DataType.STOCK_REALTIME,
            'symbol': f'00000{i}.SZ',
            'data_source': DataSource.AUTO
        }
        for i in range(1, 6)
    ]
    
    print(f"发起 {len(requests)} 个并发请求...")
    start_time = asyncio.get_event_loop().time()
    
    # 并发执行请求
    tasks = [process_market_data_request(req) for req in requests]
    results = await asyncio.gather(*tasks)
    
    end_time = asyncio.get_event_loop().time()
    duration = end_time - start_time
    
    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    total_records = sum(r.get('record_count', 0) for r in results if r['success'])
    
    print(f"✅ 并发请求完成")
    print(f"总耗时: {duration:.2f} 秒")
    print(f"成功请求: {success_count}/{len(requests)}")
    print(f"总记录数: {total_records}")
    
    # 显示失败的请求
    for i, result in enumerate(results):
        if not result['success']:
            print(f"❌ 请求 {i+1} 失败: {result['error']}")


async def demo_request_validation():
    """演示请求验证"""
    print("\n=== 请求验证演示 ===")
    
    # 测试无效的股票代码
    print("--- 测试无效股票代码 ---")
    invalid_requests = [
        {
            'data_type': DataType.STOCK_REALTIME,
            'symbol': 'INVALID_CODE'
        },
        {
            'data_type': DataType.STOCK_REALTIME,
            'symbol': '00001'  # 长度不足
        },
        {
            'data_type': DataType.STOCK_HISTORY,
            'symbol': '000001.SZ'
            # 缺少日期参数
        },
        {
            'data_type': DataType.FUND_FLOW
            # 缺少股票代码
        }
    ]
    
    for i, request_data in enumerate(invalid_requests):
        result = await process_market_data_request(request_data)
        if not result['success']:
            print(f"❌ 请求 {i+1} 验证失败: {result['error']}")
        else:
            print(f"⚠️  请求 {i+1} 意外成功")


async def demo_performance_monitoring():
    """演示性能监控"""
    print("\n=== 性能监控演示 ===")
    
    # 执行一些请求以生成性能数据
    requests = [
        {
            'data_type': DataType.STOCK_REALTIME,
            'symbol': '000001.SZ'
        },
        {
            'data_type': DataType.DRAGON_TIGER,
            'trade_date': '2024-01-15'
        },
        {
            'data_type': DataType.FUND_FLOW,
            'symbol': '000002.SZ'
        }
    ]
    
    print("执行测试请求...")
    for request_data in requests:
        await process_market_data_request(request_data)
    
    # 获取性能统计
    stats = unified_request_interface.get_performance_stats()
    active_requests = unified_request_interface.get_active_requests()
    
    print("📊 性能统计:")
    if stats:
        print(f"总请求数: {stats.get('total_requests', 0)}")
        print(f"平均响应时间: {stats.get('avg_duration', 0):.3f} 秒")
        print(f"最大响应时间: {stats.get('max_duration', 0):.3f} 秒")
        print(f"最小响应时间: {stats.get('min_duration', 0):.3f} 秒")
        print(f"错误率: {stats.get('error_rate', 0):.2%}")
        print(f"缓存命中率: {stats.get('cache_hit_rate', 0):.2%}")
        print(f"总记录数: {stats.get('total_records', 0)}")
    else:
        print("暂无性能数据")
    
    print(f"当前活跃请求数: {active_requests}")


async def demo_error_handling():
    """演示错误处理"""
    print("\n=== 错误处理演示 ===")
    
    # 模拟各种错误情况
    error_cases = [
        {
            'name': '无效数据类型',
            'request': {
                'data_type': 'invalid_type',
                'symbol': '000001.SZ'
            }
        },
        {
            'name': '无效股票代码',
            'request': {
                'data_type': DataType.STOCK_REALTIME,
                'symbol': 'INVALID'
            }
        },
        {
            'name': '缺少必需参数',
            'request': {
                'data_type': DataType.STOCK_REALTIME
            }
        },
        {
            'name': '无效日期格式',
            'request': {
                'data_type': DataType.STOCK_HISTORY,
                'symbol': '000001.SZ',
                'start_date': '2024/01/01'
            }
        }
    ]
    
    for case in error_cases:
        print(f"\n--- {case['name']} ---")
        result = await process_market_data_request(case['request'])
        
        if not result['success']:
            print(f"✅ 正确捕获错误: {result['error']}")
        else:
            print(f"⚠️  未捕获到预期错误")


async def main():
    """主演示函数"""
    print("🚀 统一数据请求接口演示开始")
    print("=" * 50)
    
    try:
        # 基本功能演示
        await demo_stock_realtime_data()
        await demo_stock_history_data()
        await demo_dragon_tiger_data()
        await demo_fund_flow_data()
        await demo_limitup_reason_data()
        await demo_etf_data()
        
        # 高级功能演示
        await demo_concurrent_requests()
        await demo_request_validation()
        await demo_performance_monitoring()
        await demo_error_handling()
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 统一数据请求接口演示完成")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())