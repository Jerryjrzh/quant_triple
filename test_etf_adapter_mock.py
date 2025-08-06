"""
ETF适配器模拟测试

用于验证ETF适配器结构和逻辑的正确性，不依赖外部网络连接。
"""

import asyncio
import pandas as pd
from datetime import datetime
from stock_analysis_system.data.etf_adapter import ETFAdapter, ETFRequest, ETFResponse


class MockETFAdapter(ETFAdapter):
    """模拟ETF适配器"""
    
    async def _make_request(self, url: str, params: dict):
        """模拟HTTP请求"""
        # 根据不同的API端点返回不同的模拟数据
        if 'clist' in url and params.get('fields') == 'f12,f13':
            # ETF代码映射数据
            return {
                "data": {
                    "diff": [
                        {"f12": "159707", "f13": "0"},  # 中证500ETF
                        {"f12": "513500", "f13": "1"},  # 标普500ETF
                        {"f12": "510300", "f13": "1"},  # 沪深300ETF
                    ]
                }
            }
        elif 'clist' in url:
            # ETF实时行情数据 - 包含所有必需字段
            return {
                "data": {
                    "total": 2,
                    "diff": [
                        {
                            "f12": "159707",    # 代码
                            "f14": "中证500ETF", # 名称
                            "f2": 1.250,        # 最新价
                            "f3": 2.50,         # 涨跌幅
                            "f4": 0.031,        # 涨跌额
                            "f5": 15000000,     # 成交量
                            "f6": 18750000,     # 成交额
                            "f17": 1.220,       # 开盘价
                            "f15": 1.255,       # 最高价
                            "f16": 1.215,       # 最低价
                            "f18": 1.219,       # 昨收
                            "f8": 3.2,          # 换手率
                            "f21": 500000000,   # 流通市值
                            "f20": 500000000    # 总市值
                        },
                        {
                            "f12": "513500",    # 代码
                            "f14": "标普500ETF", # 名称
                            "f2": 2.180,        # 最新价
                            "f3": -1.20,        # 涨跌幅
                            "f4": -0.026,       # 涨跌额
                            "f5": 8000000,      # 成交量
                            "f6": 17440000,     # 成交额
                            "f17": 2.200,       # 开盘价
                            "f15": 2.205,       # 最高价
                            "f16": 2.175,       # 最低价
                            "f18": 2.206,       # 昨收
                            "f8": 2.8,          # 换手率
                            "f21": 800000000,   # 流通市值
                            "f20": 800000000    # 总市值
                        }
                    ]
                }
            }
        elif 'kline' in url:
            # K线数据
            return {
                "data": {
                    "klines": [
                        "2024-01-01,1.200,1.250,1.260,1.195,15000000,18750000,5.42,4.17,0.050,3.2",
                        "2024-01-02,1.250,1.280,1.285,1.240,18000000,23040000,3.60,2.40,0.030,3.8"
                    ]
                }
            }
        elif 'trends2' in url:
            # 分时数据
            return {
                "data": {
                    "trends": [
                        "2024-01-01 09:30:00,1.200,1.210,1.210,1.200,1000000,1210000,1.210",
                        "2024-01-01 09:31:00,1.210,1.215,1.215,1.205,1200000,1458000,1.215"
                    ]
                }
            }
        
        return None
    
    async def _fetch_all_pages(self, url: str, base_params: dict):
        """模拟获取所有分页数据"""
        response = await self._make_request(url, base_params)
        if response and response.get("data", {}).get("diff"):
            return response["data"]["diff"]
        return []


async def test_mock_etf_adapter():
    """测试模拟ETF适配器"""
    print("🔍 测试模拟ETF适配器")
    print("=" * 50)
    
    adapter = MockETFAdapter()
    
    # 测试ETF实时数据
    print("1. 测试ETF实时数据...")
    realtime_response = await adapter.get_etf_realtime_data(['159707', '513500'])
    print(f"   成功: {realtime_response.success}")
    print(f"   响应时间: {realtime_response.response_time:.3f}秒")
    print(f"   数据行数: {len(realtime_response.data)}")
    if not realtime_response.data.empty:
        print(f"   数据预览:")
        print(realtime_response.data[['代码', '名称', '最新价', '涨跌幅']].head())
    
    # 测试ETF历史数据
    print("\n2. 测试ETF历史数据...")
    history_request = ETFRequest(
        symbol="159707",
        data_type="history",
        period="daily",
        start_date="20240101",
        end_date="20241231"
    )
    history_response = await adapter.get_etf_history_data(history_request)
    print(f"   成功: {history_response.success}")
    print(f"   响应时间: {history_response.response_time:.3f}秒")
    print(f"   数据行数: {len(history_response.data)}")
    if not history_response.data.empty:
        print(f"   数据预览:")
        print(history_response.data[['开盘', '收盘', '最高', '最低']].head())
    
    # 测试ETF分时数据（K线）
    print("\n3. 测试ETF分时数据（K线）...")
    intraday_request = ETFRequest(
        symbol="159707",
        data_type="intraday",
        period="5"
    )
    intraday_response = await adapter.get_etf_intraday_data(intraday_request)
    print(f"   成功: {intraday_response.success}")
    print(f"   响应时间: {intraday_response.response_time:.3f}秒")
    print(f"   数据行数: {len(intraday_response.data)}")
    if not intraday_response.data.empty:
        print(f"   数据预览:")
        print(intraday_response.data[['开盘', '收盘', '涨跌幅']].head())
    
    # 测试ETF分时数据（1分钟）
    print("\n4. 测试ETF分时数据（1分钟）...")
    intraday_1min_request = ETFRequest(
        symbol="159707",
        data_type="intraday",
        period="1"
    )
    intraday_1min_response = await adapter.get_etf_intraday_data(intraday_1min_request)
    print(f"   成功: {intraday_1min_response.success}")
    print(f"   响应时间: {intraday_1min_response.response_time:.3f}秒")
    print(f"   数据行数: {len(intraday_1min_response.data)}")
    if not intraday_1min_response.data.empty:
        print(f"   数据预览:")
        print(intraday_1min_response.data[['开盘', '收盘', '最新价']].head())
    
    # 测试ETF特有指标
    print("\n5. 测试ETF特有指标...")
    if not realtime_response.data.empty:
        indicators = adapter.get_etf_special_indicators(realtime_response.data)
        print(f"   计算出 {len(indicators)} 个指标")
        for key, value in list(indicators.items())[:3]:
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    
    # 健康检查
    print("\n6. 健康检查...")
    health = await adapter.health_check()
    print(f"   状态: {health['status']}")
    print(f"   响应时间: {health.get('response_time', 0):.3f}秒")
    
    # 错误统计
    print("\n7. 错误统计...")
    stats = adapter.get_error_statistics()
    print(f"   总请求数: {stats['total_requests']}")
    print(f"   成功率: {stats.get('success_rate', 0):.2%}")
    
    print("\n✅ 模拟ETF适配器测试完成!")
    
    # 验证数据结构
    print("\n8. 验证数据结构...")
    
    # 检查实时数据结构
    if not realtime_response.data.empty:
        required_cols = ['代码', '名称', '最新价', '涨跌幅', '成交量', '成交额']
        missing_cols = [col for col in required_cols if col not in realtime_response.data.columns]
        if missing_cols:
            print(f"   ❌ 实时数据缺少列: {missing_cols}")
        else:
            print(f"   ✅ 实时数据结构正确")
    
    # 检查历史数据结构
    if not history_response.data.empty:
        required_cols = ['开盘', '收盘', '最高', '最低', '成交量']
        missing_cols = [col for col in required_cols if col not in history_response.data.columns]
        if missing_cols:
            print(f"   ❌ 历史数据缺少列: {missing_cols}")
        else:
            print(f"   ✅ 历史数据结构正确")
    
    # 检查分时数据结构
    if not intraday_response.data.empty:
        required_cols = ['开盘', '收盘', '最高', '最低']
        missing_cols = [col for col in required_cols if col not in intraday_response.data.columns]
        if missing_cols:
            print(f"   ❌ 分时数据缺少列: {missing_cols}")
        else:
            print(f"   ✅ 分时数据结构正确")
    
    # 检查数据类型
    if not realtime_response.data.empty:
        numeric_cols = ['最新价', '涨跌幅', '成交量']
        for col in numeric_cols:
            if col in realtime_response.data.columns:
                if realtime_response.data[col].dtype in ['float64', 'int64']:
                    print(f"   ✅ {col} 数据类型正确")
                else:
                    print(f"   ❌ {col} 数据类型错误: {realtime_response.data[col].dtype}")
    
    print("\n9. 测试错误处理...")
    
    # 测试无效ETF代码
    invalid_request = ETFRequest(
        symbol="INVALID",
        data_type="history",
        period="daily"
    )
    invalid_response = await adapter.get_etf_history_data(invalid_request)
    print(f"   无效代码处理: {'✅' if not invalid_response.success else '❌'}")
    
    # 测试空数据处理
    empty_response = await adapter.get_etf_realtime_data([])
    print(f"   空数据处理: {'✅' if empty_response.success else '❌'}")
    
    print("\n10. 测试数据过滤...")
    
    # 测试日期过滤
    filtered_request = ETFRequest(
        symbol="159707",
        data_type="intraday",
        period="5",
        start_date="2024-01-01",
        end_date="2024-01-01"
    )
    filtered_response = await adapter.get_etf_intraday_data(filtered_request)
    print(f"   日期过滤: {'✅' if filtered_response.success else '❌'}")
    
    print("\n11. 测试缓存机制...")
    
    # 测试代码映射缓存
    start_time = time.time()
    await adapter._get_etf_code_id_map()
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    await adapter._get_etf_code_id_map()
    second_call_time = time.time() - start_time
    
    print(f"   缓存效果: {'✅' if second_call_time < first_call_time else '❌'}")
    print(f"   首次调用: {first_call_time:.4f}秒, 缓存调用: {second_call_time:.4f}秒")
    
    print("\n12. 测试并发处理...")
    
    # 测试并发请求
    import asyncio
    tasks = [
        adapter.get_etf_realtime_data(['159707']),
        adapter.get_etf_realtime_data(['513500']),
        adapter.get_etf_realtime_data(['510300'])
    ]
    
    concurrent_results = await asyncio.gather(*tasks)
    successful_concurrent = sum(1 for result in concurrent_results if result.success)
    print(f"   并发处理: {'✅' if successful_concurrent == 3 else '❌'} ({successful_concurrent}/3)")
    
    print("\n13. 性能测试...")
    
    # 测试响应时间
    performance_times = []
    for i in range(5):
        start_time = time.time()
        perf_response = await adapter.get_etf_realtime_data(['159707'])
        performance_times.append(time.time() - start_time)
    
    avg_time = sum(performance_times) / len(performance_times)
    print(f"   平均响应时间: {avg_time:.4f}秒")
    print(f"   性能评估: {'✅' if avg_time < 0.1 else '❌'} (目标 < 0.1秒)")
    
    print("\n14. 最终统计...")
    final_stats = adapter.get_error_statistics()
    print(f"   总请求数: {final_stats['total_requests']}")
    print(f"   成功率: {final_stats.get('success_rate', 0):.2%}")
    print(f"   网络错误: {final_stats['network_errors']}")
    print(f"   数据格式错误: {final_stats['data_format_errors']}")
    print(f"   API错误: {final_stats['api_errors']}")


async def test_etf_adapter_edge_cases():
    """测试ETF适配器边界情况"""
    print("\n🔍 测试ETF适配器边界情况")
    print("=" * 50)
    
    adapter = MockETFAdapter()
    
    # 测试极端参数
    print("1. 测试极端参数...")
    
    # 测试空字符串参数
    empty_request = ETFRequest(symbol="", data_type="history")
    empty_response = await adapter.get_etf_history_data(empty_request)
    print(f"   空字符串处理: {'✅' if not empty_response.success else '❌'}")
    
    # 测试无效日期格式
    invalid_date_request = ETFRequest(
        symbol="159707",
        data_type="history",
        start_date="invalid-date",
        end_date="also-invalid"
    )
    invalid_date_response = await adapter.get_etf_history_data(invalid_date_request)
    print(f"   无效日期处理: {'✅' if invalid_date_response.success else '❌'}")
    
    # 测试超长ETF列表
    long_list = [f"ETF{i:06d}" for i in range(100)]
    long_list_response = await adapter.get_etf_realtime_data(long_list)
    print(f"   超长列表处理: {'✅' if long_list_response.success else '❌'}")
    
    print("\n2. 测试数据完整性...")
    
    # 验证数据完整性
    test_response = await adapter.get_etf_realtime_data(['159707'])
    if test_response.success and not test_response.data.empty:
        # 检查是否有空值
        null_count = test_response.data.isnull().sum().sum()
        print(f"   空值检查: {'✅' if null_count == 0 else f'❌ ({null_count} 个空值)'}")
        
        # 检查数值范围合理性
        if '最新价' in test_response.data.columns:
            price_valid = (test_response.data['最新价'] > 0).all()
            print(f"   价格合理性: {'✅' if price_valid else '❌'}")
        
        if '成交量' in test_response.data.columns:
            volume_valid = (test_response.data['成交量'] >= 0).all()
            print(f"   成交量合理性: {'✅' if volume_valid else '❌'}")
    
    print("\n✅ 边界情况测试完成!")


if __name__ == "__main__":
    import time
    asyncio.run(test_mock_etf_adapter())
    asyncio.run(test_etf_adapter_edge_cases())


if __name__ == "__main__":
    asyncio.run(test_mock_etf_adapter())