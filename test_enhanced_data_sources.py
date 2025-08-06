#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Data Sources Test

This script tests the integrated data sources from the crawling interfaces
to validate their functionality and data quality.

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_analysis_system.data.enhanced_data_sources import (
    EnhancedDataSourceManager,
    MarketDataRequest,
    EastMoneyDataSource,
    DragonTigerDataSource,
    LimitUpReasonDataSource,
    ChipRaceDataSource
)


async def test_eastmoney_data_source():
    """测试东方财富数据源"""
    print("🔍 Testing EastMoney Data Source")
    print("-" * 40)
    
    source = EastMoneyDataSource()
    
    # 测试实时行情数据
    print("1. Testing realtime stock data...")
    try:
        realtime_data = source.get_stock_realtime_data()
        if not realtime_data.empty:
            print(f"   ✅ Success: Got {len(realtime_data)} stocks")
            print(f"   Sample data: {realtime_data.head(3)[['代码', '名称', '最新价', '涨跌幅']].to_string()}")
        else:
            print("   ❌ Failed: No realtime data")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 测试历史数据
    print("\n2. Testing historical stock data...")
    try:
        history_data = source.get_stock_history_data(
            symbol="000001",
            period="daily",
            start_date="20241201",
            end_date="20241231"
        )
        if not history_data.empty:
            print(f"   ✅ Success: Got {len(history_data)} records for 000001")
            print(f"   Date range: {history_data.index.min()} to {history_data.index.max()}")
        else:
            print("   ❌ Failed: No historical data")
    except Exception as e:
        print(f"   ❌ Error: {e}")


async def test_dragon_tiger_data_source():
    """测试龙虎榜数据源"""
    print("\n🐉 Testing Dragon Tiger Data Source")
    print("-" * 40)
    
    source = DragonTigerDataSource()
    
    try:
        # 获取最近一周的龙虎榜数据
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        
        dt_data = source.get_dragon_tiger_detail(start_date, end_date)
        if not dt_data.empty:
            print(f"   ✅ Success: Got {len(dt_data)} dragon tiger records")
            print(f"   Sample data:")
            print(f"   {dt_data.head(3)[['代码', '名称', '上榜日', '涨跌幅', '龙虎榜净买额']].to_string()}")
        else:
            print("   ❌ Failed: No dragon tiger data")
    except Exception as e:
        print(f"   ❌ Error: {e}")


async def test_limitup_reason_data_source():
    """测试涨停原因数据源"""
    print("\n📈 Testing Limit Up Reason Data Source")
    print("-" * 40)
    
    source = LimitUpReasonDataSource()
    
    try:
        # 获取今日涨停原因数据
        limitup_data = source.get_limitup_reason()
        if not limitup_data.empty:
            print(f"   ✅ Success: Got {len(limitup_data)} limit up records")
            print(f"   Sample data:")
            print(f"   {limitup_data.head(3)[['代码', '名称', '原因', '涨跌幅']].to_string()}")
        else:
            print("   ❌ Failed: No limit up data (may be normal if no limit up stocks today)")
    except Exception as e:
        print(f"   ❌ Error: {e}")


async def test_chip_race_data_source():
    """测试筹码竞价数据源"""
    print("\n🏁 Testing Chip Race Data Source")
    print("-" * 40)
    
    source = ChipRaceDataSource()
    
    try:
        # 获取早盘抢筹数据
        chip_data = source.get_chip_race_open()
        if not chip_data.empty:
            print(f"   ✅ Success: Got {len(chip_data)} chip race records")
            print(f"   Sample data:")
            print(f"   {chip_data.head(3)[['代码', '名称', '最新价', '涨跌幅', '抢筹幅度']].to_string()}")
        else:
            print("   ❌ Failed: No chip race data")
    except Exception as e:
        print(f"   ❌ Error: {e}")


async def test_enhanced_data_manager():
    """测试增强数据管理器"""
    print("\n🚀 Testing Enhanced Data Manager")
    print("-" * 40)
    
    manager = EnhancedDataSourceManager()
    
    # 测试不同类型的数据请求
    test_requests = [
        {
            "name": "Realtime Data",
            "request": MarketDataRequest(
                symbol="000001",
                start_date="",
                end_date="",
                data_type="realtime"
            )
        },
        {
            "name": "Historical Data",
            "request": MarketDataRequest(
                symbol="000001",
                start_date="20241201",
                end_date="20241231",
                period="daily",
                data_type="history"
            )
        },
        {
            "name": "Dragon Tiger Data",
            "request": MarketDataRequest(
                symbol="",
                start_date="20241201",
                end_date="20241231",
                data_type="dragon_tiger"
            )
        }
    ]
    
    for test in test_requests:
        print(f"\n   Testing {test['name']}...")
        try:
            data = await manager.get_enhanced_market_data(test['request'])
            if not data.empty:
                print(f"   ✅ Success: Got {len(data)} records")
            else:
                print(f"   ⚠️  Warning: No data returned")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # 测试健康检查
    print(f"\n   Testing health check...")
    try:
        health_status = await manager.health_check_enhanced_sources()
        print(f"   Health Status:")
        for source, health in health_status.items():
            status_emoji = "✅" if health.status == "healthy" else "⚠️" if health.status == "degraded" else "❌"
            print(f"   {status_emoji} {source}: {health.status} (Score: {health.reliability_score:.1f})")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")


async def test_data_quality():
    """测试数据质量"""
    print("\n📊 Testing Data Quality")
    print("-" * 40)
    
    source = EastMoneyDataSource()
    
    try:
        # 获取实时数据进行质量检查
        data = source.get_stock_realtime_data()
        
        if data.empty:
            print("   ❌ No data to analyze")
            return
        
        print(f"   Data shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        
        # 检查数据完整性
        missing_data = data.isnull().sum()
        print(f"   Missing data per column:")
        for col, missing in missing_data.items():
            if missing > 0:
                print(f"     {col}: {missing} ({missing/len(data)*100:.1f}%)")
        
        # 检查数值列的统计信息
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            print(f"   Numeric columns statistics:")
            stats = data[numeric_cols].describe()
            print(f"     {stats.loc[['count', 'mean', 'std']].to_string()}")
        
        # 检查代码格式
        if '代码' in data.columns:
            code_lengths = data['代码'].str.len().value_counts()
            print(f"   Stock code lengths: {dict(code_lengths)}")
        
        print("   ✅ Data quality check completed")
        
    except Exception as e:
        print(f"   ❌ Data quality check error: {e}")


async def main():
    """主测试函数"""
    print("🧪 Enhanced Data Sources Integration Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    # 运行所有测试
    await test_eastmoney_data_source()
    await test_dragon_tiger_data_source()
    await test_limitup_reason_data_source()
    await test_chip_race_data_source()
    await test_enhanced_data_manager()
    await test_data_quality()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print(f"Test finished at: {datetime.now()}")


if __name__ == "__main__":
    asyncio.run(main())