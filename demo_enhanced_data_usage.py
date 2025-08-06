#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Data Sources Usage Demo

This script demonstrates how to use the enhanced data sources
for practical stock analysis scenarios.

Author: Stock Analysis System Team
Date: 2025-08-06
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


async def demo_market_overview():
    """演示市场概览功能"""
    print("📊 Market Overview Demo")
    print("=" * 50)
    
    manager = EnhancedDataSourceManager()
    
    # 获取实时市场数据
    print("1. Getting real-time market data...")
    realtime_request = MarketDataRequest(
        symbol="",
        start_date="",
        end_date="",
        data_type="realtime"
    )
    
    realtime_data = await manager.get_enhanced_market_data(realtime_request)
    if not realtime_data.empty:
        print(f"   📈 Total stocks: {len(realtime_data)}")
        
        # 涨跌统计
        up_stocks = len(realtime_data[realtime_data['涨跌幅'] > 0])
        down_stocks = len(realtime_data[realtime_data['涨跌幅'] < 0])
        flat_stocks = len(realtime_data[realtime_data['涨跌幅'] == 0])
        
        print(f"   🟢 Rising: {up_stocks} stocks")
        print(f"   🔴 Falling: {down_stocks} stocks") 
        print(f"   ⚪ Flat: {flat_stocks} stocks")
        
        # 涨幅榜前5
        top_gainers = realtime_data.nlargest(5, '涨跌幅')[['代码', '名称', '最新价', '涨跌幅']]
        print(f"\n   🏆 Top 5 Gainers:")
        print(f"   {top_gainers.to_string(index=False)}")
        
        # 跌幅榜前5
        top_losers = realtime_data.nsmallest(5, '涨跌幅')[['代码', '名称', '最新价', '涨跌幅']]
        print(f"\n   📉 Top 5 Losers:")
        print(f"   {top_losers.to_string(index=False)}")


async def demo_hot_stocks_analysis():
    """演示热门股票分析"""
    print("\n🔥 Hot Stocks Analysis Demo")
    print("=" * 50)
    
    # 获取涨停股票
    limitup_source = LimitUpReasonDataSource()
    limitup_data = limitup_source.get_limitup_reason()
    
    if not limitup_data.empty:
        print(f"1. Limit Up Stocks Today: {len(limitup_data)} stocks")
        print(f"   {limitup_data[['代码', '名称', '原因']].to_string(index=False)}")
        
        # 分析涨停原因
        reasons = limitup_data['原因'].str.split('+').explode().value_counts().head(5)
        print(f"\n   📋 Top Limit Up Reasons:")
        for reason, count in reasons.items():
            print(f"   - {reason}: {count} stocks")
    
    # 获取早盘抢筹数据
    chip_source = ChipRaceDataSource()
    chip_data = chip_source.get_chip_race_open()
    
    if not chip_data.empty:
        print(f"\n2. Morning Chip Race: {len(chip_data)} stocks")
        top_chip_race = chip_data.head(5)[['代码', '名称', '最新价', '涨跌幅', '抢筹幅度']]
        print(f"   {top_chip_race.to_string(index=False)}")


async def demo_institutional_activity():
    """演示机构活动分析"""
    print("\n🏛️ Institutional Activity Demo")
    print("=" * 50)
    
    # 获取龙虎榜数据
    dt_source = DragonTigerDataSource()
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=3)).strftime("%Y%m%d")
    
    dt_data = dt_source.get_dragon_tiger_detail(start_date, end_date)
    
    if not dt_data.empty:
        print(f"1. Dragon Tiger List (Last 3 days): {len(dt_data)} records")
        
        # 按净买额排序
        top_net_buy = dt_data.nlargest(5, '龙虎榜净买额')[['代码', '名称', '上榜日', '涨跌幅', '龙虎榜净买额']]
        print(f"\n   💰 Top Net Buyers:")
        print(f"   {top_net_buy.to_string(index=False)}")
        
        # 统计上榜次数
        frequent_stocks = dt_data['名称'].value_counts().head(5)
        print(f"\n   🔄 Most Frequent Stocks:")
        for stock, count in frequent_stocks.items():
            print(f"   - {stock}: {count} times")


async def demo_stock_deep_analysis():
    """演示个股深度分析"""
    print("\n🔍 Stock Deep Analysis Demo")
    print("=" * 50)
    
    # 选择一个活跃股票进行分析
    symbol = "000001"  # 平安银行
    print(f"Analyzing stock: {symbol}")
    
    manager = EnhancedDataSourceManager()
    
    # 获取历史数据
    history_request = MarketDataRequest(
        symbol=symbol,
        start_date="20241201",
        end_date="20241231",
        period="daily",
        data_type="history"
    )
    
    history_data = await manager.get_enhanced_market_data(history_request)
    
    if not history_data.empty:
        print(f"\n1. Historical Performance (Dec 2024):")
        print(f"   📅 Trading days: {len(history_data)}")
        print(f"   💹 Price range: {history_data['最低'].min():.2f} - {history_data['最高'].max():.2f}")
        print(f"   📊 Average volume: {history_data['成交量'].mean():.0f}")
        print(f"   🎯 Total return: {((history_data['收盘'].iloc[-1] / history_data['收盘'].iloc[0] - 1) * 100):.2f}%")
        
        # 计算技术指标
        history_data['MA5'] = history_data['收盘'].rolling(5).mean()
        history_data['MA10'] = history_data['收盘'].rolling(10).mean()
        
        latest_price = history_data['收盘'].iloc[-1]
        latest_ma5 = history_data['MA5'].iloc[-1]
        latest_ma10 = history_data['MA10'].iloc[-1]
        
        print(f"\n2. Technical Analysis:")
        print(f"   📈 Latest price: {latest_price:.2f}")
        print(f"   📊 MA5: {latest_ma5:.2f}")
        print(f"   📊 MA10: {latest_ma10:.2f}")
        
        if latest_price > latest_ma5 > latest_ma10:
            trend = "🟢 Bullish"
        elif latest_price < latest_ma5 < latest_ma10:
            trend = "🔴 Bearish"
        else:
            trend = "🟡 Neutral"
        
        print(f"   🎯 Trend: {trend}")


async def demo_data_quality_monitoring():
    """演示数据质量监控"""
    print("\n🔍 Data Quality Monitoring Demo")
    print("=" * 50)
    
    manager = EnhancedDataSourceManager()
    
    # 健康检查
    print("1. Data Source Health Check:")
    health_status = await manager.health_check_enhanced_sources()
    
    for source, health in health_status.items():
        status_emoji = "✅" if health.status == "healthy" else "⚠️" if health.status == "degraded" else "❌"
        print(f"   {status_emoji} {source.upper()}: {health.status}")
        print(f"      - Reliability Score: {health.reliability_score:.1f}")
        print(f"      - Avg Response Time: {health.avg_response_time:.2f}s")
    
    # 数据完整性检查
    print(f"\n2. Data Completeness Check:")
    realtime_request = MarketDataRequest(
        symbol="",
        start_date="",
        end_date="",
        data_type="realtime"
    )
    
    data = await manager.get_enhanced_market_data(realtime_request)
    if not data.empty:
        completeness = (1 - data.isnull().sum() / len(data)) * 100
        critical_fields = ['代码', '名称', '最新价', '涨跌幅']
        
        print(f"   📊 Total records: {len(data)}")
        print(f"   🎯 Critical fields completeness:")
        for field in critical_fields:
            if field in completeness:
                print(f"      - {field}: {completeness[field]:.1f}%")


async def main():
    """主演示函数"""
    print("🚀 Enhanced Data Sources Usage Demo")
    print("=" * 60)
    print(f"Demo started at: {datetime.now()}")
    print()
    
    try:
        # 运行所有演示
        await demo_market_overview()
        await demo_hot_stocks_analysis()
        await demo_institutional_activity()
        await demo_stock_deep_analysis()
        await demo_data_quality_monitoring()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Demo finished at: {datetime.now()}")


if __name__ == "__main__":
    asyncio.run(main())