#!/usr/bin/env python3
"""Demo script for Spring Festival Analysis Engine using local data."""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from stock_analysis_system.analysis.spring_festival_engine import (
    ChineseCalendar,
    SpringFestivalAlignmentEngine
)
from stock_analysis_system.data.data_source_manager import get_data_source_manager

async def get_real_stock_data(symbol: str, years: int = 5):
    """Get real stock data from local source."""
    print(f"📊 Loading real stock data for {symbol}...")
    
    try:
        # Get data source manager
        manager = await get_data_source_manager()
        
        # Calculate date range
        end_date = date.today()
        start_date = date(end_date.year - years, 1, 1)
        
        # Fetch data
        stock_data = await manager.get_stock_data(symbol, start_date, end_date)
        
        if stock_data.empty:
            print(f"   ⚠️ No data found for {symbol}")
            return None
        
        print(f"   ✓ Loaded {len(stock_data)} trading days of data")
        print(f"   ✓ Date range: {stock_data['trade_date'].min().date()} to {stock_data['trade_date'].max().date()}")
        print(f"   ✓ Price range: {stock_data['close_price'].min():.2f} to {stock_data['close_price'].max():.2f}")
        
        return stock_data
        
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return None

async def demonstrate_real_spring_festival_analysis():
    """Demonstrate Spring Festival analysis with real local data."""
    print("🚀 Spring Festival Analysis with Real Local Data")
    print("=" * 70)
    
    # Get available stocks from local data
    print("\n📋 Getting available stocks...")
    try:
        manager = await get_data_source_manager()
        stock_list = await manager.get_stock_list()
        
        if stock_list.empty:
            print("❌ No stocks available in local data")
            return
        
        print(f"✓ Found {len(stock_list)} stocks in local data")
        
        # Select a few representative stocks for analysis
        test_symbols = []
        
        # Try to find some common large-cap stocks
        preferred_symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
        available_symbols = stock_list['stock_code'].tolist()
        
        for symbol in preferred_symbols:
            if symbol in available_symbols:
                test_symbols.append(symbol)
            if len(test_symbols) >= 3:  # Limit to 3 stocks for demo
                break
        
        # If we don't have preferred symbols, use the first few available
        if not test_symbols:
            test_symbols = available_symbols[:3]
        
        print(f"✓ Selected stocks for analysis: {test_symbols}")
        
    except Exception as e:
        print(f"❌ Error getting stock list: {e}")
        return
    
    # Analyze each stock
    results = {}
    
    for symbol in test_symbols:
        print(f"\n🎯 Analyzing {symbol}")
        print("-" * 50)
        
        # Load stock data
        stock_data = await get_real_stock_data(symbol, years=6)
        
        if stock_data is None:
            print(f"   Skipping {symbol} - no data available")
            continue
        
        try:
            # Initialize Spring Festival engine
            engine = SpringFestivalAlignmentEngine(window_days=60)
            
            # Align data to Spring Festival
            print("   🔄 Aligning data to Spring Festival dates...")
            years = [2020, 2021, 2022, 2023, 2024, 2025]
            aligned_data = engine.align_to_spring_festival(stock_data, years)
            
            print(f"   ✓ Aligned {len(aligned_data.data_points)} data points")
            print(f"   ✓ Years analyzed: {aligned_data.years}")
            
            # Identify seasonal patterns
            print("   📈 Identifying seasonal patterns...")
            pattern = engine.identify_seasonal_patterns(aligned_data)
            
            # Store results
            results[symbol] = {
                'aligned_data': aligned_data,
                'pattern': pattern,
                'data_points': len(aligned_data.data_points)
            }
            
            # Display pattern results
            print(f"   📊 PATTERN RESULTS")
            print(f"      Pattern Strength: {pattern.pattern_strength:.3f}")
            print(f"      Confidence Level: {pattern.confidence_level:.3f}")
            print(f"      Avg Return Before SF: {pattern.average_return_before:+.2f}%")
            print(f"      Avg Return After SF: {pattern.average_return_after:+.2f}%")
            print(f"      Volatility Ratio: {pattern.volatility_ratio:.2f}x")
            
            # Generate current trading signals
            position = engine.get_current_position(symbol)
            signals = engine.generate_trading_signals(pattern, position)
            
            print(f"   🚦 CURRENT SIGNALS")
            print(f"      Position: {position['position']}")
            print(f"      Signal: {signals['signal'].upper()}")
            print(f"      Strength: {signals['strength']:.2f}")
            print(f"      Action: {signals['recommended_action'].upper()}")
            
            if signals['reason']:
                print(f"      Reason: {signals['reason']}")
            
        except Exception as e:
            print(f"   ❌ Analysis failed for {symbol}: {e}")
            continue
    
    # Summary analysis
    if results:
        print(f"\n📊 SUMMARY ANALYSIS")
        print("=" * 70)
        
        print(f"Successfully analyzed {len(results)} stocks:")
        
        # Calculate aggregate statistics
        all_patterns = [r['pattern'] for r in results.values()]
        
        avg_pattern_strength = np.mean([p.pattern_strength for p in all_patterns])
        avg_confidence = np.mean([p.confidence_level for p in all_patterns])
        avg_return_before = np.mean([p.average_return_before for p in all_patterns])
        avg_return_after = np.mean([p.average_return_after for p in all_patterns])
        
        print(f"\n🎯 AGGREGATE PATTERNS")
        print(f"   Average Pattern Strength: {avg_pattern_strength:.3f}")
        print(f"   Average Confidence Level: {avg_confidence:.3f}")
        print(f"   Average Return Before SF: {avg_return_before:+.2f}%")
        print(f"   Average Return After SF: {avg_return_after:+.2f}%")
        
        # Individual stock summary
        print(f"\n📈 INDIVIDUAL STOCK SUMMARY")
        for symbol, result in results.items():
            pattern = result['pattern']
            print(f"   {symbol}:")
            print(f"      Strength: {pattern.pattern_strength:.2f}, "
                  f"Before SF: {pattern.average_return_before:+.1f}%, "
                  f"After SF: {pattern.average_return_after:+.1f}%")
        
        # Market insights
        print(f"\n💡 MARKET INSIGHTS")
        
        bullish_before_count = sum(1 for p in all_patterns if p.is_bullish_before)
        bullish_after_count = sum(1 for p in all_patterns if p.is_bullish_after)
        
        print(f"   Stocks bullish before Spring Festival: {bullish_before_count}/{len(all_patterns)}")
        print(f"   Stocks bullish after Spring Festival: {bullish_after_count}/{len(all_patterns)}")
        
        if avg_return_before > 0:
            print(f"   📈 Market tends to rise before Spring Festival")
        else:
            print(f"   📉 Market tends to decline before Spring Festival")
        
        if avg_return_after > 0:
            print(f"   📈 Market tends to rise after Spring Festival")
        else:
            print(f"   📉 Market tends to decline after Spring Festival")
        
        # High-confidence patterns
        high_confidence_patterns = [p for p in all_patterns if p.confidence_level > 0.7]
        if high_confidence_patterns:
            print(f"\n🎯 HIGH-CONFIDENCE PATTERNS ({len(high_confidence_patterns)} stocks)")
            for symbol, result in results.items():
                if result['pattern'].confidence_level > 0.7:
                    p = result['pattern']
                    print(f"   {symbol}: {p.confidence_level:.2f} confidence, "
                          f"{p.pattern_strength:.2f} strength")
    
    else:
        print("\n❌ No successful analyses completed")
        print("   Please check if local data is available and accessible")

async def demonstrate_spring_festival_calendar():
    """Demonstrate Spring Festival calendar functionality."""
    print("\n🏮 Spring Festival Calendar Analysis")
    print("-" * 50)
    
    # Show upcoming Spring Festival dates
    print("Upcoming Spring Festival dates:")
    current_year = date.today().year
    for year in range(current_year, current_year + 5):
        sf_date = ChineseCalendar.get_spring_festival(year)
        if sf_date:
            days_from_now = (sf_date - date.today()).days
            weekday = sf_date.strftime('%A')
            print(f"   {year}: {sf_date} ({weekday}) - {days_from_now} days from now")
    
    # Current position analysis
    today = date.today()
    sf_date = ChineseCalendar.get_spring_festival(today.year)
    
    if sf_date:
        is_in_period, _ = ChineseCalendar.is_spring_festival_period(today, window_days=60)
        days_to_sf = (sf_date - today).days
        
        print(f"\nCurrent market position:")
        print(f"   Today: {today}")
        print(f"   Next Spring Festival: {sf_date}")
        print(f"   Days to Spring Festival: {days_to_sf}")
        print(f"   In analysis window (±60 days): {is_in_period}")
        
        if is_in_period:
            if days_to_sf > 0:
                print(f"   📅 Pre-Spring Festival period - Watch for seasonal patterns")
            else:
                print(f"   📅 Post-Spring Festival period - Monitor recovery patterns")
        else:
            print(f"   📅 Normal trading period - Outside seasonal analysis window")

async def main():
    """Main demonstration function."""
    try:
        # Check if local data is available
        manager = await get_data_source_manager()
        health = await manager.health_check()
        
        from stock_analysis_system.data.data_source_manager import DataSourceType
        local_health = health.get(DataSourceType.LOCAL)
        
        if not local_health or local_health.status.value != 'healthy':
            print("❌ Local data source is not available or healthy")
            print("   Please ensure TDX data files are accessible")
            return
        
        print("✅ Local data source is healthy and ready")
        
        # Run demonstrations
        await demonstrate_spring_festival_calendar()
        await demonstrate_real_spring_festival_analysis()
        
        print(f"\n🎉 Spring Festival analysis with real data completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())