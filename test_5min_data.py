#!/usr/bin/env python3
"""Test script for 5-minute data reading functionality."""

import asyncio
import os
from datetime import date, timedelta
from stock_analysis_system.data.data_source_manager import get_data_source_manager, DataSourceType

async def test_5min_data_reading():
    """Test 5-minute data reading functionality."""
    print("🧪 Testing 5-Minute Data Reading")
    print("=" * 60)
    
    # Get data source manager
    manager = await get_data_source_manager()
    
    # Check if local data source is available
    if DataSourceType.LOCAL not in manager.data_sources:
        print("❌ Local data source not initialized")
        return
    
    local_source = manager.data_sources[DataSourceType.LOCAL]
    
    # Test connection
    print("\n1. Testing connection...")
    try:
        is_connected = await local_source.test_connection()
        if is_connected:
            print("✅ Local data source connection successful")
            print(f"   Base path: {local_source.base_path}")
        else:
            print("❌ Local data source connection failed")
            return
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return
    
    # Check if 5-minute data files exist
    print("\n2. Checking 5-minute data files...")
    base_path = local_source.base_path
    
    fzline_files_found = 0
    for market in ['sz', 'sh']:
        fzline_path = os.path.join(base_path, market, 'fzline')
        if os.path.exists(fzline_path):
            lc5_files = [f for f in os.listdir(fzline_path) if f.endswith('.lc5')]
            fzline_files_found += len(lc5_files)
            print(f"   {market.upper()} market: {len(lc5_files)} .lc5 files")
        else:
            print(f"   {market.upper()} market: fzline directory not found")
    
    if fzline_files_found == 0:
        print("❌ No 5-minute data files found")
        print("   Please ensure TDX has downloaded 5-minute data")
        return
    
    print(f"✅ Found {fzline_files_found} 5-minute data files")
    
    # Get a test symbol
    print("\n3. Getting test symbol...")
    try:
        stock_list = await local_source.get_stock_list()
        if stock_list.empty:
            print("❌ No stocks available")
            return
        
        # Use the first available stock
        test_symbol = stock_list.iloc[0]['stock_code']
        print(f"   Using test symbol: {test_symbol}")
        
    except Exception as e:
        print(f"❌ Error getting stock list: {e}")
        return
    
    # Test different timeframes
    timeframes = ['5min', '15min', '30min', '60min']
    end_date = date.today()
    start_date = end_date - timedelta(days=7)  # Last week
    
    for timeframe in timeframes:
        print(f"\n4.{timeframes.index(timeframe)+1} Testing {timeframe} data...")
        
        try:
            # Test with local source directly
            if timeframe == '5min':
                data = await local_source.get_intraday_data(test_symbol, start_date, end_date, timeframe)
            else:
                data = await local_source.get_stock_data(test_symbol, start_date, end_date, timeframe)
            
            if not data.empty:
                print(f"   ✅ Retrieved {len(data)} {timeframe} data points")
                print(f"   Date range: {data['datetime'].min()} to {data['datetime'].max()}")
                print(f"   Price range: {data['close'].min():.2f} to {data['close'].max():.2f}")
                
                # Show sample data
                print("   Sample data:")
                sample_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                available_cols = [col for col in sample_cols if col in data.columns]
                print(data[available_cols].head(3).to_string(index=False))
                
            else:
                print(f"   ⚠️ No {timeframe} data found for the specified date range")
                
                # Try with longer date range
                print(f"   Trying with longer date range (30 days)...")
                longer_start = end_date - timedelta(days=30)
                
                if timeframe == '5min':
                    data = await local_source.get_intraday_data(test_symbol, longer_start, end_date, timeframe)
                else:
                    data = await local_source.get_stock_data(test_symbol, longer_start, end_date, timeframe)
                
                if not data.empty:
                    print(f"   ✅ Retrieved {len(data)} {timeframe} data points with longer range")
                    print(f"   Date range: {data['datetime'].min()} to {data['datetime'].max()}")
                else:
                    print(f"   ❌ No {timeframe} data found even with longer range")
                
        except Exception as e:
            print(f"   ❌ Error retrieving {timeframe} data: {e}")
    
    # Test data source manager integration
    print(f"\n5. Testing data source manager integration...")
    try:
        # Test with manager (should use local source as primary)
        intraday_data = await manager.get_intraday_data(test_symbol, start_date, end_date, '5min')
        
        if not intraday_data.empty:
            print(f"   ✅ Manager retrieved {len(intraday_data)} 5min data points")
            print("   Data source manager is working with 5-minute data")
        else:
            print("   ⚠️ Manager returned no 5-minute data")
            
    except Exception as e:
        print(f"   ❌ Manager integration error: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 5-minute data testing completed")

def check_5min_data_files():
    """Check if 5-minute data files exist."""
    print("📁 Checking 5-Minute Data Files")
    print("-" * 40)
    
    base_path = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/vipdoc")
    print(f"Base path: {base_path}")
    
    if not os.path.exists(base_path):
        print("❌ Base path does not exist")
        return False
    
    print("✅ Base path exists")
    
    # Check fzline directories
    markets_found = []
    total_lc5_files = 0
    
    for market in ['sz', 'sh']:
        fzline_path = os.path.join(base_path, market, 'fzline')
        if os.path.exists(fzline_path):
            lc5_files = [f for f in os.listdir(fzline_path) if f.endswith('.lc5')]
            if lc5_files:
                markets_found.append(market)
                total_lc5_files += len(lc5_files)
                print(f"✅ {market.upper()} market fzline: {len(lc5_files)} .lc5 files")
            else:
                print(f"⚠️ {market.upper()} market fzline: directory exists but no .lc5 files")
        else:
            print(f"❌ {market.upper()} market fzline: directory not found")
    
    if markets_found:
        print(f"✅ Found 5-minute data for {len(markets_found)} markets: {', '.join(markets_found).upper()}")
        print(f"✅ Total .lc5 files: {total_lc5_files}")
        return True
    else:
        print("❌ No 5-minute data found")
        return False

async def test_data_quality():
    """Test data quality of 5-minute data."""
    print("\n📊 Testing 5-Minute Data Quality")
    print("-" * 40)
    
    try:
        manager = await get_data_source_manager()
        local_source = manager.data_sources[DataSourceType.LOCAL]
        
        # Get a test symbol
        stock_list = await local_source.get_stock_list()
        test_symbol = stock_list.iloc[0]['stock_code']
        
        # Get recent 5-minute data
        end_date = date.today()
        start_date = end_date - timedelta(days=3)
        
        data = await local_source.get_intraday_data(test_symbol, start_date, end_date, '5min')
        
        if data.empty:
            print("⚠️ No data available for quality testing")
            return
        
        print(f"Testing data quality for {test_symbol}")
        print(f"Data points: {len(data)}")
        
        # Check for basic data quality issues
        issues = []
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.any():
            issues.append(f"Missing values found: {missing_values.to_dict()}")
        
        # Check OHLC relationships
        invalid_ohlc = data[
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        ]
        
        if not invalid_ohlc.empty:
            issues.append(f"Invalid OHLC relationships: {len(invalid_ohlc)} records")
        
        # Check for negative values
        negative_prices = data[
            (data['open'] <= 0) |
            (data['high'] <= 0) |
            (data['low'] <= 0) |
            (data['close'] <= 0)
        ]
        
        if not negative_prices.empty:
            issues.append(f"Negative or zero prices: {len(negative_prices)} records")
        
        # Check time sequence
        if not data['datetime'].is_monotonic_increasing:
            issues.append("Time sequence is not monotonic")
        
        if issues:
            print("⚠️ Data quality issues found:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Data quality looks good")
            
        # Show data statistics
        print(f"\nData statistics:")
        print(f"   Time range: {data['datetime'].min()} to {data['datetime'].max()}")
        print(f"   Price range: {data['close'].min():.2f} to {data['close'].max():.2f}")
        print(f"   Average volume: {data['volume'].mean():.0f}")
        
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")

async def main():
    """Main test function."""
    print("🚀 5-Minute Data Integration Test")
    print("=" * 60)
    
    # First check if 5-minute data files exist
    if not check_5min_data_files():
        print("\n💡 To use 5-minute data:")
        print("   1. Open TDX (通达信)")
        print("   2. Download 5-minute data for stocks")
        print("   3. Ensure .lc5 files are in fzline directories")
        return
    
    # Test the 5-minute data functionality
    await test_5min_data_reading()
    
    # Test data quality
    await test_data_quality()

if __name__ == "__main__":
    asyncio.run(main())