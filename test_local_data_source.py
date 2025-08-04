#!/usr/bin/env python3
"""Test script for local data source integration."""

import asyncio
import os
from datetime import date, timedelta
from stock_analysis_system.data.data_source_manager import get_data_source_manager, DataSourceType

async def test_local_data_source():
    """Test local data source functionality."""
    print("🧪 Testing Local Data Source Integration")
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
            print(f"   Base path: {local_source.base_path}")
            print("   Please check if TDX data files exist in the specified path")
            return
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return
    
    # Test stock list
    print("\n2. Testing stock list retrieval...")
    try:
        stock_list = await local_source.get_stock_list()
        if not stock_list.empty:
            print(f"✅ Found {len(stock_list)} stocks in local data")
            print("   Sample stocks:")
            for i, row in stock_list.head(5).iterrows():
                print(f"     {row['stock_code']} - {row['name']} ({row['market']})")
        else:
            print("⚠️ No stocks found in local data")
            return
    except Exception as e:
        print(f"❌ Stock list retrieval error: {e}")
        return
    
    # Test stock data retrieval
    print("\n3. Testing stock data retrieval...")
    
    # Use the first available stock
    test_symbol = stock_list.iloc[0]['stock_code']
    print(f"   Testing with symbol: {test_symbol}")
    
    # Test recent data (last 30 days)
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    
    try:
        stock_data = await local_source.get_stock_data(test_symbol, start_date, end_date)
        
        if not stock_data.empty:
            print(f"✅ Retrieved {len(stock_data)} data points")
            print(f"   Date range: {stock_data['trade_date'].min().date()} to {stock_data['trade_date'].max().date()}")
            print(f"   Price range: {stock_data['close_price'].min():.2f} to {stock_data['close_price'].max():.2f}")
            
            # Show sample data
            print("\n   Sample data:")
            print(stock_data[['trade_date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']].head(3).to_string(index=False))
            
        else:
            print("⚠️ No data found for the specified date range")
            
            # Try with a longer date range
            print("   Trying with longer date range (1 year)...")
            start_date = end_date - timedelta(days=365)
            stock_data = await local_source.get_stock_data(test_symbol, start_date, end_date)
            
            if not stock_data.empty:
                print(f"✅ Retrieved {len(stock_data)} data points with longer range")
                print(f"   Date range: {stock_data['trade_date'].min().date()} to {stock_data['trade_date'].max().date()}")
            else:
                print("❌ No data found even with longer date range")
                
    except Exception as e:
        print(f"❌ Stock data retrieval error: {e}")
        return
    
    # Test data source manager integration
    print("\n4. Testing data source manager integration...")
    try:
        # Test with manager (should use local source as primary)
        manager_data = await manager.get_stock_data(test_symbol, start_date, end_date)
        
        if not manager_data.empty:
            print(f"✅ Manager retrieved {len(manager_data)} data points")
            print("   Data source manager is working with local data")
        else:
            print("⚠️ Manager returned no data")
            
    except Exception as e:
        print(f"❌ Manager integration error: {e}")
    
    # Test health check
    print("\n5. Testing health check...")
    try:
        health_status = await manager.health_check()
        local_health = health_status.get(DataSourceType.LOCAL)
        
        if local_health:
            print(f"✅ Local data source health: {local_health.status.value}")
            print(f"   Reliability score: {local_health.reliability_score:.2f}")
        else:
            print("❌ No health status for local data source")
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Local data source testing completed")

def check_local_data_files():
    """Check if local data files exist."""
    print("📁 Checking Local Data Files")
    print("-" * 40)
    
    base_path = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/vipdoc")
    print(f"Base path: {base_path}")
    
    if not os.path.exists(base_path):
        print("❌ Base path does not exist")
        print("   Please ensure TDX data is available at the expected location")
        return False
    
    print("✅ Base path exists")
    
    # Check market directories
    markets_found = []
    for market in ['sz', 'sh']:
        market_path = os.path.join(base_path, market, 'lday')
        if os.path.exists(market_path):
            day_files = [f for f in os.listdir(market_path) if f.endswith('.day')]
            if day_files:
                markets_found.append(market)
                print(f"✅ {market.upper()} market: {len(day_files)} data files")
            else:
                print(f"⚠️ {market.upper()} market: directory exists but no .day files")
        else:
            print(f"❌ {market.upper()} market: directory not found")
    
    if markets_found:
        print(f"✅ Found data for {len(markets_found)} markets: {', '.join(markets_found).upper()}")
        return True
    else:
        print("❌ No market data found")
        return False

async def main():
    """Main test function."""
    print("🚀 Local Data Source Integration Test")
    print("=" * 60)
    
    # First check if data files exist
    if not check_local_data_files():
        print("\n💡 To use local data source:")
        print("   1. Install and run TDX (通达信)")
        print("   2. Download stock data in TDX")
        print("   3. Ensure data files are in the expected location")
        print("   4. Or modify the base_path in LocalDataSource")
        return
    
    # Test the local data source
    await test_local_data_source()

if __name__ == "__main__":
    asyncio.run(main())