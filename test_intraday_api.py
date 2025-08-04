#!/usr/bin/env python3
"""Test script for intraday data API endpoint."""

import requests
import json
from datetime import datetime

def test_intraday_api():
    """Test intraday data API endpoint."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Intraday Data API")
    print("=" * 60)
    
    # Test health endpoint first
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed")
            
            # Check local data source status
            data_sources = health_data.get('data_sources', {})
            sources = data_sources.get('sources', {})
            local_status = sources.get('local', {}).get('status', 'unknown')
            print(f"   Local data source: {local_status}")
            
            if local_status != 'healthy':
                print("⚠️ Local data source not healthy - intraday tests may fail")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test different timeframes
    symbol = "000001.SZ"
    timeframes = ['5min', '15min', '30min', '60min']
    
    for timeframe in timeframes:
        print(f"\n2.{timeframes.index(timeframe)+1} Testing {timeframe} data...")
        
        try:
            response = requests.get(
                f"{base_url}/api/v1/stocks/{symbol}/intraday?timeframe={timeframe}&days=3",
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'error' in data:
                    print(f"⚠️ API returned error: {data['error']}")
                    continue
                
                print(f"✅ {timeframe} data retrieved")
                print(f"   Symbol: {data.get('symbol', 'N/A')}")
                print(f"   Timeframe: {data.get('timeframe', 'N/A')}")
                print(f"   Data points: {data.get('count', 0)}")
                print(f"   Date range: {data.get('start_date', 'N/A')} to {data.get('end_date', 'N/A')}")
                
                # Show sample data
                records = data.get('data', [])
                if records:
                    sample = records[0]
                    print(f"   Sample: {sample.get('datetime', 'N/A')} - "
                          f"O:{sample.get('open', 'N/A')} H:{sample.get('high', 'N/A')} "
                          f"L:{sample.get('low', 'N/A')} C:{sample.get('close', 'N/A')} "
                          f"V:{sample.get('volume', 'N/A')}")
                    
                    # Show last few records to see time progression
                    if len(records) > 3:
                        print(f"   Recent data:")
                        for record in records[-3:]:
                            dt = record.get('datetime', 'N/A')
                            close = record.get('close', 'N/A')
                            volume = record.get('volume', 'N/A')
                            print(f"     {dt}: Close {close}, Volume {volume}")
                
            else:
                print(f"❌ {timeframe} data failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"❌ {timeframe} data error: {e}")
    
    # Test invalid timeframe
    print(f"\n3. Testing invalid timeframe...")
    try:
        response = requests.get(
            f"{base_url}/api/v1/stocks/{symbol}/intraday?timeframe=invalid&days=1",
            timeout=30
        )
        
        if response.status_code == 400:
            print("✅ Invalid timeframe correctly rejected")
            error_data = response.json()
            print(f"   Error message: {error_data.get('detail', 'N/A')}")
        else:
            print(f"⚠️ Expected 400 error, got {response.status_code}")
            
    except Exception as e:
        print(f"❌ Invalid timeframe test error: {e}")
    
    # Test different date ranges
    print(f"\n4. Testing different date ranges...")
    
    date_ranges = [1, 7, 30]  # 1 day, 1 week, 1 month
    
    for days in date_ranges:
        try:
            response = requests.get(
                f"{base_url}/api/v1/stocks/{symbol}/intraday?timeframe=5min&days={days}",
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                count = data.get('count', 0)
                print(f"   {days} days: {count} data points")
            else:
                print(f"   {days} days: Failed ({response.status_code})")
                
        except Exception as e:
            print(f"   {days} days: Error - {e}")
    
    # Test with different symbols
    print(f"\n5. Testing different symbols...")
    
    test_symbols = ["000001.SZ", "000002.SZ", "600000.SH"]
    
    for test_symbol in test_symbols:
        try:
            response = requests.get(
                f"{base_url}/api/v1/stocks/{test_symbol}/intraday?timeframe=5min&days=1",
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                count = data.get('count', 0)
                print(f"   {test_symbol}: {count} data points")
            else:
                print(f"   {test_symbol}: Failed ({response.status_code})")
                
        except Exception as e:
            print(f"   {test_symbol}: Error - {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Intraday data API testing completed")

def start_server_if_needed():
    """Check if server is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
    except:
        pass
    
    print("❌ Server is not running")
    print("Please start the server with: python start_server.py")
    return False

if __name__ == "__main__":
    print("🚀 Intraday Data API Testing")
    print("=" * 60)
    
    if start_server_if_needed():
        print()
        test_intraday_api()
    else:
        print("\nServer needs to be started first.")