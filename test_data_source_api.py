#!/usr/bin/env python3
"""Test script for data source manager API integration."""

import asyncio
import requests
import json
from datetime import datetime, timedelta

def test_api_endpoints():
    """Test API endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Stock Analysis System API")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed")
            print(f"   Database: {health_data.get('database', 'unknown')}")
            print(f"   Data sources: {health_data.get('data_sources', {}).get('total_sources', 0)} total")
            print(f"   Healthy sources: {health_data.get('data_sources', {}).get('healthy_sources', 0)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test API info endpoint
    print("\n2. Testing API info endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/info", timeout=10)
        if response.status_code == 200:
            info_data = response.json()
            print("✅ API info retrieved")
            print(f"   App: {info_data.get('app_name', 'unknown')}")
            print(f"   Version: {info_data.get('app_version', 'unknown')}")
            print(f"   Features: {len(info_data.get('features', []))} available")
        else:
            print(f"❌ API info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API info error: {e}")
    
    # Test stock list endpoint
    print("\n3. Testing stock list endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/stocks?limit=5", timeout=30)
        if response.status_code == 200:
            stocks_data = response.json()
            print("✅ Stock list retrieved")
            print(f"   Total stocks: {stocks_data.get('total', 0)}")
            print(f"   Returned: {len(stocks_data.get('stocks', []))}")
            
            # Show first few stocks
            for i, stock in enumerate(stocks_data.get('stocks', [])[:3]):
                print(f"   [{i+1}] {stock.get('symbol', 'N/A')} - {stock.get('name', 'N/A')}")
        else:
            print(f"❌ Stock list failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Stock list error: {e}")
    
    # Test stock data endpoint
    print("\n4. Testing stock data endpoint...")
    try:
        # Test with a common stock symbol
        symbol = "000001.SZ"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        response = requests.get(
            f"{base_url}/api/v1/stocks/{symbol}/data?start_date={start_date}&end_date={end_date}",
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Stock data retrieved")
            print(f"   Symbol: {data.get('symbol', 'N/A')}")
            print(f"   Data points: {data.get('count', 0)}")
            print(f"   Date range: {data.get('start_date', 'N/A')} to {data.get('end_date', 'N/A')}")
            
            # Show sample data point
            if data.get('data') and len(data['data']) > 0:
                sample = data['data'][0]
                print(f"   Sample: {sample.get('date', 'N/A')} - Close: {sample.get('close', 'N/A')}")
        else:
            print(f"❌ Stock data failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Stock data error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 API testing completed")


if __name__ == "__main__":
    print("Starting API tests...")
    print("Make sure the server is running with: python start_server.py")
    print()
    
    test_api_endpoints()