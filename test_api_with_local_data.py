#!/usr/bin/env python3
"""Test API endpoints with local data integration."""

import requests
import json
import time
from datetime import datetime

def test_api_with_local_data():
    """Test API endpoints using local data."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing API with Local Data Integration")
    print("=" * 60)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed")
            print(f"   Database: {health_data.get('database', 'unknown')}")
            
            data_sources = health_data.get('data_sources', {})
            print(f"   Data sources: {data_sources.get('total_sources', 0)} total")
            print(f"   Healthy sources: {data_sources.get('healthy_sources', 0)}")
            
            # Show data source details
            sources = data_sources.get('sources', {})
            for source_name, source_info in sources.items():
                status = source_info.get('status', 'unknown')
                reliability = source_info.get('reliability_score', 0)
                print(f"     {source_name}: {status} (reliability: {reliability:.2f})")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test stock list endpoint
    print("\n2. Testing stock list endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/stocks?limit=10", timeout=30)
        if response.status_code == 200:
            stocks_data = response.json()
            print("✅ Stock list retrieved")
            print(f"   Total stocks: {stocks_data.get('total', 0)}")
            print(f"   Returned: {len(stocks_data.get('stocks', []))}")
            
            # Show first few stocks
            for i, stock in enumerate(stocks_data.get('stocks', [])[:5]):
                print(f"   [{i+1}] {stock.get('symbol', 'N/A')} - {stock.get('name', 'N/A')}")
        else:
            print(f"❌ Stock list failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Stock list error: {e}")
    
    # Test stock data endpoint
    print("\n3. Testing stock data endpoint...")
    try:
        # Test with a common stock symbol
        symbol = "000001.SZ"
        
        response = requests.get(
            f"{base_url}/api/v1/stocks/{symbol}/data?days=30",
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
    
    # Test Spring Festival analysis endpoint
    print("\n4. Testing Spring Festival analysis endpoint...")
    try:
        symbol = "000001.SZ"
        
        print(f"   Analyzing {symbol} for Spring Festival patterns...")
        response = requests.get(
            f"{base_url}/api/v1/stocks/{symbol}/spring-festival?years=5",
            timeout=120  # Longer timeout for analysis
        )
        
        if response.status_code == 200:
            analysis = response.json()
            
            if 'error' in analysis:
                print(f"⚠️ Analysis returned error: {analysis['error']}")
                return
            
            print("✅ Spring Festival analysis completed")
            print(f"   Symbol: {analysis.get('symbol', 'N/A')}")
            print(f"   Analysis period: {analysis.get('analysis_period', 'N/A')}")
            print(f"   Data points: {analysis.get('data_points', 0)}")
            
            # Pattern results
            pattern = analysis.get('spring_festival_pattern', {})
            print(f"\n   📊 PATTERN ANALYSIS")
            print(f"      Pattern Strength: {pattern.get('pattern_strength', 0):.3f}")
            print(f"      Confidence Score: {pattern.get('confidence_score', 0):.3f}")
            print(f"      Avg Return Before SF: {pattern.get('average_return_before', 0):+.2f}%")
            print(f"      Avg Return After SF: {pattern.get('average_return_after', 0):+.2f}%")
            print(f"      Volatility Ratio: {pattern.get('volatility_ratio', 0):.2f}x")
            
            # Current position
            position = analysis.get('current_position', {})
            print(f"\n   📅 CURRENT POSITION")
            print(f"      Position: {position.get('position', 'unknown')}")
            print(f"      Days to SF: {position.get('days_to_spring_festival', 'N/A')}")
            print(f"      In Analysis Window: {position.get('in_analysis_window', False)}")
            
            # Trading signals
            signals = analysis.get('trading_signals', {})
            print(f"\n   🚦 TRADING SIGNALS")
            print(f"      Signal: {signals.get('signal', 'unknown').upper()}")
            print(f"      Strength: {signals.get('strength', 0):.2f}")
            print(f"      Recommended Action: {signals.get('recommended_action', 'unknown').upper()}")
            if signals.get('reason'):
                print(f"      Reason: {signals.get('reason', '')}")
            
            # Yearly data
            yearly_data = analysis.get('yearly_data', [])
            if yearly_data:
                print(f"\n   📈 YEARLY BREAKDOWN ({len(yearly_data)} years)")
                for year_info in yearly_data[:3]:  # Show first 3 years
                    year = year_info.get('year', 'N/A')
                    sf_date = year_info.get('spring_festival_date', 'N/A')
                    return_before = year_info.get('return_before', 0)
                    return_after = year_info.get('return_after', 0)
                    print(f"      {year} (SF: {sf_date}): Before {return_before:+.1f}%, After {return_after:+.1f}%")
            
            # Recommendations
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                print(f"\n   💡 RECOMMENDATIONS")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"      {i}. {rec}")
            
        else:
            print(f"❌ Spring Festival analysis failed: {response.status_code}")
            print(f"   Response: {response.text[:300]}")
    except Exception as e:
        print(f"❌ Spring Festival analysis error: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 API testing with local data completed")

def start_server_if_needed():
    """Check if server is running, if not provide instructions."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is already running")
            return True
    except:
        pass
    
    print("❌ Server is not running")
    print("Please start the server with: python start_server.py")
    print("Then run this test again.")
    return False

if __name__ == "__main__":
    print("🚀 API Testing with Local Data Integration")
    print("=" * 60)
    
    if start_server_if_needed():
        print()
        test_api_with_local_data()
    else:
        print("\nServer needs to be started first.")