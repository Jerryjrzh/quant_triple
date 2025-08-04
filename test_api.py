#!/usr/bin/env python3
"""Simple script to test the API endpoints."""

import requests
import json
import sys


def test_endpoint(url: str, description: str) -> bool:
    """Test an API endpoint."""
    try:
        print(f"🔄 Testing {description}...")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ {description} - OK")
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ {description} - Failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ {description} - Connection failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Testing Stock Analysis System API")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    endpoints = [
        (f"{base_url}/", "Root endpoint"),
        (f"{base_url}/health", "Health check"),
        (f"{base_url}/api/v1/info", "API info"),
        (f"{base_url}/api/v1/stocks", "List stocks"),
        (f"{base_url}/api/v1/stocks?search=银行", "Search stocks"),
        (f"{base_url}/api/v1/stocks/000001.SZ", "Get stock info"),
        (f"{base_url}/api/v1/stocks/000001.SZ/spring-festival", "Spring Festival analysis"),
        (f"{base_url}/api/v1/screening", "Stock screening"),
        (f"{base_url}/api/v1/screening?sector=金融&max_pe_ratio=10", "Stock screening with filters"),
    ]
    
    all_passed = True
    
    for url, description in endpoints:
        success = test_endpoint(url, description)
        if not success:
            all_passed = False
        print()
    
    # Test protected endpoint (should fail without auth)
    print("🔄 Testing protected endpoint (should fail without auth)...")
    try:
        response = requests.get(f"{base_url}/api/v1/alerts", timeout=10)
        if response.status_code in [401, 403]:  # Both are acceptable for missing auth
            print("✅ Protected endpoint correctly requires authentication")
        else:
            print(f"❌ Protected endpoint should return 401/403, got {response.status_code}")
            all_passed = False
    except requests.exceptions.RequestException as e:
        print(f"❌ Protected endpoint test failed: {e}")
        all_passed = False
    print()
    
    if all_passed:
        print("🎉 All API tests passed!")
        return 0
    else:
        print("❌ Some API tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())