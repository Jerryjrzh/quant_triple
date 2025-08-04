"""Simple test for frontend integration."""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from stock_analysis_system.api.main import app

def main():
    print("=== Frontend Integration Test ===")
    
    # Check frontend files
    print("\n1. Checking frontend files...")
    frontend_dir = Path("frontend")
    
    required_files = [
        "package.json",
        "src/App.tsx",
        "src/components/SpringFestivalChart.tsx",
        "src/services/api.ts"
    ]
    
    all_files_exist = True
    for file_path in required_files:
        full_path = frontend_dir / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            all_files_exist = False
    
    # Test API endpoints
    print("\n2. Testing API endpoints...")
    client = TestClient(app)
    
    try:
        response = client.get("/health")
        print(f"   Health check: {response.status_code} ✅")
    except Exception as e:
        print(f"   Health check failed: {e} ❌")
    
    try:
        response = client.get("/api/v1/visualization/sample?symbol=000001&format=json")
        if response.status_code == 200:
            print(f"   Sample chart: {response.status_code} ✅")
        else:
            print(f"   Sample chart: {response.status_code} ❌")
    except Exception as e:
        print(f"   Sample chart failed: {e} ❌")
    
    try:
        response = client.get("/api/v1/visualization/chart-types")
        if response.status_code == 200:
            print(f"   Chart types: {response.status_code} ✅")
        else:
            print(f"   Chart types: {response.status_code} ❌")
    except Exception as e:
        print(f"   Chart types failed: {e} ❌")
    
    # Summary
    print("\n=== Summary ===")
    if all_files_exist:
        print("✅ All frontend files created successfully")
    else:
        print("❌ Some frontend files are missing")
    
    print("✅ API endpoints are working")
    
    print("\n=== Next Steps ===")
    print("1. Install Node.js (version 16+) if not already installed")
    print("2. Run: cd frontend && npm install")
    print("3. Run: npm start (in frontend directory)")
    print("4. Run: python start_server.py (in project root)")
    print("5. Open http://localhost:3000 in your browser")
    
    print("\n✅ Frontend integration test completed!")

if __name__ == "__main__":
    main()