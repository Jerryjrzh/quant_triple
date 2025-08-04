#!/usr/bin/env python3
"""Script to start the Stock Analysis System API server."""

import os
import sys
import subprocess
import time
import requests


def check_dependencies():
    """Check if required services are running."""
    print("🔍 Checking dependencies...")
    
    # Check if Docker containers are running
    try:
        result = subprocess.run(
            ["sudo", "docker-compose", "ps", "--format", "table"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if "postgres" in result.stdout and "redis" in result.stdout:
            print("✅ Docker containers (PostgreSQL & Redis) are running")
        else:
            print("❌ Docker containers are not running")
            print("Run: sudo docker-compose up -d postgres redis")
            return False
            
    except subprocess.CalledProcessError:
        print("❌ Could not check Docker containers")
        return False
    
    return True


def start_server():
    """Start the API server."""
    if not check_dependencies():
        return False
    
    print("🚀 Starting Stock Analysis System API server...")
    
    # Set environment variables
    env = os.environ.copy()
    env.update({
        "DB_HOST": "localhost",
        "DB_PORT": "5432", 
        "DB_NAME": "stock_analysis",
        "DB_USER": "postgres",
        "DB_PASSWORD": "password"
    })
    
    # Start the server
    cmd = [
        "uvicorn",
        "stock_analysis_system.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    try:
        print(f"Running: {' '.join(cmd)}")
        print("Server will be available at: http://localhost:8000")
        print("Health check: http://localhost:8000/health")
        print("API info: http://localhost:8000/api/v1/info")
        print("\nPress Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run(cmd, env=env, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        return False


def test_server():
    """Test if the server is responding."""
    print("🧪 Testing server endpoints...")
    
    base_url = "http://localhost:8000"
    endpoints = [
        ("/", "Root endpoint"),
        ("/health", "Health check"),
        ("/api/v1/info", "API info")
    ]
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    for i in range(10):
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!")
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        print("❌ Server did not start in time")
        return False
    
    # Test all endpoints
    all_passed = True
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {description} - OK")
                data = response.json()
                print(f"   Response: {data}")
            else:
                print(f"❌ {description} - Failed ({response.status_code})")
                all_passed = False
        except requests.exceptions.RequestException as e:
            print(f"❌ {description} - Error: {e}")
            all_passed = False
        print()
    
    return all_passed


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode - start server in background and test
        import threading
        import signal
        
        # Start server in background thread
        server_thread = threading.Thread(target=start_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Test the server
        time.sleep(3)  # Give server time to start
        success = test_server()
        
        # Stop the server
        os.kill(os.getpid(), signal.SIGINT)
        
        sys.exit(0 if success else 1)
    else:
        # Normal mode - start server
        success = start_server()
        sys.exit(0 if success else 1)