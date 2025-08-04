#!/usr/bin/env python3
"""Script to verify the Docker setup is working correctly."""

import subprocess
import time
import sys


def run_command(command: str, description: str, use_sudo: bool = False) -> bool:
    """Run a command and return success status."""
    if use_sudo:
        command = f"sudo {command}"
    
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} - OK")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        if e.stderr.strip():
            print(f"   Error: {e.stderr.strip()}")
        return False


def main():
    """Main verification function."""
    print("🔍 Verifying Docker Setup for Stock Analysis System")
    print("=" * 60)
    
    # Check Docker installation
    if not run_command("docker --version", "Check Docker installation"):
        print("Please install Docker first")
        sys.exit(1)
    
    if not run_command("docker-compose --version", "Check Docker Compose installation"):
        print("Please install Docker Compose first")
        sys.exit(1)
    
    # Check if we need sudo
    need_sudo = not run_command("docker ps", "Check Docker permissions")
    
    if need_sudo:
        print("⚠️  Docker requires sudo. Using sudo for Docker commands...")
        docker_cmd = "sudo docker-compose"
    else:
        docker_cmd = "docker-compose"
    
    # Start services
    print("\n🚀 Starting Docker services...")
    if not run_command(f"{docker_cmd} up -d postgres redis", "Start PostgreSQL and Redis", need_sudo):
        sys.exit(1)
    
    # Wait for services to be ready
    print("\n⏳ Waiting for services to be ready...")
    time.sleep(10)
    
    # Check service status
    run_command(f"{docker_cmd} ps", "Check service status", need_sudo)
    
    # Check service health
    run_command(f"{docker_cmd} exec postgres pg_isready -U postgres", "Check PostgreSQL health", need_sudo)
    run_command(f"{docker_cmd} exec redis redis-cli ping", "Check Redis health", need_sudo)
    
    print("\n✅ Docker setup verification completed!")
    print("\nNext steps:")
    print("1. Activate your Python virtual environment")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run migrations: alembic upgrade head")
    print("4. Start the app: uvicorn stock_analysis_system.api.main:app --reload")


if __name__ == "__main__":
    main()