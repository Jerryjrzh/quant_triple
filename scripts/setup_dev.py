#!/usr/bin/env python3
"""Development environment setup script."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def setup_git_hooks():
    """Set up git hooks."""
    if not Path(".git").exists():
        print("⚠️  Git repository not found. Initializing...")
        if not run_command("git init", "Initialize Git repository"):
            return False
    
    return run_command("pre-commit install", "Install pre-commit hooks")


def create_directories():
    """Create necessary directories."""
    directories = [
        "logs",
        "data/raw",
        "data/processed",
        "data/cache",
        "docs",
        "tests/integration",
        "tests/fixtures",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def setup_environment():
    """Set up environment file."""
    if not Path(".env").exists():
        if Path(".env.example").exists():
            run_command("cp .env.example .env", "Copy environment file")
            print("⚠️  Please edit .env file with your configuration")
        else:
            print("❌ .env.example not found")
            return False
    else:
        print("✅ .env file already exists")
    return True


def main():
    """Main setup function."""
    print("🚀 Setting up development environment for Stock Analysis System")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    create_directories()
    
    # Set up environment
    if not setup_environment():
        sys.exit(1)
    
    # Install dependencies
    if not run_command("pip install -e .[dev]", "Install dependencies"):
        sys.exit(1)
    
    # Set up git hooks
    if not setup_git_hooks():
        print("⚠️  Pre-commit hooks setup failed, but continuing...")
    
    # Run initial code quality checks
    print("\n🔍 Running initial code quality checks...")
    run_command("black --check stock_analysis_system tests", "Check code formatting")
    run_command("isort --check-only stock_analysis_system tests", "Check import sorting")
    run_command("flake8 stock_analysis_system tests", "Run linting")
    
    print("\n🎉 Development environment setup completed!")
    print("\nNext steps:")
    print("1. Edit .env file with your configuration")
    print("2. Set up PostgreSQL and Redis")
    print("3. Run database migrations: alembic upgrade head")
    print("4. Start the development server: uvicorn stock_analysis_system.api.main:app --reload")


if __name__ == "__main__":
    main()