# Development Guide

## 📋 Overview

This guide provides comprehensive instructions for setting up a development environment, understanding the codebase, and contributing to the Stock Analysis System. It covers everything from initial setup to advanced development workflows.

## 🚀 Quick Start

### Prerequisites

```bash
# Required software versions
Python >= 3.9 (recommended: 3.12)
Node.js >= 16 (recommended: 18)
Docker >= 20.10
Docker Compose >= 2.0
Git >= 2.30
```

### One-Command Setup

```bash
# Clone and setup everything
git clone https://github.com/your-org/stock-analysis-system.git
cd stock-analysis-system
make setup-dev
```

### Manual Setup

```bash
# 1. Clone repository
git clone https://github.com/your-org/stock-analysis-system.git
cd stock-analysis-system

# 2. Setup Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Setup environment variables
cp .env.example .env
# Edit .env with your configuration

# 4. Start infrastructure services
docker-compose up -d postgres redis

# 5. Run database migrations
alembic upgrade head

# 6. Setup frontend
cd frontend
npm install
cd ..

# 7. Verify setup
python verify_setup.py
```

## 🏗️ Development Environment

### Directory Structure

```
stock-analysis-system/
├── .github/                    # GitHub workflows and templates
├── .kiro/                      # Kiro AI assistant configuration
├── alembic/                    # Database migrations
├── config/                     # Configuration files
├── data/                       # Development data storage
├── docs/                       # Documentation
├── frontend/                   # React frontend application
├── logs/                       # Application logs
├── scripts/                    # Utility scripts
├── stock_analysis_system/      # Main Python package
├── tests/                      # Test suite
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
├── .pre-commit-config.yaml    # Pre-commit hooks
├── docker-compose.yml         # Development services
├── Dockerfile                 # Container definition
├── Makefile                   # Development commands
├── pyproject.toml            # Python project configuration
├── README.md                 # Project overview
└── requirements.txt          # Python dependencies
```

### Development Services

```yaml
# docker-compose.yml services for development
services:
  postgres:    # Database server
  redis:       # Cache and message broker
  celery:      # Background task worker (optional)
  flower:      # Celery monitoring (optional)
```

### Environment Configuration

#### Development Environment (`.env`)

```bash
# Application
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database (using Docker services)
DATABASE_URL=postgresql://postgres:password@localhost:5432/stock_analysis
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock_analysis
DB_USER=postgres
DB_PASSWORD=password

# Redis (using Docker services)
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1

# Data Sources (get your own tokens)
TUSHARE_TOKEN=your_tushare_token_here
AKSHARE_TIMEOUT=30

# Security (development keys)
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET_KEY=dev-jwt-secret-change-in-production

# Development Features
ENABLE_DEBUG_TOOLBAR=true
ENABLE_PROFILING=true
MOCK_EXTERNAL_APIS=false
```

## 🛠️ Development Workflow

### Daily Development Commands

```bash
# Start development environment
make dev-start

# Run backend API server
make run-api

# Run frontend development server
make run-frontend

# Run background workers
make run-celery

# Run tests
make test

# Code formatting and linting
make lint
make format

# Database operations
make db-upgrade    # Apply migrations
make db-downgrade  # Rollback migrations
make db-reset      # Reset database

# Stop development environment
make dev-stop
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add new feature"

# Push and create pull request
git push origin feature/your-feature-name
# Create PR on GitHub

# After PR approval, merge and cleanup
git checkout main
git pull origin main
git branch -d feature/your-feature-name
```

### Code Quality Tools

#### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203,W503"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ["-r", "stock_analysis_system"]

  - repo: https://github.com/gitguardian/ggshield
    rev: v1.25.0
    hooks:
      - id: ggshield
        language: python
        stages: [commit]
```

#### Setup Pre-commit

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/                      # Unit tests
│   ├── test_data/            # Data layer tests
│   ├── test_analysis/        # Analysis engine tests
│   ├── test_api/             # API tests
│   └── test_utils/           # Utility tests
├── integration/              # Integration tests
│   ├── test_workflows/       # End-to-end workflows
│   └── test_external/        # External API integration
├── fixtures/                 # Test data and fixtures
├── conftest.py              # Pytest configuration
└── __init__.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=stock_analysis_system --cov-report=html

# Run specific test categories
pytest -m unit                # Unit tests only
pytest -m integration         # Integration tests only
pytest -m slow               # Slow tests only

# Run specific test files
pytest tests/test_spring_festival_engine.py

# Run with verbose output
pytest -v

# Run with debugging
pytest -s --pdb

# Run tests in parallel
pytest -n auto
```

### Writing Tests

#### Unit Test Example

```python
# tests/unit/test_spring_festival_engine.py
import pytest
import pandas as pd
from datetime import date
from stock_analysis_system.analysis.spring_festival_engine import SpringFestivalAlignmentEngine

class TestSpringFestivalEngine:
    @pytest.fixture
    def engine(self):
        return SpringFestivalAlignmentEngine(window_days=60)
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'stock_code': ['000001.SZ'] * 100,
            'trade_date': pd.date_range('2023-01-01', periods=100),
            'close_price': range(100, 200),
            'volume': range(1000, 1100)
        })
    
    def test_spring_festival_date_calculation(self, engine):
        """Test Spring Festival date calculation for known years."""
        assert engine.chinese_calendar.get_spring_festival(2024) == date(2024, 2, 10)
        assert engine.chinese_calendar.get_spring_festival(2023) == date(2023, 1, 22)
    
    def test_data_alignment(self, engine, sample_data):
        """Test data alignment to Spring Festival dates."""
        aligned_data = engine.align_to_spring_festival(sample_data, years=[2023])
        
        assert aligned_data is not None
        assert len(aligned_data.data_points) > 0
        assert all(dp.year == 2023 for dp in aligned_data.data_points)
    
    def test_pattern_identification(self, engine, sample_data):
        """Test seasonal pattern identification."""
        aligned_data = engine.align_to_spring_festival(sample_data, years=[2023])
        pattern = engine.identify_seasonal_patterns(aligned_data)
        
        assert pattern.pattern_strength >= 0.0
        assert pattern.pattern_strength <= 1.0
        assert pattern.confidence_level >= 0.0
        assert pattern.confidence_level <= 1.0
    
    @pytest.mark.parametrize("window_days", [30, 60, 90])
    def test_different_window_sizes(self, sample_data, window_days):
        """Test engine with different window sizes."""
        engine = SpringFestivalAlignmentEngine(window_days=window_days)
        aligned_data = engine.align_to_spring_festival(sample_data, years=[2023])
        
        assert aligned_data.window_days == window_days
```

#### Integration Test Example

```python
# tests/integration/test_api_workflow.py
import pytest
from fastapi.testclient import TestClient
from stock_analysis_system.api.main import app

class TestAPIWorkflow:
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_complete_analysis_workflow(self, client):
        """Test complete workflow from search to analysis."""
        
        # 1. Search for stocks
        response = client.get("/api/v1/stocks?q=000001")
        assert response.status_code == 200
        stocks = response.json()["stocks"]
        assert len(stocks) > 0
        
        # 2. Get stock details
        symbol = stocks[0]["symbol"]
        response = client.get(f"/api/v1/stocks/{symbol}")
        assert response.status_code == 200
        
        # 3. Perform Spring Festival analysis
        analysis_request = {
            "symbol": symbol,
            "years": [2022, 2023, 2024],
            "window_days": 60
        }
        response = client.post("/api/v1/analysis/spring-festival", json=analysis_request)
        assert response.status_code == 200
        
        analysis = response.json()
        assert "seasonal_pattern" in analysis
        assert "trading_signals" in analysis
        
        # 4. Generate visualization
        viz_request = {
            "symbol": symbol,
            "years": [2022, 2023, 2024],
            "chart_type": "overlay"
        }
        response = client.post("/api/v1/visualization/spring-festival-chart", json=viz_request)
        assert response.status_code == 200
        
        chart_data = response.json()
        assert "chart_data" in chart_data
```

### Test Configuration

```python
# tests/conftest.py
import pytest
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from stock_analysis_system.core.database import Base, get_session
from stock_analysis_system.api.main import app

# Test database
TEST_DATABASE_URL = "sqlite:///./test.db"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine."""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_session(test_engine):
    """Create test database session."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    yield session
    session.close()

@pytest.fixture
def override_get_session(test_session):
    """Override database session for testing."""
    def _override_get_session():
        yield test_session
    
    app.dependency_overrides[get_session] = _override_get_session
    yield
    app.dependency_overrides.clear()
```

## 🔧 Development Tools

### IDE Configuration

#### VS Code Settings

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".coverage": true,
        "htmlcov": true
    }
}
```

#### VS Code Extensions

```json
// .vscode/extensions.json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.flake8",
        "ms-python.mypy-type-checker",
        "ms-python.black-formatter",
        "ms-python.isort",
        "bradlc.vscode-tailwindcss",
        "esbenp.prettier-vscode",
        "ms-vscode.vscode-typescript-next",
        "ms-vscode-remote.remote-containers",
        "ms-azuretools.vscode-docker"
    ]
}
```

### Debugging

#### Python Debugging

```python
# Debug configuration for VS Code
# .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "FastAPI Server",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/venv/bin/uvicorn",
            "args": [
                "stock_analysis_system.api.main:app",
                "--reload",
                "--host", "0.0.0.0",
                "--port", "8000"
            ],
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env"
        },
        {
            "name": "Pytest",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${file}"],
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env"
        }
    ]
}
```

#### Remote Debugging

```python
# Add to your code for remote debugging
import debugpy
debugpy.listen(("0.0.0.0", 5678))
debugpy.wait_for_client()  # Optional: wait for debugger to attach
```

### Performance Profiling

#### Memory Profiling

```python
# memory_profiler usage
from memory_profiler import profile

@profile
def analyze_large_dataset():
    # Your code here
    pass

# Run with: python -m memory_profiler your_script.py
```

#### CPU Profiling

```python
# cProfile usage
import cProfile
import pstats

def profile_function():
    pr = cProfile.Profile()
    pr.enable()
    
    # Your code here
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

#### Line Profiling

```python
# line_profiler usage
@profile
def slow_function():
    # Your code here
    pass

# Run with: kernprof -l -v your_script.py
```

## 📊 Database Development

### Migration Management

```bash
# Create new migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback migrations
alembic downgrade -1

# Show migration history
alembic history

# Show current revision
alembic current
```

### Database Schema Changes

```python
# Example migration file
"""Add user preferences table

Revision ID: abc123
Revises: def456
Create Date: 2025-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'abc123'
down_revision = 'def456'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'user_preferences',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(50), nullable=False),
        sa.Column('preferences', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_user_preferences_user_id', 'user_id')
    )

def downgrade():
    op.drop_table('user_preferences')
```

### Database Testing

```python
# Database test utilities
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from stock_analysis_system.data.models import Base

@pytest.fixture
def db_session():
    # Create in-memory SQLite database for testing
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()
```

## 🎨 Frontend Development

### Setup and Commands

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test

# Run linting
npm run lint

# Fix linting issues
npm run lint:fix
```

### Component Development

```typescript
// Example component with TypeScript
import React, { useState, useEffect } from 'react';
import { Card, Spin, Alert } from 'antd';
import { StockInfo, ChartConfig } from '../services/api';

interface StockChartProps {
  stock: StockInfo | null;
  config: ChartConfig;
}

export const StockChart: React.FC<StockChartProps> = ({ stock, config }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chartData, setChartData] = useState<any>(null);

  useEffect(() => {
    if (!stock) return;

    const fetchChartData = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await api.getSpringFestivalChart({
          symbol: stock.symbol,
          years: config.years,
          chartType: config.chartType
        });
        setChartData(response.chartData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchChartData();
  }, [stock, config]);

  if (loading) {
    return (
      <Card>
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <Spin size="large" tip="正在生成图表..." />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <Alert
          message="图表加载失败"
          description={error}
          type="error"
          showIcon
        />
      </Card>
    );
  }

  return (
    <Card title={`${stock?.name} 春节分析图表`}>
      <div dangerouslySetInnerHTML={{ __html: chartData }} />
    </Card>
  );
};
```

### State Management

```typescript
// Redux store setup
import { configureStore } from '@reduxjs/toolkit';
import { stockSlice } from './slices/stockSlice';
import { chartSlice } from './slices/chartSlice';

export const store = configureStore({
  reducer: {
    stock: stockSlice.reducer,
    chart: chartSlice.reducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
```

## 🔌 API Development

### Adding New Endpoints

```python
# Example: Adding a new API endpoint
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from stock_analysis_system.analysis.spring_festival_engine import SpringFestivalAlignmentEngine

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])

class AnalysisRequest(BaseModel):
    symbol: str
    years: List[int]
    window_days: int = 60

class AnalysisResponse(BaseModel):
    symbol: str
    pattern_strength: float
    confidence_level: float
    trading_signals: dict

@router.post("/custom-analysis", response_model=AnalysisResponse)
async def custom_analysis(
    request: AnalysisRequest,
    engine: SpringFestivalAlignmentEngine = Depends(get_analysis_engine)
):
    """Perform custom Spring Festival analysis."""
    try:
        # Fetch data
        data = await get_stock_data(request.symbol, request.years)
        
        # Perform analysis
        aligned_data = engine.align_to_spring_festival(data, request.years)
        pattern = engine.identify_seasonal_patterns(aligned_data)
        signals = engine.generate_trading_signals(pattern)
        
        return AnalysisResponse(
            symbol=request.symbol,
            pattern_strength=pattern.pattern_strength,
            confidence_level=pattern.confidence_level,
            trading_signals=signals
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### API Testing

```python
# API test example
from fastapi.testclient import TestClient
from stock_analysis_system.api.main import app

client = TestClient(app)

def test_custom_analysis_endpoint():
    """Test custom analysis endpoint."""
    request_data = {
        "symbol": "000001.SZ",
        "years": [2022, 2023, 2024],
        "window_days": 60
    }
    
    response = client.post("/api/v1/analysis/custom-analysis", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "pattern_strength" in data
    assert "confidence_level" in data
    assert "trading_signals" in data
```

## 🔄 Background Tasks

### Celery Development

```python
# Adding new Celery tasks
from celery import Celery
from stock_analysis_system.etl.celery_app import celery_app

@celery_app.task(bind=True, max_retries=3)
def custom_analysis_task(self, symbol: str, years: list):
    """Custom analysis background task."""
    try:
        # Your task logic here
        result = perform_analysis(symbol, years)
        return result
    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))

# Task monitoring
@celery_app.task
def monitor_task_health():
    """Monitor Celery task health."""
    inspector = celery_app.control.inspect()
    stats = inspector.stats()
    return stats
```

### Task Testing

```python
# Testing Celery tasks
import pytest
from unittest.mock import patch
from stock_analysis_system.etl.tasks import custom_analysis_task

def test_custom_analysis_task():
    """Test custom analysis task."""
    with patch('stock_analysis_system.etl.tasks.perform_analysis') as mock_analysis:
        mock_analysis.return_value = {"status": "success"}
        
        result = custom_analysis_task.apply(args=["000001.SZ", [2023, 2024]])
        
        assert result.successful()
        assert result.result["status"] == "success"
```

## 📈 Performance Optimization

### Database Optimization

```python
# Query optimization examples
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

# Efficient querying with joins
def get_stock_with_data(session, symbol: str):
    stmt = (
        select(Stock)
        .options(selectinload(Stock.daily_data))
        .where(Stock.symbol == symbol)
    )
    return session.execute(stmt).scalar_one_or_none()

# Bulk operations
def bulk_insert_data(session, data_list):
    session.bulk_insert_mappings(StockDailyData, data_list)
    session.commit()

# Aggregation queries
def get_stock_statistics(session, symbol: str):
    stmt = (
        select(
            func.avg(StockDailyData.close_price).label('avg_price'),
            func.max(StockDailyData.close_price).label('max_price'),
            func.min(StockDailyData.close_price).label('min_price'),
            func.count(StockDailyData.id).label('record_count')
        )
        .where(StockDailyData.stock_code == symbol)
    )
    return session.execute(stmt).first()
```

### Caching Strategies

```python
# Redis caching implementation
import redis
import json
from functools import wraps

redis_client = redis.Redis.from_url(REDIS_URL)

def cache_result(expiration=3600):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            
            return result
        return wrapper
    return decorator

# Usage
@cache_result(expiration=1800)  # 30 minutes
async def expensive_analysis(symbol: str, years: list):
    # Expensive computation here
    return result
```

### Memory Optimization

```python
# Memory-efficient data processing
import pandas as pd
from typing import Iterator

def process_large_dataset_chunked(file_path: str, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
    """Process large datasets in chunks to manage memory."""
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process chunk
        processed_chunk = chunk.copy()
        # Your processing logic here
        yield processed_chunk

# Memory monitoring
import psutil
import os

def get_memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }
```

## 🐛 Debugging and Troubleshooting

### Common Issues

#### Import Errors
```bash
# Fix Python path issues
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Or add to your script
import sys
sys.path.append('.')
```

#### Database Connection Issues
```python
# Debug database connections
from sqlalchemy import create_engine, text

engine = create_engine(DATABASE_URL, echo=True)  # echo=True for SQL logging
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print(result.fetchone())
```

#### Async/Await Issues
```python
# Common async patterns
import asyncio

# Running async functions in sync context
def sync_function():
    result = asyncio.run(async_function())
    return result

# Proper async context manager usage
async def async_function():
    async with async_context_manager() as resource:
        # Use resource
        pass
```

### Logging and Monitoring

```python
# Structured logging setup
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

# Setup logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info("Processing started", extra={'symbol': '000001.SZ', 'years': [2023, 2024]})
```

## 📚 Documentation

### Code Documentation

```python
# Docstring standards (Google style)
def analyze_spring_festival_pattern(
    stock_data: pd.DataFrame,
    years: List[int],
    window_days: int = 60
) -> SeasonalPattern:
    """Analyze Spring Festival seasonal patterns for a stock.
    
    This function performs temporal alignment of stock price data relative to
    Spring Festival dates and identifies recurring seasonal patterns using
    machine learning techniques.
    
    Args:
        stock_data: DataFrame containing stock OHLCV data with columns:
            - trade_date: Trading date
            - close_price: Closing price
            - volume: Trading volume
        years: List of years to include in the analysis
        window_days: Number of days before/after Spring Festival to analyze
    
    Returns:
        SeasonalPattern object containing:
            - pattern_strength: Float between 0-1 indicating pattern consistency
            - confidence_level: Statistical confidence in the pattern
            - average_return_before: Average return before Spring Festival
            - average_return_after: Average return after Spring Festival
    
    Raises:
        ValueError: If stock_data is empty or years list is invalid
        DataQualityError: If data quality is insufficient for analysis
    
    Example:
        >>> data = get_stock_data("000001.SZ", start_date, end_date)
        >>> pattern = analyze_spring_festival_pattern(data, [2022, 2023, 2024])
        >>> print(f"Pattern strength: {pattern.pattern_strength:.2f}")
    """
    # Implementation here
    pass
```

### API Documentation

```python
# FastAPI automatic documentation
from fastapi import FastAPI, Query, Path
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(
    title="Stock Analysis System API",
    description="Comprehensive stock analysis with Spring Festival temporal patterns",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class AnalysisRequest(BaseModel):
    """Request model for Spring Festival analysis."""
    
    symbol: str = Field(
        ...,
        description="Stock symbol in format XXXXXX.XX (e.g., 000001.SZ)",
        example="000001.SZ",
        regex=r"^\d{6}\.(SZ|SH)$"
    )
    years: List[int] = Field(
        ...,
        description="List of years to include in analysis",
        example=[2022, 2023, 2024],
        min_items=1,
        max_items=10
    )
    window_days: Optional[int] = Field(
        60,
        description="Number of days before/after Spring Festival to analyze",
        ge=30,
        le=120
    )

@app.post(
    "/api/v1/analysis/spring-festival",
    response_model=AnalysisResponse,
    summary="Perform Spring Festival Analysis",
    description="Analyze stock performance patterns relative to Spring Festival dates",
    tags=["Analysis"]
)
async def spring_festival_analysis(
    request: AnalysisRequest,
    include_signals: bool = Query(
        True,
        description="Include trading signals in response"
    )
):
    """
    Perform comprehensive Spring Festival temporal analysis.
    
    This endpoint analyzes stock price patterns relative to Chinese New Year
    dates, identifying seasonal trends and generating trading signals.
    """
    # Implementation here
    pass
```

## 🚀 Deployment Preparation

### Production Checklist

```bash
# Pre-deployment checklist script
#!/bin/bash

echo "🔍 Running pre-deployment checks..."

# Check code quality
echo "Checking code quality..."
black --check stock_analysis_system/
isort --check-only stock_analysis_system/
flake8 stock_analysis_system/
mypy stock_analysis_system/

# Run tests
echo "Running tests..."
pytest --cov=stock_analysis_system --cov-fail-under=80

# Check security
echo "Checking security..."
bandit -r stock_analysis_system/
safety check

# Check dependencies
echo "Checking dependencies..."
pip-audit

# Build Docker image
echo "Building Docker image..."
docker build -t stock-analysis-system:latest .

# Test Docker image
echo "Testing Docker image..."
docker run --rm stock-analysis-system:latest python -c "import stock_analysis_system; print('Import successful')"

echo "✅ All checks passed!"
```

### Environment Preparation

```python
# Production configuration validation
from pydantic import BaseSettings, validator
import os

class ProductionSettings(BaseSettings):
    environment: str = "production"
    debug: bool = False
    
    @validator('debug')
    def debug_must_be_false_in_production(cls, v, values):
        if values.get('environment') == 'production' and v:
            raise ValueError('Debug must be False in production')
        return v
    
    @validator('secret_key')
    def secret_key_must_be_secure(cls, v):
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters')
        if v in ['dev-secret-key', 'change-me']:
            raise ValueError('Secret key must be changed from default')
        return v

# Validate before deployment
try:
    settings = ProductionSettings()
    print("✅ Production configuration is valid")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    exit(1)
```

## 📋 Summary

This development guide provides comprehensive coverage of:

- ✅ **Quick Setup**: One-command development environment setup
- ✅ **Development Workflow**: Git workflow, code quality, and daily commands
- ✅ **Testing**: Unit, integration, and performance testing strategies
- ✅ **Debugging**: Tools and techniques for troubleshooting
- ✅ **Database Development**: Migrations, schema changes, and optimization
- ✅ **Frontend Development**: React/TypeScript development patterns
- ✅ **API Development**: FastAPI endpoint creation and testing
- ✅ **Performance**: Optimization strategies for database, caching, and memory
- ✅ **Documentation**: Code documentation and API documentation standards
- ✅ **Deployment Preparation**: Production readiness checklists

The guide is designed to help developers quickly become productive while maintaining high code quality and following best practices.

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Maintained By**: Development Team