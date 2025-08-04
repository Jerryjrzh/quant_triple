"""Pytest configuration and fixtures."""

import asyncio
import os
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from stock_analysis_system.core.database import Base, get_db_session


# Test database URL
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql://postgres:password@localhost/test_stock_analysis"
)
TEST_ASYNC_DATABASE_URL = TEST_DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def engine():
    """Create test database engine."""
    engine = create_engine(TEST_DATABASE_URL, echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="session")
def async_engine():
    """Create async test database engine."""
    engine = create_async_engine(TEST_ASYNC_DATABASE_URL, echo=False)
    yield engine
    engine.sync_engine.dispose()


@pytest.fixture
def db_session(engine):
    """Create database session for testing."""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest_asyncio.fixture
async def async_db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async database session for testing."""
    async_session = sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
def sample_stock_data():
    """Sample stock data for testing."""
    return {
        "stock_code": "000001",
        "stock_name": "平安银行",
        "trade_date": "2023-01-01",
        "open_price": 10.0,
        "high_price": 10.5,
        "low_price": 9.8,
        "close_price": 10.2,
        "volume": 1000000,
        "amount": 10200000.0,
        "adj_factor": 1.0,
    }


@pytest.fixture
def sample_institutional_data():
    """Sample institutional data for testing."""
    return {
        "stock_code": "000001",
        "institution_name": "Test Fund",
        "institution_type": "mutual_fund",
        "activity_date": "2023-01-01",
        "activity_type": "new_entry",
        "position_change": 1000000,
        "total_position": 1000000,
        "confidence_score": 0.85,
    }


@pytest.fixture
def mock_spring_festival_dates():
    """Mock Spring Festival dates for testing."""
    return {
        2020: "2020-01-25",
        2021: "2021-02-12",
        2022: "2022-02-01",
        2023: "2023-01-22",
        2024: "2024-02-10",
    }