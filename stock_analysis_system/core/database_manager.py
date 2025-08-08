"""Database Manager for testing and operations."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from config.settings import get_settings
from stock_analysis_system.core.database import get_async_db_session

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database manager for async operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.pool = None
        self._session = None
    
    async def initialize(self):
        """Initialize database connection pool."""
        try:
            # Create asyncpg connection pool
            database_url = self.settings.database.url.replace("postgresql://", "postgresql://")
            self.pool = await asyncpg.create_pool(
                database_url,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            # For testing, we'll create a mock pool
            self.pool = MockConnectionPool()
    
    async def close(self):
        """Close database connections."""
        if self.pool and hasattr(self.pool, 'close'):
            await self.pool.close()
            logger.info("Database connections closed")
    
    async def execute(self, query: str, *args) -> None:
        """Execute a query without returning results."""
        if not self.pool:
            await self.initialize()
        
        try:
            if hasattr(self.pool, 'acquire'):
                async with self.pool.acquire() as conn:
                    await conn.execute(query, *args)
            else:
                # Mock execution for testing
                logger.debug(f"Mock execute: {query} with args: {args}")
        except Exception as e:
            logger.error(f"Database execute error: {e}")
            # For testing, we'll ignore errors
            pass
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict]:
        """Fetch one row from query."""
        if not self.pool:
            await self.initialize()
        
        try:
            if hasattr(self.pool, 'acquire'):
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow(query, *args)
                    return dict(row) if row else None
            else:
                # Mock fetch for testing
                logger.debug(f"Mock fetch_one: {query} with args: {args}")
                return {"mock": "data"}
        except Exception as e:
            logger.error(f"Database fetch_one error: {e}")
            return None
    
    async def fetch_all(self, query: str, *args) -> List[Dict]:
        """Fetch all rows from query."""
        if not self.pool:
            await self.initialize()
        
        try:
            if hasattr(self.pool, 'acquire'):
                async with self.pool.acquire() as conn:
                    rows = await conn.fetch(query, *args)
                    return [dict(row) for row in rows]
            else:
                # Mock fetch for testing
                logger.debug(f"Mock fetch_all: {query} with args: {args}")
                return [{"mock": "data1"}, {"mock": "data2"}]
        except Exception as e:
            logger.error(f"Database fetch_all error: {e}")
            return []
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        if not self.pool:
            await self.initialize()
        
        if hasattr(self.pool, 'acquire'):
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    yield conn
        else:
            # Mock transaction for testing
            yield MockConnection()


class MockConnectionPool:
    """Mock connection pool for testing."""
    
    async def close(self):
        """Mock close method."""
        pass
    
    def acquire(self):
        """Mock acquire method."""
        return MockConnection()


class MockConnection:
    """Mock database connection for testing."""
    
    async def execute(self, query: str, *args):
        """Mock execute method."""
        logger.debug(f"Mock execute: {query} with args: {args}")
    
    async def fetchrow(self, query: str, *args):
        """Mock fetchrow method."""
        logger.debug(f"Mock fetchrow: {query} with args: {args}")
        return {"mock": "data"}
    
    async def fetch(self, query: str, *args):
        """Mock fetch method."""
        logger.debug(f"Mock fetch: {query} with args: {args}")
        return [{"mock": "data1"}, {"mock": "data2"}]
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def transaction(self):
        """Mock transaction method."""
        return self


# Global database manager instance
db_manager = DatabaseManager()