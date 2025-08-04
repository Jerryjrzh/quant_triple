"""Database configuration and base models."""

from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config.settings import get_settings

settings = get_settings()

# Create synchronous engine for migrations
engine = create_engine(
    settings.database.url,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    echo=settings.debug,
)

# Create async engine for application
async_engine = create_async_engine(
    settings.database.url.replace("postgresql://", "postgresql+asyncpg://"),
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    echo=settings.debug,
)

# Create session factories
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

# Create declarative base
Base = declarative_base()


def get_db_session():
    """Get database session for synchronous operations."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for async operations."""
    async with AsyncSessionLocal() as session:
        yield session