"""SQLite settings for testing without PostgreSQL."""

import os
from config.settings import Settings as BaseSettings


class SQLiteSettings(BaseSettings):
    """SQLite-based settings for testing."""
    
    @property
    def database_url(self) -> str:
        """Get SQLite database URL."""
        return "sqlite:///./stock_analysis.db"


def get_sqlite_settings():
    """Get SQLite settings for testing."""
    return SQLiteSettings()