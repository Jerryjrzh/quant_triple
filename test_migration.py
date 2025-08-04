#!/usr/bin/env python3
"""Test script to run migrations with SQLite."""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables for SQLite
os.environ["DATABASE_URL"] = "sqlite:///./test_stock_analysis.db"

# Import after setting environment
from alembic.config import Config
from alembic import command

def test_migration():
    """Test the migration with SQLite."""
    print("🔄 Testing database migration with SQLite...")
    
    # Create alembic config
    alembic_cfg = Config("alembic.ini")
    
    # Override the database URL to use SQLite
    alembic_cfg.set_main_option("sqlalchemy.url", "sqlite:///./test_stock_analysis.db")
    
    try:
        # Run the migration
        command.upgrade(alembic_cfg, "head")
        print("✅ Migration completed successfully!")
        
        # Check if database file was created
        if Path("test_stock_analysis.db").exists():
            print("✅ Database file created: test_stock_analysis.db")
            
            # Show database info
            import sqlite3
            conn = sqlite3.connect("test_stock_analysis.db")
            cursor = conn.cursor()
            
            # Get table list
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            print(f"✅ Created {len(tables)} tables:")
            for table in tables:
                print(f"   - {table[0]}")
            
            conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False
    
    finally:
        # Clean up test database
        if Path("test_stock_analysis.db").exists():
            Path("test_stock_analysis.db").unlink()
            print("🧹 Cleaned up test database")

if __name__ == "__main__":
    success = test_migration()
    sys.exit(0 if success else 1)