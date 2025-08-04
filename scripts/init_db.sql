-- Database initialization script for Stock Analysis System
-- This script is automatically executed when PostgreSQL container starts

-- Create database if it doesn't exist (handled by POSTGRES_DB environment variable)
-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for better performance
-- These will be created by Alembic migrations, but we can prepare the database

-- Set timezone
SET timezone = 'Asia/Shanghai';

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'Stock Analysis System database initialized successfully';
END $$;