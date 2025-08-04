"""Application settings and configuration management."""

import os
from typing import Any, Dict, List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    model_config = {"env_prefix": "DB_"}

    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    name: str = Field(default="stock_analysis", env="DB_NAME")
    user: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")

    @property
    def url(self) -> str:
        """Get database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    model_config = {"env_prefix": "REDIS_"}

    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")

    @property
    def url(self) -> str:
        """Get Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class APISettings(BaseSettings):
    """API configuration settings."""
    
    model_config = {"env_prefix": "API_"}

    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    reload: bool = Field(default=False, env="API_RELOAD")
    workers: int = Field(default=1, env="API_WORKERS")
    
    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


class DataSourceSettings(BaseSettings):
    """Data source configuration settings."""
    
    model_config = {"env_prefix": "DATA_"}

    # Tushare
    tushare_token: Optional[str] = Field(default=None, env="TUSHARE_TOKEN")
    tushare_timeout: int = Field(default=30, env="TUSHARE_TIMEOUT")
    
    # AkShare
    akshare_timeout: int = Field(default=30, env="AKSHARE_TIMEOUT")
    
    # Rate limiting
    requests_per_minute: int = Field(default=200, env="DATA_REQUESTS_PER_MINUTE")
    retry_attempts: int = Field(default=3, env="DATA_RETRY_ATTEMPTS")
    retry_delay: int = Field(default=1, env="DATA_RETRY_DELAY")


class CelerySettings(BaseSettings):
    """Celery configuration settings."""
    
    model_config = {"env_prefix": "CELERY_"}

    broker_url: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    result_backend: str = Field(default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")
    task_serializer: str = Field(default="json", env="CELERY_TASK_SERIALIZER")
    result_serializer: str = Field(default="json", env="CELERY_RESULT_SERIALIZER")
    accept_content: List[str] = Field(default=["json"], env="CELERY_ACCEPT_CONTENT")
    timezone: str = Field(default="Asia/Shanghai", env="CELERY_TIMEZONE")
    enable_utc: bool = Field(default=True, env="CELERY_ENABLE_UTC")


class MLSettings(BaseSettings):
    """Machine learning configuration settings."""
    
    model_config = {"env_prefix": "ML_", "protected_namespaces": ("settings_",)}

    mlflow_tracking_uri: str = Field(default="sqlite:///mlruns.db", env="MLFLOW_TRACKING_URI")
    mlflow_registry_uri: Optional[str] = Field(default=None, env="MLFLOW_REGISTRY_URI")
    experiment_name: str = Field(default="stock-analysis", env="ML_EXPERIMENT_NAME")
    
    # Model settings
    retrain_days: int = Field(default=30, env="MODEL_RETRAIN_DAYS")
    drift_threshold: float = Field(default=0.1, env="MODEL_DRIFT_THRESHOLD")


class DaskSettings(BaseSettings):
    """Dask parallel processing configuration settings."""
    
    model_config = {"env_prefix": "DASK_"}

    # Worker configuration
    n_workers: int = Field(default=4, env="DASK_N_WORKERS")
    threads_per_worker: int = Field(default=2, env="DASK_THREADS_PER_WORKER")
    memory_limit: str = Field(default="2GB", env="DASK_MEMORY_LIMIT")
    
    # Processing configuration
    chunk_size: int = Field(default=100, env="DASK_CHUNK_SIZE")
    enable_distributed: bool = Field(default=False, env="DASK_ENABLE_DISTRIBUTED")
    scheduler_address: Optional[str] = Field(default=None, env="DASK_SCHEDULER_ADDRESS")
    
    # Performance tuning
    memory_target_fraction: float = Field(default=0.6, env="DASK_MEMORY_TARGET_FRACTION")
    memory_spill_fraction: float = Field(default=0.7, env="DASK_MEMORY_SPILL_FRACTION")
    memory_pause_fraction: float = Field(default=0.8, env="DASK_MEMORY_PAUSE_FRACTION")
    memory_terminate_fraction: float = Field(default=0.95, env="DASK_MEMORY_TERMINATE_FRACTION")
    
    # Timeouts
    connect_timeout: str = Field(default="60s", env="DASK_CONNECT_TIMEOUT")
    tcp_timeout: str = Field(default="60s", env="DASK_TCP_TIMEOUT")


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    model_config = {"env_prefix": "LOG_"}

    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    max_file_size: int = Field(default=10485760, env="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = {
        "extra": "ignore",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }

    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Application specific
    app_name: str = Field(default="Stock Analysis System", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    
    # Spring Festival Analysis
    spring_festival_window_days: int = Field(default=60, env="SPRING_FESTIVAL_WINDOW_DAYS")
    min_years_for_analysis: int = Field(default=3, env="MIN_YEARS_FOR_ANALYSIS")
    
    # Risk Management
    default_confidence_level: float = Field(default=0.95, env="DEFAULT_CONFIDENCE_LEVEL")
    max_position_size: float = Field(default=0.1, env="MAX_POSITION_SIZE")  # 10%
    
    # Caching
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")  # 1 hour
    
    # Component settings - initialized separately to avoid conflicts
    @property
    def database(self) -> DatabaseSettings:
        return DatabaseSettings()
    
    @property
    def redis(self) -> RedisSettings:
        return RedisSettings()
    
    @property
    def api(self) -> APISettings:
        return APISettings()
    
    @property
    def data_sources(self) -> DataSourceSettings:
        return DataSourceSettings()
    
    @property
    def celery(self) -> CelerySettings:
        return CelerySettings()
    
    @property
    def ml(self) -> MLSettings:
        return MLSettings()
    
    @property
    def dask(self) -> DaskSettings:
        return DaskSettings()
    
    @property
    def logging(self) -> LoggingSettings:
        return LoggingSettings()


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings