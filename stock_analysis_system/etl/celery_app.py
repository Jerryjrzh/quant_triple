"""Celery application configuration for ETL tasks."""

import os

from celery import Celery

from config.settings import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    "stock_analysis_etl",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
    include=[
        "stock_analysis_system.etl.tasks",
    ],
)

# Configure Celery
celery_app.conf.update(
    task_serializer=settings.celery.task_serializer,
    result_serializer=settings.celery.result_serializer,
    accept_content=settings.celery.accept_content,
    timezone=settings.celery.timezone,
    enable_utc=settings.celery.enable_utc,
    # Task routing
    task_routes={
        "stock_analysis_system.etl.tasks.daily_data_ingestion": {
            "queue": "data_ingestion"
        },
        "stock_analysis_system.etl.tasks.data_quality_check": {"queue": "data_quality"},
        "stock_analysis_system.etl.tasks.data_transformation": {
            "queue": "data_processing"
        },
    },
    # Task execution settings
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    # Retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    # Result settings
    result_expires=3600,  # 1 hour
    # Beat schedule for periodic tasks
    beat_schedule={
        "daily-market-data-ingestion": {
            "task": "stock_analysis_system.etl.tasks.daily_data_ingestion",
            "schedule": 60.0 * 60.0 * 24.0,  # Daily at midnight
            "options": {"queue": "data_ingestion"},
        },
        "weekly-data-quality-report": {
            "task": "stock_analysis_system.etl.tasks.generate_quality_report",
            "schedule": 60.0 * 60.0 * 24.0 * 7.0,  # Weekly
            "options": {"queue": "data_quality"},
        },
    },
)

# Auto-discover tasks
celery_app.autodiscover_tasks()

if __name__ == "__main__":
    celery_app.start()
