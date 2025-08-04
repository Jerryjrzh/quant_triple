#!/usr/bin/env python3
"""Demo script for ETL Pipeline."""

import asyncio
import pandas as pd
from datetime import datetime, date, timedelta
from stock_analysis_system.etl.pipeline import ETLPipeline, ETLJobManager, ETLJobConfig

async def demonstrate_etl_pipeline():
    """Demonstrate ETL pipeline capabilities."""
    print("🚀 ETL Pipeline Demonstration")
    print("=" * 60)
    
    # Initialize pipeline
    print("\n🔧 Initializing ETL Pipeline...")
    pipeline = ETLPipeline()
    await pipeline.initialize()
    print("   ✓ Pipeline initialized successfully")
    
    # Create sample job configuration
    print("\n📋 Creating ETL Job Configuration...")
    config = ETLJobConfig(
        job_name="demo_etl_job",
        symbols=["000001.SZ", "000002.SZ"],
        start_date=date.today() - timedelta(days=30),
        end_date=date.today() - timedelta(days=1),
        batch_size=20,
        quality_threshold=0.6,
        retry_failed=True,
        skip_existing=True,
        validate_data=True,
        clean_data=True
    )
    
    print(f"   Job Name: {config.job_name}")
    print(f"   Symbols: {config.symbols}")
    print(f"   Date Range: {config.start_date} to {config.end_date}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Quality Threshold: {config.quality_threshold}")
    
    # Run ETL job
    print(f"\n🔄 Running ETL Job...")
    print("   This may take a moment as it fetches real data...")
    
    try:
        metrics = await pipeline.run_job(config)
        
        print(f"\n📊 ETL JOB RESULTS")
        print("-" * 40)
        print(f"Status: {metrics.status.value.upper()}")
        print(f"Stage: {metrics.stage.value}")
        print(f"Duration: {metrics.duration}")
        print(f"Records Extracted: {metrics.records_extracted}")
        print(f"Records Transformed: {metrics.records_transformed}")
        print(f"Records Loaded: {metrics.records_loaded}")
        print(f"Records Failed: {metrics.records_failed}")
        print(f"Success Rate: {metrics.success_rate:.1f}%")
        print(f"Quality Score: {metrics.quality_score:.2f}")
        
        if metrics.error_messages:
            print(f"\n⚠️ ERRORS ({len(metrics.error_messages)}):")
            for i, error in enumerate(metrics.error_messages[:5], 1):
                print(f"   {i}. {error}")
            if len(metrics.error_messages) > 5:
                print(f"   ... and {len(metrics.error_messages) - 5} more errors")
        
        # Status interpretation
        if metrics.status.value == 'success':
            print(f"\n✅ ETL job completed successfully!")
        elif metrics.status.value == 'partial_success':
            print(f"\n⚠️ ETL job completed with some issues")
        else:
            print(f"\n❌ ETL job failed")
            
    except Exception as e:
        print(f"\n❌ ETL job failed with exception: {e}")
        return None
    
    return metrics

async def demonstrate_etl_manager():
    """Demonstrate ETL job manager capabilities."""
    print(f"\n🎯 ETL Job Manager Demonstration")
    print("-" * 40)
    
    # Initialize manager
    manager = ETLJobManager()
    
    # Create different types of jobs
    print("\n📅 Creating Daily Update Job...")
    daily_config = await manager.create_daily_update_job(['000001.SZ'])
    print(f"   Job: {daily_config.job_name}")
    print(f"   Symbols: {len(daily_config.symbols)}")
    print(f"   Date Range: {daily_config.start_date} to {daily_config.end_date}")
    
    print("\n📚 Creating Historical Backfill Job...")
    backfill_config = await manager.create_historical_backfill_job(['000001.SZ'], years=1)
    print(f"   Job: {backfill_config.job_name}")
    print(f"   Symbols: {len(backfill_config.symbols)}")
    print(f"   Date Range: {backfill_config.start_date} to {backfill_config.end_date}")
    print(f"   Days of Data: {(backfill_config.end_date - backfill_config.start_date).days}")
    
    # Run a small job
    print(f"\n🏃 Running Small Demo Job...")
    small_config = ETLJobConfig(
        job_name="small_demo_job",
        symbols=["000001.SZ"],
        start_date=date.today() - timedelta(days=7),
        end_date=date.today() - timedelta(days=1),
        batch_size=10,
        quality_threshold=0.5,
        validate_data=False,  # Skip validation for speed
        clean_data=False
    )
    
    try:
        metrics = await manager.run_job_async(small_config)
        
        print(f"   Status: {metrics.status.value}")
        print(f"   Records: {metrics.records_extracted} extracted, {metrics.records_loaded} loaded")
        print(f"   Duration: {metrics.duration}")
        
        # Check job tracking
        active_jobs = manager.get_active_jobs()
        print(f"\n📋 Active Jobs: {len(active_jobs)}")
        for job_id, job_info in active_jobs.items():
            print(f"   {job_id}: {job_info.get('status', 'unknown')}")
            
    except Exception as e:
        print(f"   ❌ Small job failed: {e}")
    
    return manager

async def demonstrate_data_quality_integration():
    """Demonstrate data quality integration in ETL."""
    print(f"\n🔍 Data Quality Integration Demonstration")
    print("-" * 40)
    
    # Create pipeline
    pipeline = ETLPipeline()
    await pipeline.initialize()
    
    # Create job with strict quality requirements
    config = ETLJobConfig(
        job_name="quality_demo_job",
        symbols=["000001.SZ"],
        start_date=date.today() - timedelta(days=10),
        end_date=date.today() - timedelta(days=1),
        batch_size=5,
        quality_threshold=0.8,  # High quality threshold
        validate_data=True,
        clean_data=True
    )
    
    print(f"   Quality Threshold: {config.quality_threshold}")
    print(f"   Data Validation: {config.validate_data}")
    print(f"   Data Cleaning: {config.clean_data}")
    
    try:
        # Run with quality checks
        print(f"\n   Running ETL with quality validation...")
        metrics = await pipeline.run_job(config)
        
        print(f"   Quality Score: {metrics.quality_score:.2f}")
        print(f"   Status: {metrics.status.value}")
        
        if metrics.quality_score >= config.quality_threshold:
            print(f"   ✅ Data quality meets requirements")
        else:
            print(f"   ⚠️ Data quality below threshold (cleaned: {config.clean_data})")
            
    except Exception as e:
        print(f"   ❌ Quality validation failed: {e}")

def demonstrate_celery_tasks():
    """Demonstrate Celery task integration."""
    print(f"\n⚙️ Celery Tasks Demonstration")
    print("-" * 40)
    
    try:
        from stock_analysis_system.etl.tasks import (
            schedule_daily_ingestion,
            schedule_quality_check,
            health_check
        )
        
        print("   ✓ Celery tasks imported successfully")
        print("   Available tasks:")
        print("     - daily_data_ingestion")
        print("     - historical_backfill")
        print("     - data_quality_check")
        print("     - generate_quality_report")
        print("     - cleanup_old_data")
        print("     - health_check")
        
        # Test health check (synchronous task)
        print(f"\n   Testing health check task...")
        try:
            result = health_check()
            print(f"   ✅ Health check result: {result['status']}")
        except Exception as e:
            print(f"   ⚠️ Health check requires Celery worker: {e}")
        
        print(f"\n   📝 To run Celery tasks:")
        print(f"   1. Start Redis: redis-server")
        print(f"   2. Start Celery worker: celery -A stock_analysis_system.etl.celery_app worker --loglevel=info")
        print(f"   3. Start Celery beat: celery -A stock_analysis_system.etl.celery_app beat --loglevel=info")
        print(f"   4. Tasks will run automatically or can be triggered via API")
        
    except ImportError as e:
        print(f"   ⚠️ Celery tasks not available: {e}")
        print(f"   Install Celery and Redis to use background tasks")

async def main():
    """Main demonstration function."""
    try:
        print("Starting ETL Pipeline demonstration...")
        print("This demo will show the capabilities of the ETL system")
        print()
        
        # Core pipeline demonstration
        etl_metrics = await demonstrate_etl_pipeline()
        
        # Job manager demonstration
        etl_manager = await demonstrate_etl_manager()
        
        # Data quality integration
        await demonstrate_data_quality_integration()
        
        # Celery tasks demonstration
        demonstrate_celery_tasks()
        
        print(f"\n🎉 ETL Pipeline demonstration completed!")
        print("=" * 60)
        
        if etl_metrics:
            print(f"✅ Successfully demonstrated ETL pipeline with {etl_metrics.records_loaded} records loaded")
        
        print(f"📚 Key Features Demonstrated:")
        print(f"   ✓ Data extraction from multiple sources with failover")
        print(f"   ✓ Data quality validation and cleaning")
        print(f"   ✓ Batch processing and error handling")
        print(f"   ✓ Job management and tracking")
        print(f"   ✓ Celery integration for background processing")
        print(f"   ✓ Comprehensive metrics and monitoring")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())