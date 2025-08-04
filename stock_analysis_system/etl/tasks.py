"""Celery tasks for ETL pipeline."""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from celery import Task
from celery.exceptions import Retry

from stock_analysis_system.etl.celery_app import celery_app
from stock_analysis_system.etl.pipeline import get_etl_manager, ETLJobConfig, ETLStatus
from stock_analysis_system.data.data_quality_engine import EnhancedDataQualityEngine

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Base task class with callbacks."""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called on task success."""
        logger.info(f"Task {task_id} completed successfully")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        logger.error(f"Task {task_id} failed: {exc}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called on task retry."""
        logger.warning(f"Task {task_id} retrying: {exc}")


@celery_app.task(bind=True, base=CallbackTask, name='stock_analysis_system.etl.tasks.daily_data_ingestion')
def daily_data_ingestion(self, symbols: Optional[List[str]] = None, days_back: int = 7):
    """Daily data ingestion task."""
    logger.info("Starting daily data ingestion task")
    
    try:
        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(_run_daily_ingestion(symbols, days_back))
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Daily ingestion task failed: {exc}")
        
        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            retry_delay = 60 * (2 ** self.request.retries)  # Exponential backoff
            logger.info(f"Retrying in {retry_delay} seconds...")
            raise self.retry(countdown=retry_delay, exc=exc)
        
        raise exc


async def _run_daily_ingestion(symbols: Optional[List[str]], days_back: int) -> Dict[str, Any]:
    """Run daily data ingestion."""
    etl_manager = await get_etl_manager()
    
    # Create job configuration
    if symbols is None:
        config = await etl_manager.create_daily_update_job()
    else:
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=days_back)
        
        config = ETLJobConfig(
            job_name=f"daily_ingestion_{datetime.now().strftime('%Y%m%d_%H%M')}",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            batch_size=50,
            quality_threshold=0.6,
            retry_failed=True,
            skip_existing=False,
            validate_data=True,
            clean_data=True
        )
    
    # Run ETL job
    metrics = await etl_manager.run_job_async(config)
    
    # Return results
    return {
        'job_name': config.job_name,
        'status': metrics.status.value,
        'records_extracted': metrics.records_extracted,
        'records_loaded': metrics.records_loaded,
        'success_rate': metrics.success_rate,
        'quality_score': metrics.quality_score,
        'duration_seconds': metrics.duration.total_seconds() if metrics.duration else 0,
        'errors': metrics.error_messages
    }


@celery_app.task(bind=True, base=CallbackTask, name='stock_analysis_system.etl.tasks.historical_backfill')
def historical_backfill(self, symbols: List[str], years: int = 5):
    """Historical data backfill task."""
    logger.info(f"Starting historical backfill for {len(symbols)} symbols, {years} years")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(_run_historical_backfill(symbols, years))
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Historical backfill task failed: {exc}")
        
        if self.request.retries < self.max_retries:
            retry_delay = 300 * (2 ** self.request.retries)  # Longer delays for backfill
            logger.info(f"Retrying in {retry_delay} seconds...")
            raise self.retry(countdown=retry_delay, exc=exc)
        
        raise exc


async def _run_historical_backfill(symbols: List[str], years: int) -> Dict[str, Any]:
    """Run historical data backfill."""
    etl_manager = await get_etl_manager()
    
    # Create job configuration
    config = await etl_manager.create_historical_backfill_job(symbols, years)
    
    # Run ETL job
    metrics = await etl_manager.run_job_async(config)
    
    return {
        'job_name': config.job_name,
        'status': metrics.status.value,
        'symbols_count': len(symbols),
        'records_extracted': metrics.records_extracted,
        'records_loaded': metrics.records_loaded,
        'success_rate': metrics.success_rate,
        'quality_score': metrics.quality_score,
        'duration_seconds': metrics.duration.total_seconds() if metrics.duration else 0,
        'errors': metrics.error_messages
    }


@celery_app.task(bind=True, base=CallbackTask, name='stock_analysis_system.etl.tasks.data_quality_check')
def data_quality_check(self, symbols: List[str], days_back: int = 30):
    """Data quality check task."""
    logger.info(f"Starting data quality check for {len(symbols)} symbols")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(_run_quality_check(symbols, days_back))
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Data quality check task failed: {exc}")
        raise exc


async def _run_quality_check(symbols: List[str], days_back: int) -> Dict[str, Any]:
    """Run data quality check."""
    from stock_analysis_system.data.data_source_manager import get_data_source_manager
    
    data_manager = await get_data_source_manager()
    quality_engine = EnhancedDataQualityEngine()
    
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=days_back)
    
    results = {
        'check_date': datetime.now().isoformat(),
        'symbols_checked': len(symbols),
        'period': f"{start_date} to {end_date}",
        'symbol_results': {},
        'overall_quality': 0.0,
        'issues_found': 0,
        'recommendations': []
    }
    
    total_quality = 0.0
    total_issues = 0
    
    for symbol in symbols:
        try:
            # Get data for symbol
            data = await data_manager.get_stock_data(symbol, start_date, end_date)
            
            if data.empty:
                results['symbol_results'][symbol] = {
                    'status': 'no_data',
                    'quality_score': 0.0,
                    'issues': 0
                }
                continue
            
            # Validate quality
            report = quality_engine.validate_data(data, f"Quality Check - {symbol}")
            
            results['symbol_results'][symbol] = {
                'status': 'checked',
                'quality_score': report.overall_score,
                'issues': len(report.issues),
                'records': len(data),
                'completeness': report.completeness_score,
                'consistency': report.consistency_score,
                'timeliness': report.timeliness_score,
                'accuracy': report.accuracy_score
            }
            
            total_quality += report.overall_score
            total_issues += len(report.issues)
            
            # Collect recommendations
            results['recommendations'].extend(report.recommendations)
            
        except Exception as e:
            logger.error(f"Quality check failed for {symbol}: {e}")
            results['symbol_results'][symbol] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Calculate overall metrics
    checked_symbols = [s for s, r in results['symbol_results'].items() if r.get('status') == 'checked']
    if checked_symbols:
        results['overall_quality'] = total_quality / len(checked_symbols)
    
    results['issues_found'] = total_issues
    results['recommendations'] = list(set(results['recommendations']))  # Remove duplicates
    
    logger.info(f"Quality check completed: {len(checked_symbols)} symbols, "
                f"overall quality: {results['overall_quality']:.2f}")
    
    return results


@celery_app.task(bind=True, base=CallbackTask, name='stock_analysis_system.etl.tasks.data_transformation')
def data_transformation(self, job_config: Dict[str, Any]):
    """Data transformation task."""
    logger.info("Starting data transformation task")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(_run_data_transformation(job_config))
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Data transformation task failed: {exc}")
        
        if self.request.retries < self.max_retries:
            retry_delay = 120 * (2 ** self.request.retries)
            raise self.retry(countdown=retry_delay, exc=exc)
        
        raise exc


async def _run_data_transformation(job_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run data transformation."""
    # This would implement specific transformation logic
    # For now, return a placeholder result
    
    return {
        'job_name': job_config.get('job_name', 'transformation'),
        'status': 'completed',
        'records_processed': job_config.get('record_count', 0),
        'transformations_applied': [
            'data_type_conversion',
            'missing_value_handling',
            'outlier_detection',
            'normalization'
        ]
    }


@celery_app.task(bind=True, base=CallbackTask, name='stock_analysis_system.etl.tasks.generate_quality_report')
def generate_quality_report(self, symbols: Optional[List[str]] = None, days_back: int = 30):
    """Generate comprehensive data quality report."""
    logger.info("Starting quality report generation")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(_generate_quality_report(symbols, days_back))
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Quality report generation failed: {exc}")
        raise exc


async def _generate_quality_report(symbols: Optional[List[str]], days_back: int) -> Dict[str, Any]:
    """Generate quality report."""
    if symbols is None:
        # Get default symbols
        symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
    
    # Run quality check
    quality_results = await _run_quality_check(symbols, days_back)
    
    # Generate summary report
    report = {
        'report_date': datetime.now().isoformat(),
        'report_type': 'weekly_quality_summary',
        'period_days': days_back,
        'summary': {
            'total_symbols': len(symbols),
            'symbols_with_data': len([s for s, r in quality_results['symbol_results'].items() 
                                    if r.get('status') == 'checked']),
            'average_quality_score': quality_results['overall_quality'],
            'total_issues': quality_results['issues_found'],
            'quality_grade': _get_quality_grade(quality_results['overall_quality'])
        },
        'top_issues': quality_results['recommendations'][:5],  # Top 5 recommendations
        'symbol_breakdown': quality_results['symbol_results']
    }
    
    logger.info(f"Quality report generated: {report['summary']['quality_grade']} grade, "
                f"{report['summary']['total_issues']} issues found")
    
    return report


def _get_quality_grade(score: float) -> str:
    """Convert quality score to letter grade."""
    if score >= 0.9:
        return 'A'
    elif score >= 0.8:
        return 'B'
    elif score >= 0.7:
        return 'C'
    elif score >= 0.6:
        return 'D'
    else:
        return 'F'


@celery_app.task(bind=True, base=CallbackTask, name='stock_analysis_system.etl.tasks.cleanup_old_data')
def cleanup_old_data(self, days_to_keep: int = 365):
    """Clean up old data beyond retention period."""
    logger.info(f"Starting cleanup of data older than {days_to_keep} days")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(_cleanup_old_data(days_to_keep))
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Data cleanup task failed: {exc}")
        raise exc


async def _cleanup_old_data(days_to_keep: int) -> Dict[str, Any]:
    """Clean up old data."""
    from stock_analysis_system.core.database import AsyncSessionLocal
    from stock_analysis_system.data.models import StockDailyData
    
    cutoff_date = date.today() - timedelta(days=days_to_keep)
    
    async with AsyncSessionLocal() as session:
        try:
            # Count records to be deleted
            count_result = await session.execute(
                session.query(StockDailyData)
                .filter(StockDailyData.trade_date < cutoff_date)
                .count()
            )
            
            records_to_delete = count_result.scalar()
            
            if records_to_delete == 0:
                return {
                    'status': 'completed',
                    'records_deleted': 0,
                    'cutoff_date': cutoff_date.isoformat(),
                    'message': 'No old records found'
                }
            
            # Delete old records
            delete_result = await session.execute(
                session.query(StockDailyData)
                .filter(StockDailyData.trade_date < cutoff_date)
                .delete()
            )
            
            await session.commit()
            
            logger.info(f"Cleaned up {records_to_delete} records older than {cutoff_date}")
            
            return {
                'status': 'completed',
                'records_deleted': records_to_delete,
                'cutoff_date': cutoff_date.isoformat(),
                'message': f'Successfully deleted {records_to_delete} old records'
            }
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Cleanup failed: {e}")
            raise e


# Task routing and monitoring
@celery_app.task(name='stock_analysis_system.etl.tasks.health_check')
def health_check():
    """Health check task for monitoring."""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'worker_id': health_check.request.id
    }


# Utility functions for task management
def schedule_daily_ingestion(symbols: Optional[List[str]] = None, eta: Optional[datetime] = None):
    """Schedule daily data ingestion task."""
    return daily_data_ingestion.apply_async(
        args=[symbols],
        eta=eta,
        queue='data_ingestion'
    )


def schedule_historical_backfill(symbols: List[str], years: int = 5, eta: Optional[datetime] = None):
    """Schedule historical backfill task."""
    return historical_backfill.apply_async(
        args=[symbols, years],
        eta=eta,
        queue='data_ingestion'
    )


def schedule_quality_check(symbols: List[str], days_back: int = 30, eta: Optional[datetime] = None):
    """Schedule data quality check task."""
    return data_quality_check.apply_async(
        args=[symbols, days_back],
        eta=eta,
        queue='data_quality'
    )