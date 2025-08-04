"""ETL Pipeline implementation with data ingestion, transformation, and loading."""

import logging
import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from config.settings import get_settings
from stock_analysis_system.core.database import get_db_session, AsyncSessionLocal
from stock_analysis_system.data.models import StockDailyData, SystemConfig
from stock_analysis_system.data.data_source_manager import get_data_source_manager
from stock_analysis_system.data.data_quality_engine import EnhancedDataQualityEngine

logger = logging.getLogger(__name__)
settings = get_settings()


class ETLStage(Enum):
    """ETL pipeline stages."""
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
    VALIDATE = "validate"
    COMPLETE = "complete"
    FAILED = "failed"


class ETLStatus(Enum):
    """ETL execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"


@dataclass
class ETLMetrics:
    """ETL execution metrics."""
    start_time: datetime
    end_time: Optional[datetime] = None
    stage: ETLStage = ETLStage.EXTRACT
    status: ETLStatus = ETLStatus.PENDING
    records_extracted: int = 0
    records_transformed: int = 0
    records_loaded: int = 0
    records_failed: int = 0
    quality_score: float = 0.0
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate execution duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total_records = self.records_extracted
        if total_records == 0:
            return 0.0
        return (self.records_loaded / total_records) * 100


@dataclass
class ETLJobConfig:
    """ETL job configuration."""
    job_name: str
    symbols: List[str]
    start_date: date
    end_date: date
    batch_size: int = 100
    quality_threshold: float = 0.7
    retry_failed: bool = True
    skip_existing: bool = True
    validate_data: bool = True
    clean_data: bool = True


class ETLPipeline:
    """Main ETL pipeline for stock data processing."""
    
    def __init__(self):
        self.data_manager = None
        self.quality_engine = EnhancedDataQualityEngine()
        self.metrics = None
        
    async def initialize(self):
        """Initialize pipeline components."""
        self.data_manager = await get_data_source_manager()
        logger.info("ETL Pipeline initialized")
    
    async def run_job(self, config: ETLJobConfig) -> ETLMetrics:
        """Run complete ETL job."""
        logger.info(f"Starting ETL job: {config.job_name}")
        
        # Initialize metrics
        self.metrics = ETLMetrics(
            start_time=datetime.now(),
            stage=ETLStage.EXTRACT,
            status=ETLStatus.RUNNING
        )
        
        try:
            # Ensure pipeline is initialized
            if not self.data_manager:
                await self.initialize()
            
            # Extract data
            raw_data = await self._extract_data(config)
            self.metrics.stage = ETLStage.TRANSFORM
            self.metrics.records_extracted = len(raw_data) if raw_data is not None else 0
            
            if raw_data is None or raw_data.empty:
                self.metrics.status = ETLStatus.FAILED
                self.metrics.error_messages.append("No data extracted")
                return self._finalize_metrics()
            
            # Validate data quality
            if config.validate_data:
                self.metrics.stage = ETLStage.VALIDATE
                quality_report = await self._validate_data_quality(raw_data, config.job_name)
                self.metrics.quality_score = quality_report.overall_score
                
                if quality_report.overall_score < config.quality_threshold:
                    logger.warning(f"Data quality below threshold: {quality_report.overall_score:.2f}")
                    
                    if config.clean_data:
                        raw_data = self.quality_engine.clean_data(raw_data, quality_report)
                        logger.info(f"Applied automatic data cleaning")
                    else:
                        self.metrics.error_messages.append(
                            f"Data quality too low: {quality_report.overall_score:.2f}"
                        )
            
            # Transform data
            self.metrics.stage = ETLStage.TRANSFORM
            transformed_data = await self._transform_data(raw_data, config)
            self.metrics.records_transformed = len(transformed_data) if transformed_data is not None else 0
            
            if transformed_data is None or transformed_data.empty:
                self.metrics.status = ETLStatus.FAILED
                self.metrics.error_messages.append("Data transformation failed")
                return self._finalize_metrics()
            
            # Load data
            self.metrics.stage = ETLStage.LOAD
            load_result = await self._load_data(transformed_data, config)
            self.metrics.records_loaded = load_result['loaded']
            self.metrics.records_failed = load_result['failed']
            
            # Determine final status
            if self.metrics.records_failed == 0:
                self.metrics.status = ETLStatus.SUCCESS
            elif self.metrics.records_loaded > 0:
                self.metrics.status = ETLStatus.PARTIAL_SUCCESS
            else:
                self.metrics.status = ETLStatus.FAILED
            
            self.metrics.stage = ETLStage.COMPLETE
            
        except Exception as e:
            logger.error(f"ETL job failed: {str(e)}")
            self.metrics.status = ETLStatus.FAILED
            self.metrics.stage = ETLStage.FAILED
            self.metrics.error_messages.append(str(e))
        
        return self._finalize_metrics()
    
    async def _extract_data(self, config: ETLJobConfig) -> Optional[pd.DataFrame]:
        """Extract data from data sources."""
        logger.info(f"Extracting data for {len(config.symbols)} symbols")
        
        all_data = []
        failed_symbols = []
        
        for symbol in config.symbols:
            try:
                logger.debug(f"Extracting data for {symbol}")
                symbol_data = await self.data_manager.get_stock_data(
                    symbol=symbol,
                    start_date=config.start_date,
                    end_date=config.end_date
                )
                
                if not symbol_data.empty:
                    all_data.append(symbol_data)
                    logger.debug(f"Extracted {len(symbol_data)} records for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
                    failed_symbols.append(symbol)
                    
            except Exception as e:
                logger.error(f"Failed to extract data for {symbol}: {str(e)}")
                failed_symbols.append(symbol)
                self.metrics.error_messages.append(f"Extract failed for {symbol}: {str(e)}")
        
        if failed_symbols:
            logger.warning(f"Failed to extract data for {len(failed_symbols)} symbols: {failed_symbols}")
        
        if not all_data:
            logger.error("No data extracted from any source")
            return None
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Successfully extracted {len(combined_data)} total records")
        
        return combined_data
    
    async def _validate_data_quality(self, data: pd.DataFrame, job_name: str):
        """Validate data quality."""
        logger.info("Validating data quality")
        
        try:
            # Train ML detector if not already trained
            if not self.quality_engine.ml_detector.is_fitted:
                # Use a subset of clean data for training
                clean_subset = data.dropna().head(1000)  # Use first 1000 clean records
                if len(clean_subset) > 10:
                    self.quality_engine.train_ml_detector(clean_subset)
                    logger.info("Trained ML anomaly detector")
            
            # Validate data
            report = self.quality_engine.validate_data(data, job_name)
            
            logger.info(f"Data quality validation completed:")
            logger.info(f"  Overall score: {report.overall_score:.2f}")
            logger.info(f"  Issues found: {len(report.issues)}")
            
            return report
            
        except Exception as e:
            logger.error(f"Data quality validation failed: {str(e)}")
            # Return a minimal report
            from stock_analysis_system.data.data_quality_engine import DataQualityReport
            return DataQualityReport(
                dataset_name=job_name,
                total_rows=len(data),
                total_columns=len(data.columns),
                issues=[],
                overall_score=0.5,  # Neutral score when validation fails
                completeness_score=0.5,
                consistency_score=0.5,
                timeliness_score=0.5,
                accuracy_score=0.5,
                recommendations=["Data quality validation failed - manual review recommended"]
            )
    
    async def _transform_data(self, data: pd.DataFrame, config: ETLJobConfig) -> Optional[pd.DataFrame]:
        """Transform raw data for loading."""
        logger.info("Transforming data")
        
        try:
            transformed_data = data.copy()
            
            # Ensure required columns exist
            required_columns = ['stock_code', 'trade_date', 'close_price']
            missing_columns = [col for col in required_columns if col not in transformed_data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return None
            
            # Standardize data types
            if 'trade_date' in transformed_data.columns:
                transformed_data['trade_date'] = pd.to_datetime(transformed_data['trade_date']).dt.date
            
            # Convert numeric columns
            numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount']
            for col in numeric_columns:
                if col in transformed_data.columns:
                    transformed_data[col] = pd.to_numeric(transformed_data[col], errors='coerce')
            
            # Add metadata columns
            transformed_data['created_at'] = datetime.now()
            transformed_data['updated_at'] = datetime.now()
            
            # Remove duplicates based on key columns
            if config.skip_existing:
                transformed_data = transformed_data.drop_duplicates(
                    subset=['stock_code', 'trade_date'], 
                    keep='last'
                )
            
            # Sort by stock_code and trade_date
            transformed_data = transformed_data.sort_values(['stock_code', 'trade_date'])
            
            logger.info(f"Data transformation completed: {len(transformed_data)} records")
            return transformed_data
            
        except Exception as e:
            logger.error(f"Data transformation failed: {str(e)}")
            self.metrics.error_messages.append(f"Transform failed: {str(e)}")
            return None
    
    async def _load_data(self, data: pd.DataFrame, config: ETLJobConfig) -> Dict[str, int]:
        """Load transformed data into database."""
        logger.info(f"Loading {len(data)} records to database")
        
        loaded_count = 0
        failed_count = 0
        batch_size = config.batch_size
        
        try:
            # Process data in batches
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i + batch_size]
                
                try:
                    batch_result = await self._load_batch(batch, config)
                    loaded_count += batch_result['loaded']
                    failed_count += batch_result['failed']
                    
                    logger.debug(f"Loaded batch {i//batch_size + 1}: {batch_result['loaded']} records")
                    
                except Exception as e:
                    logger.error(f"Failed to load batch {i//batch_size + 1}: {str(e)}")
                    failed_count += len(batch)
                    self.metrics.error_messages.append(f"Batch load failed: {str(e)}")
            
            logger.info(f"Data loading completed: {loaded_count} loaded, {failed_count} failed")
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            self.metrics.error_messages.append(f"Load failed: {str(e)}")
            failed_count = len(data)
        
        return {
            'loaded': loaded_count,
            'failed': failed_count
        }
    
    async def _load_batch(self, batch: pd.DataFrame, config: ETLJobConfig) -> Dict[str, int]:
        """Load a single batch of data."""
        loaded_count = 0
        failed_count = 0
        
        async with AsyncSessionLocal() as session:
            try:
                for _, row in batch.iterrows():
                    try:
                        # Check if record already exists (if skip_existing is True)
                        if config.skip_existing:
                            existing = await session.execute(
                                session.query(StockDailyData).filter(
                                    and_(
                                        StockDailyData.stock_code == row['stock_code'],
                                        StockDailyData.trade_date == row['trade_date']
                                    )
                                )
                            )
                            
                            if existing.first():
                                continue  # Skip existing record
                        
                        # Create new record
                        record = StockDailyData(
                            stock_code=row['stock_code'],
                            trade_date=row['trade_date'],
                            open_price=row.get('open_price'),
                            high_price=row.get('high_price'),
                            low_price=row.get('low_price'),
                            close_price=row.get('close_price'),
                            volume=row.get('volume'),
                            amount=row.get('amount'),
                            adj_factor=row.get('adj_factor', 1.0)
                        )
                        
                        session.add(record)
                        loaded_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to load record: {str(e)}")
                        failed_count += 1
                
                # Commit batch
                await session.commit()
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Batch commit failed: {str(e)}")
                failed_count = len(batch)
                loaded_count = 0
        
        return {
            'loaded': loaded_count,
            'failed': failed_count
        }
    
    def _finalize_metrics(self) -> ETLMetrics:
        """Finalize metrics and return."""
        self.metrics.end_time = datetime.now()
        
        logger.info(f"ETL job completed:")
        logger.info(f"  Status: {self.metrics.status.value}")
        logger.info(f"  Duration: {self.metrics.duration}")
        logger.info(f"  Records extracted: {self.metrics.records_extracted}")
        logger.info(f"  Records loaded: {self.metrics.records_loaded}")
        logger.info(f"  Success rate: {self.metrics.success_rate:.1f}%")
        
        if self.metrics.error_messages:
            logger.info(f"  Errors: {len(self.metrics.error_messages)}")
        
        return self.metrics
    
    async def get_last_update_date(self, symbol: str) -> Optional[date]:
        """Get the last update date for a symbol."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    session.query(StockDailyData.trade_date)
                    .filter(StockDailyData.stock_code == symbol)
                    .order_by(StockDailyData.trade_date.desc())
                    .limit(1)
                )
                
                row = result.first()
                return row[0] if row else None
                
        except Exception as e:
            logger.error(f"Failed to get last update date for {symbol}: {str(e)}")
            return None
    
    async def get_missing_dates(self, symbol: str, start_date: date, end_date: date) -> List[date]:
        """Get list of missing dates for a symbol in the given range."""
        try:
            async with AsyncSessionLocal() as session:
                # Get existing dates
                result = await session.execute(
                    session.query(StockDailyData.trade_date)
                    .filter(
                        and_(
                            StockDailyData.stock_code == symbol,
                            StockDailyData.trade_date >= start_date,
                            StockDailyData.trade_date <= end_date
                        )
                    )
                    .order_by(StockDailyData.trade_date)
                )
                
                existing_dates = {row[0] for row in result.all()}
                
                # Generate expected date range (excluding weekends)
                expected_dates = []
                current_date = start_date
                while current_date <= end_date:
                    # Skip weekends (assuming stock market is closed)
                    if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                        expected_dates.append(current_date)
                    current_date += timedelta(days=1)
                
                # Find missing dates
                missing_dates = [d for d in expected_dates if d not in existing_dates]
                
                return missing_dates
                
        except Exception as e:
            logger.error(f"Failed to get missing dates for {symbol}: {str(e)}")
            return []


class ETLJobManager:
    """Manager for ETL jobs and scheduling."""
    
    def __init__(self):
        self.pipeline = ETLPipeline()
        self.active_jobs = {}
    
    async def create_daily_update_job(self, symbols: List[str] = None) -> ETLJobConfig:
        """Create a job configuration for daily data updates."""
        if symbols is None:
            # Get all active symbols from database
            symbols = await self._get_active_symbols()
        
        # Update data for the last 7 days to catch any late updates
        end_date = date.today() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=7)  # Last week
        
        return ETLJobConfig(
            job_name=f"daily_update_{datetime.now().strftime('%Y%m%d')}",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            batch_size=50,
            quality_threshold=0.6,  # Lower threshold for daily updates
            retry_failed=True,
            skip_existing=False,  # Update existing records
            validate_data=True,
            clean_data=True
        )
    
    async def create_historical_backfill_job(self, symbols: List[str], years: int = 5) -> ETLJobConfig:
        """Create a job configuration for historical data backfill."""
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=365 * years)
        
        return ETLJobConfig(
            job_name=f"backfill_{datetime.now().strftime('%Y%m%d')}",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            batch_size=100,
            quality_threshold=0.7,
            retry_failed=True,
            skip_existing=True,
            validate_data=True,
            clean_data=True
        )
    
    async def _get_active_symbols(self) -> List[str]:
        """Get list of active stock symbols."""
        try:
            # This would typically come from a configuration table
            # For now, return a default list
            return [
                '000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ'
            ]
        except Exception as e:
            logger.error(f"Failed to get active symbols: {str(e)}")
            return []
    
    async def run_job_async(self, config: ETLJobConfig) -> ETLMetrics:
        """Run ETL job asynchronously."""
        job_id = f"{config.job_name}_{datetime.now().strftime('%H%M%S')}"
        
        try:
            self.active_jobs[job_id] = {
                'config': config,
                'start_time': datetime.now(),
                'status': 'running'
            }
            
            await self.pipeline.initialize()
            metrics = await self.pipeline.run_job(config)
            
            self.active_jobs[job_id]['status'] = metrics.status.value
            self.active_jobs[job_id]['metrics'] = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            self.active_jobs[job_id]['status'] = 'failed'
            self.active_jobs[job_id]['error'] = str(e)
            raise
        
        finally:
            # Clean up old jobs (keep last 10)
            if len(self.active_jobs) > 10:
                oldest_jobs = sorted(
                    self.active_jobs.keys(),
                    key=lambda x: self.active_jobs[x]['start_time']
                )[:-10]
                for job_id in oldest_jobs:
                    del self.active_jobs[job_id]
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        return self.active_jobs.get(job_id)
    
    def get_active_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get all active jobs."""
        return self.active_jobs.copy()


# Global instance
etl_manager = ETLJobManager()


async def get_etl_manager() -> ETLJobManager:
    """Get the global ETL manager instance."""
    return etl_manager