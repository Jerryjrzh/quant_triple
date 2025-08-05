"""Tests for ETL Pipeline."""

import asyncio
import os
import tempfile
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from stock_analysis_system.etl.pipeline import (
    ETLJobConfig,
    ETLJobManager,
    ETLMetrics,
    ETLPipeline,
    ETLStage,
    ETLStatus,
)


class TestETLJobConfig:
    """Test ETL job configuration."""

    def test_job_config_creation(self):
        """Test creating ETL job configuration."""
        config = ETLJobConfig(
            job_name="test_job",
            symbols=["000001.SZ", "000002.SZ"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            batch_size=50,
            quality_threshold=0.8,
        )

        assert config.job_name == "test_job"
        assert len(config.symbols) == 2
        assert config.batch_size == 50
        assert config.quality_threshold == 0.8
        assert config.retry_failed is True  # Default value
        assert config.skip_existing is True  # Default value


class TestETLMetrics:
    """Test ETL metrics."""

    def test_metrics_creation(self):
        """Test creating ETL metrics."""
        start_time = datetime.now()
        metrics = ETLMetrics(start_time=start_time)

        assert metrics.start_time == start_time
        assert metrics.stage == ETLStage.EXTRACT
        assert metrics.status == ETLStatus.PENDING
        assert metrics.records_extracted == 0
        assert metrics.error_messages == []

    def test_metrics_duration(self):
        """Test duration calculation."""
        start_time = datetime.now()
        metrics = ETLMetrics(start_time=start_time)

        # No end time yet
        assert metrics.duration is None

        # Set end time
        metrics.end_time = start_time + timedelta(minutes=5)
        assert metrics.duration == timedelta(minutes=5)

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ETLMetrics(start_time=datetime.now())

        # No records
        assert metrics.success_rate == 0.0

        # Some records
        metrics.records_extracted = 100
        metrics.records_loaded = 80
        assert metrics.success_rate == 80.0


class TestETLPipeline:
    """Test ETL pipeline."""

    @pytest.fixture
    def sample_config(self):
        """Create sample ETL job configuration."""
        return ETLJobConfig(
            job_name="test_pipeline",
            symbols=["000001.SZ"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
            batch_size=10,
            quality_threshold=0.7,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample stock data."""
        return pd.DataFrame(
            {
                "stock_code": ["000001.SZ"] * 5,
                "trade_date": pd.date_range("2024-01-01", periods=5),
                "open_price": [10.0, 11.0, 12.0, 13.0, 14.0],
                "high_price": [11.0, 12.0, 13.0, 14.0, 15.0],
                "low_price": [9.0, 10.0, 11.0, 12.0, 13.0],
                "close_price": [10.5, 11.5, 12.5, 13.5, 14.5],
                "volume": [1000, 1100, 1200, 1300, 1400],
                "amount": [10500, 12650, 15000, 17550, 20300],
            }
        )

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = ETLPipeline()

        # Should not be initialized yet
        assert pipeline.data_manager is None

        # Mock the data manager
        with patch(
            "stock_analysis_system.etl.pipeline.get_data_source_manager"
        ) as mock_get_manager:
            mock_manager = AsyncMock()
            mock_get_manager.return_value = mock_manager

            await pipeline.initialize()

            assert pipeline.data_manager is not None
            mock_get_manager.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_data_success(self, sample_config, sample_data):
        """Test successful data extraction."""
        pipeline = ETLPipeline()

        # Mock data manager
        mock_manager = AsyncMock()
        mock_manager.get_stock_data.return_value = sample_data
        pipeline.data_manager = mock_manager

        # Initialize metrics
        pipeline.metrics = ETLMetrics(start_time=datetime.now())

        result = await pipeline._extract_data(sample_config)

        assert not result.empty
        assert len(result) == len(sample_data)
        mock_manager.get_stock_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_data_failure(self, sample_config):
        """Test data extraction failure."""
        pipeline = ETLPipeline()

        # Mock data manager to raise exception
        mock_manager = AsyncMock()
        mock_manager.get_stock_data.side_effect = Exception("Data source error")
        pipeline.data_manager = mock_manager

        # Initialize metrics
        pipeline.metrics = ETLMetrics(start_time=datetime.now())

        result = await pipeline._extract_data(sample_config)

        assert result is None
        assert len(pipeline.metrics.error_messages) > 0

    @pytest.mark.asyncio
    async def test_transform_data_success(self, sample_config, sample_data):
        """Test successful data transformation."""
        pipeline = ETLPipeline()

        result = await pipeline._transform_data(sample_data, sample_config)

        assert not result.empty
        assert "created_at" in result.columns
        assert "updated_at" in result.columns

        # Check data types
        assert result["trade_date"].dtype == "object"  # Should be date objects
        assert pd.api.types.is_numeric_dtype(result["close_price"])

    @pytest.mark.asyncio
    async def test_transform_data_missing_columns(self, sample_config):
        """Test transformation with missing required columns."""
        pipeline = ETLPipeline()

        # Data missing required columns
        bad_data = pd.DataFrame(
            {
                "symbol": ["000001.SZ"],  # Wrong column name
                "price": [10.0],  # Missing trade_date, close_price
            }
        )

        result = await pipeline._transform_data(bad_data, sample_config)

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_data_quality(self, sample_data):
        """Test data quality validation."""
        pipeline = ETLPipeline()

        report = await pipeline._validate_data_quality(sample_data, "test_job")

        assert report is not None
        assert report.dataset_name == "test_job"
        assert report.total_rows == len(sample_data)
        assert 0.0 <= report.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_load_data_success(self, sample_config, sample_data):
        """Test successful data loading."""
        pipeline = ETLPipeline()

        # Mock the batch loading
        with patch.object(pipeline, "_load_batch") as mock_load_batch:
            mock_load_batch.return_value = {"loaded": 5, "failed": 0}

            result = await pipeline._load_data(sample_data, sample_config)

            assert result["loaded"] == 5
            assert result["failed"] == 0
            mock_load_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_job_complete_success(self, sample_config, sample_data):
        """Test complete ETL job execution."""
        pipeline = ETLPipeline()

        # Mock all dependencies
        mock_manager = AsyncMock()
        mock_manager.get_stock_data.return_value = sample_data

        with patch(
            "stock_analysis_system.etl.pipeline.get_data_source_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_manager

            with patch.object(pipeline, "_load_batch") as mock_load_batch:
                mock_load_batch.return_value = {"loaded": 5, "failed": 0}

                metrics = await pipeline.run_job(sample_config)

                assert metrics.status == ETLStatus.SUCCESS
                assert metrics.records_extracted == len(sample_data)
                assert metrics.records_loaded == 5
                assert metrics.end_time is not None

    @pytest.mark.asyncio
    async def test_run_job_with_quality_issues(self, sample_config):
        """Test ETL job with data quality issues."""
        pipeline = ETLPipeline()

        # Create problematic data
        bad_data = pd.DataFrame(
            {
                "stock_code": ["000001.SZ"],
                "trade_date": [date.today()],
                "close_price": [None],  # Missing value
                "volume": [-1000],  # Negative volume
            }
        )

        mock_manager = AsyncMock()
        mock_manager.get_stock_data.return_value = bad_data

        with patch(
            "stock_analysis_system.etl.pipeline.get_data_source_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_manager

            with patch.object(pipeline, "_load_batch") as mock_load_batch:
                mock_load_batch.return_value = {"loaded": 1, "failed": 0}

                # Set low quality threshold
                sample_config.quality_threshold = 0.9
                sample_config.clean_data = True

                metrics = await pipeline.run_job(sample_config)

                # Should still succeed due to cleaning
                assert metrics.status in [ETLStatus.SUCCESS, ETLStatus.PARTIAL_SUCCESS]
                assert metrics.quality_score < 0.9


class TestETLJobManager:
    """Test ETL job manager."""

    @pytest.mark.asyncio
    async def test_create_daily_update_job(self):
        """Test creating daily update job configuration."""
        manager = ETLJobManager()

        with patch.object(manager, "_get_active_symbols") as mock_get_symbols:
            mock_get_symbols.return_value = ["000001.SZ", "000002.SZ"]

            config = await manager.create_daily_update_job()

            assert config.job_name.startswith("daily_update_")
            assert len(config.symbols) == 2
            assert config.end_date == date.today() - timedelta(days=1)
            assert config.start_date == config.end_date - timedelta(days=7)

    @pytest.mark.asyncio
    async def test_create_historical_backfill_job(self):
        """Test creating historical backfill job configuration."""
        manager = ETLJobManager()
        symbols = ["000001.SZ", "000002.SZ"]

        config = await manager.create_historical_backfill_job(symbols, years=2)

        assert config.job_name.startswith("backfill_")
        assert config.symbols == symbols
        assert config.end_date == date.today() - timedelta(days=1)
        assert config.start_date == config.end_date - timedelta(days=365 * 2)

    @pytest.mark.asyncio
    async def test_run_job_async(self):
        """Test running job asynchronously."""
        manager = ETLJobManager()

        config = ETLJobConfig(
            job_name="test_async_job",
            symbols=["000001.SZ"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
            batch_size=10,
        )

        # Mock the pipeline
        with patch.object(manager.pipeline, "initialize") as mock_init:
            with patch.object(manager.pipeline, "run_job") as mock_run:
                mock_metrics = ETLMetrics(start_time=datetime.now())
                mock_metrics.status = ETLStatus.SUCCESS
                mock_run.return_value = mock_metrics

                result = await manager.run_job_async(config)

                assert result.status == ETLStatus.SUCCESS
                mock_init.assert_called_once()
                mock_run.assert_called_once_with(config)

                # Check job tracking
                assert len(manager.active_jobs) == 1

    def test_get_job_status(self):
        """Test getting job status."""
        manager = ETLJobManager()

        # Add a mock job
        job_id = "test_job_123"
        manager.active_jobs[job_id] = {
            "config": Mock(),
            "start_time": datetime.now(),
            "status": "running",
        }

        status = manager.get_job_status(job_id)
        assert status is not None
        assert status["status"] == "running"

        # Non-existent job
        assert manager.get_job_status("non_existent") is None

    def test_get_active_jobs(self):
        """Test getting all active jobs."""
        manager = ETLJobManager()

        # Add mock jobs
        manager.active_jobs["job1"] = {"status": "running"}
        manager.active_jobs["job2"] = {"status": "completed"}

        active_jobs = manager.get_active_jobs()

        assert len(active_jobs) == 2
        assert "job1" in active_jobs
        assert "job2" in active_jobs

        # Should be a copy, not the original
        active_jobs["job3"] = {"status": "new"}
        assert "job3" not in manager.active_jobs


class TestETLIntegration:
    """Integration tests for ETL pipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self):
        """Test end-to-end ETL pipeline execution."""
        # This would be a more comprehensive integration test
        # For now, we'll test the basic flow

        pipeline = ETLPipeline()

        config = ETLJobConfig(
            job_name="integration_test",
            symbols=["000001.SZ"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
            batch_size=5,
            quality_threshold=0.5,  # Lower threshold for test
            validate_data=False,  # Skip validation for speed
            clean_data=False,
        )

        # Mock all external dependencies
        sample_data = pd.DataFrame(
            {
                "stock_code": ["000001.SZ"],
                "trade_date": [date(2024, 1, 1)],
                "open_price": [10.0],
                "high_price": [11.0],
                "low_price": [9.0],
                "close_price": [10.5],
                "volume": [1000],
                "amount": [10500],
            }
        )

        mock_manager = AsyncMock()
        mock_manager.get_stock_data.return_value = sample_data

        with patch(
            "stock_analysis_system.etl.pipeline.get_data_source_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_manager

            with patch.object(pipeline, "_load_batch") as mock_load_batch:
                mock_load_batch.return_value = {"loaded": 1, "failed": 0}

                metrics = await pipeline.run_job(config)

                # Verify the pipeline completed
                assert metrics is not None
                assert metrics.status in [ETLStatus.SUCCESS, ETLStatus.PARTIAL_SUCCESS]
                assert metrics.records_extracted > 0
                assert metrics.end_time is not None
                assert metrics.duration is not None


if __name__ == "__main__":
    pytest.main([__file__])
