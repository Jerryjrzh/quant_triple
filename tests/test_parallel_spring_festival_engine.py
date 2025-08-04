"""Tests for parallel Spring Festival Engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from stock_analysis_system.analysis.parallel_spring_festival_engine import (
    ParallelSpringFestivalEngine,
    ParallelProcessingConfig,
    DaskResourceManager,
    BatchProcessingResult,
    create_dask_dataframe_from_stock_data,
    optimize_dask_config_for_spring_festival_analysis
)
from stock_analysis_system.analysis.spring_festival_engine import SeasonalPattern


class TestParallelProcessingConfig:
    """Test ParallelProcessingConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ParallelProcessingConfig()
        
        assert config.n_workers == 4
        assert config.threads_per_worker == 2
        assert config.memory_limit == "2GB"
        assert config.chunk_size == 100
        assert config.enable_distributed is False
        assert config.scheduler_address is None
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ParallelProcessingConfig(
            n_workers=8,
            threads_per_worker=4,
            memory_limit="4GB",
            chunk_size=200,
            enable_distributed=True,
            scheduler_address="tcp://localhost:8786"
        )
        
        assert config.n_workers == 8
        assert config.threads_per_worker == 4
        assert config.memory_limit == "4GB"
        assert config.chunk_size == 200
        assert config.enable_distributed is True
        assert config.scheduler_address == "tcp://localhost:8786"
    
    @patch('stock_analysis_system.analysis.parallel_spring_festival_engine.get_settings')
    def test_from_settings(self, mock_get_settings):
        """Test configuration from settings."""
        mock_settings = Mock()
        mock_dask = Mock()
        mock_dask.n_workers = 6
        mock_dask.threads_per_worker = 3
        mock_dask.memory_limit = "3GB"
        mock_dask.chunk_size = 150
        mock_dask.enable_distributed = True
        mock_dask.scheduler_address = "tcp://localhost:8787"
        mock_settings.dask = mock_dask
        mock_get_settings.return_value = mock_settings
        
        config = ParallelProcessingConfig.from_settings()
        
        assert config.n_workers == 6
        assert config.threads_per_worker == 3
        assert config.memory_limit == "3GB"
        assert config.chunk_size == 150
        assert config.enable_distributed is True
        assert config.scheduler_address == "tcp://localhost:8787"


class TestBatchProcessingResult:
    """Test BatchProcessingResult."""
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        successful = {"AAPL": Mock(), "GOOGL": Mock()}
        failed = {"MSFT": "Error message"}
        
        result = BatchProcessingResult(
            successful_analyses=successful,
            failed_analyses=failed,
            processing_time=10.5,
            total_stocks=3,
            memory_usage={}
        )
        
        assert abs(result.success_rate - 2/3) < 1e-10
        assert abs(result.failure_rate - 1/3) < 1e-10
    
    def test_empty_result(self):
        """Test empty result."""
        result = BatchProcessingResult(
            successful_analyses={},
            failed_analyses={},
            processing_time=0.0,
            total_stocks=0,
            memory_usage={}
        )
        
        assert result.success_rate == 0.0
        assert result.failure_rate == 1.0  # When total_stocks is 0, failure_rate is 1.0


class TestDaskResourceManager:
    """Test DaskResourceManager."""
    
    def test_init(self):
        """Test initialization."""
        config = ParallelProcessingConfig()
        manager = DaskResourceManager(config)
        
        assert manager.config == config
        assert manager.client is None
        assert manager._cluster is None
    
    @patch('stock_analysis_system.analysis.parallel_spring_festival_engine.Client')
    def test_start_distributed_cluster(self, mock_client_class):
        """Test starting distributed cluster."""
        mock_client = Mock()
        mock_client.dashboard_link = "http://localhost:8787"
        mock_client_class.return_value = mock_client
        
        config = ParallelProcessingConfig(
            enable_distributed=True,
            scheduler_address="tcp://localhost:8786"
        )
        manager = DaskResourceManager(config)
        
        client = manager.start_cluster()
        
        assert client == mock_client
        assert manager.client == mock_client
        mock_client_class.assert_called_once_with("tcp://localhost:8786")
    
    @patch('dask.distributed.LocalCluster')
    @patch('dask.distributed.Client')
    def test_start_local_cluster(self, mock_client_class, mock_local_cluster_class):
        """Test starting local cluster."""
        mock_cluster = Mock()
        mock_client = Mock()
        mock_client.dashboard_link = "http://localhost:8787"
        
        mock_local_cluster_class.return_value = mock_cluster
        mock_client_class.return_value = mock_client
        
        config = ParallelProcessingConfig(enable_distributed=False)
        manager = DaskResourceManager(config)
        
        client = manager.start_cluster()
        
        assert client == mock_client
        assert manager.client == mock_client
        assert manager._cluster == mock_cluster
        
        mock_local_cluster_class.assert_called_once_with(
            n_workers=4,
            threads_per_worker=2,
            memory_limit="2GB",
            silence_logs=30  # logging.WARNING
        )
    
    def test_context_manager(self):
        """Test context manager functionality."""
        config = ParallelProcessingConfig()
        manager = DaskResourceManager(config)
        
        with patch.object(manager, 'start_cluster') as mock_start:
            with patch.object(manager, 'shutdown_cluster') as mock_shutdown:
                with manager as rm:
                    assert rm == manager
                    mock_start.assert_called_once()
                mock_shutdown.assert_called_once()
    
    def test_get_cluster_info_not_connected(self):
        """Test get_cluster_info when not connected."""
        config = ParallelProcessingConfig()
        manager = DaskResourceManager(config)
        
        info = manager.get_cluster_info()
        assert info == {"status": "not_connected"}
    
    def test_get_cluster_info_connected(self):
        """Test get_cluster_info when connected."""
        config = ParallelProcessingConfig()
        manager = DaskResourceManager(config)
        
        mock_client = Mock()
        mock_client.dashboard_link = "http://localhost:8787"
        mock_client.scheduler_info.return_value = {
            "workers": {
                "worker1": {"nthreads": 2, "memory_limit": 2000000000},
                "worker2": {"nthreads": 2, "memory_limit": 2000000000}
            }
        }
        manager.client = mock_client
        
        info = manager.get_cluster_info()
        
        expected = {
            "status": "connected",
            "workers": 2,
            "total_cores": 4,
            "total_memory": 4000000000,
            "dashboard_link": "http://localhost:8787"
        }
        assert info == expected


class TestParallelSpringFestivalEngine:
    """Test ParallelSpringFestivalEngine."""
    
    @pytest.fixture
    def sample_stock_data(self):
        """Create sample stock data for testing."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        
        stock_data = {}
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            data = pd.DataFrame({
                'stock_code': symbol,
                'trade_date': dates,
                'open_price': np.random.uniform(100, 200, len(dates)),
                'high_price': np.random.uniform(100, 200, len(dates)),
                'low_price': np.random.uniform(100, 200, len(dates)),
                'close_price': np.random.uniform(100, 200, len(dates)),
                'volume': np.random.randint(1000000, 10000000, len(dates))
            })
            # Ensure high >= low and close is within range
            data['high_price'] = np.maximum(data['high_price'], data['low_price'])
            data['close_price'] = np.clip(data['close_price'], data['low_price'], data['high_price'])
            stock_data[symbol] = data
        
        return stock_data
    
    def test_init(self):
        """Test initialization."""
        config = ParallelProcessingConfig()
        engine = ParallelSpringFestivalEngine(window_days=60, config=config)
        
        assert engine.window_days == 60
        assert engine.config == config
        assert isinstance(engine.resource_manager, DaskResourceManager)
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        with patch('stock_analysis_system.analysis.parallel_spring_festival_engine.ParallelProcessingConfig.from_settings') as mock_from_settings:
            mock_config = ParallelProcessingConfig()
            mock_from_settings.return_value = mock_config
            
            engine = ParallelSpringFestivalEngine()
            
            assert engine.config == mock_config
            mock_from_settings.assert_called_once()
    
    @patch('stock_analysis_system.analysis.parallel_spring_festival_engine.SpringFestivalAlignmentEngine.align_to_spring_festival')
    @patch('stock_analysis_system.analysis.parallel_spring_festival_engine.SpringFestivalAlignmentEngine.identify_seasonal_patterns')
    def test_analyze_stock_chunk(self, mock_identify_patterns, mock_align, sample_stock_data):
        """Test analyzing a chunk of stocks."""
        # Setup mocks
        mock_aligned_data = Mock()
        mock_pattern = Mock()
        mock_align.return_value = mock_aligned_data
        mock_identify_patterns.return_value = mock_pattern
        
        engine = ParallelSpringFestivalEngine()
        
        # Test with subset of data
        chunk_data = {k: v for k, v in list(sample_stock_data.items())[:2]}
        result = engine._analyze_stock_chunk(chunk_data, [2020, 2021, 2022])
        
        assert len(result['successful']) == 2
        assert len(result['failed']) == 0
        assert 'AAPL' in result['successful']
        assert 'GOOGL' in result['successful']
    
    @patch('stock_analysis_system.analysis.parallel_spring_festival_engine.SpringFestivalAlignmentEngine.align_to_spring_festival')
    def test_analyze_stock_chunk_with_error(self, mock_align, sample_stock_data):
        """Test analyzing a chunk with errors."""
        # Setup mock to raise exception
        mock_align.side_effect = Exception("Test error")
        
        engine = ParallelSpringFestivalEngine()
        
        chunk_data = {'AAPL': sample_stock_data['AAPL']}
        result = engine._analyze_stock_chunk(chunk_data, [2020, 2021, 2022])
        
        assert len(result['successful']) == 0
        assert len(result['failed']) == 1
        assert 'AAPL' in result['failed']
        assert result['failed']['AAPL'] == "Test error"
    
    @patch('stock_analysis_system.analysis.parallel_spring_festival_engine.ThreadPoolExecutor')
    def test_analyze_with_threads(self, mock_executor_class, sample_stock_data):
        """Test thread-based analysis."""
        # Setup mock executor
        mock_executor = Mock()
        mock_future = Mock()
        mock_future.result.return_value = {
            'successful': {'AAPL': Mock()},
            'failed': {}
        }
        mock_executor.submit.return_value = mock_future
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=None)
        mock_executor_class.return_value = mock_executor
        
        # Mock as_completed
        with patch('stock_analysis_system.analysis.parallel_spring_festival_engine.as_completed') as mock_as_completed:
            mock_as_completed.return_value = [mock_future]
            
            engine = ParallelSpringFestivalEngine()
            result = engine._analyze_with_threads({'AAPL': sample_stock_data['AAPL']}, [2020, 2021, 2022])
            
            assert len(result['successful']) == 1
            assert len(result['failed']) == 0
            assert 'AAPL' in result['successful']
    
    def test_optimize_for_memory_usage(self, sample_stock_data):
        """Test memory optimization."""
        engine = ParallelSpringFestivalEngine()
        
        # Add some extra columns to test removal
        for symbol, data in sample_stock_data.items():
            data['extra_column'] = 'test'
            data['another_column'] = 123
        
        optimized_data = engine.optimize_for_memory_usage(sample_stock_data)
        
        # Check that data is optimized
        for symbol, data in optimized_data.items():
            # Should only have required columns
            expected_cols = ['stock_code', 'trade_date', 'close_price', 'volume']
            assert set(data.columns) == set(expected_cols)
            
            # Check data types are optimized
            assert pd.api.types.is_float_dtype(data['close_price'])
            assert pd.api.types.is_integer_dtype(data['volume'])
            assert pd.api.types.is_datetime64_any_dtype(data['trade_date'])
    
    def test_get_processing_recommendations(self):
        """Test processing recommendations."""
        engine = ParallelSpringFestivalEngine()
        
        recommendations = engine.get_processing_recommendations(
            total_stocks=1000,
            avg_data_points_per_stock=2000
        )
        
        assert 'total_stocks' in recommendations
        assert 'estimated_memory_gb' in recommendations
        assert 'recommended_chunk_size' in recommendations
        assert 'recommended_workers' in recommendations
        assert 'estimated_sequential_time_minutes' in recommendations
        assert 'estimated_parallel_time_minutes' in recommendations
        assert 'speedup_factor' in recommendations
        assert 'use_distributed' in recommendations
        assert 'memory_optimization_needed' in recommendations
        
        assert recommendations['total_stocks'] == 1000
        assert recommendations['speedup_factor'] > 1
    
    def test_get_memory_usage(self):
        """Test memory usage retrieval."""
        with patch('psutil.Process') as mock_process_class:
            # Setup mock
            mock_process = Mock()
            mock_memory_info = Mock()
            mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB
            mock_memory_info.vms = 1024 * 1024 * 200  # 200 MB
            mock_process.memory_info.return_value = mock_memory_info
            mock_process.memory_percent.return_value = 5.5
            mock_process_class.return_value = mock_process
            
            engine = ParallelSpringFestivalEngine()
            memory_usage = engine._get_memory_usage()
            
            assert memory_usage['rss_mb'] == 100.0
            assert memory_usage['vms_mb'] == 200.0
            assert memory_usage['percent'] == 5.5
    
    def test_get_memory_usage_no_psutil(self):
        """Test memory usage when psutil is not available."""
        with patch('psutil.Process', side_effect=ImportError("No module named 'psutil'")):
            engine = ParallelSpringFestivalEngine()
            memory_usage = engine._get_memory_usage()
            
            assert memory_usage == {}


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_dask_dataframe_from_stock_data(self):
        """Test creating Dask DataFrame from stock data."""
        # Create sample data
        stock_data = {
            'AAPL': pd.DataFrame({
                'trade_date': pd.date_range('2020-01-01', periods=10),
                'close_price': np.random.uniform(100, 200, 10)
            }),
            'GOOGL': pd.DataFrame({
                'trade_date': pd.date_range('2020-01-01', periods=10),
                'close_price': np.random.uniform(100, 200, 10)
            })
        }
        
        dask_df = create_dask_dataframe_from_stock_data(stock_data)
        
        # Check that it's a Dask DataFrame
        import dask.dataframe as dd
        assert isinstance(dask_df, dd.DataFrame)
        
        # Check that data is combined correctly
        computed_df = dask_df.compute()
        assert len(computed_df) == 20  # 10 rows per stock
        assert 'stock_code' in computed_df.columns
        assert set(computed_df['stock_code'].unique()) == {'AAPL', 'GOOGL'}
    
    @patch('stock_analysis_system.analysis.parallel_spring_festival_engine.dask')
    def test_optimize_dask_config(self, mock_dask):
        """Test Dask configuration optimization."""
        optimize_dask_config_for_spring_festival_analysis()
        
        # Check that dask.config.set was called
        mock_dask.config.set.assert_called_once()
        
        # Check some of the configuration values
        config_dict = mock_dask.config.set.call_args[0][0]
        assert 'dataframe.query-planning' in config_dict
        assert 'array.chunk-size' in config_dict
        assert 'distributed.worker.memory.target' in config_dict


class TestIntegration:
    """Integration tests."""
    
    @pytest.fixture
    def small_stock_data(self):
        """Create small stock data for integration testing."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        
        stock_data = {}
        for symbol in ['TEST1', 'TEST2']:
            data = pd.DataFrame({
                'stock_code': symbol,
                'trade_date': dates,
                'close_price': 100 + np.random.randn(len(dates)).cumsum() * 0.1,
                'volume': np.random.randint(1000000, 10000000, len(dates))
            })
            stock_data[symbol] = data
        
        return stock_data
    
    @patch('stock_analysis_system.analysis.parallel_spring_festival_engine.DaskResourceManager')
    def test_analyze_multiple_stocks_parallel_fallback(self, mock_resource_manager_class, small_stock_data):
        """Test parallel analysis with fallback to sequential processing."""
        # Setup mock resource manager that fails
        mock_rm = Mock()
        mock_rm.client = None
        mock_rm.__enter__ = Mock(return_value=mock_rm)
        mock_rm.__exit__ = Mock(return_value=None)
        mock_resource_manager_class.return_value = mock_rm
        
        engine = ParallelSpringFestivalEngine()
        
        # This should fallback to sequential processing
        with patch.object(engine, 'align_to_spring_festival') as mock_align:
            with patch.object(engine, 'identify_seasonal_patterns') as mock_identify:
                mock_aligned = Mock()
                mock_pattern = Mock()
                mock_align.return_value = mock_aligned
                mock_identify.return_value = mock_pattern
                
                result = engine.analyze_multiple_stocks_parallel(small_stock_data, [2020])
                
                assert isinstance(result, BatchProcessingResult)
                assert len(result.successful_analyses) == 2
                assert len(result.failed_analyses) == 0
                assert result.total_stocks == 2
    
    def test_stream_data_batches_dict(self):
        """Test streaming data batches from dictionary."""
        engine = ParallelSpringFestivalEngine()
        
        data_source = {f'STOCK{i}': f'data{i}' for i in range(10)}
        
        batches = list(engine._stream_data_batches(data_source, batch_size=3))
        
        assert len(batches) == 4  # 3 + 3 + 3 + 1
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1
    
    def test_stream_data_batches_iterator(self):
        """Test streaming data batches from iterator."""
        engine = ParallelSpringFestivalEngine()
        
        def data_iterator():
            for i in range(5):
                yield f'STOCK{i}', f'data{i}'
        
        batches = list(engine._stream_data_batches(data_iterator(), batch_size=2))
        
        assert len(batches) == 3  # 2 + 2 + 1
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1


if __name__ == '__main__':
    pytest.main([__file__])