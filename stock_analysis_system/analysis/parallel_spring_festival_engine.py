"""Parallel processing extension for Spring Festival Engine using Dask."""

import gc
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from dask.distributed import as_completed as dask_as_completed

from config.settings import get_settings

from .spring_festival_engine import (
    AlignedDataPoint,
    AlignedTimeSeries,
    ChineseCalendar,
    SeasonalPattern,
    SpringFestivalAlignmentEngine,
    SpringFestivalWindow,
)

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ParallelProcessingConfig:
    """Configuration for parallel processing."""

    n_workers: int = 4
    threads_per_worker: int = 2
    memory_limit: str = "2GB"
    chunk_size: int = 100  # Number of stocks per chunk
    enable_distributed: bool = False
    scheduler_address: Optional[str] = None

    @classmethod
    def from_settings(cls) -> "ParallelProcessingConfig":
        """Create config from application settings."""
        settings_instance = get_settings()
        dask_settings = settings_instance.dask
        return cls(
            n_workers=dask_settings.n_workers,
            threads_per_worker=dask_settings.threads_per_worker,
            memory_limit=dask_settings.memory_limit,
            chunk_size=dask_settings.chunk_size,
            enable_distributed=dask_settings.enable_distributed,
            scheduler_address=dask_settings.scheduler_address,
        )


@dataclass
class BatchProcessingResult:
    """Result of batch processing multiple stocks."""

    successful_analyses: Dict[str, SeasonalPattern]
    failed_analyses: Dict[str, str]  # symbol -> error message
    processing_time: float
    total_stocks: int
    memory_usage: Dict[str, float]

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_stocks == 0:
            return 0.0
        return len(self.successful_analyses) / self.total_stocks

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate


class DaskResourceManager:
    """Manages Dask cluster resources and memory optimization."""

    def __init__(self, config: ParallelProcessingConfig):
        self.config = config
        self.client: Optional[Client] = None
        self._cluster = None

    def __enter__(self):
        """Context manager entry."""
        self.start_cluster()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown_cluster()

    def start_cluster(self) -> Client:
        """Start Dask cluster."""
        try:
            if self.config.enable_distributed and self.config.scheduler_address:
                # Connect to existing distributed cluster
                logger.info(
                    f"Connecting to distributed Dask cluster at {self.config.scheduler_address}"
                )
                self.client = Client(self.config.scheduler_address)
            else:
                # Create local cluster
                logger.info(
                    f"Starting local Dask cluster with {self.config.n_workers} workers"
                )
                from dask.distributed import LocalCluster

                self._cluster = LocalCluster(
                    n_workers=self.config.n_workers,
                    threads_per_worker=self.config.threads_per_worker,
                    memory_limit=self.config.memory_limit,
                    silence_logs=logging.WARNING,
                )
                self.client = Client(self._cluster)

            logger.info(f"Dask cluster started: {self.client.dashboard_link}")
            return self.client

        except Exception as e:
            logger.error(f"Failed to start Dask cluster: {e}")
            # Fallback to synchronous processing
            self.client = None
            return None

    def shutdown_cluster(self):
        """Shutdown Dask cluster."""
        if self.client:
            logger.info("Shutting down Dask cluster")
            self.client.close()
            self.client = None

        if self._cluster:
            self._cluster.close()
            self._cluster = None

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information."""
        if not self.client:
            return {"status": "not_connected"}

        try:
            info = self.client.scheduler_info()
            return {
                "status": "connected",
                "workers": len(info.get("workers", {})),
                "total_cores": sum(
                    w.get("nthreads", 0) for w in info.get("workers", {}).values()
                ),
                "total_memory": sum(
                    w.get("memory_limit", 0) for w in info.get("workers", {}).values()
                ),
                "dashboard_link": self.client.dashboard_link,
            }
        except Exception as e:
            logger.error(f"Failed to get cluster info: {e}")
            return {"status": "error", "error": str(e)}

    def optimize_memory(self):
        """Optimize memory usage."""
        if self.client:
            try:
                # Run garbage collection on all workers
                self.client.run(gc.collect)
                logger.info("Memory optimization completed on all workers")
            except Exception as e:
                logger.warning(f"Memory optimization failed: {e}")


class ParallelSpringFestivalEngine(SpringFestivalAlignmentEngine):
    """Extended Spring Festival Engine with parallel processing capabilities."""

    def __init__(
        self, window_days: int = None, config: ParallelProcessingConfig = None
    ):
        super().__init__(window_days)
        self.config = config or ParallelProcessingConfig.from_settings()
        self.resource_manager = DaskResourceManager(self.config)

    def analyze_multiple_stocks_parallel(
        self, stock_data_dict: Dict[str, pd.DataFrame], years: List[int] = None
    ) -> BatchProcessingResult:
        """Analyze multiple stocks in parallel using Dask."""
        start_time = datetime.now()
        logger.info(f"Starting parallel analysis of {len(stock_data_dict)} stocks")

        successful_analyses = {}
        failed_analyses = {}

        try:
            with self.resource_manager as rm:
                if rm.client:
                    # Use Dask distributed processing
                    result = self._analyze_with_dask(stock_data_dict, years, rm.client)
                else:
                    # Fallback to thread-based parallel processing
                    logger.warning(
                        "Dask cluster not available, falling back to thread-based processing"
                    )
                    result = self._analyze_with_threads(stock_data_dict, years)

                successful_analyses = result["successful"]
                failed_analyses = result["failed"]

        except Exception as e:
            logger.error(f"Parallel analysis failed: {e}")
            # Fallback to sequential processing
            logger.info("Falling back to sequential processing")
            for symbol, data in stock_data_dict.items():
                try:
                    aligned_data = self.align_to_spring_festival(data, years)
                    pattern = self.identify_seasonal_patterns(aligned_data)
                    successful_analyses[symbol] = pattern
                except Exception as stock_error:
                    failed_analyses[symbol] = str(stock_error)

        processing_time = (datetime.now() - start_time).total_seconds()

        # Get memory usage info
        memory_usage = self._get_memory_usage()

        result = BatchProcessingResult(
            successful_analyses=successful_analyses,
            failed_analyses=failed_analyses,
            processing_time=processing_time,
            total_stocks=len(stock_data_dict),
            memory_usage=memory_usage,
        )

        logger.info(f"Parallel analysis completed in {processing_time:.2f}s")
        logger.info(
            f"Success rate: {result.success_rate:.1%} ({len(successful_analyses)}/{len(stock_data_dict)})"
        )

        return result

    def _analyze_with_dask(
        self, stock_data_dict: Dict[str, pd.DataFrame], years: List[int], client: Client
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze stocks using Dask distributed processing."""
        logger.info("Using Dask distributed processing")

        # Create delayed tasks
        delayed_tasks = []
        symbols = list(stock_data_dict.keys())

        # Process in chunks to manage memory
        chunks = [
            symbols[i : i + self.config.chunk_size]
            for i in range(0, len(symbols), self.config.chunk_size)
        ]

        for chunk in chunks:
            chunk_data = {symbol: stock_data_dict[symbol] for symbol in chunk}
            delayed_task = delayed(self._analyze_stock_chunk)(chunk_data, years)
            delayed_tasks.append(delayed_task)

        # Execute tasks with progress bar
        logger.info(f"Processing {len(chunks)} chunks with {len(delayed_tasks)} tasks")

        with ProgressBar():
            results = dask.compute(*delayed_tasks)

        # Combine results
        successful_analyses = {}
        failed_analyses = {}

        for chunk_result in results:
            successful_analyses.update(chunk_result["successful"])
            failed_analyses.update(chunk_result["failed"])

        return {"successful": successful_analyses, "failed": failed_analyses}

    def _analyze_with_threads(
        self, stock_data_dict: Dict[str, pd.DataFrame], years: List[int]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze stocks using thread-based parallel processing."""
        logger.info("Using thread-based parallel processing")

        successful_analyses = {}
        failed_analyses = {}

        # Process in chunks
        symbols = list(stock_data_dict.keys())
        chunks = [
            symbols[i : i + self.config.chunk_size]
            for i in range(0, len(symbols), self.config.chunk_size)
        ]

        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            # Submit chunk processing tasks
            future_to_chunk = {}
            for chunk in chunks:
                chunk_data = {symbol: stock_data_dict[symbol] for symbol in chunk}
                future = executor.submit(self._analyze_stock_chunk, chunk_data, years)
                future_to_chunk[future] = chunk

            # Collect results
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_result = future.result()
                    successful_analyses.update(chunk_result["successful"])
                    failed_analyses.update(chunk_result["failed"])
                    logger.info(f"Completed chunk with {len(chunk)} stocks")
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
                    for symbol in chunk:
                        failed_analyses[symbol] = str(e)

        return {"successful": successful_analyses, "failed": failed_analyses}

    def _analyze_stock_chunk(
        self, chunk_data: Dict[str, pd.DataFrame], years: List[int]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze a chunk of stocks."""
        successful = {}
        failed = {}

        for symbol, data in chunk_data.items():
            try:
                # Create a new engine instance for thread safety
                engine = SpringFestivalAlignmentEngine(self.window_days)
                aligned_data = engine.align_to_spring_festival(data, years)
                pattern = engine.identify_seasonal_patterns(aligned_data)
                successful[symbol] = pattern

            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {e}")
                failed[symbol] = str(e)

        return {"successful": successful, "failed": failed}

    def analyze_large_dataset_streaming(
        self,
        data_source: Any,  # Could be database connection, file iterator, etc.
        batch_size: int = 1000,
        years: List[int] = None,
    ) -> BatchProcessingResult:
        """Analyze large datasets using streaming processing."""
        logger.info(f"Starting streaming analysis with batch size {batch_size}")

        start_time = datetime.now()
        successful_analyses = {}
        failed_analyses = {}
        total_processed = 0

        try:
            with self.resource_manager as rm:
                # Process data in batches
                for batch_data in self._stream_data_batches(data_source, batch_size):
                    batch_result = self.analyze_multiple_stocks_parallel(
                        batch_data, years
                    )

                    successful_analyses.update(batch_result.successful_analyses)
                    failed_analyses.update(batch_result.failed_analyses)
                    total_processed += len(batch_data)

                    logger.info(
                        f"Processed batch: {len(batch_data)} stocks, "
                        f"Total: {total_processed}, "
                        f"Success rate: {len(successful_analyses)/total_processed:.1%}"
                    )

                    # Memory optimization after each batch
                    if rm.client:
                        rm.optimize_memory()

                    # Force garbage collection
                    gc.collect()

        except Exception as e:
            logger.error(f"Streaming analysis failed: {e}")

        processing_time = (datetime.now() - start_time).total_seconds()
        memory_usage = self._get_memory_usage()

        return BatchProcessingResult(
            successful_analyses=successful_analyses,
            failed_analyses=failed_analyses,
            processing_time=processing_time,
            total_stocks=total_processed,
            memory_usage=memory_usage,
        )

    def _stream_data_batches(self, data_source: Any, batch_size: int):
        """Stream data in batches from data source."""
        # This is a placeholder - implementation depends on data source type
        # Could be database cursor, file reader, API paginator, etc.

        if hasattr(data_source, "items"):
            # Dictionary-like source
            items = list(data_source.items())
            for i in range(0, len(items), batch_size):
                batch = dict(items[i : i + batch_size])
                yield batch
        else:
            # Custom iterator
            batch = {}
            for symbol, data in data_source:
                batch[symbol] = data
                if len(batch) >= batch_size:
                    yield batch
                    batch = {}

            if batch:  # Yield remaining items
                yield batch

    def optimize_for_memory_usage(
        self, stock_data_dict: Dict[str, pd.DataFrame], max_memory_gb: float = 4.0
    ) -> Dict[str, pd.DataFrame]:
        """Optimize data for memory-efficient processing."""
        logger.info(f"Optimizing data for memory usage (max: {max_memory_gb}GB)")

        optimized_data = {}

        for symbol, data in stock_data_dict.items():
            # Convert to more memory-efficient dtypes
            optimized_df = data.copy()

            # Optimize numeric columns
            for col in ["open_price", "high_price", "low_price", "close_price"]:
                if col in optimized_df.columns:
                    optimized_df[col] = pd.to_numeric(
                        optimized_df[col], downcast="float"
                    )

            if "volume" in optimized_df.columns:
                optimized_df["volume"] = pd.to_numeric(
                    optimized_df["volume"], downcast="integer"
                )

            # Optimize date column
            if "trade_date" in optimized_df.columns:
                optimized_df["trade_date"] = pd.to_datetime(optimized_df["trade_date"])

            # Remove unnecessary columns for analysis
            required_cols = ["stock_code", "trade_date", "close_price", "volume"]
            available_cols = [
                col for col in required_cols if col in optimized_df.columns
            ]
            optimized_df = optimized_df[available_cols]

            optimized_data[symbol] = optimized_df

        # Check memory usage
        total_memory = sum(
            df.memory_usage(deep=True).sum() for df in optimized_data.values()
        )
        memory_gb = total_memory / (1024**3)

        logger.info(f"Optimized data memory usage: {memory_gb:.2f}GB")

        if memory_gb > max_memory_gb:
            logger.warning(
                f"Memory usage ({memory_gb:.2f}GB) exceeds limit ({max_memory_gb}GB)"
            )
            logger.info("Consider processing in smaller batches")

        return optimized_data

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "rss_mb": memory_info.rss / (1024**2),  # Resident Set Size
                "vms_mb": memory_info.vms / (1024**2),  # Virtual Memory Size
                "percent": process.memory_percent(),
            }
        except ImportError:
            logger.warning("psutil not available, cannot get memory usage")
            return {}
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return {}

    def get_processing_recommendations(
        self, total_stocks: int, avg_data_points_per_stock: int = 2000
    ) -> Dict[str, Any]:
        """Get recommendations for optimal processing configuration."""

        # Estimate memory requirements
        estimated_memory_per_stock = (
            avg_data_points_per_stock * 8 * 10
        )  # Rough estimate in bytes
        total_estimated_memory_gb = (total_stocks * estimated_memory_per_stock) / (
            1024**3
        )

        # Recommend chunk size based on memory
        if total_estimated_memory_gb > 8:
            recommended_chunk_size = max(
                50, min(200, int(8 * 1024**3 / estimated_memory_per_stock))
            )
        else:
            recommended_chunk_size = min(500, total_stocks)

        # Recommend number of workers
        try:
            import psutil

            cpu_count = psutil.cpu_count()
            recommended_workers = min(cpu_count, max(2, cpu_count // 2))
        except ImportError:
            recommended_workers = 4

        # Processing time estimate
        estimated_time_per_stock = 0.1  # seconds
        sequential_time = total_stocks * estimated_time_per_stock
        parallel_time = sequential_time / recommended_workers

        return {
            "total_stocks": total_stocks,
            "estimated_memory_gb": total_estimated_memory_gb,
            "recommended_chunk_size": recommended_chunk_size,
            "recommended_workers": recommended_workers,
            "estimated_sequential_time_minutes": sequential_time / 60,
            "estimated_parallel_time_minutes": parallel_time / 60,
            "speedup_factor": sequential_time / parallel_time,
            "use_distributed": total_estimated_memory_gb > 4 or total_stocks > 1000,
            "memory_optimization_needed": total_estimated_memory_gb > 8,
        }


# Utility functions for Dask integration


def create_dask_dataframe_from_stock_data(
    stock_data_dict: Dict[str, pd.DataFrame]
) -> dd.DataFrame:
    """Create a Dask DataFrame from stock data dictionary."""
    # Combine all stock data into a single DataFrame
    all_data = []
    for symbol, data in stock_data_dict.items():
        data_copy = data.copy()
        if "stock_code" not in data_copy.columns:
            data_copy["stock_code"] = symbol
        all_data.append(data_copy)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Convert to Dask DataFrame
    dask_df = dd.from_pandas(
        combined_df, npartitions=max(1, len(stock_data_dict) // 100)
    )

    return dask_df


def optimize_dask_config_for_spring_festival_analysis():
    """Optimize Dask configuration for Spring Festival analysis workloads."""
    dask.config.set(
        {
            "dataframe.query-planning": False,  # Use legacy query planning for stability
            "array.chunk-size": "128MB",
            "distributed.worker.memory.target": 0.6,
            "distributed.worker.memory.spill": 0.7,
            "distributed.worker.memory.pause": 0.8,
            "distributed.worker.memory.terminate": 0.95,
            "distributed.comm.timeouts.connect": "60s",
            "distributed.comm.timeouts.tcp": "60s",
        }
    )

    logger.info("Dask configuration optimized for Spring Festival analysis")
