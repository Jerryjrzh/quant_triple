"""Demo script for parallel Spring Festival analysis using Dask."""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import time
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_analysis_system.analysis.parallel_spring_festival_engine import (
    ParallelSpringFestivalEngine,
    ParallelProcessingConfig,
    optimize_dask_config_for_spring_festival_analysis
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_stock_data(symbols: list, start_date: str = '2018-01-01', end_date: str = '2023-12-31') -> dict:
    """Generate synthetic stock data for testing."""
    logger.info(f"Generating synthetic data for {len(symbols)} stocks from {start_date} to {end_date}")
    
    dates = pd.date_range(start_date, end_date, freq='D')
    stock_data = {}
    
    for i, symbol in enumerate(symbols):
        # Create realistic price movements with some seasonal patterns
        np.random.seed(42 + i)  # Reproducible data
        
        # Base price with trend
        base_price = 50 + i * 10
        trend = np.linspace(0, 20, len(dates))
        
        # Add seasonal component (stronger around Spring Festival)
        seasonal = np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) * 5
        
        # Add Spring Festival specific patterns
        sf_effect = np.zeros(len(dates))
        for year in range(2018, 2024):
            # Approximate Spring Festival dates
            sf_dates = {
                2018: date(2018, 2, 16),
                2019: date(2019, 2, 5),
                2020: date(2020, 1, 25),
                2021: date(2021, 2, 12),
                2022: date(2022, 2, 1),
                2023: date(2023, 1, 22)
            }
            
            if year in sf_dates:
                sf_date = sf_dates[year]
                for j, d in enumerate(dates):
                    days_to_sf = (sf_date - d.date()).days
                    if abs(days_to_sf) <= 60:
                        # Add pattern: decline before, recovery after
                        if days_to_sf > 0:
                            sf_effect[j] += -2 * np.exp(-days_to_sf/20)
                        else:
                            sf_effect[j] += 3 * np.exp(days_to_sf/30)
        
        # Random walk component
        random_walk = np.cumsum(np.random.randn(len(dates)) * 0.5)
        
        # Combine all components
        close_prices = base_price + trend + seasonal + sf_effect + random_walk
        close_prices = np.maximum(close_prices, 1.0)  # Ensure positive prices
        
        # Generate OHLV data
        data = pd.DataFrame({
            'stock_code': symbol,
            'trade_date': dates,
            'open_price': close_prices * (1 + np.random.randn(len(dates)) * 0.01),
            'high_price': close_prices * (1 + np.abs(np.random.randn(len(dates))) * 0.02),
            'low_price': close_prices * (1 - np.abs(np.random.randn(len(dates))) * 0.02),
            'close_price': close_prices,
            'volume': np.random.randint(100000, 5000000, len(dates))
        })
        
        # Ensure OHLC consistency
        data['high_price'] = np.maximum(data['high_price'], data[['open_price', 'close_price']].max(axis=1))
        data['low_price'] = np.minimum(data['low_price'], data[['open_price', 'close_price']].min(axis=1))
        
        stock_data[symbol] = data
    
    logger.info(f"Generated {len(stock_data)} stock datasets with {len(dates)} days each")
    return stock_data


def demonstrate_sequential_vs_parallel(stock_data: dict, years: list):
    """Demonstrate the difference between sequential and parallel processing."""
    logger.info("=== Sequential vs Parallel Processing Comparison ===")
    
    # Sequential processing
    logger.info("Starting sequential processing...")
    sequential_engine = ParallelSpringFestivalEngine()
    
    start_time = time.time()
    sequential_results = {}
    sequential_errors = {}
    
    for symbol, data in stock_data.items():
        try:
            aligned_data = sequential_engine.align_to_spring_festival(data, years)
            pattern = sequential_engine.identify_seasonal_patterns(aligned_data)
            sequential_results[symbol] = pattern
        except Exception as e:
            sequential_errors[symbol] = str(e)
    
    sequential_time = time.time() - start_time
    
    logger.info(f"Sequential processing completed in {sequential_time:.2f} seconds")
    logger.info(f"Sequential results: {len(sequential_results)} successful, {len(sequential_errors)} failed")
    
    # Parallel processing
    logger.info("Starting parallel processing...")
    parallel_config = ParallelProcessingConfig(
        n_workers=4,
        threads_per_worker=2,
        chunk_size=50,
        enable_distributed=False  # Use local processing for demo
    )
    parallel_engine = ParallelSpringFestivalEngine(config=parallel_config)
    
    start_time = time.time()
    parallel_result = parallel_engine.analyze_multiple_stocks_parallel(stock_data, years)
    parallel_time = time.time() - start_time
    
    logger.info(f"Parallel processing completed in {parallel_time:.2f} seconds")
    logger.info(f"Parallel results: {len(parallel_result.successful_analyses)} successful, {len(parallel_result.failed_analyses)} failed")
    
    # Calculate speedup
    if parallel_time > 0:
        speedup = sequential_time / parallel_time
        logger.info(f"Speedup factor: {speedup:.2f}x")
    
    return {
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'sequential_results': len(sequential_results),
        'parallel_results': len(parallel_result.successful_analyses),
        'speedup': speedup if parallel_time > 0 else 0
    }


def demonstrate_memory_optimization(stock_data: dict):
    """Demonstrate memory optimization features."""
    logger.info("=== Memory Optimization Demonstration ===")
    
    engine = ParallelSpringFestivalEngine()
    
    # Calculate original memory usage
    original_memory = sum(df.memory_usage(deep=True).sum() for df in stock_data.values())
    logger.info(f"Original data memory usage: {original_memory / (1024**2):.2f} MB")
    
    # Optimize for memory
    optimized_data = engine.optimize_for_memory_usage(stock_data, max_memory_gb=2.0)
    
    # Calculate optimized memory usage
    optimized_memory = sum(df.memory_usage(deep=True).sum() for df in optimized_data.values())
    logger.info(f"Optimized data memory usage: {optimized_memory / (1024**2):.2f} MB")
    
    memory_reduction = (original_memory - optimized_memory) / original_memory * 100
    logger.info(f"Memory reduction: {memory_reduction:.1f}%")
    
    return {
        'original_memory_mb': original_memory / (1024**2),
        'optimized_memory_mb': optimized_memory / (1024**2),
        'reduction_percent': memory_reduction
    }


def demonstrate_processing_recommendations(stock_data: dict):
    """Demonstrate processing recommendations."""
    logger.info("=== Processing Recommendations ===")
    
    engine = ParallelSpringFestivalEngine()
    
    # Calculate average data points per stock
    avg_data_points = np.mean([len(df) for df in stock_data.values()])
    
    recommendations = engine.get_processing_recommendations(
        total_stocks=len(stock_data),
        avg_data_points_per_stock=int(avg_data_points)
    )
    
    logger.info("Processing recommendations:")
    for key, value in recommendations.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return recommendations


def demonstrate_batch_processing(stock_data: dict, years: list):
    """Demonstrate batch processing with different configurations."""
    logger.info("=== Batch Processing with Different Configurations ===")
    
    configurations = [
        ParallelProcessingConfig(n_workers=2, chunk_size=25),
        ParallelProcessingConfig(n_workers=4, chunk_size=50),
        ParallelProcessingConfig(n_workers=6, chunk_size=100),
    ]
    
    results = []
    
    for i, config in enumerate(configurations):
        logger.info(f"Testing configuration {i+1}: {config.n_workers} workers, chunk size {config.chunk_size}")
        
        engine = ParallelSpringFestivalEngine(config=config)
        
        start_time = time.time()
        result = engine.analyze_multiple_stocks_parallel(stock_data, years)
        processing_time = time.time() - start_time
        
        config_result = {
            'config': f"{config.n_workers}w_{config.chunk_size}c",
            'processing_time': processing_time,
            'success_rate': result.success_rate,
            'successful_count': len(result.successful_analyses),
            'failed_count': len(result.failed_analyses)
        }
        
        results.append(config_result)
        
        logger.info(f"  Time: {processing_time:.2f}s, Success rate: {result.success_rate:.1%}")
    
    # Find best configuration
    best_config = min(results, key=lambda x: x['processing_time'])
    logger.info(f"Best configuration: {best_config['config']} ({best_config['processing_time']:.2f}s)")
    
    return results


def demonstrate_pattern_analysis(stock_data: dict, years: list):
    """Demonstrate pattern analysis results."""
    logger.info("=== Pattern Analysis Results ===")
    
    engine = ParallelSpringFestivalEngine()
    
    # Analyze a subset of stocks
    sample_stocks = dict(list(stock_data.items())[:5])
    result = engine.analyze_multiple_stocks_parallel(sample_stocks, years)
    
    logger.info(f"Analyzed {len(result.successful_analyses)} stocks successfully")
    
    # Display pattern statistics
    patterns = list(result.successful_analyses.values())
    if patterns:
        avg_pattern_strength = np.mean([p.pattern_strength for p in patterns])
        avg_return_before = np.mean([p.average_return_before for p in patterns])
        avg_return_after = np.mean([p.average_return_after for p in patterns])
        avg_confidence = np.mean([p.confidence_level for p in patterns])
        
        logger.info(f"Average pattern strength: {avg_pattern_strength:.3f}")
        logger.info(f"Average return before SF: {avg_return_before:.2f}%")
        logger.info(f"Average return after SF: {avg_return_after:.2f}%")
        logger.info(f"Average confidence level: {avg_confidence:.3f}")
        
        # Show individual stock patterns
        logger.info("Individual stock patterns:")
        for symbol, pattern in list(result.successful_analyses.items())[:3]:
            logger.info(f"  {symbol}: strength={pattern.pattern_strength:.3f}, "
                       f"before={pattern.average_return_before:.2f}%, "
                       f"after={pattern.average_return_after:.2f}%")
    
    return result


def main():
    """Main demo function."""
    logger.info("Starting Parallel Spring Festival Engine Demo")
    
    # Optimize Dask configuration
    optimize_dask_config_for_spring_festival_analysis()
    
    # Generate test data
    symbols = [f"STOCK{i:03d}" for i in range(200)]  # 200 stocks for meaningful parallel processing
    stock_data = generate_synthetic_stock_data(symbols)
    
    years = [2020, 2021, 2022, 2023]
    
    try:
        # Run demonstrations
        demo_results = {}
        
        # 1. Sequential vs Parallel comparison
        demo_results['performance'] = demonstrate_sequential_vs_parallel(stock_data, years)
        
        # 2. Memory optimization
        demo_results['memory'] = demonstrate_memory_optimization(stock_data)
        
        # 3. Processing recommendations
        demo_results['recommendations'] = demonstrate_processing_recommendations(stock_data)
        
        # 4. Batch processing configurations
        demo_results['batch_configs'] = demonstrate_batch_processing(stock_data, years)
        
        # 5. Pattern analysis
        demo_results['patterns'] = demonstrate_pattern_analysis(stock_data, years)
        
        # Summary
        logger.info("=== Demo Summary ===")
        logger.info(f"Processed {len(stock_data)} stocks across {len(years)} years")
        logger.info(f"Parallel speedup: {demo_results['performance']['speedup']:.2f}x")
        logger.info(f"Memory reduction: {demo_results['memory']['reduction_percent']:.1f}%")
        logger.info(f"Best batch config processing time: {min(demo_results['batch_configs'], key=lambda x: x['processing_time'])['processing_time']:.2f}s")
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()