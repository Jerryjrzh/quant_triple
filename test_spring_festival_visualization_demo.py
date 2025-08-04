"""Demo script for Spring Festival visualization capabilities."""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import sys
import os
import webbrowser
import tempfile

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_analysis_system.visualization.spring_festival_charts import (
    SpringFestivalChartEngine,
    SpringFestivalChartConfig,
    create_sample_chart
)
from stock_analysis_system.analysis.spring_festival_engine import (
    SpringFestivalAlignmentEngine,
    AlignedTimeSeries,
    AlignedDataPoint
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_realistic_stock_data(
    symbol: str, 
    start_date: str = '2018-01-01', 
    end_date: str = '2023-12-31'
) -> pd.DataFrame:
    """Generate realistic stock data with Spring Festival patterns."""
    logger.info(f"Generating realistic stock data for {symbol}")
    
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Remove weekends (simplified)
    dates = [d for d in dates if d.weekday() < 5]
    
    # Base parameters
    base_price = 50 + hash(symbol) % 100  # Different base price for each symbol
    volatility = 0.02
    trend = 0.0001  # Small upward trend
    
    prices = []
    current_price = base_price
    
    for i, date in enumerate(dates):
        # Add trend
        current_price *= (1 + trend)
        
        # Add Spring Festival effect
        sf_effect = get_spring_festival_effect(date)
        
        # Add random walk
        random_change = np.random.normal(0, volatility)
        
        # Combine effects
        daily_return = trend + sf_effect + random_change
        current_price *= (1 + daily_return)
        
        # Ensure price doesn't go negative
        current_price = max(current_price, 1.0)
        
        prices.append(current_price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'stock_code': symbol,
        'trade_date': dates,
        'open_price': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high_price': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low_price': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close_price': prices,
        'volume': [np.random.randint(1000000, 10000000) for _ in prices]
    })
    
    # Ensure OHLC consistency
    df['high_price'] = df[['open_price', 'close_price', 'high_price']].max(axis=1)
    df['low_price'] = df[['open_price', 'close_price', 'low_price']].min(axis=1)
    
    return df


def get_spring_festival_effect(current_date: datetime) -> float:
    """Calculate Spring Festival effect for a given date."""
    from datetime import date as date_class
    
    # Approximate Spring Festival dates
    sf_dates = {
        2018: date_class(2018, 2, 16),
        2019: date_class(2019, 2, 5),
        2020: date_class(2020, 1, 25),
        2021: date_class(2021, 2, 12),
        2022: date_class(2022, 2, 1),
        2023: date_class(2023, 1, 22)
    }
    
    year = current_date.year
    if year not in sf_dates:
        return 0.0
    
    sf_date = sf_dates[year]
    days_to_sf = (sf_date - current_date.date()).days
    
    # Create Spring Festival pattern
    if abs(days_to_sf) <= 60:
        # Pattern: decline before SF, recovery after
        if days_to_sf > 0:
            # Before Spring Festival - gradual decline
            effect = -0.001 * np.exp(-days_to_sf/30)
        else:
            # After Spring Festival - recovery
            effect = 0.002 * np.exp(days_to_sf/20)
        
        # Add some randomness
        effect += np.random.normal(0, 0.0005)
        return effect
    
    return 0.0


def demonstrate_single_stock_visualization():
    """Demonstrate single stock visualization."""
    logger.info("=== Single Stock Visualization Demo ===")
    
    # Generate sample data
    stock_data = generate_realistic_stock_data("000001", "2020-01-01", "2023-12-31")
    
    # Create Spring Festival alignment
    sf_engine = SpringFestivalAlignmentEngine()
    aligned_data = sf_engine.align_to_spring_festival(stock_data, [2020, 2021, 2022, 2023])
    
    # Create chart engine
    chart_engine = SpringFestivalChartEngine()
    
    # Create overlay chart
    fig = chart_engine.create_overlay_chart(
        aligned_data,
        title="000001 春节对齐分析演示",
        show_pattern_info=True
    )
    
    # Save and display
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        html_content = chart_engine.export_chart(fig, f.name, 'html')
        logger.info(f"Single stock chart saved to: {f.name}")
        
        # Open in browser
        webbrowser.open(f'file://{f.name}')
    
    return fig


def demonstrate_multi_stock_comparison():
    """Demonstrate multi-stock comparison visualization."""
    logger.info("=== Multi-Stock Comparison Demo ===")
    
    # Generate data for multiple stocks
    symbols = ["000001", "000002", "600000", "600036", "000858"]
    stock_data_dict = {}
    
    for symbol in symbols:
        stock_data_dict[symbol] = generate_realistic_stock_data(
            symbol, "2020-01-01", "2023-12-31"
        )
    
    # Create aligned data for all stocks
    sf_engine = SpringFestivalAlignmentEngine()
    aligned_data_dict = {}
    patterns = {}
    
    for symbol, data in stock_data_dict.items():
        try:
            aligned_data = sf_engine.align_to_spring_festival(data, [2020, 2021, 2022, 2023])
            pattern = sf_engine.identify_seasonal_patterns(aligned_data)
            
            aligned_data_dict[symbol] = aligned_data
            patterns[symbol] = pattern
            
            logger.info(f"{symbol}: 模式强度={pattern.pattern_strength:.3f}, "
                       f"春节前收益={pattern.average_return_before:.2f}%, "
                       f"春节后收益={pattern.average_return_after:.2f}%")
        except Exception as e:
            logger.warning(f"Failed to process {symbol}: {e}")
    
    # Create chart engine
    chart_engine = SpringFestivalChartEngine()
    
    # Create pattern summary chart
    fig_summary = chart_engine.create_pattern_summary_chart(
        patterns, 
        title="多股票春节模式对比分析"
    )
    
    # Save pattern summary
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        chart_engine.export_chart(fig_summary, f.name, 'html')
        logger.info(f"Pattern summary chart saved to: {f.name}")
        webbrowser.open(f'file://{f.name}')
    
    # Create cluster visualization
    try:
        fig_cluster = chart_engine.create_cluster_visualization(
            aligned_data_dict,
            n_clusters=3,
            title="春节模式聚类分析"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            chart_engine.export_chart(fig_cluster, f.name, 'html')
            logger.info(f"Cluster analysis chart saved to: {f.name}")
            webbrowser.open(f'file://{f.name}')
            
    except Exception as e:
        logger.warning(f"Cluster visualization failed: {e}")
    
    return fig_summary


def demonstrate_interactive_dashboard():
    """Demonstrate interactive dashboard."""
    logger.info("=== Interactive Dashboard Demo ===")
    
    # Generate data for dashboard
    symbols = ["000001", "600000", "000858"]
    stock_data_dict = {}
    aligned_data_dict = {}
    
    sf_engine = SpringFestivalAlignmentEngine()
    
    for symbol in symbols:
        stock_data = generate_realistic_stock_data(symbol, "2020-01-01", "2023-12-31")
        aligned_data = sf_engine.align_to_spring_festival(stock_data, [2020, 2021, 2022, 2023])
        
        stock_data_dict[symbol] = stock_data
        aligned_data_dict[symbol] = aligned_data
    
    # Create dashboard
    chart_engine = SpringFestivalChartEngine()
    fig_dashboard = chart_engine.create_interactive_dashboard(
        aligned_data_dict,
        title="春节分析综合仪表板"
    )
    
    # Save and display
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        chart_engine.export_chart(fig_dashboard, f.name, 'html')
        logger.info(f"Interactive dashboard saved to: {f.name}")
        webbrowser.open(f'file://{f.name}')
    
    return fig_dashboard


def demonstrate_export_capabilities():
    """Demonstrate chart export capabilities."""
    logger.info("=== Export Capabilities Demo ===")
    
    # Create a sample chart
    fig = create_sample_chart("EXPORT_TEST", [2021, 2022, 2023])
    
    chart_engine = SpringFestivalChartEngine()
    
    # Test different export formats
    export_formats = ['html', 'png', 'svg']
    
    for format in export_formats:
        try:
            with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as f:
                if format == 'html':
                    content = chart_engine.export_chart(fig, f.name, format)
                    logger.info(f"Exported HTML chart to: {f.name}")
                else:
                    content = chart_engine.export_chart(fig, f.name, format)
                    logger.info(f"Exported {format.upper()} chart to: {f.name} ({len(content)} bytes)")
                    
        except Exception as e:
            logger.error(f"Failed to export {format}: {e}")


def demonstrate_configuration_options():
    """Demonstrate configuration options."""
    logger.info("=== Configuration Options Demo ===")
    
    # Create custom configuration
    custom_config = SpringFestivalChartConfig()
    custom_config.width = 1600
    custom_config.height = 1000
    custom_config.background_color = '#f8f9fa'
    custom_config.sf_line_color = '#dc3545'
    custom_config.sf_line_width = 4
    
    # Create chart engine with custom config
    chart_engine = SpringFestivalChartEngine(custom_config)
    
    # Create sample chart
    fig = create_sample_chart("CONFIG_TEST", [2022, 2023])
    
    # Display configuration info
    logger.info(f"Chart dimensions: {custom_config.width}x{custom_config.height}")
    logger.info(f"Background color: {custom_config.background_color}")
    logger.info(f"Spring Festival line color: {custom_config.sf_line_color}")
    
    # Save with custom styling
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        chart_engine.export_chart(fig, f.name, 'html')
        logger.info(f"Custom styled chart saved to: {f.name}")
        webbrowser.open(f'file://{f.name}')


def demonstrate_performance_analysis():
    """Demonstrate performance with larger datasets."""
    logger.info("=== Performance Analysis Demo ===")
    
    import time
    
    # Test with different data sizes
    test_sizes = [1, 5, 10]
    
    for size in test_sizes:
        logger.info(f"Testing with {size} stocks...")
        
        start_time = time.time()
        
        # Generate data
        symbols = [f"TEST{i:03d}" for i in range(size)]
        aligned_data_dict = {}
        
        sf_engine = SpringFestivalAlignmentEngine()
        
        for symbol in symbols:
            stock_data = generate_realistic_stock_data(symbol, "2021-01-01", "2023-12-31")
            aligned_data = sf_engine.align_to_spring_festival(stock_data, [2021, 2022, 2023])
            aligned_data_dict[symbol] = aligned_data
        
        # Create charts
        chart_engine = SpringFestivalChartEngine()
        
        if size == 1:
            fig = chart_engine.create_overlay_chart(
                list(aligned_data_dict.values())[0],
                title=f"Performance Test - {size} Stock"
            )
        else:
            patterns = {}
            for symbol, aligned_data in aligned_data_dict.items():
                try:
                    patterns[symbol] = sf_engine.identify_seasonal_patterns(aligned_data)
                except:
                    continue
            
            if patterns:
                fig = chart_engine.create_pattern_summary_chart(
                    patterns,
                    title=f"Performance Test - {size} Stocks"
                )
        
        processing_time = time.time() - start_time
        logger.info(f"  Processing time: {processing_time:.2f} seconds")
        logger.info(f"  Average time per stock: {processing_time/size:.2f} seconds")


def main():
    """Main demo function."""
    logger.info("Starting Spring Festival Visualization Demo")
    
    try:
        # Run all demonstrations
        logger.info("Running visualization demonstrations...")
        
        # 1. Single stock visualization
        demonstrate_single_stock_visualization()
        
        # Wait a bit between demos
        import time
        time.sleep(2)
        
        # 2. Multi-stock comparison
        demonstrate_multi_stock_comparison()
        time.sleep(2)
        
        # 3. Interactive dashboard
        demonstrate_interactive_dashboard()
        time.sleep(2)
        
        # 4. Export capabilities
        demonstrate_export_capabilities()
        time.sleep(1)
        
        # 5. Configuration options
        demonstrate_configuration_options()
        time.sleep(1)
        
        # 6. Performance analysis
        demonstrate_performance_analysis()
        
        logger.info("=== Demo Summary ===")
        logger.info("✅ Single stock Spring Festival overlay charts")
        logger.info("✅ Multi-stock pattern comparison")
        logger.info("✅ Interactive clustering visualization")
        logger.info("✅ Comprehensive dashboard")
        logger.info("✅ Multiple export formats (HTML, PNG, SVG)")
        logger.info("✅ Customizable chart configuration")
        logger.info("✅ Performance optimization")
        
        logger.info("Demo completed successfully!")
        logger.info("Check your browser for the generated charts.")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()