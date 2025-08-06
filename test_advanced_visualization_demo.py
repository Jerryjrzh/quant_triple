"""Comprehensive demo of the Advanced Visualization Engine capabilities."""

import asyncio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

from stock_analysis_system.visualization.webgl_chart_engine import (
    WebGLChartEngine,
    WebGLChartConfig,
    create_webgl_time_series_chart,
    create_webgl_stock_chart
)
from stock_analysis_system.visualization.chart_interaction_system import (
    ChartInteractionSystem,
    ChartAnnotation,
    AnnotationType,
    create_interactive_chart
)
from stock_analysis_system.visualization.institutional_network_viz import (
    InstitutionalNetworkVisualizer,
    create_simple_institutional_network,
    analyze_network_metrics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_stock_data(symbol: str, days: int = 252) -> pd.DataFrame:
    """Generate realistic sample stock data."""
    
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate price data with realistic patterns
    np.random.seed(42)  # For reproducible results
    
    # Start with base price
    base_price = 100.0
    
    # Generate returns with some autocorrelation
    returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]  # Add some momentum
    
    # Calculate prices
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))
        
        # High and low based on close with some randomness
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        
        volume = int(np.random.lognormal(15, 0.5))  # Log-normal volume
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'symbol': symbol
        })
    
    return pd.DataFrame(data).set_index('date')


def generate_institutional_data() -> tuple:
    """Generate sample institutional and holdings data."""
    
    # Institutional data
    institutions = [
        {'id': 1, 'name': 'Vanguard Total Stock Market', 'type': 'mutual_fund', 'assets': 1.2e12, 'performance': 0.08},
        {'id': 2, 'name': 'BlackRock iShares Core S&P 500', 'type': 'mutual_fund', 'assets': 800e9, 'performance': 0.09},
        {'id': 3, 'name': 'Berkshire Hathaway', 'type': 'hedge_fund', 'assets': 600e9, 'performance': 0.12},
        {'id': 4, 'name': 'California Public Employees', 'type': 'pension_fund', 'assets': 450e9, 'performance': 0.07},
        {'id': 5, 'name': 'Government Pension Fund Norway', 'type': 'sovereign_fund', 'assets': 1.4e12, 'performance': 0.06},
        {'id': 6, 'name': 'Bridgewater Associates', 'type': 'hedge_fund', 'assets': 150e9, 'performance': 0.15},
        {'id': 7, 'name': 'State Street Global Advisors', 'type': 'mutual_fund', 'assets': 3.5e12, 'performance': 0.08},
        {'id': 8, 'name': 'Fidelity Investments', 'type': 'mutual_fund', 'assets': 4.2e12, 'performance': 0.09}
    ]
    
    institutional_df = pd.DataFrame([
        {
            'institution_id': inst['id'],
            'institution_name': inst['name'],
            'institution_type': inst['type'],
            'total_assets': inst['assets'],
            'performance': inst['performance']
        }
        for inst in institutions
    ])
    
    # Holdings data
    stocks = [
        {'code': 'AAPL', 'name': 'Apple Inc.', 'sector': 'technology', 'market_cap': 2.8e12},
        {'code': 'MSFT', 'name': 'Microsoft Corp.', 'sector': 'technology', 'market_cap': 2.4e12},
        {'code': 'GOOGL', 'name': 'Alphabet Inc.', 'sector': 'technology', 'market_cap': 1.7e12},
        {'code': 'AMZN', 'name': 'Amazon.com Inc.', 'sector': 'consumer_discretionary', 'market_cap': 1.5e12},
        {'code': 'TSLA', 'name': 'Tesla Inc.', 'sector': 'automotive', 'market_cap': 800e9},
        {'code': 'BRK.A', 'name': 'Berkshire Hathaway', 'sector': 'financial', 'market_cap': 700e9},
        {'code': 'JNJ', 'name': 'Johnson & Johnson', 'sector': 'healthcare', 'market_cap': 450e9},
        {'code': 'JPM', 'name': 'JPMorgan Chase', 'sector': 'financial', 'market_cap': 400e9}
    ]
    
    # Generate holdings relationships
    holdings_data = []
    np.random.seed(42)
    
    for inst in institutions:
        # Each institution holds 3-6 stocks
        num_holdings = np.random.randint(3, 7)
        selected_stocks = np.random.choice(stocks, num_holdings, replace=False)
        
        for stock in selected_stocks:
            holding_pct = np.random.uniform(0.01, 0.08)  # 1% to 8% holding
            holding_value = inst['assets'] * holding_pct
            
            holdings_data.append({
                'institution_id': inst['id'],
                'stock_code': stock['code'],
                'stock_name': stock['name'],
                'holding_percentage': holding_pct,
                'holding_value': holding_value,
                'sector': stock['sector'],
                'market_cap': stock['market_cap']
            })
    
    holdings_df = pd.DataFrame(holdings_data)
    
    return institutional_df, holdings_df


async def demo_webgl_charts():
    """Demonstrate WebGL-accelerated chart capabilities."""
    
    print("\n" + "="*60)
    print("WEBGL CHART ENGINE DEMONSTRATION")
    print("="*60)
    
    # Initialize WebGL engine
    config = WebGLChartConfig(
        max_points_per_trace=5000,
        enable_webgl=True,
        enable_data_decimation=True,
        enable_animations=True
    )
    engine = WebGLChartEngine(config)
    
    print(f"WebGL Engine initialized with config:")
    print(f"- Max points per trace: {config.max_points_per_trace}")
    print(f"- WebGL enabled: {config.enable_webgl}")
    print(f"- Data decimation: {config.enable_data_decimation}")
    
    # 1. High-performance time series chart
    print("\n1. Creating high-performance time series chart...")
    
    # Generate large dataset
    dates = pd.date_range('2020-01-01', periods=10000, freq='h')
    large_data = {
        'Price': pd.Series(np.random.randn(10000).cumsum() + 100, index=dates),
        'Volume': pd.Series(np.random.lognormal(10, 1, 10000), index=dates),
        'RSI': pd.Series(np.random.uniform(20, 80, 10000), index=dates)
    }
    
    fig1 = engine.create_high_performance_line_chart(
        large_data,
        title="High-Performance Time Series (10,000 points)",
        x_title="Time",
        y_title="Value"
    )
    
    # Save chart
    fig1.write_html("output/webgl_time_series_chart.html")
    print("✓ Time series chart saved to output/webgl_time_series_chart.html")
    
    # 2. WebGL candlestick chart
    print("\n2. Creating WebGL candlestick chart...")
    
    stock_data = generate_sample_stock_data('AAPL', 500)
    volume_data = stock_data['volume']
    
    fig2 = engine.create_webgl_candlestick_chart(
        stock_data[['open', 'high', 'low', 'close']],
        volume_data=volume_data,
        title="AAPL Stock Chart with Volume"
    )
    
    fig2.write_html("output/webgl_candlestick_chart.html")
    print("✓ Candlestick chart saved to output/webgl_candlestick_chart.html")
    
    # 3. High-performance scatter plot
    print("\n3. Creating high-performance scatter plot...")
    
    n_points = 50000
    x_data = np.random.randn(n_points)
    y_data = x_data * 2 + np.random.randn(n_points) * 0.5
    color_data = x_data + y_data
    
    fig3 = engine.create_webgl_scatter_plot(
        x_data,
        y_data,
        color_data=color_data,
        title=f"High-Performance Scatter Plot ({n_points:,} points)"
    )
    
    fig3.write_html("output/webgl_scatter_plot.html")
    print("✓ Scatter plot saved to output/webgl_scatter_plot.html")
    
    # 4. Real-time chart
    print("\n4. Creating real-time optimized chart...")
    
    real_time_data = pd.Series(
        np.random.randn(1000).cumsum(),
        index=pd.date_range('2023-01-01', periods=1000, freq='min')
    )
    
    fig4 = engine.create_real_time_chart(
        real_time_data,
        title="Real-Time Data Visualization",
        max_points=1000
    )
    
    fig4.write_html("output/webgl_realtime_chart.html")
    print("✓ Real-time chart saved to output/webgl_realtime_chart.html")
    
    # Performance statistics
    stats = engine.get_performance_stats()
    print(f"\nPerformance Statistics:")
    for key, value in stats.items():
        print(f"- {key}: {value}")


async def demo_chart_interactions():
    """Demonstrate comprehensive chart interaction system."""
    
    print("\n" + "="*60)
    print("CHART INTERACTION SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Initialize interaction system
    interaction_system = ChartInteractionSystem()
    
    # Create base chart
    stock_data = generate_sample_stock_data('TSLA', 200)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['close'],
        mode='lines',
        name='TSLA Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    print("1. Adding advanced zoom capabilities...")
    fig = interaction_system.enable_advanced_zoom(fig)
    
    print("2. Adding advanced pan capabilities...")
    fig = interaction_system.enable_advanced_pan(fig)
    
    print("3. Adding selection tools...")
    fig = interaction_system.enable_selection_tools(fig)
    
    print("4. Adding crosshair system...")
    fig = interaction_system.enable_crosshair_system(fig)
    
    print("5. Adding annotation tools...")
    fig = interaction_system.add_annotation_tools(fig)
    
    print("6. Adding measurement tools...")
    fig = interaction_system.add_measurement_tools(fig)
    
    print("7. Adding custom tooltip system...")
    fig = interaction_system.add_custom_tooltip_system(
        fig,
        tooltip_template="<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>"
    )
    
    # Add some sample annotations
    print("\n8. Adding sample annotations...")
    
    # Text annotation
    text_annotation = ChartAnnotation(
        id="peak_annotation",
        type=AnnotationType.TEXT,
        x=stock_data.index[100],
        y=stock_data['close'].iloc[100],
        text="Local Peak",
        color="#ff0000",
        size=12
    )
    fig = interaction_system.add_annotation(fig, text_annotation)
    
    # Line annotation
    line_annotation = ChartAnnotation(
        id="trend_line",
        type=AnnotationType.LINE,
        x=[stock_data.index[50], stock_data.index[150]],
        y=[stock_data['close'].iloc[50], stock_data['close'].iloc[150]],
        color="#00ff00"
    )
    fig = interaction_system.add_annotation(fig, line_annotation)
    
    # Arrow annotation
    arrow_annotation = ChartAnnotation(
        id="breakout_arrow",
        type=AnnotationType.ARROW,
        x=[stock_data.index[75], stock_data.index[80]],
        y=[stock_data['close'].iloc[75], stock_data['close'].iloc[80]],
        text="Breakout",
        color="#ff7f0e"
    )
    fig = interaction_system.add_annotation(fig, arrow_annotation)
    
    # Save interactive chart
    fig.write_html("output/interactive_chart_demo.html")
    print("✓ Interactive chart saved to output/interactive_chart_demo.html")
    
    # Export annotations
    annotations_export = interaction_system.export_annotations()
    print(f"\nExported {len(annotations_export)} annotations:")
    for ann_id, ann_data in annotations_export.items():
        print(f"- {ann_id}: {ann_data['type']} - {ann_data['text']}")
    
    # Demonstrate zoom state management
    print("\n9. Testing zoom state management...")
    interaction_system.save_zoom_state((0, 100), (90, 110))
    interaction_system.save_zoom_state((25, 75), (95, 105))
    
    print(f"Saved {len(interaction_system.zoom_history)} zoom states")
    
    # Create utility chart
    print("\n10. Creating utility interactive chart...")
    sample_data = pd.DataFrame({
        'series1': np.random.randn(100).cumsum(),
        'series2': np.random.randn(100).cumsum()
    })
    
    utility_fig = create_interactive_chart(sample_data, chart_type="line")
    utility_fig.write_html("output/utility_interactive_chart.html")
    print("✓ Utility interactive chart saved to output/utility_interactive_chart.html")


async def demo_institutional_network():
    """Demonstrate institutional network visualization."""
    
    print("\n" + "="*60)
    print("INSTITUTIONAL NETWORK VISUALIZATION DEMONSTRATION")
    print("="*60)
    
    # Initialize network visualizer
    visualizer = InstitutionalNetworkVisualizer()
    
    # Generate sample data
    print("1. Generating sample institutional data...")
    institutional_data, holdings_data = generate_institutional_data()
    
    print(f"- {len(institutional_data)} institutions")
    print(f"- {len(holdings_data)} holdings relationships")
    print(f"- {holdings_data['stock_code'].nunique()} unique stocks")
    
    # Create main institutional network
    print("\n2. Creating institutional network visualization...")
    
    fig1 = visualizer.create_institutional_network(
        institutional_data,
        holdings_data,
        correlation_threshold=0.6,
        min_holding_size=0.02
    )
    
    fig1.write_html("output/institutional_network.html")
    print("✓ Institutional network saved to output/institutional_network.html")
    
    # Analyze network metrics
    print("\n3. Analyzing network metrics...")
    metrics = analyze_network_metrics(visualizer)
    
    print("Network Statistics:")
    for key, value in metrics.items():
        if key not in ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'eigenvector_centrality']:
            print(f"- {key}: {value}")
    
    # Create interactive dashboard
    print("\n4. Creating interactive network dashboard...")
    
    dashboard_fig = visualizer.create_interactive_network_dashboard(
        institutional_data,
        holdings_data
    )
    
    dashboard_fig.write_html("output/institutional_network_dashboard.html")
    print("✓ Network dashboard saved to output/institutional_network_dashboard.html")
    
    # Add dynamic filtering
    print("\n5. Adding dynamic filtering...")
    
    filter_options = {
        'institution_type': ['mutual_fund', 'hedge_fund', 'pension_fund', 'sovereign_fund'],
        'sector': ['technology', 'financial', 'healthcare', 'automotive']
    }
    
    filtered_fig = visualizer.create_institutional_network(
        institutional_data,
        holdings_data
    )
    filtered_fig = visualizer.add_dynamic_filtering(filtered_fig, filter_options)
    
    filtered_fig.write_html("output/institutional_network_filtered.html")
    print("✓ Filtered network saved to output/institutional_network_filtered.html")
    
    # Export network data
    print("\n6. Exporting network data...")
    
    network_data = visualizer.export_network_data(format="json")
    
    with open("output/institutional_network_data.json", "w") as f:
        import json
        json.dump(network_data, f, indent=2)
    
    print(f"✓ Network data exported:")
    print(f"- {len(network_data['nodes'])} nodes")
    print(f"- {len(network_data['edges'])} edges")
    
    # Create simple network using utility function
    print("\n7. Creating simple network using utility function...")
    
    simple_institutions = ['Vanguard', 'BlackRock', 'Fidelity', 'State Street']
    simple_holdings = {
        'Vanguard': ['AAPL', 'MSFT', 'GOOGL'],
        'BlackRock': ['AAPL', 'AMZN', 'TSLA'],
        'Fidelity': ['MSFT', 'GOOGL', 'JNJ'],
        'State Street': ['AAPL', 'JPM', 'BRK.A']
    }
    
    simple_fig = create_simple_institutional_network(
        simple_institutions,
        simple_holdings
    )
    
    simple_fig.write_html("output/simple_institutional_network.html")
    print("✓ Simple network saved to output/simple_institutional_network.html")


async def demo_integration_scenarios():
    """Demonstrate integration between different visualization components."""
    
    print("\n" + "="*60)
    print("INTEGRATION SCENARIOS DEMONSTRATION")
    print("="*60)
    
    # Scenario 1: WebGL chart with advanced interactions
    print("1. WebGL chart with advanced interactions...")
    
    # Create high-performance chart
    engine = WebGLChartEngine()
    large_data = {
        'Price': pd.Series(
            np.random.randn(5000).cumsum() + 100,
            index=pd.date_range('2020-01-01', periods=5000, freq='h')
        )
    }
    
    webgl_fig = engine.create_high_performance_line_chart(
        large_data,
        title="WebGL Chart with Interactions"
    )
    
    # Add interactions
    interaction_system = ChartInteractionSystem()
    webgl_fig = interaction_system.enable_advanced_zoom(webgl_fig)
    webgl_fig = interaction_system.enable_crosshair_system(webgl_fig)
    webgl_fig = interaction_system.add_annotation_tools(webgl_fig)
    
    webgl_fig.write_html("output/webgl_with_interactions.html")
    print("✓ WebGL + Interactions saved to output/webgl_with_interactions.html")
    
    # Scenario 2: Multi-chart synchronized dashboard
    print("\n2. Multi-chart synchronized dashboard...")
    
    # Create multiple related charts
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL']
    charts = []
    
    for symbol in stock_symbols:
        stock_data = generate_sample_stock_data(symbol, 200)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['close'],
            mode='lines',
            name=f'{symbol} Price'
        ))
        
        fig.update_layout(
            title=f'{symbol} Stock Price',
            xaxis_title='Date',
            yaxis_title='Price ($)'
        )
        
        charts.append(fig)
    
    # Enable synchronization (simplified)
    interaction_system = ChartInteractionSystem()
    synchronized_charts = interaction_system.enable_chart_synchronization(
        charts,
        sync_zoom=True,
        sync_pan=True
    )
    
    # Save synchronized charts
    for i, fig in enumerate(synchronized_charts):
        fig.write_html(f"output/synchronized_chart_{i+1}.html")
    
    print(f"✓ {len(synchronized_charts)} synchronized charts saved")
    
    # Scenario 3: Network with performance overlay
    print("\n3. Network with performance overlay...")
    
    # Create network with performance-based coloring
    visualizer = InstitutionalNetworkVisualizer()
    institutional_data, holdings_data = generate_institutional_data()
    
    # Create network
    network_fig = visualizer.create_institutional_network(
        institutional_data,
        holdings_data
    )
    
    # Add performance heatmap overlay (conceptual)
    network_fig.update_layout(
        title="Institutional Network with Performance Overlay",
        annotations=[
            dict(
                text="Node colors represent performance:<br>Green = High, Red = Low",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                font=dict(size=10)
            )
        ]
    )
    
    network_fig.write_html("output/network_with_performance.html")
    print("✓ Network with performance overlay saved")


async def main():
    """Run all visualization demos."""
    
    print("ADVANCED VISUALIZATION ENGINE COMPREHENSIVE DEMO")
    print("=" * 80)
    print("This demo showcases the capabilities of the Advanced Visualization Engine")
    print("including WebGL acceleration, chart interactions, and network visualization.")
    print("=" * 80)
    
    try:
        # Create output directory
        import os
        os.makedirs("output", exist_ok=True)
        
        # Run all demos
        await demo_webgl_charts()
        await demo_chart_interactions()
        await demo_institutional_network()
        await demo_integration_scenarios()
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("All visualization files have been saved to the 'output' directory.")
        print("Open the HTML files in your browser to explore the interactive features.")
        print("\nKey files to check out:")
        print("- output/webgl_time_series_chart.html - High-performance time series")
        print("- output/interactive_chart_demo.html - Full interaction capabilities")
        print("- output/institutional_network.html - Network visualization")
        print("- output/institutional_network_dashboard.html - Interactive dashboard")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())