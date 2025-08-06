"""Simple demo to test the Advanced Visualization Engine basic functionality."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import os

from stock_analysis_system.visualization.webgl_chart_engine import (
    WebGLChartEngine,
    WebGLChartConfig
)
from stock_analysis_system.visualization.chart_interaction_system import (
    ChartInteractionSystem,
    ChartAnnotation,
    AnnotationType
)
from stock_analysis_system.visualization.institutional_network_viz import (
    InstitutionalNetworkVisualizer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_webgl_engine():
    """Test WebGL chart engine basic functionality."""
    print("Testing WebGL Chart Engine...")
    
    # Create engine
    config = WebGLChartConfig(max_points_per_trace=1000, enable_webgl=True)
    engine = WebGLChartEngine(config)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    data = {
        'Price': pd.Series(np.random.randn(500).cumsum() + 100, index=dates),
        'Volume': pd.Series(np.random.lognormal(10, 1, 500), index=dates)
    }
    
    # Create chart
    fig = engine.create_high_performance_line_chart(
        data,
        title="Test WebGL Chart",
        x_title="Date",
        y_title="Value"
    )
    
    # Save chart
    os.makedirs("output", exist_ok=True)
    fig.write_html("output/test_webgl_chart.html")
    print("✓ WebGL chart saved to output/test_webgl_chart.html")
    
    return True


def test_interaction_system():
    """Test chart interaction system basic functionality."""
    print("Testing Chart Interaction System...")
    
    # Create interaction system
    interaction_system = ChartInteractionSystem()
    
    # Create base chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[1, 2, 3, 4, 5],
        y=[2, 4, 3, 5, 1],
        mode='lines+markers',
        name='Test Data'
    ))
    
    # Add interactions
    fig = interaction_system.enable_advanced_zoom(fig)
    fig = interaction_system.enable_crosshair_system(fig)
    
    # Add annotation
    annotation = ChartAnnotation(
        id="test_annotation",
        type=AnnotationType.TEXT,
        x=3,
        y=4,
        text="Test Point",
        color="#ff0000"
    )
    fig = interaction_system.add_annotation(fig, annotation)
    
    # Save chart
    fig.write_html("output/test_interaction_chart.html")
    print("✓ Interactive chart saved to output/test_interaction_chart.html")
    
    return True


def test_network_visualizer():
    """Test institutional network visualizer basic functionality."""
    print("Testing Institutional Network Visualizer...")
    
    # Create visualizer
    visualizer = InstitutionalNetworkVisualizer()
    
    # Create sample data
    institutional_data = pd.DataFrame({
        'institution_id': [1, 2, 3],
        'institution_name': ['Fund A', 'Fund B', 'Fund C'],
        'institution_type': ['mutual_fund', 'hedge_fund', 'pension_fund'],
        'total_assets': [1e9, 2e9, 1.5e9],
        'performance': [0.08, 0.12, 0.06]
    })
    
    holdings_data = pd.DataFrame({
        'institution_id': [1, 1, 2, 2, 3, 3],
        'stock_code': ['AAPL', 'GOOGL', 'AAPL', 'MSFT', 'GOOGL', 'TSLA'],
        'stock_name': ['Apple', 'Google', 'Apple', 'Microsoft', 'Google', 'Tesla'],
        'holding_percentage': [0.05, 0.03, 0.04, 0.06, 0.02, 0.08],
        'holding_value': [50e6, 30e6, 40e6, 60e6, 20e6, 80e6],
        'sector': ['technology', 'technology', 'technology', 'technology', 'technology', 'automotive'],
        'market_cap': [2e12, 1.5e12, 2e12, 2.2e12, 1.5e12, 800e9]
    })
    
    # Create network
    fig = visualizer.create_institutional_network(
        institutional_data,
        holdings_data,
        correlation_threshold=0.5,
        min_holding_size=0.01
    )
    
    # Save chart
    fig.write_html("output/test_network_chart.html")
    print("✓ Network chart saved to output/test_network_chart.html")
    
    return True


def main():
    """Run simple tests."""
    print("ADVANCED VISUALIZATION ENGINE - SIMPLE TEST")
    print("=" * 50)
    
    try:
        # Test each component
        success = True
        
        success &= test_webgl_engine()
        success &= test_interaction_system()
        success &= test_network_visualizer()
        
        if success:
            print("\n" + "=" * 50)
            print("ALL TESTS PASSED!")
            print("Check the output directory for generated charts.")
        else:
            print("\nSome tests failed.")
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()