"""Tests for the Advanced Visualization Engine components."""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch

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
    InteractionMode,
    create_interactive_chart
)
from stock_analysis_system.visualization.institutional_network_viz import (
    InstitutionalNetworkVisualizer,
    NetworkNode,
    NetworkEdge,
    NodeType,
    EdgeType,
    create_simple_institutional_network
)


class TestWebGLChartEngine:
    """Test WebGL-accelerated chart rendering engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = WebGLChartConfig(
            max_points_per_trace=1000,
            enable_webgl=True,
            enable_data_decimation=True
        )
        self.engine = WebGLChartEngine(self.config)
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.sample_data = {
            'series1': pd.Series(np.random.randn(100).cumsum(), index=dates),
            'series2': pd.Series(np.random.randn(100).cumsum(), index=dates)
        }
        
        # Create OHLC data
        self.ohlc_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(110, 120, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(100, 110, 50)
        }, index=pd.date_range('2023-01-01', periods=50, freq='D'))
        
        # Ensure OHLC consistency
        for i in range(len(self.ohlc_data)):
            row = self.ohlc_data.iloc[i]
            high = max(row['open'], row['close']) + np.random.uniform(0, 5)
            low = min(row['open'], row['close']) - np.random.uniform(0, 5)
            self.ohlc_data.iloc[i, self.ohlc_data.columns.get_loc('high')] = high
            self.ohlc_data.iloc[i, self.ohlc_data.columns.get_loc('low')] = low
    
    def test_webgl_config_initialization(self):
        """Test WebGL configuration initialization."""
        config = WebGLChartConfig()
        
        assert config.max_points_per_trace == 10000
        assert config.enable_webgl is True
        assert config.enable_data_decimation is True
        assert len(config.color_palette) == 10
    
    def test_high_performance_line_chart_creation(self):
        """Test creation of high-performance line chart."""
        fig = self.engine.create_high_performance_line_chart(
            self.sample_data,
            title="Test Chart",
            x_title="Date",
            y_title="Value"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Two series
        assert fig.layout.title.text == "Test Chart"
        assert fig.layout.xaxis.title.text == "Date"
        assert fig.layout.yaxis.title.text == "Value"
    
    def test_webgl_candlestick_chart_creation(self):
        """Test creation of WebGL candlestick chart."""
        fig = self.engine.create_webgl_candlestick_chart(
            self.ohlc_data,
            title="Candlestick Chart"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # One candlestick trace
        assert fig.layout.title.text == "Candlestick Chart"
        
        # Check candlestick data
        candlestick_trace = fig.data[0]
        assert hasattr(candlestick_trace, 'open')
        assert hasattr(candlestick_trace, 'high')
        assert hasattr(candlestick_trace, 'low')
        assert hasattr(candlestick_trace, 'close')
    
    def test_webgl_candlestick_with_volume(self):
        """Test candlestick chart with volume data."""
        volume_data = pd.Series(
            np.random.uniform(1000000, 5000000, len(self.ohlc_data)),
            index=self.ohlc_data.index
        )
        
        fig = self.engine.create_webgl_candlestick_chart(
            self.ohlc_data,
            volume_data=volume_data,
            title="Candlestick with Volume"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Candlestick + volume
        assert fig.layout.title.text == "Candlestick with Volume"
    
    def test_webgl_heatmap_creation(self):
        """Test creation of WebGL heatmap."""
        heatmap_data = pd.DataFrame(
            np.random.randn(20, 15),
            index=[f'Row_{i}' for i in range(20)],
            columns=[f'Col_{i}' for i in range(15)]
        )
        
        fig = self.engine.create_webgl_heatmap(
            heatmap_data,
            title="Test Heatmap"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.layout.title.text == "Test Heatmap"
    
    def test_webgl_scatter_plot_creation(self):
        """Test creation of WebGL scatter plot."""
        x_data = np.random.randn(100)
        y_data = np.random.randn(100)
        color_data = np.random.uniform(0, 1, 100)
        
        fig = self.engine.create_webgl_scatter_plot(
            x_data,
            y_data,
            color_data=color_data,
            title="Scatter Plot"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.layout.title.text == "Scatter Plot"
    
    def test_real_time_chart_creation(self):
        """Test creation of real-time optimized chart."""
        initial_data = pd.Series(
            np.random.randn(100).cumsum(),
            index=pd.date_range('2023-01-01', periods=100, freq='h')
        )
        
        fig = self.engine.create_real_time_chart(
            initial_data,
            title="Real-time Chart",
            max_points=500
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.layout.title.text == "Real-time Chart"
    
    def test_data_decimation(self):
        """Test data decimation functionality."""
        large_data = np.random.randn(50000)
        x_data = np.arange(50000)
        
        x_processed, y_processed = self.engine._prepare_data_for_webgl(
            x_data, large_data, max_points=1000
        )
        
        assert len(x_processed) <= 1000
        assert len(y_processed) <= 1000
        assert len(x_processed) == len(y_processed)
    
    def test_lttb_decimation(self):
        """Test Largest Triangle Three Buckets decimation."""
        x_data = np.arange(10000)
        y_data = np.sin(x_data / 100) + np.random.randn(10000) * 0.1
        
        x_decimated, y_decimated = self.engine._lttb_decimation(
            x_data, y_data, max_points=1000
        )
        
        assert len(x_decimated) <= 1000
        assert len(y_decimated) <= 1000
        assert x_decimated[0] == x_data[0]  # First point preserved
        assert x_decimated[-1] == x_data[-1]  # Last point preserved
    
    def test_performance_stats(self):
        """Test performance statistics retrieval."""
        stats = self.engine.get_performance_stats()
        
        assert 'webgl_enabled' in stats
        assert 'max_points_per_trace' in stats
        assert 'decimation_enabled' in stats
        assert stats['webgl_enabled'] == self.config.enable_webgl
    
    def test_utility_functions(self):
        """Test utility functions for WebGL charts."""
        # Test time series chart creation
        fig1 = create_webgl_time_series_chart(
            pd.DataFrame(self.sample_data),
            title="Time Series Test"
        )
        assert isinstance(fig1, go.Figure)
        
        # Test stock chart creation
        fig2 = create_webgl_stock_chart(
            self.ohlc_data,
            title="Stock Chart Test"
        )
        assert isinstance(fig2, go.Figure)


class TestChartInteractionSystem:
    """Test comprehensive chart interaction system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.interaction_system = ChartInteractionSystem()
        
        # Create sample figure
        self.fig = go.Figure()
        self.fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4, 5],
            y=[2, 4, 3, 5, 1],
            mode='lines+markers',
            name='Sample Data'
        ))
    
    def test_interaction_system_initialization(self):
        """Test interaction system initialization."""
        assert len(self.interaction_system.annotations) == 0
        assert self.interaction_system.crosshair is None
        assert self.interaction_system.selection is None
        assert len(self.interaction_system.zoom_history) == 0
        assert self.interaction_system.current_mode == InteractionMode.ZOOM
    
    def test_enable_advanced_zoom(self):
        """Test enabling advanced zoom capabilities."""
        fig = self.interaction_system.enable_advanced_zoom(self.fig)
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.dragmode == 'zoom'
        assert len(fig.layout.updatemenus) > 0
    
    def test_enable_advanced_pan(self):
        """Test enabling advanced pan capabilities."""
        fig = self.interaction_system.enable_advanced_pan(self.fig)
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.dragmode == 'pan'
    
    def test_enable_selection_tools(self):
        """Test enabling selection tools."""
        fig = self.interaction_system.enable_selection_tools(self.fig)
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.dragmode == 'select'
    
    def test_enable_crosshair_system(self):
        """Test enabling crosshair system."""
        fig = self.interaction_system.enable_crosshair_system(self.fig)
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.hovermode == 'x unified'
    
    def test_add_annotation_tools(self):
        """Test adding annotation tools."""
        fig = self.interaction_system.add_annotation_tools(self.fig)
        
        assert isinstance(fig, go.Figure)
        assert hasattr(fig.layout, 'newshape')
    
    def test_add_measurement_tools(self):
        """Test adding measurement tools."""
        fig = self.interaction_system.add_measurement_tools(self.fig)
        
        assert isinstance(fig, go.Figure)
    
    def test_add_custom_tooltip_system(self):
        """Test adding custom tooltip system."""
        fig = self.interaction_system.add_custom_tooltip_system(
            self.fig,
            tooltip_template="<b>Custom:</b> %{y}<extra></extra>"
        )
        
        assert isinstance(fig, go.Figure)
        assert hasattr(fig.layout, 'hoverlabel')
    
    def test_add_annotation(self):
        """Test adding annotations to chart."""
        annotation = ChartAnnotation(
            id="test_annotation",
            type=AnnotationType.TEXT,
            x=2.5,
            y=3.0,
            text="Test Annotation",
            color="#ff0000"
        )
        
        fig = self.interaction_system.add_annotation(self.fig, annotation)
        
        assert isinstance(fig, go.Figure)
        assert "test_annotation" in self.interaction_system.annotations
        assert len(fig.layout.annotations) > 0
    
    def test_add_line_annotation(self):
        """Test adding line annotation."""
        annotation = ChartAnnotation(
            id="line_annotation",
            type=AnnotationType.LINE,
            x=[1, 4],
            y=[2, 5],
            color="#00ff00"
        )
        
        fig = self.interaction_system.add_annotation(self.fig, annotation)
        
        assert isinstance(fig, go.Figure)
        assert "line_annotation" in self.interaction_system.annotations
    
    def test_remove_annotation(self):
        """Test removing annotations."""
        annotation = ChartAnnotation(
            id="temp_annotation",
            type=AnnotationType.TEXT,
            x=1,
            y=1,
            text="Temporary"
        )
        
        self.interaction_system.add_annotation(self.fig, annotation)
        assert "temp_annotation" in self.interaction_system.annotations
        
        self.interaction_system.remove_annotation(self.fig, "temp_annotation")
        assert "temp_annotation" not in self.interaction_system.annotations
    
    def test_event_callbacks(self):
        """Test event callback system."""
        callback_called = False
        
        def test_callback(event_data):
            nonlocal callback_called
            callback_called = True
        
        self.interaction_system.add_event_callback('test_event', test_callback)
        self.interaction_system.trigger_event('test_event', {'data': 'test'})
        
        assert callback_called
    
    def test_zoom_state_management(self):
        """Test zoom state saving and restoration."""
        x_range = (0, 10)
        y_range = (0, 5)
        
        self.interaction_system.save_zoom_state(x_range, y_range)
        
        assert len(self.interaction_system.zoom_history) == 1
        assert self.interaction_system.zoom_history[0].x_range == x_range
        assert self.interaction_system.zoom_history[0].y_range == y_range
        
        # Test restoration
        fig = self.interaction_system.restore_zoom_state(self.fig)
        assert isinstance(fig, go.Figure)
    
    def test_export_import_annotations(self):
        """Test annotation export and import."""
        annotation = ChartAnnotation(
            id="export_test",
            type=AnnotationType.TEXT,
            x=1,
            y=1,
            text="Export Test"
        )
        
        self.interaction_system.add_annotation(self.fig, annotation)
        
        # Export annotations
        exported = self.interaction_system.export_annotations()
        assert "export_test" in exported
        assert exported["export_test"]["text"] == "Export Test"
        
        # Clear and import
        self.interaction_system.annotations.clear()
        fig = self.interaction_system.import_annotations(self.fig, exported)
        
        assert "export_test" in self.interaction_system.annotations
        assert isinstance(fig, go.Figure)
    
    def test_utility_functions(self):
        """Test utility functions for chart interactions."""
        data = pd.DataFrame({
            'value1': [1, 2, 3, 4, 5],
            'value2': [2, 4, 3, 5, 1]
        })
        
        fig = create_interactive_chart(data, chart_type="line")
        assert isinstance(fig, go.Figure)


class TestInstitutionalNetworkVisualizer:
    """Test institutional network visualization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = InstitutionalNetworkVisualizer()
        
        # Create sample institutional data
        self.institutional_data = pd.DataFrame({
            'institution_id': [1, 2, 3, 4],
            'institution_name': ['Fund A', 'Fund B', 'Fund C', 'Fund D'],
            'institution_type': ['mutual_fund', 'hedge_fund', 'pension_fund', 'insurance'],
            'total_assets': [1e9, 2e9, 1.5e9, 3e9],
            'performance': [0.08, 0.12, 0.06, 0.10]
        })
        
        # Create sample holdings data
        self.holdings_data = pd.DataFrame({
            'institution_id': [1, 1, 2, 2, 3, 3, 4, 4],
            'stock_code': ['AAPL', 'GOOGL', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'MSFT', 'TSLA'],
            'stock_name': ['Apple', 'Google', 'Apple', 'Microsoft', 'Google', 'Tesla', 'Microsoft', 'Tesla'],
            'holding_percentage': [0.05, 0.03, 0.04, 0.06, 0.02, 0.08, 0.05, 0.04],
            'holding_value': [50e6, 30e6, 40e6, 60e6, 20e6, 80e6, 50e6, 40e6],
            'sector': ['technology', 'technology', 'technology', 'technology', 'technology', 'automotive', 'technology', 'automotive'],
            'market_cap': [2e12, 1.5e12, 2e12, 2.2e12, 1.5e12, 800e9, 2.2e12, 800e9]
        })
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        assert len(self.visualizer.nodes) == 0
        assert len(self.visualizer.edges) == 0
        assert len(self.visualizer.graph.nodes) == 0
        assert len(self.visualizer.color_schemes) > 0
    
    def test_create_institutional_network(self):
        """Test creation of institutional network."""
        fig = self.visualizer.create_institutional_network(
            self.institutional_data,
            self.holdings_data,
            correlation_threshold=0.5,
            min_holding_size=0.01
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Should have traces for nodes and edges
        assert fig.layout.title.text == "Institutional Network Visualization"
    
    def test_network_graph_building(self):
        """Test network graph building."""
        self.visualizer._build_network_graph(
            self.institutional_data,
            self.holdings_data,
            correlation_threshold=0.5,
            min_holding_size=0.01
        )
        
        # Check that nodes were created
        assert len(self.visualizer.nodes) > 0
        assert len(self.visualizer.edges) > 0
        assert len(self.visualizer.graph.nodes) > 0
        
        # Check node types
        institution_nodes = [n for n in self.visualizer.nodes.values() if n.node_type == NodeType.INSTITUTION]
        stock_nodes = [n for n in self.visualizer.nodes.values() if n.node_type == NodeType.STOCK]
        
        assert len(institution_nodes) == len(self.institutional_data)
        assert len(stock_nodes) > 0
    
    def test_force_directed_layout(self):
        """Test force-directed layout calculation."""
        self.visualizer._build_network_graph(
            self.institutional_data,
            self.holdings_data,
            correlation_threshold=0.5,
            min_holding_size=0.01
        )
        
        layout = self.visualizer._calculate_force_directed_layout()
        
        assert isinstance(layout, dict)
        assert len(layout) == len(self.visualizer.graph.nodes)
        
        # Check that positions are assigned
        for node_id, (x, y) in layout.items():
            assert isinstance(x, (int, float))
            assert isinstance(y, (int, float))
    
    def test_node_size_calculation(self):
        """Test node size calculation."""
        institution = self.institutional_data.iloc[0]
        size = self.visualizer._calculate_institution_size(institution)
        
        assert isinstance(size, float)
        assert 10 <= size <= 50  # Within expected range
    
    def test_edge_width_calculation(self):
        """Test edge width calculation."""
        width = self.visualizer._calculate_edge_width(0.05)
        
        assert isinstance(width, float)
        assert 0.5 <= width <= 5.0  # Within expected range
    
    def test_color_assignment(self):
        """Test color assignment for nodes."""
        institution = self.institutional_data.iloc[0]
        color = self.visualizer._get_institution_color(institution)
        
        assert isinstance(color, str)
        assert color.startswith('#')  # Should be hex color
    
    def test_network_node_creation(self):
        """Test NetworkNode creation."""
        node = NetworkNode(
            id="test_node",
            label="Test Node",
            node_type=NodeType.INSTITUTION,
            size=20.0,
            color="#ff0000"
        )
        
        assert node.id == "test_node"
        assert node.label == "Test Node"
        assert node.node_type == NodeType.INSTITUTION
        assert node.size == 20.0
        assert node.color == "#ff0000"
    
    def test_network_edge_creation(self):
        """Test NetworkEdge creation."""
        edge = NetworkEdge(
            source="node1",
            target="node2",
            edge_type=EdgeType.HOLDING,
            weight=0.5,
            color="#999999"
        )
        
        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.edge_type == EdgeType.HOLDING
        assert edge.weight == 0.5
        assert edge.color == "#999999"
    
    def test_export_network_data(self):
        """Test network data export."""
        self.visualizer._build_network_graph(
            self.institutional_data,
            self.holdings_data,
            correlation_threshold=0.5,
            min_holding_size=0.01
        )
        
        # Test JSON export
        json_data = self.visualizer.export_network_data(format="json")
        
        assert isinstance(json_data, dict)
        assert "nodes" in json_data
        assert "edges" in json_data
        assert len(json_data["nodes"]) > 0
        assert len(json_data["edges"]) > 0
    
    def test_interactive_network_dashboard(self):
        """Test interactive network dashboard creation."""
        fig = self.visualizer.create_interactive_network_dashboard(
            self.institutional_data,
            self.holdings_data
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Institutional Network Analysis Dashboard"
    
    def test_utility_functions(self):
        """Test utility functions for network visualization."""
        institutions = ['Fund A', 'Fund B', 'Fund C']
        holdings = {
            'Fund A': ['AAPL', 'GOOGL'],
            'Fund B': ['AAPL', 'MSFT'],
            'Fund C': ['GOOGL', 'TSLA']
        }
        
        fig = create_simple_institutional_network(institutions, holdings)
        assert isinstance(fig, go.Figure)


class TestAdvancedVisualizationIntegration:
    """Test integration between visualization components."""
    
    def test_webgl_with_interactions(self):
        """Test WebGL charts with interaction system."""
        # Create WebGL chart
        engine = WebGLChartEngine()
        data = {
            'series1': pd.Series(np.random.randn(100).cumsum(), 
                               index=pd.date_range('2023-01-01', periods=100))
        }
        fig = engine.create_high_performance_line_chart(data)
        
        # Add interactions
        interaction_system = ChartInteractionSystem()
        fig = interaction_system.enable_advanced_zoom(fig)
        fig = interaction_system.enable_crosshair_system(fig)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_network_with_interactions(self):
        """Test network visualization with interactions."""
        visualizer = InstitutionalNetworkVisualizer()
        
        # Create sample data
        institutional_data = pd.DataFrame({
            'institution_id': [1, 2],
            'institution_name': ['Fund A', 'Fund B'],
            'institution_type': ['mutual_fund', 'hedge_fund'],
            'total_assets': [1e9, 2e9]
        })
        
        holdings_data = pd.DataFrame({
            'institution_id': [1, 2],
            'stock_code': ['AAPL', 'GOOGL'],
            'stock_name': ['Apple', 'Google'],
            'holding_percentage': [0.05, 0.03],
            'sector': ['technology', 'technology']
        })
        
        fig = visualizer.create_institutional_network(
            institutional_data,
            holdings_data
        )
        
        # Add filtering
        filter_options = {
            'institution_type': ['mutual_fund', 'hedge_fund'],
            'sector': ['technology']
        }
        fig = visualizer.add_dynamic_filtering(fig, filter_options)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.updatemenus) > 0


if __name__ == "__main__":
    pytest.main([__file__])