"""Tests for Spring Festival visualization charts."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

import plotly.graph_objects as go

from stock_analysis_system.visualization.spring_festival_charts import (
    SpringFestivalChartEngine,
    SpringFestivalChartConfig,
    create_sample_chart
)
from stock_analysis_system.analysis.spring_festival_engine import (
    AlignedTimeSeries,
    AlignedDataPoint,
    SeasonalPattern
)


class TestSpringFestivalChartConfig:
    """Test SpringFestivalChartConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SpringFestivalChartConfig()
        
        assert config.width == 1200
        assert config.height == 800
        assert config.background_color == '#ffffff'
        assert config.enable_zoom is True
        assert config.enable_hover is True
        assert 'png' in config.export_formats


class TestSpringFestivalChartEngine:
    """Test SpringFestivalChartEngine."""
    
    @pytest.fixture
    def chart_engine(self):
        """Create chart engine for testing."""
        return SpringFestivalChartEngine()
    
    @pytest.fixture
    def sample_aligned_data(self):
        """Create sample aligned data for testing."""
        data_points = []
        years = [2020, 2021, 2022]
        
        for year in years:
            sf_date = date(year, 2, 1)
            
            for day_offset in range(-30, 31):
                relative_day = day_offset
                actual_date = sf_date + timedelta(days=day_offset)
                price = 100 + day_offset * 0.1 + np.random.normal(0, 1)
                normalized_price = day_offset * 0.5 + np.random.normal(0, 2)
                
                point = AlignedDataPoint(
                    original_date=actual_date,
                    relative_day=relative_day,
                    spring_festival_date=sf_date,
                    year=year,
                    price=price,
                    normalized_price=normalized_price
                )
                data_points.append(point)
        
        return AlignedTimeSeries(
            symbol="TEST001",
            data_points=data_points,
            window_days=30,
            years=years,
            baseline_price=100.0
        )
    
    @pytest.fixture
    def sample_pattern(self):
        """Create sample seasonal pattern."""
        return SeasonalPattern(
            symbol="TEST001",
            pattern_strength=0.75,
            average_return_before=2.5,
            average_return_after=3.2,
            volatility_before=1.8,
            volatility_after=2.1,
            consistency_score=0.68,
            confidence_level=0.82,
            years_analyzed=[2020, 2021, 2022],
            peak_day=15,
            trough_day=-10
        )
    
    def test_init(self):
        """Test chart engine initialization."""
        engine = SpringFestivalChartEngine()
        
        assert engine.config is not None
        assert engine.sf_engine is not None
        assert isinstance(engine.config, SpringFestivalChartConfig)
    
    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = SpringFestivalChartConfig()
        config.width = 800
        config.height = 600
        
        engine = SpringFestivalChartEngine(config)
        
        assert engine.config.width == 800
        assert engine.config.height == 600
    
    def test_create_overlay_chart(self, chart_engine, sample_aligned_data):
        """Test creating overlay chart."""
        fig = chart_engine.create_overlay_chart(sample_aligned_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # Three years of data
        
        # Check that Spring Festival line is added
        shapes = fig.layout.shapes
        assert len(shapes) >= 1  # Should have vertical line for Spring Festival
        
        # Check layout
        assert fig.layout.title.text is not None
        assert "相对春节天数" in fig.layout.xaxis.title.text
        assert "标准化收益率" in fig.layout.yaxis.title.text
    
    def test_create_overlay_chart_with_selected_years(self, chart_engine, sample_aligned_data):
        """Test creating overlay chart with selected years."""
        selected_years = [2020, 2022]
        fig = chart_engine.create_overlay_chart(
            sample_aligned_data, 
            selected_years=selected_years
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Only two years selected
        
        # Check that only selected years are in legend
        trace_names = [trace.name for trace in fig.data]
        assert "2020年" in trace_names
        assert "2022年" in trace_names
        assert "2021年" not in trace_names
    
    def test_create_overlay_chart_empty_data(self, chart_engine):
        """Test creating chart with empty data."""
        empty_data = AlignedTimeSeries(
            symbol="EMPTY",
            data_points=[],
            window_days=30,
            years=[],
            baseline_price=100.0
        )
        
        with pytest.raises(ValueError, match="No aligned data points available"):
            chart_engine.create_overlay_chart(empty_data)
    
    def test_create_pattern_summary_chart(self, chart_engine, sample_pattern):
        """Test creating pattern summary chart."""
        patterns = {
            "TEST001": sample_pattern,
            "TEST002": SeasonalPattern(
                symbol="TEST002",
                pattern_strength=0.65,
                average_return_before=1.8,
                average_return_after=2.5,
                volatility_before=1.5,
                volatility_after=1.9,
                consistency_score=0.72,
                confidence_level=0.78,
                years_analyzed=[2020, 2021, 2022],
                peak_day=12,
                trough_day=-8
            )
        }
        
        fig = chart_engine.create_pattern_summary_chart(patterns)
        
        assert isinstance(fig, go.Figure)
        # Should have 4 subplots
        assert len(fig.data) == 4
        
        # Check subplot titles
        annotations = fig.layout.annotations
        subplot_titles = [ann.text for ann in annotations if ann.text]
        assert "模式强度对比" in subplot_titles
        assert "春节前平均收益" in subplot_titles
    
    def test_create_pattern_summary_chart_empty(self, chart_engine):
        """Test creating pattern summary chart with empty data."""
        with pytest.raises(ValueError, match="No patterns provided"):
            chart_engine.create_pattern_summary_chart({})
    
    @patch('sklearn.cluster.KMeans')
    @patch('sklearn.preprocessing.StandardScaler')
    @patch('sklearn.decomposition.PCA')
    def test_create_cluster_visualization(
        self, 
        mock_pca, 
        mock_scaler, 
        mock_kmeans, 
        chart_engine, 
        sample_aligned_data
    ):
        """Test creating cluster visualization."""
        # Mock sklearn components
        mock_scaler_instance = Mock()
        mock_scaler_instance.fit_transform.return_value = np.random.randn(3, 7)
        mock_scaler.return_value = mock_scaler_instance
        
        mock_kmeans_instance = Mock()
        mock_kmeans_instance.fit_predict.return_value = np.array([0, 1, 0])
        mock_kmeans.return_value = mock_kmeans_instance
        
        mock_pca_instance = Mock()
        mock_pca_instance.fit_transform.return_value = np.random.randn(3, 3)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.5, 0.3, 0.2])
        mock_pca.return_value = mock_pca_instance
        
        # Mock pattern identification
        with patch.object(chart_engine.sf_engine, 'identify_seasonal_patterns') as mock_identify:
            mock_pattern = Mock()
            mock_pattern.pattern_strength = 0.7
            mock_pattern.average_return_before = 2.0
            mock_pattern.average_return_after = 3.0
            mock_pattern.volatility_before = 1.5
            mock_pattern.volatility_after = 2.0
            mock_pattern.consistency_score = 0.8
            mock_pattern.confidence_level = 0.9
            mock_identify.return_value = mock_pattern
            
            aligned_data_dict = {
                "TEST001": sample_aligned_data,
                "TEST002": sample_aligned_data,
                "TEST003": sample_aligned_data
            }
            
            fig = chart_engine.create_cluster_visualization(aligned_data_dict)
            
            assert isinstance(fig, go.Figure)
            # Should have traces for clusters
            assert len(fig.data) >= 1
    
    def test_create_cluster_visualization_insufficient_data(self, chart_engine):
        """Test cluster visualization with insufficient data."""
        with pytest.raises(ValueError, match="No aligned data provided"):
            chart_engine.create_cluster_visualization({})
    
    def test_export_chart_html(self, chart_engine, sample_aligned_data):
        """Test exporting chart as HTML."""
        fig = chart_engine.create_overlay_chart(sample_aligned_data)
        
        html_content = chart_engine.export_chart(fig, None, "html")
        
        assert isinstance(html_content, str)
        assert "<html>" in html_content
        assert "plotly" in html_content.lower()
    
    @patch('plotly.io.to_image')
    def test_export_chart_png(self, mock_to_image, chart_engine, sample_aligned_data):
        """Test exporting chart as PNG."""
        mock_to_image.return_value = b"fake_png_data"
        
        fig = chart_engine.create_overlay_chart(sample_aligned_data)
        
        png_data = chart_engine.export_chart(fig, None, "png")
        
        assert png_data == b"fake_png_data"
        mock_to_image.assert_called_once()
    
    def test_export_chart_unsupported_format(self, chart_engine, sample_aligned_data):
        """Test exporting chart with unsupported format."""
        fig = chart_engine.create_overlay_chart(sample_aligned_data)
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            chart_engine.export_chart(fig, None, "unsupported")
    
    @patch.object(SpringFestivalChartEngine, '_add_pattern_annotations')
    def test_create_overlay_chart_with_pattern_info(
        self, 
        mock_add_annotations, 
        chart_engine, 
        sample_aligned_data
    ):
        """Test creating overlay chart with pattern information."""
        with patch.object(chart_engine.sf_engine, 'identify_seasonal_patterns') as mock_identify:
            mock_identify.return_value = Mock()
            
            fig = chart_engine.create_overlay_chart(
                sample_aligned_data, 
                show_pattern_info=True
            )
            
            assert isinstance(fig, go.Figure)
            mock_identify.assert_called_once()
            mock_add_annotations.assert_called_once()
    
    def test_add_pattern_annotations(self, chart_engine, sample_pattern):
        """Test adding pattern annotations to chart."""
        fig = go.Figure()
        
        chart_engine._add_pattern_annotations(fig, sample_pattern)
        
        # Check that annotations were added
        annotations = fig.layout.annotations
        assert len(annotations) >= 1  # Should have at least pattern summary
    
    def test_create_interactive_dashboard(self, chart_engine, sample_aligned_data):
        """Test creating interactive dashboard."""
        aligned_data_dict = {
            "TEST001": sample_aligned_data,
            "TEST002": sample_aligned_data
        }
        
        with patch.object(chart_engine.sf_engine, 'identify_seasonal_patterns') as mock_identify:
            mock_pattern = Mock()
            mock_pattern.pattern_strength = 0.7
            mock_pattern.average_return_before = 2.0
            mock_pattern.average_return_after = 3.0
            mock_pattern.volatility_before = 1.5
            mock_pattern.volatility_after = 2.0
            mock_identify.return_value = mock_pattern
            
            fig = chart_engine.create_interactive_dashboard(aligned_data_dict)
            
            assert isinstance(fig, go.Figure)
            # Should have multiple traces for different subplots
            assert len(fig.data) >= 4


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_sample_chart(self):
        """Test creating sample chart."""
        fig = create_sample_chart("TEST001", [2020, 2021])
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Two years
        
        # Check that it has proper title
        assert "TEST001" in fig.layout.title.text
        assert "春节对齐分析示例" in fig.layout.title.text
    
    def test_create_sample_chart_default_years(self):
        """Test creating sample chart with default years."""
        fig = create_sample_chart("TEST002")
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4  # Default years [2020, 2021, 2022, 2023]


class TestIntegration:
    """Integration tests."""
    
    def test_full_chart_creation_workflow(self):
        """Test complete chart creation workflow."""
        # Create sample data
        data_points = []
        years = [2021, 2022]
        
        for year in years:
            sf_date = date(year, 2, 1)
            
            for day_offset in range(-20, 21):
                point = AlignedDataPoint(
                    original_date=sf_date + timedelta(days=day_offset),
                    relative_day=day_offset,
                    spring_festival_date=sf_date,
                    year=year,
                    price=100 + day_offset * 0.2,
                    normalized_price=day_offset * 0.3
                )
                data_points.append(point)
        
        aligned_data = AlignedTimeSeries(
            symbol="INTEGRATION_TEST",
            data_points=data_points,
            window_days=20,
            years=years,
            baseline_price=100.0
        )
        
        # Create chart engine
        engine = SpringFestivalChartEngine()
        
        # Create overlay chart
        fig = engine.create_overlay_chart(aligned_data, title="集成测试图表")
        
        # Verify chart properties
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert "集成测试图表" in fig.layout.title.text
        
        # Test export
        html_content = engine.export_chart(fig, None, "html")
        assert isinstance(html_content, str)
        assert len(html_content) > 1000  # Should be substantial HTML content


if __name__ == '__main__':
    pytest.main([__file__])