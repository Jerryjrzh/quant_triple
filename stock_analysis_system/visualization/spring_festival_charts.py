"""Spring Festival visualization charts using Plotly."""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import base64
from io import BytesIO

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.colors import qualitative

from stock_analysis_system.analysis.spring_festival_engine import (
    SpringFestivalAlignmentEngine,
    AlignedTimeSeries,
    SeasonalPattern,
    AlignedDataPoint
)
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SpringFestivalChartConfig:
    """Configuration for Spring Festival charts."""
    
    def __init__(self):
        # Chart dimensions
        self.width = 1200
        self.height = 800
        
        # Color scheme
        self.colors = qualitative.Set3
        self.background_color = '#ffffff'
        self.grid_color = '#e0e0e0'
        self.text_color = '#333333'
        
        # Spring Festival marker
        self.sf_line_color = '#ff0000'
        self.sf_line_width = 3
        self.sf_line_dash = 'dash'
        
        # Interactive features
        self.enable_zoom = True
        self.enable_pan = True
        self.enable_hover = True
        self.enable_crossfilter = True
        
        # Export settings
        self.export_formats = ['png', 'svg', 'pdf', 'html']
        self.export_scale = 2  # For high-resolution exports


class SpringFestivalChartEngine:
    """Engine for creating Spring Festival alignment charts."""
    
    def __init__(self, config: SpringFestivalChartConfig = None):
        self.config = config or SpringFestivalChartConfig()
        self.sf_engine = SpringFestivalAlignmentEngine()
        
        # Set Plotly default template
        pio.templates.default = "plotly_white"
        
    def create_overlay_chart(
        self, 
        aligned_data: AlignedTimeSeries,
        title: str = None,
        show_pattern_info: bool = True,
        selected_years: List[int] = None
    ) -> go.Figure:
        """Create Spring Festival overlay chart with multiple years."""
        
        if not aligned_data.data_points:
            raise ValueError("No aligned data points available for visualization")
        
        # Filter years if specified
        if selected_years:
            filtered_points = [
                point for point in aligned_data.data_points 
                if point.year in selected_years
            ]
            years_to_show = selected_years
        else:
            filtered_points = aligned_data.data_points
            years_to_show = aligned_data.years
        
        if not filtered_points:
            raise ValueError("No data points available for selected years")
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each year
        for i, year in enumerate(years_to_show):
            year_points = [p for p in filtered_points if p.year == year]
            if not year_points:
                continue
            
            # Sort by relative day
            year_points.sort(key=lambda x: x.relative_day)
            
            # Extract data
            relative_days = [p.relative_day for p in year_points]
            normalized_prices = [p.normalized_price for p in year_points]
            actual_prices = [p.price for p in year_points]
            dates = [p.original_date for p in year_points]
            
            # Create hover text
            hover_text = [
                f"日期: {date}<br>"
                f"相对春节: {rel_day}天<br>"
                f"实际价格: ¥{price:.2f}<br>"
                f"标准化收益: {norm_price:.2f}%"
                for date, rel_day, price, norm_price in 
                zip(dates, relative_days, actual_prices, normalized_prices)
            ]
            
            # Add trace
            color = self.config.colors[i % len(self.config.colors)]
            fig.add_trace(go.Scatter(
                x=relative_days,
                y=normalized_prices,
                mode='lines+markers',
                name=f'{year}年',
                line=dict(color=color, width=2),
                marker=dict(size=4, color=color),
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=True
            ))
        
        # Add Spring Festival vertical line
        fig.add_vline(
            x=0,
            line_dash=self.config.sf_line_dash,
            line_color=self.config.sf_line_color,
            line_width=self.config.sf_line_width,
            annotation_text="春节",
            annotation_position="top"
        )
        
        # Add pattern information if requested
        if show_pattern_info:
            try:
                pattern = self.sf_engine.identify_seasonal_patterns(aligned_data)
                self._add_pattern_annotations(fig, pattern)
            except Exception as e:
                logger.warning(f"Failed to add pattern annotations: {e}")
        
        # Update layout
        chart_title = title or f"{aligned_data.symbol} 春节对齐分析"
        fig.update_layout(
            title=dict(
                text=chart_title,
                x=0.5,
                font=dict(size=20, color=self.config.text_color)
            ),
            xaxis=dict(
                title="相对春节天数",
                gridcolor=self.config.grid_color,
                zeroline=True,
                zerolinecolor=self.config.sf_line_color,
                zerolinewidth=1
            ),
            yaxis=dict(
                title="标准化收益率 (%)",
                gridcolor=self.config.grid_color,
                zeroline=True,
                zerolinecolor=self.config.grid_color
            ),
            plot_bgcolor=self.config.background_color,
            paper_bgcolor=self.config.background_color,
            font=dict(color=self.config.text_color),
            width=self.config.width,
            height=self.config.height,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Configure interactivity
        if self.config.enable_zoom and self.config.enable_pan:
            fig.update_layout(dragmode='zoom')
        elif self.config.enable_pan:
            fig.update_layout(dragmode='pan')
        
        return fig
    
    def create_pattern_summary_chart(
        self, 
        patterns: Dict[str, SeasonalPattern],
        title: str = "春节模式对比分析"
    ) -> go.Figure:
        """Create pattern summary comparison chart."""
        
        if not patterns:
            raise ValueError("No patterns provided for visualization")
        
        # Extract data for comparison
        symbols = list(patterns.keys())
        pattern_strengths = [patterns[symbol].pattern_strength for symbol in symbols]
        returns_before = [patterns[symbol].average_return_before for symbol in symbols]
        returns_after = [patterns[symbol].average_return_after for symbol in symbols]
        confidence_levels = [patterns[symbol].confidence_level for symbol in symbols]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '模式强度对比', '春节前平均收益',
                '春节后平均收益', '置信度分布'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Pattern strength bar chart
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=pattern_strengths,
                name='模式强度',
                marker_color='lightblue',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Returns before Spring Festival
        colors_before = ['green' if r > 0 else 'red' for r in returns_before]
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=returns_before,
                name='春节前收益',
                marker_color=colors_before,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Returns after Spring Festival
        colors_after = ['green' if r > 0 else 'red' for r in returns_after]
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=returns_after,
                name='春节后收益',
                marker_color=colors_after,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Confidence level scatter
        fig.add_trace(
            go.Scatter(
                x=symbols,
                y=confidence_levels,
                mode='markers',
                name='置信度',
                marker=dict(
                    size=10,
                    color=confidence_levels,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="置信度")
                ),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18, color=self.config.text_color)
            ),
            plot_bgcolor=self.config.background_color,
            paper_bgcolor=self.config.background_color,
            font=dict(color=self.config.text_color),
            width=self.config.width,
            height=self.config.height,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="股票代码", row=1, col=1)
        fig.update_yaxes(title_text="强度", row=1, col=1)
        
        fig.update_xaxes(title_text="股票代码", row=1, col=2)
        fig.update_yaxes(title_text="收益率 (%)", row=1, col=2)
        
        fig.update_xaxes(title_text="股票代码", row=2, col=1)
        fig.update_yaxes(title_text="收益率 (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="股票代码", row=2, col=2)
        fig.update_yaxes(title_text="置信度", row=2, col=2)
        
        return fig
    
    def create_cluster_visualization(
        self, 
        aligned_data_dict: Dict[str, AlignedTimeSeries],
        n_clusters: int = 3,
        title: str = "春节模式聚类分析"
    ) -> go.Figure:
        """Create cluster visualization of Spring Festival patterns."""
        
        if not aligned_data_dict:
            raise ValueError("No aligned data provided for clustering")
        
        # Extract features for clustering
        features = []
        symbols = []
        
        for symbol, aligned_data in aligned_data_dict.items():
            try:
                pattern = self.sf_engine.identify_seasonal_patterns(aligned_data)
                
                # Create feature vector
                feature_vector = [
                    pattern.pattern_strength,
                    pattern.average_return_before,
                    pattern.average_return_after,
                    pattern.volatility_before,
                    pattern.volatility_after,
                    pattern.consistency_score,
                    pattern.confidence_level
                ]
                
                features.append(feature_vector)
                symbols.append(symbol)
                
            except Exception as e:
                logger.warning(f"Failed to extract features for {symbol}: {e}")
                continue
        
        if len(features) < n_clusters:
            raise ValueError(f"Not enough valid patterns ({len(features)}) for {n_clusters} clusters")
        
        # Perform clustering
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Create 3D scatter plot using first 3 principal components
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(features_scaled)
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot for each cluster
        cluster_colors = px.colors.qualitative.Set1[:n_clusters]
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_symbols = [symbols[i] for i in range(len(symbols)) if cluster_mask[i]]
            
            if not any(cluster_mask):
                continue
            
            fig.add_trace(go.Scatter3d(
                x=features_pca[cluster_mask, 0],
                y=features_pca[cluster_mask, 1],
                z=features_pca[cluster_mask, 2],
                mode='markers+text',
                text=cluster_symbols,
                textposition='top center',
                name=f'聚类 {cluster_id + 1}',
                marker=dict(
                    size=8,
                    color=cluster_colors[cluster_id],
                    opacity=0.8
                ),
                hovertemplate='<b>%{text}</b><br>' +
                            'PC1: %{x:.2f}<br>' +
                            'PC2: %{y:.2f}<br>' +
                            'PC3: %{z:.2f}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18, color=self.config.text_color)
            ),
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%})',
                bgcolor=self.config.background_color
            ),
            plot_bgcolor=self.config.background_color,
            paper_bgcolor=self.config.background_color,
            font=dict(color=self.config.text_color),
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def _add_pattern_annotations(self, fig: go.Figure, pattern: SeasonalPattern):
        """Add pattern information annotations to the chart."""
        
        # Add peak and trough annotations
        if hasattr(pattern, 'peak_day') and hasattr(pattern, 'trough_day'):
            # Peak annotation
            fig.add_annotation(
                x=pattern.peak_day,
                y=0,  # Will be adjusted based on actual data
                text=f"峰值<br>{pattern.peak_day}天",
                showarrow=True,
                arrowhead=2,
                arrowcolor="green",
                bgcolor="rgba(0,255,0,0.1)",
                bordercolor="green"
            )
            
            # Trough annotation
            fig.add_annotation(
                x=pattern.trough_day,
                y=0,  # Will be adjusted based on actual data
                text=f"谷值<br>{pattern.trough_day}天",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                bgcolor="rgba(255,0,0,0.1)",
                bordercolor="red"
            )
        
        # Add pattern summary text box
        pattern_text = (
            f"模式强度: {pattern.pattern_strength:.3f}<br>"
            f"春节前收益: {pattern.average_return_before:.2f}%<br>"
            f"春节后收益: {pattern.average_return_after:.2f}%<br>"
            f"置信度: {pattern.confidence_level:.3f}"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=pattern_text,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=10)
        )
    
    def export_chart(
        self, 
        fig: go.Figure, 
        filename: str, 
        format: str = 'png',
        **kwargs
    ) -> Union[str, bytes]:
        """Export chart to various formats."""
        
        if format not in self.config.export_formats:
            raise ValueError(f"Unsupported export format: {format}")
        
        try:
            if format == 'html':
                # Export as HTML
                html_str = pio.to_html(
                    fig, 
                    include_plotlyjs=True,
                    config={'displayModeBar': True}
                )
                
                if filename:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(html_str)
                
                return html_str
            
            elif format in ['png', 'svg', 'pdf']:
                # Export as image
                img_bytes = pio.to_image(
                    fig,
                    format=format,
                    scale=self.config.export_scale,
                    **kwargs
                )
                
                if filename:
                    with open(filename, 'wb') as f:
                        f.write(img_bytes)
                
                return img_bytes
            
        except Exception as e:
            logger.error(f"Failed to export chart: {e}")
            raise
    
    def create_interactive_dashboard(
        self, 
        aligned_data_dict: Dict[str, AlignedTimeSeries],
        title: str = "春节分析仪表板"
    ) -> go.Figure:
        """Create an interactive dashboard with multiple charts."""
        
        if not aligned_data_dict:
            raise ValueError("No data provided for dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '多股票春节对齐', '模式强度分布',
                '收益率散点图', '风险收益分析'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Extract patterns for all stocks
        patterns = {}
        for symbol, aligned_data in aligned_data_dict.items():
            try:
                pattern = self.sf_engine.identify_seasonal_patterns(aligned_data)
                patterns[symbol] = pattern
            except Exception as e:
                logger.warning(f"Failed to analyze pattern for {symbol}: {e}")
                continue
        
        if not patterns:
            raise ValueError("No valid patterns found for dashboard")
        
        # 1. Multi-stock overlay (simplified)
        colors = px.colors.qualitative.Set3
        for i, (symbol, aligned_data) in enumerate(list(aligned_data_dict.items())[:5]):  # Limit to 5 stocks
            # Calculate average normalized prices by relative day
            daily_averages = {}
            for point in aligned_data.data_points:
                if point.relative_day not in daily_averages:
                    daily_averages[point.relative_day] = []
                daily_averages[point.relative_day].append(point.normalized_price)
            
            relative_days = sorted(daily_averages.keys())
            avg_prices = [np.mean(daily_averages[day]) for day in relative_days]
            
            fig.add_trace(
                go.Scatter(
                    x=relative_days,
                    y=avg_prices,
                    mode='lines',
                    name=symbol,
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 2. Pattern strength distribution
        symbols = list(patterns.keys())
        strengths = [patterns[symbol].pattern_strength for symbol in symbols]
        
        fig.add_trace(
            go.Histogram(
                x=strengths,
                nbinsx=10,
                name='模式强度分布',
                marker_color='lightblue',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Return scatter plot
        returns_before = [patterns[symbol].average_return_before for symbol in symbols]
        returns_after = [patterns[symbol].average_return_after for symbol in symbols]
        
        fig.add_trace(
            go.Scatter(
                x=returns_before,
                y=returns_after,
                mode='markers+text',
                text=symbols,
                textposition='top center',
                name='收益率分布',
                marker=dict(
                    size=8,
                    color=strengths,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="模式强度", x=1.1)
                ),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Risk-return analysis
        volatilities = [
            (patterns[symbol].volatility_before + patterns[symbol].volatility_after) / 2
            for symbol in symbols
        ]
        total_returns = [
            patterns[symbol].average_return_before + patterns[symbol].average_return_after
            for symbol in symbols
        ]
        
        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=total_returns,
                mode='markers+text',
                text=symbols,
                textposition='top center',
                name='风险收益',
                marker=dict(
                    size=10,
                    color='orange',
                    opacity=0.7
                ),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20, color=self.config.text_color)
            ),
            plot_bgcolor=self.config.background_color,
            paper_bgcolor=self.config.background_color,
            font=dict(color=self.config.text_color),
            width=self.config.width * 1.2,
            height=self.config.height,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="相对春节天数", row=1, col=1)
        fig.update_yaxes(title_text="标准化收益率 (%)", row=1, col=1)
        
        fig.update_xaxes(title_text="模式强度", row=1, col=2)
        fig.update_yaxes(title_text="频次", row=1, col=2)
        
        fig.update_xaxes(title_text="春节前收益率 (%)", row=2, col=1)
        fig.update_yaxes(title_text="春节后收益率 (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="波动率", row=2, col=2)
        fig.update_yaxes(title_text="总收益率 (%)", row=2, col=2)
        
        return fig


def create_sample_chart(symbol: str = "000001", years: List[int] = None) -> go.Figure:
    """Create a sample Spring Festival chart for demonstration."""
    
    # Generate sample data
    if years is None:
        years = [2020, 2021, 2022, 2023]
    
    # Create sample aligned data
    from stock_analysis_system.analysis.spring_festival_engine import AlignedDataPoint, AlignedTimeSeries
    
    data_points = []
    base_price = 100
    
    for year in years:
        sf_date = date(year, 2, 1)  # Approximate Spring Festival date
        
        for day_offset in range(-60, 61):  # 60 days before and after
            relative_day = day_offset
            actual_date = sf_date + timedelta(days=day_offset)
            
            # Generate synthetic price with Spring Festival pattern
            seasonal_effect = 5 * np.sin(day_offset * np.pi / 60)  # Seasonal pattern
            random_noise = np.random.normal(0, 2)
            price = base_price + seasonal_effect + random_noise
            
            # Normalize price
            normalized_price = (price - base_price) / base_price * 100
            
            point = AlignedDataPoint(
                original_date=actual_date,
                relative_day=relative_day,
                spring_festival_date=sf_date,
                year=year,
                price=price,
                normalized_price=normalized_price
            )
            data_points.append(point)
    
    aligned_data = AlignedTimeSeries(
        symbol=symbol,
        data_points=data_points,
        window_days=60,
        years=years,
        baseline_price=base_price
    )
    
    # Create chart
    chart_engine = SpringFestivalChartEngine()
    return chart_engine.create_overlay_chart(aligned_data, title=f"{symbol} 春节对齐分析示例")