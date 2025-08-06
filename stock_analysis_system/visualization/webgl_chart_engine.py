"""WebGL-accelerated chart rendering engine for high-performance visualization."""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WebGLChartConfig:
    """Configuration for WebGL-accelerated charts."""
    
    # Performance settings
    max_points_per_trace: int = 10000
    enable_webgl: bool = True
    enable_gpu_acceleration: bool = True
    
    # Visual settings
    line_width: float = 1.5
    marker_size: int = 3
    opacity: float = 0.8
    
    # Interaction settings
    enable_hover: bool = True
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_selection: bool = True
    
    # Animation settings
    enable_animations: bool = True
    animation_duration: int = 500
    
    # Memory management
    enable_data_decimation: bool = True
    decimation_threshold: int = 50000
    
    # Color scheme
    color_palette: List[str] = None
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]


class WebGLChartEngine:
    """High-performance WebGL-accelerated chart rendering engine."""
    
    def __init__(self, config: WebGLChartConfig = None):
        self.config = config or WebGLChartConfig()
        self._trace_counter = 0
        
    def create_high_performance_line_chart(
        self,
        data: Dict[str, pd.Series],
        title: str = "High Performance Chart",
        x_title: str = "X Axis",
        y_title: str = "Y Axis",
        **kwargs
    ) -> go.Figure:
        """Create a high-performance line chart with WebGL acceleration."""
        
        fig = go.Figure()
        
        for i, (name, series) in enumerate(data.items()):
            # Decimate data if necessary
            x_data, y_data = self._prepare_data_for_webgl(series.index, series.values)
            
            # Choose appropriate trace type based on data size
            trace_type = self._get_optimal_trace_type(len(x_data))
            
            color = self.config.color_palette[i % len(self.config.color_palette)]
            
            if trace_type == 'scattergl':
                trace = go.Scattergl(
                    x=x_data,
                    y=y_data,
                    mode='lines+markers' if len(x_data) < 1000 else 'lines',
                    name=name,
                    line=dict(
                        color=color,
                        width=self.config.line_width
                    ),
                    marker=dict(
                        size=self.config.marker_size,
                        opacity=self.config.opacity
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 f'{x_title}: %{{x}}<br>' +
                                 f'{y_title}: %{{y:.2f}}<extra></extra>',
                    connectgaps=False
                )
            else:
                trace = go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines+markers' if len(x_data) < 1000 else 'lines',
                    name=name,
                    line=dict(
                        color=color,
                        width=self.config.line_width
                    ),
                    marker=dict(
                        size=self.config.marker_size,
                        opacity=self.config.opacity
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 f'{x_title}: %{{x}}<br>' +
                                 f'{y_title}: %{{y:.2f}}<extra></extra>',
                    connectgaps=False
                )
            
            fig.add_trace(trace)
        
        # Configure layout for optimal performance
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18)
            ),
            xaxis=dict(
                title=x_title,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False
            ),
            yaxis=dict(
                title=y_title,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False
            ),
            hovermode='x unified' if self.config.enable_hover else False,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            # Performance optimizations
            uirevision=True,  # Preserve UI state on updates
        )
        
        # Configure interactions (this would be passed to the frontend)
        # For now, we just configure the layout
        pass
        
        fig.update_layout(
            dragmode='zoom' if self.config.enable_zoom else False,
            selectdirection='d' if self.config.enable_selection else None
        )
        
        return fig
    
    def create_webgl_candlestick_chart(
        self,
        data: pd.DataFrame,
        title: str = "Candlestick Chart",
        volume_data: pd.Series = None,
        **kwargs
    ) -> go.Figure:
        """Create a high-performance candlestick chart with optional volume."""
        
        # Prepare OHLC data
        x_data, ohlc_data = self._prepare_ohlc_data_for_webgl(data)
        
        if volume_data is not None:
            # Create subplot with volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(title, 'Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Add candlestick
            fig.add_trace(
                go.Candlestick(
                    x=x_data,
                    open=ohlc_data['open'],
                    high=ohlc_data['high'],
                    low=ohlc_data['low'],
                    close=ohlc_data['close'],
                    name="OHLC",
                    increasing_line_color='#00ff00',
                    decreasing_line_color='#ff0000'
                ),
                row=1, col=1
            )
            
            # Add volume bars using WebGL if data is large
            volume_aligned = volume_data.reindex(x_data).fillna(0)
            
            if len(volume_aligned) > self.config.max_points_per_trace:
                # Use Scattergl for large volume data
                fig.add_trace(
                    go.Scattergl(
                        x=x_data,
                        y=volume_aligned.values,
                        mode='lines',
                        fill='tonexty',
                        name='Volume',
                        line=dict(color='rgba(0,100,200,0.5)', width=0),
                        fillcolor='rgba(0,100,200,0.3)',
                        hovertemplate='<b>Date:</b> %{x}<br>' +
                                     '<b>Volume:</b> %{y:,.0f}<extra></extra>'
                    ),
                    row=2, col=1
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=x_data,
                        y=volume_aligned.values,
                        name='Volume',
                        marker_color='rgba(0,100,200,0.5)',
                        hovertemplate='<b>Date:</b> %{x}<br>' +
                                     '<b>Volume:</b> %{y:,.0f}<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Update layout for subplots
            fig.update_layout(
                title=dict(text=title, x=0.5, font=dict(size=18)),
                xaxis2_title="Date",
                yaxis_title="Price",
                yaxis2_title="Volume",
                showlegend=False,
                height=600
            )
            
        else:
            # Single candlestick chart
            fig = go.Figure()
            
            fig.add_trace(
                go.Candlestick(
                    x=x_data,
                    open=ohlc_data['open'],
                    high=ohlc_data['high'],
                    low=ohlc_data['low'],
                    close=ohlc_data['close'],
                    name="OHLC",
                    increasing_line_color='#00ff00',
                    decreasing_line_color='#ff0000'
                )
            )
            
            fig.update_layout(
                title=dict(text=title, x=0.5, font=dict(size=18)),
                xaxis_title="Date",
                yaxis_title="Price",
                showlegend=False
            )
        
        # Apply performance optimizations
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            uirevision=True,
            hovermode='x unified' if self.config.enable_hover else False
        )
        
        return fig
    
    def create_webgl_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Heatmap",
        colorscale: str = 'RdYlBu_r',
        **kwargs
    ) -> go.Figure:
        """Create a high-performance heatmap using WebGL acceleration."""
        
        # For very large heatmaps, we might need to use alternative approaches
        if data.size > 100000:  # 100k cells
            logger.warning(f"Large heatmap detected ({data.size} cells). Consider data aggregation.")
            # Optionally downsample the data
            if self.config.enable_data_decimation:
                data = self._downsample_heatmap_data(data)
        
        fig = go.Figure()
        
        # Use Heatmapgl for better performance with large datasets
        if data.size > 10000:
            # Note: Heatmapgl is not available in standard Plotly
            # We'll use regular Heatmap with optimizations
            fig.add_trace(
                go.Heatmap(
                    z=data.values,
                    x=data.columns,
                    y=data.index,
                    colorscale=colorscale,
                    hoverongaps=False,
                    hovertemplate='<b>X:</b> %{x}<br>' +
                                 '<b>Y:</b> %{y}<br>' +
                                 '<b>Value:</b> %{z:.2f}<extra></extra>',
                    showscale=True
                )
            )
        else:
            fig.add_trace(
                go.Heatmap(
                    z=data.values,
                    x=data.columns,
                    y=data.index,
                    colorscale=colorscale,
                    text=np.round(data.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False,
                    hovertemplate='<b>X:</b> %{x}<br>' +
                                 '<b>Y:</b> %{y}<br>' +
                                 '<b>Value:</b> %{z:.2f}<extra></extra>',
                    showscale=True
                )
            )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_webgl_scatter_plot(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        color_data: np.ndarray = None,
        size_data: np.ndarray = None,
        text_data: List[str] = None,
        title: str = "Scatter Plot",
        **kwargs
    ) -> go.Figure:
        """Create a high-performance scatter plot with WebGL acceleration."""
        
        # Prepare data for WebGL
        x_processed, y_processed = self._prepare_data_for_webgl(x_data, y_data)
        
        fig = go.Figure()
        
        # Determine marker properties
        marker_props = dict(
            size=self.config.marker_size if size_data is None else size_data,
            opacity=self.config.opacity,
            line=dict(width=0.5, color='rgba(0,0,0,0.3)')
        )
        
        if color_data is not None:
            marker_props.update({
                'color': color_data,
                'colorscale': 'Viridis',
                'showscale': True,
                'colorbar': dict(title="Color Scale")
            })
        else:
            marker_props['color'] = self.config.color_palette[0]
        
        # Choose trace type based on data size
        if len(x_processed) > self.config.max_points_per_trace:
            trace = go.Scattergl(
                x=x_processed,
                y=y_processed,
                mode='markers',
                marker=marker_props,
                text=text_data,
                hovertemplate='<b>X:</b> %{x:.2f}<br>' +
                             '<b>Y:</b> %{y:.2f}<br>' +
                             ('<b>Text:</b> %{text}<br>' if text_data else '') +
                             '<extra></extra>'
            )
        else:
            trace = go.Scatter(
                x=x_processed,
                y=y_processed,
                mode='markers',
                marker=marker_props,
                text=text_data,
                hovertemplate='<b>X:</b> %{x:.2f}<br>' +
                             '<b>Y:</b> %{y:.2f}<br>' +
                             ('<b>Text:</b> %{text}<br>' if text_data else '') +
                             '<extra></extra>'
            )
        
        fig.add_trace(trace)
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            hovermode='closest' if self.config.enable_hover else False
        )
        
        return fig
    
    def create_real_time_chart(
        self,
        initial_data: pd.Series,
        title: str = "Real-time Chart",
        max_points: int = 1000,
        **kwargs
    ) -> go.Figure:
        """Create a chart optimized for real-time updates."""
        
        # Prepare initial data
        x_data, y_data = self._prepare_data_for_webgl(
            initial_data.index, 
            initial_data.values,
            max_points=max_points
        )
        
        fig = go.Figure()
        
        # Use Scattergl for real-time performance
        fig.add_trace(
            go.Scattergl(
                x=x_data,
                y=y_data,
                mode='lines',
                name='Real-time Data',
                line=dict(
                    color=self.config.color_palette[0],
                    width=self.config.line_width
                ),
                hovertemplate='<b>Time:</b> %{x}<br>' +
                             '<b>Value:</b> %{y:.2f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            xaxis=dict(
                title="Time",
                type='date',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                title="Value",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            # Real-time optimizations
            uirevision=True,
            transition=dict(
                duration=self.config.animation_duration if self.config.enable_animations else 0,
                easing='cubic-in-out'
            )
        )
        
        return fig
    
    def _prepare_data_for_webgl(
        self, 
        x_data: Union[np.ndarray, pd.Index], 
        y_data: np.ndarray,
        max_points: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for WebGL rendering with optional decimation."""
        
        max_points = max_points or self.config.max_points_per_trace
        
        # Convert to numpy arrays
        if isinstance(x_data, pd.Index):
            x_array = x_data.values
        else:
            x_array = np.array(x_data)
        
        y_array = np.array(y_data)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x_array) | np.isnan(y_array))
        x_clean = x_array[valid_mask]
        y_clean = y_array[valid_mask]
        
        # Decimate data if necessary
        if len(x_clean) > max_points and self.config.enable_data_decimation:
            x_clean, y_clean = self._decimate_data(x_clean, y_clean, max_points)
        
        return x_clean, y_clean
    
    def _prepare_ohlc_data_for_webgl(
        self, 
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare OHLC data for WebGL candlestick charts."""
        
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Clean data
        clean_data = data[required_columns].dropna()
        
        # Decimate if necessary
        if len(clean_data) > self.config.max_points_per_trace and self.config.enable_data_decimation:
            # For OHLC data, we need to be more careful with decimation
            step = len(clean_data) // self.config.max_points_per_trace
            clean_data = clean_data.iloc[::step]
        
        x_data = clean_data.index.values
        ohlc_data = {
            'open': clean_data['open'].values,
            'high': clean_data['high'].values,
            'low': clean_data['low'].values,
            'close': clean_data['close'].values
        }
        
        return x_data, ohlc_data
    
    def _decimate_data(
        self, 
        x_data: np.ndarray, 
        y_data: np.ndarray, 
        max_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decimate data while preserving important features."""
        
        if len(x_data) <= max_points:
            return x_data, y_data
        
        # Use largest triangle three buckets (LTTB) algorithm for better decimation
        try:
            return self._lttb_decimation(x_data, y_data, max_points)
        except Exception as e:
            logger.warning(f"LTTB decimation failed: {e}. Using simple decimation.")
            # Fallback to simple decimation
            step = len(x_data) // max_points
            return x_data[::step], y_data[::step]
    
    def _lttb_decimation(
        self, 
        x_data: np.ndarray, 
        y_data: np.ndarray, 
        max_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Largest Triangle Three Buckets decimation algorithm."""
        
        if len(x_data) <= max_points:
            return x_data, y_data
        
        # Always include first and last points
        if max_points < 3:
            return np.array([x_data[0], x_data[-1]]), np.array([y_data[0], y_data[-1]])
        
        # Convert datetime to numeric if needed
        x_numeric = x_data
        if x_data.dtype.kind == 'M':  # datetime64
            x_numeric = x_data.astype('datetime64[ns]').astype(np.int64)
        
        # Calculate bucket size
        bucket_size = (len(x_numeric) - 2) / (max_points - 2)
        
        # Initialize result arrays
        x_result = [x_data[0]]
        y_result = [y_data[0]]
        
        a = 0  # Initially a is the first point in the triangle
        
        for i in range(max_points - 2):
            # Calculate point average for next bucket
            avg_range_start = int(np.floor((i + 1) * bucket_size) + 1)
            avg_range_end = int(np.floor((i + 2) * bucket_size) + 1)
            avg_range_end = min(avg_range_end, len(x_numeric))
            
            avg_x = np.mean(x_numeric[avg_range_start:avg_range_end])
            avg_y = np.mean(y_data[avg_range_start:avg_range_end])
            
            # Get the range for this bucket
            range_offs = int(np.floor(i * bucket_size) + 1)
            range_to = int(np.floor((i + 1) * bucket_size) + 1)
            
            # Point a
            point_a_x = x_numeric[a]
            point_a_y = y_data[a]
            
            max_area = -1
            next_a = range_offs
            
            for j in range(range_offs, range_to):
                # Calculate triangle area
                area = abs((point_a_x - avg_x) * (y_data[j] - point_a_y) - 
                          (point_a_x - x_numeric[j]) * (avg_y - point_a_y)) * 0.5
                
                if area > max_area:
                    max_area = area
                    next_a = j
            
            x_result.append(x_data[next_a])
            y_result.append(y_data[next_a])
            a = next_a
        
        # Always add last point
        x_result.append(x_data[-1])
        y_result.append(y_data[-1])
        
        return np.array(x_result), np.array(y_result)
    
    def _downsample_heatmap_data(self, data: pd.DataFrame, max_size: int = 10000) -> pd.DataFrame:
        """Downsample heatmap data to reduce size while preserving patterns."""
        
        if data.size <= max_size:
            return data
        
        # Calculate downsampling factors
        rows, cols = data.shape
        target_rows = int(np.sqrt(max_size * rows / cols))
        target_cols = int(np.sqrt(max_size * cols / rows))
        
        # Ensure minimum size
        target_rows = max(target_rows, 10)
        target_cols = max(target_cols, 10)
        
        # Downsample using block averaging
        row_step = max(1, rows // target_rows)
        col_step = max(1, cols // target_cols)
        
        downsampled = data.iloc[::row_step, ::col_step]
        
        logger.info(f"Downsampled heatmap from {data.shape} to {downsampled.shape}")
        
        return downsampled
    
    def _get_optimal_trace_type(self, data_size: int) -> str:
        """Determine optimal trace type based on data size."""
        
        if not self.config.enable_webgl:
            return 'scatter'
        
        if data_size > self.config.max_points_per_trace:
            return 'scattergl'
        else:
            return 'scatter'
    
    def _get_disabled_modebar_buttons(self) -> List[str]:
        """Get list of modebar buttons to disable based on configuration."""
        
        disabled_buttons = []
        
        if not self.config.enable_pan:
            disabled_buttons.extend(['pan2d', 'autoScale2d'])
        
        if not self.config.enable_zoom:
            disabled_buttons.extend(['zoom2d', 'zoomIn2d', 'zoomOut2d'])
        
        if not self.config.enable_selection:
            disabled_buttons.extend(['select2d', 'lasso2d'])
        
        # Always disable some buttons for performance
        disabled_buttons.extend(['hoverClosestCartesian', 'hoverCompareCartesian'])
        
        return disabled_buttons
    
    def update_chart_data(
        self, 
        fig: go.Figure, 
        new_data: Dict[str, Any],
        trace_index: int = 0
    ) -> go.Figure:
        """Update chart data efficiently for real-time updates."""
        
        if trace_index >= len(fig.data):
            raise ValueError(f"Trace index {trace_index} out of range")
        
        # Update data using Plotly's efficient update methods
        with fig.batch_update():
            if 'x' in new_data:
                fig.data[trace_index].x = new_data['x']
            if 'y' in new_data:
                fig.data[trace_index].y = new_data['y']
            if 'z' in new_data:  # For heatmaps
                fig.data[trace_index].z = new_data['z']
        
        return fig
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the chart engine."""
        
        return {
            'webgl_enabled': self.config.enable_webgl,
            'max_points_per_trace': self.config.max_points_per_trace,
            'decimation_enabled': self.config.enable_data_decimation,
            'decimation_threshold': self.config.decimation_threshold,
            'traces_created': self._trace_counter,
            'gpu_acceleration': self.config.enable_gpu_acceleration
        }


# Utility functions for WebGL chart creation
def create_webgl_time_series_chart(
    data: pd.DataFrame,
    columns: List[str] = None,
    title: str = "Time Series Chart",
    **kwargs
) -> go.Figure:
    """Convenience function to create WebGL time series chart."""
    
    engine = WebGLChartEngine()
    
    if columns is None:
        columns = data.columns.tolist()
    
    chart_data = {col: data[col] for col in columns if col in data.columns}
    
    return engine.create_high_performance_line_chart(
        chart_data,
        title=title,
        x_title="Time",
        y_title="Value",
        **kwargs
    )


def create_webgl_stock_chart(
    ohlc_data: pd.DataFrame,
    volume_data: pd.Series = None,
    title: str = "Stock Chart",
    **kwargs
) -> go.Figure:
    """Convenience function to create WebGL stock chart."""
    
    engine = WebGLChartEngine()
    
    return engine.create_webgl_candlestick_chart(
        ohlc_data,
        title=title,
        volume_data=volume_data,
        **kwargs
    )


def create_webgl_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Correlation Heatmap",
    **kwargs
) -> go.Figure:
    """Convenience function to create WebGL correlation heatmap."""
    
    engine = WebGLChartEngine()
    
    return engine.create_webgl_heatmap(
        correlation_matrix,
        title=title,
        colorscale='RdBu',
        **kwargs
    )