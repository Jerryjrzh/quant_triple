"""Comprehensive chart interaction system with advanced zoom, pan, selection, and annotation capabilities."""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, date
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = logging.getLogger(__name__)


class InteractionMode(Enum):
    """Chart interaction modes."""
    ZOOM = "zoom"
    PAN = "pan"
    SELECT = "select"
    ANNOTATE = "annotate"
    CROSSHAIR = "crosshair"
    MEASURE = "measure"


class AnnotationType(Enum):
    """Types of annotations."""
    TEXT = "text"
    ARROW = "arrow"
    LINE = "line"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    TREND_LINE = "trend_line"
    FIBONACCI = "fibonacci"
    SUPPORT_RESISTANCE = "support_resistance"


@dataclass
class ChartAnnotation:
    """Chart annotation data structure."""
    id: str
    type: AnnotationType
    x: Union[float, List[float]]
    y: Union[float, List[float]]
    text: str = ""
    color: str = "#000000"
    size: int = 12
    opacity: float = 1.0
    visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CrosshairData:
    """Crosshair data structure."""
    x: float
    y: float
    visible: bool = True
    color: str = "#666666"
    width: int = 1
    dash: str = "dash"


@dataclass
class SelectionData:
    """Selection data structure."""
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    selected_points: List[int]
    selection_type: str = "rectangle"  # rectangle, lasso, etc.


@dataclass
class ZoomState:
    """Zoom state data structure."""
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    zoom_level: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0


class ChartInteractionSystem:
    """Comprehensive chart interaction system."""
    
    def __init__(self):
        self.annotations: Dict[str, ChartAnnotation] = {}
        self.crosshair: Optional[CrosshairData] = None
        self.selection: Optional[SelectionData] = None
        self.zoom_history: List[ZoomState] = []
        self.current_mode: InteractionMode = InteractionMode.ZOOM
        self.synchronized_charts: List[str] = []
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
    def enable_advanced_zoom(
        self,
        fig: go.Figure,
        enable_box_zoom: bool = True,
        enable_wheel_zoom: bool = True,
        enable_double_click_reset: bool = True,
        zoom_sensitivity: float = 1.0
    ) -> go.Figure:
        """Enable advanced zoom capabilities."""
        
        # Configure zoom settings
        fig.update_layout(
            dragmode='zoom' if enable_box_zoom else 'pan'
        )
        
        # Add zoom controls
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"dragmode": "zoom"}],
                            label="Box Zoom",
                            method="relayout"
                        ),
                        dict(
                            args=[{"dragmode": "pan"}],
                            label="Pan",
                            method="relayout"
                        ),
                        dict(
                            args=[{"xaxis.autorange": True, "yaxis.autorange": True}],
                            label="Auto Scale",
                            method="relayout"
                        ),
                        dict(
                            args=[{"xaxis.range": None, "yaxis.range": None}],
                            label="Reset Zoom",
                            method="relayout"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )
        
        # Add zoom event handling
        self._add_zoom_event_handlers(fig)
        
        return fig
    
    def enable_advanced_pan(
        self,
        fig: go.Figure,
        enable_drag_pan: bool = True,
        enable_keyboard_pan: bool = True,
        pan_sensitivity: float = 1.0
    ) -> go.Figure:
        """Enable advanced pan capabilities."""
        
        if enable_drag_pan:
            fig.update_layout(dragmode='pan')
        
        # Add pan controls
        pan_buttons = [
            dict(
                args=[{"xaxis.range[0]": "xaxis.range[0] - (xaxis.range[1] - xaxis.range[0]) * 0.1"}],
                label="← Pan Left",
                method="relayout"
            ),
            dict(
                args=[{"xaxis.range[0]": "xaxis.range[0] + (xaxis.range[1] - xaxis.range[0]) * 0.1"}],
                label="Pan Right →",
                method="relayout"
            ),
            dict(
                args=[{"yaxis.range[0]": "yaxis.range[0] + (yaxis.range[1] - yaxis.range[0]) * 0.1"}],
                label="↑ Pan Up",
                method="relayout"
            ),
            dict(
                args=[{"yaxis.range[0]": "yaxis.range[0] - (yaxis.range[1] - yaxis.range[0]) * 0.1"}],
                label="Pan Down ↓",
                method="relayout"
            )
        ]
        
        new_menu = dict(
            type="buttons",
            direction="left",
            buttons=pan_buttons,
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.01,
            xanchor="left",
            y=0.95,
            yanchor="top"
        )
        
        if fig.layout.updatemenus:
            existing_menus = list(fig.layout.updatemenus)
            existing_menus.append(new_menu)
            fig.update_layout(updatemenus=existing_menus)
        else:
            fig.update_layout(updatemenus=[new_menu])
        
        return fig
    
    def enable_selection_tools(
        self,
        fig: go.Figure,
        enable_box_select: bool = True,
        enable_lasso_select: bool = True,
        selection_callback: Optional[Callable] = None
    ) -> go.Figure:
        """Enable advanced selection tools."""
        
        # Configure selection modes
        config_updates = {}
        
        if enable_box_select:
            config_updates['dragmode'] = 'select'
        
        if enable_lasso_select:
            # Add lasso selection button
            lasso_button = dict(
                args=[{"dragmode": "lasso"}],
                label="Lasso Select",
                method="relayout"
            )
            
            if fig.layout.updatemenus:
                # Convert tuple to list, modify, then update
                existing_menus = list(fig.layout.updatemenus)
                if existing_menus:
                    existing_buttons = list(existing_menus[0].buttons)
                    existing_buttons.append(lasso_button)
                    existing_menus[0].buttons = existing_buttons
                    fig.update_layout(updatemenus=existing_menus)
            else:
                fig.update_layout(
                    updatemenus=[
                        dict(
                            type="buttons",
                            direction="left",
                            buttons=[lasso_button],
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=0.01,
                            xanchor="left",
                            y=0.88,
                            yanchor="top"
                        )
                    ]
                )
        
        # Add selection event handling
        if selection_callback:
            self.add_event_callback('selection', selection_callback)
        
        fig.update_layout(**config_updates)
        
        return fig
    
    def enable_crosshair_system(
        self,
        fig: go.Figure,
        crosshair_color: str = "#666666",
        crosshair_width: int = 1,
        show_coordinates: bool = True
    ) -> go.Figure:
        """Enable crosshair system with coordinate display."""
        
        # Add crosshair lines (initially hidden)
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color=crosshair_color,
            line_width=crosshair_width,
            visible=False,
            annotation_text="",
            annotation_position="top right"
        )
        
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color=crosshair_color,
            line_width=crosshair_width,
            visible=False,
            annotation_text="",
            annotation_position="top right"
        )
        
        # Enable hover mode for crosshair
        fig.update_layout(hovermode='x unified')
        
        # Add crosshair toggle button
        crosshair_button = dict(
            args=[{"hovermode": "x unified"}],
            label="Toggle Crosshair",
            method="relayout"
        )
        
        if fig.layout.updatemenus:
            # Convert tuple to list, modify, then update
            existing_menus = list(fig.layout.updatemenus)
            if existing_menus:
                existing_buttons = list(existing_menus[0].buttons)
                existing_buttons.append(crosshair_button)
                existing_menus[0].buttons = existing_buttons
                fig.update_layout(updatemenus=existing_menus)
        else:
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        buttons=[crosshair_button],
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.01,
                        xanchor="left",
                        y=0.81,
                        yanchor="top"
                    )
                ]
            )
        
        return fig
    
    def add_annotation_tools(
        self,
        fig: go.Figure,
        enable_text_annotations: bool = True,
        enable_drawing_tools: bool = True,
        enable_trend_lines: bool = True
    ) -> go.Figure:
        """Add comprehensive annotation tools."""
        
        annotation_buttons = []
        
        if enable_text_annotations:
            annotation_buttons.append(
                dict(
                    args=[{"dragmode": "drawrect"}],
                    label="Add Text",
                    method="relayout"
                )
            )
        
        if enable_drawing_tools:
            annotation_buttons.extend([
                dict(
                    args=[{"dragmode": "drawline"}],
                    label="Draw Line",
                    method="relayout"
                ),
                dict(
                    args=[{"dragmode": "drawrect"}],
                    label="Draw Rectangle",
                    method="relayout"
                ),
                dict(
                    args=[{"dragmode": "drawcircle"}],
                    label="Draw Circle",
                    method="relayout"
                )
            ])
        
        if enable_trend_lines:
            annotation_buttons.append(
                dict(
                    args=[{"dragmode": "drawline"}],
                    label="Trend Line",
                    method="relayout"
                )
            )
        
        # Add annotation controls
        if annotation_buttons:
            new_menu = dict(
                type="buttons",
                direction="left",
                buttons=annotation_buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=0.74,
                yanchor="top"
            )
            
            if fig.layout.updatemenus:
                existing_menus = list(fig.layout.updatemenus)
                existing_menus.append(new_menu)
                fig.update_layout(updatemenus=existing_menus)
            else:
                fig.update_layout(updatemenus=[new_menu])
        
        # Configure drawing tools
        fig.update_layout(
            newshape=dict(
                line_color="red",
                line_width=2,
                opacity=0.8,
                fillcolor="rgba(255,0,0,0.1)"
            ),
            modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 
                        'drawcircle', 'drawrect', 'eraseshape']
        )
        
        return fig
    
    def enable_chart_synchronization(
        self,
        charts: List[go.Figure],
        sync_zoom: bool = True,
        sync_pan: bool = True,
        sync_selection: bool = False
    ) -> List[go.Figure]:
        """Enable synchronization between multiple charts."""
        
        # Add unique IDs to charts for synchronization
        for i, fig in enumerate(charts):
            chart_id = f"chart_{i}"
            self.synchronized_charts.append(chart_id)
            
            # Add synchronization configuration
            if sync_zoom or sync_pan:
                fig.update_layout(
                    uirevision=chart_id,  # Preserve UI state
                    # Add custom JavaScript for synchronization would go here
                    # This is a simplified version - full implementation would require
                    # custom JavaScript callbacks
                )
        
        return charts
    
    def add_measurement_tools(
        self,
        fig: go.Figure,
        enable_distance_measurement: bool = True,
        enable_angle_measurement: bool = True,
        enable_area_measurement: bool = True
    ) -> go.Figure:
        """Add measurement tools to the chart."""
        
        measurement_buttons = []
        
        if enable_distance_measurement:
            measurement_buttons.append(
                dict(
                    args=[{"dragmode": "drawline"}],
                    label="Measure Distance",
                    method="relayout"
                )
            )
        
        if enable_angle_measurement:
            measurement_buttons.append(
                dict(
                    args=[{"dragmode": "drawline"}],
                    label="Measure Angle",
                    method="relayout"
                )
            )
        
        if enable_area_measurement:
            measurement_buttons.append(
                dict(
                    args=[{"dragmode": "drawrect"}],
                    label="Measure Area",
                    method="relayout"
                )
            )
        
        if measurement_buttons:
            new_menu = dict(
                type="buttons",
                direction="left",
                buttons=measurement_buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=0.67,
                yanchor="top"
            )
            
            if fig.layout.updatemenus:
                existing_menus = list(fig.layout.updatemenus)
                existing_menus.append(new_menu)
                fig.update_layout(updatemenus=existing_menus)
            else:
                fig.update_layout(updatemenus=[new_menu])
        
        return fig
    
    def add_custom_tooltip_system(
        self,
        fig: go.Figure,
        tooltip_template: str = None,
        show_multiple_traces: bool = True,
        custom_hover_data: Dict[str, Any] = None
    ) -> go.Figure:
        """Add advanced tooltip system."""
        
        # Configure hover mode
        if show_multiple_traces:
            fig.update_layout(hovermode='x unified')
        else:
            fig.update_layout(hovermode='closest')
        
        # Update hover templates for all traces
        if tooltip_template:
            for trace in fig.data:
                if hasattr(trace, 'hovertemplate'):
                    trace.hovertemplate = tooltip_template
        
        # Add custom hover data if provided
        if custom_hover_data:
            for trace in fig.data:
                if hasattr(trace, 'customdata'):
                    trace.customdata = custom_hover_data.get(trace.name, [])
        
        # Configure hover appearance
        fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="black",
                font_size=12,
                font_family="Arial"
            )
        )
        
        return fig
    
    def add_annotation(
        self,
        fig: go.Figure,
        annotation: ChartAnnotation
    ) -> go.Figure:
        """Add an annotation to the chart."""
        
        self.annotations[annotation.id] = annotation
        
        if annotation.type == AnnotationType.TEXT:
            fig.add_annotation(
                x=annotation.x,
                y=annotation.y,
                text=annotation.text,
                showarrow=False,
                font=dict(size=annotation.size, color=annotation.color),
                opacity=annotation.opacity,
                visible=annotation.visible
            )
        
        elif annotation.type == AnnotationType.ARROW:
            fig.add_annotation(
                x=annotation.x[1] if isinstance(annotation.x, list) else annotation.x,
                y=annotation.y[1] if isinstance(annotation.y, list) else annotation.y,
                ax=annotation.x[0] if isinstance(annotation.x, list) else annotation.x - 1,
                ay=annotation.y[0] if isinstance(annotation.y, list) else annotation.y - 1,
                text=annotation.text,
                showarrow=True,
                arrowhead=2,
                arrowcolor=annotation.color,
                font=dict(size=annotation.size, color=annotation.color),
                opacity=annotation.opacity,
                visible=annotation.visible
            )
        
        elif annotation.type == AnnotationType.LINE:
            if isinstance(annotation.x, list) and isinstance(annotation.y, list):
                fig.add_shape(
                    type="line",
                    x0=annotation.x[0],
                    y0=annotation.y[0],
                    x1=annotation.x[1],
                    y1=annotation.y[1],
                    line=dict(color=annotation.color, width=annotation.size),
                    opacity=annotation.opacity,
                    visible=annotation.visible
                )
        
        elif annotation.type == AnnotationType.RECTANGLE:
            if isinstance(annotation.x, list) and isinstance(annotation.y, list):
                fig.add_shape(
                    type="rect",
                    x0=annotation.x[0],
                    y0=annotation.y[0],
                    x1=annotation.x[1],
                    y1=annotation.y[1],
                    line=dict(color=annotation.color, width=2),
                    fillcolor=f"rgba({annotation.color[1:3]},{annotation.color[3:5]},{annotation.color[5:7]},0.1)",
                    opacity=annotation.opacity,
                    visible=annotation.visible
                )
        
        elif annotation.type == AnnotationType.CIRCLE:
            if isinstance(annotation.x, list) and isinstance(annotation.y, list):
                fig.add_shape(
                    type="circle",
                    x0=annotation.x[0],
                    y0=annotation.y[0],
                    x1=annotation.x[1],
                    y1=annotation.y[1],
                    line=dict(color=annotation.color, width=2),
                    fillcolor=f"rgba({annotation.color[1:3]},{annotation.color[3:5]},{annotation.color[5:7]},0.1)",
                    opacity=annotation.opacity,
                    visible=annotation.visible
                )
        
        return fig
    
    def remove_annotation(
        self,
        fig: go.Figure,
        annotation_id: str
    ) -> go.Figure:
        """Remove an annotation from the chart."""
        
        if annotation_id in self.annotations:
            del self.annotations[annotation_id]
            
            # Remove from figure (simplified - would need more complex logic for shapes)
            # This is a placeholder for the actual implementation
            logger.info(f"Removed annotation {annotation_id}")
        
        return fig
    
    def add_event_callback(
        self,
        event_type: str,
        callback: Callable
    ) -> None:
        """Add event callback for chart interactions."""
        
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        
        self.event_callbacks[event_type].append(callback)
    
    def trigger_event(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Trigger event callbacks."""
        
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(event_data)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
    
    def save_zoom_state(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float]
    ) -> None:
        """Save current zoom state."""
        
        zoom_state = ZoomState(
            x_range=x_range,
            y_range=y_range,
            zoom_level=self._calculate_zoom_level(x_range, y_range),
            center_x=(x_range[0] + x_range[1]) / 2,
            center_y=(y_range[0] + y_range[1]) / 2
        )
        
        self.zoom_history.append(zoom_state)
        
        # Keep only last 10 zoom states
        if len(self.zoom_history) > 10:
            self.zoom_history.pop(0)
    
    def restore_zoom_state(
        self,
        fig: go.Figure,
        state_index: int = -1
    ) -> go.Figure:
        """Restore a previous zoom state."""
        
        if self.zoom_history and abs(state_index) <= len(self.zoom_history):
            zoom_state = self.zoom_history[state_index]
            
            fig.update_layout(
                xaxis_range=list(zoom_state.x_range),
                yaxis_range=list(zoom_state.y_range)
            )
        
        return fig
    
    def export_annotations(self) -> Dict[str, Any]:
        """Export all annotations to a dictionary."""
        
        return {
            annotation_id: {
                'type': annotation.type.value,
                'x': annotation.x,
                'y': annotation.y,
                'text': annotation.text,
                'color': annotation.color,
                'size': annotation.size,
                'opacity': annotation.opacity,
                'visible': annotation.visible,
                'metadata': annotation.metadata,
                'created_at': annotation.created_at.isoformat()
            }
            for annotation_id, annotation in self.annotations.items()
        }
    
    def import_annotations(
        self,
        fig: go.Figure,
        annotations_data: Dict[str, Any]
    ) -> go.Figure:
        """Import annotations from a dictionary."""
        
        for annotation_id, data in annotations_data.items():
            annotation = ChartAnnotation(
                id=annotation_id,
                type=AnnotationType(data['type']),
                x=data['x'],
                y=data['y'],
                text=data['text'],
                color=data['color'],
                size=data['size'],
                opacity=data['opacity'],
                visible=data['visible'],
                metadata=data['metadata'],
                created_at=datetime.fromisoformat(data['created_at'])
            )
            
            fig = self.add_annotation(fig, annotation)
        
        return fig
    
    def _add_zoom_event_handlers(self, fig: go.Figure) -> None:
        """Add zoom event handlers (placeholder for actual implementation)."""
        
        # This would typically involve JavaScript callbacks
        # For now, we'll just log that handlers are added
        logger.info("Zoom event handlers added")
    
    def _calculate_zoom_level(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float]
    ) -> float:
        """Calculate zoom level based on axis ranges."""
        
        x_span = x_range[1] - x_range[0]
        y_span = y_range[1] - y_range[0]
        
        # Simple zoom level calculation (would be more sophisticated in practice)
        return 1.0 / (x_span * y_span) if x_span > 0 and y_span > 0 else 1.0


class AdvancedTooltipSystem:
    """Advanced tooltip system with custom formatting and multiple data sources."""
    
    def __init__(self):
        self.tooltip_templates = {}
        self.custom_data_sources = {}
    
    def create_multi_line_tooltip(
        self,
        data_sources: Dict[str, pd.DataFrame],
        x_column: str = 'date',
        format_functions: Dict[str, Callable] = None
    ) -> str:
        """Create multi-line tooltip template."""
        
        template_parts = ["<b>%{x}</b><br>"]
        
        for source_name, df in data_sources.items():
            if format_functions and source_name in format_functions:
                formatter = format_functions[source_name]
                template_parts.append(f"<b>{source_name}:</b> {formatter('%{y}')}<br>")
            else:
                template_parts.append(f"<b>{source_name}:</b> %{{y:.2f}}<br>")
        
        template_parts.append("<extra></extra>")
        
        return "".join(template_parts)
    
    def create_financial_tooltip(
        self,
        include_ohlc: bool = True,
        include_volume: bool = True,
        include_indicators: List[str] = None
    ) -> str:
        """Create specialized financial data tooltip."""
        
        template_parts = ["<b>Date:</b> %{x}<br>"]
        
        if include_ohlc:
            template_parts.extend([
                "<b>Open:</b> %{open:.2f}<br>",
                "<b>High:</b> %{high:.2f}<br>",
                "<b>Low:</b> %{low:.2f}<br>",
                "<b>Close:</b> %{close:.2f}<br>"
            ])
        
        if include_volume:
            template_parts.append("<b>Volume:</b> %{customdata[0]:,.0f}<br>")
        
        if include_indicators:
            for i, indicator in enumerate(include_indicators):
                template_parts.append(f"<b>{indicator}:</b> %{{customdata[{i+1}]:.2f}}<br>")
        
        template_parts.append("<extra></extra>")
        
        return "".join(template_parts)


# Utility functions for chart interactions
def create_interactive_chart(
    data: pd.DataFrame,
    chart_type: str = "line",
    enable_all_interactions: bool = True,
    **kwargs
) -> go.Figure:
    """Create a fully interactive chart with all interaction features enabled."""
    
    interaction_system = ChartInteractionSystem()
    
    # Create base chart
    if chart_type == "line":
        fig = go.Figure()
        for column in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[column],
                mode='lines',
                name=column
            ))
    elif chart_type == "candlestick":
        fig = go.Figure(data=go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close']
        ))
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    if enable_all_interactions:
        # Enable all interaction features
        fig = interaction_system.enable_advanced_zoom(fig)
        fig = interaction_system.enable_advanced_pan(fig)
        fig = interaction_system.enable_selection_tools(fig)
        fig = interaction_system.enable_crosshair_system(fig)
        fig = interaction_system.add_annotation_tools(fig)
        fig = interaction_system.add_measurement_tools(fig)
        fig = interaction_system.add_custom_tooltip_system(fig)
    
    return fig


def synchronize_charts(
    charts: List[go.Figure],
    sync_zoom: bool = True,
    sync_pan: bool = True
) -> List[go.Figure]:
    """Synchronize multiple charts for coordinated interaction."""
    
    interaction_system = ChartInteractionSystem()
    
    return interaction_system.enable_chart_synchronization(
        charts,
        sync_zoom=sync_zoom,
        sync_pan=sync_pan
    )