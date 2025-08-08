"""
Chart Template Manager

Advanced template management system for creating, storing, and applying
custom chart templates with:
- Template creation and editing
- Template library management
- Theme and style customization
- Template sharing and export
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ChartStyle:
    """Chart styling configuration"""
    color_scheme: str = "plotly"
    background_color: str = "#ffffff"
    grid_color: str = "#e6e6e6"
    text_color: str = "#2e2e2e"
    font_family: str = "Arial, sans-serif"
    font_size: int = 12
    line_width: int = 2
    marker_size: int = 6
    opacity: float = 0.8
    border_width: int = 1
    border_color: str = "#cccccc"


@dataclass
class ChartLayout:
    """Chart layout configuration"""
    title: str = ""
    title_font_size: int = 16
    width: int = 800
    height: int = 600
    margin_top: int = 50
    margin_bottom: int = 50
    margin_left: int = 50
    margin_right: int = 50
    show_legend: bool = True
    legend_position: str = "top"
    x_axis_title: str = ""
    y_axis_title: str = ""
    show_grid: bool = True
    show_toolbar: bool = True


@dataclass
class ChartTemplate:
    """Complete chart template definition"""
    id: str
    name: str
    description: str
    category: str
    chart_type: str
    style: ChartStyle
    layout: ChartLayout
    custom_config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    author: str = "system"
    version: str = "1.0"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ChartTemplateManager:
    """Manager for chart templates"""
    
    def __init__(self, templates_dir: str = "chart_templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        self.templates: Dict[str, ChartTemplate] = {}
        self.load_templates()
        
        # Initialize default templates
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default chart templates"""
        default_templates = [
            self._create_candlestick_template(),
            self._create_line_chart_template(),
            self._create_volume_profile_template(),
            self._create_technical_analysis_template(),
            self._create_correlation_heatmap_template(),
            self._create_3d_surface_template(),
            self._create_spring_festival_template(),
            self._create_risk_dashboard_template()
        ]
        
        for template in default_templates:
            if template.id not in self.templates:
                self.save_template(template)
    
    def _create_candlestick_template(self) -> ChartTemplate:
        """Create professional candlestick chart template"""
        return ChartTemplate(
            id="professional_candlestick",
            name="Professional Candlestick",
            description="Professional-grade candlestick chart with volume and indicators",
            category="Technical Analysis",
            chart_type="candlestick",
            style=ChartStyle(
                color_scheme="plotly_dark",
                background_color="#1e1e1e",
                grid_color="#404040",
                text_color="#ffffff",
                font_family="Roboto, sans-serif"
            ),
            layout=ChartLayout(
                title="Stock Price Analysis",
                width=1200,
                height=800,
                show_legend=True,
                legend_position="top",
                x_axis_title="Date",
                y_axis_title="Price"
            ),
            custom_config={
                "show_volume": True,
                "volume_height_ratio": 0.3,
                "indicators": ["ma_20", "ma_50", "bb_upper", "bb_lower"],
                "candlestick_colors": {
                    "increasing": "#00ff88",
                    "decreasing": "#ff4444"
                },
                "volume_colors": {
                    "increasing": "#00ff8844",
                    "decreasing": "#ff444444"
                }
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["candlestick", "technical", "professional"]
        )
    
    def _create_line_chart_template(self) -> ChartTemplate:
        """Create elegant line chart template"""
        return ChartTemplate(
            id="elegant_line_chart",
            name="Elegant Line Chart",
            description="Clean and elegant line chart for price trends",
            category="Basic Charts",
            chart_type="line",
            style=ChartStyle(
                color_scheme="plotly_white",
                background_color="#fafafa",
                grid_color="#e0e0e0",
                text_color="#333333",
                font_family="Inter, sans-serif",
                line_width=3
            ),
            layout=ChartLayout(
                title="Price Trend Analysis",
                width=1000,
                height=600,
                show_legend=True,
                legend_position="bottom"
            ),
            custom_config={
                "line_smoothing": True,
                "fill_area": True,
                "gradient_fill": True,
                "hover_mode": "x unified",
                "colors": ["#667eea", "#764ba2", "#f093fb", "#f5576c"]
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["line", "trend", "elegant"]
        )
    
    def _create_volume_profile_template(self) -> ChartTemplate:
        """Create volume profile template"""
        return ChartTemplate(
            id="volume_profile",
            name="Volume Profile Analysis",
            description="Advanced volume profile with price distribution",
            category="Volume Analysis",
            chart_type="volume_profile",
            style=ChartStyle(
                color_scheme="ggplot2",
                background_color="#f8f9fa",
                grid_color="#dee2e6",
                text_color="#495057"
            ),
            layout=ChartLayout(
                title="Volume Profile Analysis",
                width=1400,
                height=900,
                show_legend=True
            ),
            custom_config={
                "profile_bins": 50,
                "show_poc": True,  # Point of Control
                "show_value_area": True,
                "value_area_percentage": 70,
                "profile_colors": {
                    "high_volume": "#ff6b6b",
                    "medium_volume": "#feca57", 
                    "low_volume": "#48dbfb"
                }
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["volume", "profile", "advanced"]
        )
    
    def _create_technical_analysis_template(self) -> ChartTemplate:
        """Create comprehensive technical analysis template"""
        return ChartTemplate(
            id="technical_analysis_pro",
            name="Technical Analysis Pro",
            description="Comprehensive technical analysis with multiple indicators",
            category="Technical Analysis",
            chart_type="multi_indicator",
            style=ChartStyle(
                color_scheme="plotly_dark",
                background_color="#0d1117",
                grid_color="#30363d",
                text_color="#f0f6fc",
                font_family="JetBrains Mono, monospace"
            ),
            layout=ChartLayout(
                title="Technical Analysis Dashboard",
                width=1600,
                height=1000,
                show_legend=True,
                legend_position="right"
            ),
            custom_config={
                "subplots": [
                    {"type": "candlestick", "height_ratio": 0.6},
                    {"type": "volume", "height_ratio": 0.15},
                    {"type": "rsi", "height_ratio": 0.125},
                    {"type": "macd", "height_ratio": 0.125}
                ],
                "indicators": {
                    "moving_averages": [5, 10, 20, 50, 200],
                    "bollinger_bands": True,
                    "rsi": {"period": 14, "overbought": 70, "oversold": 30},
                    "macd": {"fast": 12, "slow": 26, "signal": 9},
                    "volume_sma": 20
                },
                "color_palette": {
                    "ma_5": "#ff6b6b",
                    "ma_10": "#4ecdc4", 
                    "ma_20": "#45b7d1",
                    "ma_50": "#f9ca24",
                    "ma_200": "#f0932b",
                    "bb_upper": "#6c5ce7",
                    "bb_lower": "#6c5ce7",
                    "rsi": "#a29bfe",
                    "macd": "#fd79a8",
                    "macd_signal": "#fdcb6e"
                }
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["technical", "indicators", "professional", "multi-panel"]
        )
    
    def _create_correlation_heatmap_template(self) -> ChartTemplate:
        """Create correlation heatmap template"""
        return ChartTemplate(
            id="correlation_heatmap",
            name="Correlation Heatmap",
            description="Interactive correlation heatmap for portfolio analysis",
            category="Portfolio Analysis",
            chart_type="heatmap",
            style=ChartStyle(
                color_scheme="RdYlBu",
                background_color="#ffffff",
                text_color="#2c3e50"
            ),
            layout=ChartLayout(
                title="Stock Correlation Matrix",
                width=800,
                height=800,
                show_legend=True
            ),
            custom_config={
                "colorscale": "RdYlBu",
                "show_values": True,
                "value_format": ".2f",
                "cluster_method": "ward",
                "show_dendrograms": True
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["correlation", "heatmap", "portfolio"]
        )
    
    def _create_3d_surface_template(self) -> ChartTemplate:
        """Create 3D surface plot template"""
        return ChartTemplate(
            id="3d_surface_plot",
            name="3D Surface Analysis",
            description="3D surface plot for multi-dimensional analysis",
            category="3D Visualization",
            chart_type="surface_3d",
            style=ChartStyle(
                color_scheme="viridis",
                background_color="#1a1a1a",
                text_color="#ffffff"
            ),
            layout=ChartLayout(
                title="3D Market Surface Analysis",
                width=1000,
                height=800
            ),
            custom_config={
                "colorscale": "Viridis",
                "show_contours": True,
                "lighting": {
                    "ambient": 0.4,
                    "diffuse": 0.8,
                    "specular": 0.05
                },
                "camera": {
                    "eye": {"x": 1.2, "y": 1.2, "z": 0.6}
                }
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["3d", "surface", "advanced"]
        )
    
    def _create_spring_festival_template(self) -> ChartTemplate:
        """Create Spring Festival analysis template"""
        return ChartTemplate(
            id="spring_festival_analysis",
            name="Spring Festival Analysis",
            description="Specialized template for Spring Festival pattern analysis",
            category="Seasonal Analysis",
            chart_type="spring_festival",
            style=ChartStyle(
                color_scheme="plotly",
                background_color="#fff8f0",
                grid_color="#ffe4cc",
                text_color="#8b4513",
                font_family="Noto Sans SC, sans-serif"
            ),
            layout=ChartLayout(
                title="春节效应分析 (Spring Festival Effect Analysis)",
                width=1200,
                height=700,
                show_legend=True,
                legend_position="top"
            ),
            custom_config={
                "spring_festival_marker": {
                    "symbol": "star",
                    "size": 15,
                    "color": "#ff4444"
                },
                "pattern_colors": {
                    "strong_pattern": "#ff6b6b",
                    "medium_pattern": "#feca57",
                    "weak_pattern": "#48dbfb"
                },
                "show_confidence_bands": True,
                "show_pattern_strength": True,
                "chinese_labels": True
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["spring_festival", "seasonal", "chinese", "pattern"]
        )
    
    def _create_risk_dashboard_template(self) -> ChartTemplate:
        """Create risk dashboard template"""
        return ChartTemplate(
            id="risk_dashboard",
            name="Risk Management Dashboard",
            description="Comprehensive risk analysis dashboard",
            category="Risk Management",
            chart_type="risk_dashboard",
            style=ChartStyle(
                color_scheme="plotly_dark",
                background_color="#2c3e50",
                grid_color="#34495e",
                text_color="#ecf0f1"
            ),
            layout=ChartLayout(
                title="Risk Management Dashboard",
                width=1600,
                height=1200,
                show_legend=True
            ),
            custom_config={
                "risk_metrics": ["var", "cvar", "max_drawdown", "sharpe_ratio"],
                "gauge_charts": True,
                "risk_colors": {
                    "low": "#2ecc71",
                    "medium": "#f39c12", 
                    "high": "#e74c3c",
                    "extreme": "#8e44ad"
                },
                "alert_thresholds": {
                    "var_95": 0.05,
                    "max_drawdown": 0.15,
                    "sharpe_ratio": 1.0
                }
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["risk", "dashboard", "var", "metrics"]
        )
    
    def save_template(self, template: ChartTemplate):
        """Save a chart template"""
        template.updated_at = datetime.now()
        self.templates[template.id] = template
        
        # Save to file
        template_file = self.templates_dir / f"{template.id}.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            # Convert dataclass to dict for JSON serialization
            template_dict = asdict(template)
            # Convert datetime objects to ISO strings
            template_dict['created_at'] = template.created_at.isoformat()
            template_dict['updated_at'] = template.updated_at.isoformat()
            json.dump(template_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved template: {template.name} ({template.id})")
    
    def load_templates(self):
        """Load all templates from files"""
        if not self.templates_dir.exists():
            return
        
        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_dict = json.load(f)
                
                # Convert ISO strings back to datetime objects
                template_dict['created_at'] = datetime.fromisoformat(template_dict['created_at'])
                template_dict['updated_at'] = datetime.fromisoformat(template_dict['updated_at'])
                
                # Convert nested dicts back to dataclasses
                template_dict['style'] = ChartStyle(**template_dict['style'])
                template_dict['layout'] = ChartLayout(**template_dict['layout'])
                
                template = ChartTemplate(**template_dict)
                self.templates[template.id] = template
                
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
    
    def get_template(self, template_id: str) -> Optional[ChartTemplate]:
        """Get a template by ID"""
        return self.templates.get(template_id)
    
    def list_templates(self, category: str = None) -> List[ChartTemplate]:
        """List all templates, optionally filtered by category"""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        return sorted(templates, key=lambda t: t.name)
    
    def delete_template(self, template_id: str) -> bool:
        """Delete a template"""
        if template_id not in self.templates:
            return False
        
        # Remove from memory
        del self.templates[template_id]
        
        # Remove file
        template_file = self.templates_dir / f"{template_id}.json"
        if template_file.exists():
            template_file.unlink()
        
        logger.info(f"Deleted template: {template_id}")
        return True
    
    def duplicate_template(self, template_id: str, new_name: str, new_id: str = None) -> Optional[ChartTemplate]:
        """Duplicate an existing template"""
        original = self.get_template(template_id)
        if not original:
            return None
        
        if new_id is None:
            new_id = f"{template_id}_copy"
        
        # Create a copy
        new_template = ChartTemplate(
            id=new_id,
            name=new_name,
            description=f"Copy of {original.description}",
            category=original.category,
            chart_type=original.chart_type,
            style=ChartStyle(**asdict(original.style)),
            layout=ChartLayout(**asdict(original.layout)),
            custom_config=original.custom_config.copy(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            author=original.author,
            version="1.0",
            tags=original.tags.copy()
        )
        
        self.save_template(new_template)
        return new_template
    
    def search_templates(self, query: str) -> List[ChartTemplate]:
        """Search templates by name, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for template in self.templates.values():
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                any(query_lower in tag.lower() for tag in template.tags)):
                results.append(template)
        
        return sorted(results, key=lambda t: t.name)
    
    def get_categories(self) -> List[str]:
        """Get all available template categories"""
        categories = set(template.category for template in self.templates.values())
        return sorted(categories)
    
    def export_template(self, template_id: str, export_path: str) -> bool:
        """Export a template to a file"""
        template = self.get_template(template_id)
        if not template:
            return False
        
        try:
            template_dict = asdict(template)
            template_dict['created_at'] = template.created_at.isoformat()
            template_dict['updated_at'] = template.updated_at.isoformat()
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(template_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported template {template_id} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting template {template_id}: {e}")
            return False
    
    def import_template(self, import_path: str, new_id: str = None) -> Optional[ChartTemplate]:
        """Import a template from a file"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                template_dict = json.load(f)
            
            # Convert ISO strings back to datetime objects
            template_dict['created_at'] = datetime.fromisoformat(template_dict['created_at'])
            template_dict['updated_at'] = datetime.fromisoformat(template_dict['updated_at'])
            
            # Convert nested dicts back to dataclasses
            template_dict['style'] = ChartStyle(**template_dict['style'])
            template_dict['layout'] = ChartLayout(**template_dict['layout'])
            
            # Use new ID if provided
            if new_id:
                template_dict['id'] = new_id
            
            template = ChartTemplate(**template_dict)
            self.save_template(template)
            
            logger.info(f"Imported template from {import_path}")
            return template
            
        except Exception as e:
            logger.error(f"Error importing template from {import_path}: {e}")
            return None