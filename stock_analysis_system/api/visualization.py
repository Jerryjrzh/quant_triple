"""API endpoints for visualization services."""

import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from stock_analysis_system.analysis.spring_festival_engine import (
    SpringFestivalAlignmentEngine,
)
from stock_analysis_system.data.data_source_manager import DataSourceManager
from stock_analysis_system.visualization.spring_festival_charts import (
    SpringFestivalChartConfig,
    SpringFestivalChartEngine,
    create_sample_chart,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/visualization", tags=["visualization"])


class ChartRequest(BaseModel):
    """Request model for chart generation."""

    symbol: str = Field(..., description="Stock symbol")
    years: Optional[List[int]] = Field(None, description="Years to include in analysis")
    start_date: Optional[date] = Field(None, description="Start date for data")
    end_date: Optional[date] = Field(None, description="End date for data")
    chart_type: str = Field("overlay", description="Type of chart to generate")
    title: Optional[str] = Field(None, description="Custom chart title")
    show_pattern_info: bool = Field(True, description="Show pattern information")


class MultiStockChartRequest(BaseModel):
    """Request model for multi-stock chart generation."""

    symbols: List[str] = Field(..., description="List of stock symbols")
    years: Optional[List[int]] = Field(None, description="Years to include in analysis")
    chart_type: str = Field("comparison", description="Type of chart to generate")
    title: Optional[str] = Field(None, description="Custom chart title")


class ExportRequest(BaseModel):
    """Request model for chart export."""

    chart_data: Dict[str, Any] = Field(..., description="Chart configuration data")
    format: str = Field("png", description="Export format (png, svg, pdf, html)")
    filename: Optional[str] = Field(None, description="Output filename")


@router.get("/sample")
async def get_sample_chart(
    symbol: str = Query("000001", description="Stock symbol for sample"),
    format: str = Query("html", description="Response format"),
):
    """Get a sample Spring Festival chart for demonstration."""
    try:
        fig = create_sample_chart(symbol)

        if format == "html":
            html_content = fig.to_html(include_plotlyjs=True)
            return HTMLResponse(content=html_content)
        elif format == "json":
            return {"chart_data": fig.to_dict()}
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    except Exception as e:
        logger.error(f"Failed to generate sample chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spring-festival-chart")
async def create_spring_festival_chart(request: ChartRequest):
    """Create Spring Festival alignment chart for a single stock."""
    try:
        # Initialize engines
        sf_engine = SpringFestivalAlignmentEngine()
        chart_engine = SpringFestivalChartEngine()
        data_manager = DataSourceManager()

        # Get stock data
        end_date = request.end_date or date.today()
        start_date = request.start_date or date(end_date.year - 10, 1, 1)

        stock_data = await data_manager.get_stock_data(
            symbol=request.symbol, start_date=start_date, end_date=end_date
        )

        if stock_data.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for symbol {request.symbol}"
            )

        # Perform Spring Festival alignment
        aligned_data = sf_engine.align_to_spring_festival(stock_data, request.years)

        # Create chart
        fig = chart_engine.create_overlay_chart(
            aligned_data=aligned_data,
            title=request.title,
            show_pattern_info=request.show_pattern_info,
            selected_years=request.years,
        )

        return {"chart_data": fig.to_dict()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create Spring Festival chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-stock-chart")
async def create_multi_stock_chart(request: MultiStockChartRequest):
    """Create comparison chart for multiple stocks."""
    try:
        if len(request.symbols) > 10:
            raise HTTPException(
                status_code=400, detail="Maximum 10 stocks allowed for comparison"
            )

        # Initialize engines
        sf_engine = SpringFestivalAlignmentEngine()
        chart_engine = SpringFestivalChartEngine()
        data_manager = DataSourceManager()

        # Get data for all stocks
        aligned_data_dict = {}

        for symbol in request.symbols:
            try:
                # Get stock data
                stock_data = await data_manager.get_stock_data(
                    symbol=symbol, start_date=date(2018, 1, 1), end_date=date.today()
                )

                if not stock_data.empty:
                    aligned_data = sf_engine.align_to_spring_festival(
                        stock_data, request.years
                    )
                    aligned_data_dict[symbol] = aligned_data

            except Exception as e:
                logger.warning(f"Failed to process {symbol}: {e}")
                continue

        if not aligned_data_dict:
            raise HTTPException(
                status_code=404,
                detail="No valid data found for any of the requested symbols",
            )

        # Create appropriate chart based on type
        if request.chart_type == "comparison":
            fig = chart_engine.create_pattern_summary_chart(
                patterns={
                    symbol: sf_engine.identify_seasonal_patterns(data)
                    for symbol, data in aligned_data_dict.items()
                },
                title=request.title or "多股票春节模式对比",
            )
        elif request.chart_type == "cluster":
            fig = chart_engine.create_cluster_visualization(
                aligned_data_dict=aligned_data_dict,
                title=request.title or "春节模式聚类分析",
            )
        elif request.chart_type == "dashboard":
            fig = chart_engine.create_interactive_dashboard(
                aligned_data_dict=aligned_data_dict,
                title=request.title or "春节分析仪表板",
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported chart type: {request.chart_type}"
            )

        return {"chart_data": fig.to_dict()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create multi-stock chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_chart(request: ExportRequest):
    """Export chart to various formats."""
    try:
        # Reconstruct figure from chart data
        import plotly.graph_objects as go

        fig = go.Figure(request.chart_data)

        # Initialize chart engine for export
        chart_engine = SpringFestivalChartEngine()

        # Export chart
        if request.format == "html":
            html_content = chart_engine.export_chart(
                fig=fig, filename=request.filename, format="html"
            )
            return HTMLResponse(content=html_content)

        elif request.format in ["png", "svg", "pdf"]:
            image_bytes = chart_engine.export_chart(
                fig=fig, filename=request.filename, format=request.format
            )

            # Determine content type
            content_types = {
                "png": "image/png",
                "svg": "image/svg+xml",
                "pdf": "application/pdf",
            }

            return Response(
                content=image_bytes,
                media_type=content_types[request.format],
                headers={
                    "Content-Disposition": f"attachment; filename=chart.{request.format}"
                },
            )

        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported export format: {request.format}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chart-types")
async def get_available_chart_types():
    """Get list of available chart types."""
    return {
        "single_stock": [
            {
                "type": "overlay",
                "name": "春节对齐叠加图",
                "description": "多年数据叠加显示",
            },
            {
                "type": "pattern",
                "name": "模式分析图",
                "description": "显示季节性模式信息",
            },
        ],
        "multi_stock": [
            {
                "type": "comparison",
                "name": "对比分析图",
                "description": "多股票模式对比",
            },
            {
                "type": "cluster",
                "name": "聚类分析图",
                "description": "基于模式的聚类可视化",
            },
            {
                "type": "dashboard",
                "name": "综合仪表板",
                "description": "多维度分析仪表板",
            },
        ],
        "export_formats": ["png", "svg", "pdf", "html"],
    }


@router.get("/config")
async def get_chart_config():
    """Get current chart configuration."""
    config = SpringFestivalChartConfig()
    return {
        "width": config.width,
        "height": config.height,
        "colors": config.colors,
        "background_color": config.background_color,
        "export_formats": config.export_formats,
        "interactive_features": {
            "zoom": config.enable_zoom,
            "pan": config.enable_pan,
            "hover": config.enable_hover,
            "crossfilter": config.enable_crossfilter,
        },
    }


@router.get("/health")
async def visualization_health_check():
    """Health check for visualization service."""
    try:
        # Test chart creation
        fig = create_sample_chart("TEST", [2023])

        return {
            "status": "healthy",
            "service": "visualization",
            "timestamp": datetime.now().isoformat(),
            "features": {
                "plotly_available": True,
                "chart_generation": True,
                "export_capabilities": True,
            },
        }
    except Exception as e:
        logger.error(f"Visualization health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "visualization",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }
