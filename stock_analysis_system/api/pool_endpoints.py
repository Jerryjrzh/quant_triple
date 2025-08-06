"""
Stock Pool Management API Endpoints

This module provides FastAPI endpoints for the stock pool management system
including pool CRUD operations, analytics, export/import, and dashboard functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Query
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json
import io
import logging

from ..pool import (
    StockPoolManager,
    PoolType,
    PoolStatus,
    PoolAnalyticsDashboard,
    PoolExportImport
)

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/pools", tags=["Stock Pool Management"])

# Global instances (in production, these would be dependency injected)
pool_manager = StockPoolManager()
analytics_dashboard = PoolAnalyticsDashboard(pool_manager)
export_import = PoolExportImport(pool_manager)

# Pydantic models for request/response
class CreatePoolRequest(BaseModel):
    name: str = Field(..., description="Pool name")
    pool_type: PoolType = Field(..., description="Pool type")
    description: str = Field("", description="Pool description")
    max_stocks: int = Field(100, description="Maximum number of stocks")
    rebalance_frequency: str = Field("monthly", description="Rebalancing frequency")

class AddStockRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    name: str = Field("", description="Stock name")
    weight: float = Field(0.0, description="Stock weight in pool")
    notes: str = Field("", description="Notes about the stock")
    tags: List[str] = Field([], description="Stock tags")

class UpdatePoolFromScreeningRequest(BaseModel):
    screening_results: List[Dict[str, Any]] = Field(..., description="Screening results")
    max_additions: int = Field(10, description="Maximum stocks to add")
    replace_existing: bool = Field(False, description="Replace existing stocks")

class SetAutoUpdateRulesRequest(BaseModel):
    rules: Dict[str, Any] = Field(..., description="Auto-update rules")

class ExportPoolRequest(BaseModel):
    format_type: str = Field("json", description="Export format")
    include_history: bool = Field(True, description="Include pool history")
    include_analytics: bool = Field(True, description="Include analytics")

class CreateBackupRequest(BaseModel):
    include_full_history: bool = Field(True, description="Include full history")
    backup_name: Optional[str] = Field(None, description="Custom backup name")

class CreateShareLinkRequest(BaseModel):
    access_level: str = Field("read_only", description="Access level")
    expiry_days: int = Field(30, description="Link expiry in days")
    include_analytics: bool = Field(True, description="Include analytics")

# Pool CRUD endpoints
@router.post("/", response_model=Dict[str, Any])
async def create_pool(request: CreatePoolRequest):
    """Create a new stock pool"""
    try:
        pool_id = await pool_manager.create_pool(
            name=request.name,
            pool_type=request.pool_type,
            description=request.description,
            max_stocks=request.max_stocks,
            rebalance_frequency=request.rebalance_frequency
        )
        
        return {
            "success": True,
            "pool_id": pool_id,
            "message": f"Pool '{request.name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create pool: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=Dict[str, Any])
async def list_pools():
    """List all pools"""
    try:
        pools_info = []
        
        for pool_id, pool in pool_manager.pools.items():
            pools_info.append({
                "pool_id": pool_id,
                "name": pool.name,
                "pool_type": pool.pool_type.value,
                "description": pool.description,
                "status": pool.status.value,
                "total_stocks": len(pool.stocks),
                "created_date": pool.created_date.isoformat(),
                "last_modified": pool.last_modified.isoformat()
            })
        
        return {
            "success": True,
            "pools": pools_info,
            "total_pools": len(pools_info)
        }
        
    except Exception as e:
        logger.error(f"Failed to list pools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{pool_id}", response_model=Dict[str, Any])
async def get_pool(pool_id: str):
    """Get pool details"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        pool = pool_manager.pools[pool_id]
        
        pool_data = {
            "pool_id": pool_id,
            "name": pool.name,
            "pool_type": pool.pool_type.value,
            "description": pool.description,
            "status": pool.status.value,
            "max_stocks": pool.max_stocks,
            "rebalance_frequency": pool.rebalance_frequency,
            "created_date": pool.created_date.isoformat(),
            "last_modified": pool.last_modified.isoformat(),
            "stocks": [
                {
                    "symbol": stock.symbol,
                    "name": stock.name,
                    "added_date": stock.added_date.isoformat(),
                    "added_price": stock.added_price,
                    "current_price": stock.current_price,
                    "weight": stock.weight,
                    "notes": stock.notes,
                    "tags": stock.tags,
                    "return_pct": ((stock.current_price - stock.added_price) / stock.added_price * 100) if stock.added_price > 0 else 0
                }
                for stock in pool.stocks
            ],
            "metrics": {
                "total_return": pool.metrics.total_return,
                "annualized_return": pool.metrics.annualized_return,
                "volatility": pool.metrics.volatility,
                "sharpe_ratio": pool.metrics.sharpe_ratio,
                "max_drawdown": pool.metrics.max_drawdown,
                "win_rate": pool.metrics.win_rate,
                "risk_score": pool.metrics.risk_score,
                "last_updated": pool.metrics.last_updated.isoformat() if pool.metrics.last_updated else None
            }
        }
        
        return {
            "success": True,
            "pool": pool_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pool {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{pool_id}", response_model=Dict[str, Any])
async def delete_pool(pool_id: str):
    """Delete a pool"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        pool_name = pool_manager.pools[pool_id].name
        del pool_manager.pools[pool_id]
        
        # Clean up history
        if pool_id in pool_manager.pool_history:
            del pool_manager.pool_history[pool_id]
        
        return {
            "success": True,
            "message": f"Pool '{pool_name}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete pool {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Stock management endpoints
@router.post("/{pool_id}/stocks", response_model=Dict[str, Any])
async def add_stock_to_pool(pool_id: str, request: AddStockRequest):
    """Add a stock to a pool"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        success = await pool_manager.add_stock_to_pool(
            pool_id=pool_id,
            symbol=request.symbol,
            name=request.name,
            weight=request.weight,
            notes=request.notes,
            tags=request.tags
        )
        
        if success:
            return {
                "success": True,
                "message": f"Stock {request.symbol} added to pool"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to add stock to pool")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add stock to pool {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{pool_id}/stocks/{symbol}", response_model=Dict[str, Any])
async def remove_stock_from_pool(pool_id: str, symbol: str):
    """Remove a stock from a pool"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        success = await pool_manager.remove_stock_from_pool(pool_id, symbol)
        
        if success:
            return {
                "success": True,
                "message": f"Stock {symbol} removed from pool"
            }
        else:
            raise HTTPException(status_code=404, detail="Stock not found in pool")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove stock from pool {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{pool_id}/update-from-screening", response_model=Dict[str, Any])
async def update_pool_from_screening(pool_id: str, request: UpdatePoolFromScreeningRequest):
    """Update pool based on screening results"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        result = await pool_manager.update_pool_from_screening(
            pool_id=pool_id,
            screening_results=request.screening_results,
            max_additions=request.max_additions,
            replace_existing=request.replace_existing
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update pool from screening {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@router.get("/{pool_id}/analytics", response_model=Dict[str, Any])
async def get_pool_analytics(pool_id: str):
    """Get comprehensive pool analytics"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        analytics = await pool_manager.get_pool_analytics(pool_id)
        
        if "error" in analytics:
            raise HTTPException(status_code=400, detail=analytics["error"])
        
        return {
            "success": True,
            "analytics": analytics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analytics for pool {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare", response_model=Dict[str, Any])
async def compare_pools(pool_ids: List[str]):
    """Compare multiple pools"""
    try:
        # Validate all pools exist
        for pool_id in pool_ids:
            if pool_id not in pool_manager.pools:
                raise HTTPException(status_code=404, detail=f"Pool {pool_id} not found")
        
        comparison = await pool_manager.compare_pools(pool_ids)
        
        if "error" in comparison:
            raise HTTPException(status_code=400, detail=comparison["error"])
        
        return {
            "success": True,
            "comparison": comparison
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare pools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dashboard endpoints
@router.get("/{pool_id}/dashboard", response_model=Dict[str, Any])
async def get_pool_dashboard(pool_id: str):
    """Get pool performance dashboard"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        dashboard_data = await analytics_dashboard.create_pool_performance_dashboard(pool_id)
        
        if "error" in dashboard_data:
            raise HTTPException(status_code=400, detail=dashboard_data["error"])
        
        return {
            "success": True,
            "dashboard": dashboard_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create dashboard for pool {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dashboard/comparison", response_model=Dict[str, Any])
async def get_multi_pool_dashboard(pool_ids: List[str]):
    """Get multi-pool comparison dashboard"""
    try:
        # Validate all pools exist
        for pool_id in pool_ids:
            if pool_id not in pool_manager.pools:
                raise HTTPException(status_code=404, detail=f"Pool {pool_id} not found")
        
        dashboard_data = await analytics_dashboard.create_multi_pool_comparison_dashboard(pool_ids)
        
        if "error" in dashboard_data:
            raise HTTPException(status_code=400, detail=dashboard_data["error"])
        
        return {
            "success": True,
            "dashboard": dashboard_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create comparison dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{pool_id}/sector-analysis", response_model=Dict[str, Any])
async def get_sector_analysis(pool_id: str):
    """Get sector and industry analysis"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        analysis = await analytics_dashboard.create_sector_industry_analysis(pool_id)
        
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])
        
        return {
            "success": True,
            "analysis": analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get sector analysis for pool {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk-analysis", response_model=Dict[str, Any])
async def get_risk_analysis(pool_ids: List[str]):
    """Get risk distribution analysis across pools"""
    try:
        # Validate all pools exist
        for pool_id in pool_ids:
            if pool_id not in pool_manager.pools:
                raise HTTPException(status_code=404, detail=f"Pool {pool_id} not found")
        
        analysis = await analytics_dashboard.create_risk_distribution_analysis(pool_ids)
        
        return {
            "success": True,
            "analysis": analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get risk analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{pool_id}/optimization", response_model=Dict[str, Any])
async def get_optimization_recommendations(pool_id: str):
    """Get pool optimization recommendations"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        optimization = await analytics_dashboard.create_pool_optimization_recommendations(pool_id)
        
        if "error" in optimization:
            raise HTTPException(status_code=400, detail=optimization["error"])
        
        return {
            "success": True,
            "optimization": optimization
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization for pool {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Auto-update endpoints
@router.post("/{pool_id}/auto-update-rules", response_model=Dict[str, Any])
async def set_auto_update_rules(pool_id: str, request: SetAutoUpdateRulesRequest):
    """Set automatic update rules for a pool"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        success = await pool_manager.set_auto_update_rules(pool_id, request.rules)
        
        if success:
            return {
                "success": True,
                "message": "Auto-update rules set successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to set auto-update rules")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set auto-update rules for pool {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{pool_id}/auto-update-rules", response_model=Dict[str, Any])
async def get_auto_update_rules(pool_id: str):
    """Get automatic update rules for a pool"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        pool = pool_manager.pools[pool_id]
        
        return {
            "success": True,
            "rules": pool.auto_update_rules
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get auto-update rules for pool {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Export endpoints
@router.post("/{pool_id}/export", response_model=Dict[str, Any])
async def export_pool(pool_id: str, request: ExportPoolRequest):
    """Export a pool to specified format"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        result = await export_import.export_pool(
            pool_id=pool_id,
            format_type=request.format_type,
            include_history=request.include_history,
            include_analytics=request.include_analytics
        )
        
        if result.get("success"):
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Export failed"))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export pool {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{pool_id}/export/download")
async def download_pool_export(
    pool_id: str,
    format_type: str = Query("json", description="Export format"),
    include_history: bool = Query(True, description="Include history"),
    include_analytics: bool = Query(True, description="Include analytics")
):
    """Download pool export file"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        pool_name = pool_manager.pools[pool_id].name
        filename = f"{pool_name.replace(' ', '_')}_export.{format_type}"
        
        result = await export_import.export_pool(
            pool_id=pool_id,
            format_type=format_type,
            include_history=include_history,
            include_analytics=include_analytics,
            output_path=filename
        )
        
        if result.get("success"):
            return FileResponse(
                path=result["file_path"],
                filename=filename,
                media_type='application/octet-stream'
            )
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Export failed"))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download pool export {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export/multiple", response_model=Dict[str, Any])
async def export_multiple_pools(
    pool_ids: List[str],
    format_type: str = Query("json", description="Export format"),
    create_archive: bool = Query(True, description="Create archive")
):
    """Export multiple pools"""
    try:
        # Validate all pools exist
        for pool_id in pool_ids:
            if pool_id not in pool_manager.pools:
                raise HTTPException(status_code=404, detail=f"Pool {pool_id} not found")
        
        result = await export_import.export_multiple_pools(
            pool_ids=pool_ids,
            format_type=format_type,
            create_archive=create_archive
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export multiple pools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Import endpoints
@router.post("/import", response_model=Dict[str, Any])
async def import_pool(
    file: UploadFile = File(...),
    format_type: Optional[str] = Query(None, description="File format"),
    merge_strategy: str = Query("replace", description="Merge strategy"),
    validate_data: bool = Query(True, description="Validate data")
):
    """Import a pool from uploaded file"""
    try:
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            result = await export_import.import_pool(
                file_path=tmp_file_path,
                format_type=format_type,
                merge_strategy=merge_strategy,
                validate_data=validate_data
            )
            
            return result
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
    except Exception as e:
        logger.error(f"Failed to import pool: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Backup and restore endpoints
@router.post("/{pool_id}/backup", response_model=Dict[str, Any])
async def create_pool_backup(pool_id: str, request: CreateBackupRequest):
    """Create a backup of a pool"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        result = await export_import.create_pool_backup(
            pool_id=pool_id,
            include_full_history=request.include_full_history,
            backup_name=request.backup_name
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create backup for pool {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/restore", response_model=Dict[str, Any])
async def restore_pool_from_backup(
    file: UploadFile = File(...),
    restore_strategy: str = Query("replace", description="Restore strategy")
):
    """Restore a pool from backup file"""
    try:
        # Save uploaded backup file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            result = await export_import.restore_pool_from_backup(
                backup_path=tmp_file_path,
                restore_strategy=restore_strategy
            )
            
            return result
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
    except Exception as e:
        logger.error(f"Failed to restore pool from backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Sharing endpoints
@router.post("/{pool_id}/share", response_model=Dict[str, Any])
async def create_shareable_link(pool_id: str, request: CreateShareLinkRequest):
    """Create a shareable link for pool collaboration"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        result = await export_import.create_shareable_pool_link(
            pool_id=pool_id,
            access_level=request.access_level,
            expiry_days=request.expiry_days,
            include_analytics=request.include_analytics
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create shareable link for pool {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# External tool integration endpoints
@router.post("/{pool_id}/export/external/{tool_type}", response_model=Dict[str, Any])
async def export_for_external_tool(pool_id: str, tool_type: str):
    """Export pool for external portfolio management tools"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        result = await export_import.export_for_external_tools(
            pool_id=pool_id,
            tool_type=tool_type
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export for external tool {tool_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# History endpoints
@router.get("/{pool_id}/history", response_model=Dict[str, Any])
async def get_pool_history(pool_id: str, limit: int = Query(100, description="History limit")):
    """Get pool modification history"""
    try:
        if pool_id not in pool_manager.pools:
            raise HTTPException(status_code=404, detail="Pool not found")
        
        history = await pool_manager.get_pool_history(pool_id, limit)
        
        return {
            "success": True,
            "history": history,
            "total_entries": len(history)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get history for pool {pool_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check for pool management system"""
    try:
        return {
            "success": True,
            "status": "healthy",
            "total_pools": len(pool_manager.pools),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))