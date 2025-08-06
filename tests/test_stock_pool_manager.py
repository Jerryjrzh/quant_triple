"""
Tests for Stock Pool Management System

This module contains comprehensive tests for the stock pool management functionality
including pool creation, stock management, analytics, and automated updates.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from stock_analysis_system.pool import (
    StockPoolManager,
    StockPool,
    StockInfo,
    PoolMetrics,
    PoolType,
    PoolStatus,
    PoolAnalyticsDashboard,
    PoolExportImport
)

@pytest.fixture
def pool_manager():
    """Create a pool manager instance for testing"""
    data_source_mock = Mock()
    risk_engine_mock = Mock()
    
    manager = StockPoolManager(
        data_source_manager=data_source_mock,
        risk_engine=risk_engine_mock
    )
    
    return manager

@pytest.fixture
async def sample_pool(pool_manager):
    """Create a sample pool for testing"""
    pool_id = await pool_manager.create_pool(
        name="Test Pool",
        pool_type=PoolType.WATCHLIST,
        description="A test pool for unit testing",
        max_stocks=50
    )
    
    # Add some sample stocks
    await pool_manager.add_stock_to_pool(
        pool_id=pool_id,
        symbol="AAPL",
        name="Apple Inc.",
        weight=0.3,
        notes="Tech stock"
    )
    
    await pool_manager.add_stock_to_pool(
        pool_id=pool_id,
        symbol="GOOGL",
        name="Alphabet Inc.",
        weight=0.25,
        notes="Search engine"
    )
    
    await pool_manager.add_stock_to_pool(
        pool_id=pool_id,
        symbol="MSFT",
        name="Microsoft Corp.",
        weight=0.2,
        notes="Software company"
    )
    
    return pool_id

class TestStockPoolManager:
    """Test cases for StockPoolManager"""
    
    @pytest.mark.asyncio
    async def test_create_pool(self, pool_manager):
        """Test pool creation"""
        pool_id = await pool_manager.create_pool(
            name="Test Pool",
            pool_type=PoolType.CORE_HOLDINGS,
            description="Test description",
            max_stocks=100
        )
        
        assert pool_id is not None
        assert pool_id in pool_manager.pools
        
        pool = pool_manager.pools[pool_id]
        assert pool.name == "Test Pool"
        assert pool.pool_type == PoolType.CORE_HOLDINGS
        assert pool.description == "Test description"
        assert pool.max_stocks == 100
        assert pool.status == PoolStatus.ACTIVE
        assert len(pool.stocks) == 0
    
    @pytest.mark.asyncio
    async def test_add_stock_to_pool(self, pool_manager, sample_pool):
        """Test adding stocks to a pool"""
        pool_id = await sample_pool
        pool = pool_manager.pools[pool_id]
        
        # Check that stocks were added
        assert len(pool.stocks) == 3
        
        # Check stock details
        aapl_stock = next((s for s in pool.stocks if s.symbol == "AAPL"), None)
        assert aapl_stock is not None
        assert aapl_stock.name == "Apple Inc."
        assert aapl_stock.weight == 0.3
        assert aapl_stock.notes == "Tech stock"
    
    @pytest.mark.asyncio
    async def test_remove_stock_from_pool(self, pool_manager, sample_pool):
        """Test removing stocks from a pool"""
        pool_id = sample_pool
        
        # Remove a stock
        success = await pool_manager.remove_stock_from_pool(pool_id, "AAPL")
        assert success is True
        
        pool = pool_manager.pools[pool_id]
        assert len(pool.stocks) == 2
        
        # Verify stock is removed
        aapl_stock = next((s for s in pool.stocks if s.symbol == "AAPL"), None)
        assert aapl_stock is None
    
    @pytest.mark.asyncio
    async def test_pool_capacity_limit(self, pool_manager):
        """Test pool capacity limits"""
        pool_id = await pool_manager.create_pool(
            name="Small Pool",
            pool_type=PoolType.WATCHLIST,
            max_stocks=2
        )
        
        # Add stocks up to limit
        success1 = await pool_manager.add_stock_to_pool(pool_id, "AAPL", "Apple")
        success2 = await pool_manager.add_stock_to_pool(pool_id, "GOOGL", "Google")
        success3 = await pool_manager.add_stock_to_pool(pool_id, "MSFT", "Microsoft")
        
        assert success1 is True
        assert success2 is True
        assert success3 is False  # Should fail due to capacity limit
        
        pool = pool_manager.pools[pool_id]
        assert len(pool.stocks) == 2
    
    @pytest.mark.asyncio
    async def test_duplicate_stock_prevention(self, pool_manager, sample_pool):
        """Test prevention of duplicate stocks in pool"""
        pool_id = sample_pool
        
        # Try to add duplicate stock
        success = await pool_manager.add_stock_to_pool(
            pool_id=pool_id,
            symbol="AAPL",
            name="Apple Inc. Duplicate"
        )
        
        assert success is False
        
        pool = pool_manager.pools[pool_id]
        assert len(pool.stocks) == 3  # Should remain unchanged
    
    @pytest.mark.asyncio
    async def test_update_pool_from_screening(self, pool_manager, sample_pool):
        """Test updating pool from screening results"""
        pool_id = sample_pool
        
        screening_results = [
            {"symbol": "TSLA", "name": "Tesla Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corp."},
            {"symbol": "AMD", "name": "Advanced Micro Devices"}
        ]
        
        result = await pool_manager.update_pool_from_screening(
            pool_id=pool_id,
            screening_results=screening_results,
            max_additions=2
        )
        
        assert result["success"] is True
        assert len(result["added_stocks"]) == 2
        
        pool = pool_manager.pools[pool_id]
        assert len(pool.stocks) == 5  # 3 original + 2 new
    
    @pytest.mark.asyncio
    async def test_get_pool_analytics(self, pool_manager, sample_pool):
        """Test pool analytics generation"""
        pool_id = sample_pool
        
        analytics = await pool_manager.get_pool_analytics(pool_id)
        
        assert "error" not in analytics
        assert "basic_info" in analytics
        assert "performance_metrics" in analytics
        assert "stock_analysis" in analytics
        assert "risk_analysis" in analytics
        assert "sector_analysis" in analytics
        assert "recommendations" in analytics
        
        # Check basic info
        basic_info = analytics["basic_info"]
        assert basic_info["pool_id"] == pool_id
        assert basic_info["name"] == "Test Pool"
        assert basic_info["total_stocks"] == 3
    
    @pytest.mark.asyncio
    async def test_compare_pools(self, pool_manager):
        """Test pool comparison functionality"""
        # Create two pools
        pool_id1 = await pool_manager.create_pool("Pool 1", PoolType.WATCHLIST)
        pool_id2 = await pool_manager.create_pool("Pool 2", PoolType.CORE_HOLDINGS)
        
        # Add stocks to both pools
        await pool_manager.add_stock_to_pool(pool_id1, "AAPL", "Apple")
        await pool_manager.add_stock_to_pool(pool_id2, "GOOGL", "Google")
        
        comparison = await pool_manager.compare_pools([pool_id1, pool_id2])
        
        assert "error" not in comparison
        assert "pools" in comparison
        assert "comparative_metrics" in comparison
        assert "rankings" in comparison
        assert "correlation_analysis" in comparison
        
        assert pool_id1 in comparison["pools"]
        assert pool_id2 in comparison["pools"]
    
    @pytest.mark.asyncio
    async def test_auto_update_rules(self, pool_manager, sample_pool):
        """Test automatic update rules"""
        pool_id = sample_pool
        
        rules = {
            "enabled": True,
            "interval_hours": 24,
            "remove_poor_performers": True,
            "poor_performance_threshold": -0.2
        }
        
        success = await pool_manager.set_auto_update_rules(pool_id, rules)
        assert success is True
        
        pool = pool_manager.pools[pool_id]
        assert pool.auto_update_rules == rules
    
    @pytest.mark.asyncio
    async def test_pool_history_tracking(self, pool_manager, sample_pool):
        """Test pool history tracking"""
        pool_id = sample_pool
        
        # Perform some actions to generate history
        await pool_manager.add_stock_to_pool(pool_id, "TSLA", "Tesla")
        await pool_manager.remove_stock_from_pool(pool_id, "MSFT")
        
        history = await pool_manager.get_pool_history(pool_id)
        
        assert len(history) > 0
        assert any(entry["action"] == "stock_added" for entry in history)
        assert any(entry["action"] == "stock_removed" for entry in history)

class TestPoolAnalyticsDashboard:
    """Test cases for PoolAnalyticsDashboard"""
    
    @pytest.fixture
    def dashboard(self, pool_manager):
        """Create dashboard instance"""
        return PoolAnalyticsDashboard(pool_manager)
    
    @pytest.mark.asyncio
    async def test_create_pool_performance_dashboard(self, dashboard, pool_manager, sample_pool):
        """Test performance dashboard creation"""
        pool_id = sample_pool
        
        dashboard_data = await dashboard.create_pool_performance_dashboard(pool_id)
        
        assert "error" not in dashboard_data
        assert "pool_info" in dashboard_data
        assert "charts" in dashboard_data
        assert "metrics" in dashboard_data
        assert "recommendations" in dashboard_data
        
        # Check charts
        charts = dashboard_data["charts"]
        assert "performance_overview" in charts
        assert "stock_breakdown" in charts
        assert "sector_distribution" in charts
        assert "risk_analysis" in charts
        assert "performance_timeline" in charts
    
    @pytest.mark.asyncio
    async def test_multi_pool_comparison_dashboard(self, dashboard, pool_manager):
        """Test multi-pool comparison dashboard"""
        # Create multiple pools
        pool_id1 = await pool_manager.create_pool("Pool 1", PoolType.WATCHLIST)
        pool_id2 = await pool_manager.create_pool("Pool 2", PoolType.CORE_HOLDINGS)
        
        await pool_manager.add_stock_to_pool(pool_id1, "AAPL", "Apple")
        await pool_manager.add_stock_to_pool(pool_id2, "GOOGL", "Google")
        
        dashboard_data = await dashboard.create_multi_pool_comparison_dashboard([pool_id1, pool_id2])
        
        assert "error" not in dashboard_data
        assert "comparison_data" in dashboard_data
        assert "charts" in dashboard_data
        assert "summary" in dashboard_data
        
        # Check charts
        charts = dashboard_data["charts"]
        assert "performance_comparison" in charts
        assert "risk_return_scatter" in charts
        assert "correlation_heatmap" in charts
        assert "sector_comparison" in charts
    
    @pytest.mark.asyncio
    async def test_sector_industry_analysis(self, dashboard, pool_manager, sample_pool):
        """Test sector and industry analysis"""
        pool_id = sample_pool
        
        analysis = await dashboard.create_sector_industry_analysis(pool_id)
        
        assert "error" not in analysis
        assert "sector_breakdown" in analysis
        assert "industry_breakdown" in analysis
        assert "charts" in analysis
        assert "insights" in analysis
        
        # Check charts
        charts = analysis["charts"]
        assert "sector_pie" in charts
        assert "industry_bars" in charts
        assert "sector_performance" in charts
    
    @pytest.mark.asyncio
    async def test_risk_distribution_analysis(self, dashboard, pool_manager):
        """Test risk distribution analysis"""
        # Create pools for risk analysis
        pool_id1 = await pool_manager.create_pool("Risk Pool 1", PoolType.HIGH_RISK)
        pool_id2 = await pool_manager.create_pool("Risk Pool 2", PoolType.VALUE_STOCKS)
        
        await pool_manager.add_stock_to_pool(pool_id1, "AAPL", "Apple")
        await pool_manager.add_stock_to_pool(pool_id2, "GOOGL", "Google")
        
        analysis = await dashboard.create_risk_distribution_analysis([pool_id1, pool_id2])
        
        assert "pool_risk_metrics" in analysis
        assert "charts" in analysis
        assert "risk_summary" in analysis
        assert "recommendations" in analysis
        
        # Check charts
        charts = analysis["charts"]
        assert "risk_distribution" in charts
        assert "var_analysis" in charts
        assert "concentration_risk" in charts
        assert "risk_adjusted_returns" in charts
    
    @pytest.mark.asyncio
    async def test_pool_optimization_recommendations(self, dashboard, pool_manager, sample_pool):
        """Test pool optimization recommendations"""
        pool_id = sample_pool
        
        optimization = await dashboard.create_pool_optimization_recommendations(pool_id)
        
        assert "error" not in optimization
        assert "current_analysis" in optimization
        assert "optimization_opportunities" in optimization
        assert "rebalancing_suggestions" in optimization
        assert "charts" in optimization
        assert "action_plan" in optimization
        
        # Check charts
        charts = optimization["charts"]
        assert "current_vs_optimal" in charts
        assert "rebalancing_impact" in charts
        assert "efficiency_frontier" in charts

class TestPoolExportImport:
    """Test cases for PoolExportImport"""
    
    @pytest.fixture
    def export_import(self, pool_manager):
        """Create export/import instance"""
        return PoolExportImport(pool_manager)
    
    @pytest.mark.asyncio
    async def test_export_pool_json(self, export_import, pool_manager, sample_pool):
        """Test pool export to JSON format"""
        pool_id = sample_pool
        
        result = await export_import.export_pool(
            pool_id=pool_id,
            format_type='json',
            output_path='test_pool_export.json'
        )
        
        assert result["success"] is True
        assert result["format"] == "json"
        assert "file_path" in result
        
        # Verify file exists and contains data
        import json
        from pathlib import Path
        
        file_path = Path(result["file_path"])
        assert file_path.exists()
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        assert "pool_info" in data
        assert "stocks" in data
        assert len(data["stocks"]) == 3
        
        # Cleanup
        file_path.unlink()
    
    @pytest.mark.asyncio
    async def test_export_pool_csv(self, export_import, pool_manager, sample_pool):
        """Test pool export to CSV format"""
        pool_id = sample_pool
        
        result = await export_import.export_pool(
            pool_id=pool_id,
            format_type='csv',
            output_path='test_pool_export.csv'
        )
        
        assert result["success"] is True
        assert result["format"] == "csv"
        
        # Verify CSV file
        import pandas as pd
        from pathlib import Path
        
        file_path = Path(result["file_path"])
        assert file_path.exists()
        
        df = pd.read_csv(file_path)
        assert len(df) == 3
        assert "symbol" in df.columns
        assert "name" in df.columns
        
        # Cleanup
        file_path.unlink()
    
    @pytest.mark.asyncio
    async def test_export_pool_excel(self, export_import, pool_manager, sample_pool):
        """Test pool export to Excel format"""
        pool_id = sample_pool
        
        result = await export_import.export_pool(
            pool_id=pool_id,
            format_type='excel',
            output_path='test_pool_export.xlsx'
        )
        
        assert result["success"] is True
        assert result["format"] == "excel"
        
        # Verify Excel file
        import pandas as pd
        from pathlib import Path
        
        file_path = Path(result["file_path"])
        assert file_path.exists()
        
        # Read Excel file
        excel_file = pd.ExcelFile(file_path)
        assert 'Pool Info' in excel_file.sheet_names
        assert 'Stocks' in excel_file.sheet_names
        
        stocks_df = pd.read_excel(file_path, sheet_name='Stocks')
        assert len(stocks_df) == 3
        
        # Cleanup
        file_path.unlink()
    
    @pytest.mark.asyncio
    async def test_export_multiple_pools(self, export_import, pool_manager):
        """Test exporting multiple pools"""
        # Create multiple pools
        pool_id1 = await pool_manager.create_pool("Pool 1", PoolType.WATCHLIST)
        pool_id2 = await pool_manager.create_pool("Pool 2", PoolType.CORE_HOLDINGS)
        
        await pool_manager.add_stock_to_pool(pool_id1, "AAPL", "Apple")
        await pool_manager.add_stock_to_pool(pool_id2, "GOOGL", "Google")
        
        result = await export_import.export_multiple_pools(
            pool_ids=[pool_id1, pool_id2],
            format_type='json',
            create_archive=True,
            output_path='test_pools_export.zip'
        )
        
        assert result["success"] is True
        assert result["exported_pools"] == 2
        assert "archive_path" in result
        
        # Verify archive exists
        from pathlib import Path
        archive_path = Path(result["archive_path"])
        assert archive_path.exists()
        
        # Cleanup
        archive_path.unlink()
    
    @pytest.mark.asyncio
    async def test_import_pool_json(self, export_import, pool_manager, sample_pool):
        """Test pool import from JSON format"""
        pool_id = sample_pool
        
        # First export a pool
        export_result = await export_import.export_pool(
            pool_id=pool_id,
            format_type='json',
            output_path='test_import.json'
        )
        
        # Then import it
        import_result = await export_import.import_pool(
            file_path='test_import.json',
            format_type='json'
        )
        
        assert import_result["success"] is True
        assert "pool_id" in import_result
        assert import_result["added_stocks"] == 3
        
        # Verify imported pool
        imported_pool_id = import_result["pool_id"]
        assert imported_pool_id in pool_manager.pools
        
        imported_pool = pool_manager.pools[imported_pool_id]
        assert len(imported_pool.stocks) == 3
        
        # Cleanup
        from pathlib import Path
        Path('test_import.json').unlink()
    
    @pytest.mark.asyncio
    async def test_create_pool_backup(self, export_import, pool_manager, sample_pool):
        """Test pool backup creation"""
        pool_id = sample_pool
        
        result = await export_import.create_pool_backup(
            pool_id=pool_id,
            include_full_history=True,
            backup_name="test_backup"
        )
        
        assert result["success"] is True
        assert "backup_path" in result
        assert "backup_size" in result
        
        # Verify backup file
        from pathlib import Path
        backup_path = Path(result["backup_path"])
        assert backup_path.exists()
        
        # Verify backup content
        import json
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)
        
        assert "backup_info" in backup_data
        assert "pool_data" in backup_data
        assert "system_info" in backup_data
        
        # Cleanup
        backup_path.unlink()
    
    @pytest.mark.asyncio
    async def test_restore_pool_from_backup(self, export_import, pool_manager, sample_pool):
        """Test pool restoration from backup"""
        pool_id = sample_pool
        
        # Create backup
        backup_result = await export_import.create_pool_backup(
            pool_id=pool_id,
            backup_name="test_restore_backup"
        )
        
        backup_path = backup_result["backup_path"]
        
        # Restore from backup
        restore_result = await export_import.restore_pool_from_backup(
            backup_path=backup_path,
            restore_strategy='replace'
        )
        
        assert restore_result["success"] is True
        assert "restored_pool_id" in restore_result
        
        # Verify restored pool
        restored_pool_id = restore_result["restored_pool_id"]
        assert restored_pool_id in pool_manager.pools
        
        restored_pool = pool_manager.pools[restored_pool_id]
        assert len(restored_pool.stocks) == 3
        
        # Cleanup
        from pathlib import Path
        Path(backup_path).unlink()
    
    @pytest.mark.asyncio
    async def test_create_shareable_pool_link(self, export_import, pool_manager, sample_pool):
        """Test creating shareable pool links"""
        pool_id = sample_pool
        
        result = await export_import.create_shareable_pool_link(
            pool_id=pool_id,
            access_level='read_only',
            expiry_days=30,
            include_analytics=True
        )
        
        assert result["success"] is True
        assert "share_id" in result
        assert "share_url" in result
        assert "share_token" in result
        assert result["access_level"] == "read_only"
    
    @pytest.mark.asyncio
    async def test_export_for_external_tools(self, export_import, pool_manager, sample_pool):
        """Test export for external portfolio management tools"""
        pool_id = sample_pool
        
        # Test Portfolio Visualizer export
        result = await export_import.export_for_external_tools(
            pool_id=pool_id,
            tool_type='portfolio_visualizer',
            output_path='test_pv_export.csv'
        )
        
        assert result["success"] is True
        assert result["format"] == "portfolio_visualizer_csv"
        
        # Verify file
        import pandas as pd
        from pathlib import Path
        
        file_path = Path(result["file_path"])
        assert file_path.exists()
        
        df = pd.read_csv(file_path)
        assert "Symbol" in df.columns
        assert "Weight" in df.columns
        assert len(df) == 3
        
        # Cleanup
        file_path.unlink()

class TestIntegration:
    """Integration tests for the complete pool management system"""
    
    @pytest.mark.asyncio
    async def test_complete_pool_workflow(self, pool_manager):
        """Test complete pool management workflow"""
        # Create pool
        pool_id = await pool_manager.create_pool(
            name="Integration Test Pool",
            pool_type=PoolType.GROWTH_STOCKS,
            description="Complete workflow test"
        )
        
        # Add stocks
        stocks_to_add = [
            ("AAPL", "Apple Inc.", 0.25),
            ("GOOGL", "Alphabet Inc.", 0.25),
            ("MSFT", "Microsoft Corp.", 0.25),
            ("TSLA", "Tesla Inc.", 0.25)
        ]
        
        for symbol, name, weight in stocks_to_add:
            success = await pool_manager.add_stock_to_pool(
                pool_id=pool_id,
                symbol=symbol,
                name=name,
                weight=weight
            )
            assert success is True
        
        # Get analytics
        analytics = await pool_manager.get_pool_analytics(pool_id)
        assert "error" not in analytics
        assert analytics["basic_info"]["total_stocks"] == 4
        
        # Create dashboard
        dashboard = PoolAnalyticsDashboard(pool_manager)
        dashboard_data = await dashboard.create_pool_performance_dashboard(pool_id)
        assert "error" not in dashboard_data
        
        # Export pool
        export_import = PoolExportImport(pool_manager)
        export_result = await export_import.export_pool(
            pool_id=pool_id,
            format_type='json'
        )
        assert export_result["success"] is True
        
        # Create backup
        backup_result = await export_import.create_pool_backup(pool_id)
        assert backup_result["success"] is True
        
        # Update from screening
        screening_results = [
            {"symbol": "NVDA", "name": "NVIDIA Corp."},
            {"symbol": "AMD", "name": "Advanced Micro Devices"}
        ]
        
        update_result = await pool_manager.update_pool_from_screening(
            pool_id=pool_id,
            screening_results=screening_results,
            max_additions=2
        )
        assert update_result["success"] is True
        assert len(update_result["added_stocks"]) == 2
        
        # Final verification
        final_pool = pool_manager.pools[pool_id]
        assert len(final_pool.stocks) == 6
        
        # Cleanup
        from pathlib import Path
        if Path(export_result["file_path"]).exists():
            Path(export_result["file_path"]).unlink()
        if Path(backup_result["backup_path"]).exists():
            Path(backup_result["backup_path"]).unlink()

if __name__ == "__main__":
    pytest.main([__file__])