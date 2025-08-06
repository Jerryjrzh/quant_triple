#!/usr/bin/env python3
"""
Stock Pool Management System Demo

This script demonstrates the comprehensive stock pool management capabilities
including pool creation, analytics, visualization, and export/import functionality.
"""

import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Import the pool management system
from stock_analysis_system.pool import (
    StockPoolManager,
    PoolType,
    PoolAnalyticsDashboard,
    PoolExportImport
)

async def demo_basic_pool_operations():
    """Demonstrate basic pool operations"""
    print("=" * 60)
    print("STOCK POOL MANAGEMENT SYSTEM DEMO")
    print("=" * 60)
    
    # Initialize pool manager
    pool_manager = StockPoolManager()
    
    print("\n1. Creating Stock Pools")
    print("-" * 30)
    
    # Create different types of pools
    watchlist_id = await pool_manager.create_pool(
        name="Tech Watchlist",
        pool_type=PoolType.WATCHLIST,
        description="Technology stocks to monitor",
        max_stocks=20
    )
    print(f"✓ Created watchlist pool: {watchlist_id}")
    
    core_holdings_id = await pool_manager.create_pool(
        name="Core Holdings",
        pool_type=PoolType.CORE_HOLDINGS,
        description="Long-term investment positions",
        max_stocks=15
    )
    print(f"✓ Created core holdings pool: {core_holdings_id}")
    
    growth_stocks_id = await pool_manager.create_pool(
        name="Growth Opportunities",
        pool_type=PoolType.GROWTH_STOCKS,
        description="High-growth potential stocks",
        max_stocks=25
    )
    print(f"✓ Created growth stocks pool: {growth_stocks_id}")
    
    print(f"\nTotal pools created: {len(pool_manager.pools)}")
    
    return pool_manager, watchlist_id, core_holdings_id, growth_stocks_id

async def demo_stock_management(pool_manager, watchlist_id, core_holdings_id, growth_stocks_id):
    """Demonstrate stock management within pools"""
    print("\n2. Adding Stocks to Pools")
    print("-" * 30)
    
    # Add stocks to watchlist
    watchlist_stocks = [
        ("AAPL", "Apple Inc.", 0.20, "iPhone maker, strong ecosystem"),
        ("GOOGL", "Alphabet Inc.", 0.18, "Search and cloud dominance"),
        ("MSFT", "Microsoft Corp.", 0.15, "Enterprise software leader"),
        ("TSLA", "Tesla Inc.", 0.12, "EV and energy innovation"),
        ("NVDA", "NVIDIA Corp.", 0.10, "AI and GPU technology"),
        ("META", "Meta Platforms", 0.08, "Social media and metaverse"),
        ("AMZN", "Amazon.com Inc.", 0.17, "E-commerce and AWS")
    ]
    
    for symbol, name, weight, notes in watchlist_stocks:
        success = await pool_manager.add_stock_to_pool(
            pool_id=watchlist_id,
            symbol=symbol,
            name=name,
            weight=weight,
            notes=notes,
            tags=["tech", "large-cap"]
        )
        if success:
            print(f"  ✓ Added {symbol} ({name}) to watchlist")
    
    # Add stocks to core holdings
    core_stocks = [
        ("AAPL", "Apple Inc.", 0.25, "Core technology position"),
        ("MSFT", "Microsoft Corp.", 0.20, "Stable dividend growth"),
        ("JNJ", "Johnson & Johnson", 0.15, "Healthcare defensive"),
        ("PG", "Procter & Gamble", 0.15, "Consumer staples"),
        ("KO", "Coca-Cola Co.", 0.10, "Dividend aristocrat"),
        ("VTI", "Vanguard Total Stock", 0.15, "Market diversification")
    ]
    
    for symbol, name, weight, notes in core_stocks:
        success = await pool_manager.add_stock_to_pool(
            pool_id=core_holdings_id,
            symbol=symbol,
            name=name,
            weight=weight,
            notes=notes,
            tags=["core", "dividend"]
        )
        if success:
            print(f"  ✓ Added {symbol} ({name}) to core holdings")
    
    # Add stocks to growth pool
    growth_stocks = [
        ("TSLA", "Tesla Inc.", 0.20, "EV market leader"),
        ("NVDA", "NVIDIA Corp.", 0.18, "AI revolution beneficiary"),
        ("AMD", "Advanced Micro Devices", 0.15, "CPU/GPU competition"),
        ("PLTR", "Palantir Technologies", 0.12, "Big data analytics"),
        ("ROKU", "Roku Inc.", 0.10, "Streaming platform"),
        ("SQ", "Block Inc.", 0.10, "Fintech innovation"),
        ("SHOP", "Shopify Inc.", 0.15, "E-commerce platform")
    ]
    
    for symbol, name, weight, notes in growth_stocks:
        success = await pool_manager.add_stock_to_pool(
            pool_id=growth_stocks_id,
            symbol=symbol,
            name=name,
            weight=weight,
            notes=notes,
            tags=["growth", "high-risk"]
        )
        if success:
            print(f"  ✓ Added {symbol} ({name}) to growth pool")
    
    # Display pool summaries
    print(f"\nPool Summaries:")
    for pool_id, pool in pool_manager.pools.items():
        print(f"  {pool.name}: {len(pool.stocks)} stocks")

async def demo_pool_analytics(pool_manager, pool_ids):
    """Demonstrate pool analytics capabilities"""
    print("\n3. Pool Analytics and Performance")
    print("-" * 30)
    
    dashboard = PoolAnalyticsDashboard(pool_manager)
    
    # Analyze each pool
    for pool_id in pool_ids:
        pool = pool_manager.pools[pool_id]
        print(f"\nAnalyzing {pool.name}:")
        
        analytics = await pool_manager.get_pool_analytics(pool_id)
        
        if "error" not in analytics:
            basic_info = analytics["basic_info"]
            metrics = analytics["performance_metrics"]
            
            print(f"  Total Stocks: {basic_info['total_stocks']}")
            print(f"  Total Return: {metrics['total_return']:.2%}")
            print(f"  Volatility: {metrics['volatility']:.2%}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"  Win Rate: {metrics['win_rate']:.2%}")
            
            # Show top performers
            stock_analysis = analytics["stock_analysis"]
            if stock_analysis["top_performers"]:
                print("  Top Performers:")
                for stock in stock_analysis["top_performers"][:3]:
                    print(f"    {stock['symbol']}: {stock['return']:.2%}")
    
    # Create comprehensive dashboard for watchlist
    watchlist_id = pool_ids[0]
    print(f"\nCreating comprehensive dashboard for {pool_manager.pools[watchlist_id].name}...")
    
    dashboard_data = await dashboard.create_pool_performance_dashboard(watchlist_id)
    
    if "error" not in dashboard_data:
        print("  ✓ Performance overview chart created")
        print("  ✓ Stock breakdown chart created")
        print("  ✓ Sector distribution chart created")
        print("  ✓ Risk analysis chart created")
        print("  ✓ Performance timeline chart created")
    
    # Multi-pool comparison
    print(f"\nComparing all pools...")
    comparison = await pool_manager.compare_pools(pool_ids)
    
    if "error" not in comparison:
        rankings = comparison["rankings"]
        print("  Pool Rankings by Return:")
        for i, (pool_id, return_val) in enumerate(rankings["by_return"], 1):
            pool_name = pool_manager.pools[pool_id].name
            print(f"    {i}. {pool_name}: {return_val:.2%}")

async def demo_screening_integration(pool_manager, watchlist_id):
    """Demonstrate screening integration"""
    print("\n4. Screening Integration")
    print("-" * 30)
    
    # Simulate screening results
    screening_results = [
        {"symbol": "CRM", "name": "Salesforce Inc.", "score": 85},
        {"symbol": "ADBE", "name": "Adobe Inc.", "score": 82},
        {"symbol": "NOW", "name": "ServiceNow Inc.", "score": 80},
        {"symbol": "SNOW", "name": "Snowflake Inc.", "score": 78},
        {"symbol": "ZM", "name": "Zoom Video", "score": 75}
    ]
    
    print("Screening results received:")
    for result in screening_results:
        print(f"  {result['symbol']} ({result['name']}): Score {result['score']}")
    
    # Update pool from screening
    update_result = await pool_manager.update_pool_from_screening(
        pool_id=watchlist_id,
        screening_results=screening_results,
        max_additions=3,
        replace_existing=False
    )
    
    if update_result["success"]:
        print(f"\n✓ Added {len(update_result['added_stocks'])} stocks from screening:")
        for symbol in update_result["added_stocks"]:
            print(f"  - {symbol}")
        print(f"Total stocks in watchlist: {update_result['total_stocks']}")

async def demo_auto_update_rules(pool_manager, growth_stocks_id):
    """Demonstrate automatic update rules"""
    print("\n5. Automatic Update Rules")
    print("-" * 30)
    
    # Set up auto-update rules
    auto_rules = {
        "enabled": True,
        "interval_hours": 24,
        "remove_poor_performers": True,
        "poor_performance_threshold": -0.25,  # Remove stocks with >25% loss
        "rebalance_weights": True,
        "max_position_size": 0.20  # No single stock >20%
    }
    
    success = await pool_manager.set_auto_update_rules(growth_stocks_id, auto_rules)
    
    if success:
        print("✓ Auto-update rules configured:")
        print(f"  - Check interval: {auto_rules['interval_hours']} hours")
        print(f"  - Remove poor performers: {auto_rules['remove_poor_performers']}")
        print(f"  - Performance threshold: {auto_rules['poor_performance_threshold']:.1%}")
        print(f"  - Rebalance weights: {auto_rules['rebalance_weights']}")
        print(f"  - Max position size: {auto_rules['max_position_size']:.1%}")

async def demo_export_import(pool_manager, pool_ids):
    """Demonstrate export/import functionality"""
    print("\n6. Export/Import Functionality")
    print("-" * 30)
    
    export_import = PoolExportImport(pool_manager)
    
    # Export single pool to different formats
    watchlist_id = pool_ids[0]
    pool_name = pool_manager.pools[watchlist_id].name
    
    print(f"Exporting '{pool_name}' to multiple formats:")
    
    # JSON export
    json_result = await export_import.export_pool(
        pool_id=watchlist_id,
        format_type='json',
        include_analytics=True,
        output_path=f"demo_export_{pool_name.replace(' ', '_')}.json"
    )
    
    if json_result["success"]:
        print(f"  ✓ JSON export: {json_result['file_path']}")
    
    # CSV export
    csv_result = await export_import.export_pool(
        pool_id=watchlist_id,
        format_type='csv',
        output_path=f"demo_export_{pool_name.replace(' ', '_')}.csv"
    )
    
    if csv_result["success"]:
        print(f"  ✓ CSV export: {csv_result['file_path']}")
    
    # Excel export
    excel_result = await export_import.export_pool(
        pool_id=watchlist_id,
        format_type='excel',
        output_path=f"demo_export_{pool_name.replace(' ', '_')}.xlsx"
    )
    
    if excel_result["success"]:
        print(f"  ✓ Excel export: {excel_result['file_path']}")
    
    # Export all pools as archive
    print(f"\nExporting all pools as archive:")
    archive_result = await export_import.export_multiple_pools(
        pool_ids=pool_ids,
        format_type='json',
        create_archive=True,
        output_path="demo_all_pools_export.zip"
    )
    
    if archive_result["success"]:
        print(f"  ✓ Archive created: {archive_result['archive_path']}")
        print(f"  ✓ Exported {archive_result['exported_pools']} pools")
    
    # Create backup
    print(f"\nCreating backup for '{pool_name}':")
    backup_result = await export_import.create_pool_backup(
        pool_id=watchlist_id,
        include_full_history=True,
        backup_name=f"demo_backup_{pool_name.replace(' ', '_')}"
    )
    
    if backup_result["success"]:
        print(f"  ✓ Backup created: {backup_result['backup_path']}")
        print(f"  ✓ Backup size: {backup_result['backup_size']} bytes")
    
    # Create shareable link
    print(f"\nCreating shareable link for '{pool_name}':")
    share_result = await export_import.create_shareable_pool_link(
        pool_id=watchlist_id,
        access_level='read_only',
        expiry_days=30,
        include_analytics=True
    )
    
    if share_result["success"]:
        print(f"  ✓ Share URL: {share_result['share_url']}")
        print(f"  ✓ Access level: {share_result['access_level']}")
        print(f"  ✓ Expires: {share_result['expiry_date']}")
    
    # Export for external tools
    print(f"\nExporting for external portfolio tools:")
    
    # Portfolio Visualizer format
    pv_result = await export_import.export_for_external_tools(
        pool_id=watchlist_id,
        tool_type='portfolio_visualizer',
        output_path="demo_portfolio_visualizer.csv"
    )
    
    if pv_result["success"]:
        print(f"  ✓ Portfolio Visualizer: {pv_result['file_path']}")
    
    # Morningstar format
    ms_result = await export_import.export_for_external_tools(
        pool_id=watchlist_id,
        tool_type='morningstar',
        output_path="demo_morningstar.csv"
    )
    
    if ms_result["success"]:
        print(f"  ✓ Morningstar: {ms_result['file_path']}")
    
    return [json_result, csv_result, excel_result, archive_result, backup_result]

async def demo_advanced_analytics(pool_manager, pool_ids):
    """Demonstrate advanced analytics features"""
    print("\n7. Advanced Analytics")
    print("-" * 30)
    
    dashboard = PoolAnalyticsDashboard(pool_manager)
    
    # Sector analysis
    watchlist_id = pool_ids[0]
    print(f"Sector analysis for {pool_manager.pools[watchlist_id].name}:")
    
    sector_analysis = await dashboard.create_sector_industry_analysis(watchlist_id)
    
    if "error" not in sector_analysis:
        sector_breakdown = sector_analysis["sector_breakdown"]
        print(f"  Diversification score: {sector_breakdown['diversification_score']:.1f}%")
        
        if sector_breakdown["sector_distribution"]:
            print("  Sector allocation:")
            for sector, weight in sector_breakdown["sector_distribution"].items():
                print(f"    {sector}: {weight:.1%}")
    
    # Risk analysis across all pools
    print(f"\nRisk analysis across all pools:")
    risk_analysis = await dashboard.create_risk_distribution_analysis(pool_ids)
    
    if "pool_risk_metrics" in risk_analysis:
        print("  Risk scores by pool:")
        for pool_id in pool_ids:
            pool_name = pool_manager.pools[pool_id].name
            if pool_id in risk_analysis["pool_risk_metrics"]:
                risk_score = risk_analysis["pool_risk_metrics"][pool_id].get("risk_score", 0)
                print(f"    {pool_name}: {risk_score:.1f}/100")
    
    # Optimization recommendations
    print(f"\nOptimization recommendations for {pool_manager.pools[watchlist_id].name}:")
    optimization = await dashboard.create_pool_optimization_recommendations(watchlist_id)
    
    if "optimization_opportunities" in optimization:
        for opportunity in optimization["optimization_opportunities"]:
            print(f"  {opportunity['type'].upper()}: {opportunity['description']}")
            print(f"    Priority: {opportunity['priority']}, Impact: {opportunity['impact']}")

async def demo_pool_history_and_restore(pool_manager, watchlist_id):
    """Demonstrate pool history tracking and restore functionality"""
    print("\n8. Pool History and Restore")
    print("-" * 30)
    
    # Show pool history
    history = await pool_manager.get_pool_history(watchlist_id, limit=10)
    
    print(f"Recent pool history (last {len(history)} actions):")
    for entry in history[-5:]:  # Show last 5 actions
        timestamp = entry["timestamp"]
        action = entry["action"]
        details = entry.get("details", {})
        print(f"  {timestamp}: {action}")
        if "symbol" in details:
            print(f"    Symbol: {details['symbol']}")
    
    # Demonstrate pool state management
    print(f"\nPool state management:")
    pool = pool_manager.pools[watchlist_id]
    print(f"  Current stocks: {len(pool.stocks)}")
    print(f"  Last modified: {pool.last_modified}")
    print(f"  Status: {pool.status.value}")

def display_summary(pool_manager, export_results):
    """Display demo summary"""
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    
    print(f"\nPools Created: {len(pool_manager.pools)}")
    for pool_id, pool in pool_manager.pools.items():
        print(f"  {pool.name} ({pool.pool_type.value}): {len(pool.stocks)} stocks")
    
    print(f"\nFiles Generated:")
    for result in export_results:
        if result and result.get("success"):
            if "file_path" in result:
                print(f"  ✓ {result['file_path']}")
            elif "archive_path" in result:
                print(f"  ✓ {result['archive_path']}")
            elif "backup_path" in result:
                print(f"  ✓ {result['backup_path']}")
    
    print(f"\nKey Features Demonstrated:")
    print("  ✓ Multiple pool types and management")
    print("  ✓ Stock addition, removal, and weighting")
    print("  ✓ Comprehensive analytics and performance metrics")
    print("  ✓ Interactive dashboards and visualizations")
    print("  ✓ Screening integration and automated updates")
    print("  ✓ Export/import in multiple formats")
    print("  ✓ Backup and restore capabilities")
    print("  ✓ Shareable links and collaboration")
    print("  ✓ External tool integration")
    print("  ✓ Risk analysis and optimization recommendations")
    print("  ✓ Pool history tracking")
    
    print(f"\nNext Steps:")
    print("  - Integrate with real market data sources")
    print("  - Connect to screening engines")
    print("  - Set up automated monitoring")
    print("  - Configure real-time updates")
    print("  - Deploy web interface")

async def cleanup_demo_files():
    """Clean up demo files"""
    print(f"\nCleaning up demo files...")
    
    demo_files = [
        "demo_export_Tech_Watchlist.json",
        "demo_export_Tech_Watchlist.csv",
        "demo_export_Tech_Watchlist.xlsx",
        "demo_all_pools_export.zip",
        "demo_portfolio_visualizer.csv",
        "demo_morningstar.csv"
    ]
    
    # Also check for backup files
    backup_dir = Path("pool_backups")
    if backup_dir.exists():
        for backup_file in backup_dir.glob("demo_backup_*.json"):
            demo_files.append(str(backup_file))
    
    cleaned_count = 0
    for file_path in demo_files:
        path = Path(file_path)
        if path.exists():
            try:
                path.unlink()
                cleaned_count += 1
                print(f"  ✓ Removed {file_path}")
            except Exception as e:
                print(f"  ✗ Failed to remove {file_path}: {e}")
    
    print(f"Cleaned up {cleaned_count} demo files")

async def main():
    """Main demo function"""
    try:
        # Run the complete demo
        pool_manager, watchlist_id, core_holdings_id, growth_stocks_id = await demo_basic_pool_operations()
        pool_ids = [watchlist_id, core_holdings_id, growth_stocks_id]
        
        await demo_stock_management(pool_manager, watchlist_id, core_holdings_id, growth_stocks_id)
        await demo_pool_analytics(pool_manager, pool_ids)
        await demo_screening_integration(pool_manager, watchlist_id)
        await demo_auto_update_rules(pool_manager, growth_stocks_id)
        
        export_results = await demo_export_import(pool_manager, pool_ids)
        await demo_advanced_analytics(pool_manager, pool_ids)
        await demo_pool_history_and_restore(pool_manager, watchlist_id)
        
        display_summary(pool_manager, export_results)
        
        # Ask user if they want to keep demo files
        try:
            keep_files = input(f"\nKeep demo files? (y/N): ").lower().strip()
            if keep_files != 'y':
                await cleanup_demo_files()
        except KeyboardInterrupt:
            print(f"\nDemo interrupted by user")
        
    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nDemo completed!")

if __name__ == "__main__":
    asyncio.run(main())