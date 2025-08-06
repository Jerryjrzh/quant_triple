#!/usr/bin/env python3
"""
Advanced Screening Interface Demo

This script demonstrates the comprehensive stock screening system with
real-time updates, template management, and result analysis.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from stock_analysis_system.screening import (
    ScreeningInterface, ScreeningEngine, ScreeningCriteriaBuilder,
    PredefinedTemplates, TechnicalCriteria, SeasonalCriteria,
    InstitutionalCriteria, RiskCriteria
)
from stock_analysis_system.data.data_source_manager import DataSourceManager
from stock_analysis_system.analysis.spring_festival_engine import SpringFestivalAlignmentEngine
from stock_analysis_system.analysis.institutional_attention_scoring import InstitutionalAttentionScoringSystem
from stock_analysis_system.analysis.risk_management_engine import EnhancedRiskManagementEngine


class MockDataSourceManager:
    """Mock data source manager for demo purposes."""
    
    async def get_stock_basic_info(self, stock_code: str):
        """Mock stock basic info."""
        return {
            'stock_code': stock_code,
            'stock_name': f"Stock {stock_code}",
            'sector': 'Technology',
            'industry': 'Software',
            'market_cap': 1000000000
        }


class MockSpringFestivalEngine:
    """Mock Spring Festival engine for demo purposes."""
    
    def get_current_position(self, stock_code: str, current_date=None):
        """Mock Spring Festival position analysis."""
        import random
        return {
            'days_to_spring_festival': random.randint(-60, 60),
            'pattern_strength': random.uniform(0.3, 0.9),
            'confidence_level': random.uniform(0.4, 0.8)
        }


class MockInstitutionalEngine:
    """Mock institutional attention engine for demo purposes."""
    
    async def calculate_stock_attention_profile(self, stock_code: str):
        """Mock institutional attention profile."""
        import random
        from types import SimpleNamespace
        
        # Create a mock profile object
        profile = SimpleNamespace()
        profile.stock_code = stock_code
        profile.overall_attention_score = random.uniform(20, 95)
        profile.institution_scores = [
            SimpleNamespace(institution_id=f"inst_{i}", score=random.uniform(30, 90))
            for i in range(random.randint(1, 5))
        ]
        return profile


class MockRiskEngine:
    """Mock risk management engine for demo purposes."""
    
    async def calculate_comprehensive_risk_metrics(self, stock_code: str, lookback_days: int = 20):
        """Mock risk metrics."""
        import random
        return {
            'volatility': random.uniform(0.15, 0.45),
            'var_95': random.uniform(0.02, 0.08),
            'sharpe_ratio': random.uniform(-0.5, 2.0),
            'beta': random.uniform(0.5, 1.8),
            'max_drawdown': random.uniform(0.05, 0.25)
        }


async def demo_basic_screening():
    """Demonstrate basic screening functionality."""
    print("=" * 60)
    print("BASIC SCREENING DEMO")
    print("=" * 60)
    
    # Initialize components
    data_source = MockDataSourceManager()
    sf_engine = MockSpringFestivalEngine()
    inst_engine = MockInstitutionalEngine()
    risk_engine = MockRiskEngine()
    
    screening_engine = ScreeningEngine(data_source, sf_engine, inst_engine, risk_engine)
    interface = ScreeningInterface(screening_engine)
    
    print(f"✓ Initialized screening interface with {len(interface.templates)} predefined templates")
    
    # List available templates
    print("\nAvailable Templates:")
    templates = interface.get_template_list()
    for i, template in enumerate(templates, 1):
        print(f"  {i}. {template['name']}: {template['description']}")
        print(f"     Tags: {', '.join(template['tags'])}")
        print(f"     Criteria: Technical={template['has_technical']}, "
              f"Seasonal={template['has_seasonal']}, "
              f"Institutional={template['has_institutional']}, "
              f"Risk={template['has_risk']}")
    
    # Run screening with predefined template
    print(f"\n📊 Running screening with 'Growth Momentum' template...")
    start_time = datetime.now()
    
    result = await interface.run_screening(
        template_name="Growth Momentum",
        stock_universe=[f"{i:06d}" for i in range(1, 101)],  # Sample 100 stocks
        max_results=20
    )
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✓ Screening completed in {execution_time:.2f} seconds")
    print(f"  - Screened: {result.total_stocks_screened} stocks")
    print(f"  - Passed: {result.stocks_passed} stocks")
    print(f"  - Average score: {result.avg_composite_score:.1f}")
    
    # Display top results
    print(f"\n🏆 Top 10 Results:")
    top_stocks = result.get_top_stocks(10)
    
    print(f"{'Rank':<4} {'Code':<8} {'Name':<15} {'Score':<6} {'Tech':<5} {'Seas':<5} {'Inst':<5} {'Risk':<5}")
    print("-" * 70)
    
    for i, stock in enumerate(top_stocks, 1):
        print(f"{i:<4} {stock.stock_code:<8} {stock.stock_name:<15} "
              f"{stock.composite_score:<6.1f} {stock.technical_score:<5.1f} "
              f"{stock.seasonal_score:<5.1f} {stock.institutional_score:<5.1f} "
              f"{stock.risk_score:<5.1f}")
    
    return interface, result


async def demo_custom_template():
    """Demonstrate custom template creation."""
    print("\n" + "=" * 60)
    print("CUSTOM TEMPLATE DEMO")
    print("=" * 60)
    
    # Get interface from previous demo
    data_source = MockDataSourceManager()
    sf_engine = MockSpringFestivalEngine()
    inst_engine = MockInstitutionalEngine()
    risk_engine = MockRiskEngine()
    
    screening_engine = ScreeningEngine(data_source, sf_engine, inst_engine, risk_engine)
    interface = ScreeningInterface(screening_engine)
    
    # Create custom template
    print("🔧 Creating custom template: 'Conservative Growth'...")
    
    template_name = await interface.create_custom_template(
        name="Conservative Growth",
        description="Conservative stocks with steady growth and low risk",
        technical_params={
            'price_change_pct_min': 1.0,
            'price_change_pct_max': 8.0,
            'rsi_min': 40.0,
            'rsi_max': 70.0,
            'ma20_position': 'above'
        },
        seasonal_params={
            'spring_festival_pattern_strength': 0.5,
            'pattern_confidence_min': 0.6
        },
        institutional_params={
            'attention_score_min': 50.0,
            'mutual_fund_activity': True
        },
        risk_params={
            'volatility_max': 0.3,
            'sharpe_ratio_min': 0.4,
            'max_drawdown_max': 0.2
        },
        tags=['conservative', 'growth', 'low_risk']
    )
    
    print(f"✓ Created template: {template_name}")
    
    # Get template details
    template_details = interface.get_template_details(template_name)
    print(f"\nTemplate Details:")
    print(f"  Name: {template_details['name']}")
    print(f"  Description: {template_details['description']}")
    print(f"  Tags: {', '.join(template_details['tags'])}")
    
    # Run screening with custom template
    print(f"\n📊 Running screening with custom template...")
    
    result = await interface.run_screening(
        template_name=template_name,
        stock_universe=[f"{i:06d}" for i in range(1, 51)],  # Sample 50 stocks
        max_results=15
    )
    
    print(f"✓ Custom screening completed")
    print(f"  - Found {result.stocks_passed} stocks matching criteria")
    print(f"  - Average composite score: {result.avg_composite_score:.1f}")
    
    # Show score distribution
    print(f"\n📈 Score Distribution:")
    for category, count in result.score_distribution.items():
        print(f"  {category.capitalize()}: {count} stocks")
    
    return interface


async def demo_real_time_screening():
    """Demonstrate real-time screening capabilities."""
    print("\n" + "=" * 60)
    print("REAL-TIME SCREENING DEMO")
    print("=" * 60)
    
    # Get interface
    data_source = MockDataSourceManager()
    sf_engine = MockSpringFestivalEngine()
    inst_engine = MockInstitutionalEngine()
    risk_engine = MockRiskEngine()
    
    screening_engine = ScreeningEngine(data_source, sf_engine, inst_engine, risk_engine)
    interface = ScreeningInterface(screening_engine)
    
    # Real-time callback function
    update_count = 0
    
    async def screening_callback(session_id: str, result):
        nonlocal update_count
        update_count += 1
        print(f"📡 Real-time update #{update_count} (Session: {session_id[:8]}...)")
        print(f"   Found {result.stocks_passed} stocks, avg score: {result.avg_composite_score:.1f}")
        
        if update_count >= 3:  # Stop after 3 updates for demo
            await interface.stop_real_time_screening(session_id)
            print(f"🛑 Stopped real-time screening session")
    
    # Start real-time screening
    print("🚀 Starting real-time screening with 'Institutional Following' template...")
    print("   Update interval: 5 seconds (demo purposes)")
    
    session_id = await interface.start_real_time_screening(
        template_name="Institutional Following",
        update_interval_seconds=5,  # Short interval for demo
        callback=screening_callback
    )
    
    print(f"✓ Started real-time session: {session_id[:8]}...")
    
    # Wait for updates
    print("⏳ Waiting for real-time updates...")
    await asyncio.sleep(20)  # Wait for a few updates
    
    # Check session status
    sessions = interface.get_real_time_sessions()
    print(f"\n📊 Real-time Sessions Status:")
    for session in sessions:
        print(f"  Session: {session['session_id'][:8]}...")
        print(f"  Template: {session['template_name']}")
        print(f"  Updates: {session['update_count']}")
        print(f"  Active: {session['active']}")


async def demo_template_management():
    """Demonstrate template management features."""
    print("\n" + "=" * 60)
    print("TEMPLATE MANAGEMENT DEMO")
    print("=" * 60)
    
    # Get interface
    data_source = MockDataSourceManager()
    sf_engine = MockSpringFestivalEngine()
    inst_engine = MockInstitutionalEngine()
    risk_engine = MockRiskEngine()
    
    screening_engine = ScreeningEngine(data_source, sf_engine, inst_engine, risk_engine)
    interface = ScreeningInterface(screening_engine)
    
    # Create a test template
    builder = ScreeningCriteriaBuilder()
    template = builder.with_technical_criteria(
        rsi_min=30.0,
        rsi_max=70.0,
        ma20_position='above'
    ).with_risk_criteria(
        volatility_max=0.35,
        sharpe_ratio_min=0.3
    ).build_template(
        name="Test Export Template",
        description="Template for testing export/import functionality",
        tags=['test', 'export']
    )
    
    interface.templates[template.name] = template
    
    # Export template
    print("📤 Exporting template to JSON...")
    exported_json = await interface.export_template("Test Export Template")
    print(f"✓ Template exported ({len(exported_json)} characters)")
    
    # Delete template
    print("\n🗑️ Deleting template...")
    deleted = await interface.delete_template("Test Export Template")
    print(f"✓ Template deleted: {deleted}")
    
    # Import template back
    print("\n📥 Importing template from JSON...")
    imported_name = await interface.import_template(exported_json)
    print(f"✓ Template imported: {imported_name}")
    
    # Verify import
    template_details = interface.get_template_details(imported_name)
    if template_details:
        print(f"✓ Import verified - Template has {len(template_details['tags'])} tags")
    
    # Update template
    print(f"\n✏️ Updating template description...")
    updated = await interface.update_template(
        imported_name,
        description="Updated description after import"
    )
    print(f"✓ Template updated: {updated}")


async def demo_result_analysis():
    """Demonstrate comprehensive result analysis."""
    print("\n" + "=" * 60)
    print("RESULT ANALYSIS DEMO")
    print("=" * 60)
    
    # Get interface and run screening
    data_source = MockDataSourceManager()
    sf_engine = MockSpringFestivalEngine()
    inst_engine = MockInstitutionalEngine()
    risk_engine = MockRiskEngine()
    
    screening_engine = ScreeningEngine(data_source, sf_engine, inst_engine, risk_engine)
    interface = ScreeningInterface(screening_engine)
    
    # Run screening
    print("📊 Running screening for analysis...")
    result = await interface.run_screening(
        template_name="Growth Momentum",
        stock_universe=[f"{i:06d}" for i in range(1, 201)],  # Larger sample
        max_results=50
    )
    
    # Analyze results
    print("🔍 Performing comprehensive analysis...")
    analysis = await interface.analyze_screening_result(result)
    
    # Display analysis
    exec_summary = analysis['execution_summary']
    print(f"\n📈 Execution Summary:")
    print(f"  Template: {exec_summary['template_name']}")
    print(f"  Duration: {exec_summary['duration_ms']}ms")
    print(f"  Pass Rate: {exec_summary['pass_rate']:.1f}%")
    
    score_dist = analysis['score_distribution']
    print(f"\n📊 Score Distribution:")
    print(f"  Mean: {score_dist['mean']:.1f}")
    print(f"  Median: {score_dist['median']:.1f}")
    print(f"  Std Dev: {score_dist['std']:.1f}")
    print(f"  Range: {score_dist['min']:.1f} - {score_dist['max']:.1f}")
    
    print(f"\n🏆 Score Ranges:")
    for range_name, count in score_dist['score_ranges'].items():
        print(f"  {range_name.capitalize()}: {count} stocks")
    
    # Sector analysis
    if 'sector_breakdown' in analysis:
        print(f"\n🏢 Sector Breakdown:")
        for sector, data in analysis['sector_breakdown'].items():
            print(f"  {sector}: {data['stock_count']} stocks "
                  f"(avg score: {data['avg_score']:.1f})")
    
    # Criteria effectiveness
    criteria_eff = analysis['criteria_effectiveness']
    print(f"\n⚡ Criteria Effectiveness:")
    for criteria_type, data in criteria_eff.items():
        if data['stocks_with_score'] > 0:
            print(f"  {criteria_type.replace('_', ' ').title()}: "
                  f"{data['effectiveness']:.1%} effective "
                  f"({data['stocks_with_score']} stocks)")


async def demo_screening_history():
    """Demonstrate screening history and performance tracking."""
    print("\n" + "=" * 60)
    print("SCREENING HISTORY DEMO")
    print("=" * 60)
    
    # Get interface
    data_source = MockDataSourceManager()
    sf_engine = MockSpringFestivalEngine()
    inst_engine = MockInstitutionalEngine()
    risk_engine = MockRiskEngine()
    
    screening_engine = ScreeningEngine(data_source, sf_engine, inst_engine, risk_engine)
    interface = ScreeningInterface(screening_engine)
    
    # Run multiple screenings
    templates = ["Growth Momentum", "Low Risk Value", "Institutional Following"]
    
    print("🔄 Running multiple screenings for history...")
    for template in templates:
        print(f"  Running {template}...")
        await interface.run_screening(
            template_name=template,
            stock_universe=[f"{i:06d}" for i in range(1, 51)],
            max_results=20
        )
        await asyncio.sleep(0.5)  # Small delay between screenings
    
    # Get screening history
    history = await interface.get_screening_history(limit=10)
    
    print(f"\n📚 Screening History ({len(history)} entries):")
    print(f"{'Template':<25} {'Duration':<10} {'Found':<6} {'Success':<8} {'Time'}")
    print("-" * 70)
    
    for entry in history:
        duration = f"{entry['duration_ms']}ms"
        timestamp = datetime.fromisoformat(entry['execution_time']).strftime("%H:%M:%S")
        success = "✓" if entry['success'] else "✗"
        
        print(f"{entry['template_name']:<25} {duration:<10} "
              f"{entry['stocks_found']:<6} {success:<8} {timestamp}")
    
    # Cache statistics
    cache_stats = await interface.get_cache_stats()
    print(f"\n💾 Cache Statistics:")
    print(f"  Total entries: {cache_stats['total_entries']}")
    print(f"  Valid entries: {cache_stats['valid_entries']}")
    print(f"  Expired entries: {cache_stats['expired_entries']}")
    print(f"  TTL: {cache_stats['cache_ttl_minutes']} minutes")


async def main():
    """Run all screening demos."""
    print("🚀 Advanced Stock Screening System Demo")
    print("=" * 60)
    
    try:
        # Run all demos
        await demo_basic_screening()
        await demo_custom_template()
        await demo_real_time_screening()
        await demo_template_management()
        await demo_result_analysis()
        await demo_screening_history()
        
        print("\n" + "=" * 60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n📋 Demo Summary:")
        print("  ✓ Basic screening with predefined templates")
        print("  ✓ Custom template creation and management")
        print("  ✓ Real-time screening with live updates")
        print("  ✓ Template export/import functionality")
        print("  ✓ Comprehensive result analysis")
        print("  ✓ Screening history and performance tracking")
        
        print(f"\n🎯 Key Features Demonstrated:")
        print("  • Multi-dimensional screening (Technical, Seasonal, Institutional, Risk)")
        print("  • Real-time screening with callbacks")
        print("  • Template management and persistence")
        print("  • Advanced result analysis and visualization")
        print("  • Performance optimization with caching")
        print("  • Comprehensive scoring and ranking system")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)