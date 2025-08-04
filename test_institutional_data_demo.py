"""
Demo script for Institutional Data Collector

This script demonstrates dragon-tiger list collection, shareholder data collection,
block trades, and institutional classification implemented in task 6.1.
"""

import asyncio
import pandas as pd
from datetime import date, datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_analysis_system.analysis.institutional_data_collector import (
    InstitutionalDataCollector,
    InstitutionClassifier,
    InstitutionType,
    ActivityType
)


async def demonstrate_institution_classification():
    """Demonstrate institutional investor classification."""
    
    print("=== Institutional Investor Classification Demo ===\n")
    
    classifier = InstitutionClassifier()
    
    # Test cases for different institution types
    test_institutions = [
        # Mutual Funds
        ("易方达基金管理有限公司", "Major mutual fund company"),
        ("华夏基金管理有限公司", "Leading fund management company"),
        ("嘉实基金管理有限公司", "Well-known asset manager"),
        ("某某资产管理有限公司", "Generic asset management company"),
        
        # Social Security
        ("全国社会保障基金理事会", "National Social Security Fund"),
        ("社保基金一零一组合", "Social Security Fund Portfolio"),
        
        # QFII
        ("摩根士丹利QFII", "Morgan Stanley QFII"),
        ("高盛集团", "Goldman Sachs Group"),
        ("瑞银集团", "UBS Group"),
        
        # Insurance
        ("中国人寿保险股份有限公司", "China Life Insurance"),
        ("中国平安人寿保险股份有限公司", "Ping An Life Insurance"),
        ("太平洋保险", "Pacific Insurance"),
        
        # Securities Firms
        ("中信证券股份有限公司", "CITIC Securities"),
        ("华泰证券股份有限公司", "Huatai Securities"),
        ("国泰君安证券股份有限公司", "Guotai Junan Securities"),
        
        # Banks
        ("中国工商银行股份有限公司", "Industrial and Commercial Bank of China"),
        ("中国建设银行股份有限公司", "China Construction Bank"),
        
        # Hot Money
        ("某某游资", "Speculative capital"),
        ("热钱机构", "Hot money institution"),
        ("个人投资者", "Individual investor"),
        
        # Unknown
        ("神秘机构XYZ", "Unknown institution"),
        ("Random Company Ltd", "Foreign unknown company")
    ]
    
    print("Classification Results:")
    print(f"{'Institution Name':<40} {'Type':<20} {'Confidence':<12} {'Description'}")
    print("-" * 90)
    
    type_counts = {}
    
    for name, description in test_institutions:
        institution_type, confidence = classifier.classify_institution(name)
        
        # Count by type
        type_counts[institution_type] = type_counts.get(institution_type, 0) + 1
        
        print(f"{name:<40} {institution_type.value:<20} {confidence:<12.1%} {description}")
    
    print(f"\n=== Classification Summary ===")
    print(f"Total institutions classified: {len(test_institutions)}")
    
    for inst_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{inst_type.value.replace('_', ' ').title()}: {count}")
    
    # Demonstrate institution object creation
    print(f"\n=== Institution Object Creation ===")
    
    sample_institution = classifier.create_institution("华夏基金管理有限公司")
    
    print(f"Institution ID: {sample_institution.institution_id}")
    print(f"Name: {sample_institution.name}")
    print(f"Type: {sample_institution.institution_type.value}")
    print(f"Confidence: {sample_institution.confidence_score:.1%}")
    print(f"First Seen: {sample_institution.first_seen}")
    print(f"Name Patterns: {sample_institution.name_patterns}")


async def demonstrate_data_collection():
    """Demonstrate comprehensive institutional data collection."""
    
    print("\n\n=== Institutional Data Collection Demo ===\n")
    
    # Initialize the main collector
    collector = InstitutionalDataCollector()
    
    # Define test parameters
    stock_codes = ["000001", "000002", "600000"]  # Mix of SZ and SH stocks
    start_date = date(2023, 1, 1)
    end_date = date(2023, 3, 31)  # Q1 2023
    
    print(f"Collecting institutional data for:")
    print(f"  Stock Codes: {', '.join(stock_codes)}")
    print(f"  Date Range: {start_date} to {end_date}")
    print(f"  Duration: {(end_date - start_date).days} days")
    
    print(f"\nStarting data collection...")
    
    try:
        # Collect all institutional data
        all_data = await collector.collect_all_data(stock_codes, start_date, end_date)
        
        print(f"✓ Data collection completed successfully!")
        
        # Display collection summary
        print(f"\n=== Collection Summary ===")
        
        total_dragon_tiger = 0
        total_shareholders = 0
        total_block_trades = 0
        total_activities = 0
        
        for stock_code, data in all_data.items():
            dt_count = len(data['dragon_tiger'])
            sh_count = len(data['shareholders'])
            bt_count = len(data['block_trades'])
            
            total_dragon_tiger += dt_count
            total_shareholders += sh_count
            total_block_trades += bt_count
            
            activities = collector.get_institution_activity_timeline(stock_code)
            activity_count = len(activities)
            total_activities += activity_count
            
            print(f"\n{stock_code}:")
            print(f"  Dragon-Tiger Records: {dt_count}")
            print(f"  Shareholder Records: {sh_count}")
            print(f"  Block Trade Records: {bt_count}")
            print(f"  Total Activities: {activity_count}")
        
        print(f"\nOverall Totals:")
        print(f"  Dragon-Tiger Records: {total_dragon_tiger}")
        print(f"  Shareholder Records: {total_shareholders}")
        print(f"  Block Trade Records: {total_block_trades}")
        print(f"  Total Activities: {total_activities}")
        
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        return None, None
    
    return collector, all_data


async def demonstrate_dragon_tiger_analysis(collector, all_data):
    """Demonstrate Dragon-Tiger list analysis."""
    
    print(f"\n\n=== Dragon-Tiger List Analysis ===\n")
    
    if not all_data:
        print("No data available for analysis")
        return
    
    # Analyze Dragon-Tiger data for first stock
    stock_code = list(all_data.keys())[0]
    dragon_tiger_data = all_data[stock_code]['dragon_tiger']
    
    if not dragon_tiger_data:
        print(f"No Dragon-Tiger data available for {stock_code}")
        return
    
    print(f"Analyzing Dragon-Tiger data for {stock_code}")
    print(f"Total Dragon-Tiger records: {len(dragon_tiger_data)}")
    
    # Analyze by seat type
    buy_records = [r for r in dragon_tiger_data if r.seat_type == "buy"]
    sell_records = [r for r in dragon_tiger_data if r.seat_type == "sell"]
    
    print(f"\nTrading Direction Analysis:")
    print(f"  Buy Records: {len(buy_records)}")
    print(f"  Sell Records: {len(sell_records)}")
    
    if buy_records:
        total_buy_amount = sum(r.amount for r in buy_records)
        avg_buy_amount = total_buy_amount / len(buy_records)
        print(f"  Total Buy Amount: ¥{total_buy_amount:,.0f}")
        print(f"  Average Buy Amount: ¥{avg_buy_amount:,.0f}")
    
    if sell_records:
        total_sell_amount = sum(r.amount for r in sell_records)
        avg_sell_amount = total_sell_amount / len(sell_records)
        print(f"  Total Sell Amount: ¥{total_sell_amount:,.0f}")
        print(f"  Average Sell Amount: ¥{avg_sell_amount:,.0f}")
    
    # Analyze by institution type
    print(f"\nInstitution Type Analysis:")
    type_analysis = {}
    
    for record in dragon_tiger_data:
        if record.institution:
            inst_type = record.institution.institution_type.value
            if inst_type not in type_analysis:
                type_analysis[inst_type] = {'count': 0, 'total_amount': 0}
            
            type_analysis[inst_type]['count'] += 1
            type_analysis[inst_type]['total_amount'] += record.amount
    
    for inst_type, data in sorted(type_analysis.items(), key=lambda x: x[1]['total_amount'], reverse=True):
        count = data['count']
        total_amount = data['total_amount']
        avg_amount = total_amount / count if count > 0 else 0
        
        print(f"  {inst_type.replace('_', ' ').title()}:")
        print(f"    Records: {count}")
        print(f"    Total Amount: ¥{total_amount:,.0f}")
        print(f"    Average Amount: ¥{avg_amount:,.0f}")
    
    # Show top trading seats
    print(f"\nTop Trading Seats (by amount):")
    
    seat_amounts = {}
    for record in dragon_tiger_data:
        seat_amounts[record.seat_name] = seat_amounts.get(record.seat_name, 0) + record.amount
    
    top_seats = sorted(seat_amounts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for i, (seat_name, total_amount) in enumerate(top_seats, 1):
        print(f"  {i}. {seat_name}")
        print(f"     Total Amount: ¥{total_amount:,.0f}")


async def demonstrate_shareholder_analysis(collector, all_data):
    """Demonstrate shareholder data analysis."""
    
    print(f"\n\n=== Shareholder Analysis ===\n")
    
    if not all_data:
        print("No data available for analysis")
        return
    
    # Analyze shareholder data for first stock
    stock_code = list(all_data.keys())[0]
    shareholder_data = all_data[stock_code]['shareholders']
    
    if not shareholder_data:
        print(f"No shareholder data available for {stock_code}")
        return
    
    print(f"Analyzing shareholder data for {stock_code}")
    print(f"Total shareholder records: {len(shareholder_data)}")
    
    # Group by report date
    by_date = {}
    for record in shareholder_data:
        report_date = record.report_date
        if report_date not in by_date:
            by_date[report_date] = []
        by_date[report_date].append(record)
    
    print(f"\nQuarterly Reports: {len(by_date)} quarters")
    
    for report_date in sorted(by_date.keys()):
        records = by_date[report_date]
        print(f"\n{report_date} (Top 10 Shareholders):")
        
        # Sort by rank
        records.sort(key=lambda x: x.rank)
        
        total_institutional_holding = 0
        institutional_count = 0
        
        for record in records[:5]:  # Show top 5
            print(f"  {record.rank}. {record.shareholder_name}")
            print(f"     Shareholding: {record.shareholding_ratio:.2f}% ({record.shares_held:,} shares)")
            
            if record.institution and record.institution.institution_type != InstitutionType.OTHER:
                print(f"     Type: {record.institution.institution_type.value.replace('_', ' ').title()}")
                total_institutional_holding += record.shareholding_ratio
                institutional_count += 1
        
        print(f"  ...")
        print(f"  Total Institutional Holding (Top 10): {total_institutional_holding:.2f}%")
        print(f"  Institutional Shareholders: {institutional_count}/10")
    
    # Analyze institution types in shareholding
    print(f"\nInstitutional Shareholder Analysis:")
    
    type_holdings = {}
    for record in shareholder_data:
        if record.institution and record.institution.institution_type != InstitutionType.OTHER:
            inst_type = record.institution.institution_type.value
            if inst_type not in type_holdings:
                type_holdings[inst_type] = {'count': 0, 'total_holding': 0}
            
            type_holdings[inst_type]['count'] += 1
            type_holdings[inst_type]['total_holding'] += record.shareholding_ratio
    
    for inst_type, data in sorted(type_holdings.items(), key=lambda x: x[1]['total_holding'], reverse=True):
        count = data['count']
        total_holding = data['total_holding']
        avg_holding = total_holding / count if count > 0 else 0
        
        print(f"  {inst_type.replace('_', ' ').title()}:")
        print(f"    Appearances: {count}")
        print(f"    Total Holding: {total_holding:.2f}%")
        print(f"    Average Holding: {avg_holding:.2f}%")


async def demonstrate_block_trade_analysis(collector, all_data):
    """Demonstrate block trade analysis."""
    
    print(f"\n\n=== Block Trade Analysis ===\n")
    
    if not all_data:
        print("No data available for analysis")
        return
    
    # Analyze block trade data for first stock
    stock_code = list(all_data.keys())[0]
    block_trade_data = all_data[stock_code]['block_trades']
    
    if not block_trade_data:
        print(f"No block trade data available for {stock_code}")
        return
    
    print(f"Analyzing block trade data for {stock_code}")
    print(f"Total block trade records: {len(block_trade_data)}")
    
    # Basic statistics
    total_volume = sum(r.volume for r in block_trade_data)
    total_amount = sum(r.total_amount for r in block_trade_data)
    avg_price = sum(r.price for r in block_trade_data) / len(block_trade_data)
    avg_volume = total_volume / len(block_trade_data)
    
    print(f"\nTrading Statistics:")
    print(f"  Total Volume: {total_volume:,} shares")
    print(f"  Total Amount: ¥{total_amount:,.0f}")
    print(f"  Average Price: ¥{avg_price:.2f}")
    print(f"  Average Volume per Trade: {avg_volume:,.0f} shares")
    
    # Discount/Premium analysis
    discounted_trades = [r for r in block_trade_data if r.discount_rate is not None]
    premium_trades = [r for r in block_trade_data if r.premium_rate is not None]
    
    print(f"\nPrice Analysis:")
    print(f"  Discounted Trades: {len(discounted_trades)}")
    if discounted_trades:
        avg_discount = sum(r.discount_rate for r in discounted_trades) / len(discounted_trades)
        print(f"    Average Discount: {avg_discount:.2%}")
    
    print(f"  Premium Trades: {len(premium_trades)}")
    if premium_trades:
        avg_premium = sum(r.premium_rate for r in premium_trades) / len(premium_trades)
        print(f"    Average Premium: {avg_premium:.2%}")
    
    # Institution analysis
    print(f"\nInstitutional Participation:")
    
    buyer_types = {}
    seller_types = {}
    
    for record in block_trade_data:
        if record.buyer_institution:
            buyer_type = record.buyer_institution.institution_type.value
            buyer_types[buyer_type] = buyer_types.get(buyer_type, 0) + 1
        
        if record.seller_institution:
            seller_type = record.seller_institution.institution_type.value
            seller_types[seller_type] = seller_types.get(seller_type, 0) + 1
    
    print(f"  Buyer Institution Types:")
    for inst_type, count in sorted(buyer_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    {inst_type.replace('_', ' ').title()}: {count} trades")
    
    print(f"  Seller Institution Types:")
    for inst_type, count in sorted(seller_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    {inst_type.replace('_', ' ').title()}: {count} trades")


async def demonstrate_activity_timeline(collector, all_data):
    """Demonstrate institutional activity timeline analysis."""
    
    print(f"\n\n=== Institutional Activity Timeline ===\n")
    
    if not all_data:
        print("No data available for analysis")
        return
    
    # Analyze activity timeline for first stock
    stock_code = list(all_data.keys())[0]
    
    # Get all activities
    all_activities = collector.get_institution_activity_timeline(stock_code)
    
    if not all_activities:
        print(f"No activities found for {stock_code}")
        return
    
    print(f"Activity timeline for {stock_code}")
    print(f"Total activities: {len(all_activities)}")
    
    # Activity type breakdown
    activity_types = {}
    for activity in all_activities:
        activity_type = activity.activity_type.value
        activity_types[activity_type] = activity_types.get(activity_type, 0) + 1
    
    print(f"\nActivity Type Breakdown:")
    for activity_type, count in sorted(activity_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {activity_type.replace('_', ' ').title()}: {count}")
    
    # Institution type participation
    institution_participation = {}
    for activity in all_activities:
        inst_type = activity.institution.institution_type.value
        if inst_type not in institution_participation:
            institution_participation[inst_type] = {'activities': 0, 'total_amount': 0}
        
        institution_participation[inst_type]['activities'] += 1
        if activity.amount:
            institution_participation[inst_type]['total_amount'] += activity.amount
    
    print(f"\nInstitution Type Participation:")
    for inst_type, data in sorted(institution_participation.items(), 
                                key=lambda x: x[1]['total_amount'], reverse=True):
        activities = data['activities']
        total_amount = data['total_amount']
        
        print(f"  {inst_type.replace('_', ' ').title()}:")
        print(f"    Activities: {activities}")
        print(f"    Total Amount: ¥{total_amount:,.0f}")
    
    # Recent activity analysis (last 30 days)
    if all_activities:
        latest_date = max(a.activity_date for a in all_activities)
        cutoff_date = latest_date - timedelta(days=30)
        recent_activities = [a for a in all_activities if a.activity_date >= cutoff_date]
        
        print(f"\nRecent Activity (Last 30 days from {latest_date}):")
        print(f"  Recent Activities: {len(recent_activities)}")
        print(f"  Activity Rate: {len(recent_activities)/30:.1f} activities/day")
        
        if recent_activities:
            recent_amount = sum(a.amount for a in recent_activities if a.amount)
            print(f"  Recent Total Amount: ¥{recent_amount:,.0f}")
    
    # Show timeline sample (first 10 activities)
    print(f"\nActivity Timeline Sample (First 10 activities):")
    
    for i, activity in enumerate(all_activities[:10], 1):
        print(f"  {i}. {activity.activity_date} - {activity.activity_type.value.replace('_', ' ').title()}")
        print(f"     Institution: {activity.institution.name}")
        print(f"     Type: {activity.institution.institution_type.value.replace('_', ' ').title()}")
        if activity.amount:
            print(f"     Amount: ¥{activity.amount:,.0f}")
        print(f"     Source: {activity.source_type}")


async def demonstrate_summary_statistics(collector, all_data):
    """Demonstrate summary statistics and insights."""
    
    print(f"\n\n=== Summary Statistics & Insights ===\n")
    
    if not all_data:
        print("No data available for analysis")
        return
    
    # Overall statistics across all stocks
    total_stocks = len(all_data)
    
    overall_stats = {
        'dragon_tiger_records': 0,
        'shareholder_records': 0,
        'block_trade_records': 0,
        'total_activities': 0,
        'unique_institutions': set(),
        'institution_types': set(),
        'total_amount': 0
    }
    
    for stock_code, data in all_data.items():
        overall_stats['dragon_tiger_records'] += len(data['dragon_tiger'])
        overall_stats['shareholder_records'] += len(data['shareholders'])
        overall_stats['block_trade_records'] += len(data['block_trades'])
        
        activities = collector.get_institution_activity_timeline(stock_code)
        overall_stats['total_activities'] += len(activities)
        
        for activity in activities:
            overall_stats['unique_institutions'].add(activity.institution.institution_id)
            overall_stats['institution_types'].add(activity.institution.institution_type)
            if activity.amount:
                overall_stats['total_amount'] += activity.amount
    
    print(f"Overall Collection Statistics:")
    print(f"  Stocks Analyzed: {total_stocks}")
    print(f"  Dragon-Tiger Records: {overall_stats['dragon_tiger_records']:,}")
    print(f"  Shareholder Records: {overall_stats['shareholder_records']:,}")
    print(f"  Block Trade Records: {overall_stats['block_trade_records']:,}")
    print(f"  Total Activities: {overall_stats['total_activities']:,}")
    print(f"  Unique Institutions: {len(overall_stats['unique_institutions']):,}")
    print(f"  Institution Types: {len(overall_stats['institution_types'])}")
    print(f"  Total Transaction Amount: ¥{overall_stats['total_amount']:,.0f}")
    
    # Institution type distribution
    print(f"\nInstitution Type Distribution:")
    type_counts = {}
    
    for stock_code in all_data.keys():
        activities = collector.get_institution_activity_timeline(stock_code)
        for activity in activities:
            inst_type = activity.institution.institution_type.value
            type_counts[inst_type] = type_counts.get(inst_type, 0) + 1
    
    total_activities = sum(type_counts.values())
    
    for inst_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_activities) * 100 if total_activities > 0 else 0
        print(f"  {inst_type.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")
    
    # Data quality assessment
    print(f"\nData Quality Assessment:")
    
    total_records = (overall_stats['dragon_tiger_records'] + 
                    overall_stats['shareholder_records'] + 
                    overall_stats['block_trade_records'])
    
    if total_records > 0:
        classification_rate = (overall_stats['total_activities'] / total_records) * 100
        print(f"  Classification Success Rate: {classification_rate:.1f}%")
    
    # Calculate average confidence scores
    confidence_scores = []
    for stock_code in all_data.keys():
        activities = collector.get_institution_activity_timeline(stock_code)
        confidence_scores.extend([a.confidence_score for a in activities])
    
    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        print(f"  Average Classification Confidence: {avg_confidence:.1%}")
    
    print(f"\nKey Insights:")
    print(f"  • Institutional activity is dominated by {max(type_counts.items(), key=lambda x: x[1])[0].replace('_', ' ')} institutions")
    print(f"  • Average of {overall_stats['total_activities']/total_stocks:.1f} activities per stock")
    print(f"  • Data collection covers {(overall_stats['total_activities']/90):.1f} activities per day on average")
    
    # Export sample data
    print(f"\n=== Data Export Sample ===")
    
    sample_stock = list(all_data.keys())[0]
    dataframes = collector.export_data_to_dataframes(sample_stock)
    
    if 'activities' in dataframes:
        activities_df = dataframes['activities']
        print(f"\nSample Activities DataFrame for {sample_stock}:")
        print(f"  Shape: {activities_df.shape}")
        print(f"  Columns: {list(activities_df.columns)}")
        
        if len(activities_df) > 0:
            print(f"\nFirst 3 records:")
            print(activities_df.head(3).to_string(index=False))


if __name__ == "__main__":
    print("Institutional Data Collector - Comprehensive Demo")
    print("=" * 60)
    
    async def run_full_demo():
        # Run classification demo
        await demonstrate_institution_classification()
        
        # Run data collection demo
        collector, all_data = await demonstrate_data_collection()
        
        if collector and all_data:
            # Run analysis demos
            await demonstrate_dragon_tiger_analysis(collector, all_data)
            await demonstrate_shareholder_analysis(collector, all_data)
            await demonstrate_block_trade_analysis(collector, all_data)
            await demonstrate_activity_timeline(collector, all_data)
            await demonstrate_summary_statistics(collector, all_data)
        
        print(f"\n" + "=" * 60)
        print("Demo completed successfully!")
    
    asyncio.run(run_full_demo())