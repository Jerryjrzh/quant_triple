#!/usr/bin/env python3
"""
Institutional Attention Scoring Demo

This script demonstrates the institutional attention scoring system including
comprehensive scoring, behavior pattern classification, stock screening, and alert generation.
"""

import asyncio
import sys
import os
from datetime import date, timedelta
import json
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from stock_analysis_system.analysis.institutional_data_collector import InstitutionalDataCollector
from stock_analysis_system.analysis.institutional_graph_analytics import InstitutionalGraphAnalytics
from stock_analysis_system.analysis.institutional_attention_scoring import (
    InstitutionalAttentionScoringSystem, AttentionLevel, BehaviorPattern, ActivityIntensity
)

async def main():
    """Main demo function"""
    
    print("🎯 Institutional Attention Scoring Demo")
    print("=" * 50)
    
    try:
        # Initialize components
        print("\n📊 Initializing institutional attention scoring system...")
        data_collector = InstitutionalDataCollector()
        graph_analytics = InstitutionalGraphAnalytics(data_collector)
        scoring_system = InstitutionalAttentionScoringSystem(
            data_collector=data_collector,
            graph_analytics=graph_analytics
        )
        
        # Define analysis parameters
        stock_codes = ["000001", "000002", "600000", "600036", "000858"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 3, 31)
        
        print(f"📈 Analyzing stocks: {', '.join(stock_codes)}")
        print(f"📅 Date range: {start_date} to {end_date}")
        
        # Step 1: Calculate attention profiles for individual stocks
        print("\n🎯 Calculating institutional attention profiles...")
        
        stock_profiles = {}
        for stock_code in stock_codes:
            print(f"\n   📊 Analyzing {stock_code}...")
            
            profile = await scoring_system.calculate_stock_attention_profile(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                min_attention_score=10.0  # Include institutions with score >= 10
            )
            
            stock_profiles[stock_code] = profile
            
            print(f"   ✅ Profile calculated:")
            print(f"      - Total Attention Score: {profile.total_attention_score:.1f}")
            print(f"      - Institutions Tracked: {profile.institutional_count}")
            print(f"      - Recently Active: {profile.active_institutional_count}")
            print(f"      - Total Activities: {profile.total_activities}")
            print(f"      - Recent Activities: {profile.recent_activities}")
            print(f"      - Activity Trend: {profile.activity_trend:.2f}")
            print(f"      - Coordination Score: {profile.coordination_score:.2f}")
            
            # Show top institutions for this stock
            if profile.institution_scores:
                print(f"      - Top 3 Institutions:")
                top_institutions = sorted(
                    profile.institution_scores, 
                    key=lambda s: s.overall_score, 
                    reverse=True
                )[:3]
                
                for i, score in enumerate(top_institutions, 1):
                    print(f"        {i}. {score.institution.name}")
                    print(f"           Score: {score.overall_score:.1f} ({score.attention_level.value})")
                    print(f"           Pattern: {score.behavior_pattern.value}")
                    print(f"           Intensity: {score.activity_intensity.value}")
                    print(f"           Activities: {score.total_activities} total, {score.recent_activities} recent")
        
        # Step 2: Analyze attention patterns across all stocks
        print("\n📈 Analyzing attention patterns across all stocks...")
        
        # Aggregate statistics
        total_institutions = sum(p.institutional_count for p in stock_profiles.values())
        total_activities = sum(p.total_activities for p in stock_profiles.values())
        avg_attention_score = sum(p.total_attention_score for p in stock_profiles.values()) / len(stock_profiles)
        
        print(f"   📊 Aggregate Statistics:")
        print(f"      - Total Unique Institutions: {total_institutions}")
        print(f"      - Total Activities Analyzed: {total_activities}")
        print(f"      - Average Attention Score: {avg_attention_score:.1f}")
        
        # Attention level distribution
        attention_distribution = {}
        behavior_distribution = {}
        intensity_distribution = {}
        
        for profile in stock_profiles.values():
            for score in profile.institution_scores:
                # Attention levels
                level = score.attention_level.value
                attention_distribution[level] = attention_distribution.get(level, 0) + 1
                
                # Behavior patterns
                pattern = score.behavior_pattern.value
                behavior_distribution[pattern] = behavior_distribution.get(pattern, 0) + 1
                
                # Activity intensity
                intensity = score.activity_intensity.value
                intensity_distribution[intensity] = intensity_distribution.get(intensity, 0) + 1
        
        print(f"\n   🎯 Attention Level Distribution:")
        for level, count in sorted(attention_distribution.items(), 
                                 key=lambda x: ['very_low', 'low', 'moderate', 'high', 'very_high'].index(x[0])):
            percentage = (count / total_institutions) * 100 if total_institutions > 0 else 0
            print(f"      - {level.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        print(f"\n   🎭 Behavior Pattern Distribution:")
        for pattern, count in sorted(behavior_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_institutions) * 100 if total_institutions > 0 else 0
            print(f"      - {pattern.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        print(f"\n   ⚡ Activity Intensity Distribution:")
        for intensity, count in sorted(intensity_distribution.items(),
                                     key=lambda x: ['dormant', 'light', 'moderate', 'heavy', 'extreme'].index(x[0])):
            percentage = (count / total_institutions) * 100 if total_institutions > 0 else 0
            print(f"      - {intensity.title()}: {count} ({percentage:.1f}%)")
        
        # Step 3: Stock screening by attention criteria
        print("\n🔍 Screening stocks by attention criteria...")
        
        screening_criteria = {
            'high_attention': 50.0,        # Attention score >= 50
            'coordinated_activity': 0.3,   # Coordination score >= 0.3
            'recent_activity_days': 7,     # Recent activity required
            'min_institutions': 2,         # At least 2 institutions
            'positive_trend': True         # Positive activity trend
        }
        
        print(f"   📋 Screening Criteria:")
        for criterion, value in screening_criteria.items():
            print(f"      - {criterion.replace('_', ' ').title()}: {value}")
        
        screening_results = await scoring_system.screen_stocks_by_attention(
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            criteria=screening_criteria
        )
        
        print(f"\n   ✅ Screening Results: {len(screening_results)} stocks meet criteria")
        
        if screening_results:
            print(f"\n   🏆 Top Stocks by Attention:")
            
            for i, result in enumerate(screening_results[:5], 1):  # Top 5
                print(f"\n      {i}. {result['stock_code']}")
                print(f"         Attention Score: {result['total_attention_score']:.1f}")
                print(f"         Institutions: {result['institutional_count']} total, {result['active_institutional_count']} active")
                print(f"         Recent Activities: {result['recent_activities']}")
                print(f"         Activity Trend: {result['activity_trend']:.2f}")
                print(f"         Coordination Score: {result['coordination_score']:.2f}")
                print(f"         Dominant Pattern: {result['dominant_pattern']}")
                
                print(f"         Screening Reasons:")
                for reason in result['screening_reasons']:
                    print(f"           • {reason}")
                
                print(f"         Top Institutions:")
                for j, inst in enumerate(result['top_institutions'][:3], 1):
                    print(f"           {j}. {inst['name']} ({inst['type']})")
                    print(f"              Score: {inst['attention_score']:.1f}, Pattern: {inst['behavior_pattern']}")
        
        # Step 4: Generate attention alerts
        print("\n🚨 Generating attention alerts...")
        
        alerts = scoring_system.generate_attention_alerts(
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            alert_threshold=60.0  # Alert for scores >= 60
        )
        
        print(f"   📢 Generated {len(alerts)} alerts")
        
        if alerts:
            print(f"\n   🔔 High Priority Alerts:")
            
            high_priority_alerts = [a for a in alerts if a['priority'] == 'high']
            medium_priority_alerts = [a for a in alerts if a['priority'] == 'medium']
            
            if high_priority_alerts:
                print(f"\n      🔴 High Priority ({len(high_priority_alerts)} alerts):")
                for alert in high_priority_alerts[:3]:  # Top 3
                    print(f"         • {alert['stock_code']} - {alert['alert_type']}")
                    print(f"           Score: {alert['attention_score']:.1f}")
                    print(f"           Message: {alert['message']}")
                    print(f"           Timestamp: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            if medium_priority_alerts:
                print(f"\n      🟡 Medium Priority ({len(medium_priority_alerts)} alerts):")
                for alert in medium_priority_alerts[:2]:  # Top 2
                    print(f"         • {alert['stock_code']} - {alert['alert_type']}")
                    print(f"           Score: {alert['attention_score']:.1f}")
                    print(f"           Message: {alert['message'][:100]}...")
        
        # Step 5: Institution-specific analysis
        print("\n🏢 Institution-specific attention analysis...")
        
        # Get a sample institution for detailed analysis
        sample_institution_id = None
        for profile in stock_profiles.values():
            if profile.institution_scores:
                sample_institution_id = profile.institution_scores[0].institution.institution_id
                break
        
        if sample_institution_id:
            institution_summary = scoring_system.get_institution_attention_summary(sample_institution_id)
            
            if 'error' not in institution_summary:
                print(f"\n   📊 Sample Institution Analysis:")
                print(f"      Institution: {institution_summary['institution_name']}")
                print(f"      Type: {institution_summary['institution_type']}")
                print(f"      Average Attention Score: {institution_summary['average_attention_score']:.1f}")
                print(f"      Stocks Tracked: {institution_summary['total_stocks_tracked']}")
                print(f"      High Attention Stocks: {institution_summary['high_attention_stocks']}")
                print(f"      Total Activities: {institution_summary['total_activities']}")
                print(f"      Recent Activities: {institution_summary['recent_activities']}")
                
                print(f"      Dominant Patterns:")
                for pattern, count in list(institution_summary['dominant_patterns'].items())[:3]:
                    print(f"         • {pattern.replace('_', ' ').title()}: {count}")
                
                print(f"      Top Stocks by Attention:")
                for stock in institution_summary['top_stocks'][:3]:
                    print(f"         • {stock['stock_code']}: {stock['attention_score']:.1f} ({stock['behavior_pattern']})")
        
        # Step 6: Export results
        print("\n💾 Exporting analysis results...")
        
        # Prepare export data
        export_data = {
            'analysis_summary': {
                'analysis_date': date.today().isoformat(),
                'stock_codes': stock_codes,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'total_institutions': total_institutions,
                'total_activities': total_activities,
                'average_attention_score': avg_attention_score
            },
            'stock_profiles': {},
            'screening_results': screening_results,
            'alerts': alerts,
            'distributions': {
                'attention_levels': attention_distribution,
                'behavior_patterns': behavior_distribution,
                'activity_intensity': intensity_distribution
            }
        }
        
        # Add stock profiles (simplified for JSON serialization)
        for stock_code, profile in stock_profiles.items():
            export_data['stock_profiles'][stock_code] = {
                'total_attention_score': profile.total_attention_score,
                'institutional_count': profile.institutional_count,
                'active_institutional_count': profile.active_institutional_count,
                'total_activities': profile.total_activities,
                'recent_activities': profile.recent_activities,
                'activity_trend': profile.activity_trend,
                'coordination_score': profile.coordination_score,
                'dominant_patterns': [(pattern.value, count) for pattern, count in profile.dominant_patterns],
                'attention_distribution': {level.value: count for level, count in profile.attention_distribution.items()},
                'top_institutions': [
                    {
                        'name': score.institution.name,
                        'type': score.institution.institution_type.value,
                        'attention_score': score.overall_score,
                        'behavior_pattern': score.behavior_pattern.value,
                        'activity_intensity': score.activity_intensity.value,
                        'total_activities': score.total_activities,
                        'recent_activities': score.recent_activities
                    }
                    for score in sorted(profile.institution_scores, key=lambda s: s.overall_score, reverse=True)[:5]
                ]
            }
        
        # Save to file
        export_file = "institutional_attention_analysis.json"
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"   💾 Analysis results saved to: {export_file}")
        
        # Create summary CSV for easy analysis
        csv_data = []
        for stock_code, profile in stock_profiles.items():
            csv_data.append({
                'stock_code': stock_code,
                'attention_score': profile.total_attention_score,
                'institutional_count': profile.institutional_count,
                'active_institutions': profile.active_institutional_count,
                'total_activities': profile.total_activities,
                'recent_activities': profile.recent_activities,
                'activity_trend': profile.activity_trend,
                'coordination_score': profile.coordination_score,
                'dominant_pattern': profile.dominant_patterns[0][0].value if profile.dominant_patterns else 'none',
                'data_quality': profile.data_quality_score
            })
        
        df = pd.DataFrame(csv_data)
        csv_file = "institutional_attention_summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"   📊 Summary CSV saved to: {csv_file}")
        
        # Step 7: Performance and insights summary
        print("\n⚡ Performance Summary:")
        print(f"   - Stocks Analyzed: {len(stock_codes)}")
        print(f"   - Institutions Tracked: {total_institutions}")
        print(f"   - Activities Processed: {total_activities}")
        print(f"   - Stocks Meeting Screening Criteria: {len(screening_results)}")
        print(f"   - Alerts Generated: {len(alerts)}")
        
        print("\n💡 Key Insights:")
        
        # Highest attention stock
        if stock_profiles:
            highest_attention_stock = max(stock_profiles.items(), key=lambda x: x[1].total_attention_score)
            print(f"   🏆 Highest Attention Stock: {highest_attention_stock[0]} (Score: {highest_attention_stock[1].total_attention_score:.1f})")
        
        # Most active institutions
        if total_institutions > 0:
            most_common_pattern = max(behavior_distribution.items(), key=lambda x: x[1])
            print(f"   🎭 Most Common Behavior Pattern: {most_common_pattern[0].replace('_', ' ').title()} ({most_common_pattern[1]} institutions)")
        
        # Activity trends
        positive_trend_stocks = [s for s in stock_profiles.values() if s.activity_trend > 0]
        print(f"   📈 Stocks with Positive Activity Trends: {len(positive_trend_stocks)}/{len(stock_codes)}")
        
        # Coordination analysis
        high_coordination_stocks = [s for s in stock_profiles.values() if s.coordination_score > 0.5]
        print(f"   🤝 Stocks with High Coordination: {len(high_coordination_stocks)}/{len(stock_codes)}")
        
        print("\n✅ Institutional Attention Scoring Demo completed successfully!")
        print("\nGenerated files:")
        print("   - institutional_attention_analysis.json")
        print("   - institutional_attention_summary.csv")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)