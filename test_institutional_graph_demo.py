#!/usr/bin/env python3
"""
Institutional Graph Analytics Demo

This script demonstrates the institutional graph analytics functionality including
relationship detection, network analysis, and visualization capabilities.
"""

import asyncio
import sys
import os
from datetime import date, timedelta
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from stock_analysis_system.analysis.institutional_data_collector import InstitutionalDataCollector
from stock_analysis_system.analysis.institutional_graph_analytics import InstitutionalGraphAnalytics

async def main():
    """Main demo function"""
    
    print("🔗 Institutional Graph Analytics Demo")
    print("=" * 50)
    
    try:
        # Initialize components
        print("\n📊 Initializing institutional data collector and graph analytics...")
        data_collector = InstitutionalDataCollector()
        graph_analytics = InstitutionalGraphAnalytics(data_collector)
        
        # Define analysis parameters
        stock_codes = ["000001", "000002", "600000", "600036", "000858"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 3, 31)
        
        print(f"📈 Analyzing stocks: {', '.join(stock_codes)}")
        print(f"📅 Date range: {start_date} to {end_date}")
        
        # Step 1: Build institutional network
        print("\n🏗️  Building institutional relationship network...")
        network_graph = await graph_analytics.build_institutional_network(
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"✅ Network built successfully!")
        print(f"   - Nodes (Institutions): {network_graph.number_of_nodes()}")
        print(f"   - Edges (Relationships): {network_graph.number_of_edges()}")
        
        # Step 2: Display network metrics
        print("\n📊 Network Metrics:")
        if graph_analytics.network_metrics:
            metrics = graph_analytics.network_metrics
            print(f"   - Network Density: {metrics.density:.3f}")
            print(f"   - Average Clustering Coefficient: {metrics.avg_clustering_coefficient:.3f}")
            print(f"   - Average Path Length: {metrics.avg_path_length:.3f}")
            print(f"   - Number of Communities: {len(metrics.communities)}")
            print(f"   - Modularity: {metrics.modularity:.3f}")
            
            # Top institutions by centrality
            print("\n🏆 Top Institutions by Degree Centrality:")
            top_degree = sorted(
                metrics.degree_centrality.items(),
                key=lambda x: x[1], reverse=True
            )[:5]
            
            for i, (inst_id, centrality) in enumerate(top_degree, 1):
                inst_name = graph_analytics.institutions[inst_id].name if inst_id in graph_analytics.institutions else inst_id
                print(f"   {i}. {inst_name}: {centrality:.3f}")
            
            print("\n🌉 Top Institutions by Betweenness Centrality:")
            top_betweenness = sorted(
                metrics.betweenness_centrality.items(),
                key=lambda x: x[1], reverse=True
            )[:5]
            
            for i, (inst_id, centrality) in enumerate(top_betweenness, 1):
                inst_name = graph_analytics.institutions[inst_id].name if inst_id in graph_analytics.institutions else inst_id
                print(f"   {i}. {inst_name}: {centrality:.3f}")
        
        # Step 3: Analyze relationships
        print(f"\n🤝 Institutional Relationships: {len(graph_analytics.relationships)}")
        
        if graph_analytics.relationships:
            # Group relationships by type
            relationship_types = {}
            for relationship in graph_analytics.relationships.values():
                rel_type = relationship.relationship_type.value
                if rel_type not in relationship_types:
                    relationship_types[rel_type] = []
                relationship_types[rel_type].append(relationship)
            
            for rel_type, relationships in relationship_types.items():
                print(f"\n   📋 {rel_type.replace('_', ' ').title()}: {len(relationships)} relationships")
                
                # Show top 3 strongest relationships of this type
                top_relationships = sorted(relationships, key=lambda r: r.strength_score, reverse=True)[:3]
                
                for i, rel in enumerate(top_relationships, 1):
                    print(f"      {i}. {rel.institution_a.name} ↔ {rel.institution_b.name}")
                    print(f"         Strength: {rel.strength_score:.3f}")
                    if rel.common_stocks:
                        print(f"         Common Stocks: {len(rel.common_stocks)}")
                    if rel.coordinated_activities:
                        print(f"         Coordinated Activities: {len(rel.coordinated_activities)}")
        
        # Step 4: Detect coordinated patterns
        print("\n🎯 Detecting coordinated activity patterns...")
        coordinated_patterns = await graph_analytics.detect_coordinated_patterns(
            min_institutions=2,
            min_correlation=0.6
        )
        
        print(f"✅ Found {len(coordinated_patterns)} coordinated patterns")
        
        if coordinated_patterns:
            print("\n📈 Top Coordinated Patterns:")
            
            # Sort patterns by correlation strength
            top_patterns = sorted(
                coordinated_patterns,
                key=lambda p: p.activity_correlation,
                reverse=True
            )[:3]
            
            for i, pattern in enumerate(top_patterns, 1):
                print(f"\n   Pattern {i}:")
                print(f"   - Institutions: {len(pattern.institutions)}")
                print(f"   - Stocks: {', '.join(pattern.stock_codes)}")
                print(f"   - Activity Type: {pattern.activity_type.value}")
                print(f"   - Activity Correlation: {pattern.activity_correlation:.3f}")
                print(f"   - Volume Correlation: {pattern.volume_correlation:.3f}")
                print(f"   - Time Window: {pattern.time_window.days} days")
                
                # List participating institutions
                inst_names = [inst.name for inst in pattern.institutions]
                print(f"   - Participating Institutions:")
                for inst_name in inst_names:
                    print(f"     • {inst_name}")
        
        # Step 5: Institution type analysis
        print("\n🏢 Institution Type Distribution:")
        type_counts = {}
        for institution in graph_analytics.institutions.values():
            inst_type = institution.institution_type.value
            type_counts[inst_type] = type_counts.get(inst_type, 0) + 1
        
        for inst_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(graph_analytics.institutions)) * 100
            print(f"   - {inst_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # Step 6: Generate network summary
        print("\n📋 Generating comprehensive network summary...")
        network_summary = graph_analytics.get_network_summary()
        
        if "error" not in network_summary:
            print("✅ Network summary generated successfully!")
            
            # Save summary to file
            summary_file = "institutional_network_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                # Convert sets to lists for JSON serialization
                serializable_summary = json.loads(json.dumps(network_summary, default=str))
                json.dump(serializable_summary, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Network summary saved to: {summary_file}")
        
        # Step 7: Create network visualization
        print("\n🎨 Creating network visualization...")
        try:
            # Create visualization with different layouts
            layouts = ["spring", "circular", "kamada_kawai"]
            
            for layout in layouts:
                print(f"   📊 Generating {layout} layout visualization...")
                fig = graph_analytics.create_network_visualization(
                    layout=layout,
                    node_size_metric="degree_centrality",
                    color_by="institution_type"
                )
                
                # Save visualization
                viz_file = f"institutional_network_{layout}.html"
                fig.write_html(viz_file)
                print(f"   💾 Saved: {viz_file}")
            
            print("✅ Network visualizations created successfully!")
            
        except Exception as e:
            print(f"⚠️  Visualization creation failed: {e}")
        
        # Step 8: Analyze specific institution relationships
        print("\n🔍 Analyzing specific institution relationships...")
        
        if graph_analytics.institutions:
            # Pick a random institution for detailed analysis
            sample_inst_id = list(graph_analytics.institutions.keys())[0]
            sample_inst = graph_analytics.institutions[sample_inst_id]
            
            print(f"\n📊 Detailed analysis for: {sample_inst.name}")
            print(f"   - Type: {sample_inst.institution_type.value}")
            print(f"   - Confidence Score: {sample_inst.confidence_score:.3f}")
            
            # Get relationships for this institution
            relationships = graph_analytics.get_institution_relationships(sample_inst_id)
            print(f"   - Number of Relationships: {len(relationships)}")
            
            if relationships:
                print("   - Related Institutions:")
                for rel in relationships[:5]:  # Show top 5
                    other_inst = (rel.institution_b if rel.institution_a.institution_id == sample_inst_id 
                                else rel.institution_a)
                    print(f"     • {other_inst.name} ({rel.relationship_type.value}, strength: {rel.strength_score:.3f})")
        
        # Step 9: Performance summary
        print("\n⚡ Performance Summary:")
        print(f"   - Total Institutions Analyzed: {len(graph_analytics.institutions)}")
        print(f"   - Total Relationships Detected: {len(graph_analytics.relationships)}")
        print(f"   - Coordinated Patterns Found: {len(coordinated_patterns)}")
        print(f"   - Network Density: {graph_analytics.network_metrics.density:.3f}")
        
        # Step 10: Recommendations
        print("\n💡 Analysis Insights:")
        
        if graph_analytics.network_metrics:
            density = graph_analytics.network_metrics.density
            
            if density > 0.3:
                print("   🔗 High network density indicates strong institutional interconnectedness")
            elif density > 0.1:
                print("   🔗 Moderate network density shows selective institutional relationships")
            else:
                print("   🔗 Low network density suggests fragmented institutional landscape")
            
            if len(coordinated_patterns) > 0:
                print(f"   🎯 {len(coordinated_patterns)} coordinated patterns detected - monitor for market impact")
            
            if graph_analytics.network_metrics.modularity > 0.3:
                print("   🏘️  Strong community structure detected - institutions form distinct groups")
        
        print("\n✅ Institutional Graph Analytics Demo completed successfully!")
        print("\nGenerated files:")
        print("   - institutional_network_summary.json")
        print("   - institutional_network_spring.html")
        print("   - institutional_network_circular.html")
        print("   - institutional_network_kamada_kawai.html")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)