"""
Tests for Institutional Graph Analytics

This module contains comprehensive tests for the institutional graph analytics
functionality including relationship detection, network analysis, and visualization.
"""

import asyncio
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from stock_analysis_system.analysis.institutional_data_collector import (
    ActivityType,
    InstitutionalActivity,
    InstitutionalDataCollector,
    InstitutionalInvestor,
    InstitutionType,
)
from stock_analysis_system.analysis.institutional_graph_analytics import (
    CoordinatedActivityPattern,
    InstitutionalGraphAnalytics,
    InstitutionalRelationship,
    NetworkMetrics,
    RelationshipType,
)


class TestInstitutionalGraphAnalytics:
    """Test cases for InstitutionalGraphAnalytics"""

    @pytest.fixture
    def mock_data_collector(self):
        """Create a mock data collector with sample data"""
        collector = Mock(spec=InstitutionalDataCollector)

        # Create sample institutions
        inst1 = InstitutionalInvestor(
            institution_id="fund_001",
            name="易方达基金管理有限公司",
            institution_type=InstitutionType.MUTUAL_FUND,
            confidence_score=0.9,
        )

        inst2 = InstitutionalInvestor(
            institution_id="fund_002",
            name="华夏基金管理有限公司",
            institution_type=InstitutionType.MUTUAL_FUND,
            confidence_score=0.9,
        )

        inst3 = InstitutionalInvestor(
            institution_id="qfii_001",
            name="摩根士丹利投资管理公司",
            institution_type=InstitutionType.QFII,
            confidence_score=0.8,
        )

        # Create sample activities
        base_date = date(2024, 1, 1)

        activities = [
            # Coordinated activities between inst1 and inst2
            InstitutionalActivity(
                activity_id="act_001",
                activity_date=base_date,
                stock_code="000001",
                institution=inst1,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
                amount=10000000,
                volume=1000000,
                source_type="dragon_tiger",
                confidence_score=0.9,
            ),
            InstitutionalActivity(
                activity_id="act_002",
                activity_date=base_date + timedelta(days=1),
                stock_code="000001",
                institution=inst2,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
                amount=8000000,
                volume=800000,
                source_type="dragon_tiger",
                confidence_score=0.9,
            ),
            # Same stock activity
            InstitutionalActivity(
                activity_id="act_003",
                activity_date=base_date + timedelta(days=5),
                stock_code="000002",
                institution=inst1,
                activity_type=ActivityType.BLOCK_TRADE_BUY,
                amount=15000000,
                volume=1500000,
                source_type="block_trade",
                confidence_score=0.9,
            ),
            InstitutionalActivity(
                activity_id="act_004",
                activity_date=base_date + timedelta(days=7),
                stock_code="000002",
                institution=inst2,
                activity_type=ActivityType.SHAREHOLDING_INCREASE,
                volume=2000000,
                source_type="shareholder",
                confidence_score=0.9,
            ),
            # QFII activity
            InstitutionalActivity(
                activity_id="act_005",
                activity_date=base_date + timedelta(days=3),
                stock_code="000001",
                institution=inst3,
                activity_type=ActivityType.BLOCK_TRADE_BUY,
                amount=20000000,
                volume=2000000,
                source_type="block_trade",
                confidence_score=0.8,
            ),
        ]

        # Mock the activity timeline
        collector.activity_timeline = {
            "000001": [activities[0], activities[1], activities[4]],
            "000002": [activities[2], activities[3]],
        }

        # Mock collect_all_data method
        async def mock_collect_all_data(stock_codes, start_date, end_date):
            return {
                "000001": {
                    "dragon_tiger": [activities[0], activities[1]],
                    "shareholders": [],
                    "block_trades": [activities[4]],
                },
                "000002": {
                    "dragon_tiger": [],
                    "shareholders": [activities[3]],
                    "block_trades": [activities[2]],
                },
            }

        collector.collect_all_data = AsyncMock(side_effect=mock_collect_all_data)

        return collector

    @pytest.fixture
    def graph_analytics(self, mock_data_collector):
        """Create InstitutionalGraphAnalytics instance with mock data"""
        return InstitutionalGraphAnalytics(mock_data_collector)

    @pytest.mark.asyncio
    async def test_build_institutional_network(self, graph_analytics):
        """Test building institutional network"""

        stock_codes = ["000001", "000002"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        # Build network
        graph = await graph_analytics.build_institutional_network(
            stock_codes, start_date, end_date
        )

        # Verify graph structure
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() > 0
        assert len(graph_analytics.institutions) > 0

        # Verify institutions are registered
        assert "fund_001" in graph_analytics.institutions
        assert "fund_002" in graph_analytics.institutions
        assert "qfii_001" in graph_analytics.institutions

        # Verify network metrics are calculated
        assert graph_analytics.network_metrics is not None
        assert isinstance(graph_analytics.network_metrics, NetworkMetrics)

    @pytest.mark.asyncio
    async def test_detect_coordinated_trading(self, graph_analytics):
        """Test coordinated trading detection"""

        # Build network first
        await graph_analytics.build_institutional_network(
            ["000001", "000002"], date(2024, 1, 1), date(2024, 1, 31)
        )

        # Check for relationships
        assert len(graph_analytics.relationships) > 0

        # Find coordinated trading relationship
        coordinated_rel = None
        for relationship in graph_analytics.relationships.values():
            if relationship.relationship_type == RelationshipType.COORDINATED_TRADING:
                coordinated_rel = relationship
                break

        if coordinated_rel:
            assert coordinated_rel.strength_score > 0
            assert len(coordinated_rel.coordinated_activities) > 0
            assert len(coordinated_rel.common_stocks) > 0

    def test_calculate_coordination_score(self, graph_analytics):
        """Test coordination score calculation"""

        # Create sample activities
        inst1 = InstitutionalInvestor(
            institution_id="test_001",
            name="Test Fund 1",
            institution_type=InstitutionType.MUTUAL_FUND,
        )

        inst2 = InstitutionalInvestor(
            institution_id="test_002",
            name="Test Fund 2",
            institution_type=InstitutionType.MUTUAL_FUND,
        )

        base_date = date(2024, 1, 1)

        activity1 = InstitutionalActivity(
            activity_id="test_act_1",
            activity_date=base_date,
            stock_code="000001",
            institution=inst1,
            activity_type=ActivityType.DRAGON_TIGER_BUY,
            amount=10000000,
            volume=1000000,
        )

        activity2 = InstitutionalActivity(
            activity_id="test_act_2",
            activity_date=base_date + timedelta(days=1),
            stock_code="000001",
            institution=inst2,
            activity_type=ActivityType.DRAGON_TIGER_BUY,
            amount=8000000,
            volume=800000,
        )

        # Calculate coordination score
        score = graph_analytics._calculate_coordination_score(activity1, activity2)

        assert 0.0 <= score <= 1.0
        assert score > 0  # Should have some coordination due to similar activities

    def test_are_activities_coordinated(self, graph_analytics):
        """Test activity coordination detection"""

        # Test coordinated buy activities
        assert graph_analytics._are_activities_coordinated(
            ActivityType.DRAGON_TIGER_BUY, ActivityType.BLOCK_TRADE_BUY
        )

        # Test coordinated sell activities
        assert graph_analytics._are_activities_coordinated(
            ActivityType.DRAGON_TIGER_SELL, ActivityType.SHAREHOLDING_DECREASE
        )

        # Test same type activities
        assert graph_analytics._are_activities_coordinated(
            ActivityType.DRAGON_TIGER_BUY, ActivityType.DRAGON_TIGER_BUY
        )

        # Test non-coordinated activities
        assert not graph_analytics._are_activities_coordinated(
            ActivityType.DRAGON_TIGER_BUY, ActivityType.DRAGON_TIGER_SELL
        )

    def test_detect_fund_family_relationship(self, graph_analytics):
        """Test fund family relationship detection"""

        # Create funds from same family
        fund1 = InstitutionalInvestor(
            institution_id="fund_001",
            name="易方达价值精选基金",
            institution_type=InstitutionType.MUTUAL_FUND,
        )

        fund2 = InstitutionalInvestor(
            institution_id="fund_002",
            name="易方达成长优选基金",
            institution_type=InstitutionType.MUTUAL_FUND,
        )

        # Test fund family detection
        relationship = graph_analytics._detect_fund_family_relationship(fund1, fund2)

        assert relationship is not None
        assert relationship.relationship_type == RelationshipType.FUND_FAMILY
        assert relationship.strength_score == 0.9

        # Test different fund families
        fund3 = InstitutionalInvestor(
            institution_id="fund_003",
            name="华夏成长基金",
            institution_type=InstitutionType.MUTUAL_FUND,
        )

        relationship2 = graph_analytics._detect_fund_family_relationship(fund1, fund3)
        assert relationship2 is None

    def test_extract_fund_company(self, graph_analytics):
        """Test fund company name extraction"""

        # Test various fund name patterns
        test_cases = [
            ("易方达基金管理有限公司", "易方达"),
            ("华夏基金管理有限公司", "华夏"),
            ("南方资产管理有限公司", "南方"),
            ("博时投资管理有限公司", "博时"),
            ("嘉实基金", "嘉实"),
        ]

        for fund_name, expected_company in test_cases:
            company = graph_analytics._extract_fund_company(fund_name)
            assert company == expected_company

    @pytest.mark.asyncio
    async def test_detect_coordinated_patterns(self, graph_analytics):
        """Test coordinated pattern detection"""

        # Build network first
        await graph_analytics.build_institutional_network(
            ["000001", "000002"], date(2024, 1, 1), date(2024, 1, 31)
        )

        # Detect patterns
        patterns = await graph_analytics.detect_coordinated_patterns(
            min_institutions=2, min_correlation=0.5
        )

        assert isinstance(patterns, list)

        # If patterns are found, verify their structure
        for pattern in patterns:
            assert isinstance(pattern, CoordinatedActivityPattern)
            assert len(pattern.institutions) >= 2
            assert pattern.activity_correlation >= 0.5
            assert len(pattern.stock_codes) > 0

    def test_create_time_windows(self, graph_analytics):
        """Test time window creation"""

        # Create sample activities
        base_date = date(2024, 1, 1)
        activities = []

        for i in range(10):
            activity = InstitutionalActivity(
                activity_id=f"act_{i}",
                activity_date=base_date + timedelta(days=i),
                stock_code="000001",
                institution=Mock(),
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            )
            activities.append(activity)

        # Create time windows
        windows = graph_analytics._create_time_windows(activities, timedelta(days=3))

        assert len(windows) > 0

        # Verify window structure
        for window_start, window_activities in windows.items():
            assert isinstance(window_start, date)
            assert isinstance(window_activities, list)
            assert len(window_activities) > 0

    def test_calculate_network_metrics(self, graph_analytics):
        """Test network metrics calculation"""

        # Create a simple test graph
        graph_analytics.graph.add_node("node1", institution=Mock())
        graph_analytics.graph.add_node("node2", institution=Mock())
        graph_analytics.graph.add_node("node3", institution=Mock())
        graph_analytics.graph.add_edge("node1", "node2", weight=0.8)
        graph_analytics.graph.add_edge("node2", "node3", weight=0.6)

        # Calculate metrics
        metrics = graph_analytics._calculate_network_metrics()

        assert isinstance(metrics, NetworkMetrics)
        assert metrics.total_nodes == 3
        assert metrics.total_edges == 2
        assert 0.0 <= metrics.density <= 1.0
        assert 0.0 <= metrics.avg_clustering_coefficient <= 1.0

        # Verify centrality measures
        assert len(metrics.degree_centrality) == 3
        assert len(metrics.betweenness_centrality) == 3
        assert len(metrics.closeness_centrality) == 3

        # Verify all centrality values are between 0 and 1
        for centrality_dict in [
            metrics.degree_centrality,
            metrics.betweenness_centrality,
            metrics.closeness_centrality,
        ]:
            for value in centrality_dict.values():
                assert 0.0 <= value <= 1.0

    def test_create_network_visualization(self, graph_analytics):
        """Test network visualization creation"""

        # Create a simple test graph with institutions
        inst1 = InstitutionalInvestor(
            institution_id="test_001",
            name="Test Fund 1",
            institution_type=InstitutionType.MUTUAL_FUND,
        )

        inst2 = InstitutionalInvestor(
            institution_id="test_002",
            name="Test Fund 2",
            institution_type=InstitutionType.QFII,
        )

        graph_analytics.institutions = {"test_001": inst1, "test_002": inst2}

        graph_analytics.graph.add_node("test_001", institution=inst1)
        graph_analytics.graph.add_node("test_002", institution=inst2)
        graph_analytics.graph.add_edge("test_001", "test_002", weight=0.8)

        # Calculate network metrics
        graph_analytics.network_metrics = graph_analytics._calculate_network_metrics()

        # Create visualization
        fig = graph_analytics.create_network_visualization()

        assert fig is not None
        assert len(fig.data) > 0  # Should have traces for nodes and edges

        # Test different layout options
        layouts = ["spring", "circular", "kamada_kawai"]
        for layout in layouts:
            fig = graph_analytics.create_network_visualization(layout=layout)
            assert fig is not None

    def test_create_network_visualization_empty_graph(self, graph_analytics):
        """Test network visualization with empty graph"""

        # Create visualization with empty graph
        fig = graph_analytics.create_network_visualization()

        assert fig is not None
        # Should have annotation about no data
        assert len(fig.layout.annotations) > 0

    def test_get_institution_relationships(self, graph_analytics):
        """Test getting relationships for specific institution"""

        # Create sample relationship
        inst1 = InstitutionalInvestor(
            institution_id="test_001",
            name="Test Fund 1",
            institution_type=InstitutionType.MUTUAL_FUND,
        )

        inst2 = InstitutionalInvestor(
            institution_id="test_002",
            name="Test Fund 2",
            institution_type=InstitutionType.MUTUAL_FUND,
        )

        relationship = InstitutionalRelationship(
            institution_a=inst1,
            institution_b=inst2,
            relationship_type=RelationshipType.FUND_FAMILY,
            strength_score=0.9,
        )

        graph_analytics.relationships[("test_001", "test_002")] = relationship

        # Get relationships
        relationships = graph_analytics.get_institution_relationships("test_001")

        assert len(relationships) == 1
        assert relationships[0] == relationship

        # Test non-existent institution
        relationships = graph_analytics.get_institution_relationships("non_existent")
        assert len(relationships) == 0

    def test_get_network_summary(self, graph_analytics):
        """Test network summary generation"""

        # Create test data
        inst1 = InstitutionalInvestor(
            institution_id="test_001",
            name="Test Fund 1",
            institution_type=InstitutionType.MUTUAL_FUND,
        )

        inst2 = InstitutionalInvestor(
            institution_id="test_002",
            name="Test Fund 2",
            institution_type=InstitutionType.QFII,
        )

        graph_analytics.institutions = {"test_001": inst1, "test_002": inst2}

        relationship = InstitutionalRelationship(
            institution_a=inst1,
            institution_b=inst2,
            relationship_type=RelationshipType.COORDINATED_TRADING,
            strength_score=0.8,
        )

        graph_analytics.relationships[("test_001", "test_002")] = relationship

        # Calculate network metrics
        graph_analytics.graph.add_node("test_001", institution=inst1)
        graph_analytics.graph.add_node("test_002", institution=inst2)
        graph_analytics.graph.add_edge("test_001", "test_002", weight=0.8)
        graph_analytics.network_metrics = graph_analytics._calculate_network_metrics()

        # Get summary
        summary = graph_analytics.get_network_summary()

        assert "network_metrics" in summary
        assert "top_institutions" in summary
        assert "distributions" in summary

        # Verify network metrics
        network_metrics = summary["network_metrics"]
        assert network_metrics["total_institutions"] == 2
        assert network_metrics["total_relationships"] == 1
        assert 0.0 <= network_metrics["network_density"] <= 1.0

        # Verify distributions
        distributions = summary["distributions"]
        assert "institution_types" in distributions
        assert "relationship_types" in distributions

        # Verify institution types
        inst_types = distributions["institution_types"]
        assert inst_types["mutual_fund"] == 1
        assert inst_types["qfii"] == 1

        # Verify relationship types
        rel_types = distributions["relationship_types"]
        assert rel_types["coordinated_trading"] == 1

    def test_get_network_summary_no_metrics(self, graph_analytics):
        """Test network summary with no calculated metrics"""

        summary = graph_analytics.get_network_summary()

        assert "error" in summary
        assert summary["error"] == "Network metrics not calculated"


class TestRelationshipDetection:
    """Test cases for relationship detection algorithms"""

    def test_same_stock_activity_detection(self):
        """Test same stock activity relationship detection"""

        # Create test institutions
        inst1 = InstitutionalInvestor(
            institution_id="test_001",
            name="Test Fund 1",
            institution_type=InstitutionType.MUTUAL_FUND,
        )

        inst2 = InstitutionalInvestor(
            institution_id="test_002",
            name="Test Fund 2",
            institution_type=InstitutionType.MUTUAL_FUND,
        )

        # Create activities with overlapping stocks
        activities1 = [
            InstitutionalActivity(
                activity_id="act_1",
                activity_date=date(2024, 1, 1),
                stock_code="000001",
                institution=inst1,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            ),
            InstitutionalActivity(
                activity_id="act_2",
                activity_date=date(2024, 1, 2),
                stock_code="000002",
                institution=inst1,
                activity_type=ActivityType.BLOCK_TRADE_BUY,
            ),
        ]

        activities2 = [
            InstitutionalActivity(
                activity_id="act_3",
                activity_date=date(2024, 1, 3),
                stock_code="000001",
                institution=inst2,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            ),
            InstitutionalActivity(
                activity_id="act_4",
                activity_date=date(2024, 1, 4),
                stock_code="000003",
                institution=inst2,
                activity_type=ActivityType.SHAREHOLDING_INCREASE,
            ),
        ]

        # Create analytics instance
        mock_collector = Mock()
        analytics = InstitutionalGraphAnalytics(mock_collector)

        # Test relationship detection
        relationship = analytics._detect_same_stock_activity(
            inst1, inst2, activities1, activities2
        )

        assert relationship is not None
        assert relationship.relationship_type == RelationshipType.SAME_STOCK_ACTIVITY
        assert len(relationship.common_stocks) == 1
        assert "000001" in relationship.common_stocks
        assert relationship.activity_overlap_ratio > 0

    def test_temporal_correlation_detection(self):
        """Test temporal correlation detection"""

        # Create test institutions
        inst1 = InstitutionalInvestor(
            institution_id="test_001",
            name="Test Fund 1",
            institution_type=InstitutionType.MUTUAL_FUND,
        )

        inst2 = InstitutionalInvestor(
            institution_id="test_002",
            name="Test Fund 2",
            institution_type=InstitutionType.MUTUAL_FUND,
        )

        # Create correlated activities (same dates)
        base_date = date(2024, 1, 1)
        activities1 = []
        activities2 = []

        for i in range(10):
            # Both institutions active on same dates
            activity_date = base_date + timedelta(days=i)

            activities1.append(
                InstitutionalActivity(
                    activity_id=f"act1_{i}",
                    activity_date=activity_date,
                    stock_code="000001",
                    institution=inst1,
                    activity_type=ActivityType.DRAGON_TIGER_BUY,
                )
            )

            activities2.append(
                InstitutionalActivity(
                    activity_id=f"act2_{i}",
                    activity_date=activity_date,
                    stock_code="000002",
                    institution=inst2,
                    activity_type=ActivityType.DRAGON_TIGER_BUY,
                )
            )

        # Create analytics instance
        mock_collector = Mock()
        analytics = InstitutionalGraphAnalytics(mock_collector)

        # Test correlation detection
        relationship = analytics._detect_temporal_correlation(
            inst1, inst2, activities1, activities2
        )

        assert relationship is not None
        assert relationship.relationship_type == RelationshipType.TEMPORAL_CORRELATION
        assert relationship.temporal_correlation > 0.8  # Should be highly correlated

    def test_activity_correlation_calculation(self):
        """Test activity correlation calculation"""

        # Create test activities
        base_date = date(2024, 1, 1)

        # Perfectly correlated activities (same dates)
        activities1 = [
            InstitutionalActivity(
                activity_id="act1_1",
                activity_date=base_date,
                stock_code="000001",
                institution=Mock(),
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            ),
            InstitutionalActivity(
                activity_id="act1_2",
                activity_date=base_date + timedelta(days=1),
                stock_code="000001",
                institution=Mock(),
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            ),
        ]

        activities2 = [
            InstitutionalActivity(
                activity_id="act2_1",
                activity_date=base_date,
                stock_code="000002",
                institution=Mock(),
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            ),
            InstitutionalActivity(
                activity_id="act2_2",
                activity_date=base_date + timedelta(days=1),
                stock_code="000002",
                institution=Mock(),
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            ),
        ]

        # Create analytics instance
        mock_collector = Mock()
        analytics = InstitutionalGraphAnalytics(mock_collector)

        # Calculate correlation
        correlation = analytics._calculate_activity_correlation(
            activities1, activities2
        )

        assert correlation == 1.0  # Perfect correlation

        # Test with empty activities
        correlation = analytics._calculate_activity_correlation([], activities2)
        assert correlation == 0.0

        # Test with single date
        single_activity = [activities1[0]]
        correlation = analytics._calculate_activity_correlation(
            single_activity, activities2
        )
        assert correlation == 0.0  # Not enough data points


if __name__ == "__main__":
    pytest.main([__file__])
