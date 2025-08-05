"""
Tests for Institutional Attention Scoring System

This module contains comprehensive tests for the institutional attention scoring
functionality including score calculation, behavior pattern detection, and screening.
"""

import asyncio
from collections import defaultdict
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from stock_analysis_system.analysis.institutional_attention_scoring import (
    ActivityIntensity,
    AttentionLevel,
    AttentionScore,
    AttentionScoreCalculator,
    BehaviorPattern,
    InstitutionalAttentionScoringSystem,
    StockAttentionProfile,
)
from stock_analysis_system.analysis.institutional_data_collector import (
    ActivityType,
    InstitutionalActivity,
    InstitutionalDataCollector,
    InstitutionalInvestor,
    InstitutionType,
)
from stock_analysis_system.analysis.institutional_graph_analytics import (
    InstitutionalGraphAnalytics,
)


class TestAttentionScoreCalculator:
    """Test cases for AttentionScoreCalculator"""

    @pytest.fixture
    def calculator(self):
        """Create AttentionScoreCalculator instance"""
        return AttentionScoreCalculator()

    @pytest.fixture
    def sample_institution(self):
        """Create sample institution"""
        return InstitutionalInvestor(
            institution_id="test_fund_001",
            name="测试基金管理有限公司",
            institution_type=InstitutionType.MUTUAL_FUND,
            confidence_score=0.9,
        )

    @pytest.fixture
    def sample_activities(self, sample_institution):
        """Create sample activities for testing"""
        base_date = date(2024, 1, 1)
        activities = []

        # Create varied activities over time
        for i in range(10):
            activity = InstitutionalActivity(
                activity_id=f"test_act_{i}",
                activity_date=base_date + timedelta(days=i * 3),
                stock_code="000001",
                institution=sample_institution,
                activity_type=(
                    ActivityType.DRAGON_TIGER_BUY
                    if i % 2 == 0
                    else ActivityType.DRAGON_TIGER_SELL
                ),
                amount=1000000 * (i + 1),
                volume=100000 * (i + 1),
                source_type="dragon_tiger",
                confidence_score=0.9,
            )
            activities.append(activity)

        return activities

    def test_calculate_attention_score_basic(
        self, calculator, sample_institution, sample_activities
    ):
        """Test basic attention score calculation"""

        score = calculator.calculate_attention_score(
            stock_code="000001",
            institution=sample_institution,
            activities=sample_activities,
            coordination_score=50.0,
            reference_date=date(2024, 2, 1),
        )

        assert isinstance(score, AttentionScore)
        assert score.stock_code == "000001"
        assert score.institution == sample_institution
        assert 0.0 <= score.overall_score <= 100.0
        assert 0.0 <= score.activity_score <= 100.0
        assert 0.0 <= score.recency_score <= 100.0
        assert 0.0 <= score.volume_score <= 100.0
        assert 0.0 <= score.frequency_score <= 100.0
        assert score.coordination_score == 50.0
        assert score.total_activities == len(sample_activities)
        assert isinstance(score.attention_level, AttentionLevel)
        assert isinstance(score.behavior_pattern, BehaviorPattern)
        assert isinstance(score.activity_intensity, ActivityIntensity)

    def test_calculate_attention_score_empty_activities(
        self, calculator, sample_institution
    ):
        """Test attention score calculation with no activities"""

        score = calculator.calculate_attention_score(
            stock_code="000001",
            institution=sample_institution,
            activities=[],
            reference_date=date(2024, 2, 1),
        )

        assert score.overall_score == 0.0
        assert score.activity_score == 0.0
        assert score.recency_score == 0.0
        assert score.volume_score == 0.0
        assert score.frequency_score == 0.0
        assert score.coordination_score == 0.0
        assert score.attention_level == AttentionLevel.VERY_LOW
        assert score.activity_intensity == ActivityIntensity.DORMANT
        assert score.total_activities == 0
        assert score.recent_activities == 0

    def test_calculate_activity_score(self, calculator):
        """Test activity score calculation"""

        # Test with different activity counts
        test_cases = [
            ([], 0.0),
            ([Mock()], 20.0),
            ([Mock()] * 5, 20 + 60 * (np.log(5) / np.log(50))),
            ([Mock()] * 50, 80.0),  # Should be close to max
        ]

        for activities, expected_min in test_cases:
            score = calculator._calculate_activity_score(activities)
            assert 0.0 <= score <= 100.0
            if expected_min > 0:
                assert score >= expected_min * 0.8  # Allow some tolerance

    def test_calculate_recency_score(self, calculator, sample_institution):
        """Test recency score calculation"""

        reference_date = date(2024, 2, 1)

        # Recent activities should score higher
        recent_activities = [
            InstitutionalActivity(
                activity_id="recent_1",
                activity_date=reference_date - timedelta(days=1),
                stock_code="000001",
                institution=sample_institution,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            ),
            InstitutionalActivity(
                activity_id="recent_2",
                activity_date=reference_date - timedelta(days=5),
                stock_code="000001",
                institution=sample_institution,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            ),
        ]

        # Old activities should score lower
        old_activities = [
            InstitutionalActivity(
                activity_id="old_1",
                activity_date=reference_date - timedelta(days=100),
                stock_code="000001",
                institution=sample_institution,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            ),
            InstitutionalActivity(
                activity_id="old_2",
                activity_date=reference_date - timedelta(days=200),
                stock_code="000001",
                institution=sample_institution,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            ),
        ]

        recent_score = calculator._calculate_recency_score(
            recent_activities, reference_date
        )
        old_score = calculator._calculate_recency_score(old_activities, reference_date)

        assert recent_score > old_score
        assert 0.0 <= recent_score <= 100.0
        assert 0.0 <= old_score <= 100.0

    def test_calculate_volume_score(self, calculator, sample_institution):
        """Test volume score calculation"""

        # High volume activities
        high_volume_activities = [
            InstitutionalActivity(
                activity_id="high_vol_1",
                activity_date=date(2024, 1, 1),
                stock_code="000001",
                institution=sample_institution,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
                volume=10000000,  # 10M shares
            )
        ]

        # Low volume activities
        low_volume_activities = [
            InstitutionalActivity(
                activity_id="low_vol_1",
                activity_date=date(2024, 1, 1),
                stock_code="000001",
                institution=sample_institution,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
                volume=50000,  # 50K shares
            )
        ]

        # No volume data
        no_volume_activities = [
            InstitutionalActivity(
                activity_id="no_vol_1",
                activity_date=date(2024, 1, 1),
                stock_code="000001",
                institution=sample_institution,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            )
        ]

        high_score = calculator._calculate_volume_score(high_volume_activities)
        low_score = calculator._calculate_volume_score(low_volume_activities)
        no_vol_score = calculator._calculate_volume_score(no_volume_activities)

        assert high_score > low_score
        assert no_vol_score == 50.0  # Neutral score
        assert 0.0 <= high_score <= 100.0
        assert 0.0 <= low_score <= 100.0

    def test_calculate_frequency_score(self, calculator, sample_institution):
        """Test frequency score calculation"""

        reference_date = date(2024, 2, 1)

        # Regular activities (weekly)
        regular_activities = []
        for i in range(8):  # 8 weeks
            activity = InstitutionalActivity(
                activity_id=f"regular_{i}",
                activity_date=reference_date - timedelta(weeks=i),
                stock_code="000001",
                institution=sample_institution,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            )
            regular_activities.append(activity)

        # Irregular activities (clustered)
        irregular_activities = []
        for i in range(4):
            activity = InstitutionalActivity(
                activity_id=f"irregular_{i}",
                activity_date=reference_date - timedelta(days=i),  # All within 4 days
                stock_code="000001",
                institution=sample_institution,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            )
            irregular_activities.append(activity)

        regular_score = calculator._calculate_frequency_score(
            regular_activities, reference_date
        )
        irregular_score = calculator._calculate_frequency_score(
            irregular_activities, reference_date
        )

        assert regular_score > irregular_score
        assert 0.0 <= regular_score <= 100.0
        assert 0.0 <= irregular_score <= 100.0

    def test_classify_attention_level(self, calculator):
        """Test attention level classification"""

        test_cases = [
            (10.0, AttentionLevel.VERY_LOW),
            (30.0, AttentionLevel.LOW),
            (50.0, AttentionLevel.MODERATE),
            (70.0, AttentionLevel.HIGH),
            (90.0, AttentionLevel.VERY_HIGH),
        ]

        for score, expected_level in test_cases:
            level = calculator._classify_attention_level(score)
            assert level == expected_level

    def test_detect_behavior_pattern(self, calculator, sample_institution):
        """Test behavior pattern detection"""

        reference_date = date(2024, 1, 1)

        # Accumulating pattern (mostly buy activities)
        accumulating_activities = []
        for i in range(5):
            activity = InstitutionalActivity(
                activity_id=f"acc_{i}",
                activity_date=reference_date + timedelta(days=i * 7),
                stock_code="000001",
                institution=sample_institution,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            )
            accumulating_activities.append(activity)

        # Distributing pattern (mostly sell activities)
        distributing_activities = []
        for i in range(5):
            activity = InstitutionalActivity(
                activity_id=f"dist_{i}",
                activity_date=reference_date + timedelta(days=i * 7),
                stock_code="000001",
                institution=sample_institution,
                activity_type=ActivityType.DRAGON_TIGER_SELL,
            )
            distributing_activities.append(activity)

        # Swing trading pattern (short time frame, many activities)
        swing_activities = []
        for i in range(8):
            activity = InstitutionalActivity(
                activity_id=f"swing_{i}",
                activity_date=reference_date + timedelta(days=i * 2),  # Every 2 days
                stock_code="000001",
                institution=sample_institution,
                activity_type=(
                    ActivityType.DRAGON_TIGER_BUY
                    if i % 2 == 0
                    else ActivityType.DRAGON_TIGER_SELL
                ),
            )
            swing_activities.append(activity)

        acc_pattern = calculator._detect_behavior_pattern(accumulating_activities)
        dist_pattern = calculator._detect_behavior_pattern(distributing_activities)
        swing_pattern = calculator._detect_behavior_pattern(swing_activities)

        assert acc_pattern == BehaviorPattern.ACCUMULATING
        assert dist_pattern == BehaviorPattern.DISTRIBUTING
        assert swing_pattern == BehaviorPattern.SWING_TRADING

    def test_is_buy_sell_activity(self, calculator):
        """Test buy/sell activity classification"""

        buy_types = [
            ActivityType.DRAGON_TIGER_BUY,
            ActivityType.BLOCK_TRADE_BUY,
            ActivityType.SHAREHOLDING_INCREASE,
            ActivityType.NEW_POSITION,
        ]

        sell_types = [
            ActivityType.DRAGON_TIGER_SELL,
            ActivityType.BLOCK_TRADE_SELL,
            ActivityType.SHAREHOLDING_DECREASE,
            ActivityType.POSITION_EXIT,
        ]

        for buy_type in buy_types:
            assert calculator._is_buy_activity(buy_type)
            assert not calculator._is_sell_activity(buy_type)

        for sell_type in sell_types:
            assert calculator._is_sell_activity(sell_type)
            assert not calculator._is_buy_activity(sell_type)

    def test_classify_activity_intensity(self, calculator):
        """Test activity intensity classification"""

        test_cases = [
            ([], ActivityIntensity.DORMANT),
            ([Mock()], ActivityIntensity.LIGHT),
            ([Mock()] * 5, ActivityIntensity.MODERATE),
            ([Mock()] * 15, ActivityIntensity.HEAVY),
            ([Mock()] * 25, ActivityIntensity.EXTREME),
        ]

        for activities, expected_intensity in test_cases:
            intensity = calculator._classify_activity_intensity(activities)
            assert intensity == expected_intensity

    def test_calculate_trend_consistency(self, calculator, sample_institution):
        """Test trend consistency calculation"""

        # All buy activities (positive consistency)
        buy_activities = []
        for i in range(5):
            activity = InstitutionalActivity(
                activity_id=f"buy_{i}",
                activity_date=date(2024, 1, 1) + timedelta(days=i),
                stock_code="000001",
                institution=sample_institution,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            )
            buy_activities.append(activity)

        # All sell activities (negative consistency)
        sell_activities = []
        for i in range(5):
            activity = InstitutionalActivity(
                activity_id=f"sell_{i}",
                activity_date=date(2024, 1, 1) + timedelta(days=i),
                stock_code="000001",
                institution=sample_institution,
                activity_type=ActivityType.DRAGON_TIGER_SELL,
            )
            sell_activities.append(activity)

        # Mixed activities (neutral consistency)
        mixed_activities = buy_activities[:3] + sell_activities[:2]

        buy_consistency = calculator._calculate_trend_consistency(buy_activities)
        sell_consistency = calculator._calculate_trend_consistency(sell_activities)
        mixed_consistency = calculator._calculate_trend_consistency(mixed_activities)

        assert buy_consistency == 1.0
        assert sell_consistency == -1.0
        assert -1.0 <= mixed_consistency <= 1.0
        assert abs(mixed_consistency) < 1.0  # Should be between extremes

    def test_calculate_activity_regularity(self, calculator, sample_institution):
        """Test activity regularity calculation"""

        # Regular activities (same interval)
        regular_activities = []
        for i in range(5):
            activity = InstitutionalActivity(
                activity_id=f"regular_{i}",
                activity_date=date(2024, 1, 1) + timedelta(days=i * 7),  # Weekly
                stock_code="000001",
                institution=sample_institution,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            )
            regular_activities.append(activity)

        # Irregular activities (random intervals)
        irregular_activities = []
        intervals = [1, 5, 2, 15, 3]  # Irregular intervals
        current_date = date(2024, 1, 1)
        for i, interval in enumerate(intervals):
            current_date += timedelta(days=interval)
            activity = InstitutionalActivity(
                activity_id=f"irregular_{i}",
                activity_date=current_date,
                stock_code="000001",
                institution=sample_institution,
                activity_type=ActivityType.DRAGON_TIGER_BUY,
            )
            irregular_activities.append(activity)

        regular_score = calculator._calculate_activity_regularity(regular_activities)
        irregular_score = calculator._calculate_activity_regularity(
            irregular_activities
        )

        assert regular_score > irregular_score
        assert 0.0 <= regular_score <= 1.0
        assert 0.0 <= irregular_score <= 1.0


class TestInstitutionalAttentionScoringSystem:
    """Test cases for InstitutionalAttentionScoringSystem"""

    @pytest.fixture
    def mock_data_collector(self):
        """Create mock data collector"""
        collector = Mock(spec=InstitutionalDataCollector)

        # Create sample institution
        institution = InstitutionalInvestor(
            institution_id="test_fund_001",
            name="测试基金管理有限公司",
            institution_type=InstitutionType.MUTUAL_FUND,
            confidence_score=0.9,
        )

        # Create sample activities
        activities = []
        base_date = date(2024, 1, 1)

        for i in range(10):
            activity = InstitutionalActivity(
                activity_id=f"test_act_{i}",
                activity_date=base_date + timedelta(days=i * 3),
                stock_code="000001",
                institution=institution,
                activity_type=(
                    ActivityType.DRAGON_TIGER_BUY
                    if i % 2 == 0
                    else ActivityType.DRAGON_TIGER_SELL
                ),
                amount=1000000 * (i + 1),
                volume=100000 * (i + 1),
                source_type="dragon_tiger",
                confidence_score=0.9,
            )
            activities.append(activity)

        # Mock activity timeline
        collector.activity_timeline = {"000001": activities}

        # Mock collect_all_data method
        async def mock_collect_all_data(stock_codes, start_date, end_date):
            return {
                "000001": {
                    "dragon_tiger": activities,
                    "shareholders": [],
                    "block_trades": [],
                }
            }

        collector.collect_all_data = AsyncMock(side_effect=mock_collect_all_data)

        return collector

    @pytest.fixture
    def mock_graph_analytics(self):
        """Create mock graph analytics"""
        analytics = Mock(spec=InstitutionalGraphAnalytics)

        # Mock build_institutional_network
        async def mock_build_network(stock_codes, start_date, end_date):
            return Mock()

        analytics.build_institutional_network = AsyncMock(
            side_effect=mock_build_network
        )

        # Mock get_institution_relationships
        def mock_get_relationships(institution_id):
            # Return mock relationships with varying strengths
            mock_rel = Mock()
            mock_rel.strength_score = 0.7
            return [mock_rel]

        analytics.get_institution_relationships = Mock(
            side_effect=mock_get_relationships
        )

        return analytics

    @pytest.fixture
    def scoring_system(self, mock_data_collector, mock_graph_analytics):
        """Create InstitutionalAttentionScoringSystem instance"""
        return InstitutionalAttentionScoringSystem(
            data_collector=mock_data_collector, graph_analytics=mock_graph_analytics
        )

    @pytest.mark.asyncio
    async def test_calculate_stock_attention_profile(self, scoring_system):
        """Test stock attention profile calculation"""

        stock_code = "000001"
        start_date = date(2024, 1, 1)
        end_date = date(2024, 2, 1)

        profile = await scoring_system.calculate_stock_attention_profile(
            stock_code, start_date, end_date
        )

        assert isinstance(profile, StockAttentionProfile)
        assert profile.stock_code == stock_code
        assert profile.institutional_count > 0
        assert 0.0 <= profile.total_attention_score <= 100.0
        assert len(profile.institution_scores) > 0
        assert profile.calculation_date is not None

        # Check that scores are properly calculated
        for score in profile.institution_scores:
            assert isinstance(score, AttentionScore)
            assert 0.0 <= score.overall_score <= 100.0
            assert isinstance(score.attention_level, AttentionLevel)
            assert isinstance(score.behavior_pattern, BehaviorPattern)

    @pytest.mark.asyncio
    async def test_calculate_stock_attention_profile_empty_data(self, scoring_system):
        """Test attention profile calculation with no data"""

        # Mock empty data
        scoring_system.data_collector.activity_timeline = {}

        profile = await scoring_system.calculate_stock_attention_profile(
            "000002", date(2024, 1, 1), date(2024, 2, 1)
        )

        assert profile.stock_code == "000002"
        assert profile.institutional_count == 0
        assert profile.total_attention_score == 0.0
        assert len(profile.institution_scores) == 0

    @pytest.mark.asyncio
    async def test_screen_stocks_by_attention(self, scoring_system):
        """Test stock screening by attention criteria"""

        stock_codes = ["000001", "000002"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 2, 1)

        # Custom criteria
        criteria = {
            "high_attention": 30.0,  # Lower threshold for testing
            "coordinated_activity": 0.3,
            "recent_activity_days": 7,
        }

        results = await scoring_system.screen_stocks_by_attention(
            stock_codes, start_date, end_date, criteria
        )

        assert isinstance(results, list)

        # Check result structure
        for result in results:
            assert "stock_code" in result
            assert "total_attention_score" in result
            assert "institutional_count" in result
            assert "active_institutional_count" in result
            assert "top_institutions" in result
            assert "screening_reasons" in result

            # Verify screening criteria are met
            assert result["total_attention_score"] >= criteria["high_attention"]

    def test_meets_screening_criteria(self, scoring_system):
        """Test screening criteria evaluation"""

        # Create mock profile
        profile = Mock(spec=StockAttentionProfile)
        profile.total_attention_score = 75.0
        profile.coordination_score = 0.6
        profile.active_institutional_count = 3
        profile.institutional_count = 5
        profile.activity_trend = 0.3
        profile.dominant_patterns = [(BehaviorPattern.ACCUMULATING, 3)]

        # Test various criteria
        criteria_pass = {
            "high_attention": 70.0,
            "coordinated_activity": 0.5,
            "min_institutions": 3,
        }

        criteria_fail = {
            "high_attention": 80.0,  # Too high
            "coordinated_activity": 0.7,  # Too high
            "min_institutions": 10,  # Too high
        }

        assert scoring_system._meets_screening_criteria(profile, criteria_pass)
        assert not scoring_system._meets_screening_criteria(profile, criteria_fail)

    def test_get_screening_reasons(self, scoring_system):
        """Test screening reasons generation"""

        profile = Mock(spec=StockAttentionProfile)
        profile.total_attention_score = 75.0
        profile.coordination_score = 0.6
        profile.active_institutional_count = 3
        profile.activity_trend = 0.3
        profile.dominant_patterns = [(BehaviorPattern.ACCUMULATING, 3)]

        criteria = {"high_attention": 70.0, "coordinated_activity": 0.5}

        reasons = scoring_system._get_screening_reasons(profile, criteria)

        assert isinstance(reasons, list)
        assert len(reasons) > 0

        # Check that reasons contain expected information
        reasons_text = " ".join(reasons)
        assert "attention score" in reasons_text.lower()
        assert "coordinated activity" in reasons_text.lower()

    def test_generate_attention_alerts(self, scoring_system):
        """Test attention alert generation"""

        # Create mock profile with high attention
        high_attention_profile = Mock(spec=StockAttentionProfile)
        high_attention_profile.total_attention_score = 85.0
        high_attention_profile.institutional_count = 5
        high_attention_profile.active_institutional_count = 3
        high_attention_profile.coordination_score = 0.7
        high_attention_profile.activity_trend = 0.4
        high_attention_profile.dominant_patterns = [(BehaviorPattern.COORDINATED, 3)]
        high_attention_profile.data_quality_score = 0.9

        # Add to cached profiles
        scoring_system.stock_profiles = {"000001": high_attention_profile}

        alerts = scoring_system.generate_attention_alerts(
            ["000001"], date(2024, 1, 1), date(2024, 2, 1), alert_threshold=60.0
        )

        assert isinstance(alerts, list)
        assert len(alerts) > 0

        alert = alerts[0]
        assert alert["stock_code"] == "000001"
        assert alert["attention_score"] == 85.0
        assert "alert_type" in alert
        assert "priority" in alert
        assert "message" in alert
        assert "timestamp" in alert

    def test_classify_alert(self, scoring_system):
        """Test alert classification"""

        # High coordination profile
        coord_profile = Mock(spec=StockAttentionProfile)
        coord_profile.total_attention_score = 80.0
        coord_profile.coordination_score = 0.8
        coord_profile.activity_trend = 0.2

        alert_type, priority = scoring_system._classify_alert(coord_profile)
        assert alert_type == "coordinated_activity"
        assert priority in ["high", "medium", "low"]

        # High trend profile
        trend_profile = Mock(spec=StockAttentionProfile)
        trend_profile.total_attention_score = 75.0
        trend_profile.coordination_score = 0.3
        trend_profile.activity_trend = 0.6

        alert_type, priority = scoring_system._classify_alert(trend_profile)
        assert alert_type == "increasing_attention"

        # Very high attention profile
        high_profile = Mock(spec=StockAttentionProfile)
        high_profile.total_attention_score = 95.0
        high_profile.coordination_score = 0.3
        high_profile.activity_trend = 0.2

        alert_type, priority = scoring_system._classify_alert(high_profile)
        assert alert_type == "very_high_attention"

    def test_generate_alert_message(self, scoring_system):
        """Test alert message generation"""

        profile = Mock(spec=StockAttentionProfile)
        profile.stock_code = "000001"
        profile.total_attention_score = 85.0
        profile.institutional_count = 5
        profile.active_institutional_count = 3
        profile.dominant_patterns = [(BehaviorPattern.ACCUMULATING, 3)]
        profile.coordination_score = 0.7
        profile.activity_trend = 0.4

        message = scoring_system._generate_alert_message(profile)

        assert isinstance(message, str)
        assert len(message) > 0
        assert "000001" in message
        assert "85.1" in message  # Score should be included
        assert "5 institutions" in message
        assert "3 recently active" in message

    def test_get_institution_attention_summary(self, scoring_system):
        """Test institution attention summary"""

        # Create mock attention scores
        institution = InstitutionalInvestor(
            institution_id="test_fund_001",
            name="测试基金管理有限公司",
            institution_type=InstitutionType.MUTUAL_FUND,
        )

        scores = []
        for i in range(3):
            score = Mock(spec=AttentionScore)
            score.institution = institution
            score.overall_score = 70.0 + i * 10
            score.behavior_pattern = BehaviorPattern.ACCUMULATING
            score.total_activities = 5 + i
            score.recent_activities = 2 + i
            score.stock_code = f"00000{i+1}"
            scores.append(score)

        # Add to cached scores
        for i, score in enumerate(scores):
            scoring_system.institution_scores[("test_fund_001", f"00000{i+1}")] = score

        summary = scoring_system.get_institution_attention_summary("test_fund_001")

        assert "institution_id" in summary
        assert "institution_name" in summary
        assert "institution_type" in summary
        assert "average_attention_score" in summary
        assert "total_stocks_tracked" in summary
        assert "high_attention_stocks" in summary
        assert "total_activities" in summary
        assert "recent_activities" in summary
        assert "dominant_patterns" in summary
        assert "top_stocks" in summary

        assert summary["institution_id"] == "test_fund_001"
        assert summary["total_stocks_tracked"] == 3
        assert summary["average_attention_score"] > 0

    def test_get_institution_attention_summary_no_data(self, scoring_system):
        """Test institution summary with no data"""

        summary = scoring_system.get_institution_attention_summary("nonexistent")

        assert "error" in summary
        assert summary["error"] == "No attention data found for institution"


class TestDataStructures:
    """Test cases for data structures and enums"""

    def test_attention_level_enum(self):
        """Test AttentionLevel enum values"""

        levels = list(AttentionLevel)
        expected_levels = ["very_low", "low", "moderate", "high", "very_high"]

        assert len(levels) == len(expected_levels)
        for level in levels:
            assert level.value in expected_levels

    def test_behavior_pattern_enum(self):
        """Test BehaviorPattern enum values"""

        patterns = list(BehaviorPattern)
        expected_patterns = [
            "accumulating",
            "distributing",
            "momentum_following",
            "contrarian",
            "swing_trading",
            "long_term_holding",
            "coordinated",
            "opportunistic",
        ]

        assert len(patterns) == len(expected_patterns)
        for pattern in patterns:
            assert pattern.value in expected_patterns

    def test_activity_intensity_enum(self):
        """Test ActivityIntensity enum values"""

        intensities = list(ActivityIntensity)
        expected_intensities = ["dormant", "light", "moderate", "heavy", "extreme"]

        assert len(intensities) == len(expected_intensities)
        for intensity in intensities:
            assert intensity.value in expected_intensities

    def test_attention_score_dataclass(self):
        """Test AttentionScore dataclass"""

        institution = InstitutionalInvestor(
            institution_id="test_001",
            name="Test Institution",
            institution_type=InstitutionType.MUTUAL_FUND,
        )

        score = AttentionScore(
            stock_code="000001",
            institution=institution,
            overall_score=75.0,
            activity_score=70.0,
            recency_score=80.0,
            volume_score=75.0,
            frequency_score=70.0,
            coordination_score=60.0,
            attention_level=AttentionLevel.HIGH,
            behavior_pattern=BehaviorPattern.ACCUMULATING,
            activity_intensity=ActivityIntensity.MODERATE,
            total_activities=10,
            recent_activities=3,
        )

        assert score.stock_code == "000001"
        assert score.institution == institution
        assert score.overall_score == 75.0
        assert score.attention_level == AttentionLevel.HIGH
        assert score.behavior_pattern == BehaviorPattern.ACCUMULATING
        assert score.activity_intensity == ActivityIntensity.MODERATE
        assert isinstance(score.calculation_date, datetime)

    def test_stock_attention_profile_dataclass(self):
        """Test StockAttentionProfile dataclass"""

        profile = StockAttentionProfile(
            stock_code="000001",
            total_attention_score=75.0,
            institutional_count=5,
            active_institutional_count=3,
        )

        assert profile.stock_code == "000001"
        assert profile.total_attention_score == 75.0
        assert profile.institutional_count == 5
        assert profile.active_institutional_count == 3
        assert isinstance(profile.calculation_date, datetime)
        assert isinstance(profile.institution_scores, list)
        assert isinstance(profile.dominant_patterns, list)
        assert isinstance(profile.attention_distribution, dict)


if __name__ == "__main__":
    pytest.main([__file__])
