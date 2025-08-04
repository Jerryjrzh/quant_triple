"""
Institutional Attention Scoring System

This module implements comprehensive institutional attention scoring including
time-weighted scoring, behavior pattern classification, and integration with
stock screening and alert systems for the stock analysis system.

Requirements addressed: 3.1, 3.2, 3.3, 3.4
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
import json
import logging
from collections import defaultdict, Counter
import math
from abc import ABC, abstractmethod

from .institutional_data_collector import (
    InstitutionalDataCollector, InstitutionalInvestor, InstitutionalActivity,
    ActivityType, InstitutionType
)
from .institutional_graph_analytics import InstitutionalGraphAnalytics

logger = logging.getLogger(__name__)

class AttentionLevel(str, Enum):
    """Levels of institutional attention"""
    VERY_LOW = "very_low"      # 0-20
    LOW = "low"                # 21-40
    MODERATE = "moderate"      # 41-60
    HIGH = "high"              # 61-80
    VERY_HIGH = "very_high"    # 81-100

class BehaviorPattern(str, Enum):
    """Institutional behavior patterns"""
    ACCUMULATING = "accumulating"           # Consistent buying over time
    DISTRIBUTING = "distributing"           # Consistent selling over time
    MOMENTUM_FOLLOWING = "momentum_following" # Following price trends
    CONTRARIAN = "contrarian"               # Against price trends
    SWING_TRADING = "swing_trading"         # Short-term trading
    LONG_TERM_HOLDING = "long_term_holding" # Minimal trading activity
    COORDINATED = "coordinated"             # Acting with other institutions
    OPPORTUNISTIC = "opportunistic"         # Irregular, event-driven activity

class ActivityIntensity(str, Enum):
    """Activity intensity levels"""
    DORMANT = "dormant"         # No recent activity
    LIGHT = "light"             # Minimal activity
    MODERATE = "moderate"       # Regular activity
    HEAVY = "heavy"             # High activity
    EXTREME = "extreme"         # Exceptional activity

@dataclass
class AttentionScore:
    """Institutional attention score for a specific stock"""
    stock_code: str
    institution: InstitutionalInvestor
    
    # Core scores (0-100 scale)
    overall_score: float
    activity_score: float
    recency_score: float
    volume_score: float
    frequency_score: float
    coordination_score: float
    
    # Classification
    attention_level: AttentionLevel
    behavior_pattern: BehaviorPattern
    activity_intensity: ActivityIntensity
    
    # Supporting metrics
    total_activities: int
    recent_activities: int  # Last 30 days
    total_volume: Optional[int] = None
    total_amount: Optional[float] = None
    avg_activity_size: Optional[float] = None
    
    # Time-based metrics
    first_activity_date: Optional[date] = None
    last_activity_date: Optional[date] = None
    activity_span_days: int = 0
    
    # Pattern characteristics
    trend_consistency: float = 0.0  # -1 to 1 (sell to buy consistency)
    activity_regularity: float = 0.0  # 0 to 1 (irregular to regular)
    coordination_strength: float = 0.0  # 0 to 1
    
    # Metadata
    calculation_date: datetime = field(default_factory=datetime.now)
    confidence_score: float = 1.0

@dataclass
class StockAttentionProfile:
    """Comprehensive attention profile for a stock"""
    stock_code: str
    
    # Aggregate scores
    total_attention_score: float  # 0-100
    institutional_count: int
    active_institutional_count: int  # Active in last 30 days
    
    # Institution breakdown
    institution_scores: List[AttentionScore] = field(default_factory=list)
    
    # Pattern analysis
    dominant_patterns: List[Tuple[BehaviorPattern, int]] = field(default_factory=list)
    attention_distribution: Dict[AttentionLevel, int] = field(default_factory=dict)
    
    # Activity metrics
    total_activities: int = 0
    recent_activities: int = 0
    activity_trend: float = 0.0  # -1 to 1 (decreasing to increasing)
    
    # Institution type breakdown
    type_distribution: Dict[InstitutionType, int] = field(default_factory=dict)
    type_attention_scores: Dict[InstitutionType, float] = field(default_factory=dict)
    
    # Coordination metrics
    coordination_clusters: List[List[str]] = field(default_factory=list)  # Institution ID clusters
    coordination_score: float = 0.0
    
    # Metadata
    calculation_date: datetime = field(default_factory=datetime.now)
    data_quality_score: float = 1.0

class AttentionScoreCalculator:
    """Calculator for institutional attention scores"""
    
    def __init__(self):
        # Scoring weights
        self.weights = {
            'activity': 0.25,      # Raw activity count
            'recency': 0.30,       # Time decay factor
            'volume': 0.20,        # Transaction volume
            'frequency': 0.15,     # Activity frequency
            'coordination': 0.10   # Coordination with others
        }
        
        # Time decay parameters
        self.recency_half_life = 30  # Days for 50% weight decay
        self.max_lookback_days = 365  # Maximum historical lookback
        
        # Activity thresholds
        self.activity_thresholds = {
            ActivityIntensity.DORMANT: 0,
            ActivityIntensity.LIGHT: 1,
            ActivityIntensity.MODERATE: 5,
            ActivityIntensity.HEAVY: 15,
            ActivityIntensity.EXTREME: 30
        }
        
        # Pattern detection parameters
        self.pattern_min_activities = 3
        self.coordination_threshold = 0.3
    
    def calculate_attention_score(self, 
                                stock_code: str,
                                institution: InstitutionalInvestor,
                                activities: List[InstitutionalActivity],
                                coordination_score: float = 0.0,
                                reference_date: Optional[date] = None) -> AttentionScore:
        """
        Calculate comprehensive attention score for an institution-stock pair.
        
        Args:
            stock_code: Stock code
            institution: Institution object
            activities: List of activities for this institution-stock pair
            coordination_score: Coordination score from graph analytics
            reference_date: Reference date for calculations (default: today)
            
        Returns:
            AttentionScore object with comprehensive metrics
        """
        
        if reference_date is None:
            reference_date = date.today()
        
        if not activities:
            return self._create_zero_score(stock_code, institution, reference_date)
        
        # Filter activities within lookback period
        cutoff_date = reference_date - timedelta(days=self.max_lookback_days)
        relevant_activities = [
            a for a in activities 
            if a.activity_date >= cutoff_date
        ]
        
        if not relevant_activities:
            return self._create_zero_score(stock_code, institution, reference_date)
        
        # Calculate component scores
        activity_score = self._calculate_activity_score(relevant_activities)
        recency_score = self._calculate_recency_score(relevant_activities, reference_date)
        volume_score = self._calculate_volume_score(relevant_activities)
        frequency_score = self._calculate_frequency_score(relevant_activities, reference_date)
        
        # Calculate overall score
        overall_score = (
            activity_score * self.weights['activity'] +
            recency_score * self.weights['recency'] +
            volume_score * self.weights['volume'] +
            frequency_score * self.weights['frequency'] +
            coordination_score * self.weights['coordination']
        )
        
        # Classify attention level
        attention_level = self._classify_attention_level(overall_score)
        
        # Detect behavior pattern
        behavior_pattern = self._detect_behavior_pattern(relevant_activities)
        
        # Classify activity intensity
        recent_activities = [
            a for a in relevant_activities 
            if (reference_date - a.activity_date).days <= 30
        ]
        activity_intensity = self._classify_activity_intensity(recent_activities)
        
        # Calculate supporting metrics
        total_volume = sum(a.volume for a in relevant_activities if a.volume)
        total_amount = sum(a.amount for a in relevant_activities if a.amount)
        avg_activity_size = total_amount / len(relevant_activities) if total_amount else None
        
        # Time-based metrics
        activity_dates = [a.activity_date for a in relevant_activities]
        first_activity_date = min(activity_dates)
        last_activity_date = max(activity_dates)
        activity_span_days = (last_activity_date - first_activity_date).days
        
        # Pattern characteristics
        trend_consistency = self._calculate_trend_consistency(relevant_activities)
        activity_regularity = self._calculate_activity_regularity(relevant_activities)
        
        return AttentionScore(
            stock_code=stock_code,
            institution=institution,
            overall_score=overall_score,
            activity_score=activity_score,
            recency_score=recency_score,
            volume_score=volume_score,
            frequency_score=frequency_score,
            coordination_score=coordination_score,
            attention_level=attention_level,
            behavior_pattern=behavior_pattern,
            activity_intensity=activity_intensity,
            total_activities=len(relevant_activities),
            recent_activities=len(recent_activities),
            total_volume=total_volume,
            total_amount=total_amount,
            avg_activity_size=avg_activity_size,
            first_activity_date=first_activity_date,
            last_activity_date=last_activity_date,
            activity_span_days=activity_span_days,
            trend_consistency=trend_consistency,
            activity_regularity=activity_regularity,
            coordination_strength=coordination_score,
            calculation_date=datetime.now(),
            confidence_score=min(institution.confidence_score, 1.0)
        )
    
    def _create_zero_score(self, stock_code: str, institution: InstitutionalInvestor, 
                          reference_date: date) -> AttentionScore:
        """Create a zero attention score for institutions with no activity"""
        
        return AttentionScore(
            stock_code=stock_code,
            institution=institution,
            overall_score=0.0,
            activity_score=0.0,
            recency_score=0.0,
            volume_score=0.0,
            frequency_score=0.0,
            coordination_score=0.0,
            attention_level=AttentionLevel.VERY_LOW,
            behavior_pattern=BehaviorPattern.LONG_TERM_HOLDING,
            activity_intensity=ActivityIntensity.DORMANT,
            total_activities=0,
            recent_activities=0,
            calculation_date=datetime.now(),
            confidence_score=institution.confidence_score
        )
    
    def _calculate_activity_score(self, activities: List[InstitutionalActivity]) -> float:
        """Calculate activity score based on number of activities"""
        
        activity_count = len(activities)
        
        # Logarithmic scaling to prevent extreme scores
        if activity_count == 0:
            return 0.0
        elif activity_count == 1:
            return 20.0
        else:
            # Scale logarithmically, max out around 50 activities
            score = 20 + 60 * (math.log(activity_count) / math.log(50))
            return min(score, 100.0)
    
    def _calculate_recency_score(self, activities: List[InstitutionalActivity], 
                               reference_date: date) -> float:
        """Calculate recency score with exponential time decay"""
        
        if not activities:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for activity in activities:
            days_ago = (reference_date - activity.activity_date).days
            
            # Exponential decay: weight = 0.5^(days_ago / half_life)
            weight = 0.5 ** (days_ago / self.recency_half_life)
            
            # Base activity value (can be enhanced with activity importance)
            activity_value = 1.0
            
            total_weighted_score += activity_value * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Normalize and scale to 0-100
        avg_weighted_score = total_weighted_score / total_weight
        return min(avg_weighted_score * 100, 100.0)
    
    def _calculate_volume_score(self, activities: List[InstitutionalActivity]) -> float:
        """Calculate volume score based on transaction volumes"""
        
        volumes = [a.volume for a in activities if a.volume is not None and a.volume > 0]
        amounts = [a.amount for a in activities if a.amount is not None and a.amount > 0]
        
        if not volumes and not amounts:
            return 50.0  # Neutral score when volume data is unavailable
        
        # Use volume if available, otherwise use amount
        values = volumes if volumes else amounts
        
        if not values:
            return 50.0
        
        total_value = sum(values)
        avg_value = total_value / len(values)
        
        # Logarithmic scaling for volume/amount
        if total_value <= 0:
            return 0.0
        
        # Scale based on total value (adjust thresholds as needed)
        if volumes:
            # Volume-based scoring (shares)
            if total_value < 100000:  # < 100K shares
                score = 20.0
            elif total_value < 1000000:  # < 1M shares
                score = 40.0
            elif total_value < 10000000:  # < 10M shares
                score = 60.0
            elif total_value < 50000000:  # < 50M shares
                score = 80.0
            else:
                score = 100.0
        else:
            # Amount-based scoring (yuan)
            if total_value < 1000000:  # < 1M yuan
                score = 20.0
            elif total_value < 10000000:  # < 10M yuan
                score = 40.0
            elif total_value < 100000000:  # < 100M yuan
                score = 60.0
            elif total_value < 500000000:  # < 500M yuan
                score = 80.0
            else:
                score = 100.0
        
        return score
    
    def _calculate_frequency_score(self, activities: List[InstitutionalActivity], 
                                 reference_date: date) -> float:
        """Calculate frequency score based on activity distribution over time"""
        
        if len(activities) < 2:
            return 20.0 if activities else 0.0
        
        # Group activities by week
        activity_weeks = defaultdict(int)
        
        for activity in activities:
            # Calculate week number from reference date
            days_ago = (reference_date - activity.activity_date).days
            week_number = days_ago // 7
            activity_weeks[week_number] += 1
        
        if not activity_weeks:
            return 0.0
        
        # Calculate frequency metrics
        total_weeks = max(activity_weeks.keys()) + 1
        active_weeks = len(activity_weeks)
        
        # Frequency ratio (active weeks / total weeks)
        frequency_ratio = active_weeks / total_weeks if total_weeks > 0 else 0.0
        
        # Consistency score (lower variance in weekly activity = higher score)
        weekly_counts = list(activity_weeks.values())
        if len(weekly_counts) > 1:
            mean_count = np.mean(weekly_counts)
            std_count = np.std(weekly_counts)
            consistency = 1.0 - min(std_count / mean_count, 1.0) if mean_count > 0 else 0.0
        else:
            consistency = 1.0
        
        # Combined frequency score
        frequency_score = (frequency_ratio * 0.7 + consistency * 0.3) * 100
        
        return min(frequency_score, 100.0)
    
    def _classify_attention_level(self, overall_score: float) -> AttentionLevel:
        """Classify attention level based on overall score"""
        
        if overall_score <= 20:
            return AttentionLevel.VERY_LOW
        elif overall_score <= 40:
            return AttentionLevel.LOW
        elif overall_score <= 60:
            return AttentionLevel.MODERATE
        elif overall_score <= 80:
            return AttentionLevel.HIGH
        else:
            return AttentionLevel.VERY_HIGH
    
    def _detect_behavior_pattern(self, activities: List[InstitutionalActivity]) -> BehaviorPattern:
        """Detect institutional behavior pattern from activities"""
        
        if len(activities) < self.pattern_min_activities:
            return BehaviorPattern.OPPORTUNISTIC
        
        # Analyze activity types
        buy_activities = [a for a in activities if self._is_buy_activity(a.activity_type)]
        sell_activities = [a for a in activities if self._is_sell_activity(a.activity_type)]
        
        buy_ratio = len(buy_activities) / len(activities)
        sell_ratio = len(sell_activities) / len(activities)
        
        # Analyze time distribution
        activity_dates = [a.activity_date for a in activities]
        date_range = (max(activity_dates) - min(activity_dates)).days
        
        # Pattern detection logic
        if buy_ratio > 0.8:
            return BehaviorPattern.ACCUMULATING
        elif sell_ratio > 0.8:
            return BehaviorPattern.DISTRIBUTING
        elif date_range <= 30 and len(activities) >= 5:
            return BehaviorPattern.SWING_TRADING
        elif date_range > 180 and len(activities) < 10:
            return BehaviorPattern.LONG_TERM_HOLDING
        elif abs(buy_ratio - sell_ratio) < 0.2:
            # Balanced buy/sell activity
            if self._has_momentum_pattern(activities):
                return BehaviorPattern.MOMENTUM_FOLLOWING
            else:
                return BehaviorPattern.CONTRARIAN
        else:
            return BehaviorPattern.OPPORTUNISTIC
    
    def _is_buy_activity(self, activity_type: ActivityType) -> bool:
        """Check if activity type represents buying"""
        return activity_type in [
            ActivityType.DRAGON_TIGER_BUY,
            ActivityType.BLOCK_TRADE_BUY,
            ActivityType.SHAREHOLDING_INCREASE,
            ActivityType.NEW_POSITION
        ]
    
    def _is_sell_activity(self, activity_type: ActivityType) -> bool:
        """Check if activity type represents selling"""
        return activity_type in [
            ActivityType.DRAGON_TIGER_SELL,
            ActivityType.BLOCK_TRADE_SELL,
            ActivityType.SHAREHOLDING_DECREASE,
            ActivityType.POSITION_EXIT
        ]
    
    def _has_momentum_pattern(self, activities: List[InstitutionalActivity]) -> bool:
        """Check if activities follow momentum pattern (simplified)"""
        
        # This is a simplified implementation
        # In practice, would need price data to determine momentum
        
        # For now, check if activities are clustered in time (momentum-like)
        activity_dates = sorted([a.activity_date for a in activities])
        
        if len(activity_dates) < 3:
            return False
        
        # Check for clustering (activities within short time periods)
        clusters = 0
        current_cluster_start = activity_dates[0]
        
        for i in range(1, len(activity_dates)):
            if (activity_dates[i] - current_cluster_start).days > 14:  # New cluster
                clusters += 1
                current_cluster_start = activity_dates[i]
        
        # Momentum pattern if activities are clustered
        return clusters >= 2
    
    def _classify_activity_intensity(self, recent_activities: List[InstitutionalActivity]) -> ActivityIntensity:
        """Classify activity intensity based on recent activities"""
        
        activity_count = len(recent_activities)
        
        if activity_count == 0:
            return ActivityIntensity.DORMANT
        elif activity_count <= 2:
            return ActivityIntensity.LIGHT
        elif activity_count <= 8:
            return ActivityIntensity.MODERATE
        elif activity_count <= 20:
            return ActivityIntensity.HEAVY
        else:
            return ActivityIntensity.EXTREME
    
    def _calculate_trend_consistency(self, activities: List[InstitutionalActivity]) -> float:
        """Calculate trend consistency (-1 to 1, sell to buy bias)"""
        
        if not activities:
            return 0.0
        
        buy_count = sum(1 for a in activities if self._is_buy_activity(a.activity_type))
        sell_count = sum(1 for a in activities if self._is_sell_activity(a.activity_type))
        total_directional = buy_count + sell_count
        
        if total_directional == 0:
            return 0.0
        
        # Calculate bias: -1 (all sell) to +1 (all buy)
        bias = (buy_count - sell_count) / total_directional
        
        return bias
    
    def _calculate_activity_regularity(self, activities: List[InstitutionalActivity]) -> float:
        """Calculate activity regularity (0 to 1, irregular to regular)"""
        
        if len(activities) < 2:
            return 0.0
        
        # Calculate intervals between activities
        activity_dates = sorted([a.activity_date for a in activities])
        intervals = []
        
        for i in range(1, len(activity_dates)):
            interval = (activity_dates[i] - activity_dates[i-1]).days
            intervals.append(interval)
        
        if not intervals:
            return 0.0
        
        # Calculate coefficient of variation (lower = more regular)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval == 0:
            return 0.0
        
        cv = std_interval / mean_interval
        
        # Convert to regularity score (0 to 1)
        regularity = max(0.0, 1.0 - min(cv, 2.0) / 2.0)
        
        return regularity

class InstitutionalAttentionScoringSystem:
    """
    Main system for institutional attention scoring and analysis
    """
    
    def __init__(self, 
                 data_collector: InstitutionalDataCollector,
                 graph_analytics: Optional[InstitutionalGraphAnalytics] = None):
        self.data_collector = data_collector
        self.graph_analytics = graph_analytics
        self.calculator = AttentionScoreCalculator()
        
        # Cached results
        self.stock_profiles = {}  # stock_code -> StockAttentionProfile
        self.institution_scores = {}  # (institution_id, stock_code) -> AttentionScore
        
        # Configuration
        self.min_score_for_alerts = 60.0
        self.screening_thresholds = {
            'high_attention': 70.0,
            'coordinated_activity': 0.5,
            'recent_activity_days': 7
        }
    
    async def calculate_stock_attention_profile(self, 
                                              stock_code: str,
                                              start_date: date,
                                              end_date: date,
                                              min_attention_score: float = 0.0) -> StockAttentionProfile:
        """
        Calculate comprehensive attention profile for a stock.
        
        Args:
            stock_code: Stock code to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            min_attention_score: Minimum score threshold for inclusion
            
        Returns:
            StockAttentionProfile with comprehensive metrics
        """
        
        try:
            logger.info(f"Calculating attention profile for {stock_code}")
            
            # Collect institutional data
            stock_data = await self.data_collector.collect_all_data(
                [stock_code], start_date, end_date
            )
            
            if stock_code not in self.data_collector.activity_timeline:
                return self._create_empty_profile(stock_code)
            
            activities = self.data_collector.activity_timeline[stock_code]
            
            # Group activities by institution
            institution_activities = defaultdict(list)
            institutions = {}
            
            for activity in activities:
                inst_id = activity.institution.institution_id
                institution_activities[inst_id].append(activity)
                institutions[inst_id] = activity.institution
            
            # Get coordination scores if graph analytics available
            coordination_scores = {}
            if self.graph_analytics:
                try:
                    # Build network for coordination analysis
                    await self.graph_analytics.build_institutional_network(
                        [stock_code], start_date, end_date
                    )
                    
                    # Extract coordination scores
                    for inst_id in institutions.keys():
                        relationships = self.graph_analytics.get_institution_relationships(inst_id)
                        if relationships:
                            avg_strength = np.mean([r.strength_score for r in relationships])
                            coordination_scores[inst_id] = avg_strength * 100  # Scale to 0-100
                        else:
                            coordination_scores[inst_id] = 0.0
                except Exception as e:
                    logger.warning(f"Could not calculate coordination scores: {e}")
                    coordination_scores = {inst_id: 0.0 for inst_id in institutions.keys()}
            else:
                coordination_scores = {inst_id: 0.0 for inst_id in institutions.keys()}
            
            # Calculate attention scores for each institution
            institution_scores = []
            
            for inst_id, inst_activities in institution_activities.items():
                institution = institutions[inst_id]
                coordination_score = coordination_scores.get(inst_id, 0.0)
                
                attention_score = self.calculator.calculate_attention_score(
                    stock_code=stock_code,
                    institution=institution,
                    activities=inst_activities,
                    coordination_score=coordination_score,
                    reference_date=end_date
                )
                
                if attention_score.overall_score >= min_attention_score:
                    institution_scores.append(attention_score)
                    
                    # Cache the score
                    self.institution_scores[(inst_id, stock_code)] = attention_score
            
            # Create comprehensive profile
            profile = self._create_stock_profile(stock_code, institution_scores, activities)
            
            # Cache the profile
            self.stock_profiles[stock_code] = profile
            
            logger.info(f"Calculated attention profile for {stock_code}: "
                       f"{profile.institutional_count} institutions, "
                       f"score {profile.total_attention_score:.1f}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error calculating attention profile for {stock_code}: {e}")
            return self._create_empty_profile(stock_code)
    
    def _create_empty_profile(self, stock_code: str) -> StockAttentionProfile:
        """Create empty attention profile for stocks with no data"""
        
        return StockAttentionProfile(
            stock_code=stock_code,
            total_attention_score=0.0,
            institutional_count=0,
            active_institutional_count=0,
            calculation_date=datetime.now(),
            data_quality_score=0.0
        )
    
    def _create_stock_profile(self, 
                            stock_code: str,
                            institution_scores: List[AttentionScore],
                            activities: List[InstitutionalActivity]) -> StockAttentionProfile:
        """Create comprehensive stock attention profile"""
        
        if not institution_scores:
            return self._create_empty_profile(stock_code)
        
        # Calculate aggregate metrics
        total_attention_score = np.mean([score.overall_score for score in institution_scores])
        institutional_count = len(institution_scores)
        active_institutional_count = len([s for s in institution_scores if s.recent_activities > 0])
        
        # Analyze patterns
        pattern_counts = Counter([score.behavior_pattern for score in institution_scores])
        dominant_patterns = pattern_counts.most_common(5)
        
        # Attention level distribution
        attention_distribution = Counter([score.attention_level for score in institution_scores])
        
        # Institution type analysis
        type_distribution = Counter([score.institution.institution_type for score in institution_scores])
        type_attention_scores = {}
        
        for inst_type in type_distribution.keys():
            type_scores = [s.overall_score for s in institution_scores 
                          if s.institution.institution_type == inst_type]
            type_attention_scores[inst_type] = np.mean(type_scores) if type_scores else 0.0
        
        # Activity trend analysis
        recent_activities = sum(score.recent_activities for score in institution_scores)
        total_activities = sum(score.total_activities for score in institution_scores)
        
        # Simple trend calculation (recent vs historical)
        if total_activities > recent_activities:
            historical_rate = (total_activities - recent_activities) / max(1, total_activities - recent_activities)
            recent_rate = recent_activities / 30  # Activities per day in last 30 days
            activity_trend = (recent_rate - historical_rate) / max(historical_rate, 0.1)
            activity_trend = max(-1.0, min(1.0, activity_trend))  # Clamp to [-1, 1]
        else:
            activity_trend = 1.0 if recent_activities > 0 else 0.0
        
        # Coordination analysis
        coordination_scores = [score.coordination_strength for score in institution_scores]
        coordination_score = np.mean(coordination_scores) if coordination_scores else 0.0
        
        # Simple coordination clustering (institutions with high coordination)
        high_coord_institutions = [
            score.institution.institution_id for score in institution_scores
            if score.coordination_strength > 0.5
        ]
        coordination_clusters = [high_coord_institutions] if len(high_coord_institutions) > 1 else []
        
        # Data quality assessment
        confidence_scores = [score.confidence_score for score in institution_scores]
        data_quality_score = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return StockAttentionProfile(
            stock_code=stock_code,
            total_attention_score=total_attention_score,
            institutional_count=institutional_count,
            active_institutional_count=active_institutional_count,
            institution_scores=institution_scores,
            dominant_patterns=dominant_patterns,
            attention_distribution=dict(attention_distribution),
            total_activities=total_activities,
            recent_activities=recent_activities,
            activity_trend=activity_trend,
            type_distribution=dict(type_distribution),
            type_attention_scores=dict(type_attention_scores),
            coordination_clusters=coordination_clusters,
            coordination_score=coordination_score,
            calculation_date=datetime.now(),
            data_quality_score=data_quality_score
        )
    
    async def screen_stocks_by_attention(self, 
                                       stock_codes: List[str],
                                       start_date: date,
                                       end_date: date,
                                       criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Screen stocks based on institutional attention criteria.
        
        Args:
            stock_codes: List of stock codes to screen
            start_date: Start date for analysis
            end_date: End date for analysis
            criteria: Screening criteria dictionary
            
        Returns:
            List of stocks meeting criteria with attention metrics
        """
        
        if criteria is None:
            criteria = self.screening_thresholds.copy()
        
        results = []
        
        try:
            logger.info(f"Screening {len(stock_codes)} stocks by attention criteria")
            
            for stock_code in stock_codes:
                try:
                    # Calculate attention profile
                    profile = await self.calculate_stock_attention_profile(
                        stock_code, start_date, end_date
                    )
                    
                    # Apply screening criteria
                    if self._meets_screening_criteria(profile, criteria):
                        
                        # Prepare result data
                        result = {
                            'stock_code': stock_code,
                            'total_attention_score': profile.total_attention_score,
                            'institutional_count': profile.institutional_count,
                            'active_institutional_count': profile.active_institutional_count,
                            'recent_activities': profile.recent_activities,
                            'activity_trend': profile.activity_trend,
                            'coordination_score': profile.coordination_score,
                            'dominant_pattern': profile.dominant_patterns[0][0].value if profile.dominant_patterns else 'none',
                            'top_institutions': [
                                {
                                    'name': score.institution.name,
                                    'type': score.institution.institution_type.value,
                                    'attention_score': score.overall_score,
                                    'behavior_pattern': score.behavior_pattern.value,
                                    'recent_activities': score.recent_activities
                                }
                                for score in sorted(profile.institution_scores, 
                                                  key=lambda s: s.overall_score, reverse=True)[:5]
                            ],
                            'screening_reasons': self._get_screening_reasons(profile, criteria)
                        }
                        
                        results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Error screening {stock_code}: {e}")
                    continue
            
            # Sort by attention score
            results.sort(key=lambda x: x['total_attention_score'], reverse=True)
            
            logger.info(f"Found {len(results)} stocks meeting attention criteria")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in attention screening: {e}")
            return []
    
    def _meets_screening_criteria(self, profile: StockAttentionProfile, 
                                criteria: Dict[str, Any]) -> bool:
        """Check if stock profile meets screening criteria"""
        
        # High attention threshold
        if 'high_attention' in criteria:
            if profile.total_attention_score < criteria['high_attention']:
                return False
        
        # Coordinated activity threshold
        if 'coordinated_activity' in criteria:
            if profile.coordination_score < criteria['coordinated_activity']:
                return False
        
        # Recent activity requirement
        if 'recent_activity_days' in criteria:
            if profile.active_institutional_count == 0:
                return False
        
        # Minimum institutional count
        if 'min_institutions' in criteria:
            if profile.institutional_count < criteria['min_institutions']:
                return False
        
        # Activity trend requirement
        if 'positive_trend' in criteria and criteria['positive_trend']:
            if profile.activity_trend <= 0:
                return False
        
        # Specific behavior patterns
        if 'required_patterns' in criteria:
            required_patterns = set(criteria['required_patterns'])
            profile_patterns = {pattern for pattern, count in profile.dominant_patterns}
            if not required_patterns.intersection(profile_patterns):
                return False
        
        return True
    
    def _get_screening_reasons(self, profile: StockAttentionProfile, 
                             criteria: Dict[str, Any]) -> List[str]:
        """Get reasons why stock meets screening criteria"""
        
        reasons = []
        
        if profile.total_attention_score >= criteria.get('high_attention', 0):
            reasons.append(f"High attention score: {profile.total_attention_score:.1f}")
        
        if profile.coordination_score >= criteria.get('coordinated_activity', 0):
            reasons.append(f"Coordinated activity: {profile.coordination_score:.2f}")
        
        if profile.active_institutional_count > 0:
            reasons.append(f"Recent activity: {profile.active_institutional_count} active institutions")
        
        if profile.activity_trend > 0:
            reasons.append(f"Increasing activity trend: {profile.activity_trend:.2f}")
        
        if profile.dominant_patterns:
            top_pattern = profile.dominant_patterns[0][0].value
            reasons.append(f"Dominant pattern: {top_pattern}")
        
        return reasons
    
    def generate_attention_alerts(self, 
                                stock_codes: List[str],
                                start_date: date,
                                end_date: date,
                                alert_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Generate alerts for stocks with high institutional attention.
        
        Args:
            stock_codes: List of stock codes to monitor
            start_date: Start date for analysis
            end_date: End date for analysis
            alert_threshold: Minimum attention score for alerts
            
        Returns:
            List of alert dictionaries
        """
        
        if alert_threshold is None:
            alert_threshold = self.min_score_for_alerts
        
        alerts = []
        
        try:
            for stock_code in stock_codes:
                if stock_code in self.stock_profiles:
                    profile = self.stock_profiles[stock_code]
                    
                    if profile.total_attention_score >= alert_threshold:
                        
                        # Determine alert type and priority
                        alert_type, priority = self._classify_alert(profile)
                        
                        alert = {
                            'stock_code': stock_code,
                            'alert_type': alert_type,
                            'priority': priority,
                            'attention_score': profile.total_attention_score,
                            'institutional_count': profile.institutional_count,
                            'active_institutions': profile.active_institutional_count,
                            'coordination_score': profile.coordination_score,
                            'activity_trend': profile.activity_trend,
                            'dominant_pattern': profile.dominant_patterns[0][0].value if profile.dominant_patterns else 'unknown',
                            'message': self._generate_alert_message(profile),
                            'timestamp': datetime.now(),
                            'data_quality': profile.data_quality_score
                        }
                        
                        alerts.append(alert)
            
            # Sort by priority and attention score
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            alerts.sort(key=lambda x: (priority_order.get(x['priority'], 0), x['attention_score']), 
                       reverse=True)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating attention alerts: {e}")
            return []
    
    def _classify_alert(self, profile: StockAttentionProfile) -> Tuple[str, str]:
        """Classify alert type and priority"""
        
        score = profile.total_attention_score
        coordination = profile.coordination_score
        trend = profile.activity_trend
        
        # Determine alert type
        if coordination > 0.7:
            alert_type = "coordinated_activity"
        elif trend > 0.5:
            alert_type = "increasing_attention"
        elif score > 85:
            alert_type = "very_high_attention"
        else:
            alert_type = "high_attention"
        
        # Determine priority
        if score > 90 or (coordination > 0.8 and trend > 0.3):
            priority = "high"
        elif score > 75 or coordination > 0.5:
            priority = "medium"
        else:
            priority = "low"
        
        return alert_type, priority
    
    def _generate_alert_message(self, profile: StockAttentionProfile) -> str:
        """Generate human-readable alert message"""
        
        stock_code = profile.stock_code
        score = profile.total_attention_score
        inst_count = profile.institutional_count
        active_count = profile.active_institutional_count
        
        message = f"Stock {stock_code} has high institutional attention (score: {score:.1f}). "
        message += f"{inst_count} institutions tracked, {active_count} recently active. "
        
        if profile.dominant_patterns:
            pattern = profile.dominant_patterns[0][0].value.replace('_', ' ').title()
            message += f"Dominant behavior: {pattern}. "
        
        if profile.coordination_score > 0.5:
            message += f"Coordinated activity detected (strength: {profile.coordination_score:.2f}). "
        
        if profile.activity_trend > 0.3:
            message += "Activity trend is increasing. "
        elif profile.activity_trend < -0.3:
            message += "Activity trend is decreasing. "
        
        return message.strip()
    
    def get_institution_attention_summary(self, institution_id: str) -> Dict[str, Any]:
        """Get attention summary for a specific institution across all stocks"""
        
        institution_scores = [
            score for (inst_id, stock_code), score in self.institution_scores.items()
            if inst_id == institution_id
        ]
        
        if not institution_scores:
            return {'error': 'No attention data found for institution'}
        
        # Calculate summary metrics
        avg_attention = np.mean([score.overall_score for score in institution_scores])
        total_stocks = len(institution_scores)
        high_attention_stocks = len([s for s in institution_scores if s.overall_score > 70])
        
        # Pattern analysis
        patterns = Counter([score.behavior_pattern for score in institution_scores])
        
        # Activity analysis
        total_activities = sum(score.total_activities for score in institution_scores)
        recent_activities = sum(score.recent_activities for score in institution_scores)
        
        # Top stocks by attention
        top_stocks = sorted(institution_scores, key=lambda s: s.overall_score, reverse=True)[:10]
        
        return {
            'institution_id': institution_id,
            'institution_name': institution_scores[0].institution.name,
            'institution_type': institution_scores[0].institution.institution_type.value,
            'average_attention_score': avg_attention,
            'total_stocks_tracked': total_stocks,
            'high_attention_stocks': high_attention_stocks,
            'total_activities': total_activities,
            'recent_activities': recent_activities,
            'dominant_patterns': dict(patterns.most_common(5)),
            'top_stocks': [
                {
                    'stock_code': score.stock_code,
                    'attention_score': score.overall_score,
                    'behavior_pattern': score.behavior_pattern.value,
                    'recent_activities': score.recent_activities
                }
                for score in top_stocks
            ]
        }