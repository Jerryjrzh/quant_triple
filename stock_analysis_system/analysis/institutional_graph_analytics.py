"""
Institutional Graph Analytics

This module implements graph analytics for institutional relationships including
NetworkX integration, coordinated activity detection, network visualization,
and relationship strength scoring for the stock analysis system.

Requirements addressed: 3.1, 3.2, 3.3, 3.4
"""

import asyncio
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .institutional_data_collector import (
    ActivityType,
    InstitutionalActivity,
    InstitutionalDataCollector,
    InstitutionalInvestor,
    InstitutionType,
)

logger = logging.getLogger(__name__)


class RelationshipType(str, Enum):
    """Types of institutional relationships"""

    COORDINATED_TRADING = "coordinated_trading"
    SAME_STOCK_ACTIVITY = "same_stock_activity"
    TEMPORAL_CORRELATION = "temporal_correlation"
    SIMILAR_PORTFOLIO = "similar_portfolio"
    PARENT_SUBSIDIARY = "parent_subsidiary"
    FUND_FAMILY = "fund_family"
    GEOGRAPHIC_PROXIMITY = "geographic_proximity"
    SECTOR_SPECIALIZATION = "sector_specialization"


@dataclass
class InstitutionalRelationship:
    """Represents a relationship between two institutions"""

    institution_a: InstitutionalInvestor
    institution_b: InstitutionalInvestor
    relationship_type: RelationshipType
    strength_score: float  # 0.0 to 1.0

    # Supporting evidence
    common_stocks: Set[str] = field(default_factory=set)
    coordinated_activities: List[
        Tuple[InstitutionalActivity, InstitutionalActivity]
    ] = field(default_factory=list)
    temporal_correlation: float = 0.0
    activity_overlap_ratio: float = 0.0

    # Metadata
    first_observed: Optional[datetime] = None
    last_observed: Optional[datetime] = None
    confidence_score: float = 1.0

    def __post_init__(self):
        if not self.first_observed:
            self.first_observed = datetime.now()
        if not self.last_observed:
            self.last_observed = datetime.now()


@dataclass
class CoordinatedActivityPattern:
    """Represents a detected coordinated activity pattern"""

    pattern_id: str
    institutions: List[InstitutionalInvestor]
    stock_codes: Set[str]
    activity_type: ActivityType

    # Pattern characteristics
    time_window: timedelta
    activity_correlation: float
    volume_correlation: float
    price_impact_correlation: float

    # Pattern instances
    instances: List[List[InstitutionalActivity]] = field(default_factory=list)

    # Statistical measures
    frequency: int = 0
    avg_profit: Optional[float] = None
    success_rate: Optional[float] = None

    # Detection metadata
    detection_date: datetime = field(default_factory=datetime.now)
    confidence_score: float = 1.0


@dataclass
class NetworkMetrics:
    """Network-level metrics for institutional relationships"""

    total_nodes: int
    total_edges: int
    density: float
    avg_clustering_coefficient: float
    avg_path_length: float

    # Centrality measures
    degree_centrality: Dict[str, float] = field(default_factory=dict)
    betweenness_centrality: Dict[str, float] = field(default_factory=dict)
    closeness_centrality: Dict[str, float] = field(default_factory=dict)
    eigenvector_centrality: Dict[str, float] = field(default_factory=dict)

    # Community detection
    communities: List[List[str]] = field(default_factory=list)
    modularity: float = 0.0

    # Temporal metrics
    calculation_date: datetime = field(default_factory=datetime.now)


class InstitutionalGraphAnalytics:
    """
    Main class for institutional graph analytics including relationship detection,
    network analysis, and visualization
    """

    def __init__(self, data_collector: InstitutionalDataCollector):
        self.data_collector = data_collector
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()  # For directional relationships

        # Analysis parameters
        self.coordination_time_window = timedelta(
            days=3
        )  # Time window for coordinated activity
        self.min_relationship_strength = (
            0.3  # Minimum strength for relationship inclusion
        )
        self.min_activity_overlap = 0.2  # Minimum activity overlap ratio

        # Cached results
        self.relationships = {}  # institution_pair -> InstitutionalRelationship
        self.coordinated_patterns = []  # List[CoordinatedActivityPattern]
        self.network_metrics = None  # NetworkMetrics

        # Institution registry
        self.institutions = {}  # institution_id -> InstitutionalInvestor

    async def build_institutional_network(
        self,
        stock_codes: List[str],
        start_date: date,
        end_date: date,
        rebuild: bool = False,
    ) -> nx.Graph:
        """
        Build the institutional relationship network based on activity data.

        Args:
            stock_codes: List of stock codes to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            rebuild: Whether to rebuild the network from scratch

        Returns:
            NetworkX graph representing institutional relationships
        """

        try:
            if rebuild:
                self.graph.clear()
                self.directed_graph.clear()
                self.relationships.clear()
                self.institutions.clear()

            logger.info(f"Building institutional network for {len(stock_codes)} stocks")

            # Collect institutional activity data
            activity_data = await self.data_collector.collect_all_data(
                stock_codes, start_date, end_date
            )

            # Extract all institutions and their activities
            all_activities = []
            for stock_code, data in activity_data.items():
                if stock_code in self.data_collector.activity_timeline:
                    all_activities.extend(
                        self.data_collector.activity_timeline[stock_code]
                    )

            # Register institutions
            for activity in all_activities:
                if activity.institution.institution_id not in self.institutions:
                    self.institutions[activity.institution.institution_id] = (
                        activity.institution
                    )
                    self.graph.add_node(
                        activity.institution.institution_id,
                        institution=activity.institution,
                        name=activity.institution.name,
                        type=activity.institution.institution_type.value,
                        confidence=activity.institution.confidence_score,
                    )

            logger.info(f"Registered {len(self.institutions)} institutions")

            # Detect relationships
            await self._detect_relationships(all_activities)

            # Add edges to graph
            self._build_graph_edges()

            # Calculate network metrics
            self.network_metrics = self._calculate_network_metrics()

            logger.info(
                f"Built network with {self.graph.number_of_nodes()} nodes "
                f"and {self.graph.number_of_edges()} edges"
            )

            return self.graph

        except Exception as e:
            logger.error(f"Error building institutional network: {e}")
            raise

    async def _detect_relationships(
        self, activities: List[InstitutionalActivity]
    ) -> None:
        """Detect relationships between institutions based on their activities"""

        # Group activities by institution
        institution_activities = defaultdict(list)
        for activity in activities:
            institution_activities[activity.institution.institution_id].append(activity)

        # Analyze all pairs of institutions
        institution_ids = list(institution_activities.keys())

        for i, inst_a_id in enumerate(institution_ids):
            for inst_b_id in institution_ids[i + 1 :]:

                inst_a = self.institutions[inst_a_id]
                inst_b = self.institutions[inst_b_id]

                activities_a = institution_activities[inst_a_id]
                activities_b = institution_activities[inst_b_id]

                # Detect various types of relationships
                relationships = []

                # 1. Coordinated trading detection
                coord_rel = await self._detect_coordinated_trading(
                    inst_a, inst_b, activities_a, activities_b
                )
                if coord_rel:
                    relationships.append(coord_rel)

                # 2. Same stock activity detection
                same_stock_rel = self._detect_same_stock_activity(
                    inst_a, inst_b, activities_a, activities_b
                )
                if same_stock_rel:
                    relationships.append(same_stock_rel)

                # 3. Temporal correlation detection
                temporal_rel = self._detect_temporal_correlation(
                    inst_a, inst_b, activities_a, activities_b
                )
                if temporal_rel:
                    relationships.append(temporal_rel)

                # 4. Fund family detection
                fund_family_rel = self._detect_fund_family_relationship(inst_a, inst_b)
                if fund_family_rel:
                    relationships.append(fund_family_rel)

                # Store the strongest relationship
                if relationships:
                    best_relationship = max(
                        relationships, key=lambda r: r.strength_score
                    )
                    if (
                        best_relationship.strength_score
                        >= self.min_relationship_strength
                    ):
                        pair_key = tuple(sorted([inst_a_id, inst_b_id]))
                        self.relationships[pair_key] = best_relationship

    async def _detect_coordinated_trading(
        self,
        inst_a: InstitutionalInvestor,
        inst_b: InstitutionalInvestor,
        activities_a: List[InstitutionalActivity],
        activities_b: List[InstitutionalActivity],
    ) -> Optional[InstitutionalRelationship]:
        """Detect coordinated trading patterns between two institutions"""

        coordinated_pairs = []

        # Look for activities in the same stock within the time window
        for activity_a in activities_a:
            for activity_b in activities_b:

                # Must be same stock
                if activity_a.stock_code != activity_b.stock_code:
                    continue

                # Must be within time window
                time_diff = abs(
                    (activity_a.activity_date - activity_b.activity_date).days
                )
                if time_diff > self.coordination_time_window.days:
                    continue

                # Check for coordination patterns
                coordination_score = self._calculate_coordination_score(
                    activity_a, activity_b
                )

                if coordination_score > 0.5:  # Threshold for coordination
                    coordinated_pairs.append(
                        (activity_a, activity_b, coordination_score)
                    )

        if not coordinated_pairs:
            return None

        # Calculate overall relationship strength
        avg_coordination = np.mean([score for _, _, score in coordinated_pairs])

        # Get common stocks
        stocks_a = {a.stock_code for a in activities_a}
        stocks_b = {a.stock_code for a in activities_b}
        common_stocks = stocks_a.intersection(stocks_b)

        relationship = InstitutionalRelationship(
            institution_a=inst_a,
            institution_b=inst_b,
            relationship_type=RelationshipType.COORDINATED_TRADING,
            strength_score=avg_coordination,
            common_stocks=common_stocks,
            coordinated_activities=[(pair[0], pair[1]) for pair in coordinated_pairs],
            confidence_score=min(inst_a.confidence_score, inst_b.confidence_score),
        )

        return relationship

    def _calculate_coordination_score(
        self, activity_a: InstitutionalActivity, activity_b: InstitutionalActivity
    ) -> float:
        """Calculate coordination score between two activities"""

        score = 0.0

        # Time proximity (closer in time = higher score)
        time_diff_days = abs((activity_a.activity_date - activity_b.activity_date).days)
        time_score = max(0, 1.0 - (time_diff_days / self.coordination_time_window.days))
        score += time_score * 0.3

        # Activity type coordination
        if self._are_activities_coordinated(
            activity_a.activity_type, activity_b.activity_type
        ):
            score += 0.4

        # Volume correlation (if available)
        if activity_a.volume and activity_b.volume:
            volume_ratio = min(activity_a.volume, activity_b.volume) / max(
                activity_a.volume, activity_b.volume
            )
            score += volume_ratio * 0.2

        # Amount correlation (if available)
        if activity_a.amount and activity_b.amount:
            amount_ratio = min(activity_a.amount, activity_b.amount) / max(
                activity_a.amount, activity_b.amount
            )
            score += amount_ratio * 0.1

        return min(score, 1.0)

    def _are_activities_coordinated(
        self, type_a: ActivityType, type_b: ActivityType
    ) -> bool:
        """Check if two activity types indicate coordination"""

        coordinated_patterns = [
            # Both buying
            (ActivityType.DRAGON_TIGER_BUY, ActivityType.BLOCK_TRADE_BUY),
            (ActivityType.DRAGON_TIGER_BUY, ActivityType.SHAREHOLDING_INCREASE),
            (ActivityType.BLOCK_TRADE_BUY, ActivityType.SHAREHOLDING_INCREASE),
            # Both selling
            (ActivityType.DRAGON_TIGER_SELL, ActivityType.BLOCK_TRADE_SELL),
            (ActivityType.DRAGON_TIGER_SELL, ActivityType.SHAREHOLDING_DECREASE),
            (ActivityType.BLOCK_TRADE_SELL, ActivityType.SHAREHOLDING_DECREASE),
            # Same type activities
            (ActivityType.DRAGON_TIGER_BUY, ActivityType.DRAGON_TIGER_BUY),
            (ActivityType.DRAGON_TIGER_SELL, ActivityType.DRAGON_TIGER_SELL),
            (ActivityType.BLOCK_TRADE_BUY, ActivityType.BLOCK_TRADE_BUY),
            (ActivityType.BLOCK_TRADE_SELL, ActivityType.BLOCK_TRADE_SELL),
        ]

        return (type_a, type_b) in coordinated_patterns or (
            type_b,
            type_a,
        ) in coordinated_patterns

    def _detect_same_stock_activity(
        self,
        inst_a: InstitutionalInvestor,
        inst_b: InstitutionalInvestor,
        activities_a: List[InstitutionalActivity],
        activities_b: List[InstitutionalActivity],
    ) -> Optional[InstitutionalRelationship]:
        """Detect relationships based on activity in the same stocks"""

        stocks_a = {a.stock_code for a in activities_a}
        stocks_b = {a.stock_code for a in activities_b}

        common_stocks = stocks_a.intersection(stocks_b)

        if not common_stocks:
            return None

        # Calculate overlap ratio
        total_stocks = len(stocks_a.union(stocks_b))
        overlap_ratio = len(common_stocks) / total_stocks

        if overlap_ratio < self.min_activity_overlap:
            return None

        relationship = InstitutionalRelationship(
            institution_a=inst_a,
            institution_b=inst_b,
            relationship_type=RelationshipType.SAME_STOCK_ACTIVITY,
            strength_score=overlap_ratio,
            common_stocks=common_stocks,
            activity_overlap_ratio=overlap_ratio,
            confidence_score=min(inst_a.confidence_score, inst_b.confidence_score),
        )

        return relationship

    def _detect_temporal_correlation(
        self,
        inst_a: InstitutionalInvestor,
        inst_b: InstitutionalInvestor,
        activities_a: List[InstitutionalActivity],
        activities_b: List[InstitutionalActivity],
    ) -> Optional[InstitutionalRelationship]:
        """Detect temporal correlation in trading activities"""

        if len(activities_a) < 5 or len(activities_b) < 5:
            return None  # Need sufficient data for correlation

        # Create time series of activity counts
        all_dates = set()
        for activity in activities_a + activities_b:
            all_dates.add(activity.activity_date)

        if len(all_dates) < 10:
            return None  # Need sufficient time points

        sorted_dates = sorted(all_dates)

        # Count activities per date
        counts_a = []
        counts_b = []

        for date in sorted_dates:
            count_a = sum(1 for a in activities_a if a.activity_date == date)
            count_b = sum(1 for a in activities_b if a.activity_date == date)
            counts_a.append(count_a)
            counts_b.append(count_b)

        # Calculate correlation
        correlation = np.corrcoef(counts_a, counts_b)[0, 1]

        if np.isnan(correlation) or correlation < 0.5:
            return None

        relationship = InstitutionalRelationship(
            institution_a=inst_a,
            institution_b=inst_b,
            relationship_type=RelationshipType.TEMPORAL_CORRELATION,
            strength_score=correlation,
            temporal_correlation=correlation,
            confidence_score=min(inst_a.confidence_score, inst_b.confidence_score),
        )

        return relationship

    def _detect_fund_family_relationship(
        self, inst_a: InstitutionalInvestor, inst_b: InstitutionalInvestor
    ) -> Optional[InstitutionalRelationship]:
        """Detect fund family relationships based on name similarity"""

        # Only apply to funds
        if inst_a.institution_type not in [
            InstitutionType.MUTUAL_FUND
        ] or inst_b.institution_type not in [InstitutionType.MUTUAL_FUND]:
            return None

        # Extract company names
        company_a = self._extract_fund_company(inst_a.name)
        company_b = self._extract_fund_company(inst_b.name)

        if not company_a or not company_b:
            return None

        # Check for same company
        if company_a == company_b:
            relationship = InstitutionalRelationship(
                institution_a=inst_a,
                institution_b=inst_b,
                relationship_type=RelationshipType.FUND_FAMILY,
                strength_score=0.9,  # High strength for same fund family
                confidence_score=min(inst_a.confidence_score, inst_b.confidence_score),
            )
            return relationship

        return None

    def _extract_fund_company(self, fund_name: str) -> Optional[str]:
        """Extract fund company name from fund name"""

        # Common patterns for fund company extraction
        patterns = [
            r"(.+?)基金管理",
            r"(.+?)资产管理",
            r"(.+?)投资管理",
            r"(.+?)基金",
        ]

        import re

        for pattern in patterns:
            match = re.search(pattern, fund_name)
            if match:
                return match.group(1).strip()

        return None

    def _build_graph_edges(self) -> None:
        """Build graph edges from detected relationships"""

        for pair_key, relationship in self.relationships.items():
            inst_a_id, inst_b_id = pair_key

            # Add edge to undirected graph
            self.graph.add_edge(
                inst_a_id,
                inst_b_id,
                relationship=relationship,
                weight=relationship.strength_score,
                type=relationship.relationship_type.value,
                common_stocks=len(relationship.common_stocks),
                confidence=relationship.confidence_score,
            )

            # Add edges to directed graph if there's directionality
            if relationship.coordinated_activities:
                # Analyze coordination direction
                buy_sell_balance = self._analyze_coordination_direction(relationship)
                if buy_sell_balance != 0:
                    if buy_sell_balance > 0:
                        # inst_a tends to lead
                        self.directed_graph.add_edge(
                            inst_a_id, inst_b_id, **self.graph[inst_a_id][inst_b_id]
                        )
                    else:
                        # inst_b tends to lead
                        self.directed_graph.add_edge(
                            inst_b_id, inst_a_id, **self.graph[inst_a_id][inst_b_id]
                        )

    def _analyze_coordination_direction(
        self, relationship: InstitutionalRelationship
    ) -> float:
        """Analyze the direction of coordination between institutions"""

        direction_score = 0.0

        for activity_a, activity_b in relationship.coordinated_activities:
            # Check temporal precedence
            if activity_a.activity_date < activity_b.activity_date:
                direction_score += 1.0  # inst_a leads
            elif activity_a.activity_date > activity_b.activity_date:
                direction_score -= 1.0  # inst_b leads
            # Equal dates contribute 0

        return (
            direction_score / len(relationship.coordinated_activities)
            if relationship.coordinated_activities
            else 0.0
        )

    def _calculate_network_metrics(self) -> NetworkMetrics:
        """Calculate comprehensive network metrics"""

        if self.graph.number_of_nodes() == 0:
            return NetworkMetrics(
                total_nodes=0,
                total_edges=0,
                density=0.0,
                avg_clustering_coefficient=0.0,
                avg_path_length=0.0,
            )

        # Basic metrics
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()
        density = nx.density(self.graph)

        # Clustering coefficient
        clustering_coeffs = nx.clustering(self.graph)
        avg_clustering = (
            np.mean(list(clustering_coeffs.values())) if clustering_coeffs else 0.0
        )

        # Average path length (for connected components)
        if nx.is_connected(self.graph):
            avg_path_length = nx.average_shortest_path_length(self.graph)
        else:
            # Calculate for largest connected component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            avg_path_length = (
                nx.average_shortest_path_length(subgraph)
                if len(largest_cc) > 1
                else 0.0
            )

        # Centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        closeness_centrality = nx.closeness_centrality(self.graph)

        try:
            eigenvector_centrality = nx.eigenvector_centrality(
                self.graph, max_iter=1000
            )
        except:
            eigenvector_centrality = {node: 0.0 for node in self.graph.nodes()}

        # Community detection
        try:
            communities = list(nx.community.greedy_modularity_communities(self.graph))
            modularity = nx.community.modularity(self.graph, communities)
            communities_list = [list(community) for community in communities]
        except:
            communities_list = []
            modularity = 0.0

        return NetworkMetrics(
            total_nodes=total_nodes,
            total_edges=total_edges,
            density=density,
            avg_clustering_coefficient=avg_clustering,
            avg_path_length=avg_path_length,
            degree_centrality=degree_centrality,
            betweenness_centrality=betweenness_centrality,
            closeness_centrality=closeness_centrality,
            eigenvector_centrality=eigenvector_centrality,
            communities=communities_list,
            modularity=modularity,
        )

    async def detect_coordinated_patterns(
        self, min_institutions: int = 3, min_correlation: float = 0.7
    ) -> List[CoordinatedActivityPattern]:
        """
        Detect coordinated activity patterns involving multiple institutions.

        Args:
            min_institutions: Minimum number of institutions for a pattern
            min_correlation: Minimum correlation threshold

        Returns:
            List of detected coordinated activity patterns
        """

        patterns = []

        try:
            # Group activities by stock and time window
            stock_activities = defaultdict(list)

            for stock_code, activities in self.data_collector.activity_timeline.items():
                stock_activities[stock_code] = activities

            # Analyze each stock for coordinated patterns
            for stock_code, activities in stock_activities.items():

                # Group activities by time windows
                time_windows = self._create_time_windows(
                    activities, self.coordination_time_window
                )

                for window_start, window_activities in time_windows.items():

                    if len(window_activities) < min_institutions:
                        continue

                    # Group by institution
                    inst_activities = defaultdict(list)
                    for activity in window_activities:
                        inst_activities[activity.institution.institution_id].append(
                            activity
                        )

                    if len(inst_activities) < min_institutions:
                        continue

                    # Check for coordination
                    pattern = self._analyze_coordination_pattern(
                        stock_code, window_start, inst_activities, min_correlation
                    )

                    if pattern:
                        patterns.append(pattern)

            # Store detected patterns
            self.coordinated_patterns = patterns

            logger.info(f"Detected {len(patterns)} coordinated activity patterns")
            return patterns

        except Exception as e:
            logger.error(f"Error detecting coordinated patterns: {e}")
            return []

    def _create_time_windows(
        self, activities: List[InstitutionalActivity], window_size: timedelta
    ) -> Dict[datetime, List[InstitutionalActivity]]:
        """Create time windows for activity analysis"""

        if not activities:
            return {}

        # Sort activities by date
        sorted_activities = sorted(activities, key=lambda a: a.activity_date)

        windows = {}
        current_window_start = sorted_activities[0].activity_date
        current_window_activities = []

        for activity in sorted_activities:
            activity_date = activity.activity_date

            # Check if activity falls within current window
            if (activity_date - current_window_start).days <= window_size.days:
                current_window_activities.append(activity)
            else:
                # Save current window if it has activities
                if current_window_activities:
                    windows[current_window_start] = current_window_activities.copy()

                # Start new window
                current_window_start = activity_date
                current_window_activities = [activity]

        # Don't forget the last window
        if current_window_activities:
            windows[current_window_start] = current_window_activities

        return windows

    def _analyze_coordination_pattern(
        self,
        stock_code: str,
        window_start: datetime,
        inst_activities: Dict[str, List[InstitutionalActivity]],
        min_correlation: float,
    ) -> Optional[CoordinatedActivityPattern]:
        """Analyze a time window for coordination patterns"""

        institutions = [
            self.institutions[inst_id] for inst_id in inst_activities.keys()
        ]

        # Calculate activity correlations
        correlations = []
        activity_types = []

        inst_ids = list(inst_activities.keys())

        for i, inst_a_id in enumerate(inst_ids):
            for inst_b_id in inst_ids[i + 1 :]:

                activities_a = inst_activities[inst_a_id]
                activities_b = inst_activities[inst_b_id]

                # Calculate correlation metrics
                correlation = self._calculate_activity_correlation(
                    activities_a, activities_b
                )
                correlations.append(correlation)

                # Track activity types
                types_a = {a.activity_type for a in activities_a}
                types_b = {a.activity_type for a in activities_b}
                activity_types.extend(list(types_a.union(types_b)))

        if not correlations:
            return None

        avg_correlation = np.mean(correlations)

        if avg_correlation < min_correlation:
            return None

        # Determine dominant activity type
        activity_counter = Counter(activity_types)
        dominant_activity_type = activity_counter.most_common(1)[0][0]

        # Create pattern
        pattern_id = f"pattern_{stock_code}_{window_start.strftime('%Y%m%d')}_{len(institutions)}"

        pattern = CoordinatedActivityPattern(
            pattern_id=pattern_id,
            institutions=institutions,
            stock_codes={stock_code},
            activity_type=dominant_activity_type,
            time_window=self.coordination_time_window,
            activity_correlation=avg_correlation,
            volume_correlation=self._calculate_volume_correlation(inst_activities),
            price_impact_correlation=0.0,  # Would need price data
            instances=[list(inst_activities.values())],
            frequency=1,
            confidence_score=avg_correlation,
        )

        return pattern

    def _calculate_activity_correlation(
        self,
        activities_a: List[InstitutionalActivity],
        activities_b: List[InstitutionalActivity],
    ) -> float:
        """Calculate correlation between two sets of activities"""

        if not activities_a or not activities_b:
            return 0.0

        # Create activity vectors based on dates
        all_dates = set()
        for activity in activities_a + activities_b:
            all_dates.add(activity.activity_date)

        if len(all_dates) < 2:
            return 0.0

        sorted_dates = sorted(all_dates)

        # Count activities per date
        counts_a = []
        counts_b = []

        for date in sorted_dates:
            count_a = sum(1 for a in activities_a if a.activity_date == date)
            count_b = sum(1 for a in activities_b if a.activity_date == date)
            counts_a.append(count_a)
            counts_b.append(count_b)

        # Calculate correlation
        try:
            correlation = np.corrcoef(counts_a, counts_b)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    def _calculate_volume_correlation(
        self, inst_activities: Dict[str, List[InstitutionalActivity]]
    ) -> float:
        """Calculate volume correlation across institutions"""

        # Extract volumes for each institution
        inst_volumes = {}

        for inst_id, activities in inst_activities.items():
            volumes = [a.volume for a in activities if a.volume is not None]
            if volumes:
                inst_volumes[inst_id] = np.mean(volumes)

        if len(inst_volumes) < 2:
            return 0.0

        # Calculate pairwise correlations
        correlations = []
        inst_ids = list(inst_volumes.keys())

        for i, inst_a_id in enumerate(inst_ids):
            for inst_b_id in inst_ids[i + 1 :]:
                # Simple correlation based on volume similarity
                vol_a = inst_volumes[inst_a_id]
                vol_b = inst_volumes[inst_b_id]

                similarity = min(vol_a, vol_b) / max(vol_a, vol_b)
                correlations.append(similarity)

        return np.mean(correlations) if correlations else 0.0

    def create_network_visualization(
        self,
        layout: str = "spring",
        node_size_metric: str = "degree_centrality",
        edge_width_metric: str = "weight",
        color_by: str = "institution_type",
    ) -> go.Figure:
        """
        Create interactive network visualization using Plotly.

        Args:
            layout: Layout algorithm ("spring", "circular", "kamada_kawai")
            node_size_metric: Metric for node sizing
            edge_width_metric: Metric for edge width
            color_by: Attribute for node coloring

        Returns:
            Plotly figure object
        """

        if self.graph.number_of_nodes() == 0:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No institutional network data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=16),
            )
            return fig

        # Calculate layout positions
        if layout == "spring":
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)

        # Extract node information
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_colors = []
        node_sizes = []

        # Color mapping for institution types
        type_colors = {
            "mutual_fund": "#1f77b4",
            "social_security": "#ff7f0e",
            "qfii": "#2ca02c",
            "insurance": "#d62728",
            "securities_firm": "#9467bd",
            "bank": "#8c564b",
            "hot_money": "#e377c2",
            "other": "#7f7f7f",
        }

        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Get node attributes
            node_data = self.graph.nodes[node]
            institution = node_data.get("institution")

            if institution:
                node_text.append(institution.name)
                node_info.append(
                    f"Name: {institution.name}<br>"
                    f"Type: {institution.institution_type.value}<br>"
                    f"Confidence: {institution.confidence_score:.2f}"
                )

                # Color by institution type
                node_colors.append(
                    type_colors.get(institution.institution_type.value, "#7f7f7f")
                )

                # Size by centrality metric
                if self.network_metrics and node_size_metric in [
                    "degree_centrality",
                    "betweenness_centrality",
                    "closeness_centrality",
                ]:
                    centrality_dict = getattr(
                        self.network_metrics, node_size_metric, {}
                    )
                    centrality = centrality_dict.get(node, 0.0)
                    node_sizes.append(20 + centrality * 30)  # Scale between 20-50
                else:
                    node_sizes.append(25)
            else:
                node_text.append(str(node))
                node_info.append(f"Node: {node}")
                node_colors.append("#7f7f7f")
                node_sizes.append(25)

        # Extract edge information
        edge_x = []
        edge_y = []
        edge_info = []
        edge_widths = []

        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # Get edge attributes
            edge_data = self.graph.edges[edge]
            relationship = edge_data.get("relationship")

            if relationship:
                edge_info.append(
                    f"Type: {relationship.relationship_type.value}<br>"
                    f"Strength: {relationship.strength_score:.2f}<br>"
                    f"Common Stocks: {len(relationship.common_stocks)}"
                )
                edge_widths.append(
                    1 + relationship.strength_score * 4
                )  # Scale between 1-5
            else:
                edge_info.append("Relationship")
                edge_widths.append(1)

        # Create figure
        fig = go.Figure()

        # Add edges
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=2, color="rgba(125,125,125,0.5)"),
                hoverinfo="none",
                mode="lines",
                name="Relationships",
            )
        )

        # Add nodes
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                hoverinfo="text",
                hovertext=node_info,
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2, color="white"),
                    opacity=0.8,
                ),
                name="Institutions",
            )
        )

        # Update layout
        fig.update_layout(
            title="Institutional Relationship Network",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Institutional relationships based on coordinated activities and common stock holdings",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(color="gray", size=12),
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
        )

        return fig

    def get_institution_relationships(
        self, institution_id: str
    ) -> List[InstitutionalRelationship]:
        """Get all relationships for a specific institution"""

        relationships = []

        for pair_key, relationship in self.relationships.items():
            if institution_id in pair_key:
                relationships.append(relationship)

        return relationships

    def get_network_summary(self) -> Dict[str, Any]:
        """Get a summary of the institutional network"""

        if not self.network_metrics:
            return {"error": "Network metrics not calculated"}

        # Top institutions by centrality
        top_by_degree = sorted(
            self.network_metrics.degree_centrality.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        top_by_betweenness = sorted(
            self.network_metrics.betweenness_centrality.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        # Institution type distribution
        type_distribution = Counter()
        for inst in self.institutions.values():
            type_distribution[inst.institution_type.value] += 1

        # Relationship type distribution
        relationship_types = Counter()
        for relationship in self.relationships.values():
            relationship_types[relationship.relationship_type.value] += 1

        return {
            "network_metrics": {
                "total_institutions": self.network_metrics.total_nodes,
                "total_relationships": self.network_metrics.total_edges,
                "network_density": self.network_metrics.density,
                "avg_clustering": self.network_metrics.avg_clustering_coefficient,
                "avg_path_length": self.network_metrics.avg_path_length,
                "modularity": self.network_metrics.modularity,
                "num_communities": len(self.network_metrics.communities),
            },
            "top_institutions": {
                "by_degree_centrality": [
                    {
                        "institution_id": inst_id,
                        "name": (
                            self.institutions[inst_id].name
                            if inst_id in self.institutions
                            else inst_id
                        ),
                        "centrality": centrality,
                    }
                    for inst_id, centrality in top_by_degree
                ],
                "by_betweenness_centrality": [
                    {
                        "institution_id": inst_id,
                        "name": (
                            self.institutions[inst_id].name
                            if inst_id in self.institutions
                            else inst_id
                        ),
                        "centrality": centrality,
                    }
                    for inst_id, centrality in top_by_betweenness
                ],
            },
            "distributions": {
                "institution_types": dict(type_distribution),
                "relationship_types": dict(relationship_types),
            },
            "coordinated_patterns": len(self.coordinated_patterns),
        }
