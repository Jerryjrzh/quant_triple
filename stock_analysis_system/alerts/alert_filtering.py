"""
Smart Alert Filtering and Aggregation System

This module provides intelligent alert deduplication, clustering, summarization,
and adaptive threshold management based on market conditions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

from .alert_engine import Alert, AlertPriority, AlertTriggerType

logger = logging.getLogger(__name__)


class FilterAction(str, Enum):
    """Actions that can be taken on filtered alerts"""
    ALLOW = "allow"
    SUPPRESS = "suppress"
    AGGREGATE = "aggregate"
    DELAY = "delay"


@dataclass
class FilterRule:
    """Rule for filtering alerts"""
    id: str
    name: str
    conditions: Dict[str, Any]
    action: FilterAction
    priority: int = 0  # Higher priority rules are applied first
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def matches(self, alert: Alert) -> bool:
        """Check if the alert matches this filter rule"""
        for field, condition in self.conditions.items():
            if not self._check_condition(alert, field, condition):
                return False
        return True
    
    def _check_condition(self, alert: Alert, field: str, condition: Any) -> bool:
        """Check a single condition against the alert"""
        try:
            if field == 'priority':
                return alert.priority.value == condition
            elif field == 'stock_code':
                return alert.stock_code == condition
            elif field == 'trigger_type':
                return alert.trigger.trigger_type.value == condition
            elif field == 'name_contains':
                return condition.lower() in alert.name.lower()
            elif field == 'description_contains':
                return condition.lower() in alert.description.lower()
            elif field == 'created_within_minutes':
                time_diff = datetime.now() - alert.created_at
                return time_diff.total_seconds() / 60 <= condition
            else:
                return False
        except Exception:
            return False


@dataclass
class AlertCluster:
    """Represents a cluster of similar alerts"""
    id: str
    alerts: List[Alert]
    representative_alert: Alert
    similarity_score: float
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_alert(self, alert: Alert) -> None:
        """Add an alert to the cluster"""
        self.alerts.append(alert)
        # Update representative alert if new alert has higher priority
        if alert.priority.value == 'critical' or (
            self.representative_alert.priority.value != 'critical' and 
            alert.priority.value == 'high'
        ):
            self.representative_alert = alert
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the cluster"""
        return {
            'cluster_id': self.id,
            'alert_count': len(self.alerts),
            'representative_alert': {
                'id': self.representative_alert.id,
                'name': self.representative_alert.name,
                'priority': self.representative_alert.priority.value
            },
            'priority_distribution': self._get_priority_distribution(),
            'stock_codes': list(set(alert.stock_code for alert in self.alerts if alert.stock_code)),
            'created_at': self.created_at.isoformat(),
            'similarity_score': self.similarity_score
        }
    
    def _get_priority_distribution(self) -> Dict[str, int]:
        """Get distribution of priorities in the cluster"""
        distribution = {}
        for alert in self.alerts:
            priority = alert.priority.value
            distribution[priority] = distribution.get(priority, 0) + 1
        return distribution


@dataclass
class MarketCondition:
    """Represents current market conditions for adaptive thresholds"""
    volatility_level: str  # 'low', 'medium', 'high', 'extreme'
    trend_direction: str   # 'up', 'down', 'sideways'
    volume_level: str      # 'low', 'medium', 'high'
    market_hours: bool
    updated_at: datetime = field(default_factory=datetime.now)


class SmartAlertFilter:
    """
    Intelligent alert filtering system with deduplication and adaptive thresholds
    """
    
    def __init__(self):
        self.filter_rules: List[FilterRule] = []
        self.suppressed_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.deduplication_cache: Dict[str, datetime] = {}
        self.adaptive_thresholds: Dict[str, float] = {}
        self.market_conditions: Optional[MarketCondition] = None
        
        # Initialize default filter rules
        self._initialize_default_rules()
        
        logger.info("SmartAlertFilter initialized")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default filter rules"""
        # Rule to suppress duplicate alerts within 5 minutes
        duplicate_rule = FilterRule(
            id="suppress_duplicates",
            name="Suppress Duplicate Alerts",
            conditions={
                'created_within_minutes': 5
            },
            action=FilterAction.SUPPRESS,
            priority=100
        )
        
        # Rule to aggregate low priority alerts
        aggregate_rule = FilterRule(
            id="aggregate_low_priority",
            name="Aggregate Low Priority Alerts",
            conditions={
                'priority': 'low'
            },
            action=FilterAction.AGGREGATE,
            priority=50
        )
        
        self.filter_rules = [duplicate_rule, aggregate_rule]
    
    async def add_filter_rule(self, rule: FilterRule) -> None:
        """Add a new filter rule"""
        self.filter_rules.append(rule)
        # Sort by priority (higher first)
        self.filter_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added filter rule: {rule.name}")
    
    async def remove_filter_rule(self, rule_id: str) -> bool:
        """Remove a filter rule"""
        for i, rule in enumerate(self.filter_rules):
            if rule.id == rule_id:
                del self.filter_rules[i]
                logger.info(f"Removed filter rule: {rule_id}")
                return True
        return False
    
    async def filter_alert(self, alert: Alert) -> Tuple[FilterAction, Optional[str]]:
        """Filter an alert and return the action to take"""
        try:
            # Check for deduplication first
            if await self._is_duplicate(alert):
                return FilterAction.SUPPRESS, "Duplicate alert suppressed"
            
            # Apply filter rules
            for rule in self.filter_rules:
                if rule.enabled and rule.matches(alert):
                    logger.info(f"Alert {alert.id} matched rule: {rule.name}")
                    return rule.action, f"Matched rule: {rule.name}"
            
            # Check adaptive thresholds
            if await self._check_adaptive_thresholds(alert):
                return FilterAction.SUPPRESS, "Below adaptive threshold"
            
            # Default action is to allow
            return FilterAction.ALLOW, "No filter rules matched"
            
        except Exception as e:
            logger.error(f"Error filtering alert {alert.id}: {e}")
            return FilterAction.ALLOW, "Error in filtering, allowing by default"
    
    async def _is_duplicate(self, alert: Alert) -> bool:
        """Check if the alert is a duplicate"""
        # Create a hash of the alert's key characteristics
        alert_signature = self._create_alert_signature(alert)
        
        # Check if we've seen this signature recently
        if alert_signature in self.deduplication_cache:
            last_seen = self.deduplication_cache[alert_signature]
            time_diff = datetime.now() - last_seen
            
            # Consider it a duplicate if seen within the last 5 minutes
            if time_diff.total_seconds() < 300:  # 5 minutes
                return True
        
        # Update cache
        self.deduplication_cache[alert_signature] = datetime.now()
        
        # Clean old entries from cache
        await self._clean_deduplication_cache()
        
        return False
    
    def _create_alert_signature(self, alert: Alert) -> str:
        """Create a unique signature for an alert"""
        signature_data = f"{alert.name}|{alert.stock_code}|{alert.trigger.trigger_type.value}"
        return hashlib.md5(signature_data.encode()).hexdigest()
    
    async def _clean_deduplication_cache(self) -> None:
        """Clean old entries from deduplication cache"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        keys_to_remove = [
            key for key, timestamp in self.deduplication_cache.items()
            if timestamp < cutoff_time
        ]
        
        for key in keys_to_remove:
            del self.deduplication_cache[key]
    
    async def _check_adaptive_thresholds(self, alert: Alert) -> bool:
        """Check if alert should be suppressed based on adaptive thresholds"""
        if not self.market_conditions:
            return False
        
        # Get threshold for this alert type
        threshold_key = f"{alert.trigger.trigger_type.value}_{alert.priority.value}"
        base_threshold = self.adaptive_thresholds.get(threshold_key, 0.5)
        
        # Adjust threshold based on market conditions
        adjusted_threshold = self._adjust_threshold_for_market_conditions(
            base_threshold, alert
        )
        
        # For this example, we'll use a simple scoring mechanism
        alert_score = self._calculate_alert_score(alert)
        
        return alert_score < adjusted_threshold
    
    def _adjust_threshold_for_market_conditions(self, base_threshold: float, alert: Alert) -> float:
        """Adjust threshold based on current market conditions"""
        if not self.market_conditions:
            return base_threshold
        
        adjustment_factor = 1.0
        
        # Adjust based on volatility
        if self.market_conditions.volatility_level == 'extreme':
            adjustment_factor *= 0.7  # Lower threshold (more alerts)
        elif self.market_conditions.volatility_level == 'high':
            adjustment_factor *= 0.85
        elif self.market_conditions.volatility_level == 'low':
            adjustment_factor *= 1.2  # Higher threshold (fewer alerts)
        
        # Adjust based on market hours
        if not self.market_conditions.market_hours:
            adjustment_factor *= 1.5  # Higher threshold outside market hours
        
        # Adjust based on alert priority
        if alert.priority == AlertPriority.CRITICAL:
            adjustment_factor *= 0.5  # Always allow critical alerts
        elif alert.priority == AlertPriority.HIGH:
            adjustment_factor *= 0.8
        
        return base_threshold * adjustment_factor
    
    def _calculate_alert_score(self, alert: Alert) -> float:
        """Calculate a score for the alert (higher = more important)"""
        score = 0.5  # Base score
        
        # Priority scoring
        priority_scores = {
            AlertPriority.CRITICAL: 1.0,
            AlertPriority.HIGH: 0.8,
            AlertPriority.MEDIUM: 0.6,
            AlertPriority.LOW: 0.4
        }
        score = priority_scores.get(alert.priority, 0.5)
        
        # Adjust based on trigger count (frequent alerts get lower scores)
        if alert.trigger_count > 10:
            score *= 0.8
        elif alert.trigger_count > 5:
            score *= 0.9
        
        return score
    
    async def update_market_conditions(self, conditions: MarketCondition) -> None:
        """Update current market conditions for adaptive thresholds"""
        self.market_conditions = conditions
        logger.info(f"Updated market conditions: {conditions.volatility_level} volatility, "
                   f"{conditions.trend_direction} trend")
    
    async def get_suppressed_alerts(self, limit: int = 100) -> List[Alert]:
        """Get recently suppressed alerts"""
        return self.suppressed_alerts[-limit:]
    
    async def get_filter_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        total_processed = len(self.alert_history)
        total_suppressed = len(self.suppressed_alerts)
        
        if total_processed == 0:
            return {
                'total_processed': 0,
                'total_suppressed': 0,
                'suppression_rate': 0.0,
                'active_rules': len([r for r in self.filter_rules if r.enabled])
            }
        
        return {
            'total_processed': total_processed,
            'total_suppressed': total_suppressed,
            'suppression_rate': (total_suppressed / total_processed) * 100,
            'active_rules': len([r for r in self.filter_rules if r.enabled]),
            'cache_size': len(self.deduplication_cache)
        }


class AlertAggregator:
    """
    Alert aggregation and clustering system
    """
    
    def __init__(self):
        self.clusters: Dict[str, AlertCluster] = {}
        self.pending_alerts: List[Alert] = []
        self.aggregation_rules: Dict[str, Any] = {}
        
        # Initialize default aggregation settings
        self._initialize_default_settings()
        
        logger.info("AlertAggregator initialized")
    
    def _initialize_default_settings(self) -> None:
        """Initialize default aggregation settings"""
        self.aggregation_rules = {
            'similarity_threshold': 0.7,
            'max_cluster_size': 10,
            'aggregation_window_minutes': 15,
            'min_alerts_for_clustering': 3
        }
    
    async def add_alert_for_aggregation(self, alert: Alert) -> Optional[str]:
        """Add an alert to the aggregation queue"""
        self.pending_alerts.append(alert)
        
        # Check if we should trigger aggregation
        if len(self.pending_alerts) >= self.aggregation_rules['min_alerts_for_clustering']:
            return await self._perform_aggregation()
        
        return None
    
    async def _perform_aggregation(self) -> Optional[str]:
        """Perform alert aggregation and clustering"""
        try:
            if len(self.pending_alerts) < 2:
                return None
            
            # Create feature vectors for clustering
            features = self._create_feature_vectors(self.pending_alerts)
            
            if features is None or len(features) == 0:
                return None
            
            # Perform clustering
            clusters = self._cluster_alerts(features, self.pending_alerts)
            
            # Process clusters
            cluster_id = None
            for cluster_alerts in clusters:
                if len(cluster_alerts) > 1:
                    cluster_id = await self._create_alert_cluster(cluster_alerts)
            
            # Clear pending alerts
            self.pending_alerts.clear()
            
            return cluster_id
            
        except Exception as e:
            logger.error(f"Error in alert aggregation: {e}")
            return None
    
    def _create_feature_vectors(self, alerts: List[Alert]) -> Optional[np.ndarray]:
        """Create feature vectors for alert clustering"""
        try:
            # Create text features from alert names and descriptions
            texts = []
            for alert in alerts:
                text = f"{alert.name} {alert.description}"
                texts.append(text)
            
            if not texts:
                return None
            
            # Use TF-IDF vectorization
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            text_features = vectorizer.fit_transform(texts).toarray()
            
            # Add categorical features
            categorical_features = []
            for alert in alerts:
                features = [
                    1 if alert.priority == AlertPriority.CRITICAL else 0,
                    1 if alert.priority == AlertPriority.HIGH else 0,
                    1 if alert.priority == AlertPriority.MEDIUM else 0,
                    1 if alert.trigger.trigger_type == AlertTriggerType.SEASONAL else 0,
                    1 if alert.trigger.trigger_type == AlertTriggerType.INSTITUTIONAL else 0,
                    1 if alert.trigger.trigger_type == AlertTriggerType.RISK else 0,
                    1 if alert.trigger.trigger_type == AlertTriggerType.TECHNICAL else 0,
                ]
                categorical_features.append(features)
            
            categorical_features = np.array(categorical_features)
            
            # Combine features
            combined_features = np.hstack([text_features, categorical_features])
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Error creating feature vectors: {e}")
            return None
    
    def _cluster_alerts(self, features: np.ndarray, alerts: List[Alert]) -> List[List[Alert]]:
        """Cluster alerts based on similarity"""
        try:
            # Use DBSCAN for clustering
            clustering = DBSCAN(
                eps=1 - self.aggregation_rules['similarity_threshold'],
                min_samples=2,
                metric='cosine'
            )
            
            cluster_labels = clustering.fit_predict(features)
            
            # Group alerts by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label != -1:  # -1 indicates noise/outliers
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(alerts[i])
            
            return list(clusters.values())
            
        except Exception as e:
            logger.error(f"Error clustering alerts: {e}")
            return []
    
    async def _create_alert_cluster(self, alerts: List[Alert]) -> str:
        """Create an alert cluster"""
        try:
            # Find representative alert (highest priority)
            representative = max(alerts, key=lambda a: self._get_priority_weight(a.priority))
            
            # Calculate similarity score
            similarity_score = self._calculate_cluster_similarity(alerts)
            
            # Create cluster
            cluster_id = f"cluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.clusters)}"
            
            cluster = AlertCluster(
                id=cluster_id,
                alerts=alerts,
                representative_alert=representative,
                similarity_score=similarity_score
            )
            
            self.clusters[cluster_id] = cluster
            
            logger.info(f"Created alert cluster {cluster_id} with {len(alerts)} alerts")
            
            return cluster_id
            
        except Exception as e:
            logger.error(f"Error creating alert cluster: {e}")
            return ""
    
    def _get_priority_weight(self, priority: AlertPriority) -> int:
        """Get numeric weight for priority"""
        weights = {
            AlertPriority.CRITICAL: 4,
            AlertPriority.HIGH: 3,
            AlertPriority.MEDIUM: 2,
            AlertPriority.LOW: 1
        }
        return weights.get(priority, 1)
    
    def _calculate_cluster_similarity(self, alerts: List[Alert]) -> float:
        """Calculate average similarity within a cluster"""
        if len(alerts) < 2:
            return 1.0
        
        try:
            # Create feature vectors
            features = self._create_feature_vectors(alerts)
            if features is None:
                return 0.5
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(features)
            
            # Get average similarity (excluding diagonal)
            total_similarity = 0
            count = 0
            
            for i in range(len(similarities)):
                for j in range(i + 1, len(similarities)):
                    total_similarity += similarities[i][j]
                    count += 1
            
            return total_similarity / count if count > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating cluster similarity: {e}")
            return 0.5
    
    async def get_cluster(self, cluster_id: str) -> Optional[AlertCluster]:
        """Get a cluster by ID"""
        return self.clusters.get(cluster_id)
    
    async def list_clusters(self, limit: int = 50) -> List[AlertCluster]:
        """List recent clusters"""
        clusters = list(self.clusters.values())
        clusters.sort(key=lambda c: c.created_at, reverse=True)
        return clusters[:limit]
    
    async def get_cluster_summary(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a cluster"""
        cluster = self.clusters.get(cluster_id)
        if cluster:
            return cluster.get_summary()
        return None
    
    async def force_aggregation(self) -> List[str]:
        """Force aggregation of pending alerts"""
        cluster_ids = []
        
        if self.pending_alerts:
            cluster_id = await self._perform_aggregation()
            if cluster_id:
                cluster_ids.append(cluster_id)
        
        return cluster_ids
    
    async def get_aggregation_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        total_clusters = len(self.clusters)
        total_alerts_in_clusters = sum(len(cluster.alerts) for cluster in self.clusters.values())
        
        avg_cluster_size = total_alerts_in_clusters / total_clusters if total_clusters > 0 else 0
        
        return {
            'total_clusters': total_clusters,
            'total_alerts_in_clusters': total_alerts_in_clusters,
            'average_cluster_size': round(avg_cluster_size, 2),
            'pending_alerts': len(self.pending_alerts),
            'similarity_threshold': self.aggregation_rules['similarity_threshold']
        }