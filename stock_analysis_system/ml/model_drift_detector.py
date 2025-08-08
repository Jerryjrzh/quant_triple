"""
Model Drift Detection and Monitoring System

This module implements comprehensive model drift detection using statistical methods,
performance monitoring, and automated retraining scheduling for the Stock Analysis System.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine, text
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftType(str, Enum):
    """Types of model drift."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DRIFT = "performance_drift"

class DriftSeverity(str, Enum):
    """Severity levels for drift detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DriftAlert:
    """Model drift alert information."""
    model_id: str
    drift_type: DriftType
    severity: DriftSeverity
    drift_score: float
    threshold: float
    detected_at: datetime
    description: str
    recommendations: List[str]
    metadata: Dict[str, Any]

@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics for monitoring."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    custom_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}

@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    model_id: str
    detection_timestamp: datetime
    data_drift_score: float
    concept_drift_score: float
    performance_drift_score: float
    overall_drift_score: float
    alerts: List[DriftAlert]
    recommendations: List[str]
    should_retrain: bool

class ModelDriftDetector:
    """
    Comprehensive model drift detection and monitoring system.
    
    Features:
    - Statistical drift detection using KL divergence and Jensen-Shannon distance
    - Performance monitoring with configurable thresholds
    - Automated retraining scheduling
    - A/B testing framework for model comparison
    """
    
    def __init__(self, database_url: str, mlflow_tracking_uri: str):
        """
        Initialize the model drift detector.
        
        Args:
            database_url: Database connection string
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.database_url = database_url
        self.engine = create_engine(database_url)
        
        # MLflow setup
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
        # Drift detection thresholds
        self.drift_thresholds = {
            DriftType.DATA_DRIFT: {
                DriftSeverity.LOW: 0.1,
                DriftSeverity.MEDIUM: 0.2,
                DriftSeverity.HIGH: 0.3,
                DriftSeverity.CRITICAL: 0.5
            },
            DriftType.CONCEPT_DRIFT: {
                DriftSeverity.LOW: 0.05,
                DriftSeverity.MEDIUM: 0.1,
                DriftSeverity.HIGH: 0.15,
                DriftSeverity.CRITICAL: 0.25
            },
            DriftType.PERFORMANCE_DRIFT: {
                DriftSeverity.LOW: 0.02,
                DriftSeverity.MEDIUM: 0.05,
                DriftSeverity.HIGH: 0.1,
                DriftSeverity.CRITICAL: 0.15
            }
        }
        
        # Model monitoring state
        self.monitored_models = {}
        self.baseline_data = {}
        self.baseline_performance = {}
        self.retraining_schedule = {}
        
        logger.info("ModelDriftDetector initialized successfully")
    
    async def register_model_for_monitoring(self, model_id: str, model_name: str,
                                          baseline_data: np.ndarray,
                                          baseline_performance: ModelPerformanceMetrics,
                                          monitoring_config: Dict[str, Any] = None) -> bool:
        """
        Register a model for drift monitoring.
        
        Args:
            model_id: Unique model identifier
            model_name: Human-readable model name
            baseline_data: Reference data for drift detection
            baseline_performance: Baseline performance metrics
            monitoring_config: Configuration for monitoring parameters
            
        Returns:
            bool: True if registration successful
        """
        try:
            # Default monitoring configuration
            default_config = {
                'check_frequency_hours': 24,
                'min_samples_for_drift': 100,
                'enable_data_drift': True,
                'enable_concept_drift': True,
                'enable_performance_drift': True,
                'auto_retrain_threshold': DriftSeverity.HIGH,
                'notification_channels': ['database', 'mlflow']
            }
            
            config = {**default_config, **(monitoring_config or {})}
            
            # Store model information
            self.monitored_models[model_id] = {
                'model_name': model_name,
                'registered_at': datetime.now(),
                'last_check': None,
                'config': config,
                'status': 'active'
            }
            
            # Store baseline data and performance
            self.baseline_data[model_id] = baseline_data
            self.baseline_performance[model_id] = baseline_performance
            
            # Log to MLflow
            with mlflow.start_run(run_name=f"drift_monitoring_{model_id}"):
                mlflow.log_params({
                    'model_id': model_id,
                    'model_name': model_name,
                    'monitoring_registered': True
                })
                mlflow.log_metrics({
                    'baseline_accuracy': baseline_performance.accuracy,
                    'baseline_precision': baseline_performance.precision,
                    'baseline_recall': baseline_performance.recall,
                    'baseline_f1': baseline_performance.f1_score
                })
            
            # Store in database
            await self._store_monitoring_registration(model_id, model_name, config)
            
            logger.info(f"Model {model_id} registered for drift monitoring")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id} for monitoring: {str(e)}")
            return False
    
    async def detect_drift(self, model_id: str, new_data: np.ndarray,
                          new_predictions: np.ndarray = None,
                          true_labels: np.ndarray = None) -> DriftDetectionResult:
        """
        Detect drift for a monitored model.
        
        Args:
            model_id: Model identifier
            new_data: New input data for drift detection
            new_predictions: Model predictions on new data
            true_labels: True labels for performance drift detection
            
        Returns:
            DriftDetectionResult: Comprehensive drift detection results
        """
        if model_id not in self.monitored_models:
            raise ValueError(f"Model {model_id} is not registered for monitoring")
        
        detection_timestamp = datetime.now()
        alerts = []
        recommendations = []
        
        # Get baseline data and performance
        baseline_data = self.baseline_data[model_id]
        baseline_performance = self.baseline_performance[model_id]
        config = self.monitored_models[model_id]['config']
        
        # Initialize drift scores
        data_drift_score = 0.0
        concept_drift_score = 0.0
        performance_drift_score = 0.0
        
        # 1. Data Drift Detection
        if config['enable_data_drift'] and len(new_data) >= config['min_samples_for_drift']:
            data_drift_score = await self._detect_data_drift(baseline_data, new_data)
            
            # Check for data drift alerts
            data_drift_alerts = self._evaluate_drift_severity(
                model_id, DriftType.DATA_DRIFT, data_drift_score
            )
            alerts.extend(data_drift_alerts)
        
        # 2. Concept Drift Detection (requires predictions)
        if (config['enable_concept_drift'] and new_predictions is not None and 
            len(new_predictions) >= config['min_samples_for_drift']):
            concept_drift_score = await self._detect_concept_drift(
                baseline_data, new_data, new_predictions
            )
            
            # Check for concept drift alerts
            concept_drift_alerts = self._evaluate_drift_severity(
                model_id, DriftType.CONCEPT_DRIFT, concept_drift_score
            )
            alerts.extend(concept_drift_alerts)
        
        # 3. Performance Drift Detection (requires true labels)
        if (config['enable_performance_drift'] and new_predictions is not None and 
            true_labels is not None and len(true_labels) >= config['min_samples_for_drift']):
            
            # Calculate current performance
            current_performance = ModelPerformanceMetrics(
                accuracy=accuracy_score(true_labels, new_predictions),
                precision=precision_score(true_labels, new_predictions, average='weighted', zero_division=0),
                recall=recall_score(true_labels, new_predictions, average='weighted', zero_division=0),
                f1_score=f1_score(true_labels, new_predictions, average='weighted', zero_division=0)
            )
            
            performance_drift_score = await self._detect_performance_drift(
                baseline_performance, current_performance
            )
            
            # Check for performance drift alerts
            performance_drift_alerts = self._evaluate_drift_severity(
                model_id, DriftType.PERFORMANCE_DRIFT, performance_drift_score
            )
            alerts.extend(performance_drift_alerts)
        
        # Calculate overall drift score
        overall_drift_score = max(data_drift_score, concept_drift_score, performance_drift_score)
        
        # Generate recommendations
        recommendations = self._generate_drift_recommendations(
            data_drift_score, concept_drift_score, performance_drift_score, alerts
        )
        
        # Determine if retraining is needed
        should_retrain = self._should_trigger_retraining(alerts, config)
        
        # Create result
        result = DriftDetectionResult(
            model_id=model_id,
            detection_timestamp=detection_timestamp,
            data_drift_score=data_drift_score,
            concept_drift_score=concept_drift_score,
            performance_drift_score=performance_drift_score,
            overall_drift_score=overall_drift_score,
            alerts=alerts,
            recommendations=recommendations,
            should_retrain=should_retrain
        )
        
        # Log results
        await self._log_drift_detection_results(result)
        
        # Update monitoring state
        self.monitored_models[model_id]['last_check'] = detection_timestamp
        
        logger.info(f"Drift detection completed for model {model_id}. Overall score: {overall_drift_score:.4f}")
        
        return result
    
    async def _detect_data_drift(self, baseline_data: np.ndarray, new_data: np.ndarray) -> float:
        """
        Detect data drift using statistical methods.
        
        Args:
            baseline_data: Reference data distribution
            new_data: New data distribution
            
        Returns:
            float: Data drift score (0-1, higher means more drift)
        """
        try:
            # Ensure data is 2D
            if baseline_data.ndim == 1:
                baseline_data = baseline_data.reshape(-1, 1)
            if new_data.ndim == 1:
                new_data = new_data.reshape(-1, 1)
            
            drift_scores = []
            
            # Calculate drift for each feature
            for feature_idx in range(baseline_data.shape[1]):
                baseline_feature = baseline_data[:, feature_idx]
                new_feature = new_data[:, feature_idx]
                
                # Remove NaN values
                baseline_feature = baseline_feature[~np.isnan(baseline_feature)]
                new_feature = new_feature[~np.isnan(new_feature)]
                
                if len(baseline_feature) == 0 or len(new_feature) == 0:
                    continue
                
                # Method 1: Jensen-Shannon Distance
                try:
                    # Create histograms
                    min_val = min(baseline_feature.min(), new_feature.min())
                    max_val = max(baseline_feature.max(), new_feature.max())
                    bins = np.linspace(min_val, max_val, 50)
                    
                    baseline_hist, _ = np.histogram(baseline_feature, bins=bins, density=True)
                    new_hist, _ = np.histogram(new_feature, bins=bins, density=True)
                    
                    # Normalize to probability distributions
                    baseline_hist = baseline_hist / np.sum(baseline_hist)
                    new_hist = new_hist / np.sum(new_hist)
                    
                    # Add small epsilon to avoid division by zero
                    epsilon = 1e-10
                    baseline_hist += epsilon
                    new_hist += epsilon
                    
                    # Calculate Jensen-Shannon distance
                    js_distance = jensenshannon(baseline_hist, new_hist)
                    drift_scores.append(js_distance)
                    
                except Exception as e:
                    logger.warning(f"JS distance calculation failed for feature {feature_idx}: {str(e)}")
                
                # Method 2: Kolmogorov-Smirnov test
                try:
                    ks_statistic, ks_p_value = stats.ks_2samp(baseline_feature, new_feature)
                    # Convert p-value to drift score (lower p-value = higher drift)
                    ks_drift_score = 1.0 - ks_p_value
                    drift_scores.append(ks_drift_score)
                    
                except Exception as e:
                    logger.warning(f"KS test failed for feature {feature_idx}: {str(e)}")
            
            # Return average drift score across all features
            return np.mean(drift_scores) if drift_scores else 0.0
            
        except Exception as e:
            logger.error(f"Data drift detection failed: {str(e)}")
            return 0.0
    
    async def _detect_concept_drift(self, baseline_data: np.ndarray, new_data: np.ndarray,
                                  new_predictions: np.ndarray) -> float:
        """
        Detect concept drift by analyzing prediction patterns.
        
        Args:
            baseline_data: Reference input data
            new_data: New input data
            new_predictions: Model predictions on new data
            
        Returns:
            float: Concept drift score (0-1, higher means more drift)
        """
        try:
            # Simple concept drift detection based on prediction distribution changes
            # In practice, this would be more sophisticated
            
            # Calculate prediction statistics
            pred_mean = np.mean(new_predictions)
            pred_std = np.std(new_predictions)
            
            # Compare with expected ranges (simplified approach)
            # This would typically use historical prediction patterns
            expected_mean_range = (0.3, 0.7)  # Example range
            expected_std_range = (0.1, 0.4)   # Example range
            
            # Calculate drift based on deviation from expected ranges
            mean_drift = 0.0
            if pred_mean < expected_mean_range[0]:
                mean_drift = (expected_mean_range[0] - pred_mean) / expected_mean_range[0]
            elif pred_mean > expected_mean_range[1]:
                mean_drift = (pred_mean - expected_mean_range[1]) / expected_mean_range[1]
            
            std_drift = 0.0
            if pred_std < expected_std_range[0]:
                std_drift = (expected_std_range[0] - pred_std) / expected_std_range[0]
            elif pred_std > expected_std_range[1]:
                std_drift = (pred_std - expected_std_range[1]) / expected_std_range[1]
            
            # Combine drift scores
            concept_drift_score = min(1.0, (mean_drift + std_drift) / 2.0)
            
            return concept_drift_score
            
        except Exception as e:
            logger.error(f"Concept drift detection failed: {str(e)}")
            return 0.0
    
    async def _detect_performance_drift(self, baseline_performance: ModelPerformanceMetrics,
                                      current_performance: ModelPerformanceMetrics) -> float:
        """
        Detect performance drift by comparing current vs baseline metrics.
        
        Args:
            baseline_performance: Reference performance metrics
            current_performance: Current performance metrics
            
        Returns:
            float: Performance drift score (0-1, higher means more drift)
        """
        try:
            # Calculate relative performance changes
            accuracy_drift = abs(baseline_performance.accuracy - current_performance.accuracy) / baseline_performance.accuracy
            precision_drift = abs(baseline_performance.precision - current_performance.precision) / baseline_performance.precision
            recall_drift = abs(baseline_performance.recall - current_performance.recall) / baseline_performance.recall
            f1_drift = abs(baseline_performance.f1_score - current_performance.f1_score) / baseline_performance.f1_score
            
            # Weight the metrics (accuracy and F1 are more important)
            weighted_drift = (
                accuracy_drift * 0.3 +
                precision_drift * 0.2 +
                recall_drift * 0.2 +
                f1_drift * 0.3
            )
            
            return min(1.0, weighted_drift)
            
        except Exception as e:
            logger.error(f"Performance drift detection failed: {str(e)}")
            return 0.0
    
    def _evaluate_drift_severity(self, model_id: str, drift_type: DriftType, 
                               drift_score: float) -> List[DriftAlert]:
        """
        Evaluate drift severity and create alerts if necessary.
        
        Args:
            model_id: Model identifier
            drift_type: Type of drift detected
            drift_score: Drift score value
            
        Returns:
            List[DriftAlert]: List of drift alerts
        """
        alerts = []
        thresholds = self.drift_thresholds[drift_type]
        
        # Determine severity level
        severity = None
        threshold = 0.0
        
        if drift_score >= thresholds[DriftSeverity.CRITICAL]:
            severity = DriftSeverity.CRITICAL
            threshold = thresholds[DriftSeverity.CRITICAL]
        elif drift_score >= thresholds[DriftSeverity.HIGH]:
            severity = DriftSeverity.HIGH
            threshold = thresholds[DriftSeverity.HIGH]
        elif drift_score >= thresholds[DriftSeverity.MEDIUM]:
            severity = DriftSeverity.MEDIUM
            threshold = thresholds[DriftSeverity.MEDIUM]
        elif drift_score >= thresholds[DriftSeverity.LOW]:
            severity = DriftSeverity.LOW
            threshold = thresholds[DriftSeverity.LOW]
        
        # Create alert if drift detected
        if severity:
            recommendations = self._get_drift_recommendations(drift_type, severity)
            
            alert = DriftAlert(
                model_id=model_id,
                drift_type=drift_type,
                severity=severity,
                drift_score=drift_score,
                threshold=threshold,
                detected_at=datetime.now(),
                description=f"{drift_type.value} detected with {severity.value} severity (score: {drift_score:.4f})",
                recommendations=recommendations,
                metadata={
                    'model_name': self.monitored_models[model_id]['model_name'],
                    'detection_method': 'statistical_analysis'
                }
            )
            
            alerts.append(alert)
        
        return alerts
    
    def _get_drift_recommendations(self, drift_type: DriftType, severity: DriftSeverity) -> List[str]:
        """
        Get recommendations based on drift type and severity.
        
        Args:
            drift_type: Type of drift detected
            severity: Severity level
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        if drift_type == DriftType.DATA_DRIFT:
            if severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                recommendations.extend([
                    "Consider retraining the model with recent data",
                    "Investigate data source changes or data quality issues",
                    "Review feature engineering pipeline for consistency"
                ])
            else:
                recommendations.extend([
                    "Monitor data drift trends closely",
                    "Consider updating data preprocessing steps"
                ])
        
        elif drift_type == DriftType.CONCEPT_DRIFT:
            if severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                recommendations.extend([
                    "Immediate model retraining recommended",
                    "Review model architecture and feature selection",
                    "Consider ensemble methods or adaptive learning"
                ])
            else:
                recommendations.extend([
                    "Increase monitoring frequency",
                    "Prepare for potential model update"
                ])
        
        elif drift_type == DriftType.PERFORMANCE_DRIFT:
            if severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                recommendations.extend([
                    "Urgent model retraining required",
                    "Investigate root cause of performance degradation",
                    "Consider rolling back to previous model version"
                ])
            else:
                recommendations.extend([
                    "Schedule model performance review",
                    "Collect more recent training data"
                ])
        
        return recommendations
    
    def _generate_drift_recommendations(self, data_drift_score: float, concept_drift_score: float,
                                      performance_drift_score: float, alerts: List[DriftAlert]) -> List[str]:
        """
        Generate comprehensive recommendations based on all drift scores.
        
        Args:
            data_drift_score: Data drift score
            concept_drift_score: Concept drift score
            performance_drift_score: Performance drift score
            alerts: List of drift alerts
            
        Returns:
            List[str]: Comprehensive recommendations
        """
        recommendations = []
        
        # Overall drift assessment
        max_drift = max(data_drift_score, concept_drift_score, performance_drift_score)
        
        if max_drift > 0.3:
            recommendations.append("High drift detected - immediate attention required")
        elif max_drift > 0.2:
            recommendations.append("Moderate drift detected - schedule model review")
        elif max_drift > 0.1:
            recommendations.append("Low drift detected - continue monitoring")
        
        # Specific recommendations from alerts
        for alert in alerts:
            recommendations.extend(alert.recommendations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _should_trigger_retraining(self, alerts: List[DriftAlert], config: Dict[str, Any]) -> bool:
        """
        Determine if automatic retraining should be triggered.
        
        Args:
            alerts: List of drift alerts
            config: Model monitoring configuration
            
        Returns:
            bool: True if retraining should be triggered
        """
        auto_retrain_threshold = config.get('auto_retrain_threshold', DriftSeverity.HIGH)
        
        for alert in alerts:
            if alert.severity == DriftSeverity.CRITICAL:
                return True
            elif alert.severity == DriftSeverity.HIGH and auto_retrain_threshold in [DriftSeverity.HIGH, DriftSeverity.MEDIUM]:
                return True
            elif alert.severity == DriftSeverity.MEDIUM and auto_retrain_threshold == DriftSeverity.MEDIUM:
                return True
        
        return False
    
    async def _log_drift_detection_results(self, result: DriftDetectionResult) -> None:
        """
        Log drift detection results to MLflow and database.
        
        Args:
            result: Drift detection results
        """
        try:
            # Log to MLflow
            with mlflow.start_run(run_name=f"drift_detection_{result.model_id}_{result.detection_timestamp.strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_metrics({
                    'data_drift_score': result.data_drift_score,
                    'concept_drift_score': result.concept_drift_score,
                    'performance_drift_score': result.performance_drift_score,
                    'overall_drift_score': result.overall_drift_score,
                    'num_alerts': len(result.alerts),
                    'should_retrain': int(result.should_retrain)
                })
                
                mlflow.log_params({
                    'model_id': result.model_id,
                    'detection_timestamp': result.detection_timestamp.isoformat()
                })
                
                # Log alerts as artifacts
                if result.alerts:
                    alerts_data = [asdict(alert) for alert in result.alerts]
                    mlflow.log_dict(alerts_data, "drift_alerts.json")
            
            # Log to database
            await self._store_drift_results_in_database(result)
            
        except Exception as e:
            logger.error(f"Failed to log drift detection results: {str(e)}")
    
    async def _store_monitoring_registration(self, model_id: str, model_name: str, 
                                           config: Dict[str, Any]) -> None:
        """Store model monitoring registration in database."""
        try:
            query = text("""
                INSERT INTO model_monitoring_registry 
                (model_id, model_name, config, registered_at, status)
                VALUES (:model_id, :model_name, :config, :registered_at, :status)
                ON CONFLICT (model_id) DO UPDATE SET
                    model_name = EXCLUDED.model_name,
                    config = EXCLUDED.config,
                    registered_at = EXCLUDED.registered_at,
                    status = EXCLUDED.status
            """)
            
            with self.engine.connect() as conn:
                conn.execute(query, {
                    'model_id': model_id,
                    'model_name': model_name,
                    'config': json.dumps(config),
                    'registered_at': datetime.now(),
                    'status': 'active'
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store monitoring registration: {str(e)}")
    
    async def _store_drift_results_in_database(self, result: DriftDetectionResult) -> None:
        """Store drift detection results in database."""
        try:
            # Store main drift detection result
            query = text("""
                INSERT INTO model_drift_detection_results 
                (model_id, detection_timestamp, data_drift_score, concept_drift_score, 
                 performance_drift_score, overall_drift_score, should_retrain, 
                 recommendations, created_at)
                VALUES (:model_id, :detection_timestamp, :data_drift_score, :concept_drift_score,
                        :performance_drift_score, :overall_drift_score, :should_retrain,
                        :recommendations, :created_at)
            """)
            
            with self.engine.connect() as conn:
                conn.execute(query, {
                    'model_id': result.model_id,
                    'detection_timestamp': result.detection_timestamp,
                    'data_drift_score': result.data_drift_score,
                    'concept_drift_score': result.concept_drift_score,
                    'performance_drift_score': result.performance_drift_score,
                    'overall_drift_score': result.overall_drift_score,
                    'should_retrain': result.should_retrain,
                    'recommendations': json.dumps(result.recommendations),
                    'created_at': datetime.now()
                })
                
                # Store alerts
                for alert in result.alerts:
                    alert_query = text("""
                        INSERT INTO model_drift_alerts 
                        (model_id, drift_type, severity, drift_score, threshold, 
                         detected_at, description, recommendations, metadata, created_at)
                        VALUES (:model_id, :drift_type, :severity, :drift_score, :threshold,
                                :detected_at, :description, :recommendations, :metadata, :created_at)
                    """)
                    
                    conn.execute(alert_query, {
                        'model_id': alert.model_id,
                        'drift_type': alert.drift_type.value,
                        'severity': alert.severity.value,
                        'drift_score': alert.drift_score,
                        'threshold': alert.threshold,
                        'detected_at': alert.detected_at,
                        'description': alert.description,
                        'recommendations': json.dumps(alert.recommendations),
                        'metadata': json.dumps(alert.metadata),
                        'created_at': datetime.now()
                    })
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store drift results in database: {str(e)}")
    
    async def get_model_drift_history(self, model_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get drift detection history for a model.
        
        Args:
            model_id: Model identifier
            days: Number of days of history to retrieve
            
        Returns:
            List[Dict]: Drift detection history
        """
        try:
            query = text("""
                SELECT * FROM model_drift_detection_results 
                WHERE model_id = :model_id 
                AND detection_timestamp >= :start_date
                ORDER BY detection_timestamp DESC
            """)
            
            start_date = datetime.now() - timedelta(days=days)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {
                    'model_id': model_id,
                    'start_date': start_date
                })
                
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            logger.error(f"Failed to get drift history: {str(e)}")
            return []
    
    async def schedule_automated_retraining(self, model_id: str, schedule_config: Dict[str, Any]) -> bool:
        """
        Schedule automated model retraining.
        
        Args:
            model_id: Model identifier
            schedule_config: Retraining schedule configuration
            
        Returns:
            bool: True if scheduling successful
        """
        try:
            default_config = {
                'frequency': 'weekly',  # daily, weekly, monthly
                'drift_threshold': DriftSeverity.HIGH,
                'min_samples': 1000,
                'validation_split': 0.2,
                'notification_channels': ['email', 'slack']
            }
            
            config = {**default_config, **schedule_config}
            
            # Calculate next retraining date
            next_retrain = self._calculate_next_retrain_date(config['frequency'])
            
            self.retraining_schedule[model_id] = {
                'config': config,
                'next_retrain': next_retrain,
                'last_retrain': None,
                'status': 'scheduled'
            }
            
            # Store in database
            query = text("""
                INSERT INTO model_retraining_schedule 
                (model_id, schedule_config, next_retrain, status, created_at)
                VALUES (:model_id, :schedule_config, :next_retrain, :status, :created_at)
                ON CONFLICT (model_id) DO UPDATE SET
                    schedule_config = EXCLUDED.schedule_config,
                    next_retrain = EXCLUDED.next_retrain,
                    status = EXCLUDED.status,
                    updated_at = :created_at
            """)
            
            with self.engine.connect() as conn:
                conn.execute(query, {
                    'model_id': model_id,
                    'schedule_config': json.dumps(config),
                    'next_retrain': next_retrain,
                    'status': 'scheduled',
                    'created_at': datetime.now()
                })
                conn.commit()
            
            logger.info(f"Automated retraining scheduled for model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule automated retraining: {str(e)}")
            return False
    
    def _calculate_next_retrain_date(self, frequency: str) -> datetime:
        """Calculate next retraining date based on frequency."""
        current_time = datetime.now()
        
        if frequency == 'daily':
            return current_time + timedelta(days=1)
        elif frequency == 'weekly':
            return current_time + timedelta(weeks=1)
        elif frequency == 'monthly':
            return current_time + timedelta(days=30)
        else:
            return current_time + timedelta(weeks=1)  # Default to weekly
    
    async def check_models_due_for_retraining(self) -> List[str]:
        """
        Check which models are due for retraining.
        
        Returns:
            List[str]: List of model IDs due for retraining
        """
        due_models = []
        current_time = datetime.now()
        
        for model_id, schedule_info in self.retraining_schedule.items():
            if (schedule_info['status'] == 'scheduled' and 
                current_time >= schedule_info['next_retrain']):
                due_models.append(model_id)
        
        return due_models
    
    async def cleanup_old_drift_data(self, retention_days: int = 90) -> int:
        """
        Clean up old drift detection data.
        
        Args:
            retention_days: Number of days to retain data
            
        Returns:
            int: Number of records cleaned up
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            with self.engine.connect() as conn:
                # Clean up drift detection results
                result_query = text("""
                    DELETE FROM model_drift_detection_results 
                    WHERE detection_timestamp < :cutoff_date
                """)
                
                result = conn.execute(result_query, {'cutoff_date': cutoff_date})
                results_deleted = result.rowcount
                
                # Clean up drift alerts
                alert_query = text("""
                    DELETE FROM model_drift_alerts 
                    WHERE detected_at < :cutoff_date
                """)
                
                conn.execute(alert_query, {'cutoff_date': cutoff_date})
                conn.commit()
                
                logger.info(f"Cleaned up {results_deleted} old drift detection records")
                return results_deleted
                
        except Exception as e:
            logger.error(f"Failed to cleanup old drift data: {str(e)}")
            return 0