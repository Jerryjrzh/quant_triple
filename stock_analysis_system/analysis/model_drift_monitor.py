"""
Model Drift Detection and Monitoring System

This module provides advanced model drift detection and monitoring capabilities including:
- Statistical drift detection using multiple methods (KL divergence, KS test, PSI)
- Model performance monitoring and degradation detection
- Automated alerting system for drift and performance issues
- A/B testing framework for model comparison
- Comprehensive monitoring dashboards and reports
"""

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, entropy, ks_2samp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from config.settings import get_settings

from .ml_model_manager import DriftDetectionResult, MLModelManager, ModelInfo

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    """Types of drift that can be detected."""

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Model drift alert information."""

    alert_id: str
    model_id: str
    drift_type: DriftType
    severity: AlertSeverity
    drift_score: float
    threshold: float
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class PerformanceMetrics:
    """Model performance metrics over time."""

    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """A/B test comparison result."""

    test_id: str
    model_a_id: str
    model_b_id: str
    metric_name: str
    model_a_score: float
    model_b_score: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    winner: Optional[str]
    test_duration_days: int
    sample_size: int


class PopulationStabilityIndex:
    """Population Stability Index (PSI) calculator for drift detection."""

    @staticmethod
    def calculate_psi(
        expected: np.ndarray, actual: np.ndarray, bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index between expected and actual distributions.

        Args:
            expected: Reference/expected distribution
            actual: Actual/new distribution
            bins: Number of bins for discretization

        Returns:
            PSI value (0 = no drift, >0.25 = significant drift)
        """
        try:
            # Create bins based on expected distribution
            _, bin_edges = np.histogram(expected, bins=bins)

            # Calculate frequencies for both distributions
            expected_freq, _ = np.histogram(expected, bins=bin_edges, density=True)
            actual_freq, _ = np.histogram(actual, bins=bin_edges, density=True)

            # Normalize to get percentages
            expected_pct = expected_freq / expected_freq.sum()
            actual_pct = actual_freq / actual_freq.sum()

            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            expected_pct = np.maximum(expected_pct, epsilon)
            actual_pct = np.maximum(actual_pct, epsilon)

            # Calculate PSI
            psi = np.sum(
                (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
            )

            return float(psi)

        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0


class ModelDriftMonitor:
    """
    Advanced model drift detection and monitoring system.

    This class provides comprehensive monitoring capabilities including:
    - Multiple drift detection methods
    - Performance monitoring and alerting
    - A/B testing framework
    - Automated monitoring workflows
    """

    def __init__(self, ml_manager: MLModelManager):
        """
        Initialize Model Drift Monitor.

        Args:
            ml_manager: MLModelManager instance for model operations
        """
        self.ml_manager = ml_manager
        self.settings = get_settings()

        # Monitoring state
        self.alerts: List[DriftAlert] = []
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.drift_history: Dict[str, List[DriftDetectionResult]] = {}
        self.ab_tests: Dict[str, ABTestResult] = {}

        # Configuration
        self.drift_thresholds = {
            DriftType.DATA_DRIFT: 0.1,
            DriftType.CONCEPT_DRIFT: 0.15,
            DriftType.PREDICTION_DRIFT: 0.1,
            DriftType.PERFORMANCE_DRIFT: 0.05,
        }

        self.psi_calculator = PopulationStabilityIndex()

        logger.info("ModelDriftMonitor initialized")

    async def detect_comprehensive_drift(
        self,
        model_id: str,
        new_data: np.ndarray,
        reference_data: np.ndarray,
        new_labels: Optional[np.ndarray] = None,
        reference_labels: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[DriftType, DriftDetectionResult]:
        """
        Perform comprehensive drift detection using multiple methods.

        Args:
            model_id: ID of the model to monitor
            new_data: New input data
            reference_data: Reference input data (training data)
            new_labels: New labels (if available)
            reference_labels: Reference labels (if available)
            feature_names: Feature names for detailed analysis

        Returns:
            Dictionary of drift detection results by drift type
        """
        results = {}

        try:
            # 1. Data Drift Detection
            data_drift = await self._detect_data_drift(
                model_id, new_data, reference_data, feature_names
            )
            results[DriftType.DATA_DRIFT] = data_drift

            # 2. Prediction Drift Detection (if model is available)
            prediction_drift = await self._detect_prediction_drift(
                model_id, new_data, reference_data
            )
            if prediction_drift:
                results[DriftType.PREDICTION_DRIFT] = prediction_drift

            # 3. Concept Drift Detection (if labels are available)
            if new_labels is not None and reference_labels is not None:
                concept_drift = await self._detect_concept_drift(
                    model_id, new_data, reference_data, new_labels, reference_labels
                )
                results[DriftType.CONCEPT_DRIFT] = concept_drift

            # 4. Performance Drift Detection
            performance_drift = await self._detect_performance_drift(
                model_id, new_data, new_labels
            )
            if performance_drift:
                results[DriftType.PERFORMANCE_DRIFT] = performance_drift

            # Store drift history
            if model_id not in self.drift_history:
                self.drift_history[model_id] = []

            for drift_type, drift_result in results.items():
                drift_result.details["drift_type"] = drift_type.value
                self.drift_history[model_id].append(drift_result)

            # Generate alerts if necessary
            await self._generate_drift_alerts(model_id, results)

            logger.info(f"Comprehensive drift detection completed for {model_id}")
            return results

        except Exception as e:
            logger.error(f"Error in comprehensive drift detection for {model_id}: {e}")
            return {}

    async def _detect_data_drift(
        self,
        model_id: str,
        new_data: np.ndarray,
        reference_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> DriftDetectionResult:
        """Detect data drift using multiple statistical methods."""
        try:
            if new_data.ndim == 1:
                new_data = new_data.reshape(-1, 1)
                reference_data = reference_data.reshape(-1, 1)

            n_features = new_data.shape[1]
            feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]

            drift_scores = []
            feature_details = {}

            for i in range(n_features):
                feature_name = feature_names[i]

                # KS Test
                ks_stat, ks_p_value = ks_2samp(reference_data[:, i], new_data[:, i])

                # KL Divergence
                try:
                    ref_hist, bins = np.histogram(
                        reference_data[:, i], bins=50, density=True
                    )
                    new_hist, _ = np.histogram(new_data[:, i], bins=bins, density=True)

                    epsilon = 1e-10
                    ref_hist = ref_hist + epsilon
                    new_hist = new_hist + epsilon

                    ref_hist = ref_hist / ref_hist.sum()
                    new_hist = new_hist / new_hist.sum()

                    kl_div = entropy(new_hist, ref_hist)
                except Exception:
                    kl_div = 0.0

                # Population Stability Index
                psi = self.psi_calculator.calculate_psi(
                    reference_data[:, i], new_data[:, i]
                )

                # Combined drift score
                drift_score = max(ks_stat, min(kl_div, 1.0), min(psi, 1.0))
                drift_scores.append(drift_score)

                feature_details[feature_name] = {
                    "ks_statistic": ks_stat,
                    "ks_p_value": ks_p_value,
                    "kl_divergence": kl_div,
                    "psi": psi,
                    "drift_score": drift_score,
                }

            overall_drift_score = np.mean(drift_scores)
            drift_detected = (
                overall_drift_score > self.drift_thresholds[DriftType.DATA_DRIFT]
            )

            # Calculate confidence
            confidence = 1.0 - (np.std(drift_scores) / (np.mean(drift_scores) + 1e-10))
            confidence = max(0.0, min(confidence, 1.0))

            return DriftDetectionResult(
                drift_detected=drift_detected,
                drift_score=overall_drift_score,
                drift_type=DriftType.DATA_DRIFT.value,
                confidence=confidence,
                details={
                    "feature_drift_scores": drift_scores,
                    "feature_drift_details": feature_details,
                    "n_features": n_features,
                    "threshold": self.drift_thresholds[DriftType.DATA_DRIFT],
                },
            )

        except Exception as e:
            logger.error(f"Error in data drift detection: {e}")
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type=DriftType.DATA_DRIFT.value,
                confidence=0.0,
                details={"error": str(e)},
            )

    async def _detect_prediction_drift(
        self, model_id: str, new_data: np.ndarray, reference_data: np.ndarray
    ) -> Optional[DriftDetectionResult]:
        """Detect prediction drift by comparing model outputs."""
        try:
            # Load model
            model = await self.ml_manager.load_model(model_id, stage="Production")
            if model is None:
                model = await self.ml_manager.load_model(model_id, stage="Staging")

            if model is None:
                logger.warning(
                    f"Could not load model {model_id} for prediction drift detection"
                )
                return None

            # Get predictions
            ref_predictions = model.predict(reference_data)
            new_predictions = model.predict(new_data)

            # For classification, compare prediction distributions
            if hasattr(model, "predict_proba"):
                ref_proba = model.predict_proba(reference_data)
                new_proba = model.predict_proba(new_data)

                # Compare probability distributions
                drift_scores = []
                for class_idx in range(ref_proba.shape[1]):
                    ks_stat, _ = ks_2samp(
                        ref_proba[:, class_idx], new_proba[:, class_idx]
                    )
                    drift_scores.append(ks_stat)

                overall_drift_score = np.mean(drift_scores)
            else:
                # For regression, compare prediction distributions
                ks_stat, ks_p_value = ks_2samp(ref_predictions, new_predictions)
                overall_drift_score = ks_stat

            drift_detected = (
                overall_drift_score > self.drift_thresholds[DriftType.PREDICTION_DRIFT]
            )

            return DriftDetectionResult(
                drift_detected=drift_detected,
                drift_score=overall_drift_score,
                drift_type=DriftType.PREDICTION_DRIFT.value,
                confidence=0.8,  # Fixed confidence for prediction drift
                details={
                    "reference_predictions_mean": float(np.mean(ref_predictions)),
                    "new_predictions_mean": float(np.mean(new_predictions)),
                    "reference_predictions_std": float(np.std(ref_predictions)),
                    "new_predictions_std": float(np.std(new_predictions)),
                    "threshold": self.drift_thresholds[DriftType.PREDICTION_DRIFT],
                },
            )

        except Exception as e:
            logger.error(f"Error in prediction drift detection: {e}")
            return None

    async def _detect_concept_drift(
        self,
        model_id: str,
        new_data: np.ndarray,
        reference_data: np.ndarray,
        new_labels: np.ndarray,
        reference_labels: np.ndarray,
    ) -> DriftDetectionResult:
        """Detect concept drift by analyzing label distributions and relationships."""
        try:
            # Compare label distributions
            if len(np.unique(reference_labels)) <= 10:  # Categorical
                # Chi-square test for categorical labels
                ref_counts = np.bincount(reference_labels.astype(int))
                new_counts = np.bincount(
                    new_labels.astype(int), minlength=len(ref_counts)
                )

                # Ensure same length
                max_len = max(len(ref_counts), len(new_counts))
                ref_counts = np.pad(ref_counts, (0, max_len - len(ref_counts)))
                new_counts = np.pad(new_counts, (0, max_len - len(new_counts)))

                try:
                    chi2_stat, p_value = stats.chisquare(new_counts + 1, ref_counts + 1)
                    drift_score = min(chi2_stat / 100.0, 1.0)  # Normalize
                except Exception:
                    drift_score = 0.0
                    p_value = 1.0
            else:
                # KS test for continuous labels
                ks_stat, p_value = ks_2samp(reference_labels, new_labels)
                drift_score = ks_stat

            drift_detected = (
                drift_score > self.drift_thresholds[DriftType.CONCEPT_DRIFT]
            )

            return DriftDetectionResult(
                drift_detected=drift_detected,
                drift_score=drift_score,
                drift_type=DriftType.CONCEPT_DRIFT.value,
                confidence=1.0 - p_value,
                details={
                    "p_value": p_value,
                    "reference_label_mean": float(np.mean(reference_labels)),
                    "new_label_mean": float(np.mean(new_labels)),
                    "reference_label_std": float(np.std(reference_labels)),
                    "new_label_std": float(np.std(new_labels)),
                    "threshold": self.drift_thresholds[DriftType.CONCEPT_DRIFT],
                },
            )

        except Exception as e:
            logger.error(f"Error in concept drift detection: {e}")
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type=DriftType.CONCEPT_DRIFT.value,
                confidence=0.0,
                details={"error": str(e)},
            )

    async def _detect_performance_drift(
        self,
        model_id: str,
        new_data: np.ndarray,
        new_labels: Optional[np.ndarray] = None,
    ) -> Optional[DriftDetectionResult]:
        """Detect performance drift by comparing current vs historical performance."""
        try:
            if new_labels is None:
                logger.info("No labels provided for performance drift detection")
                return None

            # Load model
            model = await self.ml_manager.load_model(model_id, stage="Production")
            if model is None:
                model = await self.ml_manager.load_model(model_id, stage="Staging")

            if model is None:
                logger.warning(
                    f"Could not load model {model_id} for performance drift detection"
                )
                return None

            # Calculate current performance
            predictions = model.predict(new_data)
            current_accuracy = accuracy_score(new_labels, predictions)

            # Get historical performance
            if model_id not in self.performance_history:
                # First measurement, store as baseline
                self.performance_history[model_id] = []
                baseline_accuracy = current_accuracy
            else:
                # Calculate baseline from recent history
                recent_metrics = self.performance_history[model_id][
                    -10:
                ]  # Last 10 measurements
                if recent_metrics:
                    baseline_accuracy = np.mean([m.accuracy for m in recent_metrics])
                else:
                    baseline_accuracy = current_accuracy

            # Calculate performance drift
            performance_change = abs(current_accuracy - baseline_accuracy)
            drift_score = performance_change
            drift_detected = (
                drift_score > self.drift_thresholds[DriftType.PERFORMANCE_DRIFT]
            )

            # Store current performance
            current_metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                accuracy=current_accuracy,
                precision=precision_score(new_labels, predictions, average="weighted"),
                recall=recall_score(new_labels, predictions, average="weighted"),
                f1_score=f1_score(new_labels, predictions, average="weighted"),
            )

            if model_id not in self.performance_history:
                self.performance_history[model_id] = []
            self.performance_history[model_id].append(current_metrics)

            return DriftDetectionResult(
                drift_detected=drift_detected,
                drift_score=drift_score,
                drift_type=DriftType.PERFORMANCE_DRIFT.value,
                confidence=0.9,
                details={
                    "current_accuracy": current_accuracy,
                    "baseline_accuracy": baseline_accuracy,
                    "performance_change": performance_change,
                    "threshold": self.drift_thresholds[DriftType.PERFORMANCE_DRIFT],
                },
            )

        except Exception as e:
            logger.error(f"Error in performance drift detection: {e}")
            return None

    async def _generate_drift_alerts(
        self, model_id: str, drift_results: Dict[DriftType, DriftDetectionResult]
    ) -> None:
        """Generate alerts based on drift detection results."""
        try:
            for drift_type, result in drift_results.items():
                if result.drift_detected:
                    # Determine severity
                    if result.drift_score > 0.5:
                        severity = AlertSeverity.CRITICAL
                    elif result.drift_score > 0.3:
                        severity = AlertSeverity.HIGH
                    elif result.drift_score > 0.15:
                        severity = AlertSeverity.MEDIUM
                    else:
                        severity = AlertSeverity.LOW

                    # Create alert
                    alert = DriftAlert(
                        alert_id=f"{model_id}_{drift_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        model_id=model_id,
                        drift_type=drift_type,
                        severity=severity,
                        drift_score=result.drift_score,
                        threshold=self.drift_thresholds[drift_type],
                        message=f"{drift_type.value.replace('_', ' ').title()} detected for model {model_id}. "
                        f"Score: {result.drift_score:.4f}, Threshold: {self.drift_thresholds[drift_type]:.4f}",
                        timestamp=datetime.now(),
                        details=result.details,
                    )

                    self.alerts.append(alert)
                    logger.warning(f"Drift alert generated: {alert.message}")

        except Exception as e:
            logger.error(f"Error generating drift alerts: {e}")

    async def run_ab_test(
        self,
        test_id: str,
        model_a_id: str,
        model_b_id: str,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        metric_name: str = "accuracy",
        confidence_level: float = 0.95,
    ) -> ABTestResult:
        """
        Run A/B test between two models.

        Args:
            test_id: Unique identifier for the test
            model_a_id: ID of first model (control)
            model_b_id: ID of second model (treatment)
            test_data: Test data for comparison
            test_labels: Test labels
            metric_name: Metric to compare (accuracy, precision, recall, f1_score)
            confidence_level: Statistical confidence level

        Returns:
            ABTestResult with comparison results
        """
        try:
            # Load models
            model_a = await self.ml_manager.load_model(model_a_id, stage="Production")
            if model_a is None:
                model_a = await self.ml_manager.load_model(model_a_id, stage="Staging")

            model_b = await self.ml_manager.load_model(model_b_id, stage="Production")
            if model_b is None:
                model_b = await self.ml_manager.load_model(model_b_id, stage="Staging")

            if model_a is None or model_b is None:
                raise ValueError("Could not load one or both models for A/B testing")

            # Get predictions
            predictions_a = model_a.predict(test_data)
            predictions_b = model_b.predict(test_data)

            # Calculate metrics
            if metric_name == "accuracy":
                score_a = accuracy_score(test_labels, predictions_a)
                score_b = accuracy_score(test_labels, predictions_b)
            elif metric_name == "precision":
                score_a = precision_score(
                    test_labels, predictions_a, average="weighted"
                )
                score_b = precision_score(
                    test_labels, predictions_b, average="weighted"
                )
            elif metric_name == "recall":
                score_a = recall_score(test_labels, predictions_a, average="weighted")
                score_b = recall_score(test_labels, predictions_b, average="weighted")
            elif metric_name == "f1_score":
                score_a = f1_score(test_labels, predictions_a, average="weighted")
                score_b = f1_score(test_labels, predictions_b, average="weighted")
            else:
                raise ValueError(f"Unsupported metric: {metric_name}")

            # Statistical significance test (paired t-test approximation)
            # For simplicity, we'll use a basic comparison
            # In practice, you'd want more sophisticated statistical testing

            score_diff = score_b - score_a

            # Simple significance test based on score difference
            # This is a simplified approach - in practice, use proper statistical tests
            if abs(score_diff) > 0.01:  # 1% difference threshold
                is_significant = True
                p_value = 0.01  # Simplified
            else:
                is_significant = False
                p_value = 0.5

            # Confidence interval (simplified)
            margin_of_error = 0.02  # Simplified 2% margin
            confidence_interval = (
                score_diff - margin_of_error,
                score_diff + margin_of_error,
            )

            # Determine winner
            if is_significant:
                winner = model_b_id if score_b > score_a else model_a_id
            else:
                winner = None

            result = ABTestResult(
                test_id=test_id,
                model_a_id=model_a_id,
                model_b_id=model_b_id,
                metric_name=metric_name,
                model_a_score=score_a,
                model_b_score=score_b,
                p_value=p_value,
                confidence_interval=confidence_interval,
                is_significant=is_significant,
                winner=winner,
                test_duration_days=1,  # Simplified
                sample_size=len(test_data),
            )

            self.ab_tests[test_id] = result

            logger.info(
                f"A/B test completed: {test_id}, Winner: {winner}, "
                f"Scores: {score_a:.4f} vs {score_b:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in A/B test {test_id}: {e}")
            raise

    async def get_monitoring_dashboard_data(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive monitoring data for dashboard display."""
        try:
            dashboard_data = {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "alerts": [],
                "performance_history": [],
                "drift_history": [],
                "current_status": "healthy",
            }

            # Get recent alerts
            recent_alerts = [
                alert
                for alert in self.alerts
                if alert.model_id == model_id and not alert.resolved
            ]
            dashboard_data["alerts"] = [
                {
                    "alert_id": alert.alert_id,
                    "drift_type": alert.drift_type.value,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "drift_score": alert.drift_score,
                }
                for alert in recent_alerts[-10:]  # Last 10 alerts
            ]

            # Get performance history
            if model_id in self.performance_history:
                dashboard_data["performance_history"] = [
                    {
                        "timestamp": metrics.timestamp.isoformat(),
                        "accuracy": metrics.accuracy,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1_score": metrics.f1_score,
                    }
                    for metrics in self.performance_history[model_id][
                        -50:
                    ]  # Last 50 measurements
                ]

            # Get drift history
            if model_id in self.drift_history:
                dashboard_data["drift_history"] = [
                    {
                        "drift_type": result.details.get("drift_type", "unknown"),
                        "drift_score": result.drift_score,
                        "drift_detected": result.drift_detected,
                        "confidence": result.confidence,
                    }
                    for result in self.drift_history[model_id][
                        -20:
                    ]  # Last 20 drift checks
                ]

            # Determine current status
            if any(
                alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]
                for alert in recent_alerts
            ):
                dashboard_data["current_status"] = "critical"
            elif any(alert.severity == AlertSeverity.MEDIUM for alert in recent_alerts):
                dashboard_data["current_status"] = "warning"
            elif recent_alerts:
                dashboard_data["current_status"] = "attention"

            return dashboard_data

        except Exception as e:
            logger.error(f"Error getting dashboard data for {model_id}: {e}")
            return {"error": str(e)}

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a drift alert."""
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"Alert {alert_id} acknowledged")
                    return True

            logger.warning(f"Alert {alert_id} not found")
            return False

        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False

    async def resolve_alert(self, alert_id: str) -> bool:
        """Mark a drift alert as resolved."""
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.acknowledged = True
                    logger.info(f"Alert {alert_id} resolved")
                    return True

            logger.warning(f"Alert {alert_id} not found")
            return False

        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False

    def get_active_alerts(self, model_id: Optional[str] = None) -> List[DriftAlert]:
        """Get active (unresolved) alerts."""
        active_alerts = [alert for alert in self.alerts if not alert.resolved]

        if model_id:
            active_alerts = [
                alert for alert in active_alerts if alert.model_id == model_id
            ]

        return active_alerts

    def get_ab_test_results(
        self, test_id: Optional[str] = None
    ) -> Union[ABTestResult, List[ABTestResult]]:
        """Get A/B test results."""
        if test_id:
            return self.ab_tests.get(test_id)
        else:
            return list(self.ab_tests.values())
