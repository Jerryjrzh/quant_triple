"""
Tests for Model Drift Monitor

This module contains comprehensive tests for the model drift detection and monitoring system.
"""

import asyncio
import shutil
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from stock_analysis_system.analysis.ml_model_manager import MLModelManager, ModelMetrics
from stock_analysis_system.analysis.model_drift_monitor import (
    ABTestResult,
    AlertSeverity,
    DriftAlert,
    DriftType,
    ModelDriftMonitor,
    PerformanceMetrics,
    PopulationStabilityIndex,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": [f"feature_{i}" for i in range(10)],
    }


@pytest.fixture
def sample_models(sample_data):
    """Create sample trained models."""
    # Model A (Random Forest)
    model_a = RandomForestClassifier(n_estimators=50, random_state=42)
    model_a.fit(sample_data["X_train"], sample_data["y_train"])

    # Model B (Random Forest with different parameters)
    model_b = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=43)
    model_b.fit(sample_data["X_train"], sample_data["y_train"])

    return {"model_a": model_a, "model_b": model_b}


@pytest.fixture
def temp_mlflow_dir():
    """Create temporary directory for MLflow tracking."""
    temp_dir = tempfile.mkdtemp()
    mlflow_uri = f"file://{temp_dir}/mlruns"
    yield mlflow_uri
    shutil.rmtree(temp_dir)


@pytest.fixture
async def drift_monitor(temp_mlflow_dir, sample_data, sample_models):
    """Create ModelDriftMonitor instance for testing."""
    # Initialize ML Manager
    ml_manager = MLModelManager(mlflow_tracking_uri=temp_mlflow_dir)

    # Register sample models
    metrics = ModelMetrics(0.85, 0.83, 0.87, 0.85, {})

    model_a_id = await ml_manager.register_model(
        model_name="test_model_a",
        model_object=sample_models["model_a"],
        metrics=metrics,
    )

    model_b_id = await ml_manager.register_model(
        model_name="test_model_b",
        model_object=sample_models["model_b"],
        metrics=metrics,
    )

    await ml_manager.promote_model_to_production(model_a_id)
    await ml_manager.promote_model_to_production(model_b_id)

    # Create drift monitor
    monitor = ModelDriftMonitor(ml_manager)

    yield monitor, model_a_id, model_b_id


class TestPopulationStabilityIndex:
    """Test cases for Population Stability Index calculator."""

    def test_psi_no_drift(self):
        """Test PSI calculation with no drift."""
        psi_calc = PopulationStabilityIndex()

        # Same distribution
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)

        psi = psi_calc.calculate_psi(expected, actual)

        # PSI should be low for similar distributions
        assert psi >= 0.0
        assert psi < 0.1  # Typically PSI < 0.1 indicates no significant drift

    def test_psi_with_drift(self):
        """Test PSI calculation with drift."""
        psi_calc = PopulationStabilityIndex()

        # Different distributions
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(2, 1.5, 1000)  # Shifted and scaled

        psi = psi_calc.calculate_psi(expected, actual)

        # PSI should be higher for different distributions
        assert psi > 0.1  # Significant drift threshold

    def test_psi_edge_cases(self):
        """Test PSI calculation with edge cases."""
        psi_calc = PopulationStabilityIndex()

        # Empty arrays
        psi = psi_calc.calculate_psi(np.array([]), np.array([]))
        assert psi == 0.0

        # Single value arrays
        psi = psi_calc.calculate_psi(np.array([1.0]), np.array([1.0]))
        assert psi >= 0.0


class TestModelDriftMonitor:
    """Test cases for Model Drift Monitor."""

    @pytest.mark.asyncio
    async def test_initialization(self, temp_mlflow_dir):
        """Test ModelDriftMonitor initialization."""
        ml_manager = MLModelManager(mlflow_tracking_uri=temp_mlflow_dir)
        monitor = ModelDriftMonitor(ml_manager)

        assert monitor.ml_manager == ml_manager
        assert len(monitor.alerts) == 0
        assert len(monitor.performance_history) == 0
        assert len(monitor.drift_history) == 0
        assert DriftType.DATA_DRIFT in monitor.drift_thresholds

    @pytest.mark.asyncio
    async def test_data_drift_detection_no_drift(self, drift_monitor, sample_data):
        """Test data drift detection with no drift."""
        monitor, model_a_id, model_b_id = drift_monitor

        # Use similar data (no drift)
        reference_data = sample_data["X_train"][:400]
        new_data = sample_data["X_train"][400:600]

        result = await monitor._detect_data_drift(
            model_a_id, new_data, reference_data, sample_data["feature_names"]
        )

        assert result.drift_type == DriftType.DATA_DRIFT.value
        assert result.drift_score >= 0.0
        assert "feature_drift_details" in result.details
        assert "n_features" in result.details

    @pytest.mark.asyncio
    async def test_data_drift_detection_with_drift(self, drift_monitor, sample_data):
        """Test data drift detection with significant drift."""
        monitor, model_a_id, model_b_id = drift_monitor

        # Create drifted data
        reference_data = sample_data["X_train"][:400]
        new_data = sample_data["X_train"][400:600].copy()
        new_data = new_data + np.random.normal(
            0, 3, new_data.shape
        )  # Add significant noise
        new_data = new_data + 5  # Shift distribution

        result = await monitor._detect_data_drift(
            model_a_id, new_data, reference_data, sample_data["feature_names"]
        )

        assert result.drift_type == DriftType.DATA_DRIFT.value
        assert result.drift_score > 0.0
        # With significant drift, score should be high
        assert result.drift_score > 0.1 or result.drift_detected

    @pytest.mark.asyncio
    async def test_prediction_drift_detection(self, drift_monitor, sample_data):
        """Test prediction drift detection."""
        monitor, model_a_id, model_b_id = drift_monitor

        reference_data = sample_data["X_train"][:400]
        new_data = sample_data["X_test"][:200]

        result = await monitor._detect_prediction_drift(
            model_a_id, new_data, reference_data
        )

        assert result is not None
        assert result.drift_type == DriftType.PREDICTION_DRIFT.value
        assert result.drift_score >= 0.0
        assert "reference_predictions_mean" in result.details
        assert "new_predictions_mean" in result.details

    @pytest.mark.asyncio
    async def test_concept_drift_detection(self, drift_monitor, sample_data):
        """Test concept drift detection."""
        monitor, model_a_id, model_b_id = drift_monitor

        reference_data = sample_data["X_train"][:400]
        new_data = sample_data["X_train"][400:600]
        reference_labels = sample_data["y_train"][:400]
        new_labels = sample_data["y_train"][400:600]

        result = await monitor._detect_concept_drift(
            model_a_id, new_data, reference_data, new_labels, reference_labels
        )

        assert result.drift_type == DriftType.CONCEPT_DRIFT.value
        assert result.drift_score >= 0.0
        assert "reference_label_mean" in result.details
        assert "new_label_mean" in result.details

    @pytest.mark.asyncio
    async def test_performance_drift_detection(self, drift_monitor, sample_data):
        """Test performance drift detection."""
        monitor, model_a_id, model_b_id = drift_monitor

        new_data = sample_data["X_test"][:200]
        new_labels = sample_data["y_test"][:200]

        result = await monitor._detect_performance_drift(
            model_a_id, new_data, new_labels
        )

        assert result is not None
        assert result.drift_type == DriftType.PERFORMANCE_DRIFT.value
        assert result.drift_score >= 0.0
        assert "current_accuracy" in result.details
        assert "baseline_accuracy" in result.details

        # Check that performance history was updated
        assert model_a_id in monitor.performance_history
        assert len(monitor.performance_history[model_a_id]) > 0

    @pytest.mark.asyncio
    async def test_comprehensive_drift_detection(self, drift_monitor, sample_data):
        """Test comprehensive drift detection."""
        monitor, model_a_id, model_b_id = drift_monitor

        reference_data = sample_data["X_train"][:400]
        new_data = sample_data["X_test"][:200]
        reference_labels = sample_data["y_train"][:400]
        new_labels = sample_data["y_test"][:200]

        results = await monitor.detect_comprehensive_drift(
            model_a_id,
            new_data,
            reference_data,
            new_labels,
            reference_labels,
            sample_data["feature_names"],
        )

        assert isinstance(results, dict)
        assert DriftType.DATA_DRIFT in results
        assert DriftType.PREDICTION_DRIFT in results
        assert DriftType.CONCEPT_DRIFT in results
        assert DriftType.PERFORMANCE_DRIFT in results

        # Check that drift history was updated
        assert model_a_id in monitor.drift_history
        assert len(monitor.drift_history[model_a_id]) > 0

    @pytest.mark.asyncio
    async def test_alert_generation(self, drift_monitor, sample_data):
        """Test drift alert generation."""
        monitor, model_a_id, model_b_id = drift_monitor

        # Create high drift scenario
        reference_data = sample_data["X_train"][:400]
        new_data = sample_data["X_train"][400:600].copy()
        new_data = new_data + 10  # Extreme shift to trigger alerts

        # Lower thresholds to ensure alerts are generated
        monitor.drift_thresholds[DriftType.DATA_DRIFT] = 0.01

        results = await monitor.detect_comprehensive_drift(
            model_a_id,
            new_data,
            reference_data,
            feature_names=sample_data["feature_names"],
        )

        # Check that alerts were generated
        active_alerts = monitor.get_active_alerts(model_a_id)
        assert len(active_alerts) > 0

        # Check alert properties
        alert = active_alerts[0]
        assert alert.model_id == model_a_id
        assert alert.drift_type in [DriftType.DATA_DRIFT, DriftType.PREDICTION_DRIFT]
        assert alert.severity in [
            AlertSeverity.LOW,
            AlertSeverity.MEDIUM,
            AlertSeverity.HIGH,
            AlertSeverity.CRITICAL,
        ]
        assert not alert.acknowledged
        assert not alert.resolved

    @pytest.mark.asyncio
    async def test_ab_testing(self, drift_monitor, sample_data):
        """Test A/B testing functionality."""
        monitor, model_a_id, model_b_id = drift_monitor

        test_data = sample_data["X_test"]
        test_labels = sample_data["y_test"]

        result = await monitor.run_ab_test(
            test_id="test_ab_001",
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            test_data=test_data,
            test_labels=test_labels,
            metric_name="accuracy",
        )

        assert isinstance(result, ABTestResult)
        assert result.test_id == "test_ab_001"
        assert result.model_a_id == model_a_id
        assert result.model_b_id == model_b_id
        assert result.metric_name == "accuracy"
        assert 0.0 <= result.model_a_score <= 1.0
        assert 0.0 <= result.model_b_score <= 1.0
        assert result.sample_size == len(test_data)

        # Check that test was stored
        stored_result = monitor.get_ab_test_results("test_ab_001")
        assert stored_result == result

    @pytest.mark.asyncio
    async def test_alert_management(self, drift_monitor):
        """Test alert acknowledgment and resolution."""
        monitor, model_a_id, model_b_id = drift_monitor

        # Create a test alert
        alert = DriftAlert(
            alert_id="test_alert_001",
            model_id=model_a_id,
            drift_type=DriftType.DATA_DRIFT,
            severity=AlertSeverity.MEDIUM,
            drift_score=0.15,
            threshold=0.1,
            message="Test alert",
            timestamp=datetime.now(),
        )

        monitor.alerts.append(alert)

        # Test acknowledgment
        success = await monitor.acknowledge_alert("test_alert_001")
        assert success
        assert alert.acknowledged

        # Test resolution
        success = await monitor.resolve_alert("test_alert_001")
        assert success
        assert alert.resolved
        assert alert.acknowledged

        # Test non-existent alert
        success = await monitor.acknowledge_alert("non_existent")
        assert not success

    @pytest.mark.asyncio
    async def test_monitoring_dashboard_data(self, drift_monitor, sample_data):
        """Test monitoring dashboard data generation."""
        monitor, model_a_id, model_b_id = drift_monitor

        # Add some test data
        alert = DriftAlert(
            alert_id="dashboard_test_alert",
            model_id=model_a_id,
            drift_type=DriftType.DATA_DRIFT,
            severity=AlertSeverity.HIGH,
            drift_score=0.25,
            threshold=0.1,
            message="Dashboard test alert",
            timestamp=datetime.now(),
        )
        monitor.alerts.append(alert)

        # Add performance metrics
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
        )
        monitor.performance_history[model_a_id] = [metrics]

        # Get dashboard data
        dashboard_data = await monitor.get_monitoring_dashboard_data(model_a_id)

        assert "model_id" in dashboard_data
        assert "timestamp" in dashboard_data
        assert "alerts" in dashboard_data
        assert "performance_history" in dashboard_data
        assert "current_status" in dashboard_data

        assert dashboard_data["model_id"] == model_a_id
        assert len(dashboard_data["alerts"]) > 0
        assert len(dashboard_data["performance_history"]) > 0
        assert dashboard_data["current_status"] in [
            "healthy",
            "attention",
            "warning",
            "critical",
        ]

    @pytest.mark.asyncio
    async def test_multiple_metrics_ab_test(self, drift_monitor, sample_data):
        """Test A/B testing with different metrics."""
        monitor, model_a_id, model_b_id = drift_monitor

        test_data = sample_data["X_test"]
        test_labels = sample_data["y_test"]

        metrics_to_test = ["accuracy", "precision", "recall", "f1_score"]

        for metric in metrics_to_test:
            result = await monitor.run_ab_test(
                test_id=f"test_{metric}",
                model_a_id=model_a_id,
                model_b_id=model_b_id,
                test_data=test_data,
                test_labels=test_labels,
                metric_name=metric,
            )

            assert result.metric_name == metric
            assert 0.0 <= result.model_a_score <= 1.0
            assert 0.0 <= result.model_b_score <= 1.0

    @pytest.mark.asyncio
    async def test_drift_threshold_configuration(self, drift_monitor):
        """Test drift threshold configuration."""
        monitor, model_a_id, model_b_id = drift_monitor

        # Test default thresholds
        assert DriftType.DATA_DRIFT in monitor.drift_thresholds
        assert DriftType.CONCEPT_DRIFT in monitor.drift_thresholds
        assert DriftType.PREDICTION_DRIFT in monitor.drift_thresholds
        assert DriftType.PERFORMANCE_DRIFT in monitor.drift_thresholds

        # Test threshold modification
        original_threshold = monitor.drift_thresholds[DriftType.DATA_DRIFT]
        monitor.drift_thresholds[DriftType.DATA_DRIFT] = 0.05

        assert monitor.drift_thresholds[DriftType.DATA_DRIFT] == 0.05
        assert monitor.drift_thresholds[DriftType.DATA_DRIFT] != original_threshold

    def test_get_active_alerts_filtering(self, drift_monitor):
        """Test active alerts filtering."""
        monitor, model_a_id, model_b_id = drift_monitor

        # Create test alerts
        alert1 = DriftAlert(
            alert_id="alert1",
            model_id=model_a_id,
            drift_type=DriftType.DATA_DRIFT,
            severity=AlertSeverity.HIGH,
            drift_score=0.2,
            threshold=0.1,
            message="Alert 1",
            timestamp=datetime.now(),
        )

        alert2 = DriftAlert(
            alert_id="alert2",
            model_id=model_b_id,
            drift_type=DriftType.CONCEPT_DRIFT,
            severity=AlertSeverity.MEDIUM,
            drift_score=0.15,
            threshold=0.1,
            message="Alert 2",
            timestamp=datetime.now(),
            resolved=True,
        )

        alert3 = DriftAlert(
            alert_id="alert3",
            model_id=model_a_id,
            drift_type=DriftType.PERFORMANCE_DRIFT,
            severity=AlertSeverity.LOW,
            drift_score=0.08,
            threshold=0.05,
            message="Alert 3",
            timestamp=datetime.now(),
        )

        monitor.alerts.extend([alert1, alert2, alert3])

        # Test getting all active alerts
        active_alerts = monitor.get_active_alerts()
        assert len(active_alerts) == 2  # alert2 is resolved

        # Test filtering by model_id
        model_a_alerts = monitor.get_active_alerts(model_a_id)
        assert len(model_a_alerts) == 2
        assert all(alert.model_id == model_a_id for alert in model_a_alerts)

        model_b_alerts = monitor.get_active_alerts(model_b_id)
        assert len(model_b_alerts) == 0  # alert2 is resolved


if __name__ == "__main__":
    pytest.main([__file__])
