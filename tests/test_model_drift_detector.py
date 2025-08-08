"""
Tests for Model Drift Detection and Monitoring System
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

from stock_analysis_system.ml.model_drift_detector import (
    ModelDriftDetector,
    DriftType,
    DriftSeverity,
    DriftAlert,
    ModelPerformanceMetrics,
    DriftDetectionResult
)


class TestModelDriftDetector:
    """Test cases for ModelDriftDetector."""
    
    @pytest.fixture
    def mock_database_url(self):
        """Mock database URL for testing."""
        return "sqlite:///:memory:"
    
    @pytest.fixture
    def mock_mlflow_uri(self):
        """Mock MLflow URI for testing."""
        return "sqlite:///mlflow.db"
    
    @pytest.fixture
    def drift_detector(self, mock_database_url, mock_mlflow_uri):
        """Create ModelDriftDetector instance for testing."""
        with patch('stock_analysis_system.ml.model_drift_detector.create_engine'):
            with patch('stock_analysis_system.ml.model_drift_detector.mlflow'):
                detector = ModelDriftDetector(mock_database_url, mock_mlflow_uri)
                return detector
    
    @pytest.fixture
    def sample_baseline_data(self):
        """Sample baseline data for testing."""
        np.random.seed(42)
        return np.random.normal(0, 1, (1000, 5))
    
    @pytest.fixture
    def sample_baseline_performance(self):
        """Sample baseline performance metrics."""
        return ModelPerformanceMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            custom_metrics={'auc': 0.90}
        )
    
    @pytest.fixture
    def sample_new_data_no_drift(self):
        """Sample new data with no drift."""
        np.random.seed(43)
        return np.random.normal(0, 1, (500, 5))
    
    @pytest.fixture
    def sample_new_data_with_drift(self):
        """Sample new data with drift."""
        np.random.seed(44)
        return np.random.normal(2, 1.5, (500, 5))  # Different mean and std
    
    @pytest.mark.asyncio
    async def test_register_model_for_monitoring(self, drift_detector, sample_baseline_data, 
                                               sample_baseline_performance):
        """Test model registration for monitoring."""
        # Mock database operations
        drift_detector._store_monitoring_registration = AsyncMock(return_value=None)
        
        # Mock MLflow operations
        with patch('mlflow.start_run'), patch('mlflow.log_params'), patch('mlflow.log_metrics'):
            result = await drift_detector.register_model_for_monitoring(
                model_id="test_model_1",
                model_name="Test Model",
                baseline_data=sample_baseline_data,
                baseline_performance=sample_baseline_performance
            )
        
        assert result is True
        assert "test_model_1" in drift_detector.monitored_models
        assert drift_detector.monitored_models["test_model_1"]["model_name"] == "Test Model"
        assert drift_detector.monitored_models["test_model_1"]["status"] == "active"
    
    @pytest.mark.asyncio
    async def test_detect_data_drift_no_drift(self, drift_detector, sample_baseline_data, 
                                            sample_new_data_no_drift, sample_baseline_performance):
        """Test data drift detection with no drift."""
        # Register model
        drift_detector._store_monitoring_registration = AsyncMock(return_value=None)
        with patch('mlflow.start_run'), patch('mlflow.log_params'), patch('mlflow.log_metrics'):
            await drift_detector.register_model_for_monitoring(
                model_id="test_model_1",
                model_name="Test Model",
                baseline_data=sample_baseline_data,
                baseline_performance=sample_baseline_performance
            )
        
        # Mock database operations
        drift_detector._log_drift_detection_results = AsyncMock(return_value=None)
        
        # Detect drift
        result = await drift_detector.detect_drift(
            model_id="test_model_1",
            new_data=sample_new_data_no_drift
        )
        
        assert isinstance(result, DriftDetectionResult)
        assert result.model_id == "test_model_1"
        assert result.data_drift_score < 0.2  # Should be low drift
        assert len(result.alerts) == 0  # No alerts for low drift
    
    @pytest.mark.asyncio
    async def test_detect_data_drift_with_drift(self, drift_detector, sample_baseline_data, 
                                              sample_new_data_with_drift, sample_baseline_performance):
        """Test data drift detection with significant drift."""
        # Register model
        drift_detector._store_monitoring_registration = AsyncMock(return_value=None)
        with patch('mlflow.start_run'), patch('mlflow.log_params'), patch('mlflow.log_metrics'):
            await drift_detector.register_model_for_monitoring(
                model_id="test_model_1",
                model_name="Test Model",
                baseline_data=sample_baseline_data,
                baseline_performance=sample_baseline_performance
            )
        
        # Mock database operations
        drift_detector._log_drift_detection_results = AsyncMock(return_value=None)
        
        # Detect drift
        result = await drift_detector.detect_drift(
            model_id="test_model_1",
            new_data=sample_new_data_with_drift
        )
        
        assert isinstance(result, DriftDetectionResult)
        assert result.model_id == "test_model_1"
        assert result.data_drift_score > 0.3  # Should be high drift
        assert len(result.alerts) > 0  # Should have alerts
        
        # Check alert details
        data_drift_alerts = [alert for alert in result.alerts if alert.drift_type == DriftType.DATA_DRIFT]
        assert len(data_drift_alerts) > 0
        assert data_drift_alerts[0].severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_detect_performance_drift(self, drift_detector, sample_baseline_data, 
                                          sample_new_data_no_drift, sample_baseline_performance):
        """Test performance drift detection."""
        # Register model
        drift_detector._store_monitoring_registration = AsyncMock(return_value=None)
        with patch('mlflow.start_run'), patch('mlflow.log_params'), patch('mlflow.log_metrics'):
            await drift_detector.register_model_for_monitoring(
                model_id="test_model_1",
                model_name="Test Model",
                baseline_data=sample_baseline_data,
                baseline_performance=sample_baseline_performance
            )
        
        # Mock database operations
        drift_detector._log_drift_detection_results = AsyncMock(return_value=None)
        
        # Create degraded predictions and labels
        new_predictions = np.random.choice([0, 1], size=500, p=[0.7, 0.3])  # Biased predictions
        true_labels = np.random.choice([0, 1], size=500, p=[0.5, 0.5])  # Balanced labels
        
        # Detect drift
        result = await drift_detector.detect_drift(
            model_id="test_model_1",
            new_data=sample_new_data_no_drift,
            new_predictions=new_predictions,
            true_labels=true_labels
        )
        
        assert isinstance(result, DriftDetectionResult)
        assert result.performance_drift_score > 0  # Should detect performance degradation
    
    @pytest.mark.asyncio
    async def test_drift_severity_evaluation(self, drift_detector):
        """Test drift severity evaluation."""
        # Test different drift scores
        test_cases = [
            (0.05, DriftSeverity.LOW),
            (0.15, DriftSeverity.MEDIUM),
            (0.25, DriftSeverity.HIGH),
            (0.6, DriftSeverity.CRITICAL)
        ]
        
        # Mock monitored model
        drift_detector.monitored_models["test_model"] = {
            "model_name": "Test Model"
        }
        
        for drift_score, expected_severity in test_cases:
            alerts = drift_detector._evaluate_drift_severity(
                "test_model", DriftType.DATA_DRIFT, drift_score
            )
            
            if expected_severity:
                assert len(alerts) == 1
                assert alerts[0].severity == expected_severity
                assert alerts[0].drift_score == drift_score
            else:
                assert len(alerts) == 0
    
    @pytest.mark.asyncio
    async def test_retraining_recommendation(self, drift_detector):
        """Test retraining recommendation logic."""
        # Create alerts with different severities
        critical_alert = DriftAlert(
            model_id="test_model",
            drift_type=DriftType.PERFORMANCE_DRIFT,
            severity=DriftSeverity.CRITICAL,
            drift_score=0.8,
            threshold=0.5,
            detected_at=datetime.now(),
            description="Critical drift",
            recommendations=[],
            metadata={}
        )
        
        high_alert = DriftAlert(
            model_id="test_model",
            drift_type=DriftType.DATA_DRIFT,
            severity=DriftSeverity.HIGH,
            drift_score=0.4,
            threshold=0.3,
            detected_at=datetime.now(),
            description="High drift",
            recommendations=[],
            metadata={}
        )
        
        medium_alert = DriftAlert(
            model_id="test_model",
            drift_type=DriftType.CONCEPT_DRIFT,
            severity=DriftSeverity.MEDIUM,
            drift_score=0.2,
            threshold=0.1,
            detected_at=datetime.now(),
            description="Medium drift",
            recommendations=[],
            metadata={}
        )
        
        # Test different configurations
        config_high_threshold = {'auto_retrain_threshold': DriftSeverity.HIGH}
        config_medium_threshold = {'auto_retrain_threshold': DriftSeverity.MEDIUM}
        
        # Critical alert should always trigger retraining
        assert drift_detector._should_trigger_retraining([critical_alert], config_high_threshold) is True
        assert drift_detector._should_trigger_retraining([critical_alert], config_medium_threshold) is True
        
        # High alert should trigger with high or medium threshold
        assert drift_detector._should_trigger_retraining([high_alert], config_high_threshold) is True
        assert drift_detector._should_trigger_retraining([high_alert], config_medium_threshold) is True
        
        # Medium alert should only trigger with medium threshold
        assert drift_detector._should_trigger_retraining([medium_alert], config_high_threshold) is False
        assert drift_detector._should_trigger_retraining([medium_alert], config_medium_threshold) is True
    
    @pytest.mark.asyncio
    async def test_schedule_automated_retraining(self, drift_detector):
        """Test automated retraining scheduling."""
        # Mock database operations
        drift_detector.engine.connect = Mock()
        mock_conn = Mock()
        mock_conn.execute = Mock()
        mock_conn.commit = Mock()
        drift_detector.engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        drift_detector.engine.connect.return_value.__exit__ = Mock(return_value=None)
        
        schedule_config = {
            'frequency': 'weekly',
            'drift_threshold': DriftSeverity.HIGH,
            'min_samples': 1000
        }
        
        result = await drift_detector.schedule_automated_retraining(
            "test_model", schedule_config
        )
        
        assert result is True
        assert "test_model" in drift_detector.retraining_schedule
        assert drift_detector.retraining_schedule["test_model"]["config"]["frequency"] == "weekly"
    
    @pytest.mark.asyncio
    async def test_check_models_due_for_retraining(self, drift_detector):
        """Test checking models due for retraining."""
        # Set up retraining schedule with past due date
        past_date = datetime.now() - timedelta(days=1)
        future_date = datetime.now() + timedelta(days=1)
        
        drift_detector.retraining_schedule = {
            "model_due": {
                "status": "scheduled",
                "next_retrain": past_date
            },
            "model_not_due": {
                "status": "scheduled",
                "next_retrain": future_date
            },
            "model_inactive": {
                "status": "inactive",
                "next_retrain": past_date
            }
        }
        
        due_models = await drift_detector.check_models_due_for_retraining()
        
        assert len(due_models) == 1
        assert "model_due" in due_models
        assert "model_not_due" not in due_models
        assert "model_inactive" not in due_models
    
    def test_drift_recommendations(self, drift_detector):
        """Test drift recommendation generation."""
        # Test data drift recommendations
        data_drift_recs = drift_detector._get_drift_recommendations(
            DriftType.DATA_DRIFT, DriftSeverity.HIGH
        )
        assert any("retraining" in rec.lower() for rec in data_drift_recs)
        assert any("data source" in rec.lower() for rec in data_drift_recs)
        
        # Test concept drift recommendations
        concept_drift_recs = drift_detector._get_drift_recommendations(
            DriftType.CONCEPT_DRIFT, DriftSeverity.CRITICAL
        )
        assert any("immediate" in rec.lower() for rec in concept_drift_recs)
        assert any("architecture" in rec.lower() for rec in concept_drift_recs)
        
        # Test performance drift recommendations
        perf_drift_recs = drift_detector._get_drift_recommendations(
            DriftType.PERFORMANCE_DRIFT, DriftSeverity.HIGH
        )
        assert any("urgent" in rec.lower() for rec in perf_drift_recs)
        assert any("root cause" in rec.lower() for rec in perf_drift_recs)
    
    @pytest.mark.asyncio
    async def test_cleanup_old_drift_data(self, drift_detector):
        """Test cleanup of old drift data."""
        # Mock database operations
        drift_detector.engine.connect = Mock()
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.rowcount = 150
        mock_conn.execute = Mock(return_value=mock_result)
        mock_conn.commit = Mock()
        drift_detector.engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        drift_detector.engine.connect.return_value.__exit__ = Mock(return_value=None)
        
        deleted_count = await drift_detector.cleanup_old_drift_data(retention_days=30)
        
        assert deleted_count == 150
        assert mock_conn.execute.call_count == 2  # Two delete queries
        assert mock_conn.commit.called
    
    @pytest.mark.asyncio
    async def test_get_model_drift_history(self, drift_detector):
        """Test getting model drift history."""
        # Mock database operations
        drift_detector.engine.connect = Mock()
        mock_conn = Mock()
        mock_result = Mock()
        mock_row1 = Mock()
        mock_row1._mapping = {
            'model_id': 'test_model',
            'detection_timestamp': datetime.now(),
            'data_drift_score': 0.15,
            'overall_drift_score': 0.15
        }
        mock_row2 = Mock()
        mock_row2._mapping = {
            'model_id': 'test_model',
            'detection_timestamp': datetime.now() - timedelta(days=1),
            'data_drift_score': 0.12,
            'overall_drift_score': 0.12
        }
        mock_result.__iter__ = Mock(return_value=iter([mock_row1, mock_row2]))
        mock_conn.execute = Mock(return_value=mock_result)
        drift_detector.engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        drift_detector.engine.connect.return_value.__exit__ = Mock(return_value=None)
        
        history = await drift_detector.get_model_drift_history("test_model", days=30)
        
        assert len(history) == 2
        assert history[0]['model_id'] == 'test_model'
        assert history[0]['data_drift_score'] == 0.15
    
    def test_calculate_next_retrain_date(self, drift_detector):
        """Test calculation of next retraining date."""
        current_time = datetime.now()
        
        # Test daily frequency
        daily_date = drift_detector._calculate_next_retrain_date('daily')
        assert (daily_date - current_time).days == 1
        
        # Test weekly frequency
        weekly_date = drift_detector._calculate_next_retrain_date('weekly')
        assert (weekly_date - current_time).days == 7
        
        # Test monthly frequency
        monthly_date = drift_detector._calculate_next_retrain_date('monthly')
        assert (monthly_date - current_time).days == 30
        
        # Test default frequency
        default_date = drift_detector._calculate_next_retrain_date('unknown')
        assert (default_date - current_time).days == 7


@pytest.mark.asyncio
async def test_integration_drift_detection_workflow():
    """Integration test for complete drift detection workflow."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        database_url = f"sqlite:///{db_path}"
        mlflow_uri = "sqlite:///test_mlflow.db"
        
        with patch('stock_analysis_system.ml.model_drift_detector.create_engine'):
            with patch('stock_analysis_system.ml.model_drift_detector.mlflow'):
                detector = ModelDriftDetector(database_url, mlflow_uri)
                
                # Mock database operations
                detector._store_monitoring_registration = AsyncMock(return_value=None)
                detector._log_drift_detection_results = AsyncMock(return_value=None)
                
                # Create test data
                baseline_data = np.random.normal(0, 1, (1000, 3))
                baseline_performance = ModelPerformanceMetrics(
                    accuracy=0.85, precision=0.82, recall=0.88, f1_score=0.85
                )
                
                # Register model
                with patch('mlflow.start_run'), patch('mlflow.log_params'), patch('mlflow.log_metrics'):
                    success = await detector.register_model_for_monitoring(
                        model_id="integration_test_model",
                        model_name="Integration Test Model",
                        baseline_data=baseline_data,
                        baseline_performance=baseline_performance
                    )
                
                assert success is True
                
                # Test with no drift data
                no_drift_data = np.random.normal(0, 1, (500, 3))
                result_no_drift = await detector.detect_drift(
                    model_id="integration_test_model",
                    new_data=no_drift_data
                )
                
                assert result_no_drift.data_drift_score < 0.3
                assert len(result_no_drift.alerts) == 0
                
                # Test with drift data
                drift_data = np.random.normal(3, 2, (500, 3))
                result_with_drift = await detector.detect_drift(
                    model_id="integration_test_model",
                    new_data=drift_data
                )
                
                assert result_with_drift.data_drift_score > 0.3
                assert len(result_with_drift.alerts) > 0
                
                # Test retraining scheduling
                schedule_success = await detector.schedule_automated_retraining(
                    "integration_test_model",
                    {'frequency': 'weekly', 'drift_threshold': DriftSeverity.HIGH}
                )
                
                assert schedule_success is True
                
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
        if os.path.exists("test_mlflow.db"):
            os.unlink("test_mlflow.db")


if __name__ == "__main__":
    pytest.main([__file__])