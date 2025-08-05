"""
Tests for ML Model Manager

This module contains comprehensive tests for the ML model lifecycle management system.
"""

import asyncio
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from stock_analysis_system.analysis.ml_model_manager import (
    DriftDetectionResult,
    MLModelManager,
    ModelInfo,
    ModelMetrics,
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
        X, y, test_size=0.2, random_state=42
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": [f"feature_{i}" for i in range(10)],
    }


@pytest.fixture
def sample_model(sample_data):
    """Create a sample trained model."""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(sample_data["X_train"], sample_data["y_train"])
    return model


@pytest.fixture
def sample_metrics():
    """Create sample model metrics."""
    return ModelMetrics(
        accuracy=0.85,
        precision=0.83,
        recall=0.87,
        f1_score=0.85,
        custom_metrics={"auc_roc": 0.89, "log_loss": 0.35},
    )


@pytest.fixture
def temp_mlflow_dir():
    """Create temporary directory for MLflow tracking."""
    temp_dir = tempfile.mkdtemp()
    mlflow_uri = f"file://{temp_dir}/mlruns"
    yield mlflow_uri
    shutil.rmtree(temp_dir)


@pytest.fixture
async def ml_manager(temp_mlflow_dir):
    """Create ML Model Manager instance for testing."""
    manager = MLModelManager(mlflow_tracking_uri=temp_mlflow_dir)
    yield manager
    # Cleanup
    mlflow.end_run()


class TestMLModelManager:
    """Test cases for ML Model Manager."""

    @pytest.mark.asyncio
    async def test_initialization(self, temp_mlflow_dir):
        """Test ML Model Manager initialization."""
        manager = MLModelManager(mlflow_tracking_uri=temp_mlflow_dir)

        assert manager.mlflow_uri == temp_mlflow_dir
        assert manager.drift_threshold > 0
        assert isinstance(manager.models, dict)
        assert isinstance(manager.retraining_schedule, dict)

    @pytest.mark.asyncio
    async def test_register_model(self, ml_manager, sample_model, sample_metrics):
        """Test model registration."""
        model_name = "test_model"
        tags = {"version": "1.0", "type": "classification"}
        description = "Test model for unit testing"

        model_id = await ml_manager.register_model(
            model_name=model_name,
            model_object=sample_model,
            metrics=sample_metrics,
            tags=tags,
            description=description,
        )

        assert model_id is not None
        assert model_id.startswith(f"{model_name}_v")
        assert model_id in ml_manager.models

        model_info = ml_manager.models[model_id]
        assert model_info.model_name == model_name
        assert model_info.status == "staging"
        assert model_info.metrics.accuracy == sample_metrics.accuracy
        assert model_info.tags == tags
        assert model_info.description == description

    @pytest.mark.asyncio
    async def test_promote_model_to_production(
        self, ml_manager, sample_model, sample_metrics
    ):
        """Test model promotion to production."""
        # First register a model
        model_id = await ml_manager.register_model(
            model_name="test_promotion_model",
            model_object=sample_model,
            metrics=sample_metrics,
        )

        # Promote to production
        success = await ml_manager.promote_model_to_production(model_id)

        assert success is True
        assert ml_manager.models[model_id].status == "production"

    @pytest.mark.asyncio
    async def test_promote_nonexistent_model(self, ml_manager):
        """Test promoting a non-existent model."""
        success = await ml_manager.promote_model_to_production("nonexistent_model")
        assert success is False

    @pytest.mark.asyncio
    async def test_detect_model_drift_no_drift(self, ml_manager, sample_data):
        """Test drift detection when no drift is present."""
        model_id = "test_model_drift"

        # Use same distribution for reference and new data
        reference_data = sample_data["X_train"][:400]
        new_data = sample_data["X_train"][400:600]

        # Add model to manager
        ml_manager.models[model_id] = ModelInfo(
            model_id=model_id,
            model_name="drift_test",
            version="1",
            status="production",
            created_at=datetime.now(),
            last_updated=datetime.now(),
            metrics=ModelMetrics(0.8, 0.8, 0.8, 0.8, {}),
            drift_score=0.0,
            tags={},
        )

        result = await ml_manager.detect_model_drift(
            model_id=model_id,
            new_data=new_data,
            reference_data=reference_data,
            feature_names=sample_data["feature_names"],
        )

        assert isinstance(result, DriftDetectionResult)
        assert result.drift_type == "data"
        assert result.drift_score >= 0.0
        assert "overall_drift_score" in result.details
        assert "feature_drift_details" in result.details

    @pytest.mark.asyncio
    async def test_detect_model_drift_with_drift(self, ml_manager, sample_data):
        """Test drift detection when drift is present."""
        model_id = "test_model_drift_present"

        # Create reference data
        reference_data = sample_data["X_train"][:400]

        # Create drifted data by adding noise and shifting distribution
        new_data = sample_data["X_train"][400:600].copy()
        new_data = new_data + np.random.normal(
            0, 2, new_data.shape
        )  # Add significant noise
        new_data = new_data + 5  # Shift distribution

        # Add model to manager
        ml_manager.models[model_id] = ModelInfo(
            model_id=model_id,
            model_name="drift_test_present",
            version="1",
            status="production",
            created_at=datetime.now(),
            last_updated=datetime.now(),
            metrics=ModelMetrics(0.8, 0.8, 0.8, 0.8, {}),
            drift_score=0.0,
            tags={},
        )

        result = await ml_manager.detect_model_drift(
            model_id=model_id,
            new_data=new_data,
            reference_data=reference_data,
            feature_names=sample_data["feature_names"],
        )

        assert isinstance(result, DriftDetectionResult)
        assert result.drift_score > 0.0
        # With significant drift, it should be detected
        assert result.drift_detected is True or result.drift_score > 0.1

    @pytest.mark.asyncio
    async def test_schedule_retraining_periodic(
        self, ml_manager, sample_model, sample_metrics
    ):
        """Test periodic retraining scheduling."""
        # Register a model first
        model_id = await ml_manager.register_model(
            model_name="test_schedule_model",
            model_object=sample_model,
            metrics=sample_metrics,
        )

        # Schedule periodic retraining
        await ml_manager.schedule_retraining(
            model_id=model_id,
            schedule_type="periodic",
            schedule_config={"interval_days": 7},
        )

        assert model_id in ml_manager.retraining_schedule
        schedule_info = ml_manager.retraining_schedule[model_id]
        assert schedule_info["schedule_type"] == "periodic"
        assert schedule_info["schedule_config"]["interval_days"] == 7

    @pytest.mark.asyncio
    async def test_schedule_retraining_drift_based(
        self, ml_manager, sample_model, sample_metrics
    ):
        """Test drift-based retraining scheduling."""
        # Register a model first
        model_id = await ml_manager.register_model(
            model_name="test_drift_schedule_model",
            model_object=sample_model,
            metrics=sample_metrics,
        )

        # Schedule drift-based retraining
        await ml_manager.schedule_retraining(
            model_id=model_id, schedule_type="drift_based"
        )

        assert model_id in ml_manager.retraining_schedule
        schedule_info = ml_manager.retraining_schedule[model_id]
        assert schedule_info["schedule_type"] == "drift_based"

    @pytest.mark.asyncio
    async def test_check_retraining_due(self, ml_manager, sample_model, sample_metrics):
        """Test checking for models due for retraining."""
        # Register a model
        model_id = await ml_manager.register_model(
            model_name="test_due_model",
            model_object=sample_model,
            metrics=sample_metrics,
        )

        # Schedule retraining with past due date
        past_date = datetime.now() - timedelta(days=1)
        ml_manager.retraining_schedule[model_id] = {
            "schedule_type": "periodic",
            "schedule_config": {"interval_days": 7},
            "last_retrain": past_date,
            "next_retrain": past_date,
            "retrain_count": 0,
        }

        due_models = await ml_manager.check_retraining_due()
        assert model_id in due_models

    @pytest.mark.asyncio
    async def test_load_model(self, ml_manager, sample_model, sample_metrics):
        """Test loading a model from MLflow."""
        # Register and promote a model
        model_id = await ml_manager.register_model(
            model_name="test_load_model",
            model_object=sample_model,
            metrics=sample_metrics,
        )

        await ml_manager.promote_model_to_production(model_id)

        # Load the model
        loaded_model = await ml_manager.load_model(model_id, stage="Production")

        # The loaded model should be able to make predictions
        assert loaded_model is not None
        assert hasattr(loaded_model, "predict")

    @pytest.mark.asyncio
    async def test_compare_models(
        self, ml_manager, sample_model, sample_metrics, sample_data
    ):
        """Test comparing multiple models."""
        # Register two models
        model_id_1 = await ml_manager.register_model(
            model_name="test_compare_model_1",
            model_object=sample_model,
            metrics=sample_metrics,
        )

        model_id_2 = await ml_manager.register_model(
            model_name="test_compare_model_2",
            model_object=sample_model,
            metrics=sample_metrics,
        )

        # Promote both to production
        await ml_manager.promote_model_to_production(model_id_1)
        await ml_manager.promote_model_to_production(model_id_2)

        # Compare models
        results = await ml_manager.compare_models(
            model_ids=[model_id_1, model_id_2],
            test_data=sample_data["X_test"],
            test_labels=sample_data["y_test"],
            metrics=["accuracy", "precision"],
        )

        assert len(results) == 2
        assert model_id_1 in results
        assert model_id_2 in results
        assert "accuracy" in results[model_id_1]
        assert "precision" in results[model_id_1]

    @pytest.mark.asyncio
    async def test_get_model_info(self, ml_manager, sample_model, sample_metrics):
        """Test getting model information."""
        model_id = await ml_manager.register_model(
            model_name="test_info_model",
            model_object=sample_model,
            metrics=sample_metrics,
        )

        model_info = await ml_manager.get_model_info(model_id)

        assert model_info is not None
        assert model_info.model_id == model_id
        assert model_info.model_name == "test_info_model"
        assert model_info.status == "staging"

    @pytest.mark.asyncio
    async def test_list_models(self, ml_manager, sample_model, sample_metrics):
        """Test listing models with filters."""
        # Register multiple models
        model_id_1 = await ml_manager.register_model(
            model_name="test_list_model_1",
            model_object=sample_model,
            metrics=sample_metrics,
        )

        model_id_2 = await ml_manager.register_model(
            model_name="test_list_model_2",
            model_object=sample_model,
            metrics=sample_metrics,
        )

        # Promote one to production
        await ml_manager.promote_model_to_production(model_id_1)

        # List all models
        all_models = await ml_manager.list_models()
        assert len(all_models) >= 2

        # List only production models
        prod_models = await ml_manager.list_models(status_filter="production")
        assert len(prod_models) >= 1
        assert any(m.model_id == model_id_1 for m in prod_models)

        # List models by name filter
        filtered_models = await ml_manager.list_models(name_filter="test_list")
        assert len(filtered_models) >= 2

    @pytest.mark.asyncio
    async def test_archive_model(self, ml_manager, sample_model, sample_metrics):
        """Test archiving a model."""
        model_id = await ml_manager.register_model(
            model_name="test_archive_model",
            model_object=sample_model,
            metrics=sample_metrics,
        )

        success = await ml_manager.archive_model(model_id)

        assert success is True
        assert ml_manager.models[model_id].status == "archived"

    @pytest.mark.asyncio
    async def test_archive_nonexistent_model(self, ml_manager):
        """Test archiving a non-existent model."""
        success = await ml_manager.archive_model("nonexistent_model")
        assert success is False


class TestModelMetrics:
    """Test cases for ModelMetrics class."""

    def test_model_metrics_creation(self):
        """Test creating ModelMetrics instance."""
        metrics = ModelMetrics(
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            custom_metrics={"auc_roc": 0.89},
        )

        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.83
        assert metrics.recall == 0.87
        assert metrics.f1_score == 0.85
        assert metrics.custom_metrics["auc_roc"] == 0.89

    def test_model_metrics_to_dict(self):
        """Test converting ModelMetrics to dictionary."""
        metrics = ModelMetrics(
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            custom_metrics={"auc_roc": 0.89, "log_loss": 0.35},
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict["accuracy"] == 0.85
        assert metrics_dict["precision"] == 0.83
        assert metrics_dict["recall"] == 0.87
        assert metrics_dict["f1_score"] == 0.85
        assert metrics_dict["auc_roc"] == 0.89
        assert metrics_dict["log_loss"] == 0.35


class TestModelInfo:
    """Test cases for ModelInfo class."""

    def test_model_info_creation(self):
        """Test creating ModelInfo instance."""
        metrics = ModelMetrics(0.85, 0.83, 0.87, 0.85, {})

        model_info = ModelInfo(
            model_id="test_model_v1",
            model_name="test_model",
            version="1",
            status="staging",
            created_at=datetime.now(),
            last_updated=datetime.now(),
            metrics=metrics,
            drift_score=0.05,
            tags={"type": "classification"},
            description="Test model",
        )

        assert model_info.model_id == "test_model_v1"
        assert model_info.model_name == "test_model"
        assert model_info.version == "1"
        assert model_info.status == "staging"
        assert model_info.drift_score == 0.05
        assert model_info.tags["type"] == "classification"
        assert model_info.description == "Test model"

    def test_model_info_to_dict(self):
        """Test converting ModelInfo to dictionary."""
        metrics = ModelMetrics(0.85, 0.83, 0.87, 0.85, {})
        created_at = datetime.now()
        last_updated = datetime.now()

        model_info = ModelInfo(
            model_id="test_model_v1",
            model_name="test_model",
            version="1",
            status="staging",
            created_at=created_at,
            last_updated=last_updated,
            metrics=metrics,
            drift_score=0.05,
            tags={"type": "classification"},
        )

        info_dict = model_info.to_dict()

        assert info_dict["model_id"] == "test_model_v1"
        assert info_dict["model_name"] == "test_model"
        assert info_dict["version"] == "1"
        assert info_dict["status"] == "staging"
        assert info_dict["created_at"] == created_at.isoformat()
        assert info_dict["last_updated"] == last_updated.isoformat()
        assert info_dict["drift_score"] == 0.05
        assert info_dict["tags"]["type"] == "classification"


class TestDriftDetectionResult:
    """Test cases for DriftDetectionResult class."""

    def test_drift_detection_result_creation(self):
        """Test creating DriftDetectionResult instance."""
        result = DriftDetectionResult(
            drift_detected=True,
            drift_score=0.15,
            drift_type="data",
            confidence=0.85,
            details={"feature_count": 10},
        )

        assert result.drift_detected is True
        assert result.drift_score == 0.15
        assert result.drift_type == "data"
        assert result.confidence == 0.85
        assert result.details["feature_count"] == 10


if __name__ == "__main__":
    pytest.main([__file__])
