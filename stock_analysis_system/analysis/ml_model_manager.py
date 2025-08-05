"""
ML Model Management System with MLflow Integration

This module provides comprehensive ML model lifecycle management including:
- Model registration and versioning with MLflow
- Model promotion workflows (staging -> production)
- Model drift detection and monitoring
- Automated retraining scheduling
- A/B testing framework for model comparison
"""

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    custom_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            **self.custom_metrics,
        }


@dataclass
class ModelInfo:
    """Model information and metadata."""

    model_id: str
    model_name: str
    version: str
    status: str  # 'training', 'staging', 'production', 'archived'
    created_at: datetime
    last_updated: datetime
    metrics: ModelMetrics
    drift_score: float
    tags: Dict[str, str]
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert model info to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "metrics": self.metrics.to_dict(),
            "drift_score": self.drift_score,
            "tags": self.tags,
            "description": self.description,
        }


@dataclass
class DriftDetectionResult:
    """Model drift detection result."""

    drift_detected: bool
    drift_score: float
    drift_type: str  # 'data', 'concept', 'prediction'
    confidence: float
    details: Dict[str, Any]


class MLModelManager:
    """
    Comprehensive ML model lifecycle management with MLflow integration.

    This class handles:
    - Model registration and versioning
    - Model promotion workflows
    - Drift detection and monitoring
    - Automated retraining scheduling
    - A/B testing framework
    """

    def __init__(self, mlflow_tracking_uri: Optional[str] = None):
        """
        Initialize ML Model Manager.

        Args:
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.settings = get_settings()
        self.mlflow_uri = mlflow_tracking_uri or self.settings.ml.mlflow_tracking_uri

        # Initialize MLflow
        mlflow.set_tracking_uri(self.mlflow_uri)

        # Set experiment
        try:
            mlflow.set_experiment(self.settings.ml.experiment_name)
        except Exception as e:
            logger.warning(f"Could not set experiment: {e}")
            mlflow.create_experiment(self.settings.ml.experiment_name)
            mlflow.set_experiment(self.settings.ml.experiment_name)

        # Internal state
        self.models: Dict[str, ModelInfo] = {}
        self.drift_threshold = self.settings.ml.drift_threshold
        self.retraining_schedule: Dict[str, Dict[str, Any]] = {}
        self.client = mlflow.tracking.MlflowClient()

        logger.info(f"MLModelManager initialized with tracking URI: {self.mlflow_uri}")

    async def register_model(
        self,
        model_name: str,
        model_object: Any,
        metrics: ModelMetrics,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a new model with MLflow.

        Args:
            model_name: Name of the model
            model_object: The trained model object
            metrics: Model performance metrics
            tags: Optional tags for the model
            description: Optional model description
            artifacts: Optional additional artifacts to log

        Returns:
            Model ID for the registered model
        """
        try:
            with mlflow.start_run() as run:
                # Log model
                mlflow.sklearn.log_model(
                    model_object, model_name, registered_model_name=model_name
                )

                # Log metrics
                for metric_name, metric_value in metrics.to_dict().items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log parameters (if model has them)
                if hasattr(model_object, "get_params"):
                    params = model_object.get_params()
                    for param_name, param_value in params.items():
                        if isinstance(param_value, (int, float, str, bool)):
                            mlflow.log_param(param_name, param_value)

                # Log tags
                if tags:
                    mlflow.set_tags(tags)

                # Log description
                if description:
                    mlflow.set_tag("description", description)

                # Log additional artifacts
                if artifacts:
                    for artifact_name, artifact_data in artifacts.items():
                        if isinstance(artifact_data, (dict, list)):
                            # Save as JSON
                            artifact_path = f"{artifact_name}.json"
                            with open(artifact_path, "w") as f:
                                json.dump(artifact_data, f, indent=2)
                            mlflow.log_artifact(artifact_path)
                            Path(artifact_path).unlink()  # Clean up
                        elif isinstance(artifact_data, pd.DataFrame):
                            # Save as CSV
                            artifact_path = f"{artifact_name}.csv"
                            artifact_data.to_csv(artifact_path, index=False)
                            mlflow.log_artifact(artifact_path)
                            Path(artifact_path).unlink()  # Clean up

                run_id = run.info.run_id

                # Get the latest version of the registered model
                latest_version = self.client.get_latest_versions(
                    model_name, stages=["None"]
                )[0]

                model_id = f"{model_name}_v{latest_version.version}"

                # Store model info
                self.models[model_id] = ModelInfo(
                    model_id=model_id,
                    model_name=model_name,
                    version=latest_version.version,
                    status="staging",
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    metrics=metrics,
                    drift_score=0.0,
                    tags=tags or {},
                    description=description,
                )

                logger.info(f"Model {model_id} registered successfully")
                return model_id

        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            raise

    async def promote_model_to_production(self, model_id: str) -> bool:
        """
        Promote a model from staging to production.

        Args:
            model_id: ID of the model to promote

        Returns:
            True if promotion was successful
        """
        try:
            if model_id not in self.models:
                logger.error(f"Model {model_id} not found")
                return False

            model_info = self.models[model_id]

            # First, archive current production model (if any)
            try:
                current_prod_versions = self.client.get_latest_versions(
                    model_info.model_name, stages=["Production"]
                )
                for version in current_prod_versions:
                    self.client.transition_model_version_stage(
                        name=model_info.model_name,
                        version=version.version,
                        stage="Archived",
                    )
                    logger.info(
                        f"Archived previous production model version {version.version}"
                    )
            except Exception as e:
                logger.warning(f"Could not archive previous production model: {e}")

            # Promote new model to production
            self.client.transition_model_version_stage(
                name=model_info.model_name,
                version=model_info.version,
                stage="Production",
            )

            # Update local status
            model_info.status = "production"
            model_info.last_updated = datetime.now()

            logger.info(f"Model {model_id} promoted to production")
            return True

        except Exception as e:
            logger.error(f"Failed to promote model {model_id}: {e}")
            return False

    async def detect_model_drift(
        self,
        model_id: str,
        new_data: np.ndarray,
        reference_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> DriftDetectionResult:
        """
        Detect model drift using statistical tests.

        Args:
            model_id: ID of the model to check for drift
            new_data: New data to compare
            reference_data: Reference data (training data)
            feature_names: Optional feature names for detailed analysis

        Returns:
            DriftDetectionResult with drift information
        """
        try:
            drift_details = {}

            # Data drift detection using KL divergence and KS test
            data_drift_scores = []
            feature_drift_details = {}

            if new_data.ndim == 1:
                new_data = new_data.reshape(-1, 1)
                reference_data = reference_data.reshape(-1, 1)

            n_features = new_data.shape[1]

            for i in range(n_features):
                feature_name = feature_names[i] if feature_names else f"feature_{i}"

                # KS test for distribution comparison
                ks_stat, ks_p_value = ks_2samp(reference_data[:, i], new_data[:, i])

                # KL divergence calculation
                try:
                    # Create histograms
                    ref_hist, bins = np.histogram(
                        reference_data[:, i], bins=50, density=True
                    )
                    new_hist, _ = np.histogram(new_data[:, i], bins=bins, density=True)

                    # Add small epsilon to avoid log(0)
                    epsilon = 1e-10
                    ref_hist = ref_hist + epsilon
                    new_hist = new_hist + epsilon

                    # Normalize
                    ref_hist = ref_hist / ref_hist.sum()
                    new_hist = new_hist / new_hist.sum()

                    # Calculate KL divergence
                    kl_div = entropy(new_hist, ref_hist)

                except Exception as e:
                    logger.warning(
                        f"Could not calculate KL divergence for {feature_name}: {e}"
                    )
                    kl_div = 0.0

                feature_drift_score = max(
                    ks_stat, min(kl_div, 1.0)
                )  # Cap KL div at 1.0
                data_drift_scores.append(feature_drift_score)

                feature_drift_details[feature_name] = {
                    "ks_statistic": ks_stat,
                    "ks_p_value": ks_p_value,
                    "kl_divergence": kl_div,
                    "drift_score": feature_drift_score,
                }

            # Overall drift score
            overall_drift_score = np.mean(data_drift_scores)

            # Determine if drift is detected
            drift_detected = overall_drift_score > self.drift_threshold

            # Calculate confidence based on consistency across features
            drift_consistency = 1.0 - np.std(data_drift_scores) / (
                np.mean(data_drift_scores) + 1e-10
            )
            confidence = min(max(drift_consistency, 0.0), 1.0)

            drift_details.update(
                {
                    "overall_drift_score": overall_drift_score,
                    "feature_drift_scores": data_drift_scores,
                    "feature_drift_details": feature_drift_details,
                    "n_features": n_features,
                    "n_samples_new": len(new_data),
                    "n_samples_reference": len(reference_data),
                }
            )

            # Update model info
            if model_id in self.models:
                self.models[model_id].drift_score = overall_drift_score
                self.models[model_id].last_updated = datetime.now()

            result = DriftDetectionResult(
                drift_detected=drift_detected,
                drift_score=overall_drift_score,
                drift_type="data",
                confidence=confidence,
                details=drift_details,
            )

            logger.info(
                f"Drift detection completed for {model_id}: "
                f"score={overall_drift_score:.4f}, detected={drift_detected}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to detect drift for model {model_id}: {e}")
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type="error",
                confidence=0.0,
                details={"error": str(e)},
            )

    async def schedule_retraining(
        self,
        model_id: str,
        schedule_type: str = "periodic",
        schedule_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Schedule automatic model retraining.

        Args:
            model_id: ID of the model to schedule retraining for
            schedule_type: Type of schedule ('periodic', 'drift_based', 'performance_based')
            schedule_config: Configuration for the schedule
        """
        try:
            if model_id not in self.models:
                logger.error(f"Model {model_id} not found")
                return

            config = schedule_config or {}

            if schedule_type == "periodic":
                interval_days = config.get(
                    "interval_days", self.settings.ml.retrain_days
                )
                next_retrain = datetime.now() + timedelta(days=interval_days)
            elif schedule_type == "drift_based":
                # Check drift daily, retrain if threshold exceeded
                next_retrain = datetime.now() + timedelta(days=1)
            elif schedule_type == "performance_based":
                # Check performance weekly
                next_retrain = datetime.now() + timedelta(days=7)
            else:
                logger.error(f"Unknown schedule type: {schedule_type}")
                return

            self.retraining_schedule[model_id] = {
                "schedule_type": schedule_type,
                "schedule_config": config,
                "last_retrain": datetime.now(),
                "next_retrain": next_retrain,
                "retrain_count": 0,
            }

            logger.info(
                f"Retraining scheduled for {model_id}: {schedule_type}, next: {next_retrain}"
            )

        except Exception as e:
            logger.error(f"Failed to schedule retraining for {model_id}: {e}")

    async def check_retraining_due(self) -> List[str]:
        """
        Check which models are due for retraining.

        Returns:
            List of model IDs that need retraining
        """
        due_models = []
        current_time = datetime.now()

        for model_id, schedule_info in self.retraining_schedule.items():
            try:
                if current_time >= schedule_info["next_retrain"]:
                    # Additional checks based on schedule type
                    schedule_type = schedule_info["schedule_type"]

                    if schedule_type == "drift_based":
                        # Check if drift threshold is exceeded
                        if model_id in self.models:
                            if self.models[model_id].drift_score > self.drift_threshold:
                                due_models.append(model_id)
                    elif schedule_type == "performance_based":
                        # Check if performance has degraded (simplified check)
                        # In practice, this would involve more sophisticated performance monitoring
                        due_models.append(model_id)
                    else:
                        # Periodic retraining
                        due_models.append(model_id)

            except Exception as e:
                logger.error(f"Error checking retraining schedule for {model_id}: {e}")

        if due_models:
            logger.info(f"Models due for retraining: {due_models}")

        return due_models

    async def load_model(self, model_id: str, stage: str = "Production") -> Any:
        """
        Load a model from MLflow.

        Args:
            model_id: ID of the model to load
            stage: Model stage to load ('Production', 'Staging', etc.)

        Returns:
            Loaded model object
        """
        try:
            if model_id not in self.models:
                logger.error(f"Model {model_id} not found")
                return None

            model_info = self.models[model_id]

            # Load model from MLflow
            model_uri = f"models:/{model_info.model_name}/{stage}"
            model = mlflow.sklearn.load_model(model_uri)

            logger.info(f"Model {model_id} loaded successfully from stage {stage}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None

    async def compare_models(
        self,
        model_ids: List[str],
        test_data: np.ndarray,
        test_labels: np.ndarray,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on test data.

        Args:
            model_ids: List of model IDs to compare
            test_data: Test data
            test_labels: Test labels
            metrics: List of metrics to calculate

        Returns:
            Dictionary with model comparison results
        """
        try:
            if metrics is None:
                metrics = ["accuracy", "precision", "recall", "f1_score"]

            results = {}

            for model_id in model_ids:
                try:
                    # Load model
                    model = await self.load_model(model_id, stage="Production")
                    if model is None:
                        model = await self.load_model(model_id, stage="Staging")

                    if model is None:
                        logger.warning(f"Could not load model {model_id}")
                        continue

                    # Make predictions
                    predictions = model.predict(test_data)

                    # Calculate metrics
                    model_metrics = {}
                    if "accuracy" in metrics:
                        model_metrics["accuracy"] = accuracy_score(
                            test_labels, predictions
                        )
                    if "precision" in metrics:
                        model_metrics["precision"] = precision_score(
                            test_labels, predictions, average="weighted"
                        )
                    if "recall" in metrics:
                        model_metrics["recall"] = recall_score(
                            test_labels, predictions, average="weighted"
                        )
                    if "f1_score" in metrics:
                        model_metrics["f1_score"] = f1_score(
                            test_labels, predictions, average="weighted"
                        )

                    results[model_id] = model_metrics

                except Exception as e:
                    logger.error(f"Error comparing model {model_id}: {e}")
                    results[model_id] = {"error": str(e)}

            logger.info(f"Model comparison completed for {len(results)} models")
            return results

        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return {}

    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model.

        Args:
            model_id: ID of the model

        Returns:
            ModelInfo object or None if not found
        """
        return self.models.get(model_id)

    async def list_models(
        self, status_filter: Optional[str] = None, name_filter: Optional[str] = None
    ) -> List[ModelInfo]:
        """
        List all registered models.

        Args:
            status_filter: Filter by model status
            name_filter: Filter by model name (partial match)

        Returns:
            List of ModelInfo objects
        """
        models = list(self.models.values())

        if status_filter:
            models = [m for m in models if m.status == status_filter]

        if name_filter:
            models = [m for m in models if name_filter.lower() in m.model_name.lower()]

        return models

    async def archive_model(self, model_id: str) -> bool:
        """
        Archive a model.

        Args:
            model_id: ID of the model to archive

        Returns:
            True if archiving was successful
        """
        try:
            if model_id not in self.models:
                logger.error(f"Model {model_id} not found")
                return False

            model_info = self.models[model_id]

            # Transition to archived in MLflow
            self.client.transition_model_version_stage(
                name=model_info.model_name, version=model_info.version, stage="Archived"
            )

            # Update local status
            model_info.status = "archived"
            model_info.last_updated = datetime.now()

            logger.info(f"Model {model_id} archived successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to archive model {model_id}: {e}")
            return False

    def _calculate_next_retrain_date(
        self, schedule_type: str, config: Dict[str, Any]
    ) -> datetime:
        """Calculate next retraining date based on schedule configuration."""
        current_time = datetime.now()

        if schedule_type == "periodic":
            interval_days = config.get("interval_days", 30)
            return current_time + timedelta(days=interval_days)
        elif schedule_type == "drift_based":
            return current_time + timedelta(days=1)  # Check daily
        elif schedule_type == "performance_based":
            return current_time + timedelta(days=7)  # Check weekly
        else:
            return current_time + timedelta(days=30)  # Default to monthly
