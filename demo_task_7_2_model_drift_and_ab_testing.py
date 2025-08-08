"""
Demo Script for Task 7.2: Model Drift Detection and A/B Testing Framework

This script demonstrates the comprehensive model drift detection and monitoring system
along with the A/B testing framework for model comparison.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import uuid
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock imports for demo (in real implementation, these would be actual imports)
class MockMLflowClient:
    def __init__(self):
        pass

class MockEngine:
    def connect(self):
        return MockConnection()

class MockConnection:
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def execute(self, query, params=None):
        return MockResult()
    
    def commit(self):
        pass

class MockResult:
    def __init__(self):
        self.rowcount = 10
    
    def fetchall(self):
        return []
    
    def __iter__(self):
        return iter([])

# Import our modules (with mocking for demo)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the database and MLflow dependencies
import unittest.mock as mock

with mock.patch('sqlalchemy.create_engine', return_value=MockEngine()):
    with mock.patch('mlflow.set_tracking_uri'):
        with mock.patch('mlflow.tracking.MlflowClient', return_value=MockMLflowClient()):
            from stock_analysis_system.ml.model_drift_detector import (
                ModelDriftDetector, ModelPerformanceMetrics, DriftType, DriftSeverity
            )
            from stock_analysis_system.ml.ab_testing_framework import (
                ABTestingFramework, ExperimentConfig, ModelVariant, 
                TrafficSplitMethod, ExperimentStatus
            )


class ModelDriftDetectionDemo:
    """Demonstration of model drift detection capabilities."""
    
    def __init__(self):
        self.detector = ModelDriftDetector(
            database_url="sqlite:///demo.db",
            mlflow_tracking_uri="sqlite:///mlflow.db"
        )
        
        # Mock database operations for demo
        self.detector._store_monitoring_registration = self._mock_async_operation
        self.detector._log_drift_detection_results = self._mock_async_operation
        self.detector._store_drift_results_in_database = self._mock_async_operation
    
    async def _mock_async_operation(self, *args, **kwargs):
        """Mock async database operation."""
        await asyncio.sleep(0.01)  # Simulate async operation
        return True
    
    def generate_baseline_data(self, n_samples: int = 1000, n_features: int = 5) -> np.ndarray:
        """Generate baseline training data."""
        np.random.seed(42)
        return np.random.normal(0, 1, (n_samples, n_features))
    
    def generate_baseline_performance(self) -> ModelPerformanceMetrics:
        """Generate baseline performance metrics."""
        return ModelPerformanceMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            custom_metrics={
                'auc_roc': 0.90,
                'log_loss': 0.35,
                'matthews_corr': 0.68
            }
        )
    
    def generate_new_data_no_drift(self, n_samples: int = 500, n_features: int = 5) -> np.ndarray:
        """Generate new data with no significant drift."""
        np.random.seed(43)
        # Similar distribution to baseline
        return np.random.normal(0.1, 1.1, (n_samples, n_features))
    
    def generate_new_data_with_drift(self, n_samples: int = 500, n_features: int = 5) -> np.ndarray:
        """Generate new data with significant drift."""
        np.random.seed(44)
        # Different distribution (shifted mean and variance)
        return np.random.normal(2.5, 2.0, (n_samples, n_features))
    
    def generate_degraded_predictions(self, n_samples: int = 500) -> tuple:
        """Generate predictions and labels showing performance degradation."""
        np.random.seed(45)
        
        # Simulate biased/poor predictions
        predictions = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        
        # True labels are more balanced
        true_labels = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
        
        return predictions, true_labels
    
    async def demonstrate_model_registration(self):
        """Demonstrate model registration for drift monitoring."""
        logger.info("=== Model Registration Demo ===")
        
        # Generate baseline data and performance
        baseline_data = self.generate_baseline_data()
        baseline_performance = self.generate_baseline_performance()
        
        logger.info(f"Generated baseline data: {baseline_data.shape}")
        logger.info(f"Baseline performance: Accuracy={baseline_performance.accuracy:.3f}, "
                   f"F1={baseline_performance.f1_score:.3f}")
        
        # Register model for monitoring
        with mock.patch('mlflow.start_run'), mock.patch('mlflow.log_params'), mock.patch('mlflow.log_metrics'):
            success = await self.detector.register_model_for_monitoring(
                model_id="stock_prediction_model_v1",
                model_name="Stock Price Prediction Model V1",
                baseline_data=baseline_data,
                baseline_performance=baseline_performance,
                monitoring_config={
                    'check_frequency_hours': 6,
                    'min_samples_for_drift': 100,
                    'enable_data_drift': True,
                    'enable_concept_drift': True,
                    'enable_performance_drift': True,
                    'auto_retrain_threshold': DriftSeverity.HIGH
                }
            )
        
        if success:
            logger.info("✅ Model successfully registered for drift monitoring")
            logger.info(f"Monitoring configuration: {self.detector.monitored_models['stock_prediction_model_v1']['config']}")
        else:
            logger.error("❌ Failed to register model for monitoring")
        
        return success
    
    async def demonstrate_no_drift_detection(self):
        """Demonstrate drift detection with no significant drift."""
        logger.info("\n=== No Drift Detection Demo ===")
        
        # Generate new data with no drift
        new_data = self.generate_new_data_no_drift()
        logger.info(f"Generated new data (no drift): {new_data.shape}")
        logger.info(f"New data statistics: mean={np.mean(new_data):.3f}, std={np.std(new_data):.3f}")
        
        # Detect drift
        result = await self.detector.detect_drift(
            model_id="stock_prediction_model_v1",
            new_data=new_data
        )
        
        logger.info(f"Drift Detection Results:")
        logger.info(f"  Data Drift Score: {result.data_drift_score:.4f}")
        logger.info(f"  Concept Drift Score: {result.concept_drift_score:.4f}")
        logger.info(f"  Performance Drift Score: {result.performance_drift_score:.4f}")
        logger.info(f"  Overall Drift Score: {result.overall_drift_score:.4f}")
        logger.info(f"  Number of Alerts: {len(result.alerts)}")
        logger.info(f"  Should Retrain: {result.should_retrain}")
        
        if result.recommendations:
            logger.info("  Recommendations:")
            for rec in result.recommendations:
                logger.info(f"    - {rec}")
        
        return result
    
    async def demonstrate_data_drift_detection(self):
        """Demonstrate drift detection with significant data drift."""
        logger.info("\n=== Data Drift Detection Demo ===")
        
        # Generate new data with drift
        new_data = self.generate_new_data_with_drift()
        logger.info(f"Generated new data (with drift): {new_data.shape}")
        logger.info(f"New data statistics: mean={np.mean(new_data):.3f}, std={np.std(new_data):.3f}")
        
        # Detect drift
        result = await self.detector.detect_drift(
            model_id="stock_prediction_model_v1",
            new_data=new_data
        )
        
        logger.info(f"Drift Detection Results:")
        logger.info(f"  Data Drift Score: {result.data_drift_score:.4f}")
        logger.info(f"  Overall Drift Score: {result.overall_drift_score:.4f}")
        logger.info(f"  Number of Alerts: {len(result.alerts)}")
        logger.info(f"  Should Retrain: {result.should_retrain}")
        
        # Display alerts
        for alert in result.alerts:
            logger.info(f"  🚨 Alert: {alert.drift_type.value} - {alert.severity.value}")
            logger.info(f"     Score: {alert.drift_score:.4f} (threshold: {alert.threshold:.4f})")
            logger.info(f"     Description: {alert.description}")
            if alert.recommendations:
                logger.info(f"     Recommendations:")
                for rec in alert.recommendations:
                    logger.info(f"       - {rec}")
        
        return result
    
    async def demonstrate_performance_drift_detection(self):
        """Demonstrate performance drift detection."""
        logger.info("\n=== Performance Drift Detection Demo ===")
        
        # Generate new data and degraded predictions
        new_data = self.generate_new_data_no_drift()
        predictions, true_labels = self.generate_degraded_predictions()
        
        logger.info(f"Generated predictions and labels: {len(predictions)} samples")
        logger.info(f"Prediction distribution: {np.bincount(predictions) / len(predictions)}")
        logger.info(f"True label distribution: {np.bincount(true_labels) / len(true_labels)}")
        
        # Detect drift
        result = await self.detector.detect_drift(
            model_id="stock_prediction_model_v1",
            new_data=new_data,
            new_predictions=predictions,
            true_labels=true_labels
        )
        
        logger.info(f"Drift Detection Results:")
        logger.info(f"  Performance Drift Score: {result.performance_drift_score:.4f}")
        logger.info(f"  Overall Drift Score: {result.overall_drift_score:.4f}")
        logger.info(f"  Number of Alerts: {len(result.alerts)}")
        logger.info(f"  Should Retrain: {result.should_retrain}")
        
        # Display performance-related alerts
        perf_alerts = [alert for alert in result.alerts if alert.drift_type == DriftType.PERFORMANCE_DRIFT]
        for alert in perf_alerts:
            logger.info(f"  🚨 Performance Alert: {alert.severity.value}")
            logger.info(f"     Score: {alert.drift_score:.4f}")
            logger.info(f"     Description: {alert.description}")
        
        return result
    
    async def demonstrate_retraining_scheduling(self):
        """Demonstrate automated retraining scheduling."""
        logger.info("\n=== Automated Retraining Scheduling Demo ===")
        
        # Mock database operations
        self.detector.engine.connect = mock.MagicMock()
        mock_conn = mock.MagicMock()
        mock_conn.execute = mock.MagicMock()
        mock_conn.commit = mock.MagicMock()
        self.detector.engine.connect.return_value.__enter__ = mock.MagicMock(return_value=mock_conn)
        self.detector.engine.connect.return_value.__exit__ = mock.MagicMock(return_value=None)
        
        # Schedule retraining
        schedule_config = {
            'frequency': 'weekly',
            'drift_threshold': DriftSeverity.HIGH,
            'min_samples': 1000,
            'validation_split': 0.2,
            'notification_channels': ['email', 'slack']
        }
        
        success = await self.detector.schedule_automated_retraining(
            "stock_prediction_model_v1",
            schedule_config
        )
        
        if success:
            logger.info("✅ Automated retraining scheduled successfully")
            schedule_info = self.detector.retraining_schedule["stock_prediction_model_v1"]
            logger.info(f"Schedule: {schedule_info['config']['frequency']}")
            logger.info(f"Next retraining: {schedule_info['next_retrain']}")
            logger.info(f"Drift threshold: {schedule_info['config']['drift_threshold']}")
        
        # Check models due for retraining
        due_models = await self.detector.check_models_due_for_retraining()
        logger.info(f"Models due for retraining: {due_models}")
        
        return success


class ABTestingDemo:
    """Demonstration of A/B testing framework capabilities."""
    
    def __init__(self):
        self.framework = ABTestingFramework(
            database_url="sqlite:///demo.db",
            mlflow_tracking_uri="sqlite:///mlflow.db"
        )
        
        # Mock database operations for demo
        self.framework._store_experiment_config = self._mock_async_operation
        self.framework._update_experiment_status = self._mock_async_operation
        self.framework._store_experiment_metrics = self._mock_async_operation
        self.framework._log_experiment_results = self._mock_async_operation
    
    async def _mock_async_operation(self, *args, **kwargs):
        """Mock async database operation."""
        await asyncio.sleep(0.01)
        return True
    
    def create_experiment_config(self) -> ExperimentConfig:
        """Create a sample experiment configuration."""
        experiment_id = str(uuid.uuid4())
        
        variants = [
            ModelVariant(
                variant_id="control_v1",
                model_id="stock_model_v1",
                model_name="Current Production Model",
                traffic_percentage=40.0,
                description="Current LSTM-based stock prediction model",
                parameters={
                    "model_type": "LSTM",
                    "layers": 2,
                    "hidden_units": 128,
                    "dropout": 0.2
                },
                is_control=True
            ),
            ModelVariant(
                variant_id="treatment_v2",
                model_id="stock_model_v2",
                model_name="Enhanced Transformer Model",
                traffic_percentage=30.0,
                description="New transformer-based model with attention mechanism",
                parameters={
                    "model_type": "Transformer",
                    "attention_heads": 8,
                    "hidden_dim": 256,
                    "dropout": 0.1
                },
                is_control=False
            ),
            ModelVariant(
                variant_id="treatment_v3",
                model_id="stock_model_v3",
                model_name="Ensemble Model",
                traffic_percentage=30.0,
                description="Ensemble of LSTM and Transformer models",
                parameters={
                    "model_type": "Ensemble",
                    "base_models": ["LSTM", "Transformer"],
                    "ensemble_method": "weighted_average"
                },
                is_control=False
            )
        ]
        
        return ExperimentConfig(
            experiment_id=experiment_id,
            experiment_name="Stock Prediction Model Comparison",
            description="A/B testing different model architectures for stock price prediction",
            variants=variants,
            traffic_split_method=TrafficSplitMethod.HASH_BASED,
            primary_metric="prediction_accuracy",
            secondary_metrics=["sharpe_ratio", "max_drawdown", "prediction_latency"],
            minimum_sample_size=1000,
            confidence_level=0.95,
            statistical_power=0.8,
            max_duration_days=14,
            early_stopping_enabled=True,
            significance_threshold=0.05
        )
    
    async def demonstrate_experiment_creation(self):
        """Demonstrate A/B testing experiment creation."""
        logger.info("=== A/B Testing Experiment Creation Demo ===")
        
        # Create experiment configuration
        config = self.create_experiment_config()
        
        logger.info(f"Experiment ID: {config.experiment_id}")
        logger.info(f"Experiment Name: {config.experiment_name}")
        logger.info(f"Number of Variants: {len(config.variants)}")
        logger.info(f"Traffic Split Method: {config.traffic_split_method.value}")
        logger.info(f"Primary Metric: {config.primary_metric}")
        
        logger.info("Variants:")
        for variant in config.variants:
            logger.info(f"  - {variant.variant_id}: {variant.model_name} ({variant.traffic_percentage}%)")
            logger.info(f"    Control: {variant.is_control}")
            logger.info(f"    Parameters: {variant.parameters}")
        
        # Create experiment
        with mock.patch('mlflow.create_experiment', return_value="test_experiment_mlflow_id"):
            success = await self.framework.create_experiment(config)
        
        if success:
            logger.info("✅ A/B testing experiment created successfully")
        else:
            logger.error("❌ Failed to create A/B testing experiment")
        
        return config, success
    
    async def demonstrate_experiment_execution(self, config: ExperimentConfig):
        """Demonstrate running an A/B testing experiment."""
        logger.info("\n=== A/B Testing Experiment Execution Demo ===")
        
        # Start experiment
        self.framework.traffic_router.configure_experiment = mock.AsyncMock(return_value=None)
        
        with mock.patch('mlflow.start_run'), mock.patch('mlflow.log_params'):
            start_success = await self.framework.start_experiment(config.experiment_id)
        
        if not start_success:
            logger.error("❌ Failed to start experiment")
            return
        
        logger.info("✅ Experiment started successfully")
        
        # Simulate traffic routing and metric collection
        logger.info("Simulating user traffic and metric collection...")
        
        # Mock traffic router to return variants based on hash
        def mock_route_user(experiment_id, user_id, context=None):
            hash_val = hash(user_id) % 100
            if hash_val < 40:
                return "control_v1"
            elif hash_val < 70:
                return "treatment_v2"
            else:
                return "treatment_v3"
        
        self.framework.traffic_router.route_user = mock.AsyncMock(side_effect=mock_route_user)
        
        # Simulate 1500 user interactions
        variant_metrics = {
            "control_v1": [],
            "treatment_v2": [],
            "treatment_v3": []
        }
        
        for i in range(1500):
            user_id = f"user_{i}"
            
            # Route traffic
            variant_id = await self.framework.route_traffic(config.experiment_id, user_id)
            
            # Simulate different model performance
            if variant_id == "control_v1":
                # Current model performance
                accuracy = np.random.normal(0.72, 0.08)
                sharpe_ratio = np.random.normal(1.2, 0.3)
                max_drawdown = np.random.normal(0.15, 0.05)
                latency = np.random.normal(50, 10)  # ms
            elif variant_id == "treatment_v2":
                # Transformer model - better accuracy, slightly higher latency
                accuracy = np.random.normal(0.76, 0.07)
                sharpe_ratio = np.random.normal(1.4, 0.3)
                max_drawdown = np.random.normal(0.12, 0.04)
                latency = np.random.normal(75, 15)
            else:  # treatment_v3
                # Ensemble model - best accuracy, highest latency
                accuracy = np.random.normal(0.78, 0.06)
                sharpe_ratio = np.random.normal(1.5, 0.25)
                max_drawdown = np.random.normal(0.10, 0.03)
                latency = np.random.normal(120, 20)
            
            # Ensure metrics are within reasonable bounds
            metrics = {
                "prediction_accuracy": max(0.5, min(0.95, accuracy)),
                "sharpe_ratio": max(0.5, min(3.0, sharpe_ratio)),
                "max_drawdown": max(0.05, min(0.3, max_drawdown)),
                "prediction_latency": max(20, min(200, latency))
            }
            
            variant_metrics[variant_id].append(metrics)
            
            # Record metrics
            await self.framework.record_metric(
                config.experiment_id,
                variant_id,
                user_id,
                metrics
            )
        
        # Display traffic distribution
        logger.info("Traffic Distribution:")
        for variant_id, metrics_list in variant_metrics.items():
            logger.info(f"  {variant_id}: {len(metrics_list)} users")
        
        return variant_metrics
    
    async def demonstrate_experiment_analysis(self, config: ExperimentConfig, variant_metrics: Dict):
        """Demonstrate experiment analysis and results."""
        logger.info("\n=== A/B Testing Experiment Analysis Demo ===")
        
        # Create mock experiment data for analysis
        experiment_data_rows = []
        for variant_id, metrics_list in variant_metrics.items():
            for i, metrics in enumerate(metrics_list):
                row = {
                    'variant_id': variant_id,
                    'user_id': f'user_{variant_id}_{i}',
                    'recorded_at': datetime.now() - timedelta(minutes=i),
                    **metrics
                }
                experiment_data_rows.append(row)
        
        experiment_data = pd.DataFrame(experiment_data_rows)
        self.framework._fetch_experiment_data = mock.AsyncMock(return_value=experiment_data)
        
        # Analyze experiment
        summary = await self.framework.analyze_experiment(config.experiment_id)
        
        if not summary:
            logger.error("❌ Failed to analyze experiment")
            return
        
        logger.info("📊 Experiment Analysis Results:")
        logger.info(f"  Experiment: {summary.experiment_name}")
        logger.info(f"  Status: {summary.status.value}")
        logger.info(f"  Total Samples: {summary.total_samples}")
        logger.info(f"  Duration: {summary.duration_days} days")
        logger.info(f"  Winning Variant: {summary.winning_variant}")
        
        logger.info("\n📈 Variant Performance:")
        for result in summary.results:
            variant_name = next(v.model_name for v in config.variants if v.variant_id == result.variant_id)
            logger.info(f"  {result.variant_id} ({variant_name}):")
            logger.info(f"    Sample Size: {result.sample_size}")
            logger.info(f"    Primary Metric (Accuracy): {result.primary_metric_value:.4f} ± {result.primary_metric_std:.4f}")
            logger.info(f"    Confidence Interval: ({result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f})")
            logger.info(f"    Statistical Significance: {'✅ Yes' if result.statistical_significance else '❌ No'}")
            logger.info(f"    P-value: {result.p_value:.6f}")
            logger.info(f"    Effect Size: {result.effect_size:.4f}")
            
            logger.info(f"    Secondary Metrics:")
            for metric, value in result.secondary_metrics.items():
                logger.info(f"      {metric}: {value:.4f}")
        
        logger.info(f"\n📋 Statistical Summary:")
        for key, value in summary.statistical_summary.items():
            logger.info(f"  {key}: {value}")
        
        logger.info(f"\n💡 Recommendations:")
        for rec in summary.recommendations:
            logger.info(f"  - {rec}")
        
        return summary
    
    async def demonstrate_experiment_completion(self, config: ExperimentConfig):
        """Demonstrate experiment completion."""
        logger.info("\n=== A/B Testing Experiment Completion Demo ===")
        
        # Mock final analysis
        self.framework.analyze_experiment = mock.AsyncMock(return_value=mock.MagicMock())
        
        # Stop experiment
        with mock.patch('mlflow.start_run'), mock.patch('mlflow.log_params'):
            success = await self.framework.stop_experiment(
                config.experiment_id,
                "Demo completion - sufficient statistical power achieved"
            )
        
        if success:
            logger.info("✅ Experiment completed successfully")
            experiment = self.framework.active_experiments[config.experiment_id]
            logger.info(f"Final Status: {experiment['status'].value}")
            logger.info(f"End Date: {experiment['end_date']}")
        else:
            logger.error("❌ Failed to complete experiment")
        
        return success


async def main():
    """Main demo function."""
    logger.info("🚀 Starting Task 7.2 Demo: Model Drift Detection and A/B Testing Framework")
    logger.info("=" * 80)
    
    try:
        # Part 1: Model Drift Detection Demo
        logger.info("PART 1: MODEL DRIFT DETECTION AND MONITORING")
        logger.info("=" * 50)
        
        drift_demo = ModelDriftDetectionDemo()
        
        # Register model for monitoring
        await drift_demo.demonstrate_model_registration()
        
        # Demonstrate no drift scenario
        await drift_demo.demonstrate_no_drift_detection()
        
        # Demonstrate data drift scenario
        await drift_demo.demonstrate_data_drift_detection()
        
        # Demonstrate performance drift scenario
        await drift_demo.demonstrate_performance_drift_detection()
        
        # Demonstrate retraining scheduling
        await drift_demo.demonstrate_retraining_scheduling()
        
        # Part 2: A/B Testing Framework Demo
        logger.info("\n" + "=" * 80)
        logger.info("PART 2: A/B TESTING FRAMEWORK")
        logger.info("=" * 50)
        
        ab_demo = ABTestingDemo()
        
        # Create experiment
        config, success = await ab_demo.demonstrate_experiment_creation()
        
        if success:
            # Execute experiment
            variant_metrics = await ab_demo.demonstrate_experiment_execution(config)
            
            # Analyze results
            await ab_demo.demonstrate_experiment_analysis(config, variant_metrics)
            
            # Complete experiment
            await ab_demo.demonstrate_experiment_completion(config)
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 Task 7.2 Demo completed successfully!")
        logger.info("=" * 80)
        
        # Summary of capabilities demonstrated
        logger.info("\n📋 CAPABILITIES DEMONSTRATED:")
        logger.info("Model Drift Detection:")
        logger.info("  ✅ Model registration for monitoring")
        logger.info("  ✅ Data drift detection using statistical methods")
        logger.info("  ✅ Concept drift detection")
        logger.info("  ✅ Performance drift detection")
        logger.info("  ✅ Automated alert generation")
        logger.info("  ✅ Retraining recommendations")
        logger.info("  ✅ Automated retraining scheduling")
        
        logger.info("\nA/B Testing Framework:")
        logger.info("  ✅ Multi-variant experiment configuration")
        logger.info("  ✅ Traffic routing and splitting")
        logger.info("  ✅ Metric collection and tracking")
        logger.info("  ✅ Statistical significance testing")
        logger.info("  ✅ Comprehensive result analysis")
        logger.info("  ✅ Experiment lifecycle management")
        logger.info("  ✅ MLflow integration for tracking")
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())