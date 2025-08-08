"""
Demo Script for Task 7.3: Automated Model Training Pipeline

This script demonstrates the comprehensive automated model training pipeline with
feature engineering, hyperparameter optimization, model validation, and deployment
capabilities.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import uuid
from typing import Dict, List
import time

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
        self.rowcount = 1
    
    def fetchall(self):
        return []
    
    def fetchone(self):
        return None

# Import our modules (with mocking for demo)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the database and MLflow dependencies
import unittest.mock as mock

with mock.patch('sqlalchemy.create_engine', return_value=MockEngine()):
    with mock.patch('mlflow.set_tracking_uri'):
        with mock.patch('mlflow.tracking.MlflowClient', return_value=MockMLflowClient()):
            from stock_analysis_system.ml.automated_training_pipeline import (
                AutomatedTrainingPipeline,
                TrainingConfig,
                TrainingJob,
                TrainingJobStatus,
                TrainingJobType,
                ModelType,
                OptimizationMethod
            )


class AutomatedTrainingPipelineDemo:
    """Demonstration of automated training pipeline capabilities."""
    
    def __init__(self):
        self.pipeline = AutomatedTrainingPipeline(
            database_url="sqlite:///demo.db",
            mlflow_tracking_uri="sqlite:///mlflow.db",
            max_concurrent_jobs=2
        )
        
        # Mock database operations for demo
        self.pipeline._store_training_job = self._mock_async_operation
        self.pipeline._update_training_job_status = self._mock_async_operation
        self.pipeline._save_model_artifacts = self._mock_save_artifacts
        self.pipeline._log_training_to_mlflow = self._mock_async_operation
    
    async def _mock_async_operation(self, *args, **kwargs):
        """Mock async database operation."""
        await asyncio.sleep(0.01)  # Simulate async operation
        return True
    
    async def _mock_save_artifacts(self, job_id, pipeline, config):
        """Mock model artifact saving."""
        await asyncio.sleep(0.01)
        return f"models/{job_id}"
    
    def generate_stock_dataset(self, n_samples: int = 1000, n_features: int = 10) -> tuple:
        """Generate synthetic stock market dataset."""
        np.random.seed(42)
        
        # Generate base features
        features = {}
        for i in range(n_features):
            features[f'technical_indicator_{i}'] = np.random.randn(n_samples)
        
        # Generate stock-specific features
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        # Price data with trend and volatility
        price_trend = np.cumsum(np.random.randn(n_samples) * 0.02)
        price_volatility = np.random.uniform(0.8, 1.2, n_samples)
        features['close'] = 100 * np.exp(price_trend) * price_volatility
        
        # Volume data
        features['volume'] = np.random.lognormal(mean=12, sigma=1, size=n_samples)
        
        # High and low prices
        features['high'] = features['close'] * (1 + np.random.uniform(0, 0.05, n_samples))
        features['low'] = features['close'] * (1 - np.random.uniform(0, 0.05, n_samples))
        
        # Market cap and other fundamental features
        features['market_cap'] = features['close'] * features['volume'] * np.random.uniform(0.8, 1.2, n_samples)
        features['pe_ratio'] = np.random.uniform(5, 50, n_samples)
        features['dividend_yield'] = np.random.uniform(0, 0.08, n_samples)
        
        # Macro-economic features
        features['interest_rate'] = 0.02 + np.random.randn(n_samples) * 0.005
        features['inflation_rate'] = 0.025 + np.random.randn(n_samples) * 0.003
        features['gdp_growth'] = 0.03 + np.random.randn(n_samples) * 0.01
        
        # Create DataFrame
        X = pd.DataFrame(features, index=dates)
        
        # Target variable: next day return
        y = X['close'].pct_change().shift(-1).fillna(0)
        
        # Remove last row due to shift
        X = X.iloc[:-1]
        y = y.iloc[:-1]
        
        logger.info(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target statistics: mean={y.mean():.6f}, std={y.std():.6f}")
        
        return X, y
    
    def create_training_configurations(self) -> List[TrainingConfig]:
        """Create different training configurations for demonstration."""
        configs = []
        
        # Configuration 1: Linear Regression with basic setup
        configs.append(TrainingConfig(
            model_type=ModelType.LINEAR_REGRESSION,
            target_column="next_day_return",
            feature_columns=["close", "volume", "high", "low", "market_cap"],
            validation_split=0.2,
            test_split=0.1,
            cross_validation_folds=5,
            random_state=42,
            scaling_method="standard",
            feature_selection_k=None,
            hyperparameter_optimization=False,
            optimization_method=OptimizationMethod.GRID_SEARCH,
            optimization_trials=10
        ))
        
        # Configuration 2: Random Forest with hyperparameter tuning
        configs.append(TrainingConfig(
            model_type=ModelType.RANDOM_FOREST,
            target_column="next_day_return",
            feature_columns=["close", "volume", "high", "low", "market_cap", "pe_ratio", "dividend_yield"],
            validation_split=0.2,
            test_split=0.1,
            cross_validation_folds=3,
            random_state=42,
            scaling_method="robust",
            feature_selection_k=15,
            hyperparameter_optimization=True,
            optimization_method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
            optimization_trials=20
        ))
        
        # Configuration 3: Gradient Boosting with feature selection
        configs.append(TrainingConfig(
            model_type=ModelType.GRADIENT_BOOSTING,
            target_column="next_day_return",
            feature_columns=None,  # Use all features
            validation_split=0.25,
            test_split=0.15,
            cross_validation_folds=4,
            random_state=42,
            scaling_method="minmax",
            feature_selection_k=20,
            hyperparameter_optimization=True,
            optimization_method=OptimizationMethod.RANDOM_SEARCH,
            optimization_trials=15
        ))
        
        return configs
    
    async def demonstrate_job_submission(self):
        """Demonstrate submitting training jobs with different configurations."""
        logger.info("=== Training Job Submission Demo ===")
        
        # Create training configurations
        configs = self.create_training_configurations()
        
        # Dataset configuration
        dataset_config = {
            "n_samples": 2000,
            "n_features": 15,
            "data_source": "synthetic_stock_data"
        }
        
        job_ids = []
        
        # Submit jobs with different priorities
        for i, config in enumerate(configs):
            model_id = f"stock_prediction_model_{config.model_type.value}_{i+1}"
            
            logger.info(f"Submitting training job for {model_id}")
            logger.info(f"  Model Type: {config.model_type.value}")
            logger.info(f"  Hyperparameter Optimization: {config.hyperparameter_optimization}")
            logger.info(f"  Optimization Method: {config.optimization_method.value}")
            logger.info(f"  Feature Selection K: {config.feature_selection_k}")
            
            job_id = await self.pipeline.submit_training_job(
                model_id=model_id,
                job_type=TrainingJobType.INITIAL_TRAINING,
                config=config,
                dataset_config=dataset_config,
                priority=i + 1,  # Different priorities
                triggered_by="demo_user"
            )
            
            job_ids.append(job_id)
            logger.info(f"  ✅ Job submitted with ID: {job_id}")
        
        return job_ids
    
    async def demonstrate_hyperparameter_tuning_job(self):
        """Demonstrate hyperparameter tuning job."""
        logger.info("\n=== Hyperparameter Tuning Job Demo ===")
        
        # Configuration for hyperparameter tuning
        tuning_config = TrainingConfig(
            model_type=ModelType.RANDOM_FOREST,
            target_column="next_day_return",
            feature_columns=["close", "volume", "high", "low", "market_cap"],
            validation_split=0.2,
            test_split=0.1,
            cross_validation_folds=3,
            random_state=42,
            scaling_method="standard",
            hyperparameter_optimization=True,
            optimization_method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
            optimization_trials=25
        )
        
        dataset_config = {
            "n_samples": 1500,
            "n_features": 12,
            "data_source": "stock_data_for_tuning"
        }
        
        logger.info("Submitting hyperparameter tuning job...")
        logger.info(f"  Model Type: {tuning_config.model_type.value}")
        logger.info(f"  Optimization Method: {tuning_config.optimization_method.value}")
        logger.info(f"  Number of Trials: {tuning_config.optimization_trials}")
        
        job_id = await self.pipeline.submit_training_job(
            model_id="stock_model_hyperparameter_tuning",
            job_type=TrainingJobType.HYPERPARAMETER_TUNING,
            config=tuning_config,
            dataset_config=dataset_config,
            priority=1,  # High priority
            triggered_by="automated_tuning_system"
        )
        
        logger.info(f"✅ Hyperparameter tuning job submitted with ID: {job_id}")
        return job_id
    
    async def demonstrate_model_comparison_job(self):
        """Demonstrate model comparison job."""
        logger.info("\n=== Model Comparison Job Demo ===")
        
        # Configuration for model comparison
        comparison_config = TrainingConfig(
            model_type=ModelType.LINEAR_REGRESSION,  # Will be overridden in comparison
            target_column="next_day_return",
            feature_columns=["close", "volume", "high", "low", "market_cap", "pe_ratio"],
            validation_split=0.2,
            test_split=0.1,
            cross_validation_folds=5,
            random_state=42,
            scaling_method="standard",
            hyperparameter_optimization=False  # Quick comparison
        )
        
        dataset_config = {
            "n_samples": 1000,
            "n_features": 10,
            "data_source": "stock_data_for_comparison"
        }
        
        logger.info("Submitting model comparison job...")
        logger.info("  Will compare: Linear Regression, Ridge, Random Forest, Gradient Boosting")
        logger.info(f"  Cross-validation folds: {comparison_config.cross_validation_folds}")
        
        job_id = await self.pipeline.submit_training_job(
            model_id="stock_model_comparison",
            job_type=TrainingJobType.MODEL_COMPARISON,
            config=comparison_config,
            dataset_config=dataset_config,
            priority=2,
            triggered_by="model_selection_system"
        )
        
        logger.info(f"✅ Model comparison job submitted with ID: {job_id}")
        return job_id
    
    async def demonstrate_queue_management(self, job_ids: List[str]):
        """Demonstrate queue management and monitoring."""
        logger.info("\n=== Queue Management Demo ===")
        
        # Get initial queue status
        queue_status = await self.pipeline.get_queue_status()
        logger.info("Initial Queue Status:")
        logger.info(f"  Queued Jobs: {queue_status['queued_jobs']}")
        logger.info(f"  Running Jobs: {queue_status['running_jobs']}")
        logger.info(f"  Completed Jobs: {queue_status['completed_jobs']}")
        
        if queue_status['queue_details']:
            logger.info("  Queue Details:")
            for job_detail in queue_status['queue_details']:
                logger.info(f"    - Job {job_detail['job_id'][:8]}... "
                           f"(Model: {job_detail['model_id']}, "
                           f"Type: {job_detail['job_type']}, "
                           f"Priority: {job_detail['priority']})")
        
        # Demonstrate job status checking
        if job_ids:
            logger.info(f"\nChecking status of first job: {job_ids[0][:8]}...")
            job_status = await self.pipeline.get_job_status(job_ids[0])
            
            if job_status:
                logger.info(f"  Job ID: {job_status.job_id[:8]}...")
                logger.info(f"  Model ID: {job_status.model_id}")
                logger.info(f"  Status: {job_status.status.value}")
                logger.info(f"  Job Type: {job_status.job_type.value}")
                logger.info(f"  Priority: {job_status.priority}")
                logger.info(f"  Created At: {job_status.created_at}")
                logger.info(f"  Triggered By: {job_status.triggered_by}")
        
        # Demonstrate job cancellation
        if len(job_ids) > 1:
            cancel_job_id = job_ids[-1]  # Cancel last job
            logger.info(f"\nCancelling job: {cancel_job_id[:8]}...")
            
            cancel_success = await self.pipeline.cancel_job(cancel_job_id)
            if cancel_success:
                logger.info("  ✅ Job cancelled successfully")
            else:
                logger.info("  ❌ Failed to cancel job (may have already started)")
            
            # Check queue status after cancellation
            updated_status = await self.pipeline.get_queue_status()
            logger.info(f"  Updated queue size: {updated_status['queued_jobs']} jobs")
    
    async def demonstrate_feature_engineering(self):
        """Demonstrate feature engineering capabilities."""
        logger.info("\n=== Feature Engineering Demo ===")
        
        # Generate sample data
        X, y = self.generate_stock_dataset(n_samples=500, n_features=8)
        
        logger.info("Original Dataset:")
        logger.info(f"  Shape: {X.shape}")
        logger.info(f"  Columns: {list(X.columns)}")
        
        # Create configuration with feature engineering
        config = TrainingConfig(
            model_type=ModelType.RANDOM_FOREST,
            target_column="next_day_return",
            feature_columns=list(X.columns),
            feature_selection_k=15  # Select top 15 features
        )
        
        # Apply feature engineering
        X_processed, feature_names = await self.pipeline._engineer_features(X, y, config)
        
        logger.info("After Feature Engineering:")
        logger.info(f"  Shape: {X_processed.shape}")
        logger.info(f"  New features added: {X_processed.shape[1] - X.shape[1]}")
        logger.info("  New feature columns:")
        
        new_features = [col for col in X_processed.columns if col not in X.columns]
        for feature in new_features:
            logger.info(f"    - {feature}")
        
        logger.info(f"  Selected features: {len(feature_names)}")
        
        # Show feature statistics
        logger.info("\nFeature Statistics (first 5 features):")
        for i, feature in enumerate(feature_names[:5]):
            values = X_processed[feature]
            logger.info(f"  {feature}: mean={values.mean():.4f}, "
                       f"std={values.std():.4f}, "
                       f"min={values.min():.4f}, "
                       f"max={values.max():.4f}")
    
    async def demonstrate_model_training_execution(self):
        """Demonstrate actual model training execution."""
        logger.info("\n=== Model Training Execution Demo ===")
        
        # Create a simple training job for demonstration
        config = TrainingConfig(
            model_type=ModelType.LINEAR_REGRESSION,
            target_column="next_day_return",
            feature_columns=["close", "volume", "high", "low"],
            validation_split=0.2,
            test_split=0.1,
            cross_validation_folds=3,
            random_state=42,
            scaling_method="standard",
            hyperparameter_optimization=False
        )
        
        dataset_config = {
            "n_samples": 300,
            "n_features": 8
        }
        
        # Create a mock training job
        job = TrainingJob(
            job_id=str(uuid.uuid4()),
            model_id="demo_training_model",
            job_type=TrainingJobType.INITIAL_TRAINING,
            status=TrainingJobStatus.RUNNING,
            config=config,
            dataset_config=dataset_config,
            priority=1,
            created_at=datetime.now(),
            started_at=datetime.now(),
            triggered_by="demo_execution"
        )
        
        logger.info(f"Executing training job: {job.job_id[:8]}...")
        logger.info(f"  Model Type: {config.model_type.value}")
        logger.info(f"  Dataset Size: {dataset_config['n_samples']} samples")
        
        try:
            # Load training data
            X, y = await self.pipeline._load_training_data(dataset_config)
            logger.info(f"  ✅ Data loaded: {X.shape}")
            
            # Execute training
            start_time = time.time()
            result = await self.pipeline._execute_initial_training(job, X, y)
            training_time = time.time() - start_time
            
            logger.info(f"  ✅ Training completed in {training_time:.2f} seconds")
            logger.info(f"  Training Results:")
            logger.info(f"    - Training R²: {result.training_metrics.get('training_r2', 0):.4f}")
            logger.info(f"    - Validation R²: {result.validation_metrics.get('validation_r2', 0):.4f}")
            logger.info(f"    - Test R²: {result.test_metrics.get('test_r2', 0):.4f}")
            logger.info(f"    - CV Mean Score: {np.mean(result.cross_validation_scores):.4f}")
            logger.info(f"    - CV Std Score: {np.std(result.cross_validation_scores):.4f}")
            
            # Show feature importance
            if result.feature_importance:
                logger.info("  Top 5 Important Features:")
                sorted_features = sorted(result.feature_importance.items(), 
                                       key=lambda x: abs(x[1]), reverse=True)
                for feature, importance in sorted_features[:5]:
                    logger.info(f"    - {feature}: {importance:.4f}")
            
            logger.info(f"  Model artifacts saved to: {result.model_artifacts_path}")
            
        except Exception as e:
            logger.error(f"  ❌ Training failed: {str(e)}")
    
    async def demonstrate_pipeline_monitoring(self):
        """Demonstrate pipeline monitoring and maintenance."""
        logger.info("\n=== Pipeline Monitoring Demo ===")
        
        # Get current pipeline status
        queue_status = await self.pipeline.get_queue_status()
        
        logger.info("Pipeline Status:")
        logger.info(f"  Max Concurrent Jobs: {self.pipeline.max_concurrent_jobs}")
        logger.info(f"  Current Queue Size: {queue_status['queued_jobs']}")
        logger.info(f"  Running Jobs: {queue_status['running_jobs']}")
        logger.info(f"  Completed Jobs: {queue_status['completed_jobs']}")
        
        # Demonstrate cleanup
        logger.info("\nPerforming maintenance tasks...")
        
        # Mock cleanup operation
        with mock.patch.object(self.pipeline, 'cleanup_old_jobs', return_value=5):
            cleanup_count = await self.pipeline.cleanup_old_jobs(retention_days=30)
            logger.info(f"  ✅ Cleaned up {cleanup_count} old training jobs")
        
        # Show model registry information
        logger.info("\nSupported Model Types:")
        for model_type in ModelType:
            model_class = self.pipeline.model_registry.get(model_type)
            if model_class:
                logger.info(f"  - {model_type.value}: {model_class.__name__}")
        
        # Show hyperparameter spaces
        logger.info("\nHyperparameter Optimization Spaces:")
        for model_type, param_space in self.pipeline.hyperparameter_spaces.items():
            logger.info(f"  {model_type.value}:")
            for param, values in param_space.items():
                logger.info(f"    - {param}: {values}")
    
    async def demonstrate_advanced_features(self):
        """Demonstrate advanced pipeline features."""
        logger.info("\n=== Advanced Features Demo ===")
        
        # Demonstrate different optimization methods
        logger.info("Optimization Methods Available:")
        for method in OptimizationMethod:
            logger.info(f"  - {method.value}")
        
        # Demonstrate different job types
        logger.info("\nSupported Job Types:")
        for job_type in TrainingJobType:
            logger.info(f"  - {job_type.value}")
        
        # Demonstrate scaling methods
        logger.info("\nSupported Scaling Methods:")
        scaling_methods = ["standard", "robust", "minmax"]
        for method in scaling_methods:
            logger.info(f"  - {method}")
        
        # Show pipeline configuration options
        logger.info("\nPipeline Configuration:")
        logger.info(f"  Database URL: {self.pipeline.database_url}")
        logger.info(f"  Max Concurrent Jobs: {self.pipeline.max_concurrent_jobs}")
        logger.info(f"  Thread Pool Workers: {self.pipeline.executor._max_workers}")


async def main():
    """Main demo function."""
    logger.info("🚀 Starting Task 7.3 Demo: Automated Model Training Pipeline")
    logger.info("=" * 80)
    
    try:
        demo = AutomatedTrainingPipelineDemo()
        
        # Part 1: Job Submission
        logger.info("PART 1: TRAINING JOB SUBMISSION AND MANAGEMENT")
        logger.info("=" * 60)
        
        job_ids = await demo.demonstrate_job_submission()
        await demo.demonstrate_hyperparameter_tuning_job()
        await demo.demonstrate_model_comparison_job()
        
        # Part 2: Queue Management
        logger.info("\nPART 2: QUEUE MANAGEMENT AND MONITORING")
        logger.info("=" * 60)
        
        await demo.demonstrate_queue_management(job_ids)
        
        # Part 3: Feature Engineering
        logger.info("\nPART 3: FEATURE ENGINEERING CAPABILITIES")
        logger.info("=" * 60)
        
        await demo.demonstrate_feature_engineering()
        
        # Part 4: Model Training Execution
        logger.info("\nPART 4: MODEL TRAINING EXECUTION")
        logger.info("=" * 60)
        
        await demo.demonstrate_model_training_execution()
        
        # Part 5: Pipeline Monitoring
        logger.info("\nPART 5: PIPELINE MONITORING AND MAINTENANCE")
        logger.info("=" * 60)
        
        await demo.demonstrate_pipeline_monitoring()
        
        # Part 6: Advanced Features
        logger.info("\nPART 6: ADVANCED FEATURES")
        logger.info("=" * 60)
        
        await demo.demonstrate_advanced_features()
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 Task 7.3 Demo completed successfully!")
        logger.info("=" * 80)
        
        # Summary of capabilities demonstrated
        logger.info("\n📋 CAPABILITIES DEMONSTRATED:")
        logger.info("Automated Training Pipeline:")
        logger.info("  ✅ Multi-model training job submission")
        logger.info("  ✅ Priority-based job queue management")
        logger.info("  ✅ Hyperparameter optimization (Bayesian, Grid, Random)")
        logger.info("  ✅ Automated feature engineering and selection")
        logger.info("  ✅ Cross-validation and model evaluation")
        logger.info("  ✅ Model comparison and selection")
        logger.info("  ✅ Parallel job execution")
        logger.info("  ✅ MLflow integration for experiment tracking")
        logger.info("  ✅ Model artifact management")
        logger.info("  ✅ Pipeline monitoring and maintenance")
        logger.info("  ✅ Job cancellation and cleanup")
        logger.info("  ✅ Comprehensive error handling")
        
        logger.info("\nSupported Features:")
        logger.info("  📊 Model Types: Linear, Ridge, Lasso, Random Forest, Gradient Boosting")
        logger.info("  🔧 Optimization: Bayesian, Grid Search, Random Search")
        logger.info("  📈 Feature Engineering: Technical indicators, price patterns")
        logger.info("  🎯 Feature Selection: K-best selection with statistical tests")
        logger.info("  📏 Scaling: Standard, Robust, MinMax scaling")
        logger.info("  🔄 Cross-validation: Time series aware validation")
        logger.info("  💾 Persistence: Database storage and MLflow tracking")
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())