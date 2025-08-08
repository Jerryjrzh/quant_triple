"""
Tests for Automated Model Training Pipeline
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import tempfile
import os
import json

from stock_analysis_system.ml.automated_training_pipeline import (
    AutomatedTrainingPipeline,
    TrainingConfig,
    TrainingJob,
    TrainingResult,
    TrainingJobStatus,
    TrainingJobType,
    ModelType,
    OptimizationMethod
)


class TestAutomatedTrainingPipeline:
    """Test cases for AutomatedTrainingPipeline."""
    
    @pytest.fixture
    def mock_database_url(self):
        """Mock database URL for testing."""
        return "sqlite:///:memory:"
    
    @pytest.fixture
    def mock_mlflow_uri(self):
        """Mock MLflow URI for testing."""
        return "sqlite:///mlflow.db"
    
    @pytest.fixture
    def training_pipeline(self, mock_database_url, mock_mlflow_uri):
        """Create AutomatedTrainingPipeline instance for testing."""
        with patch('stock_analysis_system.ml.automated_training_pipeline.create_engine'):
            with patch('stock_analysis_system.ml.automated_training_pipeline.mlflow'):
                pipeline = AutomatedTrainingPipeline(
                    database_url=mock_database_url,
                    mlflow_tracking_uri=mock_mlflow_uri,
                    max_concurrent_jobs=2
                )
                return pipeline
    
    @pytest.fixture
    def sample_training_config(self):
        """Sample training configuration for testing."""
        return TrainingConfig(
            model_type=ModelType.RANDOM_FOREST,
            target_column="target",
            feature_columns=["feature_1", "feature_2", "feature_3"],
            validation_split=0.2,
            test_split=0.1,
            cross_validation_folds=3,
            random_state=42,
            scaling_method="standard",
            feature_selection_k=None,
            hyperparameter_optimization=True,
            optimization_method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
            optimization_trials=10
        )
    
    @pytest.fixture
    def sample_dataset_config(self):
        """Sample dataset configuration for testing."""
        return {
            "n_samples": 1000,
            "n_features": 10,
            "data_source": "synthetic"
        }
    
    @pytest.fixture
    def sample_training_data(self):
        """Sample training data for testing."""
        np.random.seed(42)
        n_samples, n_features = 100, 5
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add stock-like columns
        X['close'] = 100 + np.cumsum(np.random.randn(n_samples) * 0.1)
        X['volume'] = np.random.lognormal(10, 1, n_samples)
        X['high'] = X['close'] * (1 + np.random.uniform(0, 0.05, n_samples))
        X['low'] = X['close'] * (1 - np.random.uniform(0, 0.05, n_samples))
        
        # Target variable
        y = X['close'].pct_change().shift(-1).fillna(0)
        
        return X.iloc[:-1], y.iloc[:-1]
    
    @pytest.mark.asyncio
    async def test_submit_training_job(self, training_pipeline, sample_training_config, 
                                     sample_dataset_config):
        """Test submitting a training job."""
        # Mock database operations
        training_pipeline._store_training_job = AsyncMock(return_value=None)
        training_pipeline._process_job_queue = AsyncMock(return_value=None)
        
        # Submit job
        job_id = await training_pipeline.submit_training_job(
            model_id="test_model_1",
            job_type=TrainingJobType.INITIAL_TRAINING,
            config=sample_training_config,
            dataset_config=sample_dataset_config,
            priority=5,
            triggered_by="test_user"
        )
        
        assert job_id is not None
        assert len(training_pipeline.job_queue) == 1
        
        job = training_pipeline.job_queue[0]
        assert job.model_id == "test_model_1"
        assert job.job_type == TrainingJobType.INITIAL_TRAINING
        assert job.status == TrainingJobStatus.QUEUED
        assert job.priority == 5
        assert job.triggered_by == "test_user"
    
    @pytest.mark.asyncio
    async def test_job_queue_priority_ordering(self, training_pipeline, sample_training_config, 
                                             sample_dataset_config):
        """Test that jobs are ordered by priority in the queue."""
        training_pipeline._store_training_job = AsyncMock(return_value=None)
        training_pipeline._process_job_queue = AsyncMock(return_value=None)
        
        # Submit jobs with different priorities
        job_id_1 = await training_pipeline.submit_training_job(
            model_id="model_1", job_type=TrainingJobType.INITIAL_TRAINING,
            config=sample_training_config, dataset_config=sample_dataset_config,
            priority=10
        )
        
        job_id_2 = await training_pipeline.submit_training_job(
            model_id="model_2", job_type=TrainingJobType.INITIAL_TRAINING,
            config=sample_training_config, dataset_config=sample_dataset_config,
            priority=1
        )
        
        job_id_3 = await training_pipeline.submit_training_job(
            model_id="model_3", job_type=TrainingJobType.INITIAL_TRAINING,
            config=sample_training_config, dataset_config=sample_dataset_config,
            priority=5
        )
        
        # Check queue ordering (lower priority number = higher priority)
        assert len(training_pipeline.job_queue) == 3
        assert training_pipeline.job_queue[0].priority == 1  # Highest priority
        assert training_pipeline.job_queue[1].priority == 5
        assert training_pipeline.job_queue[2].priority == 10  # Lowest priority
    
    def test_create_model(self, training_pipeline):
        """Test model creation."""
        # Test creating different model types
        linear_model = training_pipeline._create_model(ModelType.LINEAR_REGRESSION)
        assert linear_model.__class__.__name__ == "LinearRegression"
        
        rf_model = training_pipeline._create_model(ModelType.RANDOM_FOREST)
        assert rf_model.__class__.__name__ == "RandomForestRegressor"
        
        # Test with parameters
        rf_model_with_params = training_pipeline._create_model(
            ModelType.RANDOM_FOREST, 
            {"n_estimators": 50, "max_depth": 10}
        )
        assert rf_model_with_params.n_estimators == 50
        assert rf_model_with_params.max_depth == 10
    
    def test_create_pipeline(self, training_pipeline, sample_training_config):
        """Test pipeline creation."""
        model = training_pipeline._create_model(ModelType.LINEAR_REGRESSION)
        pipeline = training_pipeline._create_pipeline(model, sample_training_config)
        
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "scaler"
        assert pipeline.steps[1][0] == "model"
        assert pipeline.steps[0][1].__class__.__name__ == "StandardScaler"
    
    def test_evaluate_model(self, training_pipeline, sample_training_data):
        """Test model evaluation."""
        X, y = sample_training_data
        
        # Create and fit a simple model
        model = training_pipeline._create_model(ModelType.LINEAR_REGRESSION)
        config = TrainingConfig(
            model_type=ModelType.LINEAR_REGRESSION,
            target_column="target",
            feature_columns=list(X.columns)
        )
        pipeline = training_pipeline._create_pipeline(model, config)
        pipeline.fit(X, y)
        
        # Evaluate model
        metrics = training_pipeline._evaluate_model(pipeline, X, y, "test")
        
        assert "test_mse" in metrics
        assert "test_mae" in metrics
        assert "test_r2" in metrics
        assert "test_rmse" in metrics
        
        # Check that metrics are reasonable
        assert metrics["test_mse"] >= 0
        assert metrics["test_mae"] >= 0
        assert metrics["test_rmse"] >= 0
    
    def test_cross_validate_model(self, training_pipeline, sample_training_data, 
                                sample_training_config):
        """Test cross-validation."""
        X, y = sample_training_data
        
        model = training_pipeline._create_model(ModelType.LINEAR_REGRESSION)
        pipeline = training_pipeline._create_pipeline(model, sample_training_config)
        
        cv_scores = training_pipeline._cross_validate_model(pipeline, X, y, sample_training_config)
        
        assert len(cv_scores) == sample_training_config.cross_validation_folds
        assert all(isinstance(score, (int, float)) for score in cv_scores)
    
    def test_get_feature_importance(self, training_pipeline, sample_training_data):
        """Test feature importance extraction."""
        X, y = sample_training_data
        feature_names = X.columns.tolist()
        
        # Test with Random Forest (has feature_importances_)
        rf_model = training_pipeline._create_model(ModelType.RANDOM_FOREST)
        config = TrainingConfig(
            model_type=ModelType.RANDOM_FOREST,
            target_column="target",
            feature_columns=feature_names
        )
        rf_pipeline = training_pipeline._create_pipeline(rf_model, config)
        rf_pipeline.fit(X, y)
        
        rf_importance = training_pipeline._get_feature_importance(rf_pipeline, feature_names)
        assert len(rf_importance) == len(feature_names)
        assert all(isinstance(imp, (int, float)) for imp in rf_importance.values())
        
        # Test with Linear Regression (has coef_)
        lr_model = training_pipeline._create_model(ModelType.LINEAR_REGRESSION)
        lr_config = TrainingConfig(
            model_type=ModelType.LINEAR_REGRESSION,
            target_column="target",
            feature_columns=feature_names
        )
        lr_pipeline = training_pipeline._create_pipeline(lr_model, lr_config)
        lr_pipeline.fit(X, y)
        
        lr_importance = training_pipeline._get_feature_importance(lr_pipeline, feature_names)
        assert len(lr_importance) == len(feature_names)
    
    @pytest.mark.asyncio
    async def test_engineer_features(self, training_pipeline, sample_training_config):
        """Test feature engineering."""
        # Create sample data with stock-like columns
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
            'volume': np.random.lognormal(10, 1, n_samples),
            'high': np.random.uniform(100, 110, n_samples),
            'low': np.random.uniform(90, 100, n_samples),
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples)
        })
        
        y = pd.Series(np.random.randn(n_samples))
        
        X_processed, feature_names = await training_pipeline._engineer_features(
            X, y, sample_training_config
        )
        
        # Check that new features were created
        assert X_processed.shape[1] > X.shape[1]
        assert 'price_change' in X_processed.columns
        assert 'price_volatility' in X_processed.columns
        assert 'price_momentum' in X_processed.columns
        assert 'volume_change' in X_processed.columns
        assert 'price_range' in X_processed.columns
        
        # Check that feature names match
        assert len(feature_names) == X_processed.shape[1]
        assert feature_names == X_processed.columns.tolist()
    
    @pytest.mark.asyncio
    async def test_load_training_data(self, training_pipeline, sample_dataset_config):
        """Test training data loading."""
        X, y = await training_pipeline._load_training_data(sample_dataset_config)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == sample_dataset_config['n_samples'] - 1  # Due to shift in target
        
        # Check that stock-like columns are present
        assert 'close' in X.columns
        assert 'volume' in X.columns
        assert 'high' in X.columns
        assert 'low' in X.columns
    
    @pytest.mark.asyncio
    async def test_save_model_artifacts(self, training_pipeline, sample_training_data, 
                                      sample_training_config):
        """Test model artifact saving."""
        X, y = sample_training_data
        
        # Create and train a model
        model = training_pipeline._create_model(ModelType.LINEAR_REGRESSION)
        pipeline = training_pipeline._create_pipeline(model, sample_training_config)
        pipeline.fit(X, y)
        
        # Mock file operations
        with patch('joblib.dump') as mock_dump, \
             patch('builtins.open', create=True) as mock_open:
            
            artifacts_path = await training_pipeline._save_model_artifacts(
                "test_job_id", pipeline, sample_training_config
            )
            
            assert artifacts_path == "models/test_job_id"
            mock_dump.assert_called_once()
            mock_open.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_job_status(self, training_pipeline, sample_training_config, 
                                sample_dataset_config):
        """Test getting job status."""
        # Mock database operations
        training_pipeline._store_training_job = AsyncMock(return_value=None)
        training_pipeline._process_job_queue = AsyncMock(return_value=None)
        
        # Submit a job
        job_id = await training_pipeline.submit_training_job(
            model_id="test_model",
            job_type=TrainingJobType.INITIAL_TRAINING,
            config=sample_training_config,
            dataset_config=sample_dataset_config
        )
        
        # Get job status
        job_status = await training_pipeline.get_job_status(job_id)
        
        assert job_status is not None
        assert job_status.job_id == job_id
        assert job_status.status == TrainingJobStatus.QUEUED
    
    @pytest.mark.asyncio
    async def test_cancel_job(self, training_pipeline, sample_training_config, 
                            sample_dataset_config):
        """Test job cancellation."""
        # Mock database operations
        training_pipeline._store_training_job = AsyncMock(return_value=None)
        training_pipeline._process_job_queue = AsyncMock(return_value=None)
        training_pipeline._update_training_job_status = AsyncMock(return_value=None)
        
        # Submit a job
        job_id = await training_pipeline.submit_training_job(
            model_id="test_model",
            job_type=TrainingJobType.INITIAL_TRAINING,
            config=sample_training_config,
            dataset_config=sample_dataset_config
        )
        
        # Cancel the job
        success = await training_pipeline.cancel_job(job_id)
        
        assert success is True
        assert len(training_pipeline.job_queue) == 0  # Job removed from queue
    
    @pytest.mark.asyncio
    async def test_get_queue_status(self, training_pipeline, sample_training_config, 
                                  sample_dataset_config):
        """Test getting queue status."""
        # Mock database operations
        training_pipeline._store_training_job = AsyncMock(return_value=None)
        training_pipeline._process_job_queue = AsyncMock(return_value=None)
        
        # Submit multiple jobs
        for i in range(3):
            await training_pipeline.submit_training_job(
                model_id=f"test_model_{i}",
                job_type=TrainingJobType.INITIAL_TRAINING,
                config=sample_training_config,
                dataset_config=sample_dataset_config,
                priority=i + 1
            )
        
        # Get queue status
        status = await training_pipeline.get_queue_status()
        
        assert status['queued_jobs'] == 3
        assert status['running_jobs'] == 0
        assert status['completed_jobs'] == 0
        assert len(status['queue_details']) == 3
        assert len(status['running_details']) == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_old_jobs(self, training_pipeline):
        """Test cleanup of old jobs."""
        # Mock database operations
        training_pipeline.engine.connect = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_conn.execute = MagicMock(return_value=mock_result)
        mock_conn.commit = MagicMock()
        training_pipeline.engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        training_pipeline.engine.connect.return_value.__exit__ = MagicMock(return_value=None)
        
        # Cleanup old jobs
        deleted_count = await training_pipeline.cleanup_old_jobs(retention_days=30)
        
        assert deleted_count == 5
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_bayesian_optimization(self, training_pipeline, sample_training_data, 
                                       sample_training_config):
        """Test Bayesian optimization."""
        X, y = sample_training_data
        
        # Create pipeline
        model = training_pipeline._create_model(ModelType.RANDOM_FOREST)
        pipeline = training_pipeline._create_pipeline(model, sample_training_config)
        
        # Mock optuna to avoid long optimization
        with patch('optuna.create_study') as mock_create_study:
            mock_study = MagicMock()
            mock_study.best_params = {'n_estimators': 100, 'max_depth': 10}
            mock_study.best_value = 0.85
            mock_create_study.return_value = mock_study
            
            # Set small number of trials for testing
            sample_training_config.optimization_trials = 2
            
            best_params, best_score = await training_pipeline._bayesian_optimization(
                pipeline, X, y, sample_training_config
            )
            
            assert isinstance(best_params, dict)
            assert isinstance(best_score, (int, float))
            assert 'n_estimators' in best_params
            assert 'max_depth' in best_params
    
    @pytest.mark.asyncio
    async def test_grid_search_optimization(self, training_pipeline, sample_training_data, 
                                          sample_training_config):
        """Test grid search optimization."""
        X, y = sample_training_data
        
        # Create pipeline
        model = training_pipeline._create_model(ModelType.RIDGE_REGRESSION)
        sample_training_config.model_type = ModelType.RIDGE_REGRESSION
        pipeline = training_pipeline._create_pipeline(model, sample_training_config)
        
        # Reduce CV folds for faster testing
        sample_training_config.cross_validation_folds = 2
        
        best_params, best_score = await training_pipeline._grid_search_optimization(
            pipeline, X, y, sample_training_config
        )
        
        assert isinstance(best_params, dict)
        assert isinstance(best_score, (int, float))
        assert 'alpha' in best_params
    
    @pytest.mark.asyncio
    async def test_random_search_optimization(self, training_pipeline, sample_training_data, 
                                            sample_training_config):
        """Test random search optimization."""
        X, y = sample_training_data
        
        # Create pipeline
        model = training_pipeline._create_model(ModelType.RIDGE_REGRESSION)
        sample_training_config.model_type = ModelType.RIDGE_REGRESSION
        sample_training_config.optimization_trials = 5  # Small number for testing
        pipeline = training_pipeline._create_pipeline(model, sample_training_config)
        
        # Reduce CV folds for faster testing
        sample_training_config.cross_validation_folds = 2
        
        best_params, best_score = await training_pipeline._random_search_optimization(
            pipeline, X, y, sample_training_config
        )
        
        assert isinstance(best_params, dict)
        assert isinstance(best_score, (int, float))
        assert 'alpha' in best_params


class TestTrainingConfig:
    """Test cases for TrainingConfig."""
    
    def test_training_config_creation(self):
        """Test creating training configuration."""
        config = TrainingConfig(
            model_type=ModelType.RANDOM_FOREST,
            target_column="target",
            feature_columns=["feature_1", "feature_2"]
        )
        
        assert config.model_type == ModelType.RANDOM_FOREST
        assert config.target_column == "target"
        assert config.feature_columns == ["feature_1", "feature_2"]
        assert config.validation_split == 0.2  # Default value
        assert config.random_state == 42  # Default value
    
    def test_training_config_with_custom_values(self):
        """Test creating training configuration with custom values."""
        config = TrainingConfig(
            model_type=ModelType.LINEAR_REGRESSION,
            target_column="price_change",
            feature_columns=["close", "volume"],
            validation_split=0.3,
            test_split=0.15,
            cross_validation_folds=10,
            random_state=123,
            scaling_method="robust",
            feature_selection_k=20,
            hyperparameter_optimization=False,
            optimization_method=OptimizationMethod.GRID_SEARCH,
            optimization_trials=50
        )
        
        assert config.model_type == ModelType.LINEAR_REGRESSION
        assert config.validation_split == 0.3
        assert config.test_split == 0.15
        assert config.cross_validation_folds == 10
        assert config.random_state == 123
        assert config.scaling_method == "robust"
        assert config.feature_selection_k == 20
        assert config.hyperparameter_optimization is False
        assert config.optimization_method == OptimizationMethod.GRID_SEARCH
        assert config.optimization_trials == 50


class TestTrainingJob:
    """Test cases for TrainingJob."""
    
    def test_training_job_creation(self):
        """Test creating training job."""
        config = TrainingConfig(
            model_type=ModelType.RANDOM_FOREST,
            target_column="target",
            feature_columns=["feature_1", "feature_2"]
        )
        
        job = TrainingJob(
            job_id="test_job_123",
            model_id="test_model_456",
            job_type=TrainingJobType.INITIAL_TRAINING,
            status=TrainingJobStatus.QUEUED,
            config=config,
            dataset_config={"n_samples": 1000},
            priority=5
        )
        
        assert job.job_id == "test_job_123"
        assert job.model_id == "test_model_456"
        assert job.job_type == TrainingJobType.INITIAL_TRAINING
        assert job.status == TrainingJobStatus.QUEUED
        assert job.config == config
        assert job.dataset_config == {"n_samples": 1000}
        assert job.priority == 5


@pytest.mark.asyncio
async def test_integration_training_pipeline_workflow():
    """Integration test for complete training pipeline workflow."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        database_url = f"sqlite:///{db_path}"
        mlflow_uri = "sqlite:///test_mlflow.db"
        
        with patch('stock_analysis_system.ml.automated_training_pipeline.create_engine'):
            with patch('stock_analysis_system.ml.automated_training_pipeline.mlflow'):
                pipeline = AutomatedTrainingPipeline(
                    database_url=database_url,
                    mlflow_tracking_uri=mlflow_uri,
                    max_concurrent_jobs=1
                )
                
                # Mock database operations
                pipeline._store_training_job = AsyncMock(return_value=None)
                pipeline._update_training_job_status = AsyncMock(return_value=None)
                pipeline._save_model_artifacts = AsyncMock(return_value="models/test_job")
                pipeline._log_training_to_mlflow = AsyncMock(return_value=None)
                
                # Create training configuration
                config = TrainingConfig(
                    model_type=ModelType.LINEAR_REGRESSION,
                    target_column="target",
                    feature_columns=["feature_1", "feature_2"],
                    validation_split=0.2,
                    test_split=0.1,
                    cross_validation_folds=2,  # Small for testing
                    hyperparameter_optimization=False  # Disable for faster testing
                )
                
                dataset_config = {
                    "n_samples": 100,
                    "n_features": 5
                }
                
                # Submit training job
                job_id = await pipeline.submit_training_job(
                    model_id="integration_test_model",
                    job_type=TrainingJobType.INITIAL_TRAINING,
                    config=config,
                    dataset_config=dataset_config,
                    priority=1,
                    triggered_by="integration_test"
                )
                
                assert job_id is not None
                
                # Check initial queue status
                queue_status = await pipeline.get_queue_status()
                assert queue_status['queued_jobs'] >= 0  # Job might have started already
                
                # Wait a bit for job to potentially start
                await asyncio.sleep(0.1)
                
                # Check job status
                job_status = await pipeline.get_job_status(job_id)
                assert job_status is not None
                assert job_status.job_id == job_id
                
                # Test job cancellation
                cancel_success = await pipeline.cancel_job(job_id)
                # Note: Success depends on job state, so we just check it doesn't error
                
                # Test cleanup
                cleanup_count = await pipeline.cleanup_old_jobs(retention_days=0)
                assert isinstance(cleanup_count, int)
                
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
        if os.path.exists("test_mlflow.db"):
            os.unlink("test_mlflow.db")


if __name__ == "__main__":
    pytest.main([__file__])