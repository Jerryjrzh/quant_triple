"""
Automated Model Training Pipeline

This module implements a comprehensive automated model training pipeline with
feature engineering, hyperparameter optimization, model validation, and deployment
capabilities for the Stock Analysis System.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import optuna
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine, text
import json
import joblib
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingJobStatus(str, Enum):
    """Status of training jobs."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TrainingJobType(str, Enum):
    """Types of training jobs."""
    INITIAL_TRAINING = "initial_training"
    RETRAINING = "retraining"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    FEATURE_SELECTION = "feature_selection"
    MODEL_COMPARISON = "model_comparison"

class ModelType(str, Enum):
    """Supported model types."""
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"

class OptimizationMethod(str, Enum):
    """Hyperparameter optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_type: ModelType
    target_column: str
    feature_columns: List[str]
    validation_split: float = 0.2
    test_split: float = 0.1
    cross_validation_folds: int = 5
    random_state: int = 42
    scaling_method: str = "standard"  # standard, robust, minmax
    feature_selection_k: Optional[int] = None
    hyperparameter_optimization: bool = True
    optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN_OPTIMIZATION
    optimization_trials: int = 100
    early_stopping_rounds: Optional[int] = 10
    model_parameters: Dict[str, Any] = None

@dataclass
class TrainingJob:
    """Training job definition."""
    job_id: str
    model_id: str
    job_type: TrainingJobType
    status: TrainingJobStatus
    config: TrainingConfig
    dataset_config: Dict[str, Any]
    priority: int = 5
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    final_metrics: Optional[Dict[str, float]] = None
    model_artifacts_path: Optional[str] = None
    error_message: Optional[str] = None
    triggered_by: Optional[str] = None
    parent_job_id: Optional[str] = None

@dataclass
class TrainingResult:
    """Result of model training."""
    job_id: str
    model_id: str
    model_type: ModelType
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    cross_validation_scores: List[float]
    feature_importance: Dict[str, float]
    best_parameters: Dict[str, Any]
    model_artifacts_path: str
    training_duration: float
    data_shape: Tuple[int, int]
    feature_names: List[str]

class AutomatedTrainingPipeline:
    """
    Comprehensive automated model training pipeline.
    
    Features:
    - Automated feature engineering and selection
    - Hyperparameter optimization using multiple methods
    - Cross-validation and model evaluation
    - Automated model deployment
    - Training job queue management
    - MLflow integration for experiment tracking
    """
    
    def __init__(self, database_url: str, mlflow_tracking_uri: str, 
                 max_concurrent_jobs: int = 4):
        """
        Initialize the automated training pipeline.
        
        Args:
            database_url: Database connection string
            mlflow_tracking_uri: MLflow tracking server URI
            max_concurrent_jobs: Maximum number of concurrent training jobs
        """
        self.database_url = database_url
        self.engine = create_engine(database_url)
        
        # MLflow setup
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
        # Job management
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_queue = []
        self.running_jobs = {}
        self.completed_jobs = {}
        
        # Model registry
        self.model_registry = {
            ModelType.LINEAR_REGRESSION: LinearRegression,
            ModelType.RIDGE_REGRESSION: Ridge,
            ModelType.LASSO_REGRESSION: Lasso,
            ModelType.RANDOM_FOREST: RandomForestRegressor,
            ModelType.GRADIENT_BOOSTING: GradientBoostingRegressor
        }
        
        # Default hyperparameter spaces
        self.hyperparameter_spaces = {
            ModelType.RIDGE_REGRESSION: {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            ModelType.LASSO_REGRESSION: {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            ModelType.RANDOM_FOREST: {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            ModelType.GRADIENT_BOOSTING: {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        # Executor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        
        logger.info("AutomatedTrainingPipeline initialized successfully")
    
    async def submit_training_job(self, model_id: str, job_type: TrainingJobType,
                                config: TrainingConfig, dataset_config: Dict[str, Any],
                                priority: int = 5, triggered_by: str = None) -> str:
        """
        Submit a new training job to the queue.
        
        Args:
            model_id: Model identifier
            job_type: Type of training job
            config: Training configuration
            dataset_config: Dataset configuration
            priority: Job priority (lower number = higher priority)
            triggered_by: Who/what triggered the job
            
        Returns:
            str: Job ID
        """
        job_id = str(uuid.uuid4())
        
        job = TrainingJob(
            job_id=job_id,
            model_id=model_id,
            job_type=job_type,
            status=TrainingJobStatus.QUEUED,
            config=config,
            dataset_config=dataset_config,
            priority=priority,
            created_at=datetime.now(),
            triggered_by=triggered_by
        )
        
        # Store job in database
        await self._store_training_job(job)
        
        # Add to queue
        self.job_queue.append(job)
        self.job_queue.sort(key=lambda x: x.priority)  # Sort by priority
        
        logger.info(f"Training job {job_id} submitted for model {model_id}")
        
        # Try to start job if resources available
        await self._process_job_queue()
        
        return job_id
    
    async def _process_job_queue(self):
        """Process the job queue and start jobs if resources are available."""
        while (len(self.running_jobs) < self.max_concurrent_jobs and 
               len(self.job_queue) > 0):
            
            job = self.job_queue.pop(0)  # Get highest priority job
            
            # Update job status
            job.status = TrainingJobStatus.RUNNING
            job.started_at = datetime.now()
            
            # Store in running jobs
            self.running_jobs[job.job_id] = job
            
            # Update database
            await self._update_training_job_status(job.job_id, TrainingJobStatus.RUNNING)
            
            # Start job asynchronously
            asyncio.create_task(self._execute_training_job(job))
            
            logger.info(f"Started training job {job.job_id}")
    
    async def _execute_training_job(self, job: TrainingJob):
        """Execute a training job."""
        try:
            logger.info(f"Executing training job {job.job_id} for model {job.model_id}")
            
            # Load and prepare data
            X, y = await self._load_training_data(job.dataset_config)
            
            # Execute training based on job type
            if job.job_type == TrainingJobType.INITIAL_TRAINING:
                result = await self._execute_initial_training(job, X, y)
            elif job.job_type == TrainingJobType.RETRAINING:
                result = await self._execute_retraining(job, X, y)
            elif job.job_type == TrainingJobType.HYPERPARAMETER_TUNING:
                result = await self._execute_hyperparameter_tuning(job, X, y)
            elif job.job_type == TrainingJobType.FEATURE_SELECTION:
                result = await self._execute_feature_selection(job, X, y)
            elif job.job_type == TrainingJobType.MODEL_COMPARISON:
                result = await self._execute_model_comparison(job, X, y)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")
            
            # Update job completion
            job.status = TrainingJobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.duration_seconds = int((job.completed_at - job.started_at).total_seconds())
            job.final_metrics = result.validation_metrics
            job.model_artifacts_path = result.model_artifacts_path
            
            # Store result
            self.completed_jobs[job.job_id] = result
            
            logger.info(f"Training job {job.job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Training job {job.job_id} failed: {str(e)}")
            
            job.status = TrainingJobStatus.FAILED
            job.completed_at = datetime.now()
            job.error_message = str(e)
            
        finally:
            # Remove from running jobs
            if job.job_id in self.running_jobs:
                del self.running_jobs[job.job_id]
            
            # Update database
            await self._update_training_job_status(job.job_id, job.status)
            
            # Process next job in queue
            await self._process_job_queue()
    
    async def _execute_initial_training(self, job: TrainingJob, X: pd.DataFrame, 
                                      y: pd.Series) -> TrainingResult:
        """Execute initial model training."""
        config = job.config
        
        # Feature engineering and selection
        X_processed, feature_names = await self._engineer_features(X, y, config)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_processed, y, test_size=config.validation_split + config.test_split,
            random_state=config.random_state
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=config.test_split / (config.validation_split + config.test_split),
            random_state=config.random_state
        )
        
        # Create and train model
        model = self._create_model(config.model_type, config.model_parameters)
        
        # Create pipeline with preprocessing
        pipeline = self._create_pipeline(model, config)
        
        # Train model
        start_time = datetime.now()
        pipeline.fit(X_train, y_train)
        training_duration = (datetime.now() - start_time).total_seconds()
        
        # Evaluate model
        training_metrics = self._evaluate_model(pipeline, X_train, y_train, "training")
        validation_metrics = self._evaluate_model(pipeline, X_val, y_val, "validation")
        test_metrics = self._evaluate_model(pipeline, X_test, y_test, "test")
        
        # Cross-validation
        cv_scores = self._cross_validate_model(pipeline, X_processed, y, config)
        
        # Feature importance
        feature_importance = self._get_feature_importance(pipeline, feature_names)
        
        # Save model artifacts
        artifacts_path = await self._save_model_artifacts(job.job_id, pipeline, config)
        
        # Log to MLflow
        await self._log_training_to_mlflow(job, pipeline, training_metrics, 
                                         validation_metrics, test_metrics, cv_scores)
        
        return TrainingResult(
            job_id=job.job_id,
            model_id=job.model_id,
            model_type=config.model_type,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            cross_validation_scores=cv_scores,
            feature_importance=feature_importance,
            best_parameters=pipeline.get_params(),
            model_artifacts_path=artifacts_path,
            training_duration=training_duration,
            data_shape=X_processed.shape,
            feature_names=feature_names
        )
    
    async def _execute_hyperparameter_tuning(self, job: TrainingJob, X: pd.DataFrame, 
                                           y: pd.Series) -> TrainingResult:
        """Execute hyperparameter tuning."""
        config = job.config
        
        # Feature engineering
        X_processed, feature_names = await self._engineer_features(X, y, config)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=config.test_split, random_state=config.random_state
        )
        
        # Create base model
        base_model = self._create_model(config.model_type)
        pipeline = self._create_pipeline(base_model, config)
        
        # Perform hyperparameter optimization
        if config.optimization_method == OptimizationMethod.BAYESIAN_OPTIMIZATION:
            best_params, best_score = await self._bayesian_optimization(
                pipeline, X_train, y_train, config
            )
        elif config.optimization_method == OptimizationMethod.GRID_SEARCH:
            best_params, best_score = await self._grid_search_optimization(
                pipeline, X_train, y_train, config
            )
        else:  # Random search
            best_params, best_score = await self._random_search_optimization(
                pipeline, X_train, y_train, config
            )
        
        # Train final model with best parameters
        final_pipeline = self._create_pipeline(
            self._create_model(config.model_type, best_params), config
        )
        
        start_time = datetime.now()
        final_pipeline.fit(X_train, y_train)
        training_duration = (datetime.now() - start_time).total_seconds()
        
        # Evaluate final model
        training_metrics = self._evaluate_model(final_pipeline, X_train, y_train, "training")
        test_metrics = self._evaluate_model(final_pipeline, X_test, y_test, "test")
        
        # Cross-validation with best parameters
        cv_scores = self._cross_validate_model(final_pipeline, X_processed, y, config)
        
        # Feature importance
        feature_importance = self._get_feature_importance(final_pipeline, feature_names)
        
        # Save model artifacts
        artifacts_path = await self._save_model_artifacts(job.job_id, final_pipeline, config)
        
        # Log to MLflow
        await self._log_training_to_mlflow(job, final_pipeline, training_metrics, 
                                         test_metrics, test_metrics, cv_scores, best_params)
        
        return TrainingResult(
            job_id=job.job_id,
            model_id=job.model_id,
            model_type=config.model_type,
            training_metrics=training_metrics,
            validation_metrics=test_metrics,  # Using test as validation for tuning
            test_metrics=test_metrics,
            cross_validation_scores=cv_scores,
            feature_importance=feature_importance,
            best_parameters=best_params,
            model_artifacts_path=artifacts_path,
            training_duration=training_duration,
            data_shape=X_processed.shape,
            feature_names=feature_names
        )
    
    async def _bayesian_optimization(self, pipeline: Pipeline, X: pd.DataFrame, 
                                   y: pd.Series, config: TrainingConfig) -> Tuple[Dict, float]:
        """Perform Bayesian optimization using Optuna."""
        
        def objective(trial):
            # Get hyperparameter space for model type
            param_space = self.hyperparameter_spaces.get(config.model_type, {})
            
            # Sample parameters
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values[0], int):
                    params[f'model__{param_name}'] = trial.suggest_int(
                        param_name, min(param_values), max(param_values)
                    )
                elif isinstance(param_values[0], float):
                    params[f'model__{param_name}'] = trial.suggest_float(
                        param_name, min(param_values), max(param_values)
                    )
                else:
                    params[f'model__{param_name}'] = trial.suggest_categorical(
                        param_name, param_values
                    )
            
            # Set parameters
            pipeline.set_params(**params)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=config.cross_validation_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                
                pipeline.fit(X_fold_train, y_fold_train)
                y_pred = pipeline.predict(X_fold_val)
                score = r2_score(y_fold_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=config.optimization_trials)
        
        # Get best parameters
        best_params = {}
        for key, value in study.best_params.items():
            best_params[key.replace('model__', '')] = value
        
        return best_params, study.best_value
    
    async def _grid_search_optimization(self, pipeline: Pipeline, X: pd.DataFrame, 
                                      y: pd.Series, config: TrainingConfig) -> Tuple[Dict, float]:
        """Perform grid search optimization."""
        param_space = self.hyperparameter_spaces.get(config.model_type, {})
        
        # Prefix parameters for pipeline
        param_grid = {f'model__{k}': v for k, v in param_space.items()}
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=config.cross_validation_folds)
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=tscv, scoring='r2', n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        # Extract best parameters (remove model__ prefix)
        best_params = {}
        for key, value in grid_search.best_params_.items():
            best_params[key.replace('model__', '')] = value
        
        return best_params, grid_search.best_score_
    
    async def _random_search_optimization(self, pipeline: Pipeline, X: pd.DataFrame, 
                                        y: pd.Series, config: TrainingConfig) -> Tuple[Dict, float]:
        """Perform random search optimization."""
        param_space = self.hyperparameter_spaces.get(config.model_type, {})
        
        # Prefix parameters for pipeline
        param_distributions = {f'model__{k}': v for k, v in param_space.items()}
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=config.cross_validation_folds)
        
        # Random search
        random_search = RandomizedSearchCV(
            pipeline, param_distributions, n_iter=config.optimization_trials,
            cv=tscv, scoring='r2', n_jobs=-1, random_state=config.random_state
        )
        
        random_search.fit(X, y)
        
        # Extract best parameters (remove model__ prefix)
        best_params = {}
        for key, value in random_search.best_params_.items():
            best_params[key.replace('model__', '')] = value
        
        return best_params, random_search.best_score_
    
    async def _engineer_features(self, X: pd.DataFrame, y: pd.Series, 
                               config: TrainingConfig) -> Tuple[pd.DataFrame, List[str]]:
        """Perform feature engineering and selection."""
        X_processed = X.copy()
        
        # Basic feature engineering for stock data
        if 'close' in X_processed.columns:
            # Price-based features
            X_processed['price_change'] = X_processed['close'].pct_change()
            X_processed['price_volatility'] = X_processed['close'].rolling(window=20).std()
            X_processed['price_momentum'] = X_processed['close'] / X_processed['close'].shift(10) - 1
            
        if 'volume' in X_processed.columns:
            # Volume-based features
            X_processed['volume_change'] = X_processed['volume'].pct_change()
            X_processed['volume_ma'] = X_processed['volume'].rolling(window=20).mean()
            
        # Technical indicators
        if 'high' in X_processed.columns and 'low' in X_processed.columns:
            X_processed['price_range'] = (X_processed['high'] - X_processed['low']) / X_processed['close']
        
        # Remove NaN values
        X_processed = X_processed.fillna(method='ffill').fillna(method='bfill')
        
        # Feature selection if specified
        if config.feature_selection_k and config.feature_selection_k < X_processed.shape[1]:
            selector = SelectKBest(score_func=f_regression, k=config.feature_selection_k)
            X_processed = pd.DataFrame(
                selector.fit_transform(X_processed, y),
                columns=X_processed.columns[selector.get_support()],
                index=X_processed.index
            )
        
        feature_names = X_processed.columns.tolist()
        
        return X_processed, feature_names
    
    def _create_model(self, model_type: ModelType, parameters: Dict[str, Any] = None) -> Any:
        """Create a model instance."""
        model_class = self.model_registry[model_type]
        
        if parameters:
            return model_class(**parameters)
        else:
            return model_class()
    
    def _create_pipeline(self, model: Any, config: TrainingConfig) -> Pipeline:
        """Create a preprocessing and modeling pipeline."""
        steps = []
        
        # Scaling
        if config.scaling_method == "standard":
            steps.append(('scaler', StandardScaler()))
        elif config.scaling_method == "robust":
            steps.append(('scaler', RobustScaler()))
        elif config.scaling_method == "minmax":
            steps.append(('scaler', MinMaxScaler()))
        
        # Model
        steps.append(('model', model))
        
        return Pipeline(steps)
    
    def _evaluate_model(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, 
                       dataset_name: str) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = pipeline.predict(X)
        
        return {
            f'{dataset_name}_mse': mean_squared_error(y, y_pred),
            f'{dataset_name}_mae': mean_absolute_error(y, y_pred),
            f'{dataset_name}_r2': r2_score(y, y_pred),
            f'{dataset_name}_rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
    
    def _cross_validate_model(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, 
                            config: TrainingConfig) -> List[float]:
        """Perform cross-validation."""
        tscv = TimeSeriesSplit(n_splits=config.cross_validation_folds)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            score = r2_score(y_val, y_pred)
            scores.append(score)
        
        return scores
    
    def _get_feature_importance(self, pipeline: Pipeline, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        model = pipeline.named_steps['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            return {}
        
        return dict(zip(feature_names, importances))
    
    async def _load_training_data(self, dataset_config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Load training data based on configuration."""
        # This would typically load from database or file system
        # For demo purposes, we'll generate synthetic data
        
        n_samples = dataset_config.get('n_samples', 1000)
        n_features = dataset_config.get('n_features', 10)
        
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add some stock-like columns
        X['close'] = 100 + np.cumsum(np.random.randn(n_samples) * 0.1)
        X['volume'] = np.random.lognormal(10, 1, n_samples)
        X['high'] = X['close'] * (1 + np.random.uniform(0, 0.05, n_samples))
        X['low'] = X['close'] * (1 - np.random.uniform(0, 0.05, n_samples))
        
        # Target variable (next day return)
        y = X['close'].pct_change().shift(-1).fillna(0)
        
        return X.iloc[:-1], y.iloc[:-1]  # Remove last row due to shift
    
    async def _save_model_artifacts(self, job_id: str, pipeline: Pipeline, 
                                  config: TrainingConfig) -> str:
        """Save model artifacts to disk."""
        artifacts_path = f"models/{job_id}"
        
        # Save pipeline
        joblib.dump(pipeline, f"{artifacts_path}/model.pkl")
        
        # Save configuration
        with open(f"{artifacts_path}/config.json", 'w') as f:
            json.dump(asdict(config), f, default=str)
        
        return artifacts_path
    
    async def _log_training_to_mlflow(self, job: TrainingJob, pipeline: Pipeline,
                                    training_metrics: Dict[str, float],
                                    validation_metrics: Dict[str, float],
                                    test_metrics: Dict[str, float],
                                    cv_scores: List[float],
                                    best_params: Dict[str, Any] = None):
        """Log training results to MLflow."""
        with mlflow.start_run(run_name=f"training_{job.job_id}"):
            # Log parameters
            mlflow.log_params({
                'job_id': job.job_id,
                'model_id': job.model_id,
                'model_type': job.config.model_type.value,
                'job_type': job.job_type.value
            })
            
            if best_params:
                mlflow.log_params(best_params)
            
            # Log metrics
            mlflow.log_metrics(training_metrics)
            mlflow.log_metrics(validation_metrics)
            mlflow.log_metrics(test_metrics)
            mlflow.log_metrics({
                'cv_mean_score': np.mean(cv_scores),
                'cv_std_score': np.std(cv_scores)
            })
            
            # Log model
            mlflow.sklearn.log_model(pipeline, "model")
    
    async def _store_training_job(self, job: TrainingJob):
        """Store training job in database."""
        try:
            query = text("""
                INSERT INTO model_training_jobs 
                (job_id, model_id, job_type, status, priority, training_config, 
                 dataset_config, created_at, triggered_by, parent_job_id)
                VALUES (:job_id, :model_id, :job_type, :status, :priority, 
                        :training_config, :dataset_config, :created_at, :triggered_by, :parent_job_id)
            """)
            
            with self.engine.connect() as conn:
                conn.execute(query, {
                    'job_id': job.job_id,
                    'model_id': job.model_id,
                    'job_type': job.job_type.value,
                    'status': job.status.value,
                    'priority': job.priority,
                    'training_config': json.dumps(asdict(job.config), default=str),
                    'dataset_config': json.dumps(job.dataset_config),
                    'created_at': job.created_at,
                    'triggered_by': job.triggered_by,
                    'parent_job_id': job.parent_job_id
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store training job: {str(e)}")
    
    async def _update_training_job_status(self, job_id: str, status: TrainingJobStatus):
        """Update training job status in database."""
        try:
            query = text("""
                UPDATE model_training_jobs 
                SET status = :status, updated_at = :updated_at
                WHERE job_id = :job_id
            """)
            
            with self.engine.connect() as conn:
                conn.execute(query, {
                    'job_id': job_id,
                    'status': status.value,
                    'updated_at': datetime.now()
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update training job status: {str(e)}")
    
    async def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job status."""
        # Check running jobs
        if job_id in self.running_jobs:
            return self.running_jobs[job_id]
        
        # Check queue
        for job in self.job_queue:
            if job.job_id == job_id:
                return job
        
        # Check database for completed jobs
        try:
            query = text("""
                SELECT * FROM model_training_jobs WHERE job_id = :job_id
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'job_id': job_id})
                row = result.fetchone()
                
                if row:
                    # Convert to TrainingJob object
                    config_dict = json.loads(row.training_config)
                    config = TrainingConfig(**config_dict)
                    
                    return TrainingJob(
                        job_id=row.job_id,
                        model_id=row.model_id,
                        job_type=TrainingJobType(row.job_type),
                        status=TrainingJobStatus(row.status),
                        config=config,
                        dataset_config=json.loads(row.dataset_config),
                        priority=row.priority,
                        created_at=row.created_at,
                        started_at=row.started_at,
                        completed_at=row.completed_at,
                        duration_seconds=row.duration_seconds,
                        final_metrics=json.loads(row.final_metrics) if row.final_metrics else None,
                        model_artifacts_path=row.model_artifacts_path,
                        error_message=row.error_message,
                        triggered_by=row.triggered_by,
                        parent_job_id=row.parent_job_id
                    )
                
        except Exception as e:
            logger.error(f"Failed to get job status: {str(e)}")
        
        return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        # Check if job is in queue
        for i, job in enumerate(self.job_queue):
            if job.job_id == job_id:
                job.status = TrainingJobStatus.CANCELLED
                self.job_queue.pop(i)
                await self._update_training_job_status(job_id, TrainingJobStatus.CANCELLED)
                logger.info(f"Cancelled queued job {job_id}")
                return True
        
        # Check if job is running (more complex to cancel)
        if job_id in self.running_jobs:
            # For now, just mark as cancelled in database
            # In a production system, you'd need to implement proper job cancellation
            await self._update_training_job_status(job_id, TrainingJobStatus.CANCELLED)
            logger.info(f"Marked running job {job_id} for cancellation")
            return True
        
        return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            'queued_jobs': len(self.job_queue),
            'running_jobs': len(self.running_jobs),
            'completed_jobs': len(self.completed_jobs),
            'queue_details': [
                {
                    'job_id': job.job_id,
                    'model_id': job.model_id,
                    'job_type': job.job_type.value,
                    'priority': job.priority,
                    'created_at': job.created_at.isoformat()
                }
                for job in self.job_queue
            ],
            'running_details': [
                {
                    'job_id': job.job_id,
                    'model_id': job.model_id,
                    'job_type': job.job_type.value,
                    'started_at': job.started_at.isoformat() if job.started_at else None
                }
                for job in self.running_jobs.values()
            ]
        }
    
    async def cleanup_old_jobs(self, retention_days: int = 30) -> int:
        """Clean up old completed training jobs."""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            query = text("""
                DELETE FROM model_training_jobs 
                WHERE status IN ('completed', 'failed', 'cancelled') 
                AND completed_at < :cutoff_date
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'cutoff_date': cutoff_date})
                deleted_count = result.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old training jobs")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old jobs: {str(e)}")
            return 0
    
    async def shutdown(self):
        """Shutdown the training pipeline gracefully."""
        logger.info("Shutting down automated training pipeline...")
        
        # Cancel all queued jobs
        for job in self.job_queue:
            job.status = TrainingJobStatus.CANCELLED
            await self._update_training_job_status(job.job_id, TrainingJobStatus.CANCELLED)
        
        # Wait for running jobs to complete (with timeout)
        timeout = 300  # 5 minutes
        start_time = datetime.now()
        
        while self.running_jobs and (datetime.now() - start_time).seconds < timeout:
            await asyncio.sleep(5)
            logger.info(f"Waiting for {len(self.running_jobs)} running jobs to complete...")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Automated training pipeline shutdown complete")


# Additional utility functions for specific training scenarios

async def _execute_retraining(self, job: TrainingJob, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
    """Execute model retraining with existing parameters."""
    # Similar to initial training but may use existing hyperparameters
    return await self._execute_initial_training(job, X, y)

async def _execute_feature_selection(self, job: TrainingJob, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
    """Execute feature selection optimization."""
    config = job.config
    best_features = None
    best_score = -np.inf
    
    # Try different numbers of features
    for k in range(5, min(50, X.shape[1]), 5):
        config.feature_selection_k = k
        
        # Quick evaluation with cross-validation
        X_processed, _ = await self._engineer_features(X, y, config)
        model = self._create_model(config.model_type)
        pipeline = self._create_pipeline(model, config)
        
        cv_scores = self._cross_validate_model(pipeline, X_processed, y, config)
        mean_score = np.mean(cv_scores)
        
        if mean_score > best_score:
            best_score = mean_score
            best_features = k
    
    # Train final model with best feature count
    config.feature_selection_k = best_features
    return await self._execute_initial_training(job, X, y)

async def _execute_model_comparison(self, job: TrainingJob, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
    """Execute comparison of multiple model types."""
    config = job.config
    model_results = {}
    
    # Test different model types
    model_types = [ModelType.LINEAR_REGRESSION, ModelType.RIDGE_REGRESSION, 
                   ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]
    
    for model_type in model_types:
        config.model_type = model_type
        
        # Quick evaluation
        X_processed, _ = await self._engineer_features(X, y, config)
        model = self._create_model(model_type)
        pipeline = self._create_pipeline(model, config)
        
        cv_scores = self._cross_validate_model(pipeline, X_processed, y, config)
        model_results[model_type.value] = np.mean(cv_scores)
    
    # Select best model type
    best_model_type = max(model_results, key=model_results.get)
    config.model_type = ModelType(best_model_type)
    
    # Train final model with best type
    return await self._execute_initial_training(job, X, y)