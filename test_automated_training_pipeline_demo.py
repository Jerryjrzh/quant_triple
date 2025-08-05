"""
Automated Training Pipeline Demo

This script demonstrates the comprehensive automated ML training capabilities including:
- Automated feature engineering and selection
- Hyperparameter optimization using Bayesian methods
- Model validation and cross-validation frameworks
- Automated model deployment and rollback capabilities
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_analysis_system.analysis.automated_training_pipeline import (
    AutomatedTrainingPipeline,
    AutomatedFeatureEngineer,
    BayesianHyperparameterOptimizer,
    FeatureEngineeringConfig,
    ModelConfig,
    TrainingConfig,
    DEFAULT_MODEL_CONFIGS,
    create_default_training_config
)
from stock_analysis_system.analysis.ml_model_manager import MLModelManager
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_stock_data(n_days: int = 500) -> pd.DataFrame:
    """Create realistic sample stock data for demonstration."""
    logger.info(f"Creating sample stock data with {n_days} days")
    
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Generate realistic stock price data with trends and volatility
    base_price = 100.0
    trend = 0.0003  # Small upward trend
    volatility = 0.025  # 2.5% daily volatility
    
    # Generate price series with some autocorrelation
    prices = [base_price]
    for i in range(1, n_days):
        # Add some mean reversion and momentum
        prev_change = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
        momentum = 0.1 * prev_change  # 10% momentum
        mean_reversion = -0.05 * (prices[-1] - base_price) / base_price  # Mean reversion
        
        change = np.random.normal(trend + momentum + mean_reversion, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Ensure positive prices
    
    # Create OHLCV data
    data = pd.DataFrame({
        'trade_date': dates,
        'stock_code': '000001',
        'open_price': prices,
        'high_price': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
        'low_price': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
        'close_price': prices,
        'volume': np.random.randint(500000, 5000000, n_days),
        'amount': [p * v * (1 + np.random.normal(0, 0.1)) for p, v in 
                  zip(prices, np.random.randint(500000, 5000000, n_days))]
    })
    
    # Create multiple target variables for different prediction tasks
    
    # 1. Next day direction (up/down)
    data['next_day_up'] = (data['close_price'].shift(-1) > data['close_price']).astype(int)
    
    # 2. Strong movement (>2% change)
    data['strong_movement'] = (
        abs(data['close_price'].pct_change().shift(-1)) > 0.02
    ).astype(int)
    
    # 3. Outperform market (assuming market returns 0.05% daily on average)
    market_return = 0.0005
    data['outperform_market'] = (
        data['close_price'].pct_change().shift(-1) > market_return
    ).astype(int)
    
    # Remove rows with NaN values
    data = data.dropna()
    
    logger.info(f"Created stock data: {len(data)} rows, {len(data.columns)} columns")
    return data


async def demonstrate_feature_engineering():
    """Demonstrate automated feature engineering capabilities."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING AUTOMATED FEATURE ENGINEERING")
    logger.info("="*60)
    
    # Create sample data
    data = create_sample_stock_data(200)
    logger.info(f"Original data shape: {data.shape}")
    logger.info(f"Original columns: {list(data.columns)}")
    
    # Configure feature engineering
    config = FeatureEngineeringConfig(
        technical_indicators=True,
        statistical_features=True,
        lag_features=True,
        rolling_features=True,
        interaction_features=True,
        polynomial_features=True,
        max_lag_periods=10,
        rolling_windows=[5, 10, 20],
        polynomial_degree=2
    )
    
    # Create feature engineer
    engineer = AutomatedFeatureEngineer(config)
    
    # Apply feature engineering
    start_time = datetime.now()
    engineered_data = await engineer.engineer_features(data)
    engineering_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Feature engineering completed in {engineering_time:.2f} seconds")
    logger.info(f"Engineered data shape: {engineered_data.shape}")
    logger.info(f"Features added: {engineered_data.shape[1] - data.shape[1]}")
    
    # Show some example features
    new_features = [col for col in engineered_data.columns if col not in data.columns]
    logger.info(f"Example new features: {new_features[:10]}")
    
    # Show feature categories
    technical_features = [col for col in new_features if any(
        indicator in col for indicator in ['sma', 'ema', 'rsi', 'macd', 'bb', 'stoch', 'adx', 'atr']
    )]
    lag_features = [col for col in new_features if '_lag_' in col]
    rolling_features = [col for col in new_features if '_rolling_' in col]
    interaction_features = [col for col in new_features if '_x_' in col]
    polynomial_features = [col for col in new_features if '_poly_' in col]
    
    logger.info(f"Technical indicators: {len(technical_features)}")
    logger.info(f"Lag features: {len(lag_features)}")
    logger.info(f"Rolling features: {len(rolling_features)}")
    logger.info(f"Interaction features: {len(interaction_features)}")
    logger.info(f"Polynomial features: {len(polynomial_features)}")
    
    return engineered_data


async def demonstrate_hyperparameter_optimization():
    """Demonstrate Bayesian hyperparameter optimization."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING HYPERPARAMETER OPTIMIZATION")
    logger.info("="*60)
    
    # Create sample data for optimization
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.randint(0, 2, 200)
    
    # Test different model configurations
    model_configs = [
        ModelConfig(
            model_class=RandomForestClassifier,
            param_space={
                'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
                'max_depth': {'type': 'int', 'low': 3, 'high': 15},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 5}
            },
            name="random_forest_optimized",
            requires_scaling=False
        ),
        ModelConfig(
            model_class=LogisticRegression,
            param_space={
                'C': {'type': 'real', 'low': 0.01, 'high': 100},
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            name="logistic_regression_optimized",
            requires_scaling=True
        )
    ]
    
    for model_config in model_configs:
        logger.info(f"\nOptimizing {model_config.name}...")
        
        optimizer = BayesianHyperparameterOptimizer(model_config, "f1_weighted")
        
        start_time = datetime.now()
        best_params, best_score = await optimizer.optimize(
            X, y, cv_folds=5, n_calls=20
        )
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")


async def demonstrate_full_training_pipeline():
    """Demonstrate the complete automated training pipeline."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING FULL AUTOMATED TRAINING PIPELINE")
    logger.info("="*60)
    
    # Create sample data
    data = create_sample_stock_data(400)
    
    # Initialize ML manager (mock for demo)
    try:
        ml_manager = MLModelManager("sqlite:///demo_mlflow.db")
    except Exception as e:
        logger.warning(f"Could not initialize MLflow: {e}. Using mock manager.")
        from unittest.mock import Mock, AsyncMock
        ml_manager = Mock()
        ml_manager.register_model = AsyncMock(return_value="demo_model_id")
        ml_manager.promote_model_to_production = AsyncMock(return_value=True)
        ml_manager.list_models = AsyncMock(return_value=[])
    
    # Create training pipeline
    pipeline = AutomatedTrainingPipeline(ml_manager)
    
    # Test different prediction tasks
    prediction_tasks = [
        {
            'target': 'next_day_up',
            'description': 'Predict next day price direction (up/down)'
        },
        {
            'target': 'strong_movement',
            'description': 'Predict strong price movements (>2%)'
        }
    ]
    
    for task in prediction_tasks:
        logger.info(f"\n--- Training models for: {task['description']} ---")
        
        # Create training configuration
        config = TrainingConfig(
            target_column=task['target'],
            feature_engineering=FeatureEngineeringConfig(
                technical_indicators=True,
                statistical_features=True,
                lag_features=True,
                rolling_features=True,
                interaction_features=False,  # Disable for speed
                polynomial_features=False,   # Disable for speed
                max_lag_periods=5,
                rolling_windows=[5, 10, 20]
            ),
            models=[
                ModelConfig(
                    model_class=RandomForestClassifier,
                    param_space={
                        'n_estimators': [50, 100, 150],
                        'max_depth': [5, 10, 15],
                        'min_samples_split': [2, 5, 10]
                    },
                    name=f"random_forest_{task['target']}",
                    requires_scaling=False
                ),
                ModelConfig(
                    model_class=GradientBoostingClassifier,
                    param_space={
                        'n_estimators': [50, 100],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    },
                    name=f"gradient_boosting_{task['target']}",
                    requires_scaling=False
                )
            ],
            cv_folds=5,
            test_size=0.2,
            validation_size=0.2,
            optimization_method="grid",  # Use grid for demo (faster than Bayesian)
            n_optimization_calls=10,
            scoring_metric="f1_weighted",
            feature_selection_k=20,
            auto_deploy=False  # Disable auto-deployment for demo
        )
        
        # Train models
        start_time = datetime.now()
        results = await pipeline.train_models(data, config)
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Trained {len(results)} models")
        
        # Display results
        for i, result in enumerate(results, 1):
            logger.info(f"\nModel {i}: {result.model_name}")
            logger.info(f"  Best CV Score: {result.best_score:.4f}")
            logger.info(f"  Best Parameters: {result.best_params}")
            logger.info(f"  Training Time: {result.training_time:.2f} seconds")
            logger.info(f"  Features Used: {len(result.feature_names)}")
            
            # Show validation metrics
            logger.info("  Validation Metrics:")
            for metric, value in result.validation_metrics.items():
                logger.info(f"    {metric}: {value:.4f}")
            
            # Show top features by importance
            if result.feature_importance:
                top_features = sorted(
                    result.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                logger.info("  Top 5 Important Features:")
                for feature, importance in top_features:
                    logger.info(f"    {feature}: {importance:.4f}")
        
        # Find best model
        if results:
            best_result = max(results, key=lambda x: x.best_score)
            logger.info(f"\nBest model for {task['description']}: {best_result.model_name}")
            logger.info(f"Best score: {best_result.best_score:.4f}")


async def demonstrate_model_comparison():
    """Demonstrate model comparison and selection."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING MODEL COMPARISON AND SELECTION")
    logger.info("="*60)
    
    # Create sample data
    data = create_sample_stock_data(300)
    
    # Mock ML manager for demo
    from unittest.mock import Mock, AsyncMock
    ml_manager = Mock()
    ml_manager.register_model = AsyncMock(return_value="demo_model_id")
    ml_manager.promote_model_to_production = AsyncMock(return_value=True)
    ml_manager.list_models = AsyncMock(return_value=[])
    
    pipeline = AutomatedTrainingPipeline(ml_manager)
    
    # Create configuration with multiple models
    config = await create_default_training_config('next_day_up')
    config.models = DEFAULT_MODEL_CONFIGS[:2]  # Use first 2 default models
    config.cv_folds = 3
    config.n_optimization_calls = 5
    config.feature_selection_k = 15
    
    # Train models
    results = await pipeline.train_models(data, config)
    
    # Compare models
    logger.info("Model Comparison Results:")
    logger.info("-" * 40)
    
    comparison_data = []
    for result in results:
        comparison_data.append({
            'Model': result.model_name,
            'CV Score': result.best_score,
            'Test Accuracy': result.validation_metrics.get('test_accuracy', 0),
            'Test F1': result.validation_metrics.get('test_f1', 0),
            'Training Time': result.training_time,
            'Features': len(result.feature_names)
        })
    
    # Sort by CV score
    comparison_data.sort(key=lambda x: x['CV Score'], reverse=True)
    
    # Display comparison table
    headers = ['Rank', 'Model', 'CV Score', 'Test Acc', 'Test F1', 'Time(s)', 'Features']
    logger.info(f"{headers[0]:<4} {headers[1]:<20} {headers[2]:<8} {headers[3]:<8} {headers[4]:<8} {headers[5]:<8} {headers[6]:<8}")
    logger.info("-" * 70)
    
    for i, model_data in enumerate(comparison_data, 1):
        logger.info(
            f"{i:<4} {model_data['Model']:<20} "
            f"{model_data['CV Score']:<8.4f} {model_data['Test Accuracy']:<8.4f} "
            f"{model_data['Test F1']:<8.4f} {model_data['Training Time']:<8.1f} "
            f"{model_data['Features']:<8}"
        )
    
    # Recommend best model
    best_model = comparison_data[0]
    logger.info(f"\nRecommended model: {best_model['Model']}")
    logger.info(f"Reason: Highest CV score ({best_model['CV Score']:.4f})")


async def demonstrate_deployment_simulation():
    """Demonstrate automated deployment simulation."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING AUTOMATED DEPLOYMENT SIMULATION")
    logger.info("="*60)
    
    # Create sample data
    data = create_sample_stock_data(200)
    
    # Mock ML manager with existing production model
    from unittest.mock import Mock, AsyncMock
    from stock_analysis_system.analysis.ml_model_manager import ModelInfo, ModelMetrics
    
    # Create mock existing production model
    existing_model = ModelInfo(
        model_id="existing_model_v1",
        model_name="existing_random_forest",
        version="1",
        status="production",
        created_at=datetime.now() - timedelta(days=30),
        last_updated=datetime.now() - timedelta(days=30),
        metrics=ModelMetrics(
            accuracy=0.65,
            precision=0.63,
            recall=0.67,
            f1_score=0.65,
            custom_metrics={'cv_score': 0.62}
        ),
        drift_score=0.05,
        tags={'version': 'v1'}
    )
    
    ml_manager = Mock()
    ml_manager.register_model = AsyncMock(return_value="new_model_id")
    ml_manager.promote_model_to_production = AsyncMock(return_value=True)
    ml_manager.list_models = AsyncMock(return_value=[existing_model])
    
    pipeline = AutomatedTrainingPipeline(ml_manager)
    
    # Test different deployment scenarios
    scenarios = [
        {
            'name': 'Significant Improvement',
            'rollback_threshold': 0.02,
            'description': 'New model significantly better than existing'
        },
        {
            'name': 'Marginal Improvement',
            'rollback_threshold': 0.05,
            'description': 'New model only marginally better'
        },
        {
            'name': 'Conservative Deployment',
            'rollback_threshold': 0.01,
            'description': 'Very conservative deployment threshold'
        }
    ]
    
    for scenario in scenarios:
        logger.info(f"\n--- Scenario: {scenario['name']} ---")
        logger.info(f"Description: {scenario['description']}")
        logger.info(f"Rollback threshold: {scenario['rollback_threshold']}")
        
        # Create configuration with auto-deployment enabled
        config = TrainingConfig(
            target_column='next_day_up',
            feature_engineering=FeatureEngineeringConfig(
                technical_indicators=True,
                statistical_features=True,
                lag_features=False,
                rolling_features=False,
                max_lag_periods=3,
                rolling_windows=[5, 10]
            ),
            models=[ModelConfig(
                model_class=RandomForestClassifier,
                param_space={'n_estimators': [50], 'max_depth': [5]},
                name="test_model",
                requires_scaling=False
            )],
            cv_folds=3,
            test_size=0.3,
            validation_size=0.3,
            optimization_method="grid",
            n_optimization_calls=1,
            scoring_metric="f1_weighted",
            feature_selection_k=10,
            auto_deploy=True,
            rollback_threshold=scenario['rollback_threshold']
        )
        
        # Reset mock calls
        ml_manager.register_model.reset_mock()
        ml_manager.promote_model_to_production.reset_mock()
        
        # Train and potentially deploy
        results = await pipeline.train_models(data, config)
        
        # Check deployment decision
        if results:
            new_score = results[0].best_score
            existing_score = existing_model.metrics.custom_metrics['cv_score']
            improvement = new_score - existing_score
            
            logger.info(f"Existing model CV score: {existing_score:.4f}")
            logger.info(f"New model CV score: {new_score:.4f}")
            logger.info(f"Performance improvement: {improvement:.4f}")
            
            if improvement >= scenario['rollback_threshold']:
                logger.info("✅ Model would be deployed (improvement above threshold)")
            else:
                logger.info("❌ Model would NOT be deployed (improvement below threshold)")
        
        # Verify mock calls
        assert ml_manager.register_model.called, "Model should always be registered"
        
        if ml_manager.promote_model_to_production.called:
            logger.info("🚀 Model was promoted to production")
        else:
            logger.info("⏸️  Model remained in staging")


async def main():
    """Run all demonstrations."""
    logger.info("Starting Automated Training Pipeline Demonstration")
    logger.info("=" * 80)
    
    try:
        # Run demonstrations
        await demonstrate_feature_engineering()
        await demonstrate_hyperparameter_optimization()
        await demonstrate_full_training_pipeline()
        await demonstrate_model_comparison()
        await demonstrate_deployment_simulation()
        
        logger.info("\n" + "="*80)
        logger.info("DEMONSTRATION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        logger.info("\nKey Features Demonstrated:")
        logger.info("✓ Automated feature engineering with technical indicators")
        logger.info("✓ Statistical and lag feature generation")
        logger.info("✓ Bayesian hyperparameter optimization")
        logger.info("✓ Cross-validation and model selection")
        logger.info("✓ Automated feature selection")
        logger.info("✓ Model comparison and ranking")
        logger.info("✓ Automated deployment with rollback protection")
        logger.info("✓ Comprehensive validation metrics")
        logger.info("✓ Feature importance analysis")
        
        logger.info("\nThe automated training pipeline provides:")
        logger.info("• End-to-end ML workflow automation")
        logger.info("• Intelligent feature engineering for stock data")
        logger.info("• Robust model validation and selection")
        logger.info("• Safe deployment with performance monitoring")
        logger.info("• Comprehensive logging and monitoring")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())