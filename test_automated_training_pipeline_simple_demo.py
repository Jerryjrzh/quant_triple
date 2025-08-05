"""
Simple Automated Training Pipeline Demo

This script demonstrates the core automated ML training capabilities with a simplified approach.
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
    FeatureEngineeringConfig,
    ModelConfig,
    TrainingConfig,
    create_default_training_config
)
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import Mock, AsyncMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_simple_stock_data(n_days: int = 100) -> pd.DataFrame:
    """Create simple sample stock data for demonstration."""
    logger.info(f"Creating simple stock data with {n_days} days")
    
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Generate simple price data
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, n_days):
        change = np.random.normal(0.001, 0.02)  # Small daily changes
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'trade_date': dates,
        'stock_code': '000001',
        'open_price': prices,
        'high_price': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low_price': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close_price': prices,
        'volume': np.random.randint(100000, 1000000, n_days),
    })
    
    # Create simple target: 1 if next day price goes up, 0 otherwise
    data['target'] = (data['close_price'].shift(-1) > data['close_price']).astype(int)
    
    # Remove last row with NaN target
    data = data.dropna()
    
    logger.info(f"Created stock data: {len(data)} rows, {len(data.columns)} columns")
    return data


async def demonstrate_simple_feature_engineering():
    """Demonstrate basic feature engineering."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING SIMPLE FEATURE ENGINEERING")
    logger.info("="*60)
    
    # Create sample data
    data = create_simple_stock_data(50)
    logger.info(f"Original data shape: {data.shape}")
    
    # Configure simple feature engineering
    config = FeatureEngineeringConfig(
        technical_indicators=True,
        statistical_features=True,
        lag_features=False,  # Disable to avoid NaN issues
        rolling_features=False,  # Disable to avoid NaN issues
        interaction_features=False,
        polynomial_features=False
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
    logger.info(f"New features: {new_features[:5]}...")
    
    return engineered_data


async def demonstrate_simple_training():
    """Demonstrate simple model training."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING SIMPLE MODEL TRAINING")
    logger.info("="*60)
    
    # Create sample data
    data = create_simple_stock_data(80)
    
    # Mock ML manager
    ml_manager = Mock()
    ml_manager.register_model = AsyncMock(return_value="demo_model_id")
    ml_manager.promote_model_to_production = AsyncMock(return_value=True)
    ml_manager.list_models = AsyncMock(return_value=[])
    
    # Create training pipeline
    pipeline = AutomatedTrainingPipeline(ml_manager)
    
    # Create simple training configuration
    config = TrainingConfig(
        target_column='target',
        feature_engineering=FeatureEngineeringConfig(
            technical_indicators=True,
            statistical_features=True,
            lag_features=False,
            rolling_features=False,
            interaction_features=False,
            polynomial_features=False
        ),
        models=[ModelConfig(
            model_class=RandomForestClassifier,
            param_space={
                'n_estimators': [10, 20],
                'max_depth': [3, 5]
            },
            name="simple_random_forest",
            requires_scaling=False
        )],
        cv_folds=3,
        test_size=0.3,
        validation_size=0.3,
        optimization_method="grid",
        n_optimization_calls=4,
        scoring_metric="accuracy",
        feature_selection_k=5,
        auto_deploy=False
    )
    
    # Train models
    start_time = datetime.now()
    results = await pipeline.train_models(data, config)
    training_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Trained {len(results)} models")
    
    # Display results
    if results:
        result = results[0]
        logger.info(f"\nModel: {result.model_name}")
        logger.info(f"Best CV Score: {result.best_score:.4f}")
        logger.info(f"Best Parameters: {result.best_params}")
        logger.info(f"Training Time: {result.training_time:.2f} seconds")
        logger.info(f"Features Used: {len(result.feature_names)}")
        
        # Show validation metrics
        logger.info("Validation Metrics:")
        for metric, value in result.validation_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Show feature importance
        if result.feature_importance:
            top_features = sorted(
                result.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            logger.info("Top 3 Important Features:")
            for feature, importance in top_features:
                logger.info(f"  {feature}: {importance:.4f}")
    else:
        logger.warning("No models were successfully trained")
    
    return results


async def demonstrate_feature_selection():
    """Demonstrate automated feature selection."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING AUTOMATED FEATURE SELECTION")
    logger.info("="*60)
    
    # Create sample data with many features
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    # Create random features
    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    X.columns = [f'feature_{i}' for i in range(n_features)]
    
    # Create target with some correlation to first few features
    y = pd.Series((X.iloc[:, 0] + X.iloc[:, 1] + np.random.randn(n_samples) * 0.5) > 0).astype(int)
    
    # Mock ML manager
    ml_manager = Mock()
    ml_manager.register_model = AsyncMock(return_value="demo_model_id")
    ml_manager.promote_model_to_production = AsyncMock(return_value=True)
    ml_manager.list_models = AsyncMock(return_value=[])
    
    pipeline = AutomatedTrainingPipeline(ml_manager)
    
    # Test feature selection
    X_train = X.iloc[:70]
    X_val = X.iloc[70:85]
    X_test = X.iloc[85:]
    y_train = y.iloc[:70]
    
    logger.info(f"Original features: {X_train.shape[1]}")
    
    X_train_sel, X_val_sel, X_test_sel, selected_features = await pipeline._select_features(
        X_train, X_val, X_test, y_train, k=5
    )
    
    logger.info(f"Selected features: {len(selected_features)}")
    logger.info(f"Selected feature names: {selected_features}")
    
    # Verify shapes
    assert X_train_sel.shape[1] == len(selected_features)
    assert X_val_sel.shape[1] == len(selected_features)
    assert X_test_sel.shape[1] == len(selected_features)
    
    logger.info("✓ Feature selection completed successfully")


async def demonstrate_data_splitting():
    """Demonstrate time-series aware data splitting."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING TIME-SERIES DATA SPLITTING")
    logger.info("="*60)
    
    # Create sample data
    data = create_simple_stock_data(100)
    
    # Mock ML manager
    ml_manager = Mock()
    pipeline = AutomatedTrainingPipeline(ml_manager)
    
    # Prepare data
    X, y = await pipeline._prepare_data(data, 'target')
    
    logger.info(f"Total samples: {len(X)}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = await pipeline._split_data(
        X, y, test_size=0.2, validation_size=0.2
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Verify no overlap (time-series splitting)
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)
    
    assert len(train_indices & val_indices) == 0, "Training and validation sets overlap"
    assert len(train_indices & test_indices) == 0, "Training and test sets overlap"
    assert len(val_indices & test_indices) == 0, "Validation and test sets overlap"
    
    # Verify chronological order (for time series)
    assert max(X_train.index) < min(X_val.index), "Training set should come before validation"
    assert max(X_val.index) < min(X_test.index), "Validation set should come before test"
    
    logger.info("✓ Time-series data splitting completed successfully")


async def demonstrate_configuration_creation():
    """Demonstrate configuration creation utilities."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING CONFIGURATION CREATION")
    logger.info("="*60)
    
    # Create default configuration
    config = await create_default_training_config('target')
    
    logger.info(f"Target column: {config.target_column}")
    logger.info(f"Number of models: {len(config.models)}")
    logger.info(f"CV folds: {config.cv_folds}")
    logger.info(f"Test size: {config.test_size}")
    logger.info(f"Validation size: {config.validation_size}")
    logger.info(f"Optimization method: {config.optimization_method}")
    logger.info(f"Feature selection k: {config.feature_selection_k}")
    
    # Show model configurations
    logger.info("Model configurations:")
    for i, model_config in enumerate(config.models):
        logger.info(f"  {i+1}. {model_config.name}")
        logger.info(f"     Requires scaling: {model_config.requires_scaling}")
        logger.info(f"     Parameters: {list(model_config.param_space.keys())}")
    
    # Show feature engineering configuration
    fe_config = config.feature_engineering
    logger.info("Feature engineering configuration:")
    logger.info(f"  Technical indicators: {fe_config.technical_indicators}")
    logger.info(f"  Statistical features: {fe_config.statistical_features}")
    logger.info(f"  Lag features: {fe_config.lag_features}")
    logger.info(f"  Rolling features: {fe_config.rolling_features}")
    logger.info(f"  Max lag periods: {fe_config.max_lag_periods}")
    logger.info(f"  Rolling windows: {fe_config.rolling_windows}")
    
    logger.info("✓ Configuration creation completed successfully")


async def main():
    """Run all simple demonstrations."""
    logger.info("Starting Simple Automated Training Pipeline Demonstration")
    logger.info("=" * 80)
    
    try:
        # Run demonstrations
        await demonstrate_simple_feature_engineering()
        await demonstrate_feature_selection()
        await demonstrate_data_splitting()
        await demonstrate_configuration_creation()
        await demonstrate_simple_training()
        
        logger.info("\n" + "="*80)
        logger.info("SIMPLE DEMONSTRATION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        logger.info("\nKey Features Demonstrated:")
        logger.info("✓ Basic automated feature engineering")
        logger.info("✓ Automated feature selection")
        logger.info("✓ Time-series aware data splitting")
        logger.info("✓ Configuration management")
        logger.info("✓ Model training with hyperparameter optimization")
        logger.info("✓ Model validation and metrics calculation")
        logger.info("✓ Feature importance analysis")
        
        logger.info("\nThe automated training pipeline provides:")
        logger.info("• End-to-end ML workflow automation")
        logger.info("• Intelligent feature engineering for stock data")
        logger.info("• Robust model validation and selection")
        logger.info("• Comprehensive logging and error handling")
        logger.info("• Flexible configuration system")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())