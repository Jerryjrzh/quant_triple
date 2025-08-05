"""
Tests for Automated Training Pipeline

This module tests the comprehensive automated ML training capabilities including:
- Automated feature engineering and selection
- Hyperparameter optimization using Bayesian methods
- Model validation and cross-validation frameworks
- Automated model deployment and rollback capabilities
"""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from stock_analysis_system.analysis.automated_training_pipeline import (
    DEFAULT_MODEL_CONFIGS,
    AutomatedFeatureEngineer,
    AutomatedTrainingPipeline,
    BayesianHyperparameterOptimizer,
    FeatureEngineeringConfig,
    ModelConfig,
    TrainingConfig,
    TrainingResult,
    create_default_training_config,
)
from stock_analysis_system.analysis.ml_model_manager import MLModelManager, ModelMetrics


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    np.random.seed(42)

    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
    n_days = len(dates)

    # Generate realistic stock price data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Create OHLCV data
    data = pd.DataFrame(
        {
            "trade_date": dates,
            "open_price": prices,
            "high_price": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "low_price": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "close_price": prices,
            "volume": np.random.randint(1000000, 10000000, n_days),
            "stock_code": ["000001"] * n_days,
        }
    )

    # Create a simple target variable (1 if next day price goes up, 0 otherwise)
    data["target"] = (data["close_price"].shift(-1) > data["close_price"]).astype(int)
    data = data.dropna()

    return data


@pytest.fixture
def feature_engineering_config():
    """Create feature engineering configuration for testing."""
    return FeatureEngineeringConfig(
        technical_indicators=True,
        statistical_features=True,
        lag_features=True,
        rolling_features=True,
        interaction_features=False,  # Disable to speed up tests
        polynomial_features=False,  # Disable to speed up tests
        max_lag_periods=5,
        rolling_windows=[5, 10],
    )


@pytest.fixture
def simple_model_config():
    """Create a simple model configuration for testing."""
    return ModelConfig(
        model_class=RandomForestClassifier,
        param_space={"n_estimators": [10, 20], "max_depth": [3, 5]},
        name="test_random_forest",
        requires_scaling=False,
    )


@pytest.fixture
def training_config(feature_engineering_config, simple_model_config):
    """Create training configuration for testing."""
    return TrainingConfig(
        target_column="target",
        feature_engineering=feature_engineering_config,
        models=[simple_model_config],
        cv_folds=3,
        test_size=0.2,
        validation_size=0.2,
        optimization_method="grid",  # Use grid search for faster testing
        n_optimization_calls=4,
        scoring_metric="accuracy",
        feature_selection_k=10,
        auto_deploy=False,
    )


@pytest.fixture
async def mock_ml_manager():
    """Create a mock ML manager for testing."""
    manager = Mock(spec=MLModelManager)
    manager.register_model = AsyncMock(return_value="test_model_id")
    manager.promote_model_to_production = AsyncMock(return_value=True)
    manager.list_models = AsyncMock(return_value=[])
    return manager


class TestAutomatedFeatureEngineer:
    """Test cases for AutomatedFeatureEngineer."""

    @pytest.mark.asyncio
    async def test_feature_engineering_basic(
        self, sample_stock_data, feature_engineering_config
    ):
        """Test basic feature engineering functionality."""
        engineer = AutomatedFeatureEngineer(feature_engineering_config)

        # Use a smaller subset for faster testing
        test_data = sample_stock_data.head(100).copy()

        result = await engineer.engineer_features(test_data)

        # Check that features were added
        assert len(result.columns) > len(test_data.columns)

        # Check for specific technical indicators
        expected_indicators = ["sma_5", "sma_10", "rsi_14", "macd", "bb_upper"]
        for indicator in expected_indicators:
            assert indicator in result.columns, f"Missing indicator: {indicator}"

        # Check for lag features
        lag_features = [col for col in result.columns if "_lag_" in col]
        assert len(lag_features) > 0, "No lag features found"

        # Check for rolling features
        rolling_features = [col for col in result.columns if "_rolling_" in col]
        assert len(rolling_features) > 0, "No rolling features found"

    @pytest.mark.asyncio
    async def test_technical_indicators(
        self, sample_stock_data, feature_engineering_config
    ):
        """Test technical indicators generation."""
        engineer = AutomatedFeatureEngineer(feature_engineering_config)
        test_data = sample_stock_data.head(100).copy()

        result = await engineer._add_technical_indicators(test_data)

        # Check specific indicators
        technical_indicators = [
            "sma_5",
            "sma_10",
            "sma_20",
            "ema_12",
            "ema_26",
            "macd",
            "macd_signal",
            "rsi_14",
            "bb_upper",
            "bb_lower",
            "stoch_k",
            "adx",
            "atr",
        ]

        for indicator in technical_indicators:
            assert (
                indicator in result.columns
            ), f"Missing technical indicator: {indicator}"
            # Check that indicator has reasonable values (not all NaN)
            assert (
                not result[indicator].isna().all()
            ), f"Indicator {indicator} is all NaN"

    @pytest.mark.asyncio
    async def test_statistical_features(
        self, sample_stock_data, feature_engineering_config
    ):
        """Test statistical features generation."""
        engineer = AutomatedFeatureEngineer(feature_engineering_config)
        test_data = sample_stock_data.head(100).copy()

        result = await engineer._add_statistical_features(test_data)

        # Check specific statistical features
        stat_features = [
            "price_range",
            "price_change",
            "price_change_pct",
            "volume_price_trend",
            "high_low_pct",
        ]

        for feature in stat_features:
            assert feature in result.columns, f"Missing statistical feature: {feature}"

    @pytest.mark.asyncio
    async def test_lag_features(self, sample_stock_data, feature_engineering_config):
        """Test lag features generation."""
        engineer = AutomatedFeatureEngineer(feature_engineering_config)
        test_data = sample_stock_data.head(100).copy()

        # Add a simple feature first
        test_data["test_feature"] = test_data["close_price"]

        result = await engineer._add_lag_features(test_data)

        # Check that lag features were created
        lag_features = [col for col in result.columns if "_lag_" in col]
        assert len(lag_features) > 0, "No lag features created"

        # Check specific lag feature
        if "close_price_lag_1" in result.columns:
            # Verify lag is correct (ignoring NaN values)
            valid_indices = ~result["close_price_lag_1"].isna()
            if valid_indices.sum() > 1:
                original_values = test_data["close_price"].iloc[1:].values
                lagged_values = result["close_price_lag_1"].iloc[1:].values
                lagged_values = lagged_values[~np.isnan(lagged_values)]
                if len(lagged_values) > 0 and len(original_values) > 0:
                    # Check first few values
                    np.testing.assert_array_almost_equal(
                        original_values[: len(lagged_values) - 1],
                        lagged_values[1 : len(original_values)],
                        decimal=2,
                    )

    @pytest.mark.asyncio
    async def test_missing_columns_error(self, feature_engineering_config):
        """Test error handling for missing required columns."""
        engineer = AutomatedFeatureEngineer(feature_engineering_config)

        # Create data missing required columns
        incomplete_data = pd.DataFrame(
            {
                "close_price": [100, 101, 102],
                "volume": [1000, 1100, 1200],
                # Missing open_price, high_price, low_price
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            await engineer.engineer_features(incomplete_data)


class TestBayesianHyperparameterOptimizer:
    """Test cases for BayesianHyperparameterOptimizer."""

    @pytest.mark.asyncio
    async def test_grid_search_optimization(self, simple_model_config):
        """Test hyperparameter optimization using grid search."""
        optimizer = BayesianHyperparameterOptimizer(simple_model_config, "accuracy")

        # Create simple test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        best_params, best_score = await optimizer.optimize(X, y, cv_folds=3, n_calls=4)

        # Check that we got valid results
        assert isinstance(best_params, dict)
        assert isinstance(best_score, float)
        assert best_score > 0  # Should be positive for accuracy

        # Check that parameters are from the specified space
        assert best_params["n_estimators"] in [10, 20]
        assert best_params["max_depth"] in [3, 5]

    @pytest.mark.asyncio
    async def test_optimization_with_invalid_data(self, simple_model_config):
        """Test optimization with invalid data."""
        optimizer = BayesianHyperparameterOptimizer(simple_model_config, "accuracy")

        # Create data that will cause model fitting to fail
        X = np.full((10, 5), np.nan)
        y = np.random.randint(0, 2, 10)

        # Should handle errors gracefully
        best_params, best_score = await optimizer.optimize(X, y, cv_folds=3, n_calls=4)

        # Should return some result even with problematic data
        assert isinstance(best_params, dict)
        assert isinstance(best_score, float)


class TestAutomatedTrainingPipeline:
    """Test cases for AutomatedTrainingPipeline."""

    @pytest.mark.asyncio
    async def test_full_training_pipeline(
        self, sample_stock_data, training_config, mock_ml_manager
    ):
        """Test the complete training pipeline."""
        pipeline = AutomatedTrainingPipeline(mock_ml_manager)

        # Use smaller dataset for faster testing
        test_data = sample_stock_data.head(200).copy()

        results = await pipeline.train_models(test_data, training_config)

        # Check that we got results
        assert len(results) > 0, "No training results returned"

        # Check result structure
        result = results[0]
        assert isinstance(result, TrainingResult)
        assert result.model_id is not None
        assert result.model_name == "test_random_forest"
        assert isinstance(result.best_score, float)
        assert isinstance(result.best_params, dict)
        assert len(result.feature_names) > 0
        assert result.model_object is not None

    @pytest.mark.asyncio
    async def test_data_preparation(self, sample_stock_data, mock_ml_manager):
        """Test data preparation functionality."""
        pipeline = AutomatedTrainingPipeline(mock_ml_manager)

        test_data = sample_stock_data.head(100).copy()

        X, y = await pipeline._prepare_data(test_data, "target")

        # Check that target was separated correctly
        assert "target" not in X.columns
        assert len(y) == len(test_data)
        assert len(X) == len(test_data)

        # Check that only numeric columns are kept
        assert all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number)))

    @pytest.mark.asyncio
    async def test_data_splitting(self, sample_stock_data, mock_ml_manager):
        """Test data splitting functionality."""
        pipeline = AutomatedTrainingPipeline(mock_ml_manager)

        test_data = sample_stock_data.head(100).copy()
        X, y = await pipeline._prepare_data(test_data, "target")

        X_train, X_val, X_test, y_train, y_val, y_test = await pipeline._split_data(
            X, y, test_size=0.2, validation_size=0.2
        )

        # Check split sizes
        total_samples = len(X)
        expected_test_size = int(total_samples * 0.2)
        expected_val_size = int((total_samples - expected_test_size) * 0.2)
        expected_train_size = total_samples - expected_test_size - expected_val_size

        assert len(X_test) == expected_test_size
        assert len(X_val) == expected_val_size
        assert len(X_train) == expected_train_size

        # Check that splits don't overlap
        assert len(set(X_train.index) & set(X_val.index)) == 0
        assert len(set(X_train.index) & set(X_test.index)) == 0
        assert len(set(X_val.index) & set(X_test.index)) == 0

    @pytest.mark.asyncio
    async def test_feature_selection(self, sample_stock_data, mock_ml_manager):
        """Test feature selection functionality."""
        pipeline = AutomatedTrainingPipeline(mock_ml_manager)

        # Create data with many features
        np.random.seed(42)
        n_samples = 100
        n_features = 50

        X_train = pd.DataFrame(np.random.randn(n_samples, n_features))
        X_val = pd.DataFrame(np.random.randn(20, n_features))
        X_test = pd.DataFrame(np.random.randn(20, n_features))
        y_train = pd.Series(np.random.randint(0, 2, n_samples))

        k = 10
        X_train_sel, X_val_sel, X_test_sel, selected_features = (
            await pipeline._select_features(X_train, X_val, X_test, y_train, k)
        )

        # Check that feature selection worked
        assert len(selected_features) <= k
        assert X_train_sel.shape[1] == len(selected_features)
        assert X_val_sel.shape[1] == len(selected_features)
        assert X_test_sel.shape[1] == len(selected_features)

        # Check that selected features are from original features
        assert all(feature in X_train.columns for feature in selected_features)

    @pytest.mark.asyncio
    async def test_single_model_training(
        self, sample_stock_data, simple_model_config, training_config, mock_ml_manager
    ):
        """Test training of a single model."""
        pipeline = AutomatedTrainingPipeline(mock_ml_manager)

        # Prepare simple test data
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        X_train = pd.DataFrame(np.random.randn(n_samples, n_features))
        X_val = pd.DataFrame(np.random.randn(20, n_features))
        X_test = pd.DataFrame(np.random.randn(20, n_features))
        y_train = pd.Series(np.random.randint(0, 2, n_samples))
        y_val = pd.Series(np.random.randint(0, 2, 20))
        y_test = pd.Series(np.random.randint(0, 2, 20))

        feature_names = [f"feature_{i}" for i in range(n_features)]

        result = await pipeline._train_single_model(
            simple_model_config,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            feature_names,
            training_config,
        )

        # Check result structure
        assert isinstance(result, TrainingResult)
        assert result.model_name == "test_random_forest"
        assert isinstance(result.best_score, float)
        assert isinstance(result.best_params, dict)
        assert len(result.cv_scores) == training_config.cv_folds
        assert len(result.feature_names) == n_features
        assert result.model_object is not None

        # Check validation metrics
        required_metrics = [
            "val_accuracy",
            "val_precision",
            "val_recall",
            "val_f1",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
        ]
        for metric in required_metrics:
            assert metric in result.validation_metrics
            assert isinstance(result.validation_metrics[metric], float)

    @pytest.mark.asyncio
    async def test_auto_deployment(
        self, sample_stock_data, training_config, mock_ml_manager
    ):
        """Test automated model deployment."""
        # Enable auto deployment
        training_config.auto_deploy = True

        pipeline = AutomatedTrainingPipeline(mock_ml_manager)

        # Use smaller dataset
        test_data = sample_stock_data.head(100).copy()

        results = await pipeline.train_models(test_data, training_config)

        # Check that model was registered
        mock_ml_manager.register_model.assert_called_once()

        # Check that model was promoted (since no existing production models)
        mock_ml_manager.promote_model_to_production.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_target_column(
        self, sample_stock_data, training_config, mock_ml_manager
    ):
        """Test error handling for invalid target column."""
        training_config.target_column = "nonexistent_column"

        pipeline = AutomatedTrainingPipeline(mock_ml_manager)

        with pytest.raises(
            ValueError, match="Target column 'nonexistent_column' not found"
        ):
            await pipeline.train_models(sample_stock_data.head(50), training_config)


class TestDefaultConfigurations:
    """Test cases for default configurations."""

    def test_default_model_configs(self):
        """Test that default model configurations are valid."""
        assert len(DEFAULT_MODEL_CONFIGS) > 0

        for config in DEFAULT_MODEL_CONFIGS:
            assert isinstance(config, ModelConfig)
            assert config.model_class is not None
            assert isinstance(config.param_space, dict)
            assert isinstance(config.name, str)
            assert isinstance(config.requires_scaling, bool)

    @pytest.mark.asyncio
    async def test_create_default_training_config(self):
        """Test creation of default training configuration."""
        config = await create_default_training_config("target")

        assert isinstance(config, TrainingConfig)
        assert config.target_column == "target"
        assert isinstance(config.feature_engineering, FeatureEngineeringConfig)
        assert len(config.models) > 0
        assert config.cv_folds > 0
        assert 0 < config.test_size < 1
        assert 0 < config.validation_size < 1


class TestIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, sample_stock_data, mock_ml_manager):
        """Test the complete end-to-end pipeline."""
        # Create a minimal configuration for faster testing
        config = TrainingConfig(
            target_column="target",
            feature_engineering=FeatureEngineeringConfig(
                technical_indicators=True,
                statistical_features=True,
                lag_features=False,  # Disable for speed
                rolling_features=False,  # Disable for speed
                interaction_features=False,
                polynomial_features=False,
            ),
            models=[
                ModelConfig(
                    model_class=RandomForestClassifier,
                    param_space={"n_estimators": [10], "max_depth": [3]},
                    name="minimal_rf",
                    requires_scaling=False,
                )
            ],
            cv_folds=2,
            test_size=0.3,
            validation_size=0.3,
            optimization_method="grid",
            n_optimization_calls=1,
            scoring_metric="accuracy",
            feature_selection_k=5,
            auto_deploy=False,
        )

        pipeline = AutomatedTrainingPipeline(mock_ml_manager)

        # Use very small dataset for speed
        test_data = sample_stock_data.head(50).copy()

        results = await pipeline.train_models(test_data, config)

        # Verify we got valid results
        assert len(results) == 1
        result = results[0]

        assert result.model_name == "minimal_rf"
        assert isinstance(result.best_score, float)
        assert result.model_object is not None
        assert len(result.feature_names) > 0
        assert len(result.validation_metrics) > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_real_data_structure(self, mock_ml_manager):
        """Test pipeline with realistic stock data structure."""
        # Create more realistic stock data
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", end="2023-06-30", freq="D")
        n_days = len(dates)

        # Simulate stock price with trend and volatility
        base_price = 50.0
        trend = 0.0002  # Small upward trend
        volatility = 0.02

        prices = [base_price]
        for i in range(1, n_days):
            change = np.random.normal(trend, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Ensure positive prices

        data = pd.DataFrame(
            {
                "trade_date": dates,
                "stock_code": "000001",
                "open_price": prices,
                "high_price": [
                    p * (1 + abs(np.random.normal(0, 0.005))) for p in prices
                ],
                "low_price": [
                    p * (1 - abs(np.random.normal(0, 0.005))) for p in prices
                ],
                "close_price": prices,
                "volume": np.random.randint(100000, 1000000, n_days),
                "amount": [
                    p * v
                    for p, v in zip(prices, np.random.randint(100000, 1000000, n_days))
                ],
            }
        )

        # Create target: 1 if price goes up next day, 0 otherwise
        data["target"] = (data["close_price"].shift(-1) > data["close_price"]).astype(
            int
        )
        data = data.dropna()

        # Simple configuration
        config = await create_default_training_config("target")
        config.models = [DEFAULT_MODEL_CONFIGS[0]]  # Use only Random Forest
        config.cv_folds = 2
        config.n_optimization_calls = 5
        config.feature_selection_k = 15

        pipeline = AutomatedTrainingPipeline(mock_ml_manager)

        results = await pipeline.train_models(data, config)

        assert len(results) == 1
        result = results[0]

        # Verify the model was trained successfully
        assert result.model_object is not None
        assert len(result.feature_names) > 0
        assert result.best_score > 0

        # Verify feature engineering worked
        assert len(result.feature_names) <= config.feature_selection_k

        # Verify validation metrics are reasonable
        for metric_name, metric_value in result.validation_metrics.items():
            assert (
                0 <= metric_value <= 1
            ), f"Metric {metric_name} has invalid value: {metric_value}"


if __name__ == "__main__":
    pytest.main([__file__])
