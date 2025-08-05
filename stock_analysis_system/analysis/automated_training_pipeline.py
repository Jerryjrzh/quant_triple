"""
Automated Model Training Pipeline

This module provides comprehensive automated ML training capabilities including:
- Automated feature engineering and selection
- Hyperparameter optimization using Bayesian methods
- Model validation and cross-validation frameworks
- Automated model deployment and rollback capabilities
"""

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    ParameterGrid,
    ParameterSampler,
    TimeSeriesSplit,
    cross_val_score,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC

# Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Categorical, Integer, Real
    from skopt.utils import use_named_args

    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    logging.warning("scikit-optimize not available. Falling back to grid search.")

# Feature engineering
try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Technical indicators will be limited.")

from scipy import stats

from config.settings import get_settings

from .ml_model_manager import MLModelManager, ModelMetrics

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringConfig:
    """Configuration for automated feature engineering."""

    technical_indicators: bool = True
    statistical_features: bool = True
    lag_features: bool = True
    rolling_features: bool = True
    interaction_features: bool = False
    polynomial_features: bool = False
    max_lag_periods: int = 20
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    polynomial_degree: int = 2
    interaction_threshold: float = 0.1


@dataclass
class ModelConfig:
    """Configuration for a single model type."""

    model_class: Any
    param_space: Dict[str, Any]
    name: str
    requires_scaling: bool = True


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""

    target_column: str
    feature_engineering: FeatureEngineeringConfig
    models: List[ModelConfig]
    cv_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.2
    optimization_method: str = "bayesian"  # "bayesian", "grid", "random"
    n_optimization_calls: int = 50
    scoring_metric: str = "f1_weighted"
    early_stopping_rounds: int = 10
    feature_selection_k: int = 50
    auto_deploy: bool = False
    rollback_threshold: float = 0.05  # Performance degradation threshold


@dataclass
class TrainingResult:
    """Result of the training pipeline."""

    model_id: str
    model_name: str
    best_score: float
    best_params: Dict[str, Any]
    cv_scores: List[float]
    feature_importance: Dict[str, float]
    training_time: float
    validation_metrics: Dict[str, float]
    model_object: Any
    feature_names: List[str]
    preprocessing_pipeline: Any


class AutomatedFeatureEngineer:
    """Automated feature engineering for stock market data."""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    async def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply automated feature engineering to the dataset.

        Args:
            data: Input DataFrame with OHLCV data

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting automated feature engineering")

        # Make a copy to avoid modifying original data
        df = data.copy()

        # Ensure required columns exist
        required_cols = [
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Technical indicators
        if self.config.technical_indicators:
            df = await self._add_technical_indicators(df)

        # Statistical features
        if self.config.statistical_features:
            df = await self._add_statistical_features(df)

        # Lag features
        if self.config.lag_features:
            df = await self._add_lag_features(df)

        # Rolling features
        if self.config.rolling_features:
            df = await self._add_rolling_features(df)

        # Interaction features
        if self.config.interaction_features:
            df = await self._add_interaction_features(df)

        # Polynomial features
        if self.config.polynomial_features:
            df = await self._add_polynomial_features(df)

        # Remove rows with NaN values (from lag/rolling features)
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)

        logger.info(
            f"Feature engineering completed. Rows: {initial_rows} -> {final_rows}"
        )
        logger.info(f"Features: {len(df.columns)} total")

        return df

    async def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using TA-Lib or manual calculations."""
        logger.info("Adding technical indicators")

        high = df["high_price"].values
        low = df["low_price"].values
        close = df["close_price"].values
        volume = df["volume"].values

        if TALIB_AVAILABLE:
            # Use TA-Lib for better performance and accuracy
            df["sma_5"] = talib.SMA(close, timeperiod=5)
            df["sma_10"] = talib.SMA(close, timeperiod=10)
            df["sma_20"] = talib.SMA(close, timeperiod=20)
            df["ema_12"] = talib.EMA(close, timeperiod=12)
            df["ema_26"] = talib.EMA(close, timeperiod=26)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            df["macd"] = macd
            df["macd_signal"] = macd_signal
            df["macd_histogram"] = macd_hist

            # RSI
            df["rsi_14"] = talib.RSI(close, timeperiod=14)

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            df["bb_upper"] = bb_upper
            df["bb_middle"] = bb_middle
            df["bb_lower"] = bb_lower
            df["bb_width"] = (bb_upper - bb_lower) / bb_middle
            df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)

            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            df["stoch_k"] = slowk
            df["stoch_d"] = slowd

            # ADX
            df["adx"] = talib.ADX(high, low, close, timeperiod=14)

            # Volume indicators
            df["ad"] = talib.AD(high, low, close, volume)
            df["obv"] = talib.OBV(close, volume)

            # Volatility indicators
            df["atr"] = talib.ATR(high, low, close, timeperiod=14)
            df["natr"] = talib.NATR(high, low, close, timeperiod=14)
        else:
            # Manual calculations as fallback
            df = await self._add_manual_technical_indicators(df)

        return df

    async def _add_manual_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using manual calculations."""
        logger.info("Using manual technical indicator calculations")

        close = df["close_price"]
        high = df["high_price"]
        low = df["low_price"]
        volume = df["volume"]

        # Simple Moving Averages
        df["sma_5"] = close.rolling(window=5).mean()
        df["sma_10"] = close.rolling(window=10).mean()
        df["sma_20"] = close.rolling(window=20).mean()

        # Exponential Moving Averages
        df["ema_12"] = close.ewm(span=12).mean()
        df["ema_26"] = close.ewm(span=26).mean()

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # RSI (simplified calculation)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["bb_middle"] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # Stochastic (simplified)
        lowest_low = low.rolling(window=14).min()
        highest_high = high.rolling(window=14).max()
        df["stoch_k"] = 100 * (close - lowest_low) / (highest_high - lowest_low)
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

        # Average True Range (ATR)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=14).mean()
        df["natr"] = df["atr"] / close * 100

        # On Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv

        # Accumulation/Distribution Line (simplified)
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_volume = money_flow_multiplier * volume
        df["ad"] = money_flow_volume.cumsum()

        # ADX (simplified directional movement)
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)

        plus_di = 100 * (plus_dm.rolling(window=14).mean() / df["atr"])
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / df["atr"])
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df["adx"] = dx.rolling(window=14).mean()

        return df

    async def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        logger.info("Adding statistical features")

        # Price-based statistical features
        df["price_range"] = df["high_price"] - df["low_price"]
        df["price_change"] = df["close_price"] - df["open_price"]
        df["price_change_pct"] = df["price_change"] / df["open_price"]

        # Volume-based features
        df["volume_price_trend"] = df["volume"] * df["price_change_pct"]
        df["volume_sma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

        # Volatility features
        df["high_low_pct"] = (df["high_price"] - df["low_price"]) / df["close_price"]
        df["open_close_pct"] = (df["close_price"] - df["open_price"]) / df["open_price"]

        return df

    async def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        logger.info("Adding lag features")

        key_columns = ["close_price", "volume", "price_change_pct", "rsi_14"]

        for col in key_columns:
            if col in df.columns:
                for lag in range(1, min(self.config.max_lag_periods + 1, 21)):
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        return df

    async def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features."""
        logger.info("Adding rolling features")

        key_columns = ["close_price", "volume", "price_change_pct"]

        for col in key_columns:
            if col in df.columns:
                for window in self.config.rolling_windows:
                    df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window).mean()
                    df[f"{col}_rolling_std_{window}"] = df[col].rolling(window).std()
                    df[f"{col}_rolling_min_{window}"] = df[col].rolling(window).min()
                    df[f"{col}_rolling_max_{window}"] = df[col].rolling(window).max()

                    # Rolling ratios
                    rolling_mean = df[col].rolling(window).mean()
                    df[f"{col}_ratio_to_mean_{window}"] = df[col] / rolling_mean

        return df

    async def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between key variables."""
        logger.info("Adding interaction features")

        # Select numeric columns for interactions
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Limit to most important features to avoid explosion
        important_features = [
            "close_price",
            "volume",
            "rsi_14",
            "macd",
            "bb_position",
            "price_change_pct",
            "atr",
            "adx",
        ]

        interaction_cols = [col for col in important_features if col in numeric_cols]

        # Create interactions between selected features
        for i, col1 in enumerate(interaction_cols):
            for col2 in interaction_cols[i + 1 :]:
                # Calculate correlation to decide if interaction is worth adding
                corr = df[col1].corr(df[col2])
                if abs(corr) > self.config.interaction_threshold:
                    df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

        return df

    async def _add_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features for key variables."""
        logger.info("Adding polynomial features")

        # Select key features for polynomial expansion
        poly_features = [
            "rsi_14",
            "bb_position",
            "price_change_pct",
            "volume_sma_ratio",
        ]

        for feature in poly_features:
            if feature in df.columns:
                for degree in range(2, self.config.polynomial_degree + 1):
                    df[f"{feature}_poly_{degree}"] = df[feature] ** degree

        return df


class BayesianHyperparameterOptimizer:
    """Bayesian hyperparameter optimization using scikit-optimize."""

    def __init__(self, model_config: ModelConfig, scoring_metric: str = "f1_weighted"):
        self.model_config = model_config
        self.scoring_metric = scoring_metric
        self.best_score = -np.inf
        self.best_params = {}

    async def optimize(
        self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5, n_calls: int = 50
    ) -> Tuple[Dict[str, Any], float]:
        """
        Perform Bayesian hyperparameter optimization.

        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            n_calls: Number of optimization calls

        Returns:
            Tuple of (best_params, best_score)
        """
        if not BAYESIAN_OPT_AVAILABLE:
            logger.warning(
                "Bayesian optimization not available, falling back to grid search"
            )
            return await self._grid_search_fallback(X, y, cv_folds)

        logger.info(f"Starting Bayesian optimization for {self.model_config.name}")

        # Convert parameter space to skopt format
        dimensions = []
        param_names = []

        for param_name, param_config in self.model_config.param_space.items():
            param_names.append(param_name)

            if isinstance(param_config, dict):
                if param_config["type"] == "real":
                    dimensions.append(
                        Real(param_config["low"], param_config["high"], name=param_name)
                    )
                elif param_config["type"] == "int":
                    dimensions.append(
                        Integer(
                            param_config["low"], param_config["high"], name=param_name
                        )
                    )
                elif param_config["type"] == "categorical":
                    dimensions.append(
                        Categorical(param_config["choices"], name=param_name)
                    )
            else:
                # Assume it's a list of values (categorical)
                dimensions.append(Categorical(param_config, name=param_name))

        # Define objective function
        @use_named_args(dimensions)
        def objective(**params):
            try:
                # Create model with current parameters
                model = self.model_config.model_class(**params)

                # Perform cross-validation
                cv_scores = cross_val_score(
                    model, X, y, cv=cv_folds, scoring=self.scoring_metric, n_jobs=-1
                )

                # Return negative score (skopt minimizes)
                score = np.mean(cv_scores)
                return -score

            except Exception as e:
                logger.warning(f"Error in objective function: {e}")
                return 0  # Return worst possible score

        # Perform optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=42,
            n_initial_points=min(10, n_calls // 5),
        )

        # Extract best parameters
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun

        logger.info(f"Bayesian optimization completed. Best score: {best_score:.4f}")

        return best_params, best_score

    async def _grid_search_fallback(
        self, X: np.ndarray, y: np.ndarray, cv_folds: int
    ) -> Tuple[Dict[str, Any], float]:
        """Fallback to grid search if Bayesian optimization is not available."""
        logger.info(f"Performing grid search for {self.model_config.name}")

        # Convert parameter space to grid search format
        param_grid = {}
        for param_name, param_config in self.model_config.param_space.items():
            if isinstance(param_config, dict):
                if param_config["type"] in ["real", "int"]:
                    # Create a reasonable grid
                    low, high = param_config["low"], param_config["high"]
                    if param_config["type"] == "int":
                        param_grid[param_name] = list(
                            range(low, high + 1, max(1, (high - low) // 5))
                        )
                    else:
                        param_grid[param_name] = np.linspace(low, high, 5).tolist()
                elif param_config["type"] == "categorical":
                    param_grid[param_name] = param_config["choices"]
            else:
                param_grid[param_name] = param_config

        # Limit grid size to avoid excessive computation
        total_combinations = np.prod([len(values) for values in param_grid.values()])
        if total_combinations > 100:
            # Use random sampling instead
            param_list = list(ParameterSampler(param_grid, n_iter=50, random_state=42))
        else:
            param_list = list(ParameterGrid(param_grid))

        best_score = -np.inf
        best_params = {}

        for params in param_list:
            try:
                model = self.model_config.model_class(**params)
                cv_scores = cross_val_score(
                    model, X, y, cv=cv_folds, scoring=self.scoring_metric, n_jobs=-1
                )
                score = np.mean(cv_scores)

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                logger.warning(f"Error with parameters {params}: {e}")
                continue

        logger.info(f"Grid search completed. Best score: {best_score:.4f}")

        return best_params, best_score


class AutomatedTrainingPipeline:
    """
    Comprehensive automated ML training pipeline.

    This class orchestrates the entire training process including:
    - Feature engineering
    - Feature selection
    - Hyperparameter optimization
    - Model validation
    - Automated deployment
    """

    def __init__(self, ml_manager: MLModelManager):
        self.ml_manager = ml_manager
        self.settings = get_settings()

    async def train_models(
        self, data: pd.DataFrame, config: TrainingConfig
    ) -> List[TrainingResult]:
        """
        Train multiple models with automated pipeline.

        Args:
            data: Input DataFrame
            config: Training configuration

        Returns:
            List of training results
        """
        logger.info("Starting automated training pipeline")
        start_time = datetime.now()

        try:
            # Step 1: Feature Engineering
            logger.info("Step 1: Feature Engineering")
            feature_engineer = AutomatedFeatureEngineer(config.feature_engineering)
            engineered_data = await feature_engineer.engineer_features(data)

            # Step 2: Prepare target variable
            if config.target_column not in engineered_data.columns:
                raise ValueError(
                    f"Target column '{config.target_column}' not found in data"
                )

            # Step 3: Split data
            X, y = await self._prepare_data(engineered_data, config.target_column)
            X_train, X_val, X_test, y_train, y_val, y_test = await self._split_data(
                X, y, config.test_size, config.validation_size
            )

            # Step 4: Feature Selection
            logger.info("Step 4: Feature Selection")
            X_train_selected, X_val_selected, X_test_selected, selected_features = (
                await self._select_features(
                    X_train, X_val, X_test, y_train, config.feature_selection_k
                )
            )

            # Step 5: Train models
            logger.info("Step 5: Model Training and Optimization")
            results = []

            for model_config in config.models:
                try:
                    result = await self._train_single_model(
                        model_config,
                        X_train_selected,
                        X_val_selected,
                        X_test_selected,
                        y_train,
                        y_val,
                        y_test,
                        selected_features,
                        config,
                    )
                    results.append(result)

                except Exception as e:
                    logger.error(f"Failed to train model {model_config.name}: {e}")
                    continue

            # Step 6: Model Selection and Deployment
            if results and config.auto_deploy:
                logger.info("Step 6: Automated Model Deployment")
                best_result = max(results, key=lambda x: x.best_score)
                await self._deploy_model(best_result, config)

            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Training pipeline completed in {total_time:.2f} seconds")

            return results

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

    async def _prepare_data(
        self, data: pd.DataFrame, target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variables."""

        # Separate features and target
        y = data[target_column]
        X = data.drop(columns=[target_column])

        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]

        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")

        return X, y

    async def _split_data(
        self, X: pd.DataFrame, y: pd.Series, test_size: float, validation_size: float
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """Split data into train, validation, and test sets."""

        # For time series data, use time-based splitting
        n_samples = len(X)

        # Calculate split indices
        test_start = int(n_samples * (1 - test_size))
        val_start = int(test_start * (1 - validation_size))

        # Split data
        X_train = X.iloc[:val_start]
        X_val = X.iloc[val_start:test_start]
        X_test = X.iloc[test_start:]

        y_train = y.iloc[:val_start]
        y_val = y.iloc[val_start:test_start]
        y_test = y.iloc[test_start:]

        logger.info(
            f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    async def _select_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        k: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        """Perform automated feature selection."""

        logger.info(f"Selecting top {k} features from {X_train.shape[1]} available")

        # Use multiple feature selection methods and combine results
        selectors = [SelectKBest(f_classif, k=k), SelectKBest(mutual_info_classif, k=k)]

        selected_features_sets = []

        for selector in selectors:
            try:
                selector.fit(X_train, y_train)
                selected_features = X_train.columns[selector.get_support()].tolist()
                selected_features_sets.append(set(selected_features))
            except Exception as e:
                logger.warning(f"Feature selector failed: {e}")
                continue

        # Take intersection of selected features (features selected by multiple methods)
        if selected_features_sets:
            final_features = list(set.intersection(*selected_features_sets))

            # If intersection is too small, take union and limit to k
            if len(final_features) < k // 2:
                final_features = list(set.union(*selected_features_sets))[:k]
        else:
            # Fallback: use all features or top k by variance
            final_features = X_train.columns.tolist()[:k]

        logger.info(f"Selected {len(final_features)} features")

        return (
            X_train[final_features],
            X_val[final_features],
            X_test[final_features],
            final_features,
        )

    async def _train_single_model(
        self,
        model_config: ModelConfig,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        feature_names: List[str],
        config: TrainingConfig,
    ) -> TrainingResult:
        """Train a single model with hyperparameter optimization."""

        logger.info(f"Training model: {model_config.name}")
        model_start_time = datetime.now()

        # Prepare data for training
        if model_config.requires_scaling:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            preprocessing_pipeline = scaler
        else:
            X_train_scaled = X_train.values
            X_val_scaled = X_val.values
            X_test_scaled = X_test.values
            preprocessing_pipeline = None

        # Hyperparameter optimization
        optimizer = BayesianHyperparameterOptimizer(model_config, config.scoring_metric)
        best_params, best_cv_score = await optimizer.optimize(
            X_train_scaled, y_train, config.cv_folds, config.n_optimization_calls
        )

        # Train final model with best parameters
        final_model = model_config.model_class(**best_params)
        final_model.fit(X_train_scaled, y_train)

        # Validation
        val_predictions = final_model.predict(X_val_scaled)
        test_predictions = final_model.predict(X_test_scaled)

        # Calculate validation metrics
        validation_metrics = {
            "val_accuracy": accuracy_score(y_val, val_predictions),
            "val_precision": precision_score(
                y_val, val_predictions, average="weighted"
            ),
            "val_recall": recall_score(y_val, val_predictions, average="weighted"),
            "val_f1": f1_score(y_val, val_predictions, average="weighted"),
            "test_accuracy": accuracy_score(y_test, test_predictions),
            "test_precision": precision_score(
                y_test, test_predictions, average="weighted"
            ),
            "test_recall": recall_score(y_test, test_predictions, average="weighted"),
            "test_f1": f1_score(y_test, test_predictions, average="weighted"),
        }

        # Feature importance (if available)
        feature_importance = {}
        if hasattr(final_model, "feature_importances_"):
            feature_importance = dict(
                zip(feature_names, final_model.feature_importances_)
            )
        elif hasattr(final_model, "coef_"):
            feature_importance = dict(
                zip(feature_names, np.abs(final_model.coef_).flatten())
            )

        # Cross-validation scores
        cv_scores = cross_val_score(
            final_model,
            X_train_scaled,
            y_train,
            cv=config.cv_folds,
            scoring=config.scoring_metric,
        )

        training_time = (datetime.now() - model_start_time).total_seconds()

        # Create model ID
        model_id = f"{model_config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        result = TrainingResult(
            model_id=model_id,
            model_name=model_config.name,
            best_score=best_cv_score,
            best_params=best_params,
            cv_scores=cv_scores.tolist(),
            feature_importance=feature_importance,
            training_time=training_time,
            validation_metrics=validation_metrics,
            model_object=final_model,
            feature_names=feature_names,
            preprocessing_pipeline=preprocessing_pipeline,
        )

        logger.info(
            f"Model {model_config.name} trained successfully. CV Score: {best_cv_score:.4f}"
        )

        return result

    async def _deploy_model(
        self, result: TrainingResult, config: TrainingConfig
    ) -> None:
        """Deploy the best model automatically."""

        logger.info(f"Deploying model: {result.model_name}")

        try:
            # Create model metrics
            metrics = ModelMetrics(
                accuracy=result.validation_metrics["test_accuracy"],
                precision=result.validation_metrics["test_precision"],
                recall=result.validation_metrics["test_recall"],
                f1_score=result.validation_metrics["test_f1"],
                custom_metrics={
                    "cv_score": result.best_score,
                    "training_time": result.training_time,
                },
            )

            # Register model with ML manager
            model_id = await self.ml_manager.register_model(
                model_name=result.model_name,
                model_object=result.model_object,
                metrics=metrics,
                tags={
                    "automated_training": "true",
                    "feature_count": str(len(result.feature_names)),
                    "training_date": datetime.now().isoformat(),
                },
                description=f"Automatically trained {result.model_name} model",
                artifacts={
                    "feature_names": result.feature_names,
                    "feature_importance": result.feature_importance,
                    "best_params": result.best_params,
                    "validation_metrics": result.validation_metrics,
                },
            )

            # Check if we should promote to production
            current_production_models = await self.ml_manager.list_models(
                status_filter="production"
            )

            should_promote = True

            if current_production_models:
                # Compare with current production model
                current_model = current_production_models[
                    0
                ]  # Assume one production model

                # Check if new model is significantly better
                performance_improvement = (
                    result.best_score
                    - current_model.metrics.custom_metrics.get("cv_score", 0)
                )

                if performance_improvement < config.rollback_threshold:
                    should_promote = False
                    logger.info(
                        f"New model performance improvement ({performance_improvement:.4f}) "
                        f"below threshold ({config.rollback_threshold}). Not promoting."
                    )

            if should_promote:
                success = await self.ml_manager.promote_model_to_production(model_id)
                if success:
                    logger.info(f"Model {model_id} successfully deployed to production")
                else:
                    logger.error(f"Failed to promote model {model_id} to production")

        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            raise


# Default model configurations
DEFAULT_MODEL_CONFIGS = [
    ModelConfig(
        model_class=RandomForestClassifier,
        param_space={
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "max_depth": {"type": "int", "low": 3, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": ["sqrt", "log2", None],
        },
        name="random_forest",
        requires_scaling=False,
    ),
    ModelConfig(
        model_class=GradientBoostingClassifier,
        param_space={
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "learning_rate": {"type": "real", "low": 0.01, "high": 0.3},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "subsample": {"type": "real", "low": 0.6, "high": 1.0},
        },
        name="gradient_boosting",
        requires_scaling=False,
    ),
    ModelConfig(
        model_class=LogisticRegression,
        param_space={
            "C": {"type": "real", "low": 0.01, "high": 100},
            "penalty": ["l1", "l2", "elasticnet"],
            "solver": ["liblinear", "saga"],
        },
        name="logistic_regression",
        requires_scaling=True,
    ),
]


async def create_default_training_config(target_column: str) -> TrainingConfig:
    """Create a default training configuration."""

    return TrainingConfig(
        target_column=target_column,
        feature_engineering=FeatureEngineeringConfig(),
        models=DEFAULT_MODEL_CONFIGS,
        cv_folds=5,
        test_size=0.2,
        validation_size=0.2,
        optimization_method="bayesian",
        n_optimization_calls=30,
        scoring_metric="f1_weighted",
        early_stopping_rounds=10,
        feature_selection_k=50,
        auto_deploy=False,
        rollback_threshold=0.02,
    )
