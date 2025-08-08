"""
Tests for A/B Testing Framework
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
import uuid

from stock_analysis_system.ml.ab_testing_framework import (
    ABTestingFramework,
    TrafficRouter,
    ExperimentConfig,
    ModelVariant,
    ExperimentResult,
    ExperimentSummary,
    ExperimentStatus,
    TrafficSplitMethod,
    StatisticalTest
)


class TestABTestingFramework:
    """Test cases for ABTestingFramework."""
    
    @pytest.fixture
    def mock_database_url(self):
        """Mock database URL for testing."""
        return "sqlite:///:memory:"
    
    @pytest.fixture
    def mock_mlflow_uri(self):
        """Mock MLflow URI for testing."""
        return "sqlite:///mlflow.db"
    
    @pytest.fixture
    def ab_framework(self, mock_database_url, mock_mlflow_uri):
        """Create ABTestingFramework instance for testing."""
        with patch('stock_analysis_system.ml.ab_testing_framework.create_engine'):
            with patch('stock_analysis_system.ml.ab_testing_framework.mlflow'):
                framework = ABTestingFramework(mock_database_url, mock_mlflow_uri)
                return framework
    
    @pytest.fixture
    def sample_experiment_config(self):
        """Sample experiment configuration for testing."""
        experiment_id = str(uuid.uuid4())
        
        variants = [
            ModelVariant(
                variant_id="control",
                model_id="model_v1",
                model_name="Control Model",
                traffic_percentage=50.0,
                description="Current production model",
                parameters={"version": "1.0"},
                is_control=True
            ),
            ModelVariant(
                variant_id="treatment",
                model_id="model_v2",
                model_name="Treatment Model",
                traffic_percentage=50.0,
                description="New experimental model",
                parameters={"version": "2.0"},
                is_control=False
            )
        ]
        
        return ExperimentConfig(
            experiment_id=experiment_id,
            experiment_name="Model Comparison Test",
            description="Testing new model against current production model",
            variants=variants,
            traffic_split_method=TrafficSplitMethod.RANDOM,
            primary_metric="accuracy",
            secondary_metrics=["precision", "recall"],
            minimum_sample_size=1000,
            confidence_level=0.95,
            statistical_power=0.8,
            max_duration_days=30,
            early_stopping_enabled=True,
            significance_threshold=0.05
        )
    
    @pytest.mark.asyncio
    async def test_create_experiment(self, ab_framework, sample_experiment_config):
        """Test experiment creation."""
        # Mock database and MLflow operations
        ab_framework._store_experiment_config = AsyncMock(return_value=None)
        
        with patch('mlflow.create_experiment', return_value="test_experiment_id"):
            result = await ab_framework.create_experiment(sample_experiment_config)
        
        assert result is True
        assert sample_experiment_config.experiment_id in ab_framework.active_experiments
        
        experiment = ab_framework.active_experiments[sample_experiment_config.experiment_id]
        assert experiment['status'] == ExperimentStatus.DRAFT
        assert experiment['config'] == sample_experiment_config
    
    @pytest.mark.asyncio
    async def test_start_experiment(self, ab_framework, sample_experiment_config):
        """Test starting an experiment."""
        # Create experiment first
        ab_framework._store_experiment_config = AsyncMock(return_value=None)
        ab_framework._update_experiment_status = AsyncMock(return_value=None)
        
        with patch('mlflow.create_experiment', return_value="test_experiment_id"):
            await ab_framework.create_experiment(sample_experiment_config)
        
        # Mock traffic router configuration
        ab_framework.traffic_router.configure_experiment = AsyncMock(return_value=None)
        
        # Start experiment
        with patch('mlflow.start_run'), patch('mlflow.log_params'):
            result = await ab_framework.start_experiment(sample_experiment_config.experiment_id)
        
        assert result is True
        
        experiment = ab_framework.active_experiments[sample_experiment_config.experiment_id]
        assert experiment['status'] == ExperimentStatus.RUNNING
        assert experiment['start_date'] is not None
    
    @pytest.mark.asyncio
    async def test_route_traffic(self, ab_framework, sample_experiment_config):
        """Test traffic routing."""
        # Set up experiment
        ab_framework._store_experiment_config = AsyncMock(return_value=None)
        ab_framework._update_experiment_status = AsyncMock(return_value=None)
        ab_framework.traffic_router.configure_experiment = AsyncMock(return_value=None)
        
        with patch('mlflow.create_experiment', return_value="test_experiment_id"):
            await ab_framework.create_experiment(sample_experiment_config)
        
        with patch('mlflow.start_run'), patch('mlflow.log_params'):
            await ab_framework.start_experiment(sample_experiment_config.experiment_id)
        
        # Mock traffic router
        ab_framework.traffic_router.route_user = AsyncMock(return_value="control")
        
        # Route traffic
        variant_id = await ab_framework.route_traffic(
            sample_experiment_config.experiment_id,
            "user_123"
        )
        
        assert variant_id == "control"
        
        # Check sample count updated
        experiment = ab_framework.active_experiments[sample_experiment_config.experiment_id]
        assert experiment['sample_counts']['control'] == 1
    
    @pytest.mark.asyncio
    async def test_record_metric(self, ab_framework, sample_experiment_config):
        """Test metric recording."""
        # Set up running experiment
        ab_framework._store_experiment_config = AsyncMock(return_value=None)
        ab_framework._update_experiment_status = AsyncMock(return_value=None)
        ab_framework.traffic_router.configure_experiment = AsyncMock(return_value=None)
        ab_framework._store_experiment_metrics = AsyncMock(return_value=None)
        ab_framework._analyze_experiment_results = AsyncMock(return_value=None)
        
        with patch('mlflow.create_experiment', return_value="test_experiment_id"):
            await ab_framework.create_experiment(sample_experiment_config)
        
        with patch('mlflow.start_run'), patch('mlflow.log_params'):
            await ab_framework.start_experiment(sample_experiment_config.experiment_id)
        
        # Record metrics
        metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88
        }
        
        result = await ab_framework.record_metric(
            sample_experiment_config.experiment_id,
            "control",
            "user_123",
            metrics
        )
        
        assert result is True
        ab_framework._store_experiment_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_experiment(self, ab_framework, sample_experiment_config):
        """Test experiment analysis."""
        # Set up experiment
        ab_framework._store_experiment_config = AsyncMock(return_value=None)
        ab_framework._update_experiment_status = AsyncMock(return_value=None)
        ab_framework.traffic_router.configure_experiment = AsyncMock(return_value=None)
        ab_framework._log_experiment_results = AsyncMock(return_value=None)
        
        with patch('mlflow.create_experiment', return_value="test_experiment_id"):
            await ab_framework.create_experiment(sample_experiment_config)
        
        with patch('mlflow.start_run'), patch('mlflow.log_params'):
            await ab_framework.start_experiment(sample_experiment_config.experiment_id)
        
        # Mock experiment data
        experiment_data = pd.DataFrame({
            'variant_id': ['control'] * 500 + ['treatment'] * 500,
            'user_id': [f'user_{i}' for i in range(1000)],
            'accuracy': np.concatenate([
                np.random.normal(0.80, 0.05, 500),  # Control
                np.random.normal(0.85, 0.05, 500)   # Treatment (better)
            ]),
            'precision': np.concatenate([
                np.random.normal(0.78, 0.05, 500),
                np.random.normal(0.83, 0.05, 500)
            ]),
            'recall': np.concatenate([
                np.random.normal(0.82, 0.05, 500),
                np.random.normal(0.87, 0.05, 500)
            ])
        })
        
        ab_framework._fetch_experiment_data = AsyncMock(return_value=experiment_data)
        
        # Analyze experiment
        summary = await ab_framework.analyze_experiment(sample_experiment_config.experiment_id)
        
        assert isinstance(summary, ExperimentSummary)
        assert summary.experiment_id == sample_experiment_config.experiment_id
        assert len(summary.results) == 2  # Control and treatment
        assert summary.total_samples == 1000
        
        # Check that treatment variant has higher accuracy
        treatment_result = next(r for r in summary.results if r.variant_id == "treatment")
        control_result = next(r for r in summary.results if r.variant_id == "control")
        assert treatment_result.primary_metric_value > control_result.primary_metric_value
    
    @pytest.mark.asyncio
    async def test_stop_experiment(self, ab_framework, sample_experiment_config):
        """Test stopping an experiment."""
        # Set up running experiment
        ab_framework._store_experiment_config = AsyncMock(return_value=None)
        ab_framework._update_experiment_status = AsyncMock(return_value=None)
        ab_framework.traffic_router.configure_experiment = AsyncMock(return_value=None)
        ab_framework.analyze_experiment = AsyncMock(return_value=Mock())
        
        with patch('mlflow.create_experiment', return_value="test_experiment_id"):
            await ab_framework.create_experiment(sample_experiment_config)
        
        with patch('mlflow.start_run'), patch('mlflow.log_params'):
            await ab_framework.start_experiment(sample_experiment_config.experiment_id)
        
        # Stop experiment
        with patch('mlflow.start_run'), patch('mlflow.log_params'):
            result = await ab_framework.stop_experiment(
                sample_experiment_config.experiment_id,
                "Test completion"
            )
        
        assert result is True
        
        experiment = ab_framework.active_experiments[sample_experiment_config.experiment_id]
        assert experiment['status'] == ExperimentStatus.COMPLETED
        assert experiment['end_date'] is not None
    
    def test_validate_experiment_config_valid(self, ab_framework, sample_experiment_config):
        """Test validation of valid experiment configuration."""
        result = ab_framework._validate_experiment_config(sample_experiment_config)
        assert result is True
    
    def test_validate_experiment_config_invalid_traffic(self, ab_framework, sample_experiment_config):
        """Test validation with invalid traffic percentages."""
        # Modify traffic percentages to not sum to 100
        sample_experiment_config.variants[0].traffic_percentage = 60.0
        sample_experiment_config.variants[1].traffic_percentage = 30.0  # Total = 90%
        
        result = ab_framework._validate_experiment_config(sample_experiment_config)
        assert result is False
    
    def test_validate_experiment_config_no_control(self, ab_framework, sample_experiment_config):
        """Test validation with no control variant."""
        # Remove control flag from all variants
        for variant in sample_experiment_config.variants:
            variant.is_control = False
        
        result = ab_framework._validate_experiment_config(sample_experiment_config)
        assert result is False
    
    def test_validate_experiment_config_multiple_controls(self, ab_framework, sample_experiment_config):
        """Test validation with multiple control variants."""
        # Set both variants as control
        for variant in sample_experiment_config.variants:
            variant.is_control = True
        
        result = ab_framework._validate_experiment_config(sample_experiment_config)
        assert result is False
    
    def test_statistical_tests(self, ab_framework):
        """Test statistical test methods."""
        # Generate test data
        group_a = np.random.normal(0.8, 0.1, 500)
        group_b = np.random.normal(0.85, 0.1, 500)  # Slightly better
        
        # Test t-test
        t_stat, p_value = ab_framework._perform_t_test(group_a, group_b)
        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
        
        # Test Mann-Whitney test
        u_stat, p_value = ab_framework._perform_mann_whitney_test(group_a, group_b)
        assert isinstance(u_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
        
        # Test bootstrap test
        diff, p_value = ab_framework._perform_bootstrap_test(group_a, group_b, n_bootstrap=1000)
        assert isinstance(diff, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
    
    def test_confidence_interval_calculation(self, ab_framework):
        """Test confidence interval calculation."""
        data = np.random.normal(0.8, 0.1, 1000)
        ci_lower, ci_upper = ab_framework._calculate_confidence_interval(data, 0.95)
        
        assert ci_lower < np.mean(data) < ci_upper
        assert ci_upper - ci_lower > 0  # Interval should have positive width
    
    def test_determine_winning_variant(self, ab_framework, sample_experiment_config):
        """Test winning variant determination."""
        # Create mock results
        control_result = ExperimentResult(
            experiment_id=sample_experiment_config.experiment_id,
            variant_id="control",
            sample_size=500,
            primary_metric_value=0.80,
            primary_metric_std=0.05,
            secondary_metrics={},
            confidence_interval=(0.75, 0.85),
            statistical_significance=False,
            p_value=0.1,
            effect_size=0.0
        )
        
        treatment_result = ExperimentResult(
            experiment_id=sample_experiment_config.experiment_id,
            variant_id="treatment",
            sample_size=500,
            primary_metric_value=0.85,
            primary_metric_std=0.05,
            secondary_metrics={},
            confidence_interval=(0.80, 0.90),
            statistical_significance=True,
            p_value=0.01,
            effect_size=1.0
        )
        
        results = [control_result, treatment_result]
        
        winning_variant = ab_framework._determine_winning_variant(results, sample_experiment_config)
        assert winning_variant == "treatment"
        
        # Test case with no significant variants
        treatment_result.statistical_significance = False
        winning_variant = ab_framework._determine_winning_variant(results, sample_experiment_config)
        assert winning_variant == "control"  # Should default to control
    
    def test_generate_experiment_recommendations(self, ab_framework, sample_experiment_config):
        """Test experiment recommendation generation."""
        # Create mock results with small sample size
        results = [
            ExperimentResult(
                experiment_id=sample_experiment_config.experiment_id,
                variant_id="control",
                sample_size=50,  # Below minimum
                primary_metric_value=0.80,
                primary_metric_std=0.05,
                secondary_metrics={},
                confidence_interval=(0.75, 0.85),
                statistical_significance=False,
                p_value=0.1,
                effect_size=0.2
            )
        ]
        
        recommendations = ab_framework._generate_experiment_recommendations(
            results, sample_experiment_config
        )
        
        assert len(recommendations) > 0
        assert any("sample size" in rec.lower() for rec in recommendations)
        assert any("no statistically significant" in rec.lower() for rec in recommendations)


class TestTrafficRouter:
    """Test cases for TrafficRouter."""
    
    @pytest.fixture
    def traffic_router(self):
        """Create TrafficRouter instance for testing."""
        return TrafficRouter()
    
    @pytest.fixture
    def sample_variants(self):
        """Sample variants for testing."""
        return [
            ModelVariant(
                variant_id="control",
                model_id="model_v1",
                model_name="Control",
                traffic_percentage=30.0,
                description="Control variant",
                parameters={},
                is_control=True
            ),
            ModelVariant(
                variant_id="treatment_a",
                model_id="model_v2",
                model_name="Treatment A",
                traffic_percentage=35.0,
                description="Treatment A",
                parameters={},
                is_control=False
            ),
            ModelVariant(
                variant_id="treatment_b",
                model_id="model_v3",
                model_name="Treatment B",
                traffic_percentage=35.0,
                description="Treatment B",
                parameters={},
                is_control=False
            )
        ]
    
    def test_random_routing(self, traffic_router, sample_variants):
        """Test random traffic routing."""
        # Test multiple routing decisions
        results = []
        for _ in range(1000):
            variant_id = traffic_router._random_routing(sample_variants)
            results.append(variant_id)
        
        # Check that all variants are represented
        unique_variants = set(results)
        expected_variants = {"control", "treatment_a", "treatment_b"}
        assert unique_variants == expected_variants
        
        # Check approximate distribution (should be close to specified percentages)
        control_count = results.count("control")
        treatment_a_count = results.count("treatment_a")
        treatment_b_count = results.count("treatment_b")
        
        # Allow for some variance in random distribution
        assert 250 <= control_count <= 350  # ~30% ± 5%
        assert 300 <= treatment_a_count <= 400  # ~35% ± 5%
        assert 300 <= treatment_b_count <= 400  # ~35% ± 5%
    
    def test_hash_based_routing(self, traffic_router, sample_variants):
        """Test hash-based consistent routing."""
        # Test that same user always gets same variant
        user_id = "test_user_123"
        
        variant_1 = traffic_router._hash_based_routing(user_id, sample_variants)
        variant_2 = traffic_router._hash_based_routing(user_id, sample_variants)
        variant_3 = traffic_router._hash_based_routing(user_id, sample_variants)
        
        assert variant_1 == variant_2 == variant_3
        
        # Test that different users can get different variants
        variants = set()
        for i in range(100):
            user_id = f"user_{i}"
            variant = traffic_router._hash_based_routing(user_id, sample_variants)
            variants.add(variant)
        
        # Should have multiple variants represented
        assert len(variants) > 1
    
    def test_time_based_routing(self, traffic_router, sample_variants):
        """Test time-based routing."""
        # Mock datetime to control current hour
        with patch('stock_analysis_system.ml.ab_testing_framework.datetime') as mock_datetime:
            # Test different hours
            for hour in range(24):
                mock_datetime.now.return_value.hour = hour
                variant_id = traffic_router._time_based_routing(sample_variants)
                
                expected_index = hour % len(sample_variants)
                expected_variant = sample_variants[expected_index].variant_id
                
                assert variant_id == expected_variant
    
    @pytest.mark.asyncio
    async def test_route_user_integration(self, traffic_router, sample_variants):
        """Test complete user routing integration."""
        # Create mock experiment config
        experiment_config = Mock()
        experiment_config.variants = sample_variants
        experiment_config.traffic_split_method = TrafficSplitMethod.HASH_BASED
        
        # Configure experiment
        await traffic_router.configure_experiment("test_experiment", experiment_config)
        
        # Route user
        variant_id = await traffic_router.route_user("test_experiment", "test_user", {})
        
        assert variant_id in ["control", "treatment_a", "treatment_b"]
        
        # Test consistency
        variant_id_2 = await traffic_router.route_user("test_experiment", "test_user", {})
        assert variant_id == variant_id_2


@pytest.mark.asyncio
async def test_integration_ab_testing_workflow():
    """Integration test for complete A/B testing workflow."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        database_url = f"sqlite:///{db_path}"
        mlflow_uri = "sqlite:///test_mlflow.db"
        
        with patch('stock_analysis_system.ml.ab_testing_framework.create_engine'):
            with patch('stock_analysis_system.ml.ab_testing_framework.mlflow'):
                framework = ABTestingFramework(database_url, mlflow_uri)
                
                # Mock database operations
                framework._store_experiment_config = AsyncMock(return_value=None)
                framework._update_experiment_status = AsyncMock(return_value=None)
                framework._store_experiment_metrics = AsyncMock(return_value=None)
                framework._log_experiment_results = AsyncMock(return_value=None)
                
                # Create experiment configuration
                experiment_id = str(uuid.uuid4())
                variants = [
                    ModelVariant(
                        variant_id="control",
                        model_id="model_v1",
                        model_name="Control Model",
                        traffic_percentage=50.0,
                        description="Current model",
                        parameters={"version": "1.0"},
                        is_control=True
                    ),
                    ModelVariant(
                        variant_id="treatment",
                        model_id="model_v2",
                        model_name="New Model",
                        traffic_percentage=50.0,
                        description="Improved model",
                        parameters={"version": "2.0"},
                        is_control=False
                    )
                ]
                
                config = ExperimentConfig(
                    experiment_id=experiment_id,
                    experiment_name="Integration Test",
                    description="Testing A/B framework",
                    variants=variants,
                    traffic_split_method=TrafficSplitMethod.RANDOM,
                    primary_metric="accuracy",
                    secondary_metrics=["precision"],
                    minimum_sample_size=100,
                    confidence_level=0.95,
                    statistical_power=0.8,
                    max_duration_days=7,
                    early_stopping_enabled=True,
                    significance_threshold=0.05
                )
                
                # Create experiment
                with patch('mlflow.create_experiment', return_value="test_exp_id"):
                    create_success = await framework.create_experiment(config)
                assert create_success is True
                
                # Start experiment
                framework.traffic_router.configure_experiment = AsyncMock(return_value=None)
                with patch('mlflow.start_run'), patch('mlflow.log_params'):
                    start_success = await framework.start_experiment(experiment_id)
                assert start_success is True
                
                # Simulate traffic routing and metric recording
                framework.traffic_router.route_user = AsyncMock(side_effect=["control", "treatment"] * 50)
                
                for i in range(100):
                    user_id = f"user_{i}"
                    variant_id = await framework.route_traffic(experiment_id, user_id)
                    
                    # Simulate different performance for variants
                    if variant_id == "control":
                        accuracy = np.random.normal(0.80, 0.05)
                    else:
                        accuracy = np.random.normal(0.85, 0.05)  # Better performance
                    
                    metrics = {"accuracy": max(0, min(1, accuracy))}
                    
                    await framework.record_metric(experiment_id, variant_id, user_id, metrics)
                
                # Mock experiment data for analysis
                experiment_data = pd.DataFrame({
                    'variant_id': ['control'] * 50 + ['treatment'] * 50,
                    'user_id': [f'user_{i}' for i in range(100)],
                    'accuracy': np.concatenate([
                        np.random.normal(0.80, 0.05, 50),
                        np.random.normal(0.85, 0.05, 50)
                    ])
                })
                framework._fetch_experiment_data = AsyncMock(return_value=experiment_data)
                
                # Analyze experiment
                summary = await framework.analyze_experiment(experiment_id)
                assert isinstance(summary, ExperimentSummary)
                assert len(summary.results) == 2
                
                # Stop experiment
                with patch('mlflow.start_run'), patch('mlflow.log_params'):
                    stop_success = await framework.stop_experiment(experiment_id, "Integration test complete")
                assert stop_success is True
                
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
        if os.path.exists("test_mlflow.db"):
            os.unlink("test_mlflow.db")


if __name__ == "__main__":
    pytest.main([__file__])