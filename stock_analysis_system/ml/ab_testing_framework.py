"""
A/B Testing Framework for Model Comparison

This module implements a comprehensive A/B testing framework for comparing
different model versions and strategies in the Stock Analysis System.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine, text
import json
import uuid
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentStatus(str, Enum):
    """Status of A/B testing experiments."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class TrafficSplitMethod(str, Enum):
    """Methods for splitting traffic between model variants."""
    RANDOM = "random"
    HASH_BASED = "hash_based"
    GEOGRAPHIC = "geographic"
    TIME_BASED = "time_based"

class StatisticalTest(str, Enum):
    """Statistical tests for significance analysis."""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    CHI_SQUARE = "chi_square"
    BOOTSTRAP = "bootstrap"

@dataclass
class ModelVariant:
    """Model variant configuration for A/B testing."""
    variant_id: str
    model_id: str
    model_name: str
    traffic_percentage: float
    description: str
    parameters: Dict[str, Any]
    is_control: bool = False

@dataclass
class ExperimentConfig:
    """A/B testing experiment configuration."""
    experiment_id: str
    experiment_name: str
    description: str
    variants: List[ModelVariant]
    traffic_split_method: TrafficSplitMethod
    primary_metric: str
    secondary_metrics: List[str]
    minimum_sample_size: int
    confidence_level: float
    statistical_power: float
    max_duration_days: int
    early_stopping_enabled: bool
    significance_threshold: float

@dataclass
class ExperimentResult:
    """Results of an A/B testing experiment."""
    experiment_id: str
    variant_id: str
    sample_size: int
    primary_metric_value: float
    primary_metric_std: float
    secondary_metrics: Dict[str, float]
    confidence_interval: Tuple[float, float]
    statistical_significance: bool
    p_value: float
    effect_size: float
    conversion_rate: Optional[float] = None

@dataclass
class ExperimentSummary:
    """Summary of A/B testing experiment results."""
    experiment_id: str
    experiment_name: str
    status: ExperimentStatus
    start_date: datetime
    end_date: Optional[datetime]
    duration_days: int
    total_samples: int
    winning_variant: Optional[str]
    confidence_level: float
    results: List[ExperimentResult]
    recommendations: List[str]
    statistical_summary: Dict[str, Any]

class ABTestingFramework:
    """
    Comprehensive A/B testing framework for model comparison.
    
    Features:
    - Multiple traffic splitting methods
    - Statistical significance testing
    - Early stopping capabilities
    - Comprehensive result analysis
    - Integration with MLflow for experiment tracking
    """
    
    def __init__(self, database_url: str, mlflow_tracking_uri: str):
        """
        Initialize the A/B testing framework.
        
        Args:
            database_url: Database connection string
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.database_url = database_url
        self.engine = create_engine(database_url)
        
        # MLflow setup
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
        # Active experiments
        self.active_experiments = {}
        self.traffic_router = TrafficRouter()
        
        # Statistical test configurations
        self.statistical_tests = {
            StatisticalTest.T_TEST: self._perform_t_test,
            StatisticalTest.MANN_WHITNEY: self._perform_mann_whitney_test,
            StatisticalTest.CHI_SQUARE: self._perform_chi_square_test,
            StatisticalTest.BOOTSTRAP: self._perform_bootstrap_test
        }
        
        logger.info("ABTestingFramework initialized successfully")
    
    async def create_experiment(self, config: ExperimentConfig) -> bool:
        """
        Create a new A/B testing experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            bool: True if experiment created successfully
        """
        try:
            # Validate experiment configuration
            if not self._validate_experiment_config(config):
                return False
            
            # Store experiment in database
            await self._store_experiment_config(config)
            
            # Initialize experiment in MLflow
            experiment_name = f"ab_test_{config.experiment_name}_{config.experiment_id}"
            mlflow_experiment = mlflow.create_experiment(experiment_name)
            
            # Store experiment state
            self.active_experiments[config.experiment_id] = {
                'config': config,
                'status': ExperimentStatus.DRAFT,
                'start_date': None,
                'end_date': None,
                'mlflow_experiment_id': mlflow_experiment,
                'results': {},
                'sample_counts': {variant.variant_id: 0 for variant in config.variants}
            }
            
            logger.info(f"A/B testing experiment {config.experiment_id} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create experiment {config.experiment_id}: {str(e)}")
            return False
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """
        Start an A/B testing experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            bool: True if experiment started successfully
        """
        try:
            if experiment_id not in self.active_experiments:
                logger.error(f"Experiment {experiment_id} not found")
                return False
            
            experiment = self.active_experiments[experiment_id]
            
            if experiment['status'] != ExperimentStatus.DRAFT:
                logger.error(f"Experiment {experiment_id} is not in draft status")
                return False
            
            # Update experiment status
            experiment['status'] = ExperimentStatus.RUNNING
            experiment['start_date'] = datetime.now()
            
            # Configure traffic router
            await self.traffic_router.configure_experiment(
                experiment_id, experiment['config']
            )
            
            # Update database
            await self._update_experiment_status(experiment_id, ExperimentStatus.RUNNING)
            
            # Log to MLflow
            with mlflow.start_run(experiment_id=experiment['mlflow_experiment_id']):
                mlflow.log_params({
                    'experiment_id': experiment_id,
                    'start_date': experiment['start_date'].isoformat(),
                    'status': ExperimentStatus.RUNNING.value
                })
            
            logger.info(f"A/B testing experiment {experiment_id} started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start experiment {experiment_id}: {str(e)}")
            return False
    
    async def route_traffic(self, experiment_id: str, user_id: str, 
                          context: Dict[str, Any] = None) -> Optional[str]:
        """
        Route traffic to appropriate model variant.
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier for consistent routing
            context: Additional context for routing decisions
            
        Returns:
            Optional[str]: Variant ID to use, None if experiment not active
        """
        try:
            if experiment_id not in self.active_experiments:
                return None
            
            experiment = self.active_experiments[experiment_id]
            
            if experiment['status'] != ExperimentStatus.RUNNING:
                return None
            
            # Route traffic using configured method
            variant_id = await self.traffic_router.route_user(
                experiment_id, user_id, context
            )
            
            # Update sample count
            if variant_id:
                experiment['sample_counts'][variant_id] += 1
            
            return variant_id
            
        except Exception as e:
            logger.error(f"Failed to route traffic for experiment {experiment_id}: {str(e)}")
            return None
    
    async def record_metric(self, experiment_id: str, variant_id: str, 
                          user_id: str, metrics: Dict[str, float]) -> bool:
        """
        Record metrics for a user interaction.
        
        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            user_id: User identifier
            metrics: Dictionary of metric values
            
        Returns:
            bool: True if metrics recorded successfully
        """
        try:
            if experiment_id not in self.active_experiments:
                return False
            
            # Store metrics in database
            await self._store_experiment_metrics(
                experiment_id, variant_id, user_id, metrics
            )
            
            # Check if we should analyze results
            experiment = self.active_experiments[experiment_id]
            total_samples = sum(experiment['sample_counts'].values())
            
            if (total_samples >= experiment['config'].minimum_sample_size and
                total_samples % 1000 == 0):  # Analyze every 1000 samples
                await self._analyze_experiment_results(experiment_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record metrics for experiment {experiment_id}: {str(e)}")
            return False
    
    async def analyze_experiment(self, experiment_id: str) -> Optional[ExperimentSummary]:
        """
        Analyze experiment results and determine statistical significance.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Optional[ExperimentSummary]: Experiment analysis results
        """
        try:
            if experiment_id not in self.active_experiments:
                return None
            
            experiment = self.active_experiments[experiment_id]
            config = experiment['config']
            
            # Fetch experiment data
            experiment_data = await self._fetch_experiment_data(experiment_id)
            
            if not experiment_data:
                return None
            
            # Analyze each variant
            results = []
            for variant in config.variants:
                variant_data = experiment_data[experiment_data['variant_id'] == variant.variant_id]
                
                if len(variant_data) == 0:
                    continue
                
                # Calculate primary metric statistics
                primary_values = variant_data[config.primary_metric].values
                primary_mean = np.mean(primary_values)
                primary_std = np.std(primary_values)
                
                # Calculate confidence interval
                confidence_interval = self._calculate_confidence_interval(
                    primary_values, config.confidence_level
                )
                
                # Calculate secondary metrics
                secondary_metrics = {}
                for metric in config.secondary_metrics:
                    if metric in variant_data.columns:
                        secondary_metrics[metric] = np.mean(variant_data[metric].values)
                
                result = ExperimentResult(
                    experiment_id=experiment_id,
                    variant_id=variant.variant_id,
                    sample_size=len(variant_data),
                    primary_metric_value=primary_mean,
                    primary_metric_std=primary_std,
                    secondary_metrics=secondary_metrics,
                    confidence_interval=confidence_interval,
                    statistical_significance=False,  # Will be calculated later
                    p_value=0.0,  # Will be calculated later
                    effect_size=0.0  # Will be calculated later
                )
                
                results.append(result)
            
            # Perform statistical significance testing
            if len(results) >= 2:
                results = await self._perform_statistical_tests(results, experiment_data, config)
            
            # Determine winning variant
            winning_variant = self._determine_winning_variant(results, config)
            
            # Generate recommendations
            recommendations = self._generate_experiment_recommendations(results, config)
            
            # Calculate statistical summary
            statistical_summary = self._calculate_statistical_summary(results, config)
            
            # Create experiment summary
            summary = ExperimentSummary(
                experiment_id=experiment_id,
                experiment_name=config.experiment_name,
                status=experiment['status'],
                start_date=experiment['start_date'],
                end_date=experiment.get('end_date'),
                duration_days=self._calculate_experiment_duration(experiment),
                total_samples=sum(result.sample_size for result in results),
                winning_variant=winning_variant,
                confidence_level=config.confidence_level,
                results=results,
                recommendations=recommendations,
                statistical_summary=statistical_summary
            )
            
            # Log results to MLflow
            await self._log_experiment_results(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to analyze experiment {experiment_id}: {str(e)}")
            return None
    
    async def stop_experiment(self, experiment_id: str, reason: str = "Manual stop") -> bool:
        """
        Stop a running experiment.
        
        Args:
            experiment_id: Experiment identifier
            reason: Reason for stopping the experiment
            
        Returns:
            bool: True if experiment stopped successfully
        """
        try:
            if experiment_id not in self.active_experiments:
                return False
            
            experiment = self.active_experiments[experiment_id]
            
            if experiment['status'] != ExperimentStatus.RUNNING:
                return False
            
            # Update experiment status
            experiment['status'] = ExperimentStatus.COMPLETED
            experiment['end_date'] = datetime.now()
            
            # Update database
            await self._update_experiment_status(experiment_id, ExperimentStatus.COMPLETED)
            
            # Perform final analysis
            final_summary = await self.analyze_experiment(experiment_id)
            
            # Log to MLflow
            with mlflow.start_run(experiment_id=experiment['mlflow_experiment_id']):
                mlflow.log_params({
                    'end_date': experiment['end_date'].isoformat(),
                    'status': ExperimentStatus.COMPLETED.value,
                    'stop_reason': reason
                })
            
            logger.info(f"A/B testing experiment {experiment_id} stopped: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop experiment {experiment_id}: {str(e)}")
            return False
    
    def _validate_experiment_config(self, config: ExperimentConfig) -> bool:
        """Validate experiment configuration."""
        try:
            # Check traffic percentages sum to 100%
            total_traffic = sum(variant.traffic_percentage for variant in config.variants)
            if abs(total_traffic - 100.0) > 0.01:
                logger.error(f"Traffic percentages must sum to 100%, got {total_traffic}")
                return False
            
            # Check for exactly one control variant
            control_variants = [v for v in config.variants if v.is_control]
            if len(control_variants) != 1:
                logger.error("Exactly one control variant must be specified")
                return False
            
            # Check minimum sample size
            if config.minimum_sample_size < 100:
                logger.error("Minimum sample size must be at least 100")
                return False
            
            # Check confidence level
            if not 0.8 <= config.confidence_level <= 0.99:
                logger.error("Confidence level must be between 0.8 and 0.99")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    async def _perform_statistical_tests(self, results: List[ExperimentResult],
                                       experiment_data: pd.DataFrame,
                                       config: ExperimentConfig) -> List[ExperimentResult]:
        """Perform statistical significance tests between variants."""
        try:
            # Find control variant
            control_result = next((r for r in results 
                                 for v in config.variants 
                                 if v.variant_id == r.variant_id and v.is_control), None)
            
            if not control_result:
                return results
            
            # Get control data
            control_data = experiment_data[
                experiment_data['variant_id'] == control_result.variant_id
            ][config.primary_metric].values
            
            # Test each variant against control
            for result in results:
                if result.variant_id == control_result.variant_id:
                    continue  # Skip control vs control
                
                # Get variant data
                variant_data = experiment_data[
                    experiment_data['variant_id'] == result.variant_id
                ][config.primary_metric].values
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(variant_data, control_data)
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(variant_data) - 1) * np.var(variant_data, ddof=1) +
                                    (len(control_data) - 1) * np.var(control_data, ddof=1)) /
                                   (len(variant_data) + len(control_data) - 2))
                
                effect_size = (np.mean(variant_data) - np.mean(control_data)) / pooled_std
                
                # Update result
                result.p_value = p_value
                result.effect_size = effect_size
                result.statistical_significance = p_value < config.significance_threshold
            
            return results
            
        except Exception as e:
            logger.error(f"Statistical testing failed: {str(e)}")
            return results
    
    def _perform_t_test(self, group_a: np.ndarray, group_b: np.ndarray) -> Tuple[float, float]:
        """Perform independent t-test."""
        t_stat, p_value = stats.ttest_ind(group_a, group_b)
        return t_stat, p_value
    
    def _perform_mann_whitney_test(self, group_a: np.ndarray, group_b: np.ndarray) -> Tuple[float, float]:
        """Perform Mann-Whitney U test."""
        u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
        return u_stat, p_value
    
    def _perform_chi_square_test(self, group_a: np.ndarray, group_b: np.ndarray) -> Tuple[float, float]:
        """Perform Chi-square test."""
        # Convert to contingency table for categorical data
        contingency_table = np.array([[np.sum(group_a), len(group_a) - np.sum(group_a)],
                                     [np.sum(group_b), len(group_b) - np.sum(group_b)]])
        chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
        return chi2_stat, p_value
    
    def _perform_bootstrap_test(self, group_a: np.ndarray, group_b: np.ndarray,
                              n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Perform bootstrap test."""
        # Calculate observed difference
        observed_diff = np.mean(group_a) - np.mean(group_b)
        
        # Combine groups for null hypothesis
        combined = np.concatenate([group_a, group_b])
        n_a, n_b = len(group_a), len(group_b)
        
        # Bootstrap sampling
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            np.random.shuffle(combined)
            bootstrap_a = combined[:n_a]
            bootstrap_b = combined[n_a:n_a+n_b]
            bootstrap_diff = np.mean(bootstrap_a) - np.mean(bootstrap_b)
            bootstrap_diffs.append(bootstrap_diff)
        
        # Calculate p-value
        bootstrap_diffs = np.array(bootstrap_diffs)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        return observed_diff, p_value
    
    def _calculate_confidence_interval(self, data: np.ndarray, confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for the mean."""
        mean = np.mean(data)
        sem = stats.sem(data)
        h = sem * stats.t.ppf((1 + confidence_level) / 2., len(data) - 1)
        return (mean - h, mean + h)
    
    def _determine_winning_variant(self, results: List[ExperimentResult], 
                                 config: ExperimentConfig) -> Optional[str]:
        """Determine the winning variant based on statistical significance."""
        # Find control variant
        control_result = next((r for r in results 
                             for v in config.variants 
                             if v.variant_id == r.variant_id and v.is_control), None)
        
        if not control_result:
            return None
        
        # Find best performing significant variant
        significant_variants = [r for r in results if r.statistical_significance]
        
        if not significant_variants:
            return control_result.variant_id  # No significant improvement
        
        # Return variant with highest primary metric value
        best_variant = max(significant_variants, key=lambda x: x.primary_metric_value)
        return best_variant.variant_id
    
    def _generate_experiment_recommendations(self, results: List[ExperimentResult],
                                           config: ExperimentConfig) -> List[str]:
        """Generate recommendations based on experiment results."""
        recommendations = []
        
        # Check sample sizes
        min_sample_size = min(result.sample_size for result in results)
        if min_sample_size < config.minimum_sample_size:
            recommendations.append(
                f"Increase sample size. Minimum variant has only {min_sample_size} samples, "
                f"need {config.minimum_sample_size}"
            )
        
        # Check for statistical significance
        significant_results = [r for r in results if r.statistical_significance]
        if not significant_results:
            recommendations.append("No statistically significant differences found. Consider:")
            recommendations.append("- Running the experiment longer to collect more data")
            recommendations.append("- Reviewing the effect size to determine practical significance")
            recommendations.append("- Adjusting the significance threshold if appropriate")
        
        # Check effect sizes
        large_effects = [r for r in results if abs(r.effect_size) > 0.8]
        if large_effects:
            recommendations.append("Large effect sizes detected - consider early implementation")
        
        # Performance recommendations
        best_result = max(results, key=lambda x: x.primary_metric_value)
        recommendations.append(f"Best performing variant: {best_result.variant_id}")
        
        return recommendations
    
    def _calculate_statistical_summary(self, results: List[ExperimentResult],
                                     config: ExperimentConfig) -> Dict[str, Any]:
        """Calculate statistical summary of experiment."""
        return {
            'total_variants': len(results),
            'significant_variants': len([r for r in results if r.statistical_significance]),
            'average_effect_size': np.mean([abs(r.effect_size) for r in results]),
            'min_p_value': min([r.p_value for r in results if r.p_value > 0]),
            'max_effect_size': max([abs(r.effect_size) for r in results]),
            'confidence_level': config.confidence_level,
            'significance_threshold': config.significance_threshold
        }
    
    def _calculate_experiment_duration(self, experiment: Dict[str, Any]) -> int:
        """Calculate experiment duration in days."""
        if not experiment['start_date']:
            return 0
        
        end_date = experiment.get('end_date', datetime.now())
        return (end_date - experiment['start_date']).days
    
    async def _store_experiment_config(self, config: ExperimentConfig) -> None:
        """Store experiment configuration in database."""
        try:
            query = text("""
                INSERT INTO ab_test_experiments 
                (experiment_id, experiment_name, description, config, status, created_at)
                VALUES (:experiment_id, :experiment_name, :description, :config, :status, :created_at)
            """)
            
            with self.engine.connect() as conn:
                conn.execute(query, {
                    'experiment_id': config.experiment_id,
                    'experiment_name': config.experiment_name,
                    'description': config.description,
                    'config': json.dumps(asdict(config)),
                    'status': ExperimentStatus.DRAFT.value,
                    'created_at': datetime.now()
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store experiment config: {str(e)}")
    
    async def _update_experiment_status(self, experiment_id: str, status: ExperimentStatus) -> None:
        """Update experiment status in database."""
        try:
            query = text("""
                UPDATE ab_test_experiments 
                SET status = :status, updated_at = :updated_at
                WHERE experiment_id = :experiment_id
            """)
            
            with self.engine.connect() as conn:
                conn.execute(query, {
                    'experiment_id': experiment_id,
                    'status': status.value,
                    'updated_at': datetime.now()
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update experiment status: {str(e)}")
    
    async def _store_experiment_metrics(self, experiment_id: str, variant_id: str,
                                      user_id: str, metrics: Dict[str, float]) -> None:
        """Store experiment metrics in database."""
        try:
            query = text("""
                INSERT INTO ab_test_metrics 
                (experiment_id, variant_id, user_id, metrics, recorded_at)
                VALUES (:experiment_id, :variant_id, :user_id, :metrics, :recorded_at)
            """)
            
            with self.engine.connect() as conn:
                conn.execute(query, {
                    'experiment_id': experiment_id,
                    'variant_id': variant_id,
                    'user_id': user_id,
                    'metrics': json.dumps(metrics),
                    'recorded_at': datetime.now()
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store experiment metrics: {str(e)}")
    
    async def _fetch_experiment_data(self, experiment_id: str) -> Optional[pd.DataFrame]:
        """Fetch experiment data from database."""
        try:
            query = text("""
                SELECT variant_id, user_id, metrics, recorded_at
                FROM ab_test_metrics 
                WHERE experiment_id = :experiment_id
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'experiment_id': experiment_id})
                rows = result.fetchall()
                
                if not rows:
                    return None
                
                # Convert to DataFrame
                data = []
                for row in rows:
                    metrics = json.loads(row.metrics)
                    record = {
                        'variant_id': row.variant_id,
                        'user_id': row.user_id,
                        'recorded_at': row.recorded_at,
                        **metrics
                    }
                    data.append(record)
                
                return pd.DataFrame(data)
                
        except Exception as e:
            logger.error(f"Failed to fetch experiment data: {str(e)}")
            return None
    
    async def _log_experiment_results(self, summary: ExperimentSummary) -> None:
        """Log experiment results to MLflow."""
        try:
            experiment = self.active_experiments[summary.experiment_id]
            
            with mlflow.start_run(experiment_id=experiment['mlflow_experiment_id']):
                # Log summary metrics
                mlflow.log_metrics({
                    'total_samples': summary.total_samples,
                    'duration_days': summary.duration_days,
                    'confidence_level': summary.confidence_level
                })
                
                # Log variant results
                for result in summary.results:
                    mlflow.log_metrics({
                        f'{result.variant_id}_sample_size': result.sample_size,
                        f'{result.variant_id}_primary_metric': result.primary_metric_value,
                        f'{result.variant_id}_p_value': result.p_value,
                        f'{result.variant_id}_effect_size': result.effect_size
                    })
                
                # Log statistical summary
                for key, value in summary.statistical_summary.items():
                    mlflow.log_metric(f'summary_{key}', value)
                
                # Log recommendations as artifact
                mlflow.log_dict(summary.recommendations, "recommendations.json")
                
        except Exception as e:
            logger.error(f"Failed to log experiment results: {str(e)}")


class TrafficRouter:
    """Traffic routing component for A/B testing."""
    
    def __init__(self):
        self.experiment_configs = {}
    
    async def configure_experiment(self, experiment_id: str, config: ExperimentConfig) -> None:
        """Configure traffic routing for an experiment."""
        self.experiment_configs[experiment_id] = config
    
    async def route_user(self, experiment_id: str, user_id: str, 
                        context: Dict[str, Any] = None) -> Optional[str]:
        """Route user to appropriate variant."""
        if experiment_id not in self.experiment_configs:
            return None
        
        config = self.experiment_configs[experiment_id]
        
        if config.traffic_split_method == TrafficSplitMethod.RANDOM:
            return self._random_routing(config.variants)
        elif config.traffic_split_method == TrafficSplitMethod.HASH_BASED:
            return self._hash_based_routing(user_id, config.variants)
        elif config.traffic_split_method == TrafficSplitMethod.TIME_BASED:
            return self._time_based_routing(config.variants)
        else:
            return self._random_routing(config.variants)
    
    def _random_routing(self, variants: List[ModelVariant]) -> str:
        """Random traffic routing."""
        rand_val = np.random.random() * 100
        cumulative = 0
        
        for variant in variants:
            cumulative += variant.traffic_percentage
            if rand_val <= cumulative:
                return variant.variant_id
        
        return variants[-1].variant_id  # Fallback
    
    def _hash_based_routing(self, user_id: str, variants: List[ModelVariant]) -> str:
        """Hash-based consistent routing."""
        hash_val = hash(user_id) % 100
        cumulative = 0
        
        for variant in variants:
            cumulative += variant.traffic_percentage
            if hash_val < cumulative:
                return variant.variant_id
        
        return variants[-1].variant_id  # Fallback
    
    def _time_based_routing(self, variants: List[ModelVariant]) -> str:
        """Time-based routing (alternating by hour)."""
        current_hour = datetime.now().hour
        variant_index = current_hour % len(variants)
        return variants[variant_index].variant_id