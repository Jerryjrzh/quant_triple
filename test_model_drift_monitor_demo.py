#!/usr/bin/env python3
"""
Model Drift Monitor Demo

This script demonstrates the comprehensive model drift detection and monitoring capabilities
including:

1. Statistical drift detection using multiple methods (KL divergence, KS test, PSI)
2. Model performance monitoring and degradation detection
3. Automated alerting system for drift and performance issues
4. A/B testing framework for model comparison
5. Comprehensive monitoring dashboards

Usage:
    python test_model_drift_monitor_demo.py
"""

import asyncio
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from stock_analysis_system.analysis.ml_model_manager import MLModelManager, ModelMetrics
from stock_analysis_system.analysis.model_drift_monitor import (
    ModelDriftMonitor,
    DriftType,
    AlertSeverity,
    PopulationStabilityIndex
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample classification data for demonstration."""
    logger.info("Creating sample classification dataset...")
    
    # Create a classification dataset
    X, y = make_classification(
        n_samples=3000,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        random_state=42
    )
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(15)]
    
    logger.info(f"Dataset created: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_names
    }


def train_models(data):
    """Train multiple models for comparison."""
    logger.info("Training models for drift monitoring demo...")
    
    # Model A: Random Forest
    model_a = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model_a.fit(data['X_train'], data['y_train'])
    
    # Model B: Gradient Boosting
    model_b = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    model_b.fit(data['X_train'], data['y_train'])
    
    logger.info("Models trained successfully")
    return {'model_a': model_a, 'model_b': model_b}


def calculate_model_metrics(model, X_val, y_val):
    """Calculate model performance metrics."""
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    
    return ModelMetrics(
        accuracy=accuracy,
        precision=accuracy,  # Simplified for demo
        recall=accuracy,
        f1_score=accuracy,
        custom_metrics={}
    )


def create_drift_scenarios(original_data):
    """Create different drift scenarios for testing."""
    logger.info("Creating drift scenarios...")
    
    scenarios = {}
    
    # Scenario 1: No drift (similar data)
    scenarios['no_drift'] = {
        'X': original_data['X_test'][:200],
        'y': original_data['y_test'][:200],
        'description': "No drift - similar to training data"
    }
    
    # Scenario 2: Moderate data drift (feature shift)
    X_moderate_drift = original_data['X_test'][200:400].copy()
    X_moderate_drift[:, :5] += np.random.normal(0, 1, (200, 5))  # Shift first 5 features
    scenarios['moderate_drift'] = {
        'X': X_moderate_drift,
        'y': original_data['y_test'][200:400],
        'description': "Moderate drift - feature distribution shift"
    }
    
    # Scenario 3: Severe data drift (major distribution change)
    X_severe_drift = original_data['X_test'][400:600].copy()
    X_severe_drift = X_severe_drift + np.random.normal(0, 3, X_severe_drift.shape)  # Add noise
    X_severe_drift = X_severe_drift + 5  # Shift entire distribution
    scenarios['severe_drift'] = {
        'X': X_severe_drift,
        'y': original_data['y_test'][400:600],
        'description': "Severe drift - major distribution change"
    }
    
    # Scenario 4: Concept drift (label distribution change)
    X_concept_drift = original_data['X_test'][600:800].copy()
    y_concept_drift = original_data['y_test'][600:800].copy()
    # Flip some labels to simulate concept drift
    flip_indices = np.random.choice(len(y_concept_drift), size=int(0.3 * len(y_concept_drift)), replace=False)
    y_concept_drift[flip_indices] = 1 - y_concept_drift[flip_indices]
    scenarios['concept_drift'] = {
        'X': X_concept_drift,
        'y': y_concept_drift,
        'description': "Concept drift - label distribution change"
    }
    
    logger.info(f"Created {len(scenarios)} drift scenarios")
    return scenarios


async def setup_models_and_monitor(data, models):
    """Set up ML models and drift monitor."""
    logger.info("Setting up ML models and drift monitor...")
    
    # Create temporary MLflow directory
    temp_dir = tempfile.mkdtemp()
    mlflow_uri = f"file://{temp_dir}/mlruns"
    
    # Initialize ML Manager
    ml_manager = MLModelManager(mlflow_tracking_uri=mlflow_uri)
    
    # Register models
    model_a_metrics = calculate_model_metrics(models['model_a'], data['X_val'], data['y_val'])
    model_b_metrics = calculate_model_metrics(models['model_b'], data['X_val'], data['y_val'])
    
    model_a_id = await ml_manager.register_model(
        model_name="drift_demo_model_a",
        model_object=models['model_a'],
        metrics=model_a_metrics,
        tags={"type": "random_forest", "demo": "drift_monitoring"},
        description="Random Forest model for drift monitoring demo"
    )
    
    model_b_id = await ml_manager.register_model(
        model_name="drift_demo_model_b",
        model_object=models['model_b'],
        metrics=model_b_metrics,
        tags={"type": "gradient_boosting", "demo": "drift_monitoring"},
        description="Gradient Boosting model for drift monitoring demo"
    )
    
    # Promote models to production
    await ml_manager.promote_model_to_production(model_a_id)
    await ml_manager.promote_model_to_production(model_b_id)
    
    # Create drift monitor
    drift_monitor = ModelDriftMonitor(ml_manager)
    
    logger.info(f"Models registered: {model_a_id}, {model_b_id}")
    
    return drift_monitor, model_a_id, model_b_id, temp_dir


async def demonstrate_psi_calculation(data):
    """Demonstrate Population Stability Index calculation."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING POPULATION STABILITY INDEX (PSI)")
    logger.info("="*60)
    
    psi_calc = PopulationStabilityIndex()
    
    # Test 1: No drift
    reference_data = data['X_train'][:, 0]  # First feature
    similar_data = data['X_val'][:, 0]      # Similar distribution
    
    psi_no_drift = psi_calc.calculate_psi(reference_data, similar_data)
    logger.info(f"PSI (no drift): {psi_no_drift:.4f}")
    
    # Test 2: Moderate drift
    moderate_drift_data = data['X_val'][:, 0] + np.random.normal(0, 1, len(data['X_val']))
    psi_moderate = psi_calc.calculate_psi(reference_data, moderate_drift_data)
    logger.info(f"PSI (moderate drift): {psi_moderate:.4f}")
    
    # Test 3: Severe drift
    severe_drift_data = data['X_val'][:, 0] + 5  # Shift distribution
    psi_severe = psi_calc.calculate_psi(reference_data, severe_drift_data)
    logger.info(f"PSI (severe drift): {psi_severe:.4f}")
    
    logger.info("\nPSI Interpretation:")
    logger.info("  < 0.1: No significant drift")
    logger.info("  0.1-0.25: Moderate drift")
    logger.info("  > 0.25: Significant drift")


async def demonstrate_comprehensive_drift_detection(drift_monitor, model_id, data, scenarios):
    """Demonstrate comprehensive drift detection across different scenarios."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING COMPREHENSIVE DRIFT DETECTION")
    logger.info("="*60)
    
    reference_data = data['X_train'][:500]
    reference_labels = data['y_train'][:500]
    
    for scenario_name, scenario_data in scenarios.items():
        logger.info(f"\n--- Testing Scenario: {scenario_name.upper()} ---")
        logger.info(f"Description: {scenario_data['description']}")
        
        # Perform comprehensive drift detection
        drift_results = await drift_monitor.detect_comprehensive_drift(
            model_id=model_id,
            new_data=scenario_data['X'],
            reference_data=reference_data,
            new_labels=scenario_data['y'],
            reference_labels=reference_labels,
            feature_names=data['feature_names']
        )
        
        # Display results
        for drift_type, result in drift_results.items():
            logger.info(f"  {drift_type.value.replace('_', ' ').title()}:")
            logger.info(f"    Detected: {result.drift_detected}")
            logger.info(f"    Score: {result.drift_score:.4f}")
            logger.info(f"    Confidence: {result.confidence:.4f}")
            
            if drift_type == DriftType.DATA_DRIFT and 'feature_drift_details' in result.details:
                # Show top 3 features with highest drift
                feature_details = result.details['feature_drift_details']
                sorted_features = sorted(
                    feature_details.items(),
                    key=lambda x: x[1]['drift_score'],
                    reverse=True
                )[:3]
                
                logger.info("    Top drifting features:")
                for feature_name, details in sorted_features:
                    logger.info(f"      {feature_name}: {details['drift_score']:.4f}")


async def demonstrate_alert_system(drift_monitor, model_id):
    """Demonstrate the alerting system."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING ALERT SYSTEM")
    logger.info("="*60)
    
    # Get active alerts
    active_alerts = drift_monitor.get_active_alerts(model_id)
    logger.info(f"Active alerts for {model_id}: {len(active_alerts)}")
    
    for alert in active_alerts:
        logger.info(f"\nAlert: {alert.alert_id}")
        logger.info(f"  Type: {alert.drift_type.value}")
        logger.info(f"  Severity: {alert.severity.value}")
        logger.info(f"  Score: {alert.drift_score:.4f}")
        logger.info(f"  Threshold: {alert.threshold:.4f}")
        logger.info(f"  Message: {alert.message}")
        logger.info(f"  Timestamp: {alert.timestamp}")
        logger.info(f"  Acknowledged: {alert.acknowledged}")
        logger.info(f"  Resolved: {alert.resolved}")
    
    # Demonstrate alert management
    if active_alerts:
        alert_to_manage = active_alerts[0]
        logger.info(f"\nDemonstrating alert management with alert: {alert_to_manage.alert_id}")
        
        # Acknowledge alert
        success = await drift_monitor.acknowledge_alert(alert_to_manage.alert_id)
        logger.info(f"Alert acknowledged: {success}")
        
        # Resolve alert
        success = await drift_monitor.resolve_alert(alert_to_manage.alert_id)
        logger.info(f"Alert resolved: {success}")
        
        # Check status after resolution
        remaining_alerts = drift_monitor.get_active_alerts(model_id)
        logger.info(f"Remaining active alerts: {len(remaining_alerts)}")


async def demonstrate_ab_testing(drift_monitor, model_a_id, model_b_id, data):
    """Demonstrate A/B testing framework."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING A/B TESTING FRAMEWORK")
    logger.info("="*60)
    
    test_data = data['X_test']
    test_labels = data['y_test']
    
    metrics_to_test = ["accuracy", "precision", "recall", "f1_score"]
    
    for metric in metrics_to_test:
        logger.info(f"\n--- A/B Test: {metric.upper()} ---")
        
        test_id = f"ab_test_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = await drift_monitor.run_ab_test(
            test_id=test_id,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            test_data=test_data,
            test_labels=test_labels,
            metric_name=metric
        )
        
        logger.info(f"Test ID: {result.test_id}")
        logger.info(f"Model A ({model_a_id}): {result.model_a_score:.4f}")
        logger.info(f"Model B ({model_b_id}): {result.model_b_score:.4f}")
        logger.info(f"Difference: {abs(result.model_b_score - result.model_a_score):.4f}")
        logger.info(f"Statistical Significance: {result.is_significant}")
        logger.info(f"Winner: {result.winner or 'No significant difference'}")
        logger.info(f"Sample Size: {result.sample_size}")
    
    # Show all A/B test results
    logger.info("\n--- All A/B Test Results Summary ---")
    all_results = drift_monitor.get_ab_test_results()
    
    for result in all_results:
        winner_info = f" (Winner: {result.winner})" if result.winner else " (No winner)"
        logger.info(f"{result.test_id}: {result.metric_name} - "
                   f"A: {result.model_a_score:.4f}, B: {result.model_b_score:.4f}{winner_info}")


async def demonstrate_performance_monitoring(drift_monitor, model_id, data, scenarios):
    """Demonstrate performance monitoring over time."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING PERFORMANCE MONITORING")
    logger.info("="*60)
    
    # Simulate performance monitoring over time with different scenarios
    for i, (scenario_name, scenario_data) in enumerate(scenarios.items()):
        logger.info(f"\nTime period {i+1}: {scenario_name}")
        
        # Detect performance drift (this will update performance history)
        performance_result = await drift_monitor._detect_performance_drift(
            model_id, scenario_data['X'], scenario_data['y']
        )
        
        if performance_result:
            logger.info(f"  Current Accuracy: {performance_result.details['current_accuracy']:.4f}")
            logger.info(f"  Baseline Accuracy: {performance_result.details['baseline_accuracy']:.4f}")
            logger.info(f"  Performance Change: {performance_result.details['performance_change']:.4f}")
            logger.info(f"  Performance Drift Detected: {performance_result.drift_detected}")
    
    # Show performance history
    if model_id in drift_monitor.performance_history:
        logger.info(f"\n--- Performance History for {model_id} ---")
        history = drift_monitor.performance_history[model_id]
        
        for i, metrics in enumerate(history):
            logger.info(f"Measurement {i+1}: "
                       f"Accuracy={metrics.accuracy:.4f}, "
                       f"Precision={metrics.precision:.4f}, "
                       f"Recall={metrics.recall:.4f}, "
                       f"F1={metrics.f1_score:.4f}")


async def demonstrate_monitoring_dashboard(drift_monitor, model_id):
    """Demonstrate monitoring dashboard data."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING MONITORING DASHBOARD")
    logger.info("="*60)
    
    dashboard_data = await drift_monitor.get_monitoring_dashboard_data(model_id)
    
    logger.info(f"Model ID: {dashboard_data['model_id']}")
    logger.info(f"Current Status: {dashboard_data['current_status']}")
    logger.info(f"Number of Active Alerts: {len(dashboard_data['alerts'])}")
    logger.info(f"Performance History Points: {len(dashboard_data['performance_history'])}")
    logger.info(f"Drift History Points: {len(dashboard_data['drift_history'])}")
    
    # Show recent alerts
    if dashboard_data['alerts']:
        logger.info("\n--- Recent Alerts ---")
        for alert in dashboard_data['alerts'][-3:]:  # Show last 3 alerts
            logger.info(f"  {alert['drift_type']} ({alert['severity']}): {alert['message']}")
    
    # Show performance trend
    if dashboard_data['performance_history']:
        logger.info("\n--- Performance Trend (Last 5 measurements) ---")
        for metrics in dashboard_data['performance_history'][-5:]:
            logger.info(f"  {metrics['timestamp']}: Accuracy={metrics['accuracy']:.4f}")
    
    # Show drift trend
    if dashboard_data['drift_history']:
        logger.info("\n--- Drift Detection History (Last 5 checks) ---")
        for drift_check in dashboard_data['drift_history'][-5:]:
            status = "DETECTED" if drift_check['drift_detected'] else "OK"
            logger.info(f"  {drift_check['drift_type']}: {status} (Score: {drift_check['drift_score']:.4f})")


async def main():
    """Main demonstration function."""
    logger.info("Starting Model Drift Monitor Comprehensive Demo")
    logger.info("=" * 80)
    
    try:
        # Create sample data
        data = create_sample_data()
        
        # Train models
        models = train_models(data)
        
        # Create drift scenarios
        scenarios = create_drift_scenarios(data)
        
        # Set up models and monitor
        drift_monitor, model_a_id, model_b_id, temp_dir = await setup_models_and_monitor(data, models)
        
        # Demonstrate all features
        await demonstrate_psi_calculation(data)
        await demonstrate_comprehensive_drift_detection(drift_monitor, model_a_id, data, scenarios)
        await demonstrate_alert_system(drift_monitor, model_a_id)
        await demonstrate_ab_testing(drift_monitor, model_a_id, model_b_id, data)
        await demonstrate_performance_monitoring(drift_monitor, model_a_id, data, scenarios)
        await demonstrate_monitoring_dashboard(drift_monitor, model_a_id)
        
        logger.info("\n" + "="*80)
        logger.info("Model Drift Monitor Demo completed successfully!")
        logger.info("="*80)
        
        # Summary
        logger.info("\nDemo Summary:")
        logger.info("✓ Population Stability Index (PSI) calculation")
        logger.info("✓ Comprehensive drift detection (data, concept, prediction, performance)")
        logger.info("✓ Automated alerting system with severity levels")
        logger.info("✓ A/B testing framework for model comparison")
        logger.info("✓ Performance monitoring over time")
        logger.info("✓ Monitoring dashboard data generation")
        logger.info("✓ Alert management (acknowledge/resolve)")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        logger.info("Temporary files cleaned up")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())