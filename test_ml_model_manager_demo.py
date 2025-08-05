#!/usr/bin/env python3
"""
ML Model Manager Demo

This script demonstrates the comprehensive ML model lifecycle management capabilities
of the Stock Analysis System, including:

1. Model registration and versioning
2. Model promotion workflows
3. Drift detection and monitoring
4. Automated retraining scheduling
5. Model comparison and A/B testing

Usage:
    python test_ml_model_manager_demo.py
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from stock_analysis_system.analysis.ml_model_manager import (
    MLModelManager,
    ModelMetrics,
    ModelInfo
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
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(20)]
    
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


def train_model(X_train, y_train, model_type='random_forest'):
    """Train a model on the training data."""
    logger.info(f"Training {model_type} model...")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    logger.info(f"{model_type} model training completed")
    
    return model


def calculate_metrics(model, X_val, y_val):
    """Calculate model performance metrics."""
    logger.info("Calculating model performance metrics...")
    
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = ModelMetrics(
        accuracy=accuracy_score(y_val, predictions),
        precision=precision_score(y_val, predictions, average='weighted'),
        recall=recall_score(y_val, predictions, average='weighted'),
        f1_score=f1_score(y_val, predictions, average='weighted'),
        custom_metrics={
            'auc_roc': 0.85,  # Placeholder - would calculate actual AUC-ROC
            'log_loss': 0.35   # Placeholder - would calculate actual log loss
        }
    )
    
    logger.info(f"Model metrics: Accuracy={metrics.accuracy:.4f}, "
               f"Precision={metrics.precision:.4f}, "
               f"Recall={metrics.recall:.4f}, "
               f"F1={metrics.f1_score:.4f}")
    
    return metrics


def create_drifted_data(X_original, drift_strength=2.0):
    """Create drifted data by adding noise and shifting distribution."""
    logger.info(f"Creating drifted data with strength {drift_strength}...")
    
    # Add noise and shift distribution
    X_drifted = X_original.copy()
    X_drifted = X_drifted + np.random.normal(0, drift_strength, X_drifted.shape)
    X_drifted = X_drifted + drift_strength
    
    logger.info("Drifted data created")
    return X_drifted


async def demonstrate_model_registration(ml_manager, data):
    """Demonstrate model registration and versioning."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING MODEL REGISTRATION AND VERSIONING")
    logger.info("="*60)
    
    # Train and register first model (Random Forest)
    rf_model = train_model(data['X_train'], data['y_train'], 'random_forest')
    rf_metrics = calculate_metrics(rf_model, data['X_val'], data['y_val'])
    
    rf_model_id = await ml_manager.register_model(
        model_name="stock_predictor",
        model_object=rf_model,
        metrics=rf_metrics,
        tags={
            "model_type": "random_forest",
            "version": "1.0",
            "purpose": "stock_prediction"
        },
        description="Random Forest model for stock prediction",
        artifacts={
            "feature_importance": rf_model.feature_importances_.tolist(),
            "training_config": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            }
        }
    )
    
    logger.info(f"Random Forest model registered with ID: {rf_model_id}")
    
    # Train and register second model (Gradient Boosting)
    gb_model = train_model(data['X_train'], data['y_train'], 'gradient_boosting')
    gb_metrics = calculate_metrics(gb_model, data['X_val'], data['y_val'])
    
    gb_model_id = await ml_manager.register_model(
        model_name="stock_predictor",
        model_object=gb_model,
        metrics=gb_metrics,
        tags={
            "model_type": "gradient_boosting",
            "version": "2.0",
            "purpose": "stock_prediction"
        },
        description="Gradient Boosting model for stock prediction",
        artifacts={
            "feature_importance": gb_model.feature_importances_.tolist(),
            "training_config": {
                "n_estimators": 100,
                "max_depth": 6,
                "random_state": 42
            }
        }
    )
    
    logger.info(f"Gradient Boosting model registered with ID: {gb_model_id}")
    
    return rf_model_id, gb_model_id


async def demonstrate_model_promotion(ml_manager, model_ids):
    """Demonstrate model promotion workflows."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING MODEL PROMOTION WORKFLOWS")
    logger.info("="*60)
    
    rf_model_id, gb_model_id = model_ids
    
    # Promote Random Forest model to production
    logger.info(f"Promoting {rf_model_id} to production...")
    success = await ml_manager.promote_model_to_production(rf_model_id)
    
    if success:
        logger.info(f"✓ {rf_model_id} successfully promoted to production")
        
        # Check model status
        model_info = await ml_manager.get_model_info(rf_model_id)
        logger.info(f"Model status: {model_info.status}")
    else:
        logger.error(f"✗ Failed to promote {rf_model_id} to production")
    
    # Later, promote Gradient Boosting model (this will archive the previous production model)
    logger.info(f"Promoting {gb_model_id} to production...")
    success = await ml_manager.promote_model_to_production(gb_model_id)
    
    if success:
        logger.info(f"✓ {gb_model_id} successfully promoted to production")
        logger.info(f"Previous production model ({rf_model_id}) has been archived")


async def demonstrate_drift_detection(ml_manager, model_ids, data):
    """Demonstrate model drift detection."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING MODEL DRIFT DETECTION")
    logger.info("="*60)
    
    rf_model_id, gb_model_id = model_ids
    
    # Test 1: No drift scenario
    logger.info("Testing drift detection with no drift...")
    
    # Use similar data (no drift)
    reference_data = data['X_train'][:400]
    new_data_no_drift = data['X_train'][400:600]
    
    drift_result = await ml_manager.detect_model_drift(
        model_id=rf_model_id,
        new_data=new_data_no_drift,
        reference_data=reference_data,
        feature_names=data['feature_names']
    )
    
    logger.info(f"No drift test - Drift detected: {drift_result.drift_detected}")
    logger.info(f"No drift test - Drift score: {drift_result.drift_score:.4f}")
    logger.info(f"No drift test - Confidence: {drift_result.confidence:.4f}")
    
    # Test 2: Drift scenario
    logger.info("Testing drift detection with significant drift...")
    
    # Create drifted data
    new_data_with_drift = create_drifted_data(data['X_test'], drift_strength=3.0)
    
    drift_result = await ml_manager.detect_model_drift(
        model_id=rf_model_id,
        new_data=new_data_with_drift,
        reference_data=reference_data,
        feature_names=data['feature_names']
    )
    
    logger.info(f"Drift test - Drift detected: {drift_result.drift_detected}")
    logger.info(f"Drift test - Drift score: {drift_result.drift_score:.4f}")
    logger.info(f"Drift test - Confidence: {drift_result.confidence:.4f}")
    
    # Show feature-level drift details
    if 'feature_drift_details' in drift_result.details:
        logger.info("Feature-level drift analysis:")
        for feature_name, details in list(drift_result.details['feature_drift_details'].items())[:5]:
            logger.info(f"  {feature_name}: drift_score={details['drift_score']:.4f}, "
                       f"ks_p_value={details['ks_p_value']:.4f}")


async def demonstrate_retraining_scheduling(ml_manager, model_ids):
    """Demonstrate automated retraining scheduling."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING AUTOMATED RETRAINING SCHEDULING")
    logger.info("="*60)
    
    rf_model_id, gb_model_id = model_ids
    
    # Schedule periodic retraining
    logger.info(f"Scheduling periodic retraining for {rf_model_id}...")
    await ml_manager.schedule_retraining(
        model_id=rf_model_id,
        schedule_type="periodic",
        schedule_config={"interval_days": 30}
    )
    logger.info("✓ Periodic retraining scheduled (every 30 days)")
    
    # Schedule drift-based retraining
    logger.info(f"Scheduling drift-based retraining for {gb_model_id}...")
    await ml_manager.schedule_retraining(
        model_id=gb_model_id,
        schedule_type="drift_based",
        schedule_config={"drift_threshold": 0.1}
    )
    logger.info("✓ Drift-based retraining scheduled")
    
    # Check which models are due for retraining
    logger.info("Checking for models due for retraining...")
    due_models = await ml_manager.check_retraining_due()
    
    if due_models:
        logger.info(f"Models due for retraining: {due_models}")
    else:
        logger.info("No models currently due for retraining")
    
    # Simulate a model being due by manipulating the schedule
    logger.info("Simulating overdue retraining scenario...")
    past_date = datetime.now() - timedelta(days=1)
    ml_manager.retraining_schedule[rf_model_id]['next_retrain'] = past_date
    
    due_models = await ml_manager.check_retraining_due()
    logger.info(f"After simulation - Models due for retraining: {due_models}")


async def demonstrate_model_comparison(ml_manager, model_ids, data):
    """Demonstrate model comparison and A/B testing."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING MODEL COMPARISON AND A/B TESTING")
    logger.info("="*60)
    
    rf_model_id, gb_model_id = model_ids
    
    # Compare models on test data
    logger.info("Comparing models on test dataset...")
    
    comparison_results = await ml_manager.compare_models(
        model_ids=[rf_model_id, gb_model_id],
        test_data=data['X_test'],
        test_labels=data['y_test'],
        metrics=['accuracy', 'precision', 'recall', 'f1_score']
    )
    
    logger.info("Model comparison results:")
    for model_id, metrics in comparison_results.items():
        if 'error' not in metrics:
            logger.info(f"  {model_id}:")
            for metric_name, metric_value in metrics.items():
                logger.info(f"    {metric_name}: {metric_value:.4f}")
        else:
            logger.error(f"  {model_id}: Error - {metrics['error']}")
    
    # Determine best model
    if len(comparison_results) >= 2 and all('error' not in m for m in comparison_results.values()):
        best_model_id = max(comparison_results.keys(), 
                           key=lambda k: comparison_results[k]['accuracy'])
        logger.info(f"Best performing model: {best_model_id}")


async def demonstrate_model_management(ml_manager, model_ids):
    """Demonstrate model management operations."""
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATING MODEL MANAGEMENT OPERATIONS")
    logger.info("="*60)
    
    # List all models
    logger.info("Listing all registered models...")
    all_models = await ml_manager.list_models()
    
    for model_info in all_models:
        logger.info(f"  Model: {model_info.model_id}")
        logger.info(f"    Name: {model_info.model_name}")
        logger.info(f"    Status: {model_info.status}")
        logger.info(f"    Accuracy: {model_info.metrics.accuracy:.4f}")
        logger.info(f"    Created: {model_info.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"    Drift Score: {model_info.drift_score:.4f}")
    
    # List only production models
    logger.info("\nListing production models...")
    prod_models = await ml_manager.list_models(status_filter="production")
    
    for model_info in prod_models:
        logger.info(f"  Production Model: {model_info.model_id}")
        logger.info(f"    Accuracy: {model_info.metrics.accuracy:.4f}")
    
    # Archive an old model
    rf_model_id, gb_model_id = model_ids
    logger.info(f"\nArchiving model {rf_model_id}...")
    
    success = await ml_manager.archive_model(rf_model_id)
    if success:
        logger.info(f"✓ Model {rf_model_id} archived successfully")
        
        # Verify status change
        model_info = await ml_manager.get_model_info(rf_model_id)
        logger.info(f"Model status after archiving: {model_info.status}")
    else:
        logger.error(f"✗ Failed to archive model {rf_model_id}")


async def main():
    """Main demonstration function."""
    logger.info("Starting ML Model Manager Comprehensive Demo")
    logger.info("=" * 80)
    
    # Create temporary MLflow directory
    temp_dir = tempfile.mkdtemp()
    mlflow_uri = f"file://{temp_dir}/mlruns"
    
    try:
        # Initialize ML Model Manager
        logger.info(f"Initializing ML Model Manager with tracking URI: {mlflow_uri}")
        ml_manager = MLModelManager(mlflow_tracking_uri=mlflow_uri)
        
        # Create sample data
        data = create_sample_data()
        
        # Demonstrate all features
        model_ids = await demonstrate_model_registration(ml_manager, data)
        await demonstrate_model_promotion(ml_manager, model_ids)
        await demonstrate_drift_detection(ml_manager, model_ids, data)
        await demonstrate_retraining_scheduling(ml_manager, model_ids)
        await demonstrate_model_comparison(ml_manager, model_ids, data)
        await demonstrate_model_management(ml_manager, model_ids)
        
        logger.info("\n" + "="*80)
        logger.info("ML Model Manager Demo completed successfully!")
        logger.info("="*80)
        
        # Summary
        logger.info("\nDemo Summary:")
        logger.info("✓ Model registration and versioning")
        logger.info("✓ Model promotion workflows")
        logger.info("✓ Drift detection and monitoring")
        logger.info("✓ Automated retraining scheduling")
        logger.info("✓ Model comparison and A/B testing")
        logger.info("✓ Model management operations")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise
    
    finally:
        # Cleanup temporary directory
        try:
            shutil.rmtree(temp_dir)
            logger.info("Temporary MLflow directory cleaned up")
        except Exception as e:
            logger.warning(f"Could not clean up temporary directory: {e}")


if __name__ == "__main__":
    asyncio.run(main())