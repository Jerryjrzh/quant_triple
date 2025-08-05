#!/usr/bin/env python3
"""
Model Drift Monitor Integration Test

Simple integration test to verify the Model Drift Monitor works correctly
with the existing stock analysis system.
"""

import asyncio
import tempfile
import shutil
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

from stock_analysis_system.analysis.ml_model_manager import MLModelManager, ModelMetrics
from stock_analysis_system.analysis.model_drift_monitor import (
    ModelDriftMonitor,
    DriftType,
    AlertSeverity,
    PopulationStabilityIndex
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_drift_monitor_integration():
    """Test basic Model Drift Monitor integration."""
    logger.info("Starting Model Drift Monitor Integration Test")
    
    # Create temporary MLflow directory
    temp_dir = tempfile.mkdtemp()
    mlflow_uri = f"file://{temp_dir}/mlruns"
    
    try:
        # Initialize ML Model Manager
        ml_manager = MLModelManager(mlflow_tracking_uri=mlflow_uri)
        logger.info("✓ ML Model Manager initialized")
        
        # Initialize Drift Monitor
        drift_monitor = ModelDriftMonitor(ml_manager)
        logger.info("✓ Model Drift Monitor initialized")
        
        # Create sample data
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X_train, X_test = X[:300], X[300:]
        y_train, y_test = y[:300], y[300:]
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        logger.info("✓ Sample model trained")
        
        # Create metrics
        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        
        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=accuracy,
            recall=accuracy,
            f1_score=accuracy,
            custom_metrics={}
        )
        logger.info("✓ Model metrics calculated")
        
        # Register model
        model_id = await ml_manager.register_model(
            model_name="drift_integration_test_model",
            model_object=model,
            metrics=metrics,
            tags={"test": "drift_integration"},
            description="Drift integration test model"
        )
        logger.info(f"✓ Model registered with ID: {model_id}")
        
        # Promote to production
        success = await ml_manager.promote_model_to_production(model_id)
        assert success, "Model promotion failed"
        logger.info("✓ Model promoted to production")
        
        # Test PSI calculation
        psi_calc = PopulationStabilityIndex()
        psi_score = psi_calc.calculate_psi(X_train[:, 0], X_test[:, 0])
        assert psi_score >= 0.0, "PSI calculation failed"
        logger.info(f"✓ PSI calculation working (score: {psi_score:.4f})")
        
        # Test comprehensive drift detection
        drift_results = await drift_monitor.detect_comprehensive_drift(
            model_id=model_id,
            new_data=X_test,
            reference_data=X_train,
            new_labels=y_test,
            reference_labels=y_train,
            feature_names=[f'feature_{i}' for i in range(10)]
        )
        
        assert isinstance(drift_results, dict), "Drift detection failed"
        assert DriftType.DATA_DRIFT in drift_results, "Data drift detection missing"
        logger.info("✓ Comprehensive drift detection working")
        
        # Test alert system
        active_alerts = drift_monitor.get_active_alerts(model_id)
        logger.info(f"✓ Alert system working ({len(active_alerts)} alerts generated)")
        
        # Test A/B testing (create second model)
        model_b = RandomForestClassifier(n_estimators=20, random_state=43)
        model_b.fit(X_train, y_train)
        
        model_b_id = await ml_manager.register_model(
            model_name="drift_integration_test_model_b",
            model_object=model_b,
            metrics=metrics,
            tags={"test": "drift_integration"},
            description="Second model for A/B testing"
        )
        
        await ml_manager.promote_model_to_production(model_b_id)
        
        ab_result = await drift_monitor.run_ab_test(
            test_id="integration_ab_test",
            model_a_id=model_id,
            model_b_id=model_b_id,
            test_data=X_test,
            test_labels=y_test,
            metric_name="accuracy"
        )
        
        assert ab_result is not None, "A/B testing failed"
        assert ab_result.test_id == "integration_ab_test", "A/B test ID mismatch"
        logger.info("✓ A/B testing working")
        
        # Test monitoring dashboard
        dashboard_data = await drift_monitor.get_monitoring_dashboard_data(model_id)
        assert 'model_id' in dashboard_data, "Dashboard data generation failed"
        assert dashboard_data['model_id'] == model_id, "Dashboard model ID mismatch"
        logger.info("✓ Monitoring dashboard working")
        
        # Test alert management
        if active_alerts:
            alert_id = active_alerts[0].alert_id
            success = await drift_monitor.acknowledge_alert(alert_id)
            assert success, "Alert acknowledgment failed"
            
            success = await drift_monitor.resolve_alert(alert_id)
            assert success, "Alert resolution failed"
            logger.info("✓ Alert management working")
        
        logger.info("🎉 All integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
            logger.info("✓ Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


async def main():
    """Main test function."""
    success = await test_drift_monitor_integration()
    if success:
        print("\n✅ Model Drift Monitor Integration Test: PASSED")
    else:
        print("\n❌ Model Drift Monitor Integration Test: FAILED")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())