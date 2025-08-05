#!/usr/bin/env python3
"""
ML Model Manager Integration Test

Simple integration test to verify the ML Model Manager works correctly
with the existing stock analysis system.
"""

import asyncio
import tempfile
import shutil
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

from stock_analysis_system.analysis.ml_model_manager import (
    MLModelManager,
    ModelMetrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_integration():
    """Test basic ML Model Manager integration."""
    logger.info("Starting ML Model Manager Integration Test")
    
    # Create temporary MLflow directory
    temp_dir = tempfile.mkdtemp()
    mlflow_uri = f"file://{temp_dir}/mlruns"
    
    try:
        # Initialize ML Model Manager
        ml_manager = MLModelManager(mlflow_tracking_uri=mlflow_uri)
        logger.info("✓ ML Model Manager initialized")
        
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        logger.info("✓ Sample model trained")
        
        # Create metrics
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        
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
            model_name="integration_test_model",
            model_object=model,
            metrics=metrics,
            tags={"test": "integration"},
            description="Integration test model"
        )
        logger.info(f"✓ Model registered with ID: {model_id}")
        
        # Promote to production
        success = await ml_manager.promote_model_to_production(model_id)
        assert success, "Model promotion failed"
        logger.info("✓ Model promoted to production")
        
        # Test drift detection
        drift_result = await ml_manager.detect_model_drift(
            model_id=model_id,
            new_data=X[:50],
            reference_data=X[50:]
        )
        assert drift_result is not None, "Drift detection failed"
        logger.info(f"✓ Drift detection completed (score: {drift_result.drift_score:.4f})")
        
        # Schedule retraining
        await ml_manager.schedule_retraining(
            model_id=model_id,
            schedule_type="periodic",
            schedule_config={"interval_days": 7}
        )
        logger.info("✓ Retraining scheduled")
        
        # Load model
        loaded_model = await ml_manager.load_model(model_id, stage="Production")
        assert loaded_model is not None, "Model loading failed"
        logger.info("✓ Model loaded successfully")
        
        # Test predictions
        test_predictions = loaded_model.predict(X[:10])
        assert len(test_predictions) == 10, "Prediction failed"
        logger.info("✓ Model predictions working")
        
        # List models
        models = await ml_manager.list_models()
        assert len(models) >= 1, "Model listing failed"
        logger.info(f"✓ Model listing working ({len(models)} models found)")
        
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
    success = await test_basic_integration()
    if success:
        print("\n✅ ML Model Manager Integration Test: PASSED")
    else:
        print("\n❌ ML Model Manager Integration Test: FAILED")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())