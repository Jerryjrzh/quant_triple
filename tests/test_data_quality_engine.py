"""Tests for Data Quality Engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import patch, Mock
import tempfile
import os

from stock_analysis_system.data.data_quality_engine import (
    EnhancedDataQualityEngine,
    CompletenessRule,
    ConsistencyRule,
    TimelinessRule,
    DuplicateRule,
    MLAnomalyDetector,
    DataQualityIssueType,
    DataQualitySeverity
)


class TestCompletenessRule:
    """Test completeness rule."""
    
    def test_missing_column(self):
        """Test detection of missing required columns."""
        rule = CompletenessRule(['stock_code', 'trade_date', 'close_price'])
        data = pd.DataFrame({
            'stock_code': ['000001.SZ'],
            'trade_date': [date.today()]
            # Missing 'close_price'
        })
        
        issues = rule.validate(data)
        assert len(issues) == 1
        assert issues[0].issue_type == DataQualityIssueType.MISSING_DATA
        assert issues[0].severity == DataQualitySeverity.CRITICAL
        assert 'close_price' in issues[0].description
    
    def test_missing_values_within_threshold(self):
        """Test missing values within acceptable threshold."""
        rule = CompletenessRule(['close_price'], max_missing_ratio=0.1)
        data = pd.DataFrame({
            'close_price': [10.0, 11.0, None, 12.0, 13.0]  # 20% missing, but threshold is 10%
        })
        
        issues = rule.validate(data)
        assert len(issues) == 1
        assert issues[0].issue_type == DataQualityIssueType.MISSING_DATA
        assert '20.00%' in issues[0].description
    
    def test_no_missing_values(self):
        """Test data with no missing values."""
        rule = CompletenessRule(['close_price'])
        data = pd.DataFrame({
            'close_price': [10.0, 11.0, 12.0, 13.0]
        })
        
        issues = rule.validate(data)
        assert len(issues) == 0


class TestConsistencyRule:
    """Test consistency rule."""
    
    def test_ohlc_consistency_violations(self):
        """Test OHLC price relationship violations."""
        rule = ConsistencyRule()
        data = pd.DataFrame({
            'open_price': [10.0, 11.0],
            'high_price': [9.0, 12.0],  # First row: high < open (violation)
            'low_price': [8.0, 10.5],
            'close_price': [9.5, 11.5]
        })
        
        issues = rule.validate(data)
        high_issues = [i for i in issues if 'high_price is not the highest' in i.description]
        assert len(high_issues) == 1
        assert 0 in high_issues[0].affected_rows
    
    def test_negative_values(self):
        """Test detection of negative values."""
        rule = ConsistencyRule()
        data = pd.DataFrame({
            'close_price': [10.0, -5.0, 12.0],  # Negative price
            'volume': [1000, 2000, -500]  # Negative volume
        })
        
        issues = rule.validate(data)
        negative_issues = [i for i in issues if 'negative values' in i.description]
        assert len(negative_issues) == 2  # One for price, one for volume
    
    def test_valid_ohlc_data(self):
        """Test valid OHLC data."""
        rule = ConsistencyRule()
        data = pd.DataFrame({
            'open_price': [10.0, 11.0],
            'high_price': [12.0, 13.0],
            'low_price': [9.0, 10.5],
            'close_price': [11.0, 12.5],
            'volume': [1000, 2000]
        })
        
        issues = rule.validate(data)
        assert len(issues) == 0


class TestTimelinessRule:
    """Test timeliness rule."""
    
    def test_stale_data(self):
        """Test detection of stale data."""
        rule = TimelinessRule('trade_date', max_age_days=7)
        old_date = datetime.now() - timedelta(days=10)
        data = pd.DataFrame({
            'trade_date': [old_date, datetime.now()]
        })
        
        issues = rule.validate(data)
        stale_issues = [i for i in issues if i.issue_type == DataQualityIssueType.STALE_DATA]
        assert len(stale_issues) == 1
        assert '1 rows with data older than 7 days' in stale_issues[0].description
    
    def test_future_dates(self):
        """Test detection of future dates."""
        rule = TimelinessRule('trade_date')
        future_date = datetime.now() + timedelta(days=1)
        data = pd.DataFrame({
            'trade_date': [datetime.now(), future_date]
        })
        
        issues = rule.validate(data)
        future_issues = [i for i in issues if 'future dates' in i.description]
        assert len(future_issues) == 1
    
    def test_invalid_date_format(self):
        """Test handling of invalid date formats."""
        rule = TimelinessRule('trade_date')
        data = pd.DataFrame({
            'trade_date': ['invalid_date', '2024-01-01']
        })
        
        issues = rule.validate(data)
        format_issues = [i for i in issues if i.issue_type == DataQualityIssueType.INVALID_FORMAT]
        assert len(format_issues) == 1


class TestDuplicateRule:
    """Test duplicate rule."""
    
    def test_duplicate_detection(self):
        """Test detection of duplicate records."""
        rule = DuplicateRule(['stock_code', 'trade_date'])
        data = pd.DataFrame({
            'stock_code': ['000001.SZ', '000001.SZ', '000002.SZ'],
            'trade_date': [date.today(), date.today(), date.today()],  # First two are duplicates
            'close_price': [10.0, 10.0, 12.0]
        })
        
        issues = rule.validate(data)
        assert len(issues) == 1
        assert issues[0].issue_type == DataQualityIssueType.DUPLICATE_DATA
        assert len(issues[0].affected_rows) == 2  # Both duplicate rows
    
    def test_no_duplicates(self):
        """Test data with no duplicates."""
        rule = DuplicateRule(['stock_code', 'trade_date'])
        data = pd.DataFrame({
            'stock_code': ['000001.SZ', '000002.SZ'],
            'trade_date': [date.today(), date.today()],
            'close_price': [10.0, 12.0]
        })
        
        issues = rule.validate(data)
        assert len(issues) == 0
    
    def test_missing_key_columns(self):
        """Test handling of missing key columns."""
        rule = DuplicateRule(['stock_code', 'missing_column'])
        data = pd.DataFrame({
            'stock_code': ['000001.SZ'],
            'close_price': [10.0]
        })
        
        issues = rule.validate(data)
        assert len(issues) == 1
        assert issues[0].issue_type == DataQualityIssueType.MISSING_DATA
        assert 'missing_column' in issues[0].description


class TestMLAnomalyDetector:
    """Test ML anomaly detector."""
    
    def test_fit_and_detect(self):
        """Test fitting model and detecting anomalies."""
        # Create normal data
        np.random.seed(42)
        normal_data = pd.DataFrame({
            'price': np.random.normal(100, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        })
        
        # Add some outliers
        outlier_data = pd.DataFrame({
            'price': [200, 300],  # Extreme prices
            'volume': [5000, 6000]  # Extreme volumes
        })
        
        test_data = pd.concat([normal_data, outlier_data], ignore_index=True)
        
        detector = MLAnomalyDetector(contamination=0.1)
        detector.fit(normal_data, ['price', 'volume'])
        
        anomaly_indices, anomaly_scores = detector.detect_anomalies(test_data)
        
        # Should detect some anomalies
        assert len(anomaly_indices) > 0
        assert len(anomaly_scores) == len(anomaly_indices)
        
        # Outliers should be in the detected anomalies
        assert any(idx >= 100 for idx in anomaly_indices)  # Outliers are at indices 100, 101
    
    def test_save_and_load_model(self):
        """Test saving and loading model."""
        data = pd.DataFrame({
            'price': [100, 110, 90, 105],
            'volume': [1000, 1100, 900, 1050]
        })
        
        detector = MLAnomalyDetector()
        detector.fit(data, ['price', 'volume'])
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            try:
                detector.save_model(tmp.name)
                
                # Create new detector and load model
                new_detector = MLAnomalyDetector()
                new_detector.load_model(tmp.name)
                
                assert new_detector.is_fitted
                assert new_detector.feature_columns == ['price', 'volume']
                
                # Should be able to detect anomalies
                anomalies, scores = new_detector.detect_anomalies(data)
                assert isinstance(anomalies, list)
                assert isinstance(scores, list)
                
            finally:
                os.unlink(tmp.name)
    
    def test_not_fitted_error(self):
        """Test error when using unfitted model."""
        detector = MLAnomalyDetector()
        data = pd.DataFrame({'price': [100, 110]})
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.detect_anomalies(data)


class TestEnhancedDataQualityEngine:
    """Test enhanced data quality engine."""
    
    @pytest.fixture
    def sample_stock_data(self):
        """Create sample stock data for testing."""
        return pd.DataFrame({
            'stock_code': ['000001.SZ'] * 5,
            'trade_date': pd.date_range('2024-01-01', periods=5),
            'open_price': [10.0, 11.0, 12.0, 13.0, 14.0],
            'high_price': [11.0, 12.0, 13.0, 14.0, 15.0],
            'low_price': [9.0, 10.0, 11.0, 12.0, 13.0],
            'close_price': [10.5, 11.5, 12.5, 13.5, 14.5],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'amount': [10500, 12650, 15000, 17550, 20300]
        })
    
    def test_validate_clean_data(self, sample_stock_data):
        """Test validation of clean data."""
        engine = EnhancedDataQualityEngine()
        report = engine.validate_data(sample_stock_data, "test_dataset")
        
        assert report.dataset_name == "test_dataset"
        assert report.total_rows == 5
        assert report.total_columns == 8
        assert report.overall_score > 0.8  # Should be high quality
        assert len(report.recommendations) >= 0
    
    def test_validate_problematic_data(self):
        """Test validation of data with issues."""
        problematic_data = pd.DataFrame({
            'stock_code': ['000001.SZ', '000001.SZ', '000002.SZ'],  # Duplicate
            'trade_date': [date.today(), date.today(), None],  # Duplicate and missing
            'open_price': [10.0, 10.0, -5.0],  # Negative price
            'high_price': [9.0, 12.0, 15.0],  # First row: high < open
            'low_price': [8.0, 10.0, 12.0],
            'close_price': [None, 11.0, 14.0],  # Missing value
            'volume': [1000, 1100, 1200]
        })
        
        engine = EnhancedDataQualityEngine()
        report = engine.validate_data(problematic_data, "problematic_dataset")
        
        assert len(report.issues) > 0
        assert report.overall_score < 0.8  # Should be lower quality
        assert len(report.recommendations) > 0
        
        # Check for specific issue types
        issue_types = [issue.issue_type for issue in report.issues]
        assert DataQualityIssueType.DUPLICATE_DATA in issue_types
        assert DataQualityIssueType.MISSING_DATA in issue_types
        assert DataQualityIssueType.BUSINESS_RULE_VIOLATION in issue_types
        assert DataQualityIssueType.INCONSISTENT_DATA in issue_types
    
    def test_ml_anomaly_detection_integration(self, sample_stock_data):
        """Test ML anomaly detection integration."""
        engine = EnhancedDataQualityEngine()
        
        # Train ML detector
        engine.train_ml_detector(sample_stock_data)
        
        # Add some outliers
        outlier_data = sample_stock_data.copy()
        outlier_data.loc[len(outlier_data)] = {
            'stock_code': '000001.SZ',
            'trade_date': pd.Timestamp('2024-01-06'),
            'open_price': 100.0,  # Extreme outlier
            'high_price': 110.0,
            'low_price': 95.0,
            'close_price': 105.0,
            'volume': 10000,  # Extreme outlier
            'amount': 1050000
        }
        
        report = engine.validate_data(outlier_data, "outlier_test")
        
        # Should detect outliers
        outlier_issues = [i for i in report.issues if i.issue_type == DataQualityIssueType.OUTLIER_DATA]
        assert len(outlier_issues) > 0
    
    def test_clean_data_functionality(self):
        """Test automatic data cleaning."""
        dirty_data = pd.DataFrame({
            'stock_code': ['000001.SZ', '000001.SZ', '000002.SZ'],  # Duplicate
            'trade_date': [date.today(), date.today(), date.today()],
            'close_price': [10.0, 10.0, 12.0]
        })
        
        engine = EnhancedDataQualityEngine()
        report = engine.validate_data(dirty_data, "dirty_data")
        cleaned_data = engine.clean_data(dirty_data, report)
        
        # Should remove duplicates
        assert len(cleaned_data) < len(dirty_data)
        assert not cleaned_data.duplicated(subset=['stock_code', 'trade_date']).any()
    
    def test_custom_rules(self, sample_stock_data):
        """Test adding and removing custom rules."""
        engine = EnhancedDataQualityEngine()
        initial_rule_count = len(engine.rules)
        
        # Add custom rule with unique name
        custom_rule = CompletenessRule(['custom_column'])
        custom_rule.name = "Custom Completeness Check"  # Give it a unique name
        engine.add_rule(custom_rule)
        assert len(engine.rules) == initial_rule_count + 1
        
        # Remove the custom rule
        engine.remove_rule('Custom Completeness Check')
        assert len(engine.rules) == initial_rule_count  # Back to original count
    
    def test_model_save_load(self, sample_stock_data):
        """Test saving and loading ML model."""
        engine = EnhancedDataQualityEngine()
        engine.train_ml_detector(sample_stock_data)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            try:
                engine.save_model(tmp.name)
                
                # Create new engine and load model
                new_engine = EnhancedDataQualityEngine()
                new_engine.load_model(tmp.name)
                
                # Should be able to validate data
                report = new_engine.validate_data(sample_stock_data, "loaded_model_test")
                assert report.overall_score > 0
                
            finally:
                os.unlink(tmp.name)


if __name__ == "__main__":
    pytest.main([__file__])