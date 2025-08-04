"""Enhanced Data Quality Engine with ML-based validation."""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

logger = logging.getLogger(__name__)


class DataQualityIssueType(Enum):
    """Types of data quality issues."""
    MISSING_DATA = "missing_data"
    DUPLICATE_DATA = "duplicate_data"
    OUTLIER_DATA = "outlier_data"
    INCONSISTENT_DATA = "inconsistent_data"
    STALE_DATA = "stale_data"
    INVALID_FORMAT = "invalid_format"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"


class DataQualitySeverity(Enum):
    """Severity levels for data quality issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DataQualityIssue:
    """Represents a data quality issue."""
    issue_type: DataQualityIssueType
    severity: DataQualitySeverity
    description: str
    affected_rows: List[int]
    affected_columns: List[str]
    confidence_score: float
    suggested_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    dataset_name: str
    total_rows: int
    total_columns: int
    issues: List[DataQualityIssue]
    overall_score: float
    completeness_score: float
    consistency_score: float
    timeliness_score: float
    accuracy_score: float
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)


class DataQualityRule:
    """Base class for data quality rules."""
    
    def __init__(self, name: str, description: str, severity: DataQualitySeverity):
        self.name = name
        self.description = description
        self.severity = severity
    
    def validate(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Validate data against this rule."""
        raise NotImplementedError


class CompletenessRule(DataQualityRule):
    """Rule to check data completeness."""
    
    def __init__(self, required_columns: List[str], max_missing_ratio: float = 0.05):
        super().__init__(
            name="Completeness Check",
            description="Ensures required columns have minimal missing values",
            severity=DataQualitySeverity.HIGH
        )
        self.required_columns = required_columns
        self.max_missing_ratio = max_missing_ratio
    
    def validate(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Check for missing data in required columns."""
        issues = []
        
        for column in self.required_columns:
            if column not in data.columns:
                issues.append(DataQualityIssue(
                    issue_type=DataQualityIssueType.MISSING_DATA,
                    severity=DataQualitySeverity.CRITICAL,
                    description=f"Required column '{column}' is missing",
                    affected_rows=[],
                    affected_columns=[column],
                    confidence_score=1.0,
                    suggested_action=f"Add missing column '{column}' to dataset"
                ))
                continue
            
            missing_count = data[column].isnull().sum()
            missing_ratio = missing_count / len(data)
            
            if missing_ratio > self.max_missing_ratio:
                missing_rows = data[data[column].isnull()].index.tolist()
                issues.append(DataQualityIssue(
                    issue_type=DataQualityIssueType.MISSING_DATA,
                    severity=self.severity,
                    description=f"Column '{column}' has {missing_ratio:.2%} missing values (threshold: {self.max_missing_ratio:.2%})",
                    affected_rows=missing_rows,
                    affected_columns=[column],
                    confidence_score=1.0,
                    suggested_action=f"Investigate and fill missing values in '{column}'"
                ))
        
        return issues


class ConsistencyRule(DataQualityRule):
    """Rule to check data consistency."""
    
    def __init__(self):
        super().__init__(
            name="Consistency Check",
            description="Ensures data follows expected patterns and relationships",
            severity=DataQualitySeverity.MEDIUM
        )
    
    def validate(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Check for data consistency issues."""
        issues = []
        
        # Check OHLC relationships for stock data
        if all(col in data.columns for col in ['open_price', 'high_price', 'low_price', 'close_price']):
            # High should be >= Open, Low, Close
            high_violations = data[
                (data['high_price'] < data['open_price']) |
                (data['high_price'] < data['low_price']) |
                (data['high_price'] < data['close_price'])
            ]
            
            if not high_violations.empty:
                issues.append(DataQualityIssue(
                    issue_type=DataQualityIssueType.INCONSISTENT_DATA,
                    severity=DataQualitySeverity.HIGH,
                    description=f"Found {len(high_violations)} rows where high_price is not the highest value",
                    affected_rows=high_violations.index.tolist(),
                    affected_columns=['open_price', 'high_price', 'low_price', 'close_price'],
                    confidence_score=1.0,
                    suggested_action="Review and correct OHLC price relationships"
                ))
            
            # Low should be <= Open, High, Close
            low_violations = data[
                (data['low_price'] > data['open_price']) |
                (data['low_price'] > data['high_price']) |
                (data['low_price'] > data['close_price'])
            ]
            
            if not low_violations.empty:
                issues.append(DataQualityIssue(
                    issue_type=DataQualityIssueType.INCONSISTENT_DATA,
                    severity=DataQualitySeverity.HIGH,
                    description=f"Found {len(low_violations)} rows where low_price is not the lowest value",
                    affected_rows=low_violations.index.tolist(),
                    affected_columns=['open_price', 'high_price', 'low_price', 'close_price'],
                    confidence_score=1.0,
                    suggested_action="Review and correct OHLC price relationships"
                ))
        
        # Check for negative prices or volumes
        numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount']
        for column in numeric_columns:
            if column in data.columns:
                negative_values = data[data[column] < 0]
                if not negative_values.empty:
                    issues.append(DataQualityIssue(
                        issue_type=DataQualityIssueType.BUSINESS_RULE_VIOLATION,
                        severity=DataQualitySeverity.HIGH,
                        description=f"Found {len(negative_values)} negative values in '{column}'",
                        affected_rows=negative_values.index.tolist(),
                        affected_columns=[column],
                        confidence_score=1.0,
                        suggested_action=f"Remove or correct negative values in '{column}'"
                    ))
        
        return issues


class TimelinessRule(DataQualityRule):
    """Rule to check data timeliness."""
    
    def __init__(self, date_column: str = 'trade_date', max_age_days: int = 7):
        super().__init__(
            name="Timeliness Check",
            description="Ensures data is recent and up-to-date",
            severity=DataQualitySeverity.MEDIUM
        )
        self.date_column = date_column
        self.max_age_days = max_age_days
    
    def validate(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Check for stale data."""
        issues = []
        
        if self.date_column not in data.columns:
            return issues
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[self.date_column]):
            try:
                data[self.date_column] = pd.to_datetime(data[self.date_column])
            except Exception as e:
                issues.append(DataQualityIssue(
                    issue_type=DataQualityIssueType.INVALID_FORMAT,
                    severity=DataQualitySeverity.HIGH,
                    description=f"Cannot parse dates in '{self.date_column}': {str(e)}",
                    affected_rows=list(range(len(data))),
                    affected_columns=[self.date_column],
                    confidence_score=1.0,
                    suggested_action=f"Fix date format in '{self.date_column}'"
                ))
                return issues
        
        # Check for stale data
        cutoff_date = datetime.now() - timedelta(days=self.max_age_days)
        stale_data = data[data[self.date_column] < cutoff_date]
        
        if not stale_data.empty:
            issues.append(DataQualityIssue(
                issue_type=DataQualityIssueType.STALE_DATA,
                severity=self.severity,
                description=f"Found {len(stale_data)} rows with data older than {self.max_age_days} days",
                affected_rows=stale_data.index.tolist(),
                affected_columns=[self.date_column],
                confidence_score=1.0,
                suggested_action="Update with more recent data"
            ))
        
        # Check for future dates
        future_data = data[data[self.date_column] > datetime.now()]
        if not future_data.empty:
            issues.append(DataQualityIssue(
                issue_type=DataQualityIssueType.INCONSISTENT_DATA,
                severity=DataQualitySeverity.HIGH,
                description=f"Found {len(future_data)} rows with future dates",
                affected_rows=future_data.index.tolist(),
                affected_columns=[self.date_column],
                confidence_score=1.0,
                suggested_action="Remove or correct future dates"
            ))
        
        return issues


class DuplicateRule(DataQualityRule):
    """Rule to check for duplicate records."""
    
    def __init__(self, key_columns: List[str]):
        super().__init__(
            name="Duplicate Check",
            description="Identifies duplicate records based on key columns",
            severity=DataQualitySeverity.MEDIUM
        )
        self.key_columns = key_columns
    
    def validate(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """Check for duplicate records."""
        issues = []
        
        # Check if key columns exist
        missing_columns = [col for col in self.key_columns if col not in data.columns]
        if missing_columns:
            issues.append(DataQualityIssue(
                issue_type=DataQualityIssueType.MISSING_DATA,
                severity=DataQualitySeverity.HIGH,
                description=f"Key columns missing for duplicate check: {missing_columns}",
                affected_rows=[],
                affected_columns=missing_columns,
                confidence_score=1.0,
                suggested_action=f"Add missing key columns: {missing_columns}"
            ))
            return issues
        
        # Find duplicates
        duplicates = data[data.duplicated(subset=self.key_columns, keep=False)]
        
        if not duplicates.empty:
            issues.append(DataQualityIssue(
                issue_type=DataQualityIssueType.DUPLICATE_DATA,
                severity=self.severity,
                description=f"Found {len(duplicates)} duplicate records based on {self.key_columns}",
                affected_rows=duplicates.index.tolist(),
                affected_columns=self.key_columns,
                confidence_score=1.0,
                suggested_action="Remove duplicate records or investigate data source"
            ))
        
        return issues


class MLAnomalyDetector:
    """ML-based anomaly detection for data quality."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame, feature_columns: List[str]) -> None:
        """Fit the anomaly detection model."""
        self.feature_columns = feature_columns
        
        # Select and prepare features
        features = data[feature_columns].copy()
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        features_imputed = pd.DataFrame(
            imputer.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        
        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features_imputed)
        
        # Fit Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(features_scaled)
        
        self.is_fitted = True
        logger.info(f"ML anomaly detector fitted on {len(data)} samples with {len(feature_columns)} features")
    
    def detect_anomalies(self, data: pd.DataFrame) -> Tuple[List[int], List[float]]:
        """Detect anomalies in the data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detecting anomalies")
        
        # Prepare features
        features = data[self.feature_columns].copy()
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        features_imputed = pd.DataFrame(
            imputer.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        
        # Scale features
        features_scaled = self.scaler.transform(features_imputed)
        
        # Predict anomalies
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = self.model.predict(features_scaled)
            scores = self.model.decision_function(features_scaled)
        
        # Get anomaly indices (predictions == -1 means anomaly)
        anomaly_indices = data.index[predictions == -1].tolist()
        anomaly_scores = scores[predictions == -1].tolist()
        
        return anomaly_indices, anomaly_scores
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'contamination': self.contamination,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"ML anomaly detector saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.contamination = model_data['contamination']
        self.random_state = model_data['random_state']
        self.is_fitted = True
        
        logger.info(f"ML anomaly detector loaded from {filepath}")


class EnhancedDataQualityEngine:
    """Enhanced data quality engine with ML-based validation."""
    
    def __init__(self):
        self.rules: List[DataQualityRule] = []
        self.ml_detector = MLAnomalyDetector()
        self.default_stock_rules()
    
    def default_stock_rules(self):
        """Set up default rules for stock market data."""
        # Required columns for stock data
        required_columns = ['stock_code', 'trade_date', 'close_price']
        self.add_rule(CompletenessRule(required_columns, max_missing_ratio=0.02))
        
        # Consistency checks
        self.add_rule(ConsistencyRule())
        
        # Timeliness checks
        self.add_rule(TimelinessRule('trade_date', max_age_days=30))
        
        # Duplicate checks
        self.add_rule(DuplicateRule(['stock_code', 'trade_date']))
    
    def add_rule(self, rule: DataQualityRule):
        """Add a data quality rule."""
        self.rules.append(rule)
        logger.info(f"Added data quality rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove a data quality rule by name."""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        logger.info(f"Removed data quality rule: {rule_name}")
    
    def train_ml_detector(self, training_data: pd.DataFrame, feature_columns: List[str] = None):
        """Train the ML-based anomaly detector."""
        if feature_columns is None:
            # Default features for stock data
            feature_columns = []
            numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount']
            for col in numeric_columns:
                if col in training_data.columns:
                    feature_columns.append(col)
        
        if not feature_columns:
            logger.warning("No suitable feature columns found for ML training")
            return
        
        self.ml_detector.fit(training_data, feature_columns)
        logger.info(f"ML anomaly detector trained with features: {feature_columns}")
    
    def validate_data(self, data: pd.DataFrame, dataset_name: str = "Unknown") -> DataQualityReport:
        """Perform comprehensive data quality validation."""
        logger.info(f"Starting data quality validation for dataset: {dataset_name}")
        
        all_issues = []
        
        # Apply all rules
        for rule in self.rules:
            try:
                rule_issues = rule.validate(data)
                all_issues.extend(rule_issues)
                logger.debug(f"Rule '{rule.name}' found {len(rule_issues)} issues")
            except Exception as e:
                logger.error(f"Error applying rule '{rule.name}': {str(e)}")
                all_issues.append(DataQualityIssue(
                    issue_type=DataQualityIssueType.INVALID_FORMAT,
                    severity=DataQualitySeverity.MEDIUM,
                    description=f"Rule '{rule.name}' failed to execute: {str(e)}",
                    affected_rows=[],
                    affected_columns=[],
                    confidence_score=0.5,
                    suggested_action=f"Check rule implementation for '{rule.name}'"
                ))
        
        # Apply ML-based anomaly detection if model is trained
        if self.ml_detector.is_fitted:
            try:
                anomaly_indices, anomaly_scores = self.ml_detector.detect_anomalies(data)
                if anomaly_indices:
                    all_issues.append(DataQualityIssue(
                        issue_type=DataQualityIssueType.OUTLIER_DATA,
                        severity=DataQualitySeverity.MEDIUM,
                        description=f"ML detector found {len(anomaly_indices)} potential anomalies",
                        affected_rows=anomaly_indices,
                        affected_columns=self.ml_detector.feature_columns,
                        confidence_score=0.8,
                        suggested_action="Review flagged records for potential data quality issues",
                        metadata={'anomaly_scores': anomaly_scores}
                    ))
                logger.info(f"ML anomaly detection found {len(anomaly_indices)} potential issues")
            except Exception as e:
                logger.error(f"ML anomaly detection failed: {str(e)}")
        
        # Calculate quality scores
        scores = self._calculate_quality_scores(data, all_issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues)
        
        # Create report
        report = DataQualityReport(
            dataset_name=dataset_name,
            total_rows=len(data),
            total_columns=len(data.columns),
            issues=all_issues,
            overall_score=scores['overall'],
            completeness_score=scores['completeness'],
            consistency_score=scores['consistency'],
            timeliness_score=scores['timeliness'],
            accuracy_score=scores['accuracy'],
            recommendations=recommendations
        )
        
        logger.info(f"Data quality validation completed. Overall score: {scores['overall']:.2f}")
        return report
    
    def _calculate_quality_scores(self, data: pd.DataFrame, issues: List[DataQualityIssue]) -> Dict[str, float]:
        """Calculate quality scores based on issues found."""
        total_rows = len(data)
        
        # Count issues by type
        issue_counts = {
            'missing': 0,
            'consistency': 0,
            'timeliness': 0,
            'accuracy': 0
        }
        
        for issue in issues:
            affected_ratio = len(issue.affected_rows) / total_rows if total_rows > 0 else 0
            severity_weight = {
                DataQualitySeverity.LOW: 0.1,
                DataQualitySeverity.MEDIUM: 0.3,
                DataQualitySeverity.HIGH: 0.6,
                DataQualitySeverity.CRITICAL: 1.0
            }[issue.severity]
            
            impact = affected_ratio * severity_weight
            
            if issue.issue_type in [DataQualityIssueType.MISSING_DATA]:
                issue_counts['missing'] += impact
            elif issue.issue_type in [DataQualityIssueType.INCONSISTENT_DATA, DataQualityIssueType.BUSINESS_RULE_VIOLATION]:
                issue_counts['consistency'] += impact
            elif issue.issue_type in [DataQualityIssueType.STALE_DATA]:
                issue_counts['timeliness'] += impact
            elif issue.issue_type in [DataQualityIssueType.OUTLIER_DATA, DataQualityIssueType.DUPLICATE_DATA]:
                issue_counts['accuracy'] += impact
        
        # Calculate scores (1.0 - impact, capped at 0.0)
        scores = {
            'completeness': max(0.0, 1.0 - issue_counts['missing']),
            'consistency': max(0.0, 1.0 - issue_counts['consistency']),
            'timeliness': max(0.0, 1.0 - issue_counts['timeliness']),
            'accuracy': max(0.0, 1.0 - issue_counts['accuracy'])
        }
        
        # Overall score is weighted average
        scores['overall'] = (
            scores['completeness'] * 0.3 +
            scores['consistency'] * 0.3 +
            scores['timeliness'] * 0.2 +
            scores['accuracy'] * 0.2
        )
        
        return scores
    
    def _generate_recommendations(self, issues: List[DataQualityIssue]) -> List[str]:
        """Generate actionable recommendations based on issues."""
        recommendations = []
        
        # Group issues by type
        issue_groups = {}
        for issue in issues:
            issue_type = issue.issue_type
            if issue_type not in issue_groups:
                issue_groups[issue_type] = []
            issue_groups[issue_type].append(issue)
        
        # Generate recommendations for each issue type
        for issue_type, type_issues in issue_groups.items():
            if issue_type == DataQualityIssueType.MISSING_DATA:
                recommendations.append(
                    f"Address {len(type_issues)} missing data issues by implementing data validation at source"
                )
            elif issue_type == DataQualityIssueType.INCONSISTENT_DATA:
                recommendations.append(
                    f"Fix {len(type_issues)} consistency issues by adding business rule validation"
                )
            elif issue_type == DataQualityIssueType.DUPLICATE_DATA:
                recommendations.append(
                    f"Remove {len(type_issues)} duplicate records and implement deduplication logic"
                )
            elif issue_type == DataQualityIssueType.OUTLIER_DATA:
                recommendations.append(
                    f"Review {len(type_issues)} potential outliers flagged by ML detector"
                )
            elif issue_type == DataQualityIssueType.STALE_DATA:
                recommendations.append(
                    f"Update {len(type_issues)} stale data records and improve data refresh frequency"
                )
        
        # Add general recommendations
        if len(issues) > 10:
            recommendations.append("Consider implementing automated data quality monitoring")
        
        if any(issue.severity == DataQualitySeverity.CRITICAL for issue in issues):
            recommendations.append("Address critical issues immediately before using data for analysis")
        
        return recommendations
    
    def clean_data(self, data: pd.DataFrame, report: DataQualityReport) -> pd.DataFrame:
        """Apply automatic data cleaning based on quality report."""
        cleaned_data = data.copy()
        
        for issue in report.issues:
            if issue.issue_type == DataQualityIssueType.DUPLICATE_DATA:
                # Remove duplicates
                if 'stock_code' in cleaned_data.columns and 'trade_date' in cleaned_data.columns:
                    cleaned_data = cleaned_data.drop_duplicates(subset=['stock_code', 'trade_date'], keep='first')
                    logger.info(f"Removed duplicate records")
            
            elif issue.issue_type == DataQualityIssueType.OUTLIER_DATA and issue.severity != DataQualitySeverity.CRITICAL:
                # Remove extreme outliers (only if not critical)
                if issue.affected_rows:
                    cleaned_data = cleaned_data.drop(index=issue.affected_rows, errors='ignore')
                    logger.info(f"Removed {len(issue.affected_rows)} outlier records")
        
        logger.info(f"Data cleaning completed. Rows: {len(data)} -> {len(cleaned_data)}")
        return cleaned_data
    
    def save_model(self, filepath: str):
        """Save the ML model."""
        self.ml_detector.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load the ML model."""
        self.ml_detector.load_model(filepath)