#!/usr/bin/env python3
"""Demo script for Data Quality Engine."""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from stock_analysis_system.data.data_quality_engine import EnhancedDataQualityEngine

def create_sample_data():
    """Create sample stock data with various quality issues."""
    print("📊 Creating sample stock data with quality issues...")
    
    # Create base data
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    
    data = pd.DataFrame({
        'stock_code': ['000001.SZ'] * 20,
        'trade_date': dates,
        'open_price': np.random.uniform(10, 15, 20),
        'high_price': np.random.uniform(12, 18, 20),
        'low_price': np.random.uniform(8, 12, 20),
        'close_price': np.random.uniform(10, 15, 20),
        'volume': np.random.randint(1000, 5000, 20),
        'amount': np.random.uniform(10000, 50000, 20)
    })
    
    # Introduce quality issues
    print("🔧 Introducing quality issues...")
    
    # 1. Missing values
    data.loc[2, 'close_price'] = None
    data.loc[5, 'volume'] = None
    print("   ✓ Added missing values")
    
    # 2. Inconsistent OHLC relationships
    data.loc[3, 'high_price'] = data.loc[3, 'low_price'] - 1  # High < Low
    data.loc[7, 'low_price'] = data.loc[7, 'high_price'] + 1  # Low > High
    print("   ✓ Added OHLC inconsistencies")
    
    # 3. Negative values
    data.loc[4, 'close_price'] = -5.0
    data.loc[8, 'volume'] = -1000
    print("   ✓ Added negative values")
    
    # 4. Duplicates
    duplicate_row = data.iloc[10].copy()
    data.loc[len(data)] = duplicate_row
    print("   ✓ Added duplicate records")
    
    # 5. Future dates
    data.loc[12, 'trade_date'] = datetime.now() + timedelta(days=30)
    print("   ✓ Added future dates")
    
    # 6. Extreme outliers
    data.loc[15, 'close_price'] = 1000.0  # Extreme price
    data.loc[16, 'volume'] = 1000000  # Extreme volume
    print("   ✓ Added extreme outliers")
    
    return data

def demonstrate_data_quality_engine():
    """Demonstrate the data quality engine capabilities."""
    print("🚀 Data Quality Engine Demonstration")
    print("=" * 60)
    
    # Create sample data
    sample_data = create_sample_data()
    print(f"\n📈 Created dataset with {len(sample_data)} rows and {len(sample_data.columns)} columns")
    
    # Initialize the engine
    print("\n🔧 Initializing Enhanced Data Quality Engine...")
    engine = EnhancedDataQualityEngine()
    print(f"   ✓ Loaded {len(engine.rules)} default rules")
    
    # Train ML anomaly detector
    print("\n🤖 Training ML anomaly detector...")
    clean_subset = sample_data.iloc[:10].copy()  # Use first 10 rows as "clean" training data
    clean_subset = clean_subset.dropna()  # Remove any NaN values for training
    
    if len(clean_subset) > 5:  # Need enough data for training
        try:
            engine.train_ml_detector(clean_subset)
            print("   ✓ ML detector trained successfully")
        except Exception as e:
            print(f"   ⚠️ ML training failed: {e}")
    else:
        print("   ⚠️ Not enough clean data for ML training")
    
    # Validate data quality
    print("\n🔍 Performing comprehensive data quality validation...")
    report = engine.validate_data(sample_data, "Demo Dataset")
    
    # Display results
    print("\n📋 DATA QUALITY REPORT")
    print("-" * 40)
    print(f"Dataset: {report.dataset_name}")
    print(f"Rows: {report.total_rows}")
    print(f"Columns: {report.total_columns}")
    print(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📊 QUALITY SCORES")
    print(f"Overall Score:     {report.overall_score:.2f}")
    print(f"Completeness:      {report.completeness_score:.2f}")
    print(f"Consistency:       {report.consistency_score:.2f}")
    print(f"Timeliness:        {report.timeliness_score:.2f}")
    print(f"Accuracy:          {report.accuracy_score:.2f}")
    
    print(f"\n🚨 ISSUES FOUND ({len(report.issues)} total)")
    print("-" * 40)
    
    # Group issues by type
    issue_groups = {}
    for issue in report.issues:
        issue_type = issue.issue_type.value
        if issue_type not in issue_groups:
            issue_groups[issue_type] = []
        issue_groups[issue_type].append(issue)
    
    for issue_type, issues in issue_groups.items():
        print(f"\n{issue_type.upper().replace('_', ' ')} ({len(issues)} issues):")
        for i, issue in enumerate(issues[:3], 1):  # Show first 3 issues of each type
            severity_icon = {
                'low': '🟢',
                'medium': '🟡', 
                'high': '🟠',
                'critical': '🔴'
            }.get(issue.severity.value, '⚪')
            
            print(f"  {severity_icon} {issue.description}")
            if issue.affected_rows:
                affected_count = len(issue.affected_rows)
                if affected_count <= 5:
                    print(f"     Affected rows: {issue.affected_rows}")
                else:
                    print(f"     Affected rows: {affected_count} rows (showing first 5: {issue.affected_rows[:5]})")
            print(f"     Suggested action: {issue.suggested_action}")
        
        if len(issues) > 3:
            print(f"  ... and {len(issues) - 3} more issues")
    
    print(f"\n💡 RECOMMENDATIONS ({len(report.recommendations)} total)")
    print("-" * 40)
    for i, recommendation in enumerate(report.recommendations, 1):
        print(f"{i}. {recommendation}")
    
    # Demonstrate data cleaning
    print(f"\n🧹 AUTOMATIC DATA CLEANING")
    print("-" * 40)
    original_rows = len(sample_data)
    cleaned_data = engine.clean_data(sample_data, report)
    cleaned_rows = len(cleaned_data)
    
    print(f"Original rows: {original_rows}")
    print(f"Cleaned rows:  {cleaned_rows}")
    print(f"Removed rows:  {original_rows - cleaned_rows}")
    
    if cleaned_rows < original_rows:
        print("✓ Automatic cleaning removed problematic records")
    
    # Validate cleaned data
    print(f"\n🔍 VALIDATING CLEANED DATA")
    print("-" * 40)
    cleaned_report = engine.validate_data(cleaned_data, "Cleaned Demo Dataset")
    
    print(f"Quality improvement:")
    print(f"  Overall Score: {report.overall_score:.2f} → {cleaned_report.overall_score:.2f}")
    print(f"  Issues Found:  {len(report.issues)} → {len(cleaned_report.issues)}")
    
    if cleaned_report.overall_score > report.overall_score:
        print("✅ Data quality improved after cleaning!")
    
    print(f"\n🎯 SUMMARY")
    print("-" * 40)
    print(f"✓ Detected {len(report.issues)} data quality issues")
    print(f"✓ Applied {len(engine.rules)} validation rules")
    if engine.ml_detector.is_fitted:
        print(f"✓ Used ML-based anomaly detection")
    print(f"✓ Generated {len(report.recommendations)} actionable recommendations")
    print(f"✓ Improved overall quality score by {cleaned_report.overall_score - report.overall_score:.2f}")
    
    return report, cleaned_report

def demonstrate_custom_rules():
    """Demonstrate adding custom validation rules."""
    print(f"\n🔧 CUSTOM RULES DEMONSTRATION")
    print("-" * 40)
    
    from stock_analysis_system.data.data_quality_engine import CompletenessRule, DataQualitySeverity
    
    # Create engine with custom rules
    engine = EnhancedDataQualityEngine()
    
    # Add custom rule for specific requirements
    custom_rule = CompletenessRule(
        required_columns=['stock_code', 'trade_date', 'close_price', 'volume'],
        max_missing_ratio=0.01  # Very strict - only 1% missing allowed
    )
    custom_rule.name = "Strict Completeness Check"
    custom_rule.severity = DataQualitySeverity.CRITICAL
    
    engine.add_rule(custom_rule)
    print(f"✓ Added custom rule: {custom_rule.name}")
    
    # Test with data that has missing values
    test_data = pd.DataFrame({
        'stock_code': ['000001.SZ'] * 10,
        'trade_date': pd.date_range('2024-01-01', periods=10),
        'close_price': [10.0] * 8 + [None, None],  # 20% missing
        'volume': [1000] * 10
    })
    
    report = engine.validate_data(test_data, "Custom Rules Test")
    
    print(f"Custom rule validation results:")
    print(f"  Issues found: {len(report.issues)}")
    for issue in report.issues:
        if "Strict Completeness" in issue.description:
            print(f"  ✓ Custom rule triggered: {issue.description}")
    
    return engine

if __name__ == "__main__":
    try:
        # Main demonstration
        report, cleaned_report = demonstrate_data_quality_engine()
        
        # Custom rules demonstration
        custom_engine = demonstrate_custom_rules()
        
        print(f"\n🎉 Data Quality Engine demonstration completed successfully!")
        print(f"The engine successfully identified and addressed multiple data quality issues.")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()