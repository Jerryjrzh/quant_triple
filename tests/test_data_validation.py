"""
数据验证器单元测试

测试数据验证规则的正确性和完整性，实现各种数据异常场景的测试用例，
添加性能测试验证验证器的效率，创建验证结果的断言和报告机制。

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import tempfile
import os

from stock_analysis_system.data.data_quality_engine import (
    EnhancedDataQualityEngine,
    DataQualityRule,
    CompletenessRule,
    ConsistencyRule,
    TimelinessRule,
    DuplicateRule,
    MLAnomalyDetector,
    DataQualityIssue,
    DataQualityReport,
    DataQualityIssueType,
    DataQualitySeverity
)


class TestDataGenerator:
    """数据验证测试数据生成器"""
    
    @staticmethod
    def generate_valid_stock_data(rows: int = 100) -> pd.DataFrame:
        """生成有效的股票数据"""
        np.random.seed(42)
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=rows),
            periods=rows,
            freq='D'
        )
        
        data = []
        for i, date in enumerate(dates):
            base_price = 10.0 + np.random.normal(0, 0.5)
            open_price = max(0.1, base_price + np.random.normal(0, 0.2))
            close_price = max(0.1, base_price + np.random.normal(0, 0.2))
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.1))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.1))
            
            data.append({
                'stock_code': f'{i%10:06d}',
                'trade_date': date,
                'open_price': round(open_price, 2),
                'high_price': round(high_price, 2),
                'low_price': round(max(0.1, low_price), 2),
                'close_price': round(close_price, 2),
                'volume': int(abs(np.random.normal(1000000, 500000))),
                'amount': round(abs(np.random.normal(10000000, 5000000)), 2)
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_data_with_missing_values(rows: int = 100, missing_ratio: float = 0.1) -> pd.DataFrame:
        """生成包含缺失值的数据"""
        df = TestDataGenerator.generate_valid_stock_data(rows)
        
        # 随机设置缺失值
        for column in ['close_price', 'volume', 'amount']:
            missing_indices = np.random.choice(
                df.index, 
                size=int(len(df) * missing_ratio), 
                replace=False
            )
            df.loc[missing_indices, column] = np.nan
        
        return df
    
    @staticmethod
    def generate_data_with_duplicates(rows: int = 100, duplicate_ratio: float = 0.1) -> pd.DataFrame:
        """生成包含重复数据的数据"""
        df = TestDataGenerator.generate_valid_stock_data(rows)
        
        # 添加重复行
        duplicate_count = int(len(df) * duplicate_ratio)
        duplicate_indices = np.random.choice(df.index, size=duplicate_count, replace=True)
        duplicate_rows = df.loc[duplicate_indices].copy()
        
        df = pd.concat([df, duplicate_rows], ignore_index=True)
        return df
    
    @staticmethod
    def generate_data_with_inconsistencies(rows: int = 100) -> pd.DataFrame:
        """生成包含不一致数据的数据"""
        df = TestDataGenerator.generate_valid_stock_data(rows)
        
        # 创建OHLC不一致的数据
        inconsistent_indices = np.random.choice(df.index, size=10, replace=False)
        for idx in inconsistent_indices:
            # 让high_price小于close_price
            df.loc[idx, 'high_price'] = df.loc[idx, 'close_price'] - 1.0
        
        # 添加负价格
        negative_indices = np.random.choice(df.index, size=5, replace=False)
        for idx in negative_indices:
            df.loc[idx, 'close_price'] = -10.0
        
        return df
    
    @staticmethod
    def generate_data_with_stale_dates(rows: int = 100) -> pd.DataFrame:
        """生成包含过期日期的数据"""
        df = TestDataGenerator.generate_valid_stock_data(rows)
        
        # 添加过期数据
        stale_indices = np.random.choice(df.index, size=20, replace=False)
        stale_date = datetime.now() - timedelta(days=60)
        for idx in stale_indices:
            df.loc[idx, 'trade_date'] = stale_date
        
        # 添加未来日期
        future_indices = np.random.choice(df.index, size=5, replace=False)
        future_date = datetime.now() + timedelta(days=10)
        for idx in future_indices:
            df.loc[idx, 'trade_date'] = future_date
        
        return df
    
    @staticmethod
    def generate_data_with_outliers(rows: int = 100) -> pd.DataFrame:
        """生成包含异常值的数据"""
        df = TestDataGenerator.generate_valid_stock_data(rows)
        
        # 添加价格异常值
        outlier_indices = np.random.choice(df.index, size=10, replace=False)
        for idx in outlier_indices:
            df.loc[idx, 'close_price'] = 1000.0  # 异常高价
            df.loc[idx, 'volume'] = 100000000  # 异常高成交量
        
        return df


class TestCompletenessRule:
    """完整性规则测试类"""
    
    @pytest.fixture
    def rule(self):
        """创建完整性规则实例"""
        return CompletenessRule(
            required_columns=['stock_code', 'trade_date', 'close_price'],
            max_missing_ratio=0.05
        )
    
    def test_validate_complete_data(self, rule):
        """测试完整数据验证"""
        data = TestDataGenerator.generate_valid_stock_data(100)
        issues = rule.validate(data)
        
        assert len(issues) == 0
    
    def test_validate_missing_column(self, rule):
        """测试缺失列验证"""
        data = TestDataGenerator.generate_valid_stock_data(100)
        data = data.drop('stock_code', axis=1)  # 删除必需列
        
        issues = rule.validate(data)
        
        assert len(issues) == 1
        assert issues[0].issue_type == DataQualityIssueType.MISSING_DATA
        assert issues[0].severity == DataQualitySeverity.CRITICAL
        assert 'stock_code' in issues[0].description
    
    def test_validate_excessive_missing_values(self, rule):
        """测试过多缺失值验证"""
        data = TestDataGenerator.generate_data_with_missing_values(100, 0.1)  # 10%缺失
        
        issues = rule.validate(data)
        
        # 应该检测到缺失值过多的问题
        missing_issues = [i for i in issues if i.issue_type == DataQualityIssueType.MISSING_DATA]
        assert len(missing_issues) > 0
        
        for issue in missing_issues:
            assert issue.severity == DataQualitySeverity.HIGH
            assert len(issue.affected_rows) > 0
    
    def test_validate_acceptable_missing_values(self, rule):
        """测试可接受的缺失值验证"""
        data = TestDataGenerator.generate_data_with_missing_values(100, 0.02)  # 2%缺失
        
        issues = rule.validate(data)
        
        # 2%的缺失值应该是可接受的（阈值是5%）
        missing_issues = [i for i in issues if i.issue_type == DataQualityIssueType.MISSING_DATA]
        assert len(missing_issues) == 0
    
    def test_different_thresholds(self):
        """测试不同阈值设置"""
        data = TestDataGenerator.generate_data_with_missing_values(100, 0.08)  # 8%缺失
        
        # 严格规则（5%阈值）
        strict_rule = CompletenessRule(['close_price'], max_missing_ratio=0.05)
        strict_issues = strict_rule.validate(data)
        
        # 宽松规则（10%阈值）
        lenient_rule = CompletenessRule(['close_price'], max_missing_ratio=0.10)
        lenient_issues = lenient_rule.validate(data)
        
        # 严格规则应该检测到问题，宽松规则不应该
        assert len(strict_issues) > 0
        assert len(lenient_issues) == 0


class TestConsistencyRule:
    """一致性规则测试类"""
    
    @pytest.fixture
    def rule(self):
        """创建一致性规则实例"""
        return ConsistencyRule()
    
    def test_validate_consistent_ohlc_data(self, rule):
        """测试一致的OHLC数据验证"""
        data = TestDataGenerator.generate_valid_stock_data(100)
        issues = rule.validate(data)
        
        # 有效数据不应该有一致性问题
        consistency_issues = [
            i for i in issues 
            if i.issue_type == DataQualityIssueType.INCONSISTENT_DATA
        ]
        assert len(consistency_issues) == 0
    
    def test_validate_inconsistent_ohlc_data(self, rule):
        """测试不一致的OHLC数据验证"""
        data = TestDataGenerator.generate_data_with_inconsistencies(100)
        issues = rule.validate(data)
        
        # 应该检测到OHLC不一致问题
        ohlc_issues = [
            i for i in issues 
            if i.issue_type == DataQualityIssueType.INCONSISTENT_DATA
            and 'high_price' in i.description
        ]
        assert len(ohlc_issues) > 0
        
        for issue in ohlc_issues:
            assert issue.severity == DataQualitySeverity.HIGH
            assert len(issue.affected_rows) > 0
            assert 'high_price' in issue.affected_columns
    
    def test_validate_negative_values(self, rule):
        """测试负值验证"""
        data = TestDataGenerator.generate_data_with_inconsistencies(100)
        issues = rule.validate(data)
        
        # 应该检测到负价格问题
        negative_issues = [
            i for i in issues 
            if i.issue_type == DataQualityIssueType.BUSINESS_RULE_VIOLATION
            and 'negative' in i.description
        ]
        assert len(negative_issues) > 0
        
        for issue in negative_issues:
            assert issue.severity == DataQualitySeverity.HIGH
            assert len(issue.affected_rows) > 0
    
    def test_validate_partial_ohlc_data(self, rule):
        """测试部分OHLC数据验证"""
        # 只有部分OHLC列的数据
        data = pd.DataFrame({
            'stock_code': ['000001', '000002'],
            'trade_date': [datetime.now(), datetime.now()],
            'close_price': [10.0, 20.0],
            'volume': [1000000, 2000000]
        })
        
        issues = rule.validate(data)
        
        # 没有完整的OHLC数据，不应该有OHLC相关的一致性问题
        ohlc_issues = [
            i for i in issues 
            if 'high_price' in i.description or 'low_price' in i.description
        ]
        assert len(ohlc_issues) == 0
    
    def test_edge_case_equal_prices(self, rule):
        """测试价格相等的边界情况"""
        data = pd.DataFrame({
            'stock_code': ['000001'],
            'trade_date': [datetime.now()],
            'open_price': [10.0],
            'high_price': [10.0],
            'low_price': [10.0],
            'close_price': [10.0],
            'volume': [1000000]
        })
        
        issues = rule.validate(data)
        
        # 所有价格相等应该是有效的
        ohlc_issues = [
            i for i in issues 
            if i.issue_type == DataQualityIssueType.INCONSISTENT_DATA
        ]
        assert len(ohlc_issues) == 0


class TestTimelinessRule:
    """时效性规则测试类"""
    
    @pytest.fixture
    def rule(self):
        """创建时效性规则实例"""
        return TimelinessRule(date_column='trade_date', max_age_days=7)
    
    def test_validate_recent_data(self, rule):
        """测试最新数据验证"""
        data = TestDataGenerator.generate_valid_stock_data(100)
        issues = rule.validate(data)
        
        # 最新数据不应该有时效性问题
        timeliness_issues = [
            i for i in issues 
            if i.issue_type == DataQualityIssueType.STALE_DATA
        ]
        assert len(timeliness_issues) == 0
    
    def test_validate_stale_data(self, rule):
        """测试过期数据验证"""
        data = TestDataGenerator.generate_data_with_stale_dates(100)
        issues = rule.validate(data)
        
        # 应该检测到过期数据问题
        stale_issues = [
            i for i in issues 
            if i.issue_type == DataQualityIssueType.STALE_DATA
        ]
        assert len(stale_issues) > 0
        
        for issue in stale_issues:
            assert issue.severity == DataQualitySeverity.MEDIUM
            assert len(issue.affected_rows) > 0
    
    def test_validate_future_dates(self, rule):
        """测试未来日期验证"""
        data = TestDataGenerator.generate_data_with_stale_dates(100)
        issues = rule.validate(data)
        
        # 应该检测到未来日期问题
        future_issues = [
            i for i in issues 
            if i.issue_type == DataQualityIssueType.INCONSISTENT_DATA
            and 'future' in i.description
        ]
        assert len(future_issues) > 0
        
        for issue in future_issues:
            assert issue.severity == DataQualitySeverity.HIGH
            assert len(issue.affected_rows) > 0
    
    def test_validate_invalid_date_format(self, rule):
        """测试无效日期格式验证"""
        data = pd.DataFrame({
            'stock_code': ['000001', '000002'],
            'trade_date': ['invalid_date', '2024-13-45'],  # 无效日期
            'close_price': [10.0, 20.0]
        })
        
        issues = rule.validate(data)
        
        # 应该检测到日期格式问题
        format_issues = [
            i for i in issues 
            if i.issue_type == DataQualityIssueType.INVALID_FORMAT
        ]
        assert len(format_issues) > 0
        
        for issue in format_issues:
            assert issue.severity == DataQualitySeverity.HIGH
            assert 'parse dates' in issue.description
    
    def test_validate_missing_date_column(self, rule):
        """测试缺失日期列验证"""
        data = pd.DataFrame({
            'stock_code': ['000001', '000002'],
            'close_price': [10.0, 20.0]
        })
        
        issues = rule.validate(data)
        
        # 缺失日期列不应该产生错误，应该被忽略
        assert len(issues) == 0
    
    def test_different_age_thresholds(self):
        """测试不同时效阈值"""
        # 创建30天前的数据
        old_date = datetime.now() - timedelta(days=30)
        data = pd.DataFrame({
            'stock_code': ['000001'],
            'trade_date': [old_date],
            'close_price': [10.0]
        })
        
        # 严格规则（7天）
        strict_rule = TimelinessRule('trade_date', max_age_days=7)
        strict_issues = strict_rule.validate(data)
        
        # 宽松规则（60天）
        lenient_rule = TimelinessRule('trade_date', max_age_days=60)
        lenient_issues = lenient_rule.validate(data)
        
        # 严格规则应该检测到问题，宽松规则不应该
        assert len(strict_issues) > 0
        assert len(lenient_issues) == 0


class TestDuplicateRule:
    """重复数据规则测试类"""
    
    @pytest.fixture
    def rule(self):
        """创建重复数据规则实例"""
        return DuplicateRule(key_columns=['stock_code', 'trade_date'])
    
    def test_validate_unique_data(self, rule):
        """测试唯一数据验证"""
        data = TestDataGenerator.generate_valid_stock_data(100)
        issues = rule.validate(data)
        
        # 唯一数据不应该有重复问题
        duplicate_issues = [
            i for i in issues 
            if i.issue_type == DataQualityIssueType.DUPLICATE_DATA
        ]
        assert len(duplicate_issues) == 0
    
    def test_validate_duplicate_data(self, rule):
        """测试重复数据验证"""
        data = TestDataGenerator.generate_data_with_duplicates(100, 0.1)
        issues = rule.validate(data)
        
        # 应该检测到重复数据问题
        duplicate_issues = [
            i for i in issues 
            if i.issue_type == DataQualityIssueType.DUPLICATE_DATA
        ]
        assert len(duplicate_issues) > 0
        
        for issue in duplicate_issues:
            assert issue.severity == DataQualitySeverity.MEDIUM
            assert len(issue.affected_rows) > 0
            assert issue.affected_columns == ['stock_code', 'trade_date']
    
    def test_validate_missing_key_columns(self, rule):
        """测试缺失关键列验证"""
        data = pd.DataFrame({
            'stock_code': ['000001', '000002'],
            'close_price': [10.0, 20.0]
            # 缺失 trade_date 列
        })
        
        issues = rule.validate(data)
        
        # 应该检测到缺失关键列问题
        missing_issues = [
            i for i in issues 
            if i.issue_type == DataQualityIssueType.MISSING_DATA
            and 'Key columns missing' in i.description
        ]
        assert len(missing_issues) > 0
        
        for issue in missing_issues:
            assert issue.severity == DataQualitySeverity.HIGH
            assert 'trade_date' in str(issue.affected_columns)
    
    def test_partial_duplicates(self, rule):
        """测试部分重复数据"""
        data = pd.DataFrame({
            'stock_code': ['000001', '000001', '000002'],
            'trade_date': [datetime.now(), datetime.now(), datetime.now()],
            'close_price': [10.0, 15.0, 20.0]  # 不同的价格
        })
        
        issues = rule.validate(data)
        
        # 前两行在关键列上重复
        duplicate_issues = [
            i for i in issues 
            if i.issue_type == DataQualityIssueType.DUPLICATE_DATA
        ]
        assert len(duplicate_issues) > 0
        assert len(duplicate_issues[0].affected_rows) == 2  # 两行重复
    
    def test_different_key_columns(self):
        """测试不同关键列组合"""
        data = pd.DataFrame({
            'stock_code': ['000001', '000001', '000002'],
            'trade_date': [datetime.now(), datetime.now() + timedelta(days=1), datetime.now()],
            'close_price': [10.0, 15.0, 20.0]
        })
        
        # 只用stock_code作为关键列
        single_key_rule = DuplicateRule(['stock_code'])
        single_key_issues = single_key_rule.validate(data)
        
        # 用stock_code和trade_date作为关键列
        double_key_rule = DuplicateRule(['stock_code', 'trade_date'])
        double_key_issues = double_key_rule.validate(data)
        
        # 单关键列应该检测到重复，双关键列不应该
        assert len(single_key_issues) > 0
        assert len(double_key_issues) == 0


class TestMLAnomalyDetector:
    """机器学习异常检测器测试类"""
    
    @pytest.fixture
    def detector(self):
        """创建异常检测器实例"""
        return MLAnomalyDetector(contamination=0.1, random_state=42)
    
    @pytest.fixture
    def training_data(self):
        """创建训练数据"""
        return TestDataGenerator.generate_valid_stock_data(1000)
    
    def test_fit_detector(self, detector, training_data):
        """测试检测器训练"""
        feature_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
        
        detector.fit(training_data, feature_columns)
        
        assert detector.is_fitted is True
        assert detector.feature_columns == feature_columns
        assert detector.model is not None
        assert detector.scaler is not None
    
    def test_detect_anomalies_normal_data(self, detector, training_data):
        """测试正常数据异常检测"""
        feature_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
        detector.fit(training_data, feature_columns)
        
        # 使用正常数据进行检测
        test_data = TestDataGenerator.generate_valid_stock_data(100)
        anomaly_indices, anomaly_scores = detector.detect_anomalies(test_data)
        
        # 正常数据应该有少量异常（根据contamination参数）
        assert len(anomaly_indices) <= len(test_data) * 0.2  # 允许一些误报
        assert len(anomaly_indices) == len(anomaly_scores)
    
    def test_detect_anomalies_with_outliers(self, detector, training_data):
        """测试包含异常值的数据检测"""
        feature_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
        detector.fit(training_data, feature_columns)
        
        # 使用包含异常值的数据进行检测
        test_data = TestDataGenerator.generate_data_with_outliers(100)
        anomaly_indices, anomaly_scores = detector.detect_anomalies(test_data)
        
        # 应该检测到更多异常
        assert len(anomaly_indices) > 0
        assert len(anomaly_indices) == len(anomaly_scores)
        
        # 异常分数应该是负数（Isolation Forest的特性）
        assert all(score < 0 for score in anomaly_scores)
    
    def test_detect_without_fitting(self, detector):
        """测试未训练时的异常检测"""
        test_data = TestDataGenerator.generate_valid_stock_data(100)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.detect_anomalies(test_data)
    
    def test_save_and_load_model(self, detector, training_data):
        """测试模型保存和加载"""
        feature_columns = ['open_price', 'high_price', 'low_price', 'close_price']
        detector.fit(training_data, feature_columns)
        
        # 保存模型
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            detector.save_model(tmp_path)
            
            # 创建新的检测器并加载模型
            new_detector = MLAnomalyDetector()
            new_detector.load_model(tmp_path)
            
            # 验证加载的模型
            assert new_detector.is_fitted is True
            assert new_detector.feature_columns == feature_columns
            assert new_detector.contamination == detector.contamination
            
            # 测试加载的模型是否工作
            test_data = TestDataGenerator.generate_valid_stock_data(50)
            anomaly_indices, anomaly_scores = new_detector.detect_anomalies(test_data)
            
            assert isinstance(anomaly_indices, list)
            assert isinstance(anomaly_scores, list)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_nonexistent_model(self, detector):
        """测试加载不存在的模型"""
        with pytest.raises(FileNotFoundError):
            detector.load_model('nonexistent_model.joblib')
    
    def test_save_without_fitting(self, detector):
        """测试未训练时保存模型"""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(ValueError, match="Model must be fitted"):
                detector.save_model(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_handle_missing_values_in_features(self, detector):
        """测试特征中缺失值的处理"""
        # 创建包含缺失值的训练数据
        training_data = TestDataGenerator.generate_data_with_missing_values(1000, 0.05)
        feature_columns = ['open_price', 'high_price', 'low_price', 'close_price']
        
        # 应该能够处理缺失值
        detector.fit(training_data, feature_columns)
        assert detector.is_fitted is True
        
        # 测试数据也包含缺失值
        test_data = TestDataGenerator.generate_data_with_missing_values(100, 0.05)
        anomaly_indices, anomaly_scores = detector.detect_anomalies(test_data)
        
        assert isinstance(anomaly_indices, list)
        assert isinstance(anomaly_scores, list)


class TestEnhancedDataQualityEngine:
    """增强数据质量引擎测试类"""
    
    @pytest.fixture
    def engine(self):
        """创建数据质量引擎实例"""
        return EnhancedDataQualityEngine()
    
    def test_default_rules_initialization(self, engine):
        """测试默认规则初始化"""
        assert len(engine.rules) > 0
        
        # 检查是否包含预期的规则类型
        rule_types = [type(rule).__name__ for rule in engine.rules]
        assert 'CompletenessRule' in rule_types
        assert 'ConsistencyRule' in rule_types
        assert 'TimelinessRule' in rule_types
        assert 'DuplicateRule' in rule_types
    
    def test_add_and_remove_rules(self, engine):
        """测试添加和删除规则"""
        initial_count = len(engine.rules)
        
        # 添加自定义规则
        custom_rule = CompletenessRule(['custom_column'])
        engine.add_rule(custom_rule)
        
        assert len(engine.rules) == initial_count + 1
        
        # 删除规则
        engine.remove_rule('Completeness Check')
        
        # 应该删除了两个完整性规则（默认的和自定义的）
        remaining_completeness_rules = [
            rule for rule in engine.rules 
            if isinstance(rule, CompletenessRule)
        ]
        assert len(remaining_completeness_rules) == 0
    
    def test_validate_clean_data(self, engine):
        """测试验证干净数据"""
        data = TestDataGenerator.generate_valid_stock_data(100)
        report = engine.validate_data(data, "test_clean_data")
        
        assert report.dataset_name == "test_clean_data"
        assert report.total_rows == 100
        assert report.total_columns == len(data.columns)
        assert report.overall_score > 0.8  # 干净数据应该有高分
        assert len(report.recommendations) >= 0
    
    def test_validate_problematic_data(self, engine):
        """测试验证有问题的数据"""
        # 创建包含多种问题的数据
        data = TestDataGenerator.generate_data_with_missing_values(100, 0.1)
        data = TestDataGenerator.generate_data_with_duplicates(100, 0.1)
        data = TestDataGenerator.generate_data_with_inconsistencies(100)
        
        report = engine.validate_data(data, "test_problematic_data")
        
        assert len(report.issues) > 0
        assert report.overall_score < 0.8  # 有问题的数据应该有低分
        assert len(report.recommendations) > 0
        
        # 检查是否检测到各种类型的问题
        issue_types = [issue.issue_type for issue in report.issues]
        assert DataQualityIssueType.MISSING_DATA in issue_types
        assert DataQualityIssueType.INCONSISTENT_DATA in issue_types or \
               DataQualityIssueType.BUSINESS_RULE_VIOLATION in issue_types
    
    def test_train_ml_detector(self, engine):
        """测试训练ML检测器"""
        training_data = TestDataGenerator.generate_valid_stock_data(1000)
        
        # 训练ML检测器
        engine.train_ml_detector(training_data)
        
        assert engine.ml_detector.is_fitted is True
        
        # 使用训练后的检测器进行验证
        test_data = TestDataGenerator.generate_data_with_outliers(100)
        report = engine.validate_data(test_data, "test_with_ml")
        
        # 应该包含ML检测到的异常
        ml_issues = [
            issue for issue in report.issues 
            if issue.issue_type == DataQualityIssueType.OUTLIER_DATA
        ]
        assert len(ml_issues) > 0
    
    def test_train_ml_detector_with_custom_features(self, engine):
        """测试使用自定义特征训练ML检测器"""
        training_data = TestDataGenerator.generate_valid_stock_data(1000)
        custom_features = ['close_price', 'volume']
        
        engine.train_ml_detector(training_data, custom_features)
        
        assert engine.ml_detector.is_fitted is True
        assert engine.ml_detector.feature_columns == custom_features
    
    def test_train_ml_detector_no_features(self, engine):
        """测试没有合适特征时的ML训练"""
        # 创建没有数值列的数据
        data = pd.DataFrame({
            'stock_code': ['000001', '000002'],
            'stock_name': ['股票A', '股票B']
        })
        
        with patch('stock_analysis_system.data.data_quality_engine.logger') as mock_logger:
            engine.train_ml_detector(data)
            
            # 应该记录警告
            mock_logger.warning.assert_called_with(
                "No suitable feature columns found for ML training"
            )
            
            assert engine.ml_detector.is_fitted is False
    
    def test_calculate_quality_scores(self, engine):
        """测试质量分数计算"""
        data = TestDataGenerator.generate_valid_stock_data(100)
        
        # 创建一些测试问题
        issues = [
            DataQualityIssue(
                issue_type=DataQualityIssueType.MISSING_DATA,
                severity=DataQualitySeverity.HIGH,
                description="Test missing data",
                affected_rows=list(range(10)),  # 10%的行受影响
                affected_columns=['test_column'],
                confidence_score=1.0,
                suggested_action="Fix missing data"
            ),
            DataQualityIssue(
                issue_type=DataQualityIssueType.INCONSISTENT_DATA,
                severity=DataQualitySeverity.MEDIUM,
                description="Test inconsistent data",
                affected_rows=list(range(5)),  # 5%的行受影响
                affected_columns=['test_column'],
                confidence_score=1.0,
                suggested_action="Fix inconsistent data"
            )
        ]
        
        scores = engine._calculate_quality_scores(data, issues)
        
        assert 'overall' in scores
        assert 'completeness' in scores
        assert 'consistency' in scores
        assert 'timeliness' in scores
        assert 'accuracy' in scores
        
        # 所有分数应该在0-1之间
        for score in scores.values():
            assert 0.0 <= score <= 1.0
        
        # 有问题的数据应该有较低的分数
        assert scores['completeness'] < 1.0  # 因为有缺失数据问题
        assert scores['consistency'] < 1.0   # 因为有一致性问题
    
    def test_generate_recommendations(self, engine):
        """测试建议生成"""
        issues = [
            DataQualityIssue(
                issue_type=DataQualityIssueType.MISSING_DATA,
                severity=DataQualitySeverity.HIGH,
                description="Missing data issue",
                affected_rows=[1, 2, 3],
                affected_columns=['column1'],
                confidence_score=1.0,
                suggested_action="Fix missing data"
            ),
            DataQualityIssue(
                issue_type=DataQualityIssueType.DUPLICATE_DATA,
                severity=DataQualitySeverity.MEDIUM,
                description="Duplicate data issue",
                affected_rows=[4, 5],
                affected_columns=['column2'],
                confidence_score=1.0,
                suggested_action="Remove duplicates"
            ),
            DataQualityIssue(
                issue_type=DataQualityIssueType.OUTLIER_DATA,
                severity=DataQualitySeverity.CRITICAL,
                description="Outlier data issue",
                affected_rows=[6],
                affected_columns=['column3'],
                confidence_score=0.8,
                suggested_action="Review outliers"
            )
        ]
        
        recommendations = engine._generate_recommendations(issues)
        
        assert len(recommendations) > 0
        
        # 检查是否包含针对不同问题类型的建议
        rec_text = ' '.join(recommendations)
        assert 'missing data' in rec_text.lower()
        assert 'duplicate' in rec_text.lower()
        assert 'outlier' in rec_text.lower()
        
        # 检查是否包含关键问题的特殊建议
        assert any('critical' in rec.lower() for rec in recommendations)
    
    def test_clean_data(self, engine):
        """测试数据清洗"""
        # 创建包含重复和异常值的数据
        data = TestDataGenerator.generate_data_with_duplicates(100, 0.1)
        
        # 创建质量报告
        report = engine.validate_data(data, "test_cleaning")
        
        # 清洗数据
        cleaned_data = engine.clean_data(data, report)
        
        # 清洗后的数据应该更少（去除了重复和异常值）
        assert len(cleaned_data) <= len(data)
        
        # 验证清洗后的数据质量
        cleaned_report = engine.validate_data(cleaned_data, "test_cleaned")
        
        # 清洗后的数据质量应该更好
        assert cleaned_report.overall_score >= report.overall_score
    
    def test_save_and_load_ml_model(self, engine):
        """测试ML模型保存和加载"""
        training_data = TestDataGenerator.generate_valid_stock_data(1000)
        engine.train_ml_detector(training_data)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # 保存模型
            engine.save_model(tmp_path)
            
            # 创建新引擎并加载模型
            new_engine = EnhancedDataQualityEngine()
            new_engine.load_model(tmp_path)
            
            assert new_engine.ml_detector.is_fitted is True
            
            # 测试加载的模型
            test_data = TestDataGenerator.generate_valid_stock_data(100)
            report = new_engine.validate_data(test_data, "test_loaded_model")
            
            assert isinstance(report, DataQualityReport)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestPerformanceAndScalability:
    """性能和可扩展性测试类"""
    
    @pytest.fixture
    def engine(self):
        """创建数据质量引擎实例"""
        return EnhancedDataQualityEngine()
    
    def test_large_dataset_validation(self, engine):
        """测试大数据集验证性能"""
        import time
        
        # 创建大数据集
        large_data = TestDataGenerator.generate_valid_stock_data(10000)
        
        start_time = time.time()
        report = engine.validate_data(large_data, "large_dataset")
        end_time = time.time()
        
        # 验证应该在合理时间内完成（例如30秒）
        assert end_time - start_time < 30
        assert report.total_rows == 10000
        assert isinstance(report.overall_score, float)
    
    def test_memory_usage_with_large_data(self, engine):
        """测试大数据的内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 处理大数据集
        large_data = TestDataGenerator.generate_valid_stock_data(50000)
        report = engine.validate_data(large_data, "memory_test")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 内存增长应该在合理范围内（例如小于1GB）
        assert memory_increase < 1024 * 1024 * 1024  # 1GB
        assert report.total_rows == 50000
    
    def test_concurrent_validation(self, engine):
        """测试并发验证"""
        import threading
        import time
        
        def validate_data_thread(thread_id):
            data = TestDataGenerator.generate_valid_stock_data(1000)
            report = engine.validate_data(data, f"thread_{thread_id}")
            return report
        
        # 创建多个线程
        threads = []
        results = {}
        
        def thread_worker(thread_id):
            results[thread_id] = validate_data_thread(thread_id)
        
        start_time = time.time()
        
        for i in range(5):
            thread = threading.Thread(target=thread_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # 所有线程应该成功完成
        assert len(results) == 5
        assert all(isinstance(report, DataQualityReport) for report in results.values())
        
        # 并发执行应该在合理时间内完成
        assert end_time - start_time < 60
    
    def test_ml_detector_scalability(self, engine):
        """测试ML检测器可扩展性"""
        import time
        
        # 训练ML检测器
        training_data = TestDataGenerator.generate_valid_stock_data(5000)
        
        start_time = time.time()
        engine.train_ml_detector(training_data)
        training_time = time.time() - start_time
        
        # 训练应该在合理时间内完成
        assert training_time < 60  # 1分钟
        
        # 测试检测性能
        test_data = TestDataGenerator.generate_data_with_outliers(10000)
        
        start_time = time.time()
        anomaly_indices, anomaly_scores = engine.ml_detector.detect_anomalies(test_data)
        detection_time = time.time() - start_time
        
        # 检测应该在合理时间内完成
        assert detection_time < 30  # 30秒
        assert isinstance(anomaly_indices, list)
        assert isinstance(anomaly_scores, list)


class TestEdgeCasesAndErrorHandling:
    """边界情况和错误处理测试类"""
    
    @pytest.fixture
    def engine(self):
        """创建数据质量引擎实例"""
        return EnhancedDataQualityEngine()
    
    def test_empty_dataframe_validation(self, engine):
        """测试空数据框验证"""
        empty_data = pd.DataFrame()
        report = engine.validate_data(empty_data, "empty_data")
        
        assert report.total_rows == 0
        assert report.total_columns == 0
        assert isinstance(report.overall_score, float)
    
    def test_single_row_dataframe(self, engine):
        """测试单行数据框"""
        single_row_data = pd.DataFrame({
            'stock_code': ['000001'],
            'trade_date': [datetime.now()],
            'close_price': [10.0]
        })
        
        report = engine.validate_data(single_row_data, "single_row")
        
        assert report.total_rows == 1
        assert isinstance(report.overall_score, float)
    
    def test_all_null_columns(self, engine):
        """测试全空列"""
        null_data = pd.DataFrame({
            'stock_code': [None, None, None],
            'trade_date': [None, None, None],
            'close_price': [None, None, None]
        })
        
        report = engine.validate_data(null_data, "all_null")
        
        # 应该检测到严重的完整性问题
        missing_issues = [
            issue for issue in report.issues 
            if issue.issue_type == DataQualityIssueType.MISSING_DATA
        ]
        assert len(missing_issues) > 0
        assert report.completeness_score < 0.5
    
    def test_mixed_data_types(self, engine):
        """测试混合数据类型"""
        mixed_data = pd.DataFrame({
            'stock_code': ['000001', 123, None],
            'trade_date': [datetime.now(), '2024-01-01', 'invalid'],
            'close_price': [10.0, 'twenty', -5]
        })
        
        # 应该能够处理混合数据类型而不崩溃
        report = engine.validate_data(mixed_data, "mixed_types")
        
        assert isinstance(report, DataQualityReport)
        assert report.total_rows == 3
    
    def test_rule_execution_error(self, engine):
        """测试规则执行错误"""
        # 创建一个会抛出异常的规则
        class FailingRule(DataQualityRule):
            def __init__(self):
                super().__init__("Failing Rule", "Always fails", DataQualitySeverity.LOW)
            
            def validate(self, data):
                raise Exception("Rule execution failed")
        
        engine.add_rule(FailingRule())
        
        data = TestDataGenerator.generate_valid_stock_data(10)
        report = engine.validate_data(data, "rule_error_test")
        
        # 应该包含规则执行错误的问题
        rule_error_issues = [
            issue for issue in report.issues 
            if 'failed to execute' in issue.description
        ]
        assert len(rule_error_issues) > 0
    
    def test_ml_detector_with_insufficient_data(self, engine):
        """测试ML检测器数据不足的情况"""
        # 创建很少的训练数据
        insufficient_data = TestDataGenerator.generate_valid_stock_data(5)
        
        # 应该能够处理数据不足的情况
        try:
            engine.train_ml_detector(insufficient_data)
            # 如果没有抛出异常，检查是否正确处理
            assert engine.ml_detector.is_fitted in [True, False]
        except Exception as e:
            # 如果抛出异常，应该是合理的异常
            assert isinstance(e, (ValueError, RuntimeError))
    
    def test_extreme_values_handling(self, engine):
        """测试极值处理"""
        extreme_data = pd.DataFrame({
            'stock_code': ['000001', '000002', '000003'],
            'trade_date': [datetime.now()] * 3,
            'close_price': [float('inf'), float('-inf'), float('nan')],
            'volume': [0, -1000000, 999999999999],
            'amount': [0.0, -999999999.99, 999999999999.99]
        })
        
        # 应该能够处理极值而不崩溃
        report = engine.validate_data(extreme_data, "extreme_values")
        
        assert isinstance(report, DataQualityReport)
        assert len(report.issues) > 0  # 应该检测到问题


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])