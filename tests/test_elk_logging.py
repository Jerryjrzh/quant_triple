"""
ELK日志分析系统测试

测试ELK日志系统的各个组件，包括日志记录、搜索、异常检测和可视化功能。
"""

import pytest
import asyncio
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

from stock_analysis_system.monitoring.elk_logging import (
    ELKLogger, LogLevel, LogCategory, LogEntry, LogPattern, LogAnomaly,
    LogAggregator, PatternMatcher, AnomalyDetector,
    initialize_elk_logging, get_elk_logger,
    log_info, log_warning, log_error, log_performance
)


class TestLogEntry:
    """测试日志条目"""
    
    def test_log_entry_creation(self):
        """测试日志条目创建"""
        timestamp = datetime.now()
        entry = LogEntry(
            timestamp=timestamp,
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            message="Test message",
            component="test_component",
            user_id="user123",
            metadata={"key": "value"}
        )
        
        assert entry.timestamp == timestamp
        assert entry.level == LogLevel.INFO
        assert entry.category == LogCategory.SYSTEM
        assert entry.message == "Test message"
        assert entry.component == "test_component"
        assert entry.user_id == "user123"
        assert entry.metadata == {"key": "value"}
    
    def test_log_entry_to_dict(self):
        """测试日志条目转换为字典"""
        timestamp = datetime.now()
        entry = LogEntry(
            timestamp=timestamp,
            level=LogLevel.ERROR,
            category=LogCategory.API,
            message="API error",
            component="api_handler"
        )
        
        data = entry.to_dict()
        assert data["timestamp"] == timestamp.isoformat()
        assert data["level"] == "ERROR"
        assert data["category"] == "api"
        assert data["message"] == "API error"
        assert data["component"] == "api_handler"


class TestLogAggregator:
    """测试日志聚合器"""
    
    def test_aggregator_initialization(self):
        """测试聚合器初始化"""
        aggregator = LogAggregator(window_size=60)
        assert aggregator.window_size == 60
        assert len(aggregator.log_counts) == 0
        assert len(aggregator.error_patterns) == 0
        assert len(aggregator.performance_metrics) == 0
    
    def test_add_log_entry(self):
        """测试添加日志条目"""
        aggregator = LogAggregator(window_size=60)
        
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            message="Test message",
            component="test_component",
            duration_ms=100.5
        )
        
        aggregator.add_log(log_entry)
        
        # 检查日志计数
        assert len(aggregator.log_counts) > 0
        
        # 检查性能指标
        assert "test_component" in aggregator.performance_metrics
        assert 100.5 in aggregator.performance_metrics["test_component"]
    
    def test_error_pattern_tracking(self):
        """测试错误模式跟踪"""
        aggregator = LogAggregator()
        
        error_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            category=LogCategory.ERROR,
            message="Database connection failed",
            component="database"
        )
        
        aggregator.add_log(error_entry)
        
        # 检查错误模式
        error_key = "database:Database connection failed"
        assert error_key in aggregator.error_patterns
        assert aggregator.error_patterns[error_key] == 1
    
    def test_get_aggregated_stats(self):
        """测试获取聚合统计"""
        aggregator = LogAggregator(window_size=60)
        
        # 添加多个日志条目
        for i in range(5):
            log_entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO if i < 3 else LogLevel.ERROR,
                category=LogCategory.SYSTEM,
                message=f"Message {i}",
                component="test_component",
                duration_ms=100.0 + i
            )
            aggregator.add_log(log_entry)
        
        stats = aggregator.get_aggregated_stats(hours=1)
        
        assert "log_counts" in stats
        assert "error_patterns" in stats
        assert "performance_summary" in stats
        
        # 检查日志计数
        assert stats["log_counts"]["INFO"] == 3
        assert stats["log_counts"]["ERROR"] == 2
        
        # 检查性能摘要
        assert "test_component" in stats["performance_summary"]
        perf_summary = stats["performance_summary"]["test_component"]
        assert perf_summary["count"] == 5
        assert perf_summary["avg_duration"] == 102.0


class TestPatternMatcher:
    """测试模式匹配器"""
    
    def test_pattern_matcher_initialization(self):
        """测试模式匹配器初始化"""
        matcher = PatternMatcher()
        assert len(matcher.patterns) > 0
        assert len(matcher.compiled_patterns) == len(matcher.patterns)
    
    def test_database_error_pattern(self):
        """测试数据库错误模式匹配"""
        matcher = PatternMatcher()
        
        messages = [
            "database connection error occurred",
            "connection to database failed",
            "database connection timeout"
        ]
        
        for message in messages:
            matches = matcher.match_patterns(message)
            assert len(matches) > 0
            assert any(pattern.name == "database_connection_error" for pattern in matches)
    
    def test_api_timeout_pattern(self):
        """测试API超时模式匹配"""
        matcher = PatternMatcher()
        
        messages = [
            "API request timeout",
            "Timeout occurred while calling API",
            "Request timeout error"
        ]
        
        for message in messages:
            matches = matcher.match_patterns(message)
            assert len(matches) > 0
            assert any(pattern.name == "api_timeout" for pattern in matches)
    
    def test_no_pattern_match(self):
        """测试无模式匹配"""
        matcher = PatternMatcher()
        
        message = "Normal operation completed successfully"
        matches = matcher.match_patterns(message)
        assert len(matches) == 0
    
    def test_multiple_pattern_match(self):
        """测试多模式匹配"""
        matcher = PatternMatcher()
        
        message = "Database connection timeout error"
        matches = matcher.match_patterns(message)
        
        # 这个消息可能匹配多个模式
        assert len(matches) >= 1
        pattern_names = [pattern.name for pattern in matches]
        assert "database_connection_error" in pattern_names


class TestAnomalyDetector:
    """测试异常检测器"""
    
    def test_anomaly_detector_initialization(self):
        """测试异常检测器初始化"""
        detector = AnomalyDetector(threshold_multiplier=2.0)
        assert detector.threshold_multiplier == 2.0
        assert len(detector.baseline_stats) == 0
        assert len(detector.anomalies) == 0
    
    def test_update_baseline(self):
        """测试更新基线统计"""
        detector = AnomalyDetector()
        
        # 添加基线数据
        for i in range(20):
            detector.update_baseline("test_component", "response_time", 100.0 + i)
        
        assert "test_component" in detector.baseline_stats
        assert "response_time" in detector.baseline_stats["test_component"]
        assert len(detector.baseline_stats["test_component"]["response_time"]) == 20
    
    def test_detect_normal_value(self):
        """测试检测正常值"""
        detector = AnomalyDetector()
        
        # 建立基线
        for i in range(20):
            detector.update_baseline("test_component", "response_time", 100.0)
        
        # 测试正常值
        is_anomaly = detector.detect_anomaly("test_component", "response_time", 101.0)
        assert not is_anomaly
    
    def test_detect_anomaly_value(self):
        """测试检测异常值"""
        detector = AnomalyDetector(threshold_multiplier=2.0)
        
        # 建立基线（平均值100，标准差很小）
        for i in range(20):
            detector.update_baseline("test_component", "response_time", 100.0)
        
        # 测试异常值
        is_anomaly = detector.detect_anomaly("test_component", "response_time", 500.0)
        assert is_anomaly
    
    def test_insufficient_baseline_data(self):
        """测试基线数据不足的情况"""
        detector = AnomalyDetector()
        
        # 只添加少量基线数据
        for i in range(5):
            detector.update_baseline("test_component", "response_time", 100.0)
        
        # 应该不检测异常（数据不足）
        is_anomaly = detector.detect_anomaly("test_component", "response_time", 500.0)
        assert not is_anomaly
    
    def test_add_and_get_anomalies(self):
        """测试添加和获取异常"""
        detector = AnomalyDetector()
        
        anomaly = LogAnomaly(
            timestamp=datetime.now(),
            pattern_name="test_pattern",
            message="Test anomaly",
            severity=LogLevel.WARNING,
            count=1,
            first_occurrence=datetime.now(),
            last_occurrence=datetime.now(),
            metadata={"component": "test"}
        )
        
        detector.add_anomaly(anomaly)
        
        recent_anomalies = detector.get_recent_anomalies(hours=1)
        assert len(recent_anomalies) == 1
        assert recent_anomalies[0].pattern_name == "test_pattern"


class TestELKLogger:
    """测试ELK日志记录器"""
    
    @pytest.fixture
    def mock_elasticsearch(self):
        """模拟Elasticsearch客户端"""
        with patch('stock_analysis_system.monitoring.elk_logging.ELASTICSEARCH_AVAILABLE', True):
            # 创建一个模拟的Elasticsearch类
            mock_es_class = Mock()
            mock_client = Mock()
            mock_es_class.return_value = mock_client
            
            # 将模拟的类添加到模块中
            import stock_analysis_system.monitoring.elk_logging as elk_module
            elk_module.Elasticsearch = mock_es_class
            
            yield mock_client
            
            # 清理
            if hasattr(elk_module, 'Elasticsearch'):
                delattr(elk_module, 'Elasticsearch')
    
    def test_elk_logger_initialization_without_es(self):
        """测试不使用Elasticsearch的初始化"""
        logger = ELKLogger()
        assert not logger.es_available
        assert logger.index_prefix == "stock-analysis"
        assert logger.buffer_size == 100
        assert isinstance(logger.aggregator, LogAggregator)
        assert isinstance(logger.pattern_matcher, PatternMatcher)
        assert isinstance(logger.anomaly_detector, AnomalyDetector)
    
    def test_elk_logger_initialization_with_es(self, mock_elasticsearch):
        """测试使用Elasticsearch的初始化"""
        logger = ELKLogger(elasticsearch_hosts=["localhost:9200"])
        assert logger.es_available
        assert logger.es_client == mock_elasticsearch
    
    def test_log_entry_creation(self):
        """测试日志条目创建"""
        logger = ELKLogger()
        
        logger.log(
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            message="Test log message",
            component="test_component",
            user_id="user123"
        )
        
        # 检查缓冲区
        assert len(logger.log_buffer) == 1
        log_entry = logger.log_buffer[0]
        assert log_entry.level == LogLevel.INFO
        assert log_entry.message == "Test log message"
        assert log_entry.component == "test_component"
        assert log_entry.user_id == "user123"
    
    def test_pattern_matching_in_log(self):
        """测试日志中的模式匹配"""
        logger = ELKLogger()
        
        # 记录包含错误模式的日志
        logger.log(
            level=LogLevel.ERROR,
            category=LogCategory.ERROR,
            message="Database connection error occurred",
            component="database"
        )
        
        # 检查是否检测到异常
        anomalies = logger.anomaly_detector.get_recent_anomalies(hours=1)
        assert len(anomalies) > 0
        assert any(anomaly.pattern_name == "database_connection_error" for anomaly in anomalies)
    
    def test_buffer_flush_on_size(self):
        """测试缓冲区大小触发刷新"""
        logger = ELKLogger(buffer_size=5)
        
        # 添加足够的日志触发刷新
        for i in range(6):
            logger.log(
                level=LogLevel.INFO,
                category=LogCategory.SYSTEM,
                message=f"Test message {i}",
                component="test"
            )
        
        # 缓冲区应该被清空
        assert len(logger.log_buffer) == 1  # 最后一个日志
    
    def test_search_logs_without_es(self):
        """测试不使用Elasticsearch的日志搜索"""
        logger = ELKLogger()
        
        results = logger.search_logs("test query")
        assert results == []
    
    def test_search_logs_with_es(self, mock_elasticsearch):
        """测试使用Elasticsearch的日志搜索"""
        mock_elasticsearch.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"message": "test log", "level": "INFO"}},
                    {"_source": {"message": "another log", "level": "ERROR"}}
                ]
            }
        }
        
        logger = ELKLogger(elasticsearch_hosts=["localhost:9200"])
        
        results = logger.search_logs(
            query="test",
            level=LogLevel.INFO,
            component="test_component"
        )
        
        assert len(results) == 2
        assert results[0]["message"] == "test log"
        assert results[1]["message"] == "another log"
        
        # 验证搜索调用
        mock_elasticsearch.search.assert_called_once()
    
    def test_get_log_statistics(self):
        """测试获取日志统计"""
        logger = ELKLogger()
        
        # 添加一些日志
        logger.log(LogLevel.INFO, LogCategory.SYSTEM, "Info message", "component1")
        logger.log(LogLevel.ERROR, LogCategory.ERROR, "Error message", "component2")
        logger.log(LogLevel.WARNING, LogCategory.API, "Warning message", "component1")
        
        stats = logger.get_log_statistics(hours=1)
        
        assert "aggregated_stats" in stats
        assert "recent_anomalies" in stats
        assert "total_anomalies" in stats
        
        # 检查聚合统计
        log_counts = stats["aggregated_stats"]["log_counts"]
        assert log_counts.get("INFO", 0) >= 1
        assert log_counts.get("ERROR", 0) >= 1
        assert log_counts.get("WARNING", 0) >= 1
    
    def test_create_dashboard_data(self):
        """测试创建仪表板数据"""
        logger = ELKLogger()
        
        # 添加一些日志
        logger.log(LogLevel.INFO, LogCategory.SYSTEM, "Normal operation", "component1")
        logger.log(LogLevel.ERROR, LogCategory.ERROR, "Database connection error", "database")
        
        dashboard_data = logger.create_dashboard_data()
        
        assert "log_levels" in dashboard_data
        assert "error_patterns" in dashboard_data
        assert "performance_metrics" in dashboard_data
        assert "anomalies" in dashboard_data
        assert "health_status" in dashboard_data
        
        # 健康状态应该基于错误率计算
        assert dashboard_data["health_status"] in ["healthy", "warning", "critical", "unknown"]
    
    def test_health_status_calculation(self):
        """测试健康状态计算"""
        logger = ELKLogger()
        
        # 测试健康状态
        for i in range(10):
            logger.log(LogLevel.INFO, LogCategory.SYSTEM, f"Info {i}", "component")
        
        dashboard_data = logger.create_dashboard_data()
        assert dashboard_data["health_status"] == "healthy"
        
        # 测试警告状态
        for i in range(3):
            logger.log(LogLevel.ERROR, LogCategory.ERROR, f"Error {i}", "component")
        
        dashboard_data = logger.create_dashboard_data()
        assert dashboard_data["health_status"] in ["warning", "critical"]
    
    def test_concurrent_logging(self):
        """测试并发日志记录"""
        logger = ELKLogger()
        
        def log_worker(worker_id):
            for i in range(10):
                logger.log(
                    LogLevel.INFO,
                    LogCategory.SYSTEM,
                    f"Worker {worker_id} message {i}",
                    f"worker_{worker_id}"
                )
        
        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 检查日志数量
        stats = logger.get_log_statistics()
        total_logs = sum(stats["aggregated_stats"]["log_counts"].values())
        assert total_logs == 50  # 5个工作线程 × 10条日志
    
    def test_logger_shutdown(self):
        """测试日志系统关闭"""
        logger = ELKLogger()
        
        # 添加一些日志
        logger.log(LogLevel.INFO, LogCategory.SYSTEM, "Test message", "component")
        
        # 关闭日志系统
        logger.shutdown()
        
        assert not logger.running


class TestGlobalLoggerFunctions:
    """测试全局日志函数"""
    
    def test_initialize_elk_logging(self):
        """测试初始化全局ELK日志"""
        logger = initialize_elk_logging(
            elasticsearch_hosts=["localhost:9200"],
            index_prefix="test-logs"
        )
        
        assert isinstance(logger, ELKLogger)
        assert logger.index_prefix == "test-logs"
        
        # 检查全局实例
        global_logger = get_elk_logger()
        assert global_logger is logger
    
    def test_global_log_functions(self):
        """测试全局日志函数"""
        # 初始化全局日志
        initialize_elk_logging()
        logger = get_elk_logger()
        
        # 测试各种日志函数
        log_info("Info message", "test_component", user_id="user123")
        log_warning("Warning message", "test_component")
        log_error("Error message", "test_component", error_code="E001")
        log_performance("Performance message", "test_component", 150.5, operation="test_op")
        
        # 检查日志是否被记录
        assert len(logger.log_buffer) == 4
        
        # 检查日志内容
        info_log = logger.log_buffer[0]
        assert info_log.level == LogLevel.INFO
        assert info_log.message == "Info message"
        assert info_log.user_id == "user123"
        
        perf_log = logger.log_buffer[3]
        assert perf_log.level == LogLevel.INFO
        assert perf_log.category == LogCategory.PERFORMANCE
        assert perf_log.duration_ms == 150.5
    
    def test_global_functions_without_logger(self):
        """测试没有初始化日志时的全局函数"""
        # 重置全局日志
        import stock_analysis_system.monitoring.elk_logging as elk_module
        elk_module._global_logger = None
        
        # 调用全局函数不应该报错
        log_info("Info message", "test_component")
        log_warning("Warning message", "test_component")
        log_error("Error message", "test_component")
        log_performance("Performance message", "test_component", 100.0)
        
        # 应该没有异常抛出
        assert get_elk_logger() is None


class TestIntegrationScenarios:
    """测试集成场景"""
    
    def test_complete_logging_workflow(self):
        """测试完整的日志工作流"""
        logger = ELKLogger(buffer_size=10)
        
        # 模拟各种日志场景
        scenarios = [
            (LogLevel.INFO, "System started successfully", "system"),
            (LogLevel.INFO, "User login successful", "auth", {"user_id": "user123"}),
            (LogLevel.WARNING, "High memory usage detected", "monitor"),
            (LogLevel.ERROR, "Database connection error", "database"),
            (LogLevel.ERROR, "API timeout occurred", "api"),
            (LogLevel.INFO, "Data processing completed", "processor", {"duration_ms": 250.0}),
            (LogLevel.CRITICAL, "System shutdown initiated", "system")
        ]
        
        for scenario in scenarios:
            level, message, component = scenario[:3]
            kwargs = scenario[3] if len(scenario) > 3 else {}
            logger.log(level, LogCategory.SYSTEM, message, component, **kwargs)
        
        # 检查统计信息
        stats = logger.get_log_statistics()
        
        # 验证日志计数
        log_counts = stats["aggregated_stats"]["log_counts"]
        assert log_counts["INFO"] == 3
        assert log_counts["WARNING"] == 1
        assert log_counts["ERROR"] == 2
        assert log_counts["CRITICAL"] == 1
        
        # 验证异常检测
        assert stats["total_anomalies"] > 0
        
        # 验证仪表板数据
        dashboard_data = logger.create_dashboard_data()
        assert dashboard_data["health_status"] in ["warning", "critical"]
    
    def test_performance_monitoring_integration(self):
        """测试性能监控集成"""
        logger = ELKLogger()
        
        # 模拟性能数据
        components = ["api", "database", "cache", "processor"]
        
        for component in components:
            for i in range(10):
                duration = 100.0 + (i * 10)  # 递增的响应时间
                logger.log(
                    LogLevel.INFO,
                    LogCategory.PERFORMANCE,
                    f"{component} operation completed",
                    component,
                    duration_ms=duration,
                    operation=f"operation_{i}"
                )
        
        # 检查性能统计
        stats = logger.get_log_statistics()
        perf_summary = stats["aggregated_stats"]["performance_summary"]
        
        for component in components:
            assert component in perf_summary
            component_stats = perf_summary[component]
            assert component_stats["count"] == 10
            assert component_stats["avg_duration"] == 145.0  # (100+190)/2
            assert component_stats["min_duration"] == 100.0
            assert component_stats["max_duration"] == 190.0
    
    def test_error_pattern_detection_integration(self):
        """测试错误模式检测集成"""
        logger = ELKLogger()
        
        # 模拟各种错误模式
        error_messages = [
            "Database connection failed - timeout",
            "API request timeout after 30 seconds",
            "Memory usage warning - 85% utilized",
            "Authentication failed for user",
            "Data validation error - invalid format",
            "Database connection error - host unreachable",
            "API timeout - service unavailable"
        ]
        
        for message in error_messages:
            logger.log(
                LogLevel.ERROR,
                LogCategory.ERROR,
                message,
                "system"
            )
        
        # 检查异常检测结果
        anomalies = logger.anomaly_detector.get_recent_anomalies(hours=1)
        
        # 应该检测到多种错误模式
        pattern_names = {anomaly.pattern_name for anomaly in anomalies}
        expected_patterns = {
            "database_connection_error",
            "api_timeout",
            "memory_warning",
            "authentication_failure",
            "data_validation_error"
        }
        
        # 至少应该检测到一些预期的模式
        assert len(pattern_names.intersection(expected_patterns)) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])