"""
ELK日志分析系统实现

该模块实现了基于Elasticsearch、Logstash和Kibana的日志分析系统，
提供结构化日志记录、搜索、异常检测和可视化功能。
"""

import json
import logging
import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import threading
from collections import defaultdict, deque
import time

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    print("Warning: elasticsearch package not available. Using mock implementation.")


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """日志分类枚举"""
    SYSTEM = "system"
    DATA_ACCESS = "data_access"
    API = "api"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS = "business"
    ERROR = "error"


@dataclass
class LogEntry:
    """结构化日志条目"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    component: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['level'] = self.level.value
        data['category'] = self.category.value
        return data


@dataclass
class LogPattern:
    """日志模式定义"""
    name: str
    pattern: str
    severity: LogLevel
    description: str
    action: Optional[str] = None


@dataclass
class LogAnomaly:
    """日志异常"""
    timestamp: datetime
    pattern_name: str
    message: str
    severity: LogLevel
    count: int
    first_occurrence: datetime
    last_occurrence: datetime
    metadata: Dict[str, Any]


class LogAggregator:
    """日志聚合器"""
    
    def __init__(self, window_size: int = 300):  # 5分钟窗口
        self.window_size = window_size
        self.log_counts = defaultdict(lambda: defaultdict(int))
        self.error_patterns = defaultdict(int)
        self.performance_metrics = defaultdict(list)
        self.lock = threading.Lock()
    
    def add_log(self, log_entry: LogEntry):
        """添加日志条目进行聚合"""
        with self.lock:
            timestamp_key = int(log_entry.timestamp.timestamp() // self.window_size)
            
            # 统计日志级别
            self.log_counts[timestamp_key][log_entry.level.value] += 1
            
            # 统计错误模式
            if log_entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                error_key = f"{log_entry.component}:{log_entry.message[:100]}"
                self.error_patterns[error_key] += 1
            
            # 收集性能指标
            if log_entry.duration_ms is not None:
                self.performance_metrics[log_entry.component].append(log_entry.duration_ms)
    
    def get_aggregated_stats(self, hours: int = 1) -> Dict[str, Any]:
        """获取聚合统计信息"""
        with self.lock:
            current_time = int(time.time() // self.window_size)
            start_time = current_time - (hours * 3600 // self.window_size)
            
            stats = {
                'log_counts': {},
                'error_patterns': dict(self.error_patterns),
                'performance_summary': {}
            }
            
            # 聚合日志计数
            for timestamp_key in range(start_time, current_time + 1):
                if timestamp_key in self.log_counts:
                    for level, count in self.log_counts[timestamp_key].items():
                        if level not in stats['log_counts']:
                            stats['log_counts'][level] = 0
                        stats['log_counts'][level] += count
            
            # 计算性能摘要
            for component, durations in self.performance_metrics.items():
                if durations:
                    stats['performance_summary'][component] = {
                        'avg_duration': sum(durations) / len(durations),
                        'max_duration': max(durations),
                        'min_duration': min(durations),
                        'count': len(durations)
                    }
            
            return stats


class PatternMatcher:
    """日志模式匹配器"""
    
    def __init__(self):
        self.patterns = [
            LogPattern(
                name="database_connection_error",
                pattern=r"database.*connection.*error|connection.*database.*failed|database.*connection.*timeout",
                severity=LogLevel.ERROR,
                description="数据库连接错误",
                action="check_database_connectivity"
            ),
            LogPattern(
                name="api_timeout",
                pattern=r"timeout.*api|api.*timeout|request.*timeout",
                severity=LogLevel.WARNING,
                description="API请求超时",
                action="check_api_performance"
            ),
            LogPattern(
                name="memory_warning",
                pattern=r"memory.*warning|out of memory|memory usage.*high",
                severity=LogLevel.WARNING,
                description="内存使用警告",
                action="monitor_memory_usage"
            ),
            LogPattern(
                name="authentication_failure",
                pattern=r"authentication.*failed|login.*failed|unauthorized",
                severity=LogLevel.ERROR,
                description="认证失败",
                action="security_review"
            ),
            LogPattern(
                name="data_validation_error",
                pattern=r"validation.*error|invalid.*data|data.*corrupt",
                severity=LogLevel.ERROR,
                description="数据验证错误",
                action="data_quality_check"
            )
        ]
        self.compiled_patterns = [
            (pattern.name, re.compile(pattern.pattern, re.IGNORECASE), pattern)
            for pattern in self.patterns
        ]
    
    def match_patterns(self, message: str) -> List[LogPattern]:
        """匹配日志模式"""
        matches = []
        for name, compiled_pattern, pattern in self.compiled_patterns:
            if compiled_pattern.search(message):
                matches.append(pattern)
        return matches


class AnomalyDetector:
    """日志异常检测器"""
    
    def __init__(self, threshold_multiplier: float = 3.0):
        self.threshold_multiplier = threshold_multiplier
        self.baseline_stats = defaultdict(lambda: defaultdict(list))
        self.anomalies = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def update_baseline(self, component: str, metric: str, value: float):
        """更新基线统计"""
        with self.lock:
            self.baseline_stats[component][metric].append(value)
            # 保持最近100个数据点
            if len(self.baseline_stats[component][metric]) > 100:
                self.baseline_stats[component][metric].pop(0)
    
    def detect_anomaly(self, component: str, metric: str, value: float) -> bool:
        """检测异常"""
        with self.lock:
            if component not in self.baseline_stats or metric not in self.baseline_stats[component]:
                return False
            
            values = self.baseline_stats[component][metric]
            if len(values) < 10:  # 需要足够的基线数据
                return False
            
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5
            
            # 如果标准差太小，使用固定阈值
            if std_dev < 1.0:
                std_dev = 1.0
            
            threshold = mean + (self.threshold_multiplier * std_dev)
            return value > threshold
    
    def add_anomaly(self, anomaly: LogAnomaly):
        """添加异常记录"""
        with self.lock:
            self.anomalies.append(anomaly)
    
    def get_recent_anomalies(self, hours: int = 1) -> List[LogAnomaly]:
        """获取最近的异常"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        with self.lock:
            return [
                anomaly for anomaly in self.anomalies
                if anomaly.timestamp >= cutoff_time
            ]


class ELKLogger:
    """ELK日志记录器"""
    
    def __init__(self, 
                 elasticsearch_hosts: List[str] = None,
                 index_prefix: str = "stock-analysis",
                 buffer_size: int = 100):
        self.elasticsearch_hosts = elasticsearch_hosts or ["localhost:9200"]
        self.index_prefix = index_prefix
        self.buffer_size = buffer_size
        
        # 初始化Elasticsearch客户端
        if ELASTICSEARCH_AVAILABLE:
            try:
                self.es_client = Elasticsearch(
                    self.elasticsearch_hosts,
                    timeout=30,
                    max_retries=3,
                    retry_on_timeout=True
                )
                self.es_available = True
            except Exception as e:
                print(f"Warning: Failed to connect to Elasticsearch: {e}")
                self.es_available = False
        else:
            self.es_available = False
        
        # 初始化组件
        self.aggregator = LogAggregator()
        self.pattern_matcher = PatternMatcher()
        self.anomaly_detector = AnomalyDetector()
        
        # 日志缓冲区
        self.log_buffer = []
        self.buffer_lock = threading.Lock()
        
        # 启动后台任务
        self.running = True
        self.flush_thread = threading.Thread(target=self._flush_logs_periodically)
        self.flush_thread.daemon = True
        self.flush_thread.start()
    
    def log(self, 
            level: LogLevel,
            category: LogCategory,
            message: str,
            component: str,
            **kwargs):
        """记录日志"""
        # 提取LogEntry的标准字段
        standard_fields = {
            'user_id': kwargs.pop('user_id', None),
            'session_id': kwargs.pop('session_id', None),
            'request_id': kwargs.pop('request_id', None),
            'stack_trace': kwargs.pop('stack_trace', None),
            'duration_ms': kwargs.pop('duration_ms', None)
        }
        
        # 剩余的kwargs作为metadata
        metadata = kwargs if kwargs else None
        
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            component=component,
            metadata=metadata,
            **{k: v for k, v in standard_fields.items() if v is not None}
        )
        
        # 添加到聚合器
        self.aggregator.add_log(log_entry)
        
        # 模式匹配
        matched_patterns = self.pattern_matcher.match_patterns(message)
        if matched_patterns:
            for pattern in matched_patterns:
                anomaly = LogAnomaly(
                    timestamp=log_entry.timestamp,
                    pattern_name=pattern.name,
                    message=message,
                    severity=pattern.severity,
                    count=1,
                    first_occurrence=log_entry.timestamp,
                    last_occurrence=log_entry.timestamp,
                    metadata={'component': component, 'action': pattern.action}
                )
                self.anomaly_detector.add_anomaly(anomaly)
        
        # 添加到缓冲区
        with self.buffer_lock:
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) >= self.buffer_size:
                self._flush_logs()
    
    def _flush_logs(self):
        """刷新日志到Elasticsearch"""
        if not self.log_buffer:
            return
        
        logs_to_flush = self.log_buffer.copy()
        self.log_buffer.clear()
        
        if self.es_available:
            try:
                # 准备批量插入数据
                actions = []
                for log_entry in logs_to_flush:
                    index_name = f"{self.index_prefix}-{log_entry.timestamp.strftime('%Y-%m-%d')}"
                    action = {
                        "_index": index_name,
                        "_source": log_entry.to_dict()
                    }
                    actions.append(action)
                
                # 批量插入
                if actions:
                    bulk(self.es_client, actions)
            except Exception as e:
                print(f"Failed to flush logs to Elasticsearch: {e}")
        else:
            # 如果Elasticsearch不可用，输出到标准日志
            for log_entry in logs_to_flush:
                print(f"[{log_entry.timestamp}] {log_entry.level.value} "
                      f"[{log_entry.component}] {log_entry.message}")
    
    def _flush_logs_periodically(self):
        """定期刷新日志"""
        while self.running:
            time.sleep(10)  # 每10秒刷新一次
            with self.buffer_lock:
                if self.log_buffer:
                    self._flush_logs()
    
    def search_logs(self, 
                   query: str = "*",
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   level: Optional[LogLevel] = None,
                   component: Optional[str] = None,
                   size: int = 100) -> List[Dict[str, Any]]:
        """搜索日志"""
        if not self.es_available:
            return []
        
        # 构建查询
        must_clauses = []
        
        if query != "*":
            must_clauses.append({
                "multi_match": {
                    "query": query,
                    "fields": ["message", "component", "metadata.*"]
                }
            })
        
        if start_time or end_time:
            time_range = {}
            if start_time:
                time_range["gte"] = start_time.isoformat()
            if end_time:
                time_range["lte"] = end_time.isoformat()
            must_clauses.append({
                "range": {
                    "timestamp": time_range
                }
            })
        
        if level:
            must_clauses.append({
                "term": {
                    "level": level.value
                }
            })
        
        if component:
            must_clauses.append({
                "term": {
                    "component": component
                }
            })
        
        search_body = {
            "query": {
                "bool": {
                    "must": must_clauses if must_clauses else [{"match_all": {}}]
                }
            },
            "sort": [
                {"timestamp": {"order": "desc"}}
            ],
            "size": size
        }
        
        try:
            # 搜索所有相关索引
            index_pattern = f"{self.index_prefix}-*"
            response = self.es_client.search(
                index=index_pattern,
                body=search_body
            )
            
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            print(f"Failed to search logs: {e}")
            return []
    
    def get_log_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """获取日志统计信息"""
        stats = self.aggregator.get_aggregated_stats(hours)
        anomalies = self.anomaly_detector.get_recent_anomalies(hours)
        
        return {
            "aggregated_stats": stats,
            "recent_anomalies": [
                {
                    "pattern_name": anomaly.pattern_name,
                    "message": anomaly.message,
                    "severity": anomaly.severity.value,
                    "count": anomaly.count,
                    "timestamp": anomaly.timestamp.isoformat()
                }
                for anomaly in anomalies
            ],
            "total_anomalies": len(anomalies)
        }
    
    def create_dashboard_data(self) -> Dict[str, Any]:
        """创建仪表板数据"""
        stats = self.get_log_statistics()
        
        return {
            "log_levels": stats["aggregated_stats"]["log_counts"],
            "error_patterns": stats["aggregated_stats"]["error_patterns"],
            "performance_metrics": stats["aggregated_stats"]["performance_summary"],
            "anomalies": stats["recent_anomalies"],
            "health_status": self._calculate_health_status(stats)
        }
    
    def _calculate_health_status(self, stats: Dict[str, Any]) -> str:
        """计算系统健康状态"""
        log_counts = stats["aggregated_stats"]["log_counts"]
        total_logs = sum(log_counts.values())
        
        if total_logs == 0:
            return "unknown"
        
        error_ratio = (log_counts.get("ERROR", 0) + log_counts.get("CRITICAL", 0)) / total_logs
        warning_ratio = log_counts.get("WARNING", 0) / total_logs
        
        if error_ratio > 0.1:  # 超过10%的错误日志
            return "critical"
        elif error_ratio > 0.05 or warning_ratio > 0.2:  # 超过5%的错误或20%的警告
            return "warning"
        else:
            return "healthy"
    
    def shutdown(self):
        """关闭日志系统"""
        self.running = False
        if hasattr(self, 'flush_thread'):
            self.flush_thread.join(timeout=5)
        
        # 最后一次刷新
        with self.buffer_lock:
            if self.log_buffer:
                self._flush_logs()


# 全局日志实例
_global_logger: Optional[ELKLogger] = None


def initialize_elk_logging(elasticsearch_hosts: List[str] = None,
                          index_prefix: str = "stock-analysis") -> ELKLogger:
    """初始化全局ELK日志系统"""
    global _global_logger
    _global_logger = ELKLogger(elasticsearch_hosts, index_prefix)
    return _global_logger


def get_elk_logger() -> Optional[ELKLogger]:
    """获取全局ELK日志实例"""
    return _global_logger


def log_info(message: str, component: str, **kwargs):
    """记录信息日志"""
    if _global_logger:
        _global_logger.log(LogLevel.INFO, LogCategory.SYSTEM, message, component, **kwargs)


def log_warning(message: str, component: str, **kwargs):
    """记录警告日志"""
    if _global_logger:
        _global_logger.log(LogLevel.WARNING, LogCategory.SYSTEM, message, component, **kwargs)


def log_error(message: str, component: str, **kwargs):
    """记录错误日志"""
    if _global_logger:
        _global_logger.log(LogLevel.ERROR, LogCategory.ERROR, message, component, **kwargs)


def log_performance(message: str, component: str, duration_ms: float, **kwargs):
    """记录性能日志"""
    if _global_logger:
        _global_logger.log(
            LogLevel.INFO, 
            LogCategory.PERFORMANCE, 
            message, 
            component, 
            duration_ms=duration_ms,
            **kwargs
        )