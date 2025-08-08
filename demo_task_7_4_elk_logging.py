#!/usr/bin/env python3
"""
Task 7.4 ELK日志分析系统演示

本演示脚本展示了ELK日志分析系统的完整功能，包括：
1. 结构化日志记录
2. 日志模式匹配和异常检测
3. 日志聚合和统计分析
4. 性能监控集成
5. 仪表板数据生成
6. 日志搜索和查询
"""

import asyncio
import time
import threading
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

from stock_analysis_system.monitoring.elk_logging import (
    ELKLogger, LogLevel, LogCategory,
    initialize_elk_logging, get_elk_logger,
    log_info, log_warning, log_error, log_performance
)


class ELKLoggingDemo:
    """ELK日志系统演示类"""
    
    def __init__(self):
        self.logger = None
        self.demo_running = False
    
    def initialize_system(self):
        """初始化日志系统"""
        print("🚀 初始化ELK日志分析系统...")
        
        # 初始化全局日志系统
        self.logger = initialize_elk_logging(
            elasticsearch_hosts=["localhost:9200"],  # 如果有ES服务器
            index_prefix="stock-analysis-demo"
        )
        
        print(f"✅ 日志系统初始化完成")
        print(f"   - Elasticsearch可用: {self.logger.es_available}")
        print(f"   - 索引前缀: {self.logger.index_prefix}")
        print(f"   - 缓冲区大小: {self.logger.buffer_size}")
        print()
    
    def demonstrate_basic_logging(self):
        """演示基本日志记录功能"""
        print("📝 演示基本日志记录功能...")
        
        # 使用不同级别记录日志
        log_info("系统启动完成", "system", version="1.0.0", startup_time=2.5)
        log_info("用户登录成功", "auth", user_id="user123", ip="192.168.1.100")
        log_warning("内存使用率较高", "monitor", memory_usage=85.5, threshold=80.0)
        log_error("数据库连接失败", "database", error_code="DB001", retry_count=3)
        log_performance("API请求处理完成", "api", 150.5, endpoint="/api/stocks", method="GET")
        
        # 直接使用logger记录更复杂的日志
        self.logger.log(
            level=LogLevel.INFO,
            category=LogCategory.BUSINESS,
            message="股票数据更新完成",
            component="data_processor",
            user_id="system",
            metadata={
                "stocks_updated": 1500,
                "update_duration": 45.2,
                "data_source": "eastmoney",
                "success_rate": 98.5
            }
        )
        
        print(f"✅ 已记录 {len(self.logger.log_buffer)} 条日志")
        print()
    
    def demonstrate_pattern_matching(self):
        """演示日志模式匹配和异常检测"""
        print("🔍 演示日志模式匹配和异常检测...")
        
        # 模拟各种错误模式
        error_scenarios = [
            ("数据库连接超时", "database", "Database connection timeout after 30 seconds"),
            ("API请求超时", "api", "API request timeout - external service unavailable"),
            ("内存不足警告", "system", "Memory usage warning - 95% of available memory used"),
            ("用户认证失败", "auth", "Authentication failed - invalid credentials provided"),
            ("数据验证错误", "validator", "Data validation error - invalid stock symbol format"),
            ("数据库连接错误", "database", "Database connection error - host unreachable"),
            ("API超时异常", "api", "Timeout occurred while calling external API")
        ]
        
        for component, category, message in error_scenarios:
            self.logger.log(
                level=LogLevel.ERROR,
                category=LogCategory.ERROR,
                message=message,
                component=component,
                timestamp_override=datetime.now()
            )
        
        # 检查检测到的异常
        anomalies = self.logger.anomaly_detector.get_recent_anomalies(hours=1)
        
        print(f"✅ 检测到 {len(anomalies)} 个异常模式:")
        for anomaly in anomalies:
            print(f"   - {anomaly.pattern_name}: {anomaly.message[:50]}...")
        print()
    
    def demonstrate_log_aggregation(self):
        """演示日志聚合和统计分析"""
        print("📊 演示日志聚合和统计分析...")
        
        # 生成大量日志数据进行聚合
        components = ["api", "database", "cache", "processor", "monitor"]
        categories = [LogCategory.SYSTEM, LogCategory.API, LogCategory.PERFORMANCE, LogCategory.DATA_ACCESS]
        levels = [LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]
        
        for i in range(50):
            component = random.choice(components)
            category = random.choice(categories)
            level = random.choice(levels)
            
            # 根据级别生成不同的消息
            if level == LogLevel.INFO:
                message = f"{component} 操作完成"
            elif level == LogLevel.WARNING:
                message = f"{component} 性能警告"
            else:
                message = f"{component} 操作失败"
            
            self.logger.log(
                level=level,
                category=category,
                message=message,
                component=component,
                duration_ms=random.uniform(50, 500),
                metadata={"operation_id": f"op_{i}"}
            )
        
        # 获取聚合统计
        stats = self.logger.get_log_statistics(hours=1)
        
        print("✅ 日志聚合统计:")
        print(f"   - 日志级别分布: {stats['aggregated_stats']['log_counts']}")
        print(f"   - 错误模式数量: {len(stats['aggregated_stats']['error_patterns'])}")
        print(f"   - 性能指标组件: {list(stats['aggregated_stats']['performance_summary'].keys())}")
        print(f"   - 检测到的异常: {stats['total_anomalies']}")
        print()
    
    def demonstrate_performance_monitoring(self):
        """演示性能监控集成"""
        print("⚡ 演示性能监控集成...")
        
        # 模拟不同组件的性能数据
        performance_scenarios = [
            ("股票数据获取", "data_fetcher", 120.5),
            ("技术指标计算", "indicator_calculator", 85.2),
            ("风险评估", "risk_assessor", 200.8),
            ("数据库查询", "database", 45.3),
            ("缓存操作", "cache", 15.7),
            ("API响应", "api", 180.4),
            ("数据验证", "validator", 35.9),
            ("报告生成", "report_generator", 350.2)
        ]
        
        for operation, component, duration in performance_scenarios:
            log_performance(
                f"{operation}完成",
                component,
                duration,
                operation=operation.lower().replace(" ", "_"),
                success=True
            )
        
        # 模拟异常性能情况
        log_performance(
            "数据库查询超时",
            "database",
            5000.0,  # 异常高的响应时间
            operation="slow_query",
            success=False,
            query="SELECT * FROM large_table"
        )
        
        # 获取性能统计
        stats = self.logger.get_log_statistics()
        perf_summary = stats['aggregated_stats']['performance_summary']
        
        print("✅ 性能监控统计:")
        for component, metrics in perf_summary.items():
            print(f"   - {component}:")
            print(f"     平均响应时间: {metrics['avg_duration']:.2f}ms")
            print(f"     最大响应时间: {metrics['max_duration']:.2f}ms")
            print(f"     操作次数: {metrics['count']}")
        print()
    
    def demonstrate_dashboard_data(self):
        """演示仪表板数据生成"""
        print("📈 演示仪表板数据生成...")
        
        dashboard_data = self.logger.create_dashboard_data()
        
        print("✅ 仪表板数据:")
        print(f"   - 系统健康状态: {dashboard_data['health_status']}")
        print(f"   - 日志级别分布: {dashboard_data['log_levels']}")
        print(f"   - 错误模式数量: {len(dashboard_data['error_patterns'])}")
        print(f"   - 性能监控组件: {len(dashboard_data['performance_metrics'])}")
        print(f"   - 异常数量: {len(dashboard_data['anomalies'])}")
        
        # 显示前几个错误模式
        if dashboard_data['error_patterns']:
            print("   - 主要错误模式:")
            for pattern, count in list(dashboard_data['error_patterns'].items())[:3]:
                print(f"     {pattern}: {count}次")
        
        # 显示最近的异常
        if dashboard_data['anomalies']:
            print("   - 最近异常:")
            for anomaly in dashboard_data['anomalies'][:3]:
                print(f"     {anomaly['pattern_name']}: {anomaly['message'][:40]}...")
        print()
    
    def demonstrate_log_search(self):
        """演示日志搜索功能"""
        print("🔎 演示日志搜索功能...")
        
        # 如果Elasticsearch可用，演示搜索功能
        if self.logger.es_available:
            print("✅ Elasticsearch可用，演示搜索功能:")
            
            # 搜索错误日志
            error_logs = self.logger.search_logs(
                query="error",
                level=LogLevel.ERROR,
                size=5
            )
            print(f"   - 找到 {len(error_logs)} 条错误日志")
            
            # 搜索特定组件的日志
            api_logs = self.logger.search_logs(
                component="api",
                size=5
            )
            print(f"   - 找到 {len(api_logs)} 条API组件日志")
            
            # 时间范围搜索
            recent_logs = self.logger.search_logs(
                start_time=datetime.now() - timedelta(hours=1),
                size=10
            )
            print(f"   - 找到 {len(recent_logs)} 条最近1小时的日志")
        else:
            print("⚠️  Elasticsearch不可用，跳过搜索演示")
            print("   - 在生产环境中，可以使用Elasticsearch进行高效的日志搜索")
            print("   - 支持全文搜索、时间范围查询、字段过滤等功能")
        print()
    
    def demonstrate_concurrent_logging(self):
        """演示并发日志记录"""
        print("🔄 演示并发日志记录...")
        
        def worker_thread(worker_id: int, log_count: int):
            """工作线程函数"""
            for i in range(log_count):
                # 随机选择日志级别和组件
                level = random.choice([LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR])
                component = f"worker_{worker_id}"
                
                self.logger.log(
                    level=level,
                    category=LogCategory.SYSTEM,
                    message=f"Worker {worker_id} 处理任务 {i}",
                    component=component,
                    worker_id=worker_id,
                    task_id=i,
                    duration_ms=random.uniform(10, 100)
                )
                
                # 模拟处理时间
                time.sleep(0.01)
        
        # 启动多个工作线程
        threads = []
        worker_count = 5
        logs_per_worker = 10
        
        start_time = time.time()
        
        for worker_id in range(worker_count):
            thread = threading.Thread(
                target=worker_thread,
                args=(worker_id, logs_per_worker)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        print(f"✅ 并发日志记录完成:")
        print(f"   - 工作线程数: {worker_count}")
        print(f"   - 每线程日志数: {logs_per_worker}")
        print(f"   - 总日志数: {worker_count * logs_per_worker}")
        print(f"   - 处理时间: {end_time - start_time:.2f}秒")
        
        # 验证日志完整性
        stats = self.logger.get_log_statistics()
        total_logs = sum(stats['aggregated_stats']['log_counts'].values())
        print(f"   - 统计中的日志总数: {total_logs}")
        print()
    
    def demonstrate_anomaly_detection(self):
        """演示异常检测功能"""
        print("🚨 演示异常检测功能...")
        
        # 建立正常的基线数据
        print("   建立性能基线...")
        for i in range(20):
            self.logger.anomaly_detector.update_baseline(
                "api_service", "response_time", 100.0 + random.uniform(-10, 10)
            )
        
        # 测试正常值
        normal_value = 105.0
        is_anomaly = self.logger.anomaly_detector.detect_anomaly(
            "api_service", "response_time", normal_value
        )
        print(f"   正常值 {normal_value}ms 是否异常: {is_anomaly}")
        
        # 测试异常值
        anomaly_value = 500.0
        is_anomaly = self.logger.anomaly_detector.detect_anomaly(
            "api_service", "response_time", anomaly_value
        )
        print(f"   异常值 {anomaly_value}ms 是否异常: {is_anomaly}")
        
        if is_anomaly:
            # 记录异常日志
            log_error(
                f"API响应时间异常: {anomaly_value}ms",
                "api_service",
                response_time=anomaly_value,
                threshold="基线+3σ"
            )
        
        print("✅ 异常检测演示完成")
        print()
    
    def demonstrate_log_lifecycle(self):
        """演示日志生命周期管理"""
        print("🔄 演示日志生命周期管理...")
        
        # 记录系统启动日志
        log_info("系统启动", "system", phase="startup")
        
        # 记录业务操作日志
        log_info("开始数据处理", "processor", batch_id="batch_001")
        log_performance("数据处理完成", "processor", 1250.5, batch_id="batch_001", records=1000)
        
        # 记录异常和恢复
        log_error("数据源连接失败", "data_source", source="eastmoney", error="timeout")
        log_info("切换到备用数据源", "data_source", source="backup", action="failover")
        log_info("数据源连接恢复", "data_source", source="eastmoney", action="recovery")
        
        # 记录系统关闭日志
        log_info("系统准备关闭", "system", phase="shutdown")
        
        print("✅ 日志生命周期演示完成")
        print(f"   - 当前缓冲区日志数: {len(self.logger.log_buffer)}")
        print()
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("📋 生成ELK日志系统演示总结报告...")
        
        # 获取最终统计
        stats = self.logger.get_log_statistics()
        dashboard_data = self.logger.create_dashboard_data()
        
        print("\n" + "="*60)
        print("ELK日志分析系统演示总结报告")
        print("="*60)
        
        print(f"\n📊 系统概览:")
        print(f"   系统健康状态: {dashboard_data['health_status']}")
        print(f"   Elasticsearch状态: {'可用' if self.logger.es_available else '不可用'}")
        print(f"   日志索引前缀: {self.logger.index_prefix}")
        
        print(f"\n📈 日志统计:")
        log_counts = stats['aggregated_stats']['log_counts']
        total_logs = sum(log_counts.values())
        print(f"   总日志数: {total_logs}")
        for level, count in log_counts.items():
            percentage = (count / total_logs * 100) if total_logs > 0 else 0
            print(f"   {level}: {count} ({percentage:.1f}%)")
        
        print(f"\n🚨 异常检测:")
        print(f"   检测到的异常总数: {stats['total_anomalies']}")
        print(f"   错误模式数量: {len(stats['aggregated_stats']['error_patterns'])}")
        
        if stats['recent_anomalies']:
            print(f"   最近异常:")
            for anomaly in stats['recent_anomalies'][:3]:
                print(f"     - {anomaly['pattern_name']}: {anomaly['severity']}")
        
        print(f"\n⚡ 性能监控:")
        perf_summary = stats['aggregated_stats']['performance_summary']
        print(f"   监控组件数: {len(perf_summary)}")
        
        if perf_summary:
            print(f"   性能概览:")
            for component, metrics in perf_summary.items():
                print(f"     {component}: 平均{metrics['avg_duration']:.1f}ms "
                      f"(最大{metrics['max_duration']:.1f}ms, {metrics['count']}次)")
        
        print(f"\n🔧 系统特性:")
        print(f"   ✅ 结构化日志记录")
        print(f"   ✅ 实时模式匹配和异常检测")
        print(f"   ✅ 日志聚合和统计分析")
        print(f"   ✅ 性能监控集成")
        print(f"   ✅ 并发安全的日志处理")
        print(f"   ✅ 仪表板数据生成")
        print(f"   {'✅' if self.logger.es_available else '⚠️ '} Elasticsearch集成")
        
        print(f"\n💡 生产环境建议:")
        print(f"   - 配置Elasticsearch集群以支持大规模日志存储")
        print(f"   - 设置Kibana仪表板进行可视化分析")
        print(f"   - 配置Logstash进行日志预处理和转换")
        print(f"   - 实施日志轮转和归档策略")
        print(f"   - 设置实时告警和通知机制")
        
        print("\n" + "="*60)
        print("演示完成！ELK日志分析系统已准备就绪。")
        print("="*60)
    
    def cleanup(self):
        """清理资源"""
        if self.logger:
            print("\n🧹 清理系统资源...")
            self.logger.shutdown()
            print("✅ 系统资源清理完成")
    
    def run_complete_demo(self):
        """运行完整演示"""
        try:
            print("🎯 开始ELK日志分析系统完整演示")
            print("="*60)
            
            # 初始化系统
            self.initialize_system()
            
            # 演示各个功能
            self.demonstrate_basic_logging()
            self.demonstrate_pattern_matching()
            self.demonstrate_log_aggregation()
            self.demonstrate_performance_monitoring()
            self.demonstrate_dashboard_data()
            self.demonstrate_log_search()
            self.demonstrate_concurrent_logging()
            self.demonstrate_anomaly_detection()
            self.demonstrate_log_lifecycle()
            
            # 生成总结报告
            self.generate_summary_report()
            
        except Exception as e:
            print(f"❌ 演示过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理资源
            self.cleanup()


def main():
    """主函数"""
    demo = ELKLoggingDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()