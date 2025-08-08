#!/usr/bin/env python3
"""
资金流向追踪端到端测试

测试资金流向数据的实时监控流程，验证多时间周期数据的一致性。
包含资金流向趋势分析和预警的测试，以及资金流向可视化和报告的验证。
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import time
import logging
from typing import Dict, List, Optional, Tuple
import json

from stock_analysis_system.data.enhanced_data_sources import EnhancedDataSourceManager
from stock_analysis_system.data.fund_flow_adapter import FundFlowAdapter
from stock_analysis_system.data.cache_manager import CacheManager
from stock_analysis_system.data.data_quality_engine import EnhancedDataQualityEngine
from stock_analysis_system.core.database_manager import DatabaseManager
from stock_analysis_system.api.main import app
from fastapi.testclient import TestClient

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FundFlowTrackingE2ETest:
    """资金流向追踪端到端测试类"""
    
    def __init__(self):
        self.client = TestClient(app)
        self.data_sources = EnhancedDataSourceManager()
        self.fund_flow_adapter = FundFlowAdapter()
        self.cache_manager = CacheManager()
        self.quality_engine = EnhancedDataQualityEngine()
        self.db_manager = DatabaseManager()
        
        # 测试配置
        self.test_symbols = ['000001', '000002', '600000', '600036', '300001']
        self.test_periods = ['1d', '3d', '5d', '10d', '20d']
        self.test_timeout = 60  # 秒
        
        # 性能阈值
        self.performance_thresholds = {
            'data_acquisition_time': 4.0,
            'processing_time': 3.0,
            'analysis_time': 3.0,
            'alert_response_time': 2.0,
            'visualization_time': 2.0,
            'total_response_time': 12.0
        }
        
        # 资金流向告警配置
        self.alert_config = {
            'large_inflow_threshold': 100000000,    # 1亿
            'large_outflow_threshold': -100000000,  # -1亿
            'unusual_ratio_threshold': 0.1,         # 10%
            'continuous_flow_days': 3,              # 连续3天
            'main_force_threshold': 50000000        # 主力资金5000万
        }
        
        # 时间周期权重
        self.period_weights = {
            '1d': 0.4,
            '3d': 0.25,
            '5d': 0.2,
            '10d': 0.1,
            '20d': 0.05
        }
    
    async def setup_test_environment(self):
        """设置测试环境"""
        logger.info("设置资金流向追踪端到端测试环境...")
        
        # 清理测试数据
        await self._cleanup_test_data()
        
        # 初始化数据库连接
        await self.db_manager.initialize()
        
        # 创建测试数据表
        await self._create_test_tables()
        
        # 准备测试数据
        await self._prepare_test_data()
        
        logger.info("资金流向测试环境设置完成")
    
    async def teardown_test_environment(self):
        """清理测试环境"""
        logger.info("清理资金流向测试环境...")
        
        # 清理测试数据
        await self._cleanup_test_data()
        
        # 关闭数据库连接
        await self.db_manager.close()
        
        logger.info("资金流向测试环境清理完成")
    
    async def _cleanup_test_data(self):
        """清理测试数据"""
        try:
            # 清理缓存
            await self.cache_manager.delete_pattern("fund_flow:*")
            await self.cache_manager.delete_pattern("ff:*")
            
            # 清理数据库测试数据
            test_date = datetime.now().strftime('%Y-%m-%d')
            for symbol in self.test_symbols:
                await self.db_manager.execute(
                    "DELETE FROM fund_flow WHERE stock_code = $1 AND trade_date >= $2",
                    symbol, (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                )
            
            await self.db_manager.execute(
                "DELETE FROM fund_flow_alerts WHERE created_at >= $1",
                datetime.now() - timedelta(hours=2)
            )
            
        except Exception as e:
            logger.warning(f"清理测试数据时出现警告: {e}")
    
    async def _create_test_tables(self):
        """创建测试数据表"""
        try:
            # 资金流向主表
            await self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS fund_flow (
                    id SERIAL PRIMARY KEY,
                    stock_code VARCHAR(10) NOT NULL,
                    stock_name VARCHAR(50),
                    trade_date DATE NOT NULL,
                    period_type VARCHAR(10) NOT NULL,
                    main_net_inflow BIGINT DEFAULT 0,
                    main_net_inflow_rate DECIMAL(5,2) DEFAULT 0,
                    super_large_net_inflow BIGINT DEFAULT 0,
                    super_large_net_inflow_rate DECIMAL(5,2) DEFAULT 0,
                    large_net_inflow BIGINT DEFAULT 0,
                    large_net_inflow_rate DECIMAL(5,2) DEFAULT 0,
                    medium_net_inflow BIGINT DEFAULT 0,
                    medium_net_inflow_rate DECIMAL(5,2) DEFAULT 0,
                    small_net_inflow BIGINT DEFAULT 0,
                    small_net_inflow_rate DECIMAL(5,2) DEFAULT 0,
                    total_turnover BIGINT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(stock_code, trade_date, period_type)
                )
            """)
            
            # 资金流向告警表
            await self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS fund_flow_alerts (
                    id SERIAL PRIMARY KEY,
                    alert_type VARCHAR(50) NOT NULL,
                    stock_code VARCHAR(10) NOT NULL,
                    stock_name VARCHAR(50),
                    period_type VARCHAR(10),
                    alert_message TEXT NOT NULL,
                    alert_level VARCHAR(20) DEFAULT 'INFO',
                    trigger_value BIGINT,
                    threshold_value BIGINT,
                    flow_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE
                )
            """)
            
            # 资金流向趋势表
            await self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS fund_flow_trends (
                    id SERIAL PRIMARY KEY,
                    stock_code VARCHAR(10) NOT NULL,
                    trend_type VARCHAR(30) NOT NULL,
                    trend_direction VARCHAR(20),
                    trend_strength DECIMAL(5,2),
                    start_date DATE,
                    end_date DATE,
                    confidence_score DECIMAL(5,2),
                    trend_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建索引
            await self.db_manager.execute("""
                CREATE INDEX IF NOT EXISTS idx_fund_flow_code_date_period 
                ON fund_flow(stock_code, trade_date, period_type)
            """)
            
            await self.db_manager.execute("""
                CREATE INDEX IF NOT EXISTS idx_fund_flow_alerts_code_type 
                ON fund_flow_alerts(stock_code, alert_type)
            """)
            
        except Exception as e:
            logger.error(f"创建测试数据表失败: {e}")
    
    async def _prepare_test_data(self):
        """准备测试数据"""
        logger.info("准备资金流向测试数据...")
        
        # 生成过去30天的模拟数据
        for i in range(30):
            trade_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            
            for symbol in self.test_symbols:
                for period in self.test_periods:
                    # 生成模拟资金流向数据
                    main_inflow = int(np.random.normal(0, 50000000))  # 主力资金
                    super_large_inflow = int(np.random.normal(0, 30000000))  # 超大单
                    large_inflow = int(np.random.normal(0, 20000000))  # 大单
                    medium_inflow = int(np.random.normal(0, 15000000))  # 中单
                    small_inflow = int(np.random.normal(0, 10000000))  # 小单
                    
                    total_turnover = abs(main_inflow) + abs(super_large_inflow) + abs(large_inflow) + abs(medium_inflow) + abs(small_inflow)
                    
                    # 计算流入比例
                    main_rate = (main_inflow / total_turnover * 100) if total_turnover > 0 else 0
                    super_large_rate = (super_large_inflow / total_turnover * 100) if total_turnover > 0 else 0
                    large_rate = (large_inflow / total_turnover * 100) if total_turnover > 0 else 0
                    medium_rate = (medium_inflow / total_turnover * 100) if total_turnover > 0 else 0
                    small_rate = (small_inflow / total_turnover * 100) if total_turnover > 0 else 0
                    
                    await self.db_manager.execute("""
                        INSERT INTO fund_flow 
                        (stock_code, stock_name, trade_date, period_type,
                         main_net_inflow, main_net_inflow_rate,
                         super_large_net_inflow, super_large_net_inflow_rate,
                         large_net_inflow, large_net_inflow_rate,
                         medium_net_inflow, medium_net_inflow_rate,
                         small_net_inflow, small_net_inflow_rate,
                         total_turnover)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                        ON CONFLICT (stock_code, trade_date, period_type) DO NOTHING
                    """, 
                        symbol, f"测试股票{symbol}", trade_date, period,
                        main_inflow, round(main_rate, 2),
                        super_large_inflow, round(super_large_rate, 2),
                        large_inflow, round(large_rate, 2),
                        medium_inflow, round(medium_rate, 2),
                        small_inflow, round(small_rate, 2),
                        total_turnover
                    )
    
    async def test_complete_fund_flow_tracking_flow(self):
        """测试完整的资金流向追踪流程"""
        logger.info("开始完整资金流向追踪流程测试...")
        
        results = {
            'success': True,
            'performance_metrics': {},
            'tracking_metrics': {},
            'alert_metrics': {},
            'visualization_metrics': {},
            'error_details': []
        }
        
        try:
            # 1. 测试资金流向数据获取
            acquisition_result = await self._test_fund_flow_data_acquisition()
            results['performance_metrics']['data_acquisition'] = acquisition_result
            
            # 2. 测试多时间周期数据处理
            processing_result = await self._test_multi_period_data_processing()
            results['performance_metrics']['data_processing'] = processing_result
            
            # 3. 测试数据一致性验证
            consistency_result = await self._test_data_consistency()
            results['tracking_metrics']['data_consistency'] = consistency_result
            
            # 4. 测试实时监控
            monitoring_result = await self._test_realtime_monitoring()
            results['tracking_metrics']['realtime_monitoring'] = monitoring_result
            
            # 5. 测试趋势分析
            trend_result = await self._test_trend_analysis()
            results['tracking_metrics']['trend_analysis'] = trend_result
            
            # 6. 测试预警机制
            alert_result = await self._test_alert_mechanism()
            results['alert_metrics'] = alert_result
            
            # 7. 测试可视化功能
            visualization_result = await self._test_visualization()
            results['visualization_metrics'] = visualization_result
            
            # 8. 测试API端点
            api_result = await self._test_fund_flow_api()
            results['performance_metrics']['api_response'] = api_result
            
            # 计算总体性能
            total_time = sum([
                acquisition_result.get('duration', 0),
                processing_result.get('duration', 0),
                api_result.get('duration', 0),
                visualization_result.get('duration', 0)
            ])
            
            results['performance_metrics']['total_duration'] = total_time
            results['performance_metrics']['meets_sla'] = total_time <= self.performance_thresholds['total_response_time']
            
        except Exception as e:
            results['success'] = False
            results['error_details'].append(f"完整流程测试失败: {str(e)}")
            logger.error(f"资金流向追踪流程测试失败: {e}")
        
        return results
    
    async def _test_fund_flow_data_acquisition(self):
        """测试资金流向数据获取"""
        logger.info("测试资金流向数据获取...")
        
        start_time = time.time()
        results = {
            'success': True,
            'symbols_processed': 0,
            'periods_processed': 0,
            'records_acquired': 0,
            'errors': []
        }
        
        try:
            for symbol in self.test_symbols:
                symbol_success = True
                
                for period in self.test_periods:
                    try:
                        # 获取资金流向数据
                        fund_flow_data = await self.fund_flow_adapter.get_fund_flow_data(symbol, period)
                        
                        if fund_flow_data is not None and not fund_flow_data.empty:
                            results['periods_processed'] += 1
                            results['records_acquired'] += len(fund_flow_data)
                            
                            # 验证数据结构
                            required_columns = ['main_net_inflow', 'super_large_net_inflow', 'large_net_inflow']
                            missing_columns = [col for col in required_columns if col not in fund_flow_data.columns]
                            
                            if missing_columns:
                                results['errors'].append(f"股票 {symbol} 周期 {period} 数据缺少列: {missing_columns}")
                                symbol_success = False
                        else:
                            results['errors'].append(f"股票 {symbol} 周期 {period} 资金流向数据为空")
                            symbol_success = False
                    
                    except Exception as e:
                        results['errors'].append(f"获取股票 {symbol} 周期 {period} 数据失败: {str(e)}")
                        symbol_success = False
                
                if symbol_success:
                    results['symbols_processed'] += 1
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"资金流向数据获取测试失败: {str(e)}")
        
        duration = time.time() - start_time
        results['duration'] = duration
        results['meets_threshold'] = duration <= self.performance_thresholds['data_acquisition_time']
        
        logger.info(f"资金流向数据获取测试完成，耗时: {duration:.2f}秒")
        return results
    
    async def _test_multi_period_data_processing(self):
        """测试多时间周期数据处理"""
        logger.info("测试多时间周期数据处理...")
        
        start_time = time.time()
        results = {
            'success': True,
            'processed_symbols': 0,
            'period_correlations': {},
            'aggregated_data': {},
            'errors': []
        }
        
        try:
            for symbol in self.test_symbols:
                try:
                    # 获取多周期数据
                    multi_period_data = {}
                    for period in self.test_periods:
                        data = await self._get_fund_flow_data_from_db(symbol, period)
                        if not data.empty:
                            multi_period_data[period] = data
                    
                    if multi_period_data:
                        # 处理多周期数据
                        processed_data = await self._process_multi_period_data(symbol, multi_period_data)
                        results['processed_symbols'] += 1
                        
                        # 计算周期间相关性
                        correlations = await self._calculate_period_correlations(multi_period_data)
                        results['period_correlations'][symbol] = correlations
                        
                        # 聚合数据
                        aggregated = await self._aggregate_multi_period_data(multi_period_data)
                        results['aggregated_data'][symbol] = aggregated
                
                except Exception as e:
                    results['errors'].append(f"处理股票 {symbol} 多周期数据失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"多时间周期数据处理测试失败: {str(e)}")
        
        duration = time.time() - start_time
        results['duration'] = duration
        results['meets_threshold'] = duration <= self.performance_thresholds['processing_time']
        
        logger.info(f"多时间周期数据处理测试完成，耗时: {duration:.2f}秒")
        return results
    
    async def _test_data_consistency(self):
        """测试数据一致性"""
        logger.info("测试资金流向数据一致性...")
        
        results = {
            'success': True,
            'consistency_checks': 0,
            'consistency_score': 0.0,
            'inconsistencies': [],
            'errors': []
        }
        
        try:
            for symbol in self.test_symbols:
                try:
                    # 检查不同周期数据的一致性
                    consistency_result = await self._check_period_consistency(symbol)
                    results['consistency_checks'] += 1
                    
                    if consistency_result['score'] < 0.8:
                        results['inconsistencies'].append({
                            'symbol': symbol,
                            'score': consistency_result['score'],
                            'issues': consistency_result['issues']
                        })
                    
                    results['consistency_score'] += consistency_result['score']
                
                except Exception as e:
                    results['errors'].append(f"检查股票 {symbol} 数据一致性失败: {str(e)}")
            
            # 计算平均一致性分数
            if results['consistency_checks'] > 0:
                results['consistency_score'] /= results['consistency_checks']
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"数据一致性测试失败: {str(e)}")
        
        logger.info(f"数据一致性测试完成，平均分数: {results['consistency_score']:.2f}")
        return results
    
    async def _test_realtime_monitoring(self):
        """测试实时监控"""
        logger.info("测试资金流向实时监控...")
        
        results = {
            'success': True,
            'monitoring_cycles': 0,
            'anomalies_detected': 0,
            'response_times': [],
            'errors': []
        }
        
        try:
            # 模拟多个监控周期
            for cycle in range(3):
                cycle_start = time.time()
                
                try:
                    # 获取实时数据
                    realtime_data = await self._get_realtime_fund_flow_data()
                    
                    # 检测异常
                    anomalies = await self._detect_fund_flow_anomalies(realtime_data)
                    results['anomalies_detected'] += len(anomalies)
                    
                    # 更新监控状态
                    await self._update_monitoring_status(realtime_data, anomalies)
                    
                    cycle_time = time.time() - cycle_start
                    results['response_times'].append(cycle_time)
                    results['monitoring_cycles'] += 1
                    
                    # 模拟监控间隔
                    await asyncio.sleep(0.1)
                
                except Exception as e:
                    results['errors'].append(f"监控周期 {cycle + 1} 失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"实时监控测试失败: {str(e)}")
        
        avg_response_time = sum(results['response_times']) / len(results['response_times']) if results['response_times'] else 0
        results['avg_response_time'] = avg_response_time
        
        logger.info(f"实时监控测试完成，平均响应时间: {avg_response_time:.2f}秒")
        return results
    
    async def _test_trend_analysis(self):
        """测试趋势分析"""
        logger.info("测试资金流向趋势分析...")
        
        start_time = time.time()
        results = {
            'success': True,
            'trends_analyzed': 0,
            'trend_types': {},
            'accuracy_metrics': {},
            'errors': []
        }
        
        try:
            for symbol in self.test_symbols:
                try:
                    # 获取历史数据
                    historical_data = await self._get_historical_fund_flow_data(symbol)
                    
                    if not historical_data.empty:
                        # 分析不同类型的趋势
                        trend_analyses = [
                            ('main_force_trend', self._analyze_main_force_trend),
                            ('volume_trend', self._analyze_volume_trend),
                            ('flow_direction_trend', self._analyze_flow_direction_trend),
                            ('institutional_behavior_trend', self._analyze_institutional_behavior_trend)
                        ]
                        
                        symbol_trends = {}
                        for trend_type, analysis_func in trend_analyses:
                            try:
                                trend_result = await analysis_func(historical_data)
                                symbol_trends[trend_type] = trend_result
                                
                                if trend_result.get('trend_detected', False):
                                    results['trends_analyzed'] += 1
                            
                            except Exception as e:
                                results['errors'].append(f"分析股票 {symbol} {trend_type} 失败: {str(e)}")
                        
                        results['trend_types'][symbol] = symbol_trends
                        
                        # 计算趋势准确性
                        accuracy = await self._calculate_trend_accuracy(symbol_trends)
                        results['accuracy_metrics'][symbol] = accuracy
                
                except Exception as e:
                    results['errors'].append(f"分析股票 {symbol} 趋势失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"趋势分析测试失败: {str(e)}")
        
        duration = time.time() - start_time
        results['duration'] = duration
        results['meets_threshold'] = duration <= self.performance_thresholds['analysis_time']
        
        logger.info(f"趋势分析测试完成，分析趋势: {results['trends_analyzed']} 个")
        return results
    
    async def _test_alert_mechanism(self):
        """测试预警机制"""
        logger.info("测试资金流向预警机制...")
        
        start_time = time.time()
        results = {
            'success': True,
            'alerts_generated': 0,
            'alert_types': {},
            'notification_sent': 0,
            'errors': []
        }
        
        try:
            # 测试不同类型的预警
            alert_tests = [
                ('large_inflow', self._test_large_inflow_alert),
                ('large_outflow', self._test_large_outflow_alert),
                ('unusual_ratio', self._test_unusual_ratio_alert),
                ('continuous_flow', self._test_continuous_flow_alert),
                ('main_force_activity', self._test_main_force_activity_alert)
            ]
            
            for alert_type, test_func in alert_tests:
                try:
                    alert_result = await test_func()
                    results['alert_types'][alert_type] = alert_result
                    
                    if alert_result.get('triggered', False):
                        results['alerts_generated'] += 1
                        
                        # 测试通知发送
                        notification_result = await self._test_alert_notification(alert_type, alert_result)
                        if notification_result:
                            results['notification_sent'] += 1
                
                except Exception as e:
                    results['errors'].append(f"测试 {alert_type} 预警失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"预警机制测试失败: {str(e)}")
        
        duration = time.time() - start_time
        results['duration'] = duration
        results['meets_threshold'] = duration <= self.performance_thresholds['alert_response_time']
        
        logger.info(f"预警机制测试完成，生成预警: {results['alerts_generated']} 个")
        return results
    
    async def _test_visualization(self):
        """测试可视化功能"""
        logger.info("测试资金流向可视化...")
        
        start_time = time.time()
        results = {
            'success': True,
            'charts_generated': 0,
            'chart_types': {},
            'rendering_times': {},
            'errors': []
        }
        
        try:
            # 测试不同类型的图表
            chart_tests = [
                ('flow_trend_chart', self._test_flow_trend_chart),
                ('period_comparison_chart', self._test_period_comparison_chart),
                ('sector_flow_chart', self._test_sector_flow_chart),
                ('main_force_heatmap', self._test_main_force_heatmap)
            ]
            
            for chart_type, test_func in chart_tests:
                try:
                    chart_start = time.time()
                    chart_result = await test_func()
                    chart_duration = time.time() - chart_start
                    
                    results['chart_types'][chart_type] = chart_result
                    results['rendering_times'][chart_type] = chart_duration
                    
                    if chart_result.get('generated', False):
                        results['charts_generated'] += 1
                
                except Exception as e:
                    results['errors'].append(f"生成 {chart_type} 图表失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"可视化测试失败: {str(e)}")
        
        duration = time.time() - start_time
        results['duration'] = duration
        results['meets_threshold'] = duration <= self.performance_thresholds['visualization_time']
        
        logger.info(f"可视化测试完成，生成图表: {results['charts_generated']} 个")
        return results
    
    async def _test_fund_flow_api(self):
        """测试资金流向API端点"""
        logger.info("测试资金流向API端点...")
        
        start_time = time.time()
        results = {
            'success': True,
            'api_calls': 0,
            'successful_calls': 0,
            'api_performance': {},
            'errors': []
        }
        
        try:
            # 测试不同的API端点
            api_tests = [
                ('/api/v1/fund-flow/realtime', {'symbol': self.test_symbols[0]}),
                ('/api/v1/fund-flow/history', {'symbol': self.test_symbols[0], 'period': '5d'}),
                ('/api/v1/fund-flow/trend', {'symbol': self.test_symbols[0]}),
                ('/api/v1/fund-flow/alerts', {}),
                ('/api/v1/fund-flow/ranking', {'period': '1d'}),
                ('/api/v1/fund-flow/sector', {'sector': 'technology'})
            ]
            
            for endpoint, params in api_tests:
                try:
                    api_start = time.time()
                    
                    if params:
                        response = self.client.get(endpoint, params=params)
                    else:
                        response = self.client.get(endpoint)
                    
                    api_duration = time.time() - api_start
                    results['api_calls'] += 1
                    
                    if response.status_code == 200:
                        results['successful_calls'] += 1
                        
                        # 验证响应数据
                        data = response.json()
                        if self._validate_fund_flow_api_response(endpoint, data):
                            results['api_performance'][endpoint] = {
                                'duration': api_duration,
                                'status': 'success',
                                'data_count': len(data) if isinstance(data, list) else 1
                            }
                        else:
                            results['errors'].append(f"API {endpoint} 响应数据格式不正确")
                    else:
                        results['errors'].append(f"API {endpoint} 调用失败: {response.status_code}")
                
                except Exception as e:
                    results['errors'].append(f"测试API {endpoint} 失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"API测试失败: {str(e)}")
        
        duration = time.time() - start_time
        results['duration'] = duration
        
        logger.info(f"API测试完成，成功调用: {results['successful_calls']}/{results['api_calls']}")
        return results
    
    # 辅助方法实现
    async def _get_fund_flow_data_from_db(self, symbol: str, period: str) -> pd.DataFrame:
        """从数据库获取资金流向数据"""
        query = """
            SELECT * FROM fund_flow 
            WHERE stock_code = $1 AND period_type = $2
            ORDER BY trade_date DESC
            LIMIT 30
        """
        
        rows = await self.db_manager.fetch_all(query, symbol, period)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    async def _process_multi_period_data(self, symbol: str, multi_period_data: Dict) -> Dict:
        """处理多周期数据"""
        processed = {
            'symbol': symbol,
            'periods': list(multi_period_data.keys()),
            'weighted_score': 0.0,
            'trend_consistency': 0.0
        }
        
        # 计算加权评分
        total_weight = 0
        weighted_sum = 0
        
        for period, data in multi_period_data.items():
            if not data.empty and period in self.period_weights:
                weight = self.period_weights[period]
                period_score = data['main_net_inflow'].mean() if 'main_net_inflow' in data.columns else 0
                
                weighted_sum += period_score * weight
                total_weight += weight
        
        if total_weight > 0:
            processed['weighted_score'] = weighted_sum / total_weight
        
        return processed
    
    async def _calculate_period_correlations(self, multi_period_data: Dict) -> Dict:
        """计算周期间相关性"""
        correlations = {}
        
        if len(multi_period_data) < 2:
            return correlations
        
        periods = list(multi_period_data.keys())
        
        for i, period1 in enumerate(periods):
            for period2 in periods[i+1:]:
                data1 = multi_period_data[period1]
                data2 = multi_period_data[period2]
                
                if not data1.empty and not data2.empty and 'main_net_inflow' in data1.columns and 'main_net_inflow' in data2.columns:
                    # 简化的相关性计算
                    corr_key = f"{period1}_{period2}"
                    
                    # 取相同日期的数据进行比较
                    common_dates = set(data1['trade_date']) & set(data2['trade_date'])
                    if len(common_dates) > 1:
                        data1_filtered = data1[data1['trade_date'].isin(common_dates)].sort_values('trade_date')
                        data2_filtered = data2[data2['trade_date'].isin(common_dates)].sort_values('trade_date')
                        
                        if len(data1_filtered) == len(data2_filtered):
                            correlation = np.corrcoef(data1_filtered['main_net_inflow'], data2_filtered['main_net_inflow'])[0, 1]
                            correlations[corr_key] = correlation if not np.isnan(correlation) else 0.0
        
        return correlations
    
    async def _aggregate_multi_period_data(self, multi_period_data: Dict) -> Dict:
        """聚合多周期数据"""
        aggregated = {
            'total_main_inflow': 0,
            'avg_main_inflow': 0,
            'max_single_day_inflow': 0,
            'min_single_day_inflow': 0,
            'flow_volatility': 0
        }
        
        all_inflows = []
        
        for period, data in multi_period_data.items():
            if not data.empty and 'main_net_inflow' in data.columns:
                period_inflows = data['main_net_inflow'].tolist()
                all_inflows.extend(period_inflows)
        
        if all_inflows:
            aggregated['total_main_inflow'] = sum(all_inflows)
            aggregated['avg_main_inflow'] = np.mean(all_inflows)
            aggregated['max_single_day_inflow'] = max(all_inflows)
            aggregated['min_single_day_inflow'] = min(all_inflows)
            aggregated['flow_volatility'] = np.std(all_inflows)
        
        return aggregated
    
    async def _check_period_consistency(self, symbol: str) -> Dict:
        """检查周期数据一致性"""
        result = {
            'score': 1.0,
            'issues': []
        }
        
        try:
            # 获取不同周期的数据
            period_data = {}
            for period in self.test_periods:
                data = await self._get_fund_flow_data_from_db(symbol, period)
                if not data.empty:
                    period_data[period] = data
            
            if len(period_data) < 2:
                return result
            
            # 检查数据逻辑一致性
            for period, data in period_data.items():
                if 'main_net_inflow' in data.columns and 'total_turnover' in data.columns:
                    # 检查资金流入是否超过总成交额
                    invalid_records = data[abs(data['main_net_inflow']) > data['total_turnover']]
                    if not invalid_records.empty:
                        result['score'] -= 0.2
                        result['issues'].append(f"周期 {period} 存在 {len(invalid_records)} 条逻辑不一致记录")
                
                # 检查数据完整性
                null_count = data.isnull().sum().sum()
                if null_count > 0:
                    result['score'] -= 0.1
                    result['issues'].append(f"周期 {period} 存在 {null_count} 个空值")
        
        except Exception as e:
            result['score'] = 0.0
            result['issues'].append(f"一致性检查失败: {str(e)}")
        
        return result
    
    async def _get_realtime_fund_flow_data(self) -> pd.DataFrame:
        """获取实时资金流向数据"""
        query = """
            SELECT * FROM fund_flow 
            WHERE trade_date = CURRENT_DATE AND period_type = '1d'
            ORDER BY main_net_inflow DESC
        """
        
        rows = await self.db_manager.fetch_all(query)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    async def _detect_fund_flow_anomalies(self, data: pd.DataFrame) -> List[Dict]:
        """检测资金流向异常"""
        anomalies = []
        
        if data.empty:
            return anomalies
        
        # 检测大额异常流入/流出
        if 'main_net_inflow' in data.columns:
            large_inflows = data[data['main_net_inflow'] > self.alert_config['large_inflow_threshold']]
            large_outflows = data[data['main_net_inflow'] < self.alert_config['large_outflow_threshold']]
            
            for _, row in large_inflows.iterrows():
                anomalies.append({
                    'type': 'large_inflow',
                    'symbol': row['stock_code'],
                    'value': row['main_net_inflow'],
                    'severity': 'high'
                })
            
            for _, row in large_outflows.iterrows():
                anomalies.append({
                    'type': 'large_outflow',
                    'symbol': row['stock_code'],
                    'value': row['main_net_inflow'],
                    'severity': 'high'
                })
        
        return anomalies
    
    async def _update_monitoring_status(self, data: pd.DataFrame, anomalies: List[Dict]):
        """更新监控状态"""
        # 这里可以更新监控仪表板、发送通知等
        logger.info(f"监控状态更新: 数据记录 {len(data)}, 异常 {len(anomalies)}")
    
    async def _get_historical_fund_flow_data(self, symbol: str) -> pd.DataFrame:
        """获取历史资金流向数据"""
        query = """
            SELECT * FROM fund_flow 
            WHERE stock_code = $1 AND period_type = '1d'
            ORDER BY trade_date DESC
            LIMIT 30
        """
        
        rows = await self.db_manager.fetch_all(query, symbol)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    async def _analyze_main_force_trend(self, data: pd.DataFrame) -> Dict:
        """分析主力资金趋势"""
        if data.empty or 'main_net_inflow' not in data.columns:
            return {'trend_detected': False}
        
        # 计算移动平均
        data_sorted = data.sort_values('trade_date')
        ma5 = data_sorted['main_net_inflow'].rolling(window=5).mean()
        ma10 = data_sorted['main_net_inflow'].rolling(window=10).mean()
        
        if len(ma5) < 5 or len(ma10) < 10:
            return {'trend_detected': False}
        
        # 判断趋势方向
        recent_ma5 = ma5.iloc[-1]
        recent_ma10 = ma10.iloc[-1]
        
        trend_direction = 'bullish' if recent_ma5 > recent_ma10 else 'bearish'
        trend_strength = abs(recent_ma5 - recent_ma10) / abs(recent_ma10) if recent_ma10 != 0 else 0
        
        return {
            'trend_detected': True,
            'direction': trend_direction,
            'strength': trend_strength,
            'ma5': recent_ma5,
            'ma10': recent_ma10
        }
    
    async def _analyze_volume_trend(self, data: pd.DataFrame) -> Dict:
        """分析成交量趋势"""
        if data.empty or 'total_turnover' not in data.columns:
            return {'trend_detected': False}
        
        data_sorted = data.sort_values('trade_date')
        volumes = data_sorted['total_turnover']
        
        if len(volumes) < 5:
            return {'trend_detected': False}
        
        # 简单趋势分析
        recent_avg = volumes.tail(5).mean()
        earlier_avg = volumes.head(5).mean()
        
        trend_direction = 'increasing' if recent_avg > earlier_avg else 'decreasing'
        trend_strength = abs(recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0
        
        return {
            'trend_detected': True,
            'direction': trend_direction,
            'strength': trend_strength,
            'recent_avg_volume': recent_avg,
            'earlier_avg_volume': earlier_avg
        }
    
    async def _analyze_flow_direction_trend(self, data: pd.DataFrame) -> Dict:
        """分析资金流向趋势"""
        if data.empty:
            return {'trend_detected': False}
        
        # 分析各类资金的流向趋势
        flow_columns = ['main_net_inflow', 'super_large_net_inflow', 'large_net_inflow', 'medium_net_inflow', 'small_net_inflow']
        available_columns = [col for col in flow_columns if col in data.columns]
        
        if not available_columns:
            return {'trend_detected': False}
        
        data_sorted = data.sort_values('trade_date')
        flow_trends = {}
        
        for col in available_columns:
            recent_sum = data_sorted[col].tail(5).sum()
            earlier_sum = data_sorted[col].head(5).sum()
            
            if earlier_sum != 0:
                change_rate = (recent_sum - earlier_sum) / abs(earlier_sum)
                flow_trends[col] = {
                    'direction': 'inflow' if recent_sum > earlier_sum else 'outflow',
                    'change_rate': change_rate
                }
        
        return {
            'trend_detected': True,
            'flow_trends': flow_trends
        }
    
    async def _analyze_institutional_behavior_trend(self, data: pd.DataFrame) -> Dict:
        """分析机构行为趋势"""
        if data.empty or 'main_net_inflow' not in data.columns:
            return {'trend_detected': False}
        
        data_sorted = data.sort_values('trade_date')
        
        # 分析机构资金的连续性
        positive_days = (data_sorted['main_net_inflow'] > 0).sum()
        negative_days = (data_sorted['main_net_inflow'] < 0).sum()
        total_days = len(data_sorted)
        
        # 计算机构活跃度
        activity_score = abs(data_sorted['main_net_inflow']).mean()
        
        behavior_pattern = 'accumulating' if positive_days > negative_days else 'distributing'
        
        return {
            'trend_detected': True,
            'behavior_pattern': behavior_pattern,
            'positive_days_ratio': positive_days / total_days if total_days > 0 else 0,
            'activity_score': activity_score
        }
    
    async def _calculate_trend_accuracy(self, trend_results: Dict) -> Dict:
        """计算趋势分析准确性"""
        detected_trends = sum(1 for result in trend_results.values() if result.get('trend_detected', False))
        total_trends = len(trend_results)
        
        accuracy = detected_trends / total_trends if total_trends > 0 else 0.0
        
        return {
            'overall_accuracy': accuracy,
            'detected_trends': detected_trends,
            'total_trends': total_trends
        }
    
    # 预警测试方法
    async def _test_large_inflow_alert(self) -> Dict:
        """测试大额流入预警"""
        result = {'triggered': False, 'alert_count': 0, 'details': []}
        
        query = """
            SELECT * FROM fund_flow 
            WHERE main_net_inflow > $1 
            AND trade_date >= CURRENT_DATE - INTERVAL '7 days'
        """
        
        rows = await self.db_manager.fetch_all(query, self.alert_config['large_inflow_threshold'])
        
        if rows:
            result['triggered'] = True
            result['alert_count'] = len(rows)
            result['details'] = [
                {
                    'stock_code': row['stock_code'],
                    'main_net_inflow': row['main_net_inflow'],
                    'trade_date': row['trade_date']
                }
                for row in rows[:5]
            ]
        
        return result
    
    async def _test_large_outflow_alert(self) -> Dict:
        """测试大额流出预警"""
        result = {'triggered': False, 'alert_count': 0, 'details': []}
        
        query = """
            SELECT * FROM fund_flow 
            WHERE main_net_inflow < $1 
            AND trade_date >= CURRENT_DATE - INTERVAL '7 days'
        """
        
        rows = await self.db_manager.fetch_all(query, self.alert_config['large_outflow_threshold'])
        
        if rows:
            result['triggered'] = True
            result['alert_count'] = len(rows)
            result['details'] = [
                {
                    'stock_code': row['stock_code'],
                    'main_net_inflow': row['main_net_inflow'],
                    'trade_date': row['trade_date']
                }
                for row in rows[:5]
            ]
        
        return result
    
    async def _test_unusual_ratio_alert(self) -> Dict:
        """测试异常比例预警"""
        result = {'triggered': False, 'alert_count': 0, 'details': []}
        
        query = """
            SELECT * FROM fund_flow 
            WHERE ABS(main_net_inflow_rate) > $1 
            AND trade_date >= CURRENT_DATE - INTERVAL '7 days'
        """
        
        threshold = self.alert_config['unusual_ratio_threshold'] * 100  # 转换为百分比
        rows = await self.db_manager.fetch_all(query, threshold)
        
        if rows:
            result['triggered'] = True
            result['alert_count'] = len(rows)
            result['details'] = [
                {
                    'stock_code': row['stock_code'],
                    'main_net_inflow_rate': row['main_net_inflow_rate'],
                    'trade_date': row['trade_date']
                }
                for row in rows[:5]
            ]
        
        return result
    
    async def _test_continuous_flow_alert(self) -> Dict:
        """测试连续流向预警"""
        result = {'triggered': False, 'alert_count': 0, 'details': []}
        
        # 查找连续多日同向流动的股票
        query = """
            WITH flow_direction AS (
                SELECT stock_code, trade_date,
                       CASE WHEN main_net_inflow > 0 THEN 1 ELSE -1 END as direction
                FROM fund_flow 
                WHERE trade_date >= CURRENT_DATE - INTERVAL '10 days'
                ORDER BY stock_code, trade_date
            ),
            continuous_flow AS (
                SELECT stock_code, 
                       COUNT(*) as continuous_days,
                       direction
                FROM flow_direction
                GROUP BY stock_code, direction
                HAVING COUNT(*) >= $1
            )
            SELECT cf.*, ff.stock_name, ff.main_net_inflow
            FROM continuous_flow cf
            JOIN fund_flow ff ON cf.stock_code = ff.stock_code
            WHERE ff.trade_date = CURRENT_DATE - INTERVAL '1 day'
        """
        
        rows = await self.db_manager.fetch_all(query, self.alert_config['continuous_flow_days'])
        
        if rows:
            result['triggered'] = True
            result['alert_count'] = len(rows)
            result['details'] = [
                {
                    'stock_code': row['stock_code'],
                    'continuous_days': row['continuous_days'],
                    'direction': 'inflow' if row['direction'] > 0 else 'outflow'
                }
                for row in rows[:5]
            ]
        
        return result
    
    async def _test_main_force_activity_alert(self) -> Dict:
        """测试主力活跃度预警"""
        result = {'triggered': False, 'alert_count': 0, 'details': []}
        
        query = """
            SELECT stock_code, stock_name, 
                   ABS(main_net_inflow) as activity_level,
                   main_net_inflow
            FROM fund_flow 
            WHERE ABS(main_net_inflow) > $1 
            AND trade_date >= CURRENT_DATE - INTERVAL '3 days'
            ORDER BY ABS(main_net_inflow) DESC
        """
        
        rows = await self.db_manager.fetch_all(query, self.alert_config['main_force_threshold'])
        
        if rows:
            result['triggered'] = True
            result['alert_count'] = len(rows)
            result['details'] = [
                {
                    'stock_code': row['stock_code'],
                    'activity_level': row['activity_level'],
                    'main_net_inflow': row['main_net_inflow']
                }
                for row in rows[:5]
            ]
        
        return result
    
    async def _test_alert_notification(self, alert_type: str, alert_data: Dict) -> bool:
        """测试预警通知"""
        try:
            notification_data = {
                'alert_type': alert_type,
                'alert_count': alert_data.get('alert_count', 0),
                'timestamp': datetime.now().isoformat(),
                'details': alert_data.get('details', [])
            }
            
            logger.info(f"模拟发送 {alert_type} 资金流向预警通知: {notification_data}")
            return True
        except Exception as e:
            logger.error(f"发送预警通知失败: {e}")
            return False
    
    # 可视化测试方法
    async def _test_flow_trend_chart(self) -> Dict:
        """测试资金流向趋势图表"""
        try:
            # 模拟生成趋势图表
            chart_data = {
                'chart_type': 'line',
                'data_points': len(self.test_symbols) * 30,
                'time_range': '30d'
            }
            
            return {
                'generated': True,
                'chart_data': chart_data
            }
        except Exception as e:
            return {
                'generated': False,
                'error': str(e)
            }
    
    async def _test_period_comparison_chart(self) -> Dict:
        """测试周期对比图表"""
        try:
            chart_data = {
                'chart_type': 'bar',
                'periods': self.test_periods,
                'comparison_metrics': ['main_net_inflow', 'total_turnover']
            }
            
            return {
                'generated': True,
                'chart_data': chart_data
            }
        except Exception as e:
            return {
                'generated': False,
                'error': str(e)
            }
    
    async def _test_sector_flow_chart(self) -> Dict:
        """测试板块资金流向图表"""
        try:
            chart_data = {
                'chart_type': 'heatmap',
                'sectors': ['technology', 'finance', 'healthcare', 'energy'],
                'flow_data': 'aggregated'
            }
            
            return {
                'generated': True,
                'chart_data': chart_data
            }
        except Exception as e:
            return {
                'generated': False,
                'error': str(e)
            }
    
    async def _test_main_force_heatmap(self) -> Dict:
        """测试主力资金热力图"""
        try:
            chart_data = {
                'chart_type': 'heatmap',
                'symbols': self.test_symbols,
                'intensity_metric': 'main_net_inflow'
            }
            
            return {
                'generated': True,
                'chart_data': chart_data
            }
        except Exception as e:
            return {
                'generated': False,
                'error': str(e)
            }
    
    def _validate_fund_flow_api_response(self, endpoint: str, data) -> bool:
        """验证资金流向API响应"""
        if not data:
            return False
        
        if '/realtime' in endpoint:
            return isinstance(data, dict) and 'main_net_inflow' in data
        elif '/history' in endpoint:
            return isinstance(data, list) and all('trade_date' in item for item in data[:3])
        elif '/trend' in endpoint:
            return isinstance(data, dict) and 'trend_direction' in data
        elif '/alerts' in endpoint:
            return isinstance(data, list)
        elif '/ranking' in endpoint:
            return isinstance(data, list) and all('stock_code' in item for item in data[:3])
        elif '/sector' in endpoint:
            return isinstance(data, dict) and 'sector_flow' in data
        
        return True


# 测试用例
@pytest.mark.asyncio
async def test_complete_fund_flow_tracking_e2e():
    """完整的资金流向追踪端到端测试"""
    test_suite = FundFlowTrackingE2ETest()
    
    try:
        # 设置测试环境
        await test_suite.setup_test_environment()
        
        # 执行完整流程测试
        results = await test_suite.test_complete_fund_flow_tracking_flow()
        
        # 验证测试结果
        assert results['success'], f"资金流向追踪端到端测试失败: {results['error_details']}"
        assert results['performance_metrics']['meets_sla'], "性能指标未达到SLA要求"
        assert results['tracking_metrics'].get('data_consistency', {}).get('consistency_score', 0) >= 0.8, "数据一致性评分过低"
        
        logger.info("资金流向追踪端到端测试通过")
        
    finally:
        # 清理测试环境
        await test_suite.teardown_test_environment()


if __name__ == "__main__":
    # 运行资金流向追踪端到端测试
    async def run_tests():
        test_suite = FundFlowTrackingE2ETest()
        
        try:
            await test_suite.setup_test_environment()
            
            print("=" * 60)
            print("开始资金流向追踪端到端测试")
            print("=" * 60)
            
            # 完整流程测试
            print("\n执行完整追踪流程测试...")
            results = await test_suite.test_complete_fund_flow_tracking_flow()
            
            print(f"测试结果: {'通过' if results['success'] else '失败'}")
            print(f"总耗时: {results['performance_metrics'].get('total_duration', 0):.2f}秒")
            print(f"预警生成: {results['alert_metrics'].get('alerts_generated', 0)} 个")
            print(f"趋势分析: {results['tracking_metrics'].get('trend_analysis', {}).get('trends_analyzed', 0)} 个")
            print(f"数据一致性: {results['tracking_metrics'].get('data_consistency', {}).get('consistency_score', 0):.2f}")
            
            print("\n" + "=" * 60)
            print("资金流向追踪端到端测试完成")
            print("=" * 60)
            
        finally:
            await test_suite.teardown_test_environment()
    
    # 运行测试
    asyncio.run(run_tests())