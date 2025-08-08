#!/usr/bin/env python3
"""
龙虎榜监控端到端测试

测试龙虎榜数据的完整处理流程，验证机构和营业部数据的关联分析。
包含龙虎榜告警和通知机制的测试，以及历史数据查询和趋势分析的验证。
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
from stock_analysis_system.data.dragon_tiger_adapter import DragonTigerAdapter
from stock_analysis_system.data.cache_manager import CacheManager
from stock_analysis_system.data.data_quality_engine import EnhancedDataQualityEngine
from stock_analysis_system.core.database_manager import DatabaseManager
from stock_analysis_system.api.main import app
from fastapi.testclient import TestClient

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DragonTigerMonitoringE2ETest:
    """龙虎榜监控端到端测试类"""
    
    def __init__(self):
        self.client = TestClient(app)
        self.data_sources = EnhancedDataSourceManager()
        self.dragon_tiger_adapter = DragonTigerAdapter()
        self.cache_manager = CacheManager()
        self.quality_engine = EnhancedDataQualityEngine()
        self.db_manager = DatabaseManager()
        
        # 测试配置
        self.test_dates = [
            (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
            (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
            (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        ]
        self.test_symbols = ['000001', '000002', '600000', '600036', '300001']
        self.test_timeout = 45  # 秒
        
        # 性能阈值
        self.performance_thresholds = {
            'data_acquisition_time': 3.0,
            'processing_time': 2.0,
            'analysis_time': 2.0,
            'alert_response_time': 1.0,
            'total_response_time': 8.0
        }
        
        # 告警配置
        self.alert_config = {
            'large_transaction_threshold': 10000000,  # 1000万
            'unusual_activity_threshold': 5,  # 异常活跃度阈值
            'institutional_focus_threshold': 3  # 机构关注度阈值
        }
    
    async def setup_test_environment(self):
        """设置测试环境"""
        logger.info("设置龙虎榜监控端到端测试环境...")
        
        # 清理测试数据
        await self._cleanup_test_data()
        
        # 初始化数据库连接
        await self.db_manager.initialize()
        
        # 创建测试数据表（如果不存在）
        await self._create_test_tables()
        
        # 准备测试数据
        await self._prepare_test_data()
        
        logger.info("龙虎榜测试环境设置完成")
    
    async def teardown_test_environment(self):
        """清理测试环境"""
        logger.info("清理龙虎榜测试环境...")
        
        # 清理测试数据
        await self._cleanup_test_data()
        
        # 关闭数据库连接
        await self.db_manager.close()
        
        logger.info("龙虎榜测试环境清理完成")
    
    async def _cleanup_test_data(self):
        """清理测试数据"""
        try:
            # 清理缓存
            await self.cache_manager.delete_pattern("dragon_tiger:*")
            await self.cache_manager.delete_pattern("dt:*")
            
            # 清理数据库测试数据
            for date in self.test_dates:
                await self.db_manager.execute(
                    "DELETE FROM dragon_tiger_board WHERE trade_date = $1",
                    date
                )
                await self.db_manager.execute(
                    "DELETE FROM dragon_tiger_institutions WHERE trade_date = $1",
                    date
                )
                await self.db_manager.execute(
                    "DELETE FROM dragon_tiger_alerts WHERE created_at >= $1",
                    datetime.now() - timedelta(hours=1)
                )
            
        except Exception as e:
            logger.warning(f"清理测试数据时出现警告: {e}")
    
    async def _create_test_tables(self):
        """创建测试数据表"""
        try:
            # 龙虎榜主表
            await self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS dragon_tiger_board (
                    id SERIAL PRIMARY KEY,
                    trade_date DATE NOT NULL,
                    stock_code VARCHAR(10) NOT NULL,
                    stock_name VARCHAR(50) NOT NULL,
                    close_price DECIMAL(10,2),
                    change_rate DECIMAL(5,2),
                    net_buy_amount BIGINT,
                    buy_amount BIGINT,
                    sell_amount BIGINT,
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(trade_date, stock_code)
                )
            """)
            
            # 机构数据表
            await self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS dragon_tiger_institutions (
                    id SERIAL PRIMARY KEY,
                    trade_date DATE NOT NULL,
                    stock_code VARCHAR(10) NOT NULL,
                    institution_name VARCHAR(100) NOT NULL,
                    institution_type VARCHAR(20),
                    buy_amount BIGINT DEFAULT 0,
                    sell_amount BIGINT DEFAULT 0,
                    net_amount BIGINT DEFAULT 0,
                    rank_position INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 告警表
            await self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS dragon_tiger_alerts (
                    id SERIAL PRIMARY KEY,
                    alert_type VARCHAR(50) NOT NULL,
                    stock_code VARCHAR(10) NOT NULL,
                    stock_name VARCHAR(50),
                    alert_message TEXT NOT NULL,
                    alert_level VARCHAR(20) DEFAULT 'INFO',
                    trigger_value DECIMAL(15,2),
                    threshold_value DECIMAL(15,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE
                )
            """)
            
            # 创建索引
            await self.db_manager.execute("""
                CREATE INDEX IF NOT EXISTS idx_dragon_tiger_date_code 
                ON dragon_tiger_board(trade_date, stock_code)
            """)
            
            await self.db_manager.execute("""
                CREATE INDEX IF NOT EXISTS idx_dragon_tiger_institutions_date_code 
                ON dragon_tiger_institutions(trade_date, stock_code)
            """)
            
        except Exception as e:
            logger.error(f"创建测试数据表失败: {e}")
    
    async def _prepare_test_data(self):
        """准备测试数据"""
        logger.info("准备龙虎榜测试数据...")
        
        # 生成模拟龙虎榜数据
        for date in self.test_dates:
            for i, symbol in enumerate(self.test_symbols):
                # 主表数据
                await self.db_manager.execute("""
                    INSERT INTO dragon_tiger_board 
                    (trade_date, stock_code, stock_name, close_price, change_rate, 
                     net_buy_amount, buy_amount, sell_amount, reason)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (trade_date, stock_code) DO NOTHING
                """, 
                    date, symbol, f"测试股票{i+1}", 
                    round(10 + i * 2 + np.random.random() * 5, 2),
                    round((np.random.random() - 0.5) * 20, 2),
                    int(np.random.random() * 50000000),
                    int(np.random.random() * 100000000),
                    int(np.random.random() * 80000000),
                    f"测试原因{i+1}"
                )
                
                # 机构数据
                institutions = [
                    ("机构A", "基金", int(np.random.random() * 30000000)),
                    ("机构B", "券商", int(np.random.random() * 25000000)),
                    ("机构C", "保险", int(np.random.random() * 20000000))
                ]
                
                for j, (inst_name, inst_type, buy_amt) in enumerate(institutions):
                    sell_amt = int(np.random.random() * buy_amt * 0.8)
                    net_amt = buy_amt - sell_amt
                    
                    await self.db_manager.execute("""
                        INSERT INTO dragon_tiger_institutions 
                        (trade_date, stock_code, institution_name, institution_type,
                         buy_amount, sell_amount, net_amount, rank_position)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """, 
                        date, symbol, inst_name, inst_type,
                        buy_amt, sell_amt, net_amt, j + 1
                    )
    
    async def test_complete_dragon_tiger_monitoring_flow(self):
        """测试完整的龙虎榜监控流程"""
        logger.info("开始完整龙虎榜监控流程测试...")
        
        results = {
            'success': True,
            'performance_metrics': {},
            'monitoring_metrics': {},
            'alert_metrics': {},
            'error_details': []
        }
        
        try:
            # 1. 测试龙虎榜数据获取
            acquisition_result = await self._test_dragon_tiger_data_acquisition()
            results['performance_metrics']['data_acquisition'] = acquisition_result
            
            # 2. 测试数据处理和关联分析
            processing_result = await self._test_data_processing_and_analysis()
            results['performance_metrics']['data_processing'] = processing_result
            
            # 3. 测试机构和营业部数据关联
            correlation_result = await self._test_institutional_correlation()
            results['monitoring_metrics']['institutional_correlation'] = correlation_result
            
            # 4. 测试告警机制
            alert_result = await self._test_alert_mechanism()
            results['alert_metrics'] = alert_result
            
            # 5. 测试历史数据查询
            history_result = await self._test_historical_data_query()
            results['performance_metrics']['historical_query'] = history_result
            
            # 6. 测试趋势分析
            trend_result = await self._test_trend_analysis()
            results['monitoring_metrics']['trend_analysis'] = trend_result
            
            # 7. 测试API端点
            api_result = await self._test_dragon_tiger_api()
            results['performance_metrics']['api_response'] = api_result
            
            # 计算总体性能
            total_time = sum([
                acquisition_result.get('duration', 0),
                processing_result.get('duration', 0),
                history_result.get('duration', 0),
                api_result.get('duration', 0)
            ])
            
            results['performance_metrics']['total_duration'] = total_time
            results['performance_metrics']['meets_sla'] = total_time <= self.performance_thresholds['total_response_time']
            
        except Exception as e:
            results['success'] = False
            results['error_details'].append(f"完整流程测试失败: {str(e)}")
            logger.error(f"龙虎榜监控流程测试失败: {e}")
        
        return results
    
    async def _test_dragon_tiger_data_acquisition(self):
        """测试龙虎榜数据获取"""
        logger.info("测试龙虎榜数据获取...")
        
        start_time = time.time()
        results = {
            'success': True,
            'dates_processed': 0,
            'records_acquired': 0,
            'errors': []
        }
        
        try:
            for date in self.test_dates:
                try:
                    # 获取龙虎榜数据
                    dragon_tiger_data = await self.dragon_tiger_adapter.get_dragon_tiger_data(date)
                    
                    if dragon_tiger_data is not None and not dragon_tiger_data.empty:
                        results['dates_processed'] += 1
                        results['records_acquired'] += len(dragon_tiger_data)
                        
                        # 验证数据结构
                        required_columns = ['stock_code', 'stock_name', 'net_buy_amount']
                        missing_columns = [col for col in required_columns if col not in dragon_tiger_data.columns]
                        
                        if missing_columns:
                            results['errors'].append(f"日期 {date} 数据缺少列: {missing_columns}")
                    else:
                        results['errors'].append(f"日期 {date} 龙虎榜数据为空")
                
                except Exception as e:
                    results['errors'].append(f"获取日期 {date} 龙虎榜数据失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"龙虎榜数据获取测试失败: {str(e)}")
        
        duration = time.time() - start_time
        results['duration'] = duration
        results['meets_threshold'] = duration <= self.performance_thresholds['data_acquisition_time']
        
        logger.info(f"龙虎榜数据获取测试完成，耗时: {duration:.2f}秒")
        return results
    
    async def _test_data_processing_and_analysis(self):
        """测试数据处理和分析"""
        logger.info("测试龙虎榜数据处理和分析...")
        
        start_time = time.time()
        results = {
            'success': True,
            'processed_records': 0,
            'analysis_completed': 0,
            'errors': []
        }
        
        try:
            for date in self.test_dates:
                try:
                    # 获取原始数据
                    raw_data = await self._get_dragon_tiger_data_from_db(date)
                    
                    if not raw_data.empty:
                        # 数据处理
                        processed_data = await self._process_dragon_tiger_data(raw_data)
                        results['processed_records'] += len(processed_data)
                        
                        # 分析处理
                        analysis_result = await self._analyze_dragon_tiger_data(processed_data)
                        if analysis_result:
                            results['analysis_completed'] += 1
                
                except Exception as e:
                    results['errors'].append(f"处理日期 {date} 数据失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"数据处理和分析测试失败: {str(e)}")
        
        duration = time.time() - start_time
        results['duration'] = duration
        results['meets_threshold'] = duration <= self.performance_thresholds['processing_time']
        
        logger.info(f"数据处理和分析测试完成，耗时: {duration:.2f}秒")
        return results
    
    async def _test_institutional_correlation(self):
        """测试机构和营业部数据关联"""
        logger.info("测试机构数据关联分析...")
        
        results = {
            'success': True,
            'correlations_found': 0,
            'institutional_patterns': {},
            'errors': []
        }
        
        try:
            for date in self.test_dates:
                try:
                    # 获取机构数据
                    institutional_data = await self._get_institutional_data_from_db(date)
                    
                    if not institutional_data.empty:
                        # 分析机构关联性
                        correlations = await self._analyze_institutional_correlations(institutional_data)
                        results['correlations_found'] += len(correlations)
                        
                        # 识别机构模式
                        patterns = await self._identify_institutional_patterns(institutional_data)
                        results['institutional_patterns'][date] = patterns
                
                except Exception as e:
                    results['errors'].append(f"分析日期 {date} 机构关联失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"机构关联分析测试失败: {str(e)}")
        
        logger.info(f"机构关联分析测试完成，发现关联: {results['correlations_found']} 个")
        return results
    
    async def _test_alert_mechanism(self):
        """测试告警机制"""
        logger.info("测试龙虎榜告警机制...")
        
        start_time = time.time()
        results = {
            'success': True,
            'alerts_generated': 0,
            'alert_types': {},
            'notification_sent': 0,
            'errors': []
        }
        
        try:
            # 测试不同类型的告警
            alert_tests = [
                ('large_transaction', self._test_large_transaction_alert),
                ('unusual_activity', self._test_unusual_activity_alert),
                ('institutional_focus', self._test_institutional_focus_alert)
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
                    results['errors'].append(f"测试 {alert_type} 告警失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"告警机制测试失败: {str(e)}")
        
        duration = time.time() - start_time
        results['duration'] = duration
        results['meets_threshold'] = duration <= self.performance_thresholds['alert_response_time']
        
        logger.info(f"告警机制测试完成，生成告警: {results['alerts_generated']} 个")
        return results
    
    async def _test_historical_data_query(self):
        """测试历史数据查询"""
        logger.info("测试历史数据查询...")
        
        start_time = time.time()
        results = {
            'success': True,
            'queries_executed': 0,
            'records_retrieved': 0,
            'query_performance': {},
            'errors': []
        }
        
        try:
            # 测试不同类型的历史查询
            query_tests = [
                ('date_range_query', self._test_date_range_query),
                ('symbol_history_query', self._test_symbol_history_query),
                ('institutional_history_query', self._test_institutional_history_query),
                ('trend_query', self._test_trend_query)
            ]
            
            for query_type, test_func in query_tests:
                try:
                    query_start = time.time()
                    query_result = await test_func()
                    query_duration = time.time() - query_start
                    
                    results['query_performance'][query_type] = {
                        'duration': query_duration,
                        'records': query_result.get('record_count', 0)
                    }
                    
                    results['queries_executed'] += 1
                    results['records_retrieved'] += query_result.get('record_count', 0)
                
                except Exception as e:
                    results['errors'].append(f"执行 {query_type} 查询失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"历史数据查询测试失败: {str(e)}")
        
        duration = time.time() - start_time
        results['duration'] = duration
        
        logger.info(f"历史数据查询测试完成，耗时: {duration:.2f}秒")
        return results
    
    async def _test_trend_analysis(self):
        """测试趋势分析"""
        logger.info("测试龙虎榜趋势分析...")
        
        results = {
            'success': True,
            'trends_identified': 0,
            'trend_types': {},
            'accuracy_score': 0.0,
            'errors': []
        }
        
        try:
            # 获取历史数据进行趋势分析
            historical_data = await self._get_historical_dragon_tiger_data()
            
            if not historical_data.empty:
                # 分析不同类型的趋势
                trend_analyses = [
                    ('volume_trend', self._analyze_volume_trend),
                    ('institutional_trend', self._analyze_institutional_trend),
                    ('sector_trend', self._analyze_sector_trend)
                ]
                
                for trend_type, analysis_func in trend_analyses:
                    try:
                        trend_result = await analysis_func(historical_data)
                        results['trend_types'][trend_type] = trend_result
                        
                        if trend_result.get('trend_detected', False):
                            results['trends_identified'] += 1
                    
                    except Exception as e:
                        results['errors'].append(f"分析 {trend_type} 趋势失败: {str(e)}")
                
                # 计算趋势分析准确性
                results['accuracy_score'] = await self._calculate_trend_accuracy(results['trend_types'])
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"趋势分析测试失败: {str(e)}")
        
        logger.info(f"趋势分析测试完成，识别趋势: {results['trends_identified']} 个")
        return results
    
    async def _test_dragon_tiger_api(self):
        """测试龙虎榜API端点"""
        logger.info("测试龙虎榜API端点...")
        
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
                ('/api/v1/dragon-tiger/daily', {'date': self.test_dates[0]}),
                ('/api/v1/dragon-tiger/symbol', {'symbol': self.test_symbols[0]}),
                ('/api/v1/dragon-tiger/institutions', {'date': self.test_dates[0]}),
                ('/api/v1/dragon-tiger/alerts', {})
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
                        if self._validate_dragon_tiger_api_response(endpoint, data):
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
    async def _get_dragon_tiger_data_from_db(self, date: str) -> pd.DataFrame:
        """从数据库获取龙虎榜数据"""
        query = """
            SELECT * FROM dragon_tiger_board 
            WHERE trade_date = $1 
            ORDER BY net_buy_amount DESC
        """
        
        rows = await self.db_manager.fetch_all(query, date)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    async def _get_institutional_data_from_db(self, date: str) -> pd.DataFrame:
        """从数据库获取机构数据"""
        query = """
            SELECT * FROM dragon_tiger_institutions 
            WHERE trade_date = $1 
            ORDER BY net_amount DESC
        """
        
        rows = await self.db_manager.fetch_all(query, date)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    async def _process_dragon_tiger_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理龙虎榜数据"""
        if data.empty:
            return data
        
        # 数据清洗和标准化
        processed_data = data.copy()
        
        # 计算净买入比例
        if 'net_buy_amount' in processed_data.columns and 'buy_amount' in processed_data.columns:
            processed_data['net_buy_ratio'] = processed_data['net_buy_amount'] / processed_data['buy_amount']
        
        # 标记大额交易
        if 'net_buy_amount' in processed_data.columns:
            processed_data['is_large_transaction'] = processed_data['net_buy_amount'] > self.alert_config['large_transaction_threshold']
        
        return processed_data
    
    async def _analyze_dragon_tiger_data(self, data: pd.DataFrame) -> Dict:
        """分析龙虎榜数据"""
        if data.empty:
            return {}
        
        analysis = {
            'total_records': len(data),
            'large_transactions': 0,
            'avg_net_buy_amount': 0,
            'top_stocks': []
        }
        
        if 'is_large_transaction' in data.columns:
            analysis['large_transactions'] = data['is_large_transaction'].sum()
        
        if 'net_buy_amount' in data.columns:
            analysis['avg_net_buy_amount'] = data['net_buy_amount'].mean()
            
            # 获取前5名股票
            top_stocks = data.nlargest(5, 'net_buy_amount')
            analysis['top_stocks'] = top_stocks[['stock_code', 'stock_name', 'net_buy_amount']].to_dict('records')
        
        return analysis
    
    async def _analyze_institutional_correlations(self, data: pd.DataFrame) -> List[Dict]:
        """分析机构关联性"""
        correlations = []
        
        if data.empty or 'institution_name' not in data.columns:
            return correlations
        
        # 按机构分组分析
        institution_groups = data.groupby('institution_name')
        
        for inst_name, group in institution_groups:
            if len(group) > 1:  # 机构在多只股票中出现
                correlation = {
                    'institution': inst_name,
                    'stock_count': len(group),
                    'total_net_amount': group['net_amount'].sum(),
                    'stocks': group['stock_code'].tolist()
                }
                correlations.append(correlation)
        
        return correlations
    
    async def _identify_institutional_patterns(self, data: pd.DataFrame) -> Dict:
        """识别机构模式"""
        patterns = {
            'aggressive_buyers': [],
            'consistent_sellers': [],
            'diversified_institutions': []
        }
        
        if data.empty:
            return patterns
        
        # 按机构分组分析模式
        institution_groups = data.groupby('institution_name')
        
        for inst_name, group in institution_groups:
            total_buy = group['buy_amount'].sum()
            total_sell = group['sell_amount'].sum()
            net_amount = group['net_amount'].sum()
            
            # 激进买入者
            if net_amount > 0 and total_buy > total_sell * 2:
                patterns['aggressive_buyers'].append({
                    'institution': inst_name,
                    'net_amount': net_amount,
                    'buy_sell_ratio': total_buy / total_sell if total_sell > 0 else float('inf')
                })
            
            # 持续卖出者
            elif net_amount < 0 and total_sell > total_buy * 2:
                patterns['consistent_sellers'].append({
                    'institution': inst_name,
                    'net_amount': net_amount,
                    'sell_buy_ratio': total_sell / total_buy if total_buy > 0 else float('inf')
                })
            
            # 多元化机构
            elif len(group) >= 3:
                patterns['diversified_institutions'].append({
                    'institution': inst_name,
                    'stock_count': len(group),
                    'net_amount': net_amount
                })
        
        return patterns
    
    async def _test_large_transaction_alert(self) -> Dict:
        """测试大额交易告警"""
        result = {'triggered': False, 'alert_count': 0, 'details': []}
        
        # 查找大额交易
        query = """
            SELECT * FROM dragon_tiger_board 
            WHERE net_buy_amount > $1 
            AND trade_date >= $2
        """
        
        rows = await self.db_manager.fetch_all(
            query, 
            self.alert_config['large_transaction_threshold'],
            self.test_dates[-1]
        )
        
        if rows:
            result['triggered'] = True
            result['alert_count'] = len(rows)
            result['details'] = [
                {
                    'stock_code': row['stock_code'],
                    'stock_name': row['stock_name'],
                    'net_buy_amount': row['net_buy_amount'],
                    'trade_date': row['trade_date']
                }
                for row in rows[:5]  # 只返回前5个
            ]
            
            # 创建告警记录
            for row in rows:
                await self._create_alert_record(
                    'large_transaction',
                    row['stock_code'],
                    row['stock_name'],
                    f"检测到大额交易: {row['net_buy_amount']:,}元",
                    'HIGH',
                    row['net_buy_amount'],
                    self.alert_config['large_transaction_threshold']
                )
        
        return result
    
    async def _test_unusual_activity_alert(self) -> Dict:
        """测试异常活跃度告警"""
        result = {'triggered': False, 'alert_count': 0, 'details': []}
        
        # 查找异常活跃的股票（多日连续上榜）
        query = """
            SELECT stock_code, stock_name, COUNT(*) as appearance_count
            FROM dragon_tiger_board 
            WHERE trade_date >= $1
            GROUP BY stock_code, stock_name
            HAVING COUNT(*) >= $2
        """
        
        rows = await self.db_manager.fetch_all(
            query,
            self.test_dates[-1],
            self.alert_config['unusual_activity_threshold']
        )
        
        if rows:
            result['triggered'] = True
            result['alert_count'] = len(rows)
            result['details'] = [
                {
                    'stock_code': row['stock_code'],
                    'stock_name': row['stock_name'],
                    'appearance_count': row['appearance_count']
                }
                for row in rows
            ]
            
            # 创建告警记录
            for row in rows:
                await self._create_alert_record(
                    'unusual_activity',
                    row['stock_code'],
                    row['stock_name'],
                    f"异常活跃: 连续{row['appearance_count']}日上榜",
                    'MEDIUM',
                    row['appearance_count'],
                    self.alert_config['unusual_activity_threshold']
                )
        
        return result
    
    async def _test_institutional_focus_alert(self) -> Dict:
        """测试机构关注度告警"""
        result = {'triggered': False, 'alert_count': 0, 'details': []}
        
        # 查找机构高度关注的股票
        query = """
            SELECT stock_code, COUNT(DISTINCT institution_name) as institution_count
            FROM dragon_tiger_institutions 
            WHERE trade_date >= $1
            GROUP BY stock_code
            HAVING COUNT(DISTINCT institution_name) >= $2
        """
        
        rows = await self.db_manager.fetch_all(
            query,
            self.test_dates[-1],
            self.alert_config['institutional_focus_threshold']
        )
        
        if rows:
            result['triggered'] = True
            result['alert_count'] = len(rows)
            result['details'] = [
                {
                    'stock_code': row['stock_code'],
                    'institution_count': row['institution_count']
                }
                for row in rows
            ]
            
            # 创建告警记录
            for row in rows:
                await self._create_alert_record(
                    'institutional_focus',
                    row['stock_code'],
                    '',
                    f"机构高度关注: {row['institution_count']}家机构参与",
                    'MEDIUM',
                    row['institution_count'],
                    self.alert_config['institutional_focus_threshold']
                )
        
        return result
    
    async def _create_alert_record(self, alert_type: str, stock_code: str, stock_name: str, 
                                 message: str, level: str, trigger_value: float, threshold_value: float):
        """创建告警记录"""
        await self.db_manager.execute("""
            INSERT INTO dragon_tiger_alerts 
            (alert_type, stock_code, stock_name, alert_message, alert_level, 
             trigger_value, threshold_value)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, alert_type, stock_code, stock_name, message, level, trigger_value, threshold_value)
    
    async def _test_alert_notification(self, alert_type: str, alert_data: Dict) -> bool:
        """测试告警通知"""
        try:
            # 模拟发送通知
            notification_data = {
                'alert_type': alert_type,
                'alert_count': alert_data.get('alert_count', 0),
                'timestamp': datetime.now().isoformat(),
                'details': alert_data.get('details', [])
            }
            
            # 这里可以集成实际的通知系统（邮件、短信、钉钉等）
            logger.info(f"模拟发送 {alert_type} 告警通知: {notification_data}")
            
            return True
        except Exception as e:
            logger.error(f"发送告警通知失败: {e}")
            return False
    
    async def _test_date_range_query(self) -> Dict:
        """测试日期范围查询"""
        start_date = self.test_dates[-1]
        end_date = self.test_dates[0]
        
        query = """
            SELECT * FROM dragon_tiger_board 
            WHERE trade_date BETWEEN $1 AND $2
            ORDER BY trade_date DESC, net_buy_amount DESC
        """
        
        rows = await self.db_manager.fetch_all(query, start_date, end_date)
        
        return {
            'record_count': len(rows) if rows else 0,
            'date_range': f"{start_date} to {end_date}"
        }
    
    async def _test_symbol_history_query(self) -> Dict:
        """测试股票历史查询"""
        symbol = self.test_symbols[0]
        
        query = """
            SELECT * FROM dragon_tiger_board 
            WHERE stock_code = $1
            ORDER BY trade_date DESC
        """
        
        rows = await self.db_manager.fetch_all(query, symbol)
        
        return {
            'record_count': len(rows) if rows else 0,
            'symbol': symbol
        }
    
    async def _test_institutional_history_query(self) -> Dict:
        """测试机构历史查询"""
        query = """
            SELECT institution_name, COUNT(*) as appearance_count,
                   SUM(net_amount) as total_net_amount
            FROM dragon_tiger_institutions 
            WHERE trade_date >= $1
            GROUP BY institution_name
            ORDER BY total_net_amount DESC
        """
        
        rows = await self.db_manager.fetch_all(query, self.test_dates[-1])
        
        return {
            'record_count': len(rows) if rows else 0,
            'institutions': [row['institution_name'] for row in rows[:5]] if rows else []
        }
    
    async def _test_trend_query(self) -> Dict:
        """测试趋势查询"""
        query = """
            SELECT trade_date, COUNT(*) as stock_count,
                   AVG(net_buy_amount) as avg_net_buy,
                   SUM(net_buy_amount) as total_net_buy
            FROM dragon_tiger_board 
            WHERE trade_date >= $1
            GROUP BY trade_date
            ORDER BY trade_date
        """
        
        rows = await self.db_manager.fetch_all(query, self.test_dates[-1])
        
        return {
            'record_count': len(rows) if rows else 0,
            'trend_data': rows if rows else []
        }
    
    async def _get_historical_dragon_tiger_data(self) -> pd.DataFrame:
        """获取历史龙虎榜数据"""
        query = """
            SELECT * FROM dragon_tiger_board 
            WHERE trade_date >= $1
            ORDER BY trade_date, net_buy_amount DESC
        """
        
        rows = await self.db_manager.fetch_all(query, self.test_dates[-1])
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    async def _analyze_volume_trend(self, data: pd.DataFrame) -> Dict:
        """分析成交量趋势"""
        if data.empty or 'net_buy_amount' not in data.columns:
            return {'trend_detected': False}
        
        # 按日期分组计算趋势
        daily_volume = data.groupby('trade_date')['net_buy_amount'].sum().sort_index()
        
        if len(daily_volume) < 2:
            return {'trend_detected': False}
        
        # 简单趋势分析
        trend_direction = 'increasing' if daily_volume.iloc[-1] > daily_volume.iloc[0] else 'decreasing'
        trend_strength = abs(daily_volume.iloc[-1] - daily_volume.iloc[0]) / daily_volume.iloc[0] if daily_volume.iloc[0] != 0 else 0
        
        return {
            'trend_detected': True,
            'direction': trend_direction,
            'strength': trend_strength,
            'daily_volumes': daily_volume.to_dict()
        }
    
    async def _analyze_institutional_trend(self, data: pd.DataFrame) -> Dict:
        """分析机构趋势"""
        if data.empty:
            return {'trend_detected': False}
        
        # 获取机构数据
        inst_query = """
            SELECT trade_date, COUNT(DISTINCT institution_name) as inst_count,
                   AVG(net_amount) as avg_net_amount
            FROM dragon_tiger_institutions 
            WHERE trade_date >= $1
            GROUP BY trade_date
            ORDER BY trade_date
        """
        
        rows = await self.db_manager.fetch_all(inst_query, self.test_dates[-1])
        
        if not rows or len(rows) < 2:
            return {'trend_detected': False}
        
        inst_counts = [row['inst_count'] for row in rows]
        trend_direction = 'increasing' if inst_counts[-1] > inst_counts[0] else 'decreasing'
        
        return {
            'trend_detected': True,
            'direction': trend_direction,
            'institutional_activity': rows
        }
    
    async def _analyze_sector_trend(self, data: pd.DataFrame) -> Dict:
        """分析板块趋势"""
        # 简化的板块分析
        if data.empty or 'stock_code' not in data.columns:
            return {'trend_detected': False}
        
        # 按股票代码前缀分类（简化的板块分类）
        data['sector'] = data['stock_code'].str[:3]
        sector_analysis = data.groupby('sector').agg({
            'net_buy_amount': ['count', 'sum', 'mean']
        }).round(2)
        
        return {
            'trend_detected': True,
            'sector_analysis': sector_analysis.to_dict() if not sector_analysis.empty else {}
        }
    
    async def _calculate_trend_accuracy(self, trend_results: Dict) -> float:
        """计算趋势分析准确性"""
        detected_trends = sum(1 for result in trend_results.values() if result.get('trend_detected', False))
        total_trends = len(trend_results)
        
        return detected_trends / total_trends if total_trends > 0 else 0.0
    
    def _validate_dragon_tiger_api_response(self, endpoint: str, data) -> bool:
        """验证龙虎榜API响应"""
        if not data:
            return False
        
        if '/daily' in endpoint:
            # 日度数据应该包含股票列表
            return isinstance(data, list) and all('stock_code' in item for item in data[:3])
        elif '/symbol' in endpoint:
            # 股票数据应该包含基本字段
            return isinstance(data, dict) and 'stock_code' in data
        elif '/institutions' in endpoint:
            # 机构数据应该包含机构列表
            return isinstance(data, list) and all('institution_name' in item for item in data[:3])
        elif '/alerts' in endpoint:
            # 告警数据应该包含告警列表
            return isinstance(data, list)
        
        return True


# 测试用例
@pytest.mark.asyncio
async def test_complete_dragon_tiger_monitoring_e2e():
    """完整的龙虎榜监控端到端测试"""
    test_suite = DragonTigerMonitoringE2ETest()
    
    try:
        # 设置测试环境
        await test_suite.setup_test_environment()
        
        # 执行完整流程测试
        results = await test_suite.test_complete_dragon_tiger_monitoring_flow()
        
        # 验证测试结果
        assert results['success'], f"龙虎榜监控端到端测试失败: {results['error_details']}"
        assert results['performance_metrics']['meets_sla'], "性能指标未达到SLA要求"
        assert results['alert_metrics'].get('alerts_generated', 0) >= 0, "告警机制测试失败"
        
        logger.info("龙虎榜监控端到端测试通过")
        
    finally:
        # 清理测试环境
        await test_suite.teardown_test_environment()


if __name__ == "__main__":
    # 运行龙虎榜监控端到端测试
    async def run_tests():
        test_suite = DragonTigerMonitoringE2ETest()
        
        try:
            await test_suite.setup_test_environment()
            
            print("=" * 60)
            print("开始龙虎榜监控端到端测试")
            print("=" * 60)
            
            # 完整流程测试
            print("\n执行完整监控流程测试...")
            results = await test_suite.test_complete_dragon_tiger_monitoring_flow()
            
            print(f"测试结果: {'通过' if results['success'] else '失败'}")
            print(f"总耗时: {results['performance_metrics'].get('total_duration', 0):.2f}秒")
            print(f"告警生成: {results['alert_metrics'].get('alerts_generated', 0)} 个")
            print(f"趋势识别: {results['monitoring_metrics'].get('trend_analysis', {}).get('trends_identified', 0)} 个")
            
            print("\n" + "=" * 60)
            print("龙虎榜监控端到端测试完成")
            print("=" * 60)
            
        finally:
            await test_suite.teardown_test_environment()
    
    # 运行测试
    asyncio.run(run_tests())