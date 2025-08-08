#!/usr/bin/env python3
"""
实时行情分析端到端测试

测试从数据获取到分析结果的完整流程，验证实时数据处理的准确性和及时性。
包含异常情况下的系统恢复能力测试和用户界面API的端到端验证。
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import time
import logging
from typing import Dict, List, Optional

from stock_analysis_system.data.enhanced_data_sources import EnhancedDataSourceManager
from stock_analysis_system.data.data_source_manager import DataSourceManager
from stock_analysis_system.data.cache_manager import CacheManager
from stock_analysis_system.data.data_quality_engine import DataQualityEngine
from stock_analysis_system.core.database import DatabaseManager
from stock_analysis_system.api.main import app
from fastapi.testclient import TestClient

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeMarketAnalysisE2ETest:
    """实时行情分析端到端测试类"""
    
    def __init__(self):
        self.client = TestClient(app)
        self.data_sources = EnhancedDataSourceManager()
        self.cache_manager = CacheManager()
        self.quality_engine = DataQualityEngine()
        self.db_manager = DatabaseManager()
        
        # 测试配置
        self.test_symbols = ['000001', '000002', '600000', '600036']
        self.test_timeout = 30  # 秒
        self.performance_thresholds = {
            'data_acquisition_time': 2.0,  # 数据获取时间阈值
            'processing_time': 1.0,        # 数据处理时间阈值
            'total_response_time': 5.0,    # 总响应时间阈值
            'accuracy_threshold': 0.95     # 数据准确性阈值
        }
    
    async def setup_test_environment(self):
        """设置测试环境"""
        logger.info("设置端到端测试环境...")
        
        # 清理测试数据
        await self._cleanup_test_data()
        
        # 初始化数据库连接
        await self.db_manager.initialize()
        
        # 预热缓存
        await self._warmup_cache()
        
        logger.info("测试环境设置完成")
    
    async def teardown_test_environment(self):
        """清理测试环境"""
        logger.info("清理端到端测试环境...")
        
        # 清理测试数据
        await self._cleanup_test_data()
        
        # 关闭数据库连接
        await self.db_manager.close()
        
        logger.info("测试环境清理完成")
    
    async def _cleanup_test_data(self):
        """清理测试数据"""
        try:
            # 清理缓存中的测试数据
            for symbol in self.test_symbols:
                await self.cache_manager.delete_pattern(f"*{symbol}*")
            
            # 清理数据库中的测试数据
            test_date = datetime.now().strftime('%Y-%m-%d')
            await self.db_manager.execute(
                "DELETE FROM stock_data WHERE symbol = ANY($1) AND trade_date = $2",
                self.test_symbols, test_date
            )
            
        except Exception as e:
            logger.warning(f"清理测试数据时出现警告: {e}")
    
    async def _warmup_cache(self):
        """预热缓存"""
        logger.info("预热缓存...")
        
        # 预加载测试股票的基础信息
        for symbol in self.test_symbols:
            try:
                await self.data_sources.get_stock_info(symbol)
            except Exception as e:
                logger.warning(f"预热股票 {symbol} 信息失败: {e}")
    
    async def test_complete_realtime_analysis_flow(self):
        """测试完整的实时分析流程"""
        logger.info("开始完整实时分析流程测试...")
        
        results = {
            'success': True,
            'performance_metrics': {},
            'data_quality_metrics': {},
            'error_details': []
        }
        
        try:
            # 1. 测试实时数据获取
            acquisition_result = await self._test_realtime_data_acquisition()
            results['performance_metrics']['data_acquisition'] = acquisition_result
            
            # 2. 测试数据处理和验证
            processing_result = await self._test_data_processing()
            results['performance_metrics']['data_processing'] = processing_result
            
            # 3. 测试数据存储
            storage_result = await self._test_data_storage()
            results['performance_metrics']['data_storage'] = storage_result
            
            # 4. 测试分析计算
            analysis_result = await self._test_analysis_computation()
            results['performance_metrics']['analysis_computation'] = analysis_result
            
            # 5. 测试API响应
            api_result = await self._test_api_response()
            results['performance_metrics']['api_response'] = api_result
            
            # 6. 验证数据质量
            quality_result = await self._test_data_quality()
            results['data_quality_metrics'] = quality_result
            
            # 7. 计算总体性能指标
            total_time = sum([
                acquisition_result.get('duration', 0),
                processing_result.get('duration', 0),
                storage_result.get('duration', 0),
                analysis_result.get('duration', 0),
                api_result.get('duration', 0)
            ])
            
            results['performance_metrics']['total_duration'] = total_time
            results['performance_metrics']['meets_sla'] = total_time <= self.performance_thresholds['total_response_time']
            
        except Exception as e:
            results['success'] = False
            results['error_details'].append(f"完整流程测试失败: {str(e)}")
            logger.error(f"完整流程测试失败: {e}")
        
        return results
    
    async def _test_realtime_data_acquisition(self):
        """测试实时数据获取"""
        logger.info("测试实时数据获取...")
        
        start_time = time.time()
        results = {
            'success': True,
            'symbols_processed': 0,
            'data_points': 0,
            'errors': []
        }
        
        try:
            for symbol in self.test_symbols:
                try:
                    # 获取实时行情数据
                    realtime_data = await self.data_sources.get_realtime_data(symbol)
                    
                    if realtime_data is not None and not realtime_data.empty:
                        results['symbols_processed'] += 1
                        results['data_points'] += len(realtime_data)
                    else:
                        results['errors'].append(f"股票 {symbol} 实时数据为空")
                
                except Exception as e:
                    results['errors'].append(f"获取股票 {symbol} 实时数据失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"实时数据获取测试失败: {str(e)}")
        
        duration = time.time() - start_time
        results['duration'] = duration
        results['meets_threshold'] = duration <= self.performance_thresholds['data_acquisition_time']
        
        logger.info(f"实时数据获取测试完成，耗时: {duration:.2f}秒")
        return results
    
    async def _test_data_processing(self):
        """测试数据处理"""
        logger.info("测试数据处理...")
        
        start_time = time.time()
        results = {
            'success': True,
            'processed_records': 0,
            'validation_passed': 0,
            'errors': []
        }
        
        try:
            for symbol in self.test_symbols:
                try:
                    # 获取原始数据
                    raw_data = await self.data_sources.get_realtime_data(symbol)
                    
                    if raw_data is not None and not raw_data.empty:
                        # 数据验证
                        validation_result = await self.quality_engine.validate_realtime_data(raw_data)
                        
                        if validation_result.is_valid:
                            results['validation_passed'] += 1
                        
                        results['processed_records'] += len(raw_data)
                    
                except Exception as e:
                    results['errors'].append(f"处理股票 {symbol} 数据失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"数据处理测试失败: {str(e)}")
        
        duration = time.time() - start_time
        results['duration'] = duration
        results['meets_threshold'] = duration <= self.performance_thresholds['processing_time']
        
        logger.info(f"数据处理测试完成，耗时: {duration:.2f}秒")
        return results
    
    async def _test_data_storage(self):
        """测试数据存储"""
        logger.info("测试数据存储...")
        
        start_time = time.time()
        results = {
            'success': True,
            'stored_records': 0,
            'cache_hits': 0,
            'errors': []
        }
        
        try:
            for symbol in self.test_symbols:
                try:
                    # 获取数据
                    data = await self.data_sources.get_realtime_data(symbol)
                    
                    if data is not None and not data.empty:
                        # 存储到数据库
                        await self._store_test_data(symbol, data)
                        results['stored_records'] += len(data)
                        
                        # 测试缓存
                        cache_key = f"realtime:{symbol}"
                        await self.cache_manager.set(cache_key, data.to_dict(), ttl=60)
                        
                        cached_data = await self.cache_manager.get(cache_key)
                        if cached_data:
                            results['cache_hits'] += 1
                
                except Exception as e:
                    results['errors'].append(f"存储股票 {symbol} 数据失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"数据存储测试失败: {str(e)}")
        
        duration = time.time() - start_time
        results['duration'] = duration
        
        logger.info(f"数据存储测试完成，耗时: {duration:.2f}秒")
        return results
    
    async def _test_analysis_computation(self):
        """测试分析计算"""
        logger.info("测试分析计算...")
        
        start_time = time.time()
        results = {
            'success': True,
            'analysis_completed': 0,
            'indicators_calculated': 0,
            'errors': []
        }
        
        try:
            for symbol in self.test_symbols:
                try:
                    # 获取历史数据用于分析
                    historical_data = await self.data_sources.get_historical_data(
                        symbol, 
                        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                        end_date=datetime.now().strftime('%Y-%m-%d')
                    )
                    
                    if historical_data is not None and not historical_data.empty:
                        # 计算技术指标
                        indicators = await self._calculate_technical_indicators(historical_data)
                        
                        if indicators:
                            results['analysis_completed'] += 1
                            results['indicators_calculated'] += len(indicators)
                
                except Exception as e:
                    results['errors'].append(f"分析股票 {symbol} 失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"分析计算测试失败: {str(e)}")
        
        duration = time.time() - start_time
        results['duration'] = duration
        
        logger.info(f"分析计算测试完成，耗时: {duration:.2f}秒")
        return results
    
    async def _test_api_response(self):
        """测试API响应"""
        logger.info("测试API响应...")
        
        start_time = time.time()
        results = {
            'success': True,
            'api_calls': 0,
            'successful_calls': 0,
            'errors': []
        }
        
        try:
            for symbol in self.test_symbols:
                try:
                    # 测试实时行情API
                    response = self.client.get(f"/api/v1/realtime/{symbol}")
                    results['api_calls'] += 1
                    
                    if response.status_code == 200:
                        results['successful_calls'] += 1
                        
                        # 验证响应数据格式
                        data = response.json()
                        if self._validate_api_response(data):
                            logger.info(f"股票 {symbol} API响应验证通过")
                        else:
                            results['errors'].append(f"股票 {symbol} API响应格式不正确")
                    else:
                        results['errors'].append(f"股票 {symbol} API调用失败: {response.status_code}")
                
                except Exception as e:
                    results['errors'].append(f"测试股票 {symbol} API失败: {str(e)}")
        
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"API响应测试失败: {str(e)}")
        
        duration = time.time() - start_time
        results['duration'] = duration
        
        logger.info(f"API响应测试完成，耗时: {duration:.2f}秒")
        return results
    
    async def _test_data_quality(self):
        """测试数据质量"""
        logger.info("测试数据质量...")
        
        results = {
            'completeness_score': 0.0,
            'accuracy_score': 0.0,
            'timeliness_score': 0.0,
            'consistency_score': 0.0,
            'overall_score': 0.0,
            'issues': []
        }
        
        try:
            total_symbols = len(self.test_symbols)
            completeness_sum = 0.0
            accuracy_sum = 0.0
            timeliness_sum = 0.0
            consistency_sum = 0.0
            
            for symbol in self.test_symbols:
                try:
                    # 获取数据进行质量检查
                    data = await self.data_sources.get_realtime_data(symbol)
                    
                    if data is not None and not data.empty:
                        # 完整性检查
                        completeness = self._check_data_completeness(data)
                        completeness_sum += completeness
                        
                        # 准确性检查
                        accuracy = self._check_data_accuracy(data)
                        accuracy_sum += accuracy
                        
                        # 时效性检查
                        timeliness = self._check_data_timeliness(data)
                        timeliness_sum += timeliness
                        
                        # 一致性检查
                        consistency = self._check_data_consistency(data)
                        consistency_sum += consistency
                    
                except Exception as e:
                    results['issues'].append(f"检查股票 {symbol} 数据质量失败: {str(e)}")
            
            # 计算平均分数
            if total_symbols > 0:
                results['completeness_score'] = completeness_sum / total_symbols
                results['accuracy_score'] = accuracy_sum / total_symbols
                results['timeliness_score'] = timeliness_sum / total_symbols
                results['consistency_score'] = consistency_sum / total_symbols
                
                results['overall_score'] = (
                    results['completeness_score'] * 0.3 +
                    results['accuracy_score'] * 0.3 +
                    results['timeliness_score'] * 0.2 +
                    results['consistency_score'] * 0.2
                )
        
        except Exception as e:
            results['issues'].append(f"数据质量测试失败: {str(e)}")
        
        logger.info(f"数据质量测试完成，总体评分: {results['overall_score']:.2f}")
        return results
    
    async def test_exception_recovery(self):
        """测试异常情况下的系统恢复能力"""
        logger.info("测试异常恢复能力...")
        
        results = {
            'network_failure_recovery': False,
            'data_source_failure_recovery': False,
            'database_failure_recovery': False,
            'cache_failure_recovery': False,
            'recovery_times': {}
        }
        
        try:
            # 1. 测试网络异常恢复
            results['network_failure_recovery'] = await self._test_network_failure_recovery()
            
            # 2. 测试数据源异常恢复
            results['data_source_failure_recovery'] = await self._test_data_source_failure_recovery()
            
            # 3. 测试数据库异常恢复
            results['database_failure_recovery'] = await self._test_database_failure_recovery()
            
            # 4. 测试缓存异常恢复
            results['cache_failure_recovery'] = await self._test_cache_failure_recovery()
        
        except Exception as e:
            logger.error(f"异常恢复测试失败: {e}")
        
        return results
    
    async def _test_network_failure_recovery(self):
        """测试网络异常恢复"""
        logger.info("测试网络异常恢复...")
        
        try:
            # 模拟网络异常
            with patch('aiohttp.ClientSession.get') as mock_get:
                mock_get.side_effect = asyncio.TimeoutError("Network timeout")
                
                # 尝试获取数据，应该触发重试机制
                symbol = self.test_symbols[0]
                start_time = time.time()
                
                try:
                    await self.data_sources.get_realtime_data(symbol)
                except Exception:
                    pass  # 预期会失败
                
                # 恢复网络连接
                mock_get.side_effect = None
                
                # 验证系统能够恢复
                recovery_start = time.time()
                data = await self.data_sources.get_realtime_data(symbol)
                recovery_time = time.time() - recovery_start
                
                if data is not None:
                    logger.info(f"网络异常恢复成功，恢复时间: {recovery_time:.2f}秒")
                    return True
        
        except Exception as e:
            logger.error(f"网络异常恢复测试失败: {e}")
        
        return False
    
    async def _test_data_source_failure_recovery(self):
        """测试数据源异常恢复"""
        logger.info("测试数据源异常恢复...")
        
        try:
            # 模拟主数据源失效，测试备用数据源切换
            symbol = self.test_symbols[0]
            
            # 禁用主数据源
            original_primary = self.data_sources.primary_source
            self.data_sources.primary_source = None
            
            # 尝试获取数据，应该自动切换到备用数据源
            data = await self.data_sources.get_realtime_data(symbol)
            
            # 恢复主数据源
            self.data_sources.primary_source = original_primary
            
            if data is not None:
                logger.info("数据源异常恢复成功")
                return True
        
        except Exception as e:
            logger.error(f"数据源异常恢复测试失败: {e}")
        
        return False
    
    async def _test_database_failure_recovery(self):
        """测试数据库异常恢复"""
        logger.info("测试数据库异常恢复...")
        
        try:
            # 模拟数据库连接异常
            with patch.object(self.db_manager, 'execute') as mock_execute:
                mock_execute.side_effect = Exception("Database connection failed")
                
                # 尝试存储数据，应该触发重试或降级机制
                symbol = self.test_symbols[0]
                test_data = pd.DataFrame({
                    'symbol': [symbol],
                    'price': [100.0],
                    'volume': [1000],
                    'timestamp': [datetime.now()]
                })
                
                try:
                    await self._store_test_data(symbol, test_data)
                except Exception:
                    pass  # 预期会失败
                
                # 恢复数据库连接
                mock_execute.side_effect = None
                
                # 验证系统能够恢复
                await self._store_test_data(symbol, test_data)
                
                logger.info("数据库异常恢复成功")
                return True
        
        except Exception as e:
            logger.error(f"数据库异常恢复测试失败: {e}")
        
        return False
    
    async def _test_cache_failure_recovery(self):
        """测试缓存异常恢复"""
        logger.info("测试缓存异常恢复...")
        
        try:
            # 模拟缓存异常
            with patch.object(self.cache_manager, 'get') as mock_get:
                mock_get.side_effect = Exception("Cache connection failed")
                
                # 尝试获取缓存数据，应该降级到数据库
                symbol = self.test_symbols[0]
                data = await self.data_sources.get_realtime_data(symbol)
                
                if data is not None:
                    logger.info("缓存异常恢复成功")
                    return True
        
        except Exception as e:
            logger.error(f"缓存异常恢复测试失败: {e}")
        
        return False
    
    # 辅助方法
    async def _store_test_data(self, symbol: str, data: pd.DataFrame):
        """存储测试数据"""
        if data.empty:
            return
        
        # 简化的数据存储逻辑
        for _, row in data.iterrows():
            await self.db_manager.execute(
                """
                INSERT INTO stock_data (symbol, trade_date, open_price, close_price, high_price, low_price, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (symbol, trade_date) DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    close_price = EXCLUDED.close_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    volume = EXCLUDED.volume
                """,
                symbol,
                datetime.now().date(),
                row.get('open', 0),
                row.get('close', 0),
                row.get('high', 0),
                row.get('low', 0),
                row.get('volume', 0)
            )
    
    async def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """计算技术指标"""
        if data.empty:
            return {}
        
        indicators = {}
        
        try:
            # 简单移动平均线
            if 'close' in data.columns:
                indicators['sma_5'] = data['close'].rolling(window=5).mean().iloc[-1]
                indicators['sma_20'] = data['close'].rolling(window=20).mean().iloc[-1]
            
            # RSI
            if 'close' in data.columns and len(data) >= 14:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            if 'close' in data.columns and len(data) >= 26:
                exp1 = data['close'].ewm(span=12).mean()
                exp2 = data['close'].ewm(span=26).mean()
                indicators['macd'] = (exp1 - exp2).iloc[-1]
        
        except Exception as e:
            logger.warning(f"计算技术指标失败: {e}")
        
        return indicators
    
    def _validate_api_response(self, data: Dict) -> bool:
        """验证API响应格式"""
        required_fields = ['symbol', 'price', 'volume', 'timestamp']
        return all(field in data for field in required_fields)
    
    def _check_data_completeness(self, data: pd.DataFrame) -> float:
        """检查数据完整性"""
        if data.empty:
            return 0.0
        
        total_cells = data.size
        non_null_cells = data.count().sum()
        return non_null_cells / total_cells if total_cells > 0 else 0.0
    
    def _check_data_accuracy(self, data: pd.DataFrame) -> float:
        """检查数据准确性"""
        if data.empty:
            return 0.0
        
        # 简单的准确性检查：价格和成交量应该为正数
        accuracy_score = 1.0
        
        if 'price' in data.columns:
            invalid_prices = (data['price'] <= 0).sum()
            accuracy_score -= (invalid_prices / len(data)) * 0.5
        
        if 'volume' in data.columns:
            invalid_volumes = (data['volume'] < 0).sum()
            accuracy_score -= (invalid_volumes / len(data)) * 0.5
        
        return max(0.0, accuracy_score)
    
    def _check_data_timeliness(self, data: pd.DataFrame) -> float:
        """检查数据时效性"""
        if data.empty or 'timestamp' not in data.columns:
            return 0.0
        
        # 检查数据是否为最近的数据
        latest_timestamp = pd.to_datetime(data['timestamp'].max())
        current_time = datetime.now()
        time_diff = (current_time - latest_timestamp).total_seconds()
        
        # 如果数据在5分钟内，认为是及时的
        if time_diff <= 300:
            return 1.0
        elif time_diff <= 900:  # 15分钟内
            return 0.7
        elif time_diff <= 1800:  # 30分钟内
            return 0.5
        else:
            return 0.0
    
    def _check_data_consistency(self, data: pd.DataFrame) -> float:
        """检查数据一致性"""
        if data.empty:
            return 0.0
        
        consistency_score = 1.0
        
        # 检查价格逻辑一致性
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # 最高价应该 >= 开盘价、收盘价
            high_consistency = ((data['high'] >= data['open']) & 
                              (data['high'] >= data['close'])).all()
            
            # 最低价应该 <= 开盘价、收盘价
            low_consistency = ((data['low'] <= data['open']) & 
                             (data['low'] <= data['close'])).all()
            
            if not high_consistency:
                consistency_score -= 0.3
            if not low_consistency:
                consistency_score -= 0.3
        
        return max(0.0, consistency_score)


# 测试用例
@pytest.mark.asyncio
async def test_complete_realtime_analysis_e2e():
    """完整的实时分析端到端测试"""
    test_suite = RealtimeMarketAnalysisE2ETest()
    
    try:
        # 设置测试环境
        await test_suite.setup_test_environment()
        
        # 执行完整流程测试
        results = await test_suite.test_complete_realtime_analysis_flow()
        
        # 验证测试结果
        assert results['success'], f"端到端测试失败: {results['error_details']}"
        assert results['performance_metrics']['meets_sla'], "性能指标未达到SLA要求"
        assert results['data_quality_metrics']['overall_score'] >= 0.8, "数据质量评分过低"
        
        logger.info("完整实时分析端到端测试通过")
        
    finally:
        # 清理测试环境
        await test_suite.teardown_test_environment()


@pytest.mark.asyncio
async def test_exception_recovery_e2e():
    """异常恢复端到端测试"""
    test_suite = RealtimeMarketAnalysisE2ETest()
    
    try:
        # 设置测试环境
        await test_suite.setup_test_environment()
        
        # 执行异常恢复测试
        results = await test_suite.test_exception_recovery()
        
        # 验证恢复能力
        assert results['network_failure_recovery'], "网络异常恢复测试失败"
        assert results['data_source_failure_recovery'], "数据源异常恢复测试失败"
        
        logger.info("异常恢复端到端测试通过")
        
    finally:
        # 清理测试环境
        await test_suite.teardown_test_environment()


if __name__ == "__main__":
    # 运行端到端测试
    async def run_tests():
        test_suite = RealtimeMarketAnalysisE2ETest()
        
        try:
            await test_suite.setup_test_environment()
            
            print("=" * 60)
            print("开始实时行情分析端到端测试")
            print("=" * 60)
            
            # 完整流程测试
            print("\n1. 执行完整分析流程测试...")
            flow_results = await test_suite.test_complete_realtime_analysis_flow()
            print(f"流程测试结果: {'通过' if flow_results['success'] else '失败'}")
            print(f"总耗时: {flow_results['performance_metrics'].get('total_duration', 0):.2f}秒")
            print(f"数据质量评分: {flow_results['data_quality_metrics'].get('overall_score', 0):.2f}")
            
            # 异常恢复测试
            print("\n2. 执行异常恢复测试...")
            recovery_results = await test_suite.test_exception_recovery()
            print(f"网络异常恢复: {'通过' if recovery_results['network_failure_recovery'] else '失败'}")
            print(f"数据源异常恢复: {'通过' if recovery_results['data_source_failure_recovery'] else '失败'}")
            
            print("\n" + "=" * 60)
            print("端到端测试完成")
            print("=" * 60)
            
        finally:
            await test_suite.teardown_test_environment()
    
    # 运行测试
    asyncio.run(run_tests())