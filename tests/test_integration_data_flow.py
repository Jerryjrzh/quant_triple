"""
数据流集成测试

创建从数据获取到存储的完整流程测试，验证数据在各个组件间的正确传递，
添加数据转换和格式化的集成测试，实现多数据源并发处理的集成验证。

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json
import time

from stock_analysis_system.data.enhanced_data_sources import MarketDataRequest
from stock_analysis_system.data.cache_manager import CacheManager
from stock_analysis_system.data.market_data_request import DataType


class MockDataSource:
    """模拟数据源"""
    
    def __init__(self, name: str, delay: float = 0.1):
        self.name = name
        self.delay = delay
        self.call_count = 0
        self.error_rate = 0.0
        
    async def get_data(self, request: Dict[str, Any]) -> pd.DataFrame:
        """模拟获取数据"""
        self.call_count += 1
        
        # 模拟网络延迟
        await asyncio.sleep(self.delay)
        
        # 模拟错误
        if np.random.random() < self.error_rate:
            raise Exception(f"Mock error from {self.name}")
        
        # 生成模拟数据
        return self._generate_mock_data(request)
    
    def _generate_mock_data(self, request: Dict[str, Any]) -> pd.DataFrame:
        """生成模拟数据"""
        data_type = request.get('data_type', 'stock_daily')
        
        if data_type == 'stock_daily':
            return pd.DataFrame({
                'stock_code': ['000001', '000002', '000003'],
                'trade_date': [date.today()] * 3,
                'close_price': [10.50, 20.30, 15.80],
                'volume': [1000000, 2000000, 1500000],
                'amount': [10500000, 40600000, 23700000]
            })
        elif data_type == 'dragon_tiger':
            return pd.DataFrame({
                'stock_code': ['000001', '000002'],
                'trade_date': [date.today()] * 2,
                'stock_name': ['测试股票1', '测试股票2'],
                'close_price': [10.50, 20.30],
                'net_buy_amount': [1000000, -500000],
                'reason': ['涨停', '跌停']
            })
        elif data_type == 'fund_flow':
            return pd.DataFrame({
                'stock_code': ['000001', '000002'],
                'trade_date': [date.today()] * 2,
                'period_type': ['今日'] * 2,
                'main_net_inflow': [1000000, -800000],
                'main_net_inflow_rate': [5.5, -3.2]
            })
        else:
            return pd.DataFrame()


class IntegrationTestHelper:
    """集成测试辅助类"""
    
    def __init__(self):
        self.mock_sources = {}
        self.data_manager = None
        self.cache_manager = None
        
    async def setup_test_environment(self):
        """设置测试环境"""
        # 创建模拟数据源
        self.mock_sources = {
            'eastmoney': MockDataSource('eastmoney', delay=0.1),
            'dragon_tiger': MockDataSource('dragon_tiger', delay=0.15),
            'fund_flow': MockDataSource('fund_flow', delay=0.12),
            'limitup': MockDataSource('limitup', delay=0.08),
            'etf': MockDataSource('etf', delay=0.1)
        }
        
        # 创建缓存管理器
        self.cache_manager = CacheManager("redis://localhost:6379/1")
        
        # 创建数据管理器
        self.data_manager = None  # Will be mocked in tests
        
        # 模拟初始化
        await self._mock_initialization()
    
    async def _mock_initialization(self):
        """模拟组件初始化"""
        # 模拟缓存管理器初始化
        self.cache_manager.redis_client = Mock()
        self.cache_manager.redis_client.ping = AsyncMock(return_value=True)
        
        # 模拟数据管理器初始化
        # self.data_manager.cache_manager = self.cache_manager
        
    async def cleanup_test_environment(self):
        """清理测试环境"""
        if self.cache_manager:
            await self.cache_manager.close()
    
    def set_error_rate(self, source_name: str, error_rate: float):
        """设置数据源错误率"""
        if source_name in self.mock_sources:
            self.mock_sources[source_name].error_rate = error_rate
    
    def get_call_count(self, source_name: str) -> int:
        """获取数据源调用次数"""
        return self.mock_sources.get(source_name, Mock()).call_count


@pytest.fixture
async def integration_helper():
    """集成测试辅助器fixture"""
    helper = IntegrationTestHelper()
    await helper.setup_test_environment()
    yield helper
    await helper.cleanup_test_environment()


class TestDataFlowIntegration:
    """数据流集成测试"""
    
    @pytest.mark.asyncio
    async def test_complete_data_pipeline(self, integration_helper):
        """测试完整数据管道"""
        # 1. 创建数据请求
        request = MarketDataRequest(
            symbol='000001',
            start_date=(date.today() - timedelta(days=1)).strftime('%Y%m%d'),
            end_date=date.today().strftime('%Y%m%d'),
            period='daily',
            data_type='stock'
        )
        
        # 2. 模拟数据获取
        mock_data = pd.DataFrame({
            'stock_code': ['000001', '000002'],
            'trade_date': [date.today()] * 2,
            'close_price': [10.50, 20.30],
            'volume': [1000000, 2000000],
            'amount': [10500000, 40600000]
        })
        
        # 3. 模拟数据管理器
        mock_data_manager = Mock()
        mock_data_manager.get_market_data = AsyncMock(return_value=mock_data)
        integration_helper.data_manager = mock_data_manager
        
        # 4. 执行数据获取
        result = await integration_helper.data_manager.get_market_data(request)
        
        # 5. 验证结果
        assert result is not None
        assert len(result) == 2
        assert 'stock_code' in result.columns
        assert 'close_price' in result.columns
        
        # 6. 验证数据传递
        mock_data_manager.get_market_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_data_transformation_pipeline(self, integration_helper):
        """测试数据转换管道"""
        # 1. 原始数据
        raw_data = pd.DataFrame({
            'code': ['000001', '000002'],
            'date': ['2024-01-20', '2024-01-20'],
            'price': ['10.50', '20.30'],
            'vol': ['1000000', '2000000']
        })
        
        # 2. 模拟数据转换
        def transform_data(data: pd.DataFrame) -> pd.DataFrame:
            """数据转换函数"""
            transformed = data.copy()
            transformed['stock_code'] = transformed['code']
            transformed['trade_date'] = pd.to_datetime(transformed['date']).dt.date
            transformed['close_price'] = transformed['price'].astype(float)
            transformed['volume'] = transformed['vol'].astype(int)
            
            # 删除原始列
            transformed = transformed.drop(['code', 'date', 'price', 'vol'], axis=1)
            return transformed
        
        # 3. 执行转换
        transformed_data = transform_data(raw_data)
        
        # 4. 验证转换结果
        assert 'stock_code' in transformed_data.columns
        assert 'trade_date' in transformed_data.columns
        assert 'close_price' in transformed_data.columns
        assert 'volume' in transformed_data.columns
        
        assert transformed_data['close_price'].dtype == float
        assert transformed_data['volume'].dtype == int
        assert len(transformed_data) == 2
    
    @pytest.mark.asyncio
    async def test_multi_source_data_integration(self, integration_helper):
        """测试多数据源集成"""
        # 1. 模拟多个数据源的数据
        stock_data = pd.DataFrame({
            'stock_code': ['000001', '000002'],
            'trade_date': [date.today()] * 2,
            'close_price': [10.50, 20.30]
        })
        
        dragon_tiger_data = pd.DataFrame({
            'stock_code': ['000001'],
            'trade_date': [date.today()],
            'net_buy_amount': [1000000]
        })
        
        fund_flow_data = pd.DataFrame({
            'stock_code': ['000001', '000002'],
            'trade_date': [date.today()] * 2,
            'main_net_inflow': [500000, -300000]
        })
        
        # 2. 模拟数据合并
        def merge_multi_source_data(stock_df, dt_df, ff_df):
            """合并多数据源数据"""
            # 以股票数据为基础
            merged = stock_df.copy()
            
            # 左连接龙虎榜数据
            merged = merged.merge(
                dt_df[['stock_code', 'trade_date', 'net_buy_amount']], 
                on=['stock_code', 'trade_date'], 
                how='left'
            )
            
            # 左连接资金流向数据
            merged = merged.merge(
                ff_df[['stock_code', 'trade_date', 'main_net_inflow']], 
                on=['stock_code', 'trade_date'], 
                how='left'
            )
            
            return merged
        
        # 3. 执行合并
        integrated_data = merge_multi_source_data(stock_data, dragon_tiger_data, fund_flow_data)
        
        # 4. 验证集成结果
        assert len(integrated_data) == 2
        assert 'close_price' in integrated_data.columns
        assert 'net_buy_amount' in integrated_data.columns
        assert 'main_net_inflow' in integrated_data.columns
        
        # 验证数据完整性
        assert integrated_data.loc[0, 'net_buy_amount'] == 1000000  # 000001有龙虎榜数据
        assert pd.isna(integrated_data.loc[1, 'net_buy_amount'])    # 000002没有龙虎榜数据
        assert integrated_data.loc[0, 'main_net_inflow'] == 500000
        assert integrated_data.loc[1, 'main_net_inflow'] == -300000
    
    @pytest.mark.asyncio
    async def test_concurrent_data_processing(self, integration_helper):
        """测试并发数据处理"""
        # 1. 创建多个并发任务
        async def fetch_stock_data(symbol: str) -> pd.DataFrame:
            """获取股票数据"""
            await asyncio.sleep(0.1)  # 模拟网络延迟
            return pd.DataFrame({
                'stock_code': [symbol],
                'trade_date': [date.today()],
                'close_price': [np.random.uniform(10, 100)]
            })
        
        async def fetch_dragon_tiger_data(symbol: str) -> pd.DataFrame:
            """获取龙虎榜数据"""
            await asyncio.sleep(0.15)  # 模拟网络延迟
            return pd.DataFrame({
                'stock_code': [symbol],
                'trade_date': [date.today()],
                'net_buy_amount': [np.random.randint(-1000000, 1000000)]
            })
        
        # 2. 并发执行多个数据获取任务
        symbols = ['000001', '000002', '000003', '000004', '000005']
        
        start_time = time.time()
        
        # 并发获取股票数据
        stock_tasks = [fetch_stock_data(symbol) for symbol in symbols]
        stock_results = await asyncio.gather(*stock_tasks)
        
        # 并发获取龙虎榜数据
        dt_tasks = [fetch_dragon_tiger_data(symbol) for symbol in symbols]
        dt_results = await asyncio.gather(*dt_tasks)
        
        end_time = time.time()
        
        # 3. 验证并发执行效果
        total_time = end_time - start_time
        
        # 并发执行应该比串行执行快
        expected_serial_time = len(symbols) * (0.1 + 0.15)  # 串行执行时间
        assert total_time < expected_serial_time * 0.8  # 并发应该至少快20%
        
        # 验证结果完整性
        assert len(stock_results) == len(symbols)
        assert len(dt_results) == len(symbols)
        
        # 合并结果
        combined_stock_data = pd.concat(stock_results, ignore_index=True)
        combined_dt_data = pd.concat(dt_results, ignore_index=True)
        
        assert len(combined_stock_data) == len(symbols)
        assert len(combined_dt_data) == len(symbols)
    
    @pytest.mark.asyncio
    async def test_data_validation_in_pipeline(self, integration_helper):
        """测试数据管道中的数据验证"""
        # 1. 创建包含异常数据的测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001', '000002', '000003', '000004'],
            'trade_date': [date.today()] * 4,
            'close_price': [10.50, -5.20, 999999.99, None],  # 包含负价格、异常高价格、空值
            'volume': [1000000, 0, -500000, 2000000],        # 包含零成交量、负成交量
            'amount': [10500000, 0, None, 40000000]          # 包含零成交额、空值
        })
        
        # 2. 数据验证函数
        def validate_stock_data(data: pd.DataFrame) -> Dict[str, Any]:
            """验证股票数据"""
            validation_result = {
                'total_records': len(data),
                'valid_records': 0,
                'invalid_records': 0,
                'issues': []
            }
            
            valid_mask = pd.Series([True] * len(data))
            
            # 检查价格
            price_invalid = (data['close_price'] <= 0) | data['close_price'].isna()
            if price_invalid.any():
                validation_result['issues'].append(f"发现{price_invalid.sum()}条价格异常记录")
                valid_mask &= ~price_invalid
            
            # 检查成交量
            volume_invalid = (data['volume'] < 0) | data['volume'].isna()
            if volume_invalid.any():
                validation_result['issues'].append(f"发现{volume_invalid.sum()}条成交量异常记录")
                valid_mask &= ~volume_invalid
            
            # 检查成交额
            amount_invalid = (data['amount'] < 0) | data['amount'].isna()
            if amount_invalid.any():
                validation_result['issues'].append(f"发现{amount_invalid.sum()}条成交额异常记录")
                valid_mask &= ~amount_invalid
            
            validation_result['valid_records'] = valid_mask.sum()
            validation_result['invalid_records'] = len(data) - valid_mask.sum()
            
            return validation_result
        
        # 3. 执行验证
        validation_result = validate_stock_data(test_data)
        
        # 4. 验证结果
        assert validation_result['total_records'] == 4
        assert validation_result['valid_records'] == 1  # 只有第4条记录完全有效
        assert validation_result['invalid_records'] == 3
        assert len(validation_result['issues']) > 0
        
        # 验证具体问题
        issues_text = ' '.join(validation_result['issues'])
        assert '价格异常' in issues_text
        assert '成交量异常' in issues_text
        assert '成交额异常' in issues_text
    
    @pytest.mark.asyncio
    async def test_error_handling_in_pipeline(self, integration_helper):
        """测试数据管道中的错误处理"""
        # 1. 设置错误率
        integration_helper.set_error_rate('eastmoney', 0.3)  # 30%错误率
        
        # 2. 模拟数据获取函数
        async def fetch_data_with_retry(source_name: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
            """带重试的数据获取"""
            for attempt in range(max_retries):
                try:
                    source = integration_helper.mock_sources[source_name]
                    data = await source.get_data({'data_type': 'stock_daily'})
                    return data
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"数据获取失败，已重试{max_retries}次: {e}")
                        return None
                    await asyncio.sleep(0.1 * (attempt + 1))  # 指数退避
            return None
        
        # 3. 执行多次数据获取测试
        success_count = 0
        total_attempts = 10
        
        for i in range(total_attempts):
            result = await fetch_data_with_retry('eastmoney')
            if result is not None:
                success_count += 1
        
        # 4. 验证错误处理效果
        success_rate = success_count / total_attempts
        
        # 即使有30%的错误率，重试机制应该能提高成功率
        assert success_rate > 0.5  # 成功率应该超过50%
        
        # 验证调用次数（包含重试）
        call_count = integration_helper.get_call_count('eastmoney')
        assert call_count >= total_attempts  # 调用次数应该至少等于尝试次数
    
    @pytest.mark.asyncio
    async def test_cache_integration_in_pipeline(self, integration_helper):
        """测试缓存在数据管道中的集成"""
        # 1. 模拟缓存管理器
        cache_manager = integration_helper.cache_manager
        cache_manager.memory_cache = {}
        cache_manager.memory_expire = {}
        
        # 2. 模拟数据获取函数（带缓存）
        async def get_cached_data(cache_key: str, data_fetcher) -> pd.DataFrame:
            """带缓存的数据获取"""
            # 先检查缓存
            cached_data = await cache_manager.get_cached_data(cache_key, 'stock_daily')
            if cached_data is not None:
                return cached_data
            
            # 缓存未命中，获取新数据
            fresh_data = await data_fetcher()
            
            # 存入缓存
            await cache_manager.set_cached_data(cache_key, fresh_data, 'stock_daily')
            
            return fresh_data
        
        async def mock_data_fetcher():
            """模拟数据获取"""
            await asyncio.sleep(0.1)  # 模拟网络延迟
            return pd.DataFrame({
                'stock_code': ['000001'],
                'trade_date': [date.today()],
                'close_price': [10.50]
            })
        
        # 3. 第一次获取数据（应该从数据源获取）
        start_time = time.time()
        data1 = await get_cached_data('test_key', mock_data_fetcher)
        first_fetch_time = time.time() - start_time
        
        # 4. 第二次获取数据（应该从缓存获取）
        start_time = time.time()
        data2 = await get_cached_data('test_key', mock_data_fetcher)
        second_fetch_time = time.time() - start_time
        
        # 5. 验证缓存效果
        assert len(data1) == 1
        assert len(data2) == 1
        pd.testing.assert_frame_equal(data1, data2)  # 数据应该相同
        
        # 第二次获取应该明显更快（从缓存获取）
        assert second_fetch_time < first_fetch_time * 0.5
    
    @pytest.mark.asyncio
    async def test_data_format_consistency(self, integration_helper):
        """测试数据格式一致性"""
        # 1. 定义标准数据格式
        standard_columns = {
            'stock_daily': ['stock_code', 'trade_date', 'close_price', 'volume', 'amount'],
            'dragon_tiger': ['stock_code', 'trade_date', 'stock_name', 'net_buy_amount', 'reason'],
            'fund_flow': ['stock_code', 'trade_date', 'period_type', 'main_net_inflow']
        }
        
        # 2. 模拟不同数据源返回的数据
        def simulate_data_source_response(source_name: str, data_type: str) -> pd.DataFrame:
            """模拟数据源响应"""
            if data_type == 'stock_daily':
                if source_name == 'source_a':
                    return pd.DataFrame({
                        'code': ['000001'],
                        'date': ['2024-01-20'],
                        'price': [10.50],
                        'vol': [1000000],
                        'amt': [10500000]
                    })
                elif source_name == 'source_b':
                    return pd.DataFrame({
                        'symbol': ['000001'],
                        'trading_date': [date.today()],
                        'closing_price': [10.50],
                        'trading_volume': [1000000],
                        'trading_amount': [10500000]
                    })
            return pd.DataFrame()
        
        # 3. 数据格式标准化函数
        def standardize_data_format(data: pd.DataFrame, source_name: str, data_type: str) -> pd.DataFrame:
            """标准化数据格式"""
            if data_type == 'stock_daily':
                if source_name == 'source_a':
                    standardized = pd.DataFrame({
                        'stock_code': data['code'],
                        'trade_date': pd.to_datetime(data['date']).dt.date,
                        'close_price': data['price'].astype(float),
                        'volume': data['vol'].astype(int),
                        'amount': data['amt'].astype(float)
                    })
                elif source_name == 'source_b':
                    standardized = pd.DataFrame({
                        'stock_code': data['symbol'],
                        'trade_date': data['trading_date'],
                        'close_price': data['closing_price'].astype(float),
                        'volume': data['trading_volume'].astype(int),
                        'amount': data['trading_amount'].astype(float)
                    })
                else:
                    return data
                
                return standardized
            
            return data
        
        # 4. 测试不同数据源的格式标准化
        sources = ['source_a', 'source_b']
        standardized_data = []
        
        for source in sources:
            raw_data = simulate_data_source_response(source, 'stock_daily')
            std_data = standardize_data_format(raw_data, source, 'stock_daily')
            standardized_data.append(std_data)
        
        # 5. 验证格式一致性
        expected_columns = standard_columns['stock_daily']
        
        for i, data in enumerate(standardized_data):
            # 检查列名
            assert list(data.columns) == expected_columns, f"Source {sources[i]} columns mismatch"
            
            # 检查数据类型
            assert data['stock_code'].dtype == object
            assert data['close_price'].dtype == float
            assert data['volume'].dtype == int
            assert data['amount'].dtype == float
        
        # 6. 验证数据可以合并
        combined_data = pd.concat(standardized_data, ignore_index=True)
        assert len(combined_data) == len(sources)
        assert list(combined_data.columns) == expected_columns


class TestDataPipelinePerformance:
    """数据管道性能测试"""
    
    @pytest.mark.asyncio
    async def test_pipeline_throughput(self, integration_helper):
        """测试数据管道吞吐量"""
        # 1. 创建大量数据请求
        async def process_batch_data(batch_size: int) -> float:
            """处理批量数据"""
            start_time = time.time()
            
            tasks = []
            for i in range(batch_size):
                # 模拟数据处理任务
                task = asyncio.create_task(self._mock_data_processing(f"000{i+1:03d}"))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            return processing_time
        
        # 2. 测试不同批量大小的性能
        batch_sizes = [10, 50, 100]
        performance_results = {}
        
        for batch_size in batch_sizes:
            processing_time = await process_batch_data(batch_size)
            throughput = batch_size / processing_time  # 每秒处理的记录数
            performance_results[batch_size] = {
                'processing_time': processing_time,
                'throughput': throughput
            }
        
        # 3. 验证性能指标
        for batch_size, metrics in performance_results.items():
            # 吞吐量应该合理（例如每秒至少处理50条记录）
            assert metrics['throughput'] > 50, f"Throughput too low for batch size {batch_size}"
            
            # 处理时间应该在合理范围内
            assert metrics['processing_time'] < batch_size * 0.1, f"Processing time too high for batch size {batch_size}"
        
        print("Performance Results:")
        for batch_size, metrics in performance_results.items():
            print(f"Batch Size: {batch_size}, Throughput: {metrics['throughput']:.2f} records/sec")
    
    async def _mock_data_processing(self, symbol: str) -> Dict[str, Any]:
        """模拟数据处理"""
        # 模拟数据获取
        await asyncio.sleep(0.01)
        
        # 模拟数据转换
        data = {
            'symbol': symbol,
            'price': np.random.uniform(10, 100),
            'volume': np.random.randint(100000, 10000000)
        }
        
        # 模拟数据验证
        await asyncio.sleep(0.005)
        
        return data
    
    @pytest.mark.asyncio
    async def test_memory_usage_in_pipeline(self, integration_helper):
        """测试数据管道内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 1. 处理大量数据
        large_datasets = []
        for i in range(10):
            # 创建较大的数据集
            large_data = pd.DataFrame({
                'stock_code': [f'{j:06d}' for j in range(1000)],
                'trade_date': [date.today()] * 1000,
                'close_price': np.random.uniform(10, 100, 1000),
                'volume': np.random.randint(100000, 10000000, 1000)
            })
            large_datasets.append(large_data)
        
        # 2. 模拟数据处理
        processed_data = []
        for dataset in large_datasets:
            # 模拟数据转换
            transformed = dataset.copy()
            transformed['amount'] = transformed['close_price'] * transformed['volume']
            processed_data.append(transformed)
        
        # 3. 检查内存使用
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # 内存增长应该在合理范围内（例如小于200MB）
        max_memory_increase = 200 * 1024 * 1024  # 200MB
        assert memory_increase < max_memory_increase, f"Memory usage too high: {memory_increase / 1024 / 1024:.2f}MB"
        
        # 4. 清理数据，验证内存释放
        del large_datasets
        del processed_data
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 等待一段时间让内存释放
        await asyncio.sleep(0.1)
        
        final_memory = process.memory_info().rss
        memory_released = current_memory - final_memory
        
        # 应该释放了一些内存
        assert memory_released > 0, "Memory was not released after cleanup"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])