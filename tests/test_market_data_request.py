"""
统一数据请求接口测试模块

测试内容：
- MarketDataRequest 数据模型验证
- 请求路由机制测试
- 参数验证测试
- 并发控制测试
- 性能监控测试

作者: Kiro
创建时间: 2024-01-01
"""

import pytest
import asyncio
from datetime import date, datetime
from unittest.mock import Mock, patch
import pandas as pd

from stock_analysis_system.data.market_data_request import (
    MarketDataRequest,
    DataType,
    DataSource,
    Period,
    RequestRouter,
    RequestValidator,
    RequestLogger,
    ConcurrencyController,
    UnifiedDataRequestInterface,
    process_market_data_request
)


class TestMarketDataRequest:
    """MarketDataRequest 数据模型测试"""
    
    def test_valid_request_creation(self):
        """测试有效请求创建"""
        request = MarketDataRequest(
            data_type=DataType.STOCK_REALTIME,
            symbol="000001.SZ",
            data_source=DataSource.EASTMONEY
        )
        
        assert request.data_type == DataType.STOCK_REALTIME
        assert request.symbol == "000001.SZ"
        assert request.data_source == DataSource.EASTMONEY
        assert request.period == Period.DAILY  # 默认值
        assert request.use_cache is True  # 默认值
    
    def test_symbol_validation(self):
        """测试股票代码验证"""
        # 有效的股票代码格式
        valid_symbols = ["000001.SZ", "600000.SH", "000001", "SZ000001"]
        
        for symbol in valid_symbols:
            request = MarketDataRequest(
                data_type=DataType.STOCK_REALTIME,
                symbol=symbol
            )
            assert request.symbol == symbol
        
        # 无效的股票代码格式
        invalid_symbols = ["00001", "000001.XX", "INVALID", ""]
        
        for symbol in invalid_symbols:
            with pytest.raises(ValueError):
                MarketDataRequest(
                    data_type=DataType.STOCK_REALTIME,
                    symbol=symbol
                )
    
    def test_symbols_list_validation(self):
        """测试股票代码列表验证"""
        # 有效的股票代码列表
        valid_symbols = ["000001.SZ", "600000.SH", "000002.SZ"]
        request = MarketDataRequest(
            data_type=DataType.STOCK_REALTIME,
            symbols=valid_symbols
        )
        assert request.symbols == valid_symbols
        
        # 包含无效代码的列表
        invalid_symbols = ["000001.SZ", "INVALID", "600000.SH"]
        with pytest.raises(ValueError):
            MarketDataRequest(
                data_type=DataType.STOCK_REALTIME,
                symbols=invalid_symbols
            )
    
    def test_date_validation(self):
        """测试日期验证"""
        # 字符串日期格式
        request = MarketDataRequest(
            data_type=DataType.STOCK_HISTORY,
            symbol="000001.SZ",
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        
        assert isinstance(request.start_date, date)
        assert isinstance(request.end_date, date)
        
        # 无效日期格式
        with pytest.raises(ValueError):
            MarketDataRequest(
                data_type=DataType.STOCK_HISTORY,
                symbol="000001.SZ",
                start_date="2024/01/01"  # 错误格式
            )
    
    def test_date_range_validation(self):
        """测试日期范围验证"""
        # 结束日期早于开始日期
        with pytest.raises(ValueError):
            MarketDataRequest(
                data_type=DataType.STOCK_HISTORY,
                symbol="000001.SZ",
                start_date="2024-01-31",
                end_date="2024-01-01"
            )
    
    def test_limit_validation(self):
        """测试限制参数验证"""
        # 有效限制
        request = MarketDataRequest(
            data_type=DataType.STOCK_REALTIME,
            symbol="000001.SZ",
            limit=1000
        )
        assert request.limit == 1000
        
        # 超出限制
        with pytest.raises(ValueError):
            MarketDataRequest(
                data_type=DataType.STOCK_REALTIME,
                symbol="000001.SZ",
                limit=20000
            )
        
        # 负数限制
        with pytest.raises(ValueError):
            MarketDataRequest(
                data_type=DataType.STOCK_REALTIME,
                symbol="000001.SZ",
                limit=-1
            )


class TestRequestRouter:
    """请求路由器测试"""
    
    def setup_method(self):
        """测试前置设置"""
        self.router = RequestRouter()
    
    def test_auto_routing(self):
        """测试自动路由"""
        request = MarketDataRequest(
            data_type=DataType.STOCK_REALTIME,
            symbol="000001.SZ",
            data_source=DataSource.AUTO
        )
        
        sources = self.router.route_request(request)
        assert len(sources) > 0
        assert DataSource.SINA in sources  # 实时数据优先使用新浪
    
    def test_specified_source_routing(self):
        """测试指定数据源路由"""
        request = MarketDataRequest(
            data_type=DataType.STOCK_REALTIME,
            symbol="000001.SZ",
            data_source=DataSource.EASTMONEY,
            backup_sources=[DataSource.SINA]
        )
        
        sources = self.router.route_request(request)
        assert sources[0] == DataSource.EASTMONEY
        assert DataSource.SINA in sources
    
    def test_dragon_tiger_routing(self):
        """测试龙虎榜数据路由"""
        request = MarketDataRequest(
            data_type=DataType.DRAGON_TIGER,
            trade_date="2024-01-01",
            data_source=DataSource.AUTO
        )
        
        sources = self.router.route_request(request)
        assert DataSource.EASTMONEY in sources  # 龙虎榜主要使用东方财富
    
    def test_get_source_info(self):
        """测试获取数据源信息"""
        info = self.router.get_source_info(DataSource.EASTMONEY)
        assert 'supported_types' in info
        assert 'rate_limit' in info
        assert 'reliability' in info


class TestRequestValidator:
    """请求验证器测试"""
    
    def test_stock_data_validation(self):
        """测试股票数据验证"""
        # 缺少股票代码
        request = MarketDataRequest(
            data_type=DataType.STOCK_REALTIME
        )
        
        errors = RequestValidator.validate_request(request)
        assert len(errors) > 0
        assert "symbol or symbols is required" in errors[0]
    
    def test_historical_data_validation(self):
        """测试历史数据验证"""
        # 缺少日期参数
        request = MarketDataRequest(
            data_type=DataType.STOCK_HISTORY,
            symbol="000001.SZ"
        )
        
        errors = RequestValidator.validate_request(request)
        assert len(errors) > 0
        assert "start_date or trade_date is required" in errors[0]
    
    def test_dragon_tiger_validation(self):
        """测试龙虎榜数据验证"""
        # 缺少日期参数
        request = MarketDataRequest(
            data_type=DataType.DRAGON_TIGER
        )
        
        errors = RequestValidator.validate_request(request)
        assert len(errors) > 0
        assert "trade_date or start_date is required" in errors[0]
    
    def test_fund_flow_validation(self):
        """测试资金流向数据验证"""
        # 缺少股票代码
        request = MarketDataRequest(
            data_type=DataType.FUND_FLOW
        )
        
        errors = RequestValidator.validate_request(request)
        assert len(errors) > 0
        assert "symbol is required" in errors[0]
    
    def test_valid_request_validation(self):
        """测试有效请求验证"""
        request = MarketDataRequest(
            data_type=DataType.STOCK_REALTIME,
            symbol="000001.SZ"
        )
        
        errors = RequestValidator.validate_request(request)
        assert len(errors) == 0


class TestRequestLogger:
    """请求日志记录器测试"""
    
    def setup_method(self):
        """测试前置设置"""
        self.logger = RequestLogger()
    
    def test_log_request_lifecycle(self):
        """测试请求生命周期日志"""
        request = MarketDataRequest(
            data_type=DataType.STOCK_REALTIME,
            symbol="000001.SZ"
        )
        
        # 记录开始
        metrics = self.logger.log_request_start(request)
        assert metrics.request_id == request.request_id
        assert metrics.start_time > 0
        
        # 记录结束
        self.logger.log_request_end(
            request.request_id,
            success=True,
            record_count=100,
            data_source="eastmoney",
            cache_hit=False
        )
        
        # 获取指标
        final_metrics = self.logger.get_metrics(request.request_id)
        assert final_metrics is not None
        assert final_metrics.duration is not None
        assert final_metrics.record_count == 100
        assert final_metrics.data_source == "eastmoney"
    
    def test_performance_stats(self):
        """测试性能统计"""
        # 创建一些测试请求
        for i in range(5):
            request = MarketDataRequest(
                data_type=DataType.STOCK_REALTIME,
                symbol=f"00000{i}.SZ"
            )
            
            metrics = self.logger.log_request_start(request)
            self.logger.log_request_end(
                request.request_id,
                success=True,
                record_count=100 + i,
                data_source="eastmoney"
            )
        
        stats = self.logger.get_performance_stats()
        assert stats['total_requests'] == 5
        assert 'avg_duration' in stats
        assert 'error_rate' in stats
        assert 'cache_hit_rate' in stats


class TestConcurrencyController:
    """并发控制器测试"""
    
    def setup_method(self):
        """测试前置设置"""
        self.controller = ConcurrencyController(max_concurrent_requests=2)
    
    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """测试并发限制"""
        requests = []
        for i in range(3):
            request = MarketDataRequest(
                data_type=DataType.STOCK_REALTIME,
                symbol=f"00000{i}.SZ"
            )
            requests.append(request)
        
        # 获取前两个请求的许可
        await self.controller.acquire(requests[0])
        await self.controller.acquire(requests[1])
        
        assert self.controller.get_active_request_count() == 2
        
        # 释放一个许可
        self.controller.release(requests[0].request_id)
        assert self.controller.get_active_request_count() == 1
        
        # 获取第三个请可
        await self.controller.acquire(requests[2])
        assert self.controller.get_active_request_count() == 2
    
    def test_rate_limiting(self):
        """测试限流机制"""
        # 模拟大量请求
        for _ in range(150):  # 超过东方财富的限制(100)
            self.controller.increment_request_count(DataSource.EASTMONEY)
        
        assert self.controller.is_rate_limited(DataSource.EASTMONEY) is True
        assert self.controller.is_rate_limited(DataSource.SINA) is False


class TestUnifiedDataRequestInterface:
    """统一数据请求接口测试"""
    
    def setup_method(self):
        """测试前置设置"""
        self.interface = UnifiedDataRequestInterface()
    
    @pytest.mark.asyncio
    async def test_successful_request(self):
        """测试成功请求"""
        request = MarketDataRequest(
            data_type=DataType.STOCK_REALTIME,
            symbol="000001.SZ"
        )
        
        result = await self.interface.process_request(request)
        
        assert result['success'] is True
        assert 'data' in result
        assert 'data_source' in result
        assert 'request_id' in result
        assert result['request_id'] == request.request_id
    
    @pytest.mark.asyncio
    async def test_invalid_request(self):
        """测试无效请求"""
        request = MarketDataRequest(
            data_type=DataType.STOCK_REALTIME
            # 缺少必需的 symbol 参数
        )
        
        result = await self.interface.process_request(request)
        
        assert result['success'] is False
        assert 'error' in result
        assert 'validation failed' in result['error'].lower()
    
    @pytest.mark.asyncio
    async def test_dragon_tiger_request(self):
        """测试龙虎榜数据请求"""
        request = MarketDataRequest(
            data_type=DataType.DRAGON_TIGER,
            trade_date="2024-01-01"
        )
        
        result = await self.interface.process_request(request)
        
        assert result['success'] is True
        assert isinstance(result['data'], pd.DataFrame)
    
    def test_performance_stats(self):
        """测试性能统计"""
        stats = self.interface.get_performance_stats()
        assert isinstance(stats, dict)
    
    def test_active_requests(self):
        """测试活跃请求数"""
        count = self.interface.get_active_requests()
        assert isinstance(count, int)
        assert count >= 0


class TestProcessMarketDataRequest:
    """便捷函数测试"""
    
    @pytest.mark.asyncio
    async def test_valid_request_data(self):
        """测试有效请求数据"""
        request_data = {
            'data_type': 'stock_realtime',
            'symbol': '000001.SZ',
            'data_source': 'eastmoney'
        }
        
        result = await process_market_data_request(request_data)
        
        assert result['success'] is True
        assert 'data' in result
    
    @pytest.mark.asyncio
    async def test_invalid_request_data(self):
        """测试无效请求数据"""
        request_data = {
            'data_type': 'invalid_type',  # 无效的数据类型
            'symbol': '000001.SZ'
        }
        
        result = await process_market_data_request(request_data)
        
        assert result['success'] is False
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        """测试缺少必需字段"""
        request_data = {
            'symbol': '000001.SZ'
            # 缺少 data_type
        }
        
        result = await process_market_data_request(request_data)
        
        assert result['success'] is False
        assert 'error' in result


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])