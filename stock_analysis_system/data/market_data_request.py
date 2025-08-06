"""
统一数据请求接口模块

该模块实现了统一的市场数据请求接口，包括：
- MarketDataRequest 数据类定义
- 请求路由机制
- 参数验证和预处理
- 请求日志记录和性能监控
- 异步数据获取和并发控制

作者: Kiro
创建时间: 2024-01-01
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, field_validator, Field
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import json

# 配置日志
logger = logging.getLogger(__name__)


class DataType(str, Enum):
    """数据类型枚举"""
    STOCK_REALTIME = "stock_realtime"  # 股票实时行情
    STOCK_HISTORY = "stock_history"    # 股票历史数据
    STOCK_INTRADAY = "stock_intraday"  # 股票分时数据
    DRAGON_TIGER = "dragon_tiger"      # 龙虎榜数据
    FUND_FLOW = "fund_flow"           # 资金流向数据
    LIMITUP_REASON = "limitup_reason"  # 涨停原因数据
    ETF_DATA = "etf_data"             # ETF数据
    SECTOR_DATA = "sector_data"        # 板块数据
    INDEX_DATA = "index_data"          # 指数数据


class DataSource(str, Enum):
    """数据源枚举"""
    EASTMONEY = "eastmoney"
    TONGHUASHUN = "tonghuashun"
    SINA = "sina"
    TENCENT = "tencent"
    AUTO = "auto"  # 自动选择最佳数据源


class Period(str, Enum):
    """时间周期枚举"""
    REALTIME = "realtime"  # 实时
    MINUTE_1 = "1min"      # 1分钟
    MINUTE_5 = "5min"      # 5分钟
    MINUTE_15 = "15min"    # 15分钟
    MINUTE_30 = "30min"    # 30分钟
    HOUR_1 = "1hour"       # 1小时
    DAILY = "daily"        # 日线
    WEEKLY = "weekly"      # 周线
    MONTHLY = "monthly"    # 月线


class MarketDataRequest(BaseModel):
    """市场数据请求模型"""
    
    # 基本参数
    data_type: DataType = Field(..., description="数据类型")
    symbol: Optional[str] = Field(None, description="股票代码")
    symbols: Optional[List[str]] = Field(None, description="股票代码列表")
    
    # 时间参数
    start_date: Optional[Union[str, date]] = Field(None, description="开始日期")
    end_date: Optional[Union[str, date]] = Field(None, description="结束日期")
    trade_date: Optional[Union[str, date]] = Field(None, description="交易日期")
    period: Period = Field(Period.DAILY, description="时间周期")
    
    # 数据源参数
    data_source: DataSource = Field(DataSource.AUTO, description="数据源")
    backup_sources: List[DataSource] = Field(default_factory=list, description="备用数据源")
    
    # 查询参数
    limit: Optional[int] = Field(None, ge=1, le=10000, description="返回记录数限制")
    offset: Optional[int] = Field(0, ge=0, description="偏移量")
    fields: Optional[List[str]] = Field(None, description="指定返回字段")
    
    # 过滤参数
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")
    sort_by: Optional[str] = Field(None, description="排序字段")
    sort_order: str = Field("desc", pattern="^(asc|desc)$", description="排序方向")
    
    # 缓存参数
    use_cache: bool = Field(True, description="是否使用缓存")
    cache_ttl: Optional[int] = Field(None, ge=0, description="缓存过期时间(秒)")
    force_refresh: bool = Field(False, description="强制刷新缓存")
    
    # 请求元数据
    request_id: str = Field(default_factory=lambda: f"req_{int(time.time() * 1000)}", description="请求ID")
    timeout: int = Field(30, ge=1, le=300, description="请求超时时间(秒)")
    retry_count: int = Field(3, ge=0, le=10, description="重试次数")
    
    class Config:
        use_enum_values = True
        
    @field_validator('symbol', 'symbols')
    @classmethod
    def validate_symbol(cls, v):
        """验证股票代码格式"""
        if v is None:
            return v
            
        if isinstance(v, str):
            # 验证单个股票代码
            if not cls._is_valid_symbol(v):
                raise ValueError(f"Invalid symbol format: {v}")
        elif isinstance(v, list):
            # 验证股票代码列表
            for symbol in v:
                if not cls._is_valid_symbol(symbol):
                    raise ValueError(f"Invalid symbol format: {symbol}")
        
        return v
    
    @staticmethod
    def _is_valid_symbol(symbol: str) -> bool:
        """验证股票代码格式"""
        if not symbol or len(symbol) < 6:
            return False
        
        # 支持的格式: 000001.SZ, 600000.SH, 000001, SZ000001
        if '.' in symbol:
            code, market = symbol.split('.')
            return len(code) == 6 and code.isdigit() and market in ['SZ', 'SH']
        elif symbol.startswith(('SZ', 'SH')):
            return len(symbol) == 8 and symbol[2:].isdigit()
        else:
            return len(symbol) == 6 and symbol.isdigit()
    
    @field_validator('start_date', 'end_date', 'trade_date')
    @classmethod
    def validate_dates(cls, v):
        """验证日期格式"""
        if v is None:
            return v
            
        if isinstance(v, str):
            try:
                return datetime.strptime(v, '%Y-%m-%d').date()
            except ValueError:
                raise ValueError(f"Invalid date format: {v}. Expected YYYY-MM-DD")
        
        return v
    
    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v, info):
        """验证日期范围"""
        if v is None:
            return v
            
        # Get start_date from context
        if hasattr(info, 'data') and 'start_date' in info.data:
            start_date = info.data['start_date']
            if start_date is None:
                return v
                
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            
            if isinstance(v, str):
                v = datetime.strptime(v, '%Y-%m-%d').date()
                
            if v < start_date:
                raise ValueError("end_date must be greater than or equal to start_date")
        
        return v


@dataclass
class RequestMetrics:
    """请求性能指标"""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    data_source: Optional[str] = None
    cache_hit: bool = False
    record_count: int = 0
    error: Optional[str] = None
    retry_count: int = 0


class RequestRouter:
    """请求路由器"""
    
    def __init__(self):
        self.data_source_priority = {
            DataType.STOCK_REALTIME: [DataSource.SINA, DataSource.TENCENT, DataSource.EASTMONEY],
            DataType.STOCK_HISTORY: [DataSource.EASTMONEY, DataSource.TONGHUASHUN],
            DataType.DRAGON_TIGER: [DataSource.EASTMONEY],
            DataType.FUND_FLOW: [DataSource.EASTMONEY],
            DataType.LIMITUP_REASON: [DataSource.TONGHUASHUN, DataSource.EASTMONEY],
            DataType.ETF_DATA: [DataSource.EASTMONEY, DataSource.SINA],
        }
        
        self.source_capabilities = {
            DataSource.EASTMONEY: {
                'supported_types': [
                    DataType.STOCK_REALTIME, DataType.STOCK_HISTORY, 
                    DataType.DRAGON_TIGER, DataType.FUND_FLOW, DataType.ETF_DATA
                ],
                'rate_limit': 100,  # 每分钟请求数
                'reliability': 0.95
            },
            DataSource.SINA: {
                'supported_types': [DataType.STOCK_REALTIME, DataType.ETF_DATA],
                'rate_limit': 200,
                'reliability': 0.90
            },
            DataSource.TONGHUASHUN: {
                'supported_types': [DataType.LIMITUP_REASON, DataType.STOCK_HISTORY],
                'rate_limit': 50,
                'reliability': 0.85
            },
            DataSource.TENCENT: {
                'supported_types': [DataType.STOCK_REALTIME],
                'rate_limit': 150,
                'reliability': 0.88
            }
        }
    
    def route_request(self, request: MarketDataRequest) -> List[DataSource]:
        """路由请求到合适的数据源"""
        if request.data_source != DataSource.AUTO:
            # 指定了数据源，直接返回
            sources = [request.data_source]
            sources.extend(request.backup_sources)
            return sources
        
        # 自动选择数据源
        available_sources = []
        
        # 获取支持该数据类型的数据源
        for source, capabilities in self.source_capabilities.items():
            if request.data_type in capabilities['supported_types']:
                available_sources.append(source)
        
        # 按优先级排序
        if request.data_type in self.data_source_priority:
            priority_sources = self.data_source_priority[request.data_type]
            sorted_sources = []
            
            # 先添加优先级高的数据源
            for source in priority_sources:
                if source in available_sources:
                    sorted_sources.append(source)
            
            # 再添加其他可用数据源
            for source in available_sources:
                if source not in sorted_sources:
                    sorted_sources.append(source)
            
            return sorted_sources
        
        return available_sources
    
    def get_source_info(self, source: DataSource) -> Dict[str, Any]:
        """获取数据源信息"""
        return self.source_capabilities.get(source, {})


class RequestValidator:
    """请求验证器"""
    
    @staticmethod
    def validate_request(request: MarketDataRequest) -> List[str]:
        """验证请求参数"""
        errors = []
        
        # 验证必需参数
        if request.data_type in [DataType.STOCK_REALTIME, DataType.STOCK_HISTORY, DataType.STOCK_INTRADAY]:
            if not request.symbol and not request.symbols:
                errors.append("symbol or symbols is required for stock data")
        
        # 验证日期参数
        if request.data_type == DataType.STOCK_HISTORY:
            if not request.start_date and not request.trade_date:
                errors.append("start_date or trade_date is required for historical data")
        
        # 验证龙虎榜参数
        if request.data_type == DataType.DRAGON_TIGER:
            if not request.trade_date and not request.start_date:
                errors.append("trade_date or start_date is required for dragon tiger data")
        
        # 验证资金流向参数
        if request.data_type == DataType.FUND_FLOW:
            if not request.symbol:
                errors.append("symbol is required for fund flow data")
        
        # 验证分页参数
        if request.limit and request.limit > 10000:
            errors.append("limit cannot exceed 10000")
        
        return errors


class RequestLogger:
    """请求日志记录器"""
    
    def __init__(self):
        self.metrics_storage = {}  # 在实际应用中应该使用数据库或缓存
        
    def log_request_start(self, request: MarketDataRequest) -> RequestMetrics:
        """记录请求开始"""
        metrics = RequestMetrics(
            request_id=request.request_id,
            start_time=time.time()
        )
        
        self.metrics_storage[request.request_id] = metrics
        
        logger.info(f"Request started: {request.request_id}", extra={
            'request_id': request.request_id,
            'data_type': request.data_type,
            'symbol': request.symbol,
            'data_source': request.data_source
        })
        
        return metrics
    
    def log_request_end(self, request_id: str, success: bool = True, 
                       error: Optional[str] = None, record_count: int = 0,
                       data_source: Optional[str] = None, cache_hit: bool = False):
        """记录请求结束"""
        if request_id not in self.metrics_storage:
            return
            
        metrics = self.metrics_storage[request_id]
        metrics.end_time = time.time()
        metrics.duration = metrics.end_time - metrics.start_time
        metrics.record_count = record_count
        metrics.data_source = data_source
        metrics.cache_hit = cache_hit
        
        if error:
            metrics.error = error
            logger.error(f"Request failed: {request_id}", extra={
                'request_id': request_id,
                'error': error,
                'duration': metrics.duration
            })
        else:
            logger.info(f"Request completed: {request_id}", extra={
                'request_id': request_id,
                'duration': metrics.duration,
                'record_count': record_count,
                'data_source': data_source,
                'cache_hit': cache_hit
            })
    
    def get_metrics(self, request_id: str) -> Optional[RequestMetrics]:
        """获取请求指标"""
        return self.metrics_storage.get(request_id)
    
    def get_performance_stats(self, time_window: int = 3600) -> Dict[str, Any]:
        """获取性能统计"""
        current_time = time.time()
        recent_metrics = [
            m for m in self.metrics_storage.values()
            if m.start_time > current_time - time_window and m.duration is not None
        ]
        
        if not recent_metrics:
            return {}
        
        durations = [m.duration for m in recent_metrics]
        error_count = len([m for m in recent_metrics if m.error])
        cache_hits = len([m for m in recent_metrics if m.cache_hit])
        
        return {
            'total_requests': len(recent_metrics),
            'avg_duration': sum(durations) / len(durations),
            'max_duration': max(durations),
            'min_duration': min(durations),
            'error_rate': error_count / len(recent_metrics),
            'cache_hit_rate': cache_hits / len(recent_metrics),
            'total_records': sum(m.record_count for m in recent_metrics)
        }


class ConcurrencyController:
    """并发控制器"""
    
    def __init__(self, max_concurrent_requests: int = 50):
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.active_requests = {}
        self.request_counts = {}  # 数据源请求计数
        
    async def acquire(self, request: MarketDataRequest) -> bool:
        """获取并发许可"""
        await self.semaphore.acquire()
        self.active_requests[request.request_id] = request
        return True
    
    def release(self, request_id: str):
        """释放并发许可"""
        if request_id in self.active_requests:
            del self.active_requests[request_id]
        self.semaphore.release()
    
    def get_active_request_count(self) -> int:
        """获取活跃请求数"""
        return len(self.active_requests)
    
    def is_rate_limited(self, data_source: DataSource) -> bool:
        """检查是否触发限流"""
        current_minute = int(time.time() // 60)
        key = f"{data_source}_{current_minute}"
        
        count = self.request_counts.get(key, 0)
        router = RequestRouter()
        source_info = router.get_source_info(data_source)
        rate_limit = source_info.get('rate_limit', 100)
        
        return count >= rate_limit
    
    def increment_request_count(self, data_source: DataSource):
        """增加请求计数"""
        current_minute = int(time.time() // 60)
        key = f"{data_source}_{current_minute}"
        self.request_counts[key] = self.request_counts.get(key, 0) + 1
        
        # 清理过期的计数
        expired_keys = [k for k in self.request_counts.keys() 
                       if int(k.split('_')[1]) < current_minute - 5]
        for key in expired_keys:
            del self.request_counts[key]


class UnifiedDataRequestInterface:
    """统一数据请求接口"""
    
    def __init__(self):
        self.router = RequestRouter()
        self.validator = RequestValidator()
        self.logger = RequestLogger()
        self.concurrency_controller = ConcurrencyController()
        
    async def process_request(self, request: MarketDataRequest) -> Dict[str, Any]:
        """处理数据请求"""
        # 记录请求开始
        metrics = self.logger.log_request_start(request)
        
        try:
            # 获取并发许可
            await self.concurrency_controller.acquire(request)
            
            # 验证请求参数
            validation_errors = self.validator.validate_request(request)
            if validation_errors:
                raise ValueError(f"Request validation failed: {', '.join(validation_errors)}")
            
            # 路由请求
            data_sources = self.router.route_request(request)
            if not data_sources:
                raise ValueError(f"No available data source for {request.data_type}")
            
            # 尝试从各个数据源获取数据
            last_error = None
            for data_source in data_sources:
                try:
                    # 检查限流
                    if self.concurrency_controller.is_rate_limited(data_source):
                        logger.warning(f"Rate limit exceeded for {data_source}")
                        continue
                    
                    # 获取数据
                    data = await self._fetch_data(request, data_source)
                    
                    # 记录成功
                    self.logger.log_request_end(
                        request.request_id, 
                        success=True,
                        record_count=len(data) if isinstance(data, pd.DataFrame) else 0,
                        data_source=data_source.value if hasattr(data_source, 'value') else str(data_source)
                    )
                    
                    # 增加请求计数
                    self.concurrency_controller.increment_request_count(data_source)
                    
                    return {
                        'success': True,
                        'data': data,
                        'data_source': data_source.value if hasattr(data_source, 'value') else str(data_source),
                        'request_id': request.request_id,
                        'record_count': len(data) if isinstance(data, pd.DataFrame) else 0,
                        'cache_hit': False  # 这里应该从实际的缓存系统获取
                    }
                    
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Failed to fetch data from {data_source}: {e}")
                    continue
            
            # 所有数据源都失败
            raise Exception(f"All data sources failed. Last error: {last_error}")
            
        except Exception as e:
            # 记录失败
            self.logger.log_request_end(request.request_id, success=False, error=str(e))
            
            return {
                'success': False,
                'error': str(e),
                'request_id': request.request_id
            }
            
        finally:
            # 释放并发许可
            self.concurrency_controller.release(request.request_id)
    
    async def _fetch_data(self, request: MarketDataRequest, data_source: DataSource) -> pd.DataFrame:
        """从指定数据源获取数据"""
        # 这里应该调用具体的数据适配器
        # 为了演示，返回模拟数据
        
        await asyncio.sleep(0.1)  # 模拟网络延迟
        
        # 根据数据类型返回不同的模拟数据
        if request.data_type == DataType.STOCK_REALTIME:
            return pd.DataFrame({
                'symbol': [request.symbol or '000001'],
                'price': [10.50],
                'change': [0.15],
                'change_pct': [1.45],
                'volume': [1000000],
                'timestamp': [datetime.now()]
            })
        elif request.data_type == DataType.DRAGON_TIGER:
            return pd.DataFrame({
                'trade_date': [request.trade_date or date.today()],
                'symbol': ['000001'],
                'name': ['平安银行'],
                'close_price': [10.50],
                'change_rate': [1.45],
                'net_buy_amount': [50000000]
            })
        else:
            return pd.DataFrame()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.logger.get_performance_stats()
    
    def get_active_requests(self) -> int:
        """获取活跃请求数"""
        return self.concurrency_controller.get_active_request_count()


# 全局实例
unified_request_interface = UnifiedDataRequestInterface()


async def process_market_data_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """处理市场数据请求的便捷函数"""
    try:
        request = MarketDataRequest(**request_data)
        return await unified_request_interface.process_request(request)
    except Exception as e:
        return {
            'success': False,
            'error': f"Request processing failed: {str(e)}"
        }


if __name__ == "__main__":
    # 测试代码
    async def test_unified_request():
        """测试统一请求接口"""
        
        # 测试股票实时数据请求
        request_data = {
            'data_type': 'stock_realtime',
            'symbol': '000001.SZ',
            'data_source': 'auto'
        }
        
        result = await process_market_data_request(request_data)
        print("Stock realtime data result:", result)
        
        # 测试龙虎榜数据请求
        request_data = {
            'data_type': 'dragon_tiger',
            'trade_date': '2024-01-01',
            'limit': 100
        }
        
        result = await process_market_data_request(request_data)
        print("Dragon tiger data result:", result)
        
        # 获取性能统计
        stats = unified_request_interface.get_performance_stats()
        print("Performance stats:", stats)
    
    # 运行测试
    asyncio.run(test_unified_request())