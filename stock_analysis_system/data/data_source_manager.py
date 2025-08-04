"""Data Source Manager with failover capabilities."""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import pandas as pd
import aiohttp
import time
from contextlib import asynccontextmanager

import akshare as ak
import tushare as ts
import os
import struct
from pathlib import Path

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DataSourceType(Enum):
    """Data source types."""
    LOCAL = "local"
    TUSHARE = "tushare"
    AKSHARE = "akshare"
    WIND = "wind"


class DataSourceStatus(Enum):
    """Data source status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class DataSourceHealth:
    """Data source health metrics."""
    source_type: DataSourceType
    status: DataSourceStatus
    last_success: Optional[datetime]
    last_failure: Optional[datetime]
    failure_count: int
    success_rate: float
    avg_response_time: float
    reliability_score: float


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int
    burst_limit: int
    cooldown_period: int  # seconds


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                return True
            return False
        else:  # half_open
            return True
    
    def on_success(self):
        """Record successful execution."""
        self.failure_count = 0
        self.state = "closed"
    
    def on_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).seconds
        return time_since_failure >= self.recovery_timeout


class RateLimiter:
    """Rate limiter implementation."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests = []
        self.burst_requests = []
    
    async def acquire(self):
        """Acquire rate limit permission."""
        now = time.time()
        
        # Clean old requests
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < 60]  # 1 minute window
        
        self.burst_requests = [req_time for req_time in self.burst_requests 
                              if now - req_time < self.config.cooldown_period]
        
        # Check rate limits
        if len(self.requests) >= self.config.requests_per_minute:
            wait_time = 60 - (now - self.requests[0])
            await asyncio.sleep(wait_time)
        
        if len(self.burst_requests) >= self.config.burst_limit:
            wait_time = self.config.cooldown_period - (now - self.burst_requests[0])
            await asyncio.sleep(wait_time)
        
        # Record request
        self.requests.append(now)
        self.burst_requests.append(now)


class BaseDataSource(ABC):
    """Base class for data sources."""
    
    def __init__(self, source_type: DataSourceType):
        self.source_type = source_type
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(RateLimitConfig(
            requests_per_minute=settings.data_sources.requests_per_minute,
            burst_limit=10,
            cooldown_period=5
        ))
        self.health = DataSourceHealth(
            source_type=source_type,
            status=DataSourceStatus.HEALTHY,
            last_success=None,
            last_failure=None,
            failure_count=0,
            success_rate=1.0,
            avg_response_time=0.0,
            reliability_score=1.0
        )
    
    @abstractmethod
    async def get_stock_data(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Get stock daily data."""
        pass
    
    async def get_intraday_data(self, symbol: str, start_date: date, end_date: date, timeframe: str = '5min') -> pd.DataFrame:
        """Get intraday stock data. Default implementation returns empty DataFrame."""
        logger.warning(f"Intraday data not supported for {self.source_type.value}")
        return pd.DataFrame()
    
    @abstractmethod
    async def get_stock_list(self) -> pd.DataFrame:
        """Get list of available stocks."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test data source connection."""
        pass
    
    async def execute_with_circuit_breaker(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if not self.circuit_breaker.can_execute():
            raise Exception(f"Circuit breaker open for {self.source_type.value}")
        
        try:
            await self.rate_limiter.acquire()
            start_time = time.time()
            
            result = await func(*args, **kwargs)
            
            # Record success
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            self.circuit_breaker.on_success()
            
            return result
            
        except Exception as e:
            self._record_failure()
            self.circuit_breaker.on_failure()
            raise e
    
    def _record_success(self, execution_time: float):
        """Record successful execution."""
        self.health.last_success = datetime.now()
        self.health.avg_response_time = (
            (self.health.avg_response_time * 0.9) + (execution_time * 0.1)
        )
        self._update_reliability_score()
    
    def _record_failure(self):
        """Record failed execution."""
        self.health.last_failure = datetime.now()
        self.health.failure_count += 1
        self._update_reliability_score()
    
    def _update_reliability_score(self):
        """Update reliability score based on recent performance."""
        total_requests = self.health.failure_count + 100  # Assume some successful requests
        success_requests = total_requests - self.health.failure_count
        self.health.success_rate = success_requests / total_requests
        
        # Factor in recency of failures
        recency_factor = 1.0
        if self.health.last_failure:
            hours_since_failure = (datetime.now() - self.health.last_failure).seconds / 3600
            recency_factor = min(1.0, hours_since_failure / 24)  # Recover over 24 hours
        
        self.health.reliability_score = self.health.success_rate * recency_factor
        
        # Update status
        if self.health.reliability_score > 0.9:
            self.health.status = DataSourceStatus.HEALTHY
        elif self.health.reliability_score > 0.5:
            self.health.status = DataSourceStatus.DEGRADED
        else:
            self.health.status = DataSourceStatus.FAILED


class TushareDataSource(BaseDataSource):
    """Tushare data source implementation."""
    
    def __init__(self):
        super().__init__(DataSourceType.TUSHARE)
        self.api = None
        if settings.data_sources.tushare_token:
            ts.set_token(settings.data_sources.tushare_token)
            self.api = ts.pro_api()
    
    async def get_stock_data(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Get stock daily data from Tushare."""
        if not self.api:
            raise Exception("Tushare token not configured")
        
        async def _fetch_data():
            # Convert symbol format (e.g., "000001.SZ" -> "000001.SZ")
            ts_symbol = symbol
            
            df = self.api.daily(
                ts_code=ts_symbol,
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d')
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # Standardize column names
            df = df.rename(columns={
                'ts_code': 'stock_code',
                'trade_date': 'trade_date',
                'open': 'open_price',
                'high': 'high_price',
                'low': 'low_price',
                'close': 'close_price',
                'vol': 'volume',
                'amount': 'amount'
            })
            
            # Convert trade_date to datetime
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            
            return df
        
        return await self.execute_with_circuit_breaker(_fetch_data)
    
    async def get_stock_list(self) -> pd.DataFrame:
        """Get list of available stocks from Tushare."""
        if not self.api:
            raise Exception("Tushare token not configured")
        
        async def _fetch_stock_list():
            df = self.api.stock_basic(exchange='', list_status='L')
            return df.rename(columns={
                'ts_code': 'stock_code',
                'symbol': 'symbol',
                'name': 'name',
                'area': 'area',
                'industry': 'industry',
                'market': 'market'
            })
        
        return await self.execute_with_circuit_breaker(_fetch_stock_list)
    
    async def test_connection(self) -> bool:
        """Test Tushare connection."""
        try:
            if not self.api:
                return False
            
            # Try to fetch a small amount of data
            df = self.api.daily(ts_code='000001.SZ', start_date='20240101', end_date='20240102')
            return True
        except Exception as e:
            logger.error(f"Tushare connection test failed: {e}")
            return False


class LocalDataSource(BaseDataSource):
    """Local data source implementation for TDX format files."""
    
    def __init__(self, base_path: str = None):
        super().__init__(DataSourceType.LOCAL)
        if base_path is None:
            self.base_path = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/vipdoc")
        else:
            self.base_path = base_path
        
        # Check if base path exists
        if not os.path.exists(self.base_path):
            logger.warning(f"Local data path does not exist: {self.base_path}")
    
    async def get_stock_data(self, symbol: str, start_date: date, end_date: date, timeframe: str = 'daily') -> pd.DataFrame:
        """Get stock data from local TDX files.
        
        Args:
            symbol: Stock symbol (e.g., '000001.SZ')
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe ('daily', '5min', '15min', '30min', '60min')
        """
        
        async def _fetch_data():
            try:
                # Convert symbol format (e.g., "000001.SZ" -> "sz000001")
                if symbol.endswith('.SZ'):
                    market = 'sz'
                    stock_code = symbol[:6]
                elif symbol.endswith('.SH'):
                    market = 'sh'  
                    stock_code = symbol[:6]
                else:
                    # Assume it's already in the correct format
                    if symbol.startswith(('sz', 'sh')):
                        market = symbol[:2]
                        stock_code = symbol[2:]
                    else:
                        # Default to SZ market for 6-digit codes starting with 0, 3
                        if symbol.startswith(('0', '3')):
                            market = 'sz'
                        else:
                            market = 'sh'
                        stock_code = symbol
                
                # Handle different timeframes
                if timeframe == 'daily':
                    # Build file path for daily data
                    data_file = os.path.join(self.base_path, market, 'lday', f'{market}{stock_code}.day')
                    
                    if not os.path.exists(data_file):
                        logger.warning(f"Local daily data file not found: {data_file}")
                        return pd.DataFrame()
                    
                    # Read daily data
                    data = self._get_daily_data(data_file)
                    
                elif timeframe == '5min':
                    # Build file path for 5-minute data
                    data_file = os.path.join(self.base_path, market, 'fzline', f'{market}{stock_code}.lc5')
                    
                    if not os.path.exists(data_file):
                        logger.warning(f"Local 5min data file not found: {data_file}")
                        return pd.DataFrame()
                    
                    # Read 5-minute data
                    data = self._get_5min_data(data_file)
                    
                elif timeframe in ['15min', '30min', '60min']:
                    # For other timeframes, read 5min data and resample
                    data_file = os.path.join(self.base_path, market, 'fzline', f'{market}{stock_code}.lc5')
                    
                    if not os.path.exists(data_file):
                        logger.warning(f"Local 5min data file not found for resampling: {data_file}")
                        return pd.DataFrame()
                    
                    # Read 5-minute data and resample
                    min5_data = self._get_5min_data(data_file)
                    if min5_data is None or min5_data.empty:
                        return pd.DataFrame()
                    
                    data = self._resample_5min_data(min5_data, timeframe)
                    
                else:
                    logger.error(f"Unsupported timeframe: {timeframe}")
                    return pd.DataFrame()
                
                if data is None or data.empty:
                    return pd.DataFrame()
                
                if data is None or data.empty:
                    return pd.DataFrame()
                
                # Filter by date range
                if timeframe == 'daily':
                    # For daily data, filter by date
                    data = data[
                        (data['date'].dt.date >= start_date) & 
                        (data['date'].dt.date <= end_date)
                    ]
                    
                    # Standardize column names to match system format
                    data = data.rename(columns={
                        'date': 'trade_date',
                        'open': 'open_price',
                        'high': 'high_price', 
                        'low': 'low_price',
                        'close': 'close_price'
                    })
                    
                    # Sort by trade_date
                    data = data.sort_values('trade_date').reset_index(drop=True)
                    
                else:
                    # For intraday data, filter by datetime
                    start_datetime = pd.Timestamp.combine(start_date, pd.Timestamp.min.time())
                    end_datetime = pd.Timestamp.combine(end_date, pd.Timestamp.max.time())
                    
                    data = data[
                        (data['datetime'] >= start_datetime) & 
                        (data['datetime'] <= end_datetime)
                    ]
                    
                    # For intraday data, keep datetime column and add trade_date
                    data['trade_date'] = data['datetime'].dt.date
                    
                    # Sort by datetime
                    data = data.sort_values('datetime').reset_index(drop=True)
                
                # Add stock_code column
                data['stock_code'] = symbol
                
                # Ensure amount column exists
                if 'amount' not in data.columns and 'volume' in data.columns:
                    if timeframe == 'daily':
                        avg_price = (data['high_price'] + data['low_price']) / 2
                        data['amount'] = data['volume'] * avg_price
                    else:
                        avg_price = (data['high'] + data['low']) / 2
                        data['amount'] = data['volume'] * avg_price
                
                return data
                
            except Exception as e:
                logger.error(f"Failed to load local data for {symbol}: {e}")
                return pd.DataFrame()
        
        return await self.execute_with_circuit_breaker(_fetch_data)
    
    def _get_daily_data(self, file_path: str) -> pd.DataFrame:
        """Read daily data from .day file."""
        data = []
        record_size = 32
        unpack_format = '<IIIIIfI'
        unpack_size = struct.calcsize(unpack_format)
        
        try:
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(record_size)
                    if len(chunk) < record_size:
                        break
                    
                    try:
                        # Unpack the data
                        date, open_p, high_p, low_p, close_p, amount, volume = struct.unpack(
                            unpack_format, chunk[:unpack_size]
                        )
                        
                        # Convert prices (divide by 100)
                        open_p, high_p, low_p, close_p = (
                            open_p / 100, high_p / 100, low_p / 100, close_p / 100
                        )
                        
                        # Skip invalid data
                        if open_p <= 0:
                            continue
                        
                        # Parse date
                        date_str = str(date)
                        if len(date_str) == 8:
                            parsed_date = datetime.strptime(date_str, '%Y%m%d')
                        else:
                            continue
                        
                        data.append({
                            'date': parsed_date,
                            'open': open_p,
                            'high': high_p,
                            'low': low_p,
                            'close': close_p,
                            'volume': volume,
                            'amount': amount
                        })
                        
                    except (struct.error, ValueError) as e:
                        continue
            
            if not data:
                return pd.DataFrame()
            
            return pd.DataFrame(data).sort_values('date').reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error reading daily data file {file_path}: {e}")
            return pd.DataFrame()
    
    def _get_5min_data(self, file_path: str) -> pd.DataFrame:
        """Read 5-minute data from .lc5 file."""
        data = []
        record_size = 32
        unpack_format = '<HHffffff'
        unpack_size = struct.calcsize(unpack_format)
        
        try:
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(record_size)
                    if len(chunk) < record_size:
                        break
                    
                    try:
                        # Unpack the data
                        packed_date, packed_time, open_p, high_p, low_p, close_p, volume, amount = struct.unpack(
                            unpack_format, chunk[:unpack_size]
                        )
                        
                        # Skip invalid data
                        if open_p <= 0:
                            continue
                        
                        # Decode date
                        year = packed_date // 2048 + 2004
                        month = (packed_date % 2048) // 100
                        day = packed_date % 100
                        
                        # Decode time
                        hour = packed_time // 60
                        minute = packed_time % 60
                        
                        # Validate date and time
                        if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
                            continue
                        
                        try:
                            # Create datetime object
                            dt = datetime(year, month, day, hour, minute)
                        except ValueError:
                            continue
                        
                        data.append({
                            'datetime': dt,
                            'open': open_p,
                            'high': high_p,
                            'low': low_p,
                            'close': close_p,
                            'volume': volume,
                            'amount': amount
                        })
                        
                    except (struct.error, ValueError):
                        continue
            
            if not data:
                return pd.DataFrame()
            
            return pd.DataFrame(data).sort_values('datetime').reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error reading 5min data file {file_path}: {e}")
            return pd.DataFrame()
    
    def _resample_5min_data(self, df_5min: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample 5-minute data to other timeframes."""
        if df_5min is None or df_5min.empty:
            return pd.DataFrame()
        
        try:
            # Set datetime as index
            df_resampled = df_5min.set_index('datetime')
            
            # Define resampling rules
            resample_rules = {
                '15min': '15min',
                '30min': '30min', 
                '60min': '60min'
            }
            
            if timeframe not in resample_rules:
                logger.error(f"Unsupported timeframe for resampling: {timeframe}")
                return pd.DataFrame()
            
            # Resample the data
            resampled = df_resampled.resample(resample_rules[timeframe]).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'amount': 'sum'
            }).dropna()
            
            # Reset index to get datetime back as column
            resampled.reset_index(inplace=True)
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling 5min data to {timeframe}: {e}")
            return pd.DataFrame()
    
    async def get_intraday_data(self, symbol: str, start_date: date, end_date: date, timeframe: str = '5min') -> pd.DataFrame:
        """Get intraday stock data from local TDX files."""
        return await self.get_stock_data(symbol, start_date, end_date, timeframe)
    
    async def get_stock_list(self) -> pd.DataFrame:
        """Get list of available stocks from local files."""
        
        async def _fetch_stock_list():
            try:
                stocks = []
                
                # Check both SZ and SH markets
                for market in ['sz', 'sh']:
                    market_path = os.path.join(self.base_path, market, 'lday')
                    
                    if not os.path.exists(market_path):
                        continue
                    
                    # List all .day files
                    for file_name in os.listdir(market_path):
                        if file_name.endswith('.day'):
                            # Extract stock code
                            stock_code = file_name[:-4]  # Remove .day extension
                            
                            # Convert to standard format
                            if stock_code.startswith('sz'):
                                symbol = stock_code[2:] + '.SZ'
                                name = f"股票{stock_code[2:]}"
                            elif stock_code.startswith('sh'):
                                symbol = stock_code[2:] + '.SH'
                                name = f"股票{stock_code[2:]}"
                            else:
                                continue
                            
                            stocks.append({
                                'stock_code': symbol,
                                'symbol': stock_code[2:] if stock_code.startswith(('sz', 'sh')) else stock_code,
                                'name': name,
                                'market': market.upper()
                            })
                
                if not stocks:
                    logger.warning("No local stock data files found")
                    return pd.DataFrame()
                
                return pd.DataFrame(stocks)
                
            except Exception as e:
                logger.error(f"Failed to get local stock list: {e}")
                return pd.DataFrame()
        
        return await self.execute_with_circuit_breaker(_fetch_stock_list)
    
    async def test_connection(self) -> bool:
        """Test local data source connection."""
        try:
            # Check if base path exists
            if not os.path.exists(self.base_path):
                return False
            
            # Check if we have any market directories
            for market in ['sz', 'sh']:
                market_path = os.path.join(self.base_path, market, 'lday')
                if os.path.exists(market_path):
                    # Check if we have any .day files
                    day_files = [f for f in os.listdir(market_path) if f.endswith('.day')]
                    if day_files:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Local data source connection test failed: {e}")
            return False


class AkshareDataSource(BaseDataSource):
    """AkShare data source implementation."""
    
    def __init__(self):
        super().__init__(DataSourceType.AKSHARE)
    
    async def get_stock_data(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Get stock daily data from AkShare."""
        
        async def _fetch_data():
            # Convert symbol format for AkShare (e.g., "000001.SZ" -> "sz000001")
            if symbol.endswith('.SZ'):
                ak_symbol = 'sz' + symbol[:6]
            elif symbol.endswith('.SH'):
                ak_symbol = 'sh' + symbol[:6]
            else:
                ak_symbol = symbol
            
            try:
                df = ak.stock_zh_a_hist(
                    symbol=ak_symbol,
                    period="daily",
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    adjust="qfq"  # Forward adjusted
                )
                
                if df.empty:
                    return pd.DataFrame()
                
                # Standardize column names
                df = df.rename(columns={
                    '日期': 'trade_date',
                    '开盘': 'open_price',
                    '最高': 'high_price',
                    '最低': 'low_price',
                    '收盘': 'close_price',
                    '成交量': 'volume',
                    '成交额': 'amount'
                })
                
                df['stock_code'] = symbol
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                
                return df
                
            except Exception as e:
                logger.error(f"AkShare data fetch failed for {symbol}: {e}")
                return pd.DataFrame()
        
        return await self.execute_with_circuit_breaker(_fetch_data)
    
    async def get_stock_list(self) -> pd.DataFrame:
        """Get list of available stocks from AkShare."""
        
        async def _fetch_stock_list():
            try:
                # Get A-share stock list
                df = ak.stock_info_a_code_name()
                
                # Standardize format
                df['stock_code'] = df['code'].apply(lambda x: 
                    f"{x}.SH" if x.startswith(('60', '68', '90')) else f"{x}.SZ"
                )
                
                return df.rename(columns={
                    'code': 'symbol',
                    'name': 'name'
                })
                
            except Exception as e:
                logger.error(f"AkShare stock list fetch failed: {e}")
                return pd.DataFrame()
        
        return await self.execute_with_circuit_breaker(_fetch_stock_list)
    
    async def test_connection(self) -> bool:
        """Test AkShare connection."""
        try:
            # Try to fetch a small amount of data
            df = ak.stock_zh_a_hist(
                symbol="sz000001",
                period="daily",
                start_date="20240101",
                end_date="20240102"
            )
            return not df.empty
        except Exception as e:
            logger.error(f"AkShare connection test failed: {e}")
            return False


class DataSourceManager:
    """Main data source manager with failover capabilities."""
    
    def __init__(self):
        self.data_sources: Dict[DataSourceType, BaseDataSource] = {}
        self.primary_source = DataSourceType.LOCAL
        self.fallback_order = [DataSourceType.LOCAL, DataSourceType.AKSHARE, DataSourceType.TUSHARE]
        
        # Initialize data sources
        self._initialize_data_sources()
    
    def _initialize_data_sources(self):
        """Initialize all available data sources."""
        try:
            self.data_sources[DataSourceType.LOCAL] = LocalDataSource()
            logger.info("Local data source initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Local data source: {e}")
        
        try:
            self.data_sources[DataSourceType.AKSHARE] = AkshareDataSource()
            logger.info("AkShare data source initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AkShare: {e}")
            
        try:
            self.data_sources[DataSourceType.TUSHARE] = TushareDataSource()
            logger.info("Tushare data source initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Tushare: {e}")
    
    async def get_stock_data(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Get stock data with automatic failover."""
        last_exception = None
        
        for source_type in self.fallback_order:
            if source_type not in self.data_sources:
                continue
            
            data_source = self.data_sources[source_type]
            
            # Skip if source is in failed state
            if data_source.health.status == DataSourceStatus.FAILED:
                logger.warning(f"Skipping {source_type.value} - marked as failed")
                continue
            
            try:
                logger.info(f"Attempting to fetch data from {source_type.value}")
                data = await data_source.get_stock_data(symbol, start_date, end_date)
                
                if not data.empty:
                    logger.info(f"Successfully fetched data from {source_type.value}")
                    return data
                else:
                    logger.warning(f"No data returned from {source_type.value}")
                    
            except Exception as e:
                logger.error(f"Failed to fetch data from {source_type.value}: {e}")
                last_exception = e
                continue
        
        # If all sources failed, raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise Exception("No data sources available")
    
    async def get_stock_list(self) -> pd.DataFrame:
        """Get stock list with automatic failover."""
        last_exception = None
        
        for source_type in self.fallback_order:
            if source_type not in self.data_sources:
                continue
            
            data_source = self.data_sources[source_type]
            
            if data_source.health.status == DataSourceStatus.FAILED:
                continue
            
            try:
                logger.info(f"Attempting to fetch stock list from {source_type.value}")
                data = await data_source.get_stock_list()
                
                if not data.empty:
                    logger.info(f"Successfully fetched stock list from {source_type.value}")
                    return data
                    
            except Exception as e:
                logger.error(f"Failed to fetch stock list from {source_type.value}: {e}")
                last_exception = e
                continue
        
        if last_exception:
            raise last_exception
        else:
            raise Exception("No data sources available")
    
    async def get_intraday_data(self, symbol: str, start_date: date, end_date: date, timeframe: str = '5min') -> pd.DataFrame:
        """Get intraday data with automatic failover."""
        last_exception = None
        
        for source_type in self.fallback_order:
            if source_type not in self.data_sources:
                continue
            
            data_source = self.data_sources[source_type]
            
            # Skip if source is in failed state
            if data_source.health.status == DataSourceStatus.FAILED:
                logger.warning(f"Skipping {source_type.value} - marked as failed")
                continue
            
            try:
                logger.info(f"Attempting to fetch intraday data from {source_type.value}")
                data = await data_source.get_intraday_data(symbol, start_date, end_date, timeframe)
                
                if not data.empty:
                    logger.info(f"Successfully fetched intraday data from {source_type.value}")
                    return data
                else:
                    logger.warning(f"No intraday data returned from {source_type.value}")
                    
            except Exception as e:
                logger.error(f"Failed to fetch intraday data from {source_type.value}: {e}")
                last_exception = e
                continue
        
        # If all sources failed, raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise Exception("No data sources available for intraday data")
    
    async def health_check(self) -> Dict[DataSourceType, DataSourceHealth]:
        """Perform health check on all data sources."""
        health_status = {}
        
        for source_type, data_source in self.data_sources.items():
            try:
                is_healthy = await data_source.test_connection()
                if is_healthy:
                    data_source.health.status = DataSourceStatus.HEALTHY
                else:
                    data_source.health.status = DataSourceStatus.DEGRADED
            except Exception as e:
                logger.error(f"Health check failed for {source_type.value}: {e}")
                data_source.health.status = DataSourceStatus.FAILED
            
            health_status[source_type] = data_source.health
        
        return health_status
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of data source health."""
        summary = {
            "total_sources": len(self.data_sources),
            "healthy_sources": 0,
            "degraded_sources": 0,
            "failed_sources": 0,
            "primary_source": self.primary_source.value,
            "sources": {}
        }
        
        for source_type, data_source in self.data_sources.items():
            status = data_source.health.status
            summary["sources"][source_type.value] = {
                "status": status.value,
                "reliability_score": data_source.health.reliability_score,
                "last_success": data_source.health.last_success,
                "last_failure": data_source.health.last_failure
            }
            
            if status == DataSourceStatus.HEALTHY:
                summary["healthy_sources"] += 1
            elif status == DataSourceStatus.DEGRADED:
                summary["degraded_sources"] += 1
            else:
                summary["failed_sources"] += 1
        
        return summary


# Global instance
data_source_manager = DataSourceManager()


async def get_data_source_manager() -> DataSourceManager:
    """Get the global data source manager instance."""
    return data_source_manager