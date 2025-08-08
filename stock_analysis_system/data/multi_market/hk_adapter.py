"""
Hong Kong Stock Exchange Data Adapter

Provides data access for Hong Kong stocks with:
- Real-time and historical data from HKEX
- Support for H-shares, Red chips, and local stocks
- Currency handling (HKD)
- Trading hours and holiday calendar
"""

import logging
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import pytz
import yfinance as yf
from ..data_source_manager import DataSourceAdapter, DataSourceHealth, DataSourceStatus

logger = logging.getLogger(__name__)


@dataclass
class HKStockInfo:
    """Hong Kong stock information"""
    symbol: str
    name_en: str
    name_cn: str
    market: str  # Main, GEM
    sector: str
    industry: str
    listing_date: date
    currency: str = "HKD"
    lot_size: int = 100


class HongKongStockAdapter(DataSourceAdapter):
    """Data adapter for Hong Kong Stock Exchange"""
    
    def __init__(self):
        super().__init__()
        self.name = "Hong Kong Stock Exchange"
        self.timezone = pytz.timezone('Asia/Hong_Kong')
        self.currency = "HKD"
        
        # Trading hours (HKT)
        self.trading_hours = {
            'morning_open': '09:30',
            'morning_close': '12:00',
            'afternoon_open': '13:00',
            'afternoon_close': '16:00'
        }
        
        # Market holidays (simplified - in practice, use a proper calendar)
        self.holidays = [
            date(2024, 1, 1),   # New Year
            date(2024, 2, 10),  # Chinese New Year
            date(2024, 2, 12),  # Chinese New Year
            date(2024, 2, 13),  # Chinese New Year
            date(2024, 4, 4),   # Ching Ming Festival
            date(2024, 5, 1),   # Labour Day
            date(2024, 6, 10),  # Dragon Boat Festival
            date(2024, 7, 1),   # HKSAR Establishment Day
            date(2024, 9, 18),  # Mid-Autumn Festival
            date(2024, 10, 1),  # National Day
            date(2024, 10, 11), # Chung Yeung Festival
            date(2024, 12, 25), # Christmas Day
            date(2024, 12, 26), # Boxing Day
        ]
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize HK stock symbol"""
        # Remove leading zeros and add .HK suffix for yfinance
        if symbol.startswith('0'):
            symbol = symbol.lstrip('0')
        
        if not symbol.endswith('.HK'):
            symbol += '.HK'
        
        return symbol
    
    def _denormalize_symbol(self, symbol: str) -> str:
        """Convert back to HK format"""
        if symbol.endswith('.HK'):
            symbol = symbol[:-3]
        
        # Add leading zeros to make it 5 digits
        return symbol.zfill(5)
    
    async def get_stock_data(self, symbol: str, start_date: date, end_date: date, 
                           timeframe: str = 'daily') -> pd.DataFrame:
        """Get Hong Kong stock data"""
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            
            # Use yfinance for HK stocks
            ticker = yf.Ticker(normalized_symbol)
            
            if timeframe == 'daily':
                data = ticker.history(start=start_date, end=end_date)
            elif timeframe == '1h':
                data = ticker.history(start=start_date, end=end_date, interval='1h')
            elif timeframe == '5min':
                # yfinance limits intraday data to last 60 days
                if (end_date - start_date).days > 60:
                    end_date = start_date + timedelta(days=60)
                data = ticker.history(start=start_date, end=end_date, interval='5m')
            else:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            if data.empty:
                logger.warning(f"No data found for HK stock {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add metadata
            data['symbol'] = self._denormalize_symbol(symbol)
            data['currency'] = self.currency
            data['market'] = 'HKEX'
            data['timezone'] = 'Asia/Hong_Kong'
            
            # Convert timezone
            if data.index.tz is None:
                data.index = data.index.tz_localize(self.timezone)
            else:
                data.index = data.index.tz_convert(self.timezone)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching HK stock data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time Hong Kong stock data"""
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            ticker = yf.Ticker(normalized_symbol)
            
            # Get current info
            info = ticker.info
            
            # Get latest price data
            hist = ticker.history(period='1d', interval='1m')
            if hist.empty:
                return {}
            
            latest = hist.iloc[-1]
            
            return {
                'symbol': self._denormalize_symbol(symbol),
                'name': info.get('longName', ''),
                'price': float(latest['Close']),
                'change': float(latest['Close'] - hist.iloc[0]['Open']),
                'change_percent': float((latest['Close'] - hist.iloc[0]['Open']) / hist.iloc[0]['Open'] * 100),
                'volume': int(latest['Volume']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'open': float(hist.iloc[0]['Open']),
                'currency': self.currency,
                'market': 'HKEX',
                'timestamp': latest.name.isoformat(),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield')
            }
            
        except Exception as e:
            logger.error(f"Error fetching realtime HK data for {symbol}: {e}")
            return {}
    
    async def search_stocks(self, query: str, limit: int = 20) -> List[HKStockInfo]:
        """Search Hong Kong stocks"""
        # This is a simplified implementation
        # In practice, you'd use a proper HK stock database or API
        
        common_hk_stocks = [
            HKStockInfo("00700", "Tencent Holdings Ltd", "腾讯控股", "Main", "Technology", "Internet", date(2004, 6, 16)),
            HKStockInfo("00941", "China Mobile Ltd", "中国移动", "Main", "Telecommunications", "Mobile Telecom", date(1997, 10, 23)),
            HKStockInfo("00939", "China Construction Bank Corp", "中国建设银行", "Main", "Financial", "Banking", date(2005, 10, 27)),
            HKStockInfo("01299", "AIA Group Ltd", "友邦保险", "Main", "Financial", "Insurance", date(2010, 10, 29)),
            HKStockInfo("00005", "HSBC Holdings PLC", "汇丰控股", "Main", "Financial", "Banking", date(1991, 7, 8)),
            HKStockInfo("00388", "Hong Kong Exchanges & Clearing Ltd", "香港交易所", "Main", "Financial", "Exchange", date(2000, 6, 27)),
            HKStockInfo("02318", "Ping An Insurance Group Co of China Ltd", "中国平安", "Main", "Financial", "Insurance", date(2004, 6, 24)),
            HKStockInfo("01398", "Industrial & Commercial Bank of China Ltd", "工商银行", "Main", "Financial", "Banking", date(2006, 10, 27)),
            HKStockInfo("00883", "CNOOC Ltd", "中海油", "Main", "Energy", "Oil & Gas", date(2001, 2, 28)),
            HKStockInfo("03988", "Bank of China Ltd", "中国银行", "Main", "Financial", "Banking", date(2006, 6, 1)),
        ]
        
        # Simple search by symbol or name
        results = []
        query_lower = query.lower()
        
        for stock in common_hk_stocks:
            if (query_lower in stock.symbol.lower() or 
                query_lower in stock.name_en.lower() or 
                query_lower in stock.name_cn.lower()):
                results.append(stock)
                
                if len(results) >= limit:
                    break
        
        return results
    
    async def get_stock_info(self, symbol: str) -> Optional[HKStockInfo]:
        """Get detailed stock information"""
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            ticker = yf.Ticker(normalized_symbol)
            info = ticker.info
            
            return HKStockInfo(
                symbol=self._denormalize_symbol(symbol),
                name_en=info.get('longName', ''),
                name_cn=info.get('longName', ''),  # yfinance doesn't provide Chinese names
                market='Main',  # Simplified
                sector=info.get('sector', ''),
                industry=info.get('industry', ''),
                listing_date=date.today(),  # Placeholder
                currency=self.currency
            )
            
        except Exception as e:
            logger.error(f"Error fetching HK stock info for {symbol}: {e}")
            return None
    
    def is_trading_day(self, check_date: date) -> bool:
        """Check if a date is a trading day in Hong Kong"""
        # Check if it's a weekend
        if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's a holiday
        if check_date in self.holidays:
            return False
        
        return True
    
    def is_market_open(self, check_time: datetime = None) -> bool:
        """Check if Hong Kong market is currently open"""
        if check_time is None:
            check_time = datetime.now(self.timezone)
        else:
            check_time = check_time.astimezone(self.timezone)
        
        # Check if it's a trading day
        if not self.is_trading_day(check_time.date()):
            return False
        
        # Check trading hours
        current_time = check_time.time()
        
        # Morning session: 09:30 - 12:00
        morning_open = datetime.strptime(self.trading_hours['morning_open'], '%H:%M').time()
        morning_close = datetime.strptime(self.trading_hours['morning_close'], '%H:%M').time()
        
        # Afternoon session: 13:00 - 16:00
        afternoon_open = datetime.strptime(self.trading_hours['afternoon_open'], '%H:%M').time()
        afternoon_close = datetime.strptime(self.trading_hours['afternoon_close'], '%H:%M').time()
        
        return ((morning_open <= current_time <= morning_close) or 
                (afternoon_open <= current_time <= afternoon_close))
    
    def get_next_trading_day(self, from_date: date) -> date:
        """Get the next trading day after the given date"""
        next_date = from_date + timedelta(days=1)
        
        while not self.is_trading_day(next_date):
            next_date += timedelta(days=1)
        
        return next_date
    
    async def get_health_status(self) -> DataSourceHealth:
        """Get health status of Hong Kong data source"""
        try:
            # Test with a popular HK stock
            test_data = await self.get_realtime_data("00700")  # Tencent
            
            if test_data and 'price' in test_data:
                return DataSourceHealth(
                    source_type="HKEX",
                    status=DataSourceStatus.HEALTHY,
                    last_update=datetime.now(),
                    response_time=0.5,  # Placeholder
                    error_rate=0.0,
                    reliability_score=0.95
                )
            else:
                return DataSourceHealth(
                    source_type="HKEX",
                    status=DataSourceStatus.DEGRADED,
                    last_update=datetime.now(),
                    response_time=2.0,
                    error_rate=0.1,
                    reliability_score=0.7
                )
                
        except Exception as e:
            logger.error(f"HK data source health check failed: {e}")
            return DataSourceHealth(
                source_type="HKEX",
                status=DataSourceStatus.FAILED,
                last_update=datetime.now(),
                response_time=10.0,
                error_rate=1.0,
                reliability_score=0.0
            )
    
    async def get_market_calendar(self, year: int) -> List[date]:
        """Get Hong Kong market trading calendar for a year"""
        # Generate all weekdays in the year
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if self.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        return trading_days
    
    async def get_sector_data(self, sector: str) -> List[Dict[str, Any]]:
        """Get stocks in a specific sector"""
        # This would typically query a database or API
        # For now, return a simplified list
        
        sector_stocks = {
            'Technology': ['00700', '01024', '09988', '09618'],
            'Financial': ['00005', '01398', '00939', '02318', '01299'],
            'Energy': ['00883', '00386', '01088'],
            'Telecommunications': ['00941', '00762', '00728']
        }
        
        stocks = sector_stocks.get(sector, [])
        results = []
        
        for symbol in stocks:
            stock_info = await self.get_stock_info(symbol)
            if stock_info:
                results.append({
                    'symbol': stock_info.symbol,
                    'name': stock_info.name_en,
                    'sector': stock_info.sector,
                    'industry': stock_info.industry
                })
        
        return results