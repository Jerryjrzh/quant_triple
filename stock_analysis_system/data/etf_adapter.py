"""
ETF数据适配器

基于 tmp/core/crawling/fund_etf_em.py 创建的ETF数据适配器，
提供ETF实时行情和历史数据获取的统一接口。

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import asyncio
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from functools import lru_cache
import pandas as pd
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ETFRequest:
    """ETF请求结构"""
    symbol: str = ""  # ETF代码
    data_type: str = "realtime"  # realtime, history, intraday
    period: str = "daily"  # daily, weekly, monthly (for history) or 1,5,15,30,60 (for intraday)
    start_date: str = ""
    end_date: str = ""
    adjust: str = ""  # "", "qfq", "hfq"


@dataclass
class ETFResponse:
    """ETF响应结构"""
    success: bool
    data: pd.DataFrame
    error_message: str = ""
    response_time: float = 0.0
    data_source: str = "eastmoney_etf"
    timestamp: datetime = None
    data_type: str = ""
    symbol: str = ""


class ETFAdapter:
    """ETF数据适配器"""
    
    def __init__(self, timeout: int = 15, max_retries: int = 3):
        """
        初始化适配器
        
        Args:
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # API端点
        self.endpoints = {
            'realtime': "http://88.push2.eastmoney.com/api/qt/clist/get",
            'history': "http://push2his.eastmoney.com/api/qt/stock/kline/get",
            'intraday_1min': "https://push2his.eastmoney.com/api/qt/stock/trends2/get",
            'code_map': "http://88.push2.eastmoney.com/api/qt/clist/get"
        }
        
        # 参数映射
        self.adjust_dict = {"qfq": "1", "hfq": "2", "": "0"}
        self.period_dict = {"daily": "101", "weekly": "102", "monthly": "103"}
        
        # 缓存ETF代码映射
        self._etf_code_cache = None
        self._cache_timestamp = None
        self._cache_ttl = 3600  # 缓存1小时 
       
        # 错误统计
        self.error_stats = {
            'network_errors': 0,
            'data_format_errors': 0,
            'api_errors': 0,
            'total_requests': 0,
            'successful_requests': 0
        }
    
    async def get_etf_realtime_data(self, symbols: Optional[List[str]] = None) -> ETFResponse:
        """
        获取ETF实时行情数据
        
        Args:
            symbols: ETF代码列表，为空时获取全市场ETF数据
            
        Returns:
            ETF响应对象
        """
        start_time = time.time()
        self.error_stats['total_requests'] += 1
        
        try:
            url = self.endpoints['realtime']
            page_size = 5000 if symbols is None else min(len(symbols), 5000)
            
            params = {
                "pn": 1,
                "pz": page_size,
                "po": "1",
                "np": "1",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": "2",
                "invt": "2",
                "wbp2u": "|0|0|0|web",
                "fid": "f12",
                "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024",
                "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152",
                "_": str(int(time.time() * 1000)),
            }
            
            # 获取所有分页数据
            all_data = await self._fetch_all_pages(url, params)
            if not all_data:
                return ETFResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message="未获取到ETF实时数据",
                    response_time=time.time() - start_time,
                    data_type="realtime"
                )
            
            # 处理数据
            df = self._process_realtime_data(all_data)
            
            # 过滤指定ETF
            if symbols:
                df = df[df['代码'].isin(symbols)]
            
            self.error_stats['successful_requests'] += 1
            
            return ETFResponse(
                success=True,
                data=df,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                data_type="realtime"
            )
            
        except Exception as e:
            logger.error(f"获取ETF实时数据失败: {e}")
            self.error_stats['api_errors'] += 1
            return ETFResponse(
                success=False,
                data=pd.DataFrame(),
                error_message=str(e),
                response_time=time.time() - start_time,
                data_type="realtime"
            )   
 
    async def get_etf_history_data(self, request: ETFRequest) -> ETFResponse:
        """
        获取ETF历史数据
        
        Args:
            request: ETF请求对象
            
        Returns:
            ETF响应对象
        """
        start_time = time.time()
        self.error_stats['total_requests'] += 1
        
        try:
            # 获取ETF代码映射
            code_id_map = await self._get_etf_code_id_map()
            if request.symbol not in code_id_map:
                return ETFResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message=f"ETF代码 {request.symbol} 未找到",
                    response_time=time.time() - start_time,
                    data_type="history",
                    symbol=request.symbol
                )
            
            url = self.endpoints['history']
            params = {
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
                "ut": "7eea3edcaed734bea9cbfc24409ed989",
                "klt": self.period_dict.get(request.period, "101"),
                "fqt": self.adjust_dict.get(request.adjust, "0"),
                "secid": f"{code_id_map[request.symbol]}.{request.symbol}",
                "beg": request.start_date or "19700101",
                "end": request.end_date or "20500101",
                "_": str(int(time.time() * 1000)),
            }
            
            response_data = await self._make_request(url, params)
            if not response_data:
                return ETFResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message="未获取到响应数据",
                    response_time=time.time() - start_time,
                    data_type="history",
                    symbol=request.symbol
                )
            
            # 检查数据
            if not (response_data.get("data") and response_data["data"].get("klines")):
                return ETFResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message="未获取到K线数据",
                    response_time=time.time() - start_time,
                    data_type="history",
                    symbol=request.symbol
                )
            
            # 处理数据
            df = self._process_history_data(response_data["data"]["klines"])
            
            self.error_stats['successful_requests'] += 1
            
            return ETFResponse(
                success=True,
                data=df,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                data_type="history",
                symbol=request.symbol
            )
            
        except Exception as e:
            logger.error(f"获取ETF历史数据失败: {e}")
            self.error_stats['api_errors'] += 1
            return ETFResponse(
                success=False,
                data=pd.DataFrame(),
                error_message=str(e),
                response_time=time.time() - start_time,
                data_type="history",
                symbol=request.symbol
            ) 
   
    async def get_etf_intraday_data(self, request: ETFRequest) -> ETFResponse:
        """
        获取ETF分时数据
        
        Args:
            request: ETF请求对象，period支持 '1', '5', '15', '30', '60'
            
        Returns:
            ETF响应对象
        """
        start_time = time.time()
        self.error_stats['total_requests'] += 1
        
        try:
            # 获取ETF代码映射
            code_id_map = await self._get_etf_code_id_map()
            if request.symbol not in code_id_map:
                return ETFResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message=f"ETF代码 {request.symbol} 未找到",
                    response_time=time.time() - start_time,
                    data_type="intraday",
                    symbol=request.symbol
                )
            
            if request.period == "1":
                # 1分钟数据使用不同的API
                url = self.endpoints['intraday_1min']
                params = {
                    "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
                    "ut": "7eea3edcaed734bea9cbfc24409ed989",
                    "ndays": "5",
                    "iscr": "0",
                    "secid": f"{code_id_map[request.symbol]}.{request.symbol}",
                    "_": str(int(time.time() * 1000)),
                }
                
                response_data = await self._make_request(url, params)
                if not response_data or not response_data.get("data", {}).get("trends"):
                    return ETFResponse(
                        success=False,
                        data=pd.DataFrame(),
                        error_message="未获取到分时数据",
                        response_time=time.time() - start_time,
                        data_type="intraday",
                        symbol=request.symbol
                    )
                
                df = self._process_intraday_1min_data(response_data["data"]["trends"])
                
            else:
                # 其他周期使用K线API
                url = self.endpoints['history']
                params = {
                    "fields1": "f1,f2,f3,f4,f5,f6",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                    "ut": "7eea3edcaed734bea9cbfc24409ed989",
                    "klt": request.period,
                    "fqt": self.adjust_dict.get(request.adjust, "0"),
                    "secid": f"{code_id_map[request.symbol]}.{request.symbol}",
                    "beg": "0",
                    "end": "20500000",
                    "_": str(int(time.time() * 1000)),
                }
                
                response_data = await self._make_request(url, params)
                if not response_data or not response_data.get("data", {}).get("klines"):
                    return ETFResponse(
                        success=False,
                        data=pd.DataFrame(),
                        error_message="未获取到K线数据",
                        response_time=time.time() - start_time,
                        data_type="intraday",
                        symbol=request.symbol
                    )
                
                df = self._process_intraday_kline_data(response_data["data"]["klines"])
            
            # 应用时间过滤
            if request.start_date or request.end_date:
                df = self._filter_by_date_range(df, request.start_date, request.end_date)
            
            self.error_stats['successful_requests'] += 1
            
            return ETFResponse(
                success=True,
                data=df,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                data_type="intraday",
                symbol=request.symbol
            )
            
        except Exception as e:
            logger.error(f"获取ETF分时数据失败: {e}")
            self.error_stats['api_errors'] += 1
            return ETFResponse(
                success=False,
                data=pd.DataFrame(),
                error_message=str(e),
                response_time=time.time() - start_time,
                data_type="intraday",
                symbol=request.symbol
            )  
  
    async def _get_etf_code_id_map(self) -> Dict[str, str]:
        """获取ETF代码和市场ID映射（带缓存）"""
        current_time = time.time()
        
        # 检查缓存
        if (self._etf_code_cache is not None and 
            self._cache_timestamp is not None and 
            current_time - self._cache_timestamp < self._cache_ttl):
            return self._etf_code_cache
        
        try:
            url = self.endpoints['code_map']
            params = {
                "pn": "1",
                "pz": "5000",
                "po": "1",
                "np": "1",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": "2",
                "invt": "2",
                "wbp2u": "|0|0|0|web",
                "fid": "f3",
                "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024",
                "fields": "f12,f13",
                "_": str(int(time.time() * 1000)),
            }
            
            response_data = await self._make_request(url, params)
            if not response_data or not response_data.get("data", {}).get("diff"):
                return self._etf_code_cache or {}
            
            data = response_data["data"]["diff"]
            code_id_dict = {item["f12"]: item["f13"] for item in data}
            
            # 更新缓存
            self._etf_code_cache = code_id_dict
            self._cache_timestamp = current_time
            
            logger.info(f"更新ETF代码映射缓存，共 {len(code_id_dict)} 个ETF")
            return code_id_dict
            
        except Exception as e:
            logger.error(f"获取ETF代码映射失败: {e}")
            return self._etf_code_cache or {}
    
    async def _make_request(self, url: str, params: Dict) -> Optional[Dict]:
        """发送HTTP请求（带重试机制）"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                self.error_stats['network_errors'] += 1
                logger.warning(f"请求超时，尝试 {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
            except requests.exceptions.RequestException as e:
                self.error_stats['network_errors'] += 1
                logger.error(f"网络请求失败: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
            except ValueError as e:
                self.error_stats['data_format_errors'] += 1
                logger.error(f"JSON解析失败: {e}")
                break
                
        return None
    
    async def _fetch_all_pages(self, url: str, base_params: Dict) -> List[Dict]:
        """获取所有分页数据"""
        all_data = []
        
        # 获取第一页数据
        response_data = await self._make_request(url, base_params)
        if not response_data or not response_data.get("data", {}).get("diff"):
            return all_data
        
        first_page_data = response_data["data"]["diff"]
        all_data.extend(first_page_data)
        
        # 获取总数据量和页数
        data_count = response_data["data"].get("total", 0)
        page_size = base_params.get("pz", 50)
        page_count = math.ceil(data_count / page_size)
        
        # 获取其他页面数据
        for page in range(2, page_count + 1):
            params = base_params.copy()
            params["pn"] = page
            
            response_data = await self._make_request(url, params)
            if response_data and response_data.get("data", {}).get("diff"):
                all_data.extend(response_data["data"]["diff"])
            
            # 避免请求过快
            await asyncio.sleep(0.1)
        
        return all_data 
   
    def _process_realtime_data(self, data: List[Dict]) -> pd.DataFrame:
        """处理ETF实时数据"""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # 重命名列
        df.rename(columns={
            "f12": "代码",
            "f14": "名称",
            "f2": "最新价",
            "f3": "涨跌幅",
            "f4": "涨跌额",
            "f5": "成交量",
            "f6": "成交额",
            "f17": "开盘价",
            "f15": "最高价",
            "f16": "最低价",
            "f18": "昨收",
            "f8": "换手率",
            "f21": "流通市值",
            "f20": "总市值",
        }, inplace=True)
        
        # 选择需要的列
        df = df[[
            "代码", "名称", "最新价", "涨跌幅", "涨跌额", "成交量", "成交额",
            "开盘价", "最高价", "最低价", "昨收", "换手率", "流通市值", "总市值"
        ]]
        
        # 数据类型转换
        numeric_cols = [
            "最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "开盘价",
            "最高价", "最低价", "昨收", "换手率", "流通市值", "总市值"
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    def _process_history_data(self, klines: List[str]) -> pd.DataFrame:
        """处理ETF历史数据"""
        if not klines:
            return pd.DataFrame()
        
        df = pd.DataFrame([item.split(",") for item in klines])
        df.columns = [
            "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
            "振幅", "涨跌幅", "涨跌额", "换手率"
        ]
        
        # 数据类型转换
        numeric_cols = [
            "开盘", "收盘", "最高", "最低", "成交量", "成交额",
            "振幅", "涨跌幅", "涨跌额", "换手率"
        ]
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df["日期"] = pd.to_datetime(df["日期"])
        df.set_index("日期", inplace=True)
        
        return df
    
    def _process_intraday_1min_data(self, trends: List[str]) -> pd.DataFrame:
        """处理ETF 1分钟分时数据"""
        if not trends:
            return pd.DataFrame()
        
        df = pd.DataFrame([item.split(",") for item in trends])
        df.columns = [
            "时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "最新价"
        ]
        
        # 数据类型转换
        numeric_cols = ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "最新价"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df["时间"] = pd.to_datetime(df["时间"])
        df.set_index("时间", inplace=True)
        
        return df
    
    def _process_intraday_kline_data(self, klines: List[str]) -> pd.DataFrame:
        """处理ETF分时K线数据"""
        if not klines:
            return pd.DataFrame()
        
        df = pd.DataFrame([item.split(",") for item in klines])
        df.columns = [
            "时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
            "振幅", "涨跌幅", "涨跌额", "换手率"
        ]
        
        # 数据类型转换
        numeric_cols = [
            "开盘", "收盘", "最高", "最低", "成交量", "成交额",
            "振幅", "涨跌幅", "涨跌额", "换手率"
        ]
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df["时间"] = pd.to_datetime(df["时间"])
        df.set_index("时间", inplace=True)
        
        # 重新排列列顺序
        df = df[[
            "开盘", "收盘", "最高", "最低", "涨跌幅", "涨跌额",
            "成交量", "成交额", "振幅", "换手率"
        ]]
        
        return df 
   
    def _filter_by_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """按日期范围过滤数据"""
        if df.empty:
            return df
        
        try:
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df.index >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]
                
        except Exception as e:
            logger.warning(f"日期过滤失败: {e}")
        
        return df
    
    def get_etf_special_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算ETF特有指标"""
        if df.empty:
            return {}
        
        indicators = {}
        
        # 基础统计指标
        if "涨跌幅" in df.columns:
            indicators["平均涨跌幅"] = df["涨跌幅"].mean()
            indicators["涨跌幅标准差"] = df["涨跌幅"].std()
            indicators["最大涨幅"] = df["涨跌幅"].max()
            indicators["最大跌幅"] = df["涨跌幅"].min()
        
        if "成交额" in df.columns:
            indicators["平均成交额"] = df["成交额"].mean()
            indicators["成交额标准差"] = df["成交额"].std()
        
        if "换手率" in df.columns:
            indicators["平均换手率"] = df["换手率"].mean()
        
        # ETF特有指标
        if "最新价" in df.columns and len(df) > 1:
            # 价格波动率
            price_returns = df["最新价"].pct_change().dropna()
            if not price_returns.empty:
                indicators["价格波动率"] = price_returns.std() * (252 ** 0.5)  # 年化波动率
        
        # 流动性指标
        if "成交额" in df.columns and "流通市值" in df.columns:
            # 流动性比率
            liquidity_ratio = df["成交额"] / df["流通市值"]
            indicators["平均流动性比率"] = liquidity_ratio.mean()
        
        return indicators
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        total_requests = self.error_stats['total_requests']
        if total_requests == 0:
            return self.error_stats
        
        return {
            **self.error_stats,
            'success_rate': self.error_stats['successful_requests'] / total_requests,
            'error_rate': (total_requests - self.error_stats['successful_requests']) / total_requests
        }
    
    def reset_error_statistics(self):
        """重置错误统计"""
        self.error_stats = {
            'network_errors': 0,
            'data_format_errors': 0,
            'api_errors': 0,
            'total_requests': 0,
            'successful_requests': 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试获取少量ETF实时数据
            response = await self.get_etf_realtime_data(['159707'])  # 中证500ETF
            
            return {
                'status': 'healthy' if response.success else 'unhealthy',
                'response_time': response.response_time,
                'error_message': response.error_message,
                'data_available': not response.data.empty,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }


# 测试和使用示例
async def test_etf_adapter():
    """测试ETF适配器"""
    print("🔍 测试ETF数据适配器")
    print("=" * 50)
    
    adapter = ETFAdapter()
    
    # 测试ETF实时数据
    print("1. 测试ETF实时数据...")
    realtime_response = await adapter.get_etf_realtime_data(['159707', '513500'])
    print(f"   成功: {realtime_response.success}")
    print(f"   响应时间: {realtime_response.response_time:.2f}秒")
    print(f"   数据行数: {len(realtime_response.data)}")
    if not realtime_response.data.empty:
        print(f"   数据列: {list(realtime_response.data.columns)}")
    
    # 测试ETF历史数据
    print("\n2. 测试ETF历史数据...")
    history_request = ETFRequest(
        symbol="159707",
        data_type="history",
        period="daily",
        start_date="20240101",
        end_date="20241231"
    )
    history_response = await adapter.get_etf_history_data(history_request)
    print(f"   成功: {history_response.success}")
    print(f"   响应时间: {history_response.response_time:.2f}秒")
    print(f"   数据行数: {len(history_response.data)}")
    
    # 测试ETF分时数据
    print("\n3. 测试ETF分时数据...")
    intraday_request = ETFRequest(
        symbol="159707",
        data_type="intraday",
        period="5"
    )
    intraday_response = await adapter.get_etf_intraday_data(intraday_request)
    print(f"   成功: {intraday_response.success}")
    print(f"   响应时间: {intraday_response.response_time:.2f}秒")
    print(f"   数据行数: {len(intraday_response.data)}")
    
    # 测试ETF特有指标
    if not realtime_response.data.empty:
        print("\n4. 测试ETF特有指标...")
        indicators = adapter.get_etf_special_indicators(realtime_response.data)
        print(f"   计算出 {len(indicators)} 个指标")
        for key, value in list(indicators.items())[:3]:
            print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
    
    # 健康检查
    print("\n5. 健康检查...")
    health = await adapter.health_check()
    print(f"   状态: {health['status']}")
    print(f"   响应时间: {health.get('response_time', 0):.2f}秒")
    
    # 错误统计
    print("\n6. 错误统计...")
    stats = adapter.get_error_statistics()
    print(f"   总请求数: {stats['total_requests']}")
    print(f"   成功率: {stats.get('success_rate', 0):.2%}")
    
    print("\n✅ ETF适配器测试完成!")


if __name__ == "__main__":
    asyncio.run(test_etf_adapter())