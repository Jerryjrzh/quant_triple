"""
东方财富数据适配器

基于 tmp/core/crawling/stock_hist_em.py 创建的统一数据适配器，
提供股票实时行情、历史数据、分时数据的统一接口。

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
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DataRequest:
    """数据请求结构"""
    symbol: str
    start_date: str = ""
    end_date: str = ""
    period: str = "daily"
    adjust: str = ""
    data_type: str = "stock"


@dataclass
class AdapterResponse:
    """适配器响应结构"""
    success: bool
    data: pd.DataFrame
    error_message: str = ""
    response_time: float = 0.0
    data_source: str = "eastmoney"
    timestamp: datetime = None


class EastMoneyAdapter:
    """东方财富数据适配器"""
    
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
        
        # API端点配置
        self.endpoints = {
            'realtime': "http://82.push2.eastmoney.com/api/qt/clist/get",
            'history': "http://push2his.eastmoney.com/api/qt/stock/kline/get",
            'intraday': "https://push2his.eastmoney.com/api/qt/stock/trends2/get",
            'code_map': "http://80.push2.eastmoney.com/api/qt/clist/get"
        }
        
        # 参数映射
        self.adjust_dict = {"qfq": "1", "hfq": "2", "": "0"}
        self.period_dict = {"daily": "101", "weekly": "102", "monthly": "103"}
        
        # 缓存代码映射
        self._code_id_cache = None
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
    
    async def get_realtime_data(self, symbols: Optional[List[str]] = None) -> AdapterResponse:
        """
        获取实时行情数据
        
        Args:
            symbols: 股票代码列表，为空时获取全市场数据
            
        Returns:
            适配器响应对象
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
                "fid": "f12",
                "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
                "fields": "f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f14,f15,f16,f17,f18,f20,f21,f22,f23,f24,f25",
                "_": str(int(time.time() * 1000)),
            }
            
            # 执行请求
            response_data = await self._make_request(url, params)
            if not response_data:
                return AdapterResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message="Failed to get response data",
                    response_time=time.time() - start_time
                )
            
            # 解析数据
            data = response_data.get("data", {}).get("diff", [])
            if not data:
                return AdapterResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message="No data returned from API",
                    response_time=time.time() - start_time
                )
            
            # 处理分页数据
            if response_data["data"].get("total", 0) > page_size:
                data = await self._fetch_all_pages(url, params, response_data)
            
            # 转换为DataFrame
            df = self._process_realtime_data(data)
            
            # 过滤指定股票
            if symbols:
                df = df[df['代码'].isin(symbols)]
            
            self.error_stats['successful_requests'] += 1
            
            return AdapterResponse(
                success=True,
                data=df,
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"获取实时数据失败: {e}")
            self.error_stats['api_errors'] += 1
            return AdapterResponse(
                success=False,
                data=pd.DataFrame(),
                error_message=str(e),
                response_time=time.time() - start_time
            )
    
    async def get_history_data(self, request: DataRequest) -> AdapterResponse:
        """
        获取历史数据
        
        Args:
            request: 数据请求对象
            
        Returns:
            适配器响应对象
        """
        start_time = time.time()
        self.error_stats['total_requests'] += 1
        
        try:
            # 获取代码映射
            code_id_map = await self._get_code_id_map()
            if request.symbol not in code_id_map:
                return AdapterResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message=f"Symbol {request.symbol} not found in code map",
                    response_time=time.time() - start_time
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
                return AdapterResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message="Failed to get response data",
                    response_time=time.time() - start_time
                )
            
            # 检查数据
            if not (response_data.get("data") and response_data["data"].get("klines")):
                return AdapterResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message="No kline data returned",
                    response_time=time.time() - start_time
                )
            
            # 处理数据
            df = self._process_history_data(response_data["data"]["klines"])
            
            self.error_stats['successful_requests'] += 1
            
            return AdapterResponse(
                success=True,
                data=df,
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            self.error_stats['api_errors'] += 1
            return AdapterResponse(
                success=False,
                data=pd.DataFrame(),
                error_message=str(e),
                response_time=time.time() - start_time
            )
    
    async def get_intraday_data(self, request: DataRequest) -> AdapterResponse:
        """
        获取分时数据
        
        Args:
            request: 数据请求对象，period支持 '1', '5', '15', '30', '60'
            
        Returns:
            适配器响应对象
        """
        start_time = time.time()
        self.error_stats['total_requests'] += 1
        
        try:
            # 获取代码映射
            code_id_map = await self._get_code_id_map()
            if request.symbol not in code_id_map:
                return AdapterResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message=f"Symbol {request.symbol} not found in code map",
                    response_time=time.time() - start_time
                )
            
            if request.period == "1":
                # 1分钟数据使用不同的API
                url = self.endpoints['intraday']
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
                    return AdapterResponse(
                        success=False,
                        data=pd.DataFrame(),
                        error_message="No trends data returned",
                        response_time=time.time() - start_time
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
                    return AdapterResponse(
                        success=False,
                        data=pd.DataFrame(),
                        error_message="No klines data returned",
                        response_time=time.time() - start_time
                    )
                
                df = self._process_intraday_kline_data(response_data["data"]["klines"])
            
            # 应用时间过滤
            if request.start_date or request.end_date:
                df = self._filter_by_date_range(df, request.start_date, request.end_date)
            
            self.error_stats['successful_requests'] += 1
            
            return AdapterResponse(
                success=True,
                data=df,
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"获取分时数据失败: {e}")
            self.error_stats['api_errors'] += 1
            return AdapterResponse(
                success=False,
                data=pd.DataFrame(),
                error_message=str(e),
                response_time=time.time() - start_time
            )
    
    async def _get_code_id_map(self) -> Dict[str, int]:
        """获取股票代码和市场ID映射（带缓存）"""
        current_time = time.time()
        
        # 检查缓存
        if (self._code_id_cache is not None and 
            self._cache_timestamp is not None and 
            current_time - self._cache_timestamp < self._cache_ttl):
            return self._code_id_cache
        
        try:
            code_id_dict = {}
            
            # 获取上海A股
            sh_codes = await self._fetch_market_codes("m:1 t:2,m:1 t:23")
            code_id_dict.update({code: 1 for code in sh_codes})
            
            # 获取深圳A股
            sz_codes = await self._fetch_market_codes("m:0 t:6,m:0 t:80")
            code_id_dict.update({code: 0 for code in sz_codes})
            
            # 获取北京A股
            bj_codes = await self._fetch_market_codes("m:0 t:81 s:2048")
            code_id_dict.update({code: 0 for code in bj_codes})
            
            # 更新缓存
            self._code_id_cache = code_id_dict
            self._cache_timestamp = current_time
            
            logger.info(f"更新代码映射缓存，共 {len(code_id_dict)} 个股票")
            return code_id_dict
            
        except Exception as e:
            logger.error(f"获取代码映射失败: {e}")
            return self._code_id_cache or {}
    
    async def _fetch_market_codes(self, market_filter: str) -> List[str]:
        """获取特定市场的股票代码"""
        codes = []
        url = self.endpoints['code_map']
        page_size = 50
        page_current = 1
        
        params = {
            "pn": page_current,
            "pz": page_size,
            "po": "1",
            "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": "2",
            "invt": "2",
            "fid": "f12",
            "fs": market_filter,
            "fields": "f12",
            "_": str(int(time.time() * 1000)),
        }
        
        try:
            response_data = await self._make_request(url, params)
            if not response_data:
                return codes
            
            data = response_data.get("data", {}).get("diff", [])
            if not data:
                return codes
            
            # 获取第一页数据
            for item in data:
                codes.append(item["f12"])
            
            # 获取其他页面
            data_count = response_data["data"].get("total", 0)
            page_count = math.ceil(data_count / page_size)
            
            for page in range(2, page_count + 1):
                params["pn"] = page
                response_data = await self._make_request(url, params)
                if response_data:
                    page_data = response_data.get("data", {}).get("diff", [])
                    for item in page_data:
                        codes.append(item["f12"])
                        
        except Exception as e:
            logger.error(f"获取市场代码失败 {market_filter}: {e}")
            
        return codes
    
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
                    await asyncio.sleep(2 ** attempt)  # 指数退避
                    
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
    
    async def _fetch_all_pages(self, url: str, base_params: Dict, first_response: Dict) -> List[Dict]:
        """获取所有分页数据"""
        all_data = first_response["data"]["diff"]
        data_count = first_response["data"]["total"]
        page_size = base_params["pz"]
        page_count = math.ceil(data_count / page_size)
        
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
        """处理实时数据"""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # 根据字段映射重命名列
        field_mapping = {
            'f2': '最新价', 'f3': '涨跌幅', 'f4': '涨跌额', 'f5': '成交量', 'f6': '成交额',
            'f7': '振幅', 'f8': '换手率', 'f9': '市盈率动', 'f10': '量比', 'f11': '5分钟涨跌',
            'f12': '代码', 'f14': '名称', 'f15': '最高', 'f16': '最低', 'f17': '今开',
            'f18': '昨收', 'f20': '总市值', 'f21': '流通市值', 'f22': '涨速', 'f23': '市净率',
            'f24': '60日涨跌幅', 'f25': '年初至今涨跌幅'
        }
        
        # 重命名存在的列
        existing_mapping = {k: v for k, v in field_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_mapping)
        
        # 数据类型转换
        numeric_cols = ["最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "振幅", 
                       "换手率", "市盈率动", "量比", "最高", "最低", "今开", "昨收"]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    def _process_history_data(self, klines: List[str]) -> pd.DataFrame:
        """处理历史数据"""
        if not klines:
            return pd.DataFrame()
        
        df = pd.DataFrame([item.split(",") for item in klines])
        df.columns = [
            "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", 
            "振幅", "涨跌幅", "涨跌额", "换手率"
        ]
        
        # 数据类型转换
        numeric_cols = ["开盘", "收盘", "最高", "最低", "成交量", "成交额", 
                       "振幅", "涨跌幅", "涨跌额", "换手率"]
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df["日期"] = pd.to_datetime(df["日期"])
        df.set_index("日期", inplace=True)
        
        return df
    
    def _process_intraday_1min_data(self, trends: List[str]) -> pd.DataFrame:
        """处理1分钟分时数据"""
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
        """处理分时K线数据"""
        if not klines:
            return pd.DataFrame()
        
        df = pd.DataFrame([item.split(",") for item in klines])
        df.columns = [
            "时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
            "振幅", "涨跌幅", "涨跌额", "换手率"
        ]
        
        # 数据类型转换
        numeric_cols = ["开盘", "收盘", "最高", "最低", "成交量", "成交额",
                       "振幅", "涨跌幅", "涨跌额", "换手率"]
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df["时间"] = pd.to_datetime(df["时间"])
        df.set_index("时间", inplace=True)
        
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
            # 测试获取少量实时数据
            response = await self.get_realtime_data(['000001'])
            
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
async def test_eastmoney_adapter():
    """测试东方财富适配器"""
    print("🔍 测试东方财富数据适配器")
    print("=" * 50)
    
    adapter = EastMoneyAdapter()
    
    # 测试实时数据
    print("1. 测试实时数据...")
    realtime_response = await adapter.get_realtime_data(['000001', '000002'])
    print(f"   成功: {realtime_response.success}")
    print(f"   响应时间: {realtime_response.response_time:.2f}秒")
    print(f"   数据行数: {len(realtime_response.data)}")
    if not realtime_response.data.empty:
        print(f"   数据列: {list(realtime_response.data.columns)}")
    
    # 测试历史数据
    print("\n2. 测试历史数据...")
    history_request = DataRequest(
        symbol="000001",
        start_date="20240101",
        end_date="20241231",
        period="daily"
    )
    history_response = await adapter.get_history_data(history_request)
    print(f"   成功: {history_response.success}")
    print(f"   响应时间: {history_response.response_time:.2f}秒")
    print(f"   数据行数: {len(history_response.data)}")
    
    # 测试分时数据
    print("\n3. 测试分时数据...")
    intraday_request = DataRequest(
        symbol="000001",
        period="5"
    )
    intraday_response = await adapter.get_intraday_data(intraday_request)
    print(f"   成功: {intraday_response.success}")
    print(f"   响应时间: {intraday_response.response_time:.2f}秒")
    print(f"   数据行数: {len(intraday_response.data)}")
    
    # 健康检查
    print("\n4. 健康检查...")
    health = await adapter.health_check()
    print(f"   状态: {health['status']}")
    print(f"   响应时间: {health.get('response_time', 0):.2f}秒")
    
    # 错误统计
    print("\n5. 错误统计...")
    stats = adapter.get_error_statistics()
    print(f"   总请求数: {stats['total_requests']}")
    print(f"   成功率: {stats.get('success_rate', 0):.2%}")
    
    print("\n✅ 东方财富适配器测试完成!")


if __name__ == "__main__":
    asyncio.run(test_eastmoney_adapter())