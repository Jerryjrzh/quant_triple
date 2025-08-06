"""
资金流向数据适配器

基于 tmp/core/crawling/stock_fund_em.py 创建的资金流向数据适配器，
提供个股资金流向和板块资金流向数据的统一接口。

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FundFlowRequest:
    """资金流向请求结构"""
    data_type: str = "individual"  # individual, sector
    indicator: str = "5日"  # 今日, 3日, 5日, 10日
    sector_type: str = "行业资金流"  # 行业资金流, 概念资金流, 地域资金流
    symbols: Optional[List[str]] = None  # 指定股票代码列表


@dataclass
class FundFlowResponse:
    """资金流向响应结构"""
    success: bool
    data: pd.DataFrame
    error_message: str = ""
    response_time: float = 0.0
    data_source: str = "eastmoney_fund_flow"
    timestamp: datetime = None
    data_type: str = ""
    indicator: str = ""


class FundFlowAdapter:
    """资金流向数据适配器"""
    
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
        self.base_url = "http://push2.eastmoney.com/api/qt/clist/get"
        
        # 个股资金流向指标映射
        self.individual_indicator_map = {
            "今日": [
                "f62",
                "f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f204,f205,f124",
            ],
            "3日": [
                "f267",
                "f12,f14,f2,f127,f267,f268,f269,f270,f271,f272,f273,f274,f275,f276,f257,f258,f124",
            ],
            "5日": [
                "f164",
                "f12,f14,f2,f109,f164,f165,f166,f167,f168,f169,f170,f171,f172,f173,f257,f258,f124",
            ],
            "10日": [
                "f174",
                "f12,f14,f2,f160,f174,f175,f176,f177,f178,f179,f180,f181,f182,f183,f260,f261,f124",
            ],
        }
        
        # 板块资金流向指标映射
        self.sector_indicator_map = {
            "今日": [
                "f62",
                "1",
                "f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f204,f205,f124",
            ],
            "5日": [
                "f164",
                "5",
                "f12,f14,f2,f109,f164,f165,f166,f167,f168,f169,f170,f171,f172,f173,f257,f258,f124",
            ],
            "10日": [
                "f174",
                "10",
                "f12,f14,f2,f160,f174,f175,f176,f177,f178,f179,f180,f181,f182,f183,f260,f261,f124",
            ],
        }
        
        # 板块类型映射
        self.sector_type_map = {
            "行业资金流": "2",
            "概念资金流": "3", 
            "地域资金流": "1"
        }
        
        # 错误统计
        self.error_stats = {
            'network_errors': 0,
            'data_format_errors': 0,
            'api_errors': 0,
            'total_requests': 0,
            'successful_requests': 0
        }
    
    async def get_individual_fund_flow(self, request: FundFlowRequest) -> FundFlowResponse:
        """
        获取个股资金流向数据
        
        Args:
            request: 资金流向请求对象
            
        Returns:
            资金流向响应对象
        """
        start_time = time.time()
        self.error_stats['total_requests'] += 1
        
        try:
            if request.indicator not in self.individual_indicator_map:
                return FundFlowResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message=f"不支持的指标: {request.indicator}",
                    response_time=time.time() - start_time,
                    data_type="individual",
                    indicator=request.indicator
                )
            
            # 构建请求参数
            indicator_config = self.individual_indicator_map[request.indicator]
            params = {
                "fid": indicator_config[0],
                "po": "1",
                "pz": 5000,  # 获取更多数据
                "pn": 1,
                "np": "1",
                "fltt": "2",
                "invt": "2",
                "ut": "b2884a393a59ad64002292a3e90d46a5",
                "fs": "m:0+t:6+f:!2,m:0+t:13+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2,m:0+t:7+f:!2,m:1+t:3+f:!2",
                "fields": indicator_config[1],
            }
            
            # 获取数据
            all_data = await self._fetch_all_pages(params)
            if not all_data:
                return FundFlowResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message="未获取到数据",
                    response_time=time.time() - start_time,
                    data_type="individual",
                    indicator=request.indicator
                )
            
            # 处理数据
            df = self._process_individual_fund_flow_data(all_data, request.indicator)
            
            # 过滤指定股票
            if request.symbols:
                df = df[df['代码'].isin(request.symbols)]
            
            # 数据去重和时间序列处理
            df = self._deduplicate_and_sort(df)
            
            self.error_stats['successful_requests'] += 1
            
            return FundFlowResponse(
                success=True,
                data=df,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                data_type="individual",
                indicator=request.indicator
            )
            
        except Exception as e:
            logger.error(f"获取个股资金流向失败: {e}")
            self.error_stats['api_errors'] += 1
            return FundFlowResponse(
                success=False,
                data=pd.DataFrame(),
                error_message=str(e),
                response_time=time.time() - start_time,
                data_type="individual",
                indicator=request.indicator
            )
    
    async def get_sector_fund_flow(self, request: FundFlowRequest) -> FundFlowResponse:
        """
        获取板块资金流向数据
        
        Args:
            request: 资金流向请求对象
            
        Returns:
            资金流向响应对象
        """
        start_time = time.time()
        self.error_stats['total_requests'] += 1
        
        try:
            if request.indicator not in self.sector_indicator_map:
                return FundFlowResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message=f"不支持的指标: {request.indicator}",
                    response_time=time.time() - start_time,
                    data_type="sector",
                    indicator=request.indicator
                )
            
            if request.sector_type not in self.sector_type_map:
                return FundFlowResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message=f"不支持的板块类型: {request.sector_type}",
                    response_time=time.time() - start_time,
                    data_type="sector",
                    indicator=request.indicator
                )
            
            # 构建请求参数
            indicator_config = self.sector_indicator_map[request.indicator]
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            params = {
                "pn": 1,
                "pz": 5000,
                "po": "1",
                "np": "1",
                "ut": "b2884a393a59ad64002292a3e90d46a5",
                "fltt": "2",
                "invt": "2",
                "fid0": indicator_config[0],
                "fs": f"m:90 t:{self.sector_type_map[request.sector_type]}",
                "stat": indicator_config[1],
                "fields": indicator_config[2],
                "rt": "52975239",
                "cb": f"jQuery18308357908311220152_{int(time.time() * 1000)}",
                "_": int(time.time() * 1000),
            }
            
            # 获取数据
            all_data = await self._fetch_sector_pages(params, headers)
            if not all_data:
                return FundFlowResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message="未获取到板块数据",
                    response_time=time.time() - start_time,
                    data_type="sector",
                    indicator=request.indicator
                )
            
            # 处理数据
            df = self._process_sector_fund_flow_data(all_data, request.indicator)
            
            # 数据去重和排序
            df = self._deduplicate_and_sort(df, sort_by='名称')
            
            self.error_stats['successful_requests'] += 1
            
            return FundFlowResponse(
                success=True,
                data=df,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                data_type="sector",
                indicator=request.indicator
            )
            
        except Exception as e:
            logger.error(f"获取板块资金流向失败: {e}")
            self.error_stats['api_errors'] += 1
            return FundFlowResponse(
                success=False,
                data=pd.DataFrame(),
                error_message=str(e),
                response_time=time.time() - start_time,
                data_type="sector",
                indicator=request.indicator
            )
    
    async def get_multi_period_fund_flow(self, symbols: List[str]) -> Dict[str, FundFlowResponse]:
        """
        获取多时间周期资金流向数据
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            多时间周期数据字典
        """
        results = {}
        periods = ["今日", "3日", "5日", "10日"]
        
        for period in periods:
            request = FundFlowRequest(
                data_type="individual",
                indicator=period,
                symbols=symbols
            )
            
            response = await self.get_individual_fund_flow(request)
            results[period] = response
            
            # 避免请求过快
            await asyncio.sleep(0.5)
        
        return results
    
    async def _fetch_all_pages(self, base_params: Dict) -> List[Dict]:
        """获取所有分页数据"""
        all_data = []
        page_current = 1
        
        while True:
            params = base_params.copy()
            params["pn"] = page_current
            
            try:
                response = self.session.get(
                    self.base_url, 
                    params=params, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                data_json = response.json()
                
                if not data_json.get("data") or not data_json["data"].get("diff"):
                    break
                
                page_data = data_json["data"]["diff"]
                all_data.extend(page_data)
                
                # 检查是否还有更多页面
                data_count = data_json["data"].get("total", 0)
                page_size = base_params.get("pz", 50)
                
                if len(all_data) >= data_count:
                    break
                
                page_current += 1
                await asyncio.sleep(0.1)  # 避免请求过快
                
            except Exception as e:
                logger.error(f"获取第{page_current}页数据失败: {e}")
                self.error_stats['network_errors'] += 1
                break
        
        return all_data
    
    async def _fetch_sector_pages(self, base_params: Dict, headers: Dict) -> List[Dict]:
        """获取板块数据所有分页"""
        all_data = []
        page_current = 1
        
        while True:
            params = base_params.copy()
            params["pn"] = page_current
            params["_"] = int(time.time() * 1000)
            
            try:
                response = self.session.get(
                    self.base_url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # 处理JSONP响应
                text_data = response.text
                if "jQuery" in text_data:
                    json_start = text_data.find("{")
                    json_end = text_data.rfind("}")
                    if json_start != -1 and json_end != -1:
                        json_str = text_data[json_start:json_end + 1]
                        data_json = json.loads(json_str)
                    else:
                        break
                else:
                    data_json = response.json()
                
                if not data_json.get("data") or not data_json["data"].get("diff"):
                    break
                
                page_data = data_json["data"]["diff"]
                all_data.extend(page_data)
                
                # 检查是否还有更多页面
                data_count = data_json["data"].get("total", 0)
                page_size = base_params.get("pz", 50)
                
                if len(all_data) >= data_count:
                    break
                
                page_current += 1
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"获取板块第{page_current}页数据失败: {e}")
                self.error_stats['network_errors'] += 1
                break
        
        return all_data
    
    def _process_individual_fund_flow_data(self, data: List[Dict], indicator: str) -> pd.DataFrame:
        """处理个股资金流向数据"""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # 过滤无效数据（如果是字典格式）
        if not df.empty and "f2" in df.columns:
            df = df[~df["f2"].isin(["-"])]
        elif not df.empty and len(df.columns) > 1:
            # 如果是列表格式，第二列通常是价格相关数据
            df = df[~df.iloc[:, 1].isin(["-"])]
        
        # 根据指标设置列名
        if indicator == "今日":
            df.columns = [
                "最新价", "今日涨跌幅", "代码", "名称", "今日主力净流入-净额", "今日超大单净流入-净额",
                "今日超大单净流入-净占比", "今日大单净流入-净额", "今日大单净流入-净占比",
                "今日中单净流入-净额", "今日中单净流入-净占比", "今日小单净流入-净额",
                "今日小单净流入-净占比", "_", "今日主力净流入-净占比", "_", "_", "_"
            ]
            
            df = df[[
                "代码", "名称", "最新价", "今日涨跌幅", "今日主力净流入-净额", "今日主力净流入-净占比",
                "今日超大单净流入-净额", "今日超大单净流入-净占比", "今日大单净流入-净额", "今日大单净流入-净占比",
                "今日中单净流入-净额", "今日中单净流入-净占比", "今日小单净流入-净额", "今日小单净流入-净占比"
            ]]
            
        elif indicator == "3日":
            df.columns = [
                "最新价", "代码", "名称", "_", "3日涨跌幅", "_", "_", "_", "3日主力净流入-净额",
                "3日主力净流入-净占比", "3日超大单净流入-净额", "3日超大单净流入-净占比",
                "3日大单净流入-净额", "3日大单净流入-净占比", "3日中单净流入-净额", "3日中单净流入-净占比",
                "3日小单净流入-净额", "3日小单净流入-净占比"
            ]
            
            df = df[[
                "代码", "名称", "最新价", "3日涨跌幅", "3日主力净流入-净额", "3日主力净流入-净占比",
                "3日超大单净流入-净额", "3日超大单净流入-净占比", "3日大单净流入-净额", "3日大单净流入-净占比",
                "3日中单净流入-净额", "3日中单净流入-净占比", "3日小单净流入-净额", "3日小单净流入-净占比"
            ]]
            
        elif indicator == "5日":
            df.columns = [
                "最新价", "代码", "名称", "5日涨跌幅", "_", "5日主力净流入-净额", "5日主力净流入-净占比",
                "5日超大单净流入-净额", "5日超大单净流入-净占比", "5日大单净流入-净额", "5日大单净流入-净占比",
                "5日中单净流入-净额", "5日中单净流入-净占比", "5日小单净流入-净额", "5日小单净流入-净占比",
                "_", "_", "_"
            ]
            
            df = df[[
                "代码", "名称", "最新价", "5日涨跌幅", "5日主力净流入-净额", "5日主力净流入-净占比",
                "5日超大单净流入-净额", "5日超大单净流入-净占比", "5日大单净流入-净额", "5日大单净流入-净占比",
                "5日中单净流入-净额", "5日中单净流入-净占比", "5日小单净流入-净额", "5日小单净流入-净占比"
            ]]
            
        elif indicator == "10日":
            df.columns = [
                "最新价", "代码", "名称", "_", "10日涨跌幅", "10日主力净流入-净额", "10日主力净流入-净占比",
                "10日超大单净流入-净额", "10日超大单净流入-净占比", "10日大单净流入-净额", "10日大单净流入-净占比",
                "10日中单净流入-净额", "10日中单净流入-净占比", "10日小单净流入-净额", "10日小单净流入-净占比",
                "_", "_", "_"
            ]
            
            df = df[[
                "代码", "名称", "最新价", "10日涨跌幅", "10日主力净流入-净额", "10日主力净流入-净占比",
                "10日超大单净流入-净额", "10日超大单净流入-净占比", "10日大单净流入-净额", "10日大单净流入-净占比",
                "10日中单净流入-净额", "10日中单净流入-净占比", "10日小单净流入-净额", "10日小单净流入-净占比"
            ]]
        
        # 数据类型转换
        self._convert_numeric_columns(df, indicator)
        
        return df
    
    def _process_sector_fund_flow_data(self, data: List[Dict], indicator: str) -> pd.DataFrame:
        """处理板块资金流向数据"""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # 过滤无效数据（如果是字典格式）
        if not df.empty and "f2" in df.columns:
            df = df[~df["f2"].isin(["-"])]
        elif not df.empty and len(df.columns) > 1:
            # 如果是列表格式，第二列通常是价格相关数据
            df = df[~df.iloc[:, 1].isin(["-"])]
        
        # 根据指标设置列名
        if indicator == "今日":
            df.columns = [
                "-", "今日涨跌幅", "_", "名称", "今日主力净流入-净额", "今日超大单净流入-净额",
                "今日超大单净流入-净占比", "今日大单净流入-净额", "今日大单净流入-净占比",
                "今日中单净流入-净额", "今日中单净流入-净占比", "今日小单净流入-净额",
                "今日小单净流入-净占比", "-", "今日主力净流入-净占比", "今日主力净流入最大股",
                "今日主力净流入最大股代码", "是否净流入"
            ]
            
            df = df[[
                "名称", "今日涨跌幅", "今日主力净流入-净额", "今日主力净流入-净占比",
                "今日超大单净流入-净额", "今日超大单净流入-净占比", "今日大单净流入-净额", "今日大单净流入-净占比",
                "今日中单净流入-净额", "今日中单净流入-净占比", "今日小单净流入-净额", "今日小单净流入-净占比",
                "今日主力净流入最大股"
            ]]
            
        elif indicator == "5日":
            df.columns = [
                "-", "_", "名称", "5日涨跌幅", "_", "5日主力净流入-净额", "5日主力净流入-净占比",
                "5日超大单净流入-净额", "5日超大单净流入-净占比", "5日大单净流入-净额", "5日大单净流入-净占比",
                "5日中单净流入-净额", "5日中单净流入-净占比", "5日小单净流入-净额", "5日小单净流入-净占比",
                "5日主力净流入最大股", "_", "_"
            ]
            
            df = df[[
                "名称", "5日涨跌幅", "5日主力净流入-净额", "5日主力净流入-净占比",
                "5日超大单净流入-净额", "5日超大单净流入-净占比", "5日大单净流入-净额", "5日大单净流入-净占比",
                "5日中单净流入-净额", "5日中单净流入-净占比", "5日小单净流入-净额", "5日小单净流入-净占比",
                "5日主力净流入最大股"
            ]]
            
        elif indicator == "10日":
            df.columns = [
                "-", "_", "名称", "_", "10日涨跌幅", "10日主力净流入-净额", "10日主力净流入-净占比",
                "10日超大单净流入-净额", "10日超大单净流入-净占比", "10日大单净流入-净额", "10日大单净流入-净占比",
                "10日中单净流入-净额", "10日中单净流入-净占比", "10日小单净流入-净额", "10日小单净流入-净占比",
                "10日主力净流入最大股", "_", "_"
            ]
            
            df = df[[
                "名称", "10日涨跌幅", "10日主力净流入-净额", "10日主力净流入-净占比",
                "10日超大单净流入-净额", "10日超大单净流入-净占比", "10日大单净流入-净额", "10日大单净流入-净占比",
                "10日中单净流入-净额", "10日中单净流入-净占比", "10日小单净流入-净额", "10日小单净流入-净占比",
                "10日主力净流入最大股"
            ]]
        
        # 数据类型转换
        self._convert_numeric_columns(df, indicator, is_sector=True)
        
        return df
    
    def _convert_numeric_columns(self, df: pd.DataFrame, indicator: str, is_sector: bool = False):
        """转换数值列的数据类型"""
        if df.empty:
            return
        
        # 基础数值列
        base_numeric_cols = [
            f"{indicator}主力净流入-净额", f"{indicator}主力净流入-净占比",
            f"{indicator}超大单净流入-净额", f"{indicator}超大单净流入-净占比",
            f"{indicator}大单净流入-净额", f"{indicator}大单净流入-净占比",
            f"{indicator}中单净流入-净额", f"{indicator}中单净流入-净占比",
            f"{indicator}小单净流入-净额", f"{indicator}小单净流入-净占比",
            f"{indicator}涨跌幅"
        ]
        
        # 个股特有列
        if not is_sector and "最新价" in df.columns:
            base_numeric_cols.append("最新价")
        
        # 转换数值类型
        for col in base_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    
    def _deduplicate_and_sort(self, df: pd.DataFrame, sort_by: str = "代码") -> pd.DataFrame:
        """数据去重和排序"""
        if df.empty:
            return df
        
        # 去重（保留最新数据）
        if sort_by in df.columns:
            df = df.drop_duplicates(subset=[sort_by], keep='last')
            # 排序
            df = df.sort_values(by=sort_by).reset_index(drop=True)
        
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
            # 测试获取少量个股资金流向数据
            request = FundFlowRequest(
                data_type="individual",
                indicator="今日",
                symbols=["000001"]
            )
            
            response = await self.get_individual_fund_flow(request)
            
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
async def test_fund_flow_adapter():
    """测试资金流向适配器"""
    print("🔍 测试资金流向数据适配器")
    print("=" * 50)
    
    adapter = FundFlowAdapter()
    
    # 测试个股资金流向
    print("1. 测试个股资金流向...")
    individual_request = FundFlowRequest(
        data_type="individual",
        indicator="5日",
        symbols=["000001", "000002"]
    )
    
    individual_response = await adapter.get_individual_fund_flow(individual_request)
    print(f"   成功: {individual_response.success}")
    print(f"   响应时间: {individual_response.response_time:.2f}秒")
    print(f"   数据行数: {len(individual_response.data)}")
    if not individual_response.data.empty:
        print(f"   数据列: {list(individual_response.data.columns)}")
    
    # 测试板块资金流向
    print("\n2. 测试板块资金流向...")
    sector_request = FundFlowRequest(
        data_type="sector",
        indicator="今日",
        sector_type="行业资金流"
    )
    
    sector_response = await adapter.get_sector_fund_flow(sector_request)
    print(f"   成功: {sector_response.success}")
    print(f"   响应时间: {sector_response.response_time:.2f}秒")
    print(f"   数据行数: {len(sector_response.data)}")
    
    # 测试多时间周期数据
    print("\n3. 测试多时间周期数据...")
    multi_period_data = await adapter.get_multi_period_fund_flow(["000001"])
    for period, response in multi_period_data.items():
        print(f"   {period}: {'✅' if response.success else '❌'} ({len(response.data)}行)")
    
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
    
    print("\n✅ 资金流向适配器测试完成!")


if __name__ == "__main__":
    asyncio.run(test_fund_flow_adapter())