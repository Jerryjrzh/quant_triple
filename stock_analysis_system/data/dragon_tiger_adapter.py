"""
龙虎榜数据适配器

基于 tmp/core/crawling/stock_lhb_em.py 创建的龙虎榜数据适配器，
提供龙虎榜详情、机构统计、营业部排行等数据的统一接口。

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import requests
from dataclasses import dataclass
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DragonTigerRequest:
    """龙虎榜请求结构"""
    data_type: str = "detail"  # detail, stock_statistic, institution_daily, institution_statistic, department_active, department_ranking, department_statistic, stock_detail
    start_date: str = ""
    end_date: str = ""
    symbol: str = ""  # 股票代码
    period: str = "近一月"  # 近一月, 近三月, 近六月, 近一年
    flag: str = "买入"  # 买入, 卖出 (仅用于stock_detail)


@dataclass
class DragonTigerResponse:
    """龙虎榜响应结构"""
    success: bool
    data: pd.DataFrame
    error_message: str = ""
    response_time: float = 0.0
    data_source: str = "eastmoney_dragon_tiger"
    timestamp: datetime = None
    data_type: str = ""
    total_pages: int = 1


class DragonTigerAdapter:
    """龙虎榜数据适配器"""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
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
        self.base_url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
        
        # 时间周期映射
        self.period_map = {
            "近一月": "01",
            "近三月": "02", 
            "近六月": "03",
            "近一年": "04"
        }
        
        # 错误统计
        self.error_stats = {
            'network_errors': 0,
            'data_format_errors': 0,
            'api_errors': 0,
            'total_requests': 0,
            'successful_requests': 0
        }
    
    async def get_dragon_tiger_detail(self, request: DragonTigerRequest) -> DragonTigerResponse:
        """
        获取龙虎榜详情数据
        
        Args:
            request: 龙虎榜请求对象
            
        Returns:
            龙虎榜响应对象
        """
        start_time = time.time()
        self.error_stats['total_requests'] += 1
        
        try:
            # 格式化日期
            start_date = self._format_date(request.start_date)
            end_date = self._format_date(request.end_date)
            
            params = {
                "sortColumns": "SECURITY_CODE,TRADE_DATE",
                "sortTypes": "1,-1",
                "pageSize": "5000",
                "pageNumber": "1",
                "reportName": "RPT_DAILYBILLBOARD_DETAILSNEW",
                "columns": "SECURITY_CODE,SECUCODE,SECURITY_NAME_ABBR,TRADE_DATE,EXPLAIN,CLOSE_PRICE,CHANGE_RATE,BILLBOARD_NET_AMT,BILLBOARD_BUY_AMT,BILLBOARD_SELL_AMT,BILLBOARD_DEAL_AMT,ACCUM_AMOUNT,DEAL_NET_RATIO,DEAL_AMOUNT_RATIO,TURNOVERRATE,FREE_MARKET_CAP,EXPLANATION,D1_CLOSE_ADJCHRATE,D2_CLOSE_ADJCHRATE,D5_CLOSE_ADJCHRATE,D10_CLOSE_ADJCHRATE,SECURITY_TYPE_CODE",
                "source": "WEB",
                "client": "WEB",
                "filter": f"(TRADE_DATE<='{end_date}')(TRADE_DATE>='{start_date}')",
            }
            
            # 获取所有分页数据
            all_data = await self._fetch_all_pages(params)
            if not all_data:
                return DragonTigerResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message="未获取到龙虎榜数据",
                    response_time=time.time() - start_time,
                    data_type="detail"
                )
            
            # 处理数据
            df = self._process_dragon_tiger_detail_data(all_data)
            
            # 数据关联性验证和完整性检查
            df = self._validate_and_clean_data(df)
            
            self.error_stats['successful_requests'] += 1
            
            return DragonTigerResponse(
                success=True,
                data=df,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                data_type="detail"
            )
            
        except Exception as e:
            logger.error(f"获取龙虎榜详情失败: {e}")
            self.error_stats['api_errors'] += 1
            return DragonTigerResponse(
                success=False,
                data=pd.DataFrame(),
                error_message=str(e),
                response_time=time.time() - start_time,
                data_type="detail"
            )
    
    async def get_stock_statistic(self, request: DragonTigerRequest) -> DragonTigerResponse:
        """
        获取个股上榜统计数据
        
        Args:
            request: 龙虎榜请求对象
            
        Returns:
            龙虎榜响应对象
        """
        start_time = time.time()
        self.error_stats['total_requests'] += 1
        
        try:
            params = {
                "sortColumns": "BILLBOARD_TIMES,LATEST_TDATE,SECURITY_CODE",
                "sortTypes": "-1,-1,1",
                "pageSize": "500",
                "pageNumber": "1",
                "reportName": "RPT_BILLBOARD_TRADEALL",
                "columns": "ALL",
                "source": "WEB",
                "client": "WEB",
                "filter": f'(STATISTICS_CYCLE="{self.period_map[request.period]}")',
            }
            
            response_data = await self._make_request(params)
            if not response_data:
                return DragonTigerResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message="未获取到个股统计数据",
                    response_time=time.time() - start_time,
                    data_type="stock_statistic"
                )
            
            # 处理数据
            df = self._process_stock_statistic_data(response_data["result"]["data"])
            
            self.error_stats['successful_requests'] += 1
            
            return DragonTigerResponse(
                success=True,
                data=df,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                data_type="stock_statistic"
            )
            
        except Exception as e:
            logger.error(f"获取个股统计失败: {e}")
            self.error_stats['api_errors'] += 1
            return DragonTigerResponse(
                success=False,
                data=pd.DataFrame(),
                error_message=str(e),
                response_time=time.time() - start_time,
                data_type="stock_statistic"
            )
    
    async def get_institution_daily_statistic(self, request: DragonTigerRequest) -> DragonTigerResponse:
        """
        获取机构买卖每日统计数据
        
        Args:
            request: 龙虎榜请求对象
            
        Returns:
            龙虎榜响应对象
        """
        start_time = time.time()
        self.error_stats['total_requests'] += 1
        
        try:
            start_date = self._format_date(request.start_date)
            end_date = self._format_date(request.end_date)
            
            params = {
                "sortColumns": "NET_BUY_AMT,TRADE_DATE,SECURITY_CODE",
                "sortTypes": "-1,-1,1",
                "pageSize": "5000",
                "pageNumber": "1",
                "reportName": "RPT_ORGANIZATION_TRADE_DETAILS",
                "columns": "ALL",
                "source": "WEB",
                "client": "WEB",
                "filter": f"(TRADE_DATE>='{start_date}')(TRADE_DATE<='{end_date}')",
            }
            
            response_data = await self._make_request(params)
            if not response_data:
                return DragonTigerResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message="未获取到机构每日统计数据",
                    response_time=time.time() - start_time,
                    data_type="institution_daily"
                )
            
            # 处理数据
            df = self._process_institution_daily_data(response_data["result"]["data"])
            
            self.error_stats['successful_requests'] += 1
            
            return DragonTigerResponse(
                success=True,
                data=df,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                data_type="institution_daily"
            )
            
        except Exception as e:
            logger.error(f"获取机构每日统计失败: {e}")
            self.error_stats['api_errors'] += 1
            return DragonTigerResponse(
                success=False,
                data=pd.DataFrame(),
                error_message=str(e),
                response_time=time.time() - start_time,
                data_type="institution_daily"
            )
    
    async def get_department_ranking(self, request: DragonTigerRequest) -> DragonTigerResponse:
        """
        获取营业部排行数据
        
        Args:
            request: 龙虎榜请求对象
            
        Returns:
            龙虎榜响应对象
        """
        start_time = time.time()
        self.error_stats['total_requests'] += 1
        
        try:
            params = {
                "sortColumns": "TOTAL_BUYER_SALESTIMES_1DAY,OPERATEDEPT_CODE",
                "sortTypes": "-1,1",
                "pageSize": "5000",
                "pageNumber": "1",
                "reportName": "RPT_RATEDEPT_RETURNT_RANKING",
                "columns": "ALL",
                "source": "WEB",
                "client": "WEB",
                "filter": f'(STATISTICSCYCLE="{self.period_map[request.period]}")',
            }
            
            # 获取所有分页数据
            all_data = await self._fetch_all_pages(params)
            if not all_data:
                return DragonTigerResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message="未获取到营业部排行数据",
                    response_time=time.time() - start_time,
                    data_type="department_ranking"
                )
            
            # 处理数据
            df = self._process_department_ranking_data(all_data)
            
            self.error_stats['successful_requests'] += 1
            
            return DragonTigerResponse(
                success=True,
                data=df,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                data_type="department_ranking"
            )
            
        except Exception as e:
            logger.error(f"获取营业部排行失败: {e}")
            self.error_stats['api_errors'] += 1
            return DragonTigerResponse(
                success=False,
                data=pd.DataFrame(),
                error_message=str(e),
                response_time=time.time() - start_time,
                data_type="department_ranking"
            )
    
    async def get_stock_detail(self, request: DragonTigerRequest) -> DragonTigerResponse:
        """
        获取个股龙虎榜详情
        
        Args:
            request: 龙虎榜请求对象，需要包含symbol和start_date
            
        Returns:
            龙虎榜响应对象
        """
        start_time = time.time()
        self.error_stats['total_requests'] += 1
        
        try:
            flag_map = {"买入": "BUY", "卖出": "SELL"}
            report_map = {
                "买入": "RPT_BILLBOARD_DAILYDETAILSBUY",
                "卖出": "RPT_BILLBOARD_DAILYDETAILSSELL",
            }
            
            date_formatted = self._format_date(request.start_date)
            
            params = {
                "reportName": report_map[request.flag],
                "columns": "ALL",
                "filter": f"""(TRADE_DATE='{date_formatted}')(SECURITY_CODE="{request.symbol}")""",
                "pageNumber": "1",
                "pageSize": "500",
                "sortTypes": "-1",
                "sortColumns": flag_map[request.flag],
                "source": "WEB",
                "client": "WEB",
                "_": str(int(time.time() * 1000)),
            }
            
            response_data = await self._make_request(params)
            if not response_data:
                return DragonTigerResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message="未获取到个股详情数据",
                    response_time=time.time() - start_time,
                    data_type="stock_detail"
                )
            
            # 处理数据
            df = self._process_stock_detail_data(response_data["result"]["data"], request.flag)
            
            self.error_stats['successful_requests'] += 1
            
            return DragonTigerResponse(
                success=True,
                data=df,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                data_type="stock_detail"
            )
            
        except Exception as e:
            logger.error(f"获取个股详情失败: {e}")
            self.error_stats['api_errors'] += 1
            return DragonTigerResponse(
                success=False,
                data=pd.DataFrame(),
                error_message=str(e),
                response_time=time.time() - start_time,
                data_type="stock_detail"
            )
    
    async def _make_request(self, params: Dict) -> Optional[Dict]:
        """发送HTTP请求（带重试机制）"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    self.base_url, 
                    params=params, 
                    timeout=self.timeout
                )
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
    
    async def _fetch_all_pages(self, base_params: Dict) -> List[Dict]:
        """获取所有分页数据"""
        all_data = []
        
        # 获取第一页数据
        response_data = await self._make_request(base_params)
        if not response_data or not response_data.get("result"):
            return all_data
        
        first_page_data = response_data["result"].get("data", [])
        all_data.extend(first_page_data)
        
        # 获取总页数
        total_pages = response_data["result"].get("pages", 1)
        
        # 获取其他页面数据
        for page in range(2, total_pages + 1):
            params = base_params.copy()
            params["pageNumber"] = str(page)
            
            response_data = await self._make_request(params)
            if response_data and response_data.get("result", {}).get("data"):
                all_data.extend(response_data["result"]["data"])
            
            # 避免请求过快
            await asyncio.sleep(0.1)
        
        return all_data
    
    def _format_date(self, date_str: str) -> str:
        """格式化日期字符串"""
        if not date_str:
            return datetime.now().strftime("%Y-%m-%d")
        
        if len(date_str) == 8:  # YYYYMMDD
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        
        return date_str
    
    def _process_dragon_tiger_detail_data(self, data: List[Dict]) -> pd.DataFrame:
        """处理龙虎榜详情数据"""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df.reset_index(inplace=True)
        df["index"] = df.index + 1
        
        # 重命名列
        df.rename(columns={
            "index": "序号",
            "SECURITY_CODE": "代码",
            "SECUCODE": "-",
            "SECURITY_NAME_ABBR": "名称",
            "TRADE_DATE": "上榜日",
            "EXPLAIN": "解读",
            "CLOSE_PRICE": "收盘价",
            "CHANGE_RATE": "涨跌幅",
            "BILLBOARD_NET_AMT": "龙虎榜净买额",
            "BILLBOARD_BUY_AMT": "龙虎榜买入额",
            "BILLBOARD_SELL_AMT": "龙虎榜卖出额",
            "BILLBOARD_DEAL_AMT": "龙虎榜成交额",
            "ACCUM_AMOUNT": "市场总成交额",
            "DEAL_NET_RATIO": "净买额占总成交比",
            "DEAL_AMOUNT_RATIO": "成交额占总成交比",
            "TURNOVERRATE": "换手率",
            "FREE_MARKET_CAP": "流通市值",
            "EXPLANATION": "上榜原因",
            "D1_CLOSE_ADJCHRATE": "上榜后1日",
            "D2_CLOSE_ADJCHRATE": "上榜后2日",
            "D5_CLOSE_ADJCHRATE": "上榜后5日",
            "D10_CLOSE_ADJCHRATE": "上榜后10日",
        }, inplace=True)
        
        # 选择需要的列
        df = df[[
            "序号", "代码", "名称", "上榜日", "解读", "收盘价", "涨跌幅",
            "龙虎榜净买额", "龙虎榜买入额", "龙虎榜卖出额", "龙虎榜成交额",
            "市场总成交额", "净买额占总成交比", "成交额占总成交比", "换手率",
            "流通市值", "上榜原因", "上榜后1日", "上榜后2日", "上榜后5日", "上榜后10日"
        ]]
        
        # 数据类型转换
        self._convert_numeric_columns(df, [
            "收盘价", "涨跌幅", "龙虎榜净买额", "龙虎榜买入额", "龙虎榜卖出额",
            "龙虎榜成交额", "市场总成交额", "净买额占总成交比", "成交额占总成交比",
            "换手率", "流通市值", "上榜后1日", "上榜后2日", "上榜后5日", "上榜后10日"
        ])
        
        # 日期转换
        df["上榜日"] = pd.to_datetime(df["上榜日"]).dt.date
        
        return df
    
    def _process_stock_statistic_data(self, data: List[Dict]) -> pd.DataFrame:
        """处理个股统计数据"""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df.reset_index(inplace=True)
        df["index"] = df.index + 1
        
        # 设置列名
        df.columns = [
            "序号", "-", "代码", "最近上榜日", "名称", "近1个月涨跌幅", "近3个月涨跌幅",
            "近6个月涨跌幅", "近1年涨跌幅", "涨跌幅", "收盘价", "-", "龙虎榜总成交额",
            "龙虎榜净买额", "-", "-", "机构买入净额", "上榜次数", "龙虎榜买入额",
            "龙虎榜卖出额", "机构买入总额", "机构卖出总额", "买方机构次数", "卖方机构次数", "-"
        ]
        
        # 选择需要的列
        df = df[[
            "序号", "代码", "名称", "最近上榜日", "收盘价", "涨跌幅", "上榜次数",
            "龙虎榜净买额", "龙虎榜买入额", "龙虎榜卖出额", "龙虎榜总成交额",
            "买方机构次数", "卖方机构次数", "机构买入净额", "机构买入总额", "机构卖出总额",
            "近1个月涨跌幅", "近3个月涨跌幅", "近6个月涨跌幅", "近1年涨跌幅"
        ]]
        
        # 日期转换
        df["最近上榜日"] = pd.to_datetime(df["最近上榜日"]).dt.date
        
        return df
    
    def _process_institution_daily_data(self, data: List[Dict]) -> pd.DataFrame:
        """处理机构每日统计数据"""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df.reset_index(inplace=True)
        df["index"] = df.index + 1
        
        # 动态设置列名，适应不同的列数
        expected_cols = [
            "序号", "-", "名称", "代码", "上榜日期", "收盘价", "涨跌幅", "买方机构数",
            "卖方机构数", "机构买入总额", "机构卖出总额", "机构买入净额", "市场总成交额",
            "机构净买额占总成交额比", "换手率", "流通市值", "上榜原因"
        ]
        
        actual_cols = len(df.columns)
        if actual_cols >= len(expected_cols):
            # 如果实际列数更多，添加占位符
            extra_cols = [f"-{i}" for i in range(actual_cols - len(expected_cols))]
            df.columns = expected_cols + extra_cols
        else:
            # 如果实际列数较少，截取对应的列名
            df.columns = expected_cols[:actual_cols]
        
        # 选择需要的列
        df = df[[
            "序号", "代码", "名称", "收盘价", "涨跌幅", "买方机构数", "卖方机构数",
            "机构买入总额", "机构卖出总额", "机构买入净额", "市场总成交额",
            "机构净买额占总成交额比", "换手率", "流通市值", "上榜原因", "上榜日期"
        ]]
        
        # 数据类型转换
        self._convert_numeric_columns(df, [
            "收盘价", "涨跌幅", "买方机构数", "卖方机构数", "机构买入总额",
            "机构卖出总额", "机构买入净额", "市场总成交额", "机构净买额占总成交额比",
            "换手率", "流通市值"
        ])
        
        # 日期转换
        df["上榜日期"] = pd.to_datetime(df["上榜日期"]).dt.date
        
        return df
    
    def _process_department_ranking_data(self, data: List[Dict]) -> pd.DataFrame:
        """处理营业部排行数据"""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df.reset_index(inplace=True)
        df["index"] = df.index + 1
        
        # 重命名列
        df.rename(columns={
            "index": "序号",
            "OPERATEDEPT_NAME": "营业部名称",
            "TOTAL_BUYER_SALESTIMES_1DAY": "上榜后1天-买入次数",
            "AVERAGE_INCREASE_1DAY": "上榜后1天-平均涨幅",
            "RISE_PROBABILITY_1DAY": "上榜后1天-上涨概率",
            "TOTAL_BUYER_SALESTIMES_2DAY": "上榜后2天-买入次数",
            "AVERAGE_INCREASE_2DAY": "上榜后2天-平均涨幅",
            "RISE_PROBABILITY_2DAY": "上榜后2天-上涨概率",
            "TOTAL_BUYER_SALESTIMES_3DAY": "上榜后3天-买入次数",
            "AVERAGE_INCREASE_3DAY": "上榜后3天-平均涨幅",
            "RISE_PROBABILITY_3DAY": "上榜后3天-上涨概率",
            "TOTAL_BUYER_SALESTIMES_5DAY": "上榜后5天-买入次数",
            "AVERAGE_INCREASE_5DAY": "上榜后5天-平均涨幅",
            "RISE_PROBABILITY_5DAY": "上榜后5天-上涨概率",
            "TOTAL_BUYER_SALESTIMES_10DAY": "上榜后10天-买入次数",
            "AVERAGE_INCREASE_10DAY": "上榜后10天-平均涨幅",
            "RISE_PROBABILITY_10DAY": "上榜后10天-上涨概率",
        }, inplace=True)
        
        # 选择需要的列
        df = df[[
            "序号", "营业部名称", "上榜后1天-买入次数", "上榜后1天-平均涨幅", "上榜后1天-上涨概率",
            "上榜后2天-买入次数", "上榜后2天-平均涨幅", "上榜后2天-上涨概率",
            "上榜后3天-买入次数", "上榜后3天-平均涨幅", "上榜后3天-上涨概率",
            "上榜后5天-买入次数", "上榜后5天-平均涨幅", "上榜后5天-上涨概率",
            "上榜后10天-买入次数", "上榜后10天-平均涨幅", "上榜后10天-上涨概率"
        ]]
        
        # 数据类型转换
        numeric_cols = [col for col in df.columns if col not in ["序号", "营业部名称"]]
        self._convert_numeric_columns(df, numeric_cols)
        
        return df
    
    def _process_stock_detail_data(self, data: List[Dict], flag: str) -> pd.DataFrame:
        """处理个股详情数据"""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df.reset_index(inplace=True)
        df["index"] = df.index + 1
        
        # 动态设置列名，适应不同的列数
        expected_cols = [
            "序号", "-", "-", "-", "-", "交易营业部名称", "类型", "-", "-", "-", "-",
            "买入金额", "卖出金额", "净额", "-", "-", "-", "-",
            "买入金额-占总成交比例", "卖出金额-占总成交比例"
        ]
        
        actual_cols = len(df.columns)
        if actual_cols >= len(expected_cols):
            # 如果实际列数更多，添加占位符
            extra_cols = [f"-{i}" for i in range(actual_cols - len(expected_cols))]
            df.columns = expected_cols + extra_cols
        else:
            # 如果实际列数较少，截取对应的列名
            df.columns = expected_cols[:actual_cols]
        
        # 选择需要的列
        df = df[[
            "序号", "交易营业部名称", "买入金额", "买入金额-占总成交比例",
            "卖出金额", "卖出金额-占总成交比例", "净额", "类型"
        ]]
        
        # 数据类型转换
        self._convert_numeric_columns(df, [
            "买入金额", "买入金额-占总成交比例", "卖出金额", "卖出金额-占总成交比例"
        ])
        
        # 按类型排序
        df.sort_values("类型", inplace=True)
        df.reset_index(inplace=True, drop=True)
        df["序号"] = range(1, len(df) + 1)
        
        return df
    
    def _convert_numeric_columns(self, df: pd.DataFrame, columns: List[str]):
        """转换数值列的数据类型"""
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据关联性验证和完整性检查"""
        if df.empty:
            return df
        
        # 去除重复数据
        if "代码" in df.columns and "上榜日" in df.columns:
            df = df.drop_duplicates(subset=["代码", "上榜日"], keep="last")
        
        # 数据完整性检查
        if "龙虎榜净买额" in df.columns:
            # 检查净买额是否等于买入额减去卖出额
            if "龙虎榜买入额" in df.columns and "龙虎榜卖出额" in df.columns:
                calculated_net = df["龙虎榜买入额"] - df["龙虎榜卖出额"]
                # 允许小的误差
                mask = abs(df["龙虎榜净买额"] - calculated_net) > 1000
                if mask.any():
                    logger.warning(f"发现 {mask.sum()} 条数据的净买额计算不一致")
        
        return df.reset_index(drop=True)
    
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
            # 测试获取少量龙虎榜数据
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
            request = DragonTigerRequest(
                data_type="detail",
                start_date=yesterday,
                end_date=yesterday
            )
            
            response = await self.get_dragon_tiger_detail(request)
            
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
async def test_dragon_tiger_adapter():
    """测试龙虎榜适配器"""
    print("🔍 测试龙虎榜数据适配器")
    print("=" * 50)
    
    adapter = DragonTigerAdapter()
    
    # 测试龙虎榜详情
    print("1. 测试龙虎榜详情...")
    detail_request = DragonTigerRequest(
        data_type="detail",
        start_date="20240101",
        end_date="20240110"
    )
    
    detail_response = await adapter.get_dragon_tiger_detail(detail_request)
    print(f"   成功: {detail_response.success}")
    print(f"   响应时间: {detail_response.response_time:.2f}秒")
    print(f"   数据行数: {len(detail_response.data)}")
    
    # 测试个股统计
    print("\n2. 测试个股统计...")
    stock_stat_request = DragonTigerRequest(
        data_type="stock_statistic",
        period="近一月"
    )
    
    stock_stat_response = await adapter.get_stock_statistic(stock_stat_request)
    print(f"   成功: {stock_stat_response.success}")
    print(f"   响应时间: {stock_stat_response.response_time:.2f}秒")
    print(f"   数据行数: {len(stock_stat_response.data)}")
    
    # 测试营业部排行
    print("\n3. 测试营业部排行...")
    dept_ranking_request = DragonTigerRequest(
        data_type="department_ranking",
        period="近一月"
    )
    
    dept_ranking_response = await adapter.get_department_ranking(dept_ranking_request)
    print(f"   成功: {dept_ranking_response.success}")
    print(f"   响应时间: {dept_ranking_response.response_time:.2f}秒")
    print(f"   数据行数: {len(dept_ranking_response.data)}")
    
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
    
    print("\n✅ 龙虎榜适配器测试完成!")


if __name__ == "__main__":
    asyncio.run(test_dragon_tiger_adapter())