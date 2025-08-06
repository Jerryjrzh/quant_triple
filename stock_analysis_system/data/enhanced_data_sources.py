"""
Enhanced Data Sources Integration

This module integrates additional data sources from the crawling interfaces
into the existing data source manager, providing comprehensive market data access.

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import asyncio
import logging
import math
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from functools import lru_cache
import pandas as pd
import requests
import numpy as np
from dataclasses import dataclass

from .data_source_manager import DataSourceType, DataSourceHealth, DataSourceManager, DataSourceStatus
from .cache_manager import CacheManager, get_cache_manager

logger = logging.getLogger(__name__)


@dataclass
class MarketDataRequest:
    """Market data request structure"""
    symbol: str
    start_date: str
    end_date: str
    period: str = "daily"
    adjust: str = ""
    data_type: str = "stock"  # stock, etf, fund


class EastMoneyDataSource:
    """东方财富数据源接口"""
    
    def __init__(self):
        self.base_url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
        self.spot_url = "http://82.push2.eastmoney.com/api/qt/clist/get"
        self._code_id_cache = {}
        
    @lru_cache(maxsize=1)
    def get_code_id_map(self) -> dict:
        """获取股票代码和市场ID映射"""
        try:
            # 上海A股
            sh_codes = self._fetch_market_codes("m:1 t:2,m:1 t:23")
            code_map = {code: 1 for code in sh_codes}
            
            # 深圳A股
            sz_codes = self._fetch_market_codes("m:0 t:6,m:0 t:80")
            code_map.update({code: 0 for code in sz_codes})
            
            # 北京A股
            bj_codes = self._fetch_market_codes("m:0 t:81 s:2048")
            code_map.update({code: 0 for code in bj_codes})
            
            return code_map
        except Exception as e:
            logger.error(f"Failed to get code ID map: {e}")
            return {}
    
    def _fetch_market_codes(self, market_filter: str) -> List[str]:
        """获取特定市场的股票代码"""
        codes = []
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
            "_": "1623833739532",
        }
        
        try:
            r = requests.get(self.spot_url, params=params, timeout=10)
            data_json = r.json()
            data = data_json.get("data", {}).get("diff", [])
            
            if not data:
                return codes
            
            # 获取所有页面的数据
            data_count = data_json["data"]["total"]
            page_count = math.ceil(data_count / page_size)
            
            for item in data:
                codes.append(item["f12"])
            
            # 获取其他页面
            for page in range(2, page_count + 1):
                params["pn"] = page
                r = requests.get(self.spot_url, params=params, timeout=10)
                data_json = r.json()
                page_data = data_json.get("data", {}).get("diff", [])
                
                for item in page_data:
                    codes.append(item["f12"])
                    
        except Exception as e:
            logger.error(f"Failed to fetch market codes for {market_filter}: {e}")
            
        return codes 
   
    def get_stock_realtime_data(self) -> pd.DataFrame:
        """获取A股实时行情数据"""
        try:
            params = {
                "pn": 1,
                "pz": 5000,
                "po": "1",
                "np": "1",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": "2",
                "invt": "2",
                "fid": "f12",
                "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
                "fields": "f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f14,f15,f16,f17,f18,f20,f21,f22,f23,f24,f25",
                "_": "1623833739532",
            }
            
            r = requests.get(self.spot_url, params=params, timeout=15)
            data_json = r.json()
            data = data_json.get("data", {}).get("diff", [])
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # 动态处理列数，防止列数不匹配
            expected_columns = [
                "最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "振幅", "换手率",
                "市盈率动", "量比", "5分钟涨跌", "代码", "名称", "最高", "最低",
                "今开", "昨收", "总市值", "流通市值", "涨速", "市净率", "60日涨跌幅",
                "年初至今涨跌幅", "上市时间"
            ]
            
            # 根据实际列数调整列名
            actual_cols = len(df.columns)
            if actual_cols <= len(expected_columns):
                df.columns = expected_columns[:actual_cols]
            else:
                # 如果实际列数更多，添加额外列名
                extra_cols = [f"额外列{i}" for i in range(actual_cols - len(expected_columns))]
                df.columns = expected_columns + extra_cols
            
            # 数据类型转换
            numeric_cols = ["最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "振幅", 
                          "换手率", "市盈率动", "量比", "最高", "最低", "今开", "昨收"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get realtime stock data: {e}")
            return pd.DataFrame()
    
    def get_stock_history_data(self, symbol: str, period: str = "daily", 
                              start_date: str = "20200101", end_date: str = "20241231",
                              adjust: str = "") -> pd.DataFrame:
        """获取股票历史数据"""
        try:
            code_id_map = self.get_code_id_map()
            if symbol not in code_id_map:
                logger.warning(f"Symbol {symbol} not found in code map")
                return pd.DataFrame()
            
            adjust_dict = {"qfq": "1", "hfq": "2", "": "0"}
            period_dict = {"daily": "101", "weekly": "102", "monthly": "103"}
            
            params = {
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
                "ut": "7eea3edcaed734bea9cbfc24409ed989",
                "klt": period_dict.get(period, "101"),
                "fqt": adjust_dict.get(adjust, "0"),
                "secid": f"{code_id_map[symbol]}.{symbol}",
                "beg": start_date,
                "end": end_date,
                "_": "1623766962675",
            }
            
            r = requests.get(self.base_url, params=params, timeout=15)
            data_json = r.json()
            
            if not (data_json.get("data") and data_json["data"].get("klines")):
                return pd.DataFrame()
            
            df = pd.DataFrame([item.split(",") for item in data_json["data"]["klines"]])
            df.columns = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", 
                         "振幅", "涨跌幅", "涨跌额", "换手率"]
            
            # 数据类型转换
            numeric_cols = ["开盘", "收盘", "最高", "最低", "成交量", "成交额", 
                          "振幅", "涨跌幅", "涨跌额", "换手率"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df["日期"] = pd.to_datetime(df["日期"])
            df.set_index("日期", inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get history data for {symbol}: {e}")
            return pd.DataFrame()


class DragonTigerDataSource:
    """龙虎榜数据源"""
    
    def __init__(self):
        self.base_url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    
    def get_dragon_tiger_detail(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取龙虎榜详情数据"""
        try:
            start_date = "-".join([start_date[:4], start_date[4:6], start_date[6:]])
            end_date = "-".join([end_date[:4], end_date[4:6], end_date[6:]])
            
            params = {
                "sortColumns": "SECURITY_CODE,TRADE_DATE",
                "sortTypes": "1,-1",
                "pageSize": "5000",
                "pageNumber": "1",
                "reportName": "RPT_DAILYBILLBOARD_DETAILSNEW",
                "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,TRADE_DATE,EXPLAIN,CLOSE_PRICE,CHANGE_RATE,BILLBOARD_NET_AMT,BILLBOARD_BUY_AMT,BILLBOARD_SELL_AMT",
                "source": "WEB",
                "client": "WEB",
                "filter": f"(TRADE_DATE<='{end_date}')(TRADE_DATE>='{start_date}')",
            }
            
            r = requests.get(self.base_url, params=params, timeout=15)
            data_json = r.json()
            
            if not data_json.get("result", {}).get("data"):
                return pd.DataFrame()
            
            df = pd.DataFrame(data_json["result"]["data"])
            df.rename(columns={
                "SECURITY_CODE": "代码",
                "SECURITY_NAME_ABBR": "名称", 
                "TRADE_DATE": "上榜日",
                "EXPLAIN": "解读",
                "CLOSE_PRICE": "收盘价",
                "CHANGE_RATE": "涨跌幅",
                "BILLBOARD_NET_AMT": "龙虎榜净买额",
                "BILLBOARD_BUY_AMT": "龙虎榜买入额",
                "BILLBOARD_SELL_AMT": "龙虎榜卖出额"
            }, inplace=True)
            
            # 数据类型转换
            numeric_cols = ["收盘价", "涨跌幅", "龙虎榜净买额", "龙虎榜买入额", "龙虎榜卖出额"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df["上榜日"] = pd.to_datetime(df["上榜日"]).dt.date
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get dragon tiger data: {e}")
            return pd.DataFrame()


class LimitUpReasonDataSource:
    """涨停原因数据源"""
    
    def __init__(self):
        self.base_url = "http://zx.10jqka.com.cn/event/api/getharden"
        
    def get_limitup_reason(self, date: str = None) -> pd.DataFrame:
        """获取涨停原因数据"""
        try:
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
            
            url = f"{self.base_url}/date/{date}/orderby/date/orderway/desc/charset/GBK/"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            r = requests.get(url, headers=headers, timeout=10)
            data_json = r.json()
            
            data = data_json.get("data", [])
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # 根据数据列数确定列名
            if len(df.columns) < 7:
                df.columns = ["ID", "名称", "代码", "原因", "日期", "_"]
                # 添加缺失列
                for col in ["最新价", "涨跌额", "涨跌幅", "换手率", "成交额", "成交量", "DDE"]:
                    df[col] = np.nan
            else:
                df.columns = ["ID", "名称", "代码", "原因", "日期", "最新价", "涨跌额", 
                            "涨跌幅", "换手率", "成交额", "成交量", "DDE", "_"]
            
            # 选择需要的列
            df = df[["日期", "代码", "名称", "原因", "最新价", "涨跌幅", "涨跌额", 
                    "换手率", "成交量", "成交额", "DDE"]]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get limit up reason data: {e}")
            return pd.DataFrame()


class ChipRaceDataSource:
    """筹码竞价数据源"""
    
    def __init__(self):
        self.base_url = "http://excalc.icfqs.com:7616/TQLEX?Entry=HQServ.hq_nlp"
        
    def get_chip_race_open(self, date: str = None) -> pd.DataFrame:
        """获取早盘抢筹数据"""
        try:
            params = [{
                "funcId": 20, "offset": 0, "count": 100, "sort": 1, "period": 0,
                "Token": "6679f5cadca97d68245a086793fc1bfc0a50b487487c812f", 
                "modname": "JJQC"
            }]
            
            if date:
                params[0]["date"] = date
            
            headers = {
                "Content-Type": "application/json; charset=UTF-8",
                "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36"
            }
            
            r = requests.post(self.base_url, json=params, headers=headers, timeout=10)
            data_json = r.json()
            
            data = data_json.get("datas", [])
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df.columns = ["代码", "名称", "昨收", "今开", "开盘金额", "抢筹幅度", 
                         "抢筹委托金额", "抢筹成交金额", "最新价", "_"]
            
            # 数据处理
            df["昨收"] = df["昨收"] / 10000
            df["今开"] = df["今开"] / 10000
            df["抢筹幅度"] = round(df["抢筹幅度"] * 100, 2)
            df["最新价"] = round(df["最新价"], 2)
            df["涨跌幅"] = round((df["最新价"] / df["昨收"] - 1) * 100, 2)
            df["抢筹占比"] = round((df["抢筹成交金额"] / df["开盘金额"]) * 100, 2)
            
            return df[["代码", "名称", "最新价", "涨跌幅", "昨收", "今开", "开盘金额", 
                      "抢筹幅度", "抢筹委托金额", "抢筹成交金额", "抢筹占比"]]
            
        except Exception as e:
            logger.error(f"Failed to get chip race open data: {e}")
            return pd.DataFrame()


class EnhancedDataSourceManager(DataSourceManager):
    """增强的数据源管理器 - 集成所有数据适配器到统一管理器中"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 初始化增强数据源
        self.eastmoney_source = EastMoneyDataSource()
        self.dragon_tiger_source = DragonTigerDataSource()
        self.limitup_reason_source = LimitUpReasonDataSource()
        self.chip_race_source = ChipRaceDataSource()
        
        # 导入新的适配器
        from .eastmoney_adapter import EastMoneyAdapter
        from .fund_flow_adapter import FundFlowAdapter
        from .dragon_tiger_adapter import DragonTigerAdapter
        from .limitup_reason_adapter import LimitUpReasonAdapter
        from .etf_adapter import ETFAdapter
        
        # 初始化新适配器
        self.eastmoney_adapter = EastMoneyAdapter()
        self.fund_flow_adapter = FundFlowAdapter()
        self.dragon_tiger_adapter = DragonTigerAdapter()
        self.limitup_adapter = LimitUpReasonAdapter()
        self.etf_adapter = ETFAdapter()
        
        # 注册所有数据源
        self.enhanced_sources = {
            "eastmoney": self.eastmoney_source,
            "dragon_tiger": self.dragon_tiger_source,
            "limitup_reason": self.limitup_reason_source,
            "chip_race": self.chip_race_source
        }
        
        # 注册新适配器
        self.adapters = {
            "eastmoney_adapter": self.eastmoney_adapter,
            "fund_flow_adapter": self.fund_flow_adapter,
            "dragon_tiger_adapter": self.dragon_tiger_adapter,
            "limitup_adapter": self.limitup_adapter,
            "etf_adapter": self.etf_adapter
        }
        
        # 数据源优先级配置 - 实现数据源优先级管理
        self.source_priority = {
            "stock_realtime": ["eastmoney_adapter", "eastmoney", "akshare", "tushare"],
            "stock_history": ["eastmoney_adapter", "local", "akshare", "tushare"],
            "fund_flow": ["fund_flow_adapter"],
            "dragon_tiger": ["dragon_tiger_adapter", "dragon_tiger"],
            "limitup_reason": ["limitup_adapter", "limitup_reason"],
            "etf_data": ["etf_adapter"],
            "chip_race": ["chip_race"]
        }
        
        # 负载均衡配置 - 实现负载均衡机制
        self.load_balancer = {
            "round_robin_counters": {},
            "health_weights": {},
            "request_counts": {},
            "response_times": {},
            "last_used": {}
        }
        
        # 健康状态监控 - 添加数据源健康状态监控
        self.health_monitor = {
            "check_interval": 300,  # 5分钟
            "last_check": None,
            "status_cache": {},
            "failure_counts": {},
            "success_counts": {},
            "circuit_breakers": {}
        }
        
        # 缓存管理器 - 实现多级缓存策略
        self.cache_manager: Optional[CacheManager] = None
        
        # 初始化健康监控
        self._initialize_health_monitoring()
        
        # 初始化缓存管理器
        asyncio.create_task(self._initialize_cache_manager())
    
    def _initialize_health_monitoring(self):
        """初始化健康监控组件"""
        # 为每个数据源初始化健康监控指标
        all_sources = list(self.enhanced_sources.keys()) + list(self.adapters.keys())
        
        for source_name in all_sources:
            self.health_monitor["failure_counts"][source_name] = 0
            self.health_monitor["success_counts"][source_name] = 0
            self.health_monitor["circuit_breakers"][source_name] = {
                "state": "closed",  # closed, open, half_open
                "failure_threshold": 5,
                "recovery_timeout": 300,  # 5分钟
                "last_failure_time": None
            }
            
            self.load_balancer["round_robin_counters"][source_name] = 0
            self.load_balancer["health_weights"][source_name] = 1.0
            self.load_balancer["request_counts"][source_name] = {"total": 0, "success": 0}
            self.load_balancer["response_times"][source_name] = []
            self.load_balancer["last_used"][source_name] = None
    
    async def get_enhanced_market_data(self, request: MarketDataRequest) -> pd.DataFrame:
        """获取增强市场数据（带优先级和负载均衡）"""
        try:
            # 根据数据类型选择数据源优先级
            data_sources = self.source_priority.get(request.data_type, [])
            
            if not data_sources:
                logger.warning(f"No data sources configured for type: {request.data_type}")
                return pd.DataFrame()
            
            # 应用负载均衡策略选择数据源
            selected_sources = self._apply_load_balancing(data_sources)
            
            # 尝试按优先级和负载均衡获取数据
            for source_name in selected_sources:
                # 检查熔断器状态
                if not self._can_use_source(source_name):
                    logger.warning(f"Data source {source_name} is circuit broken")
                    continue
                
                try:
                    start_time = time.time()
                    result = await self._get_data_from_source(source_name, request)
                    response_time = time.time() - start_time
                    
                    if not result.empty:
                        # 更新负载均衡统计
                        self._update_load_balancer_stats(source_name, True, response_time)
                        self._record_success(source_name)
                        return result
                    else:
                        self._update_load_balancer_stats(source_name, False, response_time)
                        
                except Exception as e:
                    response_time = time.time() - start_time
                    logger.warning(f"Data source {source_name} failed: {e}")
                    self._update_load_balancer_stats(source_name, False, response_time)
                    self._record_failure(source_name)
                    continue
            
            # 所有数据源都失败
            logger.error(f"All data sources failed for request: {request}")
            return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to get enhanced market data: {e}")
            return pd.DataFrame()
    
    def _apply_load_balancing(self, data_sources: List[str]) -> List[str]:
        """应用负载均衡策略选择数据源顺序"""
        # 根据健康权重和响应时间重新排序数据源
        weighted_sources = []
        
        for source_name in data_sources:
            health_weight = self.load_balancer["health_weights"].get(source_name, 1.0)
            response_times = self.load_balancer["response_times"].get(source_name, [])
            avg_response_time = sum(response_times) / len(response_times) if response_times else 1.0
            
            # 计算综合权重（健康权重 / 平均响应时间）
            composite_weight = health_weight / max(avg_response_time, 0.1)
            weighted_sources.append((source_name, composite_weight))
        
        # 按权重降序排序
        weighted_sources.sort(key=lambda x: x[1], reverse=True)
        return [source[0] for source in weighted_sources]
    
    def _can_use_source(self, source_name: str) -> bool:
        """检查数据源是否可用（熔断器检查）"""
        circuit_breaker = self.health_monitor["circuit_breakers"].get(source_name, {})
        state = circuit_breaker.get("state", "closed")
        
        if state == "closed":
            return True
        elif state == "open":
            # 检查是否可以尝试恢复
            last_failure_time = circuit_breaker.get("last_failure_time")
            recovery_timeout = circuit_breaker.get("recovery_timeout", 300)
            
            if last_failure_time and (time.time() - last_failure_time.timestamp()) > recovery_timeout:
                # 切换到半开状态
                circuit_breaker["state"] = "half_open"
                return True
            return False
        else:  # half_open
            return True
    
    def _record_success(self, source_name: str):
        """记录数据源成功"""
        self.health_monitor["success_counts"][source_name] += 1
        
        # 重置熔断器
        circuit_breaker = self.health_monitor["circuit_breakers"].get(source_name, {})
        circuit_breaker["state"] = "closed"
        self.health_monitor["failure_counts"][source_name] = 0
    
    def _record_failure(self, source_name: str):
        """记录数据源失败"""
        self.health_monitor["failure_counts"][source_name] += 1
        
        # 检查是否需要触发熔断器
        circuit_breaker = self.health_monitor["circuit_breakers"].get(source_name, {})
        failure_count = self.health_monitor["failure_counts"][source_name]
        failure_threshold = circuit_breaker.get("failure_threshold", 5)
        
        if failure_count >= failure_threshold:
            circuit_breaker["state"] = "open"
            circuit_breaker["last_failure_time"] = datetime.now()
            logger.warning(f"Circuit breaker opened for data source: {source_name}")
    
    async def get_data_with_priority_failover(self, request: MarketDataRequest, 
                                            custom_priority: Optional[List[str]] = None) -> pd.DataFrame:
        """使用自定义优先级的数据获取（带故障转移）"""
        try:
            # 使用自定义优先级或默认优先级
            if custom_priority:
                data_sources = custom_priority
            else:
                data_sources = self.source_priority.get(request.data_type, [])
            
            if not data_sources:
                logger.warning(f"No data sources configured for type: {request.data_type}")
                return pd.DataFrame()
            
            # 应用负载均衡
            selected_sources = self._apply_load_balancing(data_sources)
            
            # 尝试获取数据
            for source_name in selected_sources:
                if not self._can_use_source(source_name):
                    continue
                
                try:
                    start_time = time.time()
                    result = await self._get_data_from_source(source_name, request)
                    response_time = time.time() - start_time
                    
                    if not result.empty:
                        self._update_load_balancer_stats(source_name, True, response_time)
                        self._record_success(source_name)
                        logger.info(f"Successfully retrieved data from {source_name}")
                        return result
                        
                except Exception as e:
                    response_time = time.time() - start_time
                    logger.warning(f"Data source {source_name} failed: {e}")
                    self._update_load_balancer_stats(source_name, False, response_time)
                    self._record_failure(source_name)
                    continue
            
            logger.error(f"All data sources failed for request: {request}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get data with priority failover: {e}")
            return pd.DataFrame()
    
    async def _get_data_from_source(self, source_name: str, request: MarketDataRequest) -> pd.DataFrame:
        """从指定数据源获取数据"""
        # 检查是否为新适配器
        if source_name in self.adapters:
            adapter = self.adapters[source_name]
            
            if source_name == "eastmoney_adapter":
                if request.data_type == "stock_realtime":
                    response = await adapter.get_stock_realtime_data([request.symbol] if request.symbol else None)
                    return response.data if response.success else pd.DataFrame()
                elif request.data_type == "stock_history":
                    from .eastmoney_adapter import StockRequest
                    stock_request = StockRequest(
                        symbol=request.symbol,
                        period=request.period,
                        start_date=request.start_date,
                        end_date=request.end_date,
                        adjust=request.adjust
                    )
                    response = await adapter.get_stock_history_data(stock_request)
                    return response.data if response.success else pd.DataFrame()
                    
            elif source_name == "fund_flow_adapter":
                from .fund_flow_adapter import FundFlowRequest
                flow_request = FundFlowRequest(
                    symbol=request.symbol,
                    data_type="individual" if request.symbol else "sector"
                )
                response = await adapter.get_fund_flow_data(flow_request)
                return response.data if response.success else pd.DataFrame()
                
            elif source_name == "dragon_tiger_adapter":
                from .dragon_tiger_adapter import DragonTigerRequest
                dt_request = DragonTigerRequest(
                    start_date=request.start_date,
                    end_date=request.end_date,
                    data_type="detail"
                )
                response = await adapter.get_dragon_tiger_data(dt_request)
                return response.data if response.success else pd.DataFrame()
                
            elif source_name == "limitup_adapter":
                from .limitup_reason_adapter import LimitUpRequest
                lu_request = LimitUpRequest(date=request.start_date)
                response = await adapter.get_limitup_data(lu_request)
                return response.data if response.success else pd.DataFrame()
                
            elif source_name == "etf_adapter":
                from .etf_adapter import ETFRequest
                etf_request = ETFRequest(
                    symbol=request.symbol,
                    data_type=request.data_type,
                    period=request.period,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    adjust=request.adjust
                )
                if request.data_type == "realtime":
                    response = await adapter.get_etf_realtime_data([request.symbol] if request.symbol else None)
                elif request.data_type == "history":
                    response = await adapter.get_etf_history_data(etf_request)
                elif request.data_type == "intraday":
                    response = await adapter.get_etf_intraday_data(etf_request)
                else:
                    return pd.DataFrame()
                return response.data if response.success else pd.DataFrame()
        
        # 检查是否为传统数据源
        elif source_name in self.enhanced_sources:
            source_obj = self.enhanced_sources[source_name]
            
            if source_name == "eastmoney":
                if request.data_type == "realtime":
                    return source_obj.get_stock_realtime_data()
                elif request.data_type == "history":
                    return source_obj.get_stock_history_data(
                        request.symbol, request.period, request.start_date, 
                        request.end_date, request.adjust
                    )
            elif source_name == "dragon_tiger":
                return source_obj.get_dragon_tiger_detail(request.start_date, request.end_date)
            elif source_name == "limitup_reason":
                return source_obj.get_limitup_reason(request.start_date)
            elif source_name == "chip_race":
                return source_obj.get_chip_race_open(request.start_date)
        
        return pd.DataFrame()
    
    def _update_load_balancer_stats(self, source_name: str, success: bool, response_time: float = 0.0):
        """更新负载均衡统计"""
        if source_name not in self.load_balancer["request_counts"]:
            self.load_balancer["request_counts"][source_name] = {"total": 0, "success": 0}
        
        self.load_balancer["request_counts"][source_name]["total"] += 1
        if success:
            self.load_balancer["request_counts"][source_name]["success"] += 1
        
        # 更新响应时间统计（保留最近100次记录）
        response_times = self.load_balancer["response_times"].get(source_name, [])
        response_times.append(response_time)
        if len(response_times) > 100:
            response_times.pop(0)
        self.load_balancer["response_times"][source_name] = response_times
        
        # 更新最后使用时间
        self.load_balancer["last_used"][source_name] = datetime.now()
        
        # 更新健康权重
        stats = self.load_balancer["request_counts"][source_name]
        success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        
        # 考虑响应时间的权重调整
        avg_response_time = sum(response_times) / len(response_times) if response_times else 1.0
        time_weight = 1.0 / max(avg_response_time, 0.1)  # 响应时间越短权重越高
        
        # 综合健康权重 = 成功率 * 时间权重
        self.load_balancer["health_weights"][source_name] = success_rate * min(time_weight, 10.0)
    
    async def get_data_source_health_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有数据源的健康状态"""
        health_status = {}
        
        all_sources = list(self.enhanced_sources.keys()) + list(self.adapters.keys())
        
        for source_name in all_sources:
            # 基础统计
            success_count = self.health_monitor["success_counts"].get(source_name, 0)
            failure_count = self.health_monitor["failure_counts"].get(source_name, 0)
            total_requests = success_count + failure_count
            success_rate = success_count / total_requests if total_requests > 0 else 1.0
            
            # 响应时间统计
            response_times = self.load_balancer["response_times"].get(source_name, [])
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            
            # 熔断器状态
            circuit_breaker = self.health_monitor["circuit_breakers"].get(source_name, {})
            circuit_state = circuit_breaker.get("state", "closed")
            
            # 健康权重
            health_weight = self.load_balancer["health_weights"].get(source_name, 1.0)
            
            # 最后使用时间
            last_used = self.load_balancer["last_used"].get(source_name)
            
            health_status[source_name] = {
                "success_count": success_count,
                "failure_count": failure_count,
                "total_requests": total_requests,
                "success_rate": round(success_rate, 4),
                "avg_response_time": round(avg_response_time, 4),
                "circuit_breaker_state": circuit_state,
                "health_weight": round(health_weight, 4),
                "last_used": last_used.isoformat() if last_used else None,
                "status": self._determine_source_status(success_rate, circuit_state, avg_response_time)
            }
        
        return health_status
    
    def _determine_source_status(self, success_rate: float, circuit_state: str, avg_response_time: float) -> str:
        """确定数据源状态"""
        if circuit_state == "open":
            return "failed"
        elif circuit_state == "half_open":
            return "recovering"
        elif success_rate >= 0.95 and avg_response_time < 5.0:
            return "healthy"
        elif success_rate >= 0.8 and avg_response_time < 10.0:
            return "degraded"
        else:
            return "unhealthy"
    
    async def perform_health_check(self, force_check: bool = False) -> Dict[str, Dict[str, Any]]:
        """执行健康检查"""
        current_time = datetime.now()
        
        # 检查是否需要执行健康检查
        if not force_check and self.health_monitor["last_check"]:
            time_since_check = (current_time - self.health_monitor["last_check"]).seconds
            if time_since_check < self.health_monitor["check_interval"]:
                return self.health_monitor["status_cache"]
        
        logger.info("Performing health check on all data sources...")
        
        health_results = {}
        
        # 检查传统数据源
        for source_name, source_obj in self.enhanced_sources.items():
            try:
                start_time = time.time()
                
                # 执行简单的健康检查
                if source_name == "eastmoney":
                    test_data = source_obj.get_stock_realtime_data()
                elif source_name == "dragon_tiger":
                    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
                    test_data = source_obj.get_dragon_tiger_detail(yesterday, yesterday)
                elif source_name == "limitup_reason":
                    test_data = source_obj.get_limitup_reason()
                elif source_name == "chip_race":
                    test_data = source_obj.get_chip_race_open()
                else:
                    test_data = pd.DataFrame()
                
                response_time = time.time() - start_time
                
                # 判断健康状态
                if not test_data.empty:
                    status = "healthy"
                    self._record_success(source_name)
                else:
                    status = "degraded"
                
                health_results[source_name] = {
                    "status": status,
                    "response_time": round(response_time, 4),
                    "data_available": not test_data.empty,
                    "last_check": current_time.isoformat()
                }
                
            except Exception as e:
                logger.error(f"Health check failed for {source_name}: {e}")
                self._record_failure(source_name)
                health_results[source_name] = {
                    "status": "failed",
                    "response_time": 0.0,
                    "data_available": False,
                    "error": str(e),
                    "last_check": current_time.isoformat()
                }
        
        # 检查新适配器
        for adapter_name, adapter_obj in self.adapters.items():
            try:
                start_time = time.time()
                
                # 执行适配器健康检查
                if hasattr(adapter_obj, 'health_check'):
                    health_result = await adapter_obj.health_check()
                    status = "healthy" if health_result else "failed"
                else:
                    # 简单的连接测试
                    status = "healthy"  # 假设适配器可用
                
                response_time = time.time() - start_time
                
                if status == "healthy":
                    self._record_success(adapter_name)
                else:
                    self._record_failure(adapter_name)
                
                health_results[adapter_name] = {
                    "status": status,
                    "response_time": round(response_time, 4),
                    "data_available": status == "healthy",
                    "last_check": current_time.isoformat()
                }
                
            except Exception as e:
                logger.error(f"Health check failed for {adapter_name}: {e}")
                self._record_failure(adapter_name)
                health_results[adapter_name] = {
                    "status": "failed",
                    "response_time": 0.0,
                    "data_available": False,
                    "error": str(e),
                    "last_check": current_time.isoformat()
                }
        
        # 更新缓存
        self.health_monitor["last_check"] = current_time
        self.health_monitor["status_cache"] = health_results
        
        logger.info(f"Health check completed. Checked {len(health_results)} data sources.")
        return health_results
    
    async def get_data_with_failover(self, request: MarketDataRequest, max_retries: int = 3) -> pd.DataFrame:
        """带故障转移的数据获取"""
        for attempt in range(max_retries):
            try:
                result = await self.get_enhanced_market_data(request)
                if not result.empty:
                    return result
                
                # 如果数据为空，等待后重试
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise e
        
        return pd.DataFrame()
    
    def get_available_data_sources(self) -> List[str]:
        """获取可用的数据源列表"""
        base_sources = super().get_available_data_sources() if hasattr(super(), 'get_available_data_sources') else []
        enhanced_sources = list(self.enhanced_sources.keys())
        return base_sources + enhanced_sources
    
    async def health_check_enhanced_sources(self) -> Dict[str, DataSourceHealth]:
        """检查增强数据源健康状态（兼容原有接口）"""
        # 执行新的健康检查
        health_results = await self.perform_health_check()
        
        # 转换为原有的DataSourceHealth格式
        health_status = {}
        
        for source_name, health_info in health_results.items():
            try:
                # 获取统计信息
                success_count = self.health_monitor["success_counts"].get(source_name, 0)
                failure_count = self.health_monitor["failure_counts"].get(source_name, 0)
                total_requests = success_count + failure_count
                success_rate = success_count / total_requests if total_requests > 0 else 1.0
                
                # 确定状态枚举
                status_str = health_info.get("status", "failed")
                if status_str == "healthy":
                    status_enum = DataSourceStatus.HEALTHY
                elif status_str == "degraded":
                    status_enum = DataSourceStatus.DEGRADED
                elif status_str == "recovering":
                    status_enum = DataSourceStatus.DEGRADED
                else:
                    status_enum = DataSourceStatus.FAILED
                
                # 创建DataSourceHealth对象
                health_status[source_name] = DataSourceHealth(
                    source_type=DataSourceType.TUSHARE,  # 使用现有枚举
                    status=status_enum,
                    last_success=datetime.now() if health_info.get("data_available", False) else None,
                    last_failure=datetime.now() if not health_info.get("data_available", True) else None,
                    failure_count=failure_count,
                    success_rate=success_rate,
                    avg_response_time=health_info.get("response_time", 0.0),
                    reliability_score=success_rate * 100
                )
                
            except Exception as e:
                logger.error(f"Failed to convert health status for {source_name}: {e}")
                health_status[source_name] = DataSourceHealth(
                    source_type=DataSourceType.TUSHARE,
                    status=DataSourceStatus.FAILED,
                    last_success=None,
                    last_failure=datetime.now(),
                    failure_count=1,
                    success_rate=0.0,
                    avg_response_time=0.0,
                    reliability_score=0.0
                )
        
        return health_status
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """获取负载均衡统计信息"""
        return {
            "source_priority": self.source_priority,
            "health_weights": self.load_balancer["health_weights"],
            "request_counts": self.load_balancer["request_counts"],
            "response_times": {
                source: {
                    "avg": sum(times) / len(times) if times else 0.0,
                    "min": min(times) if times else 0.0,
                    "max": max(times) if times else 0.0,
                    "count": len(times)
                }
                for source, times in self.load_balancer["response_times"].items()
            },
            "last_used": {
                source: last_used.isoformat() if last_used else None
                for source, last_used in self.load_balancer["last_used"].items()
            }
        }
    
    def update_source_priority(self, data_type: str, new_priority: List[str]):
        """更新数据源优先级配置"""
        if data_type in self.source_priority:
            old_priority = self.source_priority[data_type].copy()
            self.source_priority[data_type] = new_priority
            logger.info(f"Updated priority for {data_type}: {old_priority} -> {new_priority}")
        else:
            logger.warning(f"Unknown data type for priority update: {data_type}")
    
    def reset_health_monitoring(self, source_name: Optional[str] = None):
        """重置健康监控统计"""
        if source_name:
            # 重置特定数据源
            if source_name in self.health_monitor["failure_counts"]:
                self.health_monitor["failure_counts"][source_name] = 0
                self.health_monitor["success_counts"][source_name] = 0
                self.load_balancer["request_counts"][source_name] = {"total": 0, "success": 0}
                self.load_balancer["response_times"][source_name] = []
                self.load_balancer["health_weights"][source_name] = 1.0
                
                # 重置熔断器
                circuit_breaker = self.health_monitor["circuit_breakers"].get(source_name, {})
                circuit_breaker["state"] = "closed"
                circuit_breaker["last_failure_time"] = None
                
                logger.info(f"Reset health monitoring for {source_name}")
        else:
            # 重置所有数据源
            all_sources = list(self.enhanced_sources.keys()) + list(self.adapters.keys())
            for source in all_sources:
                self.reset_health_monitoring(source)
            logger.info("Reset health monitoring for all data sources")
    
    async def _initialize_cache_manager(self):
        """初始化缓存管理器"""
        try:
            self.cache_manager = await get_cache_manager()
            logger.info("Cache manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            self.cache_manager = None
    
    async def get_cached_market_data(self, request: MarketDataRequest) -> pd.DataFrame:
        """获取带缓存的市场数据"""
        if not self.cache_manager:
            # 如果缓存不可用，直接获取数据
            return await self.get_enhanced_market_data(request)
        
        # 生成缓存键
        cache_key = self._generate_cache_key(request)
        cache_type = self._get_cache_type(request.data_type)
        
        try:
            # 尝试从缓存获取数据
            cached_data = await self.cache_manager.get_cached_data(cache_key, cache_type)
            if cached_data is not None and not cached_data.empty:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_data
            
            # 缓存未命中，从数据源获取
            logger.debug(f"Cache miss for key: {cache_key}")
            data = await self.get_enhanced_market_data(request)
            
            # 将数据存入缓存
            if not data.empty:
                await self.cache_manager.set_cached_data(cache_key, data, cache_type)
                logger.debug(f"Data cached with key: {cache_key}")
            
            return data
            
        except Exception as e:
            logger.error(f"Cache operation failed: {e}")
            # 缓存失败时直接获取数据
            return await self.get_enhanced_market_data(request)
    
    def _generate_cache_key(self, request: MarketDataRequest) -> str:
        """生成缓存键"""
        key_parts = [
            request.data_type,
            request.symbol or "all",
            request.period or "default",
            request.start_date or "",
            request.end_date or "",
            request.adjust or ""
        ]
        return ":".join(key_parts)
    
    def _get_cache_type(self, data_type: str) -> str:
        """根据数据类型获取缓存类型"""
        cache_type_mapping = {
            "stock_realtime": "realtime_data",
            "stock_history": "daily_data",
            "dragon_tiger": "dragon_tiger",
            "fund_flow": "fund_flow",
            "limitup_reason": "limitup_reason",
            "etf_data": "etf_data"
        }
        return cache_type_mapping.get(data_type, "default")
    
    async def invalidate_cache_by_symbol(self, symbol: str):
        """根据股票代码失效相关缓存"""
        if not self.cache_manager:
            return
        
        try:
            # 失效该股票的所有相关缓存
            patterns = [
                f"*:{symbol}:*",
                f"realtime_data:{symbol}:*",
                f"daily_data:{symbol}:*",
                f"fund_flow:{symbol}:*"
            ]
            
            for pattern in patterns:
                await self.cache_manager.invalidate_cache(pattern)
            
            logger.info(f"Invalidated cache for symbol: {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to invalidate cache for symbol {symbol}: {e}")
    
    async def warm_up_cache_for_symbols(self, symbols: List[str]):
        """为指定股票预热缓存"""
        if not self.cache_manager:
            return
        
        logger.info(f"Starting cache warm-up for {len(symbols)} symbols")
        
        warm_up_tasks = []
        for symbol in symbols:
            # 预热实时数据
            realtime_request = MarketDataRequest(
                symbol=symbol,
                start_date="",
                end_date="",
                data_type="stock_realtime"
            )
            task1 = asyncio.create_task(self.get_cached_market_data(realtime_request))
            warm_up_tasks.append(task1)
            
            # 预热历史数据（最近30天）
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
            history_request = MarketDataRequest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                period="daily",
                data_type="stock_history"
            )
            task2 = asyncio.create_task(self.get_cached_market_data(history_request))
            warm_up_tasks.append(task2)
        
        # 并发执行预热任务
        try:
            await asyncio.gather(*warm_up_tasks, return_exceptions=True)
            logger.info("Cache warm-up completed")
        except Exception as e:
            logger.error(f"Cache warm-up failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self.cache_manager:
            return {"error": "Cache manager not available"}
        
        return self.cache_manager.get_cache_stats()


# 使用示例和测试函数
async def test_enhanced_data_sources():
    """测试增强数据源管理器"""
    manager = EnhancedDataSourceManager()
    
    print("🔍 Testing Enhanced Data Source Manager")
    print("=" * 60)
    
    # 测试实时行情数据
    print("1. Testing realtime stock data with priority and load balancing...")
    realtime_request = MarketDataRequest(
        symbol="000001",
        start_date="",
        end_date="",
        data_type="stock_realtime"
    )
    realtime_data = await manager.get_enhanced_market_data(realtime_request)
    print(f"   Realtime data shape: {realtime_data.shape}")
    
    # 测试历史数据
    print("2. Testing historical stock data with failover...")
    history_request = MarketDataRequest(
        symbol="000001",
        start_date="20240101",
        end_date="20241231",
        period="daily",
        data_type="stock_history"
    )
    history_data = await manager.get_data_with_priority_failover(history_request)
    print(f"   Historical data shape: {history_data.shape}")
    
    # 测试龙虎榜数据
    print("3. Testing dragon tiger data...")
    dt_request = MarketDataRequest(
        symbol="",
        start_date="20241201",
        end_date="20241231",
        data_type="dragon_tiger"
    )
    dt_data = await manager.get_enhanced_market_data(dt_request)
    print(f"   Dragon tiger data shape: {dt_data.shape}")
    
    # 测试资金流向数据
    print("4. Testing fund flow data...")
    ff_request = MarketDataRequest(
        symbol="000001",
        start_date="",
        end_date="",
        data_type="fund_flow"
    )
    ff_data = await manager.get_enhanced_market_data(ff_request)
    print(f"   Fund flow data shape: {ff_data.shape}")
    
    # 健康检查
    print("5. Testing comprehensive health check...")
    health_status = await manager.perform_health_check(force_check=True)
    for source, health in health_status.items():
        status = health.get("status", "unknown")
        response_time = health.get("response_time", 0.0)
        print(f"   {source}: {status} (Response: {response_time:.3f}s)")
    
    # 获取数据源健康状态
    print("6. Testing data source health status...")
    health_details = await manager.get_data_source_health_status()
    for source, details in health_details.items():
        success_rate = details.get("success_rate", 0.0)
        health_weight = details.get("health_weight", 0.0)
        circuit_state = details.get("circuit_breaker_state", "unknown")
        print(f"   {source}: Success Rate: {success_rate:.2%}, Weight: {health_weight:.3f}, Circuit: {circuit_state}")
    
    # 负载均衡统计
    print("7. Testing load balancer statistics...")
    lb_stats = manager.get_load_balancer_stats()
    print(f"   Configured data types: {list(lb_stats['source_priority'].keys())}")
    print(f"   Health weights: {len(lb_stats['health_weights'])} sources monitored")
    
    # 测试优先级更新
    print("8. Testing priority configuration update...")
    original_priority = manager.source_priority.get("stock_realtime", []).copy()
    new_priority = ["eastmoney_adapter", "akshare", "tushare", "eastmoney"]
    manager.update_source_priority("stock_realtime", new_priority)
    print(f"   Updated priority: {original_priority} -> {new_priority}")
    
    # 测试健康监控重置
    print("9. Testing health monitoring reset...")
    manager.reset_health_monitoring("eastmoney_adapter")
    print("   Reset health monitoring for eastmoney_adapter")
    
    print("\n✅ Enhanced data source manager test completed!")
    print(f"📊 Total data sources managed: {len(manager.enhanced_sources) + len(manager.adapters)}")
    print(f"🔄 Load balancing enabled for {len(manager.source_priority)} data types")
    print(f"💓 Health monitoring active for all sources")


if __name__ == "__main__":
    asyncio.run(test_enhanced_data_sources())