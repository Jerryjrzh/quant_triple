"""
爬虫数据适配器单元测试

为所有数据适配器创建全面的单元测试，包括模拟数据源和网络请求的测试环境，
边界条件和异常情况的测试用例，以及测试数据生成器和验证工具。

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import asyncio
import json
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any
import requests

from stock_analysis_system.data.eastmoney_adapter import (
    EastMoneyAdapter, DataRequest, AdapterResponse
)
from stock_analysis_system.data.fund_flow_adapter import (
    FundFlowAdapter, FundFlowRequest, FundFlowResponse
)
from stock_analysis_system.data.dragon_tiger_adapter import (
    DragonTigerAdapter, DragonTigerRequest, DragonTigerResponse
)
from stock_analysis_system.data.limitup_reason_adapter import (
    LimitUpReasonAdapter, LimitUpRequest, LimitUpResponse
)
from stock_analysis_system.data.etf_adapter import (
    ETFAdapter, ETFRequest, ETFResponse
)


class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_eastmoney_realtime_response() -> Dict:
        """生成东方财富实时数据响应"""
        return {
            "data": {
                "total": 2,
                "diff": [
                    {
                        "f2": 10.50,  # 最新价
                        "f3": 2.50,   # 涨跌幅
                        "f4": 0.26,   # 涨跌额
                        "f5": 1000000,  # 成交量
                        "f6": 10500000,  # 成交额
                        "f12": "000001",  # 代码
                        "f14": "平安银行",  # 名称
                        "f15": 10.80,  # 最高
                        "f16": 10.20,  # 最低
                        "f17": 10.30,  # 今开
                        "f18": 10.24,  # 昨收
                    },
                    {
                        "f2": 15.20,
                        "f3": -1.30,
                        "f4": -0.20,
                        "f5": 800000,
                        "f6": 12160000,
                        "f12": "000002",
                        "f14": "万科A",
                        "f15": 15.50,
                        "f16": 15.00,
                        "f17": 15.40,
                        "f18": 15.40,
                    }
                ]
            }
        }
    
    @staticmethod
    def generate_eastmoney_history_response() -> Dict:
        """生成东方财富历史数据响应"""
        return {
            "data": {
                "klines": [
                    "2024-01-01,10.20,10.50,10.60,10.10,1000000,10500000,4.90,2.94,0.30,9.80",
                    "2024-01-02,10.50,10.30,10.70,10.20,1200000,12600000,4.76,-1.90,-0.20,11.76"
                ]
            }
        }
    
    @staticmethod
    def generate_fund_flow_response() -> Dict:
        """生成资金流向数据响应"""
        return {
            "data": {
                "total": 2,
                "diff": [
                    {
                        "f12": "000001",  # 代码
                        "f14": "平安银行",  # 名称
                        "f2": 10.50,  # 最新价
                        "f3": 2.50,   # 涨跌幅
                        "f62": 5000000,  # 主力净流入
                        "f184": 4.5,     # 主力净流入占比
                        "f66": 8000000,  # 超大单净流入
                        "f69": 7.2,      # 超大单净流入占比
                    }
                ]
            }
        }
    
    @staticmethod
    def generate_dragon_tiger_response() -> Dict:
        """生成龙虎榜数据响应"""
        return {
            "result": {
                "data": [
                    {
                        "SECURITY_CODE": "000001",
                        "SECURITY_NAME_ABBR": "平安银行",
                        "TRADE_DATE": "2024-01-01T00:00:00",
                        "CLOSE_PRICE": 10.50,
                        "CHANGE_RATE": 2.50,
                        "BILLBOARD_NET_AMT": 5000000,
                        "BILLBOARD_BUY_AMT": 8000000,
                        "BILLBOARD_SELL_AMT": 3000000,
                        "EXPLANATION": "日涨幅偏离值达7%的证券",
                    }
                ],
                "pages": 1
            }
        }
    
    @staticmethod
    def generate_limitup_response() -> Dict:
        """生成涨停原因数据响应"""
        return {
            "data": [
                [
                    "12345",      # ID
                    "平安银行",    # 名称
                    "000001",     # 代码
                    "政策利好",    # 原因
                    "2024-01-01", # 日期
                    10.50,        # 最新价
                    0.26,         # 涨跌额
                    2.50,         # 涨跌幅
                    9.80,         # 换手率
                    10500000,     # 成交额
                    1000000,      # 成交量
                    1.5,          # DDE
                    ""            # 占位符
                ]
            ]
        }
    
    @staticmethod
    def generate_etf_response() -> Dict:
        """生成ETF数据响应"""
        return {
            "data": {
                "total": 1,
                "diff": [
                    {
                        "f12": "159707",  # 代码
                        "f14": "中证500ETF",  # 名称
                        "f2": 1.250,      # 最新价
                        "f3": 1.20,       # 涨跌幅
                        "f4": 0.015,      # 涨跌额
                        "f5": 5000000,    # 成交量
                        "f6": 6250000,    # 成交额
                        "f17": 1.240,     # 开盘价
                        "f15": 1.255,     # 最高价
                        "f16": 1.235,     # 最低价
                        "f18": 1.235,     # 昨收
                    }
                ]
            }
        }


class TestEastMoneyAdapter:
    """东方财富适配器测试类"""
    
    @pytest.fixture
    def adapter(self):
        """创建适配器实例"""
        return EastMoneyAdapter(timeout=5, max_retries=2)
    
    @pytest.fixture
    def mock_response_data(self):
        """模拟响应数据"""
        return TestDataGenerator.generate_eastmoney_realtime_response()
    
    @pytest.mark.asyncio
    async def test_get_realtime_data_success(self, adapter, mock_response_data):
        """测试成功获取实时数据"""
        with patch.object(adapter, '_make_request', return_value=mock_response_data):
            response = await adapter.get_realtime_data(['000001', '000002'])
            
            assert response.success is True
            assert not response.data.empty
            assert len(response.data) == 2
            assert '代码' in response.data.columns
            assert '名称' in response.data.columns
            assert '最新价' in response.data.columns
            assert response.data_source == "eastmoney"
            assert response.response_time > 0
    
    @pytest.mark.asyncio
    async def test_get_realtime_data_empty_response(self, adapter):
        """测试空响应数据"""
        with patch.object(adapter, '_make_request', return_value=None):
            response = await adapter.get_realtime_data(['000001'])
            
            assert response.success is False
            assert response.data.empty
            assert "Failed to get response data" in response.error_message
    
    @pytest.mark.asyncio
    async def test_get_realtime_data_network_error(self, adapter):
        """测试网络错误"""
        with patch.object(adapter, '_make_request', side_effect=Exception("Network error")):
            response = await adapter.get_realtime_data(['000001'])
            
            assert response.success is False
            assert response.data.empty
            assert "Network error" in response.error_message
    
    @pytest.mark.asyncio
    async def test_get_history_data_success(self, adapter):
        """测试成功获取历史数据"""
        mock_code_map = {"000001": 0}
        mock_history_data = TestDataGenerator.generate_eastmoney_history_response()
        
        with patch.object(adapter, '_get_code_id_map', return_value=mock_code_map), \
             patch.object(adapter, '_make_request', return_value=mock_history_data):
            
            request = DataRequest(symbol="000001", start_date="20240101", end_date="20240131")
            response = await adapter.get_history_data(request)
            
            assert response.success is True
            assert not response.data.empty
            assert len(response.data) == 2
            assert '开盘' in response.data.columns
            assert '收盘' in response.data.columns
    
    @pytest.mark.asyncio
    async def test_get_history_data_symbol_not_found(self, adapter):
        """测试股票代码未找到"""
        mock_code_map = {}
        
        with patch.object(adapter, '_get_code_id_map', return_value=mock_code_map):
            request = DataRequest(symbol="999999")
            response = await adapter.get_history_data(request)
            
            assert response.success is False
            assert "Symbol 999999 not found in code map" in response.error_message
    
    @pytest.mark.asyncio
    async def test_get_intraday_data_success(self, adapter):
        """测试成功获取分时数据"""
        mock_code_map = {"000001": 0}
        mock_trends_data = {
            "data": {
                "trends": [
                    "2024-01-01 09:30:00,10.20,10.25,10.30,10.15,100000,1025000,10.25",
                    "2024-01-01 09:31:00,10.25,10.30,10.35,10.20,120000,1236000,10.30"
                ]
            }
        }
        
        with patch.object(adapter, '_get_code_id_map', return_value=mock_code_map), \
             patch.object(adapter, '_make_request', return_value=mock_trends_data):
            
            request = DataRequest(symbol="000001", period="1")
            response = await adapter.get_intraday_data(request)
            
            assert response.success is True
            assert not response.data.empty
            assert len(response.data) == 2
    
    def test_process_realtime_data(self, adapter):
        """测试实时数据处理"""
        mock_data = TestDataGenerator.generate_eastmoney_realtime_response()["data"]["diff"]
        df = adapter._process_realtime_data(mock_data)
        
        assert not df.empty
        assert len(df) == 2
        assert df.loc[0, '代码'] == '000001'
        assert df.loc[0, '名称'] == '平安银行'
        assert df.loc[0, '最新价'] == 10.50
        assert df.loc[1, '代码'] == '000002'
    
    def test_process_history_data(self, adapter):
        """测试历史数据处理"""
        klines = [
            "2024-01-01,10.20,10.50,10.60,10.10,1000000,10500000,4.90,2.94,0.30,9.80",
            "2024-01-02,10.50,10.30,10.70,10.20,1200000,12600000,4.76,-1.90,-0.20,11.76"
        ]
        df = adapter._process_history_data(klines)
        
        assert not df.empty
        assert len(df) == 2
        assert '开盘' in df.columns
        assert '收盘' in df.columns
        assert df.iloc[0]['开盘'] == 10.20
        assert df.iloc[0]['收盘'] == 10.50
    
    def test_error_statistics(self, adapter):
        """测试错误统计"""
        # 初始状态
        stats = adapter.get_error_statistics()
        assert stats['total_requests'] == 0
        assert stats['successful_requests'] == 0
        
        # 模拟一些错误
        adapter.error_stats['total_requests'] = 10
        adapter.error_stats['successful_requests'] = 8
        adapter.error_stats['network_errors'] = 1
        adapter.error_stats['api_errors'] = 1
        
        stats = adapter.get_error_statistics()
        assert stats['success_rate'] == 0.8
        assert stats['error_rate'] == 0.2
        
        # 重置统计
        adapter.reset_error_statistics()
        stats = adapter.get_error_statistics()
        assert stats['total_requests'] == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """测试健康检查"""
        mock_response = AdapterResponse(
            success=True,
            data=pd.DataFrame({'代码': ['000001'], '名称': ['平安银行']}),
            response_time=0.5
        )
        
        with patch.object(adapter, 'get_realtime_data', return_value=mock_response):
            health = await adapter.health_check()
            
            assert health['status'] == 'healthy'
            assert health['response_time'] == 0.5
            assert health['data_available'] is True
            assert 'timestamp' in health


class TestFundFlowAdapter:
    """资金流向适配器测试类"""
    
    @pytest.fixture
    def adapter(self):
        """创建适配器实例"""
        return FundFlowAdapter(timeout=5, max_retries=2)
    
    @pytest.mark.asyncio
    async def test_get_individual_fund_flow_success(self, adapter):
        """测试成功获取个股资金流向"""
        mock_data = TestDataGenerator.generate_fund_flow_response()["data"]["diff"]
        
        with patch.object(adapter, '_fetch_all_pages', return_value=mock_data):
            request = FundFlowRequest(
                data_type="individual",
                indicator="今日",
                symbols=["000001"]
            )
            response = await adapter.get_individual_fund_flow(request)
            
            assert response.success is True
            assert not response.data.empty
            assert response.data_type == "individual"
            assert response.indicator == "今日"
    
    @pytest.mark.asyncio
    async def test_get_individual_fund_flow_invalid_indicator(self, adapter):
        """测试无效指标"""
        request = FundFlowRequest(
            data_type="individual",
            indicator="无效指标"
        )
        response = await adapter.get_individual_fund_flow(request)
        
        assert response.success is False
        assert "不支持的指标" in response.error_message
    
    @pytest.mark.asyncio
    async def test_get_sector_fund_flow_success(self, adapter):
        """测试成功获取板块资金流向"""
        mock_data = [
            {
                "f12": "-",
                "f14": "银行",
                "f2": 2.5,
                "f62": 5000000,
                "f184": 4.5
            }
        ]
        
        with patch.object(adapter, '_fetch_sector_pages', return_value=mock_data):
            request = FundFlowRequest(
                data_type="sector",
                indicator="今日",
                sector_type="行业资金流"
            )
            response = await adapter.get_sector_fund_flow(request)
            
            assert response.success is True
            assert not response.data.empty
            assert response.data_type == "sector"
    
    @pytest.mark.asyncio
    async def test_get_sector_fund_flow_invalid_sector_type(self, adapter):
        """测试无效板块类型"""
        request = FundFlowRequest(
            data_type="sector",
            indicator="今日",
            sector_type="无效板块类型"
        )
        response = await adapter.get_sector_fund_flow(request)
        
        assert response.success is False
        assert "不支持的板块类型" in response.error_message
    
    @pytest.mark.asyncio
    async def test_get_multi_period_fund_flow(self, adapter):
        """测试多时间周期数据"""
        mock_response = FundFlowResponse(
            success=True,
            data=pd.DataFrame({'代码': ['000001'], '名称': ['平安银行']}),
            data_type="individual",
            indicator="今日"
        )
        
        with patch.object(adapter, 'get_individual_fund_flow', return_value=mock_response):
            results = await adapter.get_multi_period_fund_flow(['000001'])
            
            assert len(results) == 4  # 今日, 3日, 5日, 10日
            assert all(response.success for response in results.values())
    
    def test_process_individual_fund_flow_data(self, adapter):
        """测试个股资金流向数据处理"""
        mock_data = [
            {
                "f2": 10.50,
                "f3": 2.50,
                "f12": "000001",
                "f14": "平安银行",
                "f62": 5000000,
                "f184": 4.5
            }
        ]
        
        df = adapter._process_individual_fund_flow_data(mock_data, "今日")
        
        assert not df.empty
        assert '代码' in df.columns
        assert '名称' in df.columns
        assert '今日主力净流入-净额' in df.columns
    
    def test_convert_numeric_columns(self, adapter):
        """测试数值列转换"""
        df = pd.DataFrame({
            '代码': ['000001'],
            '名称': ['平安银行'],
            '今日主力净流入-净额': ['5000000'],
            '今日涨跌幅': ['2.50']
        })
        
        adapter._convert_numeric_columns(df, "今日")
        
        assert df['今日主力净流入-净额'].dtype in ['float64', 'int64']
        assert df['今日涨跌幅'].dtype in ['float64', 'int64']
    
    def test_deduplicate_and_sort(self, adapter):
        """测试去重和排序"""
        df = pd.DataFrame({
            '代码': ['000001', '000002', '000001'],
            '名称': ['平安银行', '万科A', '平安银行'],
            '数值': [1, 2, 3]
        })
        
        result_df = adapter._deduplicate_and_sort(df)
        
        assert len(result_df) == 2  # 去重后只有2条记录
        assert result_df.loc[result_df['代码'] == '000001', '数值'].iloc[0] == 3  # 保留最新数据


class TestDragonTigerAdapter:
    """龙虎榜适配器测试类"""
    
    @pytest.fixture
    def adapter(self):
        """创建适配器实例"""
        return DragonTigerAdapter(timeout=5, max_retries=2)
    
    @pytest.mark.asyncio
    async def test_get_dragon_tiger_detail_success(self, adapter):
        """测试成功获取龙虎榜详情"""
        mock_data = TestDataGenerator.generate_dragon_tiger_response()["result"]["data"]
        
        with patch.object(adapter, '_fetch_all_pages', return_value=mock_data):
            request = DragonTigerRequest(
                data_type="detail",
                start_date="20240101",
                end_date="20240131"
            )
            response = await adapter.get_dragon_tiger_detail(request)
            
            assert response.success is True
            assert not response.data.empty
            assert response.data_type == "detail"
            assert '代码' in response.data.columns
            assert '名称' in response.data.columns
    
    @pytest.mark.asyncio
    async def test_get_stock_statistic_success(self, adapter):
        """测试成功获取个股统计"""
        mock_response = {
            "result": {
                "data": [
                    {
                        "SECURITY_CODE": "000001",
                        "SECURITY_NAME_ABBR": "平安银行",
                        "BILLBOARD_TIMES": 5,
                        "LATEST_TDATE": "2024-01-01T00:00:00"
                    }
                ]
            }
        }
        
        with patch.object(adapter, '_make_request', return_value=mock_response):
            request = DragonTigerRequest(
                data_type="stock_statistic",
                period="近一月"
            )
            response = await adapter.get_stock_statistic(request)
            
            assert response.success is True
            assert not response.data.empty
            assert response.data_type == "stock_statistic"
    
    @pytest.mark.asyncio
    async def test_get_institution_daily_statistic_success(self, adapter):
        """测试成功获取机构每日统计"""
        mock_response = {
            "result": {
                "data": [
                    {
                        "SECURITY_CODE": "000001",
                        "SECURITY_NAME_ABBR": "平安银行",
                        "TRADE_DATE": "2024-01-01T00:00:00",
                        "NET_BUY_AMT": 5000000
                    }
                ]
            }
        }
        
        with patch.object(adapter, '_make_request', return_value=mock_response):
            request = DragonTigerRequest(
                data_type="institution_daily",
                start_date="20240101",
                end_date="20240131"
            )
            response = await adapter.get_institution_daily_statistic(request)
            
            assert response.success is True
            assert not response.data.empty
            assert response.data_type == "institution_daily"
    
    @pytest.mark.asyncio
    async def test_get_department_ranking_success(self, adapter):
        """测试成功获取营业部排行"""
        mock_data = [
            {
                "OPERATEDEPT_NAME": "某某证券营业部",
                "TOTAL_BUYER_SALESTIMES_1DAY": 10,
                "AVERAGE_INCREASE_1DAY": 5.5,
                "RISE_PROBABILITY_1DAY": 0.8
            }
        ]
        
        with patch.object(adapter, '_fetch_all_pages', return_value=mock_data):
            request = DragonTigerRequest(
                data_type="department_ranking",
                period="近一月"
            )
            response = await adapter.get_department_ranking(request)
            
            assert response.success is True
            assert not response.data.empty
            assert response.data_type == "department_ranking"
    
    @pytest.mark.asyncio
    async def test_get_stock_detail_success(self, adapter):
        """测试成功获取个股详情"""
        mock_response = {
            "result": {
                "data": [
                    {
                        "OPERATEDEPT_NAME": "某某证券营业部",
                        "BUY": 5000000,
                        "SELL": 3000000,
                        "NET": 2000000,
                        "OPERATEDEPT_TYPE": "机构专用"
                    }
                ]
            }
        }
        
        with patch.object(adapter, '_make_request', return_value=mock_response):
            request = DragonTigerRequest(
                data_type="stock_detail",
                symbol="000001",
                start_date="20240101",
                flag="买入"
            )
            response = await adapter.get_stock_detail(request)
            
            assert response.success is True
            assert not response.data.empty
            assert response.data_type == "stock_detail"
    
    def test_format_date(self, adapter):
        """测试日期格式化"""
        # 测试YYYYMMDD格式
        result = adapter._format_date("20240101")
        assert result == "2024-01-01"
        
        # 测试已格式化的日期
        result = adapter._format_date("2024-01-01")
        assert result == "2024-01-01"
        
        # 测试空日期
        result = adapter._format_date("")
        assert result == datetime.now().strftime("%Y-%m-%d")
    
    def test_process_dragon_tiger_detail_data(self, adapter):
        """测试龙虎榜详情数据处理"""
        mock_data = [
            {
                "SECURITY_CODE": "000001",
                "SECURITY_NAME_ABBR": "平安银行",
                "TRADE_DATE": "2024-01-01T00:00:00",
                "CLOSE_PRICE": 10.50,
                "CHANGE_RATE": 2.50,
                "BILLBOARD_NET_AMT": 5000000,
                "BILLBOARD_BUY_AMT": 8000000,
                "BILLBOARD_SELL_AMT": 3000000,
                "EXPLANATION": "日涨幅偏离值达7%的证券"
            }
        ]
        
        df = adapter._process_dragon_tiger_detail_data(mock_data)
        
        assert not df.empty
        assert '代码' in df.columns
        assert '名称' in df.columns
        assert '龙虎榜净买额' in df.columns
        assert df.loc[0, '代码'] == '000001'
        assert df.loc[0, '名称'] == '平安银行'
    
    def test_validate_and_clean_data(self, adapter):
        """测试数据验证和清洗"""
        df = pd.DataFrame({
            '代码': ['000001', '000001', '000002'],
            '上榜日': ['2024-01-01', '2024-01-01', '2024-01-02'],
            '龙虎榜净买额': [5000000, 6000000, 3000000],
            '龙虎榜买入额': [8000000, 9000000, 5000000],
            '龙虎榜卖出额': [3000000, 3000000, 2000000]
        })
        
        result_df = adapter._validate_and_clean_data(df)
        
        # 检查去重
        assert len(result_df) == 2  # 去重后只有2条记录
        
        # 检查数据完整性（净买额 = 买入额 - 卖出额）
        for _, row in result_df.iterrows():
            expected_net = row['龙虎榜买入额'] - row['龙虎榜卖出额']
            assert abs(row['龙虎榜净买额'] - expected_net) <= 1000  # 允许小误差


class TestLimitUpReasonAdapter:
    """涨停原因适配器测试类"""
    
    @pytest.fixture
    def adapter(self):
        """创建适配器实例"""
        return LimitUpReasonAdapter(timeout=5, max_retries=2)
    
    @pytest.mark.asyncio
    async def test_get_limitup_reason_data_success(self, adapter):
        """测试成功获取涨停原因数据"""
        mock_data = TestDataGenerator.generate_limitup_response()
        
        with patch.object(adapter, '_make_request', return_value=mock_data):
            request = LimitUpRequest(
                date="2024-01-01",
                include_detail=False
            )
            response = await adapter.get_limitup_reason_data(request)
            
            assert response.success is True
            assert not response.data.empty
            assert response.date == "2024-01-01"
            assert '代码' in response.data.columns
            assert '名称' in response.data.columns
            assert '原因' in response.data.columns
    
    @pytest.mark.asyncio
    async def test_get_limitup_reason_data_empty(self, adapter):
        """测试无涨停股票的情况"""
        mock_data = {"data": []}
        
        with patch.object(adapter, '_make_request', return_value=mock_data):
            request = LimitUpRequest(date="2024-01-01")
            response = await adapter.get_limitup_reason_data(request)
            
            assert response.success is True
            assert response.data.empty
            assert "当日无涨停股票" in response.error_message
    
    @pytest.mark.asyncio
    async def test_get_limitup_reason_data_with_detail(self, adapter):
        """测试获取详细原因"""
        mock_data = TestDataGenerator.generate_limitup_response()
        
        with patch.object(adapter, '_make_request', return_value=mock_data), \
             patch.object(adapter, '_get_stock_detail', return_value="详细政策利好信息"):
            
            request = LimitUpRequest(
                date="2024-01-01",
                include_detail=True
            )
            response = await adapter.get_limitup_reason_data(request)
            
            assert response.success is True
            assert not response.data.empty
            assert '详细原因' in response.data.columns
    
    @pytest.mark.asyncio
    async def test_get_multi_date_data(self, adapter):
        """测试多日期数据获取"""
        mock_response = LimitUpResponse(
            success=True,
            data=pd.DataFrame({'代码': ['000001'], '名称': ['平安银行']}),
            date="2024-01-01"
        )
        
        with patch.object(adapter, 'get_limitup_reason_data', return_value=mock_response):
            results = await adapter.get_multi_date_data("2024-01-01", "2024-01-03")
            
            # 应该有3天的数据（跳过周末）
            assert len(results) >= 1
            assert all(response.success for response in results.values())
    
    def test_process_basic_data(self, adapter):
        """测试基础数据处理"""
        mock_data = [
            [
                "12345", "平安银行", "000001", "政策利好", "2024-01-01",
                10.50, 0.26, 2.50, 9.80, 10500000, 1000000, 1.5, ""
            ]
        ]
        
        df = adapter._process_basic_data(mock_data)
        
        assert not df.empty
        assert len(df) == 1
        assert df.loc[0, '代码'] == '000001'
        assert df.loc[0, '名称'] == '平安银行'
        assert df.loc[0, '原因'] == '政策利好'
        assert df.loc[0, '最新价'] == 10.50
    
    def test_clean_text(self, adapter):
        """测试文本清洗"""
        # 测试HTML标签清理
        text = "<span>政策利好</span>"
        result = adapter._clean_text(text)
        assert result == "政策利好"
        
        # 测试多余空白字符清理
        text = "  政策   利好  "
        result = adapter._clean_text(text)
        assert result == "政策 利好"
        
        # 测试空值处理
        result = adapter._clean_text(None)
        assert result == ""
    
    def test_classify_reason(self, adapter):
        """测试原因分类"""
        # 测试政策利好分类
        text = "国家发改委发布新政策"
        category = adapter._classify_reason(text)
        assert category == "政策利好"
        
        # 测试业绩预增分类
        text = "公司预计净利润大幅增长"
        category = adapter._classify_reason(text)
        assert category == "业绩预增"
        
        # 测试其他分类
        text = "未知原因"
        category = adapter._classify_reason(text)
        assert category == "其他"
    
    def test_extract_tags(self, adapter):
        """测试标签提取"""
        text = "人工智能概念股受到市场关注，新能源板块表现强势"
        tags = adapter._extract_tags(text)
        
        assert len(tags) > 0
        assert any("人工智能" in tag or "AI" in tag for tag in tags)
    
    def test_get_reason_statistics(self, adapter):
        """测试原因统计"""
        df = pd.DataFrame({
            '代码': ['000001', '000002'],
            '名称': ['平安银行', '万科A'],
            '原因分类': ['政策利好', '业绩预增'],
            '涨跌幅': [2.5, 3.2],
            '换手率': [9.8, 12.5],
            '相关标签': ['政策,银行', '业绩,地产']
        })
        
        stats = adapter.get_reason_statistics(df)
        
        assert stats['总涨停数量'] == 2
        assert '分类统计' in stats
        assert stats['分类统计']['政策利好'] == 1
        assert stats['分类统计']['业绩预增'] == 1
        assert stats['平均涨跌幅'] == 2.85
        assert '热门标签' in stats


class TestETFAdapter:
    """ETF适配器测试类"""
    
    @pytest.fixture
    def adapter(self):
        """创建适配器实例"""
        return ETFAdapter(timeout=5, max_retries=2)
    
    @pytest.mark.asyncio
    async def test_get_etf_realtime_data_success(self, adapter):
        """测试成功获取ETF实时数据"""
        mock_data = TestDataGenerator.generate_etf_response()["data"]["diff"]
        
        with patch.object(adapter, '_fetch_all_pages', return_value=mock_data):
            response = await adapter.get_etf_realtime_data(['159707'])
            
            assert response.success is True
            assert not response.data.empty
            assert response.data_type == "realtime"
            assert '代码' in response.data.columns
            assert '名称' in response.data.columns
            assert '最新价' in response.data.columns
    
    @pytest.mark.asyncio
    async def test_get_etf_history_data_success(self, adapter):
        """测试成功获取ETF历史数据"""
        mock_code_map = {"159707": "1"}
        mock_history_data = {
            "data": {
                "klines": [
                    "2024-01-01,1.240,1.250,1.255,1.235,5000000,6250000,1.62,1.20,0.015,4.0",
                    "2024-01-02,1.250,1.260,1.265,1.245,4800000,6048000,1.60,0.80,0.010,3.8"
                ]
            }
        }
        
        with patch.object(adapter, '_get_etf_code_id_map', return_value=mock_code_map), \
             patch.object(adapter, '_make_request', return_value=mock_history_data):
            
            request = ETFRequest(
                symbol="159707",
                data_type="history",
                period="daily"
            )
            response = await adapter.get_etf_history_data(request)
            
            assert response.success is True
            assert not response.data.empty
            assert response.data_type == "history"
            assert response.symbol == "159707"
    
    @pytest.mark.asyncio
    async def test_get_etf_history_data_symbol_not_found(self, adapter):
        """测试ETF代码未找到"""
        mock_code_map = {}
        
        with patch.object(adapter, '_get_etf_code_id_map', return_value=mock_code_map):
            request = ETFRequest(symbol="999999")
            response = await adapter.get_etf_history_data(request)
            
            assert response.success is False
            assert "ETF代码 999999 未找到" in response.error_message
    
    @pytest.mark.asyncio
    async def test_get_etf_intraday_data_success(self, adapter):
        """测试成功获取ETF分时数据"""
        mock_code_map = {"159707": "1"}
        mock_trends_data = {
            "data": {
                "trends": [
                    "2024-01-01 09:30:00,1.240,1.245,1.250,1.235,100000,124500,1.245",
                    "2024-01-01 09:31:00,1.245,1.250,1.255,1.240,120000,150000,1.250"
                ]
            }
        }
        
        with patch.object(adapter, '_get_etf_code_id_map', return_value=mock_code_map), \
             patch.object(adapter, '_make_request', return_value=mock_trends_data):
            
            request = ETFRequest(
                symbol="159707",
                data_type="intraday",
                period="1"
            )
            response = await adapter.get_etf_intraday_data(request)
            
            assert response.success is True
            assert not response.data.empty
            assert response.data_type == "intraday"
    
    def test_process_realtime_data(self, adapter):
        """测试ETF实时数据处理"""
        mock_data = [
            {
                "f12": "159707",
                "f14": "中证500ETF",
                "f2": 1.250,
                "f3": 1.20,
                "f4": 0.015,
                "f5": 5000000,
                "f6": 6250000,
                "f17": 1.240,
                "f15": 1.255,
                "f16": 1.235,
                "f18": 1.235
            }
        ]
        
        df = adapter._process_realtime_data(mock_data)
        
        assert not df.empty
        assert len(df) == 1
        assert df.loc[0, '代码'] == '159707'
        assert df.loc[0, '名称'] == '中证500ETF'
        assert df.loc[0, '最新价'] == 1.250
    
    def test_process_history_data(self, adapter):
        """测试ETF历史数据处理"""
        klines = [
            "2024-01-01,1.240,1.250,1.255,1.235,5000000,6250000,1.62,1.20,0.015,4.0",
            "2024-01-02,1.250,1.260,1.265,1.245,4800000,6048000,1.60,0.80,0.010,3.8"
        ]
        
        df = adapter._process_history_data(klines)
        
        assert not df.empty
        assert len(df) == 2
        assert '开盘' in df.columns
        assert '收盘' in df.columns
        assert df.iloc[0]['开盘'] == 1.240
        assert df.iloc[0]['收盘'] == 1.250
    
    def test_get_etf_special_indicators(self, adapter):
        """测试ETF特有指标计算"""
        df = pd.DataFrame({
            '代码': ['159707', '513500'],
            '名称': ['中证500ETF', '标普500ETF'],
            '最新价': [1.250, 2.100],
            '涨跌幅': [1.20, -0.50],
            '成交额': [6250000, 4200000],
            '换手率': [4.0, 2.5],
            '流通市值': [156250000, 210000000]
        })
        
        indicators = adapter.get_etf_special_indicators(df)
        
        assert '平均涨跌幅' in indicators
        assert '涨跌幅标准差' in indicators
        assert '平均成交额' in indicators
        assert '平均换手率' in indicators
        assert '平均流动性比率' in indicators
        
        assert indicators['平均涨跌幅'] == 0.35  # (1.20 + (-0.50)) / 2
        assert indicators['平均换手率'] == 3.25   # (4.0 + 2.5) / 2
    
    def test_filter_by_date_range(self, adapter):
        """测试日期范围过滤"""
        df = pd.DataFrame({
            '开盘': [1.240, 1.250, 1.260],
            '收盘': [1.250, 1.260, 1.270]
        })
        df.index = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
        
        # 测试开始日期过滤
        result = adapter._filter_by_date_range(df, "2024-01-02", "")
        assert len(result) == 2
        assert result.index[0] == pd.to_datetime('2024-01-02')
        
        # 测试结束日期过滤
        result = adapter._filter_by_date_range(df, "", "2024-01-02")
        assert len(result) == 2
        assert result.index[-1] == pd.to_datetime('2024-01-02')
        
        # 测试日期范围过滤
        result = adapter._filter_by_date_range(df, "2024-01-02", "2024-01-02")
        assert len(result) == 1
        assert result.index[0] == pd.to_datetime('2024-01-02')


class TestNetworkAndErrorHandling:
    """网络和错误处理测试类"""
    
    @pytest.fixture
    def adapter(self):
        """创建适配器实例"""
        return EastMoneyAdapter(timeout=1, max_retries=2)
    
    @pytest.mark.asyncio
    async def test_network_timeout(self, adapter):
        """测试网络超时"""
        with patch.object(adapter.session, 'get', side_effect=requests.exceptions.Timeout):
            result = await adapter._make_request("http://test.com", {})
            
            assert result is None
            assert adapter.error_stats['network_errors'] > 0
    
    @pytest.mark.asyncio
    async def test_network_connection_error(self, adapter):
        """测试网络连接错误"""
        with patch.object(adapter.session, 'get', side_effect=requests.exceptions.ConnectionError):
            result = await adapter._make_request("http://test.com", {})
            
            assert result is None
            assert adapter.error_stats['network_errors'] > 0
    
    @pytest.mark.asyncio
    async def test_json_parse_error(self, adapter):
        """测试JSON解析错误"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("Invalid JSON")
        
        with patch.object(adapter.session, 'get', return_value=mock_response):
            result = await adapter._make_request("http://test.com", {})
            
            assert result is None
            assert adapter.error_stats['data_format_errors'] > 0
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, adapter):
        """测试重试机制"""
        # 第一次失败，第二次成功
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"success": True}
        
        with patch.object(adapter.session, 'get', side_effect=[
            requests.exceptions.Timeout,
            mock_response
        ]):
            result = await adapter._make_request("http://test.com", {})
            
            assert result == {"success": True}
            assert adapter.error_stats['network_errors'] == 1  # 只记录第一次失败
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, adapter):
        """测试超过最大重试次数"""
        with patch.object(adapter.session, 'get', side_effect=requests.exceptions.Timeout):
            result = await adapter._make_request("http://test.com", {})
            
            assert result is None
            assert adapter.error_stats['network_errors'] == adapter.max_retries


class TestBoundaryConditions:
    """边界条件测试类"""
    
    @pytest.fixture
    def adapter(self):
        """创建适配器实例"""
        return EastMoneyAdapter()
    
    def test_empty_data_processing(self, adapter):
        """测试空数据处理"""
        # 测试空列表
        df = adapter._process_realtime_data([])
        assert df.empty
        
        # 测试空K线数据
        df = adapter._process_history_data([])
        assert df.empty
    
    def test_invalid_data_processing(self, adapter):
        """测试无效数据处理"""
        # 测试包含无效值的数据
        mock_data = [
            {
                "f2": "-",  # 无效价格
                "f3": None,  # 空值
                "f12": "000001",
                "f14": "平安银行"
            }
        ]
        
        df = adapter._process_realtime_data(mock_data)
        
        assert not df.empty
        assert pd.isna(df.loc[0, '最新价'])  # 无效值应转换为NaN
    
    def test_large_data_processing(self, adapter):
        """测试大数据量处理"""
        # 生成大量测试数据
        large_data = []
        for i in range(10000):
            large_data.append({
                "f2": 10.0 + i * 0.01,
                "f3": 1.0,
                "f12": f"{i:06d}",
                "f14": f"股票{i}"
            })
        
        df = adapter._process_realtime_data(large_data)
        
        assert len(df) == 10000
        assert not df.empty
        assert df.loc[0, '代码'] == '000000'
        assert df.loc[9999, '代码'] == '009999'
    
    def test_special_characters_handling(self, adapter):
        """测试特殊字符处理"""
        limitup_adapter = LimitUpReasonAdapter()
        
        # 测试包含特殊字符的文本
        text = "<script>alert('test')</script>政策利好！@#$%^&*()"
        cleaned = limitup_adapter._clean_text(text)
        
        assert "<script>" not in cleaned
        assert "政策利好" in cleaned
    
    def test_date_edge_cases(self, adapter):
        """测试日期边界情况"""
        dragon_adapter = DragonTigerAdapter()
        
        # 测试各种日期格式
        assert dragon_adapter._format_date("20240101") == "2024-01-01"
        assert dragon_adapter._format_date("2024-01-01") == "2024-01-01"
        
        # 测试空日期
        result = dragon_adapter._format_date("")
        assert result == datetime.now().strftime("%Y-%m-%d")
    
    def test_numeric_conversion_edge_cases(self, adapter):
        """测试数值转换边界情况"""
        fund_adapter = FundFlowAdapter()
        
        df = pd.DataFrame({
            '代码': ['000001'],
            '名称': ['测试'],
            '今日主力净流入-净额': ['abc'],  # 无法转换的字符串
            '今日涨跌幅': [''],  # 空字符串
            '最新价': [None]  # 空值
        })
        
        fund_adapter._convert_numeric_columns(df, "今日")
        
        # 无法转换的值应该变成NaN
        assert pd.isna(df.loc[0, '今日主力净流入-净额'])
        assert pd.isna(df.loc[0, '今日涨跌幅'])


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])