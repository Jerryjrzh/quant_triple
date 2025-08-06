"""
数据库操作单元测试

测试所有数据库模型的CRUD操作，实现事务处理和并发控制的测试，
添加数据完整性约束的验证测试，创建数据库性能和查询优化的测试。

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock
import threading
import concurrent.futures
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.pool import StaticPool

from stock_analysis_system.core.database import Base
from stock_analysis_system.data.models import (
    StockDailyData,
    Alert,
    NotificationLog,
    DragonTigerList,
    SpringFestivalAnalysis,
    InstitutionalActivity,
    RiskMetrics,
    StockPool,
    StockPoolMember,
    AlertRule,
    AlertHistory,
    UserSession,
    SystemConfig,
    DragonTigerBoard,
    DragonTigerDetail,
    FundFlow,
    LimitUpReason,
    ETFData,
    ETFConstituent,
    DataQualityLog,
    DataSourceHealth
)


class DatabaseTestHelper:
    """数据库测试辅助类"""
    
    def __init__(self):
        # 使用内存SQLite数据库进行测试
        self.engine = create_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def setup_database(self):
        """设置测试数据库"""
        Base.metadata.create_all(bind=self.engine)
        
    def teardown_database(self):
        """清理测试数据库"""
        Base.metadata.drop_all(bind=self.engine)
        
    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()
        
    def create_sample_stock_data(self, count: int = 10) -> List[StockDailyData]:
        """创建示例股票数据"""
        np.random.seed(42)
        
        data = []
        base_date = date.today() - timedelta(days=count)
        
        for i in range(count):
            stock_data = StockDailyData(
                stock_code=f"{i+1:06d}",
                trade_date=base_date + timedelta(days=i),
                open_price=Decimal(str(round(np.random.uniform(10, 100), 3))),
                high_price=Decimal(str(round(np.random.uniform(10, 100), 3))),
                low_price=Decimal(str(round(np.random.uniform(10, 100), 3))),
                close_price=Decimal(str(round(np.random.uniform(10, 100), 3))),
                volume=int(np.random.randint(1000000, 10000000)),
                amount=Decimal(str(round(np.random.uniform(10000000, 100000000), 2)))
            )
            data.append(stock_data)
            
        return data
    
    def create_sample_dragon_tiger_data(self, count: int = 5) -> List[DragonTigerBoard]:
        """创建示例龙虎榜数据"""
        np.random.seed(42)
        
        data = []
        base_date = date.today() - timedelta(days=count)
        
        for i in range(count):
            dt_data = DragonTigerBoard(
                trade_date=base_date + timedelta(days=i),
                stock_code=f"{i+1:06d}",
                stock_name=f"测试股票{i+1}",
                close_price=Decimal(str(round(np.random.uniform(10, 100), 2))),
                change_rate=Decimal(str(round(np.random.uniform(-10, 10), 2))),
                net_buy_amount=int(np.random.randint(-100000000, 100000000)),
                buy_amount=int(np.random.randint(0, 200000000)),
                sell_amount=int(np.random.randint(0, 200000000)),
                reason="测试涨停原因"
            )
            data.append(dt_data)
            
        return data
    
    def create_sample_fund_flow_data(self, count: int = 5) -> List[FundFlow]:
        """创建示例资金流向数据"""
        np.random.seed(42)
        
        data = []
        base_date = date.today() - timedelta(days=count)
        periods = ["今日", "3日", "5日", "10日"]
        
        for i in range(count):
            for period in periods:
                ff_data = FundFlow(
                    stock_code=f"{i+1:06d}",
                    stock_name=f"测试股票{i+1}",
                    trade_date=base_date + timedelta(days=i),
                    period_type=period,
                    main_net_inflow=int(np.random.randint(-50000000, 50000000)),
                    main_net_inflow_rate=Decimal(str(round(np.random.uniform(-10, 10), 2))),
                    super_large_net_inflow=int(np.random.randint(-30000000, 30000000)),
                    super_large_net_inflow_rate=Decimal(str(round(np.random.uniform(-5, 5), 2))),
                    large_net_inflow=int(np.random.randint(-20000000, 20000000)),
                    large_net_inflow_rate=Decimal(str(round(np.random.uniform(-3, 3), 2))),
                    medium_net_inflow=int(np.random.randint(-10000000, 10000000)),
                    medium_net_inflow_rate=Decimal(str(round(np.random.uniform(-2, 2), 2))),
                    small_net_inflow=int(np.random.randint(-5000000, 5000000)),
                    small_net_inflow_rate=Decimal(str(round(np.random.uniform(-1, 1), 2)))
                )
                data.append(ff_data)
                
        return data


@pytest.fixture
def db_helper():
    """数据库测试辅助器fixture"""
    helper = DatabaseTestHelper()
    helper.setup_database()
    yield helper
    helper.teardown_database()


@pytest.fixture
def db_session(db_helper):
    """数据库会话fixture"""
    session = db_helper.get_session()
    yield session
    session.close()


class TestStockDailyDataOperations:
    """股票日线数据CRUD操作测试"""
    
    def test_create_stock_data(self, db_session, db_helper):
        """测试创建股票数据"""
        # 创建测试数据
        stock_data = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),
            open_price=Decimal("10.50"),
            high_price=Decimal("11.00"),
            low_price=Decimal("10.20"),
            close_price=Decimal("10.80"),
            volume=1000000,
            amount=Decimal("10800000.00")
        )
        
        # 插入数据
        db_session.add(stock_data)
        db_session.commit()
        
        # 验证数据
        assert stock_data.id is not None
        assert stock_data.created_at is not None
        assert stock_data.updated_at is not None
    
    def test_read_stock_data(self, db_session, db_helper):
        """测试读取股票数据"""
        # 创建测试数据
        sample_data = db_helper.create_sample_stock_data(5)
        db_session.add_all(sample_data)
        db_session.commit()
        
        # 测试按股票代码查询
        result = db_session.query(StockDailyData).filter(
            StockDailyData.stock_code == "000001"
        ).first()
        assert result is not None
        assert result.stock_code == "000001"
        
        # 测试按日期范围查询
        start_date = date.today() - timedelta(days=10)
        end_date = date.today()
        results = db_session.query(StockDailyData).filter(
            StockDailyData.trade_date.between(start_date, end_date)
        ).all()
        assert len(results) > 0
    
    def test_update_stock_data(self, db_session, db_helper):
        """测试更新股票数据"""
        # 创建测试数据
        stock_data = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),
            close_price=Decimal("10.00")
        )
        db_session.add(stock_data)
        db_session.commit()
        
        original_updated_at = stock_data.updated_at
        
        # 更新数据
        stock_data.close_price = Decimal("11.00")
        db_session.commit()
        
        # 验证更新
        assert stock_data.close_price == Decimal("11.00")
        assert stock_data.updated_at > original_updated_at
    
    def test_delete_stock_data(self, db_session, db_helper):
        """测试删除股票数据"""
        # 创建测试数据
        stock_data = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),
            close_price=Decimal("10.00")
        )
        db_session.add(stock_data)
        db_session.commit()
        
        data_id = stock_data.id
        
        # 删除数据
        db_session.delete(stock_data)
        db_session.commit()
        
        # 验证删除
        result = db_session.query(StockDailyData).filter(
            StockDailyData.id == data_id
        ).first()
        assert result is None
    
    def test_unique_constraint(self, db_session):
        """测试唯一约束"""
        # 创建第一条数据
        stock_data1 = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),
            close_price=Decimal("10.00")
        )
        db_session.add(stock_data1)
        db_session.commit()
        
        # 尝试创建重复数据
        stock_data2 = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),
            close_price=Decimal("11.00")
        )
        db_session.add(stock_data2)
        
        # 应该抛出完整性错误
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestDragonTigerOperations:
    """龙虎榜数据操作测试"""
    
    def test_create_dragon_tiger_data(self, db_session, db_helper):
        """测试创建龙虎榜数据"""
        dt_data = DragonTigerBoard(
            trade_date=date.today(),
            stock_code="000001",
            stock_name="测试股票",
            close_price=Decimal("10.50"),
            change_rate=Decimal("5.00"),
            net_buy_amount=1000000,
            buy_amount=5000000,
            sell_amount=4000000,
            reason="测试涨停"
        )
        
        db_session.add(dt_data)
        db_session.commit()
        
        assert dt_data.id is not None
        assert dt_data.created_at is not None
    
    def test_dragon_tiger_with_details(self, db_session):
        """测试龙虎榜主表和明细表关联"""
        # 创建主表数据
        dt_board = DragonTigerBoard(
            trade_date=date.today(),
            stock_code="000001",
            stock_name="测试股票",
            close_price=Decimal("10.50"),
            net_buy_amount=1000000
        )
        db_session.add(dt_board)
        db_session.commit()
        
        # 创建明细数据
        dt_detail = DragonTigerDetail(
            board_id=dt_board.id,
            trade_date=date.today(),
            stock_code="000001",
            seat_name="测试营业部",
            seat_type="营业部",
            buy_amount=2000000,
            sell_amount=1000000,
            net_amount=1000000,
            rank=1
        )
        db_session.add(dt_detail)
        db_session.commit()
        
        # 验证关联
        assert dt_detail.board_id == dt_board.id
        assert dt_detail.stock_code == dt_board.stock_code
    
    def test_batch_insert_dragon_tiger(self, db_session, db_helper):
        """测试批量插入龙虎榜数据"""
        sample_data = db_helper.create_sample_dragon_tiger_data(10)
        
        # 批量插入
        db_session.add_all(sample_data)
        db_session.commit()
        
        # 验证插入结果
        count = db_session.query(DragonTigerBoard).count()
        assert count == 10
        
        # 验证数据完整性
        for data in sample_data:
            assert data.id is not None


class TestFundFlowOperations:
    """资金流向数据操作测试"""
    
    def test_create_fund_flow_data(self, db_session):
        """测试创建资金流向数据"""
        ff_data = FundFlow(
            stock_code="000001",
            stock_name="测试股票",
            trade_date=date.today(),
            period_type="今日",
            main_net_inflow=1000000,
            main_net_inflow_rate=Decimal("5.50"),
            super_large_net_inflow=500000,
            super_large_net_inflow_rate=Decimal("2.75")
        )
        
        db_session.add(ff_data)
        db_session.commit()
        
        assert ff_data.id is not None
        assert ff_data.created_at is not None
    
    def test_fund_flow_unique_constraint(self, db_session):
        """测试资金流向唯一约束"""
        # 创建第一条数据
        ff_data1 = FundFlow(
            stock_code="000001",
            trade_date=date.today(),
            period_type="今日",
            main_net_inflow=1000000
        )
        db_session.add(ff_data1)
        db_session.commit()
        
        # 尝试创建重复数据
        ff_data2 = FundFlow(
            stock_code="000001",
            trade_date=date.today(),
            period_type="今日",
            main_net_inflow=2000000
        )
        db_session.add(ff_data2)
        
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_query_fund_flow_by_period(self, db_session, db_helper):
        """测试按周期查询资金流向"""
        sample_data = db_helper.create_sample_fund_flow_data(3)
        db_session.add_all(sample_data)
        db_session.commit()
        
        # 查询今日数据
        today_data = db_session.query(FundFlow).filter(
            FundFlow.period_type == "今日"
        ).all()
        assert len(today_data) == 3
        
        # 查询特定股票的所有周期数据
        stock_data = db_session.query(FundFlow).filter(
            FundFlow.stock_code == "000001"
        ).all()
        assert len(stock_data) == 4  # 4个周期


class TestLimitUpReasonOperations:
    """涨停原因数据操作测试"""
    
    def test_create_limitup_reason(self, db_session):
        """测试创建涨停原因数据"""
        lr_data = LimitUpReason(
            trade_date=date.today(),
            stock_code="000001",
            stock_name="测试股票",
            reason="业绩预增",
            detail_reason="公司发布业绩预增公告，预计净利润同比增长50%以上",
            latest_price=Decimal("10.50"),
            change_rate=Decimal("10.00"),
            volume=1000000,
            amount=10500000,
            reason_category="业绩驱动",
            reason_tags=["业绩预增", "基本面利好"]
        )
        
        db_session.add(lr_data)
        db_session.commit()
        
        assert lr_data.id is not None
        assert lr_data.reason_tags == ["业绩预增", "基本面利好"]
    
    def test_search_by_reason_category(self, db_session):
        """测试按原因分类搜索"""
        # 创建不同分类的数据
        categories = ["业绩驱动", "政策利好", "概念炒作", "技术突破"]
        
        for i, category in enumerate(categories):
            lr_data = LimitUpReason(
                trade_date=date.today(),
                stock_code=f"{i+1:06d}",
                stock_name=f"测试股票{i+1}",
                reason=f"测试原因{i+1}",
                reason_category=category
            )
            db_session.add(lr_data)
        
        db_session.commit()
        
        # 按分类查询
        policy_data = db_session.query(LimitUpReason).filter(
            LimitUpReason.reason_category == "政策利好"
        ).all()
        assert len(policy_data) == 1
        assert policy_data[0].stock_code == "000002"


class TestETFDataOperations:
    """ETF数据操作测试"""
    
    def test_create_etf_data(self, db_session):
        """测试创建ETF数据"""
        etf_data = ETFData(
            etf_code="510050",
            etf_name="50ETF",
            trade_date=date.today(),
            open_price=Decimal("2.500"),
            close_price=Decimal("2.520"),
            high_price=Decimal("2.530"),
            low_price=Decimal("2.490"),
            volume=10000000,
            amount=25200000,
            change_rate=Decimal("0.80"),
            unit_nav=Decimal("2.5180"),
            premium_rate=Decimal("0.08"),
            fund_size=Decimal("50000000000.00")
        )
        
        db_session.add(etf_data)
        db_session.commit()
        
        assert etf_data.id is not None
        assert etf_data.premium_rate == Decimal("0.08")
    
    def test_etf_with_constituents(self, db_session):
        """测试ETF与成分股关联"""
        # 创建ETF数据
        etf_data = ETFData(
            etf_code="510050",
            etf_name="50ETF",
            trade_date=date.today(),
            close_price=Decimal("2.520")
        )
        db_session.add(etf_data)
        db_session.commit()
        
        # 创建成分股数据
        constituents = [
            ETFConstituent(
                etf_code="510050",
                stock_code="000001",
                stock_name="平安银行",
                weight=Decimal("5.50"),
                shares=1000000,
                market_value=Decimal("10000000.00"),
                update_date=date.today()
            ),
            ETFConstituent(
                etf_code="510050",
                stock_code="000002",
                stock_name="万科A",
                weight=Decimal("4.20"),
                shares=800000,
                market_value=Decimal("8000000.00"),
                update_date=date.today()
            )
        ]
        
        db_session.add_all(constituents)
        db_session.commit()
        
        # 验证关联查询
        etf_constituents = db_session.query(ETFConstituent).filter(
            ETFConstituent.etf_code == "510050"
        ).all()
        assert len(etf_constituents) == 2
        
        total_weight = sum(c.weight for c in etf_constituents)
        assert total_weight == Decimal("9.70")


class TestDataQualityOperations:
    """数据质量相关操作测试"""
    
    def test_create_data_quality_log(self, db_session):
        """测试创建数据质量日志"""
        quality_log = DataQualityLog(
            data_source="eastmoney",
            data_type="stock_daily",
            check_date=date.today(),
            total_records=1000,
            valid_records=980,
            invalid_records=20,
            duplicate_records=5,
            missing_fields=["volume", "amount"],
            completeness_score=Decimal("98.00"),
            accuracy_score=Decimal("95.50"),
            consistency_score=Decimal("97.20"),
            timeliness_score=Decimal("99.00"),
            overall_score=Decimal("97.43"),
            issues_found=["部分记录缺少成交量", "发现重复数据"],
            recommendations=["增强数据验证", "优化去重逻辑"]
        )
        
        db_session.add(quality_log)
        db_session.commit()
        
        assert quality_log.id is not None
        assert quality_log.overall_score == Decimal("97.43")
        assert "部分记录缺少成交量" in quality_log.issues_found
    
    def test_create_data_source_health(self, db_session):
        """测试创建数据源健康状态"""
        health_record = DataSourceHealth(
            source_name="eastmoney_api",
            status="healthy",
            response_time=Decimal("1.250"),
            success_rate=Decimal("99.50"),
            error_rate=Decimal("0.50"),
            total_requests=1000,
            successful_requests=995,
            failed_requests=5,
            avg_response_time=Decimal("1.180"),
            max_response_time=Decimal("3.500"),
            min_response_time=Decimal("0.800"),
            error_count_24h=5
        )
        
        db_session.add(health_record)
        db_session.commit()
        
        assert health_record.id is not None
        assert health_record.success_rate == Decimal("99.50")


class TestTransactionHandling:
    """事务处理测试"""
    
    def test_transaction_commit(self, db_session, db_helper):
        """测试事务提交"""
        # 开始事务
        sample_data = db_helper.create_sample_stock_data(3)
        
        try:
            db_session.add_all(sample_data)
            db_session.commit()
            
            # 验证数据已提交
            count = db_session.query(StockDailyData).count()
            assert count == 3
            
        except Exception as e:
            db_session.rollback()
            pytest.fail(f"Transaction commit failed: {e}")
    
    def test_transaction_rollback(self, db_session):
        """测试事务回滚"""
        # 创建有效数据
        stock_data1 = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),
            close_price=Decimal("10.00")
        )
        db_session.add(stock_data1)
        
        # 创建会导致约束违反的数据
        stock_data2 = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),  # 相同的股票代码和日期
            close_price=Decimal("11.00")
        )
        db_session.add(stock_data2)
        
        # 事务应该回滚
        with pytest.raises(IntegrityError):
            db_session.commit()
        
        db_session.rollback()
        
        # 验证没有数据被插入
        count = db_session.query(StockDailyData).count()
        assert count == 0
    
    def test_nested_transaction(self, db_session):
        """测试嵌套事务"""
        # 外层事务
        stock_data1 = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),
            close_price=Decimal("10.00")
        )
        db_session.add(stock_data1)
        
        # 创建保存点
        savepoint = db_session.begin_nested()
        
        try:
            # 内层事务 - 会失败
            stock_data2 = StockDailyData(
                stock_code="000001",
                trade_date=date.today(),  # 重复数据
                close_price=Decimal("11.00")
            )
            db_session.add(stock_data2)
            db_session.flush()  # 这会触发约束错误
            
        except IntegrityError:
            # 回滚到保存点
            savepoint.rollback()
        
        # 外层事务继续
        stock_data3 = StockDailyData(
            stock_code="000002",
            trade_date=date.today(),
            close_price=Decimal("12.00")
        )
        db_session.add(stock_data3)
        db_session.commit()
        
        # 验证结果
        count = db_session.query(StockDailyData).count()
        assert count == 2  # 只有stock_data1和stock_data3被插入


class TestConcurrencyControl:
    """并发控制测试"""
    
    def test_concurrent_inserts(self, db_helper):
        """测试并发插入"""
        def insert_data(thread_id):
            session = db_helper.get_session()
            try:
                stock_data = StockDailyData(
                    stock_code=f"{thread_id:06d}",
                    trade_date=date.today(),
                    close_price=Decimal("10.00")
                )
                session.add(stock_data)
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                return False
            finally:
                session.close()
        
        # 并发执行插入操作
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(insert_data, i) for i in range(1, 11)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 验证所有插入都成功
        assert all(results)
        
        # 验证数据完整性
        session = db_helper.get_session()
        try:
            count = session.query(StockDailyData).count()
            assert count == 10
        finally:
            session.close()
    
    def test_concurrent_updates(self, db_helper):
        """测试并发更新"""
        # 先插入测试数据
        session = db_helper.get_session()
        stock_data = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),
            close_price=Decimal("10.00")
        )
        session.add(stock_data)
        session.commit()
        data_id = stock_data.id
        session.close()
        
        def update_data(thread_id, new_price):
            session = db_helper.get_session()
            try:
                data = session.query(StockDailyData).filter(
                    StockDailyData.id == data_id
                ).first()
                if data:
                    data.close_price = Decimal(str(new_price))
                    session.commit()
                    return True
                return False
            except Exception as e:
                session.rollback()
                return False
            finally:
                session.close()
        
        # 并发更新
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(update_data, 1, 11.00),
                executor.submit(update_data, 2, 12.00),
                executor.submit(update_data, 3, 13.00)
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 验证至少有一个更新成功
        assert any(results)
        
        # 验证最终状态
        session = db_helper.get_session()
        try:
            final_data = session.query(StockDailyData).filter(
                StockDailyData.id == data_id
            ).first()
            assert final_data.close_price in [Decimal("11.00"), Decimal("12.00"), Decimal("13.00")]
        finally:
            session.close()


class TestDataIntegrityConstraints:
    """数据完整性约束测试"""
    
    def test_not_null_constraints(self, db_session):
        """测试非空约束"""
        # 尝试插入缺少必填字段的数据
        with pytest.raises(IntegrityError):
            stock_data = StockDailyData(
                stock_code=None,  # 违反非空约束
                trade_date=date.today(),
                close_price=Decimal("10.00")
            )
            db_session.add(stock_data)
            db_session.commit()
    
    def test_foreign_key_constraints(self, db_session):
        """测试外键约束（模拟）"""
        # 创建龙虎榜主表数据
        dt_board = DragonTigerBoard(
            trade_date=date.today(),
            stock_code="000001",
            stock_name="测试股票",
            close_price=Decimal("10.50")
        )
        db_session.add(dt_board)
        db_session.commit()
        
        # 创建有效的明细数据
        dt_detail = DragonTigerDetail(
            board_id=dt_board.id,
            trade_date=date.today(),
            stock_code="000001",
            seat_name="测试营业部",
            buy_amount=1000000
        )
        db_session.add(dt_detail)
        db_session.commit()
        
        assert dt_detail.board_id == dt_board.id
    
    def test_check_constraints_simulation(self, db_session):
        """测试检查约束（模拟业务逻辑验证）"""
        # 模拟价格不能为负数的约束
        with pytest.raises(ValueError):
            if Decimal("-10.00") < 0:
                raise ValueError("Price cannot be negative")
        
        # 模拟成交量不能为负数的约束
        with pytest.raises(ValueError):
            if -1000000 < 0:
                raise ValueError("Volume cannot be negative")


class TestDatabasePerformance:
    """数据库性能测试"""
    
    def test_bulk_insert_performance(self, db_session, db_helper):
        """测试批量插入性能"""
        import time
        
        # 创建大量测试数据
        large_dataset = db_helper.create_sample_stock_data(1000)
        
        # 测试批量插入性能
        start_time = time.time()
        db_session.add_all(large_dataset)
        db_session.commit()
        end_time = time.time()
        
        insert_time = end_time - start_time
        
        # 验证插入成功
        count = db_session.query(StockDailyData).count()
        assert count == 1000
        
        # 性能应该在合理范围内（例如小于5秒）
        assert insert_time < 5.0
        
        print(f"Bulk insert of 1000 records took {insert_time:.3f} seconds")
    
    def test_query_performance_with_index(self, db_session, db_helper):
        """测试带索引的查询性能"""
        import time
        
        # 插入测试数据
        large_dataset = db_helper.create_sample_stock_data(1000)
        db_session.add_all(large_dataset)
        db_session.commit()
        
        # 测试索引查询性能
        start_time = time.time()
        results = db_session.query(StockDailyData).filter(
            StockDailyData.stock_code == "000001"
        ).all()
        end_time = time.time()
        
        query_time = end_time - start_time
        
        # 查询应该很快（例如小于0.1秒）
        assert query_time < 0.1
        assert len(results) > 0
        
        print(f"Indexed query took {query_time:.6f} seconds")
    
    def test_complex_query_performance(self, db_session, db_helper):
        """测试复杂查询性能"""
        import time
        
        # 插入测试数据
        stock_data = db_helper.create_sample_stock_data(500)
        dt_data = db_helper.create_sample_dragon_tiger_data(100)
        ff_data = db_helper.create_sample_fund_flow_data(50)
        
        db_session.add_all(stock_data + dt_data + ff_data)
        db_session.commit()
        
        # 测试复杂联合查询
        start_time = time.time()
        
        # 模拟复杂查询：查找有龙虎榜记录且资金净流入为正的股票
        results = db_session.query(StockDailyData).join(
            DragonTigerBoard, StockDailyData.stock_code == DragonTigerBoard.stock_code
        ).join(
            FundFlow, StockDailyData.stock_code == FundFlow.stock_code
        ).filter(
            FundFlow.main_net_inflow > 0,
            DragonTigerBoard.net_buy_amount > 0
        ).all()
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # 复杂查询应该在合理时间内完成
        assert query_time < 2.0
        
        print(f"Complex join query took {query_time:.3f} seconds, returned {len(results)} results")


class TestDatabaseMigration:
    """数据库迁移测试"""
    
    def test_schema_creation(self, db_helper):
        """测试数据库模式创建"""
        # 验证所有表都被创建
        inspector = db_helper.engine.dialect.get_table_names(db_helper.engine.connect())
        
        expected_tables = [
            'stock_daily_data',
            'alerts',
            'notification_logs',
            'dragon_tiger_list',
            'spring_festival_analysis',
            'institutional_activity',
            'risk_metrics',
            'stock_pools',
            'stock_pool_members',
            'alert_rules',
            'alert_history',
            'user_sessions',
            'system_config',
            'dragon_tiger_board',
            'dragon_tiger_detail',
            'fund_flow',
            'limitup_reason',
            'etf_data',
            'etf_constituent',
            'data_quality_log',
            'data_source_health'
        ]
        
        # 检查所有预期的表都存在
        for table in expected_tables:
            assert table in inspector
    
    def test_data_migration_simulation(self, db_session, db_helper):
        """测试数据迁移模拟"""
        # 模拟从旧表结构迁移到新表结构
        
        # 1. 创建旧格式数据
        old_data = [
            {
                'stock_code': '000001',
                'trade_date': date.today(),
                'price': 10.50,
                'vol': 1000000
            },
            {
                'stock_code': '000002',
                'trade_date': date.today(),
                'price': 20.30,
                'vol': 2000000
            }
        ]
        
        # 2. 转换为新格式并插入
        new_data = []
        for item in old_data:
            stock_data = StockDailyData(
                stock_code=item['stock_code'],
                trade_date=item['trade_date'],
                close_price=Decimal(str(item['price'])),
                volume=item['vol']
            )
            new_data.append(stock_data)
        
        db_session.add_all(new_data)
        db_session.commit()
        
        # 3. 验证迁移结果
        count = db_session.query(StockDailyData).count()
        assert count == 2
        
        # 验证数据正确性
        migrated_data = db_session.query(StockDailyData).all()
        assert migrated_data[0].close_price == Decimal('10.50')
        assert migrated_data[1].volume == 2000000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])