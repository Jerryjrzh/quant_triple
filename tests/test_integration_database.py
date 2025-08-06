"""
数据库集成测试

测试数据层与数据库的完整交互，验证复杂查询和事务处理的正确性，
添加数据一致性和完整性的集成验证，实现数据库连接池和性能的集成测试。

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import time
import threading
import concurrent.futures
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.pool import StaticPool, QueuePool

from stock_analysis_system.core.database import Base
from stock_analysis_system.data.models import (
    StockDailyData,
    DragonTigerBoard,
    DragonTigerDetail,
    FundFlow,
    LimitUpReason,
    ETFData,
    ETFConstituent,
    DataQualityLog,
    DataSourceHealth
)


class DatabaseIntegrationHelper:
    """数据库集成测试辅助类"""
    
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
    
    def create_test_data(self, session: Session):
        """创建测试数据"""
        # 创建股票日线数据
        stock_data = [
            StockDailyData(
                stock_code="000001",
                trade_date=date.today() - timedelta(days=i),
                open_price=Decimal(str(10.0 + i * 0.1)),
                high_price=Decimal(str(10.5 + i * 0.1)),
                low_price=Decimal(str(9.5 + i * 0.1)),
                close_price=Decimal(str(10.2 + i * 0.1)),
                volume=1000000 + i * 10000,
                amount=Decimal(str(10200000 + i * 102000))
            ) for i in range(10)
        ]
        
        # 创建龙虎榜数据
        dt_boards = [
            DragonTigerBoard(
                trade_date=date.today() - timedelta(days=i),
                stock_code="000001",
                stock_name="平安银行",
                close_price=Decimal(str(10.2 + i * 0.1)),
                change_rate=Decimal(str(5.0 - i * 0.5)),
                net_buy_amount=1000000 - i * 50000,
                buy_amount=5000000 - i * 100000,
                sell_amount=4000000 - i * 50000,
                reason="涨停"
            ) for i in range(5)
        ]
        
        # 创建资金流向数据
        fund_flows = []
        periods = ["今日", "3日", "5日", "10日"]
        for i in range(3):
            for period in periods:
                fund_flows.append(FundFlow(
                    stock_code="000001",
                    stock_name="平安银行",
                    trade_date=date.today() - timedelta(days=i),
                    period_type=period,
                    main_net_inflow=1000000 - i * 100000,
                    main_net_inflow_rate=Decimal(str(5.0 - i * 1.0)),
                    super_large_net_inflow=500000 - i * 50000,
                    super_large_net_inflow_rate=Decimal(str(2.5 - i * 0.5))
                ))
        
        # 批量插入数据
        session.add_all(stock_data + dt_boards + fund_flows)
        session.commit()
        
        return {
            'stock_data': stock_data,
            'dt_boards': dt_boards,
            'fund_flows': fund_flows
        }


@pytest.fixture
def db_helper():
    """数据库集成测试辅助器fixture"""
    helper = DatabaseIntegrationHelper()
    helper.setup_database()
    yield helper
    helper.teardown_database()


@pytest.fixture
def db_session(db_helper):
    """数据库会话fixture"""
    session = db_helper.get_session()
    yield session
    session.close()


class TestDatabaseIntegration:
    """数据库集成测试"""
    
    def test_complex_join_queries(self, db_session, db_helper):
        """测试复杂联合查询"""
        # 创建测试数据
        test_data = db_helper.create_test_data(db_session)
        
        # 复杂联合查询：获取有龙虎榜记录的股票的日线数据和资金流向
        query = db_session.query(
            StockDailyData.stock_code,
            StockDailyData.trade_date,
            StockDailyData.close_price,
            DragonTigerBoard.net_buy_amount,
            FundFlow.main_net_inflow
        ).join(
            DragonTigerBoard, 
            (StockDailyData.stock_code == DragonTigerBoard.stock_code) & 
            (StockDailyData.trade_date == DragonTigerBoard.trade_date)
        ).join(
            FundFlow,
            (StockDailyData.stock_code == FundFlow.stock_code) & 
            (StockDailyData.trade_date == FundFlow.trade_date) &
            (FundFlow.period_type == "今日")
        ).filter(
            DragonTigerBoard.net_buy_amount > 0,
            FundFlow.main_net_inflow > 0
        ).order_by(StockDailyData.trade_date.desc())
        
        results = query.all()
        
        # 验证查询结果
        assert len(results) > 0
        for result in results:
            assert result.net_buy_amount > 0
            assert result.main_net_inflow > 0
            assert result.stock_code == "000001"
    
    def test_aggregation_queries(self, db_session, db_helper):
        """测试聚合查询"""
        # 创建测试数据
        test_data = db_helper.create_test_data(db_session)
        
        # 聚合查询：计算股票的统计信息
        stats_query = db_session.query(
            StockDailyData.stock_code,
            func.count(StockDailyData.id).label('trading_days'),
            func.avg(StockDailyData.close_price).label('avg_price'),
            func.max(StockDailyData.close_price).label('max_price'),
            func.min(StockDailyData.close_price).label('min_price'),
            func.sum(StockDailyData.volume).label('total_volume')
        ).group_by(StockDailyData.stock_code).having(
            func.count(StockDailyData.id) > 5
        )
        
        stats_results = stats_query.all()
        
        # 验证聚合结果
        assert len(stats_results) > 0
        for stats in stats_results:
            assert stats.trading_days > 5
            assert stats.avg_price > 0
            assert stats.max_price >= stats.min_price
            assert stats.total_volume > 0
    
    def test_subquery_operations(self, db_session, db_helper):
        """测试子查询操作"""
        # 创建测试数据
        test_data = db_helper.create_test_data(db_session)
        
        # 子查询：找出价格高于平均价格的交易日
        avg_price_subquery = db_session.query(
            func.avg(StockDailyData.close_price).label('avg_price')
        ).filter(StockDailyData.stock_code == "000001").subquery()
        
        above_avg_query = db_session.query(
            StockDailyData.trade_date,
            StockDailyData.close_price
        ).filter(
            StockDailyData.stock_code == "000001",
            StockDailyData.close_price > avg_price_subquery.c.avg_price
        ).order_by(StockDailyData.trade_date)
        
        above_avg_results = above_avg_query.all()
        
        # 验证子查询结果
        if above_avg_results:
            # 计算实际平均价格进行验证
            all_prices = db_session.query(StockDailyData.close_price).filter(
                StockDailyData.stock_code == "000001"
            ).all()
            actual_avg = sum(float(p.close_price) for p in all_prices) / len(all_prices)
            
            for result in above_avg_results:
                assert float(result.close_price) > actual_avg
    
    def test_window_functions(self, db_session, db_helper):
        """测试窗口函数（如果数据库支持）"""
        # 创建测试数据
        test_data = db_helper.create_test_data(db_session)
        
        # 由于SQLite对窗口函数支持有限，我们使用简单的排序查询来模拟
        # 查询每日价格排名
        ranked_query = db_session.query(
            StockDailyData.trade_date,
            StockDailyData.close_price
        ).filter(
            StockDailyData.stock_code == "000001"
        ).order_by(StockDailyData.close_price.desc())
        
        ranked_results = ranked_query.all()
        
        # 验证排序结果
        assert len(ranked_results) > 0
        for i in range(1, len(ranked_results)):
            assert ranked_results[i-1].close_price >= ranked_results[i].close_price
    
    def test_transaction_rollback_integration(self, db_session, db_helper):
        """测试事务回滚集成"""
        # 创建初始数据
        initial_stock = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),
            close_price=Decimal("10.00")
        )
        db_session.add(initial_stock)
        db_session.commit()
        
        initial_count = db_session.query(StockDailyData).count()
        
        # 开始事务，尝试插入数据，然后回滚
        try:
            # 插入有效数据
            valid_stock = StockDailyData(
                stock_code="000002",
                trade_date=date.today(),
                close_price=Decimal("20.00")
            )
            db_session.add(valid_stock)
            
            # 插入会导致约束违反的数据
            duplicate_stock = StockDailyData(
                stock_code="000001",  # 重复的股票代码和日期
                trade_date=date.today(),
                close_price=Decimal("15.00")
            )
            db_session.add(duplicate_stock)
            
            # 提交事务（应该失败）
            db_session.commit()
            
        except IntegrityError:
            # 回滚事务
            db_session.rollback()
        
        # 验证回滚后数据状态
        final_count = db_session.query(StockDailyData).count()
        assert final_count == initial_count  # 数据应该回到初始状态
    
    def test_batch_operations_performance(self, db_session, db_helper):
        """测试批量操作性能"""
        import time
        
        # 创建大量测试数据
        batch_size = 1000
        stock_data = []
        
        for i in range(batch_size):
            stock_data.append(StockDailyData(
                stock_code=f"{i+1:06d}",
                trade_date=date.today(),
                close_price=Decimal(str(10.0 + i * 0.01)),
                volume=1000000 + i
            ))
        
        # 测试批量插入性能
        start_time = time.time()
        db_session.add_all(stock_data)
        db_session.commit()
        insert_time = time.time() - start_time
        
        # 验证插入结果
        count = db_session.query(StockDailyData).count()
        assert count == batch_size
        
        # 性能应该在合理范围内
        assert insert_time < 10.0  # 批量插入应该在10秒内完成
        
        print(f"Batch insert of {batch_size} records took {insert_time:.3f} seconds")
        
        # 测试批量查询性能
        start_time = time.time()
        results = db_session.query(StockDailyData).filter(
            StockDailyData.close_price > Decimal("15.0")
        ).all()
        query_time = time.time() - start_time
        
        assert query_time < 1.0  # 查询应该在1秒内完成
        print(f"Batch query returned {len(results)} records in {query_time:.3f} seconds")
    
    def test_concurrent_database_access(self, db_helper):
        """测试并发数据库访问"""
        def worker_function(worker_id: int, num_operations: int) -> Dict[str, Any]:
            """工作线程函数"""
            session = db_helper.get_session()
            results = {
                'worker_id': worker_id,
                'successful_operations': 0,
                'failed_operations': 0,
                'errors': []
            }
            
            try:
                for i in range(num_operations):
                    try:
                        # 插入数据
                        stock_data = StockDailyData(
                            stock_code=f"{worker_id:03d}{i:03d}",
                            trade_date=date.today(),
                            close_price=Decimal(str(10.0 + i * 0.1)),
                            volume=1000000 + i
                        )
                        session.add(stock_data)
                        session.commit()
                        results['successful_operations'] += 1
                        
                    except Exception as e:
                        session.rollback()
                        results['failed_operations'] += 1
                        results['errors'].append(str(e))
                        
            finally:
                session.close()
            
            return results
        
        # 启动多个并发工作线程
        num_workers = 5
        operations_per_worker = 20
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(worker_function, i, operations_per_worker) 
                for i in range(num_workers)
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 验证并发操作结果
        total_successful = sum(r['successful_operations'] for r in results)
        total_failed = sum(r['failed_operations'] for r in results)
        
        # 大部分操作应该成功
        success_rate = total_successful / (total_successful + total_failed)
        assert success_rate > 0.8  # 至少80%的操作应该成功
        
        # 验证数据库中的最终数据
        session = db_helper.get_session()
        try:
            final_count = session.query(StockDailyData).count()
            assert final_count == total_successful
        finally:
            session.close()
    
    def test_data_consistency_across_tables(self, db_session, db_helper):
        """测试跨表数据一致性"""
        # 创建关联数据
        # 1. 创建龙虎榜主表数据
        dt_board = DragonTigerBoard(
            trade_date=date.today(),
            stock_code="000001",
            stock_name="平安银行",
            close_price=Decimal("10.50"),
            net_buy_amount=1000000
        )
        db_session.add(dt_board)
        db_session.commit()
        
        # 2. 创建龙虎榜明细数据
        dt_details = [
            DragonTigerDetail(
                board_id=dt_board.id,
                trade_date=date.today(),
                stock_code="000001",
                seat_name="机构专用",
                seat_type="机构",
                buy_amount=600000,
                sell_amount=100000,
                net_amount=500000,
                rank=1
            ),
            DragonTigerDetail(
                board_id=dt_board.id,
                trade_date=date.today(),
                stock_code="000001",
                seat_name="某某营业部",
                seat_type="营业部",
                buy_amount=400000,
                sell_amount=0,
                net_amount=400000,
                rank=2
            )
        ]
        db_session.add_all(dt_details)
        db_session.commit()
        
        # 3. 验证数据一致性
        # 检查明细数据的净买入金额总和是否与主表一致
        detail_net_sum = db_session.query(
            func.sum(DragonTigerDetail.net_amount)
        ).filter(
            DragonTigerDetail.board_id == dt_board.id
        ).scalar()
        
        # 允许一定的误差（因为可能有其他未记录的交易）
        assert detail_net_sum <= dt_board.net_buy_amount
        
        # 验证关联关系
        board_with_details = db_session.query(DragonTigerBoard).filter(
            DragonTigerBoard.id == dt_board.id
        ).first()
        
        details_count = db_session.query(DragonTigerDetail).filter(
            DragonTigerDetail.board_id == dt_board.id
        ).count()
        
        assert board_with_details is not None
        assert details_count == 2
    
    def test_database_constraints_enforcement(self, db_session, db_helper):
        """测试数据库约束执行"""
        # 测试唯一约束
        # 创建第一条记录
        stock1 = StockDailyData(
            stock_code="000001",
            trade_date=date.today(),
            close_price=Decimal("10.00")
        )
        db_session.add(stock1)
        db_session.commit()
        
        # 尝试创建违反唯一约束的记录
        stock2 = StockDailyData(
            stock_code="000001",  # 相同股票代码
            trade_date=date.today(),  # 相同交易日期
            close_price=Decimal("11.00")
        )
        db_session.add(stock2)
        
        with pytest.raises(IntegrityError):
            db_session.commit()
        
        db_session.rollback()
        
        # 测试非空约束
        with pytest.raises((IntegrityError, SQLAlchemyError)):
            invalid_stock = StockDailyData(
                stock_code=None,  # 违反非空约束
                trade_date=date.today(),
                close_price=Decimal("10.00")
            )
            db_session.add(invalid_stock)
            db_session.commit()
    
    def test_index_performance_impact(self, db_session, db_helper):
        """测试索引对性能的影响"""
        # 创建大量测试数据
        large_dataset = []
        for i in range(1000):
            large_dataset.append(StockDailyData(
                stock_code=f"{i%100:06d}",  # 100个不同的股票代码
                trade_date=date.today() - timedelta(days=i%30),  # 30个不同的日期
                close_price=Decimal(str(10.0 + i * 0.01)),
                volume=1000000 + i
            ))
        
        db_session.add_all(large_dataset)
        db_session.commit()
        
        # 测试索引查询性能
        import time
        
        # 按股票代码查询（应该使用索引）
        start_time = time.time()
        stock_results = db_session.query(StockDailyData).filter(
            StockDailyData.stock_code == "000001"
        ).all()
        indexed_query_time = time.time() - start_time
        
        # 按交易日期查询（应该使用索引）
        start_time = time.time()
        date_results = db_session.query(StockDailyData).filter(
            StockDailyData.trade_date == date.today()
        ).all()
        date_query_time = time.time() - start_time
        
        # 复合索引查询
        start_time = time.time()
        compound_results = db_session.query(StockDailyData).filter(
            StockDailyData.stock_code == "000001",
            StockDailyData.trade_date == date.today()
        ).all()
        compound_query_time = time.time() - start_time
        
        # 验证查询性能
        assert indexed_query_time < 0.5  # 索引查询应该很快
        assert date_query_time < 0.5
        assert compound_query_time < 0.1  # 复合索引查询应该最快
        
        print(f"Indexed query times - Stock: {indexed_query_time:.4f}s, "
              f"Date: {date_query_time:.4f}s, Compound: {compound_query_time:.4f}s")


class TestDatabaseConnectionPool:
    """数据库连接池测试"""
    
    def test_connection_pool_behavior(self, db_helper):
        """测试连接池行为"""
        # 创建带连接池的引擎
        pooled_engine = create_engine(
            "sqlite:///:memory:",
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            connect_args={"check_same_thread": False}
        )
        
        # 创建表结构
        Base.metadata.create_all(bind=pooled_engine)
        
        SessionLocal = sessionmaker(bind=pooled_engine)
        
        def worker_with_pool(worker_id: int) -> Dict[str, Any]:
            """使用连接池的工作函数"""
            session = SessionLocal()
            start_time = time.time()
            
            try:
                # 执行一些数据库操作
                for i in range(10):
                    stock_data = StockDailyData(
                        stock_code=f"{worker_id:03d}{i:03d}",
                        trade_date=date.today(),
                        close_price=Decimal(str(10.0 + i * 0.1))
                    )
                    session.add(stock_data)
                
                session.commit()
                
                # 执行查询
                results = session.query(StockDailyData).filter(
                    StockDailyData.stock_code.like(f"{worker_id:03d}%")
                ).all()
                
                end_time = time.time()
                
                return {
                    'worker_id': worker_id,
                    'execution_time': end_time - start_time,
                    'records_processed': len(results),
                    'success': True
                }
                
            except Exception as e:
                return {
                    'worker_id': worker_id,
                    'execution_time': time.time() - start_time,
                    'error': str(e),
                    'success': False
                }
            finally:
                session.close()
        
        # 并发测试连接池
        num_workers = 15  # 超过连接池大小，测试overflow
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_with_pool, i) for i in range(num_workers)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 验证连接池效果
        successful_workers = [r for r in results if r['success']]
        assert len(successful_workers) == num_workers  # 所有工作线程都应该成功
        
        # 计算平均执行时间
        avg_execution_time = sum(r['execution_time'] for r in successful_workers) / len(successful_workers)
        print(f"Average execution time with connection pool: {avg_execution_time:.3f}s")
        
        # 连接池应该提供合理的性能
        assert avg_execution_time < 2.0  # 平均执行时间应该在合理范围内


class TestDatabaseMigrationIntegration:
    """数据库迁移集成测试"""
    
    def test_schema_migration_simulation(self, db_helper):
        """测试数据库模式迁移模拟"""
        session = db_helper.get_session()
        
        # 1. 创建初始数据
        initial_data = [
            StockDailyData(
                stock_code="000001",
                trade_date=date.today() - timedelta(days=i),
                close_price=Decimal(str(10.0 + i * 0.1)),
                volume=1000000 + i * 10000
            ) for i in range(5)
        ]
        
        session.add_all(initial_data)
        session.commit()
        
        # 2. 验证初始数据
        initial_count = session.query(StockDailyData).count()
        assert initial_count == 5
        
        # 3. 模拟添加新字段的迁移（通过查询模拟）
        # 在实际迁移中，这会是ALTER TABLE语句
        migration_query = text("""
            SELECT stock_code, trade_date, close_price, volume,
                   CASE WHEN close_price > 10.2 THEN 'HIGH' ELSE 'LOW' END as price_level
            FROM stock_daily_data
        """)
        
        migrated_results = session.execute(migration_query).fetchall()
        
        # 4. 验证迁移结果
        assert len(migrated_results) == initial_count
        for result in migrated_results:
            assert result.price_level in ['HIGH', 'LOW']
            if result.close_price > 10.2:
                assert result.price_level == 'HIGH'
            else:
                assert result.price_level == 'LOW'
        
        session.close()
    
    def test_data_migration_with_transformation(self, db_helper):
        """测试带数据转换的迁移"""
        session = db_helper.get_session()
        
        # 1. 创建"旧格式"数据（模拟）
        old_format_data = [
            {
                'code': '000001',
                'date': '20240120',
                'price': '10.50',
                'vol': '1000000'
            },
            {
                'code': '000002',
                'date': '20240120',
                'price': '20.30',
                'vol': '2000000'
            }
        ]
        
        # 2. 模拟数据转换和迁移过程
        migrated_data = []
        for old_record in old_format_data:
            # 数据转换逻辑
            new_record = StockDailyData(
                stock_code=old_record['code'],
                trade_date=datetime.strptime(old_record['date'], '%Y%m%d').date(),
                close_price=Decimal(old_record['price']),
                volume=int(old_record['vol'])
            )
            migrated_data.append(new_record)
        
        # 3. 插入转换后的数据
        session.add_all(migrated_data)
        session.commit()
        
        # 4. 验证迁移结果
        migrated_count = session.query(StockDailyData).count()
        assert migrated_count == len(old_format_data)
        
        # 验证数据转换正确性
        for i, record in enumerate(session.query(StockDailyData).all()):
            original = old_format_data[i]
            assert record.stock_code == original['code']
            assert record.close_price == Decimal(original['price'])
            assert record.volume == int(original['vol'])
        
        session.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])