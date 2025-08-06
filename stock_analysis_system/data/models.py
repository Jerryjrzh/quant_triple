"""Database models for the stock analysis system."""

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from stock_analysis_system.core.database import Base


class StockDailyData(Base):
    """Stock daily trading data model."""

    __tablename__ = "stock_daily_data"
    __table_args__ = (
        UniqueConstraint("stock_code", "trade_date", name="uq_stock_daily_data"),
    )

    id = Column(Integer, primary_key=True, index=True)
    stock_code = Column(String(10), nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    open_price = Column(Numeric(10, 3), nullable=True)
    high_price = Column(Numeric(10, 3), nullable=True)
    low_price = Column(Numeric(10, 3), nullable=True)
    close_price = Column(Numeric(10, 3), nullable=True)
    volume = Column(BigInteger, nullable=True)
    amount = Column(Numeric(15, 2), nullable=True)
    adj_factor = Column(Numeric(10, 6), nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Alert(Base):
    """Alert model for storing alert configurations and status."""

    __tablename__ = "alerts"

    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    stock_code = Column(String(10), nullable=True, index=True)
    trigger_type = Column(String(50), nullable=False)
    priority = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False, default="active")
    user_id = Column(String(50), nullable=True, index=True)
    trigger_count = Column(Integer, default=0)
    last_triggered = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    alert_metadata = Column(JSON, nullable=True)


class NotificationLog(Base):
    """Notification log model for tracking notification delivery."""

    __tablename__ = "notification_logs"

    id = Column(String(100), primary_key=True, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    alert_id = Column(String(50), nullable=False, index=True)
    channel = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False)
    sent_at = Column(DateTime, nullable=False)
    delivered_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    notification_metadata = Column(JSON, nullable=True)


class DragonTigerList(Base):
    """Dragon-tiger list data model for institutional activity tracking."""

    __tablename__ = "dragon_tiger_list"

    id = Column(Integer, primary_key=True, index=True)
    stock_code = Column(String(10), nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    seat_name = Column(String(200), nullable=True)
    buy_amount = Column(Numeric(15, 2), nullable=True)
    sell_amount = Column(Numeric(15, 2), nullable=True)
    net_amount = Column(Numeric(15, 2), nullable=True)
    seat_type = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)


class SpringFestivalAnalysis(Base):
    """Spring Festival alignment analysis cache model."""

    __tablename__ = "spring_festival_analysis"

    id = Column(Integer, primary_key=True, index=True)
    stock_code = Column(String(10), nullable=False, index=True)
    analysis_year = Column(Integer, nullable=False, index=True)
    spring_festival_date = Column(Date, nullable=False)
    normalized_data = Column(JSON, nullable=True)
    pattern_score = Column(Numeric(5, 2), nullable=True)
    volatility_profile = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class InstitutionalActivity(Base):
    """Institutional activity tracking model."""

    __tablename__ = "institutional_activity"

    id = Column(Integer, primary_key=True, index=True)
    stock_code = Column(String(10), nullable=False, index=True)
    institution_name = Column(String(200), nullable=False)
    institution_type = Column(String(50), nullable=False, index=True)
    activity_date = Column(Date, nullable=False, index=True)
    activity_type = Column(String(50), nullable=False)
    position_change = Column(Numeric(15, 2), nullable=True)
    total_position = Column(Numeric(15, 2), nullable=True)
    confidence_score = Column(Numeric(3, 2), nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)


class RiskMetrics(Base):
    """Risk metrics calculation results model."""

    __tablename__ = "risk_metrics"

    id = Column(Integer, primary_key=True, index=True)
    stock_code = Column(String(10), nullable=False, index=True)
    calculation_date = Column(Date, nullable=False, index=True)
    var_1d_95 = Column(Numeric(8, 6), nullable=True)  # 1-day VaR at 95%
    var_1d_99 = Column(Numeric(8, 6), nullable=True)  # 1-day VaR at 99%
    historical_volatility = Column(Numeric(8, 6), nullable=True)
    beta = Column(Numeric(6, 4), nullable=True)
    max_drawdown = Column(Numeric(8, 6), nullable=True)
    sharpe_ratio = Column(Numeric(6, 4), nullable=True)
    seasonal_risk_score = Column(Numeric(5, 2), nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)


class StockPool(Base):
    """Stock pool management model."""

    __tablename__ = "stock_pools"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    pool_type = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class StockPoolMember(Base):
    """Stock pool membership model."""

    __tablename__ = "stock_pool_members"
    __table_args__ = (UniqueConstraint("pool_id", "stock_code", name="uq_pool_member"),)

    id = Column(Integer, primary_key=True, index=True)
    pool_id = Column(Integer, nullable=False, index=True)
    stock_code = Column(String(10), nullable=False, index=True)
    added_at = Column(DateTime, default=func.now(), nullable=False)
    added_reason = Column(Text, nullable=True)


class AlertRule(Base):
    """Alert rule configuration model."""

    __tablename__ = "alert_rules"

    id = Column(Integer, primary_key=True, index=True)
    rule_id = Column(String(50), nullable=False, unique=True)
    stock_code = Column(String(10), nullable=False, index=True)
    condition_type = Column(String(50), nullable=False)
    condition_parameters = Column(JSON, nullable=False)
    notification_channels = Column(JSON, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_triggered = Column(DateTime, nullable=True)


class AlertHistory(Base):
    """Alert trigger history model."""

    __tablename__ = "alert_history"

    id = Column(Integer, primary_key=True, index=True)
    rule_id = Column(String(50), nullable=False, index=True)
    stock_code = Column(String(10), nullable=False, index=True)
    triggered_at = Column(DateTime, default=func.now(), nullable=False)
    trigger_value = Column(Numeric(15, 6), nullable=True)
    message = Column(Text, nullable=True)
    notification_sent = Column(Boolean, default=False, nullable=False)


class UserSession(Base):
    """User session management model."""

    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False, unique=True, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_activity = Column(DateTime, default=func.now(), onupdate=func.now())


class SystemConfig(Base):
    """System configuration model."""

    __tablename__ = "system_config"

    id = Column(Integer, primary_key=True, index=True)
    config_key = Column(String(100), nullable=False, unique=True)
    config_value = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


# 新增爬虫集成相关数据模型

class DragonTigerBoard(Base):
    """龙虎榜数据表 - 增强版本"""

    __tablename__ = "dragon_tiger_board"
    __table_args__ = (
        UniqueConstraint("stock_code", "trade_date", name="uq_dragon_tiger_board"),
    )

    id = Column(Integer, primary_key=True, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    stock_code = Column(String(10), nullable=False, index=True)
    stock_name = Column(String(50), nullable=False)
    close_price = Column(Numeric(10, 2), nullable=True)
    change_rate = Column(Numeric(5, 2), nullable=True)
    net_buy_amount = Column(BigInteger, nullable=True)
    buy_amount = Column(BigInteger, nullable=True)
    sell_amount = Column(BigInteger, nullable=True)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class DragonTigerDetail(Base):
    """龙虎榜详细数据表 - 机构和营业部明细"""

    __tablename__ = "dragon_tiger_detail"

    id = Column(Integer, primary_key=True, index=True)
    board_id = Column(Integer, nullable=False, index=True)  # 关联dragon_tiger_board.id
    trade_date = Column(Date, nullable=False, index=True)
    stock_code = Column(String(10), nullable=False, index=True)
    seat_name = Column(String(200), nullable=False)
    seat_type = Column(String(50), nullable=True)  # 机构/营业部
    buy_amount = Column(BigInteger, nullable=True)
    sell_amount = Column(BigInteger, nullable=True)
    net_amount = Column(BigInteger, nullable=True)
    rank = Column(Integer, nullable=True)  # 排名
    created_at = Column(DateTime, default=func.now(), nullable=False)


class FundFlow(Base):
    """资金流向数据表"""

    __tablename__ = "fund_flow"
    __table_args__ = (
        UniqueConstraint("stock_code", "trade_date", "period_type", name="uq_fund_flow"),
    )

    id = Column(Integer, primary_key=True, index=True)
    stock_code = Column(String(10), nullable=False, index=True)
    stock_name = Column(String(50), nullable=True)
    trade_date = Column(Date, nullable=False, index=True)
    period_type = Column(String(10), nullable=False, index=True)  # 今日/3日/5日/10日
    
    # 主力资金
    main_net_inflow = Column(BigInteger, nullable=True)
    main_net_inflow_rate = Column(Numeric(5, 2), nullable=True)
    
    # 超大单资金
    super_large_net_inflow = Column(BigInteger, nullable=True)
    super_large_net_inflow_rate = Column(Numeric(5, 2), nullable=True)
    
    # 大单资金
    large_net_inflow = Column(BigInteger, nullable=True)
    large_net_inflow_rate = Column(Numeric(5, 2), nullable=True)
    
    # 中单资金
    medium_net_inflow = Column(BigInteger, nullable=True)
    medium_net_inflow_rate = Column(Numeric(5, 2), nullable=True)
    
    # 小单资金
    small_net_inflow = Column(BigInteger, nullable=True)
    small_net_inflow_rate = Column(Numeric(5, 2), nullable=True)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)


class LimitUpReason(Base):
    """涨停原因数据表"""

    __tablename__ = "limitup_reason"
    __table_args__ = (
        UniqueConstraint("stock_code", "trade_date", name="uq_limitup_reason"),
    )

    id = Column(Integer, primary_key=True, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    stock_code = Column(String(10), nullable=False, index=True)
    stock_name = Column(String(50), nullable=False)
    reason = Column(String(200), nullable=True, index=True)  # 简短原因
    detail_reason = Column(Text, nullable=True)  # 详细原因
    latest_price = Column(Numeric(10, 2), nullable=True)
    change_rate = Column(Numeric(5, 2), nullable=True)
    change_amount = Column(Numeric(10, 2), nullable=True)
    turnover_rate = Column(Numeric(5, 2), nullable=True)
    volume = Column(BigInteger, nullable=True)
    amount = Column(BigInteger, nullable=True)
    dde = Column(Numeric(10, 2), nullable=True)  # DDE大单净额
    
    # 分类字段
    reason_category = Column(String(50), nullable=True, index=True)  # 原因分类
    reason_tags = Column(JSON, nullable=True)  # 原因标签
    
    created_at = Column(DateTime, default=func.now(), nullable=False)


class ETFData(Base):
    """ETF数据表"""

    __tablename__ = "etf_data"
    __table_args__ = (
        UniqueConstraint("etf_code", "trade_date", name="uq_etf_data"),
    )

    id = Column(Integer, primary_key=True, index=True)
    etf_code = Column(String(10), nullable=False, index=True)
    etf_name = Column(String(50), nullable=False)
    trade_date = Column(Date, nullable=False, index=True)
    
    # 基础行情数据
    open_price = Column(Numeric(10, 4), nullable=True)
    close_price = Column(Numeric(10, 4), nullable=True)
    high_price = Column(Numeric(10, 4), nullable=True)
    low_price = Column(Numeric(10, 4), nullable=True)
    volume = Column(BigInteger, nullable=True)
    amount = Column(BigInteger, nullable=True)
    change_rate = Column(Numeric(5, 2), nullable=True)
    turnover_rate = Column(Numeric(5, 2), nullable=True)
    
    # ETF特有指标
    unit_nav = Column(Numeric(10, 4), nullable=True)  # 单位净值
    accumulated_nav = Column(Numeric(10, 4), nullable=True)  # 累计净值
    premium_rate = Column(Numeric(5, 2), nullable=True)  # 溢价率
    discount_rate = Column(Numeric(5, 2), nullable=True)  # 折价率
    
    # 基金规模和份额
    fund_size = Column(Numeric(15, 2), nullable=True)  # 基金规模（万元）
    fund_shares = Column(BigInteger, nullable=True)  # 基金份额
    
    created_at = Column(DateTime, default=func.now(), nullable=False)


class ETFConstituent(Base):
    """ETF成分股数据表"""

    __tablename__ = "etf_constituent"

    id = Column(Integer, primary_key=True, index=True)
    etf_code = Column(String(10), nullable=False, index=True)
    stock_code = Column(String(10), nullable=False, index=True)
    stock_name = Column(String(50), nullable=True)
    weight = Column(Numeric(8, 4), nullable=True)  # 权重百分比
    shares = Column(BigInteger, nullable=True)  # 持股数量
    market_value = Column(Numeric(15, 2), nullable=True)  # 市值
    update_date = Column(Date, nullable=False, index=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)


class DataQualityLog(Base):
    """数据质量日志表"""

    __tablename__ = "data_quality_log"

    id = Column(Integer, primary_key=True, index=True)
    data_source = Column(String(50), nullable=False, index=True)
    data_type = Column(String(50), nullable=False, index=True)
    check_date = Column(Date, nullable=False, index=True)
    check_time = Column(DateTime, default=func.now(), nullable=False)
    
    # 质量指标
    total_records = Column(Integer, nullable=True)
    valid_records = Column(Integer, nullable=True)
    invalid_records = Column(Integer, nullable=True)
    duplicate_records = Column(Integer, nullable=True)
    missing_fields = Column(JSON, nullable=True)
    
    # 质量评分
    completeness_score = Column(Numeric(5, 2), nullable=True)  # 完整性评分
    accuracy_score = Column(Numeric(5, 2), nullable=True)  # 准确性评分
    consistency_score = Column(Numeric(5, 2), nullable=True)  # 一致性评分
    timeliness_score = Column(Numeric(5, 2), nullable=True)  # 时效性评分
    overall_score = Column(Numeric(5, 2), nullable=True)  # 总体评分
    
    # 问题详情
    issues_found = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)


class DataSourceHealth(Base):
    """数据源健康状态表"""

    __tablename__ = "data_source_health"

    id = Column(Integer, primary_key=True, index=True)
    source_name = Column(String(50), nullable=False, index=True)
    check_time = Column(DateTime, default=func.now(), nullable=False, index=True)
    
    # 健康状态
    status = Column(String(20), nullable=False, index=True)  # healthy/warning/error
    response_time = Column(Numeric(8, 3), nullable=True)  # 响应时间（秒）
    success_rate = Column(Numeric(5, 2), nullable=True)  # 成功率
    error_rate = Column(Numeric(5, 2), nullable=True)  # 错误率
    
    # 统计信息
    total_requests = Column(Integer, nullable=True)
    successful_requests = Column(Integer, nullable=True)
    failed_requests = Column(Integer, nullable=True)
    
    # 错误详情
    last_error = Column(Text, nullable=True)
    error_count_24h = Column(Integer, nullable=True)
    
    # 性能指标
    avg_response_time = Column(Numeric(8, 3), nullable=True)
    max_response_time = Column(Numeric(8, 3), nullable=True)
    min_response_time = Column(Numeric(8, 3), nullable=True)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
