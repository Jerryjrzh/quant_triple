"""Database models for the stock analysis system."""

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Integer,
    JSON,
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
    __table_args__ = (
        UniqueConstraint("pool_id", "stock_code", name="uq_pool_member"),
    )

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