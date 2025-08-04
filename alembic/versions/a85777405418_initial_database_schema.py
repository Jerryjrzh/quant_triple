"""Initial database schema

Revision ID: a85777405418
Revises: 
Create Date: 2025-08-01 17:22:01.786293

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a85777405418'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create stock_daily_data table
    op.create_table(
        'stock_daily_data',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('trade_date', sa.Date(), nullable=False),
        sa.Column('open_price', sa.Numeric(precision=10, scale=3), nullable=True),
        sa.Column('high_price', sa.Numeric(precision=10, scale=3), nullable=True),
        sa.Column('low_price', sa.Numeric(precision=10, scale=3), nullable=True),
        sa.Column('close_price', sa.Numeric(precision=10, scale=3), nullable=True),
        sa.Column('volume', sa.BigInteger(), nullable=True),
        sa.Column('amount', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('adj_factor', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('stock_code', 'trade_date', name='uq_stock_daily_data')
    )
    op.create_index(op.f('ix_stock_daily_data_id'), 'stock_daily_data', ['id'], unique=False)
    op.create_index(op.f('ix_stock_daily_data_stock_code'), 'stock_daily_data', ['stock_code'], unique=False)
    op.create_index(op.f('ix_stock_daily_data_trade_date'), 'stock_daily_data', ['trade_date'], unique=False)

    # Create dragon_tiger_list table
    op.create_table(
        'dragon_tiger_list',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('trade_date', sa.Date(), nullable=False),
        sa.Column('seat_name', sa.String(length=200), nullable=True),
        sa.Column('buy_amount', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('sell_amount', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('net_amount', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('seat_type', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_dragon_tiger_list_id'), 'dragon_tiger_list', ['id'], unique=False)
    op.create_index(op.f('ix_dragon_tiger_list_stock_code'), 'dragon_tiger_list', ['stock_code'], unique=False)
    op.create_index(op.f('ix_dragon_tiger_list_trade_date'), 'dragon_tiger_list', ['trade_date'], unique=False)

    # Create spring_festival_analysis table
    op.create_table(
        'spring_festival_analysis',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('analysis_year', sa.Integer(), nullable=False),
        sa.Column('spring_festival_date', sa.Date(), nullable=False),
        sa.Column('normalized_data', sa.JSON(), nullable=True),
        sa.Column('pattern_score', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('volatility_profile', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_spring_festival_analysis_id'), 'spring_festival_analysis', ['id'], unique=False)
    op.create_index(op.f('ix_spring_festival_analysis_stock_code'), 'spring_festival_analysis', ['stock_code'], unique=False)
    op.create_index(op.f('ix_spring_festival_analysis_analysis_year'), 'spring_festival_analysis', ['analysis_year'], unique=False)

    # Create institutional_activity table
    op.create_table(
        'institutional_activity',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('institution_name', sa.String(length=200), nullable=False),
        sa.Column('institution_type', sa.String(length=50), nullable=False),
        sa.Column('activity_date', sa.Date(), nullable=False),
        sa.Column('activity_type', sa.String(length=50), nullable=False),
        sa.Column('position_change', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('total_position', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('confidence_score', sa.Numeric(precision=3, scale=2), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_institutional_activity_id'), 'institutional_activity', ['id'], unique=False)
    op.create_index(op.f('ix_institutional_activity_stock_code'), 'institutional_activity', ['stock_code'], unique=False)
    op.create_index(op.f('ix_institutional_activity_institution_type'), 'institutional_activity', ['institution_type'], unique=False)
    op.create_index(op.f('ix_institutional_activity_activity_date'), 'institutional_activity', ['activity_date'], unique=False)

    # Create risk_metrics table
    op.create_table(
        'risk_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('calculation_date', sa.Date(), nullable=False),
        sa.Column('var_1d_95', sa.Numeric(precision=8, scale=6), nullable=True),
        sa.Column('var_1d_99', sa.Numeric(precision=8, scale=6), nullable=True),
        sa.Column('historical_volatility', sa.Numeric(precision=8, scale=6), nullable=True),
        sa.Column('beta', sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column('max_drawdown', sa.Numeric(precision=8, scale=6), nullable=True),
        sa.Column('sharpe_ratio', sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column('seasonal_risk_score', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_risk_metrics_id'), 'risk_metrics', ['id'], unique=False)
    op.create_index(op.f('ix_risk_metrics_stock_code'), 'risk_metrics', ['stock_code'], unique=False)
    op.create_index(op.f('ix_risk_metrics_calculation_date'), 'risk_metrics', ['calculation_date'], unique=False)

    # Create stock_pools table
    op.create_table(
        'stock_pools',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('pool_type', sa.String(length=50), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index(op.f('ix_stock_pools_id'), 'stock_pools', ['id'], unique=False)

    # Create stock_pool_members table
    op.create_table(
        'stock_pool_members',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('pool_id', sa.Integer(), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('added_at', sa.DateTime(), nullable=False),
        sa.Column('added_reason', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('pool_id', 'stock_code', name='uq_pool_member')
    )
    op.create_index(op.f('ix_stock_pool_members_id'), 'stock_pool_members', ['id'], unique=False)
    op.create_index(op.f('ix_stock_pool_members_pool_id'), 'stock_pool_members', ['pool_id'], unique=False)
    op.create_index(op.f('ix_stock_pool_members_stock_code'), 'stock_pool_members', ['stock_code'], unique=False)

    # Create alert_rules table
    op.create_table(
        'alert_rules',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rule_id', sa.String(length=50), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('condition_type', sa.String(length=50), nullable=False),
        sa.Column('condition_parameters', sa.JSON(), nullable=False),
        sa.Column('notification_channels', sa.JSON(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('last_triggered', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('rule_id')
    )
    op.create_index(op.f('ix_alert_rules_id'), 'alert_rules', ['id'], unique=False)
    op.create_index(op.f('ix_alert_rules_stock_code'), 'alert_rules', ['stock_code'], unique=False)

    # Create alert_history table
    op.create_table(
        'alert_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rule_id', sa.String(length=50), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('triggered_at', sa.DateTime(), nullable=False),
        sa.Column('trigger_value', sa.Numeric(precision=15, scale=6), nullable=True),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('notification_sent', sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_alert_history_id'), 'alert_history', ['id'], unique=False)
    op.create_index(op.f('ix_alert_history_rule_id'), 'alert_history', ['rule_id'], unique=False)
    op.create_index(op.f('ix_alert_history_stock_code'), 'alert_history', ['stock_code'], unique=False)

    # Create user_sessions table
    op.create_table(
        'user_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.String(length=100), nullable=False),
        sa.Column('user_id', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('last_activity', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id')
    )
    op.create_index(op.f('ix_user_sessions_id'), 'user_sessions', ['id'], unique=False)
    op.create_index(op.f('ix_user_sessions_session_id'), 'user_sessions', ['session_id'], unique=False)
    op.create_index(op.f('ix_user_sessions_user_id'), 'user_sessions', ['user_id'], unique=False)

    # Create system_config table
    op.create_table(
        'system_config',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('config_key', sa.String(length=100), nullable=False),
        sa.Column('config_value', sa.JSON(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('config_key')
    )
    op.create_index(op.f('ix_system_config_id'), 'system_config', ['id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop tables in reverse order
    op.drop_table('system_config')
    op.drop_table('user_sessions')
    op.drop_table('alert_history')
    op.drop_table('alert_rules')
    op.drop_table('stock_pool_members')
    op.drop_table('stock_pools')
    op.drop_table('risk_metrics')
    op.drop_table('institutional_activity')
    op.drop_table('spring_festival_analysis')
    op.drop_table('dragon_tiger_list')
    op.drop_table('stock_daily_data')
