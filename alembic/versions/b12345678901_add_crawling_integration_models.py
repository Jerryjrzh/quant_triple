"""add crawling integration models

Revision ID: b12345678901
Revises: a85777405418
Create Date: 2024-01-20 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'b12345678901'
down_revision = 'a85777405418'
branch_labels = None
depends_on = None


def upgrade():
    """Create new tables for crawling integration."""
    
    # Create dragon_tiger_board table (enhanced version)
    op.create_table('dragon_tiger_board',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('trade_date', sa.Date(), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('stock_name', sa.String(length=50), nullable=False),
        sa.Column('close_price', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('change_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('net_buy_amount', sa.BigInteger(), nullable=True),
        sa.Column('buy_amount', sa.BigInteger(), nullable=True),
        sa.Column('sell_amount', sa.BigInteger(), nullable=True),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('stock_code', 'trade_date', name='uq_dragon_tiger_board')
    )
    op.create_index(op.f('ix_dragon_tiger_board_id'), 'dragon_tiger_board', ['id'], unique=False)
    op.create_index(op.f('ix_dragon_tiger_board_trade_date'), 'dragon_tiger_board', ['trade_date'], unique=False)
    op.create_index(op.f('ix_dragon_tiger_board_stock_code'), 'dragon_tiger_board', ['stock_code'], unique=False)

    # Create dragon_tiger_detail table
    op.create_table('dragon_tiger_detail',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('board_id', sa.Integer(), nullable=False),
        sa.Column('trade_date', sa.Date(), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('seat_name', sa.String(length=200), nullable=False),
        sa.Column('seat_type', sa.String(length=50), nullable=True),
        sa.Column('buy_amount', sa.BigInteger(), nullable=True),
        sa.Column('sell_amount', sa.BigInteger(), nullable=True),
        sa.Column('net_amount', sa.BigInteger(), nullable=True),
        sa.Column('rank', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_dragon_tiger_detail_id'), 'dragon_tiger_detail', ['id'], unique=False)
    op.create_index(op.f('ix_dragon_tiger_detail_board_id'), 'dragon_tiger_detail', ['board_id'], unique=False)
    op.create_index(op.f('ix_dragon_tiger_detail_trade_date'), 'dragon_tiger_detail', ['trade_date'], unique=False)
    op.create_index(op.f('ix_dragon_tiger_detail_stock_code'), 'dragon_tiger_detail', ['stock_code'], unique=False)

    # Create fund_flow table
    op.create_table('fund_flow',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('stock_name', sa.String(length=50), nullable=True),
        sa.Column('trade_date', sa.Date(), nullable=False),
        sa.Column('period_type', sa.String(length=10), nullable=False),
        sa.Column('main_net_inflow', sa.BigInteger(), nullable=True),
        sa.Column('main_net_inflow_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('super_large_net_inflow', sa.BigInteger(), nullable=True),
        sa.Column('super_large_net_inflow_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('large_net_inflow', sa.BigInteger(), nullable=True),
        sa.Column('large_net_inflow_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('medium_net_inflow', sa.BigInteger(), nullable=True),
        sa.Column('medium_net_inflow_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('small_net_inflow', sa.BigInteger(), nullable=True),
        sa.Column('small_net_inflow_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('stock_code', 'trade_date', 'period_type', name='uq_fund_flow')
    )
    op.create_index(op.f('ix_fund_flow_id'), 'fund_flow', ['id'], unique=False)
    op.create_index(op.f('ix_fund_flow_stock_code'), 'fund_flow', ['stock_code'], unique=False)
    op.create_index(op.f('ix_fund_flow_trade_date'), 'fund_flow', ['trade_date'], unique=False)
    op.create_index(op.f('ix_fund_flow_period_type'), 'fund_flow', ['period_type'], unique=False)

    # Create limitup_reason table
    op.create_table('limitup_reason',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('trade_date', sa.Date(), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('stock_name', sa.String(length=50), nullable=False),
        sa.Column('reason', sa.String(length=200), nullable=True),
        sa.Column('detail_reason', sa.Text(), nullable=True),
        sa.Column('latest_price', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('change_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('change_amount', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('turnover_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('volume', sa.BigInteger(), nullable=True),
        sa.Column('amount', sa.BigInteger(), nullable=True),
        sa.Column('dde', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('reason_category', sa.String(length=50), nullable=True),
        sa.Column('reason_tags', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('stock_code', 'trade_date', name='uq_limitup_reason')
    )
    op.create_index(op.f('ix_limitup_reason_id'), 'limitup_reason', ['id'], unique=False)
    op.create_index(op.f('ix_limitup_reason_trade_date'), 'limitup_reason', ['trade_date'], unique=False)
    op.create_index(op.f('ix_limitup_reason_stock_code'), 'limitup_reason', ['stock_code'], unique=False)
    op.create_index(op.f('ix_limitup_reason_reason'), 'limitup_reason', ['reason'], unique=False)
    op.create_index(op.f('ix_limitup_reason_reason_category'), 'limitup_reason', ['reason_category'], unique=False)

    # Create etf_data table
    op.create_table('etf_data',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('etf_code', sa.String(length=10), nullable=False),
        sa.Column('etf_name', sa.String(length=50), nullable=False),
        sa.Column('trade_date', sa.Date(), nullable=False),
        sa.Column('open_price', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('close_price', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('high_price', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('low_price', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('volume', sa.BigInteger(), nullable=True),
        sa.Column('amount', sa.BigInteger(), nullable=True),
        sa.Column('change_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('turnover_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('unit_nav', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('accumulated_nav', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('premium_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('discount_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('fund_size', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('fund_shares', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('etf_code', 'trade_date', name='uq_etf_data')
    )
    op.create_index(op.f('ix_etf_data_id'), 'etf_data', ['id'], unique=False)
    op.create_index(op.f('ix_etf_data_etf_code'), 'etf_data', ['etf_code'], unique=False)
    op.create_index(op.f('ix_etf_data_trade_date'), 'etf_data', ['trade_date'], unique=False)

    # Create etf_constituent table
    op.create_table('etf_constituent',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('etf_code', sa.String(length=10), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('stock_name', sa.String(length=50), nullable=True),
        sa.Column('weight', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('shares', sa.BigInteger(), nullable=True),
        sa.Column('market_value', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('update_date', sa.Date(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_etf_constituent_id'), 'etf_constituent', ['id'], unique=False)
    op.create_index(op.f('ix_etf_constituent_etf_code'), 'etf_constituent', ['etf_code'], unique=False)
    op.create_index(op.f('ix_etf_constituent_stock_code'), 'etf_constituent', ['stock_code'], unique=False)
    op.create_index(op.f('ix_etf_constituent_update_date'), 'etf_constituent', ['update_date'], unique=False)

    # Create data_quality_log table
    op.create_table('data_quality_log',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('data_source', sa.String(length=50), nullable=False),
        sa.Column('data_type', sa.String(length=50), nullable=False),
        sa.Column('check_date', sa.Date(), nullable=False),
        sa.Column('check_time', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('total_records', sa.Integer(), nullable=True),
        sa.Column('valid_records', sa.Integer(), nullable=True),
        sa.Column('invalid_records', sa.Integer(), nullable=True),
        sa.Column('duplicate_records', sa.Integer(), nullable=True),
        sa.Column('missing_fields', sa.JSON(), nullable=True),
        sa.Column('completeness_score', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('accuracy_score', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('consistency_score', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('timeliness_score', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('overall_score', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('issues_found', sa.JSON(), nullable=True),
        sa.Column('recommendations', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_data_quality_log_id'), 'data_quality_log', ['id'], unique=False)
    op.create_index(op.f('ix_data_quality_log_data_source'), 'data_quality_log', ['data_source'], unique=False)
    op.create_index(op.f('ix_data_quality_log_data_type'), 'data_quality_log', ['data_type'], unique=False)
    op.create_index(op.f('ix_data_quality_log_check_date'), 'data_quality_log', ['check_date'], unique=False)

    # Create data_source_health table
    op.create_table('data_source_health',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_name', sa.String(length=50), nullable=False),
        sa.Column('check_time', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('response_time', sa.Numeric(precision=8, scale=3), nullable=True),
        sa.Column('success_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('error_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('total_requests', sa.Integer(), nullable=True),
        sa.Column('successful_requests', sa.Integer(), nullable=True),
        sa.Column('failed_requests', sa.Integer(), nullable=True),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('error_count_24h', sa.Integer(), nullable=True),
        sa.Column('avg_response_time', sa.Numeric(precision=8, scale=3), nullable=True),
        sa.Column('max_response_time', sa.Numeric(precision=8, scale=3), nullable=True),
        sa.Column('min_response_time', sa.Numeric(precision=8, scale=3), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_data_source_health_id'), 'data_source_health', ['id'], unique=False)
    op.create_index(op.f('ix_data_source_health_source_name'), 'data_source_health', ['source_name'], unique=False)
    op.create_index(op.f('ix_data_source_health_check_time'), 'data_source_health', ['check_time'], unique=False)
    op.create_index(op.f('ix_data_source_health_status'), 'data_source_health', ['status'], unique=False)

    # Create partitions for dragon_tiger_board (by month)
    op.execute("""
        -- Create partitioned table for dragon_tiger_board
        CREATE TABLE dragon_tiger_board_partitioned (
            LIKE dragon_tiger_board INCLUDING ALL
        ) PARTITION BY RANGE (trade_date);
        
        -- Create monthly partitions for current year
        CREATE TABLE dragon_tiger_board_y2024m01 PARTITION OF dragon_tiger_board_partitioned
            FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
        CREATE TABLE dragon_tiger_board_y2024m02 PARTITION OF dragon_tiger_board_partitioned
            FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
        CREATE TABLE dragon_tiger_board_y2024m03 PARTITION OF dragon_tiger_board_partitioned
            FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');
        CREATE TABLE dragon_tiger_board_y2024m04 PARTITION OF dragon_tiger_board_partitioned
            FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');
        CREATE TABLE dragon_tiger_board_y2024m05 PARTITION OF dragon_tiger_board_partitioned
            FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');
        CREATE TABLE dragon_tiger_board_y2024m06 PARTITION OF dragon_tiger_board_partitioned
            FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');
        CREATE TABLE dragon_tiger_board_y2024m07 PARTITION OF dragon_tiger_board_partitioned
            FOR VALUES FROM ('2024-07-01') TO ('2024-08-01');
        CREATE TABLE dragon_tiger_board_y2024m08 PARTITION OF dragon_tiger_board_partitioned
            FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');
        CREATE TABLE dragon_tiger_board_y2024m09 PARTITION OF dragon_tiger_board_partitioned
            FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');
        CREATE TABLE dragon_tiger_board_y2024m10 PARTITION OF dragon_tiger_board_partitioned
            FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');
        CREATE TABLE dragon_tiger_board_y2024m11 PARTITION OF dragon_tiger_board_partitioned
            FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');
        CREATE TABLE dragon_tiger_board_y2024m12 PARTITION OF dragon_tiger_board_partitioned
            FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');
    """)

    # Create additional indexes for performance optimization
    op.execute("""
        -- Composite indexes for common query patterns
        CREATE INDEX idx_dragon_tiger_board_date_code ON dragon_tiger_board(trade_date, stock_code);
        CREATE INDEX idx_fund_flow_code_date ON fund_flow(stock_code, trade_date);
        CREATE INDEX idx_limitup_date ON limitup_reason(trade_date);
        CREATE INDEX idx_etf_code_date ON etf_data(etf_code, trade_date);
        
        -- Full-text search index for limitup reasons
        CREATE INDEX idx_limitup_reason_fulltext ON limitup_reason USING gin(to_tsvector('english', reason || ' ' || COALESCE(detail_reason, '')));
        
        -- Performance indexes for data quality monitoring
        CREATE INDEX idx_data_quality_source_date ON data_quality_log(data_source, check_date);
        CREATE INDEX idx_data_source_health_name_time ON data_source_health(source_name, check_time);
    """)


def downgrade():
    """Drop tables created for crawling integration."""
    
    # Drop partitioned tables first
    op.execute("DROP TABLE IF EXISTS dragon_tiger_board_partitioned CASCADE;")
    
    # Drop main tables
    op.drop_table('data_source_health')
    op.drop_table('data_quality_log')
    op.drop_table('etf_constituent')
    op.drop_table('etf_data')
    op.drop_table('limitup_reason')
    op.drop_table('fund_flow')
    op.drop_table('dragon_tiger_detail')
    op.drop_table('dragon_tiger_board')