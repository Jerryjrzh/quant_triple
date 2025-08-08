"""Add ML Model Management Tables

Revision ID: c23456789012
Revises: b12345678901
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'c23456789012'
down_revision = 'b12345678901'
branch_labels = None
depends_on = None


def upgrade():
    """Add ML model management tables."""
    
    # Model monitoring registry table
    op.create_table(
        'model_monitoring_registry',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('model_name', sa.String(255), nullable=False),
        sa.Column('config', sa.JSON(), nullable=False),
        sa.Column('registered_at', sa.DateTime(), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='active'),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_id')
    )
    
    # Model drift detection results table
    op.create_table(
        'model_drift_detection_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('detection_timestamp', sa.DateTime(), nullable=False),
        sa.Column('data_drift_score', sa.Float(), nullable=False, default=0.0),
        sa.Column('concept_drift_score', sa.Float(), nullable=False, default=0.0),
        sa.Column('performance_drift_score', sa.Float(), nullable=False, default=0.0),
        sa.Column('overall_drift_score', sa.Float(), nullable=False, default=0.0),
        sa.Column('should_retrain', sa.Boolean(), nullable=False, default=False),
        sa.Column('recommendations', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('idx_drift_results_model_timestamp', 'model_id', 'detection_timestamp'),
        sa.Index('idx_drift_results_overall_score', 'overall_drift_score'),
        sa.Index('idx_drift_results_should_retrain', 'should_retrain')
    )
    
    # Model drift alerts table
    op.create_table(
        'model_drift_alerts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('drift_type', sa.String(50), nullable=False),
        sa.Column('severity', sa.String(50), nullable=False),
        sa.Column('drift_score', sa.Float(), nullable=False),
        sa.Column('threshold', sa.Float(), nullable=False),
        sa.Column('detected_at', sa.DateTime(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('recommendations', sa.JSON(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('acknowledged', sa.Boolean(), nullable=False, default=False),
        sa.Column('acknowledged_at', sa.DateTime(), nullable=True),
        sa.Column('acknowledged_by', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('idx_drift_alerts_model_severity', 'model_id', 'severity'),
        sa.Index('idx_drift_alerts_detected_at', 'detected_at'),
        sa.Index('idx_drift_alerts_acknowledged', 'acknowledged')
    )
    
    # Model retraining schedule table
    op.create_table(
        'model_retraining_schedule',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('schedule_config', sa.JSON(), nullable=False),
        sa.Column('next_retrain', sa.DateTime(), nullable=False),
        sa.Column('last_retrain', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, default='scheduled'),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_id'),
        sa.Index('idx_retraining_schedule_next_retrain', 'next_retrain'),
        sa.Index('idx_retraining_schedule_status', 'status')
    )
    
    # A/B testing experiments table
    op.create_table(
        'ab_test_experiments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('experiment_id', sa.String(255), nullable=False),
        sa.Column('experiment_name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('config', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='draft'),
        sa.Column('start_date', sa.DateTime(), nullable=True),
        sa.Column('end_date', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('experiment_id'),
        sa.Index('idx_ab_experiments_status', 'status'),
        sa.Index('idx_ab_experiments_dates', 'start_date', 'end_date')
    )
    
    # A/B testing metrics table
    op.create_table(
        'ab_test_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('experiment_id', sa.String(255), nullable=False),
        sa.Column('variant_id', sa.String(255), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('metrics', sa.JSON(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('idx_ab_metrics_experiment_variant', 'experiment_id', 'variant_id'),
        sa.Index('idx_ab_metrics_recorded_at', 'recorded_at'),
        sa.Index('idx_ab_metrics_user_id', 'user_id')
    )
    
    # A/B testing results table
    op.create_table(
        'ab_test_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('experiment_id', sa.String(255), nullable=False),
        sa.Column('variant_id', sa.String(255), nullable=False),
        sa.Column('sample_size', sa.Integer(), nullable=False),
        sa.Column('primary_metric_value', sa.Float(), nullable=False),
        sa.Column('primary_metric_std', sa.Float(), nullable=False),
        sa.Column('secondary_metrics', sa.JSON(), nullable=True),
        sa.Column('confidence_interval_lower', sa.Float(), nullable=False),
        sa.Column('confidence_interval_upper', sa.Float(), nullable=False),
        sa.Column('statistical_significance', sa.Boolean(), nullable=False, default=False),
        sa.Column('p_value', sa.Float(), nullable=False, default=1.0),
        sa.Column('effect_size', sa.Float(), nullable=False, default=0.0),
        sa.Column('conversion_rate', sa.Float(), nullable=True),
        sa.Column('analysis_timestamp', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('idx_ab_results_experiment_variant', 'experiment_id', 'variant_id'),
        sa.Index('idx_ab_results_significance', 'statistical_significance'),
        sa.Index('idx_ab_results_analysis_timestamp', 'analysis_timestamp')
    )
    
    # Model performance history table
    op.create_table(
        'model_performance_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('model_version', sa.String(100), nullable=False),
        sa.Column('evaluation_timestamp', sa.DateTime(), nullable=False),
        sa.Column('dataset_type', sa.String(50), nullable=False),  # train, validation, test, production
        sa.Column('sample_size', sa.Integer(), nullable=False),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('precision', sa.Float(), nullable=True),
        sa.Column('recall', sa.Float(), nullable=True),
        sa.Column('f1_score', sa.Float(), nullable=True),
        sa.Column('auc_roc', sa.Float(), nullable=True),
        sa.Column('auc_pr', sa.Float(), nullable=True),
        sa.Column('log_loss', sa.Float(), nullable=True),
        sa.Column('custom_metrics', sa.JSON(), nullable=True),
        sa.Column('feature_importance', sa.JSON(), nullable=True),
        sa.Column('confusion_matrix', sa.JSON(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('idx_performance_model_version', 'model_id', 'model_version'),
        sa.Index('idx_performance_evaluation_timestamp', 'evaluation_timestamp'),
        sa.Index('idx_performance_dataset_type', 'dataset_type')
    )
    
    # Model deployment history table
    op.create_table(
        'model_deployment_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('model_version', sa.String(100), nullable=False),
        sa.Column('deployment_id', sa.String(255), nullable=False),
        sa.Column('environment', sa.String(50), nullable=False),  # staging, production, canary
        sa.Column('deployment_type', sa.String(50), nullable=False),  # blue_green, canary, rolling
        sa.Column('status', sa.String(50), nullable=False),  # deploying, active, inactive, failed
        sa.Column('traffic_percentage', sa.Float(), nullable=False, default=0.0),
        sa.Column('deployed_at', sa.DateTime(), nullable=False),
        sa.Column('deactivated_at', sa.DateTime(), nullable=True),
        sa.Column('deployment_config', sa.JSON(), nullable=True),
        sa.Column('health_check_url', sa.String(500), nullable=True),
        sa.Column('metrics_endpoint', sa.String(500), nullable=True),
        sa.Column('rollback_version', sa.String(100), nullable=True),
        sa.Column('deployment_notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('idx_deployment_model_environment', 'model_id', 'environment'),
        sa.Index('idx_deployment_status', 'status'),
        sa.Index('idx_deployment_deployed_at', 'deployed_at')
    )
    
    # Model feature store table
    op.create_table(
        'model_feature_store',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('feature_group_id', sa.String(255), nullable=False),
        sa.Column('feature_name', sa.String(255), nullable=False),
        sa.Column('feature_type', sa.String(50), nullable=False),  # numerical, categorical, text, datetime
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('data_source', sa.String(255), nullable=False),
        sa.Column('transformation_logic', sa.Text(), nullable=True),
        sa.Column('validation_rules', sa.JSON(), nullable=True),
        sa.Column('feature_importance_score', sa.Float(), nullable=True),
        sa.Column('last_updated', sa.DateTime(), nullable=False),
        sa.Column('update_frequency', sa.String(50), nullable=False),  # real_time, hourly, daily, weekly
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('usage_count', sa.Integer(), nullable=False, default=0),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('feature_group_id', 'feature_name'),
        sa.Index('idx_feature_store_group', 'feature_group_id'),
        sa.Index('idx_feature_store_type', 'feature_type'),
        sa.Index('idx_feature_store_active', 'is_active')
    )
    
    # Model training jobs table
    op.create_table(
        'model_training_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_id', sa.String(255), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('job_type', sa.String(50), nullable=False),  # initial_training, retraining, hyperparameter_tuning
        sa.Column('status', sa.String(50), nullable=False),  # queued, running, completed, failed, cancelled
        sa.Column('priority', sa.Integer(), nullable=False, default=5),
        sa.Column('training_config', sa.JSON(), nullable=False),
        sa.Column('dataset_config', sa.JSON(), nullable=False),
        sa.Column('hyperparameters', sa.JSON(), nullable=True),
        sa.Column('resource_requirements', sa.JSON(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('final_metrics', sa.JSON(), nullable=True),
        sa.Column('model_artifacts_path', sa.String(500), nullable=True),
        sa.Column('logs_path', sa.String(500), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('triggered_by', sa.String(255), nullable=True),  # user_id or 'automated'
        sa.Column('parent_job_id', sa.String(255), nullable=True),  # for hyperparameter tuning jobs
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_id'),
        sa.Index('idx_training_jobs_model_status', 'model_id', 'status'),
        sa.Index('idx_training_jobs_priority', 'priority'),
        sa.Index('idx_training_jobs_created_at', 'created_at')
    )


def downgrade():
    """Drop ML model management tables."""
    op.drop_table('model_training_jobs')
    op.drop_table('model_feature_store')
    op.drop_table('model_deployment_history')
    op.drop_table('model_performance_history')
    op.drop_table('ab_test_results')
    op.drop_table('ab_test_metrics')
    op.drop_table('ab_test_experiments')
    op.drop_table('model_retraining_schedule')
    op.drop_table('model_drift_alerts')
    op.drop_table('model_drift_detection_results')
    op.drop_table('model_monitoring_registry')