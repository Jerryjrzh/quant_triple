"""
Infrastructure module for cost management and optimization

This module contains the infrastructure components for:
- Cost optimization and monitoring
- Intelligent auto-scaling
- Resource optimization dashboard
"""

from .cost_optimization_manager import CostOptimizationManager, CostAlert, ResourceUsage
from .intelligent_autoscaling import IntelligentAutoScaling, AutoScalingConfig, SpotInstanceConfig
from .resource_optimization_dashboard import ResourceOptimizationDashboard, BudgetPlan, DashboardConfig

__all__ = [
    'CostOptimizationManager',
    'CostAlert', 
    'ResourceUsage',
    'IntelligentAutoScaling',
    'AutoScalingConfig',
    'SpotInstanceConfig',
    'ResourceOptimizationDashboard',
    'BudgetPlan',
    'DashboardConfig'
]