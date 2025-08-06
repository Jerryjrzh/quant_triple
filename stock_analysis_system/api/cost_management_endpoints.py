"""
Cost Management API Endpoints

This module provides REST API endpoints for cost optimization,
auto-scaling, and resource optimization dashboard functionality.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..infrastructure.cost_optimization_manager import (
    CostOptimizationManager, CostAlert, ResourceType, CostCategory
)
from ..infrastructure.intelligent_autoscaling import (
    IntelligentAutoScaling, AutoScalingConfig, SpotInstanceConfig, ScalingAction
)
from ..infrastructure.resource_optimization_dashboard import (
    ResourceOptimizationDashboard, DashboardView, TimeRange, BudgetPlan, DashboardConfig
)

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/cost-management", tags=["cost-management"])

# Global instances (in production, these would be dependency injected)
cost_manager = None
autoscaling = None
dashboard = None

# Request/Response Models
class CostAlertRequest(BaseModel):
    """Request model for creating cost alerts"""
    alert_id: str
    name: str
    threshold_type: str = Field(..., description="budget, spike, trend")
    threshold_value: float
    period: str = Field(..., description="daily, weekly, monthly")
    enabled: bool = True
    notification_channels: List[str] = []

class AutoScalingConfigRequest(BaseModel):
    """Request model for auto-scaling configuration"""
    min_instances: int = Field(default=1, ge=1)
    max_instances: int = Field(default=10, ge=1)
    target_cpu_utilization: float = Field(default=70.0, ge=10.0, le=90.0)
    target_memory_utilization: float = Field(default=80.0, ge=10.0, le=90.0)
    scale_up_threshold: float = Field(default=80.0, ge=50.0, le=95.0)
    scale_down_threshold: float = Field(default=30.0, ge=5.0, le=50.0)
    cooldown_period: int = Field(default=300, description="Seconds between scaling actions")

class BudgetPlanRequest(BaseModel):
    """Request model for budget plans"""
    name: str
    period: str = Field(..., description="daily, weekly, monthly, quarterly")
    budget_amount: float = Field(..., gt=0)
    start_date: datetime
    end_date: datetime
    categories: Dict[str, float] = Field(default_factory=dict)
    alerts_enabled: bool = True
    alert_thresholds: List[float] = Field(default=[0.8, 0.9, 1.0])

# Dependency functions
async def get_cost_manager() -> CostOptimizationManager:
    """Get cost optimization manager instance"""
    global cost_manager
    if cost_manager is None:
        cost_manager = CostOptimizationManager()
    return cost_manager

async def get_autoscaling() -> IntelligentAutoScaling:
    """Get intelligent auto-scaling instance"""
    global autoscaling
    if autoscaling is None:
        config = AutoScalingConfig()
        spot_config = SpotInstanceConfig()
        autoscaling = IntelligentAutoScaling(config, spot_config)
    return autoscaling

async def get_dashboard() -> ResourceOptimizationDashboard:
    """Get resource optimization dashboard instance"""
    global dashboard
    if dashboard is None:
        cost_mgr = await get_cost_manager()
        auto_scaling = await get_autoscaling()
        dashboard = ResourceOptimizationDashboard(cost_mgr, auto_scaling)
    return dashboard

# Cost Monitoring Endpoints

@router.get("/cost/metrics")
async def get_cost_metrics(
    cost_manager: CostOptimizationManager = Depends(get_cost_manager)
):
    """Get current cost metrics and analysis"""
    try:
        metrics = await cost_manager.calculate_cost_metrics()
        return {
            "status": "success",
            "data": {
                "total_cost": metrics.total_cost,
                "daily_cost": metrics.daily_cost,
                "monthly_cost": metrics.monthly_cost,
                "cost_by_category": metrics.cost_by_category,
                "cost_by_resource": metrics.cost_by_resource,
                "cost_trend": [(t.isoformat(), c) for t, c in metrics.cost_trend],
                "optimization_potential": metrics.optimization_potential,
                "recommendations": metrics.recommendations
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting cost metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cost/usage")
async def get_resource_usage(
    cost_manager: CostOptimizationManager = Depends(get_cost_manager)
):
    """Get current resource usage metrics"""
    try:
        usage = await cost_manager.collect_resource_usage()
        return {
            "status": "success",
            "data": {
                "resource_id": usage.resource_id,
                "resource_type": usage.resource_type.value,
                "timestamp": usage.timestamp.isoformat(),
                "cpu_usage": usage.cpu_usage,
                "memory_usage": usage.memory_usage,
                "disk_usage": usage.disk_usage,
                "network_in": usage.network_in,
                "network_out": usage.network_out,
                "cost_per_hour": usage.cost_per_hour,
                "utilization_score": usage.utilization_score
            }
        }
    except Exception as e:
        logger.error(f"Error getting resource usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cost/alerts")
async def create_cost_alert(
    alert_request: CostAlertRequest,
    cost_manager: CostOptimizationManager = Depends(get_cost_manager)
):
    """Create a new cost alert"""
    try:
        alert = CostAlert(
            alert_id=alert_request.alert_id,
            name=alert_request.name,
            threshold_type=alert_request.threshold_type,
            threshold_value=alert_request.threshold_value,
            period=alert_request.period,
            enabled=alert_request.enabled,
            notification_channels=alert_request.notification_channels
        )
        
        success = await cost_manager.create_cost_alert(alert)
        
        if success:
            return {
                "status": "success",
                "message": f"Cost alert '{alert.name}' created successfully",
                "alert_id": alert.alert_id
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to create cost alert")
            
    except Exception as e:
        logger.error(f"Error creating cost alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cost/alerts")
async def get_cost_alerts(
    cost_manager: CostOptimizationManager = Depends(get_cost_manager)
):
    """Get triggered cost alerts"""
    try:
        alerts = await cost_manager.check_cost_alerts()
        return {
            "status": "success",
            "data": alerts,
            "count": len(alerts)
        }
    except Exception as e:
        logger.error(f"Error getting cost alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cost/optimization")
async def get_optimization_recommendations(
    cost_manager: CostOptimizationManager = Depends(get_cost_manager)
):
    """Get cost optimization recommendations"""
    try:
        recommendations = await cost_manager.get_optimization_recommendations()
        return {
            "status": "success",
            "data": recommendations
        }
    except Exception as e:
        logger.error(f"Error getting optimization recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Auto-scaling Endpoints

@router.get("/autoscaling/metrics")
async def get_scaling_metrics(
    autoscaling: IntelligentAutoScaling = Depends(get_autoscaling)
):
    """Get current scaling metrics"""
    try:
        metrics = await autoscaling.collect_scaling_metrics()
        return {
            "status": "success",
            "data": {
                "timestamp": metrics.timestamp.isoformat(),
                "cpu_utilization": metrics.cpu_utilization,
                "memory_utilization": metrics.memory_utilization,
                "network_utilization": metrics.network_utilization,
                "request_rate": metrics.request_rate,
                "response_time": metrics.response_time,
                "queue_length": metrics.queue_length,
                "cost_per_hour": metrics.cost_per_hour,
                "performance_score": metrics.performance_score
            }
        }
    except Exception as e:
        logger.error(f"Error getting scaling metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/autoscaling/config")
async def update_autoscaling_config(
    config_request: AutoScalingConfigRequest,
    autoscaling: IntelligentAutoScaling = Depends(get_autoscaling)
):
    """Update auto-scaling configuration"""
    try:
        new_config = AutoScalingConfig(
            min_instances=config_request.min_instances,
            max_instances=config_request.max_instances,
            target_cpu_utilization=config_request.target_cpu_utilization,
            target_memory_utilization=config_request.target_memory_utilization,
            scale_up_threshold=config_request.scale_up_threshold,
            scale_down_threshold=config_request.scale_down_threshold,
            cooldown_period=config_request.cooldown_period
        )
        
        autoscaling.config = new_config
        
        return {
            "status": "success",
            "message": "Auto-scaling configuration updated successfully",
            "config": {
                "min_instances": new_config.min_instances,
                "max_instances": new_config.max_instances,
                "target_cpu_utilization": new_config.target_cpu_utilization,
                "target_memory_utilization": new_config.target_memory_utilization,
                "scale_up_threshold": new_config.scale_up_threshold,
                "scale_down_threshold": new_config.scale_down_threshold,
                "cooldown_period": new_config.cooldown_period
            }
        }
    except Exception as e:
        logger.error(f"Error updating autoscaling config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/autoscaling/decision")
async def get_scaling_decision(
    autoscaling: IntelligentAutoScaling = Depends(get_autoscaling)
):
    """Get current scaling decision recommendation"""
    try:
        decision = await autoscaling.make_scaling_decision()
        
        if decision:
            return {
                "status": "success",
                "data": {
                    "action": decision.action.value,
                    "target_instances": decision.target_instances,
                    "target_instance_type": decision.target_instance_type.value,
                    "confidence": decision.confidence,
                    "rationale": decision.rationale,
                    "expected_cost_change": decision.expected_cost_change,
                    "expected_performance_change": decision.expected_performance_change,
                    "timestamp": decision.timestamp.isoformat()
                }
            }
        else:
            return {
                "status": "success",
                "data": None,
                "message": "No scaling decision needed (cooldown period active)"
            }
            
    except Exception as e:
        logger.error(f"Error getting scaling decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/autoscaling/execute")
async def execute_scaling_decision(
    background_tasks: BackgroundTasks,
    autoscaling: IntelligentAutoScaling = Depends(get_autoscaling)
):
    """Execute the current scaling decision"""
    try:
        decision = await autoscaling.make_scaling_decision()
        
        if not decision:
            return {
                "status": "info",
                "message": "No scaling decision to execute"
            }
        
        # Execute in background
        background_tasks.add_task(autoscaling.execute_scaling_decision, decision)
        
        return {
            "status": "success",
            "message": f"Scaling decision '{decision.action.value}' queued for execution",
            "decision": {
                "action": decision.action.value,
                "rationale": decision.rationale,
                "expected_cost_change": decision.expected_cost_change
            }
        }
        
    except Exception as e:
        logger.error(f"Error executing scaling decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/autoscaling/prediction")
async def get_load_prediction(
    minutes_ahead: int = Query(default=60, ge=1, le=1440),
    autoscaling: IntelligentAutoScaling = Depends(get_autoscaling)
):
    """Get load prediction for specified minutes ahead"""
    try:
        prediction = await autoscaling.predict_future_load(minutes_ahead)
        return {
            "status": "success",
            "data": prediction,
            "minutes_ahead": minutes_ahead
        }
    except Exception as e:
        logger.error(f"Error getting load prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/autoscaling/rightsizing")
async def get_rightsizing_recommendations(
    autoscaling: IntelligentAutoScaling = Depends(get_autoscaling)
):
    """Get resource right-sizing recommendations"""
    try:
        recommendations = await autoscaling.get_rightsizing_recommendations()
        return {
            "status": "success",
            "data": recommendations,
            "count": len(recommendations)
        }
    except Exception as e:
        logger.error(f"Error getting rightsizing recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dashboard Endpoints

@router.get("/dashboard/overview")
async def get_dashboard_overview(
    time_range: TimeRange = Query(default=TimeRange.LAST_DAY),
    dashboard: ResourceOptimizationDashboard = Depends(get_dashboard)
):
    """Get dashboard overview with key metrics and visualizations"""
    try:
        overview = await dashboard.generate_overview_dashboard(time_range)
        return {
            "status": "success",
            "data": overview
        }
    except Exception as e:
        logger.error(f"Error getting dashboard overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/forecast")
async def get_cost_forecast(
    days_ahead: int = Query(default=30, ge=1, le=365),
    dashboard: ResourceOptimizationDashboard = Depends(get_dashboard)
):
    """Get cost forecast for specified number of days"""
    try:
        forecasts = await dashboard.generate_cost_forecast(days_ahead)
        forecast_data = [
            {
                "forecast_date": f.forecast_date.isoformat(),
                "predicted_cost": f.predicted_cost,
                "confidence_interval": f.confidence_interval,
                "trend": f.trend,
                "factors": f.factors
            }
            for f in forecasts
        ]
        
        return {
            "status": "success",
            "data": forecast_data,
            "days_ahead": days_ahead,
            "count": len(forecast_data)
        }
    except Exception as e:
        logger.error(f"Error getting cost forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dashboard/budget")
async def create_budget_plan(
    budget_request: BudgetPlanRequest,
    dashboard: ResourceOptimizationDashboard = Depends(get_dashboard)
):
    """Create a new budget plan"""
    try:
        budget_plan = BudgetPlan(
            name=budget_request.name,
            period=budget_request.period,
            budget_amount=budget_request.budget_amount,
            start_date=budget_request.start_date,
            end_date=budget_request.end_date,
            categories=budget_request.categories,
            alerts_enabled=budget_request.alerts_enabled,
            alert_thresholds=budget_request.alert_thresholds
        )
        
        success = await dashboard.create_budget_plan(budget_plan)
        
        if success:
            return {
                "status": "success",
                "message": f"Budget plan '{budget_plan.name}' created successfully",
                "plan_name": budget_plan.name
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to create budget plan")
            
    except Exception as e:
        logger.error(f"Error creating budget plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/budget/{plan_name}")
async def get_budget_status(
    plan_name: str,
    dashboard: ResourceOptimizationDashboard = Depends(get_dashboard)
):
    """Get budget status for a specific plan"""
    try:
        status = await dashboard.check_budget_status(plan_name)
        
        if not status:
            raise HTTPException(status_code=404, detail=f"Budget plan '{plan_name}' not found")
        
        return {
            "status": "success",
            "data": status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting budget status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/optimization")
async def get_dashboard_optimization_recommendations(
    dashboard: ResourceOptimizationDashboard = Depends(get_dashboard)
):
    """Get comprehensive optimization recommendations from dashboard"""
    try:
        recommendations = await dashboard.generate_optimization_recommendations()
        return {
            "status": "success",
            "data": recommendations
        }
    except Exception as e:
        logger.error(f"Error getting dashboard optimization recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/export")
async def export_dashboard_data(
    view: DashboardView = Query(default=DashboardView.OVERVIEW),
    time_range: TimeRange = Query(default=TimeRange.LAST_DAY),
    format: str = Query(default="json", regex="^(json|csv)$"),
    dashboard: ResourceOptimizationDashboard = Depends(get_dashboard)
):
    """Export dashboard data in specified format"""
    try:
        exported_data = await dashboard.export_dashboard_data(view, time_range, format)
        
        if format.lower() == "json":
            return JSONResponse(
                content=exported_data,
                headers={"Content-Disposition": f"attachment; filename=dashboard_{view.value}_{time_range.value}.json"}
            )
        else:
            # For CSV or other formats
            return JSONResponse(
                content={"data": exported_data},
                headers={"Content-Disposition": f"attachment; filename=dashboard_{view.value}_{time_range.value}.{format}"}
            )
            
    except Exception as e:
        logger.error(f"Error exporting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility Endpoints

@router.get("/health")
async def health_check():
    """Health check endpoint for cost management services"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "cost_manager": "active" if cost_manager else "inactive",
                "autoscaling": "active" if autoscaling else "inactive",
                "dashboard": "active" if dashboard else "inactive"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/status")
async def get_system_status(
    cost_manager: CostOptimizationManager = Depends(get_cost_manager),
    autoscaling: IntelligentAutoScaling = Depends(get_autoscaling)
):
    """Get comprehensive system status"""
    try:
        # Get basic metrics
        cost_metrics = await cost_manager.calculate_cost_metrics()
        scaling_metrics = await autoscaling.collect_scaling_metrics()
        
        return {
            "status": "success",
            "data": {
                "cost_summary": {
                    "daily_cost": cost_metrics.daily_cost,
                    "monthly_projection": cost_metrics.monthly_cost,
                    "optimization_potential": cost_metrics.optimization_potential
                },
                "performance_summary": {
                    "cpu_utilization": scaling_metrics.cpu_utilization,
                    "memory_utilization": scaling_metrics.memory_utilization,
                    "performance_score": scaling_metrics.performance_score
                },
                "system_health": {
                    "cost_manager_active": True,
                    "autoscaling_active": True,
                    "model_trained": autoscaling.model_trained,
                    "last_update": datetime.now().isoformat()
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))