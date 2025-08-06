"""
API endpoints for the Alert and Notification System
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..alerts.alert_engine import (
    AlertEngine, Alert, AlertTrigger, AlertCondition, 
    AlertPriority, AlertTriggerType, AlertStatus
)
from ..alerts.notification_system import (
    NotificationSystem, NotificationChannel, NotificationTemplate,
    NotificationPreference
)
from ..alerts.alert_filtering import SmartAlertFilter, AlertAggregator, MarketCondition

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["alerts"])

# Pydantic models for API
class AlertConditionModel(BaseModel):
    field: str
    operator: str
    value: Any
    timeframe: Optional[str] = None

class AlertTriggerModel(BaseModel):
    trigger_type: AlertTriggerType
    conditions: List[AlertConditionModel] = []
    logic_operator: str = "AND"
    schedule: Optional[str] = None
    ml_model_id: Optional[str] = None

class CreateAlertRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=1000)
    stock_code: Optional[str] = Field(None, max_length=10)
    trigger: AlertTriggerModel
    priority: AlertPriority
    user_id: Optional[str] = None

class UpdateAlertRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    priority: Optional[AlertPriority] = None
    status: Optional[AlertStatus] = None

class AlertResponse(BaseModel):
    id: str
    name: str
    description: str
    stock_code: Optional[str]
    priority: str
    status: str
    trigger_count: int
    created_at: datetime
    last_triggered: Optional[datetime]
    user_id: Optional[str]

class NotificationTemplateModel(BaseModel):
    name: str
    channel: NotificationChannel
    subject_template: str
    body_template: str
    priority: AlertPriority

class NotificationPreferenceModel(BaseModel):
    channel: NotificationChannel
    enabled: bool = True
    quiet_hours_start: Optional[str] = None
    quiet_hours_end: Optional[str] = None
    priority_filter: List[AlertPriority] = []
    frequency_limit: Optional[int] = None

class MarketConditionModel(BaseModel):
    volatility_level: str
    trend_direction: str
    volume_level: str
    market_hours: bool

# Global instances (in production, these would be dependency injected)
alert_engine: Optional[AlertEngine] = None
notification_system: Optional[NotificationSystem] = None
alert_filter: Optional[SmartAlertFilter] = None
alert_aggregator: Optional[AlertAggregator] = None

def get_alert_engine() -> AlertEngine:
    """Get alert engine instance"""
    global alert_engine
    if alert_engine is None:
        raise HTTPException(status_code=500, detail="Alert engine not initialized")
    return alert_engine

def get_notification_system() -> NotificationSystem:
    """Get notification system instance"""
    global notification_system
    if notification_system is None:
        raise HTTPException(status_code=500, detail="Notification system not initialized")
    return notification_system

def get_alert_filter() -> SmartAlertFilter:
    """Get alert filter instance"""
    global alert_filter
    if alert_filter is None:
        alert_filter = SmartAlertFilter()
    return alert_filter

def get_alert_aggregator() -> AlertAggregator:
    """Get alert aggregator instance"""
    global alert_aggregator
    if alert_aggregator is None:
        alert_aggregator = AlertAggregator()
    return alert_aggregator

# Alert Management Endpoints

@router.post("/", response_model=Dict[str, str])
async def create_alert(
    request: CreateAlertRequest,
    db: Session = Depends(get_db),
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Create a new alert"""
    try:
        # Convert request to Alert object
        trigger_conditions = [
            AlertCondition(
                field=cond.field,
                operator=cond.operator,
                value=cond.value,
                timeframe=cond.timeframe
            )
            for cond in request.trigger.conditions
        ]
        
        trigger = AlertTrigger(
            trigger_type=request.trigger.trigger_type,
            conditions=trigger_conditions,
            logic_operator=request.trigger.logic_operator,
            schedule=request.trigger.schedule,
            ml_model_id=request.trigger.ml_model_id
        )
        
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.user_id or 'system'}"
        
        alert = Alert(
            id=alert_id,
            name=request.name,
            description=request.description,
            stock_code=request.stock_code,
            trigger=trigger,
            priority=request.priority,
            user_id=request.user_id
        )
        
        created_id = await engine.create_alert(alert)
        
        return {"alert_id": created_id, "message": "Alert created successfully"}
        
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: str,
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Get an alert by ID"""
    try:
        alert = await engine.get_alert(alert_id)
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return AlertResponse(
            id=alert.id,
            name=alert.name,
            description=alert.description,
            stock_code=alert.stock_code,
            priority=alert.priority.value,
            status=alert.status.value,
            trigger_count=alert.trigger_count,
            created_at=alert.created_at,
            last_triggered=alert.last_triggered,
            user_id=alert.user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[AlertResponse])
async def list_alerts(
    user_id: Optional[str] = Query(None),
    status: Optional[AlertStatus] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    engine: AlertEngine = Depends(get_alert_engine)
):
    """List alerts with optional filtering"""
    try:
        alerts = await engine.list_alerts(user_id=user_id, status=status)
        
        # Apply limit
        alerts = alerts[:limit]
        
        return [
            AlertResponse(
                id=alert.id,
                name=alert.name,
                description=alert.description,
                stock_code=alert.stock_code,
                priority=alert.priority.value,
                status=alert.status.value,
                trigger_count=alert.trigger_count,
                created_at=alert.created_at,
                last_triggered=alert.last_triggered,
                user_id=alert.user_id
            )
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Error listing alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{alert_id}", response_model=Dict[str, str])
async def update_alert(
    alert_id: str,
    request: UpdateAlertRequest,
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Update an alert"""
    try:
        updates = {}
        
        if request.name is not None:
            updates['name'] = request.name
        if request.description is not None:
            updates['description'] = request.description
        if request.priority is not None:
            updates['priority'] = request.priority.value
        if request.status is not None:
            updates['status'] = request.status.value
        
        success = await engine.update_alert(alert_id, updates)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"message": "Alert updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{alert_id}", response_model=Dict[str, str])
async def delete_alert(
    alert_id: str,
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Delete an alert"""
    try:
        success = await engine.delete_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"message": "Alert deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{alert_id}/acknowledge", response_model=Dict[str, str])
async def acknowledge_alert(
    alert_id: str,
    user_id: str = Query(...),
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Acknowledge a triggered alert"""
    try:
        success = await engine.acknowledge_alert(alert_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found or not triggered")
        
        return {"message": "Alert acknowledged successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{alert_id}/resolve", response_model=Dict[str, str])
async def resolve_alert(
    alert_id: str,
    user_id: str = Query(...),
    resolution_note: str = Query(""),
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Resolve a triggered alert"""
    try:
        success = await engine.resolve_alert(alert_id, user_id, resolution_note)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found or not triggered")
        
        return {"message": "Alert resolved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring and Control Endpoints

@router.post("/monitoring/start", response_model=Dict[str, str])
async def start_monitoring(
    check_interval: int = Query(60, ge=10, le=3600),
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Start alert monitoring"""
    try:
        await engine.start_monitoring(check_interval)
        return {"message": f"Alert monitoring started with {check_interval}s interval"}
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitoring/stop", response_model=Dict[str, str])
async def stop_monitoring(
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Stop alert monitoring"""
    try:
        await engine.stop_monitoring()
        return {"message": "Alert monitoring stopped"}
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[Dict[str, Any]])
async def get_alert_history(
    limit: int = Query(100, ge=1, le=500),
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Get alert trigger history"""
    try:
        history = await engine.get_alert_history(limit)
        return history
        
    except Exception as e:
        logger.error(f"Error getting alert history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=Dict[str, Any])
async def get_performance_metrics(
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Get alert performance metrics"""
    try:
        metrics = await engine.get_performance_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Notification Endpoints

@router.post("/notifications/templates", response_model=Dict[str, str])
async def create_notification_template(
    request: NotificationTemplateModel,
    system: NotificationSystem = Depends(get_notification_system)
):
    """Create a notification template"""
    try:
        template_id = f"template_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        template = NotificationTemplate(
            id=template_id,
            name=request.name,
            channel=request.channel,
            subject_template=request.subject_template,
            body_template=request.body_template,
            priority=request.priority
        )
        
        created_id = await system.create_template(template)
        
        return {"template_id": created_id, "message": "Template created successfully"}
        
    except Exception as e:
        logger.error(f"Error creating notification template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/notifications/preferences/{user_id}", response_model=Dict[str, str])
async def set_notification_preferences(
    user_id: str,
    preferences: List[NotificationPreferenceModel],
    system: NotificationSystem = Depends(get_notification_system)
):
    """Set notification preferences for a user"""
    try:
        pref_objects = [
            NotificationPreference(
                user_id=user_id,
                channel=pref.channel,
                enabled=pref.enabled,
                quiet_hours_start=pref.quiet_hours_start,
                quiet_hours_end=pref.quiet_hours_end,
                priority_filter=pref.priority_filter,
                frequency_limit=pref.frequency_limit
            )
            for pref in preferences
        ]
        
        await system.set_user_preferences(user_id, pref_objects)
        
        return {"message": "Notification preferences updated successfully"}
        
    except Exception as e:
        logger.error(f"Error setting notification preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/notifications/in-app/{user_id}", response_model=List[Dict[str, Any]])
async def get_in_app_notifications(
    user_id: str,
    unread_only: bool = Query(False),
    system: NotificationSystem = Depends(get_notification_system)
):
    """Get in-app notifications for a user"""
    try:
        notifications = await system.get_in_app_notifications(user_id, unread_only)
        return notifications
        
    except Exception as e:
        logger.error(f"Error getting in-app notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/notifications/in-app/{user_id}/{notification_id}/read", response_model=Dict[str, str])
async def mark_notification_read(
    user_id: str,
    notification_id: str,
    system: NotificationSystem = Depends(get_notification_system)
):
    """Mark an in-app notification as read"""
    try:
        success = await system.mark_notification_read(user_id, notification_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return {"message": "Notification marked as read"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification as read: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/notifications/analytics", response_model=Dict[str, Any])
async def get_notification_analytics(
    system: NotificationSystem = Depends(get_notification_system)
):
    """Get notification delivery analytics"""
    try:
        analytics = await system.get_delivery_analytics()
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting notification analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alert Filtering Endpoints

@router.post("/filtering/market-conditions", response_model=Dict[str, str])
async def update_market_conditions(
    conditions: MarketConditionModel,
    filter_system: SmartAlertFilter = Depends(get_alert_filter)
):
    """Update market conditions for adaptive filtering"""
    try:
        market_condition = MarketCondition(
            volatility_level=conditions.volatility_level,
            trend_direction=conditions.trend_direction,
            volume_level=conditions.volume_level,
            market_hours=conditions.market_hours
        )
        
        await filter_system.update_market_conditions(market_condition)
        
        return {"message": "Market conditions updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating market conditions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/filtering/statistics", response_model=Dict[str, Any])
async def get_filtering_statistics(
    filter_system: SmartAlertFilter = Depends(get_alert_filter)
):
    """Get alert filtering statistics"""
    try:
        stats = await filter_system.get_filter_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting filtering statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/aggregation/clusters", response_model=List[Dict[str, Any]])
async def list_alert_clusters(
    limit: int = Query(50, ge=1, le=200),
    aggregator: AlertAggregator = Depends(get_alert_aggregator)
):
    """List alert clusters"""
    try:
        clusters = await aggregator.list_clusters(limit)
        
        return [
            cluster.get_summary() for cluster in clusters
        ]
        
    except Exception as e:
        logger.error(f"Error listing alert clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/aggregation/clusters/{cluster_id}", response_model=Dict[str, Any])
async def get_alert_cluster(
    cluster_id: str,
    aggregator: AlertAggregator = Depends(get_alert_aggregator)
):
    """Get details of a specific alert cluster"""
    try:
        summary = await aggregator.get_cluster_summary(cluster_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="Cluster not found")
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alert cluster: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/aggregation/statistics", response_model=Dict[str, Any])
async def get_aggregation_statistics(
    aggregator: AlertAggregator = Depends(get_alert_aggregator)
):
    """Get alert aggregation statistics"""
    try:
        stats = await aggregator.get_aggregation_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting aggregation statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))