"""
Comprehensive Alert Engine

This module implements a sophisticated alert system with multiple trigger types,
priority management, and performance tracking capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..data.models import StockDailyData, Alert as AlertModel
from ..analysis.spring_festival_engine import SpringFestivalAlignmentEngine
from ..analysis.risk_management_engine import EnhancedRiskManagementEngine
from ..analysis.institutional_attention_scoring import InstitutionalAttentionScoringSystem

logger = logging.getLogger(__name__)


class AlertPriority(str, Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertTriggerType(str, Enum):
    """Types of alert triggers"""
    TIME_BASED = "time_based"
    CONDITION_BASED = "condition_based"
    ML_BASED = "ml_based"
    SEASONAL = "seasonal"
    INSTITUTIONAL = "institutional"
    RISK = "risk"
    TECHNICAL = "technical"


class AlertStatus(str, Enum):
    """Alert status"""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISABLED = "disabled"


@dataclass
class AlertCondition:
    """Defines a condition for triggering an alert"""
    field: str
    operator: str  # '>', '<', '>=', '<=', '==', '!=', 'in', 'not_in'
    value: Any
    timeframe: Optional[str] = None  # '1d', '5d', '1w', '1m'
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate the condition against provided data"""
        if self.field not in data:
            return False
        
        field_value = data[self.field]
        
        if self.operator == '>':
            return field_value > self.value
        elif self.operator == '<':
            return field_value < self.value
        elif self.operator == '>=':
            return field_value >= self.value
        elif self.operator == '<=':
            return field_value <= self.value
        elif self.operator == '==':
            return field_value == self.value
        elif self.operator == '!=':
            return field_value != self.value
        elif self.operator == 'in':
            return field_value in self.value
        elif self.operator == 'not_in':
            return field_value not in self.value
        else:
            return False


@dataclass
class AlertTrigger:
    """Defines when and how an alert should be triggered"""
    trigger_type: AlertTriggerType
    conditions: List[AlertCondition] = field(default_factory=list)
    logic_operator: str = "AND"  # "AND" or "OR"
    schedule: Optional[str] = None  # Cron-like schedule for time-based alerts
    ml_model_id: Optional[str] = None  # For ML-based alerts
    custom_function: Optional[Callable] = None  # Custom trigger function
    
    def should_trigger(self, data: Dict[str, Any]) -> bool:
        """Check if the trigger conditions are met"""
        if not self.conditions:
            return False
        
        if self.logic_operator == "AND":
            return all(condition.evaluate(data) for condition in self.conditions)
        else:  # OR
            return any(condition.evaluate(data) for condition in self.conditions)


@dataclass
class Alert:
    """Represents an alert in the system"""
    id: str
    name: str
    description: str
    stock_code: Optional[str]
    trigger: AlertTrigger
    priority: AlertPriority
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def trigger_alert(self, trigger_data: Dict[str, Any]) -> None:
        """Mark the alert as triggered"""
        self.status = AlertStatus.TRIGGERED
        self.last_triggered = datetime.now()
        self.trigger_count += 1
        self.metadata['last_trigger_data'] = trigger_data


class AlertEngine:
    """
    Comprehensive alert engine with multiple trigger types and performance tracking
    """
    
    def __init__(self, db_session: Session, spring_festival_engine: SpringFestivalAlignmentEngine,
                 risk_engine: EnhancedRiskManagementEngine, 
                 institutional_engine: InstitutionalAttentionScoringSystem):
        self.db_session = db_session
        self.spring_festival_engine = spring_festival_engine
        self.risk_engine = risk_engine
        self.institutional_engine = institutional_engine
        
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Background task for monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
        logger.info("AlertEngine initialized")
    
    async def create_alert(self, alert: Alert) -> str:
        """Create a new alert"""
        try:
            # Validate alert
            if not alert.name or not alert.trigger:
                raise ValueError("Alert must have name and trigger")
            
            # Store in active alerts
            self.active_alerts[alert.id] = alert
            
            # Persist to database
            alert_model = AlertModel(
                id=alert.id,
                name=alert.name,
                description=alert.description,
                stock_code=alert.stock_code,
                trigger_type=alert.trigger.trigger_type.value,
                priority=alert.priority.value,
                status=alert.status.value,
                user_id=alert.user_id,
                created_at=alert.created_at,
                alert_metadata=alert.metadata
            )
            
            self.db_session.add(alert_model)
            self.db_session.commit()
            
            logger.info(f"Created alert: {alert.name} ({alert.id})")
            return alert.id
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            self.db_session.rollback()
            raise
    
    async def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing alert"""
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            
            # Update alert properties
            for key, value in updates.items():
                if hasattr(alert, key):
                    setattr(alert, key, value)
            
            # Update in database
            alert_model = self.db_session.query(AlertModel).filter(
                AlertModel.id == alert_id
            ).first()
            
            if alert_model:
                for key, value in updates.items():
                    if hasattr(alert_model, key):
                        setattr(alert_model, key, value)
                
                self.db_session.commit()
                logger.info(f"Updated alert: {alert_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating alert {alert_id}: {e}")
            self.db_session.rollback()
            return False
    
    async def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert"""
        try:
            # Remove from active alerts
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
            
            # Remove from database
            alert_model = self.db_session.query(AlertModel).filter(
                AlertModel.id == alert_id
            ).first()
            
            if alert_model:
                self.db_session.delete(alert_model)
                self.db_session.commit()
                logger.info(f"Deleted alert: {alert_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting alert {alert_id}: {e}")
            self.db_session.rollback()
            return False
    
    async def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID"""
        return self.active_alerts.get(alert_id)
    
    async def list_alerts(self, user_id: Optional[str] = None, 
                         status: Optional[AlertStatus] = None) -> List[Alert]:
        """List alerts with optional filtering"""
        alerts = list(self.active_alerts.values())
        
        if user_id:
            alerts = [alert for alert in alerts if alert.user_id == user_id]
        
        if status:
            alerts = [alert for alert in alerts if alert.status == status]
        
        return alerts
    
    async def start_monitoring(self, check_interval: int = 60) -> None:
        """Start the alert monitoring background task"""
        if self._is_monitoring:
            logger.warning("Alert monitoring is already running")
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(check_interval)
        )
        logger.info(f"Started alert monitoring with {check_interval}s interval")
    
    async def stop_monitoring(self) -> None:
        """Stop the alert monitoring background task"""
        self._is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped alert monitoring")
    
    async def _monitoring_loop(self, check_interval: int) -> None:
        """Main monitoring loop"""
        while self._is_monitoring:
            try:
                await self._check_all_alerts()
                await asyncio.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(check_interval)
    
    async def _check_all_alerts(self) -> None:
        """Check all active alerts for trigger conditions"""
        active_alerts = [
            alert for alert in self.active_alerts.values() 
            if alert.status == AlertStatus.ACTIVE
        ]
        
        for alert in active_alerts:
            try:
                await self._check_single_alert(alert)
            except Exception as e:
                logger.error(f"Error checking alert {alert.id}: {e}")
    
    async def _check_single_alert(self, alert: Alert) -> None:
        """Check a single alert for trigger conditions"""
        try:
            # Get data for evaluation
            data = await self._get_alert_data(alert)
            
            # Check trigger conditions
            should_trigger = False
            
            if alert.trigger.trigger_type == AlertTriggerType.CONDITION_BASED:
                should_trigger = alert.trigger.should_trigger(data)
            
            elif alert.trigger.trigger_type == AlertTriggerType.SEASONAL:
                should_trigger = await self._check_seasonal_trigger(alert, data)
            
            elif alert.trigger.trigger_type == AlertTriggerType.INSTITUTIONAL:
                should_trigger = await self._check_institutional_trigger(alert, data)
            
            elif alert.trigger.trigger_type == AlertTriggerType.RISK:
                should_trigger = await self._check_risk_trigger(alert, data)
            
            elif alert.trigger.trigger_type == AlertTriggerType.ML_BASED:
                should_trigger = await self._check_ml_trigger(alert, data)
            
            elif alert.trigger.trigger_type == AlertTriggerType.TECHNICAL:
                should_trigger = await self._check_technical_trigger(alert, data)
            
            # Trigger alert if conditions are met
            if should_trigger:
                await self._trigger_alert(alert, data)
                
        except Exception as e:
            logger.error(f"Error checking alert {alert.id}: {e}")
    
    async def _get_alert_data(self, alert: Alert) -> Dict[str, Any]:
        """Get data needed for alert evaluation"""
        data = {}
        
        if alert.stock_code:
            # Get latest stock data
            latest_data = self.db_session.query(StockDailyData).filter(
                StockDailyData.stock_code == alert.stock_code
            ).order_by(StockDailyData.trade_date.desc()).first()
            
            if latest_data:
                data.update({
                    'stock_code': latest_data.stock_code,
                    'close_price': latest_data.close_price,
                    'volume': latest_data.volume,
                    'change_pct': latest_data.change_pct,
                    'trade_date': latest_data.trade_date
                })
        
        # Add current market data
        data.update({
            'current_time': datetime.now(),
            'market_open': self._is_market_open()
        })
        
        return data
    
    async def _check_seasonal_trigger(self, alert: Alert, data: Dict[str, Any]) -> bool:
        """Check seasonal-based triggers using Spring Festival analysis"""
        if not alert.stock_code:
            return False
        
        try:
            # Get Spring Festival analysis
            analysis = await self.spring_festival_engine.analyze_stock(alert.stock_code)
            
            if not analysis:
                return False
            
            # Check if we're in a historically significant period
            current_date = datetime.now().date()
            
            # Add seasonal data to evaluation context
            seasonal_data = {
                'days_to_spring_festival': analysis.get('days_to_spring_festival', 0),
                'historical_pattern_strength': analysis.get('pattern_strength', 0),
                'seasonal_score': analysis.get('seasonal_score', 0)
            }
            
            data.update(seasonal_data)
            return alert.trigger.should_trigger(data)
            
        except Exception as e:
            logger.error(f"Error in seasonal trigger check: {e}")
            return False
    
    async def _check_institutional_trigger(self, alert: Alert, data: Dict[str, Any]) -> bool:
        """Check institutional activity triggers"""
        if not alert.stock_code:
            return False
        
        try:
            # Get institutional attention score
            attention_score = await self.institutional_engine.calculate_attention_score(
                alert.stock_code
            )
            
            # Add institutional data to evaluation context
            institutional_data = {
                'institutional_attention_score': attention_score.overall_score,
                'recent_activity_score': attention_score.recent_activity_score,
                'fund_activity_score': attention_score.fund_activity_score
            }
            
            data.update(institutional_data)
            return alert.trigger.should_trigger(data)
            
        except Exception as e:
            logger.error(f"Error in institutional trigger check: {e}")
            return False
    
    async def _check_risk_trigger(self, alert: Alert, data: Dict[str, Any]) -> bool:
        """Check risk-based triggers"""
        if not alert.stock_code:
            return False
        
        try:
            # Get risk metrics
            risk_metrics = await self.risk_engine.calculate_comprehensive_risk(
                alert.stock_code
            )
            
            # Add risk data to evaluation context
            risk_data = {
                'var_1d': risk_metrics.var_1d,
                'volatility': risk_metrics.volatility,
                'beta': risk_metrics.beta,
                'risk_score': risk_metrics.overall_risk_score
            }
            
            data.update(risk_data)
            return alert.trigger.should_trigger(data)
            
        except Exception as e:
            logger.error(f"Error in risk trigger check: {e}")
            return False
    
    async def _check_ml_trigger(self, alert: Alert, data: Dict[str, Any]) -> bool:
        """Check ML-based triggers"""
        # Placeholder for ML-based trigger logic
        # This would integrate with the ML model manager
        return False
    
    async def _check_technical_trigger(self, alert: Alert, data: Dict[str, Any]) -> bool:
        """Check technical indicator triggers"""
        if not alert.stock_code:
            return False
        
        try:
            # Get recent price data for technical analysis
            recent_data = self.db_session.query(StockDailyData).filter(
                StockDailyData.stock_code == alert.stock_code
            ).order_by(StockDailyData.trade_date.desc()).limit(50).all()
            
            if len(recent_data) < 20:
                return False
            
            # Convert to DataFrame for technical analysis
            df = pd.DataFrame([{
                'close': d.close_price,
                'high': d.high_price,
                'low': d.low_price,
                'volume': d.volume,
                'date': d.trade_date
            } for d in reversed(recent_data)])
            
            # Calculate technical indicators
            df['sma_20'] = df['close'].rolling(20).mean()
            df['rsi'] = self._calculate_rsi(df['close'])
            
            # Add technical data to evaluation context
            latest = df.iloc[-1]
            technical_data = {
                'sma_20': latest['sma_20'],
                'rsi': latest['rsi'],
                'price_vs_sma20': (latest['close'] - latest['sma_20']) / latest['sma_20'] * 100
            }
            
            data.update(technical_data)
            return alert.trigger.should_trigger(data)
            
        except Exception as e:
            logger.error(f"Error in technical trigger check: {e}")
            return False
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def _trigger_alert(self, alert: Alert, trigger_data: Dict[str, Any]) -> None:
        """Trigger an alert and record the event"""
        try:
            # Update alert status
            alert.trigger_alert(trigger_data)
            
            # Record in history
            history_entry = {
                'alert_id': alert.id,
                'alert_name': alert.name,
                'stock_code': alert.stock_code,
                'triggered_at': alert.last_triggered,
                'trigger_data': trigger_data,
                'priority': alert.priority.value
            }
            
            self.alert_history.append(history_entry)
            
            # Update performance metrics
            self._update_performance_metrics(alert)
            
            # Update database
            await self.update_alert(alert.id, {
                'status': AlertStatus.TRIGGERED.value,
                'last_triggered': alert.last_triggered,
                'trigger_count': alert.trigger_count
            })
            
            logger.info(f"Alert triggered: {alert.name} ({alert.id})")
            
        except Exception as e:
            logger.error(f"Error triggering alert {alert.id}: {e}")
    
    def _update_performance_metrics(self, alert: Alert) -> None:
        """Update alert performance metrics"""
        if 'total_triggers' not in self.performance_metrics:
            self.performance_metrics['total_triggers'] = 0
        
        self.performance_metrics['total_triggers'] += 1
        
        # Track by priority
        priority_key = f'triggers_{alert.priority.value}'
        if priority_key not in self.performance_metrics:
            self.performance_metrics[priority_key] = 0
        self.performance_metrics[priority_key] += 1
        
        # Track by type
        type_key = f'triggers_{alert.trigger.trigger_type.value}'
        if type_key not in self.performance_metrics:
            self.performance_metrics[type_key] = 0
        self.performance_metrics[type_key] += 1
    
    def _is_market_open(self) -> bool:
        """Check if the market is currently open"""
        now = datetime.now()
        
        # Simple check for Chinese market hours (9:30-15:00, Mon-Fri)
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    async def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert trigger history"""
        return self.alert_history[-limit:]
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get alert performance metrics"""
        return self.performance_metrics.copy()
    
    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge a triggered alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        if alert.status != AlertStatus.TRIGGERED:
            return False
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.metadata['acknowledged_by'] = user_id
        alert.metadata['acknowledged_at'] = datetime.now().isoformat()
        
        await self.update_alert(alert_id, {
            'status': AlertStatus.ACKNOWLEDGED.value,
            'metadata': alert.metadata
        })
        
        logger.info(f"Alert acknowledged: {alert_id} by {user_id}")
        return True
    
    async def resolve_alert(self, alert_id: str, user_id: str, resolution_note: str = "") -> bool:
        """Resolve a triggered alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        if alert.status not in [AlertStatus.TRIGGERED, AlertStatus.ACKNOWLEDGED]:
            return False
        
        alert.status = AlertStatus.RESOLVED
        alert.metadata['resolved_by'] = user_id
        alert.metadata['resolved_at'] = datetime.now().isoformat()
        alert.metadata['resolution_note'] = resolution_note
        
        await self.update_alert(alert_id, {
            'status': AlertStatus.RESOLVED.value,
            'metadata': alert.metadata
        })
        
        logger.info(f"Alert resolved: {alert_id} by {user_id}")
        return True