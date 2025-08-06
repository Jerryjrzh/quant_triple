#!/usr/bin/env python3
"""
Alert and Notification System Demo

This script demonstrates the comprehensive alert and notification system
including alert creation, filtering, aggregation, and multi-channel notifications.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock
from sqlalchemy.orm import Session

from stock_analysis_system.alerts.alert_engine import (
    AlertEngine, Alert, AlertTrigger, AlertCondition,
    AlertPriority, AlertTriggerType, AlertStatus
)
from stock_analysis_system.alerts.notification_system import (
    NotificationSystem, NotificationChannel, NotificationTemplate,
    NotificationPreference
)
from stock_analysis_system.alerts.alert_filtering import (
    SmartAlertFilter, AlertAggregator, MarketCondition, FilterAction
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSystemDemo:
    """Comprehensive demo of the alert and notification system"""
    
    def __init__(self):
        self.setup_mock_dependencies()
        self.setup_system_components()
    
    def setup_mock_dependencies(self):
        """Set up mock dependencies for demo"""
        # Mock database session
        self.db_session = Mock(spec=Session)
        self.db_session.add = Mock()
        self.db_session.commit = Mock()
        self.db_session.rollback = Mock()
        self.db_session.query = Mock()
        
        # Mock analysis engines
        self.spring_festival_engine = Mock()
        self.spring_festival_engine.analyze_stock = Mock(return_value={
            'days_to_spring_festival': 30,
            'pattern_strength': 0.75,
            'seasonal_score': 0.8
        })
        
        self.risk_engine = Mock()
        self.risk_engine.calculate_comprehensive_risk = Mock(return_value=Mock(
            var_1d=0.02,
            volatility=0.25,
            beta=1.2,
            overall_risk_score=0.6
        ))
        
        self.institutional_engine = Mock()
        self.institutional_engine.calculate_attention_score = Mock(return_value=Mock(
            overall_score=75,
            recent_activity_score=80,
            fund_activity_score=70
        ))
    
    def setup_system_components(self):
        """Initialize system components"""
        self.alert_engine = AlertEngine(
            db_session=self.db_session,
            spring_festival_engine=self.spring_festival_engine,
            risk_engine=self.risk_engine,
            institutional_engine=self.institutional_engine
        )
        
        self.notification_system = NotificationSystem(
            db_session=self.db_session
        )
        
        self.alert_filter = SmartAlertFilter()
        self.alert_aggregator = AlertAggregator()
        
        logger.info("Alert system components initialized")
    
    async def demo_basic_alert_operations(self):
        """Demonstrate basic alert CRUD operations"""
        logger.info("\n=== Demo: Basic Alert Operations ===")
        
        # Create a price alert
        price_condition = AlertCondition(
            field="close_price",
            operator=">",
            value=150.0
        )
        
        price_trigger = AlertTrigger(
            trigger_type=AlertTriggerType.CONDITION_BASED,
            conditions=[price_condition]
        )
        
        price_alert = Alert(
            id="demo_price_alert_001",
            name="High Price Alert",
            description="Alert when stock price exceeds 150",
            stock_code="000001",
            trigger=price_trigger,
            priority=AlertPriority.HIGH,
            user_id="demo_user"
        )
        
        # Create alert
        alert_id = await self.alert_engine.create_alert(price_alert)
        logger.info(f"Created price alert: {alert_id}")
        
        # Create a volume alert
        volume_condition = AlertCondition(
            field="volume",
            operator=">",
            value=10000000
        )
        
        volume_trigger = AlertTrigger(
            trigger_type=AlertTriggerType.CONDITION_BASED,
            conditions=[volume_condition]
        )
        
        volume_alert = Alert(
            id="demo_volume_alert_001",
            name="High Volume Alert",
            description="Alert when trading volume exceeds 10M",
            stock_code="000001",
            trigger=volume_trigger,
            priority=AlertPriority.MEDIUM,
            user_id="demo_user"
        )
        
        await self.alert_engine.create_alert(volume_alert)
        logger.info(f"Created volume alert: {volume_alert.id}")
        
        # List all alerts
        all_alerts = await self.alert_engine.list_alerts()
        logger.info(f"Total alerts created: {len(all_alerts)}")
        
        # Update alert
        updates = {"priority": AlertPriority.CRITICAL.value}
        success = await self.alert_engine.update_alert(price_alert.id, updates)
        logger.info(f"Updated alert priority: {success}")
        
        # Get updated alert
        updated_alert = await self.alert_engine.get_alert(price_alert.id)
        logger.info(f"Updated alert priority: {updated_alert.priority}")
        
        return [price_alert, volume_alert]
    
    async def demo_advanced_triggers(self):
        """Demonstrate advanced trigger types"""
        logger.info("\n=== Demo: Advanced Trigger Types ===")
        
        # Seasonal trigger
        seasonal_trigger = AlertTrigger(
            trigger_type=AlertTriggerType.SEASONAL,
            conditions=[
                AlertCondition(field="days_to_spring_festival", operator="<=", value=30),
                AlertCondition(field="seasonal_score", operator=">=", value=0.7)
            ],
            logic_operator="AND"
        )
        
        seasonal_alert = Alert(
            id="demo_seasonal_alert_001",
            name="Spring Festival Opportunity",
            description="Alert for seasonal trading opportunity",
            stock_code="000001",
            trigger=seasonal_trigger,
            priority=AlertPriority.HIGH,
            user_id="demo_user"
        )
        
        await self.alert_engine.create_alert(seasonal_alert)
        logger.info("Created seasonal alert")
        
        # Institutional trigger
        institutional_trigger = AlertTrigger(
            trigger_type=AlertTriggerType.INSTITUTIONAL,
            conditions=[
                AlertCondition(field="institutional_attention_score", operator=">=", value=80)
            ]
        )
        
        institutional_alert = Alert(
            id="demo_institutional_alert_001",
            name="High Institutional Interest",
            description="Alert for high institutional attention",
            stock_code="000002",
            trigger=institutional_trigger,
            priority=AlertPriority.HIGH,
            user_id="demo_user"
        )
        
        await self.alert_engine.create_alert(institutional_alert)
        logger.info("Created institutional alert")
        
        # Risk-based trigger
        risk_trigger = AlertTrigger(
            trigger_type=AlertTriggerType.RISK,
            conditions=[
                AlertCondition(field="var_1d", operator=">", value=0.03),
                AlertCondition(field="volatility", operator=">", value=0.3)
            ],
            logic_operator="OR"
        )
        
        risk_alert = Alert(
            id="demo_risk_alert_001",
            name="High Risk Warning",
            description="Alert for elevated risk levels",
            stock_code="000003",
            trigger=risk_trigger,
            priority=AlertPriority.CRITICAL,
            user_id="demo_user"
        )
        
        await self.alert_engine.create_alert(risk_alert)
        logger.info("Created risk alert")
        
        return [seasonal_alert, institutional_alert, risk_alert]
    
    async def demo_notification_system(self):
        """Demonstrate notification system capabilities"""
        logger.info("\n=== Demo: Notification System ===")
        
        # Create notification templates
        email_template = NotificationTemplate(
            id="demo_email_template",
            name="Email Alert Template",
            channel=NotificationChannel.EMAIL,
            subject_template="🚨 Stock Alert: {{ alert_name }}",
            body_template="""
            <h2>Stock Alert Triggered</h2>
            <p><strong>Alert:</strong> {{ alert_name }}</p>
            <p><strong>Stock:</strong> {{ stock_code }}</p>
            <p><strong>Description:</strong> {{ alert_description }}</p>
            <p><strong>Priority:</strong> {{ priority }}</p>
            <p><strong>Time:</strong> {{ triggered_at }}</p>
            """,
            priority=AlertPriority.HIGH
        )
        
        await self.notification_system.create_template(email_template)
        logger.info("Created email notification template")
        
        sms_template = NotificationTemplate(
            id="demo_sms_template",
            name="SMS Alert Template",
            channel=NotificationChannel.SMS,
            subject_template="Alert: {{ alert_name }}",
            body_template="🚨 {{ alert_name }} for {{ stock_code }}. Priority: {{ priority }}",
            priority=AlertPriority.CRITICAL
        )
        
        await self.notification_system.create_template(sms_template)
        logger.info("Created SMS notification template")
        
        # Set up user notification preferences
        user_preferences = [
            NotificationPreference(
                user_id="demo_user",
                channel=NotificationChannel.EMAIL,
                enabled=True,
                priority_filter=[AlertPriority.HIGH, AlertPriority.CRITICAL]
            ),
            NotificationPreference(
                user_id="demo_user",
                channel=NotificationChannel.SMS,
                enabled=True,
                priority_filter=[AlertPriority.CRITICAL]
            ),
            NotificationPreference(
                user_id="demo_user",
                channel=NotificationChannel.IN_APP,
                enabled=True
            )
        ]
        
        await self.notification_system.set_user_preferences("demo_user", user_preferences)
        logger.info("Set up user notification preferences")
        
        # Create and trigger an alert for notification demo
        alert_condition = AlertCondition(field="close_price", operator=">", value=200.0)
        alert_trigger = AlertTrigger(
            trigger_type=AlertTriggerType.CONDITION_BASED,
            conditions=[alert_condition]
        )
        
        demo_alert = Alert(
            id="demo_notification_alert",
            name="Critical Price Alert",
            description="Stock price exceeded critical threshold",
            stock_code="000001",
            trigger=alert_trigger,
            priority=AlertPriority.CRITICAL,
            user_id="demo_user"
        )
        
        # Simulate alert triggering
        demo_alert.trigger_alert({"close_price": 205.0})
        
        # Send notifications
        notification_ids = await self.notification_system.send_alert_notification(
            demo_alert, "demo_user"
        )
        
        logger.info(f"Sent {len(notification_ids)} notifications")
        
        # Check in-app notifications
        in_app_notifications = await self.notification_system.get_in_app_notifications("demo_user")
        logger.info(f"In-app notifications: {len(in_app_notifications)}")
        
        if in_app_notifications:
            logger.info(f"Latest notification: {in_app_notifications[0]['subject']}")
        
        return demo_alert
    
    async def demo_smart_filtering(self):
        """Demonstrate smart alert filtering"""
        logger.info("\n=== Demo: Smart Alert Filtering ===")
        
        # Create similar alerts to test deduplication
        alerts = []
        for i in range(3):
            condition = AlertCondition(field="close_price", operator=">", value=100.0)
            trigger = AlertTrigger(
                trigger_type=AlertTriggerType.CONDITION_BASED,
                conditions=[condition]
            )
            
            alert = Alert(
                id=f"demo_filter_alert_{i:03d}",
                name="Duplicate Price Alert",  # Same name for deduplication test
                description="Test alert for filtering",
                stock_code="000001",  # Same stock
                trigger=trigger,
                priority=AlertPriority.MEDIUM,
                user_id="demo_user"
            )
            alerts.append(alert)
        
        # Filter alerts
        results = []
        for alert in alerts:
            action, reason = await self.alert_filter.filter_alert(alert)
            results.append((action, reason))
            logger.info(f"Alert {alert.id}: {action.value} - {reason}")
        
        # First alert should be allowed, subsequent ones suppressed
        assert results[0][0] == FilterAction.ALLOW
        assert results[1][0] == FilterAction.SUPPRESS
        assert results[2][0] == FilterAction.SUPPRESS
        
        # Update market conditions
        market_conditions = MarketCondition(
            volatility_level="extreme",
            trend_direction="down",
            volume_level="high",
            market_hours=True
        )
        
        await self.alert_filter.update_market_conditions(market_conditions)
        logger.info("Updated market conditions to extreme volatility")
        
        # Get filtering statistics
        stats = await self.alert_filter.get_filter_statistics()
        logger.info(f"Filtering statistics: {stats}")
        
        return alerts
    
    async def demo_alert_aggregation(self):
        """Demonstrate alert aggregation and clustering"""
        logger.info("\n=== Demo: Alert Aggregation ===")
        
        # Create similar alerts for clustering
        similar_alerts = []
        for i in range(5):
            condition = AlertCondition(field="close_price", operator=">", value=100.0 + i * 5)
            trigger = AlertTrigger(
                trigger_type=AlertTriggerType.CONDITION_BASED,
                conditions=[condition]
            )
            
            alert = Alert(
                id=f"demo_cluster_alert_{i:03d}",
                name=f"Price Threshold Alert {i}",
                description="Alert when price exceeds threshold level",
                stock_code="000001",
                trigger=trigger,
                priority=AlertPriority.MEDIUM,
                user_id="demo_user"
            )
            similar_alerts.append(alert)
        
        # Add alerts for aggregation
        cluster_id = None
        for alert in similar_alerts:
            result = await self.alert_aggregator.add_alert_for_aggregation(alert)
            if result:
                cluster_id = result
                logger.info(f"Created cluster: {cluster_id}")
        
        # Force aggregation if needed
        if not cluster_id:
            cluster_ids = await self.alert_aggregator.force_aggregation()
            if cluster_ids:
                cluster_id = cluster_ids[0]
                logger.info(f"Forced aggregation created cluster: {cluster_id}")
        
        # Get cluster details
        if cluster_id:
            cluster_summary = await self.alert_aggregator.get_cluster_summary(cluster_id)
            logger.info(f"Cluster summary: {cluster_summary}")
        
        # Get aggregation statistics
        agg_stats = await self.alert_aggregator.get_aggregation_statistics()
        logger.info(f"Aggregation statistics: {agg_stats}")
        
        return similar_alerts, cluster_id
    
    async def demo_monitoring_simulation(self):
        """Demonstrate alert monitoring simulation"""
        logger.info("\n=== Demo: Alert Monitoring Simulation ===")
        
        # Create alerts with different trigger conditions
        alerts_to_monitor = []
        
        # Price alert that should trigger
        price_alert = Alert(
            id="demo_monitor_price",
            name="Monitor Price Alert",
            description="Price monitoring test",
            stock_code="000001",
            trigger=AlertTrigger(
                trigger_type=AlertTriggerType.CONDITION_BASED,
                conditions=[AlertCondition(field="close_price", operator=">", value=100.0)]
            ),
            priority=AlertPriority.HIGH,
            user_id="demo_user"
        )
        
        await self.alert_engine.create_alert(price_alert)
        alerts_to_monitor.append(price_alert)
        
        # Seasonal alert
        seasonal_alert = Alert(
            id="demo_monitor_seasonal",
            name="Monitor Seasonal Alert",
            description="Seasonal monitoring test",
            stock_code="000002",
            trigger=AlertTrigger(
                trigger_type=AlertTriggerType.SEASONAL,
                conditions=[AlertCondition(field="days_to_spring_festival", operator="<=", value=35)]
            ),
            priority=AlertPriority.MEDIUM,
            user_id="demo_user"
        )
        
        await self.alert_engine.create_alert(seasonal_alert)
        alerts_to_monitor.append(seasonal_alert)
        
        # Simulate monitoring check
        logger.info("Simulating alert monitoring check...")
        
        # Mock stock data that would trigger the price alert
        mock_stock_data = Mock()
        mock_stock_data.stock_code = "000001"
        mock_stock_data.close_price = 105.0  # Above threshold
        mock_stock_data.volume = 1000000
        mock_stock_data.change_pct = 5.0
        mock_stock_data.trade_date = datetime.now().date()
        
        # Mock database query to return this data
        self.db_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_stock_data
        
        # Check alerts manually (simulating monitoring loop)
        for alert in alerts_to_monitor:
            try:
                await self.alert_engine._check_single_alert(alert)
                logger.info(f"Checked alert: {alert.name}")
            except Exception as e:
                logger.error(f"Error checking alert {alert.name}: {e}")
        
        # Get alert history
        history = await self.alert_engine.get_alert_history()
        logger.info(f"Alert history entries: {len(history)}")
        
        # Get performance metrics
        metrics = await self.alert_engine.get_performance_metrics()
        logger.info(f"Performance metrics: {metrics}")
        
        return alerts_to_monitor
    
    async def demo_complete_workflow(self):
        """Demonstrate complete alert workflow"""
        logger.info("\n=== Demo: Complete Alert Workflow ===")
        
        # 1. Create alert
        workflow_alert = Alert(
            id="demo_workflow_alert",
            name="Complete Workflow Alert",
            description="End-to-end workflow demonstration",
            stock_code="000001",
            trigger=AlertTrigger(
                trigger_type=AlertTriggerType.CONDITION_BASED,
                conditions=[AlertCondition(field="close_price", operator=">", value=120.0)]
            ),
            priority=AlertPriority.HIGH,
            user_id="demo_user"
        )
        
        await self.alert_engine.create_alert(workflow_alert)
        logger.info("1. Created workflow alert")
        
        # 2. Filter alert
        action, reason = await self.alert_filter.filter_alert(workflow_alert)
        logger.info(f"2. Filtered alert: {action.value} - {reason}")
        
        if action == FilterAction.ALLOW:
            # 3. Simulate alert triggering
            workflow_alert.trigger_alert({"close_price": 125.0})
            logger.info("3. Alert triggered")
            
            # 4. Send notifications
            notification_ids = await self.notification_system.send_alert_notification(
                workflow_alert, "demo_user"
            )
            logger.info(f"4. Sent {len(notification_ids)} notifications")
            
            # 5. User acknowledges alert
            success = await self.alert_engine.acknowledge_alert(workflow_alert.id, "demo_user")
            logger.info(f"5. Alert acknowledged: {success}")
            
            # 6. User resolves alert
            success = await self.alert_engine.resolve_alert(
                workflow_alert.id, "demo_user", "Price target reached, position adjusted"
            )
            logger.info(f"6. Alert resolved: {success}")
        
        return workflow_alert
    
    async def run_complete_demo(self):
        """Run the complete alert system demonstration"""
        logger.info("🚀 Starting Alert and Notification System Demo")
        logger.info("=" * 60)
        
        try:
            # Basic operations
            basic_alerts = await self.demo_basic_alert_operations()
            
            # Advanced triggers
            advanced_alerts = await self.demo_advanced_triggers()
            
            # Notification system
            notification_alert = await self.demo_notification_system()
            
            # Smart filtering
            filtered_alerts = await self.demo_smart_filtering()
            
            # Alert aggregation
            clustered_alerts, cluster_id = await self.demo_alert_aggregation()
            
            # Monitoring simulation
            monitored_alerts = await self.demo_monitoring_simulation()
            
            # Complete workflow
            workflow_alert = await self.demo_complete_workflow()
            
            # Final summary
            logger.info("\n" + "=" * 60)
            logger.info("📊 Demo Summary")
            logger.info("=" * 60)
            
            all_alerts = await self.alert_engine.list_alerts()
            logger.info(f"Total alerts created: {len(all_alerts)}")
            
            performance_metrics = await self.alert_engine.get_performance_metrics()
            logger.info(f"Performance metrics: {performance_metrics}")
            
            notification_analytics = await self.notification_system.get_delivery_analytics()
            logger.info(f"Notification analytics: {notification_analytics}")
            
            filter_stats = await self.alert_filter.get_filter_statistics()
            logger.info(f"Filter statistics: {filter_stats}")
            
            agg_stats = await self.alert_aggregator.get_aggregation_statistics()
            logger.info(f"Aggregation statistics: {agg_stats}")
            
            logger.info("\n✅ Alert and Notification System Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed with error: {e}")
            raise


async def main():
    """Main demo function"""
    demo = AlertSystemDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())