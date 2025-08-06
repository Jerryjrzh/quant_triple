#!/usr/bin/env python3
"""
Simplified Alert and Notification System Demo

This script demonstrates the alert and notification system with mocked dependencies
to showcase the core functionality without requiring the full system integration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock
from sqlalchemy.orm import Session

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplifiedAlertDemo:
    """Simplified demo of the alert and notification system"""
    
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
        
        # Mock analysis engines with simple return values
        self.spring_festival_engine = Mock()
        self.risk_engine = Mock()
        self.institutional_engine = Mock()
    
    def setup_system_components(self):
        """Initialize system components with mocked dependencies"""
        # Import here to avoid circular imports
        from stock_analysis_system.alerts.alert_engine import (
            Alert, AlertTrigger, AlertCondition, AlertPriority, AlertTriggerType
        )
        from stock_analysis_system.alerts.notification_system import (
            NotificationSystem, NotificationChannel, NotificationTemplate,
            NotificationPreference
        )
        from stock_analysis_system.alerts.alert_filtering import (
            SmartAlertFilter, AlertAggregator, MarketCondition
        )
        
        # Store classes for later use
        self.Alert = Alert
        self.AlertTrigger = AlertTrigger
        self.AlertCondition = AlertCondition
        self.AlertPriority = AlertPriority
        self.AlertTriggerType = AlertTriggerType
        self.NotificationSystem = NotificationSystem
        self.NotificationChannel = NotificationChannel
        self.NotificationTemplate = NotificationTemplate
        self.NotificationPreference = NotificationPreference
        self.SmartAlertFilter = SmartAlertFilter
        self.AlertAggregator = AlertAggregator
        self.MarketCondition = MarketCondition
        
        # Initialize components
        self.notification_system = NotificationSystem(db_session=self.db_session)
        self.alert_filter = SmartAlertFilter()
        self.alert_aggregator = AlertAggregator()
        
        logger.info("Simplified alert system components initialized")
    
    async def demo_basic_alert_creation(self):
        """Demonstrate basic alert creation and management"""
        logger.info("\n=== Demo: Basic Alert Creation ===")
        
        # Create a simple price alert
        price_condition = self.AlertCondition(
            field="close_price",
            operator=">",
            value=150.0
        )
        
        price_trigger = self.AlertTrigger(
            trigger_type=self.AlertTriggerType.CONDITION_BASED,
            conditions=[price_condition]
        )
        
        price_alert = self.Alert(
            id="demo_price_alert_001",
            name="High Price Alert",
            description="Alert when stock price exceeds 150",
            stock_code="000001",
            trigger=price_trigger,
            priority=self.AlertPriority.HIGH,
            user_id="demo_user"
        )
        
        logger.info(f"Created alert: {price_alert.name}")
        logger.info(f"Alert ID: {price_alert.id}")
        logger.info(f"Priority: {price_alert.priority.value}")
        logger.info(f"Stock Code: {price_alert.stock_code}")
        
        # Test condition evaluation
        test_data = {"close_price": 155.0}
        should_trigger = price_alert.trigger.should_trigger(test_data)
        logger.info(f"Should trigger with price 155.0: {should_trigger}")
        
        test_data = {"close_price": 145.0}
        should_trigger = price_alert.trigger.should_trigger(test_data)
        logger.info(f"Should trigger with price 145.0: {should_trigger}")
        
        return price_alert
    
    async def demo_notification_templates(self):
        """Demonstrate notification template creation and rendering"""
        logger.info("\n=== Demo: Notification Templates ===")
        
        # Create email template
        email_template = self.NotificationTemplate(
            id="demo_email_template",
            name="Email Alert Template",
            channel=self.NotificationChannel.EMAIL,
            subject_template="🚨 Stock Alert: {{ alert_name }}",
            body_template="""
            <h2>Stock Alert Triggered</h2>
            <p><strong>Alert:</strong> {{ alert_name }}</p>
            <p><strong>Stock:</strong> {{ stock_code }}</p>
            <p><strong>Description:</strong> {{ alert_description }}</p>
            <p><strong>Priority:</strong> {{ priority }}</p>
            <p><strong>Time:</strong> {{ triggered_at }}</p>
            """,
            priority=self.AlertPriority.HIGH
        )
        
        await self.notification_system.create_template(email_template)
        logger.info("Created email notification template")
        
        # Test template rendering
        context = {
            'alert_name': 'High Price Alert',
            'stock_code': '000001',
            'alert_description': 'Price exceeded threshold',
            'priority': 'HIGH',
            'triggered_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        rendered = email_template.render(context)
        logger.info(f"Rendered subject: {rendered['subject']}")
        logger.info("Rendered body preview: [HTML content with alert details]")
        
        return email_template
    
    async def demo_user_preferences(self):
        """Demonstrate user notification preferences"""
        logger.info("\n=== Demo: User Notification Preferences ===")
        
        # Set up user preferences
        user_preferences = [
            self.NotificationPreference(
                user_id="demo_user",
                channel=self.NotificationChannel.EMAIL,
                enabled=True,
                quiet_hours_start="22:00",
                quiet_hours_end="08:00",
                priority_filter=[self.AlertPriority.HIGH, self.AlertPriority.CRITICAL]
            ),
            self.NotificationPreference(
                user_id="demo_user",
                channel=self.NotificationChannel.SMS,
                enabled=True,
                priority_filter=[self.AlertPriority.CRITICAL]
            ),
            self.NotificationPreference(
                user_id="demo_user",
                channel=self.NotificationChannel.IN_APP,
                enabled=True
            )
        ]
        
        await self.notification_system.set_user_preferences("demo_user", user_preferences)
        logger.info("Set up user notification preferences")
        
        # Test preference evaluation
        current_time = datetime.now().replace(hour=10, minute=0)  # 10 AM
        should_send = user_preferences[0].should_send(self.AlertPriority.HIGH, current_time)
        logger.info(f"Should send HIGH priority email at 10 AM: {should_send}")
        
        current_time = datetime.now().replace(hour=23, minute=0)  # 11 PM (quiet hours)
        should_send = user_preferences[0].should_send(self.AlertPriority.HIGH, current_time)
        logger.info(f"Should send HIGH priority email at 11 PM: {should_send}")
        
        should_send = user_preferences[1].should_send(self.AlertPriority.MEDIUM, current_time)
        logger.info(f"Should send MEDIUM priority SMS: {should_send}")
        
        return user_preferences
    
    async def demo_alert_filtering(self):
        """Demonstrate smart alert filtering"""
        logger.info("\n=== Demo: Smart Alert Filtering ===")
        
        # Create similar alerts to test deduplication
        alerts = []
        for i in range(3):
            condition = self.AlertCondition(field="close_price", operator=">", value=100.0)
            trigger = self.AlertTrigger(
                trigger_type=self.AlertTriggerType.CONDITION_BASED,
                conditions=[condition]
            )
            
            alert = self.Alert(
                id=f"demo_filter_alert_{i:03d}",
                name="Duplicate Price Alert",  # Same name for deduplication test
                description="Test alert for filtering",
                stock_code="000001",  # Same stock
                trigger=trigger,
                priority=self.AlertPriority.MEDIUM,
                user_id="demo_user"
            )
            alerts.append(alert)
        
        # Filter alerts
        from stock_analysis_system.alerts.alert_filtering import FilterAction
        
        results = []
        for i, alert in enumerate(alerts):
            action, reason = await self.alert_filter.filter_alert(alert)
            results.append((action, reason))
            logger.info(f"Alert {i+1}: {action.value} - {reason}")
        
        # Update market conditions
        market_conditions = self.MarketCondition(
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
        
        return alerts, results
    
    async def demo_alert_aggregation(self):
        """Demonstrate alert aggregation and clustering"""
        logger.info("\n=== Demo: Alert Aggregation ===")
        
        # Create similar alerts for clustering
        similar_alerts = []
        for i in range(5):
            condition = self.AlertCondition(field="close_price", operator=">", value=100.0 + i * 5)
            trigger = self.AlertTrigger(
                trigger_type=self.AlertTriggerType.CONDITION_BASED,
                conditions=[condition]
            )
            
            alert = self.Alert(
                id=f"demo_cluster_alert_{i:03d}",
                name=f"Price Threshold Alert {i}",
                description="Alert when price exceeds threshold level",
                stock_code="000001",
                trigger=trigger,
                priority=self.AlertPriority.MEDIUM,
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
    
    async def demo_in_app_notifications(self):
        """Demonstrate in-app notification functionality"""
        logger.info("\n=== Demo: In-App Notifications ===")
        
        # Create a test alert
        alert = self.Alert(
            id="demo_notification_alert",
            name="Test Notification Alert",
            description="Testing in-app notifications",
            stock_code="000001",
            trigger=self.AlertTrigger(
                trigger_type=self.AlertTriggerType.CONDITION_BASED,
                conditions=[self.AlertCondition(field="close_price", operator=">", value=100.0)]
            ),
            priority=self.AlertPriority.HIGH,
            user_id="demo_user"
        )
        
        # Simulate alert triggering
        alert.trigger_alert({"close_price": 105.0})
        
        # Set up in-app notification preference
        preferences = [
            self.NotificationPreference(
                user_id="demo_user",
                channel=self.NotificationChannel.IN_APP,
                enabled=True
            )
        ]
        await self.notification_system.set_user_preferences("demo_user", preferences)
        
        # Send notification
        notification_ids = await self.notification_system.send_alert_notification(
            alert, "demo_user"
        )
        logger.info(f"Sent {len(notification_ids)} notifications")
        
        # Get in-app notifications
        notifications = await self.notification_system.get_in_app_notifications("demo_user")
        logger.info(f"In-app notifications: {len(notifications)}")
        
        if notifications:
            notification = notifications[0]
            logger.info(f"Latest notification: {notification['subject']}")
            logger.info(f"Read status: {notification['read']}")
            
            # Mark as read
            success = await self.notification_system.mark_notification_read(
                "demo_user", notification['id']
            )
            logger.info(f"Marked as read: {success}")
        
        return notifications
    
    async def demo_complete_workflow(self):
        """Demonstrate complete alert workflow"""
        logger.info("\n=== Demo: Complete Alert Workflow ===")
        
        # 1. Create alert
        workflow_alert = self.Alert(
            id="demo_workflow_alert",
            name="Complete Workflow Alert",
            description="End-to-end workflow demonstration",
            stock_code="000001",
            trigger=self.AlertTrigger(
                trigger_type=self.AlertTriggerType.CONDITION_BASED,
                conditions=[self.AlertCondition(field="close_price", operator=">", value=120.0)]
            ),
            priority=self.AlertPriority.HIGH,
            user_id="demo_user"
        )
        logger.info("1. Created workflow alert")
        
        # 2. Filter alert
        from stock_analysis_system.alerts.alert_filtering import FilterAction
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
            
            # 5. Simulate user acknowledgment
            workflow_alert.status = self.Alert.__annotations__['status'].__args__[0].ACKNOWLEDGED
            logger.info("5. Alert acknowledged by user")
            
            # 6. Simulate resolution
            workflow_alert.status = self.Alert.__annotations__['status'].__args__[0].RESOLVED
            logger.info("6. Alert resolved by user")
        
        return workflow_alert
    
    async def run_complete_demo(self):
        """Run the complete simplified alert system demonstration"""
        logger.info("🚀 Starting Simplified Alert and Notification System Demo")
        logger.info("=" * 70)
        
        try:
            # Basic alert creation
            basic_alert = await self.demo_basic_alert_creation()
            
            # Notification templates
            email_template = await self.demo_notification_templates()
            
            # User preferences
            user_preferences = await self.demo_user_preferences()
            
            # Alert filtering
            filtered_alerts, filter_results = await self.demo_alert_filtering()
            
            # Alert aggregation
            clustered_alerts, cluster_id = await self.demo_alert_aggregation()
            
            # In-app notifications
            notifications = await self.demo_in_app_notifications()
            
            # Complete workflow
            workflow_alert = await self.demo_complete_workflow()
            
            # Final summary
            logger.info("\n" + "=" * 70)
            logger.info("📊 Demo Summary")
            logger.info("=" * 70)
            
            logger.info(f"✓ Created basic alert: {basic_alert.name}")
            logger.info(f"✓ Created notification template: {email_template.name}")
            logger.info(f"✓ Set up {len(user_preferences)} user preferences")
            logger.info(f"✓ Filtered {len(filtered_alerts)} alerts")
            logger.info(f"✓ Clustered {len(clustered_alerts)} alerts")
            logger.info(f"✓ Generated {len(notifications)} in-app notifications")
            logger.info(f"✓ Completed workflow for: {workflow_alert.name}")
            
            # Get system statistics
            filter_stats = await self.alert_filter.get_filter_statistics()
            agg_stats = await self.alert_aggregator.get_aggregation_statistics()
            notification_analytics = await self.notification_system.get_delivery_analytics()
            
            logger.info(f"\n📈 System Statistics:")
            logger.info(f"Filter statistics: {filter_stats}")
            logger.info(f"Aggregation statistics: {agg_stats}")
            logger.info(f"Notification analytics: {notification_analytics}")
            
            logger.info("\n✅ Simplified Alert and Notification System Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise


async def main():
    """Main demo function"""
    demo = SimplifiedAlertDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())