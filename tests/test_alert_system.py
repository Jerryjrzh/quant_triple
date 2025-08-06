"""
Comprehensive tests for the Alert and Notification System
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.orm import Session

from stock_analysis_system.alerts.alert_engine import (
    AlertEngine, Alert, AlertTrigger, AlertCondition, 
    AlertPriority, AlertTriggerType, AlertStatus
)
from stock_analysis_system.alerts.notification_system import (
    NotificationSystem, NotificationChannel, NotificationTemplate,
    NotificationPreference, NotificationStatus
)
from stock_analysis_system.alerts.alert_filtering import (
    SmartAlertFilter, AlertAggregator, MarketCondition, FilterAction
)


class TestAlertEngine:
    """Test cases for AlertEngine"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock(spec=Session)
        session.add = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.query = Mock()
        return session
    
    @pytest.fixture
    def mock_engines(self):
        """Mock analysis engines"""
        spring_festival_engine = Mock()
        risk_engine = Mock()
        institutional_engine = Mock()
        
        return spring_festival_engine, risk_engine, institutional_engine
    
    @pytest.fixture
    def alert_engine(self, mock_db_session, mock_engines):
        """Create AlertEngine instance for testing"""
        spring_festival_engine, risk_engine, institutional_engine = mock_engines
        
        engine = AlertEngine(
            db_session=mock_db_session,
            spring_festival_engine=spring_festival_engine,
            risk_engine=risk_engine,
            institutional_engine=institutional_engine
        )
        
        return engine
    
    @pytest.fixture
    def sample_alert(self):
        """Create a sample alert for testing"""
        condition = AlertCondition(
            field="close_price",
            operator=">",
            value=100.0
        )
        
        trigger = AlertTrigger(
            trigger_type=AlertTriggerType.CONDITION_BASED,
            conditions=[condition]
        )
        
        alert = Alert(
            id="test_alert_001",
            name="Test Price Alert",
            description="Alert when price exceeds 100",
            stock_code="000001",
            trigger=trigger,
            priority=AlertPriority.MEDIUM,
            user_id="test_user"
        )
        
        return alert
    
    @pytest.mark.asyncio
    async def test_create_alert(self, alert_engine, sample_alert):
        """Test alert creation"""
        alert_id = await alert_engine.create_alert(sample_alert)
        
        assert alert_id == sample_alert.id
        assert sample_alert.id in alert_engine.active_alerts
        
        # Verify database interaction
        alert_engine.db_session.add.assert_called_once()
        alert_engine.db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_alert(self, alert_engine, sample_alert):
        """Test alert retrieval"""
        # First create the alert
        await alert_engine.create_alert(sample_alert)
        
        # Then retrieve it
        retrieved_alert = await alert_engine.get_alert(sample_alert.id)
        
        assert retrieved_alert is not None
        assert retrieved_alert.id == sample_alert.id
        assert retrieved_alert.name == sample_alert.name
    
    @pytest.mark.asyncio
    async def test_update_alert(self, alert_engine, sample_alert):
        """Test alert update"""
        # Create alert first
        await alert_engine.create_alert(sample_alert)
        
        # Update alert
        updates = {
            'name': 'Updated Alert Name',
            'priority': AlertPriority.HIGH.value
        }
        
        success = await alert_engine.update_alert(sample_alert.id, updates)
        
        assert success is True
        
        # Verify updates
        updated_alert = await alert_engine.get_alert(sample_alert.id)
        assert updated_alert.name == 'Updated Alert Name'
    
    @pytest.mark.asyncio
    async def test_delete_alert(self, alert_engine, sample_alert):
        """Test alert deletion"""
        # Create alert first
        await alert_engine.create_alert(sample_alert)
        
        # Delete alert
        success = await alert_engine.delete_alert(sample_alert.id)
        
        assert success is True
        assert sample_alert.id not in alert_engine.active_alerts
    
    @pytest.mark.asyncio
    async def test_list_alerts(self, alert_engine, sample_alert):
        """Test alert listing with filters"""
        # Create multiple alerts
        await alert_engine.create_alert(sample_alert)
        
        # Create another alert with different user
        alert2 = Alert(
            id="test_alert_002",
            name="Test Alert 2",
            description="Another test alert",
            stock_code="000002",
            trigger=sample_alert.trigger,
            priority=AlertPriority.HIGH,
            user_id="another_user"
        )
        await alert_engine.create_alert(alert2)
        
        # Test listing all alerts
        all_alerts = await alert_engine.list_alerts()
        assert len(all_alerts) == 2
        
        # Test filtering by user
        user_alerts = await alert_engine.list_alerts(user_id="test_user")
        assert len(user_alerts) == 1
        assert user_alerts[0].user_id == "test_user"
        
        # Test filtering by status
        active_alerts = await alert_engine.list_alerts(status=AlertStatus.ACTIVE)
        assert len(active_alerts) == 2
    
    @pytest.mark.asyncio
    async def test_condition_evaluation(self, alert_engine):
        """Test alert condition evaluation"""
        # Test greater than condition
        condition = AlertCondition(field="close_price", operator=">", value=100.0)
        
        # Should trigger
        data = {"close_price": 105.0}
        assert condition.evaluate(data) is True
        
        # Should not trigger
        data = {"close_price": 95.0}
        assert condition.evaluate(data) is False
        
        # Test equality condition
        condition = AlertCondition(field="stock_code", operator="==", value="000001")
        
        data = {"stock_code": "000001"}
        assert condition.evaluate(data) is True
        
        data = {"stock_code": "000002"}
        assert condition.evaluate(data) is False
    
    @pytest.mark.asyncio
    async def test_trigger_logic(self, alert_engine):
        """Test alert trigger logic (AND/OR)"""
        condition1 = AlertCondition(field="close_price", operator=">", value=100.0)
        condition2 = AlertCondition(field="volume", operator=">", value=1000000)
        
        # Test AND logic
        trigger = AlertTrigger(
            trigger_type=AlertTriggerType.CONDITION_BASED,
            conditions=[condition1, condition2],
            logic_operator="AND"
        )
        
        # Both conditions true
        data = {"close_price": 105.0, "volume": 1500000}
        assert trigger.should_trigger(data) is True
        
        # Only one condition true
        data = {"close_price": 105.0, "volume": 500000}
        assert trigger.should_trigger(data) is False
        
        # Test OR logic
        trigger.logic_operator = "OR"
        
        # Only one condition true
        data = {"close_price": 105.0, "volume": 500000}
        assert trigger.should_trigger(data) is True
        
        # No conditions true
        data = {"close_price": 95.0, "volume": 500000}
        assert trigger.should_trigger(data) is False


class TestNotificationSystem:
    """Test cases for NotificationSystem"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock(spec=Session)
        session.add = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        return session
    
    @pytest.fixture
    def notification_system(self, mock_db_session):
        """Create NotificationSystem instance for testing"""
        return NotificationSystem(db_session=mock_db_session)
    
    @pytest.fixture
    def sample_template(self):
        """Create a sample notification template"""
        return NotificationTemplate(
            id="test_template_001",
            name="Test Email Template",
            channel=NotificationChannel.EMAIL,
            subject_template="Alert: {{ alert_name }}",
            body_template="Alert {{ alert_name }} triggered for {{ stock_code }}",
            priority=AlertPriority.MEDIUM
        )
    
    @pytest.fixture
    def sample_alert(self):
        """Create a sample alert for notification testing"""
        condition = AlertCondition(field="close_price", operator=">", value=100.0)
        trigger = AlertTrigger(
            trigger_type=AlertTriggerType.CONDITION_BASED,
            conditions=[condition]
        )
        
        alert = Alert(
            id="test_alert_001",
            name="Test Price Alert",
            description="Alert when price exceeds 100",
            stock_code="000001",
            trigger=trigger,
            priority=AlertPriority.MEDIUM,
            user_id="test_user"
        )
        alert.last_triggered = datetime.now()
        
        return alert
    
    @pytest.mark.asyncio
    async def test_create_template(self, notification_system, sample_template):
        """Test notification template creation"""
        template_id = await notification_system.create_template(sample_template)
        
        assert template_id == sample_template.id
        assert sample_template.id in notification_system.templates
    
    @pytest.mark.asyncio
    async def test_template_rendering(self, notification_system, sample_template):
        """Test template rendering with context"""
        await notification_system.create_template(sample_template)
        
        context = {
            'alert_name': 'Price Alert',
            'stock_code': '000001'
        }
        
        rendered = sample_template.render(context)
        
        assert rendered['subject'] == 'Alert: Price Alert'
        assert 'Price Alert' in rendered['body']
        assert '000001' in rendered['body']
    
    @pytest.mark.asyncio
    async def test_user_preferences(self, notification_system):
        """Test user notification preferences"""
        user_id = "test_user"
        
        preferences = [
            NotificationPreference(
                user_id=user_id,
                channel=NotificationChannel.EMAIL,
                enabled=True,
                quiet_hours_start="22:00",
                quiet_hours_end="08:00"
            ),
            NotificationPreference(
                user_id=user_id,
                channel=NotificationChannel.SMS,
                enabled=False
            )
        ]
        
        await notification_system.set_user_preferences(user_id, preferences)
        
        retrieved_prefs = await notification_system.get_user_preferences(user_id)
        
        assert len(retrieved_prefs) == 2
        assert retrieved_prefs[0].channel == NotificationChannel.EMAIL
        assert retrieved_prefs[0].enabled is True
        assert retrieved_prefs[1].enabled is False
    
    @pytest.mark.asyncio
    async def test_in_app_notifications(self, notification_system, sample_alert):
        """Test in-app notification functionality"""
        user_id = "test_user"
        
        # Set up preferences for in-app notifications
        preferences = [
            NotificationPreference(
                user_id=user_id,
                channel=NotificationChannel.IN_APP,
                enabled=True
            )
        ]
        await notification_system.set_user_preferences(user_id, preferences)
        
        # Send notification
        notification_ids = await notification_system.send_alert_notification(
            sample_alert, user_id
        )
        
        assert len(notification_ids) > 0
        
        # Get in-app notifications
        notifications = await notification_system.get_in_app_notifications(user_id)
        
        assert len(notifications) > 0
        assert notifications[0]['read'] is False
        
        # Mark as read
        notification_id = notifications[0]['id']
        success = await notification_system.mark_notification_read(user_id, notification_id)
        
        assert success is True
        
        # Verify marked as read
        notifications = await notification_system.get_in_app_notifications(user_id)
        assert notifications[0]['read'] is True
    
    @pytest.mark.asyncio
    async def test_quiet_hours(self, notification_system):
        """Test quiet hours functionality"""
        user_id = "test_user"
        
        preference = NotificationPreference(
            user_id=user_id,
            channel=NotificationChannel.EMAIL,
            enabled=True,
            quiet_hours_start="22:00",
            quiet_hours_end="08:00"
        )
        
        # Test during quiet hours
        quiet_time = datetime.now().replace(hour=23, minute=0)
        should_send = preference.should_send(AlertPriority.MEDIUM, quiet_time)
        assert should_send is False
        
        # Test outside quiet hours
        active_time = datetime.now().replace(hour=10, minute=0)
        should_send = preference.should_send(AlertPriority.MEDIUM, active_time)
        assert should_send is True
    
    @pytest.mark.asyncio
    async def test_priority_filtering(self, notification_system):
        """Test priority-based notification filtering"""
        user_id = "test_user"
        
        preference = NotificationPreference(
            user_id=user_id,
            channel=NotificationChannel.EMAIL,
            enabled=True,
            priority_filter=[AlertPriority.HIGH, AlertPriority.CRITICAL]
        )
        
        # Should send high priority
        should_send = preference.should_send(AlertPriority.HIGH, datetime.now())
        assert should_send is True
        
        # Should not send medium priority
        should_send = preference.should_send(AlertPriority.MEDIUM, datetime.now())
        assert should_send is False


class TestSmartAlertFilter:
    """Test cases for SmartAlertFilter"""
    
    @pytest.fixture
    def alert_filter(self):
        """Create SmartAlertFilter instance for testing"""
        return SmartAlertFilter()
    
    @pytest.fixture
    def sample_alert(self):
        """Create a sample alert for filtering tests"""
        condition = AlertCondition(field="close_price", operator=">", value=100.0)
        trigger = AlertTrigger(
            trigger_type=AlertTriggerType.CONDITION_BASED,
            conditions=[condition]
        )
        
        return Alert(
            id="test_alert_001",
            name="Test Price Alert",
            description="Alert when price exceeds 100",
            stock_code="000001",
            trigger=trigger,
            priority=AlertPriority.MEDIUM,
            user_id="test_user"
        )
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self, alert_filter, sample_alert):
        """Test duplicate alert detection"""
        # First alert should be allowed
        action, reason = await alert_filter.filter_alert(sample_alert)
        assert action == FilterAction.ALLOW
        
        # Immediate duplicate should be suppressed
        duplicate_alert = Alert(
            id="test_alert_002",
            name=sample_alert.name,  # Same name
            description=sample_alert.description,
            stock_code=sample_alert.stock_code,  # Same stock
            trigger=sample_alert.trigger,  # Same trigger type
            priority=sample_alert.priority,
            user_id=sample_alert.user_id
        )
        
        action, reason = await alert_filter.filter_alert(duplicate_alert)
        assert action == FilterAction.SUPPRESS
        assert "Duplicate" in reason
    
    @pytest.mark.asyncio
    async def test_market_conditions_adaptation(self, alert_filter, sample_alert):
        """Test adaptive thresholds based on market conditions"""
        # Set extreme volatility conditions
        market_conditions = MarketCondition(
            volatility_level="extreme",
            trend_direction="down",
            volume_level="high",
            market_hours=True
        )
        
        await alert_filter.update_market_conditions(market_conditions)
        
        # Test that thresholds are adjusted
        base_threshold = 0.5
        adjusted = alert_filter._adjust_threshold_for_market_conditions(
            base_threshold, sample_alert
        )
        
        # Should be lower threshold (more alerts) during extreme volatility
        assert adjusted < base_threshold
    
    @pytest.mark.asyncio
    async def test_filter_statistics(self, alert_filter, sample_alert):
        """Test filter statistics collection"""
        # Process some alerts
        await alert_filter.filter_alert(sample_alert)
        
        # Create a duplicate to be suppressed
        duplicate_alert = Alert(
            id="test_alert_002",
            name=sample_alert.name,
            description=sample_alert.description,
            stock_code=sample_alert.stock_code,
            trigger=sample_alert.trigger,
            priority=sample_alert.priority,
            user_id=sample_alert.user_id
        )
        
        await alert_filter.filter_alert(duplicate_alert)
        
        # Get statistics
        stats = await alert_filter.get_filter_statistics()
        
        assert 'total_processed' in stats
        assert 'total_suppressed' in stats
        assert 'suppression_rate' in stats
        assert stats['total_processed'] >= 2


class TestAlertAggregator:
    """Test cases for AlertAggregator"""
    
    @pytest.fixture
    def alert_aggregator(self):
        """Create AlertAggregator instance for testing"""
        return AlertAggregator()
    
    @pytest.fixture
    def similar_alerts(self):
        """Create similar alerts for clustering tests"""
        alerts = []
        
        for i in range(5):
            condition = AlertCondition(field="close_price", operator=">", value=100.0 + i)
            trigger = AlertTrigger(
                trigger_type=AlertTriggerType.CONDITION_BASED,
                conditions=[condition]
            )
            
            alert = Alert(
                id=f"test_alert_{i:03d}",
                name=f"Price Alert {i}",
                description="Alert when price exceeds threshold",
                stock_code="000001",
                trigger=trigger,
                priority=AlertPriority.MEDIUM,
                user_id="test_user"
            )
            alerts.append(alert)
        
        return alerts
    
    @pytest.mark.asyncio
    async def test_alert_aggregation(self, alert_aggregator, similar_alerts):
        """Test alert aggregation and clustering"""
        # Add alerts for aggregation
        cluster_id = None
        for alert in similar_alerts:
            result = await alert_aggregator.add_alert_for_aggregation(alert)
            if result:
                cluster_id = result
        
        # Should have created a cluster
        assert cluster_id is not None
        
        # Get cluster details
        cluster = await alert_aggregator.get_cluster(cluster_id)
        assert cluster is not None
        assert len(cluster.alerts) > 1
    
    @pytest.mark.asyncio
    async def test_cluster_summary(self, alert_aggregator, similar_alerts):
        """Test cluster summary generation"""
        # Force aggregation
        for alert in similar_alerts:
            await alert_aggregator.add_alert_for_aggregation(alert)
        
        cluster_ids = await alert_aggregator.force_aggregation()
        
        if cluster_ids:
            cluster_id = cluster_ids[0]
            summary = await alert_aggregator.get_cluster_summary(cluster_id)
            
            assert summary is not None
            assert 'cluster_id' in summary
            assert 'alert_count' in summary
            assert 'representative_alert' in summary
            assert 'similarity_score' in summary
    
    @pytest.mark.asyncio
    async def test_aggregation_statistics(self, alert_aggregator, similar_alerts):
        """Test aggregation statistics"""
        # Add some alerts
        for alert in similar_alerts[:3]:
            await alert_aggregator.add_alert_for_aggregation(alert)
        
        # Get statistics
        stats = await alert_aggregator.get_aggregation_statistics()
        
        assert 'total_clusters' in stats
        assert 'pending_alerts' in stats
        assert 'similarity_threshold' in stats
        assert stats['pending_alerts'] == 3


class TestIntegration:
    """Integration tests for the complete alert system"""
    
    @pytest.fixture
    def complete_system(self):
        """Set up complete alert system for integration testing"""
        # Mock dependencies
        db_session = Mock(spec=Session)
        db_session.add = Mock()
        db_session.commit = Mock()
        db_session.rollback = Mock()
        db_session.query = Mock()
        
        spring_festival_engine = Mock()
        risk_engine = Mock()
        institutional_engine = Mock()
        
        # Create system components
        alert_engine = AlertEngine(
            db_session=db_session,
            spring_festival_engine=spring_festival_engine,
            risk_engine=risk_engine,
            institutional_engine=institutional_engine
        )
        
        notification_system = NotificationSystem(db_session=db_session)
        alert_filter = SmartAlertFilter()
        alert_aggregator = AlertAggregator()
        
        return {
            'alert_engine': alert_engine,
            'notification_system': notification_system,
            'alert_filter': alert_filter,
            'alert_aggregator': alert_aggregator
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_alert_flow(self, complete_system):
        """Test complete alert flow from creation to notification"""
        alert_engine = complete_system['alert_engine']
        notification_system = complete_system['notification_system']
        alert_filter = complete_system['alert_filter']
        
        # Create alert
        condition = AlertCondition(field="close_price", operator=">", value=100.0)
        trigger = AlertTrigger(
            trigger_type=AlertTriggerType.CONDITION_BASED,
            conditions=[condition]
        )
        
        alert = Alert(
            id="integration_test_001",
            name="Integration Test Alert",
            description="End-to-end test alert",
            stock_code="000001",
            trigger=trigger,
            priority=AlertPriority.HIGH,
            user_id="test_user"
        )
        
        # Create alert
        alert_id = await alert_engine.create_alert(alert)
        assert alert_id == alert.id
        
        # Filter alert (should be allowed)
        action, reason = await alert_filter.filter_alert(alert)
        assert action == FilterAction.ALLOW
        
        # Set up notification preferences
        preferences = [
            NotificationPreference(
                user_id="test_user",
                channel=NotificationChannel.IN_APP,
                enabled=True
            )
        ]
        await notification_system.set_user_preferences("test_user", preferences)
        
        # Trigger alert (simulate)
        alert.trigger_alert({"close_price": 105.0})
        
        # Send notification
        notification_ids = await notification_system.send_alert_notification(
            alert, "test_user"
        )
        
        assert len(notification_ids) > 0
        
        # Verify in-app notification
        notifications = await notification_system.get_in_app_notifications("test_user")
        assert len(notifications) > 0
        assert notifications[0]['alert_id'] == alert.id
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, complete_system):
        """Test system performance under load"""
        alert_engine = complete_system['alert_engine']
        alert_filter = complete_system['alert_filter']
        
        # Create many alerts quickly
        alerts = []
        for i in range(100):
            condition = AlertCondition(field="close_price", operator=">", value=100.0 + i)
            trigger = AlertTrigger(
                trigger_type=AlertTriggerType.CONDITION_BASED,
                conditions=[condition]
            )
            
            alert = Alert(
                id=f"load_test_{i:03d}",
                name=f"Load Test Alert {i}",
                description="Load testing alert",
                stock_code=f"{i:06d}",
                trigger=trigger,
                priority=AlertPriority.MEDIUM,
                user_id="load_test_user"
            )
            alerts.append(alert)
        
        # Measure creation time
        start_time = datetime.now()
        
        for alert in alerts:
            await alert_engine.create_alert(alert)
            await alert_filter.filter_alert(alert)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should process 100 alerts in reasonable time (< 5 seconds)
        assert duration < 5.0
        
        # Verify all alerts were created
        all_alerts = await alert_engine.list_alerts()
        assert len(all_alerts) >= 100