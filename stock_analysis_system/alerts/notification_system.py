"""
Multi-Channel Notification System

This module provides comprehensive notification capabilities including email, SMS, 
webhook, and in-app notifications with templates and delivery tracking.
"""

import asyncio
import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import jinja2
from sqlalchemy.orm import Session

from ..data.models import NotificationLog
from .alert_engine import Alert, AlertPriority

logger = logging.getLogger(__name__)


class NotificationChannel(str, Enum):
    """Available notification channels"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    IN_APP = "in_app"
    PUSH = "push"


class NotificationStatus(str, Enum):
    """Notification delivery status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"


@dataclass
class NotificationTemplate:
    """Template for notifications"""
    id: str
    name: str
    channel: NotificationChannel
    subject_template: str
    body_template: str
    priority: AlertPriority
    variables: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def render(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Render the template with provided context"""
        env = jinja2.Environment()
        
        subject = env.from_string(self.subject_template).render(context)
        body = env.from_string(self.body_template).render(context)
        
        return {
            'subject': subject,
            'body': body
        }


@dataclass
class NotificationPreference:
    """User notification preferences"""
    user_id: str
    channel: NotificationChannel
    enabled: bool = True
    quiet_hours_start: Optional[str] = None  # "22:00"
    quiet_hours_end: Optional[str] = None    # "08:00"
    priority_filter: List[AlertPriority] = field(default_factory=list)
    frequency_limit: Optional[int] = None  # Max notifications per hour
    
    def should_send(self, priority: AlertPriority, current_time: datetime) -> bool:
        """Check if notification should be sent based on preferences"""
        if not self.enabled:
            return False
        
        # Check priority filter
        if self.priority_filter and priority not in self.priority_filter:
            return False
        
        # Check quiet hours
        if self.quiet_hours_start and self.quiet_hours_end:
            current_hour = current_time.strftime("%H:%M")
            if self.quiet_hours_start <= current_hour <= self.quiet_hours_end:
                return False
        
        return True


@dataclass
class NotificationRecord:
    """Record of a sent notification"""
    id: str
    user_id: str
    alert_id: str
    channel: NotificationChannel
    status: NotificationStatus
    sent_at: datetime
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmailNotificationProvider:
    """Email notification provider"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
    
    async def send_email(self, to_email: str, subject: str, body: str, 
                        is_html: bool = False) -> bool:
        """Send an email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html' if is_html else 'plain'))
            
            # Send email in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_smtp_email, msg, to_email)
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False
    
    def _send_smtp_email(self, msg: MIMEMultipart, to_email: str) -> None:
        """Send email via SMTP (blocking operation)"""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg, to_addrs=[to_email])


class SMSNotificationProvider:
    """SMS notification provider (placeholder for integration with SMS service)"""
    
    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url
    
    async def send_sms(self, phone_number: str, message: str) -> bool:
        """Send an SMS notification"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'to': phone_number,
                    'message': message,
                    'api_key': self.api_key
                }
                
                async with session.post(self.api_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"SMS sent successfully to {phone_number}")
                        return True
                    else:
                        logger.error(f"SMS failed with status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send SMS to {phone_number}: {e}")
            return False


class WebhookNotificationProvider:
    """Webhook notification provider"""
    
    async def send_webhook(self, webhook_url: str, payload: Dict[str, Any]) -> bool:
        """Send a webhook notification"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status in [200, 201, 202]:
                        logger.info(f"Webhook sent successfully to {webhook_url}")
                        return True
                    else:
                        logger.error(f"Webhook failed with status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send webhook to {webhook_url}: {e}")
            return False


class NotificationSystem:
    """
    Comprehensive notification system with multi-channel support
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.templates: Dict[str, NotificationTemplate] = {}
        self.user_preferences: Dict[str, List[NotificationPreference]] = {}
        self.notification_history: List[NotificationRecord] = []
        self.delivery_analytics: Dict[str, Any] = {}
        
        # Notification providers
        self.email_provider: Optional[EmailNotificationProvider] = None
        self.sms_provider: Optional[SMSNotificationProvider] = None
        self.webhook_provider: Optional[WebhookNotificationProvider] = None
        
        # In-app notification queue
        self.in_app_notifications: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("NotificationSystem initialized")
    
    def configure_email_provider(self, smtp_host: str, smtp_port: int, 
                                username: str, password: str) -> None:
        """Configure email notification provider"""
        self.email_provider = EmailNotificationProvider(
            smtp_host, smtp_port, username, password
        )
        logger.info("Email provider configured")
    
    def configure_sms_provider(self, api_key: str, api_url: str) -> None:
        """Configure SMS notification provider"""
        self.sms_provider = SMSNotificationProvider(api_key, api_url)
        logger.info("SMS provider configured")
    
    def configure_webhook_provider(self) -> None:
        """Configure webhook notification provider"""
        self.webhook_provider = WebhookNotificationProvider()
        logger.info("Webhook provider configured")
    
    async def create_template(self, template: NotificationTemplate) -> str:
        """Create a notification template"""
        self.templates[template.id] = template
        logger.info(f"Created notification template: {template.name}")
        return template.id
    
    async def update_template(self, template_id: str, updates: Dict[str, Any]) -> bool:
        """Update a notification template"""
        if template_id not in self.templates:
            return False
        
        template = self.templates[template_id]
        for key, value in updates.items():
            if hasattr(template, key):
                setattr(template, key, value)
        
        logger.info(f"Updated notification template: {template_id}")
        return True
    
    async def delete_template(self, template_id: str) -> bool:
        """Delete a notification template"""
        if template_id in self.templates:
            del self.templates[template_id]
            logger.info(f"Deleted notification template: {template_id}")
            return True
        return False
    
    async def set_user_preferences(self, user_id: str, 
                                  preferences: List[NotificationPreference]) -> None:
        """Set notification preferences for a user"""
        self.user_preferences[user_id] = preferences
        logger.info(f"Updated notification preferences for user: {user_id}")
    
    async def get_user_preferences(self, user_id: str) -> List[NotificationPreference]:
        """Get notification preferences for a user"""
        return self.user_preferences.get(user_id, [])
    
    async def send_alert_notification(self, alert: Alert, user_id: str, 
                                    template_id: Optional[str] = None) -> List[str]:
        """Send notifications for an alert to a user"""
        notification_ids = []
        
        try:
            # Get user preferences
            preferences = await self.get_user_preferences(user_id)
            
            if not preferences:
                # Default to in-app notifications if no preferences set
                preferences = [NotificationPreference(
                    user_id=user_id,
                    channel=NotificationChannel.IN_APP
                )]
            
            # Send notification through each enabled channel
            for preference in preferences:
                if preference.should_send(alert.priority, datetime.now()):
                    notification_id = await self._send_single_notification(
                        alert, user_id, preference.channel, template_id
                    )
                    if notification_id:
                        notification_ids.append(notification_id)
            
            return notification_ids
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
            return []
    
    async def _send_single_notification(self, alert: Alert, user_id: str, 
                                      channel: NotificationChannel, 
                                      template_id: Optional[str] = None) -> Optional[str]:
        """Send a single notification through specified channel"""
        try:
            # Generate notification ID
            notification_id = f"notif_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}_{channel.value}"
            
            # Get or create template
            template = await self._get_or_create_template(channel, alert.priority, template_id)
            
            # Prepare context for template rendering
            context = {
                'alert_name': alert.name,
                'alert_description': alert.description,
                'stock_code': alert.stock_code or 'N/A',
                'priority': alert.priority.value,
                'triggered_at': alert.last_triggered.strftime('%Y-%m-%d %H:%M:%S') if alert.last_triggered else 'N/A',
                'user_id': user_id
            }
            
            # Render template
            rendered = template.render(context)
            
            # Create notification record
            record = NotificationRecord(
                id=notification_id,
                user_id=user_id,
                alert_id=alert.id,
                channel=channel,
                status=NotificationStatus.PENDING,
                sent_at=datetime.now()
            )
            
            # Send through appropriate channel
            success = False
            
            if channel == NotificationChannel.EMAIL:
                success = await self._send_email_notification(
                    user_id, rendered['subject'], rendered['body']
                )
            
            elif channel == NotificationChannel.SMS:
                success = await self._send_sms_notification(
                    user_id, rendered['body']
                )
            
            elif channel == NotificationChannel.WEBHOOK:
                success = await self._send_webhook_notification(
                    user_id, alert, rendered
                )
            
            elif channel == NotificationChannel.IN_APP:
                success = await self._send_in_app_notification(
                    user_id, alert, rendered
                )
            
            # Update record status
            record.status = NotificationStatus.SENT if success else NotificationStatus.FAILED
            
            # Store record
            self.notification_history.append(record)
            
            # Update analytics
            self._update_delivery_analytics(channel, success)
            
            # Persist to database
            await self._persist_notification_log(record)
            
            if success:
                logger.info(f"Notification sent successfully: {notification_id}")
                return notification_id
            else:
                logger.error(f"Failed to send notification: {notification_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return None
    
    async def _get_or_create_template(self, channel: NotificationChannel, 
                                    priority: AlertPriority, 
                                    template_id: Optional[str] = None) -> NotificationTemplate:
        """Get existing template or create default one"""
        if template_id and template_id in self.templates:
            return self.templates[template_id]
        
        # Create default template
        default_templates = {
            NotificationChannel.EMAIL: {
                'subject': 'Stock Alert: {{ alert_name }}',
                'body': '''
                <h2>Stock Alert Triggered</h2>
                <p><strong>Alert:</strong> {{ alert_name }}</p>
                <p><strong>Description:</strong> {{ alert_description }}</p>
                <p><strong>Stock Code:</strong> {{ stock_code }}</p>
                <p><strong>Priority:</strong> {{ priority }}</p>
                <p><strong>Triggered At:</strong> {{ triggered_at }}</p>
                '''
            },
            NotificationChannel.SMS: {
                'subject': 'Alert: {{ alert_name }}',
                'body': 'Alert: {{ alert_name }} for {{ stock_code }}. Priority: {{ priority }}. Time: {{ triggered_at }}'
            },
            NotificationChannel.IN_APP: {
                'subject': '{{ alert_name }}',
                'body': '{{ alert_description }} ({{ stock_code }})'
            },
            NotificationChannel.WEBHOOK: {
                'subject': 'Stock Alert',
                'body': '{{ alert_name }}: {{ alert_description }}'
            }
        }
        
        template_data = default_templates.get(channel, default_templates[NotificationChannel.IN_APP])
        
        template = NotificationTemplate(
            id=f"default_{channel.value}_{priority.value}",
            name=f"Default {channel.value} template",
            channel=channel,
            subject_template=template_data['subject'],
            body_template=template_data['body'],
            priority=priority
        )
        
        self.templates[template.id] = template
        return template
    
    async def _send_email_notification(self, user_id: str, subject: str, body: str) -> bool:
        """Send email notification"""
        if not self.email_provider:
            logger.warning("Email provider not configured")
            return False
        
        # Get user email (this would typically come from user database)
        user_email = f"{user_id}@example.com"  # Placeholder
        
        return await self.email_provider.send_email(user_email, subject, body, is_html=True)
    
    async def _send_sms_notification(self, user_id: str, message: str) -> bool:
        """Send SMS notification"""
        if not self.sms_provider:
            logger.warning("SMS provider not configured")
            return False
        
        # Get user phone number (this would typically come from user database)
        user_phone = f"+1234567890"  # Placeholder
        
        return await self.sms_provider.send_sms(user_phone, message)
    
    async def _send_webhook_notification(self, user_id: str, alert: Alert, 
                                       rendered: Dict[str, str]) -> bool:
        """Send webhook notification"""
        if not self.webhook_provider:
            logger.warning("Webhook provider not configured")
            return False
        
        # Get user webhook URL (this would typically come from user settings)
        webhook_url = "https://example.com/webhook"  # Placeholder
        
        payload = {
            'alert_id': alert.id,
            'alert_name': alert.name,
            'stock_code': alert.stock_code,
            'priority': alert.priority.value,
            'triggered_at': alert.last_triggered.isoformat() if alert.last_triggered else None,
            'user_id': user_id,
            'subject': rendered['subject'],
            'body': rendered['body']
        }
        
        return await self.webhook_provider.send_webhook(webhook_url, payload)
    
    async def _send_in_app_notification(self, user_id: str, alert: Alert, 
                                      rendered: Dict[str, str]) -> bool:
        """Send in-app notification"""
        try:
            if user_id not in self.in_app_notifications:
                self.in_app_notifications[user_id] = []
            
            notification = {
                'id': f"inapp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'alert_id': alert.id,
                'subject': rendered['subject'],
                'body': rendered['body'],
                'priority': alert.priority.value,
                'created_at': datetime.now().isoformat(),
                'read': False
            }
            
            self.in_app_notifications[user_id].append(notification)
            
            # Keep only last 100 notifications per user
            if len(self.in_app_notifications[user_id]) > 100:
                self.in_app_notifications[user_id] = self.in_app_notifications[user_id][-100:]
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending in-app notification: {e}")
            return False
    
    async def get_in_app_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict[str, Any]]:
        """Get in-app notifications for a user"""
        notifications = self.in_app_notifications.get(user_id, [])
        
        if unread_only:
            notifications = [n for n in notifications if not n['read']]
        
        return notifications
    
    async def mark_notification_read(self, user_id: str, notification_id: str) -> bool:
        """Mark an in-app notification as read"""
        if user_id not in self.in_app_notifications:
            return False
        
        for notification in self.in_app_notifications[user_id]:
            if notification['id'] == notification_id:
                notification['read'] = True
                return True
        
        return False
    
    def _update_delivery_analytics(self, channel: NotificationChannel, success: bool) -> None:
        """Update delivery analytics"""
        channel_key = f"{channel.value}_total"
        success_key = f"{channel.value}_success"
        
        if channel_key not in self.delivery_analytics:
            self.delivery_analytics[channel_key] = 0
        if success_key not in self.delivery_analytics:
            self.delivery_analytics[success_key] = 0
        
        self.delivery_analytics[channel_key] += 1
        if success:
            self.delivery_analytics[success_key] += 1
    
    async def _persist_notification_log(self, record: NotificationRecord) -> None:
        """Persist notification log to database"""
        try:
            log_entry = NotificationLog(
                id=record.id,
                user_id=record.user_id,
                alert_id=record.alert_id,
                channel=record.channel.value,
                status=record.status.value,
                sent_at=record.sent_at,
                delivered_at=record.delivered_at,
                error_message=record.error_message,
                notification_metadata=record.metadata
            )
            
            self.db_session.add(log_entry)
            self.db_session.commit()
            
        except Exception as e:
            logger.error(f"Error persisting notification log: {e}")
            self.db_session.rollback()
    
    async def get_delivery_analytics(self) -> Dict[str, Any]:
        """Get notification delivery analytics"""
        analytics = self.delivery_analytics.copy()
        
        # Calculate success rates
        for channel in NotificationChannel:
            total_key = f"{channel.value}_total"
            success_key = f"{channel.value}_success"
            
            if total_key in analytics and analytics[total_key] > 0:
                success_rate = (analytics.get(success_key, 0) / analytics[total_key]) * 100
                analytics[f"{channel.value}_success_rate"] = round(success_rate, 2)
        
        return analytics
    
    async def get_notification_history(self, user_id: Optional[str] = None, 
                                     limit: int = 100) -> List[NotificationRecord]:
        """Get notification history"""
        history = self.notification_history
        
        if user_id:
            history = [record for record in history if record.user_id == user_id]
        
        return history[-limit:]