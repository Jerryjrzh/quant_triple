#!/usr/bin/env python3
"""
通知系统演示脚本

演示多渠道通知系统的功能，包括邮件、短信、Webhook、应用内通知等。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock

from stock_analysis_system.alerts.notification_system import (
    NotificationSystem, NotificationChannel, NotificationTemplate, 
    NotificationPreference
)
from stock_analysis_system.alerts.alert_engine import (
    Alert, AlertPriority, AlertTrigger, AlertTriggerType, AlertStatus
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NotificationSystemDemo:
    """通知系统演示类"""
    
    def __init__(self):
        # 创建模拟数据库会话
        self.db_session = Mock()
        
        # 创建通知系统
        self.notification_system = NotificationSystem(self.db_session)
        
        # 配置通知提供者
        self._configure_providers()
        
        logger.info("通知系统演示初始化完成")
    
    def _configure_providers(self):
        """配置通知提供者"""
        # 配置邮件提供者（使用测试配置）
        self.notification_system.configure_email_provider(
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            username="test@example.com",
            password="test_password"
        )
        
        # 配置短信提供者
        self.notification_system.configure_sms_provider(
            api_key="test_sms_api_key",
            api_url="https://api.sms.example.com/send"
        )
        
        # 配置Webhook提供者
        self.notification_system.configure_webhook_provider()
        
        logger.info("通知提供者配置完成")
    
    async def demo_template_management(self):
        """演示模板管理功能"""
        logger.info("=== 演示模板管理功能 ===")
        
        # 创建邮件模板
        email_template = NotificationTemplate(
            id="price_alert_email",
            name="股价告警邮件模板",
            channel=NotificationChannel.EMAIL,
            subject_template="【股价告警】{{ stock_code }} - {{ alert_name }}",
            body_template="""
            <h2>股价告警通知</h2>
            <p><strong>股票代码：</strong>{{ stock_code }}</p>
            <p><strong>告警名称：</strong>{{ alert_name }}</p>
            <p><strong>告警描述：</strong>{{ alert_description }}</p>
            <p><strong>优先级：</strong>{{ priority }}</p>
            <p><strong>触发时间：</strong>{{ triggered_at }}</p>
            <p>请及时关注市场动态。</p>
            """,
            priority=AlertPriority.HIGH
        )
        
        template_id = await self.notification_system.create_template(email_template)
        logger.info(f"创建邮件模板成功，ID: {template_id}")
        
        # 创建短信模板
        sms_template = NotificationTemplate(
            id="price_alert_sms",
            name="股价告警短信模板",
            channel=NotificationChannel.SMS,
            subject_template="股价告警",
            body_template="【股价告警】{{ stock_code }} {{ alert_name }}，优先级：{{ priority }}，时间：{{ triggered_at }}",
            priority=AlertPriority.HIGH
        )
        
        await self.notification_system.create_template(sms_template)
        logger.info("创建短信模板成功")
        
        # 创建应用内通知模板
        in_app_template = NotificationTemplate(
            id="price_alert_in_app",
            name="股价告警应用内模板",
            channel=NotificationChannel.IN_APP,
            subject_template="{{ alert_name }}",
            body_template="{{ stock_code }} {{ alert_description }}",
            priority=AlertPriority.HIGH
        )
        
        await self.notification_system.create_template(in_app_template)
        logger.info("创建应用内通知模板成功")
        
        # 测试模板渲染
        context = {
            'stock_code': '000001',
            'alert_name': '价格突破告警',
            'alert_description': '股价突破重要阻力位',
            'priority': 'HIGH',
            'triggered_at': '2024-01-15 14:30:00'
        }
        
        rendered = email_template.render(context)
        logger.info(f"邮件模板渲染结果：")
        logger.info(f"  主题: {rendered['subject']}")
        logger.info(f"  内容: {rendered['body'][:100]}...")
        
        return {
            'email_template_id': template_id,
            'templates_created': 3
        }
    
    async def demo_user_preferences(self):
        """演示用户偏好设置功能"""
        logger.info("=== 演示用户偏好设置功能 ===")
        
        # 用户1：只接收高优先级的邮件和应用内通知
        user1_preferences = [
            NotificationPreference(
                user_id="user001",
                channel=NotificationChannel.EMAIL,
                enabled=True,
                priority_filter=[AlertPriority.HIGH, AlertPriority.CRITICAL],
                quiet_hours_start="22:00",
                quiet_hours_end="08:00"
            ),
            NotificationPreference(
                user_id="user001",
                channel=NotificationChannel.IN_APP,
                enabled=True,
                priority_filter=[AlertPriority.MEDIUM, AlertPriority.HIGH, AlertPriority.CRITICAL]
            )
        ]
        
        await self.notification_system.set_user_preferences("user001", user1_preferences)
        logger.info("设置用户001偏好：邮件（高优先级，有静默时间）+ 应用内通知（中高优先级）")
        
        # 用户2：接收所有渠道的通知
        user2_preferences = [
            NotificationPreference(
                user_id="user002",
                channel=NotificationChannel.EMAIL,
                enabled=True
            ),
            NotificationPreference(
                user_id="user002",
                channel=NotificationChannel.SMS,
                enabled=True,
                priority_filter=[AlertPriority.HIGH, AlertPriority.CRITICAL]
            ),
            NotificationPreference(
                user_id="user002",
                channel=NotificationChannel.IN_APP,
                enabled=True
            ),
            NotificationPreference(
                user_id="user002",
                channel=NotificationChannel.WEBHOOK,
                enabled=True,
                priority_filter=[AlertPriority.CRITICAL]
            )
        ]
        
        await self.notification_system.set_user_preferences("user002", user2_preferences)
        logger.info("设置用户002偏好：全渠道通知（不同优先级过滤）")
        
        # 用户3：只接收应用内通知
        user3_preferences = [
            NotificationPreference(
                user_id="user003",
                channel=NotificationChannel.IN_APP,
                enabled=True
            )
        ]
        
        await self.notification_system.set_user_preferences("user003", user3_preferences)
        logger.info("设置用户003偏好：仅应用内通知")
        
        return {
            'users_configured': 3,
            'total_preferences': len(user1_preferences) + len(user2_preferences) + len(user3_preferences)
        }
    
    def _create_sample_alert(self, alert_id: str, name: str, priority: AlertPriority, 
                           stock_code: str = "000001") -> Alert:
        """创建示例告警"""
        trigger = AlertTrigger(
            trigger_type=AlertTriggerType.CONDITION_BASED,
            conditions=[]
        )
        
        return Alert(
            id=alert_id,
            name=name,
            description=f"{name}的详细描述",
            stock_code=stock_code,
            trigger=trigger,
            priority=priority,
            status=AlertStatus.TRIGGERED,
            last_triggered=datetime.now()
        )
    
    async def demo_notification_sending(self):
        """演示通知发送功能"""
        logger.info("=== 演示通知发送功能 ===")
        
        # 创建不同优先级的告警
        high_priority_alert = self._create_sample_alert(
            "alert_high_001", "价格突破告警", AlertPriority.HIGH, "000001"
        )
        
        medium_priority_alert = self._create_sample_alert(
            "alert_medium_001", "成交量异常告警", AlertPriority.MEDIUM, "000002"
        )
        
        critical_alert = self._create_sample_alert(
            "alert_critical_001", "系统故障告警", AlertPriority.CRITICAL, "SYSTEM"
        )
        
        # 模拟发送通知（由于没有真实的SMTP/SMS服务，这里会失败，但会记录尝试）
        results = {}
        
        # 为用户001发送高优先级告警
        logger.info("为用户001发送高优先级告警...")
        user001_notifications = await self.notification_system.send_alert_notification(
            high_priority_alert, "user001"
        )
        results['user001_high'] = len(user001_notifications)
        logger.info(f"用户001收到 {len(user001_notifications)} 个通知")
        
        # 为用户001发送中优先级告警（应该只有应用内通知）
        logger.info("为用户001发送中优先级告警...")
        user001_medium_notifications = await self.notification_system.send_alert_notification(
            medium_priority_alert, "user001"
        )
        results['user001_medium'] = len(user001_medium_notifications)
        logger.info(f"用户001收到 {len(user001_medium_notifications)} 个通知")
        
        # 为用户002发送关键告警
        logger.info("为用户002发送关键告警...")
        user002_notifications = await self.notification_system.send_alert_notification(
            critical_alert, "user002"
        )
        results['user002_critical'] = len(user002_notifications)
        logger.info(f"用户002收到 {len(user002_notifications)} 个通知")
        
        # 为用户003发送告警
        logger.info("为用户003发送告警...")
        user003_notifications = await self.notification_system.send_alert_notification(
            high_priority_alert, "user003"
        )
        results['user003_high'] = len(user003_notifications)
        logger.info(f"用户003收到 {len(user003_notifications)} 个通知")
        
        return results
    
    async def demo_in_app_notifications(self):
        """演示应用内通知功能"""
        logger.info("=== 演示应用内通知功能 ===")
        
        # 获取用户001的应用内通知
        user001_notifications = await self.notification_system.get_in_app_notifications("user001")
        logger.info(f"用户001的应用内通知数量: {len(user001_notifications)}")
        
        if user001_notifications:
            latest_notification = user001_notifications[0]
            logger.info(f"最新通知: {latest_notification['subject']}")
            logger.info(f"通知内容: {latest_notification['body']}")
            logger.info(f"是否已读: {latest_notification['read']}")
            
            # 标记为已读
            await self.notification_system.mark_notification_read(
                "user001", latest_notification['id']
            )
            logger.info("已标记最新通知为已读")
        
        # 获取用户002的应用内通知
        user002_notifications = await self.notification_system.get_in_app_notifications("user002")
        logger.info(f"用户002的应用内通知数量: {len(user002_notifications)}")
        
        # 获取用户003的应用内通知
        user003_notifications = await self.notification_system.get_in_app_notifications("user003")
        logger.info(f"用户003的应用内通知数量: {len(user003_notifications)}")
        
        return {
            'user001_notifications': len(user001_notifications),
            'user002_notifications': len(user002_notifications),
            'user003_notifications': len(user003_notifications)
        }
    
    async def demo_analytics_and_history(self):
        """演示分析和历史记录功能"""
        logger.info("=== 演示分析和历史记录功能 ===")
        
        # 获取投递分析
        analytics = await self.notification_system.get_delivery_analytics()
        logger.info("投递分析统计:")
        for key, value in analytics.items():
            logger.info(f"  {key}: {value}")
        
        # 获取通知历史记录
        history = await self.notification_system.get_notification_history(limit=10)
        logger.info(f"通知历史记录数量: {len(history)}")
        
        if history:
            logger.info("最近的通知记录:")
            for record in history[-3:]:  # 显示最近3条
                logger.info(f"  ID: {record.id}, 用户: {record.user_id}, "
                          f"渠道: {record.channel.value}, 状态: {record.status.value}")
        
        # 获取特定用户的历史记录
        user001_history = await self.notification_system.get_notification_history(
            user_id="user001", limit=5
        )
        logger.info(f"用户001的通知历史记录数量: {len(user001_history)}")
        
        return {
            'total_analytics_keys': len(analytics),
            'total_history_records': len(history),
            'user001_history_records': len(user001_history)
        }
    
    async def demo_template_updates(self):
        """演示模板更新功能"""
        logger.info("=== 演示模板更新功能 ===")
        
        # 更新邮件模板
        updates = {
            'name': '更新的股价告警邮件模板',
            'subject_template': '【紧急告警】{{ stock_code }} - {{ alert_name }}'
        }
        
        result = await self.notification_system.update_template("price_alert_email", updates)
        logger.info(f"更新邮件模板结果: {result}")
        
        # 验证更新
        if "price_alert_email" in self.notification_system.templates:
            template = self.notification_system.templates["price_alert_email"]
            logger.info(f"更新后的模板名称: {template.name}")
            logger.info(f"更新后的主题模板: {template.subject_template}")
        
        return {'template_updated': result}
    
    async def run_comprehensive_demo(self):
        """运行完整演示"""
        logger.info("开始通知系统完整演示")
        logger.info("=" * 60)
        
        results = {}
        
        try:
            # 1. 模板管理演示
            template_results = await self.demo_template_management()
            results.update(template_results)
            
            # 2. 用户偏好设置演示
            preference_results = await self.demo_user_preferences()
            results.update(preference_results)
            
            # 3. 通知发送演示
            sending_results = await self.demo_notification_sending()
            results.update(sending_results)
            
            # 4. 应用内通知演示
            in_app_results = await self.demo_in_app_notifications()
            results.update(in_app_results)
            
            # 5. 分析和历史记录演示
            analytics_results = await self.demo_analytics_and_history()
            results.update(analytics_results)
            
            # 6. 模板更新演示
            update_results = await self.demo_template_updates()
            results.update(update_results)
            
            # 总结
            logger.info("=" * 60)
            logger.info("通知系统演示完成")
            logger.info("演示结果总结:")
            for key, value in results.items():
                logger.info(f"  {key}: {value}")
            
            return results
            
        except Exception as e:
            logger.error(f"演示过程中发生错误: {e}")
            raise


async def main():
    """主函数"""
    demo = NotificationSystemDemo()
    
    try:
        results = await demo.run_comprehensive_demo()
        
        print("\n" + "="*60)
        print("通知系统演示成功完成！")
        print("="*60)
        
        # 显示关键指标
        print(f"📧 邮件模板创建: {results.get('templates_created', 0)} 个")
        print(f"👥 用户偏好配置: {results.get('users_configured', 0)} 个用户")
        print(f"📱 应用内通知: {results.get('user001_notifications', 0) + results.get('user002_notifications', 0) + results.get('user003_notifications', 0)} 条")
        print(f"📊 分析指标: {results.get('total_analytics_keys', 0)} 个")
        print(f"📝 历史记录: {results.get('total_history_records', 0)} 条")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        return False


if __name__ == "__main__":
    # 运行演示
    success = asyncio.run(main())
    exit(0 if success else 1)