#!/usr/bin/env python3
"""
System Degradation Strategy Demo

This script demonstrates the system degradation strategy functionality,
including automatic degradation triggers, manual controls, and recovery mechanisms.
"""

import asyncio
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any

from stock_analysis_system.core.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
from stock_analysis_system.core.degradation_strategy import (
    DegradationStrategy, DegradationLevel, DegradationTrigger, 
    DegradationRule, ServiceConfig, ServicePriority,
    initialize_degradation_strategy
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DegradationDemo:
    """Demo class for system degradation strategy"""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.degradation_strategy = initialize_degradation_strategy(
            error_handler=self.error_handler,
            enable_auto_degradation=True
        )
        
        # Register custom degradation actions
        self._register_degradation_actions()
        
        # Demo state
        self.demo_metrics = {
            'data_collection_frequency': 100,  # percentage
            'analysis_complexity': 100,        # percentage
            'api_features': 100,              # percentage
            'cache_size': 100,                # percentage
            'visualization_quality': 100      # percentage
        }
        
        logger.info("DegradationDemo initialized")
    
    def _register_degradation_actions(self):
        """Register custom degradation actions"""
        
        # Data collection actions
        async def reduce_frequency_25():
            self.demo_metrics['data_collection_frequency'] = 75
            logger.info("🔽 Reduced data collection frequency to 75%")
        
        async def reduce_frequency_50():
            self.demo_metrics['data_collection_frequency'] = 50
            logger.info("🔽 Reduced data collection frequency to 50%")
        
        async def minimal_collection():
            self.demo_metrics['data_collection_frequency'] = 25
            logger.info("🔽 Switched to minimal data collection (25%)")
        
        async def disable_collection():
            self.demo_metrics['data_collection_frequency'] = 0
            logger.info("🛑 Disabled data collection")
        
        # Analysis actions
        async def disable_complex_analysis():
            self.demo_metrics['analysis_complexity'] = 60
            logger.info("🔽 Disabled complex analysis features (60%)")
        
        async def basic_analysis_only():
            self.demo_metrics['analysis_complexity'] = 30
            logger.info("🔽 Basic analysis only (30%)")
        
        async def disable_analysis():
            self.demo_metrics['analysis_complexity'] = 0
            logger.info("🛑 Disabled analysis")
        
        # API actions
        async def reduce_api_features():
            self.demo_metrics['api_features'] = 70
            logger.info("🔽 Reduced API features (70%)")
        
        async def basic_api_only():
            self.demo_metrics['api_features'] = 40
            logger.info("🔽 Basic API only (40%)")
        
        async def minimal_api():
            self.demo_metrics['api_features'] = 20
            logger.info("🔽 Minimal API (20%)")
        
        async def emergency_api_only():
            self.demo_metrics['api_features'] = 10
            logger.info("🚨 Emergency API only (10%)")
        
        # Cache actions
        async def reduce_cache_size():
            self.demo_metrics['cache_size'] = 70
            logger.info("🔽 Reduced cache size (70%)")
        
        async def clear_old_cache():
            self.demo_metrics['cache_size'] = 50
            logger.info("🔽 Cleared old cache entries (50%)")
        
        async def minimal_cache():
            self.demo_metrics['cache_size'] = 25
            logger.info("🔽 Minimal cache (25%)")
        
        async def disable_cache():
            self.demo_metrics['cache_size'] = 0
            logger.info("🛑 Disabled cache")
        
        # Visualization actions
        async def reduce_chart_complexity():
            self.demo_metrics['visualization_quality'] = 70
            logger.info("🔽 Reduced chart complexity (70%)")
        
        async def basic_charts_only():
            self.demo_metrics['visualization_quality'] = 40
            logger.info("🔽 Basic charts only (40%)")
        
        async def minimal_visualization():
            self.demo_metrics['visualization_quality'] = 20
            logger.info("🔽 Minimal visualization (20%)")
        
        async def disable_visualization():
            self.demo_metrics['visualization_quality'] = 0
            logger.info("🛑 Disabled visualization")
        
        # Recovery actions
        async def recover_reduce_frequency_25():
            self.demo_metrics['data_collection_frequency'] = 100
            logger.info("🔄 Restored data collection frequency to 100%")
        
        async def recover_disable_complex_analysis():
            self.demo_metrics['analysis_complexity'] = 100
            logger.info("🔄 Restored analysis complexity to 100%")
        
        async def recover_reduce_api_features():
            self.demo_metrics['api_features'] = 100
            logger.info("🔄 Restored API features to 100%")
        
        async def recover_clear_cache():
            self.demo_metrics['cache_size'] = 100
            logger.info("🔄 Restored cache to 100%")
        
        # Register all actions
        actions = {
            'reduce_frequency_25': reduce_frequency_25,
            'reduce_frequency_50': reduce_frequency_50,
            'minimal_collection': minimal_collection,
            'disable_collection': disable_collection,
            'disable_complex_analysis': disable_complex_analysis,
            'basic_analysis_only': basic_analysis_only,
            'disable_analysis': disable_analysis,
            'reduce_api_features': reduce_api_features,
            'basic_api_only': basic_api_only,
            'minimal_api': minimal_api,
            'emergency_api_only': emergency_api_only,
            'reduce_cache_size': reduce_cache_size,
            'clear_old_cache': clear_old_cache,
            'minimal_cache': minimal_cache,
            'disable_cache': disable_cache,
            'reduce_chart_complexity': reduce_chart_complexity,
            'basic_charts_only': basic_charts_only,
            'minimal_visualization': minimal_visualization,
            'disable_visualization': disable_visualization,
            'recover_reduce_frequency_25': recover_reduce_frequency_25,
            'recover_disable_complex_analysis': recover_disable_complex_analysis,
            'recover_reduce_api_features': recover_reduce_api_features,
            'recover_clear_cache': recover_clear_cache
        }
        
        for name, action in actions.items():
            self.degradation_strategy.register_degradation_action(name, action)
    
    def print_system_status(self):
        """Print current system status"""
        status = self.degradation_strategy.get_system_status()
        
        print("\n" + "="*60)
        print("📊 SYSTEM STATUS")
        print("="*60)
        print(f"Current Degradation Level: {status['current_level'].upper()}")
        print(f"Active Degradations: {len(status['active_degradations'])}")
        print(f"Degraded Services: {', '.join(status['active_degradations']) if status['active_degradations'] else 'None'}")
        print(f"Monitoring Enabled: {status['monitoring_enabled']}")
        
        print("\n📈 SERVICE METRICS:")
        for metric, value in self.demo_metrics.items():
            status_icon = "🟢" if value >= 80 else "🟡" if value >= 50 else "🔴"
            print(f"  {status_icon} {metric.replace('_', ' ').title()}: {value}%")
        
        print(f"\n📊 STATISTICS:")
        stats = status['statistics']
        print(f"  Total Degradations: {stats['total_degradations']}")
        print(f"  Auto Degradations: {stats['auto_degradations']}")
        print(f"  Manual Degradations: {stats['manual_degradations']}")
        print(f"  Successful Recoveries: {stats['successful_recoveries']}")
        
        if status['active_events']:
            print(f"\n🚨 ACTIVE EVENTS:")
            for event in status['active_events']:
                print(f"  - {event['event_id']}: {event['trigger']} -> {event['level']}")
                print(f"    Services: {', '.join(event['affected_services'])}")
                print(f"    Time: {event['timestamp']}")
    
    async def simulate_high_error_rate(self):
        """Simulate high error rate scenario"""
        print("\n🔥 SIMULATING HIGH ERROR RATE SCENARIO")
        print("-" * 50)
        
        # Generate multiple errors to trigger degradation
        for i in range(15):
            try:
                # Simulate various types of errors
                if i % 3 == 0:
                    raise ConnectionError("Database connection failed")
                elif i % 3 == 1:
                    raise TimeoutError("API request timeout")
                else:
                    raise ValueError("Data validation failed")
            except Exception as e:
                self.error_handler.handle_error(e, custom_message=f"Simulated error {i+1}")
            
            # Update error rate metric
            self.degradation_strategy.metrics.update_metric('error_rate', (i + 1) / 10)
            
            await asyncio.sleep(0.1)
        
        print(f"Generated 15 errors to simulate high error rate")
        
        # Wait for monitoring to detect and respond
        await asyncio.sleep(2)
        
        self.print_system_status()
    
    async def simulate_slow_response_time(self):
        """Simulate slow response time scenario"""
        print("\n🐌 SIMULATING SLOW RESPONSE TIME SCENARIO")
        print("-" * 50)
        
        # Simulate increasing response times
        for i in range(10):
            response_time = 2.0 + (i * 0.5)  # Gradually increase from 2s to 6.5s
            self.degradation_strategy.metrics.update_metric('response_times', response_time)
            print(f"Response time: {response_time:.1f}s")
            await asyncio.sleep(0.5)
        
        # Wait for monitoring to detect and respond
        await asyncio.sleep(2)
        
        self.print_system_status()
    
    async def simulate_high_memory_usage(self):
        """Simulate high memory usage scenario"""
        print("\n💾 SIMULATING HIGH MEMORY USAGE SCENARIO")
        print("-" * 50)
        
        # Simulate increasing memory usage
        for i in range(10):
            memory_usage = 0.7 + (i * 0.02)  # Gradually increase from 70% to 88%
            self.degradation_strategy.metrics.update_metric('memory_usage', memory_usage)
            print(f"Memory usage: {memory_usage*100:.1f}%")
            await asyncio.sleep(0.5)
        
        # Wait for monitoring to detect and respond
        await asyncio.sleep(2)
        
        self.print_system_status()
    
    async def simulate_database_issues(self):
        """Simulate database issues scenario"""
        print("\n🗄️ SIMULATING DATABASE ISSUES SCENARIO")
        print("-" * 50)
        
        # Generate database-related errors
        for i in range(10):
            try:
                if i % 2 == 0:
                    raise Exception("DatabaseError: Connection pool exhausted")
                else:
                    raise Exception("OperationalError: Database timeout")
            except Exception as e:
                self.error_handler.handle_error(e, custom_message=f"Database error {i+1}")
            
            await asyncio.sleep(0.2)
        
        # Wait for monitoring to detect and respond
        await asyncio.sleep(2)
        
        self.print_system_status()
    
    async def test_manual_degradation(self):
        """Test manual degradation controls"""
        print("\n🎛️ TESTING MANUAL DEGRADATION CONTROLS")
        print("-" * 50)
        
        # Manual degradation
        event_id = await self.degradation_strategy.manual_degradation(
            level=DegradationLevel.MODERATE,
            services=["data_collection", "analysis"],
            reason="Manual testing"
        )
        
        print(f"Triggered manual degradation: {event_id}")
        
        await asyncio.sleep(1)
        self.print_system_status()
        
        # Wait a bit
        await asyncio.sleep(3)
        
        # Manual recovery
        print("\n🔄 TESTING MANUAL RECOVERY")
        success = await self.degradation_strategy.manual_recovery(event_id)
        print(f"Manual recovery {'successful' if success else 'failed'}")
        
        await asyncio.sleep(1)
        self.print_system_status()
    
    async def test_custom_degradation_rule(self):
        """Test adding custom degradation rules"""
        print("\n⚙️ TESTING CUSTOM DEGRADATION RULE")
        print("-" * 50)
        
        # Add custom rule
        custom_rule = DegradationRule(
            rule_id="demo_custom_rule",
            name="Demo Custom Rule",
            description="Custom rule for demonstration",
            trigger=DegradationTrigger.ERROR_RATE,
            threshold=0.05,  # 5% error rate
            degradation_level=DegradationLevel.LIGHT,
            affected_services=["visualization"],
            actions=["reduce_chart_complexity"],
            recovery_threshold=0.02,
            cooldown_period=60
        )
        
        self.degradation_strategy.add_rule(custom_rule)
        print(f"Added custom rule: {custom_rule.name}")
        
        # Trigger the rule
        for i in range(8):
            try:
                raise ValueError(f"Custom test error {i+1}")
            except Exception as e:
                self.error_handler.handle_error(e)
            
            self.degradation_strategy.metrics.update_metric('error_rate', (i + 1) / 100)
            await asyncio.sleep(0.1)
        
        # Wait for monitoring
        await asyncio.sleep(2)
        
        self.print_system_status()
    
    def print_degradation_history(self):
        """Print degradation history"""
        history = self.degradation_strategy.get_degradation_history(hours=1)
        
        print("\n📜 DEGRADATION HISTORY (Last Hour)")
        print("-" * 50)
        
        if not history:
            print("No degradation events in the last hour")
            return
        
        for event in history:
            status = "✅ Resolved" if event['resolved'] else "🔄 Active"
            trigger_type = "🤖 Auto" if event['auto_triggered'] else "👤 Manual"
            
            print(f"\n{status} {trigger_type} - {event['event_id']}")
            print(f"  Time: {event['timestamp']}")
            print(f"  Trigger: {event['trigger']} -> {event['level']}")
            print(f"  Services: {', '.join(event['affected_services'])}")
            print(f"  Actions: {', '.join(event['actions_taken'])}")
            
            if event['resolved']:
                print(f"  Resolved: {event['resolution_time']}")
    
    async def run_comprehensive_demo(self):
        """Run comprehensive degradation strategy demo"""
        print("🚀 STARTING COMPREHENSIVE DEGRADATION STRATEGY DEMO")
        print("="*60)
        
        # Start monitoring
        await self.degradation_strategy.start_monitoring()
        
        try:
            # Initial status
            print("\n1️⃣ INITIAL SYSTEM STATUS")
            self.print_system_status()
            
            await asyncio.sleep(2)
            
            # Test high error rate scenario
            print("\n2️⃣ HIGH ERROR RATE SCENARIO")
            await self.simulate_high_error_rate()
            
            await asyncio.sleep(3)
            
            # Test slow response time scenario
            print("\n3️⃣ SLOW RESPONSE TIME SCENARIO")
            await self.simulate_slow_response_time()
            
            await asyncio.sleep(3)
            
            # Test high memory usage scenario
            print("\n4️⃣ HIGH MEMORY USAGE SCENARIO")
            await self.simulate_high_memory_usage()
            
            await asyncio.sleep(3)
            
            # Test database issues scenario
            print("\n5️⃣ DATABASE ISSUES SCENARIO")
            await self.simulate_database_issues()
            
            await asyncio.sleep(3)
            
            # Test manual controls
            print("\n6️⃣ MANUAL DEGRADATION CONTROLS")
            await self.test_manual_degradation()
            
            await asyncio.sleep(3)
            
            # Test custom rules
            print("\n7️⃣ CUSTOM DEGRADATION RULES")
            await self.test_custom_degradation_rule()
            
            await asyncio.sleep(3)
            
            # Final status and history
            print("\n8️⃣ FINAL STATUS AND HISTORY")
            self.print_system_status()
            self.print_degradation_history()
            
            # Export report
            report_file = f"degradation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.degradation_strategy.export_degradation_report(report_file)
            print(f"\n📄 Degradation report exported to: {report_file}")
            
        finally:
            # Stop monitoring
            await self.degradation_strategy.stop_monitoring()
        
        print("\n✅ DEGRADATION STRATEGY DEMO COMPLETED")
        print("="*60)


async def main():
    """Main demo function"""
    demo = DegradationDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())