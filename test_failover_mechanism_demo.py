#!/usr/bin/env python3
"""
Failover Mechanism Demo

This script demonstrates the failover mechanism functionality,
including resource health monitoring, automatic failover, and recovery.
"""

import asyncio
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any

from stock_analysis_system.core.error_handler import ErrorHandler
from stock_analysis_system.core.failover_mechanism import (
    FailoverManager, ResourceType, ResourceStatus, FailoverStrategy,
    ResourceConfig, HealthCheckResult, initialize_failover_manager
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FailoverDemo:
    """Demo class for failover mechanism"""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.failover_manager = initialize_failover_manager(
            error_handler=self.error_handler,
            default_strategy=FailoverStrategy.PRIORITY_BASED
        )
        
        # Demo state
        self.demo_connections = {
            'primary_db': {'status': 'connected', 'response_time': 0.1},
            'backup_db': {'status': 'standby', 'response_time': 0.15},
            'emergency_db': {'status': 'standby', 'response_time': 0.3},
            'tushare_api': {'status': 'connected', 'response_time': 0.2},
            'akshare_api': {'status': 'standby', 'response_time': 0.25},
            'wind_api': {'status': 'standby', 'response_time': 0.4},
            'redis_primary': {'status': 'connected', 'response_time': 0.05},
            'redis_backup': {'status': 'standby', 'response_time': 0.08},
            'api_server_1': {'status': 'connected', 'response_time': 0.1},
            'api_server_2': {'status': 'standby', 'response_time': 0.12}
        }
        
        # Setup resources and custom health checks
        self._setup_resources()
        self._register_custom_health_checks()
        self._register_failover_handlers()
        
        logger.info("FailoverDemo initialized")
    
    def _setup_resources(self):
        """Setup demo resources"""
        
        # Database resources
        db_resources = [
            ResourceConfig(
                resource_id="primary_db",
                resource_type=ResourceType.DATABASE,
                name="Primary PostgreSQL Database",
                connection_string="postgresql://user:pass@primary-db:5432/stockdb",
                priority=1,
                weight=1.0,
                health_check_interval=10,
                max_failures=2,
                failure_timeout=60,
                metadata={"region": "us-east-1", "instance_type": "db.r5.large"}
            ),
            ResourceConfig(
                resource_id="backup_db",
                resource_type=ResourceType.DATABASE,
                name="Backup PostgreSQL Database",
                connection_string="postgresql://user:pass@backup-db:5432/stockdb",
                priority=2,
                weight=0.8,
                health_check_interval=15,
                max_failures=3,
                failure_timeout=120,
                metadata={"region": "us-west-2", "instance_type": "db.r5.medium"}
            ),
            ResourceConfig(
                resource_id="emergency_db",
                resource_type=ResourceType.DATABASE,
                name="Emergency SQLite Database",
                connection_string="sqlite:///emergency.db",
                priority=3,
                weight=0.5,
                health_check_interval=20,
                max_failures=1,
                failure_timeout=300,
                metadata={"type": "local", "capacity": "limited"}
            )
        ]
        
        # Data source resources
        data_source_resources = [
            ResourceConfig(
                resource_id="tushare_api",
                resource_type=ResourceType.DATA_SOURCE,
                name="Tushare API",
                connection_string="https://api.tushare.pro",
                priority=1,
                weight=1.0,
                health_check_url="https://api.tushare.pro/health",
                health_check_interval=30,
                max_failures=3,
                failure_timeout=180,
                metadata={"quota": 10000, "rate_limit": "200/min"}
            ),
            ResourceConfig(
                resource_id="akshare_api",
                resource_type=ResourceType.DATA_SOURCE,
                name="AKShare API",
                connection_string="https://akshare.akfamily.xyz",
                priority=2,
                weight=0.9,
                health_check_interval=45,
                max_failures=2,
                failure_timeout=120,
                metadata={"quota": 5000, "rate_limit": "100/min"}
            ),
            ResourceConfig(
                resource_id="wind_api",
                resource_type=ResourceType.DATA_SOURCE,
                name="Wind API",
                connection_string="https://api.wind.com.cn",
                priority=3,
                weight=0.7,
                health_check_interval=60,
                max_failures=2,
                failure_timeout=300,
                metadata={"quota": 1000, "rate_limit": "50/min"}
            )
        ]
        
        # Cache resources
        cache_resources = [
            ResourceConfig(
                resource_id="redis_primary",
                resource_type=ResourceType.CACHE,
                name="Primary Redis Cache",
                connection_string="redis://primary-redis:6379/0",
                priority=1,
                weight=1.0,
                health_check_interval=15,
                max_failures=2,
                failure_timeout=60,
                metadata={"memory": "8GB", "cluster": True}
            ),
            ResourceConfig(
                resource_id="redis_backup",
                resource_type=ResourceType.CACHE,
                name="Backup Redis Cache",
                connection_string="redis://backup-redis:6379/0",
                priority=2,
                weight=0.8,
                health_check_interval=20,
                max_failures=3,
                failure_timeout=120,
                metadata={"memory": "4GB", "cluster": False}
            )
        ]
        
        # API endpoint resources
        api_resources = [
            ResourceConfig(
                resource_id="api_server_1",
                resource_type=ResourceType.API_ENDPOINT,
                name="Primary API Server",
                connection_string="https://api1.stockanalysis.com",
                priority=1,
                weight=1.0,
                health_check_url="https://api1.stockanalysis.com/health",
                health_check_interval=20,
                max_failures=3,
                failure_timeout=90,
                metadata={"region": "us-east-1", "capacity": "high"}
            ),
            ResourceConfig(
                resource_id="api_server_2",
                resource_type=ResourceType.API_ENDPOINT,
                name="Backup API Server",
                connection_string="https://api2.stockanalysis.com",
                priority=2,
                weight=0.9,
                health_check_url="https://api2.stockanalysis.com/health",
                health_check_interval=25,
                max_failures=2,
                failure_timeout=120,
                metadata={"region": "us-west-2", "capacity": "medium"}
            )
        ]
        
        # Add all resources to failover manager
        all_resources = db_resources + data_source_resources + cache_resources + api_resources
        for resource in all_resources:
            self.failover_manager.add_resource(resource)
        
        # Set failover strategies
        self.failover_manager.set_failover_strategy(ResourceType.DATABASE, FailoverStrategy.PRIORITY_BASED)
        self.failover_manager.set_failover_strategy(ResourceType.DATA_SOURCE, FailoverStrategy.LOAD_BALANCED)
        self.failover_manager.set_failover_strategy(ResourceType.CACHE, FailoverStrategy.IMMEDIATE)
        self.failover_manager.set_failover_strategy(ResourceType.API_ENDPOINT, FailoverStrategy.PRIORITY_BASED)
    
    def _register_custom_health_checks(self):
        """Register custom health check functions"""
        
        async def database_health_check(config: ResourceConfig) -> HealthCheckResult:
            """Custom database health check"""
            start_time = time.time()
            
            try:
                # Simulate database connection and query
                connection_info = self.demo_connections.get(config.resource_id, {})
                
                if connection_info.get('status') == 'failed':
                    raise Exception("Database connection failed")
                
                # Simulate query execution time
                query_time = connection_info.get('response_time', 0.1)
                await asyncio.sleep(query_time)
                
                # Random failure simulation
                if random.random() < 0.08:  # 8% failure rate
                    raise Exception("Database query timeout")
                
                response_time = time.time() - start_time
                
                # Determine status based on response time
                if response_time > 1.0:
                    status = ResourceStatus.DEGRADED
                elif response_time > 0.5:
                    status = ResourceStatus.DEGRADED
                else:
                    status = ResourceStatus.HEALTHY
                
                return HealthCheckResult(
                    resource_id=config.resource_id,
                    timestamp=datetime.now(),
                    status=status,
                    response_time=response_time,
                    metadata={
                        'connection_pool_size': 20,
                        'active_connections': random.randint(1, 15),
                        'query_success_rate': 0.95 + random.random() * 0.05
                    }
                )
                
            except Exception as e:
                return HealthCheckResult(
                    resource_id=config.resource_id,
                    timestamp=datetime.now(),
                    status=ResourceStatus.FAILED,
                    response_time=time.time() - start_time,
                    error_message=str(e)
                )
        
        async def data_source_health_check(config: ResourceConfig) -> HealthCheckResult:
            """Custom data source health check"""
            start_time = time.time()
            
            try:
                connection_info = self.demo_connections.get(config.resource_id, {})
                
                if connection_info.get('status') == 'failed':
                    raise Exception("API endpoint unreachable")
                
                # Simulate API call
                api_time = connection_info.get('response_time', 0.2)
                await asyncio.sleep(api_time)
                
                # Random failure simulation
                if random.random() < 0.06:  # 6% failure rate
                    raise Exception("API rate limit exceeded")
                
                response_time = time.time() - start_time
                
                status = ResourceStatus.HEALTHY if response_time < 1.0 else ResourceStatus.DEGRADED
                
                return HealthCheckResult(
                    resource_id=config.resource_id,
                    timestamp=datetime.now(),
                    status=status,
                    response_time=response_time,
                    metadata={
                        'api_quota_remaining': random.randint(500, 5000),
                        'rate_limit_remaining': random.randint(50, 200),
                        'last_successful_call': datetime.now().isoformat()
                    }
                )
                
            except Exception as e:
                return HealthCheckResult(
                    resource_id=config.resource_id,
                    timestamp=datetime.now(),
                    status=ResourceStatus.FAILED,
                    response_time=time.time() - start_time,
                    error_message=str(e)
                )
        
        # Register health checks
        for resource_id in ['primary_db', 'backup_db', 'emergency_db']:
            self.failover_manager.health_monitor.register_health_check(
                resource_id, database_health_check
            )
        
        for resource_id in ['tushare_api', 'akshare_api', 'wind_api']:
            self.failover_manager.health_monitor.register_health_check(
                resource_id, data_source_health_check
            )
    
    def _register_failover_handlers(self):
        """Register custom failover handlers"""
        
        async def database_failover_handler(from_resource: str, to_resource: str):
            """Handle database failover"""
            logger.info(f"🔄 Database failover: {from_resource} -> {to_resource}")
            
            # Update demo connection status
            self.demo_connections[from_resource]['status'] = 'failed'
            self.demo_connections[to_resource]['status'] = 'connected'
            
            # Simulate connection pool update
            await asyncio.sleep(0.2)
            logger.info(f"✅ Database connection pool updated to {to_resource}")
        
        async def data_source_failover_handler(from_resource: str, to_resource: str):
            """Handle data source failover"""
            logger.info(f"🔄 Data source failover: {from_resource} -> {to_resource}")
            
            # Update demo connection status
            self.demo_connections[from_resource]['status'] = 'failed'
            self.demo_connections[to_resource]['status'] = 'connected'
            
            # Simulate API client reconfiguration
            await asyncio.sleep(0.1)
            logger.info(f"✅ API client reconfigured to {to_resource}")
        
        def cache_failover_handler(from_resource: str, to_resource: str):
            """Handle cache failover (synchronous)"""
            logger.info(f"🔄 Cache failover: {from_resource} -> {to_resource}")
            
            # Update demo connection status
            self.demo_connections[from_resource]['status'] = 'failed'
            self.demo_connections[to_resource]['status'] = 'connected'
            
            logger.info(f"✅ Cache client switched to {to_resource}")
        
        # Register handlers
        self.failover_manager.register_failover_handler(
            ResourceType.DATABASE, database_failover_handler
        )
        self.failover_manager.register_failover_handler(
            ResourceType.DATA_SOURCE, data_source_failover_handler
        )
        self.failover_manager.register_failover_handler(
            ResourceType.CACHE, cache_failover_handler
        )
    
    def print_system_status(self):
        """Print current system status"""
        stats = self.failover_manager.get_failover_statistics()
        
        print("\n" + "="*60)
        print("🔧 FAILOVER SYSTEM STATUS")
        print("="*60)
        
        print(f"Total Resources: {stats['total_resources']}")
        print(f"Total Failovers: {stats['total_failovers']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Average Failover Time: {stats['average_failover_time']:.3f}s")
        
        print("\n📊 RESOURCE STATUS:")
        for status, count in stats['resource_status_summary'].items():
            status_icon = {
                'healthy': '🟢',
                'degraded': '🟡',
                'failed': '🔴',
                'recovering': '🔄',
                'maintenance': '🔧'
            }.get(status, '❓')
            print(f"  {status_icon} {status.title()}: {count}")
        
        print("\n🎯 ACTIVE RESOURCES:")
        for resource_type, resource_id in stats['active_resources'].items():
            print(f"  {resource_type.replace('_', ' ').title()}: {resource_id}")
        
        print("\n🔗 RESOURCE GROUPS:")
        for resource_type, resources in stats['resource_groups'].items():
            print(f"  {resource_type.replace('_', ' ').title()}: {', '.join(resources)}")
    
    def print_resource_health(self):
        """Print detailed resource health information"""
        print("\n" + "="*60)
        print("🏥 RESOURCE HEALTH STATUS")
        print("="*60)
        
        for resource_id, config in self.failover_manager.get_all_resources().items():
            health = self.failover_manager.health_monitor.get_health_status(resource_id)
            status = self.failover_manager.get_resource_status(resource_id)
            
            status_icon = {
                ResourceStatus.HEALTHY: '🟢',
                ResourceStatus.DEGRADED: '🟡',
                ResourceStatus.FAILED: '🔴',
                ResourceStatus.RECOVERING: '🔄',
                ResourceStatus.MAINTENANCE: '🔧'
            }.get(status, '❓')
            
            print(f"\n{status_icon} {config.name} ({resource_id})")
            print(f"  Type: {config.resource_type.value}")
            print(f"  Priority: {config.priority}")
            print(f"  Status: {status.value if status else 'unknown'}")
            
            if health:
                print(f"  Last Check: {health.timestamp.strftime('%H:%M:%S')}")
                print(f"  Response Time: {health.response_time:.3f}s")
                if health.error_message:
                    print(f"  Error: {health.error_message}")
                if health.metadata:
                    print(f"  Metadata: {health.metadata}")
    
    async def simulate_database_failure(self):
        """Simulate database failure scenario"""
        print("\n💥 SIMULATING DATABASE FAILURE")
        print("-" * 50)
        
        # Mark primary database as failed
        self.demo_connections['primary_db']['status'] = 'failed'
        
        print("Primary database marked as failed...")
        
        # Wait for health monitor to detect failure
        await asyncio.sleep(15)  # Wait for health check cycle
        
        self.print_system_status()
    
    async def simulate_data_source_failure(self):
        """Simulate data source failure scenario"""
        print("\n📡 SIMULATING DATA SOURCE FAILURE")
        print("-" * 50)
        
        # Mark Tushare API as failed
        self.demo_connections['tushare_api']['status'] = 'failed'
        
        print("Tushare API marked as failed...")
        
        # Wait for health monitor to detect failure
        await asyncio.sleep(35)  # Wait for health check cycle
        
        self.print_system_status()
    
    async def simulate_cache_failure(self):
        """Simulate cache failure scenario"""
        print("\n💾 SIMULATING CACHE FAILURE")
        print("-" * 50)
        
        # Mark primary Redis as failed
        self.demo_connections['redis_primary']['status'] = 'failed'
        
        print("Primary Redis cache marked as failed...")
        
        # Wait for health monitor to detect failure
        await asyncio.sleep(20)  # Wait for health check cycle
        
        self.print_system_status()
    
    async def simulate_cascade_failure(self):
        """Simulate cascade failure scenario"""
        print("\n⛓️ SIMULATING CASCADE FAILURE")
        print("-" * 50)
        
        # Mark multiple resources as failed
        self.demo_connections['primary_db']['status'] = 'failed'
        self.demo_connections['tushare_api']['status'] = 'failed'
        self.demo_connections['redis_primary']['status'] = 'failed'
        
        print("Multiple resources marked as failed...")
        
        # Wait for health monitors to detect failures
        await asyncio.sleep(40)  # Wait for health check cycles
        
        self.print_system_status()
    
    async def test_manual_failover(self):
        """Test manual failover trigger"""
        print("\n🎛️ TESTING MANUAL FAILOVER")
        print("-" * 50)
        
        # Trigger manual failover for database
        success = await self.failover_manager.trigger_failover(
            resource_type=ResourceType.DATABASE,
            failed_resource="primary_db",
            reason="Manual failover test",
            strategy=FailoverStrategy.PRIORITY_BASED
        )
        
        print(f"Manual failover {'successful' if success else 'failed'}")
        
        await asyncio.sleep(2)
        self.print_system_status()
    
    async def simulate_resource_recovery(self):
        """Simulate resource recovery"""
        print("\n🔄 SIMULATING RESOURCE RECOVERY")
        print("-" * 50)
        
        # Restore failed resources
        for resource_id in self.demo_connections:
            if self.demo_connections[resource_id]['status'] == 'failed':
                self.demo_connections[resource_id]['status'] = 'connected'
                print(f"Restored {resource_id}")
        
        # Wait for health monitors to detect recovery
        await asyncio.sleep(30)
        
        self.print_system_status()
    
    def print_failover_history(self):
        """Print failover history"""
        history = self.failover_manager.get_failover_history(hours=1)
        
        print("\n📜 FAILOVER HISTORY (Last Hour)")
        print("-" * 50)
        
        if not history:
            print("No failover events in the last hour")
            return
        
        for event in history:
            status_icon = "✅" if event['success'] else "❌"
            
            print(f"\n{status_icon} {event['event_id']}")
            print(f"  Time: {event['timestamp']}")
            print(f"  Type: {event['resource_type']}")
            print(f"  From: {event['from_resource']} -> To: {event['to_resource']}")
            print(f"  Reason: {event['reason']}")
            print(f"  Strategy: {event['strategy']}")
            print(f"  Response Time: {event['response_time']:.3f}s")
    
    async def run_comprehensive_demo(self):
        """Run comprehensive failover mechanism demo"""
        print("🚀 STARTING COMPREHENSIVE FAILOVER MECHANISM DEMO")
        print("="*60)
        
        # Start monitoring
        await self.failover_manager.start_monitoring()
        
        try:
            # Initial status
            print("\n1️⃣ INITIAL SYSTEM STATUS")
            self.print_system_status()
            self.print_resource_health()
            
            await asyncio.sleep(3)
            
            # Test database failure
            print("\n2️⃣ DATABASE FAILURE SCENARIO")
            await self.simulate_database_failure()
            
            await asyncio.sleep(5)
            
            # Test data source failure
            print("\n3️⃣ DATA SOURCE FAILURE SCENARIO")
            await self.simulate_data_source_failure()
            
            await asyncio.sleep(5)
            
            # Test cache failure
            print("\n4️⃣ CACHE FAILURE SCENARIO")
            await self.simulate_cache_failure()
            
            await asyncio.sleep(5)
            
            # Test manual failover
            print("\n5️⃣ MANUAL FAILOVER TEST")
            await self.test_manual_failover()
            
            await asyncio.sleep(5)
            
            # Test cascade failure
            print("\n6️⃣ CASCADE FAILURE SCENARIO")
            await self.simulate_cascade_failure()
            
            await asyncio.sleep(5)
            
            # Test resource recovery
            print("\n7️⃣ RESOURCE RECOVERY SCENARIO")
            await self.simulate_resource_recovery()
            
            await asyncio.sleep(5)
            
            # Final status and history
            print("\n8️⃣ FINAL STATUS AND HISTORY")
            self.print_system_status()
            self.print_resource_health()
            self.print_failover_history()
            
            # Export report
            report_file = f"failover_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.failover_manager.export_failover_report(report_file)
            print(f"\n📄 Failover report exported to: {report_file}")
            
        finally:
            # Stop monitoring
            await self.failover_manager.stop_monitoring()
        
        print("\n✅ FAILOVER MECHANISM DEMO COMPLETED")
        print("="*60)


async def main():
    """Main demo function"""
    demo = FailoverDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())