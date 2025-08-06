"""
Cost Optimization Manager for Infrastructure Cost Tracking and Optimization

This module provides comprehensive cost monitoring, analysis, and optimization
capabilities for the stock analysis system infrastructure.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import psutil
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ResourceType(str, Enum):
    """Resource types for cost tracking"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"

class CostCategory(str, Enum):
    """Cost categories"""
    INFRASTRUCTURE = "infrastructure"
    DATA_PROCESSING = "data_processing"
    ANALYTICS = "analytics"
    STORAGE = "storage"
    NETWORKING = "networking"

@dataclass
class ResourceUsage:
    """Resource usage metrics"""
    resource_id: str
    resource_type: ResourceType
    timestamp: datetime
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_in: float = 0.0
    network_out: float = 0.0
    cost_per_hour: float = 0.0
    utilization_score: float = 0.0

@dataclass
class CostMetrics:
    """Cost metrics for analysis"""
    total_cost: float
    daily_cost: float
    monthly_cost: float
    cost_by_category: Dict[str, float]
    cost_by_resource: Dict[str, float]
    cost_trend: List[Tuple[datetime, float]]
    optimization_potential: float
    recommendations: List[str]

class CostAlert(BaseModel):
    """Cost alert configuration"""
    alert_id: str
    name: str
    threshold_type: str = Field(..., description="budget, spike, trend")
    threshold_value: float
    period: str = Field(..., description="daily, weekly, monthly")
    enabled: bool = True
    notification_channels: List[str] = []

class CostOptimizationManager:
    """
    Comprehensive cost optimization manager for infrastructure monitoring
    and cost optimization recommendations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the cost optimization manager."""
        self.config = config or {}
        self.resource_usage_history: List[ResourceUsage] = []
        self.cost_alerts: Dict[str, CostAlert] = {}
        self.optimization_rules: Dict[str, Any] = {}
        self.aws_client = None
        self.cost_thresholds = {
            'daily_budget': self.config.get('daily_budget', 1000.0),
            'monthly_budget': self.config.get('monthly_budget', 30000.0),
            'spike_threshold': self.config.get('spike_threshold', 0.5),  # 50% increase
            'utilization_threshold': self.config.get('utilization_threshold', 0.3)  # 30% utilization
        }
        
        # Initialize AWS client if credentials are available
        try:
            self.aws_client = boto3.client('ce')  # Cost Explorer
        except Exception as e:
            logger.warning(f"AWS Cost Explorer not available: {e}")
    
    async def collect_resource_usage(self) -> ResourceUsage:
        """Collect current resource usage metrics."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Calculate utilization score
            utilization_score = (
                cpu_percent / 100 * 0.4 +
                memory.percent / 100 * 0.3 +
                (disk.used / disk.total) * 0.3
            )
            
            # Estimate cost per hour (simplified)
            base_cost_per_hour = self.config.get('base_cost_per_hour', 0.1)
            cost_multiplier = 1 + (utilization_score * 0.5)
            
            usage = ResourceUsage(
                resource_id=f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                resource_type=ResourceType.COMPUTE,
                timestamp=datetime.now(),
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                network_in=network.bytes_recv / (1024 * 1024),  # MB
                network_out=network.bytes_sent / (1024 * 1024),  # MB
                cost_per_hour=base_cost_per_hour * cost_multiplier,
                utilization_score=utilization_score
            )
            
            # Store usage history
            self.resource_usage_history.append(usage)
            
            # Keep only last 24 hours of data
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.resource_usage_history = [
                u for u in self.resource_usage_history 
                if u.timestamp > cutoff_time
            ]
            
            return usage
            
        except Exception as e:
            logger.error(f"Error collecting resource usage: {e}")
            raise
    
    async def get_aws_cost_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get AWS cost data from Cost Explorer."""
        if not self.aws_client:
            return {}
        
        try:
            response = self.aws_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            )
            
            return response
            
        except ClientError as e:
            logger.error(f"Error fetching AWS cost data: {e}")
            return {}
    
    async def calculate_cost_metrics(self) -> CostMetrics:
        """Calculate comprehensive cost metrics."""
        try:
            if not self.resource_usage_history:
                await self.collect_resource_usage()
            
            # Calculate costs from usage history
            total_cost = sum(usage.cost_per_hour for usage in self.resource_usage_history)
            
            # Daily cost (last 24 hours)
            daily_cost = total_cost
            
            # Monthly cost projection
            monthly_cost = daily_cost * 30
            
            # Cost by category (simplified)
            cost_by_category = {
                CostCategory.INFRASTRUCTURE.value: total_cost * 0.4,
                CostCategory.DATA_PROCESSING.value: total_cost * 0.3,
                CostCategory.ANALYTICS.value: total_cost * 0.2,
                CostCategory.STORAGE.value: total_cost * 0.1
            }
            
            # Cost by resource type
            cost_by_resource = {}
            for resource_type in ResourceType:
                type_cost = sum(
                    usage.cost_per_hour for usage in self.resource_usage_history
                    if usage.resource_type == resource_type
                )
                if type_cost > 0:
                    cost_by_resource[resource_type.value] = type_cost
            
            # Cost trend (last 24 hours)
            cost_trend = []
            for usage in self.resource_usage_history[-24:]:  # Last 24 data points
                cost_trend.append((usage.timestamp, usage.cost_per_hour))
            
            # Calculate optimization potential
            avg_utilization = np.mean([u.utilization_score for u in self.resource_usage_history])
            optimization_potential = max(0, (0.7 - avg_utilization) * total_cost)
            
            # Generate recommendations
            recommendations = await self._generate_cost_recommendations()
            
            return CostMetrics(
                total_cost=total_cost,
                daily_cost=daily_cost,
                monthly_cost=monthly_cost,
                cost_by_category=cost_by_category,
                cost_by_resource=cost_by_resource,
                cost_trend=cost_trend,
                optimization_potential=optimization_potential,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error calculating cost metrics: {e}")
            raise
    
    async def _generate_cost_recommendations(self) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        if not self.resource_usage_history:
            return recommendations
        
        # Analyze utilization patterns
        avg_cpu = np.mean([u.cpu_usage for u in self.resource_usage_history])
        avg_memory = np.mean([u.memory_usage for u in self.resource_usage_history])
        avg_utilization = np.mean([u.utilization_score for u in self.resource_usage_history])
        
        # Low utilization recommendations
        if avg_utilization < self.cost_thresholds['utilization_threshold']:
            recommendations.append(
                f"Consider downsizing resources - average utilization is {avg_utilization:.1%}"
            )
        
        if avg_cpu < 20:
            recommendations.append(
                f"CPU utilization is low ({avg_cpu:.1f}%) - consider smaller instance types"
            )
        
        if avg_memory < 30:
            recommendations.append(
                f"Memory utilization is low ({avg_memory:.1f}%) - consider memory-optimized instances"
            )
        
        # Cost spike detection
        if len(self.resource_usage_history) > 10:
            recent_costs = [u.cost_per_hour for u in self.resource_usage_history[-10:]]
            older_costs = [u.cost_per_hour for u in self.resource_usage_history[-20:-10]]
            
            if older_costs and np.mean(recent_costs) > np.mean(older_costs) * (1 + self.cost_thresholds['spike_threshold']):
                recommendations.append(
                    "Cost spike detected - investigate recent resource usage changes"
                )
        
        # Scheduling recommendations
        peak_hours = self._identify_peak_hours()
        if peak_hours:
            recommendations.append(
                f"Consider auto-scaling during peak hours: {', '.join(map(str, peak_hours))}"
            )
        
        # Storage optimization
        disk_usage = np.mean([u.disk_usage for u in self.resource_usage_history])
        if disk_usage > 80:
            recommendations.append(
                f"Disk usage is high ({disk_usage:.1f}%) - consider storage cleanup or expansion"
            )
        
        return recommendations
    
    def _identify_peak_hours(self) -> List[int]:
        """Identify peak usage hours."""
        if len(self.resource_usage_history) < 24:
            return []
        
        # Group by hour and calculate average utilization
        hourly_usage = {}
        for usage in self.resource_usage_history:
            hour = usage.timestamp.hour
            if hour not in hourly_usage:
                hourly_usage[hour] = []
            hourly_usage[hour].append(usage.utilization_score)
        
        # Calculate average utilization per hour
        hourly_avg = {
            hour: np.mean(scores) 
            for hour, scores in hourly_usage.items()
        }
        
        # Find hours with above-average utilization
        overall_avg = np.mean(list(hourly_avg.values()))
        peak_hours = [
            hour for hour, avg_util in hourly_avg.items()
            if avg_util > overall_avg * 1.2
        ]
        
        return sorted(peak_hours)
    
    async def create_cost_alert(self, alert: CostAlert) -> bool:
        """Create a new cost alert."""
        try:
            self.cost_alerts[alert.alert_id] = alert
            logger.info(f"Created cost alert: {alert.name}")
            return True
        except Exception as e:
            logger.error(f"Error creating cost alert: {e}")
            return False
    
    async def check_cost_alerts(self) -> List[Dict[str, Any]]:
        """Check all cost alerts and return triggered alerts."""
        triggered_alerts = []
        
        try:
            current_metrics = await self.calculate_cost_metrics()
            
            for alert_id, alert in self.cost_alerts.items():
                if not alert.enabled:
                    continue
                
                triggered = False
                message = ""
                
                if alert.threshold_type == "budget":
                    if alert.period == "daily" and current_metrics.daily_cost > alert.threshold_value:
                        triggered = True
                        message = f"Daily cost ${current_metrics.daily_cost:.2f} exceeds budget ${alert.threshold_value:.2f}"
                    elif alert.period == "monthly" and current_metrics.monthly_cost > alert.threshold_value:
                        triggered = True
                        message = f"Monthly cost projection ${current_metrics.monthly_cost:.2f} exceeds budget ${alert.threshold_value:.2f}"
                
                elif alert.threshold_type == "spike":
                    # Check for cost spikes
                    if len(current_metrics.cost_trend) > 5:
                        recent_avg = np.mean([cost for _, cost in current_metrics.cost_trend[-5:]])
                        older_avg = np.mean([cost for _, cost in current_metrics.cost_trend[-10:-5]])
                        
                        if older_avg > 0 and (recent_avg / older_avg - 1) > alert.threshold_value:
                            triggered = True
                            message = f"Cost spike detected: {(recent_avg / older_avg - 1):.1%} increase"
                
                if triggered:
                    triggered_alerts.append({
                        'alert_id': alert_id,
                        'alert_name': alert.name,
                        'message': message,
                        'timestamp': datetime.now(),
                        'severity': 'high' if 'budget' in alert.threshold_type else 'medium'
                    })
            
            return triggered_alerts
            
        except Exception as e:
            logger.error(f"Error checking cost alerts: {e}")
            return []
    
    async def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get comprehensive optimization recommendations."""
        try:
            metrics = await self.calculate_cost_metrics()
            
            # Resource right-sizing recommendations
            rightsizing_recs = await self._get_rightsizing_recommendations()
            
            # Scheduling recommendations
            scheduling_recs = await self._get_scheduling_recommendations()
            
            # Storage optimization recommendations
            storage_recs = await self._get_storage_recommendations()
            
            return {
                'cost_metrics': asdict(metrics),
                'rightsizing_recommendations': rightsizing_recs,
                'scheduling_recommendations': scheduling_recs,
                'storage_recommendations': storage_recs,
                'potential_savings': metrics.optimization_potential,
                'priority_actions': metrics.recommendations[:3]  # Top 3 recommendations
            }
            
        except Exception as e:
            logger.error(f"Error getting optimization recommendations: {e}")
            return {}
    
    async def _get_rightsizing_recommendations(self) -> List[Dict[str, Any]]:
        """Get resource right-sizing recommendations."""
        recommendations = []
        
        if not self.resource_usage_history:
            return recommendations
        
        # Analyze CPU utilization
        cpu_usage = [u.cpu_usage for u in self.resource_usage_history]
        avg_cpu = np.mean(cpu_usage)
        max_cpu = np.max(cpu_usage)
        
        if avg_cpu < 20 and max_cpu < 50:
            recommendations.append({
                'type': 'downsize',
                'resource': 'compute',
                'current_utilization': f"{avg_cpu:.1f}%",
                'recommendation': 'Consider smaller instance type',
                'potential_savings': '20-40%'
            })
        
        # Analyze memory utilization
        memory_usage = [u.memory_usage for u in self.resource_usage_history]
        avg_memory = np.mean(memory_usage)
        
        if avg_memory < 30:
            recommendations.append({
                'type': 'optimize',
                'resource': 'memory',
                'current_utilization': f"{avg_memory:.1f}%",
                'recommendation': 'Switch to compute-optimized instance',
                'potential_savings': '15-25%'
            })
        
        return recommendations
    
    async def _get_scheduling_recommendations(self) -> List[Dict[str, Any]]:
        """Get scheduling optimization recommendations."""
        recommendations = []
        
        peak_hours = self._identify_peak_hours()
        if peak_hours:
            recommendations.append({
                'type': 'auto_scaling',
                'peak_hours': peak_hours,
                'recommendation': 'Implement auto-scaling during peak hours',
                'potential_savings': '10-30%'
            })
        
        # Check for consistent low usage periods
        if len(self.resource_usage_history) > 24:
            hourly_usage = {}
            for usage in self.resource_usage_history:
                hour = usage.timestamp.hour
                if hour not in hourly_usage:
                    hourly_usage[hour] = []
                hourly_usage[hour].append(usage.utilization_score)
            
            low_usage_hours = [
                hour for hour, scores in hourly_usage.items()
                if np.mean(scores) < 0.2
            ]
            
            if len(low_usage_hours) > 4:
                recommendations.append({
                    'type': 'scheduled_shutdown',
                    'low_usage_hours': low_usage_hours,
                    'recommendation': 'Consider scheduled shutdown during low usage hours',
                    'potential_savings': '20-50%'
                })
        
        return recommendations
    
    async def _get_storage_recommendations(self) -> List[Dict[str, Any]]:
        """Get storage optimization recommendations."""
        recommendations = []
        
        if not self.resource_usage_history:
            return recommendations
        
        avg_disk_usage = np.mean([u.disk_usage for u in self.resource_usage_history])
        
        if avg_disk_usage < 30:
            recommendations.append({
                'type': 'storage_optimization',
                'current_usage': f"{avg_disk_usage:.1f}%",
                'recommendation': 'Consider smaller storage allocation',
                'potential_savings': '10-20%'
            })
        elif avg_disk_usage > 80:
            recommendations.append({
                'type': 'storage_expansion',
                'current_usage': f"{avg_disk_usage:.1f}%",
                'recommendation': 'Consider storage expansion or cleanup',
                'urgency': 'high'
            })
        
        return recommendations
    
    async def export_cost_report(self, format: str = "json") -> str:
        """Export comprehensive cost report."""
        try:
            metrics = await self.calculate_cost_metrics()
            recommendations = await self.get_optimization_recommendations()
            
            report_data = {
                'report_timestamp': datetime.now().isoformat(),
                'cost_metrics': asdict(metrics),
                'optimization_recommendations': recommendations,
                'resource_usage_summary': {
                    'total_data_points': len(self.resource_usage_history),
                    'time_range': {
                        'start': min(u.timestamp for u in self.resource_usage_history).isoformat() if self.resource_usage_history else None,
                        'end': max(u.timestamp for u in self.resource_usage_history).isoformat() if self.resource_usage_history else None
                    }
                }
            }
            
            if format.lower() == "json":
                return json.dumps(report_data, indent=2, default=str)
            else:
                # Could add CSV, PDF formats here
                return json.dumps(report_data, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error exporting cost report: {e}")
            return "{}"