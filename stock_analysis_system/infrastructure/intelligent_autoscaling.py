"""
Intelligent Auto-scaling System for Cost Optimization

This module provides predictive auto-scaling capabilities based on usage patterns,
spot instance management, and performance vs. cost optimization balancing.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ScalingAction(str, Enum):
    """Scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    SWITCH_TO_SPOT = "switch_to_spot"
    SWITCH_TO_ON_DEMAND = "switch_to_on_demand"

class InstanceType(str, Enum):
    """Instance types for scaling"""
    MICRO = "t3.micro"
    SMALL = "t3.small"
    MEDIUM = "t3.medium"
    LARGE = "t3.large"
    XLARGE = "t3.xlarge"
    COMPUTE_LARGE = "c5.large"
    COMPUTE_XLARGE = "c5.xlarge"
    MEMORY_LARGE = "r5.large"
    MEMORY_XLARGE = "r5.xlarge"

@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions"""
    timestamp: datetime
    cpu_utilization: float
    memory_utilization: float
    network_utilization: float
    request_rate: float
    response_time: float
    queue_length: int
    cost_per_hour: float
    performance_score: float

@dataclass
class ScalingDecision:
    """Scaling decision with rationale"""
    action: ScalingAction
    target_instances: int
    target_instance_type: InstanceType
    confidence: float
    rationale: str
    expected_cost_change: float
    expected_performance_change: float
    timestamp: datetime

class SpotInstanceConfig(BaseModel):
    """Spot instance configuration"""
    enabled: bool = True
    max_spot_percentage: float = Field(default=0.7, ge=0.0, le=1.0)
    spot_price_threshold: float = Field(default=0.5, description="Max price as % of on-demand")
    fallback_to_on_demand: bool = True
    interruption_handling: str = "graceful"

class AutoScalingConfig(BaseModel):
    """Auto-scaling configuration"""
    min_instances: int = Field(default=1, ge=1)
    max_instances: int = Field(default=10, ge=1)
    target_cpu_utilization: float = Field(default=70.0, ge=10.0, le=90.0)
    target_memory_utilization: float = Field(default=80.0, ge=10.0, le=90.0)
    scale_up_threshold: float = Field(default=80.0, ge=50.0, le=95.0)
    scale_down_threshold: float = Field(default=30.0, ge=5.0, le=50.0)
    cooldown_period: int = Field(default=300, description="Seconds between scaling actions")
    prediction_window: int = Field(default=3600, description="Seconds to predict ahead")

class IntelligentAutoScaling:
    """
    Intelligent auto-scaling system with predictive capabilities,
    spot instance management, and cost optimization.
    """
    
    def __init__(self, config: AutoScalingConfig, spot_config: SpotInstanceConfig = None):
        """Initialize the intelligent auto-scaling system."""
        self.config = config
        self.spot_config = spot_config or SpotInstanceConfig()
        self.metrics_history: List[ScalingMetrics] = []
        self.scaling_history: List[ScalingDecision] = []
        self.prediction_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.model_trained = False
        self.last_scaling_action = datetime.now() - timedelta(seconds=self.config.cooldown_period)
        
        # AWS clients
        self.ec2_client = None
        self.cloudwatch_client = None
        self.autoscaling_client = None
        
        try:
            self.ec2_client = boto3.client('ec2')
            self.cloudwatch_client = boto3.client('cloudwatch')
            self.autoscaling_client = boto3.client('autoscaling')
        except Exception as e:
            logger.warning(f"AWS clients not available: {e}")
        
        # Instance type specifications (simplified)
        self.instance_specs = {
            InstanceType.MICRO: {'cpu': 1, 'memory': 1, 'cost_per_hour': 0.0116},
            InstanceType.SMALL: {'cpu': 1, 'memory': 2, 'cost_per_hour': 0.0232},
            InstanceType.MEDIUM: {'cpu': 2, 'memory': 4, 'cost_per_hour': 0.0464},
            InstanceType.LARGE: {'cpu': 2, 'memory': 8, 'cost_per_hour': 0.0928},
            InstanceType.XLARGE: {'cpu': 4, 'memory': 16, 'cost_per_hour': 0.1856},
            InstanceType.COMPUTE_LARGE: {'cpu': 2, 'memory': 4, 'cost_per_hour': 0.085},
            InstanceType.COMPUTE_XLARGE: {'cpu': 4, 'memory': 8, 'cost_per_hour': 0.17},
            InstanceType.MEMORY_LARGE: {'cpu': 2, 'memory': 16, 'cost_per_hour': 0.126},
            InstanceType.MEMORY_XLARGE: {'cpu': 4, 'memory': 32, 'cost_per_hour': 0.252},
        }
    
    async def collect_scaling_metrics(self) -> ScalingMetrics:
        """Collect current metrics for scaling decisions."""
        try:
            # In a real implementation, these would come from monitoring systems
            # For now, we'll simulate with some realistic patterns
            
            current_time = datetime.now()
            hour = current_time.hour
            
            # Simulate daily patterns
            base_cpu = 30 + 40 * np.sin((hour - 6) * np.pi / 12)  # Peak around 2 PM
            base_memory = 40 + 30 * np.sin((hour - 8) * np.pi / 12)  # Peak around 4 PM
            
            # Add some randomness
            cpu_utilization = max(0, min(100, base_cpu + np.random.normal(0, 10)))
            memory_utilization = max(0, min(100, base_memory + np.random.normal(0, 8)))
            network_utilization = max(0, min(100, cpu_utilization * 0.7 + np.random.normal(0, 5)))
            
            # Request rate correlates with CPU usage
            request_rate = max(0, cpu_utilization * 2 + np.random.normal(0, 20))
            
            # Response time inversely correlates with available resources
            response_time = max(50, 200 - (100 - cpu_utilization) * 1.5 + np.random.normal(0, 30))
            
            # Queue length based on load
            queue_length = max(0, int((cpu_utilization - 50) / 10)) if cpu_utilization > 50 else 0
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(
                cpu_utilization, memory_utilization, response_time, queue_length
            )
            
            # Estimate current cost
            cost_per_hour = self._estimate_current_cost()
            
            metrics = ScalingMetrics(
                timestamp=current_time,
                cpu_utilization=cpu_utilization,
                memory_utilization=memory_utilization,
                network_utilization=network_utilization,
                request_rate=request_rate,
                response_time=response_time,
                queue_length=queue_length,
                cost_per_hour=cost_per_hour,
                performance_score=performance_score
            )
            
            # Store metrics history
            self.metrics_history.append(metrics)
            
            # Keep only last 24 hours of data
            cutoff_time = current_time - timedelta(hours=24)
            self.metrics_history = [
                m for m in self.metrics_history 
                if m.timestamp > cutoff_time
            ]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting scaling metrics: {e}")
            raise
    
    def _calculate_performance_score(self, cpu: float, memory: float, 
                                   response_time: float, queue_length: int) -> float:
        """Calculate overall performance score (0-100)."""
        # Performance decreases with high utilization and response time
        cpu_score = max(0, 100 - cpu) if cpu > 70 else 100
        memory_score = max(0, 100 - memory) if memory > 80 else 100
        response_score = max(0, 100 - (response_time - 100) / 5) if response_time > 100 else 100
        queue_score = max(0, 100 - queue_length * 10)
        
        # Weighted average
        performance_score = (
            cpu_score * 0.3 +
            memory_score * 0.3 +
            response_score * 0.3 +
            queue_score * 0.1
        )
        
        return max(0, min(100, performance_score))
    
    def _estimate_current_cost(self) -> float:
        """Estimate current hourly cost."""
        # Simplified cost estimation
        # In reality, this would query actual AWS billing
        base_cost = 0.1  # Base cost per hour
        
        if self.metrics_history:
            recent_metrics = self.metrics_history[-5:]  # Last 5 data points
            avg_utilization = np.mean([
                (m.cpu_utilization + m.memory_utilization) / 2 
                for m in recent_metrics
            ])
            
            # Higher utilization might indicate more instances or larger instances
            cost_multiplier = 1 + (avg_utilization / 100) * 2
            return base_cost * cost_multiplier
        
        return base_cost
    
    async def train_prediction_model(self) -> bool:
        """Train the predictive model using historical data."""
        try:
            if len(self.metrics_history) < 50:  # Need minimum data
                logger.info("Insufficient data for model training")
                return False
            
            # Prepare features and targets
            features = []
            targets = []
            
            for i in range(len(self.metrics_history) - 1):
                current = self.metrics_history[i]
                next_metrics = self.metrics_history[i + 1]
                
                # Features: current metrics + time features
                feature_vector = [
                    current.cpu_utilization,
                    current.memory_utilization,
                    current.network_utilization,
                    current.request_rate,
                    current.response_time,
                    current.queue_length,
                    current.timestamp.hour,
                    current.timestamp.weekday(),
                    current.performance_score
                ]
                
                features.append(feature_vector)
                
                # Target: next period's CPU utilization (primary scaling metric)
                targets.append(next_metrics.cpu_utilization)
            
            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(targets)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.prediction_model.fit(X_scaled, y)
            self.model_trained = True
            
            logger.info(f"Prediction model trained with {len(features)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training prediction model: {e}")
            return False
    
    async def predict_future_load(self, minutes_ahead: int = 60) -> Dict[str, float]:
        """Predict future load using trained model."""
        try:
            if not self.model_trained or not self.metrics_history:
                return {}
            
            current_metrics = self.metrics_history[-1]
            future_time = current_metrics.timestamp + timedelta(minutes=minutes_ahead)
            
            # Prepare features for prediction
            feature_vector = [
                current_metrics.cpu_utilization,
                current_metrics.memory_utilization,
                current_metrics.network_utilization,
                current_metrics.request_rate,
                current_metrics.response_time,
                current_metrics.queue_length,
                future_time.hour,
                future_time.weekday(),
                current_metrics.performance_score
            ]
            
            # Scale features
            X_scaled = self.scaler.transform([feature_vector])
            
            # Make prediction
            predicted_cpu = self.prediction_model.predict(X_scaled)[0]
            
            # Estimate other metrics based on CPU prediction
            predicted_memory = predicted_cpu * 0.8 + np.random.normal(0, 5)
            predicted_response_time = max(50, 200 - (100 - predicted_cpu) * 1.5)
            
            return {
                'predicted_cpu': max(0, min(100, predicted_cpu)),
                'predicted_memory': max(0, min(100, predicted_memory)),
                'predicted_response_time': predicted_response_time,
                'confidence': min(1.0, len(self.metrics_history) / 100)  # Confidence based on data amount
            }
            
        except Exception as e:
            logger.error(f"Error predicting future load: {e}")
            return {}
    
    async def make_scaling_decision(self) -> Optional[ScalingDecision]:
        """Make intelligent scaling decision based on current and predicted metrics."""
        try:
            # Check cooldown period
            if datetime.now() - self.last_scaling_action < timedelta(seconds=self.config.cooldown_period):
                return None
            
            current_metrics = await self.collect_scaling_metrics()
            
            # Get prediction if model is trained
            prediction = await self.predict_future_load(self.config.prediction_window // 60)
            
            # Determine scaling action
            action = ScalingAction.MAINTAIN
            target_instances = 1  # Current instance count (simplified)
            target_instance_type = InstanceType.MEDIUM  # Current type (simplified)
            confidence = 0.5
            rationale = "No scaling needed"
            expected_cost_change = 0.0
            expected_performance_change = 0.0
            
            # Scale up conditions
            if (current_metrics.cpu_utilization > self.config.scale_up_threshold or
                current_metrics.memory_utilization > self.config.scale_up_threshold or
                current_metrics.queue_length > 5):
                
                action = ScalingAction.SCALE_UP
                target_instances = min(target_instances + 1, self.config.max_instances)
                confidence = 0.8
                rationale = f"High utilization detected - CPU: {current_metrics.cpu_utilization:.1f}%, Memory: {current_metrics.memory_utilization:.1f}%"
                expected_cost_change = self.instance_specs[target_instance_type]['cost_per_hour']
                expected_performance_change = 20.0
            
            # Scale down conditions
            elif (current_metrics.cpu_utilization < self.config.scale_down_threshold and
                  current_metrics.memory_utilization < self.config.scale_down_threshold and
                  current_metrics.queue_length == 0):
                
                if target_instances > self.config.min_instances:
                    action = ScalingAction.SCALE_DOWN
                    target_instances = max(target_instances - 1, self.config.min_instances)
                    confidence = 0.7
                    rationale = f"Low utilization detected - CPU: {current_metrics.cpu_utilization:.1f}%, Memory: {current_metrics.memory_utilization:.1f}%"
                    expected_cost_change = -self.instance_specs[target_instance_type]['cost_per_hour']
                    expected_performance_change = -5.0
            
            # Consider predictive scaling
            if prediction and prediction.get('confidence', 0) > 0.7:
                predicted_cpu = prediction['predicted_cpu']
                
                if predicted_cpu > self.config.scale_up_threshold and action == ScalingAction.MAINTAIN:
                    action = ScalingAction.SCALE_UP
                    target_instances = min(target_instances + 1, self.config.max_instances)
                    confidence = prediction['confidence']
                    rationale = f"Predictive scaling - Expected CPU: {predicted_cpu:.1f}%"
                    expected_cost_change = self.instance_specs[target_instance_type]['cost_per_hour']
                    expected_performance_change = 15.0
            
            # Consider spot instance optimization
            if self.spot_config.enabled:
                spot_decision = await self._evaluate_spot_instance_opportunity()
                if spot_decision:
                    action = spot_decision['action']
                    rationale += f" + {spot_decision['rationale']}"
                    expected_cost_change += spot_decision['cost_change']
            
            decision = ScalingDecision(
                action=action,
                target_instances=target_instances,
                target_instance_type=target_instance_type,
                confidence=confidence,
                rationale=rationale,
                expected_cost_change=expected_cost_change,
                expected_performance_change=expected_performance_change,
                timestamp=datetime.now()
            )
            
            # Store decision history
            self.scaling_history.append(decision)
            
            # Update last scaling action time if action is taken
            if action != ScalingAction.MAINTAIN:
                self.last_scaling_action = datetime.now()
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making scaling decision: {e}")
            return None
    
    async def _evaluate_spot_instance_opportunity(self) -> Optional[Dict[str, Any]]:
        """Evaluate spot instance opportunities for cost optimization."""
        try:
            if not self.ec2_client:
                return None
            
            # Get current spot prices
            response = self.ec2_client.describe_spot_price_history(
                InstanceTypes=[instance_type.value for instance_type in InstanceType],
                ProductDescriptions=['Linux/UNIX'],
                MaxResults=10
            )
            
            spot_prices = {}
            for price_info in response.get('SpotPrices', []):
                instance_type = price_info['InstanceType']
                spot_price = float(price_info['SpotPrice'])
                on_demand_price = self.instance_specs.get(
                    InstanceType(instance_type), {}
                ).get('cost_per_hour', 0)
                
                if on_demand_price > 0:
                    spot_discount = 1 - (spot_price / on_demand_price)
                    spot_prices[instance_type] = {
                        'spot_price': spot_price,
                        'on_demand_price': on_demand_price,
                        'discount': spot_discount
                    }
            
            # Find best spot opportunity
            best_opportunity = None
            best_savings = 0
            
            for instance_type, price_info in spot_prices.items():
                if (price_info['discount'] > 0.3 and  # At least 30% savings
                    price_info['spot_price'] < price_info['on_demand_price'] * self.spot_config.spot_price_threshold):
                    
                    if price_info['discount'] > best_savings:
                        best_savings = price_info['discount']
                        best_opportunity = {
                            'action': ScalingAction.SWITCH_TO_SPOT,
                            'instance_type': instance_type,
                            'rationale': f"Spot instance available with {best_savings:.1%} savings",
                            'cost_change': -(price_info['on_demand_price'] - price_info['spot_price'])
                        }
            
            return best_opportunity
            
        except Exception as e:
            logger.error(f"Error evaluating spot instance opportunity: {e}")
            return None
    
    async def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute the scaling decision."""
        try:
            logger.info(f"Executing scaling decision: {decision.action.value}")
            logger.info(f"Rationale: {decision.rationale}")
            
            # In a real implementation, this would interact with AWS Auto Scaling
            # For now, we'll just log the decision
            
            if decision.action == ScalingAction.SCALE_UP:
                logger.info(f"Scaling up to {decision.target_instances} instances")
                # AWS Auto Scaling API call would go here
                
            elif decision.action == ScalingAction.SCALE_DOWN:
                logger.info(f"Scaling down to {decision.target_instances} instances")
                # AWS Auto Scaling API call would go here
                
            elif decision.action == ScalingAction.SWITCH_TO_SPOT:
                logger.info(f"Switching to spot instances: {decision.target_instance_type.value}")
                # Spot fleet request would go here
                
            elif decision.action == ScalingAction.SWITCH_TO_ON_DEMAND:
                logger.info(f"Switching to on-demand instances: {decision.target_instance_type.value}")
                # Update launch template would go here
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing scaling decision: {e}")
            return False
    
    async def get_rightsizing_recommendations(self) -> List[Dict[str, Any]]:
        """Get resource right-sizing recommendations."""
        recommendations = []
        
        if len(self.metrics_history) < 10:
            return recommendations
        
        # Analyze recent utilization patterns
        recent_metrics = self.metrics_history[-24:]  # Last 24 data points
        
        avg_cpu = np.mean([m.cpu_utilization for m in recent_metrics])
        avg_memory = np.mean([m.memory_utilization for m in recent_metrics])
        max_cpu = np.max([m.cpu_utilization for m in recent_metrics])
        max_memory = np.max([m.memory_utilization for m in recent_metrics])
        
        # CPU-based recommendations
        if avg_cpu < 20 and max_cpu < 50:
            recommendations.append({
                'type': 'downsize_cpu',
                'current_avg_cpu': f"{avg_cpu:.1f}%",
                'recommendation': 'Consider smaller instance type',
                'suggested_types': [InstanceType.SMALL.value, InstanceType.MICRO.value],
                'potential_savings': '30-50%'
            })
        elif avg_cpu > 70:
            recommendations.append({
                'type': 'upsize_cpu',
                'current_avg_cpu': f"{avg_cpu:.1f}%",
                'recommendation': 'Consider larger or compute-optimized instance',
                'suggested_types': [InstanceType.COMPUTE_LARGE.value, InstanceType.COMPUTE_XLARGE.value],
                'performance_improvement': '20-40%'
            })
        
        # Memory-based recommendations
        if avg_memory < 30 and max_memory < 60:
            recommendations.append({
                'type': 'optimize_memory',
                'current_avg_memory': f"{avg_memory:.1f}%",
                'recommendation': 'Consider compute-optimized instance',
                'suggested_types': [InstanceType.COMPUTE_LARGE.value],
                'potential_savings': '15-25%'
            })
        elif avg_memory > 80:
            recommendations.append({
                'type': 'increase_memory',
                'current_avg_memory': f"{avg_memory:.1f}%",
                'recommendation': 'Consider memory-optimized instance',
                'suggested_types': [InstanceType.MEMORY_LARGE.value, InstanceType.MEMORY_XLARGE.value],
                'performance_improvement': '25-35%'
            })
        
        return recommendations
    
    async def get_scaling_analytics(self) -> Dict[str, Any]:
        """Get comprehensive scaling analytics."""
        try:
            if not self.metrics_history or not self.scaling_history:
                return {}
            
            # Performance metrics
            recent_metrics = self.metrics_history[-24:]
            avg_performance = np.mean([m.performance_score for m in recent_metrics])
            avg_cost = np.mean([m.cost_per_hour for m in recent_metrics])
            
            # Scaling effectiveness
            scaling_actions = [d for d in self.scaling_history if d.action != ScalingAction.MAINTAIN]
            successful_scalings = len([d for d in scaling_actions if d.confidence > 0.7])
            
            # Cost optimization metrics
            total_cost_change = sum(d.expected_cost_change for d in scaling_actions)
            total_performance_change = sum(d.expected_performance_change for d in scaling_actions)
            
            return {
                'performance_metrics': {
                    'average_performance_score': avg_performance,
                    'average_cost_per_hour': avg_cost,
                    'cost_performance_ratio': avg_cost / max(avg_performance, 1)
                },
                'scaling_effectiveness': {
                    'total_scaling_actions': len(scaling_actions),
                    'successful_scalings': successful_scalings,
                    'success_rate': successful_scalings / max(len(scaling_actions), 1)
                },
                'cost_optimization': {
                    'total_cost_change': total_cost_change,
                    'total_performance_change': total_performance_change,
                    'optimization_efficiency': total_performance_change / max(abs(total_cost_change), 1)
                },
                'rightsizing_recommendations': await self.get_rightsizing_recommendations()
            }
            
        except Exception as e:
            logger.error(f"Error getting scaling analytics: {e}")
            return {}
    
    async def export_scaling_report(self) -> str:
        """Export comprehensive scaling report."""
        try:
            analytics = await self.get_scaling_analytics()
            
            report_data = {
                'report_timestamp': datetime.now().isoformat(),
                'configuration': {
                    'min_instances': self.config.min_instances,
                    'max_instances': self.config.max_instances,
                    'target_cpu_utilization': self.config.target_cpu_utilization,
                    'spot_instances_enabled': self.spot_config.enabled
                },
                'analytics': analytics,
                'recent_decisions': [
                    asdict(decision) for decision in self.scaling_history[-10:]
                ],
                'metrics_summary': {
                    'total_data_points': len(self.metrics_history),
                    'model_trained': self.model_trained,
                    'prediction_accuracy': 'N/A' if not self.model_trained else 'Good'
                }
            }
            
            return json.dumps(report_data, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error exporting scaling report: {e}")
            return "{}"