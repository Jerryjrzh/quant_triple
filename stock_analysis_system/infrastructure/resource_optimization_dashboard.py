"""
Resource Optimization Dashboard

This module provides comprehensive cost and resource usage visualization,
cost forecasting, budget planning tools, and resource optimization recommendations.
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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field

from .cost_optimization_manager import CostOptimizationManager, CostMetrics, ResourceUsage
from .intelligent_autoscaling import IntelligentAutoScaling, ScalingMetrics

logger = logging.getLogger(__name__)

class DashboardView(str, Enum):
    """Dashboard view types"""
    OVERVIEW = "overview"
    COST_ANALYSIS = "cost_analysis"
    RESOURCE_UTILIZATION = "resource_utilization"
    FORECASTING = "forecasting"
    OPTIMIZATION = "optimization"
    ALERTS = "alerts"

class TimeRange(str, Enum):
    """Time range options"""
    LAST_HOUR = "1h"
    LAST_DAY = "24h"
    LAST_WEEK = "7d"
    LAST_MONTH = "30d"
    LAST_QUARTER = "90d"

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    refresh_interval: int = 300  # seconds
    default_view: DashboardView = DashboardView.OVERVIEW
    default_time_range: TimeRange = TimeRange.LAST_DAY
    enable_real_time: bool = True
    cost_currency: str = "USD"
    timezone: str = "UTC"

class BudgetPlan(BaseModel):
    """Budget planning model"""
    name: str
    period: str = Field(..., description="daily, weekly, monthly, quarterly")
    budget_amount: float = Field(..., gt=0)
    start_date: datetime
    end_date: datetime
    categories: Dict[str, float] = Field(default_factory=dict)
    alerts_enabled: bool = True
    alert_thresholds: List[float] = Field(default=[0.8, 0.9, 1.0])

class CostForecast(BaseModel):
    """Cost forecast model"""
    forecast_date: datetime
    predicted_cost: float
    confidence_interval: Tuple[float, float]
    trend: str = Field(..., description="increasing, decreasing, stable")
    factors: List[str] = Field(default_factory=list)

class ResourceOptimizationDashboard:
    """
    Comprehensive resource optimization dashboard providing visualization,
    forecasting, and optimization recommendations.
    """
    
    def __init__(self, cost_manager: CostOptimizationManager, 
                 autoscaling: IntelligentAutoScaling,
                 config: DashboardConfig = None):
        """Initialize the resource optimization dashboard."""
        self.cost_manager = cost_manager
        self.autoscaling = autoscaling
        self.config = config or DashboardConfig()
        self.budget_plans: Dict[str, BudgetPlan] = {}
        self.forecast_cache: Dict[str, List[CostForecast]] = {}
        self.dashboard_data_cache: Dict[str, Any] = {}
        self.last_cache_update = datetime.now() - timedelta(seconds=self.config.refresh_interval)
    
    async def generate_overview_dashboard(self, time_range: TimeRange = None) -> Dict[str, Any]:
        """Generate overview dashboard with key metrics and visualizations."""
        try:
            time_range = time_range or self.config.default_time_range
            
            # Get cost metrics
            cost_metrics = await self.cost_manager.calculate_cost_metrics()
            
            # Get scaling analytics
            scaling_analytics = await self.autoscaling.get_scaling_analytics()
            
            # Create overview charts
            charts = await self._create_overview_charts(cost_metrics, scaling_analytics, time_range)
            
            # Calculate key performance indicators
            kpis = await self._calculate_overview_kpis(cost_metrics, scaling_analytics)
            
            # Get recent alerts
            recent_alerts = await self.cost_manager.check_cost_alerts()
            
            overview_data = {
                'timestamp': datetime.now().isoformat(),
                'time_range': time_range.value,
                'kpis': kpis,
                'charts': charts,
                'recent_alerts': recent_alerts[-5:],  # Last 5 alerts
                'optimization_summary': {
                    'potential_savings': cost_metrics.optimization_potential,
                    'top_recommendations': cost_metrics.recommendations[:3],
                    'efficiency_score': await self._calculate_efficiency_score(cost_metrics, scaling_analytics)
                }
            }
            
            return overview_data
            
        except Exception as e:
            logger.error(f"Error generating overview dashboard: {e}")
            return {}
    
    async def _create_overview_charts(self, cost_metrics: CostMetrics, 
                                    scaling_analytics: Dict[str, Any],
                                    time_range: TimeRange) -> Dict[str, str]:
        """Create overview charts for the dashboard."""
        charts = {}
        
        try:
            # Cost trend chart
            if cost_metrics.cost_trend:
                dates, costs = zip(*cost_metrics.cost_trend)
                
                fig_cost_trend = go.Figure()
                fig_cost_trend.add_trace(go.Scatter(
                    x=dates,
                    y=costs,
                    mode='lines+markers',
                    name='Cost per Hour',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6)
                ))
                
                fig_cost_trend.update_layout(
                    title='Cost Trend Over Time',
                    xaxis_title='Time',
                    yaxis_title=f'Cost ({self.config.cost_currency})',
                    template='plotly_white',
                    height=300
                )
                
                charts['cost_trend'] = fig_cost_trend.to_json()
            
            # Cost breakdown pie chart
            if cost_metrics.cost_by_category:
                fig_cost_breakdown = go.Figure(data=[go.Pie(
                    labels=list(cost_metrics.cost_by_category.keys()),
                    values=list(cost_metrics.cost_by_category.values()),
                    hole=0.3
                )])
                
                fig_cost_breakdown.update_layout(
                    title='Cost Breakdown by Category',
                    template='plotly_white',
                    height=300
                )
                
                charts['cost_breakdown'] = fig_cost_breakdown.to_json()
            
            # Resource utilization gauge
            if scaling_analytics.get('performance_metrics'):
                perf_metrics = scaling_analytics['performance_metrics']
                avg_performance = perf_metrics.get('average_performance_score', 0)
                
                fig_utilization = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=avg_performance,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Resource Efficiency"},
                    delta={'reference': 80},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig_utilization.update_layout(
                    template='plotly_white',
                    height=300
                )
                
                charts['resource_efficiency'] = fig_utilization.to_json()
            
        except Exception as e:
            logger.error(f"Error creating overview charts: {e}")
        
        return charts
    
    async def _calculate_overview_kpis(self, cost_metrics: CostMetrics,
                                     scaling_analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key performance indicators for overview."""
        kpis = {}
        
        try:
            # Cost KPIs
            kpis['total_cost'] = {
                'value': cost_metrics.total_cost,
                'unit': self.config.cost_currency,
                'change': 0.0,  # Would calculate from historical data
                'trend': 'stable'
            }
            
            kpis['daily_cost'] = {
                'value': cost_metrics.daily_cost,
                'unit': f"{self.config.cost_currency}/day",
                'change': 0.0,
                'trend': 'stable'
            }
            
            kpis['monthly_projection'] = {
                'value': cost_metrics.monthly_cost,
                'unit': f"{self.config.cost_currency}/month",
                'change': 0.0,
                'trend': 'stable'
            }
            
            # Efficiency KPIs
            if scaling_analytics.get('performance_metrics'):
                perf_metrics = scaling_analytics['performance_metrics']
                
                kpis['cost_efficiency'] = {
                    'value': perf_metrics.get('cost_performance_ratio', 0),
                    'unit': f"{self.config.cost_currency}/performance",
                    'change': 0.0,
                    'trend': 'stable'
                }
            
            # Optimization KPIs
            kpis['potential_savings'] = {
                'value': cost_metrics.optimization_potential,
                'unit': self.config.cost_currency,
                'percentage': (cost_metrics.optimization_potential / max(cost_metrics.total_cost, 1)) * 100
            }
            
            # Scaling KPIs
            if scaling_analytics.get('scaling_effectiveness'):
                scaling_eff = scaling_analytics['scaling_effectiveness']
                
                kpis['scaling_success_rate'] = {
                    'value': scaling_eff.get('success_rate', 0) * 100,
                    'unit': '%',
                    'total_actions': scaling_eff.get('total_scaling_actions', 0)
                }
            
        except Exception as e:
            logger.error(f"Error calculating overview KPIs: {e}")
        
        return kpis
    
    async def _calculate_efficiency_score(self, cost_metrics: CostMetrics,
                                        scaling_analytics: Dict[str, Any]) -> float:
        """Calculate overall efficiency score (0-100)."""
        try:
            scores = []
            
            # Cost efficiency (lower cost per performance is better)
            if scaling_analytics.get('performance_metrics'):
                perf_metrics = scaling_analytics['performance_metrics']
                cost_perf_ratio = perf_metrics.get('cost_performance_ratio', 1)
                cost_score = max(0, 100 - cost_perf_ratio * 100)
                scores.append(cost_score)
            
            # Utilization efficiency
            if cost_metrics.optimization_potential > 0:
                utilization_score = max(0, 100 - (cost_metrics.optimization_potential / max(cost_metrics.total_cost, 1)) * 100)
                scores.append(utilization_score)
            
            # Scaling efficiency
            if scaling_analytics.get('scaling_effectiveness'):
                scaling_eff = scaling_analytics['scaling_effectiveness']
                scaling_score = scaling_eff.get('success_rate', 0) * 100
                scores.append(scaling_score)
            
            return np.mean(scores) if scores else 50.0
            
        except Exception as e:
            logger.error(f"Error calculating efficiency score: {e}")
            return 50.0
    
    async def generate_cost_forecast(self, days_ahead: int = 30) -> List[CostForecast]:
        """Generate cost forecast for specified number of days."""
        try:
            cache_key = f"forecast_{days_ahead}d"
            
            # Check cache
            if (cache_key in self.forecast_cache and 
                datetime.now() - self.last_cache_update < timedelta(hours=1)):
                return self.forecast_cache[cache_key]
            
            # Get historical cost data
            cost_metrics = await self.cost_manager.calculate_cost_metrics()
            
            if not cost_metrics.cost_trend:
                return []
            
            # Extract historical data
            dates, costs = zip(*cost_metrics.cost_trend)
            
            # Simple linear regression for trend
            x = np.arange(len(costs))
            coeffs = np.polyfit(x, costs, 1)
            trend_slope = coeffs[0]
            
            # Determine trend direction
            if trend_slope > 0.01:
                trend = "increasing"
            elif trend_slope < -0.01:
                trend = "decreasing"
            else:
                trend = "stable"
            
            # Generate forecasts
            forecasts = []
            last_cost = costs[-1]
            last_date = dates[-1]
            
            # Calculate volatility for confidence intervals
            cost_volatility = np.std(costs) if len(costs) > 1 else last_cost * 0.1
            
            for day in range(1, days_ahead + 1):
                forecast_date = last_date + timedelta(days=day)
                
                # Simple trend projection with some seasonality
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * day / 7)  # Weekly pattern
                predicted_cost = last_cost + (trend_slope * day) * seasonal_factor
                
                # Confidence interval (±2 standard deviations)
                confidence_lower = predicted_cost - 2 * cost_volatility
                confidence_upper = predicted_cost + 2 * cost_volatility
                
                # Identify key factors
                factors = []
                if day % 7 in [0, 6]:  # Weekend
                    factors.append("weekend_pattern")
                if trend_slope > 0:
                    factors.append("increasing_usage")
                if day > 20:
                    factors.append("long_term_projection")
                
                forecast = CostForecast(
                    forecast_date=forecast_date,
                    predicted_cost=max(0, predicted_cost),
                    confidence_interval=(max(0, confidence_lower), confidence_upper),
                    trend=trend,
                    factors=factors
                )
                
                forecasts.append(forecast)
            
            # Cache the results
            self.forecast_cache[cache_key] = forecasts
            self.last_cache_update = datetime.now()
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Error generating cost forecast: {e}")
            return []
    
    async def create_budget_plan(self, plan: BudgetPlan) -> bool:
        """Create a new budget plan."""
        try:
            self.budget_plans[plan.name] = plan
            logger.info(f"Created budget plan: {plan.name}")
            return True
        except Exception as e:
            logger.error(f"Error creating budget plan: {e}")
            return False
    
    async def check_budget_status(self, plan_name: str) -> Dict[str, Any]:
        """Check budget status for a specific plan."""
        try:
            if plan_name not in self.budget_plans:
                return {}
            
            plan = self.budget_plans[plan_name]
            cost_metrics = await self.cost_manager.calculate_cost_metrics()
            
            # Calculate current spending based on period
            if plan.period == "daily":
                current_spending = cost_metrics.daily_cost
            elif plan.period == "monthly":
                current_spending = cost_metrics.monthly_cost
            else:
                # For weekly/quarterly, estimate based on daily cost
                days_in_period = {"weekly": 7, "quarterly": 90}.get(plan.period, 30)
                current_spending = cost_metrics.daily_cost * days_in_period
            
            # Calculate budget utilization
            budget_utilization = (current_spending / plan.budget_amount) * 100
            
            # Determine status
            if budget_utilization >= 100:
                status = "exceeded"
            elif budget_utilization >= 90:
                status = "critical"
            elif budget_utilization >= 80:
                status = "warning"
            else:
                status = "healthy"
            
            # Check category budgets
            category_status = {}
            for category, category_budget in plan.categories.items():
                category_spending = cost_metrics.cost_by_category.get(category, 0)
                category_utilization = (category_spending / category_budget) * 100
                category_status[category] = {
                    'spending': category_spending,
                    'budget': category_budget,
                    'utilization': category_utilization,
                    'status': 'exceeded' if category_utilization >= 100 else 'healthy'
                }
            
            return {
                'plan_name': plan_name,
                'period': plan.period,
                'budget_amount': plan.budget_amount,
                'current_spending': current_spending,
                'budget_utilization': budget_utilization,
                'status': status,
                'remaining_budget': max(0, plan.budget_amount - current_spending),
                'days_remaining': (plan.end_date - datetime.now()).days,
                'category_status': category_status,
                'alerts_triggered': budget_utilization >= min(plan.alert_thresholds) * 100
            }
            
        except Exception as e:
            logger.error(f"Error checking budget status: {e}")
            return {}
    
    async def generate_optimization_recommendations(self) -> Dict[str, Any]:
        """Generate comprehensive optimization recommendations."""
        try:
            # Get recommendations from cost manager
            cost_recommendations = await self.cost_manager.get_optimization_recommendations()
            
            # Get rightsizing recommendations from autoscaling
            rightsizing_recs = await self.autoscaling.get_rightsizing_recommendations()
            
            # Get cost forecast
            forecast = await self.generate_cost_forecast(30)
            
            # Prioritize recommendations
            prioritized_recs = await self._prioritize_recommendations(
                cost_recommendations, rightsizing_recs, forecast
            )
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cost_recommendations': cost_recommendations,
                'rightsizing_recommendations': rightsizing_recs,
                'prioritized_actions': prioritized_recs,
                'forecast_impact': await self._calculate_forecast_impact(forecast),
                'implementation_roadmap': await self._create_implementation_roadmap(prioritized_recs)
            }
            
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")
            return {}
    
    async def _prioritize_recommendations(self, cost_recs: Dict[str, Any],
                                        rightsizing_recs: List[Dict[str, Any]],
                                        forecast: List[CostForecast]) -> List[Dict[str, Any]]:
        """Prioritize recommendations based on impact and urgency."""
        prioritized = []
        
        try:
            # Add cost recommendations with priority scoring
            if cost_recs.get('priority_actions'):
                for i, rec in enumerate(cost_recs['priority_actions']):
                    prioritized.append({
                        'type': 'cost_optimization',
                        'recommendation': rec,
                        'priority_score': 100 - i * 10,  # Higher score for earlier recommendations
                        'category': 'cost',
                        'urgency': 'high' if i == 0 else 'medium'
                    })
            
            # Add rightsizing recommendations
            for rec in rightsizing_recs:
                priority_score = 80
                urgency = 'medium'
                
                # Increase priority for high-impact recommendations
                if 'potential_savings' in rec:
                    savings_pct = float(rec['potential_savings'].rstrip('%').split('-')[0])
                    if savings_pct > 30:
                        priority_score = 90
                        urgency = 'high'
                
                prioritized.append({
                    'type': 'rightsizing',
                    'recommendation': rec,
                    'priority_score': priority_score,
                    'category': 'performance',
                    'urgency': urgency
                })
            
            # Add forecast-based recommendations
            if forecast:
                increasing_trend = any(f.trend == "increasing" for f in forecast[:7])
                if increasing_trend:
                    prioritized.append({
                        'type': 'forecast_based',
                        'recommendation': 'Cost trend is increasing - consider immediate optimization',
                        'priority_score': 85,
                        'category': 'preventive',
                        'urgency': 'high'
                    })
            
            # Sort by priority score
            prioritized.sort(key=lambda x: x['priority_score'], reverse=True)
            
            return prioritized[:10]  # Top 10 recommendations
            
        except Exception as e:
            logger.error(f"Error prioritizing recommendations: {e}")
            return []
    
    async def _calculate_forecast_impact(self, forecast: List[CostForecast]) -> Dict[str, Any]:
        """Calculate the impact of current trends on future costs."""
        try:
            if not forecast:
                return {}
            
            # Calculate trend impact
            first_week = forecast[:7]
            last_week = forecast[-7:] if len(forecast) >= 14 else forecast[7:14]
            
            if first_week and last_week:
                first_week_avg = np.mean([f.predicted_cost for f in first_week])
                last_week_avg = np.mean([f.predicted_cost for f in last_week])
                
                trend_impact = ((last_week_avg - first_week_avg) / first_week_avg) * 100
                
                return {
                    'trend_impact_percentage': trend_impact,
                    'projected_monthly_increase': trend_impact * 4,  # Approximate monthly
                    'confidence': 'high' if len(forecast) >= 30 else 'medium',
                    'key_factors': list(set(factor for f in forecast for factor in f.factors))
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error calculating forecast impact: {e}")
            return {}
    
    async def _create_implementation_roadmap(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create implementation roadmap for recommendations."""
        roadmap = []
        
        try:
            # Group recommendations by urgency and complexity
            high_urgency = [r for r in recommendations if r.get('urgency') == 'high']
            medium_urgency = [r for r in recommendations if r.get('urgency') == 'medium']
            
            # Phase 1: Immediate actions (high urgency, low complexity)
            if high_urgency:
                roadmap.append({
                    'phase': 1,
                    'name': 'Immediate Optimizations',
                    'duration': '1-2 weeks',
                    'actions': high_urgency[:3],
                    'expected_impact': 'High cost reduction, immediate effect'
                })
            
            # Phase 2: Medium-term optimizations
            if medium_urgency:
                roadmap.append({
                    'phase': 2,
                    'name': 'Infrastructure Optimization',
                    'duration': '2-4 weeks',
                    'actions': medium_urgency[:3],
                    'expected_impact': 'Sustained cost optimization'
                })
            
            # Phase 3: Long-term strategic changes
            remaining_recs = recommendations[6:]  # After first 6 recommendations
            if remaining_recs:
                roadmap.append({
                    'phase': 3,
                    'name': 'Strategic Optimization',
                    'duration': '1-3 months',
                    'actions': remaining_recs,
                    'expected_impact': 'Long-term cost efficiency'
                })
            
            return roadmap
            
        except Exception as e:
            logger.error(f"Error creating implementation roadmap: {e}")
            return []
    
    async def export_dashboard_data(self, view: DashboardView, 
                                  time_range: TimeRange = None,
                                  format: str = "json") -> str:
        """Export dashboard data in specified format."""
        try:
            time_range = time_range or self.config.default_time_range
            
            if view == DashboardView.OVERVIEW:
                data = await self.generate_overview_dashboard(time_range)
            elif view == DashboardView.COST_ANALYSIS:
                data = await self.cost_manager.calculate_cost_metrics()
                data = asdict(data)
            elif view == DashboardView.FORECASTING:
                forecasts = await self.generate_cost_forecast()
                data = {'forecasts': [asdict(f) for f in forecasts]}
            elif view == DashboardView.OPTIMIZATION:
                data = await self.generate_optimization_recommendations()
            else:
                data = {}
            
            # Add metadata
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'view': view.value,
                'time_range': time_range.value,
                'data': data
            }
            
            if format.lower() == "json":
                return json.dumps(export_data, indent=2, default=str)
            else:
                # Could add CSV, Excel formats here
                return json.dumps(export_data, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {e}")
            return "{}"