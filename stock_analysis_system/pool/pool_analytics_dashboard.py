"""
Pool Analytics Dashboard

This module provides comprehensive pool performance visualization, sector analysis,
risk distribution analysis, and pool optimization recommendations.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

from .stock_pool_manager import StockPoolManager, StockPool, PoolType

logger = logging.getLogger(__name__)

class PoolAnalyticsDashboard:
    """
    Comprehensive Pool Analytics Dashboard
    
    Provides advanced visualization and analysis capabilities for stock pools
    including performance tracking, sector analysis, risk assessment, and optimization.
    """
    
    def __init__(self, pool_manager: StockPoolManager):
        self.pool_manager = pool_manager
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    async def create_pool_performance_dashboard(self, pool_id: str) -> Dict[str, Any]:
        """Create comprehensive performance dashboard for a single pool"""
        
        analytics = await self.pool_manager.get_pool_analytics(pool_id)
        if "error" in analytics:
            return analytics
        
        dashboard = {
            "pool_info": analytics["basic_info"],
            "charts": {},
            "metrics": analytics["performance_metrics"],
            "recommendations": analytics["recommendations"]
        }
        
        # Performance overview chart
        dashboard["charts"]["performance_overview"] = await self._create_performance_overview_chart(pool_id)
        
        # Stock performance breakdown
        dashboard["charts"]["stock_breakdown"] = await self._create_stock_breakdown_chart(pool_id)
        
        # Sector distribution
        dashboard["charts"]["sector_distribution"] = await self._create_sector_distribution_chart(pool_id)
        
        # Risk analysis
        dashboard["charts"]["risk_analysis"] = await self._create_risk_analysis_chart(pool_id)
        
        # Performance timeline
        dashboard["charts"]["performance_timeline"] = await self._create_performance_timeline_chart(pool_id)
        
        return dashboard
    
    async def create_multi_pool_comparison_dashboard(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Create comparison dashboard for multiple pools"""
        
        comparison = await self.pool_manager.compare_pools(pool_ids)
        if "error" in comparison:
            return comparison
        
        dashboard = {
            "comparison_data": comparison,
            "charts": {},
            "summary": await self._generate_comparison_summary(pool_ids)
        }
        
        # Performance comparison
        dashboard["charts"]["performance_comparison"] = await self._create_performance_comparison_chart(pool_ids)
        
        # Risk-return scatter
        dashboard["charts"]["risk_return_scatter"] = await self._create_risk_return_scatter(pool_ids)
        
        # Correlation heatmap
        dashboard["charts"]["correlation_heatmap"] = await self._create_correlation_heatmap(pool_ids)
        
        # Sector allocation comparison
        dashboard["charts"]["sector_comparison"] = await self._create_sector_comparison_chart(pool_ids)
        
        return dashboard
    
    async def create_sector_industry_analysis(self, pool_id: str) -> Dict[str, Any]:
        """Create detailed sector and industry breakdown analysis"""
        
        pool = self.pool_manager.pools.get(pool_id)
        if not pool:
            return {"error": "Pool not found"}
        
        analysis = {
            "sector_breakdown": await self._analyze_sector_breakdown(pool),
            "industry_breakdown": await self._analyze_industry_breakdown(pool),
            "charts": {},
            "insights": []
        }
        
        # Sector allocation pie chart
        analysis["charts"]["sector_pie"] = await self._create_sector_pie_chart(pool)
        
        # Industry breakdown bar chart
        analysis["charts"]["industry_bars"] = await self._create_industry_bar_chart(pool)
        
        # Sector performance comparison
        analysis["charts"]["sector_performance"] = await self._create_sector_performance_chart(pool)
        
        # Generate insights
        analysis["insights"] = await self._generate_sector_insights(pool)
        
        return analysis
    
    async def create_risk_distribution_analysis(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Create comprehensive risk distribution analysis across pools"""
        
        risk_analysis = {
            "pool_risk_metrics": {},
            "charts": {},
            "risk_summary": {},
            "recommendations": []
        }
        
        # Collect risk metrics for each pool
        for pool_id in pool_ids:
            analytics = await self.pool_manager.get_pool_analytics(pool_id)
            if "error" not in analytics:
                risk_analysis["pool_risk_metrics"][pool_id] = analytics["risk_analysis"]
        
        # Risk distribution charts
        risk_analysis["charts"]["risk_distribution"] = await self._create_risk_distribution_chart(pool_ids)
        
        # VaR analysis
        risk_analysis["charts"]["var_analysis"] = await self._create_var_analysis_chart(pool_ids)
        
        # Concentration risk
        risk_analysis["charts"]["concentration_risk"] = await self._create_concentration_risk_chart(pool_ids)
        
        # Risk-adjusted returns
        risk_analysis["charts"]["risk_adjusted_returns"] = await self._create_risk_adjusted_returns_chart(pool_ids)
        
        # Generate risk summary and recommendations
        risk_analysis["risk_summary"] = await self._generate_risk_summary(pool_ids)
        risk_analysis["recommendations"] = await self._generate_risk_recommendations(pool_ids)
        
        return risk_analysis
    
    async def create_pool_optimization_recommendations(self, pool_id: str) -> Dict[str, Any]:
        """Create detailed pool optimization recommendations"""
        
        pool = self.pool_manager.pools.get(pool_id)
        if not pool:
            return {"error": "Pool not found"}
        
        optimization = {
            "current_analysis": await self._analyze_current_pool_state(pool),
            "optimization_opportunities": [],
            "rebalancing_suggestions": [],
            "charts": {},
            "action_plan": []
        }
        
        # Identify optimization opportunities
        optimization["optimization_opportunities"] = await self._identify_optimization_opportunities(pool)
        
        # Generate rebalancing suggestions
        optimization["rebalancing_suggestions"] = await self._generate_rebalancing_suggestions(pool)
        
        # Optimization charts
        optimization["charts"]["current_vs_optimal"] = await self._create_current_vs_optimal_chart(pool)
        optimization["charts"]["rebalancing_impact"] = await self._create_rebalancing_impact_chart(pool)
        optimization["charts"]["efficiency_frontier"] = await self._create_efficiency_frontier_chart(pool)
        
        # Create action plan
        optimization["action_plan"] = await self._create_optimization_action_plan(pool)
        
        return optimization
    
    async def _create_performance_overview_chart(self, pool_id: str) -> Dict[str, Any]:
        """Create performance overview chart"""
        
        pool = self.pool_manager.pools[pool_id]
        
        # Create gauge chart for overall performance
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = pool.metrics.total_return * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Total Return (%)"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-50, 50]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-50, -10], 'color': "lightgray"},
                    {'range': [-10, 10], 'color': "gray"},
                    {'range': [10, 50], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        
        fig.update_layout(
            title=f"Performance Overview - {pool.name}",
            height=400
        )
        
        return {
            "chart": fig.to_json(),
            "type": "gauge",
            "title": "Performance Overview"
        }
    
    async def _create_stock_breakdown_chart(self, pool_id: str) -> Dict[str, Any]:
        """Create stock performance breakdown chart"""
        
        pool = self.pool_manager.pools[pool_id]
        
        if not pool.stocks:
            return {"error": "No stocks in pool"}
        
        # Calculate returns for each stock
        symbols = []
        returns = []
        weights = []
        
        for stock in pool.stocks:
            if stock.added_price > 0:
                return_pct = (stock.current_price - stock.added_price) / stock.added_price * 100
                symbols.append(stock.symbol)
                returns.append(return_pct)
                weights.append(stock.weight if stock.weight > 0 else 1.0 / len(pool.stocks))
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        colors = ['green' if r >= 0 else 'red' for r in returns]
        
        fig.add_trace(go.Bar(
            y=symbols,
            x=returns,
            orientation='h',
            marker_color=colors,
            text=[f"{r:.1f}%" for r in returns],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Stock Performance Breakdown",
            xaxis_title="Return (%)",
            yaxis_title="Stocks",
            height=max(400, len(symbols) * 30)
        )
        
        return {
            "chart": fig.to_json(),
            "type": "bar",
            "title": "Stock Performance Breakdown"
        }
    
    async def _create_sector_distribution_chart(self, pool_id: str) -> Dict[str, Any]:
        """Create sector distribution pie chart"""
        
        analytics = await self.pool_manager.get_pool_analytics(pool_id)
        sector_data = analytics["sector_analysis"]["sector_distribution"]
        
        if not sector_data:
            return {"error": "No sector data available"}
        
        fig = go.Figure(data=[go.Pie(
            labels=list(sector_data.keys()),
            values=list(sector_data.values()),
            hole=0.3
        )])
        
        fig.update_layout(
            title="Sector Distribution",
            height=400
        )
        
        return {
            "chart": fig.to_json(),
            "type": "pie",
            "title": "Sector Distribution"
        }
    
    async def _create_risk_analysis_chart(self, pool_id: str) -> Dict[str, Any]:
        """Create risk analysis radar chart"""
        
        analytics = await self.pool_manager.get_pool_analytics(pool_id)
        risk_data = analytics["risk_analysis"]
        
        # Risk metrics for radar chart
        metrics = [
            "Concentration Risk",
            "Volatility",
            "Sector Risk",
            "Liquidity Risk",
            "Overall Risk"
        ]
        
        values = [
            risk_data.get("concentration_risk", 0) * 100,
            analytics["performance_metrics"]["volatility"] * 100,
            50,  # Mock sector risk
            30,  # Mock liquidity risk
            risk_data.get("risk_score", 0)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name='Risk Profile'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Risk Analysis Profile",
            height=400
        )
        
        return {
            "chart": fig.to_json(),
            "type": "radar",
            "title": "Risk Analysis Profile"
        }
    
    async def _create_performance_timeline_chart(self, pool_id: str) -> Dict[str, Any]:
        """Create performance timeline chart"""
        
        # Mock timeline data - in real implementation, would use historical data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        
        # Generate mock performance data
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        cumulative_returns = (1 + pd.Series(returns)).cumprod() - 1
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_returns * 100,
            mode='lines',
            name='Pool Performance',
            line=dict(color='blue', width=2)
        ))
        
        # Add benchmark (mock)
        benchmark_returns = np.random.normal(0.0003, 0.015, len(dates))
        benchmark_cumulative = (1 + pd.Series(benchmark_returns)).cumprod() - 1
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_cumulative * 100,
            mode='lines',
            name='Benchmark',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title="Performance Timeline",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=400
        )
        
        return {
            "chart": fig.to_json(),
            "type": "line",
            "title": "Performance Timeline"
        }
    
    async def _create_performance_comparison_chart(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Create performance comparison chart for multiple pools"""
        
        fig = go.Figure()
        
        for i, pool_id in enumerate(pool_ids):
            pool = self.pool_manager.pools[pool_id]
            
            # Mock performance data for each pool
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            np.random.seed(42 + i)
            returns = np.random.normal(0.0005 + i*0.0001, 0.02, len(dates))
            cumulative_returns = (1 + pd.Series(returns)).cumprod() - 1
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns * 100,
                mode='lines',
                name=pool.name,
                line=dict(color=self.color_palette[i % len(self.color_palette)])
            ))
        
        fig.update_layout(
            title="Pool Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=500
        )
        
        return {
            "chart": fig.to_json(),
            "type": "line",
            "title": "Pool Performance Comparison"
        }
    
    async def _create_risk_return_scatter(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Create risk-return scatter plot"""
        
        fig = go.Figure()
        
        pool_names = []
        returns = []
        risks = []
        
        for pool_id in pool_ids:
            pool = self.pool_manager.pools[pool_id]
            pool_names.append(pool.name)
            returns.append(pool.metrics.total_return * 100)
            risks.append(pool.metrics.volatility * 100)
        
        fig.add_trace(go.Scatter(
            x=risks,
            y=returns,
            mode='markers+text',
            text=pool_names,
            textposition="top center",
            marker=dict(
                size=12,
                color=returns,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Return (%)")
            )
        ))
        
        fig.update_layout(
            title="Risk-Return Analysis",
            xaxis_title="Risk (Volatility %)",
            yaxis_title="Return (%)",
            height=500
        )
        
        return {
            "chart": fig.to_json(),
            "type": "scatter",
            "title": "Risk-Return Analysis"
        }
    
    async def _analyze_sector_breakdown(self, pool: StockPool) -> Dict[str, Any]:
        """Analyze sector breakdown for a pool"""
        
        # Mock sector analysis
        sectors = ["Technology", "Healthcare", "Finance", "Consumer", "Industrial"]
        sector_weights = {}
        sector_performance = {}
        
        for i, stock in enumerate(pool.stocks):
            sector = sectors[i % len(sectors)]
            weight = stock.weight if stock.weight > 0 else 1.0 / len(pool.stocks)
            
            if sector not in sector_weights:
                sector_weights[sector] = 0.0
                sector_performance[sector] = []
            
            sector_weights[sector] += weight
            
            if stock.added_price > 0:
                return_pct = (stock.current_price - stock.added_price) / stock.added_price
                sector_performance[sector].append(return_pct)
        
        # Calculate average performance per sector
        sector_avg_performance = {}
        for sector, performances in sector_performance.items():
            sector_avg_performance[sector] = np.mean(performances) if performances else 0.0
        
        return {
            "sector_weights": sector_weights,
            "sector_performance": sector_avg_performance,
            "diversification_score": len(sector_weights) / len(sectors) * 100
        }
    
    async def _analyze_industry_breakdown(self, pool: StockPool) -> Dict[str, Any]:
        """Analyze industry breakdown for a pool"""
        
        # Mock industry analysis
        industries = [
            "Software", "Biotechnology", "Banking", "Retail", "Manufacturing",
            "Semiconductors", "Pharmaceuticals", "Insurance", "Automotive", "Energy"
        ]
        
        industry_weights = {}
        
        for i, stock in enumerate(pool.stocks):
            industry = industries[i % len(industries)]
            weight = stock.weight if stock.weight > 0 else 1.0 / len(pool.stocks)
            
            if industry not in industry_weights:
                industry_weights[industry] = 0.0
            
            industry_weights[industry] += weight
        
        return {
            "industry_weights": industry_weights,
            "industry_count": len(industry_weights)
        }
    
    async def _generate_comparison_summary(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Generate summary for pool comparison"""
        
        summary = {
            "best_performer": None,
            "lowest_risk": None,
            "highest_sharpe": None,
            "most_diversified": None,
            "total_pools": len(pool_ids)
        }
        
        best_return = float('-inf')
        lowest_risk = float('inf')
        highest_sharpe = float('-inf')
        most_stocks = 0
        
        for pool_id in pool_ids:
            pool = self.pool_manager.pools[pool_id]
            
            if pool.metrics.total_return > best_return:
                best_return = pool.metrics.total_return
                summary["best_performer"] = {"pool_id": pool_id, "name": pool.name, "return": best_return}
            
            if pool.metrics.volatility < lowest_risk:
                lowest_risk = pool.metrics.volatility
                summary["lowest_risk"] = {"pool_id": pool_id, "name": pool.name, "risk": lowest_risk}
            
            if pool.metrics.sharpe_ratio > highest_sharpe:
                highest_sharpe = pool.metrics.sharpe_ratio
                summary["highest_sharpe"] = {"pool_id": pool_id, "name": pool.name, "sharpe": highest_sharpe}
            
            if len(pool.stocks) > most_stocks:
                most_stocks = len(pool.stocks)
                summary["most_diversified"] = {"pool_id": pool_id, "name": pool.name, "stocks": most_stocks}
        
        return summary
    
    async def _identify_optimization_opportunities(self, pool: StockPool) -> List[Dict[str, Any]]:
        """Identify optimization opportunities for a pool"""
        
        opportunities = []
        
        # Check diversification
        if len(pool.stocks) < 10:
            opportunities.append({
                "type": "diversification",
                "priority": "high",
                "description": "Pool has low diversification. Consider adding more stocks.",
                "impact": "Reduce concentration risk",
                "effort": "medium"
            })
        
        # Check sector concentration
        sector_analysis = await self._analyze_sector_breakdown(pool)
        max_sector_weight = max(sector_analysis["sector_weights"].values()) if sector_analysis["sector_weights"] else 0
        
        if max_sector_weight > 0.4:  # More than 40% in one sector
            opportunities.append({
                "type": "sector_rebalancing",
                "priority": "medium",
                "description": "High concentration in one sector detected.",
                "impact": "Improve sector diversification",
                "effort": "low"
            })
        
        # Check performance
        poor_performers = [
            stock for stock in pool.stocks
            if stock.added_price > 0 and (stock.current_price - stock.added_price) / stock.added_price < -0.2
        ]
        
        if poor_performers:
            opportunities.append({
                "type": "performance_cleanup",
                "priority": "medium",
                "description": f"{len(poor_performers)} stocks are underperforming by more than 20%.",
                "impact": "Improve overall pool performance",
                "effort": "low"
            })
        
        return opportunities
    
    async def _generate_rebalancing_suggestions(self, pool: StockPool) -> List[Dict[str, Any]]:
        """Generate rebalancing suggestions"""
        
        suggestions = []
        
        if not pool.stocks:
            return suggestions
        
        # Equal weight suggestion
        equal_weight = 1.0 / len(pool.stocks)
        suggestions.append({
            "strategy": "equal_weight",
            "description": "Rebalance to equal weights across all stocks",
            "target_weights": {stock.symbol: equal_weight for stock in pool.stocks},
            "expected_impact": "Reduce concentration risk"
        })
        
        # Performance-based weighting
        performance_weights = {}
        total_positive_return = 0
        
        for stock in pool.stocks:
            if stock.added_price > 0:
                return_pct = (stock.current_price - stock.added_price) / stock.added_price
                if return_pct > 0:
                    total_positive_return += return_pct
        
        if total_positive_return > 0:
            for stock in pool.stocks:
                if stock.added_price > 0:
                    return_pct = (stock.current_price - stock.added_price) / stock.added_price
                    weight = max(0.01, return_pct / total_positive_return) if return_pct > 0 else 0.01
                    performance_weights[stock.symbol] = weight
            
            # Normalize weights
            total_weight = sum(performance_weights.values())
            performance_weights = {k: v/total_weight for k, v in performance_weights.items()}
            
            suggestions.append({
                "strategy": "performance_weighted",
                "description": "Weight stocks based on their performance",
                "target_weights": performance_weights,
                "expected_impact": "Increase allocation to better performers"
            })
        
        return suggestions
    
    async def _create_optimization_action_plan(self, pool: StockPool) -> List[Dict[str, Any]]:
        """Create actionable optimization plan"""
        
        action_plan = []
        
        # Immediate actions
        action_plan.append({
            "phase": "immediate",
            "timeframe": "1-2 days",
            "actions": [
                "Review and remove stocks with >30% losses",
                "Update stock prices and recalculate metrics",
                "Identify sector overconcentration"
            ]
        })
        
        # Short-term actions
        action_plan.append({
            "phase": "short_term",
            "timeframe": "1-2 weeks",
            "actions": [
                "Research and add stocks to improve diversification",
                "Rebalance weights based on performance analysis",
                "Set up automated monitoring rules"
            ]
        })
        
        # Long-term actions
        action_plan.append({
            "phase": "long_term",
            "timeframe": "1-3 months",
            "actions": [
                "Implement systematic rebalancing schedule",
                "Develop custom screening criteria for pool updates",
                "Monitor and adjust risk parameters"
            ]
        })
        
        return action_plan
    
    async def _generate_risk_summary(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Generate risk summary across pools"""
        
        risk_summary = {
            "overall_risk_level": "medium",
            "highest_risk_pool": None,
            "lowest_risk_pool": None,
            "risk_distribution": {},
            "recommendations": []
        }
        
        risk_levels = []
        
        for pool_id in pool_ids:
            analytics = await self.pool_manager.get_pool_analytics(pool_id)
            if "error" not in analytics:
                risk_score = analytics["risk_analysis"].get("risk_score", 50)
                risk_levels.append((pool_id, risk_score))
        
        if risk_levels:
            risk_levels.sort(key=lambda x: x[1])
            
            risk_summary["lowest_risk_pool"] = {
                "pool_id": risk_levels[0][0],
                "risk_score": risk_levels[0][1]
            }
            
            risk_summary["highest_risk_pool"] = {
                "pool_id": risk_levels[-1][0],
                "risk_score": risk_levels[-1][1]
            }
            
            avg_risk = np.mean([r[1] for r in risk_levels])
            if avg_risk < 30:
                risk_summary["overall_risk_level"] = "low"
            elif avg_risk > 70:
                risk_summary["overall_risk_level"] = "high"
        
        return risk_summary
    
    async def _generate_risk_recommendations(self, pool_ids: List[str]) -> List[Dict[str, Any]]:
        """Generate risk-based recommendations"""
        
        recommendations = []
        
        for pool_id in pool_ids:
            analytics = await self.pool_manager.get_pool_analytics(pool_id)
            if "error" not in analytics:
                risk_score = analytics["risk_analysis"].get("risk_score", 50)
                pool_name = analytics["basic_info"]["name"]
                
                if risk_score > 80:
                    recommendations.append({
                        "pool_id": pool_id,
                        "pool_name": pool_name,
                        "type": "high_risk_warning",
                        "message": f"Pool '{pool_name}' has high risk score ({risk_score:.1f}). Consider diversification.",
                        "priority": "high"
                    })
                elif risk_score < 20:
                    recommendations.append({
                        "pool_id": pool_id,
                        "pool_name": pool_name,
                        "type": "low_risk_opportunity",
                        "message": f"Pool '{pool_name}' has very low risk. Consider adding growth opportunities.",
                        "priority": "low"
                    })
        
        return recommendations
    
    # Missing chart methods
    async def _create_sector_pie_chart(self, pool: 'StockPool') -> Dict[str, Any]:
        """Create sector allocation pie chart"""
        
        sector_analysis = await self._analyze_sector_breakdown(pool)
        sector_data = sector_analysis["sector_weights"]
        
        if not sector_data:
            return {"error": "No sector data available"}
        
        fig = go.Figure(data=[go.Pie(
            labels=list(sector_data.keys()),
            values=list(sector_data.values()),
            hole=0.3
        )])
        
        fig.update_layout(
            title="Sector Allocation",
            height=400
        )
        
        return {
            "chart": fig.to_json(),
            "type": "pie",
            "title": "Sector Allocation"
        }
    
    async def _create_industry_bar_chart(self, pool: 'StockPool') -> Dict[str, Any]:
        """Create industry breakdown bar chart"""
        
        industry_analysis = await self._analyze_industry_breakdown(pool)
        industry_data = industry_analysis["industry_weights"]
        
        if not industry_data:
            return {"error": "No industry data available"}
        
        fig = go.Figure(data=[go.Bar(
            x=list(industry_data.keys()),
            y=list(industry_data.values()),
            marker_color='lightblue'
        )])
        
        fig.update_layout(
            title="Industry Breakdown",
            xaxis_title="Industry",
            yaxis_title="Weight",
            height=400
        )
        
        return {
            "chart": fig.to_json(),
            "type": "bar",
            "title": "Industry Breakdown"
        }
    
    async def _create_sector_performance_chart(self, pool: 'StockPool') -> Dict[str, Any]:
        """Create sector performance comparison chart"""
        
        sector_analysis = await self._analyze_sector_breakdown(pool)
        sector_performance = sector_analysis["sector_performance"]
        
        if not sector_performance:
            return {"error": "No sector performance data available"}
        
        fig = go.Figure(data=[go.Bar(
            x=list(sector_performance.keys()),
            y=[v * 100 for v in sector_performance.values()],
            marker_color=['green' if v >= 0 else 'red' for v in sector_performance.values()]
        )])
        
        fig.update_layout(
            title="Sector Performance Comparison",
            xaxis_title="Sector",
            yaxis_title="Return (%)",
            height=400
        )
        
        return {
            "chart": fig.to_json(),
            "type": "bar",
            "title": "Sector Performance Comparison"
        }
    
    async def _create_correlation_heatmap(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Create correlation heatmap for multiple pools"""
        
        # Mock correlation data
        pool_names = [self.pool_manager.pools[pid].name for pid in pool_ids]
        correlation_matrix = np.random.uniform(0.3, 0.9, (len(pool_ids), len(pool_ids)))
        np.fill_diagonal(correlation_matrix, 1.0)
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=pool_names,
            y=pool_names,
            colorscale='RdYlBu_r',
            zmin=0,
            zmax=1
        ))
        
        fig.update_layout(
            title="Pool Correlation Heatmap",
            height=500
        )
        
        return {
            "chart": fig.to_json(),
            "type": "heatmap",
            "title": "Pool Correlation Heatmap"
        }
    
    async def _create_sector_comparison_chart(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Create sector allocation comparison chart"""
        
        fig = go.Figure()
        
        for i, pool_id in enumerate(pool_ids):
            pool = self.pool_manager.pools[pool_id]
            sector_analysis = await self._analyze_sector_breakdown(pool)
            sector_data = sector_analysis["sector_weights"]
            
            if sector_data:
                fig.add_trace(go.Bar(
                    name=pool.name,
                    x=list(sector_data.keys()),
                    y=list(sector_data.values()),
                    marker_color=self.color_palette[i % len(self.color_palette)]
                ))
        
        fig.update_layout(
            title="Sector Allocation Comparison",
            xaxis_title="Sector",
            yaxis_title="Weight",
            barmode='group',
            height=500
        )
        
        return {
            "chart": fig.to_json(),
            "type": "bar",
            "title": "Sector Allocation Comparison"
        }
    
    async def _create_risk_distribution_chart(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Create risk distribution chart"""
        
        pool_names = []
        risk_scores = []
        
        for pool_id in pool_ids:
            pool = self.pool_manager.pools[pool_id]
            analytics = await self.pool_manager.get_pool_analytics(pool_id)
            
            pool_names.append(pool.name)
            risk_scores.append(analytics["risk_analysis"].get("risk_score", 50))
        
        fig = go.Figure(data=[go.Bar(
            x=pool_names,
            y=risk_scores,
            marker_color=['red' if r > 70 else 'yellow' if r > 40 else 'green' for r in risk_scores]
        )])
        
        fig.update_layout(
            title="Risk Distribution Across Pools",
            xaxis_title="Pool",
            yaxis_title="Risk Score",
            height=400
        )
        
        return {
            "chart": fig.to_json(),
            "type": "bar",
            "title": "Risk Distribution"
        }
    
    async def _create_var_analysis_chart(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Create VaR analysis chart"""
        
        pool_names = []
        var_values = []
        
        for pool_id in pool_ids:
            pool = self.pool_manager.pools[pool_id]
            pool_names.append(pool.name)
            # Mock VaR calculation
            var_values.append(np.random.uniform(0.02, 0.08))
        
        fig = go.Figure(data=[go.Bar(
            x=pool_names,
            y=[v * 100 for v in var_values],
            marker_color='orange'
        )])
        
        fig.update_layout(
            title="Value at Risk (VaR) Analysis",
            xaxis_title="Pool",
            yaxis_title="VaR (%)",
            height=400
        )
        
        return {
            "chart": fig.to_json(),
            "type": "bar",
            "title": "VaR Analysis"
        }
    
    async def _create_concentration_risk_chart(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Create concentration risk chart"""
        
        pool_names = []
        concentration_scores = []
        
        for pool_id in pool_ids:
            pool = self.pool_manager.pools[pool_id]
            analytics = await self.pool_manager.get_pool_analytics(pool_id)
            
            pool_names.append(pool.name)
            concentration_scores.append(analytics["risk_analysis"].get("concentration_risk", 0.5) * 100)
        
        fig = go.Figure(data=[go.Bar(
            x=pool_names,
            y=concentration_scores,
            marker_color='purple'
        )])
        
        fig.update_layout(
            title="Concentration Risk Analysis",
            xaxis_title="Pool",
            yaxis_title="Concentration Risk (%)",
            height=400
        )
        
        return {
            "chart": fig.to_json(),
            "type": "bar",
            "title": "Concentration Risk"
        }
    
    async def _create_risk_adjusted_returns_chart(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Create risk-adjusted returns chart"""
        
        pool_names = []
        sharpe_ratios = []
        
        for pool_id in pool_ids:
            pool = self.pool_manager.pools[pool_id]
            pool_names.append(pool.name)
            sharpe_ratios.append(pool.metrics.sharpe_ratio)
        
        fig = go.Figure(data=[go.Bar(
            x=pool_names,
            y=sharpe_ratios,
            marker_color=['green' if s > 1 else 'yellow' if s > 0 else 'red' for s in sharpe_ratios]
        )])
        
        fig.update_layout(
            title="Risk-Adjusted Returns (Sharpe Ratio)",
            xaxis_title="Pool",
            yaxis_title="Sharpe Ratio",
            height=400
        )
        
        return {
            "chart": fig.to_json(),
            "type": "bar",
            "title": "Risk-Adjusted Returns"
        }
    
    async def _create_current_vs_optimal_chart(self, pool: 'StockPool') -> Dict[str, Any]:
        """Create current vs optimal allocation chart"""
        
        # Mock optimal weights
        current_weights = [stock.weight if stock.weight > 0 else 1.0/len(pool.stocks) for stock in pool.stocks]
        optimal_weights = [w * np.random.uniform(0.8, 1.2) for w in current_weights]
        
        # Normalize optimal weights
        total_optimal = sum(optimal_weights)
        optimal_weights = [w/total_optimal for w in optimal_weights]
        
        symbols = [stock.symbol for stock in pool.stocks]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current',
            x=symbols,
            y=current_weights,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Optimal',
            x=symbols,
            y=optimal_weights,
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title="Current vs Optimal Allocation",
            xaxis_title="Stock",
            yaxis_title="Weight",
            barmode='group',
            height=400
        )
        
        return {
            "chart": fig.to_json(),
            "type": "bar",
            "title": "Current vs Optimal Allocation"
        }
    
    async def _create_rebalancing_impact_chart(self, pool: 'StockPool') -> Dict[str, Any]:
        """Create rebalancing impact chart"""
        
        # Mock rebalancing impact data
        metrics = ['Return', 'Risk', 'Sharpe Ratio', 'Diversification']
        current_values = [100, 100, 100, 100]
        rebalanced_values = [105, 95, 110, 115]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current',
            x=metrics,
            y=current_values,
            marker_color='lightcoral'
        ))
        
        fig.add_trace(go.Bar(
            name='After Rebalancing',
            x=metrics,
            y=rebalanced_values,
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title="Rebalancing Impact Analysis",
            xaxis_title="Metric",
            yaxis_title="Score (Indexed to 100)",
            barmode='group',
            height=400
        )
        
        return {
            "chart": fig.to_json(),
            "type": "bar",
            "title": "Rebalancing Impact"
        }
    
    async def _create_efficiency_frontier_chart(self, pool: 'StockPool') -> Dict[str, Any]:
        """Create efficiency frontier chart"""
        
        # Mock efficiency frontier data
        risk_values = np.linspace(0.05, 0.25, 50)
        return_values = np.sqrt(risk_values) * 0.4 + np.random.normal(0, 0.01, 50)
        
        # Current pool position
        current_risk = pool.metrics.volatility
        current_return = pool.metrics.total_return
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=risk_values * 100,
            y=return_values * 100,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[current_risk * 100],
            y=[current_return * 100],
            mode='markers',
            name='Current Pool',
            marker=dict(color='red', size=10)
        ))
        
        fig.update_layout(
            title="Efficiency Frontier Analysis",
            xaxis_title="Risk (Volatility %)",
            yaxis_title="Expected Return (%)",
            height=400
        )
        
        return {
            "chart": fig.to_json(),
            "type": "scatter",
            "title": "Efficiency Frontier"
        }
    
    async def _generate_sector_insights(self, pool: 'StockPool') -> List[str]:
        """Generate sector-based insights"""
        
        insights = []
        sector_analysis = await self._analyze_sector_breakdown(pool)
        
        # Check diversification
        diversification_score = sector_analysis.get("diversification_score", 0)
        if diversification_score < 50:
            insights.append("Pool has low sector diversification. Consider adding stocks from different sectors.")
        elif diversification_score > 80:
            insights.append("Pool has excellent sector diversification across multiple industries.")
        
        # Check dominant sector
        dominant_sector = sector_analysis.get("dominant_sector")
        if dominant_sector and dominant_sector[1] > 0.5:
            insights.append(f"Pool is heavily concentrated in {dominant_sector[0]} sector ({dominant_sector[1]:.1%}). Consider rebalancing.")
        
        return insights
    
    async def _analyze_current_pool_state(self, pool: 'StockPool') -> Dict[str, Any]:
        """Analyze current pool state for optimization"""
        
        return {
            "total_stocks": len(pool.stocks),
            "total_return": pool.metrics.total_return,
            "volatility": pool.metrics.volatility,
            "sharpe_ratio": pool.metrics.sharpe_ratio,
            "diversification_level": "medium" if len(pool.stocks) > 5 else "low",
            "risk_level": "high" if pool.metrics.volatility > 0.2 else "medium" if pool.metrics.volatility > 0.1 else "low",
            "last_updated": pool.last_modified.isoformat()
        }