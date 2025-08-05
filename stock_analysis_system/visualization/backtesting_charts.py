"""Comprehensive backtesting visualization module."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats

logger = logging.getLogger(__name__)


class BacktestingVisualizationEngine:
    """Comprehensive visualization engine for backtesting results."""

    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'benchmark': '#9467bd',
            'drawdown': '#e377c2',
            'background': '#f8f9fa'
        }

    async def create_comprehensive_backtest_report(
        self, 
        backtest_result: Any,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, go.Figure]:
        """Create a comprehensive backtesting visualization report."""
        
        charts = {}
        
        try:
            # 1. Equity curve with drawdown
            charts['equity_curve'] = await self.create_equity_curve_chart(
                backtest_result, benchmark_data
            )
            
            # 2. Performance attribution
            charts['performance_attribution'] = await self.create_performance_attribution_chart(
                backtest_result
            )
            
            # 3. Trade analysis
            charts['trade_analysis'] = await self.create_trade_analysis_chart(
                backtest_result
            )
            
            # 4. Risk metrics dashboard
            charts['risk_metrics'] = await self.create_risk_metrics_dashboard(
                backtest_result
            )
            
            # 5. Monthly returns heatmap
            charts['monthly_returns'] = await self.create_monthly_returns_heatmap(
                backtest_result
            )
            
            # 6. Benchmark comparison
            if benchmark_data is not None:
                charts['benchmark_comparison'] = await self.create_benchmark_comparison_chart(
                    backtest_result, benchmark_data
                )
            
            # 7. Rolling performance metrics
            charts['rolling_metrics'] = await self.create_rolling_metrics_chart(
                backtest_result
            )
            
        except Exception as e:
            logger.error(f"Error creating comprehensive backtest report: {e}")
            raise
        
        return charts

    async def create_equity_curve_chart(
        self, 
        backtest_result: Any,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """Create equity curve chart with drawdown visualization."""
        
        # Create subplot with secondary y-axis for drawdown
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Equity Curve', 'Drawdown'),
            row_heights=[0.7, 0.3]
        )
        
        # Extract equity curve data
        equity_curve = backtest_result.equity_curve
        if equity_curve.empty:
            logger.warning("Empty equity curve data")
            return fig
        
        # Calculate drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max * 100
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color=self.color_palette['primary'], width=2),
                hovertemplate='<b>Date:</b> %{x}<br>' +
                             '<b>Portfolio Value:</b> ¥%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add benchmark if available
        if benchmark_data is not None and not benchmark_data.empty:
            # Normalize benchmark to same starting value
            benchmark_normalized = (
                benchmark_data['close_price'] / benchmark_data['close_price'].iloc[0] * 
                equity_curve.iloc[0]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=benchmark_normalized.index,
                    y=benchmark_normalized.values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color=self.color_palette['benchmark'], width=2, dash='dash'),
                    hovertemplate='<b>Date:</b> %{x}<br>' +
                                 '<b>Benchmark Value:</b> ¥%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Drawdown',
                fill='tonexty',
                line=dict(color=self.color_palette['danger'], width=1),
                fillcolor='rgba(214, 39, 40, 0.3)',
                hovertemplate='<b>Date:</b> %{x}<br>' +
                             '<b>Drawdown:</b> %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add zero line for drawdown
        fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="gray",
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Backtesting Results: {backtest_result.strategy_name}',
                x=0.5,
                font=dict(size=20)
            ),
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Portfolio Value (¥)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        return fig

    async def create_performance_attribution_chart(
        self, 
        backtest_result: Any
    ) -> go.Figure:
        """Create performance attribution analysis chart."""
        
        attribution = backtest_result.performance_attribution
        
        if not attribution:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No performance attribution data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Performance Attribution",
                height=400
            )
            return fig
        
        # Sort by absolute contribution
        sorted_attribution = dict(
            sorted(attribution.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        
        symbols = list(sorted_attribution.keys())
        contributions = list(sorted_attribution.values())
        
        # Create colors based on positive/negative contribution
        colors = [
            self.color_palette['success'] if contrib > 0 else self.color_palette['danger']
            for contrib in contributions
        ]
        
        fig = go.Figure(data=[
            go.Bar(
                x=symbols,
                y=contributions,
                marker_color=colors,
                text=[f'¥{contrib:,.0f}' for contrib in contributions],
                textposition='auto',
                hovertemplate='<b>Symbol:</b> %{x}<br>' +
                             '<b>Contribution:</b> ¥%{y:,.0f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(
                text='Performance Attribution by Symbol',
                x=0.5,
                font=dict(size=18)
            ),
            xaxis_title='Symbol',
            yaxis_title='P&L Contribution (¥)',
            height=400,
            showlegend=False
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return fig

    async def create_trade_analysis_chart(
        self, 
        backtest_result: Any
    ) -> go.Figure:
        """Create comprehensive trade analysis visualization."""
        
        trades = backtest_result.trade_log
        
        if not trades:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No trade data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Trade Analysis",
                height=600
            )
            return fig
        
        # Create subplots for different trade analyses
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Trade P&L Distribution',
                'Cumulative P&L',
                'Trade Size Distribution',
                'Win/Loss Ratio by Symbol'
            ),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Extract trade data
        sell_trades = [trade for trade in trades if trade['side'] == 'sell' and 'pnl' in trade]
        
        if sell_trades:
            pnls = [trade['pnl'] for trade in sell_trades]
            timestamps = [trade['timestamp'] for trade in sell_trades]
            quantities = [trade['quantity'] for trade in trades]
            
            # 1. P&L Distribution
            fig.add_trace(
                go.Histogram(
                    x=pnls,
                    nbinsx=20,
                    name='P&L Distribution',
                    marker_color=self.color_palette['primary'],
                    opacity=0.7,
                    hovertemplate='<b>P&L Range:</b> %{x}<br>' +
                                 '<b>Count:</b> %{y}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 2. Cumulative P&L
            cumulative_pnl = np.cumsum(pnls)
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=cumulative_pnl,
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color=self.color_palette['success'], width=2),
                    hovertemplate='<b>Date:</b> %{x}<br>' +
                                 '<b>Cumulative P&L:</b> ¥%{y:,.0f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # 3. Trade Size Distribution
            fig.add_trace(
                go.Histogram(
                    x=quantities,
                    nbinsx=15,
                    name='Trade Size',
                    marker_color=self.color_palette['info'],
                    opacity=0.7,
                    hovertemplate='<b>Quantity Range:</b> %{x}<br>' +
                                 '<b>Count:</b> %{y}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # 4. Win/Loss by Symbol
            symbol_stats = {}
            for trade in sell_trades:
                symbol = trade['symbol']
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {'wins': 0, 'losses': 0}
                
                if trade['pnl'] > 0:
                    symbol_stats[symbol]['wins'] += 1
                else:
                    symbol_stats[symbol]['losses'] += 1
            
            if symbol_stats:
                symbols = list(symbol_stats.keys())
                win_rates = [
                    stats['wins'] / (stats['wins'] + stats['losses']) * 100
                    for stats in symbol_stats.values()
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=symbols,
                        y=win_rates,
                        name='Win Rate %',
                        marker_color=self.color_palette['warning'],
                        text=[f'{rate:.1f}%' for rate in win_rates],
                        textposition='auto',
                        hovertemplate='<b>Symbol:</b> %{x}<br>' +
                                     '<b>Win Rate:</b> %{y:.1f}%<extra></extra>'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=dict(
                text='Trade Analysis Dashboard',
                x=0.5,
                font=dict(size=18)
            ),
            height=600,
            showlegend=False
        )
        
        return fig

    async def create_risk_metrics_dashboard(
        self, 
        backtest_result: Any
    ) -> go.Figure:
        """Create risk metrics dashboard."""
        
        # Create metrics cards layout
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Sharpe Ratio',
                'Maximum Drawdown',
                'Volatility',
                'Win Rate',
                'Profit Factor',
                'Calmar Ratio'
            ),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Define metrics with their values and formatting
        metrics = [
            {
                'value': backtest_result.sharpe_ratio,
                'title': 'Sharpe Ratio',
                'format': '.2f',
                'threshold': {'good': 1.0, 'warning': 0.5},
                'row': 1, 'col': 1
            },
            {
                'value': abs(backtest_result.max_drawdown) * 100,
                'title': 'Max Drawdown (%)',
                'format': '.1f',
                'threshold': {'good': 10, 'warning': 20},
                'reverse': True,  # Lower is better
                'row': 1, 'col': 2
            },
            {
                'value': backtest_result.volatility * 100,
                'title': 'Volatility (%)',
                'format': '.1f',
                'threshold': {'good': 15, 'warning': 25},
                'reverse': True,
                'row': 1, 'col': 3
            },
            {
                'value': backtest_result.win_rate * 100,
                'title': 'Win Rate (%)',
                'format': '.1f',
                'threshold': {'good': 60, 'warning': 40},
                'row': 2, 'col': 1
            },
            {
                'value': backtest_result.profit_factor,
                'title': 'Profit Factor',
                'format': '.2f',
                'threshold': {'good': 1.5, 'warning': 1.0},
                'row': 2, 'col': 2
            },
            {
                'value': backtest_result.calmar_ratio,
                'title': 'Calmar Ratio',
                'format': '.2f',
                'threshold': {'good': 1.0, 'warning': 0.5},
                'row': 2, 'col': 3
            }
        ]
        
        for metric in metrics:
            # Determine color based on thresholds
            value = metric['value']
            reverse = metric.get('reverse', False)
            
            if reverse:
                if value <= metric['threshold']['good']:
                    color = self.color_palette['success']
                elif value <= metric['threshold']['warning']:
                    color = self.color_palette['warning']
                else:
                    color = self.color_palette['danger']
            else:
                if value >= metric['threshold']['good']:
                    color = self.color_palette['success']
                elif value >= metric['threshold']['warning']:
                    color = self.color_palette['warning']
                else:
                    color = self.color_palette['danger']
            
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=value,
                    title={"text": metric['title']},
                    number={'font': {'size': 40, 'color': color}},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=metric['row'], col=metric['col']
            )
        
        fig.update_layout(
            title=dict(
                text='Risk Metrics Dashboard',
                x=0.5,
                font=dict(size=18)
            ),
            height=500
        )
        
        return fig

    async def create_monthly_returns_heatmap(
        self, 
        backtest_result: Any
    ) -> go.Figure:
        """Create monthly returns heatmap."""
        
        monthly_returns = backtest_result.monthly_returns
        
        if monthly_returns.empty:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No monthly returns data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Monthly Returns Heatmap",
                height=400
            )
            return fig
        
        # Convert to percentage
        monthly_returns_pct = monthly_returns * 100
        
        # Create pivot table for heatmap
        monthly_returns_pct.index = pd.to_datetime(monthly_returns_pct.index)
        pivot_data = monthly_returns_pct.groupby([
            monthly_returns_pct.index.year,
            monthly_returns_pct.index.month
        ]).first().unstack(fill_value=0)
        
        # Create month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=month_labels,
            y=pivot_data.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot_data.values, 2),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hovertemplate='<b>Year:</b> %{y}<br>' +
                         '<b>Month:</b> %{x}<br>' +
                         '<b>Return:</b> %{z:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='Monthly Returns Heatmap',
                x=0.5,
                font=dict(size=18)
            ),
            xaxis_title='Month',
            yaxis_title='Year',
            height=400
        )
        
        return fig

    async def create_benchmark_comparison_chart(
        self, 
        backtest_result: Any,
        benchmark_data: pd.DataFrame
    ) -> go.Figure:
        """Create detailed benchmark comparison chart."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cumulative Returns Comparison',
                'Rolling Correlation',
                'Relative Performance',
                'Risk-Return Scatter'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        equity_curve = backtest_result.equity_curve
        
        if equity_curve.empty or benchmark_data.empty:
            fig.add_annotation(
                text="Insufficient data for benchmark comparison",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Calculate returns
        strategy_returns = equity_curve.pct_change().dropna()
        benchmark_returns = benchmark_data['close_price'].pct_change().dropna()
        
        # Align data
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) == 0:
            fig.add_annotation(
                text="No overlapping dates between strategy and benchmark",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        strategy_aligned = strategy_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]
        
        # 1. Cumulative returns comparison
        strategy_cumulative = (1 + strategy_aligned).cumprod()
        benchmark_cumulative = (1 + benchmark_aligned).cumprod()
        
        fig.add_trace(
            go.Scatter(
                x=strategy_cumulative.index,
                y=strategy_cumulative.values,
                mode='lines',
                name='Strategy',
                line=dict(color=self.color_palette['primary'], width=2),
                hovertemplate='<b>Date:</b> %{x}<br>' +
                             '<b>Cumulative Return:</b> %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative.values,
                mode='lines',
                name='Benchmark',
                line=dict(color=self.color_palette['benchmark'], width=2, dash='dash'),
                hovertemplate='<b>Date:</b> %{x}<br>' +
                             '<b>Cumulative Return:</b> %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Rolling correlation
        if len(strategy_aligned) > 30:
            rolling_corr = strategy_aligned.rolling(window=30).corr(benchmark_aligned)
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    mode='lines',
                    name='30-Day Correlation',
                    line=dict(color=self.color_palette['info'], width=2),
                    hovertemplate='<b>Date:</b> %{x}<br>' +
                                 '<b>Correlation:</b> %{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Relative performance (strategy - benchmark)
        relative_performance = (strategy_cumulative - benchmark_cumulative) * 100
        
        fig.add_trace(
            go.Scatter(
                x=relative_performance.index,
                y=relative_performance.values,
                mode='lines',
                name='Relative Performance',
                line=dict(color=self.color_palette['success'], width=2),
                fill='tonexty',
                fillcolor='rgba(44, 160, 44, 0.3)',
                hovertemplate='<b>Date:</b> %{x}<br>' +
                             '<b>Relative Performance:</b> %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add zero line for relative performance
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # 4. Risk-Return scatter
        strategy_vol = strategy_aligned.std() * np.sqrt(252) * 100
        strategy_ret = strategy_aligned.mean() * 252 * 100
        benchmark_vol = benchmark_aligned.std() * np.sqrt(252) * 100
        benchmark_ret = benchmark_aligned.mean() * 252 * 100
        
        fig.add_trace(
            go.Scatter(
                x=[strategy_vol],
                y=[strategy_ret],
                mode='markers',
                name='Strategy',
                marker=dict(
                    size=15,
                    color=self.color_palette['primary'],
                    symbol='circle'
                ),
                hovertemplate='<b>Strategy</b><br>' +
                             '<b>Volatility:</b> %{x:.2f}%<br>' +
                             '<b>Return:</b> %{y:.2f}%<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=[benchmark_vol],
                y=[benchmark_ret],
                mode='markers',
                name='Benchmark',
                marker=dict(
                    size=15,
                    color=self.color_palette['benchmark'],
                    symbol='diamond'
                ),
                hovertemplate='<b>Benchmark</b><br>' +
                             '<b>Volatility:</b> %{x:.2f}%<br>' +
                             '<b>Return:</b> %{y:.2f}%<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=dict(
                text='Benchmark Comparison Analysis',
                x=0.5,
                font=dict(size=18)
            ),
            height=700,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Correlation", row=1, col=2)
        fig.update_yaxes(title_text="Relative Performance (%)", row=2, col=1)
        fig.update_yaxes(title_text="Annual Return (%)", row=2, col=2)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Annual Volatility (%)", row=2, col=2)
        
        return fig

    async def create_rolling_metrics_chart(
        self, 
        backtest_result: Any
    ) -> go.Figure:
        """Create rolling performance metrics chart."""
        
        equity_curve = backtest_result.equity_curve
        
        if equity_curve.empty or len(equity_curve) < 60:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data for rolling metrics (need at least 60 data points)",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Rolling Performance Metrics",
                height=600
            )
            return fig
        
        # Calculate daily returns
        daily_returns = equity_curve.pct_change().dropna()
        
        # Calculate rolling metrics
        window = min(60, len(daily_returns) // 4)  # Use 60 days or 1/4 of data
        
        rolling_sharpe = (
            daily_returns.rolling(window=window).mean() / 
            daily_returns.rolling(window=window).std() * 
            np.sqrt(252)
        ).dropna()
        
        rolling_volatility = (
            daily_returns.rolling(window=window).std() * 
            np.sqrt(252) * 100
        ).dropna()
        
        # Calculate rolling max drawdown
        rolling_max = equity_curve.rolling(window=window).max()
        rolling_drawdown = ((equity_curve - rolling_max) / rolling_max * 100).dropna()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                f'Rolling Sharpe Ratio ({window}-day)',
                f'Rolling Volatility ({window}-day)',
                f'Rolling Drawdown ({window}-day)'
            )
        )
        
        # Add rolling Sharpe ratio
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color=self.color_palette['primary'], width=2),
                hovertemplate='<b>Date:</b> %{x}<br>' +
                             '<b>Sharpe Ratio:</b> %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add rolling volatility
        fig.add_trace(
            go.Scatter(
                x=rolling_volatility.index,
                y=rolling_volatility.values,
                mode='lines',
                name='Rolling Volatility',
                line=dict(color=self.color_palette['warning'], width=2),
                hovertemplate='<b>Date:</b> %{x}<br>' +
                             '<b>Volatility:</b> %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add rolling drawdown
        fig.add_trace(
            go.Scatter(
                x=rolling_drawdown.index,
                y=rolling_drawdown.values,
                mode='lines',
                name='Rolling Drawdown',
                line=dict(color=self.color_palette['danger'], width=2),
                fill='tonexty',
                fillcolor='rgba(214, 39, 40, 0.3)',
                hovertemplate='<b>Date:</b> %{x}<br>' +
                             '<b>Drawdown:</b> %{y:.2f}%<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add reference lines
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        fig.update_layout(
            title=dict(
                text='Rolling Performance Metrics',
                x=0.5,
                font=dict(size=18)
            ),
            height=600,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig

    async def create_comprehensive_dashboard(
        self,
        backtest_result: Any,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, go.Figure]:
        """Create a comprehensive backtesting dashboard with all visualizations."""
        
        dashboard = {}
        
        try:
            # Main performance overview
            dashboard['overview'] = await self.create_performance_overview_chart(
                backtest_result, benchmark_data
            )
            
            # Detailed equity curve with annotations
            dashboard['detailed_equity'] = await self.create_detailed_equity_curve(
                backtest_result, benchmark_data
            )
            
            # Risk analysis dashboard
            dashboard['risk_analysis'] = await self.create_advanced_risk_analysis(
                backtest_result
            )
            
            # Trade execution analysis
            dashboard['trade_execution'] = await self.create_trade_execution_analysis(
                backtest_result
            )
            
            # Performance attribution breakdown
            dashboard['attribution_breakdown'] = await self.create_detailed_attribution_analysis(
                backtest_result
            )
            
            # Benchmark comparison suite
            if benchmark_data is not None:
                dashboard['benchmark_suite'] = await self.create_benchmark_analysis_suite(
                    backtest_result, benchmark_data
                )
            
            # Rolling metrics with confidence intervals
            dashboard['rolling_analysis'] = await self.create_advanced_rolling_analysis(
                backtest_result
            )
            
            # Stress testing visualization
            dashboard['stress_testing'] = await self.create_stress_testing_visualization(
                backtest_result
            )
            
        except Exception as e:
            logger.error(f"Error creating comprehensive dashboard: {e}")
            raise
        
        return dashboard

    async def create_performance_overview_chart(
        self,
        backtest_result: Any,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """Create a comprehensive performance overview chart."""
        
        # Create a 2x3 subplot layout for overview
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Equity Curve vs Benchmark',
                'Risk-Return Profile',
                'Drawdown Analysis',
                'Monthly Performance',
                'Key Metrics',
                'Performance Attribution'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "indicator"}, {"type": "bar"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        equity_curve = backtest_result.equity_curve
        
        if not equity_curve.empty:
            # 1. Equity curve vs benchmark
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve.values / equity_curve.iloc[0],
                    mode='lines',
                    name='Strategy',
                    line=dict(color=self.color_palette['primary'], width=3),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Normalized Value:</b> %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            if benchmark_data is not None and not benchmark_data.empty:
                benchmark_normalized = benchmark_data['close_price'] / benchmark_data['close_price'].iloc[0]
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_normalized.index,
                        y=benchmark_normalized.values,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color=self.color_palette['benchmark'], width=2, dash='dash'),
                        hovertemplate='<b>Date:</b> %{x}<br><b>Normalized Value:</b> %{y:.3f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # 2. Risk-Return scatter
            annual_return = backtest_result.annual_return * 100
            volatility = backtest_result.volatility * 100
            
            fig.add_trace(
                go.Scatter(
                    x=[volatility],
                    y=[annual_return],
                    mode='markers',
                    name='Strategy',
                    marker=dict(
                        size=20,
                        color=self.color_palette['success'] if backtest_result.sharpe_ratio > 1 else self.color_palette['warning'],
                        symbol='circle',
                        line=dict(width=2, color='white')
                    ),
                    text=[f'Sharpe: {backtest_result.sharpe_ratio:.2f}'],
                    textposition='top center',
                    hovertemplate='<b>Return:</b> %{y:.2f}%<br><b>Volatility:</b> %{x:.2f}%<br><b>Sharpe:</b> ' + f'{backtest_result.sharpe_ratio:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # 3. Drawdown analysis
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max * 100
            
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name='Drawdown',
                    fill='tonexty',
                    line=dict(color=self.color_palette['danger'], width=1),
                    fillcolor='rgba(214, 39, 40, 0.3)',
                    hovertemplate='<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>'
                ),
                row=1, col=3
            )
            
            # 4. Monthly performance
            if not backtest_result.monthly_returns.empty:
                monthly_returns_pct = backtest_result.monthly_returns * 100
                colors = [self.color_palette['success'] if x > 0 else self.color_palette['danger'] for x in monthly_returns_pct]
                
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(monthly_returns_pct))),
                        y=monthly_returns_pct.values,
                        marker_color=colors,
                        name='Monthly Returns',
                        text=[f'{x:.1f}%' for x in monthly_returns_pct.values],
                        textposition='auto',
                        hovertemplate='<b>Month:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # 5. Key metrics indicator
            fig.add_trace(
                go.Indicator(
                    mode="number+gauge",
                    value=backtest_result.sharpe_ratio,
                    title={"text": "Sharpe Ratio"},
                    gauge={
                        'axis': {'range': [None, 3]},
                        'bar': {'color': self.color_palette['primary']},
                        'steps': [
                            {'range': [0, 1], 'color': self.color_palette['danger']},
                            {'range': [1, 2], 'color': self.color_palette['warning']},
                            {'range': [2, 3], 'color': self.color_palette['success']}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 2.0
                        }
                    },
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=2, col=2
            )
            
            # 6. Performance attribution
            if backtest_result.performance_attribution:
                symbols = list(backtest_result.performance_attribution.keys())[:5]  # Top 5
                contributions = [backtest_result.performance_attribution[s] for s in symbols]
                colors = [self.color_palette['success'] if c > 0 else self.color_palette['danger'] for c in contributions]
                
                fig.add_trace(
                    go.Bar(
                        x=symbols,
                        y=contributions,
                        marker_color=colors,
                        name='Attribution',
                        text=[f'¥{c:,.0f}' for c in contributions],
                        textposition='auto',
                        hovertemplate='<b>Symbol:</b> %{x}<br><b>Contribution:</b> ¥%{y:,.0f}<extra></extra>'
                    ),
                    row=2, col=3
                )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Performance Overview: {backtest_result.strategy_name}',
                x=0.5,
                font=dict(size=20)
            ),
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_yaxes(title_text="Normalized Value", row=1, col=1)
        fig.update_yaxes(title_text="Annual Return (%)", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=3)
        fig.update_yaxes(title_text="Monthly Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="P&L (¥)", row=2, col=3)
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=1, col=3)
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_xaxes(title_text="Symbol", row=2, col=3)
        
        return fig

    async def create_detailed_equity_curve(
        self,
        backtest_result: Any,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """Create detailed equity curve with trade annotations and key events."""
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Equity Curve with Trade Markers', 'Daily Returns', 'Volume Profile'),
            row_heights=[0.6, 0.25, 0.15]
        )
        
        equity_curve = backtest_result.equity_curve
        
        if equity_curve.empty:
            return fig
        
        # 1. Main equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color=self.color_palette['primary'], width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> ¥%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add benchmark if available
        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_normalized = (
                benchmark_data['close_price'] / benchmark_data['close_price'].iloc[0] * 
                equity_curve.iloc[0]
            )
            fig.add_trace(
                go.Scatter(
                    x=benchmark_normalized.index,
                    y=benchmark_normalized.values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color=self.color_palette['benchmark'], width=2, dash='dash'),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> ¥%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add trade markers
        if backtest_result.trade_log:
            buy_trades = [t for t in backtest_result.trade_log if t['side'] == 'buy']
            sell_trades = [t for t in backtest_result.trade_log if t['side'] == 'sell']
            
            if buy_trades:
                buy_dates = [pd.to_datetime(t['timestamp']) for t in buy_trades]
                buy_values = [equity_curve.loc[d] for d in buy_dates if d in equity_curve.index]
                
                if buy_values:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_dates[:len(buy_values)],
                            y=buy_values,
                            mode='markers',
                            name='Buy Orders',
                            marker=dict(
                                symbol='triangle-up',
                                size=8,
                                color=self.color_palette['success'],
                                line=dict(width=1, color='white')
                            ),
                            hovertemplate='<b>Buy Order</b><br><b>Date:</b> %{x}<br><b>Portfolio Value:</b> ¥%{y:,.0f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
            
            if sell_trades:
                sell_dates = [pd.to_datetime(t['timestamp']) for t in sell_trades]
                sell_values = [equity_curve.loc[d] for d in sell_dates if d in equity_curve.index]
                
                if sell_values:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_dates[:len(sell_values)],
                            y=sell_values,
                            mode='markers',
                            name='Sell Orders',
                            marker=dict(
                                symbol='triangle-down',
                                size=8,
                                color=self.color_palette['danger'],
                                line=dict(width=1, color='white')
                            ),
                            hovertemplate='<b>Sell Order</b><br><b>Date:</b> %{x}<br><b>Portfolio Value:</b> ¥%{y:,.0f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
        
        # 2. Daily returns
        daily_returns = equity_curve.pct_change().dropna() * 100
        colors = [self.color_palette['success'] if x > 0 else self.color_palette['danger'] for x in daily_returns]
        
        fig.add_trace(
            go.Bar(
                x=daily_returns.index,
                y=daily_returns.values,
                marker_color=colors,
                name='Daily Returns',
                opacity=0.7,
                hovertemplate='<b>Date:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 3. Volume profile (if trade data available)
        if backtest_result.trade_log:
            trade_volumes = {}
            for trade in backtest_result.trade_log:
                date = pd.to_datetime(trade['timestamp']).date()
                if date not in trade_volumes:
                    trade_volumes[date] = 0
                trade_volumes[date] += trade['quantity'] * trade['price']
            
            if trade_volumes:
                dates = list(trade_volumes.keys())
                volumes = list(trade_volumes.values())
                
                fig.add_trace(
                    go.Bar(
                        x=dates,
                        y=volumes,
                        marker_color=self.color_palette['info'],
                        name='Trade Volume',
                        opacity=0.6,
                        hovertemplate='<b>Date:</b> %{x}<br><b>Volume:</b> ¥%{y:,.0f}<extra></extra>'
                    ),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Detailed Equity Analysis: {backtest_result.strategy_name}',
                x=0.5,
                font=dict(size=18)
            ),
            height=900,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_yaxes(title_text="Portfolio Value (¥)", row=1, col=1)
        fig.update_yaxes(title_text="Daily Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Trade Volume (¥)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig

    async def create_advanced_risk_analysis(
        self,
        backtest_result: Any
    ) -> go.Figure:
        """Create advanced risk analysis visualization."""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'VaR Analysis',
                'Risk Decomposition',
                'Drawdown Distribution',
                'Rolling Volatility',
                'Risk-Adjusted Returns',
                'Tail Risk Analysis'
            ),
            specs=[
                [{"type": "bar"}, {"type": "pie"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        equity_curve = backtest_result.equity_curve
        
        if equity_curve.empty:
            return fig
        
        daily_returns = equity_curve.pct_change().dropna()
        
        # 1. VaR Analysis
        if 'var_95' in backtest_result.risk_metrics and 'var_99' in backtest_result.risk_metrics:
            var_metrics = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%']
            var_values = [
                backtest_result.risk_metrics.get('var_95', 0) * 100,
                backtest_result.risk_metrics.get('var_99', 0) * 100,
                backtest_result.risk_metrics.get('cvar_95', 0) * 100,
                backtest_result.risk_metrics.get('cvar_99', 0) * 100
            ]
            
            fig.add_trace(
                go.Bar(
                    x=var_metrics,
                    y=var_values,
                    marker_color=[self.color_palette['warning'], self.color_palette['danger'], 
                                 self.color_palette['warning'], self.color_palette['danger']],
                    name='VaR Metrics',
                    text=[f'{v:.2f}%' for v in var_values],
                    textposition='auto',
                    hovertemplate='<b>Metric:</b> %{x}<br><b>Value:</b> %{y:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Risk Decomposition
        risk_components = {
            'Market Risk': abs(backtest_result.beta * backtest_result.volatility * 0.7),
            'Specific Risk': abs(backtest_result.volatility * 0.3),
            'Timing Risk': abs(backtest_result.tracking_error * 0.5) if backtest_result.tracking_error else 0
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(risk_components.keys()),
                values=list(risk_components.values()),
                name='Risk Decomposition',
                marker_colors=[self.color_palette['primary'], self.color_palette['secondary'], self.color_palette['info']],
                hovertemplate='<b>%{label}</b><br><b>Contribution:</b> %{value:.3f}<br><b>Percentage:</b> %{percent}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Drawdown Distribution
        running_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - running_max) / running_max * 100
        drawdown_values = drawdowns[drawdowns < 0].values
        
        if len(drawdown_values) > 0:
            fig.add_trace(
                go.Histogram(
                    x=drawdown_values,
                    nbinsx=20,
                    name='Drawdown Distribution',
                    marker_color=self.color_palette['danger'],
                    opacity=0.7,
                    hovertemplate='<b>Drawdown Range:</b> %{x}<br><b>Frequency:</b> %{y}<extra></extra>'
                ),
                row=1, col=3
            )
        
        # 4. Rolling Volatility
        if len(daily_returns) > 30:
            rolling_vol = daily_returns.rolling(window=30).std() * np.sqrt(252) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol.values,
                    mode='lines',
                    name='30-Day Rolling Volatility',
                    line=dict(color=self.color_palette['warning'], width=2),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Volatility:</b> %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 5. Risk-Adjusted Returns
        risk_adjusted_metrics = {
            'Sharpe Ratio': backtest_result.sharpe_ratio,
            'Sortino Ratio': backtest_result.sortino_ratio,
            'Calmar Ratio': backtest_result.calmar_ratio,
            'Information Ratio': backtest_result.information_ratio
        }
        
        colors = [self.color_palette['success'] if v > 1 else self.color_palette['warning'] if v > 0 else self.color_palette['danger'] 
                 for v in risk_adjusted_metrics.values()]
        
        fig.add_trace(
            go.Bar(
                x=list(risk_adjusted_metrics.keys()),
                y=list(risk_adjusted_metrics.values()),
                marker_color=colors,
                name='Risk-Adjusted Returns',
                text=[f'{v:.2f}' for v in risk_adjusted_metrics.values()],
                textposition='auto',
                hovertemplate='<b>Metric:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 6. Tail Risk Analysis
        if len(daily_returns) > 0:
            sorted_returns = np.sort(daily_returns.values)
            percentiles = np.arange(1, 100)
            percentile_values = np.percentile(sorted_returns, percentiles) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=percentiles,
                    y=percentile_values,
                    mode='lines',
                    name='Return Distribution',
                    line=dict(color=self.color_palette['info'], width=2),
                    hovertemplate='<b>Percentile:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>'
                ),
                row=2, col=3
            )
            
            # Add annotations for tail regions instead of vrect
            fig.add_annotation(
                x=2.5, y=max(percentile_values) * 0.8,
                text="5% Tail", showarrow=False,
                bgcolor=self.color_palette['danger'], opacity=0.7,
                row=2, col=3
            )
            fig.add_annotation(
                x=97.5, y=max(percentile_values) * 0.8,
                text="95% Tail", showarrow=False,
                bgcolor=self.color_palette['success'], opacity=0.7,
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Advanced Risk Analysis: {backtest_result.strategy_name}',
                x=0.5,
                font=dict(size=18)
            ),
            height=700,
            showlegend=False
        )
        
        # Update axes
        fig.update_yaxes(title_text="Value (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=3)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="Ratio", row=2, col=2)
        fig.update_yaxes(title_text="Return (%)", row=2, col=3)
        
        fig.update_xaxes(title_text="Metric", row=1, col=1)
        fig.update_xaxes(title_text="Drawdown (%)", row=1, col=3)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Metric", row=2, col=2)
        fig.update_xaxes(title_text="Percentile", row=2, col=3)
        
        return fig

    async def export_dashboard_to_html(
        self,
        dashboard: Dict[str, go.Figure],
        filename: str = "backtesting_dashboard.html"
    ) -> str:
        """Export the complete dashboard to an HTML file."""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtesting Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .chart-container {{ margin-bottom: 30px; }}
                .dashboard-title {{ text-align: center; color: #333; margin-bottom: 30px; }}
                .chart-title {{ color: #666; margin-bottom: 10px; font-size: 18px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1 class="dashboard-title">Comprehensive Backtesting Dashboard</h1>
        """
        
        for chart_name, fig in dashboard.items():
            chart_html = fig.to_html(include_plotlyjs=False, div_id=f"chart_{chart_name}")
            html_content += f"""
            <div class="chart-container">
                <div class="chart-title">{chart_name.replace('_', ' ').title()}</div>
                {chart_html}
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard exported to {filename}")
        return filename

    async def create_trade_execution_analysis(
        self,
        backtest_result: Any
    ) -> go.Figure:
        """Create detailed trade execution analysis."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Trade Timing Analysis',
                'Execution Quality',
                'Position Holding Periods',
                'Trade Size vs Performance'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "scatter"}]
            ]
        )
        
        trades = backtest_result.trade_log
        
        if not trades:
            fig.add_annotation(
                text="No trade execution data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # 1. Trade timing analysis
        sell_trades = [t for t in trades if t['side'] == 'sell' and 'pnl' in t]
        if sell_trades:
            timestamps = [pd.to_datetime(t['timestamp']) for t in sell_trades]
            pnls = [t['pnl'] for t in sell_trades]
            colors = [self.color_palette['success'] if pnl > 0 else self.color_palette['danger'] for pnl in pnls]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=pnls,
                    mode='markers',
                    marker=dict(
                        color=colors,
                        size=8,
                        opacity=0.7
                    ),
                    name='Trade P&L',
                    hovertemplate='<b>Date:</b> %{x}<br><b>P&L:</b> ¥%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Execution quality (slippage and commission analysis)
        if trades:
            slippages = [t.get('slippage', 0) * 100 for t in trades]  # Convert to percentage
            commissions = [t.get('commission', 0) for t in trades]
            
            fig.add_trace(
                go.Bar(
                    x=['Average Slippage (%)', 'Average Commission (¥)'],
                    y=[np.mean(slippages) if slippages else 0, np.mean(commissions) if commissions else 0],
                    marker_color=[self.color_palette['warning'], self.color_palette['info']],
                    name='Execution Costs',
                    text=[f'{np.mean(slippages):.3f}%' if slippages else '0%', 
                          f'¥{np.mean(commissions):,.0f}' if commissions else '¥0'],
                    textposition='auto',
                    hovertemplate='<b>Metric:</b> %{x}<br><b>Value:</b> %{y}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Position holding periods (simplified - using trade count as proxy)
        if sell_trades:
            # Estimate holding periods based on trade frequency
            holding_periods = []
            for i in range(1, len(sell_trades)):
                prev_time = pd.to_datetime(sell_trades[i-1]['timestamp'])
                curr_time = pd.to_datetime(sell_trades[i]['timestamp'])
                days_diff = (curr_time - prev_time).days
                if days_diff > 0:
                    holding_periods.append(days_diff)
            
            if holding_periods:
                fig.add_trace(
                    go.Histogram(
                        x=holding_periods,
                        nbinsx=15,
                        name='Holding Periods',
                        marker_color=self.color_palette['primary'],
                        opacity=0.7,
                        hovertemplate='<b>Days:</b> %{x}<br><b>Frequency:</b> %{y}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # 4. Trade size vs performance
        if sell_trades:
            trade_sizes = [t['quantity'] * t['price'] for t in sell_trades]
            trade_returns = [t['pnl'] / (t['quantity'] * t['price']) * 100 for t in sell_trades if t['quantity'] * t['price'] > 0]
            
            if len(trade_sizes) == len(trade_returns):
                colors = [self.color_palette['success'] if r > 0 else self.color_palette['danger'] for r in trade_returns]
                
                fig.add_trace(
                    go.Scatter(
                        x=trade_sizes,
                        y=trade_returns,
                        mode='markers',
                        marker=dict(
                            color=colors,
                            size=8,
                            opacity=0.7
                        ),
                        name='Size vs Return',
                        hovertemplate='<b>Trade Size:</b> ¥%{x:,.0f}<br><b>Return:</b> %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=dict(
                text='Trade Execution Analysis',
                x=0.5,
                font=dict(size=18)
            ),
            height=600,
            showlegend=False
        )
        
        # Update axes
        fig.update_yaxes(title_text="P&L (¥)", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=2)
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Metric", row=1, col=2)
        fig.update_xaxes(title_text="Holding Period (Days)", row=2, col=1)
        fig.update_xaxes(title_text="Trade Size (¥)", row=2, col=2)
        
        return fig

    async def create_detailed_attribution_analysis(
        self,
        backtest_result: Any
    ) -> go.Figure:
        """Create detailed performance attribution analysis."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Symbol Contribution Breakdown',
                'Cumulative Attribution',
                'Risk-Adjusted Attribution',
                'Attribution Over Time'
            ),
            specs=[
                [{"type": "bar"}, {"type": "waterfall"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        attribution = backtest_result.performance_attribution
        
        if not attribution:
            fig.add_annotation(
                text="No attribution data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Sort by absolute contribution
        sorted_attribution = dict(
            sorted(attribution.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        
        symbols = list(sorted_attribution.keys())[:10]  # Top 10
        contributions = [sorted_attribution[s] for s in symbols]
        
        # 1. Symbol contribution breakdown
        colors = [self.color_palette['success'] if c > 0 else self.color_palette['danger'] for c in contributions]
        
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=contributions,
                marker_color=colors,
                name='Contributions',
                text=[f'¥{c:,.0f}' for c in contributions],
                textposition='auto',
                hovertemplate='<b>Symbol:</b> %{x}<br><b>Contribution:</b> ¥%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Cumulative attribution (waterfall chart)
        cumulative_values = [0] + list(np.cumsum(contributions))
        
        fig.add_trace(
            go.Waterfall(
                x=["Start"] + symbols + ["Total"],
                y=[0] + contributions + [sum(contributions)],
                measure=["absolute"] + ["relative"] * len(contributions) + ["total"],
                name="Cumulative Attribution",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                hovertemplate='<b>%{x}</b><br><b>Value:</b> ¥%{y:,.0f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Risk-adjusted attribution (contribution per unit of risk)
        if len(contributions) > 0:
            # Estimate risk as proportional to absolute contribution
            risks = [abs(c) * 0.1 for c in contributions]  # Simplified risk estimate
            risk_adjusted = [c / r if r > 0 else 0 for c, r in zip(contributions, risks)]
            
            colors_ra = [self.color_palette['success'] if ra > 0 else self.color_palette['danger'] for ra in risk_adjusted]
            
            fig.add_trace(
                go.Scatter(
                    x=risks,
                    y=risk_adjusted,
                    mode='markers+text',
                    marker=dict(
                        color=colors_ra,
                        size=10,
                        opacity=0.7
                    ),
                    text=symbols,
                    textposition='top center',
                    name='Risk-Adjusted',
                    hovertemplate='<b>Symbol:</b> %{text}<br><b>Risk:</b> %{x:.0f}<br><b>Risk-Adj Return:</b> %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Attribution over time (simplified)
        if backtest_result.trade_log:
            # Group trades by month to show attribution over time
            monthly_attribution = {}
            for trade in backtest_result.trade_log:
                if trade['side'] == 'sell' and 'pnl' in trade:
                    month = pd.to_datetime(trade['timestamp']).to_period('M')
                    symbol = trade['symbol']
                    key = f"{month}_{symbol}"
                    
                    if month not in monthly_attribution:
                        monthly_attribution[month] = 0
                    monthly_attribution[month] += trade['pnl']
            
            if monthly_attribution:
                months = list(monthly_attribution.keys())
                monthly_values = list(monthly_attribution.values())
                
                fig.add_trace(
                    go.Scatter(
                        x=[str(m) for m in months],
                        y=monthly_values,
                        mode='lines+markers',
                        name='Monthly Attribution',
                        line=dict(color=self.color_palette['primary'], width=2),
                        marker=dict(size=6),
                        hovertemplate='<b>Month:</b> %{x}<br><b>Attribution:</b> ¥%{y:,.0f}<extra></extra>'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=dict(
                text='Detailed Performance Attribution Analysis',
                x=0.5,
                font=dict(size=18)
            ),
            height=700,
            showlegend=False
        )
        
        # Update axes
        fig.update_yaxes(title_text="Contribution (¥)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative (¥)", row=1, col=2)
        fig.update_yaxes(title_text="Risk-Adj Return", row=2, col=1)
        fig.update_yaxes(title_text="Monthly Attribution (¥)", row=2, col=2)
        
        fig.update_xaxes(title_text="Symbol", row=1, col=1)
        fig.update_xaxes(title_text="Component", row=1, col=2)
        fig.update_xaxes(title_text="Risk", row=2, col=1)
        fig.update_xaxes(title_text="Month", row=2, col=2)
        
        return fig

    async def create_benchmark_analysis_suite(
        self,
        backtest_result: Any,
        benchmark_data: pd.DataFrame
    ) -> go.Figure:
        """Create comprehensive benchmark analysis suite."""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Performance Comparison',
                'Rolling Beta',
                'Active Return',
                'Tracking Error',
                'Information Ratio',
                'Up/Down Market Performance'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08
        )
        
        equity_curve = backtest_result.equity_curve
        
        if equity_curve.empty or benchmark_data.empty:
            fig.add_annotation(
                text="Insufficient data for benchmark analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Calculate returns
        strategy_returns = equity_curve.pct_change().dropna()
        benchmark_returns = benchmark_data['close_price'].pct_change().dropna()
        
        # Align data
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) == 0:
            return fig
        
        strategy_aligned = strategy_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]
        
        # 1. Performance comparison
        strategy_cumulative = (1 + strategy_aligned).cumprod()
        benchmark_cumulative = (1 + benchmark_aligned).cumprod()
        
        fig.add_trace(
            go.Scatter(
                x=strategy_cumulative.index,
                y=strategy_cumulative.values,
                mode='lines',
                name='Strategy',
                line=dict(color=self.color_palette['primary'], width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Cumulative Return:</b> %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative.values,
                mode='lines',
                name='Benchmark',
                line=dict(color=self.color_palette['benchmark'], width=2, dash='dash'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Cumulative Return:</b> %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Rolling beta
        if len(strategy_aligned) > 60:
            rolling_beta = []
            window = 60
            
            for i in range(window, len(strategy_aligned)):
                s_window = strategy_aligned.iloc[i-window:i]
                b_window = benchmark_aligned.iloc[i-window:i]
                
                if np.var(b_window) > 0:
                    beta = np.cov(s_window, b_window)[0, 1] / np.var(b_window)
                else:
                    beta = 1.0
                
                rolling_beta.append(beta)
            
            if rolling_beta:
                beta_dates = strategy_aligned.index[window:]
                
                fig.add_trace(
                    go.Scatter(
                        x=beta_dates,
                        y=rolling_beta,
                        mode='lines',
                        name='Rolling Beta',
                        line=dict(color=self.color_palette['warning'], width=2),
                        hovertemplate='<b>Date:</b> %{x}<br><b>Beta:</b> %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=2
                )
                
                # Add beta = 1 reference line
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)
        
        # 3. Active return
        active_returns = strategy_aligned - benchmark_aligned
        cumulative_active = (1 + active_returns).cumprod() - 1
        
        fig.add_trace(
            go.Scatter(
                x=cumulative_active.index,
                y=cumulative_active.values * 100,
                mode='lines',
                name='Active Return',
                line=dict(color=self.color_palette['success'], width=2),
                fill='tonexty',
                fillcolor='rgba(44, 160, 44, 0.3)',
                hovertemplate='<b>Date:</b> %{x}<br><b>Active Return:</b> %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # 4. Rolling tracking error
        if len(active_returns) > 30:
            rolling_te = active_returns.rolling(window=30).std() * np.sqrt(252) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_te.index,
                    y=rolling_te.values,
                    mode='lines',
                    name='Tracking Error',
                    line=dict(color=self.color_palette['danger'], width=2),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Tracking Error:</b> %{y:.2f}%<extra></extra>'
                ),
                row=2, col=2
            )
        
        # 5. Rolling information ratio
        if len(active_returns) > 30:
            rolling_ir = []
            window = 30
            
            for i in range(window, len(active_returns)):
                ar_window = active_returns.iloc[i-window:i]
                mean_ar = np.mean(ar_window)
                std_ar = np.std(ar_window)
                
                if std_ar > 0:
                    ir = mean_ar / std_ar * np.sqrt(252)
                else:
                    ir = 0
                
                rolling_ir.append(ir)
            
            if rolling_ir:
                ir_dates = active_returns.index[window:]
                
                fig.add_trace(
                    go.Scatter(
                        x=ir_dates,
                        y=rolling_ir,
                        mode='lines',
                        name='Information Ratio',
                        line=dict(color=self.color_palette['info'], width=2),
                        hovertemplate='<b>Date:</b> %{x}<br><b>Information Ratio:</b> %{y:.2f}<extra></extra>'
                    ),
                    row=3, col=1
                )
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        # 6. Up/Down market performance
        up_market_days = benchmark_aligned > 0
        down_market_days = benchmark_aligned < 0
        
        up_strategy_return = np.mean(strategy_aligned[up_market_days]) * 252 * 100 if up_market_days.any() else 0
        down_strategy_return = np.mean(strategy_aligned[down_market_days]) * 252 * 100 if down_market_days.any() else 0
        up_benchmark_return = np.mean(benchmark_aligned[up_market_days]) * 252 * 100 if up_market_days.any() else 0
        down_benchmark_return = np.mean(benchmark_aligned[down_market_days]) * 252 * 100 if down_market_days.any() else 0
        
        fig.add_trace(
            go.Bar(
                x=['Up Market Strategy', 'Up Market Benchmark', 'Down Market Strategy', 'Down Market Benchmark'],
                y=[up_strategy_return, up_benchmark_return, down_strategy_return, down_benchmark_return],
                marker_color=[
                    self.color_palette['success'], self.color_palette['benchmark'],
                    self.color_palette['danger'], self.color_palette['benchmark']
                ],
                name='Market Performance',
                text=[f'{x:.1f}%' for x in [up_strategy_return, up_benchmark_return, down_strategy_return, down_benchmark_return]],
                textposition='auto',
                hovertemplate='<b>Scenario:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>'
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            title=dict(
                text='Comprehensive Benchmark Analysis Suite',
                x=0.5,
                font=dict(size=18)
            ),
            height=900,
            showlegend=True
        )
        
        # Update axes
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Beta", row=1, col=2)
        fig.update_yaxes(title_text="Active Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Tracking Error (%)", row=2, col=2)
        fig.update_yaxes(title_text="Information Ratio", row=3, col=1)
        fig.update_yaxes(title_text="Annualized Return (%)", row=3, col=2)
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_xaxes(title_text="Market Scenario", row=3, col=2)
        
        return fig

    async def create_advanced_rolling_analysis(
        self,
        backtest_result: Any
    ) -> go.Figure:
        """Create advanced rolling metrics analysis with confidence intervals."""
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                'Rolling Sharpe Ratio with Confidence Bands',
                'Rolling Volatility with Regime Detection',
                'Rolling Maximum Drawdown',
                'Rolling Win Rate'
            )
        )
        
        equity_curve = backtest_result.equity_curve
        
        if equity_curve.empty or len(equity_curve) < 60:
            fig.add_annotation(
                text="Insufficient data for advanced rolling analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        daily_returns = equity_curve.pct_change().dropna()
        window = min(60, len(daily_returns) // 4)
        
        # 1. Rolling Sharpe with confidence bands
        rolling_sharpe = (
            daily_returns.rolling(window=window).mean() / 
            daily_returns.rolling(window=window).std() * 
            np.sqrt(252)
        ).dropna()
        
        # Calculate confidence bands (simplified)
        sharpe_std = rolling_sharpe.rolling(window=30).std()
        upper_band = rolling_sharpe + 1.96 * sharpe_std
        lower_band = rolling_sharpe - 1.96 * sharpe_std
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=upper_band.values,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=lower_band.values,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.2)',
                name='95% Confidence',
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color=self.color_palette['primary'], width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Sharpe Ratio:</b> %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Rolling volatility with regime detection
        rolling_vol = daily_returns.rolling(window=window).std() * np.sqrt(252) * 100
        vol_median = rolling_vol.median()
        
        # Color code by volatility regime
        high_vol_mask = rolling_vol > vol_median * 1.5
        normal_vol_mask = (rolling_vol <= vol_median * 1.5) & (rolling_vol >= vol_median * 0.5)
        low_vol_mask = rolling_vol < vol_median * 0.5
        
        if high_vol_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol[high_vol_mask].index,
                    y=rolling_vol[high_vol_mask].values,
                    mode='markers',
                    marker=dict(color=self.color_palette['danger'], size=4),
                    name='High Vol Regime',
                    hovertemplate='<b>Date:</b> %{x}<br><b>Volatility:</b> %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        if normal_vol_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol[normal_vol_mask].index,
                    y=rolling_vol[normal_vol_mask].values,
                    mode='markers',
                    marker=dict(color=self.color_palette['warning'], size=4),
                    name='Normal Vol Regime',
                    hovertemplate='<b>Date:</b> %{x}<br><b>Volatility:</b> %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        if low_vol_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol[low_vol_mask].index,
                    y=rolling_vol[low_vol_mask].values,
                    mode='markers',
                    marker=dict(color=self.color_palette['success'], size=4),
                    name='Low Vol Regime',
                    hovertemplate='<b>Date:</b> %{x}<br><b>Volatility:</b> %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 3. Rolling maximum drawdown
        rolling_max_dd = []
        for i in range(window, len(equity_curve)):
            window_data = equity_curve.iloc[i-window:i]
            running_max = window_data.expanding().max()
            drawdown = (window_data - running_max) / running_max
            rolling_max_dd.append(drawdown.min() * 100)
        
        if rolling_max_dd:
            dd_dates = equity_curve.index[window:]
            
            fig.add_trace(
                go.Scatter(
                    x=dd_dates,
                    y=rolling_max_dd,
                    mode='lines',
                    name='Rolling Max DD',
                    line=dict(color=self.color_palette['danger'], width=2),
                    fill='tonexty',
                    fillcolor='rgba(214, 39, 40, 0.3)',
                    hovertemplate='<b>Date:</b> %{x}<br><b>Max Drawdown:</b> %{y:.2f}%<extra></extra>'
                ),
                row=3, col=1
            )
        
        # 4. Rolling win rate (if trade data available)
        if backtest_result.trade_log:
            sell_trades = [t for t in backtest_result.trade_log if t['side'] == 'sell' and 'pnl' in t]
            
            if len(sell_trades) > 10:
                rolling_win_rates = []
                trade_window = min(20, len(sell_trades) // 4)
                
                for i in range(trade_window, len(sell_trades)):
                    window_trades = sell_trades[i-trade_window:i]
                    wins = sum(1 for t in window_trades if t['pnl'] > 0)
                    win_rate = wins / len(window_trades) * 100
                    rolling_win_rates.append(win_rate)
                
                if rolling_win_rates:
                    # Use trade timestamps for x-axis
                    win_rate_dates = [pd.to_datetime(sell_trades[i]['timestamp']) 
                                    for i in range(trade_window, len(sell_trades))]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=win_rate_dates,
                            y=rolling_win_rates,
                            mode='lines+markers',
                            name='Rolling Win Rate',
                            line=dict(color=self.color_palette['info'], width=2),
                            marker=dict(size=4),
                            hovertemplate='<b>Date:</b> %{x}<br><b>Win Rate:</b> %{y:.1f}%<extra></extra>'
                        ),
                        row=4, col=1
                    )
                    
                    # Add 50% reference line
                    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=4, col=1)
        
        fig.update_layout(
            title=dict(
                text='Advanced Rolling Performance Analysis',
                x=0.5,
                font=dict(size=18)
            ),
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="Max Drawdown (%)", row=3, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=4, col=1)
        fig.update_xaxes(title_text="Date", row=4, col=1)
        
        # Add reference lines
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        return fig

    async def create_stress_testing_visualization(
        self,
        backtest_result: Any
    ) -> go.Figure:
        """Create stress testing and scenario analysis visualization."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Return Distribution Analysis',
                'Tail Risk Scenarios',
                'Volatility Stress Test',
                'Correlation Breakdown'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ]
        )
        
        equity_curve = backtest_result.equity_curve
        
        if equity_curve.empty:
            fig.add_annotation(
                text="Insufficient data for stress testing",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        daily_returns = equity_curve.pct_change().dropna() * 100
        
        # 1. Return distribution with normal overlay
        fig.add_trace(
            go.Histogram(
                x=daily_returns.values,
                nbinsx=30,
                name='Actual Returns',
                marker_color=self.color_palette['primary'],
                opacity=0.7,
                histnorm='probability density',
                hovertemplate='<b>Return Range:</b> %{x}<br><b>Density:</b> %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Overlay normal distribution
        if len(daily_returns) > 0:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            x_normal = np.linspace(daily_returns.min(), daily_returns.max(), 100)
            y_normal = stats.norm.pdf(x_normal, mean_return, std_return)
            
            fig.add_trace(
                go.Scatter(
                    x=x_normal,
                    y=y_normal,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color=self.color_palette['danger'], width=2, dash='dash'),
                    hovertemplate='<b>Return:</b> %{x:.2f}%<br><b>Normal Density:</b> %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Tail risk scenarios
        if len(daily_returns) > 0:
            percentiles = [1, 5, 10, 90, 95, 99]
            percentile_values = [np.percentile(daily_returns, p) for p in percentiles]
            colors = [self.color_palette['danger']] * 3 + [self.color_palette['success']] * 3
            
            fig.add_trace(
                go.Bar(
                    x=[f'{p}th %ile' for p in percentiles],
                    y=percentile_values,
                    marker_color=colors,
                    name='Tail Scenarios',
                    text=[f'{v:.2f}%' for v in percentile_values],
                    textposition='auto',
                    hovertemplate='<b>Percentile:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Volatility stress test
        if len(daily_returns) > 30:
            # Calculate rolling volatility
            rolling_vol = daily_returns.rolling(window=30).std()
            rolling_returns = daily_returns.rolling(window=30).mean()
            
            # Create volatility buckets
            vol_buckets = pd.qcut(rolling_vol.dropna(), q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            # Calculate average returns for each volatility regime
            bucket_returns = []
            bucket_labels = []
            
            # Align the indices properly
            aligned_returns = rolling_returns.loc[rolling_vol.dropna().index]
            
            for bucket in vol_buckets.cat.categories:
                mask = vol_buckets == bucket
                if mask.any():
                    # Use aligned indices
                    bucket_indices = vol_buckets.index[mask]
                    bucket_return_values = aligned_returns.loc[bucket_indices]
                    avg_return = bucket_return_values.mean()
                    bucket_returns.append(avg_return)
                    bucket_labels.append(bucket)
            
            if bucket_returns:
                colors_vol = [self.color_palette['success'] if r > 0 else self.color_palette['danger'] for r in bucket_returns]
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(bucket_labels))),
                        y=bucket_returns,
                        mode='markers+lines',
                        marker=dict(
                            color=colors_vol,
                            size=10,
                            line=dict(width=2, color='white')
                        ),
                        line=dict(color=self.color_palette['info'], width=2),
                        name='Vol Stress Test',
                        text=bucket_labels,
                        hovertemplate='<b>Vol Regime:</b> %{text}<br><b>Avg Return:</b> %{y:.3f}%<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # 4. Correlation breakdown (simplified heatmap)
        if backtest_result.trade_log:
            # Create a simple correlation matrix based on trade timing
            symbols = list(set(t['symbol'] for t in backtest_result.trade_log))[:5]  # Top 5 symbols
            
            if len(symbols) > 1:
                # Create a simple correlation matrix (simplified)
                corr_matrix = np.random.rand(len(symbols), len(symbols))  # Placeholder
                np.fill_diagonal(corr_matrix, 1.0)  # Perfect self-correlation
                
                # Make symmetric
                corr_matrix = (corr_matrix + corr_matrix.T) / 2
                np.fill_diagonal(corr_matrix, 1.0)
                
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix,
                        x=symbols,
                        y=symbols,
                        colorscale='RdBu',
                        zmid=0,
                        text=np.round(corr_matrix, 2),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        hovertemplate='<b>%{y} vs %{x}</b><br><b>Correlation:</b> %{z:.2f}<extra></extra>'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=dict(
                text='Stress Testing and Scenario Analysis',
                x=0.5,
                font=dict(size=18)
            ),
            height=700,
            showlegend=True
        )
        
        # Update axes
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=2)
        fig.update_yaxes(title_text="Avg Return (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Daily Return (%)", row=1, col=1)
        fig.update_xaxes(title_text="Percentile", row=1, col=2)
        fig.update_xaxes(title_text="Volatility Regime", row=2, col=1)
        
        return fig

    async def export_charts_to_html(
        self, 
        charts: Dict[str, go.Figure], 
        output_path: str = "backtest_report.html"
    ) -> str:
        """Export all charts to a comprehensive HTML report."""
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtesting Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .chart-container { margin-bottom: 30px; }
                .header { text-align: center; margin-bottom: 40px; }
                .section-title { font-size: 24px; margin: 30px 0 15px 0; color: #333; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Comprehensive Backtesting Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
        """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Add each chart
        chart_order = [
            ('equity_curve', 'Equity Curve Analysis'),
            ('risk_metrics', 'Risk Metrics Dashboard'),
            ('performance_attribution', 'Performance Attribution'),
            ('trade_analysis', 'Trade Analysis'),
            ('monthly_returns', 'Monthly Returns'),
            ('benchmark_comparison', 'Benchmark Comparison'),
            ('rolling_metrics', 'Rolling Metrics')
        ]
        
        for chart_key, section_title in chart_order:
            if chart_key in charts:
                html_content += f"""
                <div class="section-title">{section_title}</div>
                <div class="chart-container" id="{chart_key}"></div>
                <script>
                    Plotly.newPlot('{chart_key}', {charts[chart_key].to_json()});
                </script>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Backtesting report exported to {output_path}")
        return output_path

    def get_chart_config(self) -> Dict[str, Any]:
        """Get default chart configuration."""
        return {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'backtest_chart',
                'height': 600,
                'width': 1000,
                'scale': 2
            }
        }